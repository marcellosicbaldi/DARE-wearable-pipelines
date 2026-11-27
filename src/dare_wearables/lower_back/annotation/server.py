"""Loopback-only annotation app. No network services or extra UI dependencies."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlparse

from .data import Diary, diary_for_recording, participants, prepare_cache, subject_id
from .labels import LabelStore, RevisionConflict


class AnnotationApp:
    def __init__(self, config, *, diary=None, cache_loader=prepare_cache):
        self.config = config
        self.diary = diary or Diary(config.diary_csv)
        self.participants = participants(config,self.diary)
        self.cache_loader = cache_loader
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.RLock()
        self.status, self.error, self.subject = 'idle', None, None
        self.cache, self.store = None, None

    def open(self, subject, *, background=True):
        subject = subject_id(subject)
        if subject not in self.participants:
            raise ValueError('No diary and raw recording found for this participant.')
        with self.lock:
            if self.status == 'loading':
                raise ValueError('A participant is still loading. Please wait.')
            self.status, self.error, self.subject = 'loading', None, subject
            self.cache, self.store = None, None
        def load():
            try:
                cache = self.cache_loader(self.config,subject)
                imported = diary_for_recording(self.diary,subject,cache)
                store = LabelStore(self.config.output_root,subject,self.config.visit,cache,imported)
                record = store.load()
                if record['revision']:
                    store.write_csv(record)
                with self.lock:
                    self.cache,self.store,self.status = cache,store,'ready'
            except Exception as error:
                with self.lock:
                    self.status,self.error = 'error',str(error)
        if background:
            threading.Thread(target=load,daemon=True).start()
        else:
            load()

    def ready(self, subject):
        with self.lock:
            if self.status != 'ready' or self.subject != subject:
                raise ValueError('This participant is not loaded; refresh the recording.')
            return self.cache, self.store

    def session(self):
        with self.lock:
            result = {'status': self.status,'error':self.error,'subject':self.subject,
                      'participants':self.participants,'visit':self.config.visit,'token':self.token,
                      'default_reviewer':self.config.reviewer}
            if self.status == 'ready':
                cache,store = self.cache,self.store
                record = store.load()
                result.update({'revision':record['revision'],'reviewer':record['reviewer'],
                               'start_time':cache.boundary(0)['time'],'duration':cache.duration,'sample_rate_hz':cache.fs,
                               'n_samples':len(cache.times),'diary':store.diary['intervals'],'issues':store.diary['issues'],
                               'rows':store.display_rows(record),'output_folder':str(store.folder),
                               'csv_path':str(store.folder/'reviewed_intervals.csv')})
            return result


def make_server(app, port=8765):
    from .web import PAGE

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def response(self, value, *, status=200, content_type='application/json', filename=None):
            body = value.encode() if isinstance(value,str) else json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            if filename:
                self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.valid_host():
                return self.response({'error':'Use the local address printed by the launcher.'},status=403)
            url = urlparse(self.path)
            args = parse_qs(url.query)
            try:
                if url.path == '/':
                    return self.response(PAGE,content_type='text/html')
                if url.path == '/api/session':
                    return self.response(app.session())
                if url.path in ('/api/view','/api/export'):
                    cache,store = app.ready(args.get('subject',[''])[0])
                    if url.path == '/api/view':
                        return self.response(cache.view(float(args['left'][0]),float(args['right'][0]),int(args.get('width',['1200'])[0])))
                    return self.response(store.csv_text(),content_type='text/csv',
                                         filename=f'{store.subject}_{store.visit}_reviewed_intervals.csv')
                return self.response({'error':'Not found'},status=404)
            except (ValueError,KeyError,TypeError,OSError) as error:
                return self.response({'error':str(error)},status=400)

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Annotation-Token') != app.token:
                return self.response({'error':'Refresh the annotation page before saving.'},status=403)
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request size.')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('Expected a JSON object.')
                if self.path == '/api/open':
                    app.open(data['subject'])
                    return self.response({'status':'loading'})
                cache,store = app.ready(data.get('subject'))
                if self.path == '/api/snap':
                    start, end = cache.snap(data['start']), cache.snap(data['end'],edge='end')
                    if end['sample'] <= start['sample']:
                        raise ValueError('End must follow start by at least one sample.')
                    return self.response({'start':start,'end':end})
                if self.path == '/api/save':
                    record,warning = store.save(data['rows'],data['reviewer'],data['revision'])
                    return self.response({'revision':record['revision'],'saved_at_utc':record['saved_at_utc'],
                                          'warning':warning,'rows':store.display_rows(record)})
                return self.response({'error':'Not found'},status=404)
            except RevisionConflict as error:
                return self.response({'error':str(error)},status=409)
            except (ValueError,KeyError,TypeError,OSError) as error:
                return self.response({'error':str(error)},status=400)

    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads = True
    return server
