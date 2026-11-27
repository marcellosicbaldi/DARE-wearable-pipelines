"""Launch the independent, local diary-guided raw ACC/gyro annotation app."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dare_wearables.lower_back.annotation.data import AnnotationConfig
from dare_wearables.lower_back.annotation.server import AnnotationApp, make_server


def demo_app(folder):
    """Invented recording for UI verification; never opens participant data."""
    import numpy as np
    import pandas as pd
    from dare_wearables.lower_back.annotation.data import RawCache, write_raw_cache
    folder = Path(folder)
    bronze = folder/'bronze'
    (bronze/'900001/T0/McRoberts').mkdir(parents=True)
    source = bronze/'900001/T0/McRoberts/demo.OMX'
    source.write_bytes(b'invented annotation demo, not an OMX recording')
    fs,n = 100.,120000
    t = np.arange(n)/fs
    quiet = ((t>=120)&(t<310))|((t>=700)&(t<820))
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'datetime':pd.Timestamp('2024-01-01 08:00')+pd.to_timedelta(t,unit='s')})
    for i, name in enumerate(('ax','ay','az')):
        frame[name] = np.where(quiet,0,.05*np.sin(.9*t+i))+(1 if i==2 else 0)+rng.normal(0,.0005,n)
    for i, name in enumerate(('gx','gy','gz')):
        frame[name] = np.where(quiet,0,.1*np.sin(2*np.pi*.3*t+i))+rng.normal(0,.001,n)
    write_raw_cache(folder/'cache',frame,source=source,identity='synthetic-raw-identity',fs=fs,hours=168,policy='strict')
    cache = RawCache(folder/'cache')
    class InventedDiary:
        path = folder/'synthetic_diary.csv'
        sha256 = 'synthetic-diary'
        rows = pd.DataFrame(index=['900001'])
        def intervals(self,subject):
            return [{'id':f'diary_1_{i}','start_ns':cache.first_ns+round(a*1e9),'end_ns':cache.first_ns+round(b*1e9),
                     'diary_day':1,'period_slot':i,'diary_date':'2024-01-01','start_field':'invented_start',
                     'end_field':'invented_end','start_raw':'','end_raw':''}
                    for i,(a,b) in enumerate(((100,340),(690,840)),1)],[]
    config = AnnotationConfig(bronze,folder/'synthetic_diary.csv',folder/'labels',folder/'cache',subject='900001')
    return AnnotationApp(config,diary=InventedDiary(),cache_loader=lambda *_:cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/nonwear_annotation.local.toml')
    parser.add_argument('--subject',help='Initial participant; default comes from the private config.')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--prepare-only',action='store_true',help='Cache initial raw recording without starting the UI.')
    parser.add_argument('--demo',action='store_true',help='Use invented data in a temporary output folder.')
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix='nonwear-demo-') if args.demo else None
    try:
        if args.demo:
            app = demo_app(temporary.name)
        else:
            if not args.config.exists():
                parser.error('Private config not found. Copy configs/nonwear_annotation.example.toml to configs/nonwear_annotation.local.toml and set your paths.')
            config = AnnotationConfig.from_toml(args.config)
            if args.subject:
                config = replace(config,subject=args.subject)
            app = AnnotationApp(config)
        if args.prepare_only:
            print(f'Preparing {app.config.subject}: decoding raw ACC/gyro and building the reusable cache…',flush=True)
            app.open(app.config.subject,background=False)
            result = app.session()
            if result['status'] != 'ready':
                raise ValueError(result['error'])
            print(f"Ready: {result['subject']} / {result['visit']}, {result['sample_rate_hz']:g} Hz; {len(result['rows'])} diary proposals.",flush=True)
            print(f'Cache: {app.cache.folder}',flush=True)
            return
        server = make_server(app,args.port)
        address = f'http://127.0.0.1:{server.server_port}'
        app.open(app.config.subject)
        print(f'Nonwear annotation: {address}',flush=True)
        print(f'Labels: {app.config.output_root}',flush=True)
        print('Leave this terminal open. Press Ctrl+C to stop the app.',flush=True)
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    except (ValueError,OSError,KeyError) as error:
        parser.exit(1,f'Cannot start annotation: {error}\n')
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == '__main__':
    main()
