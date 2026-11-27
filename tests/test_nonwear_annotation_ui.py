"""Run real client event handlers against an invented DOM, without a browser."""
import os
import shutil
import subprocess
import unittest

from dare_wearables.lower_back.annotation.web import PAGE

NODE = os.environ.get('DARE_ANNOTATION_NODE') or shutil.which('node')


class AnnotationClientTests(unittest.TestCase):
    @unittest.skipUnless(NODE, 'Optional Node.js client interaction tests; app requires only Python/browser.')
    def test_boundary_drag_add_confirm_save_undo_and_time_fields(self):
        harness = r'''
const vm=require('node:vm'), assert=require('node:assert/strict'), crypto=require('node:crypto');
let source='';process.stdin.setEncoding('utf8');process.stdin.on('data',part=>source+=part);
process.stdin.on('end',async()=>{
 try{
 class Element{
  constructor(){this.value='';this.style={};this.dataset={};this.children=[];this.events={};this.clientWidth=1200;this.classList={toggle(){}};this.disabled=false;this._captured=false;}
  append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}
  addEventListener(type,fn){(this.events[type]??=[]).push(fn);}setAttribute(){}focus(){this.focused=true;}
  getBoundingClientRect(){return{left:0,top:0,width:1200,height:260};}
  getContext(){return new Proxy({}, {get:()=>()=>{},set:()=>true});}
  setPointerCapture(){this._captured=true;}hasPointerCapture(){return this._captured;}releasePointerCapture(){this._captured=false;}
  click(){return Promise.all((this.events.click||[]).map(fn=>fn()));}
 }
 const elements=new Map(),el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 el('context').value='20';
 const document={getElementById:el,createElement:()=>new Element(),createTextNode:text=>text};
 const window={devicePixelRatio:1,events:{},addEventListener(type,fn){this.events[type]=fn;}};
 const startTime='2024-01-01T08:00:00.123456789',anchor=Date.parse(startTime.slice(0,23)+'Z');
 const boundary=seconds=>({sample:Math.round(seconds*100),seconds,time:new Date(anchor+seconds*1000).toISOString().replace('Z','')});
 const initial=(id,a,b)=>({id,label:'nonwear',status:'pending',start_sample:a*100,stop_sample:b*100,notes:'',start:boundary(a),end:boundary(b)});
 const network={revision:0,rows:[initial('diary_1_1',100,340),initial('diary_1_2',690,840)],saved:[],failSave:false};
 const fetch=async(url,options={})=>{
  const body=options.body?JSON.parse(options.body):null;
  let result;
  if(url==='/api/session')result={status:'ready',token:'synthetic',subject:'900001',visit:'T0',participants:['900001'],revision:network.revision,reviewer:'',default_reviewer:'',start_time:startTime,duration:1200,sample_rate_hz:100,n_samples:120000,rows:network.rows,diary:[{start:100,end:340},{start:690,end:840}],issues:[],output_folder:'synthetic-output'};
  else if(url.startsWith('/api/view'))result={mode:'raw',samples:5,bin_seconds:.01,x:[100,200,400,800,1100],lo:Array.from({length:5},()=>[0,0,1,.1,.2,.3]),hi:null,breaks:[],gaps:[]};
  else if(url==='/api/snap'){
   if(body.end<=body.start)return{ok:false,json:async()=>({error:'End must follow start.'})};
   result={start:boundary(Math.round(body.start*100)/100),end:boundary(Math.round(body.end*100)/100)};
  }else if(url==='/api/save'){
   if(network.failSave)return{ok:false,json:async()=>({error:'Invented save failure'})};
   network.saved.push(body);network.revision++;network.rows=body.rows.map(row=>({...row,start:boundary(row.start_sample/100),end:boundary(row.stop_sample/100)}));result={revision:network.revision,rows:network.rows};
  }else throw new Error('Unexpected route '+url);
  return{ok:true,json:async()=>JSON.parse(JSON.stringify(result))};
 };
 const storage=new Map(),context=vm.createContext({document,window,fetch,crypto,Date,console,setTimeout,clearTimeout,AbortController,URLSearchParams,localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)}});
 const run=script=>vm.runInContext(script,context);
 const tick=()=>new Promise(resolve=>setTimeout(resolve,5));
 run(source);await new Promise(resolve=>setTimeout(resolve,150));
 assert.equal(run('state.rows.length'),2);assert.equal(el('start').value,'2024-01-01T08:01:40.123');
 assert.equal(run('clock(0)'),'08:00:00'); // Stored clock survives a different browser/system timezone.
 // Exercise the actual pointer handlers, not a separate copy of the drag calculation.
 run(`{const canvas=$('acc'),g=geometry(canvas),x=t=>g.l+t/1200*(g.r-g.l);
 canvas.onpointerdown({button:0,clientX:x(100),clientY:100,pointerId:1,preventDefault(){}});
 canvas.onpointermove({clientX:x(120),clientY:100,pointerId:1});canvas.onpointerup({pointerId:1});}`);
 await tick();assert.equal(run('selected().start_sample'),12000);assert.equal(run('selected().status'),'pending');
 await el('confirm').click();assert.match(el('message').textContent,/reviewer name/);assert.equal(run('selected().status'),'pending');
 el('reviewer').value='Synthetic reviewer';await el('confirm').click();assert.equal(run('selected().status'),'reviewed');
 await el('save').click();assert.equal(network.saved.length,1);assert.equal(network.saved[0].revision,0);
 assert.equal(network.saved[0].rows[0].start_sample,12000);assert.equal(run('state.dirty'),false);
 assert.equal(Object.keys(network.saved[0].rows[0]).length,6);
 // Clicking a handle without moving it must not invalidate a confirmed label.
 run(`{const canvas=$('acc'),g=geometry(canvas),x=g.l+120/1200*(g.r-g.l);
 canvas.onpointerdown({button:0,clientX:x,clientY:100,pointerId:3,preventDefault(){}});
 canvas.onpointerup({pointerId:3});}`);
 await tick();assert.equal(run('selected().status'),'reviewed');assert.equal(run('state.dirty'),false);
 // Editing typed times and clicking Confirm automatically applies them before confirming.
 el('end').value='2024-01-01T08:05:00.123';el('end').oninput();
 await el('confirm').click();assert.equal(run('selected().stop_sample'),30000);assert.equal(run('selected().status'),'reviewed');
 // Saving and selecting another interval also apply typed boundaries rather than silently dropping them.
 el('start').value='2024-01-01T08:02:10.123';el('start').oninput();await el('save').click();
 assert.equal(network.saved.at(-1).rows[0].start_sample,13000);
 await el('undo').click();assert.equal(run('selected().start_sample'),12000);assert.equal(run('state.dirty'),true);
 // Add a missed removal using the same drag path as the UI.
 await el('add-nonwear').click();
 run(`{const canvas=$('gyro'),g=geometry(canvas),x=t=>g.l+t/1200*(g.r-g.l);
 canvas.onpointerdown({button:0,clientX:x(450),clientY:100,pointerId:2,preventDefault(){}});
 canvas.onpointermove({clientX:x(470),clientY:100,pointerId:2});canvas.onpointerup({pointerId:2});}`);
 await tick();assert.equal(run('state.rows.length'),3);assert.equal(run('selected().start_sample'),45000);assert.equal(run('selected().stop_sample'),47000);
 await el('confirm').click();assert.equal(run('selected().status'),'reviewed');
 await el('reject').click();assert.equal(run('selected().status'),'rejected');
 await el('undo').click();assert.equal(run('selected().status'),'reviewed');
 // Failed writes leave the unsaved flag and current edits intact.
 network.failSave=true;await el('save').click();assert.equal(run('state.dirty'),true);assert.match(el('message').textContent,/Invented save failure/);
 network.failSave=false;await el('save').click();assert.equal(run('state.dirty'),false);
 // Independent vertical zoom and time navigation do not alter annotations.
 const before=run('JSON.stringify(payloadRows())');
 el('acc').events.wheel[0]({deltaY:-1,shiftKey:true,clientX:300,clientY:110,preventDefault(){}});
 assert.notEqual(run('state.y[0]'),null);assert.equal(run('state.y[1]'),null);
 await el('zoom-in').click();assert.equal(run('JSON.stringify(payloadRows())'),before);
 clearTimeout(run('fetchTimer'));
 console.log('Client event checks passed: pointer drag, add, confirm, numeric edit, save failure, undo, and independent zoom.');
 }catch(error){console.error(error);process.exitCode=1;}
});
'''
        source = PAGE.split('<script>')[1].split('</script>')[0]
        # Node is optional for development tests, never a runtime dependency of the app.
        result = subprocess.run([NODE,'-e',harness],input=source,text=True,capture_output=True,timeout=30,
                                env=os.environ|{'TZ':'America/New_York'})
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    unittest.main()
