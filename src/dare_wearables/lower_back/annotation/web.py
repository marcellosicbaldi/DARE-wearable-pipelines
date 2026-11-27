"""Self-contained local UI; no CDN, JavaScript packages, or external requests."""

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nonwear review</title>
<style>
:root{--ink:#172b36;--muted:#657985;--line:#dce5e8;--teal:#087f8c;--bg:#f4f7f8;--amber:#d89313;--green:#197151}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select,textarea{font:inherit}button{border:1px solid #cbd8dd;background:white;color:var(--ink);padding:8px 11px;border-radius:7px;cursor:pointer;white-space:nowrap}
button:hover{background:#edf5f6;border-color:#9fb9c2}button:disabled{opacity:.45;cursor:default}button.primary{background:var(--teal);color:white;border-color:var(--teal)}
button.active{background:#dff0f1;border-color:var(--teal)}button.danger{color:#a33935}input,select,textarea{border:1px solid #cbd8dd;border-radius:6px;padding:7px;background:white;color:var(--ink)}
input:focus,select:focus,textarea:focus,button:focus-visible{outline:2px solid #82c8ce;outline-offset:2px}label{color:var(--muted);font-size:12px}h1{font-size:23px;margin:0 0 5px}p{margin:0;line-height:1.5}
header{display:flex;align-items:center;justify-content:space-between;padding:20px 26px 15px;border-bottom:1px solid var(--line);background:white;gap:15px}.subtitle{color:var(--muted);font-size:13px}.header-actions{display:flex;align-items:center;gap:12px}
.pill{font-size:11px;background:#eaf3f2;color:#31625b;border-radius:20px;padding:5px 9px}.layout{display:grid;grid-template-columns:280px minmax(0,1fr);gap:18px;padding:18px 24px}
aside,.card{background:white;border:1px solid var(--line);border-radius:10px}aside{padding:16px;align-self:start;max-height:calc(100vh - 130px);overflow:auto}.side-title{font-weight:650;margin:0 0 9px}.recording{display:flex;gap:7px}.recording select{min-width:0;flex:1}
.reviewer{display:block;width:100%;margin:5px 0 18px}.section-line{display:flex;align-items:center;justify-content:space-between;margin:20px 0 10px}.small{font-size:12px;color:var(--muted);line-height:1.5}
#episodes{display:flex;flex-direction:column;gap:7px}.episode{width:100%;white-space:normal;text-align:left;padding:10px 11px;display:block}.episode.selected{border-color:var(--teal);background:#eef8f8;box-shadow:inset 3px 0 var(--teal)}
.episode strong{display:block;font-size:13px;margin-bottom:4px}.episode span{display:block;font-size:12px;color:var(--muted)}.episode .state{font-size:11px;margin-top:5px}.state.reviewed{color:var(--green)}.state.rejected{color:#a33935}
.add-row{display:flex;gap:6px;margin-top:12px}.add-row button{flex:1;font-size:12px}.legend{margin:18px 0 0;padding-top:14px;border-top:1px solid var(--line)}.legend div{margin:7px 0;font-size:12px;color:var(--muted)}.swatch{display:inline-block;width:17px;height:9px;margin-right:7px;border-radius:2px}
main{min-width:0}.toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.toolbar button{font-size:12px;padding:7px 10px}.toolbar .spacer{flex:1}.toolbar label{display:flex;align-items:center;gap:5px}.toolbar input{width:55px;padding:5px}
.card{overflow:hidden;margin-bottom:10px}.plot-title{display:flex;justify-content:space-between;align-items:center;padding:11px 16px 0;font-size:13px;font-weight:600}.channels{display:flex;gap:12px}.channels label{display:flex;align-items:center;gap:3px;font-weight:500}.channels input{accent-color:var(--teal);margin:0}
canvas{display:block;width:100%;height:clamp(195px,26vh,290px);touch-action:none;cursor:crosshair}.editor{padding:13px 16px;display:flex;align-items:end;gap:10px;flex-wrap:wrap}.editor label{display:flex;flex-direction:column;gap:5px}.editor input[type=datetime-local]{font-size:12px;width:213px}.editor select{width:105px}.editor .notes{flex:1;min-width:130px}.editor .notes input{width:100%;font-size:12px}.editor .actions{display:flex;gap:6px}.editor .actions button{font-size:12px}
#plot-status{font-size:12px;color:var(--muted);padding:0 2px 11px;min-height:27px}#message{margin:0 0 11px;padding:10px 13px;border:1px solid #bbdadd;border-radius:7px;background:#edf8f8;line-height:1.5;white-space:pre-wrap}#message.error{border-color:#e8b6b1;background:#fff1ef;color:#8c302a}#message:empty{display:none}
#help{font-size:12px;color:var(--muted);line-height:1.7;margin-top:8px}.saved-path{font-size:11px;color:var(--muted);overflow-wrap:anywhere;margin-top:12px}.busy{opacity:.55;pointer-events:none}#save-state{font-size:12px;color:var(--muted)}#selected-title{font-size:13px;font-weight:600;padding:12px 16px 0}#empty{display:none;text-align:center;padding:25px;color:var(--muted)}
@media(max-width:900px){.layout{grid-template-columns:220px minmax(0,1fr);padding:12px;gap:10px}header{padding:14px}.editor input[type=datetime-local]{width:195px}aside{padding:12px}}
@media(max-width:650px){.layout{display:block}aside{max-height:330px;margin-bottom:12px}header{align-items:start}h1{font-size:20px}.header-actions{flex-wrap:wrap;justify-content:end}.pill{display:none}canvas{height:220px}}
</style></head><body>
<header><div><h1>Nonwear review</h1><p class="subtitle">Raw signals, diary reports, and your reviewed boundaries</p></div>
<div class="header-actions"><span class="pill">Local · no uploads</span><span id="save-state">Loading…</span><button id="undo" disabled>Undo</button><button id="save" class="primary" disabled>Save</button></div></header>
<div class="layout"><aside>
<div class="side-title">Recording</div><div class="recording"><select id="subject" aria-label="Participant"></select><button id="load">Load</button></div><p id="record-info" class="small" style="margin-top:8px"></p>
<label for="reviewer" style="display:block;margin-top:15px">Reviewer</label><input id="reviewer" class="reviewer" placeholder="Your name" autocomplete="name" maxlength="120">
<div class="section-line"><strong>Intervals</strong><span id="count" class="small"></span></div><div id="episodes"></div><div id="empty">No diary episodes. Add an interval to begin.</div>
<div class="add-row"><button id="add-nonwear">+ Nonwear</button><button id="add-wear">+ Wear</button></div>
<div class="legend"><div><i class="swatch" style="background:#f6dfa9"></i>Original diary report</div><div><i class="swatch" style="background:#a7d9df"></i>Your nonwear interval</div><div><i class="swatch" style="background:#c4e6d0"></i>Your wear interval</div><div><i class="swatch" style="background:repeating-linear-gradient(45deg,#eee,#eee 3px,#aaa 4px,#eee 5px)"></i>Missing / time-invalid</div></div>
<button id="export" style="width:100%;margin-top:12px" disabled>Export confirmed labels (CSV)</button><p id="output" class="saved-path"></p>
</aside><main><div id="message" role="status"></div>
<div id="workarea"><div class="toolbar"><button id="previous" title="Previous interval">← Previous</button><button id="next" title="Next interval">Next →</button><button id="focus">Focus interval</button>
<label>Context <input id="context" type="number" min="1" max="720" value="20" aria-label="Context minutes"> min</label><span class="spacer"></span>
<button id="full">Whole week</button><button id="pan-left" title="Pan left">←</button><button id="zoom-in" title="Zoom in">Zoom +</button><button id="zoom-out" title="Zoom out">Zoom −</button><button id="pan-right" title="Pan right">→</button><button id="range-zoom">Drag to zoom</button><button id="fit-y">Fit Y</button></div>
<div class="card"><div class="plot-title"><span>Raw acceleration <span class="small">· g</span></span><div id="acc-channels" class="channels"></div></div><canvas id="acc" aria-label="Raw acceleration plot. Drag teal boundaries to edit; drag elsewhere to pan."></canvas></div>
<div class="card"><div class="plot-title"><span>Raw gyroscope <span class="small">· °/s</span></span><div id="gyro-channels" class="channels"></div></div><canvas id="gyro" aria-label="Raw gyroscope plot. Drag teal boundaries to edit; drag elsewhere to pan."></canvas></div>
<div id="plot-status"></div><div class="card"><div id="selected-title">Select an interval</div><div class="editor">
<label>Label<select id="label"><option value="nonwear">Nonwear</option><option value="wear">Wear</option></select></label>
<label>Start<input id="start" type="datetime-local" step="0.001"></label><label>End<input id="end" type="datetime-local" step="0.001"></label><button id="apply">Apply times</button>
<label class="notes">Notes (optional)<input id="notes" maxlength="4000" placeholder="Optional comment"></label><div class="actions"><button id="reject" class="danger">Reject</button><button id="confirm" class="primary">Confirm interval</button></div></div></div>
<p id="help">Drag either teal edge to adjust a boundary. Drag elsewhere to pan; scroll to zoom in time. <strong>Shift + scroll</strong> zooms the vertical scale of one plot; double-click resets its scale. Confirm each interval, then Save. Zoom in to see every 100 Hz sample. All times use the recording’s stored local clock.</p>
</div></main></div>
<script>
'use strict';
const $ = id => document.getElementById(id);
const colors=['#c75489','#008f7a','#4776ad'];
const channels=['ax','ay','az','gx','gy','gz'];
const state={session:null,rows:[],selected:null,dirty:false,undo:[],view:[0,1],data:null,y:[null,null],visible:[true,true,true,true,true,true],mode:'pan',drag:null,saving:false,token:'',fetchId:0};
let aborter=null,fetchTimer=null,pollTimer=null;
const clone = v => JSON.parse(JSON.stringify(v));
function message(text='',error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function fail(error){message(error.message||String(error),true);}
async function api(path,body){
 const result=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','X-Annotation-Token':state.token}:{},body:body?JSON.stringify(body):undefined});
 const value=await result.json();if(!result.ok)throw new Error(value.error||'Request failed');return value;
}
function bind(id,fn){$(id).addEventListener('click',()=>Promise.resolve().then(fn).catch(fail));}
function selected(){return state.rows.find(row=>row.id===state.selected);}
function payloadRows(){return state.rows.map(row=>Object.fromEntries(['id','label','status','start_sample','stop_sample','notes'].map(key=>[key,row[key]])));}
function checkpoint(){state.undo.push({rows:clone(state.rows),selected:state.selected});if(state.undo.length>80)state.undo.shift();}
function changed(){state.dirty=true;updateButtons();}
function anchor(){return Date.parse(state.session.start_time.slice(0,23)+'Z');}
function dateAt(seconds){return new Date(anchor()+seconds*1000);}
function clock(seconds,date=false){
 const value=dateAt(seconds),iso=value.toISOString();
 return date?iso.slice(0,10)+' '+iso.slice(11,19):iso.slice(11,19);
}
function inputTime(seconds){return dateAt(seconds).toISOString().slice(0,23);}
function inputSeconds(value){const parsed=Date.parse(value+'Z');if(!Number.isFinite(parsed))throw new Error('Enter a valid start and end date/time.');return (parsed-anchor())/1000;}
function updateButtons(){
 const ready=state.session?.status==='ready'&&!state.saving;
 $('save').disabled=!ready||!state.dirty;$('undo').disabled=!ready||!state.undo.length;
 $('save-state').textContent=state.saving?'Saving…':state.dirty?'Unsaved changes':ready?'Saved / no changes':'Loading…';
 $('workarea').classList.toggle('busy',!ready);$('load').disabled=state.session?.status==='loading'||state.saving;
 $('export').disabled=!ready;$('add-nonwear').disabled=!ready;$('add-wear').disabled=!ready;
 const i=state.rows.findIndex(row=>row.id===state.selected);
 $('previous').disabled=i<=0;$('next').disabled=i<0||i>=state.rows.length-1;
 for(const id of ['label','start','end','apply','notes','reject','confirm','focus'])$(id).disabled=!ready||!selected();
}
function renderList(){
 const holder=$('episodes');holder.replaceChildren();
 for(const row of state.rows){
  const button=document.createElement('button');button.className='episode'+(row.id===state.selected?' selected':'');button.dataset.annotationId=row.id;
  const title=document.createElement('strong');title.textContent=clock(row.start.seconds,true).slice(0,10)+' · '+(row.label==='nonwear'?'Nonwear':'Wear');
  const time=document.createElement('span');time.textContent=clock(row.start.seconds)+' – '+clock(row.end.seconds)+' · '+((row.end.seconds-row.start.seconds)/60).toFixed(1)+' min';
  const status=document.createElement('span');status.className='state '+row.status;status.textContent=(row.status==='pending'?'Not confirmed':row.status==='reviewed'?'Confirmed':'Rejected')+(row.id.startsWith('diary_')?' · diary':' · added');
  button.append(title,time,status);button.onclick=()=>selectRow(row.id).catch(fail);holder.append(button);
 }
 $('empty').style.display=state.rows.length?'none':'block';
 $('count').textContent=state.rows.filter(row=>row.status==='reviewed').length+' / '+state.rows.length+' confirmed';
 updateEditor();updateButtons();
}
function updateEditor(){
 const row=selected();$('selected-title').textContent=row?(row.id.startsWith('diary_')?'Diary interval':'Added interval')+' · '+(row.status==='reviewed'?'confirmed':row.status==='rejected'?'rejected':'not yet confirmed'):'Select or add an interval';
 for(const id of ['start','end','notes'])$(id).value=row?(id==='notes'?row.notes:inputTime(row[id].seconds)):'';
 $('label').value=row?.label||'nonwear';
}
function editorTimesChanged(){const row=selected();return row&&($('start').value!==inputTime(row.start.seconds)||$('end').value!==inputTime(row.end.seconds));}
async function applyEditor(){if(editorTimesChanged())await moveBoundary(inputSeconds($('start').value),inputSeconds($('end').value));}
async function selectRow(id,focus=true){if(state.saving||state.session?.status!=='ready')return;await applyEditor();state.selected=id;renderList();if(focus)focusRow();else draw();}
function setView(left,right,resetY=false){
 const duration=state.session.duration,span=Math.max(.5,Math.min(duration,right-left));
 left=Math.max(0,Math.min(duration-span,left));state.view=[left,left+span];if(resetY)state.y=[null,null];
 draw();requestView();
}
function focusRow(){const row=selected();if(!row)return;const context=Math.max(1,Math.min(720,Number($('context').value)||20))*60;setView(row.start.seconds-context,row.end.seconds+context,true);}
function requestView(){clearTimeout(fetchTimer);fetchTimer=setTimeout(fetchView,100);}
async function fetchView(){
 if(state.session?.status!=='ready')return;
 const requestId=++state.fetchId;aborter?.abort();aborter=new AbortController();
 const [left,right]=state.view;
 try{
  $('plot-status').textContent='Loading signal view…';
  const response=await fetch('/api/view?'+new URLSearchParams({subject:state.session.subject,left,right,width:Math.round($('acc').clientWidth)}),{signal:aborter.signal});
  const data=await response.json();if(!response.ok)throw new Error(data.error);if(requestId!==state.fetchId)return;
  state.data=data;draw();
  $('plot-status').textContent=data.mode==='raw'?`Every original sample · ${state.session.sample_rate_hz} Hz · ${data.samples.toLocaleString()} samples/channel`:data.mode==='empty'?'No observed samples in this interval':`Min/max display · bins ≈ ${data.bin_seconds.toFixed(2)} s · all ${data.samples.toLocaleString()} raw samples contribute. Zoom in for individual samples.`;
 }catch(error){if(error.name!=='AbortError')fail(error);}
}
async function moveBoundary(start,end){
 if(state.saving)return;
 const row=selected();if(!row)return;
 const snap=await api('/api/snap',{subject:state.session.subject,start,end});
 if(snap.end.sample<=snap.start.sample)throw new Error('End must follow start by at least one sample.');
 if(snap.start.sample===row.start_sample&&snap.end.sample===row.stop_sample){updateEditor();return;}
 checkpoint();row.start=snap.start;row.end=snap.end;row.start_sample=snap.start.sample;row.stop_sample=snap.end.sample;row.status='pending';changed();renderList();draw();message('Boundary updated. Confirm this interval when you are satisfied.');
}
async function addInterval(label,start,end){
 await applyEditor();
 const snap=await api('/api/snap',{subject:state.session.subject,start,end});
 if(snap.end.sample<=snap.start.sample)throw new Error('Drag a longer interval.');
 checkpoint();const row={id:'manual_'+crypto.randomUUID(),label,status:'pending',start_sample:snap.start.sample,stop_sample:snap.end.sample,notes:'',...snap};
 state.rows.push(row);state.rows.sort((a,b)=>a.start_sample-b.start_sample);state.selected=row.id;changed();renderList();draw();message('New interval added. Adjust it if needed, then Confirm.');
}
async function save(){
 await applyEditor();
 if(!state.dirty)return;state.saving=true;updateButtons();
 try{
  const result=await api('/api/save',{subject:state.session.subject,rows:payloadRows(),reviewer:$('reviewer').value,revision:state.session.revision});
  state.session.revision=result.revision;state.rows=result.rows;state.dirty=false;renderList();message(result.warning||'Saved. Only confirmed intervals are included in the CSV export.',!!result.warning);
 }finally{state.saving=false;updateButtons();}
}
async function poll(){
 const result=await api('/api/session');state.token=result.token;
 const first=!state.session;state.session=result;
 if(first){
  for(const id of result.participants){const option=document.createElement('option');option.value=id;option.textContent=id;$('subject').append(option);}
 }
 $('subject').value=result.subject||result.participants[0]||'';
 if(result.status==='loading'){
  message(`Loading ${result.subject}… The first visit decodes and caches its raw recording; this can take a minute. Later visits reuse the cache.`);updateButtons();pollTimer=setTimeout(()=>poll().catch(fail),1000);return;
 }
 if(result.status==='error'){message(result.error,true);updateButtons();return;}
 if(result.status!=='ready'){message('Choose a participant and Load.');updateButtons();return;}
 state.rows=result.rows;state.selected=state.rows.find(row=>row.status==='pending')?.id||state.rows[0]?.id||null;state.dirty=false;state.undo=[];state.data=null;
 $('reviewer').value=result.reviewer||localStorage.getItem('nonwear-reviewer')||result.default_reviewer||'';
 $('record-info').textContent=result.subject+' / '+result.visit+' · '+result.sample_rate_hz+' Hz · '+(result.duration/3600).toFixed(1)+' h';
 $('output').textContent='Saved locally in '+result.output_folder;
 message(result.issues.length?'Diary import notes:\n'+result.issues.join('\n'):'Orange shows the unchanged diary report. Teal edges are your editable proposal.');
 renderList();if(state.selected)focusRow();else setView(0,result.duration,true);
}
bind('load',async()=>{if(state.dirty)await save();aborter?.abort();await api('/api/open',{subject:$('subject').value});await poll();});
bind('save',save);
bind('undo',()=>{const item=state.undo.pop();if(!item)return;state.rows=item.rows;state.selected=item.selected;changed();renderList();draw();message('Undone. Save to keep this version.');});
bind('previous',()=>{const i=state.rows.findIndex(r=>r.id===state.selected);if(i>0)selectRow(state.rows[i-1].id);});
bind('next',()=>{const i=state.rows.findIndex(r=>r.id===state.selected);if(i>=0&&i<state.rows.length-1)selectRow(state.rows[i+1].id);});
bind('focus',focusRow);bind('full',()=>setView(0,state.session.duration,true));
bind('zoom-in',()=>zoom(.5));bind('zoom-out',()=>zoom(2));
bind('pan-left',()=>pan(-.5));bind('pan-right',()=>pan(.5));
function zoom(factor,at=.5){const [a,b]=state.view,span=(b-a)*factor,center=a+(b-a)*at;setView(center-span*at,center+span*(1-at));}
function pan(fraction){const [a,b]=state.view,delta=(b-a)*fraction;setView(a+delta,b+delta);}
bind('fit-y',()=>{state.y=[null,null];draw();});
function mode(value){state.mode=value;$('range-zoom').classList.toggle('active',value==='zoom');$('add-nonwear').classList.toggle('active',value==='new-nonwear');$('add-wear').classList.toggle('active',value==='new-wear');}
bind('range-zoom',()=>{mode(state.mode==='zoom'?'pan':'zoom');message(state.mode==='zoom'?'Drag across either plot to zoom into a time range.':'Pan mode.');});
for(const label of ['nonwear','wear'])bind('add-'+label,()=>{mode('new-'+label);message('Drag across either plot to add a '+label+' interval. Press Escape to cancel.');});
bind('apply',()=>moveBoundary(inputSeconds($('start').value),inputSeconds($('end').value)));
bind('confirm',async()=>{
 if(!selected())return;
 if(!$('reviewer').value.trim()){$('reviewer').focus();throw new Error('Enter your reviewer name before confirming an interval.');}
 await applyEditor();const row=selected();
 if(state.rows.some(other=>other.id!==row.id&&other.status==='reviewed'&&other.start_sample<row.stop_sample&&other.stop_sample>row.start_sample))throw new Error('This overlaps another confirmed interval. Adjust or reject the overlap first.');
 checkpoint();row.status='reviewed';changed();renderList();draw();message('Interval confirmed. Save to keep your labels.');
});
bind('reject',()=>{const row=selected();if(!row)return;checkpoint();row.status='rejected';changed();renderList();draw();message('Interval rejected; the original diary entry remains in the audit.');});
$('label').onchange=()=>{const row=selected();if(!row)return;checkpoint();row.label=$('label').value;row.status='pending';changed();renderList();draw();};
$('notes').onfocus=()=>{if(selected())checkpoint();};$('notes').oninput=()=>{if(selected()){selected().notes=$('notes').value;changed();}};
$('reviewer').oninput=()=>{localStorage.setItem('nonwear-reviewer',$('reviewer').value);if(state.session?.status==='ready')changed();};
$('context').onchange=focusRow;
for(const id of ['start','end'])$(id).oninput=changed;
bind('export',async()=>{if(state.dirty)await save();const link=document.createElement('a');link.href='/api/export?'+new URLSearchParams({subject:state.session.subject});link.download='reviewed_intervals.csv';link.click();});
window.addEventListener('beforeunload',event=>{if(state.dirty||state.saving){event.preventDefault();event.returnValue='';}});
window.addEventListener('keydown',event=>{if(event.key==='Escape'){state.drag=null;mode('pan');draw();message('Pan mode.');}if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='s'){event.preventDefault();save().catch(fail);}});
for(let i=0;i<6;i++){
 const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.setAttribute('aria-label','Show '+channels[i]);input.onchange=()=>{state.visible[i]=input.checked;state.y[Math.floor(i/3)]=null;draw();};label.style.color=colors[i%3];label.append(input,document.createTextNode(channels[i]));$(i<3?'acc-channels':'gyro-channels').append(label);
}
function geometry(canvas){const rect=canvas.getBoundingClientRect();return{w:rect.width,h:rect.height,l:65,r:rect.width-48,t:22,b:rect.height-38};}
function timeX(x,g){return state.view[0]+(x-g.l)/(g.r-g.l)*(state.view[1]-state.view[0]);}
function draw(){if(!state.session||state.session.status!=='ready')return;for(let panel=0;panel<2;panel++)drawPanel($(panel?'gyro':'acc'),panel);}
function drawPanel(canvas,panel){
 const g=geometry(canvas),ratio=window.devicePixelRatio||1;canvas.width=Math.round(g.w*ratio);canvas.height=Math.round(g.h*ratio);
 const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,g.w,g.h);ctx.font='11px -apple-system, sans-serif';
 const [left,right]=state.view,X=time=>g.l+(time-left)/(right-left)*(g.r-g.l);
 let minimum=Infinity,maximum=-Infinity,data=state.data;
 if(data)for(let i=0;i<data.x.length;i++)if(data.x[i]>=left&&data.x[i]<=right)for(let c=panel*3;c<panel*3+3;c++)if(state.visible[c]){minimum=Math.min(minimum,data.lo[i][c]);maximum=Math.max(maximum,data.hi?data.hi[i][c]:data.lo[i][c]);}
 if(!Number.isFinite(minimum)){minimum=-1;maximum=1;}
 const pad=Math.max((maximum-minimum)*.09,panel?.005:.005);
 const range=state.y[panel]||[minimum-pad,maximum+pad],Y=value=>g.b-(value-range[0])/(range[1]-range[0])*(g.b-g.t);
 canvas._range=range;
 ctx.save();ctx.beginPath();ctx.rect(g.l,g.t,g.r-g.l,g.b-g.t);ctx.clip();
 ctx.fillStyle='#fff';ctx.fillRect(g.l,g.t,g.r-g.l,g.b-g.t);
 for(const item of state.session.diary){ctx.fillStyle='rgba(230,159,0,.16)';ctx.fillRect(X(item.start),g.t,X(item.end)-X(item.start),g.b-g.t);ctx.strokeStyle='rgba(206,143,15,.65)';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(X(item.start),g.t);ctx.lineTo(X(item.start),g.b);ctx.moveTo(X(item.end),g.t);ctx.lineTo(X(item.end),g.b);ctx.stroke();ctx.setLineDash([]);}
 for(const row of state.rows){
  if(row.status==='rejected')continue;const active=row.id===state.selected;
  let a=row.start.seconds,b=row.end.seconds;
  if(active&&state.drag?.type==='boundary'){if(state.drag.edge==='start')a=state.drag.current;else b=state.drag.current;}
  ctx.fillStyle=row.label==='wear'?'rgba(47,151,91,.15)':active?'rgba(8,127,140,.17)':'rgba(8,127,140,.07)';
  ctx.fillRect(X(a),g.t,X(b)-X(a),g.b-g.t);
 }
 if(data)for(const [a,b] of data.gaps){const x1=X(a),x2=X(b);ctx.fillStyle='#e8e8e8';ctx.fillRect(x1,g.t,Math.max(2,x2-x1),g.b-g.t);ctx.save();ctx.beginPath();ctx.rect(x1,g.t,Math.max(2,x2-x1),g.b-g.t);ctx.clip();ctx.strokeStyle='#aaa';for(let x=x1-(g.b-g.t);x<x2+10;x+=8){ctx.beginPath();ctx.moveTo(x,g.b);ctx.lineTo(x+g.b-g.t,g.t);ctx.stroke();}ctx.restore();}
 ctx.strokeStyle='#e5ecee';ctx.lineWidth=1;for(let i=0;i<=4;i++){const y=g.t+i*(g.b-g.t)/4;ctx.beginPath();ctx.moveTo(g.l,y);ctx.lineTo(g.r,y);ctx.stroke();}
 for(let i=0;i<=5;i++){const x=g.l+i*(g.r-g.l)/5;ctx.beginPath();ctx.moveTo(x,g.t);ctx.lineTo(x,g.b);ctx.stroke();}
 if(data){
  const breaks=new Set(data.breaks);for(let c=panel*3;c<panel*3+3;c++)if(state.visible[c]){
   ctx.strokeStyle=colors[c%3];ctx.lineWidth=.8;ctx.beginPath();let pen=false;
   for(let i=0;i<data.x.length;i++){
    const x=X(data.x[i]);if(data.x[i]<left||data.x[i]>right){pen=false;continue;}
    if(data.mode==='raw'){if(!pen||breaks.has(i))ctx.moveTo(x,Y(data.lo[i][c]));else ctx.lineTo(x,Y(data.lo[i][c]));pen=true;}
    else{ctx.moveTo(x,Y(data.lo[i][c]));ctx.lineTo(x,Y(data.hi[i][c]));}
   }ctx.stroke();
  }
 }
 if(state.drag&&['zoom','new-nonwear','new-wear'].includes(state.drag.type)){ctx.fillStyle='rgba(8,127,140,.18)';ctx.fillRect(X(Math.min(state.drag.origin,state.drag.current)),g.t,Math.abs(X(state.drag.current)-X(state.drag.origin)),g.b-g.t);}
 ctx.restore();
 ctx.strokeStyle='#d2dfe3';ctx.strokeRect(g.l,g.t,g.r-g.l,g.b-g.t);
 ctx.fillStyle='#637782';ctx.textAlign='right';for(let i=0;i<=4;i++){const value=range[1]-i*(range[1]-range[0])/4;ctx.fillText(Number(value.toPrecision(4)).toString(),g.l-8,g.t+i*(g.b-g.t)/4+4);}
 ctx.textAlign='center';for(let i=0;i<=5;i++){const t=left+i*(right-left)/5;let text=clock(t);if(right-left>12*3600){const date=dateAt(t).toISOString();text=date.slice(5,10)+' '+date.slice(11,16);}ctx.fillText(text,g.l+i*(g.r-g.l)/5,g.b+20);}
 const row=selected();if(row){for(const edge of ['start','end']){let t=row[edge].seconds;if(state.drag?.type==='boundary'&&state.drag.edge===edge)t=state.drag.current;const x=X(t);if(x<g.l||x>g.r)continue;ctx.strokeStyle='#087f8c';ctx.fillStyle='#087f8c';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,g.t);ctx.lineTo(x,g.b);ctx.stroke();ctx.beginPath();ctx.moveTo(x-5,g.t-8);ctx.lineTo(x+5,g.t-8);ctx.lineTo(x,g.t);ctx.fill();ctx.textAlign=edge==='start'?'left':'right';ctx.fillText(edge,x+(edge==='start'?7:-7),g.t-9);}}
}
for(let panel=0;panel<2;panel++){
 const canvas=$(panel?'gyro':'acc');
 canvas.onpointerdown=event=>{
  if(state.session?.status!=='ready'||state.saving||event.button!==0)return;
  const g=geometry(canvas),rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
  if(x<g.l||x>g.r||y<g.t-15||y>g.b)return;
  let type=state.mode,edge=null;const row=selected(),t=timeX(x,g);
  if(type==='pan'&&row){const distances=['start','end'].map(key=>[key,Math.abs((row[key].seconds-state.view[0])/(state.view[1]-state.view[0])*(g.r-g.l)+g.l-x)]).sort((a,b)=>a[1]-b[1]);if(distances[0][1]<12){type='boundary';edge=distances[0][0];}}
  state.drag={type,edge,origin:t,current:t,view:[...state.view],pointer:event.pointerId};canvas.setPointerCapture(event.pointerId);event.preventDefault();
 };
 canvas.onpointermove=event=>{
  if(!state.drag)return;const g=geometry(canvas),x=event.clientX-canvas.getBoundingClientRect().left,drag=state.drag;
  const time=drag.view[0]+(x-g.l)/(g.r-g.l)*(drag.view[1]-drag.view[0]);
  drag.current=Math.max(0,Math.min(state.session.duration,time));
  if(drag.type==='boundary'){const row=selected(),period=1/state.session.sample_rate_hz;drag.current=drag.edge==='start'?Math.min(drag.current,row.end.seconds-period):Math.max(drag.current,row.start.seconds+period);}
  if(drag.type==='pan'){const delta=time-drag.origin,span=drag.view[1]-drag.view[0],a=Math.max(0,Math.min(state.session.duration-span,drag.view[0]-delta));state.view=[a,a+span];}
  draw();
 };
 canvas.onpointerup=event=>{
  if(!state.drag)return;const drag=state.drag;state.drag=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
  let task=Promise.resolve();
  if(drag.type==='boundary'){const row=selected();task=moveBoundary(drag.edge==='start'?drag.current:row.start.seconds,drag.edge==='end'?drag.current:row.end.seconds);}
  else if(drag.type==='pan')requestView();
  else if(Math.abs(drag.current-drag.origin)>.05){const a=Math.min(drag.current,drag.origin),b=Math.max(drag.current,drag.origin);if(drag.type==='zoom')setView(a,b,true);else task=addInterval(drag.type==='new-wear'?'wear':'nonwear',a,b);mode('pan');}
  draw();task.catch(fail);
 };
 canvas.onpointercancel=()=>{state.drag=null;draw();};
 canvas.addEventListener('wheel',event=>{
  if(state.session?.status!=='ready')return;event.preventDefault();
  const g=geometry(canvas),rect=canvas.getBoundingClientRect(),factor=event.deltaY>0?1.25:.8;
  if(event.shiftKey){const range=canvas._range,at=Math.max(0,Math.min(1,(g.b-(event.clientY-rect.top))/(g.b-g.t))),center=range[0]+at*(range[1]-range[0]),span=(range[1]-range[0])*factor;state.y[panel]=[center-span*at,center+span*(1-at)];draw();}
  else zoom(factor,Math.max(0,Math.min(1,(event.clientX-rect.left-g.l)/(g.r-g.l))));
 },{passive:false});
 canvas.ondblclick=()=>{state.y[panel]=null;draw();};
}
window.addEventListener('resize',()=>{draw();requestView();});
poll().catch(fail);
</script></body></html>'''
