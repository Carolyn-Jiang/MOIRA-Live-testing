export const MODEL_REGISTRY=Object.freeze([
  {id:'codex',name:'Codex',backbone:'gpt-5.3-codex',color:'#176c54'},
  {id:'finmem',name:'FinMem',backbone:'gpt-4.1-mini',color:'#a3772e'},
  {id:'moira',name:'Moira',backbone:'gpt-4.1-mini',color:'#527c9c'},
  {id:'moira2-top1',name:'Moira 2.0 · Top K = 1',backbone:'gpt-4.1-mini',color:'#7e619b'},
  {id:'moira2-top3',name:'Moira 2.0 · Top K = 3',backbone:'gpt-4.1-mini',color:'#ce7656'},
  {id:'moira2-top5',name:'Moira 2.0 · Top K = 5',backbone:'gpt-4.1-mini',color:'#61734b'}
]);
const IDS=MODEL_REGISTRY.map(model=>model.id),BY_ID=new Map(MODEL_REGISTRY.map(model=>[model.id,model]));
export const PERIOD=Object.freeze({start:'2026-11-02T09:30:00-05:00',last_release:'2027-05-21T15:30:00-04:00',final_settlement:'2027-05-24T10:30:00-04:00'});
export const feedUrl=new URL('state.json',import.meta.url).href;
const TOKEN_FIELDS=['input_tokens','cached_input_tokens','output_tokens','total_tokens','training_tokens','inference_tokens'];
const DISPLAY_TOKEN_FIELDS=['training_tokens','inference_tokens','input_tokens','cached_input_tokens','output_tokens','total_tokens'];
const STATUSES=['scheduled','live','pending','finished'];
const COVERAGE=['complete','partial','unavailable'];
const fail=message=>{throw new Error(message)};
function keys(value,required,optional=[]){if(!value||typeof value!=='object'||Array.isArray(value)||required.some(key=>!Object.hasOwn(value,key))||Object.keys(value).some(key=>![...required,...optional].includes(key)))fail('Invalid public aggregate fields')}
function text(value,max=500){if(typeof value!=='string'||!value.trim()||value.length>max)fail('Invalid public label')}
function count(value){if(!Number.isSafeInteger(value)||value<0)fail('Invalid verified count')}
function number(value){if(typeof value!=='number'||!Number.isFinite(value))fail('Invalid finite metric')}
export function timestamp(value){if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)||!Number.isFinite(Date.parse(value)))fail('Timezone-aware timestamp required');return Date.parse(value)}
function sameModels(rows){if(rows.length!==IDS.length||new Set(rows.map(row=>row.model_id)).size!==IDS.length||rows.some(row=>!BY_ID.has(row.model_id)))fail('All six distinct registered models are required')}
function near(a,b){return Math.abs(a-b)<=1e-8*Math.max(1,Math.abs(a),Math.abs(b))}
export function metrics(points){const returns=points.slice(1).map((point,i)=>point.equity/points[i].equity-1),mean=returns.reduce((a,b)=>a+b,0)/returns.length;const variance=returns.length>1?returns.reduce((sum,value)=>sum+(value-mean)**2,0)/(returns.length-1):0,deviation=Math.sqrt(variance),scale=Math.max(0,...returns.map(Math.abs));let peak=1,drawdown=0;for(const point of points){peak=Math.max(peak,point.equity);drawdown=Math.max(drawdown,1-point.equity/peak)}return {cumulative_return:points.at(-1).equity-1,sharpe_ratio:deviation>Math.max(1e-15,scale*1e-12)?mean/deviation:null,max_drawdown:drawdown}}
export function validateView(view){
  keys(view,['completed_rounds','rows','series']);count(view.completed_rounds);
  if(!Array.isArray(view.rows)||!Array.isArray(view.series))fail('Verified rows and paths required');
  if(view.completed_rounds===0){if(view.rows.length||view.series.length)fail('No returns before verified settlement');return view}
  sameModels(view.rows);sameModels(view.series);
  for(const row of view.rows){keys(row,['model_id','cumulative_return','sharpe_ratio','max_drawdown']);number(row.cumulative_return);number(row.max_drawdown);if(row.cumulative_return<=-1||row.max_drawdown<0||row.max_drawdown>=1)fail('Invalid return or nonnegative drawdown');if(row.sharpe_ratio!==null)number(row.sharpe_ratio)}
  for(const series of view.series){keys(series,['model_id','points']);if(!Array.isArray(series.points)||series.points.length!==view.completed_rounds+1)fail('Each completed interval needs an equity point');for(const [i,point]of series.points.entries()){keys(point,['step','equity']);if(point.step!==i)fail('Consecutive local settlement steps required');number(point.equity);if(point.equity<=0||(i===0&&point.equity!==1))fail('Every view starts at equity one')};const actual=metrics(series.points),row=view.rows.find(row=>row.model_id===series.model_id);for(const field of ['cumulative_return','max_drawdown'])if(!near(row[field],actual[field]))fail('Metrics differ from the published aggregate path');if((actual.sharpe_ratio===null)!==(row.sharpe_ratio===null)||(actual.sharpe_ratio!==null&&!near(actual.sharpe_ratio,row.sharpe_ratio)))fail('Sharpe differs from completed interval returns')}
  return view;
}
export function validateTokens(rows){
  if(!Array.isArray(rows))fail('Token rows required');sameModels(rows);
  for(const row of rows){keys(row,['model_id',...TOKEN_FIELDS],['estimated_cost_usd','cost_coverage','recorded_tokens','token_coverage']);for(const field of TOKEN_FIELDS)if(row[field]!==null)count(row[field]);if(row.cached_input_tokens!==null&&row.input_tokens!==null&&row.cached_input_tokens>row.input_tokens)fail('Cached tokens exceed input');if([row.input_tokens,row.output_tokens,row.total_tokens].every(value=>value!==null)&&row.total_tokens!==row.input_tokens+row.output_tokens)fail('Token total mismatch');if([row.training_tokens,row.inference_tokens,row.total_tokens].every(value=>value!==null)&&row.total_tokens!==row.training_tokens+row.inference_tokens)fail('Token purpose total mismatch');if(Object.hasOwn(row,'estimated_cost_usd')||Object.hasOwn(row,'cost_coverage')){if(!COVERAGE.includes(row.cost_coverage)||!Object.hasOwn(row,'estimated_cost_usd'))fail('Invalid estimated cost coverage');if(row.estimated_cost_usd!==null){number(row.estimated_cost_usd);if(row.estimated_cost_usd<0||row.cost_coverage==='unavailable')fail('Invalid estimated cost')}else if(row.cost_coverage!=='unavailable')fail('Missing estimated cost')}
    if(Object.hasOwn(row,'recorded_tokens')||Object.hasOwn(row,'token_coverage')){
      keys(row.recorded_tokens,TOKEN_FIELDS);keys(row.token_coverage,TOKEN_FIELDS);
      for(const field of TOKEN_FIELDS){const recorded=row.recorded_tokens[field],coverage=row.token_coverage[field];
        if(recorded!==null)count(recorded);if(!COVERAGE.includes(coverage))fail('Invalid token coverage');
        if(coverage==='complete'&&(row[field]===null||recorded!==row[field]))fail('Complete token count differs from the verified total');
        if(coverage==='partial'&&(row[field]!==null||recorded===null))fail('Partial usage needs a recorded subtotal and an unknown full total');
        if(coverage==='unavailable'&&(row[field]!==null||recorded!==null))fail('Unavailable usage cannot contain inferred counts');
      }
    }
  }
  return rows;
}
export function validateSharedTokens(row){keys(row,TOKEN_FIELDS);validateTokens(MODEL_REGISTRY.map(model=>({model_id:model.id,...row})));return row}
export function validateState(state){
  keys(state,['schema','status','timezone','updated_at','period','next_release','all_time','weeks'],['data_status','budget']);
  if(state.schema!=='moira.live-testing.website/1'||!STATUSES.includes(state.status)||state.timezone!=='America/New_York')fail('Wrong public feed protocol');
  keys(state.period,Object.keys(PERIOD));for(const field of Object.keys(PERIOD))if(state.period[field]!==PERIOD[field])fail('Evaluation period changed');
  if(state.updated_at!==null)timestamp(state.updated_at);validateView(state.all_time);if(state.all_time.completed_rounds&&state.updated_at===null)fail('Verified results require a publication timestamp');
  if(state.next_release!==null){keys(state.next_release,['timestamp','session_label','note']);const at=timestamp(state.next_release.timestamp);if(at<timestamp(PERIOD.start)||at>timestamp(PERIOD.last_release))fail('Next release outside evaluation period');text(state.next_release.session_label);text(state.next_release.note,1500)}
  if(!Array.isArray(state.weeks))fail('Weekly views required');const weekIds=new Set();let total=0;
  for(const week of state.weeks){keys(week,['week_id','label','start','end','completed_rounds','rows','series','tokens'],['tokens_coverage','shared_tokens','shared_tokens_coverage']);if(typeof week.week_id!=='string'||!/^\d{4}-W\d{2}$/.test(week.week_id)||weekIds.has(week.week_id))fail('Unique calendar week required');weekIds.add(week.week_id);text(week.label,100);if(timestamp(week.start)>=timestamp(week.end))fail('Invalid reporting week');if(timestamp(week.start)<timestamp('2026-11-02T00:00:00-05:00')||timestamp(week.start)>timestamp(PERIOD.final_settlement))fail('Reporting week outside evaluation period');validateView({completed_rounds:week.completed_rounds,rows:week.rows,series:week.series});validateTokens(week.tokens);if(week.tokens_coverage!==undefined&&!COVERAGE.includes(week.tokens_coverage))fail('Invalid usage coverage');if(week.shared_tokens!==undefined)validateSharedTokens(week.shared_tokens);if(week.shared_tokens_coverage!==undefined&&!COVERAGE.includes(week.shared_tokens_coverage))fail('Invalid shared usage coverage');total+=week.completed_rounds}
  if(total!==state.all_time.completed_rounds)fail('Weekly settlements must cover the all-time record');
  if(state.data_status!==undefined){keys(state.data_status,['published_batches','expected_release_points','last_release_at','coverage']);count(state.data_status.published_batches);if(state.data_status.expected_release_points!==null){count(state.data_status.expected_release_points);if(state.data_status.published_batches>state.data_status.expected_release_points)fail('Published batches exceed schedule')};if(state.data_status.last_release_at!==null)timestamp(state.data_status.last_release_at);if(!['pending',...COVERAGE].includes(state.data_status.coverage))fail('Invalid data coverage')}
  if(state.budget!==undefined){keys(state.budget,['month','spent_usd','reserved_usd','warning_usd','limit_usd','status']);if(!/^\d{4}-\d{2}$/.test(state.budget.month)||!['ok','warning','blocked','unavailable'].includes(state.budget.status)||state.budget.warning_usd!==20||state.budget.limit_usd!==100)fail('Invalid monthly budget');for(const field of ['spent_usd','reserved_usd'])if(state.budget[field]!==null){number(state.budget[field]);if(state.budget[field]<0)fail('Negative measured spend')}}
  return state;
}
export const rankRows=rows=>[...rows].sort((a,b)=>b.cumulative_return-a.cumulative_return||IDS.indexOf(a.model_id)-IDS.indexOf(b.model_id));
export const formatReturn=value=>`${value>0?'+':''}${(value*100).toFixed(2)}%`;
export const formatSharpe=value=>value===null?'N/A':value.toFixed(2);
export const formatDrawdown=value=>value===0?'0%':`${(value*100).toFixed(2)}%`;
export const formatTokens=value=>value===null?'Unavailable':value.toLocaleString('en-US');
export const formatTokenField=(row,field)=>row.token_coverage?.[field]==='partial'?`≥${formatTokens(row.recorded_tokens[field])}`:formatTokens(row[field]);
export const formatEstimatedCost=(value,coverage='complete')=>value===null?'Unavailable':`~$${value.toFixed(value>0&&value<.01?4:2)}${coverage==='partial'?'*':''}`;
export function formatNY(value){return new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value))}

function el(document,tag,textValue,className){const node=document.createElement(tag);if(textValue!==undefined)node.textContent=textValue;if(className)node.className=className;return node}
function modelCell(document,id){const model=BY_ID.get(id),cell=el(document,'td'),label=el(document,'span',undefined,'model-name'),dot=el(document,'span',undefined,'model-dot');dot.style.background=model.color;dot.setAttribute('aria-hidden','true');label.append(dot,el(document,'span',model.name));cell.append(label,el(document,'small',model.backbone,'model-subtitle'));return cell}
function svgEl(document,tag,attributes={},textValue){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value]of Object.entries(attributes))node.setAttribute(key,String(value));if(textValue!==undefined)node.textContent=textValue;return node}
export function renderChart(document,container,legend,view){
  legend.replaceChildren();container.replaceChildren();
  if(!view.completed_rounds){const empty=el(document,'div',undefined,'chart-empty');empty.append(el(document,'span','↗','empty-glyph'),el(document,'p','The first verified return path will appear here.'),el(document,'small','Each line will use completed settlement observations.'));container.append(empty);return}
  const width=Math.max(320,Math.min(1080,container.clientWidth||1080)),height=340,left=65,right=20,top=25,bottom=48,values=view.series.flatMap(series=>series.points.map(point=>(point.equity-1)*100));let lo=Math.min(0,...values),hi=Math.max(0,...values);const pad=Math.max((hi-lo)*.12,.1);lo-=pad;hi+=pad;const x=step=>left+step/view.completed_rounds*(width-left-right),y=value=>top+(hi-value)/(hi-lo)*(height-top-bottom);
  const svg=svgEl(document,'svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-labelledby':'return-path-title return-path-description'});svg.append(svgEl(document,'title',{id:'return-path-title'},'Verified cumulative return paths'),svgEl(document,'desc',{id:'return-path-description'},`Six model variants across ${view.completed_rounds} completed settlement intervals. All paths start at zero percent.`));
  for(let i=0;i<=4;i++){const value=lo+(hi-lo)*i/4,at=y(value);svg.append(svgEl(document,'line',{x1:left,x2:width-right,y1:at,y2:at,stroke:'#e3e9e2','stroke-width':1}),svgEl(document,'text',{x:left-10,y:at+5,'text-anchor':'end',fill:'#667a73','font-size':14},`${value.toFixed(1)}%`))}
  svg.append(svgEl(document,'line',{x1:left,x2:width-right,y1:y(0),y2:y(0),stroke:'#9aa99e','stroke-width':1,'stroke-dasharray':'4 4'}));
  const ticks=[...new Set([0,Math.round(view.completed_rounds/4),Math.round(view.completed_rounds/2),Math.round(view.completed_rounds*3/4),view.completed_rounds])];for(const step of ticks)svg.append(svgEl(document,'text',{x:x(step),y:height-24,'text-anchor':'middle',fill:'#667a73','font-size':14},String(step)));
  svg.append(svgEl(document,'text',{x:(left+width-right)/2,y:height-4,'text-anchor':'middle',fill:'#667a73','font-size':13},'Completed settlement interval'));
  for(const model of MODEL_REGISTRY){const series=view.series.find(row=>row.model_id===model.id),line=svgEl(document,'polyline',{points:series.points.map(point=>`${x(point.step).toFixed(3)},${y((point.equity-1)*100).toFixed(3)}`).join(' '),fill:'none',stroke:model.color,'stroke-width':2.2,'stroke-linejoin':'round','stroke-linecap':'round','vector-effect':'non-scaling-stroke'});line.append(svgEl(document,'title',{},`${model.name}: ${formatReturn(series.points.at(-1).equity-1)}`));svg.append(line);const item=el(document,'li'),key=el(document,'span',undefined,'legend-key');key.style.background=model.color;key.setAttribute('aria-hidden','true');item.append(key,el(document,'span',model.name));legend.append(item)}
  container.append(svg);
}
export function createDashboard(document,{fetcher=globalThis.fetch,now=()=>Date.now()}={}){
  const get=id=>document.getElementById(id);let current=null,viewId='all-time',usageId='',inflight=false;
  function option(value,label){const item=el(document,'option',label);item.value=value;return item}
  function ranking(view){const body=get('rankings');body.replaceChildren();const rows=view.completed_rounds?rankRows(view.rows):MODEL_REGISTRY.map(model=>({model_id:model.id}));for(const [i,row]of rows.entries()){const tr=el(document,'tr');tr.append(el(document,'td',view.completed_rounds?String(i+1):'—','rank-cell'),modelCell(document,row.model_id));for(const field of ['cumulative_return','sharpe_ratio','max_drawdown']){const cell=el(document,'td',view.completed_rounds?(field==='cumulative_return'?formatReturn(row[field]):field==='sharpe_ratio'?formatSharpe(row[field]):formatDrawdown(row[field])):'Awaiting data','number'+(view.completed_rounds?'':' unavailable'));if(view.completed_rounds&&field==='cumulative_return')cell.className+=' '+(row[field]<0?'negative':'positive');tr.append(cell)}body.append(tr)}}
  function performance(){if(!current)return;const week=current.weeks.find(week=>week.week_id===viewId),view=week||current.all_time;get('view-label').textContent=week?week.label:'All time';get('settlement-count').textContent=view.completed_rounds?`${view.completed_rounds.toLocaleString('en-US')} verified settlement${view.completed_rounds===1?'':'s'}`:'No verified settlements yet';ranking(view);renderChart(document,get('return-chart'),get('chart-legend'),view)}
  function usage(){
    const week=current?.weeks.find(week=>week.week_id===usageId),body=get('token-usage');body.replaceChildren();
    const rows=week?.tokens||MODEL_REGISTRY.map(model=>({model_id:model.id,...Object.fromEntries(TOKEN_FIELDS.map(field=>[field,null]))}));
    for(const model of MODEL_REGISTRY){
      const row=rows.find(row=>row.model_id===model.id),tr=el(document,'tr');tr.append(modelCell(document,model.id));
      for(const field of DISPLAY_TOKEN_FIELDS){const partial=row.token_coverage?.[field]==='partial',value=formatTokenField(row,field),cell=el(document,'td',value,'number'+(partial?' token-partial':row[field]===null?' unavailable':''));if(partial){cell.setAttribute('title','Recorded subtotal; some provider usage is unavailable.');cell.setAttribute('aria-label',`${value}. Recorded subtotal; some provider usage is unavailable.`)}tr.append(cell)}
      const value=row.estimated_cost_usd??null,coverage=row.cost_coverage||'unavailable';
      const cost=el(document,'td',formatEstimatedCost(value,coverage),'number'+(value===null?' unavailable':''));
      cost.setAttribute('title',coverage==='partial'?'Estimate uses available API records and conservative reservations; some usage is incomplete.':'Estimated API cost in USD for this reporting week.');
      tr.append(cost);body.append(tr);
    }
    const measured=rows.some(row=>TOKEN_FIELDS.some(field=>row[field]!==null||row.recorded_tokens?.[field]!=null));get('usage-status').textContent=measured?`${week.label} · ${week.tokens_coverage||'verified available counts'}${week.tokens_coverage==='partial'?' coverage':''}`:'No measured token usage has been published.';
    const shared=get('shared-usage');if(shared){const counts=week?.shared_tokens;shared.textContent=counts?`Shared news processing · Input ${formatTokens(counts.input_tokens)} · Cached ${formatTokens(counts.cached_input_tokens)} · Output ${formatTokens(counts.output_tokens)} · Total ${formatTokens(counts.total_tokens)} · Inference ${formatTokens(counts.inference_tokens)}${week.shared_tokens_coverage?` · ${week.shared_tokens_coverage} coverage`:''}`:'Shared news processing: token counts unavailable.'}
  }
  function clock(){const node=get('ny-clock');node.textContent=formatNY(now());node.dateTime=new Date(now()).toISOString()}
  function render(state){const oldView=viewId,oldUsage=usageId;current=state;const selector=get('performance-view'),usageSelector=get('usage-week');selector.replaceChildren(option('all-time','All time'),...state.weeks.map(week=>option(week.week_id,week.label)));viewId=state.weeks.some(week=>week.week_id===oldView)?oldView:'all-time';selector.value=viewId;usageId=state.weeks.some(week=>week.week_id===oldUsage)?oldUsage:state.weeks.at(-1)?.week_id||'';usageSelector.replaceChildren(...(state.weeks.length?state.weeks.map(week=>option(week.week_id,week.label)):[option('','Awaiting reporting weeks')]));usageSelector.value=usageId;usageSelector.disabled=!state.weeks.length;
    const badge=get('status-badge');badge.textContent=state.status[0].toUpperCase()+state.status.slice(1);badge.dataset.status=state.status;get('updated-at').textContent=state.updated_at?`Updated ${formatNY(state.updated_at)} NY`:'Awaiting first publication';get('feed-status').textContent=state.all_time.completed_rounds?'Verified aggregate results. Views refresh automatically once a minute.':'Waiting for verified results. No illustrative returns are shown.';
    get('next-release').textContent=state.next_release?formatNY(state.next_release.timestamp):state.status==='finished'?'Evaluation complete':'Awaiting schedule';get('session-label').textContent=state.next_release?.session_label||'The next market session will be confirmed by the data publisher.';get('session-note').textContent=state.next_release?.note||'Exchange holidays and session changes follow the published calendar.';
    const data=state.data_status;get('published-batches').textContent=data?`${data.published_batches.toLocaleString('en-US')}${data.expected_release_points===null?'':` / ${data.expected_release_points.toLocaleString('en-US')}`}`:'Unavailable';get('latest-release').textContent=data?.last_release_at?`${formatNY(data.last_release_at)} NY`:'Awaiting publication';get('coverage-status').textContent=data?data.coverage[0].toUpperCase()+data.coverage.slice(1):'Unavailable';
    performance();usage();clock()}
  async function refresh(){if(inflight)return;inflight=true;try{const response=await fetcher(feedUrl,{cache:'no-store',credentials:'omit'});if(!response.ok)throw Error('Feed unavailable');const next=validateState(await response.json());if(current&&((current.updated_at&&(!next.updated_at||timestamp(next.updated_at)<timestamp(current.updated_at)))||next.all_time.completed_rounds<current.all_time.completed_rounds))throw Error('Older publication rejected');render(next)}catch{get('feed-status').textContent=current?'The latest publication is unavailable. Showing the last verified aggregate results.':'Verified results are temporarily unavailable. No returns have been inferred.';if(!current){ranking({completed_rounds:0});usage();clock()}}finally{inflight=false}}
  get('performance-view').addEventListener('change',event=>{viewId=event.target.value;performance()});get('usage-week').addEventListener('change',event=>{usageId=event.target.value;usage()});
  return {refresh,clock,render:state=>render(validateState(state))};
}
if(typeof document!=='undefined'){const dashboard=createDashboard(document);dashboard.refresh();setInterval(()=>dashboard.refresh(),60000);setInterval(()=>dashboard.clock(),1000)}
