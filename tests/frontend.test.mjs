import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {MODEL_REGISTRY,PERIOD,feedUrl,validateState,validateView,validateTokens,validateSharedTokens,metrics,rankRows,formatReturn,formatSharpe,formatDrawdown,formatTokens,formatTokenField,formatEstimatedCost,formatNY,createDashboard} from '../docs/app.js';

const initial=()=>({schema:'moira.live-testing.website/1',status:'pending',timezone:'America/New_York',updated_at:null,
  period:structuredClone(PERIOD),next_release:null,all_time:{completed_rounds:0,rows:[],series:[]},
  weeks:[{week_id:'2026-W45',label:'2–6 November 2026',start:'2026-11-02T00:00:00-05:00',end:'2026-11-09T00:00:00-05:00',
    completed_rounds:0,rows:[],series:[],tokens:MODEL_REGISTRY.map(model=>({model_id:model.id,input_tokens:null,
      cached_input_tokens:null,output_tokens:null,total_tokens:null,training_tokens:null,inference_tokens:null})),tokens_coverage:'unavailable'}],
  data_status:{published_batches:0,expected_release_points:null,last_release_at:null,coverage:'unavailable'}});
function verified(){const state=initial();state.status='live';state.updated_at='2026-11-02T11:31:00-05:00';state.next_release={timestamp:'2026-11-02T12:30:00-05:00',session_label:'NYSE regular session',note:'Next scheduled data publication. Exchange holidays follow the verified calendar.'};const series=MODEL_REGISTRY.map((model,i)=>({model_id:model.id,points:[{step:0,equity:1},{step:1,equity:1+(i-2)/100},{step:2,equity:1+(i-1)/100}]}));const view={completed_rounds:2,rows:series.map(row=>({model_id:row.model_id,...metrics(row.points)})),series};state.all_time=structuredClone(view);Object.assign(state.weeks[0],structuredClone(view));state.weeks[0].tokens_coverage='partial';state.weeks[0].tokens[1]={model_id:'finmem',input_tokens:120,cached_input_tokens:20,output_tokens:30,total_tokens:150,training_tokens:0,inference_tokens:150,estimated_cost_usd:.01,cost_coverage:'complete'};state.data_status={published_batches:3,expected_release_points:850,last_release_at:'2026-11-02T11:30:00-05:00',coverage:'partial'};state.budget={month:'2026-11',spent_usd:1.25,reserved_usd:.1,warning_usd:20,limit_usd:100,status:'ok'};return state}
function fakeDocument(){const nodes=new Map();function element(tagName='div'){return {tagName,children:[],style:{},dataset:{},attributes:{},events:{},textContent:'',append(...children){this.children.push(...children)},replaceChildren(...children){this.children=[...children]},setAttribute(key,value){this.attributes[key]=value},addEventListener(event,handler){this.events[event]=handler}}}for(const id of ['status-badge','next-release','session-label','session-note','ny-clock','performance-view','view-label','settlement-count','updated-at','feed-status','return-chart','chart-legend','rankings','usage-week','usage-status','token-usage','published-batches','latest-release','coverage-status'])nodes.set(id,element());return {nodes,document:{getElementById:id=>nodes.get(id)||null,createElement:element,createElementNS:(_ns,tag)=>element(tag)}}}

test('initial feed is an honest pending six-model roster without invented performance or usage',()=>{const state=initial();assert.equal(validateState(state),state);assert.equal(state.status,'pending');assert.deepEqual(state.all_time,{completed_rounds:0,rows:[],series:[]});assert.ok(state.weeks[0].tokens.every(row=>Object.entries(row).filter(([key])=>key!=='model_id').every(([,value])=>value===null)));assert.equal(MODEL_REGISTRY.length,6);assert.deepEqual(MODEL_REGISTRY.map(model=>model.id),['codex','finmem','moira','moira2-top1','moira2-top3','moira2-top5']);assert.equal(MODEL_REGISTRY[0].backbone,'gpt-6.1-sol');assert.ok(MODEL_REGISTRY.slice(1).every(model=>model.backbone==='gpt-4.1-mini'))});

test('the current published aggregate feed remains valid as actual data and usage advance',()=>{const state=JSON.parse(readFileSync(new URL('../docs/state.json',import.meta.url),'utf8'));assert.equal(validateState(state),state)});

test('fixed evaluation boundaries and NY daylight saving time match the requested dates',()=>{assert.equal(PERIOD.start,'2026-11-02T09:30:00-05:00');assert.equal(PERIOD.last_release,'2027-05-21T15:30:00-04:00');assert.equal(PERIOD.final_settlement,'2027-05-24T10:30:00-04:00');assert.match(formatNY(PERIOD.start),/09:30/);assert.match(formatNY(PERIOD.final_settlement),/10:30/);const state=initial();state.period.final_settlement='2027-05-24T09:30:00-04:00';assert.throws(()=>validateState(state))});

test('public schema rejects private fields, incorrect models and any fake initial flat result',()=>{for(const edit of [s=>s.api_key='secret',s=>s.all_time.actions=[],s=>s.weeks[0].tokens[0].prompt='private',s=>s.next_release={timestamp:PERIOD.start,session_label:'Session',note:'Safe note',raw_news:[]},s=>s.data_status.price=100,s=>s.all_time.rows=[{model_id:'codex',cumulative_return:0,sharpe_ratio:null,max_drawdown:0}]]){const state=initial();edit(state);assert.throws(()=>validateState(state))}const good=verified();assert.doesNotThrow(()=>validateState(good));for(const edit of [s=>s.all_time.rows[0].stock1='NVDA',s=>s.all_time.series[0].points[0].action='LS',s=>s.all_time.rows[0].model_id='participant',s=>s.all_time.rows[0].max_drawdown=-.01,s=>s.all_time.rows[0].sharpe_ratio=Infinity,s=>s.all_time.rows.pop(),s=>s.all_time.series.pop()]){const state=structuredClone(good);edit(state);assert.throws(()=>validateState(state))}});

test('paths require initial equity one, consecutive verified steps and independently checked risk metrics',()=>{const state=verified();assert.doesNotThrow(()=>validateView(state.all_time));for(const edit of [v=>v.series[0].points[0].equity=1.5,v=>v.series[0].points[1].step=2,v=>v.series[0].points.pop(),v=>v.series[0].points[1].equity=0,v=>v.rows[0].cumulative_return+=.1,v=>v.rows[0].sharpe_ratio=999,v=>v.rows[0].max_drawdown=.8]){const view=structuredClone(state.all_time);edit(view);assert.throws(()=>validateView(view))}const zero=metrics([{step:0,equity:1},{step:1,equity:1}]);assert.deepEqual(zero,{cumulative_return:0,sharpe_ratio:null,max_drawdown:0});assert.equal(formatSharpe(zero.sharpe_ratio),'N/A');assert.equal(formatDrawdown(0),'0%');assert.equal(formatDrawdown(.025),'2.50%');assert.equal(formatReturn(.01),'+1.00%')});

test('weekly views are independent, rebase paths to one, and must cover total completed rounds',()=>{const state=verified();assert.doesNotThrow(()=>validateState(state));state.weeks[0].completed_rounds=1;assert.throws(()=>validateState(state));const shifted=verified();shifted.weeks[0].series[0].points[0].equity=1.1;assert.throws(()=>validateState(shifted));const duplicate=verified();duplicate.weeks.push(structuredClone(duplicate.weeks[0]));assert.throws(()=>validateState(duplicate));assert.deepEqual(rankRows(verified().all_time.rows).map(row=>row.model_id),['moira2-top5','moira2-top3','moira2-top1','moira','finmem','codex'])});

test('constant positive returns use the same floating-point zero-volatility threshold as the scorer',()=>{const points=[1,1.01,1.0201,1.030301].map((equity,step)=>({step,equity}));assert.equal(metrics(points).sharpe_ratio,null);assert.equal(metrics(points).max_drawdown,0);const view={completed_rounds:3,rows:MODEL_REGISTRY.map(model=>({model_id:model.id,...metrics(points)})),series:MODEL_REGISTRY.map(model=>({model_id:model.id,points:structuredClone(points)}))};assert.doesNotThrow(()=>validateView(view))});

test('weekly token counts distinguish unavailable from measured zero, enforce accounting and retain partial imports',()=>{const state=verified();assert.doesNotThrow(()=>validateTokens(state.weeks[0].tokens));assert.equal(formatTokens(null),'Unavailable');assert.equal(formatTokens(0),'0');assert.equal(formatTokens(20000),'20,000');for(const edit of [rows=>rows[1].cached_input_tokens=121,rows=>rows[1].total_tokens=151,rows=>rows[1].training_tokens=1,rows=>rows[1].input_tokens=-1,rows=>rows[1].model_id='codex']){const rows=structuredClone(state.weeks[0].tokens);edit(rows);assert.throws(()=>validateTokens(rows))}state.weeks[0].tokens_coverage='partial';assert.doesNotThrow(()=>validateState(state))});

const tokenFields=['input_tokens','cached_input_tokens','output_tokens','total_tokens','training_tokens','inference_tokens'];
const purposeFields=['trading_tokens','direction_decision_tokens','pair_reselection_tokens','relationship_monitor_tokens','loo_attribution_tokens','other_inference_tokens'];
function classified(row,values={}){return Object.assign(row,Object.fromEntries(purposeFields.map(field=>[field,null])),values)}
function tokenMetadata(row){const fields=Object.hasOwn(row,'trading_tokens')?[...tokenFields,...purposeFields]:tokenFields;row.recorded_tokens=Object.fromEntries(fields.map(field=>[field,row[field]]));row.token_coverage=Object.fromEntries(fields.map(field=>[field,row[field]===null?'unavailable':'complete']));return row}
function breakdown(cells,field){return cells[0].children[0].children[1].children.find(item=>item.dataset.tokenField===field).children[1]}
function partialUsage(){const state=verified(),row=tokenMetadata(state.weeks[0].tokens[0]);for(const[field,value]of Object.entries({input_tokens:12345,cached_input_tokens:0,total_tokens:13000,inference_tokens:13000})){row.recorded_tokens[field]=value;row.token_coverage[field]='partial'}tokenMetadata(state.weeks[0].tokens[1]);return state}

test('recorded token metadata distinguishes exact totals, lower bounds and genuinely unavailable fields',()=>{
  const state=partialUsage(),rows=state.weeks[0].tokens;
  assert.doesNotThrow(()=>validateState(state));assert.equal(formatTokenField(rows[0],'input_tokens'),'≥12,345');
  assert.equal(formatTokenField(rows[0],'cached_input_tokens'),'≥0');assert.equal(formatTokenField(rows[0],'output_tokens'),'Unavailable');
  assert.equal(formatTokenField(rows[1],'input_tokens'),'120');assert.equal(formatTokenField(rows[1],'training_tokens'),'0');
  assert.equal(rows[0].input_tokens,null);assert.equal(rows[0].total_tokens,null);
  // Recorded fields can cover different request subsets; complete totals keep
  // their original arithmetic checks instead of equating partial subtotals.
  rows[0].recorded_tokens.output_tokens=900;rows[0].token_coverage.output_tokens='partial';
  assert.doesNotThrow(()=>validateTokens(rows));
  rows[1].total_tokens=151;rows[1].recorded_tokens.total_tokens=151;
  assert.throws(()=>validateTokens(rows));
});

test('recorded token metadata rejects mismatched completeness, unsafe counts and private or missing fields',()=>{
  for(const edit of [rows=>delete rows[0].recorded_tokens,rows=>delete rows[0].token_coverage,
    rows=>delete rows[0].recorded_tokens.output_tokens,rows=>rows[0].recorded_tokens.prompt='private',
    rows=>rows[0].token_coverage.response_id='private',rows=>rows[0].token_coverage.input_tokens='estimated',
    rows=>rows[0].recorded_tokens.input_tokens=-1,rows=>rows[0].recorded_tokens.input_tokens=Number.MAX_SAFE_INTEGER+1,
    rows=>rows[0].recorded_tokens.input_tokens=NaN,rows=>rows[0].recorded_tokens.input_tokens=null,
    rows=>rows[0].input_tokens=12345,rows=>rows[0].recorded_tokens.output_tokens=1,
    rows=>rows[0].token_coverage.input_tokens='complete',rows=>rows[1].recorded_tokens.input_tokens=121,
    rows=>rows[1].token_coverage.input_tokens='unavailable']){
    const rows=partialUsage().weeks[0].tokens;edit(rows);assert.throws(()=>validateTokens(rows));
  }
});

test('usage cells disclose recorded lower bounds accessibly without changing costs, rankings or paths',()=>{
  const state=partialUsage(),before=structuredClone(state.all_time),{nodes,document}=fakeDocument();
  createDashboard(document).render(state);const cells=nodes.get('token-usage').children[0].children;
  assert.equal(breakdown(cells,'input_tokens').textContent,'≥12,345');assert.equal(breakdown(cells,'cached_input_tokens').textContent,'≥0');assert.equal(breakdown(cells,'output_tokens').textContent,'Unavailable');
  assert.equal(breakdown(cells,'input_tokens').attributes.title,'Recorded subtotal; some provider usage is unavailable.');
  assert.match(breakdown(cells,'input_tokens').attributes['aria-label'],/≥12,345.*Recorded subtotal/);assert.match(breakdown(cells,'input_tokens').className,/token-partial/);
  assert.equal(breakdown(nodes.get('token-usage').children[1].children,'input_tokens').textContent,'120');
  assert.equal(nodes.get('token-usage').children[1].children[4].textContent,'~$0.01');
  assert.equal(cells[2].textContent,'Unavailable');assert.equal(cells[3].textContent,'≥13,000');
  assert.deepEqual(state.all_time,before);assert.equal(nodes.get('return-chart').children[0].children.filter(n=>n.tagName==='polyline').length,6);
  const onlyPartial=initial();onlyPartial.weeks[0].tokens[0]=state.weeks[0].tokens[0];onlyPartial.weeks[0].tokens_coverage='partial';
  createDashboard(document).render(onlyPartial);assert.match(nodes.get('usage-status').textContent,/partial coverage/);
  assert.equal(nodes.get('rankings').children[0].children[2].textContent,'Awaiting data');
  const html=readFileSync(new URL('../docs/index.html',import.meta.url),'utf8');
  assert.match(html,/≥ denotes a recorded lower bound/);assert.match(html,/\* Cost estimate based on available records/);
  assert.match(html,/app\.js\?v=20261105-sol-reference/);assert.ok(!/id="budget-status"|\$20|\$100/.test(html));
});

test('budget is a NY month aggregate with fixed warning and hard stop, never inferred zero',()=>{assert.doesNotThrow(()=>validateState(verified()));for(const edit of [s=>s.budget.spent_usd=-1,s=>s.budget.limit_usd=200,s=>s.budget.reserved_usd=NaN,s=>s.budget.token='secret']){const state=verified();edit(state);assert.throws(()=>validateState(state))}const state=initial();state.budget={month:'2026-11',spent_usd:null,reserved_usd:null,warning_usd:20,limit_usd:100,status:'unavailable'};assert.doesNotThrow(()=>validateState(state))});

test('weekly estimated model costs accompany separate training and trading totals without private budget controls',()=>{
  const state=verified(),rows=state.weeks[0].tokens;
  rows[0].estimated_cost_usd=.125;rows[0].cost_coverage='partial';
  rows[2].estimated_cost_usd=0;rows[2].cost_coverage='complete';
  assert.doesNotThrow(()=>validateState(state));
  assert.equal(formatEstimatedCost(null),'Unavailable');assert.equal(formatEstimatedCost(0),'~$0.00');
  assert.equal(formatEstimatedCost(.0012),'~$0.0012');assert.equal(formatEstimatedCost(.125,'partial'),'~$0.13*');
  for(const edit of [row=>row.estimated_cost_usd=-1,row=>row.estimated_cost_usd=Infinity,row=>row.cost_coverage='unknown',row=>delete row.cost_coverage,row=>row.estimated_cost_usd=null]){
    const copy=structuredClone(rows);edit(copy[0]);assert.throws(()=>validateTokens(copy));
  }
  const {nodes,document}=fakeDocument();createDashboard(document).render(state);
  assert.equal(nodes.get('token-usage').children[0].children.length,5);
  assert.equal(nodes.get('token-usage').children[0].children[4].textContent,'~$0.13*');
  assert.match(nodes.get('token-usage').children[0].children[4].attributes.title,/incomplete/);
  const html=readFileSync(new URL('../docs/index.html',import.meta.url),'utf8');
  assert.match(html,/>Estimated Cost<\/th>/);assert.match(html,/>Training<\/th>/);assert.match(html,/>Trading<\/th>/);assert.ok(!/id="budget-status"|\$20|\$100/.test(html));
});

test('shared news-processing tokens are a separate safe aggregate, never a seventh ranked model',()=>{const state=verified(),shared={input_tokens:100,cached_input_tokens:10,output_tokens:20,total_tokens:120,training_tokens:0,inference_tokens:120};state.weeks[0].shared_tokens=shared;state.weeks[0].shared_tokens_coverage='partial';assert.doesNotThrow(()=>validateState(state));assert.equal(validateSharedTokens(shared),shared);assert.equal(state.all_time.rows.length,6);for(const edit of [value=>value.model_id='data',value=>value.raw_news=[],value=>value.total_tokens=119,value=>value.input_tokens=-1]){const copy=structuredClone(shared);edit(copy);assert.throws(()=>validateSharedTokens(copy))}const {nodes,document}=fakeDocument();nodes.set('shared-usage',document.createElement('p'));const dashboard=createDashboard(document);dashboard.render(state);assert.match(nodes.get('shared-usage').textContent,/Shared news processing/);assert.match(nodes.get('shared-usage').textContent,/Total 120/);assert.match(nodes.get('shared-usage').textContent,/partial coverage/);assert.equal(nodes.get('token-usage').children.length,6)});

test('training and trading retain independent completeness instead of subtracting from an unknown total',()=>{
  const state=verified(),rows=state.weeks[0].tokens;
  rows[2]=tokenMetadata(classified({model_id:'moira',input_tokens:300,cached_input_tokens:40,output_tokens:30,
    total_tokens:330,training_tokens:200,inference_tokens:130},{trading_tokens:90,direction_decision_tokens:80,
      pair_reselection_tokens:10,relationship_monitor_tokens:20,loo_attribution_tokens:15,other_inference_tokens:5}));
  rows[3]=tokenMetadata(classified({model_id:'moira2-top1',...Object.fromEntries(tokenFields.map(field=>[field,null])),training_tokens:80}));
  for(const[field,value]of Object.entries({input_tokens:100,cached_input_tokens:0,output_tokens:10,total_tokens:110,inference_tokens:30,trading_tokens:20,direction_decision_tokens:15,pair_reselection_tokens:5,relationship_monitor_tokens:4,loo_attribution_tokens:3,other_inference_tokens:3})){rows[3].recorded_tokens[field]=value;rows[3].token_coverage[field]='partial'}
  assert.doesNotThrow(()=>validateState(state));
  const before=structuredClone(state),{nodes,document}=fakeDocument();createDashboard(document).render(state);
  const exact=nodes.get('token-usage').children[2].children,partial=nodes.get('token-usage').children[3].children;
  assert.equal(exact[1].textContent,'200');assert.equal(exact[2].textContent,'90');
  assert.equal(partial[3].textContent,'≥110');assert.equal(partial[1].textContent,'80');
  assert.equal(partial[2].textContent,'≥20');assert.match(partial[2].attributes['aria-label'],/Recorded subtotal/);
  assert.deepEqual(state,before);
});

function classifiedUsage(){const state=verified();state.weeks[0].tokens[1]=tokenMetadata(classified(state.weeks[0].tokens[1],{trading_tokens:90,direction_decision_tokens:80,pair_reselection_tokens:10,relationship_monitor_tokens:30,loo_attribution_tokens:20,other_inference_tokens:10}));state.weeks[0].tokens[1].reserved_cost_usd=.75;return state}

test('verified classification is an all-or-none safe bundle with independent purpose arithmetic',()=>{
  const state=classifiedUsage();assert.doesNotThrow(()=>validateState(state));
  assert.equal(Object.keys(state.weeks[0].tokens[1].recorded_tokens).length,12);
  for(const edit of [r=>delete r.trading_tokens,r=>r.direction_decision_tokens=-1,
    r=>r.other_inference_tokens=Infinity,r=>r.trading_tokens=91,r=>r.loo_attribution_tokens=21,
    r=>delete r.recorded_tokens.loo_attribution_tokens,r=>delete r.token_coverage.trading_tokens,
    r=>r.recorded_tokens.raw_response='private',r=>r.trading_tokens=Number.MAX_SAFE_INTEGER+1]){
    const copy=structuredClone(state.weeks[0].tokens);edit(copy[1]);assert.throws(()=>validateTokens(copy));
  }
  // Classification may genuinely be absent from an older feed even when all
  // nontraining usage is known. It must not be relabeled as trading.
  const {nodes,document}=fakeDocument();createDashboard(document).render(verified());
  const cells=nodes.get('token-usage').children[1].children;
  assert.equal(cells[2].textContent,'Unavailable');assert.equal(cells[3].textContent,'150');
  assert.equal(breakdown(cells,'inference_tokens').textContent,'150');
  assert.equal(formatTokenField(state.weeks[0].tokens[0],'trading_tokens'),'Unavailable');
  const nulls=classified(initial().weeks[0].tokens[0]);assert.doesNotThrow(()=>validateTokens([nulls,...initial().weeks[0].tokens.slice(1)]));
});

test('model expansion retains actual usage and a separately labeled conservative reservation',()=>{
  const state=classifiedUsage(),before=structuredClone(state),{nodes,document}=fakeDocument();
  createDashboard(document).render(state);const cells=nodes.get('token-usage').children[1].children;
  assert.equal(cells.length,5);assert.deepEqual(cells.slice(1).map(cell=>cell.textContent),['0','90','150','~$0.01']);
  const details=cells[0].children[0];assert.equal(details.tagName,'details');assert.equal(details.children[0].tagName,'summary');
  assert.equal(details.children[0].children[0].className,'model-name');
  assert.equal(details.children[0].children[0].children[1].textContent,'FinMem');
  assert.equal(breakdown(cells,'input_tokens').textContent,'120');assert.equal(breakdown(cells,'cached_input_tokens').textContent,'20');
  assert.equal(breakdown(cells,'output_tokens').textContent,'30');assert.equal(breakdown(cells,'direction_decision_tokens').textContent,'80');
  assert.equal(breakdown(cells,'pair_reselection_tokens').textContent,'10');assert.equal(breakdown(cells,'relationship_monitor_tokens').textContent,'30');
  assert.equal(breakdown(cells,'loo_attribution_tokens').textContent,'20');assert.equal(breakdown(cells,'other_inference_tokens').textContent,'10');
  const reservation=details.children[1].children.at(-1);assert.equal(reservation.children[0].textContent,'Conservative reservation (USD)');assert.equal(reservation.children[1].textContent,'$0.7500');
  assert.deepEqual(state,before);assert.equal(nodes.get('return-chart').children[0].children.filter(node=>node.tagName==='polyline').length,6);
  const html=readFileSync(new URL('../docs/index.html',import.meta.url),'utf8');assert.match(html,/colspan="5"/);assert.match(html,/not an invoice/);assert.match(html,/failed, superseded and embedding/);assert.ok(!/Total = Training \+ Trading =/.test(html));
});

test('reservation is an optional independent estimate and never silently added to cost',()=>{
  const rows=initial().weeks[0].tokens;rows[0].reserved_cost_usd=.5;rows[1].reserved_cost_usd=null;
  assert.doesNotThrow(()=>validateTokens(rows));
  for(const value of [-1,NaN,Infinity,'0.5']){const copy=structuredClone(rows);copy[0].reserved_cost_usd=value;assert.throws(()=>validateTokens(copy))}
  const state=classifiedUsage();state.weeks[0].tokens[1].reserved_cost_usd=999;
  const {nodes,document}=fakeDocument();createDashboard(document).render(state);
  assert.equal(nodes.get('token-usage').children[1].children[4].textContent,'~$0.01');
  assert.match(nodes.get('token-usage').children[1].children[4].attributes.title,/reservations are separate/);
});

test('rendering preserves weekly controls, six ranked rows and safe locally generated SVG paths',()=>{const {nodes,document}=fakeDocument();const dashboard=createDashboard(document);dashboard.render(initial());assert.equal(nodes.get('rankings').children.length,6);assert.equal(nodes.get('rankings').children[0].children[2].textContent,'Awaiting data');assert.equal(nodes.get('token-usage').children[0].children[1].textContent,'Unavailable');assert.equal(nodes.get('return-chart').children[0].tagName,'div');dashboard.render(verified());assert.equal(nodes.get('rankings').children.length,6);assert.equal(nodes.get('rankings').children[0].children[1].children[0].children[1].textContent,'Moira 2.0 · Top K = 5');const svg=nodes.get('return-chart').children[0];assert.equal(svg.tagName,'svg');assert.equal(svg.attributes.role,'img');assert.equal(svg.children.filter(node=>node.tagName==='polyline').length,6);assert.equal(nodes.get('chart-legend').children.length,6);nodes.get('performance-view').events.change({target:{value:'2026-W45'}});assert.equal(nodes.get('view-label').textContent,'2–6 November 2026');dashboard.render(verified());assert.equal(nodes.get('performance-view').value,'2026-W45');assert.match(nodes.get('usage-status').textContent,/partial coverage/);assert.equal(nodes.get('token-usage').children[1].children[1].textContent,'0');assert.equal(nodes.get('token-usage').children[1].children[2].textContent,'Unavailable');assert.equal(nodes.get('token-usage').children[1].children[3].textContent,'150');assert.equal(nodes.get('token-usage').children[1].children[4].textContent,'~$0.01');assert.equal(nodes.has('budget-status'),false);assert.match(nodes.get('next-release').textContent,/12:30/);assert.ok(!JSON.stringify(svg).includes('stock1'));assert.ok(!JSON.stringify(svg).includes('api_key'))});

test('refresh only fetches the public same-origin feed without credentials and keeps last verified results on failure',async()=>{const {nodes,document}=fakeDocument();let state=verified(),fail=false;const calls=[];const dashboard=createDashboard(document,{fetcher:async(url,options)=>{calls.push([url,options]);if(fail)throw Error('unavailable');return {ok:true,json:async()=>state}}});await dashboard.refresh();const rendered=nodes.get('return-chart').children[0];assert.equal(calls[0][0],feedUrl);assert.equal(calls[0][1].credentials,'omit');assert.equal(calls[0][1].cache,'no-store');fail=true;await dashboard.refresh();assert.equal(nodes.get('return-chart').children[0],rendered);assert.match(nodes.get('feed-status').textContent,/last verified/);fail=false;state={...state,updated_at:'2026-11-02T11:00:00-05:00'};await dashboard.refresh();assert.equal(nodes.get('return-chart').children[0],rendered);assert.match(nodes.get('feed-status').textContent,/last verified/);state=initial();await dashboard.refresh();assert.equal(nodes.get('return-chart').children[0],rendered);assert.match(nodes.get('feed-status').textContent,/last verified/)});

test('an unavailable first feed remains an honest pending roster without fabricated paths',async()=>{const {nodes,document}=fakeDocument();const dashboard=createDashboard(document,{fetcher:async()=>{throw Error('offline')}});await dashboard.refresh();assert.equal(nodes.get('rankings').children.length,6);assert.match(nodes.get('feed-status').textContent,/No returns have been inferred/);assert.ok(nodes.get('rankings').children.every(row=>row.children[2].textContent==='Awaiting data'))});

test('static UI keeps model-only purpose, private dataset login link, accessible controls and responsive layout',()=>{const html=readFileSync(new URL('../docs/index.html',import.meta.url),'utf8'),css=readFileSync(new URL('../docs/style.css',import.meta.url),'utf8'),script=readFileSync(new URL('../docs/app.js',import.meta.url),'utf8');assert.match(html,/<title>MOIRA Live Testing<\/title>/);assert.match(html,/<h1 id="page-heading">MOIRA<br><span>Live Testing<\/span><\/h1>/);assert.match(html,/2 Nov 2026 · 09:30/);assert.match(html,/21 May 2027 · 15:30/);assert.match(html,/24 May 2027 · 10:30/);assert.match(html,/Final entry · 09:30/);assert.match(html,/https:\/\/huggingface\.co\/datasets\/CarolynJiang\/MOIRA-Live-testing/);assert.match(html,/sign-in and repository permission/);assert.match(html,/for="performance-view"/);assert.match(html,/for="usage-week"/);assert.match(html,/aria-live="polite"/);assert.ok(!/<form|type="password"|Submit a pair|workers\.dev|iframe/.test(html));assert.ok(!/innerHTML|localStorage|Authorization|Bearer/.test(script));assert.match(css,/@media\(max-width:720px\)/);assert.match(css,/overflow-x:auto/);assert.match(css,/:focus-visible/);assert.match(css,/\.sr-only/)});
