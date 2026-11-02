"""Explicit one-time migration; original competition files are never written."""
import copy
import json
import os
from pathlib import Path
import shutil
from .storage import write_private,read_json,file_hash,locked,now
from .registry import MODEL_IDS,TICKERS
from .accounting import new_book

def prepare(state_dir,legacy_dir,framework_root,key_file):
    root=Path(state_dir).resolve();legacy=Path(legacy_dir).resolve();repo=Path(__file__).resolve().parents[1]
    if root==repo or repo in root.parents:raise ValueError('Runtime data must remain outside the public Git repository')
    root.mkdir(parents=True,exist_ok=True,mode=0o700);root.chmod(0o700)
    os.environ['PAIR_TRADING_FIX_ROOT']=str(Path(framework_root).resolve())
    os.environ['MOIRA_LIVE_STATE_DIR']=str(root)
    with locked(root/'run.lock',nonblocking=True),locked(legacy/'run.lock',nonblocking=True):
        if (root/'bootstrap.json').exists():return read_json(root/'bootstrap.json')
        source=read_json(legacy/'state.json')
        for i in (1,2):
            rid=f'2026-11-02_{8+i:02d}:30'
            if set(source['sealed'].get(rid,{}))!={'Codex','FinMem','Moira'}:
                raise ValueError('Both original rounds need all three genuine accepted actions')
        proof={};seed=root/'seed';seed.mkdir(mode=0o700,exist_ok=True)
        def copied(source_path,destination):
            source_path=Path(source_path);destination=Path(destination)
            destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            value=source_path.read_bytes();destination.write_bytes(value);destination.chmod(0o600)
            proof[str(source_path)]={'source_sha256':file_hash(source_path),'copy':str(destination),'copy_sha256':file_hash(destination)}
            return destination
        bundle_m=read_json(legacy/'native/moira/training-inputs.json')
        bundle_f=read_json(legacy/'native/training-inputs.json')
        daily=copied(bundle_m['daily_path'],seed/'daily-close.parquet')
        for name,bundle in (('moira',bundle_m),('finmem',bundle_f)):
            copied(legacy/('native/moira/training-inputs.json' if name=='moira' else 'native/training-inputs.json'),seed/f'{name}-training-inputs-original.json')
            bundle=copy.deepcopy(bundle);bundle['daily_path']=str(daily)
            write_private(seed/f'{name}-bundle.json',bundle)
        prepared=read_json(legacy/'native/moira/prepared.json')
        copied(legacy/'native/moira/prepared.json',seed/'moira-prepared.json')
        copied(legacy/'native/company-operations-context.json',root/'company-context.json')
        recovered=Path(prepared.get('recovery',{}).get('source_directory','')).parent/'moira-phrase2-continuation-recovered-2026-11-02'
        # The installed checkpoint has an immutable external source in its preparation manifest.
        native_manifest=read_json(legacy/'native/preparation-integrity.json')
        traces_path=next((Path(p) for p in native_manifest['source_files'] if p.endswith('training-traces.json')
            and 'continuation-recovered' in p),None)
        if traces_path is None:raise ValueError('Current continued model training trace source unavailable')
        copied(traces_path,seed/'moira-training-traces.json')
        for model_id in ('moira','moira2-top1','moira2-top3','moira2-top5'):
            folder=root/'models'/model_id;folder.mkdir(parents=True,exist_ok=True,mode=0o700)
            copied(legacy/'native/moira/prepared.json',folder/'prepared.json')
            write_private(folder/'training-inputs.json',read_json(seed/'moira-bundle.json'))
        copied(legacy/'native/moira/live-state.json',root/'models/moira/live-state.json')
        fin=root/'models/finmem';epoch=fin/'epochs/initial';epoch.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        if epoch.exists():raise ValueError('Inspect partial bootstrap rather than overwrite native memory')
        shutil.copytree(legacy/'native/finmem',epoch)
        fin_prepared=read_json(epoch/'prepared.json')
        for pair,metadata in fin_prepared['pairs'].items():
            original=Path(metadata['checkpoint']);destination=fin/'warmups'/pair
            shutil.copytree(original,destination)
            metadata['checkpoint']=str(destination)
        write_private(epoch/'prepared.json',fin_prepared)
        # Repoint copies only. Never open a Qdrant client while copying source memory.
        for path in epoch.rglob('*.json'):
            try:text=path.read_text()
            except UnicodeDecodeError:continue
            replaced=text.replace(str(legacy/'native/finmem'),str(epoch))
            if replaced!=text:path.write_text(replaced);path.chmod(0o600)
        for path in fin.rglob('*'):
            if path.is_dir():path.chmod(0o700)
            elif path.is_file():path.chmod(0o600)
        write_private(fin/'active-epoch.json',{'directory':str(epoch),'started_at':source['published']['2026-11-02_09:30']['published_at']})
        for i in (1,2):
            copied(legacy/f'inputs-{i}.json',seed/f'legacy-inputs-{i}.json')
            copied(legacy/f'release-{i}.json',seed/f'legacy-release-{i}.json')
        config={'schema':'moira.live-testing.local-config/1','state_dir':str(root),
            'legacy_dir':str(legacy),'framework_root':str(Path(framework_root).resolve()),'key_file':str(Path(key_file).resolve()),
            'native_initial_checkpoint':file_hash(seed/'moira-prepared.json'),'created_at':now()}
        write_private(root/'config.json',config)
        book=new_book(MODEL_IDS,TICKERS)
        from .migration import import_legacy_actions
        import_legacy_actions(book,legacy,limit=2)
        write_private(root/'book.json',book)
        runtime={'schema':'moira.live-testing.runtime/1','published':{},'completed_events':[],
            'feedback_applied':{model:[] for model in MODEL_IDS},'model_attempts':{},'errors':{}}
        write_private(root/'runtime.json',runtime)
        result={'status':'ready','created_at':now(),'source_files':proof,'imported_real_models':3,
            'authorized_retrospective_models':['moira2-top1','moira2-top3','moira2-top5'],
            'authorization':'User explicitly authorized on 2026-11-02: preserve original three models actions; backfill first two Moira 2.0 actions with historical point-in-time data after deployment.'}
        write_private(root/'bootstrap.json',result)
        return result

def initialize_variants(root,gateway):
    """Seed three independent RRI banks from genuine pre-start training outcomes."""
    from datetime import datetime
    from .calendar import aware
    from .accounting import pair_return
    from .moira_native import initialize
    root=Path(root);prepared=read_json(root/'seed/moira-prepared.json');bundle=read_json(root/'seed/moira-bundle.json')
    traces=read_json(root/'seed/moira-training-traces.json');active=traces[-2]
    records=[];labels={'LONG_SHORT':'LS','SHORT_LONG':'SL','FLAT':'F'}
    for decision in active['decisions']:
        a,b=decision['selected_pair'].split('/');at=aware(decision['date']);rid=at.strftime('%Y-%m-%d_%H:%M')
        interval=bundle['intervals'][rid];action=labels[decision['final_executed_action']]
        cf={label:pair_return({'stock1':a,'stock2':b,'action':label},interval['entry_prices'],interval['exit_prices']) for label in ('LS','SL','F')}
        if abs(cf[action]-decision['actual_pnl'])>1e-12:raise ValueError('Historical seed return differs from immutable interval')
        reflection=(f"Historical {rid} {a}/{b}: executed {action}, realized return {cf[action]:.10f}; "
            f"counterfactual LS {cf['LS']:.10f}, SL {cf['SL']:.10f}, F {cf['F']:.10f}. "
            f"Best action in this completed interval: {max(cf,key=cf.get)}. These are retrospective known outcomes, not forecasts.")
        records.append({'unit_id':'historical-training:'+rid,'action_id':rid,'stock1':a,'stock2':b,'action':action,
            'decision_at':at.isoformat(),'entry_at':interval['entry_timestamp'],'exit_at':interval['exit_timestamp'],
            'available_at':prepared['finished_at'],'actual_return':cf[action],'counterfactual_returns':cf,'reflection':reflection,
            'entities':[a,b],'provenance':{'kind':'retained_prompt_training_episode_counterfactual',
                'source_trace_sha256':file_hash(root/'seed/moira-training-traces.json'),'source_pass':len(traces)-1,
                'source_prices_known_at':interval['exit_timestamp'],'checkpoint_training_finished_at':prepared['finished_at'],
                'unit_constructed_at':now(),'scope':'historical initialization, not a scheduled Monday training job'}})
    for model in ('moira2-top1','moira2-top3','moira2-top5'):
        initialize(prepared,bundle,root/'models'/model,records,gateway.scoped(model,'warmup','initial-counterfactual-bank'),model_id=model)
    write_private(root/'variant-initialization.json',{'status':'ready','generated_at':now(),'historical_unit_count':len(records),
        'source_checkpoint':file_hash(root/'seed/moira-prepared.json'),'initial_training_window':prepared['training_window'],
        'first_future_weekly_training':'2026-11-09T08:00:00-05:00'})
