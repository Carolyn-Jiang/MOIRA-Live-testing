"""Causally verified weekly FinMem experiences; complete provenance stays local."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from .accounting import matured_feedback, pair_return, risk_metrics, validate_book
from .calendar import aware, week_bounds
from .data import round_id
from .registry import TICKERS
from .storage import digest, read_json, file_hash

SCHEMA='moira.finmem.weekly-selection-context/1'
_FIELDS={'schema','as_of','week_start','observation_at','known_labels','weekly_return','finalized_memory'}
_CASE={'case_id','release_at','generated_at','entry_at','exit_at','available_at','action','entry_prices','exit_prices','step_return','decision_reason'}


def _same_number(a,b):
    return type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def validate_context(value):
    if not isinstance(value,dict) or set(value)!=_FIELDS or value['schema']!=SCHEMA:
        raise ValueError('An explicit verified weekly-selection context is required')
    context=copy.deepcopy(value);cutoff=aware(context['as_of']);start,_=week_bounds(cutoff)
    if aware(context['week_start'])!=start or aware(context['observation_at'])>cutoff:
        raise ValueError('Weekly context source cutoff changed')
    rows=context['finalized_memory']
    if not isinstance(rows,list) or type(context['known_labels']) is not int or context['known_labels']!=len(rows):
        raise ValueError('Known weekly labels must equal the actual matured experiences')
    seen=set();last=None;returns=[]
    for row in rows:
        if not isinstance(row,dict) or set(row)!=_CASE:raise ValueError('A complete actual matured experience is required')
        if not isinstance(row['case_id'],str) or not row['case_id'] or row['case_id'] in seen:raise ValueError('Unique compact case references required')
        seen.add(row['case_id']);release=aware(row['release_at']);entry=aware(row['entry_at']);exit=aware(row['exit_at']);generated=aware(row['generated_at']);available=aware(row['available_at'])
        if not release<entry<exit or not release<=generated<=available or not start<=exit<=available<=cutoff:
            raise ValueError('Unexited or late-available experience cannot enter weekly selection')
        if last is not None and release<=last:raise ValueError('Matured experiences must remain in source order')
        last=release;action=row['action']
        if not isinstance(action,dict) or set(action)!={'stock1','stock2','action'} or any(action.get(k) not in TICKERS for k in ('stock1','stock2')):
            raise ValueError('Actual eligible pair action required')
        symbols={action['stock1'],action['stock2']}
        if len(symbols)!=2 or any(not isinstance(row[k],dict) or set(row[k])!=symbols for k in ('entry_prices','exit_prices')):
            raise ValueError('Both immutable pair-leg prices are required')
        actual=pair_return(action,row['entry_prices'],row['exit_prices'])
        if not _same_number(row['step_return'],actual):raise ValueError('Experience return differs from immutable quotes')
        if not isinstance(row['decision_reason'],str) or not row['decision_reason'].strip():raise ValueError('Actual sealed decision reasoning is required')
        returns.append(actual)
    expected=risk_metrics(returns);metrics=context['weekly_return']
    if not isinstance(metrics,dict) or set(metrics)!=set(expected):raise ValueError('Weekly risk metrics are incomplete')
    for key,actual in expected.items():
        if actual is None:
            if metrics[key] is not None:raise ValueError('Unavailable Sharpe cannot be invented')
        elif not _same_number(metrics[key],actual):raise ValueError('Weekly metrics differ from actual matured returns')
    return context


def build_context(root,book,cutoff,observation_at):
    """Bind each finalized experience to its real native reason and old epoch.

    Pending decisions and current ungraded reflection records are never included.
    The private proof retains all epoch paths, full IDs and exact source hashes.
    """
    root=Path(root).resolve();cutoff=aware(cutoff);validate_book(book);start,_=week_bounds(cutoff)
    rows=[r for r in matured_feedback(book,'finmem',cutoff,observation_asof=cutoff) if start<=aware(r['exit_at'])<=cutoff]
    experiences=[];proof={'source_book_sha256':digest(book),'native_sources':{},'cases':{}}
    native_cache={}
    for index,row in enumerate(rows,1):
        decision=book['decisions']['finmem'][row['release_at']];directory=decision['audit'].get('native_generation')
        if not directory:
            audit=decision['audit'];seal=audit.get('source_original_seal');rid=round_id(row['release_at'])
            if (audit.get('input_protocol')!='legacy_input_summary_protocol' or not isinstance(seal,dict)
                    or digest(seal)!=audit.get('source_seal_sha256') or audit.get('source_round_id')!=rid
                    or seal.get('action')!=decision['action'] or aware(seal['sealed_at'])!=aware(decision['generated_at'])):
                raise RuntimeError('A finalized FinMem decision lacks verified native epoch provenance')
            matches=[]
            for candidate_path in (root/'models/finmem').rglob('live-state.json'):
                candidate=read_json(candidate_path);record=(candidate or {}).get('decisions',{}).get(rid,{}).get('action',{})
                if (all(record.get(k)==decision['action'][k] for k in ('stock1','stock2','action'))
                        and record.get('generated_at') and aware(record['generated_at'])==aware(decision['generated_at'])):
                    matches.append(candidate_path.parent)
            if len(matches)!=1:raise RuntimeError('Legacy native epoch must have one exact sealed-action match')
            directory=matches[0]
        directory=Path(directory).resolve()
        if root not in directory.parents:raise RuntimeError('Native epoch must stay in the private experiment state')
        path=directory/'live-state.json'
        if path not in native_cache:native_cache[path]=read_json(path);proof['native_sources'][str(path)]=file_hash(path)
        native=native_cache[path];rid=round_id(row['release_at']);native_decision=(native or {}).get('decisions',{}).get(rid);feedback=(native or {}).get('feedback',{}).get(rid)
        if not native_decision or not feedback:raise RuntimeError('Matured native reasoning and feedback are required before weekly selection')
        action=native_decision['action']
        if {k:action[k] for k in ('stock1','stock2','action')}!=decision['action'] or aware(action['generated_at'])!=aware(decision['generated_at']):
            raise RuntimeError('Native sealed experience differs from immutable accounting')
        entry=book['valuations'][row['entry_at']];exit=book['valuations'][row['exit_at']]
        available=max(aware(decision['generated_at']),aware(row['exit_at']),aware(row['settled_at']),aware(entry['observed_at']),aware(exit['observed_at']))
        if available>cutoff:raise RuntimeError('Finalized experience was not actually available at the selector cutoff')
        actual=pair_return(decision['action'],entry['prices'],exit['prices'])
        if not _same_number(feedback.get('reward'),actual) or feedback.get('entry_prices')!=entry['prices'] or feedback.get('exit_prices')!=exit['prices']:
            raise RuntimeError('Native mature feedback differs from immutable quote evidence')
        symbols=(action['stock1'],action['stock2']);ref=f'w{index}'
        experiences.append({'case_id':ref,'release_at':row['release_at'],'generated_at':decision['generated_at'],'entry_at':row['entry_at'],'exit_at':row['exit_at'],'available_at':available.isoformat(),'action':copy.deepcopy(decision['action']),'entry_prices':{s:entry['prices'][s] for s in symbols},'exit_prices':{s:exit['prices'][s] for s in symbols},'step_return':actual,'decision_reason':action.get('reason')})
        proof['cases'][ref]={'native_generation':str(directory),'native_decision_sha256':digest(native_decision),'native_feedback_sha256':digest(feedback),'book_decision_sha256':digest(decision),'settlement_sha256':digest(row),'entry_valuation_sha256':digest(entry),'exit_valuation_sha256':digest(exit),'retrieved_evidence_ids':copy.deepcopy(native_decision.get('evidence_ids',[]))}
    context=validate_context({'schema':SCHEMA,'as_of':cutoff.isoformat(),'week_start':start.isoformat(),'observation_at':aware(observation_at).isoformat(),'known_labels':len(experiences),'weekly_return':risk_metrics(x['step_return'] for x in experiences),'finalized_memory':experiences})
    if any(file_hash(Path(p))!=sha for p,sha in proof['native_sources'].items()):raise RuntimeError('Native evidence changed during weekly context construction')
    proof['context_sha256']=digest(context)
    return context,proof
