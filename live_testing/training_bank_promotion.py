"""Pure, audited transfer of one trained RRI bank into its live namespace.

No embedding, model, file or network operation is performed. The original
training snapshot remains untouched. Imported knowledge becomes available only
at actual training completion, while every historical source cutoff and exit
remains unchanged. This supports honestly timed retrospective initialization.
"""
from __future__ import annotations

import copy
from .calendar import aware
from .memory import CounterfactualMemoryBank, Namespace
from .registry import TOP_K
from .storage import digest


def promote_training_bank(snapshot, *, model_id, completed_at, training_window_end,
                          live_source_cutoff, session_age, run_id='experiment', expected_units=29):
    """Return ``{bank, proof}``, retaining learned raw importance exactly.

    ``completed_at`` is the actual completion of paid historical training, never
    a fabricated old release time. Source markets and query vectors are frozen.
    Ranks are recomputed using the real source-exit guard after namespace/time
    transfer; any changed rank, score, future event or variant mismatch rejects.
    """
    if model_id not in TOP_K: raise ValueError('Only a registered RRI variant can be promoted')
    completed, source_end, live_cutoff = map(aware,(completed_at,training_window_end,live_source_cutoff))
    if not source_end < live_cutoff <= completed: raise ValueError('Historical training must precede the known live source cutoff')
    original=copy.deepcopy(snapshot);source_namespace=Namespace(**original['namespace'])
    if source_namespace.version!=model_id or source_namespace.scope!='training':
        raise ValueError('Training banks cannot cross model/version namespaces')
    if len(original['units'])!=expected_units: raise ValueError('The complete retained finalized episode is required')
    source=CounterfactualMemoryBank.from_dict(original,namespace=source_namespace,session_age=session_age)
    for item in original['units']:
        unit=item['unit']
        if aware(unit['exit_at'])>source_end or aware(unit['available_at'])>completed:
            raise ValueError('Future source outcomes or unavailable training evidence cannot be imported')
        if unit['provenance'].get('model_id',model_id)!=model_id:
            raise ValueError('A unit from another variant cannot be imported')
    for row in original['retrievals']:
        if (aware(row['as_of'])>completed or aware(row['source_cutoff'])>source_end
                or row['top_k']!=TOP_K[model_id]):
            raise ValueError('Future or cross-variant retrieval provenance rejected')
    for row in original['feedback']:
        if aware(row['exit_at'])>source_end or aware(row['available_at'])>completed:
            raise ValueError('Future outcome feedback cannot seed a live bank')
    target=copy.deepcopy(original);target_namespace=Namespace(model_id,'live',run_id)
    target['namespace']=vars(target_namespace);stamp=completed.isoformat();source_hash=digest(original)
    original_unit_times={};original_query_times={};original_feedback_times={}
    for item in target['units']:
        unit=item['unit'];original_unit_times[unit['unit_id']]=unit['available_at'];unit['available_at']=stamp
        unit['provenance']['trained_artifact_transfer']={
            'source_namespace':vars(source_namespace),'source_snapshot_sha256':source_hash,
            'source_unit_available_at':original_unit_times[unit['unit_id']],
            'learned_artifact_available_at':stamp,'embedding_preserved_from_training':True,
            'historical_source_dates_unchanged':True}
    for row in target['retrievals']:
        original_query_times[row['retrieval_id']]=row['as_of'];row['as_of']=stamp
    for row in target['feedback']:
        original_feedback_times[row['retrieval_id']]=row['available_at'];row['available_at']=stamp
    # This restores real vectors offline, replays only actual stored feedback,
    # and recomputes every causal historical score/order, without a provider.
    promoted=CounterfactualMemoryBank.from_dict(target,namespace=target_namespace,session_age=session_age)
    importance={}
    for item in original['units']:
        key=item['unit']['unit_id']
        old=source.importance_at(key,completed,source_cutoff=source_end)
        new=promoted.importance_at(key,completed,source_cutoff=live_cutoff)
        if old!=new: raise ValueError('Learned raw importance changed during live transfer')
        importance[key]=new
    return {'bank':promoted.to_dict(),'proof':{
        'schema':'moira.trained-bank-promotion/1','model_id':model_id,'source_snapshot_sha256':source_hash,
        'source_namespace':vars(source_namespace),'target_namespace':vars(target_namespace),
        'actual_training_completed_at':stamp,'training_window_end':source_end.isoformat(),
        'first_live_source_cutoff':live_cutoff.isoformat(),'unit_count':len(original['units']),
        'retrieval_count':len(original['retrievals']),'actual_feedback_count':len(original['feedback']),
        'original_unit_availability':original_unit_times,'original_retrieval_knowledge_times':original_query_times,
        'original_feedback_availability':original_feedback_times,'learned_raw_importance':importance,
        'causal_rank_recomputed':True,'all_raw_importance_preserved':True,'new_model_or_embedding_calls':0,
        'promoted_snapshot_sha256':digest(promoted.to_dict())}}
