"""Compact model views with complete, immutable local source projections.

This adapter changes future wire inputs only. It never edits an observation,
checkpoint, retrieved unit, exact saved request, or counterfactual LOO request.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .native_source import operations_for
from .storage import digest, file_hash, locked, now, read_json, write_private

PROTOCOL = 'moira.prompt-source-projection/1'


def project_payload(payload, observation, *, gateway, directory, task_type=None, live=True):
    """Return the future prompt payload; retain full provenance privately.

    Selection may consider the eligible universe. Once a pair is supplied,
    company and observation context is limited to those two actual tickers.
    Training round reflection receives finalized CF evidence, no market news.
    """
    from .prompt_projection import (compact_observation, compact_payload_news,
        observation_reference_map, company_reference_map,
        compact_native_prompt, native_prompt_reference_map)

    original = copy.deepcopy(payload)
    reflection = 'counterfactual_summary' in original
    selected = None if task_type == 'pair_selection' else original.get('active_pair', original.get('pair'))
    if task_type == 'daily_direction' and selected is None:
        raise ValueError('A direction prompt must identify its already-selected pair')
    projected = compact_payload_news(original)
    if 'native_prompt' in original:
        projected['native_prompt'] = compact_native_prompt(original['native_prompt'])
    # R/V/I clocks, scores and update parameters remain in the actual retrieval
    # audit. The model receives the selected full experiences and their IDs.
    projected.pop('memory_policy', None)
    company_document = None
    if not reflection:
        projected['current_observation'] = compact_observation(observation, tickers=selected)
        if live:
            context_path = Path(gateway.budget.root)/'company-context.json'
            before = file_hash(context_path)
            company_document = read_json(context_path)
            projected['company_operations_context'] = operations_for(gateway.budget, observation, tickers=selected)
            if before != file_hash(context_path):
                raise RuntimeError('Company evidence changed during prompt projection')
    proof = {'schema': PROTOCOL, 'task_type': task_type,
        'original_payload': original, 'original_observation': copy.deepcopy(observation),
        'projected_payload_sha256': digest(projected),
        'news_references': observation_reference_map(observation, tickers=selected) if not reflection else {},
        'native_prompt_references': native_prompt_reference_map(original['native_prompt']) if 'native_prompt' in original else {},
        'company_context_sha256': digest(company_document) if company_document is not None else None,
        'company_references': company_reference_map(company_document, tickers=selected) if company_document is not None else {}}
    directory = Path(directory)
    # Complete company source inventory is stored once per content hash, outside
    # the model input. This also survives the next weekly context replacement.
    if company_document is not None:
        path = Path(gateway.budget.root)/'prompt-source-archive'/f'{digest(company_document)}.json'
        with locked(path.with_suffix('.lock')):
            old = read_json(path)
            if old is None: write_private(path, company_document)
            elif old != company_document: raise ValueError('Immutable company source archive changed')
    identity = digest(proof)
    path = directory/'prompt-source-projections'/f'{identity}.json'
    with locked(path.with_suffix('.lock')):
        old = read_json(path)
        if old is None: write_private(path, {**proof, 'recorded_at': now()})
        elif {k:v for k,v in old.items() if k != 'recorded_at'} != proof:
            raise ValueError('An immutable local prompt source projection changed')
    return projected
