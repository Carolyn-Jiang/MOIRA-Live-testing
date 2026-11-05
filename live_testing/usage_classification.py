"""Read-only, evidence-bound purposes for provider usage.

The catalog is private. Only aggregate counters enter the website projection;
unclassified and superseded requests remain in total usage and cost.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .calendar import aware
from .registry import LEGACY_BACKBONES, MODEL_IDS
from .storage import digest, file_hash, read_json

SCHEMA = 'moira.usage-classification/1'
CATEGORIES = ('direction_decision', 'pair_reselection', 'relationship_monitor',
              'loo_attribution', 'other_inference')
FIELDS = ('trading_tokens', *(name + '_tokens' for name in CATEGORIES))
TRAINING_STAGES = {'training', 'warmup'}


def _action(value):
    return {key: value.get(key) for key in ('stock1', 'stock2', 'action')}


def _jsonl(path):
    if not Path(path).is_file():
        return []
    result = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                result.append(value)
    return result


def _payload(wire):
    request = wire.get('request_payload', wire.get('request', {}))
    if not isinstance(request, dict):
        return None
    value = request.get('input')
    if value is None and isinstance(request.get('messages'), list):
        users = [row.get('content') for row in request['messages'] if row.get('role') == 'user']
        value = users[0] if len(users) == 1 else None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return None


def _instructions(wire):
    request = wire['request_payload']
    if isinstance(request.get('instructions'), str):
        return request['instructions']
    return '\n'.join(row.get('content', '') for row in request.get('messages', [])
                     if row.get('role') == 'system' and isinstance(row.get('content'), str))


def _response(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except ValueError:
            return {}
    return {}


def _same_direction(audit, decision):
    response = _response(audit.get('response'))
    action = decision['action']
    label = {'LONG_SHORT': 'LS', 'SHORT_LONG': 'SL', 'FLAT': 'F'}.get(response.get('action'))
    return (label == action.get('action') and
            response.get('direction_reason') == action.get('reason'))


def _observation_at(audit):
    payload = audit.get('payload') or {}
    observation = payload.get('current_observation', payload.get('observation', {}))
    return observation.get('timestamp') if isinstance(observation, dict) else None


def _within(row, start, end):
    try:
        return aware(start) <= aware(row['started_at']) <= aware(end)
    except (ValueError, TypeError, KeyError):
        return False


def _accepted_legacy_action(identity, row, model, stamp, decision):
    """Bind unknown legacy usage to the still-accepted original seal only.

    A replacement decision can share its release boundary with an old import,
    but that does not make the old, unmeasured request its trading receipt.
    """
    audit = decision.get('audit') or {}
    seal = audit.get('source_original_seal') if isinstance(audit, dict) else None
    if not isinstance(seal, dict) or not isinstance(seal.get('action'), dict):
        return False
    try:
        rid = aware(stamp).strftime('%Y-%m-%d_%H:%M')
        seal_sha256 = digest(seal)
        return (decision.get('mode') == 'live'
                and audit.get('input_protocol') == 'legacy_input_summary_protocol'
                and audit.get('source_round_id') == rid
                and audit.get('source_seal_sha256') == seal_sha256
                and seal.get('model') == LEGACY_BACKBONES[model]
                and _action(seal['action']) == _action(decision['action'])
                and aware(seal['sealed_at']) == aware(decision['generated_at'])
                and identity == f'legacy-action:{model}:{rid}:{seal_sha256}'
                and row.get('model_id') == model
                and row.get('stage') == 'inference'
                and row.get('source_protocol') == 'legacy_input_summary_protocol'
                and row.get('source_round_id') == rid
                and aware(row['started_at']) == aware(seal['sealed_at']))
    except (ValueError, TypeError, KeyError):
        return False


def build_usage_catalog(book, calls, *, evidence=(), unresolved=None):
    """Pure validation of sanitized, already-associated private evidence.

    ``calls`` is the normalized provider-call list, keyed by its canonical id.
    Direction/reselection evidence must bind to an unchanged accepted book row.
    A missed proof is explicit uncertainty, never a guessed trading request.
    """
    indexed = {row['id']: row for row in calls}
    claims = {}
    covered = defaultdict(set)
    for proof in evidence:
        identity, model, kind = (proof.get(key) for key in ('source_call_id', 'model_id', 'kind'))
        if identity not in indexed or model not in MODEL_IDS or kind not in CATEGORIES:
            raise ValueError('Unknown provider request or purpose in private usage evidence')
        row = indexed[identity]
        if row['model_id'] != model or row['stage'] in TRAINING_STAGES:
            raise ValueError('Usage evidence cannot change a model or training scope')
        if kind in ('direction_decision', 'pair_reselection'):
            accepted = book.get('decisions', {}).get(model, {}).get(proof.get('release_at'))
            if accepted is None or proof.get('decision_sha256') != digest(accepted):
                raise ValueError('Trading usage must bind to the current accepted decision')
            if kind == 'pair_reselection':
                at = aware(proof['release_at'])
                earlier = [d for stamp, d in book['decisions'][model].items() if aware(stamp) < at]
                prior = max(earlier, key=lambda d: aware(d['release_at']), default=None)
                if (at.hour, at.minute) != (15, 30) or prior is None or _action(prior['action']) == _action(accepted['action']):
                    raise ValueError('A trading reselection must be an accepted 15:30 pair change')
                if tuple(prior['action'].get(k) for k in ('stock1', 'stock2')) == tuple(accepted['action'].get(k) for k in ('stock1', 'stock2')):
                    raise ValueError('An action-only change is not pair reselection')
            covered[(model, kind)].add(proof['release_at'])
        if identity in claims and claims[identity] != {'model_id': model, 'kind': kind}:
            raise ValueError('One provider request cannot belong to two usage purposes')
        claims[identity] = {'model_id': model, 'kind': kind}
    missing = {model: {'direction_decision': [], 'pair_reselection': []} for model in MODEL_IDS}
    for model in MODEL_IDS:
        decisions = book.get('decisions', {}).get(model, {})
        missing[model]['direction_decision'] = sorted(set(decisions)-covered[(model, 'direction_decision')])
        if model.startswith('moira'):
            prior = None
            for stamp, decision in sorted(decisions.items(), key=lambda item: aware(item[0])):
                if (prior and tuple(prior['action'][key] for key in ('stock1', 'stock2'))
                        != tuple(decision['action'][key] for key in ('stock1', 'stock2'))
                        and stamp not in covered[(model, 'pair_reselection')]):
                    missing[model]['pair_reselection'].append(stamp)
                prior = decision
    for model, groups in (unresolved or {}).items():
        if model not in missing or set(groups) - {'direction_decision', 'pair_reselection'}:
            raise ValueError('Unknown unresolved trading proof')
        for kind, stamps in groups.items():
            missing[model][kind] = sorted(set(stamps) | set(missing[model][kind]))
    return {'schema': SCHEMA, 'book_sha256': digest(book), 'calls': claims, 'unresolved': missing}


def validate_catalog(catalog, book, calls, *, ledger=None):
    if (not isinstance(catalog, dict) or catalog.get('schema') != SCHEMA
            or catalog.get('book_sha256') != digest(book)):
        raise ValueError('Usage catalog must match the current accepted book')
    if catalog.get('ledger_sha256') is not None and (ledger is None or catalog['ledger_sha256'] != digest(ledger)):
        raise ValueError('Usage catalog provider ledger changed')
    indexed = {row['id']: row for row in calls}
    for identity, claim in catalog.get('calls', {}).items():
        if (identity not in indexed or claim.get('model_id') != indexed[identity]['model_id']
                or claim.get('kind') not in CATEGORIES or indexed[identity]['stage'] in TRAINING_STAGES):
            raise ValueError('Invalid private usage purpose catalog')
    if set(catalog.get('unresolved', {})) != set(MODEL_IDS):
        raise ValueError('Every model needs explicit trading proof coverage')
    return catalog


def build_catalog_from_state(root, book, ledger):
    """Read existing receipt/audit files; no SDK, credentials, or state writes.

    Missing or ambiguous receipts stay outside Trading and mark its proof
    incomplete. Provider phases alone never promote a request into Trading.
    """
    from .projection import _calls
    root = Path(root).resolve()
    calls = _calls(book, ledger, None)
    raw = {key: value for key, value in ledger.get('calls', {}).items()}
    wire = {}
    for identity, row in raw.items():
        path = root/'api-audit/http-requests'/f'{identity}.json'
        value = read_json(path)
        if (isinstance(value, dict) and digest(value.get('request_payload')) == row.get('request_sha256')
                and value.get('model_id') == row.get('model_id')
                and value.get('response_id') == row.get('response_id')):
            wire[identity] = value
    config = read_json(root/'config.json', {})
    legacy = Path(config['legacy_dir']) if config.get('legacy_dir') else None
    old = read_json(legacy/'native/api-budget.json', {}) if legacy else {}
    for item in book.get('imported_usage', []):
        identity = item['source_call_id']
        source_id = identity.rsplit(':', 1)[-1]
        source = old.get('calls', {}).get(source_id)
        if source and item.get('source_call_sha256') == digest(source):
            raw[identity] = {**source, 'model_id': item['model_id'], 'stage': item['stage']}
        elif identity.startswith('legacy-action:'):
            raw[identity] = item
    evidence = []
    unresolved = {model: {'direction_decision': [], 'pair_reselection': []} for model in MODEL_IDS}

    def claim(identity, model, kind, stamp=None):
        item = {'source_call_id': identity, 'model_id': model, 'kind': kind}
        if stamp is not None:
            item.update(release_at=stamp, decision_sha256=digest(book['decisions'][model][stamp]))
        evidence.append(item)

    def matches(audit, model):
        found = []
        for identity, row in raw.items():
            if row.get('model_id') != model or row.get('stage') in TRAINING_STAGES:
                continue
            if row.get('status') != 'completed':
                request = audit.get('request') or wire.get(identity, {}).get('request_payload')
                if (row.get('status') in ('failed', 'reserved', 'usage-unavailable')
                        and audit.get('request_sha256') == row.get('request_sha256')
                        and isinstance(request, dict) and digest(request) == row.get('request_sha256')
                        and _within(row, audit.get('started_at'), audit.get('generated_at'))):
                    found.append(identity)
                continue
            response_id = audit.get('response_id')
            if response_id:
                if response_id == row.get('response_id') and audit.get('request_sha256') == row.get('request_sha256'):
                    found.append(identity)
                continue
            actual = wire.get(identity)
            if (actual and _within(row, audit.get('started_at'), audit.get('generated_at'))
                    and digest(_payload(actual)) == audit.get('payload_sha256')):
                if _instructions(actual).startswith(audit.get('system_prompt', '\0')):
                    found.append(identity)
        return found

    for model in MODEL_IDS:
        accepted = book.get('decisions', {}).get(model, {})
        model_dir = root/'models'/model
        audits = _jsonl(model_dir/'model-calls.jsonl') if model.startswith('moira') else []
        if model.startswith('moira'):
            audits += [read_json(path) for path in sorted((model_dir/'calls').glob('*.json'))]
        native = read_json(model_dir/'live-state.json', {}) if model.startswith('moira') else {}
        # These categories require actual audited request/task association.
        associated_by_id = {}
        for audit in audits:
            if not isinstance(audit, dict) or audit.get('scope') != 'live':
                continue
            ids = matches(audit, model)
            if len(ids) != 1:
                continue
            identity = ids[0]
            associated_by_id[identity] = audit
            task, payload = audit.get('task_type'), audit.get('payload') or {}
            if task == 'daily_direction_leave_one_out':
                claim(identity, model, 'loo_attribution')
            elif task == 'critical_news_verification' or (task is None and 'statistical_deterioration' in payload):
                claim(identity, model, 'relationship_monitor')
        associated = [(audit, identity) for identity, audit in associated_by_id.items()]
        finmem_policy = []
        if model == 'finmem':
            epoch = read_json(model_dir/'active-epoch.json', {})
            directories = {d.get('audit', {}).get('native_generation') for d in accepted.values()}
            directories.add(epoch.get('directory'))
            for directory in sorted(value for value in directories if value):
                path = Path(directory).resolve()
                if path.is_relative_to(root) or (legacy and path.is_relative_to(legacy.resolve())):
                    finmem_policy.extend(_jsonl(path/'live-policy-attempts.jsonl'))
            # Old accepted native requests were audited before the current HTTP transport.
            old_finmem = _jsonl(legacy/'native/company-context-inference-audit.jsonl') if legacy else []
        gateway_audits = _jsonl(root/'api-audit/model-calls.jsonl') if model == 'codex' else []
        for stamp, decision in accepted.items():
            found = []
            if model.startswith('moira'):
                state_row = next((value for value in native.get('decisions', {}).values()
                    if value.get('timestamp') and aware(value['timestamp']) == aware(stamp)), None)
                if (state_row and _action(state_row.get('action', {})) == _action(decision['action'])
                        and aware(state_row['action']['generated_at']) == aware(decision['generated_at'])):
                    for audit, identity in associated:
                        if (audit.get('task_type') == 'daily_direction' and _observation_at(audit)
                                and aware(_observation_at(audit)) == aware(stamp)
                                and _same_direction(audit, state_row)
                                and 0 <= (aware(decision['generated_at'])-aware(audit['generated_at'])).total_seconds() < 5):
                            found.append(identity)
                    # The immutable v2 direction file also binds the exact receipt.
                    ref = state_row.get('direction_call')
                    if ref and (model_dir/ref['path']).is_file() and file_hash(model_dir/ref['path']) == ref.get('sha256'):
                        proof = read_json(model_dir/ref['path'])
                        if _same_direction(proof, state_row):
                            found.extend(matches(proof, model))
            elif model == 'finmem':
                attempts = [p for p in finmem_policy if p.get('round_id') == aware(stamp).strftime('%Y-%m-%d_%H:%M')
                            and p.get('accepted') is True and p.get('generated_at')
                            and aware(p['generated_at']) == aware(decision['generated_at'])
                            and p.get('returned_action') == decision['action'].get('action')]
                for attempt in attempts:
                    for identity, row in raw.items():
                        if (row.get('model_id') != model or row.get('requested_model', row.get('model')) != 'gpt-4.1-mini'
                                or row.get('status') != 'completed' or not row.get('response_id')
                                or not _within(row, attempt['started_at'], attempt['generated_at'])):
                            continue
                        actual = wire.get(identity, {}).get('request_payload', {})
                        native_schema = actual.get('response_format', {}).get('json_schema', {})
                        real_http = (native_schema.get('name') == 'finmem_pair_action'
                                     and bool(actual.get('messages')))
                        old_proof = any(audit.get('status') == 'completed'
                            and 'finmem' in str(audit.get('baseline', '')).lower()
                            and audit.get('response_id') == row.get('response_id')
                            and isinstance(audit.get('request_payload'), dict)
                            and set(audit['request_payload']) == {'prompt', 'schema'}
                            and digest(audit['request_payload']) == audit.get('request_payload_sha256')
                            and _within(row, audit['started_at'], audit.get('generated_at', attempt['generated_at']))
                            for audit in old_finmem)
                        if real_http or old_proof:
                            found.append(identity)
            else:
                for audit in gateway_audits:
                    payload = audit.get('payload') or {}; obs = payload.get('observation') or payload.get('current_observation') or {}
                    if (audit.get('model_id') == model and obs.get('timestamp') and aware(obs['timestamp']) == aware(stamp)
                            and _action(_response(audit.get('response'))) == _action(decision['action'])
                            and audit.get('generated_at') and 0 <= (aware(decision['generated_at'])-aware(audit['generated_at'])).total_seconds() < 5):
                        found.extend(identity for identity, row in raw.items() if row.get('model_id') == model
                            and row.get('response_id') == audit.get('response_id') and row.get('response_id'))
                # Unknown legacy usage belongs to Trading only while that exact
                # original seal remains accepted; replaced imports stay in Total.
                found.extend(identity for identity, row in raw.items()
                    if _accepted_legacy_action(identity, row, model, stamp, decision))
            found = sorted(set(found))
            if len(found) == 1:
                claim(found[0], model, 'direction_decision', stamp)
            else:
                unresolved[model]['direction_decision'].append(stamp)
            earlier = [d for at, d in accepted.items() if aware(at) < aware(stamp)]
            prior = max(earlier, key=lambda d: aware(d['release_at']), default=None)
            changed = prior and tuple(prior['action'][k] for k in ('stock1', 'stock2')) != tuple(decision['action'][k] for k in ('stock1', 'stock2'))
            if model.startswith('moira') and changed:
                selections = [(audit, identity) for audit, identity in associated if audit.get('task_type') == 'pair_selection'
                    and _response(audit.get('response')) and audit.get('status') != 'failed'
                    and _observation_at(audit) and aware(_observation_at(audit)) == aware(stamp)
                    and aware(audit['generated_at']) <= aware(decision['generated_at'])]
                if (aware(stamp).hour, aware(stamp).minute) == (15, 30) and len(selections) == 1:
                    claim(selections[0][1], model, 'pair_reselection', stamp)
                else:
                    unresolved[model]['pair_reselection'].append(stamp)
    result = build_usage_catalog(book, calls, evidence=evidence, unresolved=unresolved)
    result['ledger_sha256'] = digest(ledger)
    return result
