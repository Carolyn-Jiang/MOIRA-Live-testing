"""Read-only readiness validation for a v2 model; an invalid v2 never falls back."""
import hashlib
import math
from pathlib import Path

from .memory_v2 import PROTOCOL_VERSION
from .registry import TOP_K
from .storage import file_hash, read_json


def _finite(value):
    return value is None or type(value) in (int, float) and math.isfinite(value)


def ready(root, model_id):
    """Return False only for an actual legacy model; raise on partial/invalid v2.

    This reads no key, constructs no gateway and changes no state.  The result
    is deliberately not cached: weekly training changes its frozen checkpoint.
    """
    if model_id not in TOP_K: return False
    root = Path(root).resolve(); directory = root/'models'/model_id
    marker = read_json(directory/'memory-policy.json')
    prepared = read_json(directory/'prepared.json')
    policy = prepared.get('memory_policy_protocol') if isinstance(prepared, dict) else None
    prepared_v2 = isinstance(prepared, dict) and ('memory_policy_protocol' in prepared
        or prepared.get('schema') == 'moira.quarter-ranking-checkpoint/2' or 'ranking_v2_training' in prepared)
    if marker is None and not prepared_v2:
        # A lost prepared/marker must not route an existing schema2 native book
        # through legacy inference merely because both small files disappeared.
        state = read_json(directory/'live-state.json')
        state_v2 = isinstance(state, dict) and ('memory_policy_protocol' in state
            or state.get('schema') == 'moira.live-testing.native/2'
            or isinstance(state.get('memory_bank'), dict)
               and state['memory_bank'].get('schema') == 'moira.counterfactual-memory/2')
        if not state_v2:
            return False
        raise RuntimeError('A v2 native book lacks its complete ready checkpoint marker')
    if (not isinstance(marker, dict) or marker.get('protocol') != PROTOCOL_VERSION
            or marker.get('status') != 'ready' or marker.get('model_id') != model_id
            or not isinstance(prepared, dict) or policy != PROTOCOL_VERSION
            or prepared.get('status') != 'ready' or prepared.get('model') != 'gpt-4.1-mini'
            or prepared.get('model_id') != model_id):
        raise RuntimeError('A complete ready v2 checkpoint and matching model marker are required')
    prompt = prepared.get('prompt_memory', {}).get('active_prompt')
    if (not isinstance(prompt, str) or not prompt.strip()
            or hashlib.sha256(prompt.encode()).hexdigest() != prepared.get('frozen_prompt_sha256')
            or marker.get('checkpoint_sha256') != file_hash(directory/'prepared.json')):
        raise RuntimeError('The frozen v2 prompt/checkpoint differs from its ready marker')
    value = marker.get('training_result_path')
    if not isinstance(value, str): raise RuntimeError('The actual immutable v2 training report is required')
    path = Path(value)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or root not in path.resolve().parents or path.name != 'training-result.json'
            or marker.get('training_result_sha256') != file_hash(path)):
        raise RuntimeError('The pinned private v2 training result is missing or changed')
    report = read_json(path)
    if not isinstance(report, dict): raise RuntimeError('A complete v2 training report is required')
    count, limit = report.get('completed_cycles'), report.get('max_cycles')
    cycles = report.get('cycles')
    checkpoint = report.get('checkpoint', {})
    if (report.get('objective') != 'sample_sharpe_without_annualization'
            or type(count) is not int or type(limit) is not int or not 1 <= count <= limit <= 8
            or not isinstance(cycles, list) or len(cycles) != count
            or not isinstance(checkpoint, dict) or checkpoint.get('model') != 'gpt-4.1-mini'
            or checkpoint.get('model_id') != model_id
            or checkpoint.get('memory_policy_protocol') != PROTOCOL_VERSION
            or checkpoint.get('frozen_prompt_sha256') != prepared['frozen_prompt_sha256']
            or checkpoint.get('prompt_memory') != prepared['prompt_memory']
            or not isinstance(prepared.get('pair_memory'), list)
            or checkpoint.get('pair_memory') != prepared['pair_memory']):
        raise RuntimeError('The ready v2 checkpoint must match its genuine maximum-eight Sharpe training result')
    for cycle in cycles:
        if not isinstance(cycle, dict) or type(cycle.get('accepted')) is not bool:
            raise RuntimeError('A genuine v2 training acceptance record is required')
        baseline, candidate = cycle.get('baseline_metrics'), cycle.get('candidate_metrics')
        if not isinstance(baseline, dict) or candidate is not None and not isinstance(candidate, dict):
            raise RuntimeError('Actual v2 baseline/candidate metrics are required')
        before = baseline.get('sharpe_ratio')
        after = candidate.get('sharpe_ratio') if candidate is not None else None
        if (not _finite(before) or not _finite(after)
                or cycle['accepted'] != (before is not None and after is not None and after > before)):
            raise RuntimeError('A v2 update was not accepted by strict finite sample-Sharpe improvement')
    return True
