"""Transactional installation of three real variant-specific initial trainings.

No SDK/key is imported here. Initialization receives only already-trained
vectors, never an embedding provider. The original three models, accounting
book, decisions and dispatcher state are immutable throughout this operation.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .calendar import START_AT, aware
from .registry import TOP_K
from .storage import read_json, write_private, file_hash, digest, locked, now
from .initial_training import source_proof, phase_two_dataset


def _tree(root):
    result = {}
    for path in Path(root).rglob("*"):
        if path.is_symlink(): raise ValueError("Prepared model trees cannot contain symlinks")
        if path.is_file(): result[str(path.relative_to(root))] = file_hash(path)
    return result


def _preserved(root):
    result = {}
    for name in ("book.json", "runtime.json"):
        path = root/name
        if not path.is_file(): raise ValueError("An initialized isolated runtime is required")
        result[name] = file_hash(path)
    for model in ("codex", "finmem", "moira"):
        directory = root/"models"/model
        result["models/"+model] = _tree(directory) if directory.exists() else None
    return result


def _write_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix+".tmp")
    with os.fdopen(os.open(temporary, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, 0o600), "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path); path.chmod(0o600)


def validate_staged_results(root, training_directory):
    """Read-only proof of all three paid/native results, with no model operation."""
    root, training_directory = Path(root).resolve(), Path(training_directory).resolve()
    proof = source_proof(root)
    if len(proof) != 32: raise ValueError("All 32 original historical source files must be verified")
    bundle = read_json(root/"seed/moira-bundle.json")
    parent_sha = file_hash(root/"seed/moira-prepared.json")
    budget = read_json(root/"api-budget.json", {"calls": {}})
    results = {}
    for model in TOP_K:
        directory = training_directory/model
        status = read_json(directory/"initial-training-status.json")
        if not status or status.get("status") != "ready":
            raise RuntimeError("All three genuine initial training results must be ready")
        prepared = read_json(directory/"prepared.json"); report = read_json(directory/"training-result.json")
        seeds = read_json(directory/"seed-records.json"); bank = read_json(directory/"retained-training-bank.json")
        if (status.get("model_id") != model or status.get("backbone") != "gpt-4.1-mini"
                or prepared.get("model_id") != model or prepared.get("model") != "gpt-4.1-mini"
                or prepared.get("status") != "ready" or prepared.get("schema") != "moira.initial-rri-checkpoint/1"
                or status.get("checkpoint_sha256") != file_hash(directory/"prepared.json")
                or status.get("source_hashes_before") != proof or status.get("source_hashes_after") != proof
                or report.get("source_hashes_before") != proof or report.get("source_hashes_after") != proof
                or status.get("parent_checkpoint_sha256") != parent_sha or prepared.get("parent_checkpoint_sha256") != parent_sha
                or read_json(directory/"frozen-training-inputs.json") != bundle):
            raise ValueError("Initial training model, checkpoint or immutable source lineage differs")
        adapters = status.get("adapter_source_hashes", {})
        if len(adapters) != 4 or {Path(path).name for path in adapters} != {"initial_training.py", "moira_native.py", "memory.py", "training.py"}:
            raise ValueError("All four actual native training adapter hashes are required")
        for path, expected in adapters.items():
            if file_hash(path) != expected: raise ValueError("The actual training adapter changed")
        initial = prepared["initial_rri_training"]
        cycles = status.get("completed_cycles")
        cap = status.get("max_cycles")
        if (type(cap) is not int or not 1 <= cap <= 8 or type(cycles) is not int or not 1 <= cycles <= cap
                or report.get("completed_cycles") != cycles or len(report.get("cycles", [])) != cycles
                or report.get("max_cycles") != cap or initial.get("completed_cycles") != cycles or initial.get("max_cycles") != cap
                or initial.get("memory_top_k") != TOP_K[model] or initial.get("tau_sessions") != 252
                or initial.get("feedback_step") != 18 or status.get("actual_observations") != 29
                or status.get("actual_matured_labels") != 29):
            raise ValueError("Variant-specific actual cycle/observation evidence is incomplete")
        started, finished = aware(status["started_at"]), aware(status["finished_at"])
        if (not started <= finished <= datetime.now(timezone.utc)
                or aware(prepared["finished_at"]) != finished or aware(report["finished_at"]) != finished):
            raise ValueError("Actual training completion time is invalid or backdated")
        dataset = phase_two_dataset(bundle, training_at=started)
        if (aware(status["source_window_start"]) != dataset.window_start
                or aware(status["source_window_end"]) != dataset.window_end
                or not dataset.window_end < START_AT):
            raise ValueError("Only the authorized complete October 19–23 source window is allowed")
        active = prepared["prompt_memory"]["active_prompt"]
        if hashlib.sha256(active.encode()).hexdigest() != prepared["frozen_prompt_sha256"]:
            raise ValueError("Learned prompt hash mismatch")
        for key in ("model", "status", "prompt_memory", "pair_memory", "frozen_prompt_sha256"):
            if prepared[key] != report["checkpoint"][key]: raise ValueError("Saved learned checkpoint differs from actual training result")
        if (len(seeds) != 29 or status.get("seed_unit_count") != 29
                or file_hash(directory/"seed-records.json") != status.get("seed_records_sha256")
                or file_hash(directory/"retained-training-bank.json") != status.get("retained_training_bank_sha256")
                or bank.get("namespace", {}).get("version") != model or bank.get("namespace", {}).get("scope") != "training"
                or len(bank.get("units", [])) != 29):
            raise ValueError("Complete trained RRI units and bank proof required")
        if any(aware(seed["available_at"]) != finished
               or seed.get("provenance", {}).get("initial_training_model") != model for seed in seeds):
            raise ValueError("Seed knowledge must become available at actual variant training completion")
        calls = [row for row in budget["calls"].values() if row.get("model_id") == model
                 and row.get("stage") == "training" and row.get("event_id") == "initial-historical-rri"]
        if (not calls or len(calls) != status.get("actual_api_calls")
                or status.get("actual_completed_calls") != len(calls) or status.get("actual_failed_calls") != 0
                or any(row.get("status") != "completed" for row in calls)
                or not any(row.get("requested_model") == "gpt-4.1-mini" for row in calls)):
            raise ValueError("Actual successful version-specific native calls must back every training claim")
        results[model] = {"directory": directory, "status": status, "prepared": prepared,
            "prepared_bytes": (directory/"prepared.json").read_bytes(), "report": report,
            "seed_records": seeds, "bank": bank, "finished": finished,
            "training_window_end": dataset.window_end}
    if source_proof(root) != proof: raise ValueError("A frozen historical source advanced during validation")
    return bundle, results, proof


class CachedTrainingVectors:
    """Explicit vector aliases after changing availability/audit metadata only."""
    def __init__(self, model, seeds, snapshot):
        from .moira_native import _unit
        self.aliases, self.proof = {}, []
        originals = {item["unit"]["unit_id"]: item for item in snapshot["units"]}
        if {seed["unit_id"] for seed in seeds} != set(originals):
            raise ValueError("Initialization seeds do not match the actual retained training units")
        for seed in seeds:
            raw = originals[seed["unit_id"]]; original = _unit(raw["unit"]).to_prompt(); current = _unit(seed).to_prompt()
            for key in set(original) - {"available_at", "provenance"}:
                if original[key] != current[key]: raise ValueError("Cached vector alias changed semantic training evidence")
            if any(current["provenance"].get(key) != value for key, value in original["provenance"].items()):
                raise ValueError("Original trained-unit provenance must be retained")
            if current["provenance"].get("initial_training_model") != model:
                raise ValueError("A seed from another trained variant is not allowed")
            vector = raw["vector"]
            if (not isinstance(vector, list) or not vector or any(type(x) not in (int,float) or not math.isfinite(x) for x in vector)
                    or not any(vector)): raise ValueError("A real finite nonzero trained vector is required")
            text = json.dumps(current, sort_keys=True)
            self.aliases[text] = copy.deepcopy(vector)
            self.proof.append({"unit_id": seed["unit_id"], "initialization_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "trained_vector_sha256": digest(vector), "semantic_fields_unchanged": True,
                "availability_and_provenance_rebound_only": True, "new_embedding_calls": 0})

    def embed(self, text):
        if text not in self.aliases: raise RuntimeError("Uncached initialization embedding requested; no provider is available")
        return copy.deepcopy(self.aliases[text])


def _already_installed(root, marker):
    if marker.get("status") != "ready": raise RuntimeError("An incomplete installation requires rollback review")
    for model in TOP_K:
        target = root/"models"/model; ready = read_json(target/"initial-training-ready.json")
        if (not ready or ready.get("status") != "ready" or ready.get("model_id") != model
                or ready.get("backbone") != "gpt-4.1-mini" or ready.get("cycles_completed", 0) < 1
                or ready.get("checkpoint_sha256") != file_hash(target/"initial-training-checkpoint.json")
                or marker["checkpoints"].get(model) != ready["checkpoint_sha256"]):
            raise ValueError("An installed immutable initial training proof changed")
    permission = read_json(root/"preparation-replay-authorization.json")
    if (not permission or permission.get("status") != "authorized" or permission.get("date") != START_AT.date().isoformat()
            or aware(permission["training_completion_cutoff"]) != aware(marker["training_completion_cutoff"])):
        raise ValueError("The truthful training-completion replay boundary changed")
    return copy.deepcopy(marker)


def install_initial_training(root, training_directory):
    """Explicit installer only; caller must wait for and authorize all results."""
    root, training_directory = Path(root).resolve(), Path(training_directory).resolve()
    with locked(root/"run.lock", nonblocking=True):
        marker_path = root/"initial-training-install.json"
        existing = read_json(marker_path)
        if existing is not None: return _already_installed(root, existing)
        book, runtime = read_json(root/"book.json"), read_json(root/"runtime.json")
        for model in TOP_K:
            live = read_json(root/"models"/model/"live-state.json")
            if (not isinstance(live,dict) or book["decisions"].get(model) or live.get("decisions") or live.get("attempts")
                    or live.get("active_pair") is not None
                    or any(key.startswith(model+":") for key in runtime.get("model_attempts", {}))):
                raise ValueError("A variant with an existing live decision/attempt cannot be replaced")
        bundle, results, sources = validate_staged_results(root, training_directory)
        preserved = _preserved(root)
        seed_trees = {model:_tree(root/"models"/model) for model in TOP_K}
        transaction = root/"initial-install"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+uuid.uuid4().hex[:8])
        transaction.mkdir(parents=True, mode=0o700)
        config = read_json(root/"config.json", {})
        if config.get("framework_root"): os.environ["PAIR_TRADING_FIX_ROOT"] = config["framework_root"]
        from . import moira_native
        from .training_bank_promotion import promote_training_bank
        for model, result in results.items():
            target = transaction/"staging"/model
            vectors = CachedTrainingVectors(model, result["seed_records"], result["bank"])
            live = moira_native.initialize(result["prepared"], bundle, target, result["seed_records"],
                vectors, model_id=model, scope="live", run_id="experiment")
            promotion = promote_training_bank(result["bank"], model_id=model, completed_at=result["finished"],
                training_window_end=result["training_window_end"], live_source_cutoff=START_AT,
                session_age=moira_native.session_age, run_id="experiment", expected_units=29)
            if live.get("active_pair") is not None or live.get("decisions") or live.get("attempts"):
                raise ValueError("Initialization must not create a live pair or action")
            live["memory_bank"] = promotion["bank"]
            write_private(target/"live-state.json", live)
            _write_bytes(target/"initial-training-checkpoint.json", result["prepared_bytes"])
            proof = {**promotion["proof"], "initialization_cached_vector_aliases": vectors.proof}
            write_private(target/"initial-training-bank-promotion.json", proof)
            write_private(target/"initial-training-ready.json", {"status":"ready", "model_id":model,
                "backbone":"gpt-4.1-mini", "cycles_completed":result["status"]["completed_cycles"],
                "checkpoint_sha256":file_hash(target/"initial-training-checkpoint.json"),
                "training_completed_at":result["finished"].isoformat(), "installed_at":now(),
                "source_status_sha256":file_hash(result["directory"]/"initial-training-status.json"),
                "bank_promotion_sha256":file_hash(target/"initial-training-bank-promotion.json")})
        if _preserved(root) != preserved or any(_tree(root/"models"/model)!=seed_trees[model] for model in TOP_K):
            raise RuntimeError("Original accounts or models changed during staging")
        permission_path = root/"preparation-replay-authorization.json"
        old_permission = permission_path.read_bytes() if permission_path.exists() else None
        cutoff = max(result["finished"] for result in results.values())
        permission = read_json(permission_path, {})
        permission.update(status="authorized", date=START_AT.date().isoformat(),
            training_completion_cutoff=cutoff.isoformat(), created_at=permission.get("created_at", now()),
            authorization=permission.get("authorization", "User explicitly authorized genuine variant-specific initial training followed by point-in-time filling of already-frozen November 2 preparation gaps; real model generation times remain visible privately."),
            scope="Only November 2 release boundaries at/before actual training completion with an already-published immutable cached observation.")
        plan = {"status":"committing", "transaction":str(transaction), "source_directory":str(training_directory),
            "training_completion_cutoff":cutoff.isoformat(), "preserved_before":preserved,
            "source_sha256":sources, "archived_initial_seed_sha256":seed_trees, "new_api_or_embedding_calls":0}
        write_private(marker_path, plan)
        moved_old, moved_new = [], []
        try:
            for model in TOP_K:
                previous = transaction/"before"/model; previous.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                (root/"models"/model).rename(previous); moved_old.append(model)
            for model in TOP_K:
                (transaction/"staging"/model).rename(root/"models"/model); moved_new.append(model)
            write_private(permission_path, permission)
            if _preserved(root) != preserved or source_proof(root) != sources:
                raise RuntimeError("Original model/book/runtime or historical source changed during commit")
            if any(_tree(transaction/"before"/model)!=seed_trees[model] for model in TOP_K):
                raise RuntimeError("The exact original variant initialization archive changed")
            plan.update(status="ready", installed_at=now(), checkpoints={model:file_hash(root/"models"/model/"initial-training-checkpoint.json") for model in TOP_K},
                preserved_after=_preserved(root), all_original_state_preserved=True)
            write_private(marker_path, plan)
        except Exception:
            for model in reversed(moved_new):
                failed = transaction/"failed-new"/model; failed.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                (root/"models"/model).rename(failed)
            for model in reversed(moved_old): (transaction/"before"/model).rename(root/"models"/model)
            if old_permission is None:
                if permission_path.exists(): permission_path.unlink()
            else: _write_bytes(permission_path, old_permission)
            plan.update(status="rolled_back", rolled_back_at=now(), original_state_restored=_preserved(root)==preserved)
            write_private(marker_path, plan); raise
        return copy.deepcopy(plan)
