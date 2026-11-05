"""Explicit, read-only import of two genuine November 2 legacy decisions.

No legacy path is implicit. No model/checkpoint/Qdrant is loaded or copied.
Only the caller's new in-memory accounting book is mutated, atomically after
source hashes have been checked again. The caller owns new-state persistence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .accounting import record_decision, record_valuation, validate_book
from .calendar import DEFAULT_CALENDAR, aware
from .registry import LEGACY_BACKBONES, TICKERS

ORIGINAL_MODELS = {"codex": "Codex", "finmem": "FinMem", "moira": "Moira"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class _Source:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("An explicit legacy phase directory is required")
        self.hashes = {}

    def read(self, relative):
        path = self.root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Legacy sources must be regular files inside the explicit directory")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if relative in self.hashes and self.hashes[relative] != digest:
            raise ValueError("A legacy source changed while importing")
        self.hashes[relative] = digest
        return payload

    def json(self, relative):
        return json.loads(self.read(relative))

    def unchanged(self):
        return all(hashlib.sha256((self.root / relative).read_bytes()).hexdigest() == digest
                   for relative, digest in self.hashes.items())


def _provider_counts(row):
    usage = row.get("usage") or {}
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    # Embedding responses explicitly contain a prompt/total count and no output.
    if str(row.get("model", "")).startswith("text-embedding-") and outgoing is None:
        outgoing = 0
    for value in (incoming, outgoing, cached):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Malformed measured legacy provider usage")
    if incoming is not None and cached is not None and cached > incoming:
        raise ValueError("Legacy cached input exceeds total input")
    return incoming, cached, outgoing


def _import_usage(source, state, releases, snapshot_at):
    budget_path = source.root / "native/api-budget.json"
    budget = source.json("native/api-budget.json") if budget_path.is_file() else {"calls": {}}
    result = []
    for model_id, old_name in ORIGINAL_MODELS.items():
        runtime_path = source.root / f"native/{model_id}/runtime-records.json"
        runtime = source.json(f"native/{model_id}/runtime-records.json") if runtime_path.is_file() else None
        attempts = {} if runtime is None else runtime.get("attempts", {})
        starts = sorted(aware(item["started_at"]) for item in attempts.values())
        for release in releases:
            rid = release.strftime("%Y-%m-%d_%H:%M")
            seal = state["sealed"][rid][old_name]
            attempt = attempts.get(rid)
            matched = []
            if model_id != "codex" and attempt and attempt.get("status") == "complete":
                start = aware(attempt["started_at"])
                end = next((at for at in starts if at > start), snapshot_at)
                if not start <= aware(seal["sealed_at"]) < end:
                    raise ValueError("A legacy accepted seal is outside its actual native attempt")
                phases = {"moira-live"} if model_id == "moira" else {"finmem-live", "finmem-live-initial-selection"}
                matched = [(identity, row) for identity, row in budget["calls"].items()
                           if row.get("phase") in phases and start <= aware(row["started_at"]) < end]
            if not matched:
                result.append({"source_call_id": "legacy-action:" + model_id + ":" + rid + ":" + _digest(seal),
                    "model_id": model_id, "stage": "inference", "started_at": seal["sealed_at"],
                    "status": "usage-unavailable", "input_tokens": None, "cached_input_tokens": None,
                    "output_tokens": None, "total_tokens": None,
                    "source_protocol": "legacy_input_summary_protocol", "source_round_id": rid})
            for identity, row in matched:
                incoming, cached, outgoing = _provider_counts(row)
                measured = row.get("status") == "completed" and incoming is not None and outgoing is not None
                result.append({"source_call_id": "legacy-provider:" + source.hashes["native/api-budget.json"] + ":" + identity,
                    "model_id": model_id, "stage": "inference", "started_at": row["started_at"],
                    "status": "completed" if measured else "usage-unavailable",
                    "input_tokens": incoming if measured else None, "cached_input_tokens": cached if measured else None,
                    "output_tokens": outgoing if measured else None,
                    "total_tokens": incoming + outgoing if measured else None,
                    "actual_cost_usd": row.get("actual_cost_usd"), "source_call_sha256": _digest(row),
                    "source_protocol": "legacy_input_summary_protocol", "source_round_id": rid})
    return result


def import_legacy_actions(book, legacy_phase3_directory, calendar=DEFAULT_CALENDAR, limit=2):
    """Import up to two exact original actions; three Moira2 gaps stay pending.

    Returns a private source-proof report. Never relabel old ticker-level input
    summaries as newly generated per-article summaries. Original generation
    times stay live; no model is invoked and no old source is modified.
    """
    if type(limit) is not int or not 1 <= limit <= 2:
        raise ValueError("Only the explicitly authorized first two actions may be copied")
    from .pair_lifecycle import validate_pair_lifecycle,validate_legacy_moira_seals
    validate_book(book, calendar=calendar)
    if 'moira' in book['decisions']:
        validate_pair_lifecycle('moira',book['decisions']['moira'],calendar=calendar)
    source = _Source(legacy_phase3_directory)
    previous = book.get("legacy_import")
    if previous is not None:
        if previous.get("legacy_directory") != str(source.root) or previous.get("limit") != limit:
            raise ValueError("An existing immutable legacy import cannot be replaced")
        return copy.deepcopy(previous)
    if set(book["symbols"]) != set(TICKERS) or not set(ORIGINAL_MODELS) <= set(book["models"]):
        raise ValueError("The existing ten-stock universe and original three accounts are required")
    snapshot_at = datetime.now(timezone.utc)
    state = source.json("state.json")
    if state.get("phase_id") != "phrase-iii-2026-11-02" or state.get("schema") != "finreason.task2.phrase3-private/1":
        raise ValueError("Only the explicitly selected November 2 legacy phase can be imported")
    releases = calendar.release_nodes[:limit]
    if not releases or releases[0].isoformat() != "2026-11-02T09:30:00-05:00":
        raise ValueError("Copied source actions must match the new experiment's initial release grid")
    validate_legacy_moira_seals(state,limit=limit,calendar=calendar)
    candidate = copy.deepcopy(book)
    copied, records = 0, []
    for i, at in enumerate(releases):
        rid = at.strftime("%Y-%m-%d_%H:%M")
        published = state["published"][rid]
        published_at = aware(published["published_at"])
        if not at <= published_at <= snapshot_at:
            raise ValueError("Legacy data was not genuinely available after its boundary")
        filename = f"release-{i+1}.json"
        release = source.json(filename)
        if release.get("timestamp") != at.isoformat() or len(release.get("price_rows", [])) != len(TICKERS):
            raise ValueError("Legacy cached observation does not match its original release")
        rows = release["price_rows"]
        target = at if (at.hour, at.minute) == (9, 30) else at - timedelta(minutes=1)
        basis = "09:30 minute open" if (at.hour, at.minute) == (9, 30) else "previous minute close"
        if ({row["ticker"] for row in rows} != set(TICKERS)
                or {row["ticker"]: row["price"] for row in rows} != release["prices"]
                or any(row.get("status") != "available" or aware(row["timestamp"]) != at
                       or row.get("price_basis") != basis or aware(row["bar_start"]) != target
                       for row in rows)):
            raise ValueError("Complete exact original prices are required")
        file_proofs = {}
        for kind in ("price", "news"):
            url_path = Path(urlsplit(published[kind + "_url"]).path)
            basename = unquote(url_path.name)
            if not basename or basename in {".", ".."} or "/" in basename or "\\" in basename:
                raise ValueError("Only a local published Parquet basename may be read")
            local_name = basename
            if not (source.root / local_name).is_file():
                # The original publisher stores flat timestamp-prefixed local
                # files but publishes folder/price.parquet and folder/news.parquet.
                # Read that exact mapping; never create aliases in the old project.
                folder = unquote(url_path.parent.name)
                if (basename != kind + ".parquet" or not folder or folder in {".", ".."}
                        or "/" in folder or "\\" in folder):
                    raise ValueError("No verified local mapping for the published Parquet URL")
                local_name = folder + "_" + basename
            source.read(local_name)
            if source.hashes[local_name] != published["file_sha256"][kind]:
                raise ValueError("Legacy published Parquet bytes changed")
            file_proofs[kind] = {"filename": local_name, "sha256": source.hashes[local_name]}
        # Validate the JSON prices against the exact bytes published originally.
        import pandas as pd
        price_frame = pd.read_parquet(source.root / file_proofs["price"]["filename"])
        if price_frame.to_dict("records") != rows:
            raise ValueError("Cached JSON prices differ from originally published Parquet")
        proof = {"source_release_sha256": source.hashes[filename],
            "source_state_sha256": source.hashes["state.json"], "source_manifest_sha256": _digest(published),
            "source_published_at": published["published_at"], "source_round_id": rid,
            "input_protocol": "legacy_input_summary_protocol", "published_file_proofs": file_proofs}
        record_valuation(candidate, at, release["prices"], observed_at=published_at,
            provenance={**proof, "availability_basis": "original publication time; conservative measured availability"}, calendar=calendar)
        for model_id, old_name in ORIGINAL_MODELS.items():
            seal = state["sealed"][rid][old_name]
            generated = aware(seal["sealed_at"])
            if seal.get("model") != LEGACY_BACKBONES[model_id] or not published_at <= generated < calendar.action_window(at).entry_at:
                raise ValueError("Copied baseline must retain a genuine timely original response")
            record_decision(candidate, model_id, at, seal["action"], generated_at=generated, input_asof=at,
                mode="live", audit={**proof, "source_seal_sha256": _digest(seal),
                    "source_original_seal": copy.deepcopy(seal)}, calendar=calendar)
            copied += 1
        records.append(proof)
    imported_usage = _import_usage(source, state, releases, snapshot_at)
    if not source.unchanged():
        raise ValueError("Legacy source advanced during import; no new state was committed")
    report = {"schema": "moira.live-testing.legacy-import/1", "legacy_directory": str(source.root),
        "limit": limit, "imported_at": snapshot_at.isoformat(), "copied_decisions": copied,
        "copied_models": list(ORIGINAL_MODELS), "pending_models": [model for model in book["models"] if model not in ORIGINAL_MODELS],
        "input_protocol": "legacy_input_summary_protocol", "sources": records,
        "source_file_sha256": copy.deepcopy(source.hashes), "legacy_files_unchanged": True}
    candidate["legacy_import"] = report
    existing_usage = {row["source_call_id"]: row for row in candidate.get("imported_usage", [])}
    for row in imported_usage:
        if row["source_call_id"] in existing_usage and existing_usage[row["source_call_id"]] != row:
            raise ValueError("Imported measured usage changed")
        existing_usage[row["source_call_id"]] = row
    candidate["imported_usage"] = list(existing_usage.values())
    validate_book(candidate, calendar=calendar)
    validate_pair_lifecycle('moira',candidate['decisions']['moira'],calendar=calendar)
    book.clear(); book.update(candidate)
    return copy.deepcopy(report)
