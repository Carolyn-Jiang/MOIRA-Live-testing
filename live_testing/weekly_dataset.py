"""Holiday-aware weekly inputs without changing the sealed training engine.

The original training.py remains byte-for-byte unchanged. This compatible
dataset retains its complete canonical-price validation, with the authorized
Monday-or-Tuesday 08:00 cutoff and the same preceding NY Monday–Sunday window.
There are no model, clock, file, or network operations in this module.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import math

from .calendar import aware, week_bounds, DEFAULT_CALENDAR, is_weekly_training_time
from .training import WeeklyDataset as FrozenWeeklyDataset


class WeeklyDataset(FrozenWeeklyDataset):
    """Drop-in native training dataset using the actual authorized weekly slot."""

    @classmethod
    def build(cls, observations, executions, *, training_at):
        cutoff = aware(training_at)
        if not is_weekly_training_time(cutoff):
            raise ValueError("Weekly training requires Monday 08:00 NY, or Tuesday 08:00 only after a closed Monday")
        current_week, _ = week_bounds(cutoff)
        previous_week, _ = week_bounds(current_week-timedelta(days=1))
        rows = tuple(copy.deepcopy(row) for row in observations
                     if previous_week <= aware(row.timestamp) < current_week)
        if not rows or tuple(sorted(row.timestamp for row in rows)) != tuple(row.timestamp for row in rows):
            raise ValueError("Chronological observations from the previous week are required")
        if len({row.action_id for row in rows}) != len(rows) or len({row.timestamp for row in rows}) != len(rows):
            raise ValueError("Duplicate historical observation")
        for row in rows:
            DEFAULT_CALENDAR.action_window(row.timestamp)
            if (not row.action_id or aware(row.available_at) < aware(row.timestamp) or aware(row.available_at) > cutoff
                    or not row.prices or set(row.prices) != set(row.news)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in row.prices.values())
                    or any(not isinstance(v, list) or not all(isinstance(s, str) for s in v) for v in row.news.values())):
                raise ValueError("Complete, available historical prices/news required")
        known, ids = {}, {row.action_id: row for row in rows}
        prices_at = {aware(row.timestamp): row.prices for row in rows}
        for item in executions:
            if item.action_id not in ids:
                continue
            if aware(item.exit_at) > cutoff or aware(item.available_at) > cutoff:
                continue  # Unavailable labels stay pending, never synthetic zero.
            if item.action_id in known:
                raise ValueError("Duplicate outcome label")
            if not ids[item.action_id].timestamp <= aware(item.entry_at) < aware(item.exit_at) <= aware(item.available_at):
                raise ValueError("Canonical ordered delayed outcome evidence required")
            canonical = DEFAULT_CALENDAR.action_window(ids[item.action_id].timestamp)
            if aware(item.entry_at) != canonical.entry_at or aware(item.exit_at) != canonical.exit_at:
                raise ValueError("Training label must use the actual canonical delayed execution grid")
            symbols = set(ids[item.action_id].prices)
            if set(item.entry_prices) != symbols or set(item.exit_prices) != symbols or not item.provenance:
                raise ValueError("Complete immutable delayed price snapshots/provenance required")
            for values in (item.entry_prices, item.exit_prices):
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values.values()):
                    raise ValueError("Finite positive outcome prices required")
            if ((aware(item.entry_at) in prices_at and item.entry_prices != prices_at[aware(item.entry_at)])
                    or (aware(item.exit_at) in prices_at and item.exit_prices != prices_at[aware(item.exit_at)])):
                raise ValueError("Outcome prices differ from immutable shared published observations")
            known[item.action_id] = copy.deepcopy(item)
        return cls(cutoff, rows, known, tuple(row.action_id for row in rows if row.action_id not in known))
