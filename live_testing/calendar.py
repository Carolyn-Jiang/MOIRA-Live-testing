"""Frozen NYSE-node calendar for the independent 2026–27 live experiment.

Only the explicitly requested observation nodes are tradable. A half-day does
not introduce a 13:00 node, and model maintenance never becomes a trade node.
All instants carry IANA timezone information; no fixed UTC offset is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import cached_property
from typing import Iterable
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc
START_AT = datetime(2026, 11, 2, 9, 30, tzinfo=NY)
LAST_RELEASE_AT = datetime(2027, 5, 21, 15, 30, tzinfo=NY)
FINAL_EXIT_AT = datetime(2027, 5, 24, 10, 30, tzinfo=NY)
FULL_HOLIDAYS = frozenset({date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26)})
HALF_DAYS = frozenset({date(2026, 11, 27), date(2026, 12, 24)})
RELEASE_TIMES = tuple(time(hour, 30) for hour in (9, 10, 11, 12, 14, 15))
HALF_DAY_TIMES = RELEASE_TIMES[:4]
SPECIAL_CACHE_DATES = frozenset({date(2026, 12, 26)})


def aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(NY)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in FULL_HOLIDAYS


def weekly_training_day(day: date) -> date | None:
    """This NY ISO week's Monday, or Tuesday only when Monday is closed.

    A Tuesday fallback never shifts the observation window: its source week
    remains the preceding Monday through Sunday. There is no Wednesday fallback.
    """
    if not isinstance(day, date) or isinstance(day, datetime):
        raise ValueError("An explicit New York calendar date is required")
    monday = day-timedelta(days=day.weekday())
    if is_trading_day(monday):
        return monday
    tuesday = monday+timedelta(days=1)
    return tuesday if is_trading_day(tuesday) else None


def is_weekly_training_time(at: datetime | str) -> bool:
    """Exact bounded 08:00 NY weekly slot, including holiday-Monday fallback."""
    local = aware(at)
    return (START_AT <= local < LAST_RELEASE_AT
            and (local.hour, local.minute, local.second, local.microsecond) == (8, 0, 0, 0)
            and local.date() == weekly_training_day(local.date()))


def session_nodes(day: date) -> tuple[datetime, ...]:
    if not is_trading_day(day):
        return ()
    times = HALF_DAY_TIMES if day in HALF_DAYS else RELEASE_TIMES
    return tuple(datetime.combine(day, slot, NY) for slot in times)


def week_bounds(at: datetime | str) -> tuple[datetime, datetime]:
    """NY ISO week [Monday 00:00, next Monday 00:00), including DST changes."""
    day = aware(at).date()
    monday = day - timedelta(days=day.weekday())
    return datetime.combine(monday, time(), NY), datetime.combine(monday + timedelta(days=7), time(), NY)


def month_bounds(at: datetime | str) -> tuple[datetime, datetime]:
    """NY natural month, used by the separate shared monthly cost guard."""
    local = aware(at)
    start = datetime(local.year, local.month, 1, tzinfo=NY)
    year, month = (local.year + 1, 1) if local.month == 12 else (local.year, local.month + 1)
    return start, datetime(year, month, 1, tzinfo=NY)


@dataclass(frozen=True)
class ActionWindow:
    step: int
    release_at: datetime
    entry_at: datetime
    exit_at: datetime

    def as_dict(self) -> dict:
        return {"step": self.step, "release_at": self.release_at.isoformat(),
                "entry_at": self.entry_at.isoformat(), "exit_at": self.exit_at.isoformat()}


@dataclass(frozen=True)
class ScheduledEvent:
    kind: str
    at: datetime

    @property
    def id(self) -> str:
        return self.kind + ":" + self.at.isoformat()

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "at": self.at.isoformat(),
                "utc": self.at.astimezone(UTC).isoformat(),
                "london": self.at.astimezone(LONDON).isoformat()}


@dataclass(frozen=True)
class ExperimentCalendar:
    start_at: datetime = START_AT
    last_release_at: datetime = LAST_RELEASE_AT

    def __post_init__(self):
        start, end = aware(self.start_at), aware(self.last_release_at)
        if start > end or start not in session_nodes(start.date()) or end not in session_nodes(end.date()):
            raise ValueError("Experiment bounds must be ordered actual session nodes")
        if start < START_AT or end > LAST_RELEASE_AT:
            raise ValueError("The frozen holiday calendar covers only this experiment")
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "last_release_at", end)

    @cached_property
    def release_nodes(self) -> tuple[datetime, ...]:
        result = []
        day = self.start_at.date()
        while day <= self.last_release_at.date():
            result.extend(node for node in session_nodes(day) if self.start_at <= node <= self.last_release_at)
            day += timedelta(days=1)
        return tuple(result)

    @cached_property
    def valuation_nodes(self) -> tuple[datetime, ...]:
        """Release nodes plus exactly two terminal valuation-only boundaries."""
        result = list(self.release_nodes)
        day = self.last_release_at.date()
        terminal = []
        while len(terminal) < 2:
            terminal.extend(node for node in session_nodes(day) if node > self.last_release_at)
            day += timedelta(days=1)
        return tuple(result + terminal[:2])

    def action_window(self, release_at: datetime | str) -> ActionWindow:
        at = aware(release_at)
        try:
            i = self.release_nodes.index(at)
        except ValueError as error:
            raise ValueError("A decision requires an actual experiment release node") from error
        return ActionWindow(i + 1, at, self.valuation_nodes[i + 1], self.valuation_nodes[i + 2])

    def next_node(self, at: datetime | str, *, inclusive: bool = False) -> datetime | None:
        at = aware(at)
        return next((node for node in self.valuation_nodes if node > at or (inclusive and node == at)), None)

    def release_schedule(self) -> list[dict]:
        return [self.action_window(node).as_dict() for node in self.release_nodes]

    def events(self, *, maintenance: bool = True) -> tuple[ScheduledEvent, ...]:
        events = [ScheduledEvent("release", at) for at in self.release_nodes]
        events += [ScheduledEvent("valuation", at) for at in self.valuation_nodes[len(self.release_nodes):]]
        if maintenance:
            day = self.start_at.date()
            while day <= self.last_release_at.date():
                specifications = []
                if day == weekly_training_day(day):
                    specifications.append(("training", time(8)))
                if day.weekday() == 4 and is_trading_day(day):
                    specifications.append(("pair_reselection", time(16)))
                if day.weekday() == 6 or day in SPECIAL_CACHE_DATES:
                    specifications.append(("cache", time(9, 30)))
                for kind, slot in specifications:
                    at = datetime.combine(day, slot, NY)
                    # No fabricated historical first training or maintenance
                    # after the last decision, even during terminal valuation.
                    if self.start_at <= at < self.last_release_at:
                        events.append(ScheduledEvent(kind, at))
                day += timedelta(days=1)
        return tuple(sorted(events, key=lambda event: (event.at.timestamp(), event.kind)))

    def due_events(self, asof: datetime | str, *, completed_ids: Iterable[str] = (),
                   grace: timedelta = timedelta(minutes=25), maintenance: bool = True) -> tuple[ScheduledEvent, ...]:
        """Resolve already-reached events only; never collect before a boundary."""
        instant = aware(asof)
        if grace <= timedelta(0):
            raise ValueError("A positive late-dispatch grace is required")
        completed = set(completed_ids)
        return tuple(event for event in self.events(maintenance=maintenance)
                     if event.id not in completed and 0 <= instant.timestamp() - event.at.timestamp() <= grace.total_seconds())

    def next_event(self, after: datetime | str, *, maintenance: bool = True) -> ScheduledEvent | None:
        instant = aware(after)
        return next((event for event in self.events(maintenance=maintenance) if event.at > instant), None)

    def utc_segments(self) -> list[dict]:
        """Actual dates grouped by both NY/UK offsets; no recurrence assumption.

        ``dates`` explicitly carries each day's actual slots, so half-days and
        holidays remain correct even inside a stable-offset segment.
        """
        segments = []
        for node in self.release_nodes:
            offsets = (str(node.utcoffset()), str(node.astimezone(LONDON).utcoffset()))
            if not segments or tuple(segments[-1]["offsets"]) != offsets:
                segments.append({"offsets": list(offsets), "first_date": node.date().isoformat(),
                                 "last_date": node.date().isoformat(), "dates": {}})
            segment = segments[-1]
            segment["last_date"] = node.date().isoformat()
            segment["dates"].setdefault(node.date().isoformat(), []).append({
                "ny": node.strftime("%H:%M"), "utc": node.astimezone(UTC).strftime("%H:%M"),
                "london": node.astimezone(LONDON).strftime("%H:%M")})
        return segments


DEFAULT_CALENDAR = ExperimentCalendar()
