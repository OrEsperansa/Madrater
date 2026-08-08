#!/usr/bin/env python3
"""Simple duty scheduler: two people at 06:20 and one at 07:40."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


WEEKDAY_ALIASES = {
    0: ("monday", "mon", "שני", "יום שני"),
    1: ("tuesday", "tue", "שלישי", "יום שלישי"),
    2: ("wednesday", "wed", "רביעי", "יום רביעי"),
    3: ("thursday", "thu", "חמישי", "יום חמישי"),
    4: ("friday", "fri", "שישי", "יום שישי"),
    5: ("saturday", "sat", "שבת", "יום שבת"),
    6: ("sunday", "sun", "ראשון", "יום ראשון"),
}

DEFAULT_WEIGHTS = {
    "fairness_total": 20.0,
    "fairness_early": 8.0,
    "fairness_late": 4.0,
    "preference": 10.0,
    "room": 4.0,
}


@dataclass(frozen=True)
class Person:
    name: str
    room: str
    constraints: dict[str, Any]


@dataclass
class State:
    total: tuple[int, ...]
    early: tuple[int, ...]
    late: tuple[int, ...]
    soft_cost: float
    assignments: list[tuple[date, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="שיבוץ שני אנשים ל-06:20 ואדם נוסף ל-07:40"
    )
    parser.add_argument("config", type=Path, help="קובץ הגדרות JSON")
    parser.add_argument("--output", "-o", type=Path, help="שמירת התוצאה כ-CSV")
    parser.add_argument("--beam-width", type=int, default=250, help=argparse.SUPPRESS)
    parser.add_argument("--branch-width", type=int, default=30, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_config(path: Path) -> tuple[list[Person], list[date], dict[str, float]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"לא ניתן לקרוא את קובץ ההגדרות: {exc}") from exc

    people = []
    names = set()
    for item in raw.get("people", []):
        name = str(item.get("name", "")).strip()
        room = str(item.get("room", "")).strip()
        if not name or not room:
            raise ValueError("לכל אדם חייבים להיות name ו-room")
        if name in names:
            raise ValueError(f"השם {name!r} מופיע יותר מפעם אחת")
        names.add(name)
        people.append(Person(name, room, item.get("constraints", {})))
    if len(people) < 3:
        raise ValueError("נדרשים לפחות 3 אנשים")

    if "dates" in raw:
        dates = [date.fromisoformat(value) for value in raw["dates"]]
    else:
        start = date.fromisoformat(raw["start_date"])
        end = date.fromisoformat(raw["end_date"])
        if end < start:
            raise ValueError("end_date חייב להיות אחרי start_date")
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if not dates:
        raise ValueError("לא הוגדרו תאריכים לשיבוץ")
    if len(set(dates)) != len(dates):
        raise ValueError("רשימת התאריכים מכילה כפילויות")

    weights = DEFAULT_WEIGHTS.copy()
    for key, value in raw.get("weights", {}).items():
        if key not in weights:
            raise ValueError(f"משקל לא מוכר: {key}")
        weights[key] = float(value)
    return people, sorted(dates), weights


def normalize_constraint(value: Any) -> int | None:
    """Return None for unavailable, otherwise preference-to-avoid level 0..3."""
    if isinstance(value, bool):
        return 0 if value else None
    if value is None:
        return 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"unavailable", "no", "cannot", "אסור", "לא יכול", "לא"}:
            return None
        try:
            value = int(lowered)
        except ValueError as exc:
            raise ValueError(f"ערך אילוץ לא מוכר: {value!r}") from exc
    level = int(value)
    if level not in (0, 1, 2, 3):
        raise ValueError("רמת העדפה חייבת להיות 0, 1, 2, 3 או unavailable")
    return level


def constraint_for(person: Person, day: date) -> int | None:
    constraints = person.constraints or {}
    values = []
    date_values = constraints.get("dates", {})
    if day.isoformat() in date_values:
        values.append(normalize_constraint(date_values[day.isoformat()]))

    weekday_values = constraints.get("weekdays", {})
    normalized = {str(k).strip().lower(): v for k, v in weekday_values.items()}
    for alias in WEEKDAY_ALIASES[day.weekday()]:
        if alias in normalized:
            values.append(normalize_constraint(normalized[alias]))
    if any(value is None for value in values):
        return None
    return max(values, default=0)


def room_penalty(a: Person, b: Person, c: Person) -> float:
    if a.room == b.room == c.room:
        return 0.0
    if a.room == b.room:  # Most important: the two early risers can wake each other.
        return 0.5
    if c.room in {a.room, b.room}:
        return 2.0
    return 3.0


def fairness_cost(state: State, completed_days: int, people_count: int, weights: dict[str, float]) -> float:
    targets = (3 * completed_days / people_count, 2 * completed_days / people_count, completed_days / people_count)
    total_error = sum((x - targets[0]) ** 2 for x in state.total)
    early_error = sum((x - targets[1]) ** 2 for x in state.early)
    late_error = sum((x - targets[2]) ** 2 for x in state.late)
    return (
        weights["fairness_total"] * total_error
        + weights["fairness_early"] * early_error
        + weights["fairness_late"] * late_error
        + state.soft_cost
    )


def day_candidates(people: list[Person], day: date, weights: dict[str, float]) -> list[tuple[int, int, int, float]]:
    levels = [constraint_for(person, day) for person in people]
    available = [i for i, level in enumerate(levels) if level is not None]
    if len(available) < 3:
        names = ", ".join(people[i].name for i in available) or "אף אחד"
        raise ValueError(f"ב-{day.isoformat()} יש פחות מ-3 אנשים זמינים (זמינים: {names})")

    candidates = []
    for pos, a in enumerate(available):
        for b in available[pos + 1 :]:  # The order inside the 06:20 pair is irrelevant.
            for c in available:
                if c in (a, b):
                    continue
                preference = sum((levels[i] or 0) ** 2 for i in (a, b, c))
                cost = weights["preference"] * preference
                cost += weights["room"] * room_penalty(people[a], people[b], people[c])
                candidates.append((a, b, c, cost))
    return candidates


def add_assignment(state: State, day: date, candidate: tuple[int, int, int, float]) -> State:
    a, b, c, local_cost = candidate
    total, early, late = list(state.total), list(state.early), list(state.late)
    for i in (a, b, c):
        total[i] += 1
    early[a] += 1
    early[b] += 1
    late[c] += 1
    return State(tuple(total), tuple(early), tuple(late), state.soft_cost + local_cost, state.assignments + [(day, a, b, c)])


def make_schedule(people: list[Person], dates: list[date], weights: dict[str, float], beam_width: int, branch_width: int) -> State:
    count = len(people)
    empty = (0,) * count
    states = [State(empty, empty, empty, 0.0, [])]

    for day_number, day in enumerate(dates, start=1):
        candidates = day_candidates(people, day, weights)
        expanded = []
        for state in states:
            ranked = []
            for candidate in candidates:
                new_state = add_assignment(state, day, candidate)
                ranked.append((fairness_cost(new_state, day_number, count, weights), new_state))
            ranked.sort(key=lambda item: item[0])
            expanded.extend(ranked[:branch_width])

        # States with identical counters have identical future options; keep the cheaper history.
        best_by_counters: dict[tuple[tuple[int, ...], tuple[int, ...]], State] = {}
        for _, state in expanded:
            key = (state.total, state.early)
            previous = best_by_counters.get(key)
            if previous is None or state.soft_cost < previous.soft_cost:
                best_by_counters[key] = state
        states = sorted(
            best_by_counters.values(),
            key=lambda s: fairness_cost(s, day_number, count, weights),
        )[:beam_width]

    return min(states, key=lambda s: fairness_cost(s, len(dates), count, weights))


def print_schedule(state: State, people: list[Person]) -> None:
    print("\nשיבוץ:\n")
    for day, a, b, c in state.assignments:
        early = f"{people[a].name} (חדר {people[a].room}), {people[b].name} (חדר {people[b].room})"
        late = f"{people[c].name} (חדר {people[c].room})"
        print(f"{day.isoformat()} | 06:20: {early} | 07:40: {late}")

    print("\nסיכום עומסים:")
    for i, person in enumerate(people):
        print(f"{person.name}: סה\"כ {state.total[i]}, ב-06:20 {state.early[i]}, ב-07:40 {state.late[i]}")


def write_csv(path: Path, state: State, people: list[Person]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["תאריך", "06:20 - אדם 1", "חדר", "06:20 - אדם 2", "חדר", "07:40", "חדר"])
        for day, a, b, c in state.assignments:
            writer.writerow([day.isoformat(), people[a].name, people[a].room, people[b].name, people[b].room, people[c].name, people[c].room])


def main() -> int:
    # Windows terminals may otherwise use a legacy code page that cannot print Hebrew.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        people, dates, weights = load_config(args.config)
        state = make_schedule(people, dates, weights, args.beam_width, args.branch_width)
        print_schedule(state, people)
        if args.output:
            write_csv(args.output, state, people)
            print(f"\nהקובץ נשמר ב-{args.output}")
        return 0
    except (ValueError, KeyError) as exc:
        print(f"שגיאה: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
