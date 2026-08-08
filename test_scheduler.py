import unittest
from datetime import date

from scheduler import Person, constraint_for, make_schedule, DEFAULT_WEIGHTS


class SchedulerTests(unittest.TestCase):
    def test_hard_constraint_is_respected(self):
        people = [
            Person("A", "1", {"dates": {"2026-08-09": "unavailable"}}),
            Person("B", "1", {}),
            Person("C", "2", {}),
            Person("D", "2", {}),
        ]
        result = make_schedule(people, [date(2026, 8, 9)], DEFAULT_WEIGHTS, 100, 20)
        assigned = result.assignments[0][1:]
        self.assertNotIn(0, assigned)

    def test_hebrew_weekday_constraint(self):
        person = Person("A", "1", {"weekdays": {"ראשון": "unavailable"}})
        self.assertIsNone(constraint_for(person, date(2026, 8, 9)))

    def test_balances_total_assignments(self):
        people = [Person(chr(65 + i), str(i // 2), {}) for i in range(6)]
        days = [date(2026, 8, 9 + i) for i in range(4)]
        result = make_schedule(people, days, DEFAULT_WEIGHTS, 250, 30)
        self.assertLessEqual(max(result.total) - min(result.total), 1)

    def test_prefers_same_room_for_early_pair(self):
        people = [
            Person("A", "1", {}), Person("B", "1", {}),
            Person("C", "2", {}), Person("D", "2", {}),
        ]
        result = make_schedule(people, [date(2026, 8, 9)], DEFAULT_WEIGHTS, 100, 20)
        _, a, b, _ = result.assignments[0]
        self.assertEqual(people[a].room, people[b].room)


if __name__ == "__main__":
    unittest.main()
