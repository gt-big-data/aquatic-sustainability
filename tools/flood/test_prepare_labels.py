import unittest
from datetime import date
from prepare_labels import prepare


def sample(identifier, when, label=1, **extra):
    return {"ID": identifier, "Began": when, "Country": "Example", "lat": "10", "long": "100", "isFlood": str(label), **extra}


class LabelAuditTests(unittest.TestCase):
    def audit(self, rows, **kwargs):
        return prepare(rows, date(2019, 1, 1), date(2021, 1, 1), **kwargs)

    def test_scaler_independent_chronological_assignment(self):
        accepted, rejected, _ = self.audit([sample("a", "2018-01-01"), sample("b", "2020-01-01"), sample("c", "2022-01-01")])
        self.assertEqual([row["split"] for row in accepted], ["train", "validation", "test"])
        self.assertFalse(rejected)

    def test_unknown_negative_is_not_silently_treated_as_confirmed(self):
        accepted, rejected, _ = self.audit([sample("a", "2018-01-01", 0)])
        self.assertFalse(accepted)
        self.assertIn("negative_not_independently_verified", rejected[0]["issue"])

    def test_negative_close_to_positive_is_quarantined_even_if_marked_verified(self):
        accepted, rejected, _ = self.audit([sample("a", "2018-01-01"), sample("b", "2018-01-02", 0, negative_verified="yes")])
        self.assertEqual(len(accepted), 1)
        self.assertIn("negative_near_recorded_flood", rejected[0]["issue"])

    def test_duplicate_and_bad_coordinates(self):
        good = sample("a", "2018-01-01")
        accepted, rejected, _ = self.audit([good, good, sample("b", "2018-02-01", lat="NaN")])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 2)

    def test_same_event_cannot_cross_partitions(self):
        accepted, rejected, _ = self.audit([sample("a", "2018-12-25", event_group="storm"), sample("b", "2019-01-03", event_group="storm")])
        self.assertFalse(accepted)
        self.assertTrue(all("event_group_crosses" in row["issue"] for row in rejected))

    def test_smap_history_cutoff(self):
        accepted, rejected, _ = self.audit([sample("a", "2015-04-01")], require_smap=True)
        self.assertFalse(accepted)
        self.assertIn("smap_history", rejected[0]["issue"])

    def test_nearby_ungrouped_cross_partition_floods_are_reported(self):
        _, _, report = self.audit([sample("a", "2018-12-31"), sample("b", "2019-01-02")])
        self.assertEqual(report["nearby_cross_split_positive_pairs_for_review"], 1)


if __name__ == "__main__":
    unittest.main()
