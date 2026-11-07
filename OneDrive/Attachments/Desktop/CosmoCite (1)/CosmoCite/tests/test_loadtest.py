import loadtest


def test_summary_counts_outcomes_and_rate():
    out = loadtest.summary([(200, 1.0), (200, 3.0), (429, 0.1), (500, 0.1)], 4.0)
    assert (out["served"], out["busy"], out["errors"]) == (2, 1, 1)
    assert out["p50_s"] == 2.0 and out["per_second"] == 0.5


def test_summary_survives_nothing_served():
    assert loadtest.summary([(429, 0.1)], 1.0)["p50_s"] is None
