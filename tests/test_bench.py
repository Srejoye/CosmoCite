from bench import bench


def test_bench_reports_latency_and_exact_search_has_full_recall():
    rows = bench(("numpy",), (500,), dim=8, queries=5)
    assert rows[0]["p95_ms"] >= rows[0]["p50_ms"] > 0 and rows[0]["recall@10"] == 1.0
    assert (rows[0]["kind"], rows[0]["size"], rows[0]["data"]) == ("numpy", 500, "random")


def test_bench_hands_over_every_row_as_it_finishes():
    seen = []
    bench(("numpy",), (50, 60), dim=4, queries=2, on_row=lambda rows: seen.append(len(rows)))
    assert seen == [1, 2]

