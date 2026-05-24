from __future__ import annotations

from tifa_archivist.source_intelligence import SourceIntelligence


def test_source_intelligence_save_and_load_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "source_intelligence.json"
    intelligence = SourceIntelligence(state_path=state_path, rng_seed=1)
    intelligence.record_search("Tifa Lockhart FF7", 12)
    intelligence.record_enqueued("Tifa Lockhart FF7", "example.com")
    intelligence.record_kept("Tifa Lockhart FF7", "example.com", "wallpaper")
    intelligence.save()

    loaded = SourceIntelligence(state_path=state_path, rng_seed=1)
    loaded.load()

    assert loaded.query_stats["Tifa Lockhart FF7"]["search_runs"] == 1
    assert loaded.query_stats["Tifa Lockhart FF7"]["kept"] == 1
    assert loaded.host_stats["example.com"]["kept"] == 1


def test_should_accept_host_rejects_bad_source_when_exploration_disabled(tmp_path) -> None:
    intelligence = SourceIntelligence(
        state_path=tmp_path / "state.json",
        min_host_samples=1,
        bad_host_threshold=0.5,
        bad_host_reject_probability=1.0,
        exploration_weight=0.0,
        rng_seed=7,
    )
    intelligence.record_download_failed("Tifa Lockhart FF7", "bad.example", "status 403")

    assert intelligence.should_accept_host("Tifa Lockhart FF7", "bad.example") is False


def test_plan_queries_generates_site_variants_from_winning_hosts(tmp_path) -> None:
    base_query = "Tifa Lockhart FF7"
    intelligence = SourceIntelligence(
        state_path=tmp_path / "state.json",
        max_variants=8,
        max_site_variants=2,
        min_host_samples=1,
        bad_host_threshold=0.1,
        rng_seed=3,
    )
    intelligence.record_kept(base_query, "wallhaven.cc", "wallpaper")
    intelligence.record_kept(base_query, "wallhaven.cc", "wallpaper")
    intelligence.record_enqueued(base_query, "wallhaven.cc")

    planned = intelligence.plan_queries([base_query])

    assert base_query in planned
    assert any("site:wallhaven.cc" in query for query in planned)
