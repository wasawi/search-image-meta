"""--not: dropping images where an excluded term turns up."""

import pytest

from helpers import hit_names, run_cli, run_search

# every sample whose positive prompt mentions a lighthouse
LIGHTHOUSE = {"01_stray_loader.png", "02_stray_chain.png", "03_bypassed.png",
              "04_muted.png", "05_reroute.png", "06_set_get.png",
              "07_subgraph.png", "09_stray_loader.webp", "11_node_scope.png"}


def test_baseline(sample_dir):
    assert hit_names(run_search(sample_dir, "lighthouse")) == LIGHTHOUSE


def test_excludes_images_with_the_term(sample_dir):
    got = hit_names(run_search(sample_dir, "lighthouse", "--not", "STRAY_ckpt_krea"))
    assert got == LIGHTHOUSE - {"01_stray_loader.png", "09_stray_loader.webp"}


def test_any_excluded_term_is_enough(sample_dir):
    got = hit_names(run_search(sample_dir, "lighthouse",
                               "--not", "MAIN_ckpt_s02", "MAIN_ckpt_s03"))
    assert got == LIGHTHOUSE - {"02_stray_chain.png", "03_bypassed.png"}


def test_regex_applies_to_excluded_terms(sample_dir):
    got = hit_names(run_search(sample_dir, "lighthouse", "--regex",
                               "--not", "MAIN_ckpt_s0[1-3]"))
    assert got == LIGHTHOUSE - {"01_stray_loader.png", "02_stray_chain.png",
                                "03_bypassed.png"}


def test_case_sensitivity_applies(sample_dir):
    assert hit_names(run_search(sample_dir, "lighthouse", "-s",
                                "--not", "stray_ckpt_krea")) == LIGHTHOUSE


def test_unconnected_nodes_dont_exclude_with_only_connected(sample_dir):
    got = hit_names(run_search(sample_dir, "lighthouse", "--only-connected",
                               "--not", "STRAY_ckpt_krea"))
    assert got == LIGHTHOUSE


def test_fields_limit_where_excluded_terms_are_looked_for(sample_dir):
    # the Note node exists only in the workflow chunk
    got = hit_names(run_search(sample_dir, "lighthouse", "--fields", "prompt",
                               "Model", "--not", "NOTE_text"))
    assert got == LIGHTHOUSE
    got = hit_names(run_search(sample_dir, "lighthouse", "--not", "NOTE_text"))
    assert got == LIGHTHOUSE - {"01_stray_loader.png", "09_stray_loader.webp"}


@pytest.mark.parametrize("scope", ["field", "node"])
def test_exclusion_is_image_wide_whatever_the_scope(sample_dir, scope):
    # CHAIN_src sits in a different node (and, for field, is still excluded)
    got = hit_names(run_search(sample_dir, "lighthouse", "--scope", scope,
                               "--not", "CHAIN_src"))
    assert got == LIGHTHOUSE - {"02_stray_chain.png"}


def test_non_graph_metadata(sample_dir):
    # A1111 parameters: "a photo of a cat ... Sampler: Euler a"
    assert hit_names(run_search(sample_dir, "Sampler: Euler", "--not", "cat")) == set()
    assert hit_names(run_search(sample_dir, "Sampler: Euler", "--not", "dog")) == {"10_a1111.png"}


def test_bad_excluded_regex_is_a_usage_error(sample_dir):
    code, _out, err = run_cli(sample_dir, "lighthouse", "--regex", "--not", "(")
    assert code == 2 and "bad regex" in err


def test_results_file_remembers_exclusions(sample_dir, tmp_path):
    log = tmp_path / "results.txt"
    code, _out, _err = run_cli(sample_dir, "lighthouse", "--not", "krea",
                               "--results", log, "--progress", "never")
    assert code == 0
    code, _out, err = run_cli(sample_dir, "lighthouse", "--results", log,
                              "--progress", "never")
    assert code == 2 and "exclude" in err
