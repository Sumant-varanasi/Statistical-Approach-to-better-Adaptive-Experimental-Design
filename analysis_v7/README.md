# v7 analysis

Scripts behind the v7 figures and tables. Run from the repository root:

    python analysis_v7/compare_v7.py       # figures/smart_7_campaigns.png, figures/smart_7_candidates.png, analysis_v7/v7_summary.md
    python analysis_v7/gc_accuracy.py      # reaction mechanism vs neural network, GC of unseen runs (~10 min) -> gc_accuracy.json
    python analysis_v7/plot_mechanism.py   # figures/smart_7_mechanism.png (needs examples/discovery/finder_results.json and gc_accuracy.json)

compare_v7.py reads the campaign files of the v7 study (`benchmark.run_all(methods=benchmark.V7_STUDY)`:
18 methods, 10 seeds, 25 runs) from `results/hpc_v7final/bench` (seeds 0-6, run on the college HPC) and
`results/v7final_local/bench` (seeds 7-9, run on the laptop at the same time, identical code), and v5, v6 and the
baselines from the committed benchmark cache `results/smart_bench`. Those folders are not in git (`results/` holds
the large campaign caches); every number behind the figures is in `analysis_v7/v7_summary.md`, which is in git. Only files made by
the present definition of each method are used (`smart.benchmark.is_stale`).

plot_campaigns.py is the earlier preliminary version of the campaign figure (before the full study was run);
it is kept for reference and writes `figures/smart_7_prelim_campaigns.png`.

Simulated reactor, placeholder kinetics and prices.
