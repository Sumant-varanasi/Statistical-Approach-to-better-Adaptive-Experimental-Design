# v7 work in progress: preliminary comparison

Scripts behind the two preliminary figures (`figures/smart_7_prelim_campaigns.png`,
`figures/smart_7_prelim_mechanism.png`). Run from the repository root:

    python analysis_v7/gc_accuracy.py      # reaction mechanism vs neural network, GC of unseen runs (~10 min)
    python analysis_v7/plot_campaigns.py   # needs the campaign files in results/hpc_early/bench and results/smart_bench
    python analysis_v7/plot_mechanism.py   # needs examples/discovery/finder_results.json and gc_accuracy.json

plot_campaigns.py shows only campaign files made by the PRESENT definition of each method
(smart.benchmark.is_stale). The HPC run of 23 November 2026 made its smart_v7* files when v7 meant
the hybrid committee without network revision, and its nn_only files under an earlier
recommendation rule: they are skipped (the footer counts them), so v7 and the neural network alone
are missing from the figure until benchmark.run_all(methods=benchmark.V7_STUDY) has been run with
the final code. The v6 ablations of that run are still valid and are shown. In the mechanism figure,
the 'runs left out at random' rows give the mechanism the reactions the data were made from; only
the 'hottest 6 runs' rows use reactions found from the training runs. Treat everything here as
preliminary. Simulated reactor, placeholder kinetics and prices.
