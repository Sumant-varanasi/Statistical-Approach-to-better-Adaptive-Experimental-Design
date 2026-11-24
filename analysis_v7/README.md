# v7 work in progress: preliminary comparison

Scripts behind the two preliminary figures (`figures/smart_7_prelim_campaigns.png`,
`figures/smart_7_prelim_mechanism.png`). Run from the repository root:

    python analysis_v7/gc_accuracy.py      # reaction mechanism vs neural network, GC of unseen runs (~10 min)
    python analysis_v7/plot_campaigns.py   # needs the campaign files in results/hpc_early/bench and results/smart_bench
    python analysis_v7/plot_mechanism.py   # needs examples/discovery/finder_results.json and gc_accuracy.json

The campaign numbers come from an HPC run of 23 November 2026 on the v7 code before its final
review. They will be regenerated from the final code; until then treat them as preliminary.
Simulated reactor, placeholder kinetics and prices.
