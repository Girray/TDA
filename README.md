# TDA of HP trajectories in Slay the Spire

Code for the SF2956 project. Data is not included; see the run-history
data set linked in the report.

Run in this order:

| script | what it does |
|---|---|
| `clean_all.py` | applies the rejection rules, writes cleaned runs per Ascension level |
| `1_build_pools.py` | reservoir-samples a pool of up to 40,000 runs per level per stratum |
| `2_kappa.py` | stable ranks, the kappa matrices and the within-level baseline |
| `3_plot_traj.py` | sample HP trajectories (Figure 1) |
| `4_plot_clear_rate.py` | clear rate by level |
| `5_barcode_demo.py` | the six-point worked example (Figure 2) |
| `6_check_outliers.py` | max-HP distribution, used to set the HP cap |
| `7_covering_radius.py` | covering radius vs N (Section 5.3) |
| `8_crossings.py` | crossing points t_L and the truncated curves (Figures 3, 4) |
| `9_complete.py` | the complete-linkage check (Figure 5) |

Parameters: N = 2000, R = 20, seed 7.
