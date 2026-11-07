# TDA of HP trajectories in Slay the Spire

Code for the SF2956 project. Data is not included; see the run-history
data set linked in the report.

Run in this order:

| script | what it does |
|---|---|
| `clean_all2.py` | applies the rejection rules, writes cleaned runs per Ascension level |
| `build_pools.py` | reservoir-samples a pool of up to 40,000 runs per level per stratum |
| `kappa.py` | stable ranks, the kappa matrices and the within-level baseline (Figure 6) |
| `plot_traj.py` | sample HP trajectories (Figure 1) |
| `plot_clearrate.py` | clear rate by Ascension level, the figures quoted in Section 1.1 |
| `barcode_demo.py` | the six-point worked example (Figure 2) |
| `check_outliers.py` | max-HP distribution, used to set the HP cap |
| `covering_radius.py` | covering radius against N (Section 5.3) |
| `crossings.py` | crossing points t_L and the truncated curves (Figures 3, 4) |
| `complete.py` | the complete-linkage check (Figure 5) |

Parameters: N = 2000, R = 20, seed 7.
