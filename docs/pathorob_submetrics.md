# PathoROB submetrics, base vs tuned, per dataset

**Generated file -- do not hand-edit.**  Regenerate with
`./.venv/bin/python scripts/pathorob_submetrics.py`.

Definitions (PathoROB `robustness_index_utils.py`): `balanced_accuracy` is the kNN
classifier's balanced accuracy at `k_opt` -- the only true prediction metric here.
`ID` = SS/(SS+OS), `OOD` = SO/(SO+OO), `pooled` = (SS+SO)/(SS+SO+OS+OO) are
neighbour class-purity fractions built from the same neighbour counts as
RI = SO/(SO+OS).  `pooled` is what PathoROB names `prediction_performance`; it is
NOT a classifier accuracy.  Tuned = mean +/- SD over seeds at each seed's own
1-SE-selected checkpoint (same rule as seed_stats.py).

## phikon2

base: phikon2-base-control; seed cells: phikon2-c50-s0-step200, phikon2-c50-s1-step200, phikon2-c50-s2-step200

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 3 | 0.921 -> 0.906 +/- 0.001 (-0.014) | 0.871 -> 0.842 +/- 0.003 (-0.029) | 0.887 -> 0.851 +/- 0.003 (-0.037) | 0.874 -> 0.846 +/- 0.003 (-0.029) | 0.617 -> 0.793 +/- 0.003 (+0.175) |
| camelyon | 3 | 0.955 -> 0.965 +/- 0.000 (+0.011) | 0.942 -> 0.949 +/- 0.000 (+0.007) | 0.994 -> 0.960 +/- 0.000 (-0.033) | 0.942 -> 0.951 +/- 0.000 (+0.008) | 0.023 -> 0.775 +/- 0.020 (+0.752) |
| tolkach_esca | 3 | 0.958 -> 0.961 +/- 0.003 (+0.004) | 0.924 -> 0.918 +/- 0.005 (-0.006) | 0.948 -> 0.928 +/- 0.004 (-0.020) | 0.929 -> 0.924 +/- 0.004 (-0.006) | 0.770 -> 0.939 +/- 0.002 (+0.169) |
| mean of 3 | 3 | 0.944 -> 0.944 +/- 0.002 (-0.000) | 0.913 -> 0.903 +/- 0.003 (-0.010) | 0.943 -> 0.913 +/- 0.003 (-0.030) | 0.915 -> 0.907 +/- 0.003 (-0.009) | 0.470 -> 0.836 +/- 0.007 (+0.365) |

## midnight

base: midnight-base-control; seed cells: midnight-c50-s0-step150, midnight-c50-s1-step100, midnight-c50-s3-step100

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 3 | 0.934 -> 0.943 +/- 0.002 (+0.009) | 0.916 -> 0.902 +/- 0.007 (-0.014) | 0.911 -> 0.908 +/- 0.006 (-0.003) | 0.914 -> 0.904 +/- 0.006 (-0.010) | 0.858 -> 0.871 +/- 0.005 (+0.013) |
| camelyon | 3 | 0.976 -> 0.975 +/- 0.001 (-0.000) | 0.966 -> 0.966 +/- 0.002 (-0.000) | 0.964 -> 0.970 +/- 0.001 (+0.006) | 0.966 -> 0.967 +/- 0.002 (+0.001) | 0.478 -> 0.884 +/- 0.009 (+0.406) |
| tolkach_esca | 3 | 0.976 -> 0.977 +/- 0.001 (+0.001) | 0.958 -> 0.953 +/- 0.001 (-0.004) | 0.961 -> 0.965 +/- 0.001 (+0.004) | 0.959 -> 0.960 +/- 0.001 (+0.001) | 0.941 -> 0.969 +/- 0.001 (+0.028) |
| mean of 3 | 3 | 0.962 -> 0.965 +/- 0.001 (+0.003) | 0.947 -> 0.940 +/- 0.003 (-0.006) | 0.945 -> 0.948 +/- 0.002 (+0.002) | 0.946 -> 0.944 +/- 0.003 (-0.003) | 0.759 -> 0.908 +/- 0.002 (+0.149) |

## virchow2

base: virchow2-base-control; seed cells: virchow2-c50-s0-step100, virchow2-c50-s1-step150, virchow2-c50-s3-step100

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 3 | 0.928 -> 0.924 +/- 0.002 (-0.004) | 0.886 -> 0.876 +/- 0.005 (-0.010) | 0.905 -> 0.887 +/- 0.005 (-0.018) | 0.893 -> 0.880 +/- 0.005 (-0.013) | 0.822 -> 0.841 +/- 0.004 (+0.019) |
| camelyon | 3 | 0.988 -> 0.982 +/- 0.001 (-0.005) | 0.981 -> 0.974 +/- 0.001 (-0.007) | 0.986 -> 0.976 +/- 0.001 (-0.009) | 0.981 -> 0.974 +/- 0.001 (-0.007) | 0.806 -> 0.916 +/- 0.009 (+0.110) |
| tolkach_esca | 3 | 0.977 -> 0.980 +/- 0.001 (+0.003) | 0.949 -> 0.953 +/- 0.003 (+0.004) | 0.964 -> 0.966 +/- 0.001 (+0.001) | 0.957 -> 0.961 +/- 0.002 (+0.004) | 0.955 -> 0.969 +/- 0.000 (+0.015) |
| mean of 3 | 3 | 0.964 -> 0.962 +/- 0.001 (-0.002) | 0.939 -> 0.934 +/- 0.003 (-0.005) | 0.952 -> 0.943 +/- 0.002 (-0.009) | 0.944 -> 0.939 +/- 0.002 (-0.005) | 0.861 -> 0.909 +/- 0.004 (+0.048) |

## hoptimus0

base: hoptimus0-base-control; seed cells: hoptimus0-c50-s0-step100, hoptimus0-c50-s1-step100, hoptimus0-c50-s3-step100

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 3 | 0.939 -> 0.926 +/- 0.002 (-0.013) | 0.895 -> 0.870 +/- 0.004 (-0.026) | 0.917 -> 0.877 +/- 0.004 (-0.040) | 0.902 -> 0.873 +/- 0.004 (-0.029) | 0.802 -> 0.836 +/- 0.004 (+0.033) |
| camelyon | 3 | 0.982 -> 0.983 +/- 0.000 (+0.000) | 0.973 -> 0.972 +/- 0.001 (-0.001) | 0.969 -> 0.973 +/- 0.001 (+0.004) | 0.973 -> 0.973 +/- 0.001 (-0.000) | 0.678 -> 0.919 +/- 0.002 (+0.241) |
| tolkach_esca | 3 | 0.972 -> 0.973 +/- 0.001 (+0.001) | 0.942 -> 0.941 +/- 0.001 (-0.001) | 0.963 -> 0.948 +/- 0.002 (-0.015) | 0.951 -> 0.945 +/- 0.001 (-0.005) | 0.918 -> 0.962 +/- 0.001 (+0.044) |
| mean of 3 | 3 | 0.964 -> 0.961 +/- 0.001 (-0.004) | 0.937 -> 0.928 +/- 0.002 (-0.009) | 0.950 -> 0.933 +/- 0.002 (-0.017) | 0.942 -> 0.930 +/- 0.002 (-0.012) | 0.800 -> 0.906 +/- 0.002 (+0.106) |

## uni2h

base: uni2h-base-control; seed cells: uni2h-c50-s0-step100, uni2h-c50-s1-step150, uni2h-c50-s2-step100

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 3 | 0.942 -> 0.932 +/- 0.004 (-0.010) | 0.909 -> 0.889 +/- 0.009 (-0.020) | 0.935 -> 0.900 +/- 0.010 (-0.035) | 0.916 -> 0.894 +/- 0.010 (-0.023) | 0.803 -> 0.858 +/- 0.003 (+0.055) |
| camelyon | 3 | 0.985 -> 0.983 +/- 0.002 (-0.003) | 0.980 -> 0.975 +/- 0.003 (-0.005) | 0.995 -> 0.981 +/- 0.003 (-0.014) | 0.980 -> 0.976 +/- 0.003 (-0.004) | 0.544 -> 0.901 +/- 0.011 (+0.357) |
| tolkach_esca | 3 | 0.976 -> 0.974 +/- 0.001 (-0.002) | 0.951 -> 0.949 +/- 0.002 (-0.002) | 0.974 -> 0.962 +/- 0.004 (-0.012) | 0.959 -> 0.956 +/- 0.003 (-0.003) | 0.923 -> 0.963 +/- 0.003 (+0.040) |
| mean of 3 | 3 | 0.968 -> 0.963 +/- 0.002 (-0.005) | 0.947 -> 0.938 +/- 0.005 (-0.009) | 0.968 -> 0.948 +/- 0.006 (-0.020) | 0.952 -> 0.942 +/- 0.005 (-0.010) | 0.757 -> 0.907 +/- 0.003 (+0.151) |

## virchow1

base: virchow1-base-control; seed cells: none

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 0 | 0.909 -> -- | 0.866 -> -- | 0.871 -> -- | 0.867 -> -- | 0.761 -> -- |
| camelyon | 0 | 0.980 -> -- | 0.975 -> -- | 0.932 -> -- | 0.972 -> -- | 0.751 -> -- |
| tolkach_esca | 0 | 0.969 -> -- | 0.945 -> -- | 0.963 -> -- | 0.953 -> -- | 0.932 -> -- |
| mean of 3 | 0 | 0.953 -> -- | 0.929 -> -- | 0.922 -> -- | 0.931 -> -- | 0.815 -> -- |

## virchow2f

base: virchow2-basectrl-fp32adv; seed cells: none

| dataset | n | Bal. acc. base -> tuned | ID purity base -> tuned | OOD purity base -> tuned | Pooled purity base -> tuned | RI base -> tuned |
|---|---|---|---|---|---|---|
| tcga | 0 | 0.928 -> -- | 0.886 -> -- | 0.905 -> -- | 0.893 -> -- | 0.822 -> -- |
| camelyon | 0 | 0.988 -> -- | 0.981 -> -- | 0.986 -> -- | 0.981 -> -- | 0.806 -> -- |
| tolkach_esca | 0 | 0.977 -> -- | 0.949 -> -- | 0.964 -> -- | 0.957 -> -- | 0.955 -> -- |
| mean of 3 | 0 | 0.964 -> -- | 0.939 -> -- | 0.952 -> -- | 0.944 -> -- | 0.861 -> -- |
