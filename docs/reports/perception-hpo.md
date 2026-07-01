# Perception CNN — HPO results

Optuna study `perception_cnn_hpo` (60 complete / 60 trials). Objective = held-out **VAL-track** mean MAE over the four vision-geometry prizes. Search epochs=10, subsample=40/track; winner retrained 25 epochs at full TRAIN.

## Best config (prize MAE 0.0998)
```
conv_spec = ((32, 8, 4), (32, 4, 2), (64, 4, 2), (64, 3, 1))
features_dim = 512
lr = 0.0026248631529213733
batch_size = 128
weight_decay = 4.388968112437962e-06
signed_curvature = False
proprio_weight = 1.0
```

## Winner — full held-out per-feature MAE

| feature | val MAE | skill | test MAE | usable |
|---|---|---|---|---|
| `lateral_offset` | 0.1618 | +0.68 | 0.1813 | — |
| `heading_error` | 0.0804 | +0.44 | 0.0766 | ✅ |
| `dist_left_edge` | 0.0810 | +0.68 | 0.0896 | ✅ |
| `dist_right_edge` | 0.0810 | +0.68 | 0.0897 | ✅ |
| `speed_mps` | 0.4667 | +0.06 | 0.4815 | — |
| `yaw_rate` | 0.1740 | +0.01 | 0.1561 | — |
| `long_accel` | 0.5995 | -0.05 | 0.6284 | — |
| `lateral_velocity` | 0.2902 | +0.39 | 0.3001 | — |
| `edge_closing_rate` | 0.2178 | +0.23 | 0.2048 | — |
| `curvature_ahead` | 0.0382 | +0.02 | 0.0225 | — |
| `nearest_object_dist` | 0.0053 | n/a | 0.0046 | — |

*skill = 1 − MAE/base; ✅ = MAE<0.10 & skill>0.15.*

## Top trials

| # | prizeMAE | arch | feat_dim | lr | batch | wd | signedΩ | propW |
|---|---|---|---|---|---|---|---|---|
| 59 | 0.0998 | deep | 512 | 2.6e-03 | 128 | 4.4e-06 | False | 1.0 |
| 56 | 0.1024 | custom | 512 | 1.3e-03 | 128 | 7.0e-06 | False | 1.0 |
| 39 | 0.1034 | custom | 512 | 1.0e-03 | 128 | 1.4e-06 | False | 1.0 |
| 46 | 0.1040 | custom | 512 | 1.5e-03 | 128 | 2.9e-06 | False | 1.0 |
| 47 | 0.1046 | custom | 512 | 1.6e-03 | 128 | 1.1e-05 | False | 1.0 |
| 36 | 0.1054 | custom | 512 | 3.9e-04 | 128 | 1.3e-06 | False | 1.0 |
| 7 | 0.1075 | deep | 256 | 4.4e-04 | 128 | 3.2e-08 | True | 0.0 |
| 53 | 0.1076 | custom | 512 | 9.8e-04 | 128 | 9.9e-06 | False | 1.0 |
| 58 | 0.1079 | custom | 128 | 1.6e-03 | 128 | 6.3e-06 | False | 0.1 |
| 44 | 0.1079 | custom | 512 | 6.5e-04 | 128 | 2.8e-06 | False | 1.0 |
