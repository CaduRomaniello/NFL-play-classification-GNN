# Comparacao justa - resultados

Execucoes lidas de `output/comparacao_justa/runs` + GCN antigas de `output/results`

## Resumo (Macro F1 no teste)

| model | n | macro_f1_mean | macro_f1_std | macro_f1_min | macro_f1_max | accuracy_mean | f1_rush_mean | f1_pass_mean | per_play_ms_mean | device | train_time_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSets-tuned | 29 | 0.8404 | 0.0080 | 0.8285 | 0.8619 | 0.8447 | 0.8144 | 0.8664 | 0.9461 | cuda | 92.8448 |
| SAGE-RNG | 29 | 0.8386 | 0.0073 | 0.8264 | 0.8581 | 0.8428 | 0.8130 | 0.8643 | 1.1754 | cuda | 81.9612 |
| SAGE-DELAUNAY | 29 | 0.8383 | 0.0087 | 0.8248 | 0.8587 | 0.8421 | 0.8135 | 0.8630 | 1.3169 | cuda | 81.1963 |
| SAGE-QB-CLOSEST | 29 | 0.8380 | 0.0072 | 0.8216 | 0.8506 | 0.8417 | 0.8134 | 0.8625 | 1.2447 | cuda | 107.4279 |
| SAGE-CLOSEST | 29 | 0.8373 | 0.0082 | 0.8176 | 0.8522 | 0.8411 | 0.8129 | 0.8618 | 1.6204 | cuda | 155.3236 |
| SAGE-GABRIEL | 29 | 0.8354 | 0.0087 | 0.8197 | 0.8506 | 0.8395 | 0.8094 | 0.8613 | 1.1690 | cuda | 255.0655 |
| SAGE-MST | 29 | 0.8348 | 0.0087 | 0.8191 | 0.8498 | 0.8391 | 0.8085 | 0.8611 | 1.4077 | cuda | 104.2858 |
| DeepSets-tuned (antigo) | 29 | 0.8316 | 0.0080 | 0.8159 | 0.8459 | 0.8355 | 0.8062 | 0.8570 | 0.4174 | cuda | 343.9314 |
| DeepSets-matched | 29 | 0.8206 | 0.0085 | 0.8040 | 0.8377 | 0.8239 | 0.7966 | 0.8447 | 0.7911 | cuda | 62.3490 |
| DeepSets-matched (antigo) | 29 | 0.8186 | 0.0128 | 0.7935 | 0.8382 | 0.8244 | 0.7868 | 0.8505 | 0.3738 | cuda | 417.8926 |
| GCN-RNG | 29 | 0.8167 | 0.0077 | 0.7993 | 0.8352 | 0.8215 | 0.7872 | 0.8462 | 3.1056 | cuda | 281.9483 |
| GCN-MST | 29 | 0.8159 | 0.0070 | 0.8033 | 0.8331 | 0.8196 | 0.7900 | 0.8418 | 1.9685 | cuda | 103.7381 |
| GCN-GABRIEL | 29 | 0.8158 | 0.0083 | 0.8014 | 0.8332 | 0.8207 | 0.7859 | 0.8457 | 1.8871 | cuda | 158.7053 |
| GCN-QB-CLOSEST | 29 | 0.8158 | 0.0077 | 0.7999 | 0.8324 | 0.8205 | 0.7863 | 0.8452 | 1.8156 | cuda | 110.2608 |
| MLP-concat_team_role-tuned | 29 | 0.8138 | 0.0087 | 0.7964 | 0.8342 | 0.8186 | 0.7843 | 0.8433 | 0.0493 | cpu | 2.1278 |
| GCN-CLOSEST | 29 | 0.8132 | 0.0070 | 0.7982 | 0.8254 | 0.8168 | 0.7873 | 0.8391 | 3.3008 | cuda | 268.8434 |
| GCN-DELAUNAY | 29 | 0.8128 | 0.0084 | 0.7919 | 0.8303 | 0.8176 | 0.7828 | 0.8427 | 1.5654 | cuda | 145.3780 |
| RF-concat_team_role-tuned | 29 | 0.8055 | 0.0095 | 0.7870 | 0.8327 | 0.8100 | 0.7758 | 0.8351 | 2.3077 | cpu | 5.0455 |
| GCN-MST (texto) | 29 | 0.7925 | 0.0102 | 0.7751 | 0.8138 | 0.8003 | 0.7529 | 0.8321 | nan | None | nan |
| GCN-RNG (texto) | 29 | 0.7916 | 0.0090 | 0.7637 | 0.8158 | 0.7990 | 0.7529 | 0.8303 | nan | None | nan |
| MLP-concat_xy-tuned | 29 | 0.7884 | 0.0105 | 0.7678 | 0.8089 | 0.7943 | 0.7534 | 0.8233 | 0.0509 | cpu | 2.1334 |
| MLP-concat_team_y-tuned | 29 | 0.7875 | 0.0091 | 0.7673 | 0.8076 | 0.7952 | 0.7475 | 0.8276 | 0.0353 | cpu | 3.0315 |
| MLP-stats_team-tuned | 29 | 0.7873 | 0.0136 | 0.7547 | 0.8108 | 0.7935 | 0.7515 | 0.8232 | 0.0523 | cpu | 10.4326 |
| RF-stats_team-tuned | 29 | 0.7853 | 0.0109 | 0.7671 | 0.8058 | 0.7930 | 0.7449 | 0.8258 | 2.3916 | cpu | 5.1463 |
| GCN-CLOSEST (texto) | 29 | 0.7812 | 0.0124 | 0.7439 | 0.7995 | 0.7895 | 0.7392 | 0.8231 | nan | None | nan |
| RF-concat_xy-tuned | 29 | 0.7763 | 0.0098 | 0.7580 | 0.7942 | 0.7821 | 0.7406 | 0.8121 | 2.4385 | cpu | 9.5590 |
| GCN-GABRIEL (texto) | 29 | 0.7731 | 0.0134 | 0.7349 | 0.7922 | 0.7823 | 0.7284 | 0.8179 | nan | None | nan |
| GCN-QB-CLOSEST (texto) | 29 | 0.7720 | 0.0117 | 0.7392 | 0.7935 | 0.7815 | 0.7264 | 0.8176 | nan | None | nan |
| GCN-DELAUNAY (texto) | 29 | 0.7628 | 0.0128 | 0.7371 | 0.7850 | 0.7720 | 0.7173 | 0.8082 | nan | None | nan |
| RF-concat_team_y-tuned | 29 | 0.7568 | 0.0114 | 0.7336 | 0.7786 | 0.7664 | 0.7084 | 0.8052 | 2.3257 | cpu | 11.1892 |
| MLP-stats-tuned | 29 | 0.7485 | 0.0100 | 0.7339 | 0.7722 | 0.7568 | 0.7037 | 0.7934 | 0.0593 | cpu | 12.5221 |
| RF-stats-tuned | 29 | 0.7393 | 0.0096 | 0.7163 | 0.7601 | 0.7516 | 0.6828 | 0.7959 | 2.3357 | cpu | 0.6286 |
| MLP-mean-tuned | 29 | 0.7328 | 0.0128 | 0.7067 | 0.7655 | 0.7448 | 0.6766 | 0.7890 | 0.0270 | cpu | 1.3255 |
| RF-mean-default | 29 | 0.7281 | 0.0102 | 0.7093 | 0.7450 | 0.7413 | 0.6683 | 0.7880 | 2.4272 | cpu | 0.3965 |
| RF-mean-tuned | 29 | 0.7280 | 0.0106 | 0.7064 | 0.7530 | 0.7419 | 0.6663 | 0.7896 | 5.5634 | cpu | 0.9450 |
| RF-concat_raw-tuned | 29 | 0.7250 | 0.0117 | 0.6995 | 0.7513 | 0.7392 | 0.6623 | 0.7876 | 2.3291 | cpu | 11.9101 |
| MLP-concat_raw-tuned | 29 | 0.6931 | 0.0103 | 0.6652 | 0.7092 | 0.7099 | 0.6220 | 0.7642 | 0.0338 | cpu | 1.0703 |
| MLP-mean-default | 29 | 0.6738 | 0.0088 | 0.6560 | 0.6954 | 0.6880 | 0.6061 | 0.7415 | 0.0397 | cpu | 2.1317 |

## Friedman / Nemenyi

```
Friedman: chi2 = 811.92, p = 3.63e-152 (k = 30 modelos, N = 29 sementes)

Ranks medios (1 = melhor):
    2.62  DeepSets-tuned
    3.62  SAGE-RNG
    3.72  SAGE-DELAUNAY
    3.76  SAGE-QB-CLOSEST
    3.97  SAGE-CLOSEST
    5.14  SAGE-GABRIEL
    5.34  SAGE-MST
    9.05  DeepSets-matched
   10.90  GCN-RNG
   11.45  GCN-MST
   11.52  GCN-GABRIEL
   11.52  GCN-QB-CLOSEST
   12.55  MLP-concat_team_role-tuned
   12.64  GCN-CLOSEST
   12.90  GCN-DELAUNAY
   15.41  RF-concat_team_role-tuned
   18.38  MLP-concat_team_y-tuned
   18.45  MLP-concat_xy-tuned
   18.55  MLP-stats_team-tuned
   18.97  RF-stats_team-tuned
   20.59  RF-concat_xy-tuned
   22.24  RF-concat_team_y-tuned
   23.14  MLP-stats-tuned
   24.17  RF-stats-tuned
   25.41  MLP-mean-tuned
   26.48  RF-mean-tuned
   26.48  RF-mean-default
   27.03  RF-concat_raw-tuned
   29.07  MLP-concat_raw-tuned
   29.93  MLP-mean-default

Nemenyi: CD = 8.666 (q_0.05 = 3.749, k = 30, N = 29)
p-valores de Nemenyi salvos em nemenyi_pvalues.csv
```

## Wilcoxon pareado por semente vs `SAGE-RNG` (correcao de Holm)

| modelo | n_sementes | dif_media | dif_mediana | ref_vence | p_wilcoxon | p_holm |
|---|---|---|---|---|---|---|
| DeepSets-tuned | 29 | -0.0017 | -0.0033 | 10 | 0.1316 | 0.5264 |
| SAGE-DELAUNAY | 29 | 0.0004 | -0.0006 | 14 | 0.8647 | 1.0000 |
| SAGE-QB-CLOSEST | 29 | 0.0007 | 0.0011 | 15 | 0.5221 | 1.0000 |
| SAGE-CLOSEST | 29 | 0.0013 | 0.0012 | 17 | 0.2470 | 0.7411 |
| SAGE-GABRIEL | 29 | 0.0033 | 0.0023 | 21 | 0.0129 | 0.0647 |
| SAGE-MST | 29 | 0.0038 | 0.0049 | 21 | 0.0035 | 0.0211 |
| DeepSets-matched | 29 | 0.0180 | 0.0181 | 29 | 0.0000 | 0.0000 |
| GCN-RNG | 29 | 0.0219 | 0.0214 | 29 | 0.0000 | 0.0000 |
| GCN-MST | 29 | 0.0228 | 0.0215 | 29 | 0.0000 | 0.0000 |
| GCN-GABRIEL | 29 | 0.0229 | 0.0235 | 29 | 0.0000 | 0.0000 |
| GCN-QB-CLOSEST | 29 | 0.0229 | 0.0234 | 29 | 0.0000 | 0.0000 |
| MLP-concat_team_role-tuned | 29 | 0.0249 | 0.0239 | 29 | 0.0000 | 0.0000 |
| GCN-CLOSEST | 29 | 0.0255 | 0.0243 | 29 | 0.0000 | 0.0000 |
| GCN-DELAUNAY | 29 | 0.0259 | 0.0245 | 29 | 0.0000 | 0.0000 |
| RF-concat_team_role-tuned | 29 | 0.0332 | 0.0333 | 29 | 0.0000 | 0.0000 |
| MLP-concat_xy-tuned | 29 | 0.0503 | 0.0489 | 29 | 0.0000 | 0.0000 |
| MLP-concat_team_y-tuned | 29 | 0.0511 | 0.0505 | 29 | 0.0000 | 0.0000 |
| MLP-stats_team-tuned | 29 | 0.0513 | 0.0505 | 29 | 0.0000 | 0.0000 |
| RF-stats_team-tuned | 29 | 0.0533 | 0.0536 | 29 | 0.0000 | 0.0000 |
| RF-concat_xy-tuned | 29 | 0.0623 | 0.0600 | 29 | 0.0000 | 0.0000 |
| RF-concat_team_y-tuned | 29 | 0.0819 | 0.0800 | 29 | 0.0000 | 0.0000 |
| MLP-stats-tuned | 29 | 0.0901 | 0.0905 | 29 | 0.0000 | 0.0000 |
| RF-stats-tuned | 29 | 0.0993 | 0.0994 | 29 | 0.0000 | 0.0000 |
| MLP-mean-tuned | 29 | 0.1059 | 0.1100 | 29 | 0.0000 | 0.0000 |
| RF-mean-default | 29 | 0.1105 | 0.1109 | 29 | 0.0000 | 0.0000 |
| RF-mean-tuned | 29 | 0.1107 | 0.1117 | 29 | 0.0000 | 0.0000 |
| RF-concat_raw-tuned | 29 | 0.1137 | 0.1118 | 29 | 0.0000 | 0.0000 |
| MLP-concat_raw-tuned | 29 | 0.1455 | 0.1450 | 29 | 0.0000 | 0.0000 |
| MLP-mean-default | 29 | 0.1648 | 0.1657 | 29 | 0.0000 | 0.0000 |
