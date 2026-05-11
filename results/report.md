# Training metrics (synthetic, hold-out test split)
### binary — Pristine vs Damage

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | modal        | accuracy | 0.9887  | 0.9947  | —       |
| xgb          | modal        | accuracy | 0.9647  | 0.9747  | —       |
| rf           | modal        | accuracy | 0.9493  | 0.9580  | —       |
| cnn2d        | cfdac        | accuracy | 0.9440  | 0.9613  | —       |
| xgb          | indicators   | accuracy | 0.9260  | 0.9187  | —       |
| rf           | indicators   | accuracy | 0.9160  | 0.9240  | —       |
| transformer  | timeseries   | accuracy | 0.8760  | 0.8900  | —       |
| cnn          | frf_mag      | accuracy | 0.8527  | 0.8387  | —       |
| cnn          | timeseries   | accuracy | 0.8420  | 0.8453  | —       |
| mlp          | indicators   | accuracy | 0.8213  | 0.8260  | —       |
| transformer  | frf_mag      | accuracy | 0.8000  | 0.8000  | —       |
### type — Damage type (Pristine/Bolt/Crack/Hole/Mass)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | modal        | accuracy | 0.8767  | 0.8687  | —       |
| xgb          | modal        | accuracy | 0.8220  | 0.8067  | —       |
| rf           | modal        | accuracy | 0.8113  | 0.8153  | —       |
| cnn2d        | cfdac        | accuracy | 0.8033  | 0.7960  | —       |
| xgb          | indicators   | accuracy | 0.7593  | 0.7740  | —       |
| rf           | indicators   | accuracy | 0.7447  | 0.7567  | —       |
| mlp          | indicators   | accuracy | 0.7007  | 0.7033  | —       |
| cnn          | frf_mag      | accuracy | 0.6893  | 0.6773  | —       |
| cnn          | timeseries   | accuracy | 0.6567  | 0.6540  | —       |
| transformer  | timeseries   | accuracy | 0.5760  | 0.5573  | —       |
| transformer  | frf_mag      | accuracy | 0.5007  | 0.4760  | —       |
### severity — Severity regression (normalised [0,1] per type)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| rf           | modal        | R2       | 0.5728  | 0.5931  | 0.1300  |
| mlp          | modal        | R2       | 0.5419  | 0.5513  | 0.1453  |
| xgb          | modal        | R2       | 0.5318  | 0.5512  | 0.1374  |
| rf           | indicators   | R2       | 0.4868  | 0.4981  | 0.1463  |
| xgb          | indicators   | R2       | 0.4677  | 0.4673  | 0.1510  |
| cnn2d        | cfdac        | R2       | 0.4199  | 0.3985  | 0.1743  |
| mlp          | indicators   | R2       | 0.3441  | 0.3760  | 0.1777  |
| cnn          | timeseries   | R2       | 0.2273  | 0.2576  | 0.2109  |
| cnn          | frf_mag      | R2       | 0.2129  | 0.2530  | 0.2129  |
| transformer  | timeseries   | R2       | 0.1679  | 0.2024  | 0.2219  |
| transformer  | frf_mag      | R2       | 0.0130  | 0.0279  | 0.2491  |
### col_location — Column-damage location (storey x end)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| cnn2d        | cfdac        | accuracy | 0.4944  | 0.4917  | —       |
| mlp          | modal        | accuracy | 0.4944  | 0.5072  | —       |
| rf           | modal        | accuracy | 0.4922  | 0.5094  | —       |
| xgb          | modal        | accuracy | 0.4878  | 0.5094  | —       |
| rf           | indicators   | accuracy | 0.4811  | 0.4817  | —       |
| cnn          | timeseries   | accuracy | 0.4733  | 0.4883  | —       |
| cnn          | frf_mag      | accuracy | 0.4689  | 0.4895  | —       |
| xgb          | indicators   | accuracy | 0.4544  | 0.4795  | —       |
| mlp          | indicators   | accuracy | 0.4167  | 0.4295  | —       |
| transformer  | timeseries   | accuracy | 0.3678  | 0.3873  | —       |
| transformer  | frf_mag      | accuracy | 0.2511  | 0.2675  | —       |
### mass_location — Mass plate location (Base/F1/F2/F3)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| rf           | modal        | accuracy | 0.9900  | 1.0000  | —       |
| mlp          | modal        | accuracy | 0.9867  | 1.0000  | —       |
| xgb          | modal        | accuracy | 0.9867  | 1.0000  | —       |
| xgb          | indicators   | accuracy | 0.9733  | 0.9900  | —       |
| rf           | indicators   | accuracy | 0.9667  | 0.9800  | —       |
| mlp          | indicators   | accuracy | 0.9633  | 0.9767  | —       |
| cnn2d        | cfdac        | accuracy | 0.9533  | 0.9767  | —       |
| transformer  | timeseries   | accuracy | 0.6367  | 0.6833  | —       |
| transformer  | frf_mag      | accuracy | 0.4800  | 0.4767  | —       |
| cnn          | timeseries   | accuracy | 0.4733  | 0.4767  | —       |
| cnn          | frf_mag      | accuracy | 0.4133  | 0.4267  | —       |

# Experimental-data evaluation (61 IQS cases)

*Note: composite damage scenarios in the experimental file are reduced to a single primary op (bolt > crack > hole > mass > pristine) for label assignment; OOD generalisation is expected to be partial.*

### binary — Pristine vs Damage

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| cnn2d        | cfdac        | accuracy | 0.8689  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.8689  | —       | —       |
| mlp          | indicators   | accuracy | 0.8689  | —       | —       |
| mlp          | modal        | accuracy | 0.8689  | —       | —       |
| rf           | indicators   | accuracy | 0.8689  | —       | —       |
| rf           | modal        | accuracy | 0.8689  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.8689  | —       | —       |
| xgb          | indicators   | accuracy | 0.8689  | —       | —       |
| xgb          | modal        | accuracy | 0.8689  | —       | —       |
| transformer  | timeseries   | accuracy | 0.7377  | —       | —       |
| cnn          | timeseries   | accuracy | 0.4098  | —       | —       |
### type — Damage type (Pristine/Bolt/Crack/Hole/Mass)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| rf           | modal        | accuracy | 0.4426  | —       | —       |
| cnn2d        | cfdac        | accuracy | 0.4262  | —       | —       |
| mlp          | modal        | accuracy | 0.4262  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.3934  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.3607  | —       | —       |
| transformer  | timeseries   | accuracy | 0.2951  | —       | —       |
| xgb          | modal        | accuracy | 0.2951  | —       | —       |
| cnn          | timeseries   | accuracy | 0.2623  | —       | —       |
| rf           | indicators   | accuracy | 0.1639  | —       | —       |
| xgb          | indicators   | accuracy | 0.1475  | —       | —       |
| mlp          | indicators   | accuracy | 0.0656  | —       | —       |
### severity — Severity regression (normalised [0,1] per type)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| transformer  | frf_mag      | R2       | -0.0390 | —       | 0.2843  |
| xgb          | modal        | R2       | -0.0622 | —       | 0.2830  |
| transformer  | timeseries   | R2       | -0.1008 | —       | 0.2882  |
| rf           | modal        | R2       | -0.1513 | —       | 0.2994  |
| cnn2d        | cfdac        | R2       | -0.2110 | —       | 0.3041  |
| xgb          | indicators   | R2       | -0.2421 | —       | 0.2841  |
| rf           | indicators   | R2       | -0.4224 | —       | 0.3007  |
| cnn          | frf_mag      | R2       | -4.2455 | —       | 0.5502  |
| cnn          | timeseries   | R2       | -22.3766 | —       | 0.9760  |
| mlp          | modal        | R2       | -33.1857 | —       | 1.8363  |
| mlp          | indicators   | R2       | -671.8231 | —       | 6.4814  |
### col_location — Column-damage location (storey x end)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | modal        | accuracy | 0.4898  | —       | —       |
| mlp          | indicators   | accuracy | 0.3673  | —       | —       |
| cnn          | timeseries   | accuracy | 0.3469  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.2653  | —       | —       |
| transformer  | timeseries   | accuracy | 0.2041  | —       | —       |
| cnn2d        | cfdac        | accuracy | 0.1633  | —       | —       |
| xgb          | indicators   | accuracy | 0.1633  | —       | —       |
| rf           | modal        | accuracy | 0.0612  | —       | —       |
| rf           | indicators   | accuracy | 0.0408  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.0408  | —       | —       |
| xgb          | modal        | accuracy | 0.0204  | —       | —       |
### mass_location — Mass plate location (Base/F1/F2/F3)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| cnn2d        | cfdac        | accuracy | 0.2500  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.2500  | —       | —       |
| cnn          | timeseries   | accuracy | 0.2500  | —       | —       |
| mlp          | indicators   | accuracy | 0.2500  | —       | —       |
| mlp          | modal        | accuracy | 0.2500  | —       | —       |
| rf           | indicators   | accuracy | 0.2500  | —       | —       |
| rf           | modal        | accuracy | 0.2500  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.2500  | —       | —       |
| xgb          | modal        | accuracy | 0.2500  | —       | —       |
| transformer  | timeseries   | accuracy | 0.0000  | —       | —       |
| xgb          | indicators   | accuracy | 0.0000  | —       | —       |
