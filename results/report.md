# Training metrics (synthetic, hold-out test split)
### binary — Pristine vs Damage

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | modal        | accuracy | 0.9820  | 0.9913  | —       |
| xgb          | modal        | accuracy | 0.9620  | 0.9700  | —       |
| rf           | modal        | accuracy | 0.9513  | 0.9553  | —       |
| transformer  | timeseries   | accuracy | 0.9233  | 0.9287  | —       |
| xgb          | indicators   | accuracy | 0.9207  | 0.9087  | —       |
| rf           | indicators   | accuracy | 0.9160  | 0.9240  | —       |
| mlp          | indicators   | accuracy | 0.8300  | 0.8167  | —       |
| cnn          | frf_mag      | accuracy | 0.8240  | 0.8127  | —       |
| cnn          | timeseries   | accuracy | 0.8213  | 0.8240  | —       |
| transformer  | frf_mag      | accuracy | 0.8000  | 0.8000  | —       |
### type — Damage type (Pristine/Bolt/Crack/Hole/Mass)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | modal        | accuracy | 0.8513  | 0.8427  | —       |
| xgb          | modal        | accuracy | 0.8247  | 0.8007  | —       |
| rf           | modal        | accuracy | 0.8100  | 0.8113  | —       |
| rf           | indicators   | accuracy | 0.7520  | 0.7560  | —       |
| cnn          | frf_mag      | accuracy | 0.7493  | 0.7380  | —       |
| xgb          | indicators   | accuracy | 0.7453  | 0.7607  | —       |
| cnn          | timeseries   | accuracy | 0.7353  | 0.7367  | —       |
| mlp          | indicators   | accuracy | 0.6560  | 0.6647  | —       |
| transformer  | frf_mag      | accuracy | 0.5800  | 0.5627  | —       |
| transformer  | timeseries   | accuracy | 0.4633  | 0.4700  | —       |
### severity — Severity regression (normalised [0,1] per type)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| rf           | modal        | R2       | 0.5728  | 0.5931  | 0.1300  |
| xgb          | modal        | R2       | 0.5280  | 0.5383  | 0.1403  |
| rf           | indicators   | R2       | 0.4868  | 0.4981  | 0.1463  |
| xgb          | indicators   | R2       | 0.4645  | 0.4574  | 0.1510  |
| mlp          | indicators   | R2       | 0.3141  | 0.3333  | 0.1889  |
| mlp          | modal        | R2       | 0.3001  | 0.3339  | 0.1860  |
| cnn          | timeseries   | R2       | 0.2714  | 0.2920  | 0.2005  |
| cnn          | frf_mag      | R2       | 0.2443  | 0.2913  | 0.2021  |
| transformer  | timeseries   | R2       | 0.0970  | 0.1044  | 0.2348  |
| transformer  | frf_mag      | R2       | 0.0083  | 0.0141  | 0.2498  |
### col_location — Column-damage location (storey x end)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| rf           | modal        | accuracy | 0.5022  | 0.4972  | —       |
| cnn          | timeseries   | accuracy | 0.4989  | 0.4895  | —       |
| xgb          | modal        | accuracy | 0.4933  | 0.4972  | —       |
| mlp          | modal        | accuracy | 0.4811  | 0.4939  | —       |
| rf           | indicators   | accuracy | 0.4756  | 0.4806  | —       |
| xgb          | indicators   | accuracy | 0.4567  | 0.4528  | —       |
| cnn          | frf_mag      | accuracy | 0.4522  | 0.4440  | —       |
| mlp          | indicators   | accuracy | 0.3922  | 0.4007  | —       |
| transformer  | frf_mag      | accuracy | 0.3878  | 0.3896  | —       |
| transformer  | timeseries   | accuracy | 0.3500  | 0.3541  | —       |
### mass_location — Mass plate location (Base/F1/F2/F3)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| xgb          | modal        | accuracy | 0.9933  | 1.0000  | —       |
| rf           | modal        | accuracy | 0.9900  | 1.0000  | —       |
| mlp          | modal        | accuracy | 0.9900  | 1.0000  | —       |
| xgb          | indicators   | accuracy | 0.9733  | 0.9900  | —       |
| rf           | indicators   | accuracy | 0.9700  | 0.9767  | —       |
| mlp          | indicators   | accuracy | 0.9233  | 0.9333  | —       |
| cnn          | timeseries   | accuracy | 0.8900  | 0.8933  | —       |
| transformer  | timeseries   | accuracy | 0.7267  | 0.7200  | —       |
| transformer  | frf_mag      | accuracy | 0.5633  | 0.6000  | —       |
| cnn          | frf_mag      | accuracy | 0.5533  | 0.5733  | —       |

# Experimental-data evaluation (61 IQS cases)

*Note: composite damage scenarios in the experimental file are reduced to a single primary op (bolt > crack > hole > mass > pristine) for label assignment; OOD generalisation is expected to be partial.*

### binary — Pristine vs Damage

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | indicators   | accuracy | 0.8689  | —       | —       |
| mlp          | modal        | accuracy | 0.8689  | —       | —       |
| rf           | indicators   | accuracy | 0.8689  | —       | —       |
| rf           | modal        | accuracy | 0.8689  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.8689  | —       | —       |
| xgb          | indicators   | accuracy | 0.8689  | —       | —       |
| xgb          | modal        | accuracy | 0.8689  | —       | —       |
| transformer  | timeseries   | accuracy | 0.7869  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.7049  | —       | —       |
| cnn          | timeseries   | accuracy | 0.3115  | —       | —       |
### type — Damage type (Pristine/Bolt/Crack/Hole/Mass)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| mlp          | modal        | accuracy | 0.5738  | —       | —       |
| rf           | modal        | accuracy | 0.4754  | —       | —       |
| xgb          | modal        | accuracy | 0.4262  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.3770  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.3607  | —       | —       |
| cnn          | timeseries   | accuracy | 0.2295  | —       | —       |
| transformer  | timeseries   | accuracy | 0.1967  | —       | —       |
| rf           | indicators   | accuracy | 0.1639  | —       | —       |
| xgb          | indicators   | accuracy | 0.1475  | —       | —       |
| mlp          | indicators   | accuracy | 0.0656  | —       | —       |
### severity — Severity regression (normalised [0,1] per type)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| transformer  | timeseries   | R2       | 0.0172  | —       | 0.2787  |
| transformer  | frf_mag      | R2       | -0.0479 | —       | 0.2840  |
| xgb          | modal        | R2       | -0.0849 | —       | 0.2887  |
| rf           | modal        | R2       | -0.1513 | —       | 0.2994  |
| rf           | indicators   | R2       | -0.4224 | —       | 0.3007  |
| xgb          | indicators   | R2       | -2.9886 | —       | 0.5391  |
| cnn          | frf_mag      | R2       | -9.7351 | —       | 0.8298  |
| cnn          | timeseries   | R2       | -50.8302 | —       | 1.2889  |
| mlp          | modal        | R2       | -1156.1252 | —       | 11.1834 |
| mlp          | indicators   | R2       | -2245.5425 | —       | 14.4219 |
### col_location — Column-damage location (storey x end)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| cnn          | timeseries   | accuracy | 0.4286  | —       | —       |
| mlp          | indicators   | accuracy | 0.3673  | —       | —       |
| mlp          | modal        | accuracy | 0.3673  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.2857  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.2041  | —       | —       |
| transformer  | timeseries   | accuracy | 0.1633  | —       | —       |
| rf           | modal        | accuracy | 0.1429  | —       | —       |
| xgb          | indicators   | accuracy | 0.1224  | —       | —       |
| rf           | indicators   | accuracy | 0.0816  | —       | —       |
| xgb          | modal        | accuracy | 0.0612  | —       | —       |
### mass_location — Mass plate location (Base/F1/F2/F3)

| model        | feature      | metric   | test    | val     | mae     |
|--------------|--------------|----------|---------|---------|---------|
| cnn          | timeseries   | accuracy | 0.2500  | —       | —       |
| mlp          | indicators   | accuracy | 0.2500  | —       | —       |
| mlp          | modal        | accuracy | 0.2500  | —       | —       |
| rf           | indicators   | accuracy | 0.2500  | —       | —       |
| rf           | modal        | accuracy | 0.2500  | —       | —       |
| xgb          | modal        | accuracy | 0.2500  | —       | —       |
| cnn          | frf_mag      | accuracy | 0.0000  | —       | —       |
| transformer  | frf_mag      | accuracy | 0.0000  | —       | —       |
| transformer  | timeseries   | accuracy | 0.0000  | —       | —       |
| xgb          | indicators   | accuracy | 0.0000  | —       | —       |
