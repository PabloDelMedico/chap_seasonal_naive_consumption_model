# Seasonal naive consumption

CHAP model for the `health_supply` domain. Forecasts monthly consumption as the value of the same month one year earlier (or the average of recent months when that value is missing). Uncertainty comes from the model's own past seasonal errors. Missing entries and stock-out months are treated as not observed.

Declares `target: consumption`, `domain: health_supply` and the required covariate
`stock_on_hand` in `MLproject`. Monthly data only.

Compare against the AMC baseline:
https://github.com/PabloDelMedico/chap_naive_consumption_model
