# Model Metrics Comparison

## Validation Metrics

|               |       r2 |     mae |     mse |    rmse |
|:--------------|---------:|--------:|--------:|--------:|
| linear_reg    | 0.902939 | 3.47037 | 18.5873 | 4.3113  |
| decision_tree | 0.903875 | 3.30013 | 18.408  | 4.29046 |
| catboost      | 0.923838 | 2.91787 | 14.5851 | 3.81904 |
| xgboost       | 0.907289 | 3.27174 | 17.7543 | 4.21358 |
| nn            | 0.897646 | 3.50703 | 19.6009 | 4.42729 |

## Best Model

Based on R² score, the best model is **catboost** with an R² of 0.9238.

## Analysis

Reasons why this model performed best:


