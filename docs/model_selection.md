# Model Selection

All candidates were trained with the same data, the same stratified 5-fold cross-validation and the same seed, so the only difference between runs is the model. Every run is tracked in MLflow (experiment `bank-propensity`).

## Results (5-fold cross-validation on the training set)

| Model | Configuration | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|---|
| Random Forest | `max_depth=15`, 300 trees | 0.4673 | 0.7946 | 0.4802 |
| **HistGradientBoosting** | **`learning_rate=0.05`, 500 trees** | **0.4666** | **0.7976** | **0.4721** |
| HistGradientBoosting | `learning_rate=0.1`, 300 trees | 0.4662 | 0.7955 | 0.4720 |
| Logistic Regression (baseline) | `C=1.0` | 0.4439 | 0.7896 | 0.4482 |
| Logistic Regression | `C=0.1` | 0.4424 | 0.7896 | 0.4464 |

A random model would reach a PR-AUC close to the positive rate (0.113), so all candidates are about 4 times better than random targeting.

## Findings

- **Tree-based models beat the linear baseline** by about 0.023 PR-AUC (around 5% relative). They capture interactions between features, such as channel, previous outcome and interest rate context, that a linear model cannot represent.
- **Random Forest and HistGradientBoosting are tied.** The PR-AUC gap between the best tree models is about 0.001, smaller than the natural variation across cross-validation folds.
- **Stronger regularization did not help the logistic regression**, since the linear model is already simple.

## Decision

**Champion: HistGradientBoosting with `learning_rate=0.05` and 500 trees.**

With the main metric tied, the tie-breakers were:

1. **Best ROC-AUC** among all candidates.
2. **Smaller and faster model** than a 300-tree Random Forest with depth 15, which matters for the Model Registry, the Docker image and batch scoring of large contact lists.
3. **Lower overfitting risk**, thanks to the smaller learning rate.

**Interpretability trade-off:** the logistic regression is easier to explain but loses about 5% of PR-AUC. For a marketing campaign, where the model ranks clients to call and does not approve or deny credit, this gain justifies the less interpretable model. A credit decision model would require a different balance.
