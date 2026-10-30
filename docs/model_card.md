# Model Card: Bank Term-Deposit Propensity

## Model details

| Item | Value |
|---|---|
| Model | HistGradientBoostingClassifier (Scikit-Learn) inside a pipeline with StandardScaler and OneHotEncoder |
| Hyperparameters | `learning_rate=0.05`, `max_iter=500`, `max_leaf_nodes=31`, `class_weight="balanced"` |
| Registered as | `bank-propensity-classifier` in the MLflow Model Registry, served through the `@champion` alias |
| Input | 19 client features (9 numeric, 10 categorical), validated by the MLflow model signature |
| Output | Probability of subscribing to a term deposit (0 to 1) |
| Training code | `src/bank_propensity_pipeline/models/train.py`, reproducible with `dvc repro` |
| Owner | Breno Carvalho |

## Intended use

- **Primary use:** rank the contact list before a telemarketing campaign for term deposits, so the campaign team calls the clients with the highest propensity first.
- **Decision rule:** call clients scored at or above the profit-maximizing threshold (0.53 with the current cost and margin assumptions).
- **Users:** marketing and campaign planning teams, with the data science team responsible for monitoring.

## Out-of-scope uses

- **Credit decisions** (approval, limits, pricing): the model was not built or validated for credit risk, and its features include personal attributes that require a dedicated fairness and regulatory review in that context.
- **Products other than term deposits**, or channels other than outbound phone calls, without retraining and validation.
- **Interpreting the score as a calibrated probability:** training with balanced class weights inflates the scores, so they are useful for ranking and for the threshold, not as an exact chance of subscription.

## Training data

- **Source:** Bank Marketing dataset (UCI Machine Learning Repository, `bank-additional-full.csv`), contacts from a Portuguese bank between May 2008 and November 2010.
- **Size:** 41,188 contacts. Stratified split: 32,950 for training and 8,238 for testing (80/20, seed 42).
- **Target:** 11.3% of clients subscribed.
- **Preparation:** `last_call_duration_sec` removed (data leakage, only known after the call); `days_since_previous_contact = 999` replaced by a binary "previously contacted" flag; `"unknown"` values kept as their own category.

## Performance

**Cross-validation** (5 stratified folds on the training set): PR-AUC 0.467, ROC-AUC 0.798.

**Held-out test set** (never used for training or model selection):

| Metric | Value |
|---|---|
| PR-AUC | 0.497 (random targeting ≈ 0.113) |
| ROC-AUC | 0.816 |
| Precision at threshold 0.53 | 43.5% of called clients subscribe |
| Recall at threshold 0.53 | 64.3% of all subscribers are reached |
| Share of clients called | 16.7% |
| Lift in the top decile | 4.8x (53.9% conversion against 11.3%) |

**Business impact** (illustrative assumptions: €8 per call, €50 margin per deposit): calling every client in the test set yields a loss of €19,504, while calling the clients selected by the model yields a profit of €18,866.

The comparison with Logistic Regression and Random Forest is documented in [model selection](model_selection.md).

## Limitations and risks

- **Economic drift:** the data covers the 2008 financial crisis, and macroeconomic features (Euribor, employment, consumer confidence) strongly influence the scores. In a different interest rate scenario, performance may drop.
- **Time period and geography:** the model learned the behavior of one Portuguese bank between 2008 and 2010. It must be revalidated before use in another market or period.
- **Optimistic profit estimate:** the decision threshold was selected on the test set. A separate validation set would give an unbiased estimate.
- **Uncalibrated scores:** balanced class weights inflate the probabilities, so the threshold must be revisited whenever costs, margins or the model change.
- **Input contract:** integer columns in the model signature do not accept missing values. Missing information must keep the `"unknown"` encoding used in training.

## Ethical considerations

- The model uses **age, marital status and job** as features. Even in marketing, this can lead to some groups being contacted much less often. Conversion and contact rates should be reviewed by group before production use.
- Client data must be processed under the **LGPD** (Brazil) or the equivalent local regulation, with a legal basis for marketing use and respect for opt-out requests.
- **Contact pressure:** the analysis showed that conversion falls sharply after many calls. A cap on contacts per client protects both the customer experience and the bank's reputation.

## Monitoring plan

Scores are produced in batch before each campaign. Each scoring run logs the client identifier, the score, the model version (`@champion` resolves to a registry version) and the date. Campaign outcomes are joined to these scores when the campaign closes.

| What is monitored | How | Tool | Frequency | Action trigger |
|---|---|---|---|---|
| **Score distribution** (share of clients above the threshold, mean score) | Compare each scoring batch with the distribution observed on the test set | Scoring job logs the summary statistics as an MLflow run in a `monitoring` experiment | Every scoring run | Share above the threshold moves more than 50% from the reference (16.7%): investigate input data before the campaign starts |
| **Feature drift**, mainly macroeconomic indicators | Population Stability Index (PSI) of each feature, current batch against the training data | Monitoring script logs one PSI value per feature to MLflow | Monthly, and before each campaign | PSI above 0.25 on any key feature: revalidate the model; above 0.1: watch closely |
| **Realized performance** (PR-AUC, lift in the top decile, precision of called clients) | Recompute the metrics with `evaluate.py` functions on scores joined with real campaign outcomes | MLflow run per campaign, compared with the offline baseline (PR-AUC 0.497, lift 4.8x) | After each campaign closes | PR-AUC more than 10% below the baseline, or top-decile lift below 3x: trigger retraining |
| **Business result** (realized profit per campaign) | Real call cost and real deposit margin applied to campaign results | Campaign report shared with the marketing team | After each campaign | Realized profit below the profit of a simple rule (for example, calling only previously successful clients): review the threshold and the model |

## Retraining and release

1. Add the latest labeled campaigns to the training data and run `dvc repro`.
2. Compare the new run with the current champion in MLflow (same metrics, same test protocol).
3. If the new model is better, run `registry.py`: it creates a new version and moves `@champion`.
4. If a problem appears after release, roll back by pointing `@champion` to the previous version. No code change is needed.
