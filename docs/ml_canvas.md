# ML Canvas: Bank Term-Deposit Propensity

## Value proposition

A Portuguese retail bank runs telemarketing campaigns to sell term deposits. Only about 11% of contacted clients subscribe, so most calls are wasted cost and operator time. The model ranks clients by their probability of subscribing, so the bank calls the most promising clients first and avoids calling clients with very low propensity.

## Prediction task

- **Type:** binary classification.
- **Target:** `subscribed` (yes or no): will the client subscribe to a term deposit after the campaign contact?
- **Output:** a probability score between 0 and 1 for each client.

## Decisions

- Rank the campaign contact list by score and call the top clients first.
- Choose the decision threshold that maximizes expected campaign profit, instead of the default 0.5.
- Use the findings to cap the number of contacts per client.

## Data sources

- Bank Marketing dataset (UCI Machine Learning Repository, `bank-additional-full.csv`).
- 41,188 contacts from May 2008 to November 2010, with client profile, credit status, contact details, previous campaign history and macroeconomic indicators.

## Features

- **Client profile:** age, job, marital status, education.
- **Credit relationship:** credit default, housing loan, personal loan.
- **Current contact:** channel, month, weekday, number of contacts in the campaign.
- **Campaign history:** days since previous contact, previous contacts, previous outcome.
- **Macroeconomic context:** employment variation rate, consumer price index, consumer confidence index, Euribor 3M, number of employees.
- **Excluded:** `last_call_duration_sec`, because it is only known after the call (data leakage).

## Making predictions

Batch scoring before each campaign: the bank scores the full contact list and the campaign team receives the ranked list.

## Building models

- Classical Scikit-Learn classifiers: Logistic Regression (baseline), Random Forest and HistGradientBoosting.
- Reproducible pipeline versioned with DVC, experiments tracked with MLflow and the best model promoted in the MLflow Model Registry.
- Retraining when monitoring shows performance degradation or when a new campaign period is available.

## Offline evaluation

- **Main metric:** PR-AUC, suited to the 11% positive class.
- **Supporting metrics:** ROC-AUC, precision, recall and F1 for the subscriber class.
- **Business metrics:** lift in the top deciles and expected campaign profit (cost per call against margin per deposit).

## Monitoring

| What | How | Frequency |
|---|---|---|
| Realized conversion of called clients against predicted probability | Compare campaign results with the scores logged at scoring time | After each campaign |
| PR-AUC and lift on labeled campaign results | Recompute metrics on the latest campaign and compare with the offline baseline | After each campaign |
| Feature drift, mainly macroeconomic indicators | Compare current feature distributions with the training data (for example with the Population Stability Index) | Monthly |
| Score distribution | Track the share of clients above the decision threshold | Every scoring run |

Retraining is triggered when PR-AUC falls more than 10% below the offline baseline or when macroeconomic drift is detected.
