# Methodology and reproduction decisions

## Definitions

For policy i, let e_i be exposure in years, n_i the adjusted number of claims, and s_i the adjusted aggregate claim amount. Targets are frequency n_i/e_i, severity s_i/n_i for positive-loss policies, and pure premium s_i/e_i.

The frequency model uses a log link: E[n_i | x_i, e_i] = e_i exp(x_i beta). Fitting n_i/e_i with exposure weights is the rate-model formulation of an exposure offset. The Gamma model uses a log link and count weights. Its positive-loss training population and the benchmark's count adjustment must be remembered when interpreting frequency × severity as expected annual loss.

The direct model minimizes exposure-weighted Tweedie deviance with power 1.9 and a log link. The compound Poisson–Gamma interpretation motivates the family; it does not assert that the empirical severity distribution is exactly Gamma.

## Data preparation

Aggregate the severity table by policy ID, then left-join it to the frequency table. Missing matched severity amounts become zero. Preserve all frequency policies. Audit severity rows whose ID is absent from the frequency table rather than merging them into unrelated policies. Reject duplicate frequency IDs and nonpositive exposure.

Follow the worked example: cap policy-level aggregate loss at EUR 200,000, exposure at one year, and count at four. If a policy has zero aggregate loss, reset its claim count to zero. These rules define a **modified benchmark target**; they are not proven corrections to each original observation and do not price the excluded loss tail.

The original authors' historical R data snapshot is the default source because the OpenML endpoint was unreachable during development. Counts and fields are checked, but row counts alone do not establish equality to OpenML. The manifest distinguishes the actual download source from the corresponding OpenML catalogue IDs.

## Split and feature encoding

Use a 75/25 random split, seed 0, by unique policy ID. Fit all preprocessing on training data only: 10 quantile bins for vehicle and driver ages; one-hot encoding for brand, power, fuel, region and area; raw bonus-malus; standardized log population density. Quantile binning uses a deterministic training subsample of at most 200,000 rows. Unknown test categories are ignored rather than learned from test data.

This differs deliberately from the tutorial's full-data preprocessing. It does not remove all possible entity leakage: the data may contain repeated exposures or similar risks with distinct IDs. Group-aware and temporal validation are further work.

## Fixed hyperparameters

Poisson alpha = 0.0001, Gamma alpha = 10, Tweedie alpha = 0.1, Tweedie power = 1.9. All use the newton-cholesky solver, max_iter = 100 and tol = 1e-4. These are fixed benchmark settings, not the output of an optimization against the test set. Full categorical encodings are stabilized by L2 regularization.

## Evaluation

- Evaluate pure-premium deviance using exposure weights and the same evaluation power for every compared model.
- Exposure-weighted MAE is supplementary; it is not the sole actuarial criterion.
- Total expected losses are sum(e_i × predicted pure premium). Report expected / actual loss and group calibration.
- The ordered Lorenz curve sorts increasing predicted risk and plots cumulative loss share against cumulative exposure share. Tied scores are combined before integration. Gini = 1 − 2 × curve area. This is a raw ranking Gini, not normalized against an oracle.
- Risk deciles use approximate equal exposure under the Tweedie score and keep tied scores together. Both models are compared on the same groups. Empty groups are possible.
- The baseline is the training portfolio's total losses divided by total exposure, never the test mean.

## Sensitivity

Rebuild targets and refit models at EUR 50,000 and with no loss cap, keeping exposure/count rules, policy IDs, features and hyperparameters unchanged. Evaluate each model on its matching scenario target. Compare model performance **within** a scenario; absolute deviance across different caps is not an apples-to-apples metric. These scenarios include both the changed target and its fitted model's response, not a pure isolated parameter perturbation.

## Scope

One split cannot establish significance. Policy-level losses are sparse and heavy-tailed. The study does not model future inflation, claims development, capital, expenses or demand. Use the reproduction to explain model assumptions and diagnostics; it is not an operational quotation engine.
