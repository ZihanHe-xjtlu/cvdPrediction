#' Train Random Forest Model for CVD Prediction
#'
#' Trains a reproducible random forest model (via `ranger`) for cardiovascular disease (CVD) prediction,
#' following BIO215 requirements: no data leakage, cross-validation, and compatibility with Shiny app/R package.
#' Includes preprocessing (imputation, one-hot encoding) fitted **only on training data** to prevent leakage.
#'
#' @param cleaned_data A cleaned `tibble` from `preprocess_heart_data()` (must include `cardio` target column).
#' @param cv_folds An integer specifying the number of cross-validation folds (default: 5, aligns with ML best practices).
#' @param num_trees An integer specifying the number of trees in the random forest (default: 400, balances performance/speed).
#' @return A list with 4 components for downstream use (prediction, evaluation, Shiny):
#'   - `model`: Trained random forest model (from `caret::train()`) with optimized hyperparameters.
#'   - `prep_recipe`: Fitted `recipes` object for preprocessing (imputation + one-hot encoding, trained on training data).
#'   - `test_data`: Hold-out test set (15% of input data) for final model evaluation (never used for training/validation).
#'   - `positive_level`: Character string ("CVD") specifying the positive class for prediction/evaluation.
#' @details
#' Key steps aligned with BIO215 project rubrics:
#' 1. **Leakage Prevention**:
#'    - Data split (70% train, 15% val, 15% test) via stratified sampling (preserves `cardio` class balance).
#'    - Preprocessing (`recipes`): Median imputation for numerics, mode imputation for categorics, one-hot encoding—**all fitted on training data only**.
#' 2. **Model Optimization**:
#'    - Hyperparameter grid search (mtry: 3/7, splitrule: "gini"/"extratrees", min.node.size: 1/5) via 5-fold CV.
#'    - Parallel computing (uses all CPU cores - 1) to speed up CV.
#' 3. **Reproducibility**: Fixed random seed (42) for data splitting and model training.
#' @examples
#' # Load preprocessed data
#' cleaned_data <- preprocess_heart_data("data/heart_data.csv")
#'
#' # Train random forest model
#' cvd_model <- train_cvd_rf(cleaned_data, cv_folds = 5, num_trees = 400)
#'
#' # View best hyperparameters from CV
#' print(cvd_model$model$bestTune)
#' @export
#' @importFrom caret train createDataPartition trainControl confusionMatrix twoClassSummary
#' @importFrom recipes recipe prep bake step_impute_median step_impute_mode step_dummy all_numeric_predictors all_nominal_predictors
#' @importFrom ranger ranger
#' @importFrom doParallel registerDoParallel
#' @importFrom parallel makePSOCKcluster stopCluster
#' @importFrom parallel detectCores
#' @importFrom dplyr select
train_cvd_rf <- function(cleaned_data, cv_folds = 5, num_trees = 400) {
  # Set seed for reproducibility (BIO215 requirement: reproducible software)
  set.seed(42)

  # Step 1: Stratified data split (70% train, 15% val, 15% test)
  # First split: train (70%) vs temp (30%)
  train_idx <- createDataPartition(cleaned_data$cardio, p = 0.7, list = FALSE)
  train_raw <- cleaned_data[train_idx, ]
  temp_raw <- cleaned_data[-train_idx, ]

  # Second split: temp → val (15%) + test (15%)
  val_idx <- createDataPartition(temp_raw$cardio, p = 0.5, list = FALSE)
  val_raw <- temp_raw[val_idx, ]
  test_raw <- temp_raw[-val_idx, ]

  # Step 2: Define preprocessing recipe (fit ONLY on training data to prevent leakage)
  preprocess_recipe <- recipe(cardio ~ ., data = train_raw) %>%
    # Impute missing values (median for numerics, mode for categorics)
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    # One-hot encode categorics (required for random forest compatibility)
    step_dummy(all_nominal_predictors(), one_hot = TRUE)

  # Fit recipe on training data (store training stats for val/test preprocessing)
  prep_recipe <- prep(preprocess_recipe, training = train_raw, retain = TRUE)

  # Apply preprocessing to all splits
  train_ml <- bake(prep_recipe, new_data = train_raw)
  val_ml <- bake(prep_recipe, new_data = val_raw)
  test_ml <- bake(prep_recipe, new_data = test_raw)

  # Verify no missing values post-preprocessing
  if (anyNA(train_ml) | anyNA(val_ml) | anyNA(test_ml)) {
    stop("Preprocessing failed: Missing values remain in train/val/test sets.")
  }

  # Step 3: Set up parallel computing for faster CV
  n_cores <- max(1, detectCores() - 1)  # Avoid using all cores (prevents system lag)
  cl <- makePSOCKcluster(n_cores)
  registerDoParallel(cl)
  # Ensure cluster is stopped when function exits (even if error occurs)
  on.exit({
    try(stopCluster(cl), silent = TRUE)
    registerDoSEQ()
  }, add = TRUE)

  # Step 4: Define CV control and hyperparameter grid
  cv_control <- trainControl(
    method = "cv",
    number = cv_folds,
    classProbs = TRUE,  # Required for AUROC/AUPRC calculation
    summaryFunction = twoClassSummary,  # Optimize for ROC (key BIO215 metric)
    allowParallel = TRUE,
    savePredictions = "final"
  )

  # Hyperparameter grid (balanced for exploration vs speed)
  rf_grid <- expand.grid(
    mtry = c(3, 7),  # Number of features sampled per split
    splitrule = c("gini", "extratrees"),  # Split criterion
    min.node.size = c(1, 5)  # Minimum samples per leaf (prevents overfitting)
  )

  # Step 5: Train random forest model
  rf_model <- train(
    x = train_ml %>% select(-cardio),  # Features
    y = train_ml$cardio,  # Target
    method = "ranger",
    metric = "ROC",  # Optimize for AUROC (BIO215 required metric)
    trControl = cv_control,
    tuneGrid = rf_grid,
    num.trees = num_trees,
    importance = "impurity",  # For feature importance plots (BIO215 requirement)
    seed = 42  # Additional seed for ranger reproducibility
  )

  # Step 6: Return model and supporting objects (for prediction/Shiny)
  result <- list(
    model = rf_model,
    prep_recipe = prep_recipe,
    test_data = test_ml,
    positive_level = "CVD"
  )

  return(result)
}
