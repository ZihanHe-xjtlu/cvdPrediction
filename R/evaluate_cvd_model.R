#' Evaluate CVD Model Performance (BIO215 Required Metrics)
#'
#' Computes and returns all BIO215-mandated ML performance metrics for classification:
#' AUROC, AUPRC, accuracy, confusion matrix, and feature importance. Generates interpretable
#' outputs for the project research report (Results section) and README.
#'
#' @param model_obj A list from `train_cvd_rf()` containing:
#'   - `model`: Trained random forest model.
#'   - `test_data`: Hold-out test set (unseen data for unbiased evaluation).
#'   - `positive_level`: Positive class label ("CVD").
#' @return A list with 6 components for report/dashboard use:
#'   - `confusion_matrix`: Confusion matrix (from `caret`) with accuracy, sensitivity, specificity.
#'   - `auroc`: Numeric value of AUROC (Area Under ROC Curve, range: 0-1).
#'   - `auprc`: Numeric value of AUPRC (Area Under Precision-Recall Curve, range: 0-1).
#'   - `feature_importance`: `tibble` of feature importance (sorted by permutation importance).
#'   - `roc_curve_data`: `data.frame` for ROC plot (FPR vs TPR).
#'   - `pr_curve_data`: `data.frame` for PR plot (Recall vs Precision).
#' @details
#' Aligns with BIO215 rubrics for ML model interpretation and reporting:
#' 1. **Required Metrics**:
#'    - Classification: AUROC, AUPRC, accuracy, confusion matrix (sensitivity/specificity).
#'    - Model Interpretation: Feature importance (permutation-based, more reliable than impurity).
#' 2. **Visualization Data**: Provides ROC/PR curve data for `ggplot2` plotting (required for Results section).
#' 3. **Unbiased Evaluation**: Uses only the hold-out test set (15% of input data) to avoid overestimating performance.
#' @examples
#' # Load preprocessed data, train model, and evaluate
#' cleaned_data <- preprocess_heart_data("data/heart_data.csv")
#' cvd_model <- train_cvd_rf(cleaned_data)
#' eval_results <- evaluate_cvd_model(cvd_model)
#'
#' # Print key metrics (for report)
#' cat("Test Set AUROC:", round(eval_results$auroc, 3), "\n")
#' cat("Test Set AUPRC:", round(eval_results$auprc, 3), "\n")
#' print(eval_results$confusion_matrix)
#'
#' # Plot ROC curve (for Results section)
#' library(ggplot2)
#' ggplot(eval_results$roc_curve_data, aes(x = FPR, y = TPR)) +
#'   geom_line(linewidth = 1.2, color = "blue") +
#'   geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
#'   labs(title = "ROC Curve (Test Set)",
#'        subtitle = paste("AUROC =", round(eval_results$auroc, 3)),
#'        x = "False Positive Rate", y = "True Positive Rate") +
#'   theme_classic()
#' @export
#' @importFrom pROC roc auc
#' @importFrom PRROC pr.curve
#' @importFrom caret confusionMatrix
#' @importFrom dplyr tibble arrange desc
#' @importFrom ranger importance
evaluate_cvd_model <- function(model_obj) {
  # Extract test data and model (unseen data for unbiased evaluation)
  test_ml <- model_obj$test_data
  rf_model <- model_obj$model
  positive_level <- model_obj$positive_level

  # Step 1: Generate predictions on test set
  test_pred_class <- predict(rf_model, newdata = test_ml)
  test_pred_prob <- predict(rf_model, newdata = test_ml, type = "prob")[, positive_level]

  # Step 2: Compute confusion matrix (accuracy, sensitivity, specificity)
  cm <- confusionMatrix(test_pred_class, test_ml$cardio, positive = positive_level)

  # Step 3: Compute AUROC (BIO215 required metric)
  roc_obj <- roc(test_ml$cardio, test_pred_prob, levels = rev(levels(test_ml$cardio)), quiet = TRUE)
  auroc <- as.numeric(auc(roc_obj))
  # Format ROC curve data for plotting
  roc_curve_data <- data.frame(
    FPR = 1 - roc_obj$specificities,  # False Positive Rate
    TPR = roc_obj$sensitivities       # True Positive Rate
  )

  # Step 4: Compute AUPRC (BIO215 required metric, critical for imbalanced data)
  pr_obj <- pr.curve(
    scores.class0 = test_pred_prob[test_ml$cardio == positive_level],  # Positive class probabilities
    scores.class1 = test_pred_prob[test_ml$cardio != positive_level],  # Negative class probabilities
    curve = TRUE  # Return curve data for plotting
  )
  auprc <- pr_obj$auc.integral
  # Format PR curve data for plotting
  pr_curve_data <- data.frame(
    Recall = pr_obj$curve[, 1],
    Precision = pr_obj$curve[, 2],
    Threshold = pr_obj$curve[, 3]
  )

  # Step 5: Compute feature importance (BIO215 model interpretation requirement)
  # Use permutation importance (more reliable than impurity for reporting)
  if ("permutation" %in% names(rf_model$finalModel$variable.importance)) {
    imp_vec <- importance(rf_model$finalModel, type = "permutation")
  } else {
    # Fallback to impurity if permutation not available (e.g., early model versions)
    imp_vec <- importance(rf_model$finalModel, type = "impurity")
  }
  feature_importance <- tibble(
    Feature = names(imp_vec),
    Importance = as.numeric(imp_vec)
  ) %>% arrange(desc(Importance))

  # Step 6: Compile results for report/dashboard
  eval_result <- list(
    confusion_matrix = cm,
    auroc = auroc,
    auprc = auprc,
    feature_importance = feature_importance,
    roc_curve_data = roc_curve_data,
    pr_curve_data = pr_curve_data
  )

  return(eval_result)
}
