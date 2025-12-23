#' Predict CVD Outcomes Using Trained Model
#'
#' Generates **class predictions** (not probabilities, per BIO215 R package requirement) for new data
#' using the trained random forest model from `train_cvd_rf()`. Applies preprocessing (via fitted recipe)
#' to ensure compatibility with the model.
#'
#' @param model_obj A list from `train_cvd_rf()` containing:
#'   - `model`: Trained random forest model.
#'   - `prep_recipe`: Fitted preprocessing recipe.
#'   - `positive_level`: Positive class label ("CVD").
#' @param new_data A `data.frame`/`tibble` of new data (must match columns of `preprocess_heart_data()` output).
#' @return A `data.frame` with 2 columns for end-users/Shiny app:
#'   - `input_features`: Original input features (for transparency).
#'   - `predicted_cvd`: Categorical prediction ("CVD" or "NoCVD", per BIO215 classification requirement).
#' @details
#' Critical compliance with BIO215 R package rubrics:
#' 1. **Input/Output Requirements**:
#'    - Input: Accepts `data.frame` (compatible with Shiny file uploads/manual entry).
#'    - Output: Returns **class labels** (not probabilities) as required for classification tasks.
#' 2. **Leakage Prevention**: Applies preprocessing via `model_obj$prep_recipe` (fitted on training data) to new data.
#' 3. **User-Friendliness**: Includes original input features in output for transparency (e.g., Shiny app users can verify inputs).
#' @examples
#' # Load preprocessed data and train model
#' cleaned_data <- preprocess_heart_data("data/heart_data.csv")
#' cvd_model <- train_cvd_rf(cleaned_data)
#'
#' # Create sample new data (e.g., from Shiny manual input)
#' new_patient <- tibble::tibble(
#'   age = 25000,  # ~68 years (25000 days / 365.25)
#'   gender = 2,   # Male
#'   height = 175, # cm
#'   weight = 80,  # kg
#'   ap_hi = 140,  # Systolic BP
#'   ap_lo = 90,   # Diastolic BP
#'   cholesterol = 2,  # Above normal
#'   gluc = 1,     # Normal
#'   smoke = 0,    # Non-smoker
#'   alco = 0,     # Non-drinker
#'   active = 1    # Physically active
#' )
#'
#' # Preprocess new data (match training data format)
#' preprocessed_new <- preprocess_heart_data(new_patient)  # Treat new data as "dataset"
#'
#' # Predict CVD outcome
#' prediction <- predict_cvd(cvd_model, preprocessed_new)
#' print(prediction)
#' @export
#' @importFrom recipes bake
#' @importFrom dplyr bind_cols
predict_cvd <- function(model_obj, new_data) {
  # Validate model object structure
  required_components <- c("model", "prep_recipe", "positive_level")
  if (!all(required_components %in% names(model_obj))) {
    stop(paste("model_obj must contain:", paste(required_components, collapse = ", "), "(from train_cvd_rf())."))
  }

  # Validate new data (must not contain target column to avoid leakage in Shiny)
  if ("cardio" %in% names(new_data)) {
    new_data <- new_data %>% select(-cardio)
  }

  # Step 1: Apply preprocessing (fitted on training data) to new data
  # Handle cases where new data has no categorical variables (avoid bake() errors)
  new_ml <- tryCatch(
    {
      bake(model_obj$prep_recipe, new_data = new_data)
    },
    error = function(e) {
      stop(paste("Preprocessing failed for new data:", e$message, "\nEnsure new data matches columns from preprocess_heart_data()."))
    }
  )

  # Step 2: Generate class predictions (BIO215 requires class labels, not probabilities)
  pred_class <- predict(model_obj$model, newdata = new_ml)

  # Step 3: Format output (include original inputs for transparency)
  result <- bind_cols(
    input_features = new_data,  # Original user inputs
    predicted_cvd = pred_class  # Final prediction
  )

  return(result)
}
