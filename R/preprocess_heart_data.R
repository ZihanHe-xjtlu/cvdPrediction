#' Preprocess Heart Disease Dataset for CVD Prediction
#'
#' Cleans and engineers features for the Kaggle Heart Disease dataset (tabular, biology/medicine-related)
#' to prepare it for machine learning. Key steps include ID column removal, age unit conversion,
#' categorical variable encoding, medical feature engineering (BMI, pulse pressure), and outlier flagging.
#'
#' @param data_path A character string specifying the file path to the raw CSV dataset (e.g., "data/heart_data.csv").
#' @return A cleaned `tibble` with:
#'   - Target variable `cardio` (factor: "CVD" for positive class, "NoCVD" for negative class)
#'   - Engineered features: `age_years` (age in years), `bmi` (body mass index), `pulse_pressure` (systolic - diastolic blood pressure)
#'   - Outliers marked as `NA` (to be imputed later using training data only, preventing leakage)
#' @details
#' Critical preprocessing steps aligned with BIO215 project requirements:
#' 1. **ID Column Removal**: Drops non-predictive columns like "index", "id", or "patient_id" to avoid model bias.
#' 2. **Age Standardization**: Converts age from days to years if median age > 200 (common dataset formatting issue).
#' 3. **Categorical Encoding**: Converts variables like `gender` (1→"Female", 2→"Male"), `cholesterol`, and `smoke` to factors.
#' 4. **Medical Feature Engineering**:
#'    - BMI = weight (kg) / (height (m))² (derived from `height` and `weight` columns)
#'    - Pulse Pressure = `ap_hi` (systolic BP) - `ap_lo` (diastolic BP) (relevant for cardiovascular risk assessment)
#' 5. **Outlier Flagging**: Marks implausible values as `NA` (e.g., systolic BP <70/ >250, height <120/ >220 cm) to avoid skewing model training.
#' @examples
#' # Load and preprocess sample dataset
#' cleaned_data <- preprocess_heart_data("data/heart_data.csv")
#'
#' # View first 5 rows and structure
#' head(cleaned_data)
#' str(cleaned_data)
#' @export
#' @importFrom dplyr select mutate filter all_of
#' @importFrom tibble as_tibble
#' @importFrom stats median
preprocess_heart_data <- function(data_path) {
  # Load raw data (suppress warnings for non-standard column types)
  df_raw <- read.csv(data_path, stringsAsFactors = FALSE, warn = FALSE)
  df <- as_tibble(df_raw)

  # Step 1: Remove ID-like columns (non-predictive)
  drop_cols <- intersect(c("index", "id", "ID", "patient_id", "row_id"), names(df))
  if (length(drop_cols) > 0) {
    df <- df %>% select(-all_of(drop_cols))
  }

  # Step 2: Prepare target variable (cardio: 0→NoCVD, 1→CVD; positive class first for caret)
  if (!"cardio" %in% names(df)) {
    stop("Dataset must contain 'cardio' column (target variable for CVD prediction).")
  }
  df$cardio <- factor(df$cardio, levels = c(1, 0), labels = c("CVD", "NoCVD"))

  # Step 3: Convert age from days to years (if needed)
  if ("age" %in% names(df)) {
    df$age <- as.numeric(df$age)
    if (median(df$age, na.rm = TRUE) > 200) {
      df <- df %>% mutate(age_years = age / 365.25) %>% select(-age)
    } else {
      df <- df %>% mutate(age_years = age) %>% select(-age)
    }
  } else {
    stop("Dataset must contain 'age' column for age standardization.")
  }

  # Step 4: Encode categorical variables as factors
  cat_vars <- intersect(c("gender", "cholesterol", "gluc", "smoke", "alco", "active"), names(df))
  if (length(cat_vars) > 0) {
    df[cat_vars] <- lapply(df[cat_vars], factor)
  }

  # Standardize gender labels (1→Female, 2→Male; common Kaggle encoding)
  if ("gender" %in% names(df)) {
    df$gender <- factor(df$gender, levels = c(1, 2), labels = c("Female", "Male"))
  }

  # Step 5: Engineer medical features (BMI + Pulse Pressure)
  # BMI calculation (requires height and weight)
  if (all(c("height", "weight") %in% names(df))) {
    df <- df %>%
      mutate(
        height_m = height / 100,  # Convert cm to meters
        bmi = weight / (height_m ^ 2)
      ) %>%
      select(-height_m)
  }

  # Pulse Pressure calculation (requires systolic/diastolic BP)
  if (all(c("ap_hi", "ap_lo") %in% names(df))) {
    df <- df %>% mutate(pulse_pressure = ap_hi - ap_lo)
  }

  # Step 6: Flag outliers as NA (to impute later with training data only)
  # Blood pressure outliers
  if (all(c("ap_hi", "ap_lo") %in% names(df))) {
    df <- df %>%
      mutate(
        ap_hi = ifelse(ap_hi < 70 | ap_hi > 250, NA, ap_hi),  # Implausible systolic BP
        ap_lo = ifelse(ap_lo < 40 | ap_lo > 160, NA, ap_lo),  # Implausible diastolic BP
        ap_hi = ifelse(!is.na(ap_hi) & !is.na(ap_lo) & ap_hi <= ap_lo, NA, ap_hi)  # Logical inconsistency
      )
  }

  # Height/weight outliers
  if ("height" %in% names(df)) {
    df$height <- ifelse(df$height < 120 | df$height > 220, NA, df$height)  # Implausible height (cm)
  }
  if ("weight" %in% names(df)) {
    df$weight <- ifelse(df$weight < 35 | df$weight > 180, NA, df$weight)    # Implausible weight (kg)
  }

  return(df)
}
