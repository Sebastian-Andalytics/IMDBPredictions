# Load necessary libraries
library(ggplot2)
library(dplyr)
library(car)
library(psych)
library(gridExtra)
library(lmtest)
library(plm)
library(parallel)
library(caTools)
library(splines)


# Load the dataset
imdb_data = read.csv("/Users/marcpujol/Desktop/MGSC 401/Group Project/IMDB_data_Winter_2025.csv")
attach(imdb_data)

# View dataset structure
head(imdb_data)
str(imdb_data)
summary(imdb_data)


######################################################
################ Step 2 and 3 ########################
######################################################


# ---- Data Preparation ----
# Convert categorical variables into meaningful dummy variables

create_dummies <- function(data, column, top_n = 5) {
  top_values <- names(sort(table(data[[column]]), decreasing = TRUE))[1:top_n]
  data[[paste0(column, "_grouped")]] <- ifelse(data[[column]] %in% top_values, data[[column]], "Other")
  dummies <- model.matrix(~ get(paste0(column, "_grouped")) - 1, data = data)
  return(cbind(data, dummies))
}

imdb_data <- create_dummies(imdb_data, "distributor")
imdb_data$colour_film_dummy <- as.integer(imdb_data$colour_film == "Color")
imdb_data <- create_dummies(imdb_data, "production_company")
imdb_data$release_month <- as.factor(imdb_data$release_month)
imdb_data <- cbind(imdb_data, model.matrix(~ release_month - 1, data = imdb_data))
imdb_data <- create_dummies(imdb_data, "director")

# ---- Outlier Detection & Removal using Bonferroni Technique ----
numeric_vars <- names(imdb_data)[sapply(imdb_data, is.numeric)]
numeric_data <- imdb_data[, numeric_vars]
outlier_indices <- unique(unlist(lapply(numeric_data, function(var) {
  test <- outlierTest(lm(var ~ 1))
  if (!is.null(test)) as.numeric(names(test$rstudent[test$p < 0.05])) else NULL
})))
imdb_data <- imdb_data[-outlier_indices, ]

# ---- Correlation Matrix & Visualization ----
numeric_vars <- names(imdb_data)[sapply(imdb_data, is.numeric) & names(imdb_data) != "imdb_score"]
non_constant_vars <- numeric_vars[sapply(imdb_data[, numeric_vars], function(x) sd(x, na.rm = TRUE) > 0 & length(unique(x)) > 2)]
correlation_matrix <- cor(imdb_data[, c("imdb_score", non_constant_vars)], use = "complete.obs")
print("Correlation Matrix:")
print(round(correlation_matrix, 3))

# Extract and sort top correlated variables
imdb_correlation_df <- data.frame(
  Variable = rownames(correlation_matrix)[-1],  # Ensure proper variable extraction
  Correlation = correlation_matrix["imdb_score", -1]
)
imdb_correlation_df <- imdb_correlation_df[order(abs(imdb_correlation_df$Correlation), decreasing = TRUE), ]
top5_vars <- head(imdb_correlation_df$Variable, 5)

# ---- Histograms & Box Plots ----
plot_variable <- function(var, plot_type) {
  if (plot_type == "hist") {
    ggplot(imdb_data, aes_string(x = var)) +
      geom_histogram(fill = "steelblue", color = "black", alpha = 0.7, bins = 30) +
      labs(title = paste("Histogram of", var), x = var, y = "Frequency") +
      theme_minimal()
  } else {
    ggplot(imdb_data, aes_string(x = "factor(1)", y = var)) +  # Ensure vertical box plots
      geom_boxplot(fill = "lightblue") +
      labs(title = paste("Box Plot of", var), x = "", y = var) +
      theme_minimal()
  }
}

# Ensure no duplicate column names in the dataset
imdb_data <- imdb_data[, !duplicated(names(imdb_data))]

do.call(grid.arrange, c(lapply(top5_vars, function(var) plot_variable(var, "hist")), ncol = 3))
do.call(grid.arrange, c(lapply(top5_vars, function(var) plot_variable(var, "box")), ncol = 3))

# ---- Heteroskedasticity Test & Correction ----
ncv_results <- data.frame(Variable = character(), P_Value = numeric())
for (var in non_constant_vars) {
  test <- ncvTest(lm(as.formula(paste("imdb_score ~", var)), data = imdb_data))  # Correct formula construction
  if (!is.null(test)) ncv_results <- rbind(ncv_results, data.frame(Variable = var, P_Value = test$p))
}

# Check and print significant heteroskedasticity results
if (nrow(ncv_results) > 0 && any(ncv_results$P_Value < 0.05)) {
  print("Significant Heteroskedasticity:")
  print(ncv_results[ncv_results$P_Value < 0.05, ])
} else {
  print("No significant heteroskedasticity detected.")
}

# Apply robust standard errors if necessary
if (nrow(ncv_results) > 0 && any(ncv_results$P_Value < 0.05)) {
  imdb_data$index <- 1:nrow(imdb_data)
  panel_data <- pdata.frame(imdb_data, index = "index")
  for (var in ncv_results$Variable) {
    plm_model <- plm(as.formula(paste("imdb_score ~", var)), data = panel_data, model = "pooling")
    cat("\nCorrected Model Results for", var, ":\n")
    print(summary(plm_model))
  }
}

# ---- Variance Inflation Factor (VIF) ----
vif_model <- lm(imdb_score ~ ., data = imdb_data[, c("imdb_score", top5_vars)])
vif_values <- vif(vif_model)
print("VIF Values:")
print(vif_values)

# ---- Linear Regressions ----
regression_results <- do.call(rbind, lapply(non_constant_vars, function(var) {
  model <- lm(as.formula(paste("imdb_score ~", var)), data = imdb_data)  # Correct formula usage
  data.frame(Variable = var, R_Squared = summary(model)$r.squared, MSE = mean(model$residuals^2))
}))

print("Top 5 Variables by R-Squared:")
print(regression_results[order(-regression_results$R_Squared), ][1:5, ])


################################################
################ Step 4 ########################
################################################

# ---- Extract and sort top correlated variables ----

# Remove constant columns to avoid zero standard deviation errors
numeric_vars <- names(imdb_data)[sapply(imdb_data, is.numeric)]
non_constant_vars <- numeric_vars[sapply(imdb_data[, numeric_vars], function(x) sd(x, na.rm = TRUE) > 0)]

# Compute correlation matrix excluding constant columns
correlation_matrix <- cor(imdb_data[, non_constant_vars], use = "complete.obs")
correlation_matrix[is.na(correlation_matrix)] <- 0  # Replace NA correlations with 0

# Ensure response variable is not included as a predictor
correlation_matrix <- correlation_matrix[!rownames(correlation_matrix) %in% "imdb_score", ]

imdb_correlation_df <- data.frame(
  Variable = rownames(correlation_matrix),
  Correlation = correlation_matrix[, "imdb_score"]
)
imdb_correlation_df <- imdb_correlation_df[order(abs(imdb_correlation_df$Correlation), decreasing = TRUE), ]
top10_vars <- head(imdb_correlation_df$Variable, 10)  # Select top 10 correlated variables

# ---- Function to Compare Models ----
library(caTools)  # Ensure caTools is loaded for sample.split
library(splines)  # Ensure splines is loaded for ns()

split_data <- function(data, predictor, response, split_ratio=0.7) {
  set.seed(123)
  sample_split <- caTools::sample.split(data[[predictor]], SplitRatio = split_ratio)
  train <- subset(data, sample_split == TRUE)
  test <- subset(data, sample_split == FALSE)
  return(list(train=train, test=test))
}

compare_models <- function(data, predictor, response) {
  unique_values <- length(unique(data[[predictor]]))
  is_binary <- all(data[[predictor]] %in% c(0,1))
  
  # Split data into train and test sets
  data_split <- split_data(data, predictor, response)
  train <- data_split$train
  test <- data_split$test
  
  cat("\n============================\n")
  cat(paste("Analyzing:", predictor, "vs", response, "\n"))
  cat("============================\n")
  
  # Fit models
  models <- list("Linear" = lm(as.formula(paste(response, "~", predictor)), data=train))
  
  if (!is_binary) {
    if (unique_values >= 3) models[["Quadratic"]] <- lm(as.formula(paste(response, "~ poly(", predictor, ", 2)")), data=train)
    if (unique_values >= 4) models[["Cubic"]] <- lm(as.formula(paste(response, "~ poly(", predictor, ", 3)")), data=train)
    if (unique_values >= 5) models[["Quartic"]] <- lm(as.formula(paste(response, "~ poly(", predictor, ", 4)")), data=train)
    
    min_mse <- Inf
    best_df <- 4
    for (df in 4:10) {
      if (unique_values >= df + 1) {
        model <- lm(as.formula(paste(response, "~ ns(", predictor, ", df=", df, ")")), data=train)
        test_predictions <- predict(model, test)
        test_mse <- mean((test[[response]] - test_predictions)^2)
        if (test_mse < min_mse) {
          min_mse <- test_mse
          best_df <- df
        } else {
          break  # Stop increasing df if MSE starts increasing
        }
      }
    }
    models[[paste0("Spline df=", best_df)]] <- lm(as.formula(paste(response, "~ ns(", predictor, ", df=", best_df, ")")), data=train)
  }
  
  # Evaluate models on test data
  results <- lapply(names(models), function(model_name) {
    model <- models[[model_name]]
    predictions <- predict(model, test)
    mse <- mean((test[[response]] - predictions)^2)
    r2 <- summary(model)$r.squared
    adj_r2 <- summary(model)$adj.r.squared
    return(data.frame(Model=model_name, R2=r2, Adj_R2=adj_r2, MSE=mse))
  })
  
  results <- do.call(rbind, results)
  
  # Identify the best model
  best_model <- results[which.min(results$MSE), ]
  print(results)
  cat("\n **Best Model for", predictor, ":**", best_model$Model, "\n")
  
  # Check for non-linearity
  if (!is_binary && best_model$Model != "Linear") {
    cat(" **Non-linearity detected**! Use", best_model$Model, "for", predictor, "\n")
  } else {
    cat("No strong non-linearity detected. Linear regression is sufficient.\n")
  }
  
  # Plot diagnostics
  scatter_plot <- ggplot(data, aes_string(x=predictor, y=response)) + geom_point(alpha=0.5) +
    ggtitle(paste("Scatter plot of", predictor, "vs", response))
  
  if (!is_binary) {
    scatter_plot <- scatter_plot + geom_smooth(method="lm", formula=y~ns(x, df=6), color="blue", se=FALSE)
  }
  
  residual_plot <- ggplot(data.frame(Fitted=fitted(models[[best_model$Model]]), Residuals=residuals(models[[best_model$Model]])), 
                          aes(x=Fitted, y=Residuals)) +
    geom_point(alpha=0.5) +
    geom_hline(yintercept=0, linetype="dashed", color="red") +
    ggtitle(paste("Residual Plot for", predictor))
  
  grid.arrange(scatter_plot, residual_plot, ncol=2)
}

# Run model comparisons for the top 10 correlated numerical variables
for (var in top10_vars) {
  compare_models(imdb_data, var, "imdb_score")
}



################################################
################ Step 5 ########################
################################################

#Building Final Model

# Load required packages
library(splines)
library(glue)

# Build the final regression model
final_model <- lm(imdb_score ~ poly(duration, 4) + 
                    ns(nb_news_articles, df=4) + ns(movie_meter_IMDBpro, df=5) + 
                    ns(release_year, df=5) + drama + horror + action + 
                    colour_film_dummy, data = imdb_data)

# View model summary
summary(final_model)

library(boot)
## Odessa Prediction

odessa_data <- data.frame(
  duration = 106,
  nb_news_articles = 24,
  movie_meter_IMDBpro = 14373,
  release_year = 2025,
  drama = 1,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

odessa_pred <- predict(final_model, newdata = odessa_data)

print(odessa_pred)
glue("Odessa's Projected Score is {round(odessa_pred,4)}")


## Black Bag Prediction

blackbag_data <- data.frame(
  duration = 93,
  nb_news_articles = 104,
  movie_meter_IMDBpro = 1172,
  release_year = 2025,
  drama = 1,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

blackbag_pred <- predict(final_model, newdata = blackbag_data)

print(blackbag_pred)
glue("Black Bag's Projected Score is {round(blackbag_pred,4)}")


## High Rollers Prediction

highrollers_data <- data.frame(
  duration = 101,
  nb_news_articles = 6,
  movie_meter_IMDBpro = 8696,
  release_year = 2025,
  drama = 0,
  horror = 0,
  action = 1,
  colour_film_dummy = 1
)

highrollers_pred <- predict(final_model, newdata = highrollers_data)

print(highrollers_pred)
glue("High Roller's Projected Score is {round(highrollers_pred,4)}")


## Novocaine Prediction

novocaine_data <- data.frame(
  duration = 110,
  nb_news_articles = 57,
  movie_meter_IMDBpro = 487,
  release_year = 2025,
  drama = 0,
  horror = 0,
  action = 1,
  colour_film_dummy = 1
)

novocaine_pred <- predict(final_model, newdata = novocaine_data)

print(novocaine_pred)
glue("Novocaine's Projected Score is {round(novocaine_pred,4)}")


## The Day the Earth Blew Up Prediction

tdtebu_data <- data.frame(
  duration = 91,
  nb_news_articles = 71,
  movie_meter_IMDBpro = 2530,
  release_year = 2025,
  drama = 0,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

tdtebu_pred <- predict(final_model, newdata = tdtebu_data)

print(tdtebu_pred)
glue("the Day the Earth Blew Up's Projected Score is {round(tdtebu_pred,4)}")


## Ash Prediction

ash_data <- data.frame(
  duration = 95,
  nb_news_articles = 185,
  movie_meter_IMDBpro = 3950,
  release_year = 2025,
  drama = 0,
  horror = 1,
  action = 0,
  colour_film_dummy = 1
)

ash_pred <- predict(final_model, newdata = ash_data)

print(ash_pred)
glue("Ash's Projected Score is {round(ash_pred,4)}")


## Locked Prediction

locked_data <- data.frame(
  duration = 95,
  nb_news_articles = 29,
  movie_meter_IMDBpro = 1898,
  release_year = 2025,
  drama = 0,
  horror = 1,
  action = 0,
  colour_film_dummy = 1
)

locked_pred <- predict(final_model, newdata = locked_data)

print(locked_pred)
glue("Locked's Projected Score is {round(locked_pred,4)}")


## Snow White Prediction

snowwhite_data <- data.frame(
  duration = 125,
  nb_news_articles = 800,
  movie_meter_IMDBpro = 278,
  release_year = 2025,
  drama = 0,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

snowwhite_pred <- predict(final_model, newdata = snowwhite_data)

print(snowwhite_pred)
glue("Snow White's Projected Score is {round(snowwhite_pred,4)}")


## The Alto Knights Prediction

ak_data <- data.frame(
  duration = 123,
  nb_news_articles = 82,
  movie_meter_IMDBpro = 1434,
  release_year = 2025,
  drama = 1,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

ak_pred <- predict(final_model, newdata = ak_data)

print(ak_pred)
glue("The Alto Knights's Projected Score is {round(ak_pred,4)}")


## A Working Man Prediction

awm_data <- data.frame(
  duration = 116,
  nb_news_articles = 109,
  movie_meter_IMDBpro = 1041,
  release_year = 2025,
  drama = 0,
  horror = 0,
  action = 1,
  colour_film_dummy = 1
)

awm_pred <- predict(final_model, newdata = awm_data)

print(awm_pred)
glue("A Working Man's Projected Score is {round(awm_pred,4)}")


## My Love Will Make You Disappear Prediction

mlwmyd_data <- data.frame(
  duration = 120,
  nb_news_articles = 0,
  movie_meter_IMDBpro = 20148,
  release_year = 2025,
  drama = 0,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

mlwmyd_pred <- predict(final_model, newdata = mlwmyd_data)

print(mlwmyd_pred)
glue("My Love will Make you Disappear's Projected Score is {round(mlwmyd_pred,4)}")


## The Woman in the Yard Prediction

twity_data <- data.frame(
  duration = 85,
  nb_news_articles = 55,
  movie_meter_IMDBpro = 3387,
  release_year = 2025,
  drama = 1,
  horror = 0,
  action = 0,
  colour_film_dummy = 1
)

twity_pred <- predict(final_model, newdata = twity_data)

print(twity_pred)
glue("The Woman in the Yard's Projected Score is {round(twity_pred,4)}")

## Final Ratings at Once

glue("Odessa's Projected Score is {round(odessa_pred,4)}")
glue("Black Bag's Projected Score is {round(blackbag_pred,4)}")
glue("High Roller's Projected Score is {round(highrollers_pred,4)}")
glue("Novocaine's Projected Score is {round(novocaine_pred,4)}")
glue("the Day the Earth Blew Up's Projected Score is {round(tdtebu_pred,4)}")
glue("Ash's Projected Score is {round(ash_pred,4)}")
glue("Locked's Projected Score is {round(locked_pred,4)}")
glue("Snow White's Projected Score is {round(snowwhite_pred,4)}")
glue("The Alto Knights's Projected Score is {round(ak_pred,4)}")
glue("A Working Man's Projected Score is {round(awm_pred,4)}")
glue("My Love will Make you Disappear's Projected Score is {round(mlwmyd_pred,4)}")
glue("The Woman in the Yard's Projected Score is {round(twity_pred,4)}")


################################################
################ Step 6 ########################
################################################

####LOOCV Test
library(boot)

fit=glm(imdb_score ~ drama + horror + action + colour_film_dummy + poly(duration, 4) + 
          ns(nb_news_articles, df = 4) + ns(movie_meter_IMDBpro, df = 5) + ns(release_year, df = 5), 
        data = imdb_data) 
mse=cv.glm(imdb_data, fit)$delta[1]
mse


###K-Fold Test
fit_k=glm(imdb_score~poly(duration, 4)+drama+ns(nb_news_articles, df=4)+ns(movie_meter_IMDBpro, df=5)+
            ns(release_year, df=5)+horror+action+colour_film_dummy, data=imdb_data)
mse_k = cv.glm(imdb_data, fit, K=10)$delta[1]
mse_k













