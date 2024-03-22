library(sp) 
library(LatticeKrig)
library(spam64) # To handle large sparse matrices

data(ozone2)

x <- ozone2$lon.lat
y <- ozone2$y[16,] # Use one time-point of spatial data only
# Keep only the locations that are not 'NA'
good <- !is.na(y)
x <- x[good,]
y <- y[good]

# Split the data set into train and test sets
set.seed(1)
train <- sample(1:nrow(x), round(0.7*nrow(x)))
x_train <- x[train, ]
x_test <- x[-train, ]
y_train <- y[train]
y_test <- y[-train]

# Different values of a.wght, nlevel, nu & NC to consider
a.wght_values <- c(4.1, 6, 8, 10)
nlevel_values <- c(2, 4)
nu_values <- c(0.1, 0.3, 0.5)
NC_values <- c(10, 20, 30)
lambda_values <- 0.1

results_df <- data.frame(
  Model = numeric(),
  a.wght = numeric(),
  nlevel = numeric(),
  nu = numeric(),
  NC = numeric(),
  MAE_train = numeric(),
  RMSE_train = numeric(),
  Rsquared_train = numeric(),
  MAE_test = numeric(),
  RMSE_test = numeric(),
  Rsquared_test = numeric(),
  Runtime_train = numeric(),
  Runtime_test = numeric(),
  stringsAsFactors = FALSE
)

# Iterate over different combinations of params
for (a.wght in a.wght_values) {
  for (nlevel in nlevel_values) {
    for (nu in nu_values) {
      for (NC in NC_values) {
        
        # Train the model with the a set of params
        start_time <- Sys.time()
        obj <- LatticeKrig(x_train, y_train, a.wght = a.wght, 
                           nlevel = nlevel, nu = nu, 
                           NC = NC, lambda = lambda_values)
        end_time <- Sys.time()
        
        # Get predictions and residuals for train set
        train_predictions <- predict(obj)
        train_residuals <- train_predictions - y_train
        
        # Compute evaluation metrics for train set
        RMSE_train <- sqrt(mean(train_residuals^2))
        MAE_train <- mean(abs(train_residuals))
        Rsquared_train <- cor(y_train, train_predictions)^2
        
        # Get predictions for test set
        start_time_test <- Sys.time()
        test_predictions <- predict(obj, xnew = x_test)
        end_time_test <- Sys.time()
        test_residuals <- test_predictions - y_test
        
        # Compute evaluation metrics for test set
        MAE_test <- mean(abs(test_residuals))
        RMSE_test <- sqrt(mean(test_residuals^2))
        Rsquared_test <- cor(y_test, test_predictions)^2
        
        # Calculate runtime values
        runtime_train <- end_time - start_time
        runtime_test <- end_time_test - start_time_test
        
        # Store results into results_df data frame
        results_df <- rbind(results_df, data.frame(
          Model = nrow(results_df) + 1,
          a.wght = a.wght,
          nlevel = nlevel,
          nu = nu,
          NC = NC,
          RMSE_train = RMSE_train,
          MAE_train = MAE_train,
          Rsquared_train = Rsquared_train,
          RMSE_test = RMSE_test,
          MAE_test = MAE_test,
          Rsquared_test = Rsquared_test,
          Runtime_train = runtime_train,
          Runtime_test = runtime_test
        ))
        
        # Plot of prediction against true values for test set
        pdf(sprintf("LK_model%d_plot.pdf", nrow(results_df)))
        plot(test_predictions, y_test)
        abline(a = 0, b = 1, col = "red")
        dev.off()
        
      }
    }
  }
}

# Export results as a csv file
write.csv(results_df, file = "LK_results.csv", row.names = FALSE)

