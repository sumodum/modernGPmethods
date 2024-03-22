library("FRK")
library("sp")
library("dplyr")

data(ozone2)

x <- ozone2$lon.lat
y <- ozone2$y[16,] # Use one time-point of spatial data only
# Keep only the locations that are not 'NA'
good <- !is.na(y)
x <- x[good,]
y <- y[good]

# Split the data set into train and test sets
set.seed(1)
train <- sample(1:nrow(x), round(0.7 * nrow(x)))
x_train <- x[train,]
x_test <- x[-train,]
y_train <- y[train]
y_test <- y[-train]

coord_train <- SpatialPointsDataFrame(coords = x_train, data = data.frame(y = y_train))
coord_test <- SpatialPointsDataFrame(coords = x_test, data = data.frame(y = y_test))

# Function to run the FRK model and extract results
run_FRK_model <- function(train_data, test_data, BAUs_cellsize, basis_nres, n_EM, tol) {
  
  # Construct BAUs and basis functions for train set
  BAUs_train <- auto_BAUs(manifold = plane(), data = train_data, nonconvex_hull = FALSE,
                          cellsize = BAUs_cellsize, type = "grid")
  
  # Add 'fs' field to BAUs_train
  BAUs_train$fs <- 1
  
  # Set up basis functions
  basis_train <- auto_basis(manifold = plane(), data = train_data, nres = basis_nres)
  
  start_time <- Sys.time()
  # Construct the SRE model using train set
  S_train <- SRE(f = y ~ 1, data = list(train_data), basis = basis_train, BAUs = BAUs_train)
  # Fit the model
  S_train <- SRE.fit(S_train, n_EM = n_EM, tol = tol, print_lik = TRUE)
  end_time <- Sys.time()
  
  start_time_test <- Sys.time()
  # Predict over BAUs for test set
  pred_test <- predict(S_train, newdata = test_data)
  end_time_test <- Sys.time()
  
  # Compute evaluation metrics for test set
  observed_test <- pred_test$y
  predicted_test <- pred_test$mu
  
  rmse_test <- sqrt(mean((observed_test - predicted_test)^2))
  mae_test <- mean(abs(observed_test - predicted_test))
  SS_res <- sum((observed_test - predicted_test)^2)
  SS_tot <- sum((observed_test - mean(observed_test))^2)
  test_R_squared <- 1 - (SS_res / SS_tot)
  
  # Calculate runtime values
  runtime_train <- end_time - start_time
  runtime_test <- end_time_test - start_time_test
  
  # Plot of prediction against true values for test set
  pdf(sprintf("FRK_model%d_plot.pdf", i))
  plot(observed_test, predicted_test)
  abline(a = 0, b = 1, col = "red")
  dev.off()
  
  return(list(rmse = rmse_test, mae = mae_test, R_squared = test_R_squared, runtime_train = runtime_train, runtime_test = runtime_test))
}

# Different values of BAUs_cellsize, basis_nres, and n_EM to consider
BAUs_cellsize <- c(0.01, 0.02, 0.03)
basis_nres <- c(1, 2, 3)
n_EM <- c(1, 2, 3)
tol <- 0.01

results_df <- data.frame(Model = numeric(), BAUs_cellsize = numeric(), basis_nres = numeric(), n_EM = numeric())
i <- 1

# Iterate over parameter combinations
for (cellsize in BAUs_cellsize) {
  for (nres in basis_nres) {
    for (em_iter in n_EM) {
      
      # Run FRK model
      results <- run_FRK_model(coord_train, coord_test, cellsize, nres, em_iter, tol)

      # Store results into results data frame
      results_df <- bind_rows(results_df, data.frame(Model = i, 
                                                     BAUs_cellsize = cellsize, 
                                                     basis_nres = nres, 
                                                     n_EM = em_iter,
                                                     RMSE_test = results$rmse, 
                                                     MAE_test = results$mae,
                                                     R_squared_test = results$R_squared,
                                                     Runtime_train = results$runtime_train,
                                                     Runtime_test = results$runtime_test))
      
      
      i <- i + 1
    }
  }
}

# Export results as a csv file
write.csv(results_df, "FRK_results.csv", row.names = FALSE)
