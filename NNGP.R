library(sp)
library(spNNGP)

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

# Parameters for NNGP
sigma.sq <- 5
tau.sq <- 1
phi <- 3/0.5

starting <- list("phi" = phi, "sigma.sq" = 5, "tau.sq" = 1)
tuning <- list("phi" = 0.5, "sigma.sq" = 0.5, "tau.sq" = 0.5)
priors <- list("phi.Unif" = c(3 / 1, 3 / 0.01), "sigma.sq.IG" = c(2, 5), "tau.sq.IG" = c(2, 1))
cov.model <- "exponential"
n.report <- 500

# Function to compute RMSE, MAE, and R-squared
compute_metrics <- function(observed, predicted) {
  rmse <- sqrt(mean((observed - predicted)^2))
  mae <- mean(abs(observed - predicted))
  rsquared <- 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)
  return(c(RMSE = rmse, MAE = mae, R_squared = rsquared))
}

# Different values of n.samples & n_neighbors to consider
n_samples_values <- c(100, 500, 1000)
n_neighbors_values <- c(5, 10, 15)

results <- data.frame()
k = 1

# Iterate over different combinations of n.samples and n.neighbors
for (i in seq_along(n_neighbors_values)) {
  for (j in seq_along(n_samples_values)) {
    n_neighbors <- n_neighbors_values[i]
    n_samples <- n_samples_values[j]
    
    start_time <- Sys.time()
    # Fit the Response NNGP model
    m.r <- spNNGP(y_train ~ x_train - 1, coords = x_train, starting = starting, method = "response", 
                  n.neighbors = n_neighbors, tuning = tuning, priors = priors, cov.model = cov.model,
                  n.samples = n_samples, n.omp.threads = 1, n.report = n.report)
    end_time <- Sys.time()
  
    # Predict response values & compute evaluation metrics for train set
    p_train <- predict(m.r, X.0 = x_train, coords.0 = x_train, n.omp.threads = 1)
    metrics_train <- compute_metrics(y_train, apply(p_train$p.y.0, 1, mean))
    
    start_time_test <- Sys.time()
    # Predict response values & compute evaluation metrics for test set
    p.r_test <- predict(m.r, X.0 = x_test, coords.0 = x_test, n.omp.threads = 1)
    end_time_test <- Sys.time()
    metrics_test <- compute_metrics(y_test, apply(p.r_test$p.y.0, 1, mean))
    
    # Calculate runtime values
    runtime_train <- end_time - start_time
    runtime_test <- end_time_test - start_time_test
    
    # Store results into results data frame
    results <- rbind(results, data.frame(Model = k,
                                         n_neighbors = n_neighbors, 
                                         n_samples = n_samples,
                                         RMSE_train = metrics_train["RMSE"], 
                                         MAE_train = metrics_train["MAE"],
                                         R_squared_train = metrics_train["R_squared"],
                                         RMSE_test = metrics_test["RMSE"], 
                                         MAE_test = metrics_test["MAE"],
                                         R_squared_test = metrics_test["R_squared"],
                                         Runtime_train = runtime_train,
                                         Runtime_test = runtime_test
                                         ))
    
    # Plot of prediction against true values for test set
    pdf(sprintf("NNGP_model%d_plot.pdf", k))
    plot(apply(p.r_test$p.y.0, 1, mean), y_test)
    abline(a = 0, b = 1, col = "red")
    dev.off()
    
    k <- k+1
  }
}

# Export results as a csv file
write.csv(results, file = "NNGP_results.csv", row.names = FALSE)
