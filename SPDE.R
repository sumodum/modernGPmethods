library(sp)
library(sf)
library(INLA)
library(microbenchmark)

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
x_train <- x[train,]
x_test <- x[-train,]
y_train <- y[train]
y_test <- y[-train]

# Point dataset
x_train <- as.data.frame(x_train)
colnames(x_train) <- c("x", "y")
coordinates(x_train) <- ~x + y

# Create the prediction grid
x_grid <- expand.grid(x = seq(min(x[,1]), max(x[,1]), length.out = 100),
                      y = seq(min(x[,2]), max(x[,2]), length.out = 100))
coordinates(x_grid) <- ~x + y
gridded(x_grid) <- TRUE

# Set boundary & define mesh
x_train.bdy <- st_union(as(x_grid, "sf"))
mesh <- inla.mesh.2d(loc.domain = x_train.bdy, max.edge = c(1.5, 5), offset = c(1, 2.5))

# Different values of range & alpha to consider
range_values <- c(0.1, 0.5, 1, 1.5, 2)
alpha_values <- c(0.1, 0.5, 1, 1.5, 2)

results_list <- list()
i = 1

# Iterate over different combinations of range and alpha
for (range in range_values) {
  for (alpha in alpha_values) {
    cat("Running model with alpha =", alpha, "and range parameter =", range, "\n")
    
    # Create SPDE with a set of range and alpha values
    x_train.spde <- inla.spde2.matern(mesh = mesh, alpha = alpha, range = range)
    
    # Create data structure for training
    x_train.stack <- inla.stack(data = list(ozone = y_train),
                                A = list(inla.spde.make.A(mesh = mesh, loc = coordinates(x_train))),
                                effects = list(c(inla.spde.make.index(name = "spatial.field", n.spde = x_train.spde$n.spde), list(Intercept = 1))),
                                tag = "x_train.data")
    
    # Create data structure for prediction
    A.pred <- inla.spde.make.A(mesh = mesh, loc = rbind(coordinates(x_test), coordinates(x_grid)))
    x.pred <- inla.stack(data = list(ozone = NA),
                         A = list(A.pred),
                         effects = list(c(inla.spde.make.index(name = "spatial.field", n.spde = x_train.spde$n.spde), list(Intercept = 1))),
                         tag = "x.pred")
    
    # Join the stacks
    join.stack <- inla.stack(x_train.stack, x.pred)
    
    # Fit model
    form <- ozone ~ -1 + Intercept + f(spatial.field, model = spde)
    start_time <- Sys.time()
    m <- inla(form, data = inla.stack.data(join.stack, spde = x_train.spde),
              family = "gaussian",
              control.predictor = list(A = inla.stack.A(join.stack), compute = TRUE),
              control.compute = list(cpo = TRUE, dic = TRUE))
    end_time <- Sys.time()
    
    # Get fitted values & compute evaluation metrics for train set
    index.train <- inla.stack.index(join.stack, "x_train.data")$data
    train_predictions <- m$summary.fitted.values[index.train, "mean"]
    
    # Performance metrics for fitting process
    train_RMSE <- sqrt(mean((train_predictions - y_train)^2))
    train_MAE <- mean(abs(train_predictions - y_train))
    train_SS_res <- sum((y_train - train_predictions)^2)
    train_SS_tot <- sum((y_train - mean(y_train))^2)
    train_R_squared <- 1 - (train_SS_res / train_SS_tot)
    
    start_time_test <- Sys.time()
    # Get predictions for test set
    index.pred <- inla.stack.index(join.stack, "x.pred")$data
    test_predictions <- m$summary.fitted.values[index.pred, "mean"][1:nrow(x_test)]
    end_time_test <- Sys.time()
    
    # Compute evaluation metrics for test set
    test_RMSE <- sqrt(mean((test_predictions - y_test)^2))
    test_MAE <- mean(abs(test_predictions - y_test))
    SS_res <- sum((y_test - test_predictions)^2)
    SS_tot <- sum((y_test - mean(y_test))^2)
    test_R_squared <- 1 - (SS_res / SS_tot)
    
    # Calculate runtime values
    runtime_train <- end_time - start_time
    runtime_test <- end_time_test - start_time_test
    
    # Store results into results_list
    results_list[[length(results_list) + 1]] <- c(Range = range,
                                                  Alpha = alpha,
                                                  RMSE_train = train_RMSE, 
                                                  MAE_train = train_MAE,
                                                  R_squared_train = train_R_squared,
                                                  RMSE_test = test_RMSE, 
                                                  MAE_test = test_MAE,
                                                  R_squared_test = test_R_squared,
                                                  Runtime_train = runtime_train,
                                                  Runtime_test = runtime_test)
    
    # Plot of prediction against true values for test set
    pdf(sprintf("SPDE_model%d_plot.pdf", i))
    plot(test_predictions, y_test)
    abline(a = 0, b = 1, col = "red")
    dev.off()
    
    i <- i+1
  }
}

# Combine results into results_df 
results_df <- do.call(rbind, results_list)

# Export results as a csv file
write.csv(results_df, "SPDE_results.csv", row.names = FALSE)
