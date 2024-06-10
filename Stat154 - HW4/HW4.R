##########################
# Problem 4: PCA on MNIST
##########################
# loading library
library(factoextra)
library(ggfortify)
library(reshape2)
library(SDEFSR) # read keel data
library(aricode) # adjusted rand index, normalized mutual information

# import data
train <- read.csv("mnist_train.csv")
test <-read.csv("mnist_test.csv")

X_train <- train[, -1]
idx <- which(colSums(X_train)==0)
X_reduced <- X_train[,-idx] # dropping columns with only zeros inside
label <- train[, 1]
label <- as.factor(label)

################
# 1. Perform PCA
#################
pca <- prcomp(X_reduced, center = TRUE, scale = TRUE)
pca_summary <- summary(pca)

########################
# 2. Plot the variances
########################
variance_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(variance_explained, type = "l", xlab = "Principal Components", 
     ylab = "Cumulative Percentage of Variance Explained")

fviz_eig(pca, addlabels = TRUE)

########################
# 3. Visualize the data in two dimensions color-coding
########################
library(ggfortify)
scores <- as.data.frame(pca$x[,1:2])
scores <- cbind(scores, label)
scores$label <- as.factor(scores$label)
ggplot(scores, aes(x = PC1, y = PC2, color = label)) + 
  geom_point() + 
  labs(title = "MNIST PCA plot", x = "PC1", y = "PC2")

# Explain any patterns your see.
# The data points are separated into different sections, it means that the principal components are capturing 
# meaningful differences between the classes. However, the separate is not distinct between in class,
# the overlapping means that using only two principal components will only give a good over picture of class separation, 
# but there will certainly be ambiguous cases between the classes. Overall, we think the first two PC did well,
# we are keeping in mind that PCA does not have the class information.


################################################
# 4. Plot the first five PC as gray-scale image
#     and comment
################################################
# calculate total variance explained by each principal component 
df_pca <- data.frame(t(pca_summary$importance))

# with reconstruction
n_comp = 5
recon <- pca$x[, 1:n_comp] %*% t(pca$rotation[, 1:n_comp])
recon <- scale(recon, center = FALSE, scale = 1/pca$scale)
recon <- scale(recon, center = -1 * pca$center, scale = 1/pca$scale)
# recon_df <- data.frame(cbind(1:nrow(recon), recon))
# colnames(recon_df) <- c("x", 1:(ncol(recon_df)-1))
recon <- as.data.frame(recon)
zero_cols <- setdiff(names(X_train), names(recon))
# as logical vector: True or False
# zero_cols <- names(X_train) %in% names(recon)
recon[, c(zero_cols)] <- 0
# reorder the columns to the same as original data
# before displaying image
recon <- recon[, colnames(X_train)]
# check if we did it correctly
all(names(recon) == names(X_train)) # yes

# plot the reconstruction
r_samp = 1
image(matrix(as.matrix(recon[r_samp,]), nrow = 28, ncol = 28)[, 28:1], 
      col = gray((0:255)/255), main = paste0("Using first 5 PC on digit ", label[r_samp]))

# Comment
# The first 5 PCs seem to capture basic edges/outlines of digits 
# It captures width of digits, and slant/tilt, we can see
# some intricate patterns within digits along with some finer details


# 5. How many principal components are required to explain 70, 80, 90, and 95% of the variance?
variance_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
required_var <- c(0.7, 0.8, 0.9)
for (i in required_var){
  num_components <- which(variance_explained >= i)[1]
  print(paste0("For ", i*100, "% it requires ", + num_components, " PCs."))
}
# for component
# num_components <- which(variance_explained >= 0.9)[1]


# 6. Plot the data in two dimensions after applying
# kernel PCA and sparse PCA with your choice of kernel 
# and corresponding hyperparameter.
library(kernlab)
library(sparsepca)
# terminology for self-reference
# scores: The scores in PCA refer to the transformed data points that have been 
# projected onto the principal components (PCs). These transformed data points 
# represent the new coordinate system that is defined by the PCs. The scores are
# sometimes referred to as the "principal component scores."
# loading: The loading object contains the loadings (or weights) for each variable 
# (or feature) in the original data set, with respect to each principal component. 
# The loadings represent the contribution of each variable to the corresponding principal 
# component. Each row in the loading matrix corresponds to one variable in the original data set, 
# and each column corresponds to one principal component.
# x: The x object is the centered and scaled data matrix that is used as input to the PCA. 
# This matrix contains the observations (rows) and variables (columns) of the original data set.

# library(usethis)
# usethis::edit_r_environ()
# R_MAX_VSIZE=5Gb

# kernel PCA
X_small <- X_reduced[1:3000, ]
kpc <- kpca(~.,data=X_small, kernel="rbfdot", kpar=list(sigma=0.2))
plot(rotated(kpc), col=as.factor(label[1:2000]),
     xlab="1st Principal Component",ylab="2nd Principal Component")

## sparse PCA
pca_sparse <- spca(X_reduced, center = TRUE, scale = TRUE)
scores <- pca_sparse$scores[, 1:2]
scores <- cbind(scores, label)
scores <- as.data.frame(scores)
names(scores) <- c("PC1", "PC2", "label")
scores$label <- as.factor(scores$label)
ggplot(scores, aes(x = PC1, y = PC2, color = label)) + 
  geom_point() + 
  labs(title = "MNIST Sparse PCA plot", x = "PC1", y = "PC2")


################################################
## Problem 5: Classification on MNIST using SVMs
################################################
rm(list = ls())
train <-read.csv("mnist_train.csv")
test <-read.csv("mnist_test.csv")
dim_train <- dim(train) # 60000 by 785
dim_test <- dim(test) # 10000 by 785
mnist <- rbind(train, test) # combine the data

### Part 1: perform PCA on entire data, reduce the dimensions to 90% of total variance
X <- mnist[, -1]
label <- mnist[, 1]
label <- as.factor(label)

# Some columns only have all zeros, we need to drop them
# otherwise, standardization produces NaN values in the column
X_reduced <- X[,-(which(colSums(X)==0))] # dropping columns with only zeros inside
pca <- prcomp(X_reduced, center = TRUE, scale = TRUE)

# determine the number of components to explain 90% of the total variance
variance_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
num_components <- which(variance_explained >= 0.9)[1]

# reduce dimensions to 90% of the total variance
X_reconstruct <- predict(pca, newdata = X_reduced)[, 1:num_components]

# X_reduced: does not include the y label, it is used in dimensionality reduction in PCA
# mnist_reduced: a data frame includes x feature, which are the PC's, and y label
mnist_reduced <- cbind(label, X_reconstruct) # add back y label to the dataset
mnist_reduced <- as.data.frame(mnist_reduced) # Convert matrix to data.frame: factor only exists in data frame not matrix
mnist_reduced$label <- as.factor(mnist_reduced$label) # change it to factor so the algorithm won't perform regression

### Part 2
# load the necessary libraries
library(e1071)
library(caret)

# set up the cross-validation scheme
# cv <- trainControl(method = "repeatedcv", 
#                    number = 10,
#                    repeats = 1
#                    # repeats = 3,
#                    verboseIter = TRUE)

# define the search grid for tuning the SVM parameters
# grid <- expand.grid(sigma = seq(0.01, 2, length = 20),
#                     cost = 2^(seq(-5, 10, length = 16)))

X_train_recon <- X_reconstruct[1:nrow(train), ]
label_train_recon <- label[1:nrow(train)]
X_test_recon <- X_reconstruct[(nrow(train)+1):nrow(mnist), ]
label_test_recon <- label[(nrow(train)+1):nrow(mnist)]
svm_radial <- train(y = label_train_recon,
                    x = X_train_recon,
                    method = "svmRadial",
                    trControl = cv,
                    # preProc = c("center", "scale"),
                    metric = "Accuracy")

# print the best parameters and corresponding accuracy
svm_radial$bestTune 
# 3 0.004689845 1

# svm_radial$results[svm_radial$bestTune[1], ]
svm_radial$results

# sigma    C  Accuracy     Kappa  AccuracySD     KappaSD
# 1 0.004689845 0.25 0.9325997 0.9250865 0.003643884 0.004049874
# 2 0.004689845 0.50 0.9463330 0.9403502 0.003540210 0.003934797
# 3 0.004689845 1.00 0.9567165 0.9518912 0.002794284 0.003105768

# use the best model to make predictions on the test set
predictions <- predict(svm_radial, newdata = test)

### another try using without using caret
num_components <- which(variance_explained >= 0.9)[1]
reduced_train_x <- pca$x[, 1:num_components]

# Train SVM with radial basis function kernel
tuned_parameters <- tune(svm, train.x = reduced_train_x, train.y = label,
                         kernel = "radial", ranges = list(cost = 2^(-1:1), gamma = 2^(-1:1)))
svm_model <- tuned_parameters$best.model

# Evaluate SVM on test set
reduced_test_x <- predict(pca, newdata = test_x)[, 1:num_components]
predicted_labels <- predict(svm_model, newdata = reduced_test_x)
accuracy <- mean(predicted_labels == test_y)
cat("Test accuracy:", accuracy)

### 3. Train SVM with polynomial kernel
train_control <- trainControl(method="cv", number=10)
svm_poly <- train(y = label_train_recon,
                  x = X_train_recon,
                  method = "svmPoly",
                  trControl = train_control,
                  # preProc = c("center", "scale"),
                  metric = "Accuracy")


### another example
library(e1071)
library(kernlab)

# Load MNIST data
rm(list = ls())
train <-read.csv("mnist_train.csv")
test <-read.csv("mnist_test.csv")
mnist <- rbind(train, test) # combine the data

# Split data into training and test sets
set.seed(123)
mnist_reduced <- mnist[,-(which(colSums(mnist)==0))] # dropping columns with only zeros inside

##########
### Part 1: Perform a PCA on the entire data (by first combining the training and test data) and reduce the dimensions of the data that explain 90% of the total variance. Remember to center and scale the data before performing a PCA.
pca <- prcomp(mnist_reduced[,-1], center = TRUE, scale. = TRUE)
variance_explained <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
num_components <- which(variance_explained >= 0.9)[1] # 238
train_pca <- as.data.frame(pca$x[1:60000, 1:num_components])
train_pca$label <- as.factor(train$label)

# Perform PCA on test data using training data PCA object
# test_pca <- as.data.frame(predict(pca, newdata = test[,-1])[,1:num_components])
# test_pca$label <- test$label
test_pca <- as.data.frame(pca$x[60001:70000, 1:num_components])
test_pca$label <- as.factor(test$label)



# Define cross-validation function
cv_function <- function(cost) {
  folds <- cut(seq(1,nrow(train_pca)),breaks=10,labels=FALSE)
  cv_results <- lapply(1:10, function(i) {
    cat('fold:', i)
    test_fold_index <- which(folds==i,arr.ind=TRUE)
    train_fold_index <- setdiff(1:nrow(train_pca), test_fold_index)
    train_fold <- train_pca[train_fold_index,]
    test_fold <- train_pca[test_fold_index,]
    cat('train SVM', i)
    svm_model <- svm(label ~ ., data = train_fold, kernel = 'polynomial', cost = cost, degree = 3)
    cat('Done', i)
    predictions <- predict(svm_model, test_fold[,-ncol(test_fold)])
    accuracy <- mean(predictions == test_fold$label)
    return(accuracy)
  })
  mean(unlist(cv_results))
}

# Tune cost parameter using cross-validation
cost_values <- seq(0.1, 10, by = 0.1)
cv_results <- sapply(cost_values, cv_function)
best_cost <- cost_values[which.max(cv_results)]

# Train SVM using best cost value
svm_model <- svm(label ~ ., data = train_pca, kernel = 'polynomial', cost = best_cost, degree = 3)

# Make predictions on test data
predictions <- predict(svm_model, test_pca[,-ncol(test_pca)])

# Evaluate model performance
confusionMatrix(predictions, test_pca$label)
