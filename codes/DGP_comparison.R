library(nnet)
library(downscalr)


##### setting up DGP of skewed MNL to test model convergence and picking up signals

# Set seed for reproducibility
set.seed(10)
# Number of observations
N <- 5000
# Number of classes
K <- 4
# Number of explanatory v<ariables
P <- 6


#generate a skewing first class
X1 <- matrix(rnorm(N), ncol = 1) + 4


# Generate explanatory variables
X <- matrix(rnorm(N * (P-1)), ncol = P-1)
X <- cbind(X1,X)
colnames(X) <- paste0(rep("X",P), seq(1,P,1))

# Define coefficients for each class (including the intercept)
beta <- matrix(sample(-2:2, (P + 1) * K, replace = TRUE), nrow = K, byrow = TRUE)
# respective first class skewing
beta[,2] <- matrix(sample(-5:-1, K, replace = TRUE), nrow = K, byrow = TRUE)
# zeros for baseline
beta[1,] <- rep(0,P+1)

# Add intercept to explanatory variables
X_with_intercept <- cbind(1, X)
# Calculate the linear combination of X and beta (logits)
logits <- X_with_intercept %*% t(beta)

# Softmax function to convert log-odds to probabilities
softmax <- function(logits) {
  exp_logits <- exp(logits)
  exp_logits / rowSums(exp_logits)
}
# Convert logits to probabilities
probs <- softmax(logits)
#check skewedness of the probs
colSums(probs)


# Generate class labels based on the probabilities
y <- apply(probs, 1, function(p) sample(1:K, 1, prob = p))
Y <- matrix(0, ncol = ncol(probs), nrow = nrow(probs))
for(i in 1:length(y)){Y[i,y[i]] <- 1}

#check skewedness of the class responses
summary(as.factor(y))

# Combine explanatory variables and class labels into a data frame
df <- data.frame(X, Class = factor(y))


# Fit the multinomial logit model with neural network
formul <- paste0("Class ~", paste0(rep("X",P), seq(1,P,1), collapse = "+"))
t1 <- Sys.time()
model <- nnet::multinom(formul, data = df)
NN <- Sys.time()-t1


# Fit the multinomial logit model with downscalr mnl
t2 <- Sys.time()
model2 <- mnlogit(X_with_intercept,probs,baseline = 1,niter = 1000,nburn = 500, calc_marginal_fx = FALSE )
MNL <- Sys.time()-t2

# Display the model summary
cat("True coef: \n")
beta
cat("NN coef: \n")
model
cat("MNL coef: \n")
t(apply(model2$postb, c(1,2), mean))
cat("NN time: \n")
NN
cat("MNL time: \n")
MNL


X2 <- scale(X)
df2 <- data.frame(X2, Class = factor(y))

t3 <- Sys.time()
model <- multinom(formul, data = df2)
NN <- Sys.time()-t3

model



X2_with_intercept <- cbind(1, X2)


t4 <- Sys.time()
model2 <- mnlogit(X2_with_intercept,probs,baseline = 1,niter = 1000,nburn = 500, calc_marginal_fx = FALSE )
MNL <- Sys.time()-t4










# Predict the probabilities for the training data
predicted_probs <- predict(model, type = "probs")

# Predict the class labels
predicted_classes <- predict(model, type = "class")

# Confusion matrix to evaluate the accuracy
confusion_matrix <- table(predicted_classes, df$Class)
print(confusion_matrix)

# Overall accuracy
accuracy <- sum(predicted_classes == df$Class) / N
print(accuracy)