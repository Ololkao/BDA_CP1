# print("Hello World", quote = FALSE)
# 
# add <- function(x, y){
#     x + y
# }
# 
# print(add(1.0e10, 2.0e10))
# print(paste("one", NULL))
# print(paste(NA, "two"))
# 
# h <- c(1,2,3,4,5,6)
# M <- c("A", "B", "C", "D", "E", "F")
# barplot(h, names.arg = M)

library(Amelia)
library(mice)
library(missForest)
library(missMDA)
library(MASS)
library(softImpute)
library(dplyr)
library(tidyr)
library(ggplot2)
library(devtools)
source_url('https://raw.githubusercontent.com/R-miss-tastic/website/master/static/how-to/generate/amputation.R')

#### Simulation of the data matrix ####
set.seed(123)
n <- 1000
p <- 10
mu.X <- rep(1, 10)
Sigma.X <- diag(0.5, ncol = 10, nrow = 10) + matrix(0.5, nrow = 10, ncol = 10)
X <- mvrnorm(n, mu.X, Sigma.X)
head(X)
XproduceNA <- produce_NA(X, mechanism = "MCAR", perc.missing = 0.3)
XNA <- as.matrix(as.data.frame(XproduceNA$data.incomp))
head(XNA)

data = read.csv("data1/train.csv", header = FALSE)
head(data)

# perform softImpute
sft <- softImpute(x = XNA, rank.max = 2, lambda = 0, type = c("als", "svd"))
# compute the factorization
X.sft <- sft$u %*% diag(sft$d) %*% t(sft$v)
# replace missing values by computed values
X.sft[which(!is.na(XNA))] <- XNA[which(!is.na(XNA))]

source('https://raw.githubusercontent.com/R-miss-tastic/website/master/static/how-to/impute/CrossValidation_softImpute.R')
lambda_sft <- cv_sft(XNA)
sft <- softImpute(x = XNA, lambda = lambda_sft)
X.sft <- sft$u %*% diag(sft$d) %*% t(sft$v)
X.sft[which(!is.na(XNA))] <- XNA[which(!is.na(XNA))]
head(X.sft)

forest <- missForest(xmis = XNA, maxiter = 20, ntree = 100)
X.forest <- forest$ximp
head(X.forest)
