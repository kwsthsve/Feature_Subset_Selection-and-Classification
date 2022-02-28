library(psych)
library(leaps)
library(glmnet)
library(MASS)
library(class)


mcensus <- read.csv("mcensus.csv", header=T, sep=" ")
head(mcensus)
str(mcensus)
summary(mcensus)
describe(mcensus)

par(mfrow=c(4,4))
for(i in 1:14){
 hist(mcensus[,i],main=colnames(mcensus[i]))
}

par(mfrow=c(4,4))
for(i in 1:14){
 qqnorm(mcensus[,i],main=colnames(mcensus[i]))
 qqline(mcensus[,i],col="red")
}

pairs(mcensus,upper.panel=NULL)
cor(mcensus)

attach(mcensus)


					#Subset selection#


set.seed(123)

regfit <- regsubsets(ncrim~., data=mcensus, nvmax=13)
reg.summary <- summary(regfit)
reg.summary
regfit.fwd <- regsubsets(ncrim~., data=mcensus, nvmax=13, method="forward")
reg.fwd.summary <- summary(regfit.fwd)
reg.fwd.summary
regfit.bwd <- regsubsets(ncrim~., data=mcensus, nvmax=13, method="backward")
reg.bwd.summary <- summary(regfit.bwd)
reg.bwd.summary

#R^2, RSS#

par(mfrow=c(2,1))
plot(reg.summary$rsq, xlab="Number of variables", ylab="R^2", type="l")
max.rsq <- which.max(reg.summary$rsq) 
points(max.rsq, reg.summary$rsq[max.rsq], col="red")

plot(reg.summary$rss, xlab="Number of variables", ylab="RSS", type="l")
min.rss <- which.min(reg.summary$rss) 
points(min.rss, reg.summary$rss[min.rss], col="red")

par(mfrow=c(2,1))
plot(reg.fwd.summary$rsq, xlab="Number of variables", ylab="R^2", type="l")
max.rsq <- which.max(reg.fwd.summary$rsq) 
points(max.rsq, reg.fwd.summary$rsq[max.rsq], col="red")

plot(reg.fwd.summary$rss, xlab="Number of variables", ylab="RSS", type="l")
min.rss <- which.min(reg.fwd.summary$rss) 
points(min.rss, reg.fwd.summary$rss[min.rss], col="red")

par(mfrow=c(2,1))
plot(reg.bwd.summary$rsq, xlab="Number of variables", ylab="R^2", type="l")
max.rsq <- which.max(reg.bwd.summary$rsq) 
points(max.rsq, reg.bwd.summary$rsq[max.rsq], col="red")

plot(reg.bwd.summary$rss, xlab="Number of variables", ylab="RSS", type="l")
min.rss <- which.min(reg.bwd.summary$rss) 
points(min.rss, reg.bwd.summary$rss[min.rss], col="red")

#adjusted R^2, BIC, Cp#

par(mfrow=c(3,1))
plot(reg.summary$cp, xlab="Number of variables", ylab="Cp", type="l")
min.cp <- which.min(reg.summary$cp)
points(min.cp, reg.summary$cp[min.cp], col="red")  ##8

plot(reg.summary$bic, xlab="Number of variables", ylab="BIC", type="l")
min.bic <- which.min(reg.summary$bic)  
points(min.bic, reg.summary$bic[min.bic], col="red") ##3

plot(reg.summary$adjr2, xlab="Number of variables", ylab="Adjusted R^2", type="l")
max.adjr2 <- which.max(reg.summary$adjr2) 
points(max.adjr2, reg.summary$adj[max.adjr2], col="red")  ##9

par(mfrow=c(3,1))
plot(reg.fwd.summary$cp, xlab="Number of variables", ylab="Cp", type="l")
min.cp <- which.min(reg.fwd.summary$cp) 
points(min.cp, reg.fwd.summary$cp[min.cp], col="red")  ##8

plot(reg.fwd.summary$bic, xlab="Number of variables", ylab="BIC", type="l")
min.bic <- which.min(reg.fwd.summary$bic)
points(min.bic, reg.fwd.summary$bic[min.bic], col="red") ##3

plot(reg.fwd.summary$adjr2, xlab="Number of variables", ylab="Adjusted R^2", type="l")
max.adjr2 <- which.max(reg.fwd.summary$adjr2) 
points(max.adjr2, reg.fwd.summary$adj[max.adjr2], col="red")  ##9

par(mfrow=c(3,1))
plot(reg.bwd.summary$cp, xlab="Number of variables", ylab="Cp", type="l")
min.cp <- which.min(reg.bwd.summary$cp) 
points(min.cp, reg.bwd.summary$cp[min.cp], col="red")  ##8

plot(reg.bwd.summary$bic, xlab="Number of variables", ylab="BIC", type="l")
min.bic <- which.min(reg.bwd.summary$bic)
points(min.bic, reg.bwd.summary$bic[min.bic], col="red") ##4

plot(reg.bwd.summary$adjr2, xlab="Number of variables", ylab="Adjusted R^2", type="l")
max.adjr2 <- which.max(reg.bwd.summary$adjr2) 
points(max.adjr2, reg.bwd.summary$adj[max.adjr2], col="red")  ##9

coef(regfit, min.cp)
coef(regfit, min.bic)
coef(regfit, max.adjr2)

coef(regfit.fwd, min.cp)
coef(regfit.fwd, min.bic)
coef(regfit.fwd, max.adjr2)

coef(regfit.bwd, min.cp)
coef(regfit.bwd, min.bic)
coef(regfit.bwd, max.adjr2)


#Validation#


set.seed(1234)

train <- sample(c(TRUE,FALSE), nrow(mcensus), rep=TRUE)
test <- !train
y.test <- ncrim[test]

regfit.best <- regsubsets(ncrim~., data=mcensus[train,], nvmax=13)

test.matrix <- model.matrix(ncrim~., data=mcensus[test,])

valid.errors <- rep(NA, 13)
for (i in 1:13){
	coefi <- coef(regfit.best, i)
	pred <- test.matrix[,names(coefi)]%*%coefi
	valid.errors[i] <- mean((y.test-pred)^2)
}

valid.errors

min(valid.errors)		##21.69671

MVE <- which.min(valid.errors)	#1
coef(regfit.best, 1)

regfit.full <- regsubsets(ncrim~., data=mcensus, nvmax=13)
coef(regfit.full, 1)


#10-fold Cross Validation#


set.seed(1234)

predict.regsubsets <- function(object, newdata, id){
 form <- as.formula(object$call[[2]])
 mat <- model.matrix(form, newdata)
 coefi <- coef(object, id = id)
 xvars <- names(coefi)
 mat[, xvars]%*%coefi
}

k <- 10
folds <- sample(1:k, nrow(mcensus), replace=TRUE)

cv.errors <- matrix(NA, k, 13, dimnames=list(NULL, paste(1:13))) 

for (j in 1:k){
 best.fit <- regsubsets(ncrim~., data=mcensus[folds!=j,], nvmax=13)
 for (i in 1:13){
  pred <- predict.regsubsets(best.fit, mcensus[folds==j,], id=i)
  cv.errors[j, i] <- mean((ncrim[folds==j]-pred)^2)
 }
}

mean.cv.errors <- apply(cv.errors, 2, mean)
min(mean.cv.errors)	##44.98395

MCVE <- which.min(mean.cv.errors)	##12

regfit.full <- regsubsets(ncrim~., data=mcensus, nvmax=13)
coef(regfit.full, 9)


par(mfrow=c(1,2))
plot(valid.errors, type="b", xlab="Number of variables", ylab="Validation errors")
points(MVE, valid.errors[MVE], col="red")

plot(mean.cv.errors, type = "b", xlab="Number of variables", ylab="Cross-Validation errors")
points(MCVE, mean.cv.errors[MCVE], col="red")


#Lasso#


set.seed(12345)

x <- model.matrix(ncrim~., mcensus)[,-1]
y <- ncrim
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train

lasso <- cv.glmnet(x[train,], y[train], alpha=1)
plot(lasso, ylab="MSE Lasso")

best.l.lasso <- lasso$lambda.min	#0.04373721

pred.lasso <- predict(lasso, s=best.l.lasso, newx=x[test,])
mean((pred.lasso - y[test])^2)	#52.79899

lasso.full <- glmnet(x, y, alpha=1, lambda=best.l.lasso)
coef(lasso.full)	#11


#Ridge Regression#


set.seed(12345)

ridge <- cv.glmnet(x[train,], y[train], alpha=0)
plot(ridge, ylab="MSE Ridge Regression")

best.l.ridge <- ridge$lambda.min	#0.7295807

pred.ridge <- predict(ridge, s=best.l.ridge, newx=x[test,])
mean((pred.ridge - y[test])^2)	#51.67966

ridge.full <- glmnet(x, y, alpha=0, lambda=best.l.ridge)
coef(ridge.full)


						#Classification#


threshold <- mean(ncrim, trim=0.2)
NCRIM <- as.factor(ncrim > threshold)
levels(NCRIM) <- c("LOWER", "GREATER")

new_mcensus <- cbind(mcensus[, c("lzn","dist","highway","medval")], NCRIM)

set.seed(123)

train <- sample(c(TRUE,FALSE), nrow(new_mcensus), rep=TRUE, prob=c(0.7,0.3))
test <- !train

dim(new_mcensus[train, ])
dim(new_mcensus[test, ])


#Logistic Regression#


glm.fit <- glm(NCRIM~., data=new_mcensus[train, ], family="binomial")
summary(glm.fit)

prob <- predict(glm.fit, new_mcensus[test, ], type="response")
glm.pred <- as.factor(prob > 0.5)
levels(glm.pred) <- c("LOWER", "GREATER")

table(glm.pred, new_mcensus$NCRIM[test])

mean(glm.pred == new_mcensus$NCRIM[test])		#0.9300699

glm.fit2 <- glm(NCRIM~dist+highway, data=new_mcensus[train, ], family="binomial")
summary(glm.fit2)

prob <- predict(glm.fit2, new_mcensus[test, ], type="response")
glm.pred <- as.factor(prob > 0.5)
levels(glm.pred) <- c("LOWER", "GREATER")

table(glm.pred, new_mcensus$NCRIM[test])

mean(glm.pred == new_mcensus$NCRIM[test])		#0.951049


#LDA#


lda <- lda(NCRIM~., data=new_mcensus, subset=train)

lda.pred <- predict(lda, new_mcensus[test, ])
lda.pred$posterior

table(lda.pred$class, new_mcensus$NCRIM[test])

mean(lda.pred$class == new_mcensus$NCRIM[test])		#0.9300699


#QDA#


qda <- qda(NCRIM~dist+highway+medval, data=new_mcensus, subset=train)

qda.pred <- predict(qda, new_mcensus[test, ])
qda.pred$posterior

table(qda.pred$class, new_mcensus$NCRIM[test])

mean(qda.pred$class == new_mcensus$NCRIM[test])		#0.9300699


#K-Nearest Neighbors#


set.seed(4321)

KNN.data <- new_mcensus[,-5]

kval <- 1:20
knn.res <- matrix(ncol=2, nrow=length(kval))
knn.res[,1] <- kval
for(i in 1:length(kval)){
 knn.preds <- knn(KNN.data[train, ], KNN.data[test, ], new_mcensus$NCRIM[train], k=kval[i])
 knn.res[i,2] <- mean(knn.preds == new_mcensus$NCRIM[test])
}

plot(knn.res[,1], knn.res[,2], type="l", xlab="K values", ylab="Accuracy", col="red")

max.k <- which.max(knn.res[,2])
knn.res[max.k,]

knn.pred <- knn(KNN.data[train, ], KNN.data[test, ], new_mcensus$NCRIM[train], k=max.k)

table(knn.pred, new_mcensus$NCRIM[test])

mean(knn.pred == new_mcensus$NCRIM[test])		#0.951049
