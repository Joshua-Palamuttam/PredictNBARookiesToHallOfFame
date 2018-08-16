# Working script file
# Michael Domke
# Team Space JAM

rm(list=ls())

# Domke
setwd("C:/Users/domkema/Google Drive/Rose-Hulman/2017-2018 Senior Year/CSSE290 machine learning/Team Space JAM Project_/data")

# AJ
setwd("~/Rose-Hulman/Senior Year/Spring Quarter/CSSE290 - Machine Learning in R")

# Josh
setwd("C:/Users/Administrator/Google Drive/College/RHIT/Junior Year/Quarter 3/Machine Learning/Team Space JAM Project_/data")

rookie <- read.csv(
  "NBA Rookies by Year - All Star and HoF designations.csv", 
  stringsAsFactors = FALSE)

library(class)
library(gmodels)

###########################
######## Functions ########

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

###########################

###########################
########### kNN ###########
###########################
rookie$X3P. = NULL # Remove 3 point percentage data causing problems with kNN

rookie_n = as.data.frame(lapply(rookie[3:21], normalize)) # normalizing player data; excluding name and rookie year

rookie_train <- rookie_n[6:1500, ] # training data subset
rookie_test <- rookie_n[c(1:5,1501:1537), ] # testing data subset

# create labels for training and test data (HOF or na)
rookie_train_labels <- rookie[6:1500, 23]
rookie_test_labels <- rookie[c(1:5,1501:1537), 23]

rookie_test_pred <- knn(train = rookie_train, test = rookie_test,
                      cl = rookie_train_labels, k = 21)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = rookie_test_labels, y = rookie_test_pred,
           prop.chisq = FALSE)
###########################
###########################


###########################
####### Naive Bayes #######
###########################

rookie_raw = rookie

rookie_raw$HOF = rookie_raw$HOF+1

library(tm)

rookie_corpus <- VCorpus(VectorSource(rookie_raw$GP))



###########################
###########################

###########################
########## Tree ###########
###########################

rookie$X = NULL

#lapply(rookie$HOF, )
rookie$HOF = factor(rookie$HOF, c(0,1),c("no", "yes"))

set.seed(13)

train_sample <- sample(nrow(rookie), round(nrow(rookie)*.9))

rookie_train <- rookie[train_sample, ]
rookie_test  <- rookie[-train_sample, ]

prop.table(table(rookie_train$HOF))
prop.table(table(rookie_test$HOF))

library(C50)
library(gmodels)

rookie_model <- C5.0(rookie_train[c(-1,-2,-23,-24)], rookie_train$HOF)

summary(rookie_model)

rookie_pred <- predict(rookie_model, rookie_test)
CrossTable(rookie_test$HOF, rookie_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

summary(rookie_pred)

###########################
####### Regression ########
###########################
rookie <- read.csv(
  "NBA Rookies by Year - All Star and HoF designations.csv", 
  stringsAsFactors = FALSE)

#All Star Prediction
rookie<-rookie[4:23]

#HOF Prediction
rookie<-rookie[4:24]

str(rookie)

reg <- function(y, x) {
  x <- as.matrix(x)
  x <- cbind(Intercept = 1, x)
  b <- solve(t(x) %*% x) %*% t(x) %*% y
  colnames(b) <- "estimate"
  print(b)
}

hist(rookie$PTS)

table(rookie$PTS)

cor(rookie)
pairs(rookie)

library(psych)
pairs.panels(rookie)

rookie_model<-lm(HOF~., data=rookie)
rookie_model
summary(rookie_model)

rookie_train<-rookie[1:1530,]
rookie_test<-rookie[1531:1538,]

library(rpart)
library(rpart.plot)

m.rpart<-rpart(HOF~.,data=rookie_train)
m.rpart
rpart.plot(m.rpart,digits=3)
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)

###########################
###### Neural Nets ########
###########################

rookie_n = as.data.frame(lapply(rookie[3:24], normalize)) # normalizing player data; excluding name and rookie year
str(rookie_n)
rookie_n

rookie_train<-rookie_n[1:1000,]
rookie_test<-rookie_n[1001:1538,]

library(neuralnet)

set.seed(12345)
column_names<-paste(colnames(rookie_n)[1:21], collapse = '+')
column_names
rookie_form<-as.formula(paste('HOF','~',column_names))
rookie_model<-neuralnet(rookie_form,data=rookie_train)
plot(rookie_model)

# obtain model results
model_results <- compute(rookie_model, rookie_test[1:21])
# obtain predicted strength values
predicted_HOF <- model_results$net.result
# examine the correlation between predicted and actual values
cor(predicted_HOF, rookie_test$sHOF)

rookie_model2<-neuralnet(rookie_form,data=rookie_train, hidden=5)
plot(rookie_model2)

# obtain model results
model_results <- compute(rookie_model2, rookie_test[1:21])
# obtain predicted strength values
predicted_HOF <- model_results$net.result
# examine the correlation between predicted and actual values
cor(predicted_HOF, rookie_test$HOF)


###########################
######## k-means ##########
###########################

rookie <- read.csv(
  "NBA Rookies by Year - All Star and HoF designations.csv", 
  stringsAsFactors = FALSE)

stats<-rookie[3:22]
stats_z<-as.data.frame(lapply(stats,scale))

set.seed(3)
rookie_clusters<-kmeans(stats,2)

rookie_clusters$centers
rookie_clusters$size


rookie$cluster<-rookie_clusters$cluster
rookie[,c("cluster","Name", "PTS","All.Star","HOF")]

-###########################
###########################

# plotting 
rbPal <- colorRampPalette(c('black','red'))   # create vector of color choices

#This adds a column of color values
# based on the y values
rookie$Color <- rbPal(2)[as.numeric(cut(rookie$All.Star,breaks = 2))] # make vector of same length as data with color choices to highlight HOF'ers


plot(x = rookie$Year.Drafted, y = rookie$PTS,
     main = "Scatterplot of Rookie Year vs. Points per Game with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$MIN, y = rookie$PTS,
     main = "Scatterplot of minutes played vs. Points per Game with HOF'ers in RED",
     xlab = "Minutes per game",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$MIN,
     main = "Scatterplot of Rookie Year vs. minutes played with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Minutes per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$EFF,
     main = "Scatterplot of Rookie Year vs. player effeciency factor with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Player Effeciency factor",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$FGA,
     main = "Scatterplot of Rookie Year vs. field goal avg with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Field goals average",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$FTA,
     main = "Scatterplot of Rookie Year vs. Free throw avg with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Free throw average per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$PTS,
     main = "Scatterplot of Rookie Year vs. Points per Game with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$PTS, y = rookie$EFF,
     main = "Scatterplot of Rookie Year vs. Points per Game with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point


#### Plotting to show All Stars vs Hall of Fame and All Stars
rbPal <- colorRampPalette(c('black','yellow','blue','red'))   # create vector of color choices

#This adds a column of color values
# based on the y values
rookie$HOF = rookie$HOF * 2
rookie$rank = rookie$All.Star + rookie$HOF
rookie$Color <- rbPal(4)[as.numeric(cut(rookie$rank,breaks = 4))] # make vector of same length as data with color choices to highlight HOF'ers


plot(x = rookie$Year.Drafted, y = rookie$PTS,
     main = "Scatterplot of Rookie Year vs. Points per Game with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$MIN, y = rookie$PTS,
     main = "Scatterplot of minutes played vs. Points per Game",
     xlab = "Minutes per game",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

legend('topleft', pch=15, col=c('black','yellow','blue','red'), c('Nothing','All Star','HOFer','All Star & HOFer'), cex=1)

plot(x = rookie$Year.Drafted, y = rookie$MIN,
     main = "Scatterplot of Rookie Year vs. minutes played with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Minutes per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$EFF,
     main = "Scatterplot of Rookie Year vs. player effeciency factor with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Player Effeciency factor",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$FGA,
     main = "Scatterplot of Rookie Year vs. field goal avg with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Field goals average",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$FTA,
     main = "Scatterplot of Rookie Year vs. Free throw avg with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Free throw average per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$Year.Drafted, y = rookie$PTS,
     main = "Scatterplot of Rookie Year vs. Points per Game with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point

plot(x = rookie$PTS, y = rookie$EFF,
     main = "Scatterplot of Rookie Year vs. Points per Game with HOF'ers in RED",
     xlab = "Rookie draft year",
     ylab = "Points per game",
     col = rookie$Color,
     cex = 1,  # defines the size of the plot point
     pch = 15) # pch defines the shape of the plot point


