#### SETUP CODE ####
rm(list = ls())

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_13_beta.R")
library(rstudioapi)

current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path ))

training = read.csv("training.csv")
test = read.csv("test.csv")
training$Fund_Date = NULL # Removing this column
test$accepted = rep(0,nrow(test)) # Adding a dummy response variable for the test set
alldata = rbind(training,test)  # Combining the two data frames to preprocess them together
alldata$Id = NULL  # Removing the Id column since it should not be used for model building
alldata$Previous_Rate = NULL # Column Previous_Rate has missing values, a basic preprocessing would be to remove it

trainingProcessed = alldata[1:nrow(training),]
testProcessed = alldata[-(1:nrow(training)),]

## Randomize the training data, and split it into different folds for cross-validation

trainingProcessedRandomized = trainingProcessed[sample(nrow(trainingProcessed)),]

num_folds = 5
folds <- cut(seq(1,nrow(trainingProcessedRandomized)),breaks=num_folds,labels=FALSE)

#### END SETUP CODE ####

#### Train the model ####

# Create a function in which we pass in the training data set, and return the model
generateModel <- function(trainingSet) {
  model = glm(accepted ~ Amount_Approved + CarType, data = trainingSet, family = "binomial")
  
  return(model)
}

# Do a 5-fold cross-validation on the model

aucs <- rep(0, num_folds)

for(i in 1:num_folds) {
  # Set up the current validation and training sets using the folds
  currValidationIndexes <- which(folds==i, arr.ind=TRUE)
  currValidationSet <- trainingProcessedRandomized[currValidationIndexes,]
  currTrainingSet <- trainingProcessedRandomized[-currValidationIndexes,]
  
  # Train the model
  currModel <- generateModel(currTrainingSet)
  currPredictions <- predict(currModel, newdata=currValidationSet)
  currAUC <- auc(currValidationSet$accepted, currPredictions)
  
  aucs[i] <- currAUC
}

cat(mean(aucs))


#### Output predictions for the test set
probabilities = predict(model, newdata = testProcessed, type = "response")
submission = data.frame(Id=test$Id, Prediction = probabilities)
write.csv(submission, file = "benchmark.csv", row.names = FALSE)


cat("AUC: " + auc())