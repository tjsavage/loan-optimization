#### SETUP CODE ####
rm(list = ls())

set.seed(998)

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_13_beta.R")
library(rstudioapi)
library(caret)
library(ranger)
library(skimr)
library(ggplot2)
library(party)


current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path ))

training = read.csv("training.csv")
test = read.csv("test.csv")

### TAKE THIS OUT WHEN COMPETING
# Subset training and test data to make training models faster when testing them out

#training <- training[sample(nrow(training), 5000),]
#test <- test[sample(nrow(test), 1),]

### End section to take out

training$Fund_Date = NULL # Removing this column
test$accepted = rep(0,nrow(test)) # Adding a dummy response variable for the test set
alldata = rbind(training,test)  # Combining the two data frames to preprocess them together
alldata$Id = NULL  # Removing the Id column since it should not be used for model building
alldata$Previous_Rate = NULL # Column Previous_Rate has missing values, a basic preprocessing would be to remove it

### Simplify the columns so that we don't end up with so many dummy variables

alldata$days_since_apply_date = max(as.Date(alldata$Apply_Date)) - as.Date(alldata$Apply_Date)
alldata$days_since_approve_date = max(as.Date(alldata$Approve_Date)) - as.Date(alldata$Approve_Date)

alldata$days_between_apply_and_approve = as.Date(alldata$Approve_Date) - as.Date(alldata$Apply_Date)

# Create dummy variables for the category variables to turn them into numeric variables.

saved_accepted = alldata$accepted
dummies_model <- dummyVars(accepted ~ ., data=alldata)
allData_mat <- predict(dummies_model, newdata=alldata)
binnedAllData <- data.frame(allData_mat)
binnedAllData$accepted <- saved_accepted

# Create a convenient vector with the names of potentially interesting columns in the data
interesting_cols = c("Tier", 
                     "Primary_FICO", 
                     "Type.F", 
                     "Type.R", 
                     "Term", 
                     "New_Rate",
                     "Used_Rate", 
                     "Amount_Approved",
                     "CarType.N", 
                     "CarType.U",
                     "CarType.R", 
                     "Competition_rate", 
                     "rate", 
                     "onemonth", 
                     "termclass", 
                     "rate1", 
                     "rel_compet_rate",
                     "mp", 
                     "mp_rto_amtfinance", 
                     "partnerbin", 
                     "accepted", 
                     "days_since_approve_date", 
                     "days_since_apply_date", 
                     "days_between_apply_and_approve")

# Limit alldata just to the interesting columns

binnedAllData <- binnedAllData[,interesting_cols]


### Separate training and test data

trainingProcessed = binnedAllData[1:nrow(training),]
testProcessed = binnedAllData[-(1:nrow(training)),]

#### End Setup Code ####

trainingIndexes <- createDataPartition(
  y = trainingProcessed$accepted,
  p = .75,
  list = FALSE
)

trainingSet <- trainingProcessed[trainingIndexes,]
validationSet <- trainingProcessed[-trainingIndexes,]

### Build a few models to try

# The caret package models seem to require the outcome values to be non-numeric,
# so switching 1's and 0's to Y's and N's

caretTrainingSet <- data.frame(trainingSet)

caretTrainingSet$accepted <- ifelse(caretTrainingSet$accepted == 1, "Y", "N")

# Save some settings to use in the caret models
# "cv" tells it to do cross-validation with "number" as the number of folds

fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE,
  classProbs=TRUE,
)

caret_model <- train(
  factor(accepted) ~ .,
  data = caretTrainingSet,
  method = "glm",
  family = binomial(),
  trControl = fitControl
)

dtree_model <- train(
  factor(accepted) ~ .,
  data = caretTrainingSet,
  method = "ctree",
  trControl = fitControl
)

earth_model <- train(
  factor(accepted) ~ .,
  data = caretTrainingSet,
  method = "earth",
  trControl = fitControl
)

basic_model <- glm(accepted ~ CarType.R * New_Rate + . , data = trainingSet, family = "binomial")

benchmark_model <- glm(accepted ~ Amount_Approved + CarType.R + CarType.U + CarType.N, data = trainingSet, family = "binomial")

### End caret section

caret_predictions <- predict(caret_model, newdata = validationSet, type = "prob")
dtree_predictions <- predict(dtree_model, newdata = validationSet, type = "prob")
earth_predictions <- predict(earth_model, newdata = validationSet, type = "prob")
basic_predictions <- predict(basic_model, newdata = validationSet)
benchmark_predictions <- predict(benchmark_model, newdata = validationSet)

caret_auc <- auc(validationSet$accepted, caret_predictions$Y)
dtree_auc <- auc(validationSet$accepted, dtree_predictions$Y)
earth_auc <- auc(validationSet$accepted, earth_predictions$Y)
basic_auc <- auc(validationSet$accepted, basic_predictions)
benchmark_auc <- auc(validationSet$accepted, benchmark_predictions)


#### Build full model on all the data, only to use for the final predictions
caret_trainingProcessed <- data.frame(trainingProcessed)
caret_trainingProcessed$accepted <- ifelse(caret_trainingProcessed$accepted == 1, "Y", "N")


final_model <- train(
  factor(accepted) ~ .,
  data = caret_trainingProcessed,
  method = "earth",
  trControl = fitControl
)

# If we wanted to just use a basic model
#final_model <- glm(accepted ~ ., data = trainingProcessed, family = "binomial")

final_predictions <- predict(final_model, newdata = testProcessed, type = "prob")

submission = data.frame(Id=test$Id, Prediction = final_predictions$Y)
write.csv(submission, file = "output.csv", row.names = FALSE)
