# LOOK AT CANVAS FOR A DETAILED DESCRIPTION OF THIS CODE

# This code runs a simple logistic regression on the 
# two variables and makes predictions for the test set. 
# Then a submission file is generated that contains the prediciton.

rm(list=ls())

startTime=proc.time()[3]  # Starts a clock to measure run time

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_13_beta.R")

#setwd("C:/Users/johndoe/midterm")    # UPDATE THIS TO THE FOLDER THAT INCLUDES COMPETITION FILES

training = read.csv("training.csv")

test = read.csv("test.csv")

training$Fund_Date = NULL # Removing this column

test$accepted = rep(0,nrow(test)) # Adding a dummy response variable for the test set

alldata = rbind(training,test)  # Combining the two data frames to preprocess them together

alldata$Id = NULL  # Removing the Id column since it should not be used for model building

# Column Previous_Rate has missing values, a basic preprocessing would be to remove it

alldata$Previous_Rate = NULL 


trainingProcessed = alldata[1:nrow(training),]
testProcessed = alldata[-(1:nrow(training)),]

model = glm(accepted ~ Amount_Approved + CarType, data = trainingProcessed, family = "binomial")

probabilities = predict(model, newdata = testProcessed, type = "response")

submission = data.frame(Id=test$Id, Prediction = probabilities)

write.csv(submission, file = "benchmark.csv", row.names = FALSE)

endTime=proc.time()[3]  # records current time to calculate overall code's run-time

cat("This code took ", endTime-startTime, " seconds\n")