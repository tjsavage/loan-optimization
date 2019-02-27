rm(list = ls())

source("http://www.stanford.edu/~bayati/oit367/T367_utilities_13_beta.R")

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

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
