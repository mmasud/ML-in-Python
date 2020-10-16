# loading the required library
library(caret)

# Setting the random seed
set.seed(123)

# load the loan prediction training data

data <- read.csv(url('https://datahack-prod.s3.ap-south-1.amazonaws.com/train_file/train_u6lujuX_CVtuZ9i.csv'))

# take a look at the data
dplyr::glimpse(data)

# Is there any missing values in data?
colSums(is.na(data))

# Imputing missing values using median

preProcValues <- caret::preProcess(data, method= c('medianImpute', 'center', 'scale'))

data_processed <- predict(preProcValues, data)

#Spliting training set into two parts based on outcome: 75% and 25%

indx <- createDataPartition(data_processed$Loan_Status, p = 0.75, list = FALSE)

trainSet <- data_processed[indx,]
testSet <- data_processed[-indx, ]

#Defining the training controls for multiple models

fitControl <- trainControl(method = 'cv', number = 5, savePredictions = 'final', classProbs = T)

#Defining the predictors and outcome 

outcome <- 'Loan_Status'
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome",
              "CoapplicantIncome")

# Model training

# 1. Training the random forest model

model_rf <- train(trainSet[, predictors], trainSet[, outcome], method = 'rf', trControl = fitControl, 
                  tuneLength = 3)

#Predicting using random forest model

testSet$pred_rf <- predict(object = model_rf, testSet[, predictors])

#Checking the accuracy of the random forest model

confusionMatrix(testSet$Loan_Status, testSet$pred_rf)

# Accuracy is 78%

# 2. #Training the knn model

model_knn <- train(trainSet[, predictors], trainSet[, outcome], method = 'knn', trControl = fitControl, 
                  tuneLength = 3)

#Predicting using random forest model

testSet$pred_knn <- predict(object = model_knn, testSet[, predictors])

#Checking the accuracy of the random forest model

confusionMatrix(testSet$Loan_Status, testSet$pred_knn)

# Accuracy is 81%

# 3. #Training the Logistic regression model
model_lr <- train(trainSet[, predictors], trainSet[, outcome], method = 'glm', trControl = fitControl, 
                   tuneLength = 3)

#Predicting using random forest model

testSet$pred_lr <- predict(object = model_lr, testSet[, predictors])

#Checking the accuracy of the random forest model

confusionMatrix(testSet$Loan_Status, testSet$pred_lr)

# Accuracy is 82%


# Ensemble Model ----------------------------------------------------------

# try out different ways of forming an ensemble with these models

# 1. Averaging on model probabilities

# #Predicting the probabilities
testSet$pred_rf_prob<-predict(object = model_rf,testSet[,predictors],type='prob')
testSet$pred_knn_prob<-predict(object = model_knn,testSet[,predictors],type='prob')
testSet$pred_lr_prob<-predict(object = model_lr,testSet[,predictors],type='prob')

#Taking average of predictions

testSet$pred_avg <- (testSet$pred_rf_prob$Y + testSet$pred_knn_prob$Y + testSet$pred_lr_prob$Y) /3

#Splitting into binary classes at 0.5

testSet$pred_avg <- as.factor(ifelse(testSet$pred_avg  > 0.5, 'Y', 'N'))

# 2. Majority Voting: 

testSet$pred_majority <- as.factor(ifelse(testSet$pred_rf == 'Y' & testSet$pred_knn == 'Y', 'Y',
                                          ifelse(testSet$pred_rf == 'Y' & testSet$pred_lr == 'Y', 'Y',
                                                 ifelse(testSet$pred_knn=='Y' & testSet$pred_lr=='Y','Y','N'))))

# 3. #Taking weighted average of predictions

testSet$pred_weighted_avg<-(testSet$pred_rf_prob$Y*0.25)+(testSet$pred_knn_prob$Y*0.25)+(testSet$pred_lr_prob$Y*0.5)

#Splitting into binary classes at 0.5
testSet$pred_weighted_avg<-as.factor(ifelse(testSet$pred_weighted_avg>0.5,'Y','N'))

# calculate inter-model correlation for an ensemble in R?
library(caretEnsemble)  # for creating ensemble of models
set.seed(121)
fitControl <- trainControl(method = 'cv', # type of cross-validation
                           number = 5, 
                           index=createFolds(trainSet$Loan_Status, 5), # manually set the indexes in trainControl object
                           savePredictions = 'final', classProbs = T)
model_list = c('rf', 'knn') # list of algorithms

outcome <- 'Loan_Status'
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome",
              "CoapplicantIncome")

models = caretList(Loan_Status ~., data=trainSet[, c(outcome,predictors)], trControl=fitControl, methodList= model_list)

# after creating multiple models, we will want to compare them. To compare the models we will use the resample() function from caret and 
# then use its output in the modelCor() function to find the inter-model correlation between RandomForest and KNN.

results = resamples(models) 
# inter-model correlation
modelCor(results) # modelCor() computes the correlation between the training OOF (out of fold) predictions of the models

# Note: If the model predictions are highly correlated, then using different models might not give better results than individual models.


# 4.Stacking

# Step 1: Train the individual base layer models on training data

#Defining the training control
fitControl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = 'final', # To save out of fold predictions for best parameter combinantions
  classProbs = T # To save the class probabilities of the out of fold predictions
)

#Defining the predictors and outcome
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome",
              "CoapplicantIncome")
outcomeName<-'Loan_Status'

#Training the random forest model
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl=fitControl,tuneLength=3)
                
#Training the knn model
model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl=fitControl,tuneLength=3)

#Training the logistic regression model
model_lr<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)


# Step 2: Predict using each base layer model for training data and test data


#Predicting the out of fold prediction probabilities for training data

trainSet$OOF_pred_rf <- model_rf$pred$Y[order(model_rf$pred$rowIndex)]
trainSet$OOF_pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
trainSet$OOF_pred_lr<-model_lr$pred$Y[order(model_lr$pred$rowIndex)]


#Predicting probabilities for the test data
testSet$OOF_pred_rf<-predict(model_rf,testSet[predictors],type='prob')$Y
testSet$OOF_pred_knn<-predict(model_knn,testSet[predictors],type='prob')$Y
testSet$OOF_pred_lr<-predict(model_lr,testSet[predictors],type='prob')$Y


# Step 3: Now train the top layer model again on the predictions of the bottom layer models that has been made on the training data

#Predictors for top layer models 

predictors_top <- c("OOF_pred_rf", "OOF_pred_knn", "OOF_pred_lr")

#GBM as top layer model 

model_gbm<- 
  train(trainSet[,predictors_top],trainSet[,outcome],method='gbm',trControl=fitControl,tuneLength=3)

# Similarly, Logistic regression as top layer model
model_glm<-
  train(trainSet[,predictors_top],trainSet[,outcome],method='glm',trControl=fitControl,tuneLength=3)

# Step 4: Finally, predict using the top layer model with the predictions of bottom layer models that has been made for testing data

# Final predict using GBM/lr top layer model

testSet$gbm_stacked<- predict(model_gbm, testSet[, predictors_top])
testSet$glm_stacked<- predict(model_glm, testSet[, predictors_top])


# END ---------------------------------------------------------------------


