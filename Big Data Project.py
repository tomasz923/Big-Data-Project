# Databricks notebook source
# MAGIC %md
# MAGIC #BIG DATA FINAL PROJECT: Predicting Earnings from Census Data  
# MAGIC   
# MAGIC   
# MAGIC - Georgios Kanetounis - ID 125627 
# MAGIC - Tomasz Podyma - ID 116042
# MAGIC - Ana Gabriela Ramirez Lopez - ID 120164

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dataset preprocessing and exploration
# MAGIC
# MAGIC The United States government periodically collects demographic information by conducting a census. In this project, we are going to use census information about an individual to classify whether the person earns more or less than $50,000 per year. This data comes from the UCI Machine Learning Repository.
# MAGIC
# MAGIC The file census (CSV) contains 1994 census data for 32,561 individuals in the United States.
# MAGIC
# MAGIC Source: https://archive.ics.uci.edu/ml/datasets/adult
# MAGIC
# MAGIC The dataset includes the following 15 variables:
# MAGIC *  age = the age of the individual in years
# MAGIC * workclass = the classification of the individual’s working status (does the *person work for the federal government, work for the local government, work without pay, and so on)
# MAGIC * fnlwgt = final weight. Number of units in the target populations that the responding unit represents
# MAGIC * education = the level of education of the individual (e.g., 5th-6th grade, high school graduate, PhD, so on)
# MAGIC * education num = year numbers of educatoin
# MAGIC * maritalstatus = the marital status of the individual
# MAGIC * occupation = the type of work the individual does (e.g., administrative/clerical work, farming/fishing, sales and so on)
# MAGIC * relationship = relationship of individual to his/her household
# MAGIC * race = the individual’s race
# MAGIC * sex = the individual’s sex
# MAGIC * capitalgain = the capital gains of the individual in 1994 (from selling an asset such as a stock or bond for more than the original purchase price)
# MAGIC * capitalloss = the capital losses of the individual in 1994 (from selling an asset such as a stock or bond for less than the original purchase price)
# MAGIC * hoursperweek = the number of hours the individual works per week
# MAGIC * nativecountry = the native country of the individual
# MAGIC * income level = whether or not the individual earned more than $50,000 in 1994
# MAGIC
# MAGIC Since the fnlwft variable is not a demographic variable per se, it will be excluded from the analysis and won't be usec as a feature in the classfication models.

# COMMAND ----------

#Importing all libraries for Data preprocessing

from pyspark.sql.functions import col
from pyspark.sql.functions import col,isnan, when, count

# COMMAND ----------

#Loading dataset

# File location and type
file_location = "/FileStore/tables/adults.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
census = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(census)

# COMMAND ----------

#Drop the Fnlwft since it won't be used for any model or analysis
census = census.drop(census['fnlwgt'])

display(census)

display(census.count())

# COMMAND ----------

#Exploring categories in categorical variables to check which variables can possibly need treatment

census.select('age').distinct().show()
census.select('workclass').distinct().show()
census.select('education').distinct().show()
census.select('marital_status').distinct().show()
census.select('occupation').distinct().show()
census.select('relationship').distinct().show()
census.select('race').distinct().show()
census.select('sex').distinct().show()
census.select('native_country').distinct().show()
census.select('income_level').distinct().show()

# COMMAND ----------

#Number of individuals with no values for workclass, occupation and native_country
print (census.filter(col("workclass").contains("?")).count(),
       census.filter(col("occupation").contains("?")).count(),
       census.filter(col("native_country").contains("?")).count())

# COMMAND ----------

#Since there is no clear approach for imputations, we proceed with deleting the individuals with at least one missing value (i.e, '?')
census = census.filter(~census.native_country.contains("?") &
                               ~census.occupation.contains("?") &
                               ~census.workclass.contains("?") )
display(census.count())

# COMMAND ----------

#Checking for null / nan values
display(census.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in census.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dataset descriptive Analysis
# MAGIC
# MAGIC We start our descriptive analysis with some numerical and categorical features to get a brief idea of our data and then focus on some variables that were deemed influential or whose results were unexpected.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In the below Data Profile (second tab), one can see that the average age of participants is 38 years old, the average education in years is 10 years and the average number of hours worked is 41 per week. It is worth mentioning that more than 90% of participants did not report any loss or gain.
# MAGIC
# MAGIC It can also be observed that the majority of the participants work in the private sector, their highest level of educational attainment is high school, are married, identify their ethnicity as white, are US citizens, earn less than 50 thousand dollars per year and the number of men are double than the number of women.

# COMMAND ----------

display(census)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Income levels, Share of work classes, and Education years
# MAGIC
# MAGIC **Income levels**
# MAGIC
# MAGIC In the Income levels chart below, one can see that every fourth citizen of the US was earning more than 50 000 dollars a year.
# MAGIC
# MAGIC **The share of work classes in total**
# MAGIC
# MAGIC The types are coded as follow:
# MAGIC
# MAGIC - Private - people who work in private sector
# MAGIC - Self-emp-not-inc - people who work for themselves but not in corporate entities
# MAGIC - Local-gov - people who work for a local government
# MAGIC - State-gov - people who work for a state government
# MAGIC - Self-emp-in - people who work for themselves in corporate entities
# MAGIC - Federal-gov - people who work for the federal government
# MAGIC - Without-pay - people who do not get paid for their work
# MAGIC
# MAGIC In the Work Classes chart below, one can see that only every fourth worker is not hired by a private company. Around 13% works in the public sector, another 12% is self-employed and the number of people working for free is negligible.
# MAGIC
# MAGIC **The length of education**
# MAGIC
# MAGIC In the Education years chart below, one can see that the biggest group of workers spent between 8,5 and 11 years in schools.

# COMMAND ----------

display(census)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Demographics vs Income
# MAGIC
# MAGIC **Marital status versus Income**
# MAGIC The marital status versus Income plot shows that the vast majority of people who were divorced or never married earn less that 50 thousand dollars per year while the married people are equally split between those who earn more than 50 thousand dollars per year and those who earn less than 50 thousand dollars per year. 
# MAGIC
# MAGIC **Education versus Income**
# MAGIC From the education versus Income plot it is observed that little percentage of high school graduates and college graduates earn more than 50 thousand dollars per year while the gap closes for those who graduate with an undergraduate diploma and it is almost equally split for the Masters holders
# MAGIC
# MAGIC **Education (in years) versus Income**
# MAGIC The education (in years) versus income bar plot indicates that the people who have at least 14 years in education usually earn more than 50 thousand dollars per year
# MAGIC
# MAGIC **Occupation versus Income**
# MAGIC Occupation versus income displays that while most of the occupations compensate less than 50 thousand dollars per year, in exectutive and managerial positions it is 50% likely that someone will earn more than 50 thousand dollars per year.
# MAGIC
# MAGIC **Sex versus Income**
# MAGIC Sex versus income piechart shows that men are almost 3 times more probable to earn more than 50 thousand dollars per year than women
# MAGIC
# MAGIC **Hours per week versus income**
# MAGIC As expected, those with an income greater that 50 thousand dollars work in average more hours per week that those with lower level of income.

# COMMAND ----------

display(census)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Classification models for level of income
# MAGIC
# MAGIC In this literal, two classification models are built for the prediction of the level of income in the US (either greater than 50K or less than 50K per year) based on demographic data : a logistic regression and a decision tree. 
# MAGIC
# MAGIC In principle, the input data and output data are prepared in order to build both models. The categorical features are first encoded and then a VectorAssembler is used on all the features to transform them into a format that can be used for the machine learning algorithms in the training process. 
# MAGIC
# MAGIC Afterwards, both models are trained and evaluated. Finally, cross-validation and hyperparameter tunning is executed to improve the performance of each model.

# COMMAND ----------

#Importing libraries for creations and measurement of classification models
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

#Preparing categorical variables to start to build a model. 

data=census.select("age","workclass","education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_level")
 
categorical_variables=["workclass","education","marital_status","occupation","relationship", "race", "sex", "native_country", "income_level"]
 
for variable in categorical_variables:
    #converts string variables to numerical indices 
    indexer = StringIndexer(inputCol=variable, outputCol=variable+"index")
    data = indexer.fit(data).transform(data)
 
    #explodes the now numerical categorical variables into binary variables 
    encoder = OneHotEncoder(inputCol=variable+"index", outputCol=variable+"vec")
    model = encoder.fit(data)
    data = model.transform(data)   
 
    
display(data)

# COMMAND ----------

#We transform all features into a vector using VectorAssembler to be able to train and test model

dataSelect = data.select("age","workclassvec","educationvec", "education_num", "marital_statusvec", "occupationvec", "relationshipvec", "racevec", "sexvec", "capital_gain", "capital_loss", "hours_per_week", "native_countryvec", "income_levelindex")

predictors = ["age","workclassvec","educationvec", "education_num", "marital_statusvec", "occupationvec", "relationshipvec", "racevec", "sexvec", "capital_gain", "capital_loss", "hours_per_week", "native_countryvec"]

# Preparing the input data by creating a VectorAssembler
assembler = VectorAssembler(inputCols=predictors, outputCol="features")

dataset = assembler.transform(dataSelect)
dataset = dataset.select("income_levelindex","features")

# COMMAND ----------

# Train (70% of the data) and test (30% of the data) data sets
testDataset, trainDataset = dataset.randomSplit(weights=[0.7, 0.3],  seed=123)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Logistic regression 
# MAGIC
# MAGIC Logistic regression is a classification algorithm employed when the value of the target variable is categorical in nature. Logistic regression is most often used when the data in question have binary output, that is, when they belong to one class or another, which makes it suitable for our purposes. This model used a sigmoid function in order to map the predicted values to probabilities, being capable of mapping any real value into another value between 0 to 1.
# MAGIC
# MAGIC In this section, a logistic regression was used to predict the income level of workers in the US. A LogisticRegression object was created, and the model was fit to the training data by using the fit() function.
# MAGIC
# MAGIC The model was then used to make predictions on the test data, and the performance of the model was evaluated using metrics such as the Area Under the Receiver Operating Characteristic Curve (AUC) and Accuracy. The cross-validation process was performed in the hyperparameter tunning in order to to improve its performance.

# COMMAND ----------

#Building and fitting logistic regression model
lr = LogisticRegression(featuresCol = 'features', labelCol = 'income_levelindex', maxIter=5)
lrModel = lr.fit(trainDataset)

#Predictions
predictions = lrModel.transform(testDataset)
predictions.select('income_levelindex', 'features', 'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 3.1.1. Model Evaluation
# MAGIC
# MAGIC **Area Under ROC**
# MAGIC
# MAGIC We use BinaryClassificationEvaluator to evaluate the performance of the model which uses areaUnderROC as the default measure for evaluation. 
# MAGIC
# MAGIC ROC tells how much model is capable of distinguishing between classes. The higher the areaUnderROC, the better the model is at distinguishing between individuals whose income is greater that 50k and lower thank 50k.

# COMMAND ----------

evaluator = BinaryClassificationEvaluator().setLabelCol("income_levelindex").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
            
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC **Model Accuracy**
# MAGIC
# MAGIC Accuracy tells the number of classifications a model correctly predicts divided by the total number of predictions made

# COMMAND ----------

accuracy = predictions.filter(predictions.income_levelindex == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ",accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 3.1.2. Cross Validation and Parameter Tuning
# MAGIC In this section we will proceed with model tunning for the Logistic regression, in order to find the optimal values of hyperparameters to maximize the model performance.
# MAGIC
# MAGIC For this purpose we use GridSearch which takes a dictionary of all of the different hyperparameters that we want to test, and then reports back which combination had the highest accuracy.

# COMMAND ----------

#ParamGrid for Cross Validation

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.

paramGrid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.02,0.4,3.0])#regularization parameter
            .addGrid(lr.elasticNetParam, [0.0,0.3,1]) #Elastic Net Parameter
            .addGrid(lr.maxIter, [2,8,15]) #Number of iterations
            .build())

#A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
#Note that the evaluator here is a BinaryClassificationEvaluator and its default metric is areaUnderROC.
cv = CrossValidator(estimator = lr, estimatorParamMaps = paramGrid, evaluator=evaluator, numFolds =5)


#Run cross-validation, and choose the best set of parameters.
cvModel = cv.fit(trainDataset)

# COMMAND ----------

# Making predictions with cvModel which uses the best model found.
predictions = cvModel.transform(testDataset)
print ('Best Model Test Area Under ROC', evaluator.evaluate(predictions))

accuracy = predictions.filter(predictions.income_levelindex == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ", accuracy)

# COMMAND ----------

#Obtaining the values for the best hyper-parameters
best_model = cvModel.bestModel

print ('Best Param (regParam):', best_model._java_obj.getRegParam())
print ('Best Param (MaxIter):', best_model._java_obj.getMaxIter())
print ('Best Param (elasticNetParam):', best_model._java_obj.getElasticNetParam())

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.2 Decision Tree Model
# MAGIC
# MAGIC Classification tree models are widely used in the field of machine learning for binary classification problems. This type of model uses a tree-like structure to split the data into smaller and smaller subsets, and the final prediction is made based on the majority class of the subset that the data belongs to.
# MAGIC
# MAGIC In this section, a classification tree model was used to predict the income level of workers in the US. A DecisionTreeClassifier object was created, and the model was fit to the training data by using the fit() function.
# MAGIC
# MAGIC The model was then used to make predictions on the test data, and the performance of the model was evaluated using metrics such as the Area Under the Receiver Operating Characteristic Curve (AUC) and Accuracy. The cross-validation process was performed in the hyperparameter tunning in order to to improve its performance.

# COMMAND ----------

# Creating a DecisionTreeClassifier object
dt = DecisionTreeClassifier(labelCol="income_levelindex", featuresCol="features")

# Fitting the model to the training data
dtModel = dt.fit(trainDataset)

# Making predictions on the test data
predictions = dtModel.transform(testDataset)
predictions.select('income_levelindex', 'features', 'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.1 Model Evaluation

# COMMAND ----------

#Selecting several columns from a Spark dataframe called "predictions" 
#and converting the dataframe to a Pandas dataframe for display purposes.
predictions.select('income_levelindex', 'features', 'rawPrediction', 'prediction', 'probability').toPandas().head(5)

#Evaluating binary classification models
evaluator = BinaryClassificationEvaluator().setLabelCol("income_levelindex").setRawPredictionCol("prediction").setMetricName("areaUnderROC")

print('Test Area Under ROC', evaluator.evaluate(predictions))


# COMMAND ----------

#Calculating the accuracy of a machine learning model's predictions 
#by dividing the number of correct predictions by the total number of predictions. 
accuracy = predictions.filter(predictions.income_levelindex == predictions.prediction).count() / float(predictions.count())

print("Accuracy : ", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.2 Cross-Validation and Parameter Tuning

# COMMAND ----------

#Defining a parameter grid for a decision tree model
paramGridTwo = (ParamGridBuilder()
            .addGrid(dt.maxDepth, [2, 5, 12, 15]) #The maximum depth of the tree
            .addGrid(dt.maxBins, [8, 16, 32, 64]) #The maximum number of bins
            .build())

# COMMAND ----------

#Again, CrossValidator requires an Estimator,
cvTwo = CrossValidator(estimator = dt, estimatorParamMaps = paramGridTwo, evaluator = evaluator)

# COMMAND ----------

#Running cross-validation
cvModelTwo = cvTwo.fit(trainDataset)

# COMMAND ----------

#Getting results to compare with previous model
predictions = cvModelTwo.transform(testDataset)
print ('Best Model Test Area Under ROC', evaluator.evaluate(predictions))

accuracy = predictions.filter(predictions.income_levelindex == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ", accuracy)

# COMMAND ----------

#Obtaining the values for the best hyper-parameters
best_modelTwo = cvModelTwo.bestModel

print ('Best parameter of the maximum depth of the tree:', best_modelTwo._java_obj.getMaxDepth())
print ('Best parameter of he maximum number of bins:', best_modelTwo._java_obj.getMaxBins())

# COMMAND ----------

# MAGIC %md
# MAGIC ##4. Conclusions
# MAGIC
# MAGIC The objective of this study was to classify the income level of workers in the US based on demographic variables. To this end we first proceed with the dataset preprocessing and exploration. We then continue with a descriptive analysis of the features and the target variables. Later, we use two machine learning models for the classification of income level: a logistic regression model and a classification tree model. Both models were trained on a dataset that contained various predictors, such as age, work class, years of education, martial status, occupation, relationship status, race, sex, capital gain, capital loss, hours per week worked, and native country. Their evaluation was based on the Area Under ROC (AUC) and accuracy metric. The former was later used as chosen performance measure in the Cross-Validation for hyper-parameter tunning.
# MAGIC
# MAGIC The descriptive analysis showed some unexpected results, such as that people who have never been married are very likely to earn less than $50,000 per year and that the probability of men earning more than $50,000 per year is 3 times higher than the probability of women earning more than $50,000 per year.
# MAGIC  
# MAGIC Furthermore, after cross-validation of the classification models, the logistic regression model outperformed the classification tree model, with an AUC score of 0.758938 compared to 0.753679 for the classification tree model. This suggests that, based of the features used, the logistic regression model was more effective than the classification tree model in discriminating between individuals whose income is greater than 50K dollars per years vs. those whose income is lower than that. However, it is worth noting that for the Logistic Regression even if the parameter tunning contributed to a higher AUC, it lowered its accuracy. This resulted in having a slightly greater accuracy for the Decision Tree Model compared to the Logistic regression. 
# MAGIC
# MAGIC Finally, as future work it would be worthwhile to explore different classification models, performance metrics, as well as different subsets of features to see if performance can be improved.
