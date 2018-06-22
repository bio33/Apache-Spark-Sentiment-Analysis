from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

# from pyspark.ml import PipelineModel

sc = SparkContext()
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('Tweets.csv')

drop_list = ['tweet_id','airline_sentiment_confidence','negativereason','negativereason_confidence'
             'airline','airline_sentiment_gold','name','retweet_count','tweet_coord','tweet_created','tweet_location'
             'user_timezone','negativereason_gold','tweet_location','user_timezone','negativereason_confidence']
data= data.withColumnRenamed("airline_sentiment","Sentiment")
data = data.withColumnRenamed("text","SentimentText")
data = data.select([column for column in data.columns if column not in drop_list])

data=data.where(col("Sentiment").isNotNull())
data=data.where(col("SentimentText").isNotNull())
data.show(5)
data.printSchema()

data.groupBy("Sentiment") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

data.groupBy("SentimentText") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# set seed for reproducibility
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed = 100)
# print("Training Dataset Count: " + str(trainingData.count()))
# print("Test Dataset Count: " + str(testData.count()))

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="SentimentText", outputCol="words", pattern="\\W")
t = regexTokenizer.transform(trainingData)

# stop words
add_stopwords = ["http","https","amp","rt","t","c","the"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
t= stopwordsRemover.transform(t)
# bag of words count

countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
model = countVectors.fit(t)
t = model.transform(t)
t.show()
# t=t.fillna("unk")
# tokenized=tokenized.where(tokenized.text.isNotNull())
t=t.where(t.Sentiment.isNotNull())
# convert string labels to indexes
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")


t=t.na.drop()

t.printSchema()
# t.filter(t.Sentiment == "null" or t.Sentiment = ).collect()
r = label_stringIdx.fit(t)

t = r.transform(t)
t.show()
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
# lrModel = lr.fit(trainingData)

print("I am here")
# build the pipeline
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx, lr])

#Fit the pipeline to training documents.
    # pipelineFit = pipeline.fit(trainingData)
    # pipelineFit = pipeline.fit(data)
    #
    # predictions = pipelineFit.transform(testData)
    #
    # predictions.filter(predictions['prediction'] == 0) \
    #     .select("SentimentText","Sentiment","probability","label","prediction") \
    #     .orderBy("probability", ascending=False) \
    #     .show(n = 10, truncate = 30)
    # predictions.filter(predictions['prediction'] == 1) \
    #     .select("SentimentText","Sentiment","probability","label","prediction") \
    #     .orderBy("probability", ascending=False) \
    #     .show(n = 10, truncate = 30)
    # predictions.filter(predictions['prediction'] == 2) \
    #     .select("SentimentText","Sentiment","probability","label","prediction") \
    #     .orderBy("probability", ascending=False) \
    #     .show(n = 10, truncate = 30)
    # # Evaluate, metricName=[accuracy | f1]default f1 measure
    # evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",labelCol="label")
    # print("F1: %g" % (evaluator.evaluate(predictions)))

# save the trained model for future use
# pipelineFit.save("logreg1.model")
#
testData.show()
testData.printSchema()
model = PipelineModel.load("logreg1.model")
predictions = model.transform(testData)
predictions.show()
predictions.filter(predictions['prediction'] == 0) \
    .select("SentimentText","Sentiment","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

predictions.filter(predictions['prediction'] == 1) \
    .select("SentimentText","Sentiment","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
predictions.filter(predictions['prediction'] == 2) \
    .select("SentimentText","Sentiment","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
# Evaluate, metricName=[accuracy | f1]default f1 measure
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",labelCol="label")
print("F1: %g" % (evaluator.evaluate(predictions)))

