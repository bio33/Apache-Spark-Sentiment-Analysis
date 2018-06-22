
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml import Pipeline,PipelineModel
from collections import namedtuple


sc = SparkContext(master="local[2]", appName="Tweet Streaming App")
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 10)
sqlContext = SQLContext(sc)
ssc.checkpoint( "file:/home/ubuntu/tweets/checkpoint/")
# ssc.checkpoint("checkpoints/")
tweet_count = 0
fields = ("SentimentText")
Tweet = namedtuple( 'Tweet', fields )

pipelineFit = PipelineModel.load("logreg1.model")
def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]
def do_something(time, rdd):
    # print("========= %s =========" % str(time))
    # try:
    # Get the singleton instance of SparkSession
        spark = getSparkSessionInstance(rdd.context.getConf())
        # Convert RDD[String] to RDD[Tweet] to DataFrame
        rowRdd = rdd.map(lambda w: Tweet(w))
        linesDataFrame = spark.createDataFrame(rowRdd)
        print(type(linesDataFrame))
        # linesDataFrame.show()
        # Creates a temporary view using the DataFrame
        linesDataFrame.createOrReplaceTempView("tweets")
        # Do tweet character count on table using SQL and print it
        lineCountsDataFrame = spark.sql("select SentimentText  from tweets")
        print("line42")
        lineCountsDataFrame.printSchema()
        lineCountsDataFrame.show()
        model = PipelineModel.load("logreg1.model")
        predictions = model.transform(lineCountsDataFrame)
        predictions.show()
        # lineCountsDataFrame.coalesce(1).write.format("com.databricks.spark.csv").options(header='true',inferschema='true')\
        #     .save("dirwithcsv")
        # get_sentiment()
        # print(predictions)

    # except:
    #     print("error")
def get_sentiment():
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true')\
    .load('dirwithcsv/part-00000-3b3ee1da-e647-4d2b-9c1a-b8a4e6af5631-c000.csv')
    data.show()
    model = PipelineModel.load("logreg1.model")
    predictions = model.transform(data)
    print("predicting data")
    predictions.show()
    print(predictions)
    # predictions = pipelineFit.transform(data)
    # predictions.show()



# pipeline = PipelineModel.load("logreg.model")
# print(pipeline)
socket_stream = ssc.socketTextStream("172.31.17.157", 8000) # Internal ip of  the tweepy streamer

lines = socket_stream.window(20)
# get_sentiment()
lines.pprint(num=5)
print("working the lines")
lines.foreachRDD(do_something)

# lines.count().map(lambda x: 'Tweets in this batch: %s' % x).pprint()

### Take the stream ,for each rdd convert it into a dataframe, perform data cleaning, perform OHE and then
### sentiment prediction

#pipeline this
# lines = lines.map(lambda x: dc.remove_punctuations(x))
# lines = lines.map(lambda x: dc.emojireplace(x))
# lines = lines.map(lambda x: dc.process_tweet_text(x))
# lines = lines.map(lambda x: dc.remove_stopwords(x))
# lines = lines.map(lambda x: dc.remove_stopwords(x))
# lines = lines.map(lambda x: dc.remove_numbers(x))
# lines = lines.map(lambda x: dc.remove_shortwords(x))
# lines = lines.map(lambda x: dc.lemmatize_text(x))
# lines = lines.map(lambda x: tokenizer.transform(x))
# lines = lines.map(lambda x: model.transform(x))
# lines = lines.map(lambda x: lr.predictionCol.transform(x))








# If we want to filter hashtags only
# .filter( lambda word: word.lower().startswith("#") )
words = lines.flatMap( lambda twit: twit.split(" ") )
words.count().map(lambda x:'Words in this Tweet: %s' % x).pprint()
# pairs = words.map( lambda word: ( word.lower(), 1 ) )
# wordCounts = pairs.reduceByKey( lambda a, b: a + b ) #.transform(lambda rdd:rdd.sortBy(lambda x:-x[1]))
# wordCounts.pprint()

ssc.start()
ssc.awaitTermination()
