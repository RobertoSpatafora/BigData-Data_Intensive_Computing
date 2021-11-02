// Databricks notebook source
import scala.io.Source
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions._

//load all the data from csv files
var dataset = spark.read.option("header",true).csv(
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_03_29_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_03_30_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_02_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_03_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_04_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_05_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_06_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_07_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_08_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_09_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_10_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_11_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_12_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_13_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_14_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_15_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_16_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_17_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_18_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_19_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_20_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_21_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_22_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_23_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_24_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_25_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_26_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_27_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_28_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_29_Coronavirus_Tweets.CSV",
                                                    "dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_04_30_Coronavirus_Tweets.CSV"
                                                  )
//only one day for sentiment analysis
var dataset_small = spark.read.option("header",true).csv("dbfs:/FileStore/shared_uploads/stefanon@kth.se/2020_03_29_Coronavirus_Tweets.CSV")

// COMMAND ----------

//load map of country name and country code
var country_map = spark.read.option("header",true).csv("dbfs:/FileStore/shared_uploads/stefanon@kth.se/Countries.CSV")
country_map.printSchema()

// COMMAND ----------

//visualize data
dataset.printSchema()

// COMMAND ----------

//select used columns and visualize sample of data
dataset = dataset.select(col("text"), col("is_retweet"), col("retweet_count"), col("country_code"), col("lang"))
dataset.show()

// COMMAND ----------

//drop null text values in the tweet
dataset = dataset.na.drop(Seq("text"))
//dataset.count()

// COMMAND ----------

//Stopwords are used to remove commonly used words that not represent Meaningful information 
val stopwords = Array( 
  "0","1","2","3","4","5","6","7","8","9","10",
  
  "i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","http","https","co","","us","amp","get","re","d","ll","-",
           "un","una","unas","unos","uno","sobre","todo","también","tras","otro","algún","alguno","alguna","algunos","algunas","ser","es","soy","eres","somos","sois","estoy","esta","estamos","estais","estan","como","en","para","atras","porque","porqué","estado","estaba","ante","antes","siendo","ambos","pero","por","poder","puede","puedo","podemos","podeis","pueden","fui","fue","fuimos","fueron","hacer","hago","hace","hacemos","haceis","hacen","cada","fin","incluso","primero","desde","conseguir","consigo","consigue","consigues","conseguimos","consiguen","ir","voy","va","vamos","vais","van","vaya","gueno","ha","tener","tengo","tiene","tenemos","teneis","tienen","el","la","lo","las","los","su","aqui","mio","tuyo","ellos","ellas","nos","nosotros","vosotros","vosotras","si","dentro","solo","solamente","saber","sabes","sabe","sabemos","sabeis","saben","ultimo","largo","bastante","haces","muchos","aquellos","aquellas","sus","entonces","tiempo","verdad","verdadero","verdadera","cierto","ciertos","cierta","ciertas","intentar","intento","intenta","intentas","intentamos","intentais","intentan","dos","bajo","arriba","encima","usar","uso","usas","usa","usamos","usais","usan","emplear","empleo","empleas","emplean","ampleamos","empleais","valor","muy","era","eras","eramos","eran","modo","bien","cual","cuando","donde","mientras","quien","con","entre","sin","trabajo","trabajar","trabajas","trabaja","trabajamos","trabajais","trabajan","podria","podrias","podriamos","podrian","podriais","yo","aquel","de","n","y","o","que","des","per", "dr","p","ya","a","b","c","d","e","f","g","h","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","z","ne","je","gt","te","na","del","le","se","#","al","et","one","di","les"              )


//Remover of stopwords form column "term" and the words to keep will be in column "ok"
val remover = new StopWordsRemover()
  .setInputCol("term")
  .setOutputCol("ok")
  .setStopWords(stopwords)

// COMMAND ----------

// DBTITLE 1,Division by Country
//drop null values in column "country_code" and select only column text and 
val dataset_country = dataset
                     .select(col("text"), col("country_code"))
                     .na.drop(Seq("country_code"))
//split "text" into single words and keep them in a list called "term"
val dataset_country_words = dataset_country
                            .withColumn("text", lower(col("text")))
                            .select(col("country_code"), split(col("text"), "[^a-z0-9#-]").alias("term"))

dataset_country_words.show()

// COMMAND ----------

//apply the remover to remove stopwards
val newDataSet = remover.transform(dataset_country_words)

// COMMAND ----------

// DBTITLE 1,group the words by country 
//explode the words in serveral rows, gruop them by "country_code" and count
val wordcount = newDataSet
                .select(col("country_code"), explode(col("ok")).alias("term"))
                //.where("term != ''")
                .groupBy(col("country_code"), col("term")).count()
                .sort(-col("count"))

wordcount.show()

// COMMAND ----------

//join with another df to retrieve the country name
val country_name_count_df = wordcount.join(country_map, Seq("country_code"), "inner")//.select(col("term"),col("count"),col("country"))

country_name_count_df.show()

// COMMAND ----------

//extract the list of all the possible county in a list
val country_list = country_map.select("country").rdd.map(r => r(0)).collect()

// COMMAND ----------

//create a list with the tuple: country_name and query result
var df_list = List[(Any, org.apache.spark.sql.DataFrame)]()
var i = 1
for(country_name <- country_list){
  println (i.toString + "- the country is: " + country_name)
  df_list = df_list :+ (country_name, country_name_count_df.where(col("country") === country_name ).select(col("term"), col("count")).sort(-col("count")))
  i= i+1
}

// COMMAND ----------

//US result
print(df_list(217)._1)
display(df_list(217)._2)

// COMMAND ----------

//Sweden result
print(df_list(197)._1)
display(df_list(197)._2)

// COMMAND ----------

// DBTITLE 1,Global
//select only the text, split it into words and remove the stopwords
val dataset_all = dataset.select(col("text"))
val dataset_all_words = dataset_all.withColumn("text", lower(col("text"))).select(split(col("text"), "[^a-z0-9#-]").alias("term"))
val dataset_all_words_new = remover.transform(dataset_all_words)

// COMMAND ----------

val wordcount_global = dataset_all_words_new
                      .select(explode(col("ok")).alias("term"))
                      //.where("term != ''")
                      .groupBy(col("term")).count()
                      .sort(-col("count"))

display(wordcount_global)

// COMMAND ----------

// DBTITLE 1,By lang
//select text and lang, dropping null values
val dataset_lang = dataset.select(col("text"),col("lang")).na.drop(Seq("lang"))

//create a list of languages to include in the query and filter the dataset accordingly
val list_lang = List("en", "es", "fr", "it")
val dataset_lang_correct = dataset_lang.filter(col("lang").isin(list_lang:_*))

//divide the text into words and remove the stopwords
val dataset_lang_words = dataset_lang_correct
                         .withColumn("text", lower(col("text")))
                         .select(col("lang"), split(col("text"), "[^a-z0-9#-]").alias("term"))

val dataset_lang_words_new = remover.transform(dataset_lang_words)

// COMMAND ----------

// DBTITLE 0,Lang --> group by and count 
//group the words by lang and count
val wordcount = dataset_lang_words_new
                .select(col("lang"), explode(col("ok")).alias("term"))
                //.where("term != ''")
                .groupBy(col("lang"), col("term")).count()
                .sort(-col("count"))

wordcount.show()

// COMMAND ----------

//create a list containg a tuple: lang and query result
var df_list_lang = List[(Any, org.apache.spark.sql.DataFrame)]()
for(l <- list_lang){
  println ("the language is: " + l)
  df_list_lang = df_list_lang :+ (l, wordcount.where(col("lang") === l ).select(col("term"), col("count")).sort(-col("count")))
}

// COMMAND ----------

//English
display(df_list_lang(0)._2)

// COMMAND ----------

//Spanish
display(df_list_lang(1)._2)

// COMMAND ----------

// DBTITLE 1,Sentiment analysis 
import com.databricks.spark.corenlp.functions._

val version = "3.9.1"
val baseUrl = s"https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp"
val model = s"stanford-corenlp-$version-models.jar" // 
val url = s"$baseUrl/$version/$model"
if (!sc.listJars().exists(jar => jar.contains(model))) {
  import scala.sys.process._
  // download model
  s"wget -N $url".!!
  // make model files available to driver
  s"jar xf $model".!!
  // add model to workers
  sc.addJar(model)
}

// COMMAND ----------

//use a portion of the data due to time constraints
dataset_small = dataset_small.select(col("text")).na.drop()
dataset_small = dataset_small.sample(0.01)

// COMMAND ----------

dataset_small.printSchema()

// COMMAND ----------

// DBTITLE 1,Start sentiment query
//sentiment: Measures the sentiment of an input sentence on a scale of 0 (strong negative) to 4 (strong positive).
val output = dataset_small
            .select(cleanxml('text).as('doc))
            .select(explode(ssplit('doc)).as('sen))
            .select('sen, split(lower(col("sen")), "[^a-z0-9#-]").alias("term"), sentiment('sen).as('sentiment))
            //.select('sen, tokenize('sen).as('words), ner('sen).as('nerTags), sentiment('sen).as('sentiment))
          
output.show()

// COMMAND ----------

//filter only the positive sentences: 3,4
val positive_words = output.select(col("term"), col("sentiment")).filter(col("sentiment") >= 3)
//remove the stopwords
val dataset_good_words_new = remover.transform(positive_words)

dataset_good_words_new.show()

// COMMAND ----------

//count the term occurences
val dataset_good_wordcount = dataset_good_words_new
                            .select(explode(col("ok")).alias("term"))
                            //.where("term != ''")
                            .groupBy(col("term")).count()
                            .sort(-col("count"))

display(dataset_good_wordcount)

// COMMAND ----------

//same but for the negative class
val negative_words = output.select(col("term"), col("sentiment")).filter(col("sentiment") <= 1)

val dataset_negative_words_new = remover.transform(negative_words)

// COMMAND ----------

val wordcount = dataset_negative_words_new.select(explode(col("ok")).alias("term")).where("term != ''").groupBy(col("term")).count().sort(-col("count"))

display(wordcount)

// COMMAND ----------

// DBTITLE 1,Correlation
//study the correaltion between length and retweet count
//cleaning and preprocessing
val length_retweet = dataset
                      .na.drop(Seq("retweet_count"))
                      .select(col("text"), col("retweet_count"))
                      .distinct()
                      .withColumn("tweet_len",length('text))
                      .filter(col("retweet_count").cast("int").isNotNull)
                      .select(col("tweet_len"), col("retweet_count").cast("int"))
                      .filter(col("tweet_len")<=280)
                      .filter(col("retweet_count")<500000)
                      .groupBy(col("tweet_len").as("tweet_lenght")).avg("retweet_count").as("avg_retweet_count")
length_retweet.show
//display(length_retweet)

// COMMAND ----------

display(length_retweet)

// COMMAND ----------


