val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
// Import Spark SQL data types
import org.apache.spark.sql._
// Import mllib recommendation data types
import org.apache.spark.mllib.recommendation.{ALS,
  MatrixFactorizationModel, Rating}

  
  val ratingText = sc.textFile("/home/ubuntu/spark-2.2.0-bin-hadoop2.7/ratings/ratings.dat")
  
  // Below code is to split the input rating file based on "::" delimiter
  def splittingRatingFile(str: String): Rating= {
      val fields = str.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}

// creating resilient distributed dataset out of the rating file
val ratingDistributedDataSet = ratingText.map(splittingRatingFile).cache()

// we are using spark sql which provides DataFrames which helps in creating distributed collection of data organised in columns
//The apache spark sqlcontext provides .toDF function to convert distributed dataset into DataFrames
val ratingsDF = ratingDistributedDataSet.toDF()

//we are saving the dataframes into temptable so that we can use it in sql commands
ratingsDF.registerTempTable("ratings")

//We have divided the dataset into two parts.
//One is the training data set and the other is test data set. 0.8 is the training data set and 0.2 is the testing dataset.
val splits = ratingDistributedDataSet.randomSplit(Array(0.8, 0.2), 0L)

val trainingDistributedDataSet = splits(0).cache()
val testDistributedDataSet = splits(1).cache()


// build a ALS user product matrix model with rank=20, iterations=10
val model = (new ALS().setRank(20).setIterations(10)
  .run(trainingDistributedDataSet))

print("\n")  
//recommendProducts is an inbuilt method which we are using to suggest the users from product id.
val topRecsForUser = model.recommendProducts(4086, 20)
print("\n")

//recommendUsers is an inbuilt method which we are using to suggest the products from the users id.
val topRecsForPrd =  model.recommendUsers(343,20)
  
  