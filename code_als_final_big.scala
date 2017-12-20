val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
// Import Spark SQL data types
import org.apache.spark.sql._
// Import mllib recommendation data types
import org.apache.spark.mllib.recommendation.{ALS,
  MatrixFactorizationModel, Rating}
//Import for io
import scala.io.Source
import sys.process._
import java.net.URL
import java.io.File


  
   val ratingText = sc.textFile("/home/ubuntu/spark-2.2.0-bin-hadoop2.7/ratings/ratings.txt")
  
  ratingText.first()
  
  // changes for big data set.
  def parseRating(str: String): Rating= {
      val fields = str.split("::")
      Rating(fields(1).toInt, fields(0).toInt, fields(2).toDouble)
}


val ratingsRDD = ratingText.map(parseRating).cache()

println("Total number of ratings: " + ratingsRDD.count())

val ratingsDF = ratingsRDD.toDF()

ratingsDF.registerTempTable("ratings")

ratingsDF.printSchema()

val splits = ratingsRDD.randomSplit(Array(0.8, 0.2), 0L)

val trainingRatingsRDD = splits(0).cache()
val testRatingsRDD = splits(1).cache()

val numTraining = trainingRatingsRDD.count()
val numTest = testRatingsRDD.count()
println(s"Training: $numTraining, test: $numTest.")

// build a ALS user product matrix model with rank=20, iterations=10
val model = (new ALS().setRank(20).setIterations(10)
  .run(trainingRatingsRDD))
  
  
  val topRecsForUser = model.recommendProducts(4086, 20)
  
  val topRecsForPrd =  model.recommendUsers(343,20)
  
  