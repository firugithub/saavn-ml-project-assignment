//package com.upgrad.saavn;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

public class App {

	public static void main(String[] args) {
	    	System.setProperty("hadoop.home.dir", "C:\\winutils");
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		
		// Create a SparkSession
		SparkSession spark = SparkSession.builder().appName("KMeansCluster").master("local").getOrCreate();

		// Loads data
		Dataset<Row> rawDataset = spark.read().option("header", "true").csv("Data/")
			.toDF("user_id","timestamp","song_id","date_col");
		//rawDataset.show();

		
		//D:\\cloudera_share\\saavn_ml_project_files\\newmetadata
		//C:\\VirtualD\\saavn_ml_project_files\\newmetadata
		//D:\\cloudera_share\\saavn_ml_project_files\\notification_clicks
		//D:\\cloudera_share\\saavn_ml_project_files\\notification_actor
		
		//Load newmetadata -- artist dataset
		Dataset<Row> artistDataset = spark.read().option("header", "true").csv("D:\\\\cloudera_share\\\\saavn_ml_project_files\\\\newmetadata")
			.toDF("song_id","artist_id");
		artistDataset.show();
		
		//Load new notification -- clicks dataset
		Dataset<Row> notificationClicks = spark.read().option("header", "true").csv("D:\\\\cloudera_share\\\\saavn_ml_project_files\\\\notification_clicks")
			.toDF("notification_id","user_id","date_col");
		notificationClicks.show();
		
		//Load new notification -- artist dataset
		Dataset<Row> notificationArtist = spark.read().option("header", "true").csv("D:\\\\cloudera_share\\\\saavn_ml_project_files\\\\notification_actor")
			.toDF("notification_id","artist_id");
		notificationArtist.show();
		
		// Ignore rows having null values
		Dataset<Row> datasetClean = rawDataset.na().drop();
		//datasetClean.show();
		datasetClean.printSchema();
		
		datasetClean = datasetClean.withColumn("modified_date",functions.date_format(functions.to_date(datasetClean.col("date_col"), "yyyymmdd"), "yyyy-MM-dd"));
		datasetClean.show();
		
		Dataset<Row> datasetdbf = datasetClean.withColumn("last_lisen", functions.datediff(
			functions.current_date(),
			datasetClean.col("modified_date")));
		datasetdbf.show();
		
		
		
		
		// Recency
		Dataset<Row> datasetRecency = datasetdbf.groupBy("user_id")
				.agg(functions.min("last_lisen").alias("recency"));
		datasetRecency.show();
		
		// Frequency
		Dataset<Row> datasetFreq = datasetdbf.groupBy("user_id", "date_col").count().groupBy("user_id")
				.agg(functions.count("*").alias("frequency"));
		datasetFreq.show();
		
		Dataset<Row> datasetMf = datasetRecency
				.join(datasetFreq, datasetRecency.col("user_id").equalTo(datasetFreq.col("user_id")), "inner")
				.drop(datasetFreq.col("user_id"));
		datasetMf.show();	
		
		Dataset<Row> datasetwithsong = datasetMf
				.join(datasetdbf, datasetMf.col("user_id").equalTo(datasetdbf.col("user_id")), "inner")
				.drop(datasetdbf.col("user_id"));
		datasetwithsong.show();	
		
		StringIndexer indexer = new StringIndexer()
				  .setInputCol("song_id")
				  .setOutputCol("song_id_indexed");

		Dataset<Row> datasetwithsongindexed = indexer.fit(datasetwithsong).transform(datasetwithsong);
		datasetwithsongindexed.show();
				
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(new String[] {"recency", "frequency","last_lisen"}).setOutputCol("features");
				
		Dataset<Row> datasetRfm = assembler.transform(datasetwithsong);
		datasetRfm.show();
		 
		// Trains a k-means model
		KMeans kmeans = new KMeans().setK(30);
		KMeansModel model = kmeans.fit(datasetRfm);
		
		// Make predictions
		Dataset<Row> predictions = model.transform(datasetRfm);
		predictions.show(200);
		
		
		Dataset<Row> predictionswithArtist = predictions
				.join(artistDataset, predictions.col("song_id").equalTo(artistDataset.col("song_id")), "inner")
				.drop(artistDataset.col("song_id"));
		predictionswithArtist.show();
		
		/*Dataset<Row> distinctValuesDF = predictionswithArtist.select(predictionswithArtist.col("prediction")).distinct();
		distinctValuesDF.show();*/
		
		/*predictionswithArtist.groupBy(predictionswithArtist.col("prediction")).count();
		
		Dataset<Row> predictionswithArtistGrp = predictionswithArtist.groupBy("prediction")
			.agg(functions.count("artist_id").alias("count"));
		predictionswithArtistGrp.show();
*/
		
		Dataset<Row> datasetcnt = predictionswithArtist.groupBy("prediction","artist_id").count().orderBy(functions.asc("prediction"),functions.desc("count"));
		datasetcnt.show();
		
		WindowSpec  window = Window.partitionBy("prediction").orderBy(functions.desc("count"));

		Dataset<Row> datasetcntRnk =  datasetcnt.withColumn("rn", functions.row_number().over(window));
		datasetcntRnk.show();
		datasetcntRnk.printSchema();
		
		Dataset<Row> datasetcntRnkfilterd =  datasetcntRnk.filter(x -> x.getInt(3) == 1);
		datasetcntRnkfilterd.show();
		
		// Evaluate clustering by computing Silhouette score
		/*ClusteringEvaluator evaluator = new ClusteringEvaluator();
		double silhouette = evaluator.evaluate(predictions);
		System.out.println("Silhouette with squared euclidean distance = " + silhouette);
		// Shows the result
		Vector[] centers = model.clusterCenters();
		System.out.println("Cluster Centers: ");
		for (Vector center : centers) {	
			System.out.println(center);
		}*/
		spark.stop();
	
	}
}	
