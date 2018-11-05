

import java.sql.Date;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataType;

public class App {

	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		
		// Create a SparkSession
		SparkSession spark = SparkSession.builder().appName("KMeansCluster").master("local").getOrCreate();

		// Loads data
		Dataset<Row> rawDataset = spark.read().option("header", "true").csv("Data/sample100mb.csv")
			.toDF("user_id","timestamp","song_id","date_col");
		//rawDataset.show();

			
		// Ignore rows having null values
		Dataset<Row> datasetClean = rawDataset.na().drop();
		//datasetClean.show();
		datasetClean.printSchema();
		
		datasetClean.withColumn("modified",functions.date_format(functions.to_date(datasetClean.col("date_col"), "yyyymmdd"), "yyyy-MM-dd")).show();
		
		//datasetClean.withColumn("test123", functions.unix_timestamp(functions.to_date(datasetClean.col("date_col"),"yyyy-mm-dd"),"yyyy-mm-dd").cast("timestamp")).show();
		
		/*datasetClean = datasetClean.withColumn("timestamp", functions.to_date(datasetClean.col("date_col")));
		datasetClean.printSchema();
		datasetClean.show();*/
		
		
		/*datasetClean = datasetClean.withColumn("date_col", functions.to_date(datasetClean.col("date_col")));
		datasetClean.printSchema();
		datasetClean.show();
		*/
		/*datasetClean = datasetClean.withColumn("date_col", functions.date_format(datasetClean.col("date_col"), "yyyy-mm-dd").cast("date"));
		datasetClean.printSchema();
		datasetClean.show();*/
		
		/*datasetClean = datasetClean.withColumn("date_test", functions.date_format(datasetClean.col("date_test"), "yyyy-mm-dd").cast("date"));
		datasetClean.printSchema();
		datasetClean.show();*/
		
		/*Dataset<Row> dsDateConv = datasetClean.withColumn("date1",functions.current_date().cast("date"));
		dsDateConv.show();*/
		
		/*Dataset<Row> datasetRetail = datasetClean.withColumn("DaysBefore", functions.datediff(
			functions.current_date(),datasetClean.col("date_col")));
		datasetRetail.show();*/

		/*// Adding Total Price Column
		// Total Price = Quantity * UnitPrice
		Column totalPrice = datasetClean.col("Quantity").multiply(datasetClean.col("UnitPrice"));
		Dataset<Row> datasetTotPrice = datasetClean.withColumn("Total_Price", totalPrice);
		datasetTotPrice.show();
		
		// Convert all timestamps into dd/MM/yy HH:mm format 01/12/10 8:26:00 AM
		// Calculate RFM attributes : Recency, Frequency and Monetary Values
		Dataset<Row> datasetRetail = datasetTotPrice.withColumn("DaysBefore", functions.datediff(
				functions.current_timestamp(),
				functions.unix_timestamp(datasetTotPrice.col("InvoiceDate"), "MM/dd/yyyy HH:mm").cast("timestamp")));
		datasetRetail.show();

		// Recency
		Dataset<Row> datasetRecency = datasetRetail.groupBy("CustomerID")
				.agg(functions.min("DaysBefore").alias("Recency"));
		datasetRecency.show();
		
		// Frequency
		Dataset<Row> datasetFreq = datasetRetail.groupBy("CustomerID", "InvoiceNo").count().groupBy("CustomerID")
				.agg(functions.count("*").alias("Frequency"));
		datasetFreq.show();
		
		// Monetary
		Dataset<Row> datasetMon = datasetRetail.groupBy("CustomerID")
				.agg(functions.round(functions.sum("Total_Price"), 2).alias("Monetary"));
		datasetMon.show();

		Dataset<Row> datasetMf = datasetMon
				.join(datasetFreq, datasetMon.col("CustomerID").equalTo(datasetFreq.col("CustomerID")), "inner")
				.drop(datasetFreq.col("CustomerID"));
		datasetMf.show();	
		
		Dataset<Row> datasetRfm1 = datasetMf
				.join(datasetRecency, datasetRecency.col("CustomerID").equalTo(datasetMf.col("CustomerID")), "inner")
				.drop(datasetFreq.col("CustomerID"));
		datasetRfm1.show();
		
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(new String[] {"Monetary", "Frequency", "Recency"}).setOutputCol("features");
				
		Dataset<Row> datasetRfm = assembler.transform(datasetRfm1);
		datasetRfm.show();
		 
		// Trains a k-means model
		KMeans kmeans = new KMeans().setK(3);
		KMeansModel model = kmeans.fit(datasetRfm);
		
		// Make predictions
		Dataset<Row> predictions = model.transform(datasetRfm);
		predictions.show(200);
		
		// Evaluate clustering by computing Silhouette score
		ClusteringEvaluator evaluator = new ClusteringEvaluator();

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