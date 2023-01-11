import com.univocity.parsers.annotations.Convert;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.tools.nsc.interactive.FreshRunReq;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.function.BiFunction;


public class G21HW1 {
    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: num_partitions, <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //System.setProperty("hadoop.home.dir", "C:\\Hadoop\\");
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: num_partitions num_products file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("HW1").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read number of products to print
        int T = Integer.parseInt(args[1]);

        // Read input file and subdivide it into K random partitions
        // containing reviews (ProductID,UserID,Rating,Timestamp)
        JavaRDD<String> RawData  = sc.textFile(args[2]).repartition(K).cache();
        JavaPairRDD<String, Float> normalizedRatings;
        JavaPairRDD<String, Float> maxNormRatings;
        Random randomGenerator = new Random();

        normalizedRatings = RawData
                                .mapToPair((review)->{ //MAP PHASE (R1)
                                    String[] tokens = review.split(",");
                                    String productID = tokens[0];
                                    String userID = tokens[1];
                                    Float rating = Float.parseFloat(tokens[2]);
                                    Tuple2<String, Float> value = new Tuple2<String,Float>(productID,rating);
                                    Tuple2<String, Tuple2> pair = new Tuple2<>(userID,value);
                                    return pair;
                                }).groupByKey() //REDUCE PHASE (R1)
                                .flatMapToPair((reviewByUSer) ->{ //MAP PHASE (R2)
                                    ArrayList<String> products = new ArrayList<>();
                                    ArrayList<Float>  ratings = new ArrayList<>();
                                    ArrayList<Tuple2<String,Float>> outputpairs = new ArrayList<>();
                                    // LOAD product and rating lists in such a way that product[i] has rating [i]
                                    for(Tuple2<String,Float> element : reviewByUSer._2())
                                    {
                                        products.add(element._1()); 
                                        ratings.add(element._2()); 
                                    }
                                    // COMPUTE average rating for user
                                    float sumratings = 0;
                                    float avgrating;
                                    for(float rating : ratings){
                                        sumratings += rating;
                                    }
                                    avgrating = sumratings/ratings.size();
                                    // NORMALIZE rating for each product rated by the user
                                    for (int i=0; i < products.size(); i++) {
                                        String product = products.get(i);
                                        Float normrating = ratings.get(i) - avgrating;
                                        Tuple2<String,Float> pair = new Tuple2<>(product,normrating);
                                        outputpairs.add(pair);
                                    }
                                    return outputpairs.iterator();
                                });//NO REDUCE PHASE (R2)

        maxNormRatings = normalizedRatings.
                               mapPartitionsToPair((normreview) -> {    // MAP PHASE (R1)
                                    HashMap<String, Float> reviewByProduct = new HashMap<>();
                                    while (normreview.hasNext()){
                                        Tuple2<String, Float> review = normreview.next();
                                        // UPDATE rating of product if it is higher than before
                                        if (review._2() > reviewByProduct.getOrDefault(review._1(), 0F) ){
                                            reviewByProduct.put(review._1(), review._2());
                                        }
                                    }
                                    ArrayList<Tuple2<String, Float>> pairs = new ArrayList<>();
                                    for (Map.Entry<String, Float> e : reviewByProduct.entrySet()) {
                                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                                    }
                                    return pairs.iterator();
                               }).groupBy((normreviewpair) -> randomGenerator.nextInt(K)) // REDUCE PHASE (R1)
                               .flatMapToPair((element) -> {  // MAP PHASE (R2)
                                        HashMap<String, Float> reviewByProduct = new HashMap<>();
                                        for (Tuple2<String, Float> c : element._2()) {
                                            // UPDATE rating of product if it is higher than before
                                            if (c._2() > reviewByProduct.getOrDefault(c._1(), 0F) ){
                                                reviewByProduct.put(c._1(), c._2());
                                            }
                                        }
                                        ArrayList<Tuple2<String, Float>> pairs = new ArrayList<>();
                                        for (Map.Entry<String, Float> e : reviewByProduct.entrySet()) {
                                            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                                        }
                                        return pairs.iterator();
                                    })
                                .reduceByKey((x ,y) -> {  // REDUCE PHASE (R2)
                                    if(x > y){
                                        return x;
                                    }else {
                                        return y;
                                    }
                               });

        List<Tuple2<String,Float>> print =  maxNormRatings.mapToPair(x->x.swap()).sortByKey(false).mapToPair(x->x.swap()).take(T); // Swap pID(key) with MNR(value), apply sortByKey and swap again
        for(int i=0;i<T;i++)
            System.out.println("Product "+print.get(i)._1()+" maxNormRating  "+print.get(i)._2());


    }
}
