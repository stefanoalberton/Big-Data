import com.univocity.parsers.annotations.Convert;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.*;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.tools.nsc.interactive.FreshRunReq;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import sun.rmi.runtime.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.function.BiFunction;


public class G21HW3 {

    public static void main(String[] args) throws IOException {
        //checking arguments
        if(args.length != 6)
            throw new IllegalArgumentException("USAGE: path_file number_of_cluster expected_sample_size_per_cluster");

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //SPARK SETUP
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("Homework3").setMaster("local");
        //CloudVeneto Context
        //SparkConf conf = new SparkConf(true).setAppName("Homework3").set("spark.locality.wait", "0s");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //INPUT READING
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        //Initialization of variables to take count of running time
        long start,end;
        start=System.currentTimeMillis();
        //Read the path to text file
        String path =  args[0].toString();
        //Read the initial number of clusters
        int kstart = Integer.parseInt(args[1]);
        //Read the number of values of k that the program will test
        int h = Integer.parseInt(args[2]);
        //Read the number of iterations of Lloyd's algorithm
        int iter = Integer.parseInt(args[3]);
        //Read the expected size of the sample used to approximate the silhouette coefficient
        int M = Integer.parseInt(args[4]);
        //Read the number of partitions of the RDDs containing the input points and their clustering
        int L = Integer.parseInt(args[5]);

        //Read input file and subdivide it in L random partitions
        JavaRDD<Vector> inputPoints = sc.textFile(path).repartition(L).map(x -> strToVector(x)).cache();
        //Materialization of the RDD
        Long numberOfElements = inputPoints.count();
        end = System.currentTimeMillis();

        System.out.println("Time for input reading = "+(end-start)+" ms \n");
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //COMPUTE THE NUMBER OF CLUSTER
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long startClustering, endClustering, startSilhouette,endSilhouette;
        for(int i = kstart; i < kstart+h; i++  ){

            //Computes a clustering
            startClustering=System.currentTimeMillis();
            KMeansModel clusters  = KMeans.train(inputPoints.rdd(), i, iter);
            JavaPairRDD<Vector,Integer>  currentClustering = inputPoints.mapToPair(point -> {
                return new Tuple2<Vector, Integer>(point,clusters.predict(point));
            });
            endClustering=System.currentTimeMillis();

            int t = (int)( M / i);

            //Computes the approximate silhouette over the sample
            startSilhouette=System.currentTimeMillis();
            double silhouetteCoeff = approxSilhouette(currentClustering, t, i, sc);
            endSilhouette=System.currentTimeMillis();

            System.out.println("Number of clusters k = "+ i);
            System.out.println("Silhouette coefficient = "+ silhouetteCoeff);
            System.out.println("Time for clustering = " +(endClustering-startClustering)+" ms" );
            System.out.println("Time for silhouette computation = " +(endSilhouette-startSilhouette)+" ms \n");

        }

    }

    public static double approxSilhouette(JavaPairRDD<Vector,Integer> fullClustering, int t, int nCluster, JavaSparkContext sc){
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //SAMPLING
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaRDD<Integer> clustersSize  = fullClustering.values().cache();
        Map<Integer,Long> tempResult = clustersSize.countByValue();
        ArrayList<Tuple2<Integer,Long>> kSize = new ArrayList<>();
        for(int i=0;i<nCluster;i++)
            kSize.add(new Tuple2<Integer,Long>(0,(long)0)); //initialize kSize
        for(Map.Entry<Integer,Long> e : tempResult.entrySet())
            kSize.set(e.getKey(),new Tuple2<>(e.getKey(), e.getValue()));
        Broadcast<List<Tuple2<Integer,Long>>> sharedClusterSize = sc.broadcast(kSize);

        Map<Integer,Double> fractions = new HashMap<>();
        for(int i=0;i<nCluster;i++)
            fractions.put(i,Math.min((double)t/kSize.get(i)._2(),1));
        List<Tuple2<Vector,Integer>> clustSamp = fullClustering.mapToPair(point -> point.swap()).sampleByKey(false,fractions).mapToPair(point -> point.swap()).collect();
        Broadcast<List<Tuple2<Vector,Integer>>> clusteringSample = sc.broadcast(clustSamp);

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //APPROXSILHFULL
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        double silhouette = fullClustering.mapToPair(point -> {
            int cluster = point._2();
            double[] sum = new double[nCluster];
            double ap,sp;
            double bp = Double.POSITIVE_INFINITY;
            for(int i=0;i<clusteringSample.getValue().size();i++)
                sum[clusteringSample.getValue().get(i)._2()] += Vectors.sqdist(point._1(), clusteringSample.getValue().get(i)._1());
            ap=((double) 1/Math.min(t,sharedClusterSize.getValue().get(cluster)._2()))*sum[cluster];

            for(int i=0;i<nCluster;i++)
                if(cluster!=i)
                    bp = Math.min(bp,((double)1/Math.min(t,sharedClusterSize.getValue().get(i)._2()))*sum[i]);

            sp = (bp-ap)/(Math.max(ap,bp));
            return new Tuple2<Vector,Double>(point._1(),sp);
        }).values().reduce((point1,point2)-> point1+point2);

        return silhouette/fullClustering.count();
        //return silhouette;
    }
    public static Vector strToVector (String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length-1];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        Vector point = Vectors.dense(data);
        return point;
    }
}
