import com.univocity.parsers.annotations.Convert;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vectors;
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


public class G21HW2 {

    public static void main(String[] args) throws IOException {
        //checking arguments
        if(args.length != 3)
            throw new IllegalArgumentException("USAGE: path_file number_of_cluster expected_sample_size_per_cluster");

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //SPARK SETUP
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("HW2").setMaster("local");;
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //INPUT READING
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //Read input file and subdivide it in 5 random partitions
        JavaPairRDD<Vector,Integer> fullClustering = sc.textFile(args[0]).repartition(8).mapToPair(x -> strToTuple(x)).cache();

        //Read number of cluster
        int nCluster = Integer.parseInt(args[1]);

        //Read number of sample
        int nSample = Integer.parseInt(args[2]);

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //COMPUTE CLUSTERS SIZE
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        JavaRDD<Integer> clusterSize  = fullClustering.values().cache();
        Map<Integer,Long> tempResult = clusterSize.countByValue();
        ArrayList<Tuple2<Integer,Long>> kSize = new ArrayList<>();
        for(int i=0;i<nCluster;i++)
            kSize.add(new Tuple2<Integer,Long>(0,(long)0)); //initialize kSize
        for(Map.Entry<Integer,Long> e : tempResult.entrySet())
            kSize.set(e.getKey(),new Tuple2<>(e.getKey(), e.getValue()));
        Broadcast<List<Tuple2<Integer,Long>>> sharedClusterSize = sc.broadcast(kSize); // SHARED VARIABLE

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //SAMPLING
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        Map<Integer,Double> fractions = new HashMap<>();
        for(int i=0;i<nCluster;i++)
            fractions.put(i,Math.min((double)nSample/ kSize.get(i)._2(),1));
        List<Tuple2<Vector,Integer>> clustSamp = fullClustering.mapToPair(point -> point.swap()).sampleByKey(false,fractions).mapToPair(point -> point.swap()).collect();

        Broadcast<List<Tuple2<Vector,Integer>>> clusteringSample = sc.broadcast(clustSamp);

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //APPROXSILHFULL
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        long start,end;
        start=System.currentTimeMillis();
        double silhouette = fullClustering.mapToPair(point -> {
            int cluster = point._2();
            double[] sum = new double[nCluster];
            double ap,sp;
            double bp = Double.POSITIVE_INFINITY;
            for(int i=0;i<clusteringSample.getValue().size();i++)
                sum[clusteringSample.getValue().get(i)._2()] += Vectors.sqdist(point._1(), clusteringSample.getValue().get(i)._1());
            ap=((double) 1/Math.min(nSample,sharedClusterSize.getValue().get(cluster)._2()))*sum[cluster];

            for(int i=0;i<nCluster;i++)
                if(cluster!=i)
                    bp = Math.min(bp,((double)1/Math.min(nSample,sharedClusterSize.getValue().get(i)._2()))*sum[i]);

            sp = (bp-ap)/(Math.max(ap,bp));
            return new Tuple2<Vector,Double>(point._1(),sp);
        }).values().reduce((point1,point2)-> point1+point2);
        double approxSilhFull = silhouette/fullClustering.count();
        end = System.currentTimeMillis();
        System.out.println("Value of approxSilhFull = "+approxSilhFull);
        System.out.println("Time to compute = "+(end-start)+" ms");

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //EXACTSILHFULL
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        start= System.currentTimeMillis();
        double exactSilhSample = 0;
        for(int index=0;index<clustSamp.size();index++){
            Tuple2<Vector,Integer> point = clustSamp.get(index);
            int cluster = point._2();
            double ap;
            double bp = Double.POSITIVE_INFINITY;
            double[] sum = new double[nCluster];
            for (int i = 0; i < clustSamp.size(); i++)
                sum[clustSamp.get(i)._2()]=sum[clustSamp.get(i)._2()]+Vectors.sqdist(point._1(),clustSamp.get(i)._1());
            ap=((double) 1/Math.min(nSample,kSize.get(cluster)._2()))*sum[cluster];
            for(int i=0;i<nCluster;i++)
                if(i!=cluster)
                    bp= Math.min(bp,((double)1/Math.min(nSample,kSize.get(i)._2()))*sum[i]);

            exactSilhSample=exactSilhSample+(bp-ap)/Math.max(ap,bp);
        }
        exactSilhSample=exactSilhSample/clustSamp.size();
        end= System.currentTimeMillis();
        System.out.println("Value of exactSilhSample = "+exactSilhSample);
        System.out.println("Time to compute = "+(end-start)+" ms");

    }


    public static Tuple2<Vector, Integer> strToTuple (String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length-1];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        Vector point = Vectors.dense(data);
        Integer cluster = Integer.valueOf(tokens[tokens.length-1]);
        Tuple2<Vector, Integer> pair = new Tuple2<>(point, cluster);
        return pair;
    }
}
