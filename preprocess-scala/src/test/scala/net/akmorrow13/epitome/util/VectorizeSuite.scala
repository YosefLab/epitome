package net.akmorrow13.epitome.util

import com.google.common.io.Files

import breeze.linalg.DenseVector
import net.akmorrow13.epitome.{EpitomeConf, EpitomeFunSuite}
import org.apache.spark.rdd.RDD
import org.bdgenomics.adam.models.{SequenceRecord, SequenceDictionary, ReferenceRegion, Coverage}
import org.bdgenomics.adam.rdd.RightOuterShuffleRegionJoinAndGroupByLeft
import org.bdgenomics.formats.avro.Feature

import scala.io.Source

class VectorizerSuite extends EpitomeFunSuite {

  val readsPath = resourcePath("small.sam")
  val featurePath1 = resourcePath("features.bed")
  val featurePath2 = resourcePath("features_2.bed")

  sparkTest("joins reads and features") {

    // setup configuration
    val conf = new EpitomeConf()
    conf.setReadsPath(readsPath)
    conf.setFeaturePaths(s"${featurePath1},${featurePath2}")
    conf.setFeaturePathLabels("TF1,TF2")
    conf.setPartitions(Some(1))

    val vectorizer = Vectorizer(sc, conf)

    val featurized: RDD[(DenseVector[Int], DenseVector[Int])] =
      vectorizer.partitionAndFeaturize()

    val positives = featurized.filter(r => r._1.sum == 2)

    // should have negative examples
    assert(featurized.count > 2)
    assert(positives.count == 1)

    assert(positives.first._2.length == conf.getWindowSize)
    assert(positives.first._2.findAll(x => x > 0).length == 2)

  }

  sparkTest("saves values locally") {
    val filepath = new java.io.File(Files.createTempDir(), "tempOutput")

    val features: RDD[DenseVector[Int]] = sc.parallelize(Array(DenseVector(1,2), DenseVector(1,3)))
    val labels: RDD[DenseVector[Int]]  = sc.parallelize(Array(DenseVector(1,2,0,0,2), DenseVector(1,2,0,0,1)))

    val featuresAndLabels = features.zip(labels)

    // setup configuration
    val conf = new EpitomeConf()
    conf.setReadsPath(readsPath)
    conf.setFeaturePaths(featurePath1)
    conf.setPartitions(Some(1))
    conf.setFeaturePathLabels("TF1")

    val vectorizer = new Vectorizer(sc, conf)

    vectorizer.saveValuesLocally(featuresAndLabels, filepath.toString)

    val header="#TF1"

    // read data back in
    val labelLines = Source.fromFile(filepath + ".labels").getLines.toArray

    assert(labelLines.length == 3)
    assert(labelLines(0) == header)
    assert(labelLines(1) == "1,2")

    val featureLines = Source.fromFile(filepath + ".features").getLines.toArray

    assert(featureLines.length == 3)
    assert(featureLines(0) == header)
    assert(featureLines(1) == "1,2,0,0,2")
  }

}