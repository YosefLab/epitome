package net.akmorrow13.epitome.util

import com.google.common.io.Files

import breeze.linalg.DenseVector
import net.akmorrow13.epitome.{EpitomeArgs, EpitomeFunSuite}
import org.apache.spark.rdd.RDD
import org.bdgenomics.adam.models.ReferenceRegion

import scala.io.Source

class VectorizerSuite extends EpitomeFunSuite {

  val readsPath = resourcePath("small.sam")
  val featurePath1 = resourcePath("features.bed")
  val featurePath2 = resourcePath("features_2.bed")

  sparkTest("joins reads and features") {

    // setup configuration
    val conf = new EpitomeArgs()
    conf.readsPath = readsPath
    conf.featurePaths = s"${featurePath1},${featurePath2}"
    conf.featurePathLabels = "TF1,TF2"
    conf.partitions = 1

    val vectorizer = Vectorizer(sc, conf)

    val featurized =
      vectorizer.partitionAndFeaturize()

    val positives = featurized.filter(r => r.labels.sum == 2)

    // should have negative examples
    assert(featurized.count > 2)
    assert(positives.count == 1)

    assert(positives.first.atacCounts.length == conf.windowSize)
    assert(positives.first.atacCounts.findAll(x => x > 0).length == 2)

  }

  sparkTest("saves values locally") {
    val filepath = new java.io.File(Files.createTempDir(), "tempOutput")

    val item1 = ATACandSequenceFeature(DenseVector(1,2), DenseVector(1,2,0,0,2),  ReferenceRegion("chr1", 1, 100))
    val item2 = ATACandSequenceFeature(DenseVector(1,3), DenseVector(1,2,0,0,1), ReferenceRegion("chr1", 1, 100))

    val featuresAndLabels = sc.parallelize(Seq(item1, item2))

    // setup configuration
    val conf = new EpitomeArgs()
    conf.readsPath = readsPath
    conf.featurePaths = featurePath1
    conf.partitions = 1
    conf.featurePathLabels = "TF1"

    val vectorizer = new Vectorizer(sc, conf)

    vectorizer.saveValuesLocally(featuresAndLabels, filepath.toString)

    val header="#TF1"

    // read data back in
    val labelLines = Source.fromFile(filepath).getLines.toArray

    assert(labelLines.length == 3)
    assert(labelLines(0) == header)
    assert(labelLines(1) == "1,2;1,2,0,0,2")
  }

}