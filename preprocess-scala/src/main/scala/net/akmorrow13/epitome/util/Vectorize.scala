package net.akmorrow13.epitome.util

import org.apache.spark.SparkConf
import org.bdgenomics.formats.avro.Feature
import org.bdgenomics.formats.avro.AlignmentRecord
import org.apache.spark.rdd.RDD
import org.bdgenomics.adam.models.ReferenceRegion
import breeze.linalg.DenseVector
import org.bdgenomics.adam.rdd.read.AlignmentRecordRDD
import org.bdgenomics.adam.rdd.feature.FeatureRDD
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.adam.rdd.GenomicRDD
import org.bdgenomics.adam.rdd.InnerShuffleRegionJoinAndGroupByLeft

object Vectorizer {
  
  // edit this as needed
  val windowSize = 1000

  def expandFeaturesAndJoinReads(readsFile: String, featuresFile: String): RDD[(Feature, Iterable[AlignmentRecord])] = {

    val reads = sc.loadAlignments(readsFile)
    // expand the features to length windowSize
    val features = sc.loadFeatures(featuresFile)
      .transform(_.map(f => {
        val length = f.getEnd - f.getStart + 1
        val startAdd = (windowSize - length)/2
        val endAdd = startAdd + 1
        f.setStart(f.getStart - startAdd)
        f.setEnd(f.getEnd + endAdd)
        f}))

    val keyedReads = reads.rdd.keyBy(read => ReferenceRegion.unstranded(read))
    val keyedFeatures = features.rdd.keyBy(read => ReferenceRegion.unstranded(read))

    // InnerShuffleRegionJoinAndGroupByLeft(keyedFeatures, keyedReads).compute()

    // use this line instead of the above when getting an error for unequal number of partitions
    InnerShuffleRegionJoinAndGroupByLeft(keyedFeatures.repartition(1), keyedReads.repartition(1)).compute()

  }


  def Vectorize(featurizedCuts: RDD[(Feature, Iterable[AlignmentRecord])]): RDD[(Feature, DenseVector[Int])] = {

    // get window size
    val windowSize = ReferenceRegion.unstranded(featurizedCuts.first._1).length().toInt

    val featurized: RDD[(Feature, DenseVector[Int])] =
      featurizedCuts.map(window => {
        val region = ReferenceRegion.unstranded(window._1)

        // TODO: positive and negative strands
        val positions = DenseVector.zeros[Int](windowSize)

        val reads: Map[ReferenceRegion, Int] = window._2
          .flatMap(r => Iterable(ReferenceRegion(r.contigName, r.start, r.start + 1),
            ReferenceRegion(r.contigName, r.end, r.end+1)))
          .map(r => (r, 1))
          .groupBy(_._1)
          .map(r => (r._1, r._2.map(_._2).toArray.sum))

        reads.filter(cut => {
          val index = (cut._1.start - region.start).toInt
          index >= 0 && index < windowSize
        }).foreach(cut => {
          positions((cut._1.start - region.start).toInt) += cut._2
        })

        // positions should be size of window
        (window._1, positions)
      })

    featurized
  }


}
