package net.akmorrow13.epitome.util

import org.apache.spark.SparkConf
import org.bdgenomics.formats.avro.Feature
import org.bdgenomics.formats.avro.AlignmentRecord
import org.apache.spark.rdd.RDD
import org.bdgenomics.adam.models.ReferenceRegion
import breeze.linalg.DenseVector


object Vectorizer {

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

        reads.foreach(cut => {
          positions((cut._1.start - region.start).toInt) += cut._2
        })

        // positions should be size of window
        (window._1, positions)
      })

    featurized
  }


}
