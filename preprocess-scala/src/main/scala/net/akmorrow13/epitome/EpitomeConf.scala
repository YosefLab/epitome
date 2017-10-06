package net.akmorrow13.epitome

import scala.reflect.{BeanProperty, ClassTag}

class EpitomeConf extends Serializable {

  // ATAC-seq or DNASE-seq path
  @BeanProperty var readsPath: String = ""

  // Array of feature (binding) paths. bed, narrowpeak or adam format
  @BeanProperty var featurePaths: String = null
  @BeanProperty var featurePathsArray: Array[String] = Array.empty

  // Array of labels for feature paths
  @BeanProperty var featurePathLabels: String = null
  @BeanProperty var featurePathLabelsArray: Array[String] = Array.empty

  // Defines number of base pairs to featurize
  @BeanProperty var windowSize: Int = 1000

  // Option to partition features before joining reads and features
  @BeanProperty var partitions: Option[Int] = None

  // Path to save features to
  @BeanProperty var featurizedPath: String = ""

}


object EpitomeConf {

  // Validates configuration
  def validate(conf: EpitomeConf): Unit = {

    conf.setFeaturePathLabelsArray(conf.getFeaturePathLabels.split(","))
    conf.setFeaturePathsArray(conf.getFeaturePaths.split(","))

    require(conf.getFeaturePathsArray.length == conf.getFeaturePathLabelsArray.length,
    throw new Exception("For each Feature path, there must be a corresponding feature path label (ie. K562_EGR1.txt => EGR1)"))
  }

}

