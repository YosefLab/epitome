/**
 * Copyright 2017 Alyssa Morrow
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.akmorrow13.epitome

import net.akmorrow13.epitome.util.Vectorizer
import org.apache.log4j.{Level, Logger}
import org.apache.parquet.filter2.dsl.Dsl.{BinaryColumn, _}
import org.apache.spark.{SparkConf, SparkContext}
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.utils.cli._
import org.bdgenomics.utils.misc.Logging
import org.kohsuke.args4j.{ Argument, Option => Args4jOption }
import org.apache.log4j.{Level, Logger}


class EpitomeArgs extends Args4jBase with ParquetArgs {
  @Argument(required = true, metaVar = "-reads", usage = "ATAC/DNase-seq file", index = 0)
  var readsPath: String = null

  @Argument(required = true, metaVar = "-features", usage = "List of Chip-seq files, separated by commas (,)", index = 1)
  var featurePaths: String = null

  @Argument(required = true, metaVar = "-featurePathLabels", usage = "A list of names for Chip-seq files, separated by commas (,)", index = 2)
  var featurePathLabels: String = null

  @Argument(required = true, metaVar = "-reference", usage = "TwoBit file stored locally (not on hdfs)", index = 3)
  var referencePath: String = null

  @Argument(required = true, metaVar = "-output", usage = "Local filepath to save results to", index=4)
  var featurizedPath: String = null

  @Args4jOption(required = false, name = "-parititons", usage = "Number of partitions for reads and features")
  var partitions: Int = 0

  @Args4jOption(required = false, name = "-windowSize", usage = "Number of base pairs to window over")
  var windowSize: Int = 1000

  def validate(): Unit = {
    require(featurePaths.split(',').length == featurePathLabels.split(',').length,
    "feature paths != feature path labels lengtt")
  }

  def getFeaturePathsArray(): Array[String] = featurePaths.split(',')

  def getFeaturePathLabelsArray(): Array[String] = featurePathLabels.split(',')

}

object Epitome extends BDGCommandCompanion {
  override val commandName = "epitomepreprocess"
  override val commandDescription = "Convert ADAM nucleotide contig fragments to FASTA files"

  override def apply(cmdLine: Array[String]): Epitome =
    new Epitome(Args4j[EpitomeArgs](cmdLine))
}

class Epitome(val args: EpitomeArgs) extends BDGSparkCommand[EpitomeArgs] with Logging {
  override val companion = Epitome

  override def run(sc: SparkContext) {

    // turn off info logs
    sc.setLogLevel("ERROR")
    
    // make sure configuration is valid
    args.validate()

    // featurize reads and features
    val vectorize = new Vectorizer(sc, args)

    val featuresAndLabels = vectorize.partitionAndFeaturize()

    // save results
    vectorize.saveValuesToHdfs(featuresAndLabels, args.featurizedPath)

  }
}
