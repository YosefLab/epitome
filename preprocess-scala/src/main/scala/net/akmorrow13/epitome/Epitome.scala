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
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor



object Epitome extends Serializable  {
  val commandName = "epitome"
  val commandDescription = "learning from chromatin"
  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    if (args.size < 1) {
      println("Incorrect number of arguments...Exiting now.")
    } else {
      val configfile = scala.io.Source.fromFile(args(0))
      val configtext = try configfile.mkString finally configfile.close()
      val yaml = new Yaml(new Constructor(classOf[EpitomeConf]))
      val appConfig = yaml.load(configtext).asInstanceOf[EpitomeConf]
      val conf = new SparkConf().setAppName("Epitome")
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      conf.remove("spark.jars")
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      val sc = new SparkContext(conf)
      run(sc, appConfig)
      sc.stop()
    }
  }

  def run(sc: SparkContext, conf: EpitomeConf) {

    // make sure configuration is valid
    EpitomeConf.validate(conf)

    // featurize reads and features
    val vectorize = new Vectorizer(sc, conf)

    val featuresAndLabels = vectorize.partitionAndFeaturize()

    // save results
    vectorize.saveValuesLocally(featuresAndLabels, conf.featurizedPath)

  }
}