/**
 * Copyright 2016 Alyssa Morrow
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

import org.bdgenomics.utils.misc.SparkFunSuite
import org.scalatest.FunSuiteLike

trait EpitomeFunSuite extends SparkFunSuite with FunSuiteLike {
  override val appName: String = "epitome"
  override val properties: Map[String, String] = Map(("spark.serializer", "org.apache.spark.serializer.KryoSerializer"),
    ("spark.kryoserializer.buffer", "4"),
    ("spark.kryo.referenceTracking", "true"))

  // fetches resources
  def resourcePath(path: String) = ClassLoader.getSystemClassLoader.getResource(path).getFile
}