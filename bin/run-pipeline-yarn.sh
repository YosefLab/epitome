#!/bin/bash

# Figure out where we are.
FWDIR="$(cd `dirname $0`; pwd)"

CLASS=$1
shift

# Figure out where the Scala framework is installed
FWDIR="$(cd `dirname $0`/..; pwd)"

if [[ "$RUN_LOCAL" ]]; then
    echo "RUN_LOCAL is set, running pipeline locally"
	$FWDIR/bin/run-main.sh $CLASS "$@"
	MASTER="local[4]"
	exit 0
else
	MASTER="yarn"
fi

if [ -z "$OMP_NUM_THREADS" ]; then
    export OMP_NUM_THREADS=1 added as we were nondeterministically running into an openblas race condition
fi


echo "automatically setting OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Figure out where epitome is installed
SCRIPT_DIR="$(cd `dirname $0`/..; pwd)"

# Get list of required jars for mango
EPITOME_JARS=$("$SCRIPT_DIR"/bin/compute-epitome-jars.sh)

# Split out the CLI jar, since it will be passed to Spark as the "primary resource".
EPITOME_PREPROCESS_JAR=${EPITOME_JARS##*,}
EPITOME_JARS=$(echo "$EPITOME_JARS" | rev | cut -d',' -f2- | rev)
echo $EPITOME_JARS

# Find spark-submit script
if [ -z "$SPARK_HOME" ]; then
  SPARK_SUBMIT=$(which spark-submit)
else
  SPARK_SUBMIT="$SPARK_HOME"/bin/spark-submit
fi
if [ -z "$SPARK_SUBMIT" ]; then
  echo "SPARK_HOME not set and spark-submit not on PATH; Aborting."
  exit 1
fi
echo "Using SPARK_SUBMIT=$SPARK_SUBMIT"

echo "RUNNING ON THE CLUSTER"

echo CORES $SPARK_EXECUTOR_CORES
echo NUM EXECUTORS $SPARK_NUM_EXECUTORS


"$SPARK_SUBMIT" \
  --master $MASTER \
  --class $CLASS \
  --driver-class-path $EPITOME_JARS:$EPITOME_PREPROCESS_JAR:$HOME/hadoop/conf \
  --conf spark.serializer=org.apache.spark.serializer.JavaSerializer \
  --jars $EPITOME_JARS \
  $EPITOME_PREPROCESS_JAR \
  "$@"

#
## Set some commonly used config flags on the cluster
#"$SPARK_SUBMIT" \
#  --class $CLASS \
#  --num-executors $SPARK_NUM_EXECUTORS \
#  --driver-memory 60g \
#  --executor-cores $SPARK_EXECUTOR_CORES \
#  --driver-class-path $JARFILE:$ASSEMBLYJAR:$HOME/hadoop/conf \
#  --conf spark.executor.extraLibraryPath=/opt/amp/openblas/lib:$FWDIR/lib \
#  --conf spark.executor.extraClassPath=$JARFILE:$ASSEMBLYJAR:$HOME/hadoop/conf \
#  --conf spark.driver.extraClassPath=$JARFILE:$ASSEMBLYJAR:$HOME/hadoop/conf \
#  --conf spark.serializer=org.apache.spark.serializer.JavaSerializer \
#  --jars $ASSEMBLYJAR \
#  $JARFILE \
#  "$@"
