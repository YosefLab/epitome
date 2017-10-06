#!/bin/bash

# Figure out where we are.
FWDIR="$(cd `dirname $0`; pwd)"

CLASS=$1
shift
JARFILE=$1
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

ASSEMBLYJAR="$FWDIR"/target/scala-2.10/epitome-assembly-0.1-deps.jar

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

# Set some commonly used config flags on the cluster
"$SPARK_SUBMIT" \
  --master $MASTER \
  --class $CLASS \
  --num-executors $SPARK_NUM_EXECUTORS \
  --driver-memory 60g \
  --executor-cores $SPARK_EXECUTOR_CORES \
  --driver-class-path $JARFILE:$ASSEMBLYJAR:$HOME/hadoop/conf \
  --driver-library-path /opt/amp/gcc/lib64:/opt/amp/openblas/lib:$FWDIR/lib \
  --conf spark.executor.extraLibraryPath=/opt/amp/openblas/lib:$FWDIR/lib \
  --conf spark.executor.extraClassPath=$JARFILE:$ASSEMBLYJAR:$HOME/hadoop/conf \
  --conf spark.driver.extraClassPath=$JARFILE:$ASSEMBLYJAR:$HOME/hadoop/conf \
  --conf spark.serializer=org.apache.spark.serializer.JavaSerializer \
  --jars $ASSEMBLYJAR \
  $JARFILE \
  "$@"
