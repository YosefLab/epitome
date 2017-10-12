#!/bin/bash

# Figure out where we are.
FWDIR="$(cd `dirname $0`; pwd)"

CLASS=$1
shift

# Split args into Spark and mango args
DD=False  # DD is "double dash"
PRE_DD=()
POST_DD=()
for ARG in "$@"; do
  shift
  if [[ $ARG == "--" ]]; then
    DD=True
    POST_DD=( "$@" )
    break
  fi
  PRE_DD+=("$ARG")
done

if [[ $DD == True ]]; then
  SPARK_ARGS="${PRE_DD[@]}"
  EPITOME_ARGS="${POST_DD[@]}"
else
  SPARK_ARGS=()
  EPITOME_ARGS="${PRE_DD[@]}"
fi


# does the user have EPITOME_OPTS set? if yes, then warn
if [[ $DD == False && -n "$EPITOME_OPTS" ]]; then
    echo "WARNING: Passing Spark arguments via EPITOME_OPTS was recently removed."
fi

# Figure out where mango is installed
SCRIPT_DIR="$(cd `dirname $0`/..; pwd)"

# Get list of required jars for epitome
EPITOME_JARS=$("$SCRIPT_DIR"/bin/compute-epitome-jars.sh)

# Split out the jar, since it will be passed to Spark as the "primary resource".
EPITOME_JAR=${EPITOME_JARS##*,}
EPITOME_JARS=$(echo "$EPITOME_JARS" | rev | cut -d',' -f2- | rev)

# append EPITOME_JARS to the --jars option, if any
SPARK_ARGS=$("$SCRIPT_DIR"/bin/append_to_option.py , --jars $EPITOME_JARS $SPARK_ARGS)

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

"$SPARK_SUBMIT" \
  --class $CLASS \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryo.registrator=org.bdgenomics.adam.serialization.ADAMKryoRegistrator \
  $SPARK_ARGS \
  $EPITOME_JAR \
  $EPITOME_ARGS

