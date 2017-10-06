#!/usr/bin/env bash

set -e

# Figure out where epitome is installed
EPITOME_REPO="$(cd `dirname $0`/..; pwd)"

CLASSPATH="$(. "$EPITOME_REPO"/bin/compute-epitome-classpath.sh)"

# list of jars to ship with spark; trim off the first from the CLASSPATH --> this is /etc
# TODO: brittle? assumes appassembler always puts the $BASE/etc first and the CLI jar last
EPITOME_JARS="$(echo "$CLASSPATH" | tr ":" "\n" | tail -n +2 | perl -pe 's/\n/,/ unless eof' )"

echo "$EPITOME_JARS"
