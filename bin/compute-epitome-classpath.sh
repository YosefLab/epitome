#!/usr/bin/env bash

# Figure out where Epitome is installed
SCRIPT_DIR="$(cd `dirname $0`/..; pwd)"

# Setup CLASSPATH like appassembler

# Assume we're running in a binary distro
EPITOME_CMD="$SCRIPT_DIR/bin/epitome"
REPO="$SCRIPT_DIR/repo"

# Fallback to source repo
if [ ! -f $EPITOME_CMD ]; then
EPITOME_CMD="$SCRIPT_DIR/preprocess-scala/target/appassembler/bin/epitome"
REPO="$SCRIPT_DIR/preprocess-scala/target/appassembler/repo"
fi

if [ ! -f "$EPITOME_CMD" ]; then
  echo "Failed to find appassembler scripts in $BASEDIR/bin"
  echo "You need to build Epitome before running this program"
  exit 1
fi
eval $(cat "$EPITOME_CMD" | grep "^CLASSPATH")

echo "$CLASSPATH"