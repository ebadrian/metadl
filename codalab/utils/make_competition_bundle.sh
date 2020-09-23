#!/bin/bash
# IMPORTANT: Please execute this script directly in its directory
# 
# This script is going to create a zip file by
#
# 1) creating a bundles/tmp/ directory
#
# 2) copying in bundles/tmp/ the following files/directories:
#
# competition.yaml
# *.jpg
# *.png
# *.html
# assets/
#
# 3) zipping and moving to tmp/ the contents or the following dirs
# and naming them xxx.zip
#
# xxx=
# meta_train/
# meta_test/
# starting_kit/
#
# 4) zipping and moving to bundles/tmp/ the contents or the following dirs
# metadl/core/xxx
# xxx=
# ingestion
# scoring
#
# and naming them xxx_program.zip

# Get root dir
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"/../ # codalab/
echo 'Using ROOT_DIR: '$ROOT_DIR
PACKAGE_DIR=$ROOT_DIR/../metadl/
cd $ROOT_DIR

DATE=`date "+%Y%m%d-%H%M%S"`
BUNDLES_DIR='bundles/'
mkdir $BUNDLES_DIR
DIR=$BUNDLES_DIR'tmp/'
# Delete $DIR if exists
rm -rf $DIR
mkdir $DIR
cp *.jpg $DIR
cp *.png $DIR
cp *.gif $DIR
cp *.html $DIR
cp *.yaml $DIR
# cp -r '../assets' $DIR
cd $ROOT_DIR # codalab/

# Begin zipping each data
for filename in meta_train meta_test; do
  cd $filename;
  echo 'Zipping: '$filename;
  zip -o -r --exclude=*.git* --exclude=*__pycache__* --exclude=*.DS_Store* "../"$DIR$filename .;
  cd $ROOT_DIR; # codalab/
done

# Zipping ingestion and scoring program
cd $PACKAGE_DIR"core/"
for filename in ingestion scoring; do
  cd $filename;
  echo $filename;
  echo $(pwd)
  zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* $ROOT_DIR$DIR$filename"_program" .;
  cd ..; 
done

# Zipping starting kit
cd $PACKAGE_DIR
for filename in starting_kit; do
  cd $filename;
  echo $filename;
  echo $(pwd)
  zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* $ROOT_DIR$DIR$filename .;
  cd ..; 
done

# Zip all to make a competition bundle
cd $ROOT_DIR$DIR
zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../MetaDL_"$DATE .;
