#!/bin/bash

target_dir=./generated

mkdir -p $target_dir

for d in $(ls -d ./data/*)
do
    unzip $d -d $target_dir
done
