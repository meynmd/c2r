#!/bin/bash

wav_file=$1
csv_file=$2
out_dir=$3

function split()
{
    line=$1
    sep=$2
    sel=$3
    local IFS=$sep
    fields=($line)
    echo ${fields[${sel}]}
}

mkdir -p $out_dir

i=0
while read line; do
    (( i += 1 ))
    start=$(split $line , 0)
    stop=$(split $line , 1)
    dur=$(( $stop - $start ))
    music_name=$(split $(basename ${csv_file}) . 0)
    cmd="sox $wav_file ${out_dir}/${music_name}_${i}.wav trim $start $dur"
    echo "$cmd"
    eval "$cmd"

done < $csv_file