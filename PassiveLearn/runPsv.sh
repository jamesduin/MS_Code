#!/usr/bin/env bash

for i in $(seq 2 10)
    do echo $i
    python runPsv.py $i &
done

wait
