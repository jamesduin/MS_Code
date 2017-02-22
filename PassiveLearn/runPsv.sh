#!/usr/bin/env bash

for i in $(seq 1 10)
    do echo $i
    python runPsv.py $i &
done

wait
python plotPsv.py