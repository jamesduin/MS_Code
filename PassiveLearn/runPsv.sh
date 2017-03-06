#!/usr/bin/env bash

for i in $(seq 1 3)
    do echo $i
    python runPsv.py $i &
    #python runPsv.py $i
done

wait

for i in $(seq 4 6)
    do echo $i
    python runPsv.py $i &
    #python runPsv.py $i
done

wait

for i in $(seq 7 9)
    do echo $i
    python runPsv.py $i &
    #python runPsv.py $i
done

wait

python runPsv.py 10
echo 10