#!/usr/bin/env bash

for i in $(seq 1 5)
    do echo $i
    python runPsv.py $i &
    #python runPsv.py $i
done

wait

for i in $(seq 6 10)
    do echo $i
    python runPsv.py $i &
    #python runPsv.py $i
done

wait
#
#for i in $(seq 5 6)
#    do echo $i
#    python runPsv.py $i &
#    #python runPsv.py $i
#done
#
#wait
#
#for i in $(seq 7 8)
#    do echo $i
#    python runPsv.py $i &
#    #python runPsv.py $i
#done
#
#wait
#
#for i in $(seq 9 10)
#    do echo $i
#    python runPsv.py $i &
#    #python runPsv.py $i
#done
#
#wait