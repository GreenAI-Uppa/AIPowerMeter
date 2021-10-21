#!/bin/bash
for i in $(seq 5)
do
    echo $((i*100))
    python bert.py --input $((i*100))
done

