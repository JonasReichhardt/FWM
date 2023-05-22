#!/bin/bash
START=$(date +%s)

printf "running detector\n"
python detector.py train/ train.json
printf "done...\n\n"

printf "running evaluation\n"
python evaluate.py train/ train.json
printf "done...\n\n" 

END=$(date +%s)
DIFF=$(( $END - $START ))

echo "It took $DIFF seconds"
read