#!/bin/bash
echo "running detector"

DET_START=$(date +%s)
python detector.py train/ train.json
DET_END=$(date +%s)

DET_DIFF=$(( $DET_END - $DET_START ))
echo "Detection took $DET_DIFF seconds"
echo ""

echo "running evaluation"
EVAL_START=$(date +%s)
python evaluate.py train/ train.json
EVAL_END=$(date +%s)

EVAL_DIFF=$(( $EVAL_END - $EVAL_START ))
echo "Evaluation took $EVAL_DIFF seconds"

read