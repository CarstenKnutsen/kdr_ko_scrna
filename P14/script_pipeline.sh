#!/bin/bash
echo 'starting'
source /home/carsten/anaconda3/etc/profile.d/conda.sh

conda activate soupxR
Rscript soupX.R
conda activate kdr_ko
python initial_analysis.py
python cell_typing_by_lineage.py
python cell_typing_by_lineage_no_cc.py
python deg.py
python subcluster.py
conda activate trajectory_inference
python trajectory_inference.py



