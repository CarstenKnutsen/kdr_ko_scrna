#!/usr/bin/bash
#SBATCH --job-name=velocyto
#SBATCH --output=velocyto.%j.out
#SBATCH --error=velocyto.%j.err
#SBATCH --time=2-00:00:00
#SBATCH -p normal
#SBATCH -n 64
#SBATCH --mem=128GB
ml python/3.9.0
ml biology
ml samtools/1.16.1
source /home/groups/calvira/software/py_venvs/velocyto/bin/activate
for dir in /scratch/users/cknutsen/kdr_ko_scrnaseq/data/processed/*/
do
    dir2=${dir%*/}
    dir2="${dir2##*/}"
    echo $dir2
    velocyto run10x --verbose $dir /home/groups/calvira/software/refdata-gex-GRCm39-2024-A/genes/genes.gtf
done

