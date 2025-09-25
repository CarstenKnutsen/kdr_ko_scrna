
#!/bin/bash
#$ -N cr_kdrko
#$ -cwd
#$ -o cr_count.out
#$ -e cr_count.err
#$ -pe smp 16
#$ -l h_vmem=8G
#$ -l h_rt=48:00:00
#$ -V

# Load cellranger
module load CBI
module load cellranger #version 9.0.0


# Set paths
REF_DIR="/wynton/group/alvira/genomes/refdata-gex-GRCm39-2024-A"
FASTQ_DIR="/wynton/group/alvira/data/250909_kdr_ko_scrnaseq/RZ14432"
OUT_DIR="/wynton/group/alvira/tmp_processed_data/250909_kdr_ko_scrnaseq"

# Sample IDs (manually or automatically derived from FASTQ names)
SAMPLES=("1_P7_WT_Nor" "2_P7_KO_Nor" "3_P7_WT_Hy" "4_P7_KO_Hy")

# Loop through each sample
for SAMPLE in "${SAMPLES[@]}"; do
  echo "Running Cell Ranger count for ${SAMPLE}..."
  mkdir -p "${OUT_DIR}/${SAMPLE}"
  cd "${OUT_DIR}/${SAMPLE}"
  cellranger count \
    --id="${SAMPLE}" \
    --transcriptome="${REF_DIR}" \
    --fastqs="${FASTQ_DIR}" \
    --sample="${SAMPLE}" \
    --localcores=16 \
    --localmem=128 \
    --include-introns=true \
    --create-bam=false
done


