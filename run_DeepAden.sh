#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

FASTA_FILE=""
OUTPUT_DIR="${SCRIPT_DIR}/results"
PLM="${SCRIPT_DIR}/model/esm2_t33_650M_UR50D"
CM="${SCRIPT_DIR}/model/ABP_CM_08232024.pth"
REFERENCE="${SCRIPT_DIR}/data/template_correction.csv"
PROCESS=12
PYTHON_SCRIPTS_DIR="${SCRIPT_DIR}/src"
GAT="${SCRIPT_DIR}/model/ABP_GAT_10292024.pth"
MAX_SEP=true
TOP_K=10

function usage() {
    echo "Usage: $0 -f <fasta_file> [-o <output_dir>] [-p <plm_path>] [-c <cm_path>] [-r <reference_csv>] [-n <processes>] [-m] [-k <top_k>]"
    echo ""
    echo "Options:"
    echo "  -f <fasta_file>      Path to the input FASTA file (required)"
    echo "  -o <output_dir>      Path to the output directory (default: ${SCRIPT_DIR}/results"
    echo "  -p <plm_path>        Path to the pre-trained model weights (default: $PLM)"
    echo "  -c <cm_path>         Path to the contact map model (default: $CM)"
    echo "  -r <reference_csv>   Path to the reference CSV file (default: $REFERENCE)"
    echo "  -m                   Use maximum separation method (default: true)"
    echo "  -k <top_k>           Use top-k method with specified k value (default: 10)"
    echo "  -n <processes>       Number of processes (default: $PROCESS)"
    echo ""
    echo "Note: By default uses maximum separation method. Specify -k to use top-k instead."
    exit 1
}

while getopts "f:o:p:c:r:n:mk:h" opt; do
    case $opt in
        f) FASTA_FILE=$(realpath "$OPTARG") ;;
        o) OUTPUT_DIR=$(realpath "$OPTARG") ;;
        p) PLM=$(realpath "$OPTARG") ;;
        c) CM=$(realpath "$OPTARG") ;;
        r) REFERENCE=$(realpath "$OPTARG") ;;
        n) PROCESS=$OPTARG ;;
        m) MAX_SEP=true ;;
        k)  TOP_K=$OPTARG
            MAX_SEP=false
            ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$FASTA_FILE" ]; then
    echo -e "${RED}Error: -f (fasta_file) is required.${NC}" >&2
    usage
fi

if [ "$MAX_SEP" = true ] && [ "$TOP_K" != 10 ]; then
    echo "Error: Cannot specify both -m and -k options together" >&2
    usage
fi

python "${PYTHON_SCRIPTS_DIR}/ABP_GAT_featurization.py" --fasta "$FASTA_FILE" --feature_dir "$OUTPUT_DIR" --plm "$PLM" --cm "$CM"
python "${PYTHON_SCRIPTS_DIR}/ABP_GAT_inference.py" --fasta "$FASTA_FILE" --feature_dir "$OUTPUT_DIR" --reference "$REFERENCE" --output "$OUTPUT_DIR/" --GAT "$GAT"

BINDING_CMD=("python" "${PYTHON_SCRIPTS_DIR}/binding_prediction.py" "--input" "$OUTPUT_DIR/ABP_prediction.csv")

if [ "$MAX_SEP" = true ]; then
    BINDING_CMD+=("--max_sep")
else
    BINDING_CMD+=("--top_k" "$TOP_K")
fi

"${BINDING_CMD[@]}"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    # Clean up intermediate folders only if the pipeline succeeded
    for DIR in ei_dir emb_dir pf_dir protein_data pyg_dir; do
        TARGET="${OUTPUT_DIR}/${DIR}"
        if [ -d "$TARGET" ]; then
            echo -e "${GREEN}Deleting $TARGET ...${NC}"
            rm -rf "$TARGET"
        fi
    done
else
    echo -e "${RED}Pipeline failed, intermediate files are kept for debugging.${NC}"
fi