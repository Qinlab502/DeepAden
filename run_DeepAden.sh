#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[1;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GRAY}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
banner() {
    echo -e "\n${CYAN}$(printf '═%.0s' {1..55})${NC}"
    echo -e "${CYAN}  DeepAden  │  $*${NC}"
    echo -e "${CYAN}$(printf '═%.0s' {1..55})${NC}"
}

FASTA_FILE=""
GBK_FILE=""
GENOME_FILE=""   
INPUT_TYPE=""
OUTPUT_DIR="${SCRIPT_DIR}/results"
PLM="${SCRIPT_DIR}/model/esm2_t33_650M_UR50D"
CM="${SCRIPT_DIR}/model/contact_prediction_sigmoid_1024_0.65_1553.pth"
GAT="${SCRIPT_DIR}/model/ABP_GAT_10292024.pth"

BINDING_MODEL_DIR="${SCRIPT_DIR}/model"

REFERENCE="${SCRIPT_DIR}/data/template_correction.csv"
PROCESS=12
PYTHON_SCRIPTS_DIR="${SCRIPT_DIR}/src"
HMM_MODEL="${SCRIPT_DIR}/data/AMP-binding/PF00501.hmm"


TOP_K=3                        
MODEL_WEIGHT_NAME="all.weight"  
DEVICE="auto"                   # Device to use: auto | cpu | gpu

function usage() {
    echo "Usage: $0 (-f <fasta_file> | -g <gbk_file> | -G <genome_fasta>) [-o <output_dir>] [-p <plm_path>] [-c <cm_path>] [-d <binding_model_dir>] [-r <reference_csv>] [-n <processes>] [-k <top_k>] [-m <model_weight_name>]"
    echo ""
    echo "Input Options (choose one):"
    echo "  -f <fasta_file>           Path to the input protein FASTA file"
    echo "  -g <gbk_file>             Path to the input GBK file"
    echo "  -G <genome_fasta>         Path to the input genome FASTA file (nucleotide)"
    echo ""
    echo "Other Options:"
    echo "  -o <output_dir>           Path to the output directory (default: ${SCRIPT_DIR}/example/output)"
    echo "  -p <plm_path>             Path to the pre-trained PLM weights (default: $PLM)"
    echo "  -c <cm_path>              Path to the contact map model (default: $CM)"
    echo "  -d <binding_model_dir>    Directory containing binding prediction weights (default: $BINDING_MODEL_DIR)"
    echo "  -r <reference_csv>        Path to the reference CSV file (default: $REFERENCE)"
    echo "  -k <top_k>                Use top-k method with specified k value (default: 3)"
    echo "  -n <processes>            Number of processes (default: $PROCESS)"
    echo "  -m <model_weight_name>    Model weight to use: all.weight or benchmark.weight (default: all.weight)"
    echo "  -D <device>               Device to use: auto, cpu, or gpu (default: auto)"
    echo ""
    echo "Mapping:"
    echo "  all.weight       -> weights_for_users.pth"
    echo "  benchmark.weight -> weights_for_benchmark.pth"
    echo ""
    exit 1
}


while getopts "f:g:G:o:p:c:d:r:n:k:m:D:h" opt; do
    case $opt in
        f)
            FASTA_FILE=$(realpath "$OPTARG")
            INPUT_TYPE="fasta"
            ;;
        g)
            GBK_FILE=$(realpath "$OPTARG")
            INPUT_TYPE="gbk"
            ;;
        G)
            GENOME_FILE=$(realpath "$OPTARG")
            INPUT_TYPE="genome"
            ;;
        o) OUTPUT_DIR=$(realpath "$OPTARG") ;;
        p) PLM=$(realpath "$OPTARG") ;;
        c) CM=$(realpath "$OPTARG") ;;
        d) BINDING_MODEL_DIR=$(realpath "$OPTARG") ;;
        r) REFERENCE=$(realpath "$OPTARG") ;;
        n) PROCESS=$OPTARG ;;
        k) TOP_K=$OPTARG ;;          
        m) MODEL_WEIGHT_NAME="$OPTARG" ;;  
        D) DEVICE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done


case "$MODEL_WEIGHT_NAME" in
    all.weight)
        BINDING_WEIGHTS="${BINDING_MODEL_DIR}/weights_for_users.pth"
        ;;
    benchmark.weight)
        BINDING_WEIGHTS="${BINDING_MODEL_DIR}/weights_for_benchmark.pth"
        ;;
    *)
        error "Unknown model weight name '$MODEL_WEIGHT_NAME'. Use 'all.weight' or 'benchmark.weight'."
        exit 1
        ;;
esac

if [ ! -f "$BINDING_WEIGHTS" ]; then
    error "Binding weight file not found: $BINDING_WEIGHTS"
    exit 1
fi

case "$DEVICE" in
    auto|cpu|gpu) ;;
    *) error "-D must be 'auto', 'cpu', or 'gpu'."; exit 1 ;;
esac


if [ -z "$INPUT_TYPE" ]; then
    error "Either -f (fasta_file), -g (gbk_file) or -G (genome_fasta) is required."
    usage
fi


input_count=0
[ -n "$FASTA_FILE" ] && input_count=$((input_count+1))
[ -n "$GBK_FILE" ] && input_count=$((input_count+1))
[ -n "$GENOME_FILE" ] && input_count=$((input_count+1))

if [ "$input_count" -ne 1 ]; then
    error "Exactly one of -f, -g or -G must be specified."
    usage
fi

mkdir -p "$OUTPUT_DIR"


if [ "$INPUT_TYPE" = "gbk" ]; then
    banner "Pre-processing — GBK Extraction"
    info "Input  : $GBK_FILE"
    info "Running GBK_extract.py ..."
    python "${PYTHON_SCRIPTS_DIR}/GBK_extract.py" "$GBK_FILE" "${OUTPUT_DIR}/extracted_cds.fasta"

    if [ ! -s "${OUTPUT_DIR}/extracted_cds.fasta" ]; then
        error "Failed to extract CDS sequences from GBK file or no CDS found."
        exit 1
    fi
    info "[✓] CDS sequences extracted → ${OUTPUT_DIR}/extracted_cds.fasta"

    FASTA_FILE="${OUTPUT_DIR}/extracted_cds.fasta"
fi


if [ "$INPUT_TYPE" = "genome" ]; then
    banner "Pre-processing — Genome ORF Prediction & Domain Detection"
    info "Input  : $GENOME_FILE"

    GENOME_BASENAME=$(basename "$GENOME_FILE")
    GENOME_PREFIX="${GENOME_BASENAME%.*}"

    GENOME_GBK="${OUTPUT_DIR}/${GENOME_PREFIX}.gbk"
    GENOME_FAA="${OUTPUT_DIR}/${GENOME_PREFIX}.faa"
    DOMAINS_CSV="${OUTPUT_DIR}/${GENOME_PREFIX}_domains.csv"

    info "Running Prodigal ..."
    prodigal -i "$GENOME_FILE" -f gbk -o "$GENOME_GBK" -p single -a "$GENOME_FAA"

    if [ ! -s "$GENOME_FAA" ]; then
        error "Prodigal failed or produced empty protein FASTA (${GENOME_FAA})."
        exit 1
    fi
    info "[✓] Prodigal finished → ${GENOME_FAA}"

    info "Running domain_identification.py ..."
    python "${PYTHON_SCRIPTS_DIR}/domain_identification.py" \
        -r "${SCRIPT_DIR}/data/nrps_domains/nrpspksdomains.hmm" \
        -f "$GENOME_FAA" \
        -o "$DOMAINS_CSV"

    if [ ! -s "$DOMAINS_CSV" ]; then
        error "domain_identification.py produced empty domain CSV (${DOMAINS_CSV})."
        exit 1
    fi
    info "[✓] Domain table written → ${DOMAINS_CSV}"

    info "Running detect_nrps_modules.py ..."
    python "${PYTHON_SCRIPTS_DIR}/detect_nrps_modules.py" \
        -i "$DOMAINS_CSV" \
        -f "$GENOME_FAA" \
        -o "$OUTPUT_DIR/"

    if [ ! -s "${OUTPUT_DIR}/extracted_cds.fasta" ]; then
        error "detect_nrps_modules.py did not produce extracted_cds.fasta in ${OUTPUT_DIR}."
        exit 1
    fi
    info "[✓] A-domain FASTA written → ${OUTPUT_DIR}/extracted_cds.fasta"

    FASTA_FILE="${OUTPUT_DIR}/extracted_cds.fasta"
fi


banner "A-domain Detection — hmmscan"
info "Scanning : $FASTA_FILE"
hmmscan --domtblout "${OUTPUT_DIR}/adomains.dom" "$HMM_MODEL" "$FASTA_FILE" > /dev/null
python "${PYTHON_SCRIPTS_DIR}/extract_adomains.py" "${OUTPUT_DIR}/adomains.dom" "$FASTA_FILE" "${OUTPUT_DIR}/adomains.fasta"

if [ ! -s "${OUTPUT_DIR}/adomains.fasta" ]; then
    warn "No adenylation (A) domains detected. Exiting."
    exit 1
fi
info "[✓] A-domains extracted → ${OUTPUT_DIR}/adomains.fasta"


python "${PYTHON_SCRIPTS_DIR}/ABP_GAT_featurization.py" \
    --fasta "${OUTPUT_DIR}/adomains.fasta" \
    --feature_dir "$OUTPUT_DIR" \
    --plm "$PLM" \
    --cm "$CM" \
    --device "$DEVICE"

python "${PYTHON_SCRIPTS_DIR}/ABP_GAT_inference.py" \
    --fasta "${OUTPUT_DIR}/adomains.fasta" \
    --feature_dir "$OUTPUT_DIR" \
    --reference "$REFERENCE" \
    --output "$OUTPUT_DIR/" \
    --GAT "$GAT" \
    --device "$DEVICE"


BINDING_OUTPUT_FILE="${OUTPUT_DIR}/substrate_predictions_top${TOP_K}_${MODEL_WEIGHT_NAME}.csv"
BINDING_OUTPUT_JSON="${OUTPUT_DIR}/substrate_predictions_top${TOP_K}_${MODEL_WEIGHT_NAME}.json"


python "${PYTHON_SCRIPTS_DIR}/binding_prediction.py" \
    --input "$OUTPUT_DIR/ABP_prediction.csv" \
    --top_k "$TOP_K" \
    --weights "$BINDING_WEIGHTS" \
    --output "$BINDING_OUTPUT_FILE" \
    --device "$DEVICE"


if [ "$INPUT_TYPE" = "genome" ]; then
    NRPS_JSON="${OUTPUT_DIR}/NRPS_modules.json"
    PRED_JSON="${OUTPUT_DIR}/substrate_predictions_top${TOP_K}_${MODEL_WEIGHT_NAME}.json"
    MERGED_NRPS_JSON="${OUTPUT_DIR}/NRPS_modules_pred.json"

    if [ -f "$NRPS_JSON" ] && [ -f "$PRED_JSON" ]; then
        banner "Post-processing — Merging Predictions into NRPS JSON"
        info "Running nrps_json.py ..."
        python "${PYTHON_SCRIPTS_DIR}/nrps_json.py" \
            --nrps-json "$NRPS_JSON" \
            --pred-json "$PRED_JSON" \
            --out "$MERGED_NRPS_JSON"
        info "[✓] Merged NRPS JSON → ${MERGED_NRPS_JSON}"
    else
        warn "Skipping NRPS merge (missing NRPS or prediction JSON)."
        warn "  NRPS_JSON : $NRPS_JSON"
        warn "  PRED_JSON : $PRED_JSON"
    fi
fi


# info "Cleaning up intermediate files ..."
rm -rf "$OUTPUT_DIR/emb_dir" "$OUTPUT_DIR/pf_dir" "$OUTPUT_DIR/ei_dir" "$OUTPUT_DIR/feature_dir" "$OUTPUT_DIR/protein_data" "$OUTPUT_DIR/pyg_dir"
rm -f "$OUTPUT_DIR/adomains.fasta" "$OUTPUT_DIR/adomains.dom" "$OUTPUT_DIR/extracted_cds.fasta" "$OUTPUT_DIR"/*.gbk "$OUTPUT_DIR"/*.faa "$OUTPUT_DIR"/*_domains.csv

echo -e "\n${GREEN}${BOLD}DeepAden finished successfully.${NC}"
info "Results saved to: $OUTPUT_DIR"