import torch
import pandas as pd
import argparse
import os
import logging
import json
from datetime import datetime
from CL_utils import perform_retrieval, save_results
from CL_model import ContrastiveModel
from CL_protein_feature import precompute_protein_features
from CL_molecule_feature import precompute_molecule_features

def setup_logger(output_dir):
    """
    Initialize and configure logger for console output.
    """
    logger = logging.getLogger('RetrievalPipeline')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    return logger

def parse_arguments():
    """
    Command line interface configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input pocket CSV file path (ABP_prediction.csv)")
    parser.add_argument("--molecules", default="data/mol_db.csv", help="Molecule CSV file path")
    parser.add_argument("--output", default="results/substrate_prediction.csv", help="Output results file path")
    parser.add_argument("--protein_dir", default="results/protein_data", help="Protein feature directory")
    parser.add_argument("--molecule_dir", default="data/molecule_data", help="Molecule feature directory")
    parser.add_argument("--gmm_dir", default="model/gmm_model/", help="GMM model directory")
    parser.add_argument("--weights", default="model/conAden_04012025.pth", help="Model weights path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device to use for computation (auto: automatically choose, cpu: force CPU, cuda: force GPU)")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--max_sep", action="store_true", help="Use maximum separation method for retrieval")
    method_group.add_argument("--top_k", type=int, default=None, help="Use top-k method with specified k value")
    return parser.parse_args()

def csv_to_json(pocket_csv_path, prediction_csv_path, json_path, logger):
    """
    Convert ABP_prediction.csv and substrate_prediction_xxx.csv into the specified JSON format.
    The CSVs are directly aligned on 'id'.
    """
    try:
        pocket_df = pd.read_csv(pocket_csv_path)
        pred_df = pd.read_csv(prediction_csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV files for JSON conversion: {e}")
        return

    # Helper: parse binding_pocket_positions string into integer list
    def parse_positions(s):
        if pd.isna(s):
            return []
        return [int(x) for x in str(s).split(",") if x.strip().isdigit()]

    # Helper: extract substrate predictions from wide table format
    def parse_predictions(row):
        preds = []
        i = 1
        while f"molecule_{i}" in row and f"confidence_score_{i}" in row:
            mol = row[f"molecule_{i}"]
            conf = row[f"confidence_score_{i}"]
            if pd.notna(mol) and pd.notna(conf):
                preds.append({
                    "substrate": str(mol),
                    "confidence": float(conf)
                })
            i += 1
        return preds

    # Merge pocket and prediction table on 'id'
    merged = pd.merge(pocket_df, pred_df, on="id", how="inner")

    # Build JSON output
    results = []
    for _, row in merged.iterrows():
        results.append({
            "id": row["id"],
            "domain_sequence": row.get("domain_sequence", ""),
            "binding_pocket": row.get("binding_pocket", ""),
            "binding_pocket_positions": parse_positions(row.get("binding_pocket_positions", "")),
            "predictions": parse_predictions(row)
        })

    # Write to JSON file
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved JSON to {json_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}")

def main():
    args = parse_arguments()
    output_dir = os.path.dirname(args.output)
    # Set output filename and corresponding JSON filename based on retrieval method
    if args.top_k is not None:
        output_file = os.path.join(output_dir, "substrate_prediction_top_k.csv")
        json_file = os.path.join(output_dir, "substrate_prediction_top_k.json")
    elif args.max_sep:
        output_file = os.path.join(output_dir, "substrate_prediction_max_sep.csv")
        json_file = os.path.join(output_dir, "substrate_prediction_max_sep.json")
    else:
        output_file = args.output
        json_file = os.path.splitext(output_file)[0] + ".json"

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)
    
    try:
        logger.info("========== Binding Prediction Pipeline Started ==========")
        logger.info(f"Input protein file: {args.input}")
        logger.info(f"Input molecules file: {args.molecules}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Using retrieval method: {'max_sep' if args.max_sep else f'top_{args.top_k if args.top_k else 10}'}")
        
        # Device selection
        if args.device == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU (forced by user)")
        elif args.device == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA (forced by user)")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        else:  # auto
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'} (auto-detected)")
        
        # Step 1: Feature computation for proteins
        logger.info("Processing protein features...")
        precompute_protein_features(args.input, args.protein_dir, device=device)
        
        # Step 2: Feature computation for molecules (if not already present)
        if not os.path.exists(args.molecule_dir):
            os.makedirs(args.molecule_dir, exist_ok=True)
            logger.info("Computing molecule features...")
            precompute_molecule_features(args.molecules, device, args.molecule_dir)
        else:
            logger.info("Using existing molecule features")
        
        # Step 3: Model initialization
        logger.info("Initializing model...")
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Model weights not found at {args.weights}")
        model = ContrastiveModel()
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model = model.to(device)
        logger.info(f"Successfully loaded model from {args.weights}")
        
        # Step 4: Data preparation
        logger.info("Loading input data...")
        ids = pd.read_csv(args.input)['id']
        mol_labels = pd.read_csv(args.molecules)['labels']
        logger.info(f"Loaded {len(ids)} proteins and {len(mol_labels)} molecules")
        
        # Step 5: Retrieval process
        logger.info("Starting retrieval process...")
        results = perform_retrieval(
            model=model,
            ids=ids,
            molecule_labels=mol_labels,
            gmm_dir=args.gmm_dir,
            use_max_sep=args.max_sep,
            top_k=args.top_k if args.top_k is not None else 10,
            device=device
        )
        
        # Step 6: Save results (CSV)
        save_results(results, output_file)
        logger.info(f"Successfully saved results to {output_file}")

        # Step 7: Save results as JSON aligned with ABP_prediction.csv
        csv_to_json(
            pocket_csv_path=args.input,          # ABP_prediction.csv
            prediction_csv_path=output_file,     # substrate_prediction_max_sep.csv or substrate_prediction_top_k.csv
            json_path=json_file,                 # substrate_prediction_max_sep.json or substrate_prediction_top_k.json
            logger=logger
        )
        
        logger.info("========== Binding Prediction Pipeline Completed ==========")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()