import torch
import pandas as pd
import argparse
import os
import logging
from datetime import datetime
from CL_utils import perform_retrieval, save_results
from CL_model import ContrastiveModel
from CL_protein_feature import precompute_protein_features
from CL_molecule_feature import precompute_molecule_features

def setup_logger(output_dir):
    """Initialize and configure logger"""
    logger = logging.getLogger('RetrievalPipeline')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

def parse_arguments():
    """Command line interface configuration"""
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument("--input", required=True, 
                      help="Input pocket CSV file path")
    
    # Optional arguments with defaults
    parser.add_argument("--molecules", default="data/mol_db.csv", 
                      help="Molecule CSV file path")
    parser.add_argument("--output", default="results/output.csv",
                      help="Output results file path")
    parser.add_argument("--protein_dir", default="results/protein_data",
                      help="Protein feature directory")
    parser.add_argument("--molecule_dir", default="data/molecule_data",
                      help="Molecule feature directory")
    parser.add_argument("--gmm_dir", default="model/gmm_model/",
                      help="GMM model directory")
    parser.add_argument("--weights", default="model/contrastive_model_20250401_111117_final.pth",
                      help="Model weights path")
    
    # Retrieval method selection
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--max_sep", action="store_true",
                            help="Use maximum separation method for retrieval")
    method_group.add_argument("--top_k", type=int, default=None,
                            help="Use top-k method with specified k value")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(os.path.dirname(args.output))
    
    try:
        logger.info("========== Binding Prediction Pipeline Started ==========")
        logger.info(f"Input protein file: {args.input}")
        logger.info(f"Input molecules file: {args.molecules}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Using retrieval method: {'max_sep' if args.max_sep else f'top_{args.top_k if args.top_k else 10}'}")
        
        # 1. Feature computation
        logger.info("Processing protein features...")
        precompute_protein_features(args.input, args.protein_dir)
        
        # 2. Molecule features (compute only if needed)
        if not os.path.exists(args.molecule_dir):
            os.makedirs(args.molecule_dir, exist_ok=True)
            logger.info("Computing molecule features...")
            precompute_molecule_features(args.molecules, args.molecule_dir)
        else:
            logger.info("Using existing molecule features")
        
        # 3. Model initialization
        logger.info("Initializing model...")
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Model weights not found at {args.weights}")
            
        model = ContrastiveModel()
        model.load_state_dict(torch.load(args.weights))
        logger.info(f"Successfully loaded model from {args.weights}")
        
        # 4. Data preparation
        logger.info("Loading input data...")
        protein_ids = pd.read_csv(args.input)['id']
        mol_labels = pd.read_csv(args.molecules)['labels']
        logger.info(f"Loaded {len(protein_ids)} proteins and {len(mol_labels)} molecules")
        
        # 5. Retrieval process
        logger.info("Starting retrieval process...")
        results = perform_retrieval(
            model=model,
            protein_ids=protein_ids,
            molecule_labels=mol_labels,
            gmm_dir=args.gmm_dir,
            use_max_sep=args.max_sep,
            top_k=args.top_k if args.top_k is not None else 10
        )
        
        # 6. Save results
        save_results(results, args.output)
        logger.info(f"Successfully saved results to {args.output}")
        
        logger.info("========== Binding Prediction Pipeline Completed ==========")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()