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
    """Initialize and configure logger"""
    logger = logging.getLogger('RetrievalPipeline')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 避免重复添加 handler
    logger.handlers.clear()
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
    parser.add_argument("--output", default="example/output/substrate_predictions.csv",
                      help="Output results file path")
    parser.add_argument("--protein_dir", default="example/output/protein_data",
                      help="Protein feature directory")
    parser.add_argument("--molecule_dir", default="data/molecule_data",
                      help="Molecule feature directory")
    parser.add_argument("--calibrator_path", default="model/kde_model/kde_calibrator.pkl",
                      help="KDE calibrator pickle file path")
    parser.add_argument("--weights", default="model/weights_for_users.pth",
                      help="Model weights path")
    
    # Retrieval method selection
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--top_k", type=int, default=3,
                            help="Use top-k method with specified k value")
    
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"],
                        help="Device to use: auto, cpu, or gpu (default: auto)")
    
    return parser.parse_args()


def resolve_device(device_arg):
    """Resolve the device argument to a torch.device."""
    if device_arg == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_arg == "cpu":
        return torch.device("cpu")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def csv_to_json(pocket_csv_path, prediction_csv_path, json_path, logger):
    """
    将 ABP_prediction.csv (pocket) 和 substrate_predictions_top*.csv (prediction)
    对齐后导出为 JSON。

    现在假定 pocket_csv 结构为：
      id,region_1,region_2,region_3,region_4,binding_pocket_positions,domain_sequence,binding_pocket

    prediction_csv 结构为：
      id,Top1,Top1_score,Top2,Top2_score,Top3,Top3_score
    """
    try:
        pocket_df = pd.read_csv(pocket_csv_path)
        pred_df = pd.read_csv(prediction_csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV files for JSON conversion: {e}")
        return

    # 1. 检查 id 列是否存在
    if "id" not in pocket_df.columns:
        logger.error(f"Column 'id' not found in pocket CSV: {pocket_csv_path}")
        logger.error(f"Pocket CSV columns: {list(pocket_df.columns)}")
        return
    if "id" not in pred_df.columns:
        logger.error(f"Column 'id' not found in prediction CSV: {prediction_csv_path}")
        logger.error(f"Prediction CSV columns: {list(pred_df.columns)}")
        return

    merged = pd.merge(pocket_df, pred_df, on="id", how="inner", suffixes=("", "_pred"))

    # 2. 解析 binding_pocket_positions 列（逗号分隔字符串 -> int 列表）
    def parse_positions(pos_str):
        if pd.isna(pos_str):
            return []
        s = str(pos_str).strip()
        if not s:
            return []
        try:
            return [int(x) for x in s.split(",") if x.strip() != ""]
        except Exception as e:
            logger.warning(f"Failed to parse binding_pocket_positions='{s}': {e}")
            return []

    # 3. 解析 Top1/Top1_score, Top2/Top2_score, Top3/Top3_score 为 predictions
    def parse_predictions(row):
        preds = []
        for i in [1, 2, 3]:
            mol_col = f"Top{i}"
            score_col = f"Top{i}_score"
            if mol_col in row and score_col in row:
                mol = row[mol_col]
                score = row[score_col]
                if pd.notna(mol) and pd.notna(score):
                    preds.append({
                        "substrate": str(mol),
                        "confidence": float(score)
                    })
        return preds

    # 4. 构建 JSON 输出
    results = []
    for _, row in merged.iterrows():
        results.append({
            "id": row.get("id", ""),
            "domain_sequence": row.get("domain_sequence", ""),          # 直接用 CSV 的 domain_sequence 列
            "binding_pocket": row.get("binding_pocket", ""),            # 直接用 CSV 的 binding_pocket 列
            "binding_pocket_positions": parse_positions(
                row.get("binding_pocket_positions", "")
            ),
            "predictions": parse_predictions(row)
        })

    # 5. 写出 JSON 文件
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.debug(f"JSON saved to {json_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}")


def main():
    args = parse_arguments()

    # 解析设备（必须在所有使用 device 的地方之前）
    device = resolve_device(args.device)

    # 确保输出目录存在（兼容纯文件名情况）
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(output_dir or ".")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    import time
    t0 = time.time()

    logger.info("\n\033[1;36m" + "="*55)
    logger.info("  DeepAden  │  Step 2/2 — Substrate Prediction")
    logger.info("=" * 55 + "\033[0m")
    logger.info(f"  \033[90mInput  :\033[0m {args.input}")
    logger.info(f"  \033[90mDevice :\033[0m {device}\n")
    
    try:
        # 1. Feature computation
        logger.info(f"  \033[90m[»]\033[0m Computing pocket embeddings (ESM2)...")
        precompute_protein_features(args.input, output_dir=args.protein_dir, device=device)
        logger.info(f"  \033[90m[✓]\033[0m Pocket embeddings ready")
        
        # 2. Molecule features (compute only if needed)
        if not os.path.exists(args.molecule_dir):
            os.makedirs(args.molecule_dir, exist_ok=True)
            logger.info(f"  \033[90m[»]\033[0m Computing molecule features...")
            precompute_molecule_features(args.molecules, output_dir=args.molecule_dir, device=device)
            logger.info(f"  \033[90m[✓]\033[0m Molecule features ready")
        
        # 3. Model initialization
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Model weights not found at {args.weights}")
            
        model = ContrastiveModel()
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        logger.info(f"  \033[90m[✓]\033[0m Model loaded: Contrastive (KDE-calibrated)")
        
        # 5. Data preparation
        protein_ids = pd.read_csv(args.input)['id']
        mol_labels = pd.read_csv(args.molecules)['label']
        logger.info(f"  \033[90m[»]\033[0m Ranking {len(mol_labels)} molecules for {len(protein_ids)} query sequence(s) — top-{args.top_k}...")
        
        # 6. Retrieval process
        results = perform_retrieval(
            model=model,
            protein_ids=protein_ids,
            molecule_labels=mol_labels,
            kde_path=args.calibrator_path,
            top_k=args.top_k,
            device=device
        )
        
        # 7. Save results (CSV)
        save_results(results, args.output)

        # 8. 同步生成 JSON
        json_output = os.path.splitext(args.output)[0] + ".json"
        csv_to_json(
            pocket_csv_path=args.input,
            prediction_csv_path=args.output,
            json_path=json_output,
            logger=logger
        )

        elapsed = time.time() - t0
        logger.info(f"  \033[90m[✓]\033[0m Output → {args.output}")
        logger.info(f"  \033[90m[✓]\033[0m Output → {json_output}")
        logger.info(f"\n\033[1;32m  Step 2/2 completed in {elapsed:.1f}s\033[0m")
#         logger.info("\033[1;36m" + "="*55 + "\033[0m\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
