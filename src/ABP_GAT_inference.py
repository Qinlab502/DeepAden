import os
import torch
import pickle
import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from deepAden_model import MLP, GAT
import sys
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration class for model hyperparameters
class Config:
    def __init__(self):
        self.hidden_channels = 64
        self.out_channels = 3
        self.heads = 32
        self.heads_intermediate = 16
        self.heads_final = 8
        self.dropratio = 0.5
        self.input_dim_esm = 1280
        self.input_dim_physio = 18
        self.fusion_dim = 512

# Feature Processing Functions
def create_pyg_data_inference(output_dir: str, logger) -> None:
    esm_dir = os.path.join(output_dir, 'emb_dir')
    pf_dir = os.path.join(output_dir, 'pf_dir')
    ei_dir = os.path.join(output_dir, 'ei_dir')
    pyg_dir = os.path.join(output_dir, 'pyg_dir')

    # Ensure required input directories exist
    for dir_path in [esm_dir, pf_dir, ei_dir]:
        if not os.path.exists(dir_path):
            logger.error(f"Required directory not found: {dir_path}")
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    # Always remove existing pyg_dir and create a fresh one
    if os.path.exists(pyg_dir):
        logger.info(f"Removing existing directory: {pyg_dir}")
        shutil.rmtree(pyg_dir)
    os.makedirs(pyg_dir, exist_ok=True)
    logger.info(f"Processing features into PyG format. Output will be saved in {pyg_dir}.")

    seq_ids = set(
        file[:file.rfind('_')] for file in os.listdir(esm_dir) if file.endswith('_emb.pkl')
    )
    logger.info(f"Found {len(seq_ids)} sequences for feature processing")

    for seq_id in tqdm(seq_ids, desc="Creating PyG data objects"):
        esm_path = os.path.join(esm_dir, f"{seq_id}_emb.pkl")
        pf_path = os.path.join(pf_dir, f"{seq_id}_pf.pkl")
        ei_path = os.path.join(ei_dir, f"{seq_id}_ei.pkl")
        if not (os.path.exists(esm_path) and os.path.exists(pf_path) and os.path.exists(ei_path)):
            logger.warning(f"Warning: Sequence {seq_id} is missing features or edge index. Skipping.")
            continue
        try:
            with open(esm_path, 'rb') as f:
                esm_emb = pickle.load(f)
            with open(pf_path, 'rb') as f:
                physio_feat = pickle.load(f)
            with open(ei_path, 'rb') as f:
                edge_idx = pickle.load(f)
            data = Data(
                x_esm=torch.tensor(esm_emb, dtype=torch.float),
                x_physio=torch.tensor(physio_feat, dtype=torch.float),
                edge_index=torch.tensor(edge_idx, dtype=torch.long)
            )
            data.num_nodes = data.x_esm.shape[0]
            output_pt_path = os.path.join(pyg_dir, f"{seq_id}.pt")
            torch.save(data, output_pt_path)
        except Exception as e:
            logger.error(f"Error processing {seq_id}: {e}")
    logger.info(f"Completed processing all sequences. Output saved in '{pyg_dir}'.")

def read_fasta_to_dataframe(fasta_file, logger):
    sequences = {'id': [], 'sequence': []}
    try:
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences['id'].append(record.id)
                sequences['sequence'].append(str(record.seq))
        df = pd.DataFrame(sequences)
        logger.info(f"Successfully read {len(df)} sequences from FASTA file.")
        return df
    except Exception as e:
        logger.error(f"Error reading FASTA file: {e}")
        raise

def perform_predictions(model, config, device, seq_df, feature_dir, logger):
    predicteds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for seq_id in tqdm(seq_df.id, desc="Predicting"):
            pt_file = os.path.join(feature_dir, f"{seq_id}.pt")
            if not os.path.exists(pt_file):
                logger.warning(f"Feature file not found for {seq_id}")
                predicteds.append(None)
                continue
            try:
                data = torch.load(pt_file).to(device)
                x_aa = model.mlp_aa(data.x_esm)
                x_pc = model.mlp_pc(data.x_physio)
                x = x_aa + x_pc
                score = model(x, data.edge_index)
                _, predicted = torch.max(score.data, 1)
                predicteds.append(predicted.cpu())
            except Exception as e:
                logger.error(f"Error processing {seq_id}: {e}")
                predicteds.append(None)
    return predicteds

def slice_group(group, window_size):
    sliced_groups = []
    seen_slices = set()
    for slice_data in group:
        seq, start_index, end_index = slice_data
        if len(seq) >= window_size:
            for i in range(0, len(seq) - window_size + 1):
                new_slice = (seq[i:i + window_size], start_index + i, start_index + i + window_size - 1)
                if new_slice not in seen_slices:
                    sliced_groups.append(new_slice)
                    seen_slices.add(new_slice)
    return sliced_groups

def get_alignment_score_cached(seq1, seq2, match_score=10, mismatch_score=-1, gap_open=-100, gap_extend=-100):
    seq1 = str(seq1)
    seq2 = str(seq2)
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    alignments = aligner.align(seq1, seq2)
    return alignments[0].score if alignments else float('-inf')

def process_and_align(reference, original_sequence):
    predicted_sequence = ''.join(map(str, reference))
    window_size = 11
    slices_with_indices = [
        (predicted_sequence[i:i + window_size], i, i + window_size - 1) 
        for i in range(len(predicted_sequence) - window_size + 1)
    ]
    filtered_slices = [
        (s, start, end) for (s, start, end) in slices_with_indices 
        if s.count('1') + s.count('2') >= 3
    ]
    if not filtered_slices:
        return []
    groups = []
    current_group = []
    last_index = filtered_slices[0][1]
    for slice_data in filtered_slices:
        _, start_index, _ = slice_data
        if start_index - last_index > 1:
            if current_group:
                groups.append(current_group)
            current_group = []
        current_group.append(slice_data)
        last_index = start_index
    if current_group:
        groups.append(current_group)
    group_data = {}
    for i, group in enumerate(groups):
        group_name = f"group{i+1}"
        group_data[group_name] = list(set(group))
    slice_sizes = {'group1': 11, 'group2': 4, 'group3': 5}
    for group_name, window_size in slice_sizes.items():
        if group_name in group_data:
            group_data[group_name] = slice_group(group_data[group_name], window_size)
    table_data = []
    slice_index = 1
    for group_name, slices in group_data.items():
        for label_slice, start_index, end_index in slices:
            amino_acid_slice = original_sequence[start_index:end_index + 1]
            table_data.append({
                "Group": group_name,
                "Slice_Index": slice_index,
                "Label_Slice": label_slice,
                "Amino_Acid_Slice": amino_acid_slice,
                "Start_Index": start_index + 1,
                "End_Index": end_index + 1
            })
            slice_index += 1
    return table_data

def process_single_sequence(args):
    seq_id, sequence, prediction, target_df = args
    try:
        if prediction is None:
            return None
        if isinstance(prediction, torch.Tensor):
            pred_labels = prediction.tolist()
        elif isinstance(prediction, list):
            pred_labels = prediction
        else:
            pred_labels = list(prediction)
        if len(pred_labels) != len(sequence):
            return None
        slices = process_and_align(pred_labels, sequence)
        if not slices:
            return None
        slices_df_seq = pd.DataFrame(slices)
        best_matches = []
        for group in ['group1', 'group2', 'group3', 'group4']:
            group_slices = slices_df_seq[slices_df_seq['Group'] == group]
            if group_slices.empty:
                continue
            target_index = ['group1', 'group2', 'group3', 'group4'].index(group) + 1
            target_slice_col = f"target_s{target_index}"
            target_amino_col = f"target{target_index}"
            best_score = float('-inf')
            best_match_info = {
                'Sequence_ID': seq_id,
                'Group': group,
                'Target_ID': None,
                'Slice_Index': None,
                'Label_Slice': None,
                'Amino_Acid_Slice': None,
                'Target_Slice': None,
                'Target_Amino_Acid': None,
                'Start_Index': None,
                'End_Index': None,
                'Label_Score': None,
                'Amino_Score': None,
                'Combined_Score': None
            }
            for _, slice_row in group_slices.iterrows():
                label_slice = slice_row['Label_Slice']
                amino_slice = slice_row['Amino_Acid_Slice']
                slice_index = slice_row['Slice_Index']
                start_idx = slice_row['Start_Index']
                end_idx = slice_row['End_Index']
                for _, target_row in target_df.iterrows():
                    target_id = target_row['id']
                    target_s = target_row[target_slice_col]
                    target_amino = target_row[target_amino_col]
                    score_label = get_alignment_score_cached(label_slice, target_s)
                    score_amino = get_alignment_score_cached(amino_slice, target_amino)
                    combined_score = score_label + score_amino
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match_info.update({
                            'Target_ID': target_id,
                            'Slice_Index': slice_index,
                            'Label_Slice': label_slice,
                            'Amino_Acid_Slice': amino_slice,
                            'Target_Slice': target_s,
                            'Target_Amino_Acid': target_amino,
                            'Start_Index': start_idx,
                            'End_Index': end_idx,
                            'Label_Score': score_label,
                            'Amino_Score': score_amino,
                            'Combined_Score': combined_score
                        })
            if best_match_info['Combined_Score'] != float('-inf'):
                best_matches.append(best_match_info)
        return best_matches
    except Exception as e:
        return {'Sequence_ID': seq_id, 'Error': str(e)}

def process_sequences_individually(seq_df, predicteds, target_slices_path, logger, num_processes):
    try:
        target_df = pd.read_csv(target_slices_path)
    except Exception as e:
        logger.error(f"Error reading reference slices: {e}")
        raise
    tasks = []
    for seq_id, sequence, prediction in zip(seq_df.id, seq_df.sequence, predicteds):
        tasks.append((seq_id, sequence, prediction, target_df))
    best_matches = []
    max_workers = num_processes if num_processes > 0 else min(32, os.cpu_count() + 4)
    logger.info(f"Using {max_workers} worker processes")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_sequence, tasks), 
            total=len(tasks), 
            desc="Processing Sequences"
        ))
    for task, res in zip(tasks, results):
        seq_id = task[0]
        if res is None:
            logger.warning(f"No pocket sites found for {seq_id}")
            continue
        if isinstance(res, dict) and 'Error' in res:
            logger.error(f"Error processing {res['Sequence_ID']}: {res['Error']}")
            continue
        best_matches.extend(res)
    return pd.DataFrame(best_matches)

def main(args):
    logger = logging.getLogger('ABP_GAT_inference')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(console_handler)

    logger.info("========== ABP GAT Pipeline Started ==========")
    logger.info(f"Input FASTA: {args.fasta}")
    logger.info(f"Output Directory: {args.output}")
    logger.info(f"Feature Directory: {args.feature_dir}")
    logger.info(f"Model Path: {args.GAT}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    feature_dir = args.feature_dir
    pyg_pt_files_path = os.path.join(feature_dir, "pyg_dir")
    create_pyg_data_inference(feature_dir, logger)
    pt_files_path = pyg_pt_files_path
    logger.info(f"Using PyG data from {pt_files_path}")

    seq_df = read_fasta_to_dataframe(args.fasta, logger)

    # Initialize and load model
    config = Config()
    model = GAT(
        config=config,
        esm_input_dim=config.input_dim_esm,
        physio_input_dim=config.input_dim_physio,
        hidden_channels=config.hidden_channels,
        out_channels=config.out_channels,
        heads=config.heads,
        heads_intermediate=config.heads_intermediate,
        heads_final=config.heads_final,
        dropratio=config.dropratio
    )

    model_weights_path = args.GAT
    if not os.path.exists(model_weights_path):
        logger.error(f"Model weights not found at {model_weights_path}")
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Loaded model weights from {model_weights_path}")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise
    logger.info("Performing predictions...")
    predicteds = perform_predictions(model, config, device, seq_df, pt_files_path, logger)
    logger.info("Processing predictions and aligning with references...")
    best_match_align = process_sequences_individually(
        seq_df=seq_df,
        predicteds=predicteds,
        target_slices_path=args.reference,
        logger=logger,
        num_processes=args.processes
    )
    logger.info("Post-processing results...")

    def replace_positions(s, group_name, positions=[1, 2, 3, 8]):
        if pd.isna(s):
            return ''
        if group_name == 'group1':
            s = list(s)
            for pos in positions:
                if pos < len(s):
                    s[pos] = '-'
            return ''.join(s)
        return s

    if len(best_match_align) == 0:
        logger.warning("No matches found for any sequence!")
        final_df = pd.DataFrame(columns=[
            'id', 'region_1', 'region_2', 'region_3', 'region_4',
            'binding_pocket_positions', 'domain_sequence', 'binding_pocket'
        ])
    else:
        merged_df = best_match_align.merge(seq_df, left_on='Sequence_ID', right_on='id', how='left')
        result = []
        grouped = merged_df.groupby('Sequence_ID')
        for seq_id, group in grouped:
            sequence = group['sequence'].iloc[0]
            result_dict = {
                'id': seq_id,
                'region_1': '',
                'region_2': '',
                'region_3': '',
                'region_4': '',
                'domain_sequence': sequence,
                'binding_pocket': ''
            }
            pocket = []
            pocket_positions = []
            for i, group_name in enumerate(['group1', 'group2', 'group3', 'group4'], start=1):
                sub_group = group[group['Group'] == group_name]
                region_key = f'region_{i}'
                if sub_group.empty:
                    continue
                row = sub_group.iloc[0]
                amino_slice = row['Amino_Acid_Slice']
                modified_slice = replace_positions(amino_slice, group_name)
                start_idx = row['Start_Index']
                result_dict[region_key] = modified_slice
                if modified_slice:
                    unmasked = ''.join([aa for aa in modified_slice if aa != '-'])
                    pocket.append(unmasked)
                    for pos, aa in enumerate(modified_slice):
                        if aa != '-':
                            pocket_positions.append(str(start_idx + pos))
            result_dict['binding_pocket'] = ''.join(pocket)
            result_dict['binding_pocket_positions'] = ','.join(pocket_positions)
            result.append(result_dict)
        final_df = pd.DataFrame(result, columns=[
            'id', 'region_1', 'region_2', 'region_3', 'region_4',
            'binding_pocket_positions', 'domain_sequence', 'binding_pocket'
        ])

    output_filename = "ABP_prediction.csv"
    output_path = os.path.join(args.output, output_filename)
    try:
        final_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise
    logger.info("========== ABP GAT Pipeline Completed ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABP-GAT: Protein pocket prediction using GAT model.")
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file path.')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing feature files or where they will be created.')
    parser.add_argument('--reference', type=str, required=True, help='Reference slices CSV file path.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results.')
    parser.add_argument('--GAT', type=str, required=True, help='Path to the trained model weights file.')
    parser.add_argument('--processes', type=int, default=0, help='Number of processes for parallel processing (0=auto).')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main(args)