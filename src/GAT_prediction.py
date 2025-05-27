# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import os
from tqdm import tqdm
from Bio import SeqIO
from Bio.Align import PairwiseAligner
import argparse
import logging
from datetime import datetime
import concurrent.futures

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define GAT model
class GAT(nn.Module):
    def __init__(self, config, esm_input_dim, physio_input_dim):
        super(GAT, self).__init__()
        self.config = config

        self.mlp_aa = MLP(input_dim=esm_input_dim, hidden_dim=2 * config.fusion_dim, output_dim=config.fusion_dim)
        self.mlp_pc = MLP(input_dim=physio_input_dim, hidden_dim=2 * config.fusion_dim, output_dim=config.fusion_dim)
        
        self.conv1 = GATConv(config.fusion_dim, config.hidden_channels, heads=config.heads, dropout=config.dropratio)
        self.conv2 = GATConv(config.hidden_channels * config.heads, config.hidden_channels, heads=config.heads_intermediate, dropout=config.dropratio)
        self.conv3 = GATConv(config.hidden_channels * config.heads_intermediate, config.hidden_channels, heads=config.heads_intermediate, dropout=config.dropratio)
        self.conv4 = GATConv(config.hidden_channels * config.heads_intermediate, config.out_channels, heads=config.heads_final, concat=False, dropout=config.dropratio)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.config.dropratio, training=self.training)
        x = self.conv4(x, edge_index)
        return x

# Configuration class
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

def read_fasta_to_dataframe(fasta_file, logger):
    """Read FASTA file into a pandas DataFrame."""
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
    """Perform predictions using the trained model."""
    predicteds = []
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for seq_id in tqdm(seq_df.id, desc="Predicting"):
            pt_file = os.path.join(feature_dir, f"feature_dir/{seq_id}.pt")
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
    """Slice protein sequences into windows of specified size."""
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
    """Calculate alignment score between two sequences."""
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
    """Process predicted labels and align with original sequence."""
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

    # Group consecutive slices
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

    # Process each group with different window sizes
    group_data = {}
    for i, group in enumerate(groups):
        group_name = f"group{i+1}"
        group_data[group_name] = list(set(group))

    slice_sizes = {'group1': 11, 'group2': 4, 'group3': 5}
    for group_name, window_size in slice_sizes.items():
        if group_name in group_data:
            group_data[group_name] = slice_group(group_data[group_name], window_size)

    # Prepare output data
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
    """Process a single sequence for alignment matching."""
    seq_id, sequence, prediction, target_df = args
    
    try:
        if prediction is None:
            return None
        
        # Convert prediction to list format
        if isinstance(prediction, torch.Tensor):
            pred_labels = prediction.tolist()
        elif isinstance(prediction, list):
            pred_labels = prediction
        else:
            pred_labels = list(prediction)
            
        if len(pred_labels) != len(sequence):
            return None

        # Process and align slices
        slices = process_and_align(pred_labels, sequence)
        if not slices:
            return None

        slices_df_seq = pd.DataFrame(slices)
        best_matches = []
        
        # Process each group
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

            # Find best match for each slice
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
    """Process sequences in parallel using multiprocessing."""
    try:
        target_df = pd.read_csv(target_slices_path)
    except Exception as e:
        logger.error(f"Error reading reference slices: {e}")
        raise

    # Prepare data for multiprocessing
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
        
    # Process results
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
    """Main pipeline execution function."""
    # Initialize logging
    logger = logging.getLogger('GAT_Pipeline')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    
    # Set up console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("========== GAT Pipeline Started ==========")
    logger.info(f"Feature Directory: {args.feature}")
    logger.info(f"FASTA Path: {args.fasta}")
    logger.info(f"Reference Path: {args.reference}")
    logger.info(f"Output Path: {args.output}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    config = Config()
    model = GAT(config=config, esm_input_dim=config.input_dim_esm, physio_input_dim=config.input_dim_physio)
    
    # Load model weights
    model_weights_path = "./model/A_78_data_best_model_edge_mask_999_979.pth"
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
    
    # Read input data
    seq_df = read_fasta_to_dataframe(args.fasta, logger)
    
    # Perform predictions
    predicteds = perform_predictions(model, config, device, seq_df, args.feature, logger)

    # Process sequences and align
    best_match_align = process_sequences_individually(
        seq_df=seq_df,
        predicteds=predicteds,
        target_slices_path=args.reference,
        logger=logger,
        num_processes=args.processes
    )

    # Post-processing
    logger.info("Post-processing results...")
    best_match_align['labels'] = best_match_align['Sequence_ID'].apply(
        lambda x: x.split('|')[-1] if '|' in x else x
    )
    
    merged_df = best_match_align.merge(seq_df, left_on='Sequence_ID', right_on='id', how='left')
    
    # Define masking positions for group1
    def replace_positions(s, group_name, positions=[1, 2, 3, 8]):
        """Mask specified positions for group1."""
        if pd.isna(s):
            return ''
        
        if group_name == 'group1':
            s = list(s)
            for pos in positions:
                if pos < len(s):
                    s[pos] = '-'
        return ''.join(s)
    
    # Prepare final results
    result = []
    grouped = merged_df.groupby('Sequence_ID')
    
    for seq_id, group in grouped:
        label = group['labels'].iloc[0]
        sequence = group['sequence'].iloc[0]
        
        result_dict = {
            'id': seq_id,
            'labels': label,
            'target_1': (None, (None, None)),
            'target_2': (None, (None, None)),
            'target_3': (None, (None, None)),
            'target_4': (None, (None, None)),
            'seq': sequence,
            'pocket': ''
        }
        
        pocket = []
        for i, group_name in enumerate(['group1', 'group2', 'group3', 'group4'], start=1):
            sub_group = group[group['Group'] == group_name]
            target_key = f'target_{i}'
            
            if sub_group.empty:
                continue
            
            row = sub_group.iloc[0]
            amino_slice = row['Amino_Acid_Slice']
            modified_slice = replace_positions(amino_slice, group_name)
            
            start_idx = row['Start_Index']
            end_idx = row['End_Index']
            
            result_dict[target_key] = (modified_slice, (start_idx - 1, end_idx - 1))
            
            if modified_slice:
                pocket.append(modified_slice.replace('-', ''))
        
        result_dict['pocket'] = ''.join(pocket)
        result.append(result_dict)
    
    # Save final results
    final_df = pd.DataFrame(result, columns=[
        'id', 'labels', 'target_1', 'target_2', 'target_3', 'target_4', 'seq', 'pocket'
    ])

    output_filename = "pocket_predictions.csv"
    output_path = os.path.join(args.output, output_filename)

    try:
        final_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise
    
    logger.info("========== GAT Pipeline Completed ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein pocket prediction using GAT model.")
    parser.add_argument('--feature', type=str, required=True, help='Directory containing feature files.')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file path.')
    parser.add_argument('--reference', type=str, required=True, help='Reference slices CSV file path.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results.')
    parser.add_argument('--processes', type=int, default=0, 
                       help='Number of processes for parallel processing (0=auto).')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    main(args)