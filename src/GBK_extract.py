#!/usr/bin/env python3

import sys
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd
import numpy as np
import re


def get_CDS_features(gbk_file):
    records = SeqIO.parse(gbk_file, 'genbank')
    CDS_features = []
    
    for record in records:
        result = re.search('[a-zA-Z]+\sbio.*?gene cluster', record.description)
        if result:
            product = result.group(0)
        else:
            product = None
            
        accession = record.annotations.get('accessions', [''])[0]
        organism = record.annotations.get('organism', [''])
        
        for feature in record.features:
            if feature.type == 'CDS':
                # protein_id = feature.qualifiers.get('protein_id', [''])[0]
                locus_tags = feature.qualifiers.get('locus_tag', [''])
                sequence = feature.qualifiers.get('translation', [''])[0]
                
                if not sequence:
                    continue
                    
                adomain_feature = {
                    'accession': accession,
                    'product': product,
                    'organism': organism,
                    # 'protein_id': protein_id,
                    'locus_tags': locus_tags,
                    'sequence': sequence
                }
                CDS_features.append(adomain_feature)
    
    return CDS_features


def write_fasta(CDS_features, output_file):
    seq_records = []
    
    for i, feature in enumerate(CDS_features):
        # seq_id = feature['protein_id'] if feature['protein_id'] else f"CDS_{i+1}"
        seq_id = feature['locus_tags'][0] if feature['locus_tags'][0] else f"CDS_{i+1}"
        
        seq_record = SeqRecord(
            Seq(feature['sequence']),
            id=seq_id,
            description=""
        )
        seq_records.append(seq_record)
    
    with open(output_file, 'w') as handle:
        SeqIO.write(seq_records, handle, 'fasta')


def main():
    parser = argparse.ArgumentParser(description='Extract CDS sequences from GenBank files and save as FASTA')
    parser.add_argument('input_gbk', help='Input GenBank file path')
    parser.add_argument('output_fasta', help='Output FASTA file path')
    
    args = parser.parse_args()
    
    try:
        CDS_features = get_CDS_features(args.input_gbk)
        
        if not CDS_features:
            with open(args.output_fasta, 'w') as f:
                pass
            return
            
        write_fasta(CDS_features, args.output_fasta)
            
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_gbk}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
