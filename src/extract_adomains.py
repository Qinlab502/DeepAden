import sys
from collections import defaultdict
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import pandas as pd

def parse_hmmscan(hmmscan_file):
    domains = []
    with open(hmmscan_file) as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.split()
                target = parts[0]       
                query = parts[3]        
                evalue = float(parts[6])
                score = float(parts[7])
                start = int(parts[17]) 
                end = int(parts[18])    
                domains.append({
                    'domain_id': target,
                    'query_id': query,
                    'start': start,
                    'end': end,
                    'evalue': evalue,
                    'score': score
                })
    return pd.DataFrame(domains)

def extract_domains(protein_file, domains_df, domain_name="AMP-binding"):
    records = SeqIO.to_dict(SeqIO.parse(protein_file, 'fasta'))
    domain_records = []
    
    if domains_df.empty:
        return None
    
    domain_count = defaultdict(int)
    domains_df = domains_df.sort_values(['query_id', 'start'])
    
    for _, row in domains_df.iterrows():
        query_id = row['query_id']
        domain_count[query_id] += 1

        protein = records[query_id]
        domain_seq = protein.seq[row['start']-1:row['end']]
        
        seq_id = f"{query_id}|{domain_name}.{domain_count[query_id]}|{row['start']}-{row['end']}"
        
        record = SeqRecord(
            Seq(str(domain_seq)),
            id=seq_id,
            description=""
        )
        domain_records.append(record)
    
    return domain_records

def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_adomains.py input.dom proteins.fasta output.fasta")
        sys.exit(1)
    
    dom_file = sys.argv[1]
    protein_file = sys.argv[2]
    output_file = sys.argv[3]
    
    domains_df = parse_hmmscan(dom_file)
    
    # Filter for only AMP-binding domains
    amp_domains_df = domains_df[domains_df['domain_id'].str.contains('AMP-binding')]
    
    if amp_domains_df.empty:
        # print("Warning: No adenylation (A) domains detected")
        sys.exit(0)
    
    domain_records = extract_domains(protein_file, amp_domains_df)
    
    if domain_records:
        with open(output_file, 'w') as f:
            SeqIO.write(domain_records, f, 'fasta')
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()