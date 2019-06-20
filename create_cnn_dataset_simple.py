import os
import argparse
import subprocess
import random

from tempfile import TemporaryDirectory
import numpy as np
import pysam
import h5py
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json


OHE_SEQ = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}


def one_hot_encode_sequence(seq):
    return np.array([OHE_SEQ[s] for s in seq.upper()])


def extract_sequences(full_seq, subseq_len=50, window=5):
    if len(full_seq) % window != 0:
        raise ValueError('Window size does not divide sequence length')
    if subseq_len > len(full_seq):
        raise ValueError('Subsequence length is greater than full sequence length')

    return np.array([one_hot_encode_sequence(full_seq[i:i + subseq_len]) for i in range(0, len(full_seq) - subseq_len, window)])


parser = argparse.ArgumentParser()
parser.add_argument('--rep1', required=True)
parser.add_argument('--rep2', required=True)
parser.add_argument('--bedtools-path', default='bedtools')
parser.add_argument('--expand-to', type=int, default=500)
parser.add_argument('--input-feature-size', type=int, default=50)
parser.add_argument('--window-size', type=int, default=5)
parser.add_argument('--reference-fasta', help='Path to genome fasta', required=True)
parser.add_argument('--genome-sizes', help='Genome sizes file (BED format)', required=True)
parser.add_argument('--negative-input-ratio', type=float, default=1.0,
                    help='Ratio of generated negative regions relative to found positive regions')
parser.add_argument('--output', default='dataset.h5')
args = vars(parser.parse_args())

EXPAND = int(args['expand_to'] / 2)
tmpdir = TemporaryDirectory()

# Intersect both replicates
print('Intersecting replicates')
subprocess.call('{bedtools} intersect -a {rep1} -b {rep2} -f 0.5 -r > {output}'.format(
    bedtools=args['bedtools_path'],
    rep1=args['rep1'],
    rep2=args['rep2'],
    output=os.path.join(tmpdir.name, 'intersected.bed')
), shell=True)

# Expand midpoint of intersected regions to 500bp
print('Expanding midpoint, phase 1')
with open(os.path.join(tmpdir.name, 'expd_bed1.bed'), 'w') as expanded_bed1:
    for record in open(os.path.join(tmpdir.name, 'intersected.bed')):
        chrom, start, stop = record.strip().split('\t')[:3]
        midpoint = int(np.mean([int(start), int(stop)]))
        expanded_bed1.write('\t'.join([chrom, str(midpoint - EXPAND), str(midpoint + EXPAND)]) + '\n')

# Sort intersected regions
print('Sorting intersected regions')
subprocess.call('{bedtools} sort -i {input} > {output}'.format(
    bedtools=args['bedtools_path'],
    input=os.path.join(tmpdir.name, 'expd_bed1.bed'),
    output=os.path.join(tmpdir.name, 'expd_bed1_sorted.bed')
), shell=True)

# Merge intersected regions to remove any overlap
print('Merging overlaps')
subprocess.call('{bedtools} merge -i {input} > {output}'.format(
    bedtools=args['bedtools_path'],
    input=os.path.join(tmpdir.name, 'expd_bed1_sorted.bed'),
    output=os.path.join(tmpdir.name, 'expd_bed1_merged.bed')
), shell=True)

# Expand midpoint of merged intersected regions to 500bp
print('Expanding midpoints, phase 2')
with open(os.path.join(tmpdir.name, 'bed_final.bed'), 'w') as bed_final:
    for record in open(os.path.join(tmpdir.name, 'expd_bed1_merged.bed')):
        chrom, start, stop = record.strip().split('\t')[:3]
        midpoint = int(np.mean([int(start), int(stop)]))
        bed_final.write('\t'.join([chrom, str(midpoint - EXPAND), str(midpoint + EXPAND)]) + '\n')

print('Sorting intersected regions')
subprocess.call('{bedtools} sort -i {input} > {output}'.format(
    bedtools=args['bedtools_path'],
    input=os.path.join(tmpdir.name, 'bed_final.bed'),
    output=os.path.join(tmpdir.name, 'bed_final.sorted.bed')
), shell=True)

# Generate complement file
print('Generating complement file')
subprocess.call('{bedtools} complement -i {input} -g {genome_sizes} > {output}'.format(
    bedtools=args['bedtools_path'],
    input=os.path.join(tmpdir.name, 'bed_final.sorted.bed'),
    genome_sizes=args['genome_sizes'],
    output=os.path.join(tmpdir.name, 'bed_final_complement.bed')
), shell=True)

ref = pysam.FastaFile(args['reference_fasta'])
# Generate input data sequences
# [1, 0] = enhancer region
# [0, 1] = not enhancer region
print('Generating input sequences')
X, y, input_i = list(), list(), 1
for record in open(os.path.join(tmpdir.name, 'bed_final.sorted.bed')):
    input_i += 1
    if input_i % 1000 == 0:
        print('Generated {} data points'.format(input_i))
    chrom, start, stop = record.strip().split('\t')[:3]
    input_seqs = extract_sequences(ref.fetch(chrom, int(start), int(stop)))
    X.extend(input_seqs)
    # Format as "[region_string].[kmer_id]"
    region_string = ':'.join([chrom, start, stop]) + '.{}'
    y.extend([[1, 0, region_string.format(i)] for i in range(len(input_seqs))])

print('Generating complement regions')
complement_regions = {r for r in open(os.path.join(tmpdir.name, 'bed_final_complement.bed'))}
for j in range(1, int(input_i * args['negative_input_ratio'])):
    if j % 1000 == 0:
        print('Generated {} complement data points'.format(j))
    chrom, start, stop = random.sample(complement_regions, k=1)[0].strip().split('\t')
    if int(stop) - int(start) < args['expand_to']:
        continue
    midpoint = random.randint(int(start) + EXPAND, int(stop) - EXPAND)
    input_seqs = extract_sequences(ref.fetch(chrom, midpoint - EXPAND, midpoint + EXPAND))
    X.extend(input_seqs)
    y.extend([[0, 1, ''] for i in range(len(input_seqs))])

print('Split into train-validation-test (80/10/10) sets')
X_train, X_interim, y_train, y_interim = train_test_split(np.array(X), np.array(y), test_size=0.2)
X_test, X_valid, y_test, y_valid = train_test_split(X_interim, y_interim, test_size=0.5)

print('Generate sequence ID map, remove from labels')
sequence_id_map = defaultdict(list)
for dataset_label, dataset in (('train', y_train), ('valid', y_valid), ('test', y_test)):
    for i, label in enumerate(dataset):
        if label[2]:
            region_string, seq_id = label[2].split('.')
            sequence_id_map[region_string].append((int(seq_id), dataset_label, i))

# Remove region string and kmer ID from y labels
y_train = (y_train[:, :2]).astype(int)
y_valid = (y_valid[:, :2]).astype(int)
y_test = (y_test[:, :2]).astype(int)

print('Writing dataset')
with h5py.File(args['output'], 'w') as out:
    out.create_dataset('X_train', data=X_train)
    out.create_dataset('X_valid', data=X_valid)
    out.create_dataset('X_test', data=X_test)
    out.create_dataset('y_train', data=y_train)
    out.create_dataset('y_valid', data=y_valid)
    out.create_dataset('y_test', data=y_test)
    out.create_dataset('sequence_id_map', data=json.dumps(sequence_id_map))


print('Done')
tmpdir.cleanup()
