"""
This script will take a single sorted BED file of enhancer regions,
extend each region out by floor(region_len/2) on each side. The size of the original
region can be recalculated as ceil(extended_len/2) and the size of each extension can be
calculated as ((extended_len - orig_len)/2)

The list of expanded regions is then scanned for overlaps. If an overlap is found so


-----

This script takes a single sorted BED file of enhancer regions. This original list of enhancer regions
is retained to mark positive examples. Each region is then extended by it's size on each of the 5' and 3' end,
which are marked as adjacent negative examples. If any of the extended regions overlap with another record's
enhancer region, those bases are marked as positive examples, thus giving rise to the possibility of a single
record possible containing part of more than one enhancer.

Positive examples are given a value of 1 and negative examples are given a value of 0. Those kmers which span
over both positive and negative bases will be given an average score for the whole kmer (in this sense, all of
the kmers actually get an average score, but most will have an average of 1 or 0).
"""
import argparse
from collections import defaultdict

import pysam
import h5py
from intervaltree import IntervalTree
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


def get_kmer_score(chrom_enhancer_regions, start, stop, as_int=False):
    """
    Calculates the average enhancer score of a region. This is only non-trivial for those k-mers which
    span a change from non-enhancer region to an enhancer region.

    Ex. AGGCT
        00111  -> score: 0.6

        GTATC
        10000  -> score: 0.2

        GGTGG
        11111  -> score: 1.0

    :param chrom_enhancer_regions: IntervalTree Enhancer regions for a single chromosome
    :param start: int Start of the kmer region
    :param stop: int End of the kmer region, non-inclusive
    :param as_int: bool Whether to return as a rounded integer or as a float
    :return: the enhancer score of the kmer
    """
    interval_hit = chrom_enhancer_regions[start:stop]
    if not interval_hit:
        return 0 if as_int else 0.0

    interval_hit = list(interval_hit)[0]
    if start >= interval_hit.begin and stop <= interval_hit.end:
        return 1 if as_int else 1.0

    score = [1 if chrom_enhancer_regions[i] else 0 for i in range(start, stop)]
    score = sum(score) / len(score)
    return int(round(score)) if as_int else score


def extract_kmers(corpus, k=5, stride=2):
    """
    Given a set of nucleotides, break each sequence up into kmers of size k

    :param corpus: iterable All nucleotide sequences to be converted to kmers
    :param k: int Size of each kmer
    :param stride: int Sliding window size when breaking sequence into kmers
    :return: list For each sequence, a list of kmers of size k
    """
    kmers_corpus = list()
    for seq in corpus:
        seq_kmers = list()
        for i in range(0, len(seq) - k + 1, stride):
            seq_kmers.append(seq[i:i + k])
        kmers_corpus.append(seq_kmers)
    return kmers_corpus


def extend_region(start, stop, extend_ratio=1.0):
    """
    Given a start and a stop, extend on both ends by a ratio (default 1.0)

    :param start: int Start of the region
    :param stop: int End of the region, non-inclusive
    :param extend_ratio: float Ratio of the original size to extend on both ends
    :return: (int, int) A new start and a new stop
    """
    region_len = stop - start
    new_start = start - int(region_len * extend_ratio)
    new_stop = stop + int(region_len * extend_ratio)
    return new_start, new_stop


# Gather arguments from the user
parser = argparse.ArgumentParser()
parser.add_argument('--input-bed', help='Single BED file containing enhancer regions')
parser.add_argument('--reference-fasta', help='Indexed fasta file of the reference genome')
parser.add_argument('--outfile', default='rnn_data.h5', help='Path to file for output')
parser.add_argument('--extension-ratio', type=float, default=1.0, help=('Ratio of original region to '
                                                                        'extend on both sides'))
args = vars(parser.parse_args())
ref_fasta = pysam.FastaFile(args['reference_fasta'])

# Populate enhancer regions
print('Populating enhancer regions')
enhancer_regions = defaultdict(lambda: IntervalTree())
with open(args['input_bed']) as input_bed:
    for record in input_bed:
        chrom, start, stop, *_ = record.strip().split('\t')
        enhancer_regions[chrom][int(start):int(stop)] = 1

labeled_samples = list()
print('Calculating scores')
with open(args['input_bed']) as input_bed:
    for record in input_bed:
        chrom, start, stop, *_ = record.strip().split('\t')
        extended_start, extended_stop = extend_region(int(start), int(stop), extend_ratio=args['extension_ratio'])
        extended_seq = ref_fasta.fetch(chrom, extended_start, extended_stop)

        # Get kmer scores (right now size is hard-coded, but could easily be added to arguments)
        k = 5
        stride = 2
        scores = list()
        for i in range(0, len(extended_seq) - k + 1, stride):
            scores.append(get_kmer_score(enhancer_regions[chrom], extended_start + i, extended_start + i + k))
        kmers = extract_kmers([extended_seq])[0]
        labeled_samples.append((kmers, scores))

print('Creating word vectors')
corpus = [c for c, l in labeled_samples]
embedding = Word2Vec(corpus, size=50, min_count=1)  # WordVec size is hard-coded here, but should be an argument

print('Creating training, validation, test sets')
# Split into 80-10-10
new_labeled_samples = [([embedding.wv[s] for s in c], l) for c, l in labeled_samples]
X, y = list(zip(*new_labeled_samples))
X_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.2)
X_test, X_valid, y_test, y_valid = train_test_split(x_, y_, test_size=0.5)

print('Writing out dataset')
with h5py.File(args['outfile'], 'w') as out:
    out.create_dataset('X_train', data=X_train)
    out.create_dataset('X_valid', data=X_valid)
    out.create_dataset('X_test', data=X_test)
    out.create_dataset('y_train', data=y_train)
    out.create_dataset('y_valid', data=y_valid)
    out.create_dataset('y_test', data=y_test)
