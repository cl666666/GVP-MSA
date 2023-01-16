import torch
import sys
sys.path.append('/home/chenlin/src/esm')
import esm
import string
import numpy as np
from Bio import SeqIO
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import copy
import torch.nn.functional as F
import math

from scipy.spatial.distance import squareform, pdist, cdist

def get_esm_msa1b_rep(a2m_path='TEM.a2m',num_seqs=256,device='cuda:5',delete_first_line = False):
    processed_alignment, position_converter, unprocessed_refseq = load_alignment(a2m_path)
    assert len(processed_alignment) > 1, "Expected alignment, but received fasta"
    
    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().to(device)
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
    
    inputs = greedy_select(processed_alignment, num_seqs=num_seqs) 
    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
    if delete_first_line:
        msa_transformer_batch_tokens = msa_transformer_batch_tokens[:,1:,:]
    with torch.no_grad():
        results = msa_transformer(msa_transformer_batch_tokens, repr_layers = [12])
        # all_temp_logits = results["logits"][:,0]
        all_temp_reprs = results["representations"][12][:,0][:,1:,:]
    
    seqlen = len(unprocessed_refseq)
    out_rep = torch.zeros(1,seqlen,768)
    
    for key,value in position_converter.items():
        out_rep[:,key,:] = all_temp_reprs[:,value,:] 
        # out_rep.shape = torch.Size([1, 263, 768])
    return out_rep.to(device)


def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def load_alignment(input_filename):
    """
    Given the path to an alignment file, loads the alignment, then processes it
    to remove unaligned columns. The processed alignment is then ready to be 
    passed to the tokenization function of the MsaTransformer.
    
    Parameters
    ----------
    input_filename: str: Path to the alignment. 
    
    Returns
    -------
    processed_alignment: list of lists: Contents of an a2m or a3m alignment file
        with all unaligned columns removed. This is formatted for passage into
        the tokenization function of the MsaTransformer.
    old_to_new_pos: dict: A dictionary that relates the old index in the reference
        sequence to the new position in the processed reference sequence.
    """
    # Set up deletekeys. This code is taken right from ESM
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    
    # Load the unprocessed alignment
    unprocessed_alignment = [(record.description, str(record.seq))
                             for record in SeqIO.parse(input_filename, "fasta")]

    # Save the original reference sequence
    unprocessed_refseq = unprocessed_alignment[0][1]

    # Get a dictionary linking old position to processed position
    position_converter = build_old_to_new(unprocessed_refseq, deletekeys)

    # Process the alignment
    processed_alignment = process_alignment(unprocessed_alignment, deletekeys)
    
    # We only need the processed alignment and the dictionary of old to new
    return processed_alignment, position_converter,unprocessed_refseq


def build_old_to_new(unprocessed_refseq, deletekeys):
    """
    Processing an alignment with `process_alignment` changes the indices of the
    mutated positions relative to their original locations in the unprocessed
    sequence. This function builds a dictionary that relates the old index (in
    the unprocessed alignment) to the new index (in the processed alignment).
    
    Parameters
    ----------
    unprocessed_refseq: str: The first sequence in the unprocessed alignment. 
    deletekeys: dict: The keys to delete from all sequences in the unprocessed
        alignment. This includes all lowercase characters, ".", and "*". The
        format is {character: None} for each character to delete.
        
    Returns
    -------
    old_to_new_pos: dict: A dictionary that relates the old index in the reference
        sequence (!! 0-indexed !!) to the new position in the processed 
        reference sequence (!! also 0-indexed !!).
    """
    # Build a dictionary linking the old positions in the protein to
    # the new. Note that this dictionary is 0-indexed relative to the
    # protein sequence
    # Get the number of alphabetic characters in the reference
    n_capital_letters = sum((char.isalpha() and char.isupper()) 
                            for char in unprocessed_refseq)

    # Loop over each character in the unprocessed reference sequence
    seq_ind = -1
    processed_ind = -1
    old_to_new_pos = {}
    for char in unprocessed_refseq:
        
        # Check if the character is a letter and whether or not it is
        # in the deletekeys
        alpha_check = char.isalpha()
        delete_check = (char not in deletekeys)
        
        # If the character is a letter, increment the sequence index. Letters
        # are the only characters that match the original sequence
        if alpha_check:
            seq_ind += 1
            
        # If the character is not in the set of deletekeys, increment the
        # processed index. Characters not in the deletekeys are carried into
        # the processed sequences
        if delete_check:
            
            # Increment counter
            processed_ind += 1
            
            # Sanity check: If not a letter, then this must be "-"
            if not alpha_check:
                assert char == "-", "Unexpected character in reference sequence"
        
        # If the character is both alphabetic and not in the deletekeys, then
        # record it as a viable character that can be converted
        if alpha_check and delete_check:
            old_to_new_pos[seq_ind] = processed_ind 
            
    # Confirm that we captured all sequence elements that we could
    assert len(old_to_new_pos) == n_capital_letters
                
    return old_to_new_pos


def process_alignment(unprocessed_alignment, deletekeys):
    """
    This handles the input alignments to the MSA transformer. Specifically, it 
    reformats the alignment such that all unaligned columns are eliminated and
    duplicate sequences are deleted. Unaligned columns are those with "." and
    lowercase letters. The example code provided in ESM also omits the "*"
    character (see 
    https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb),
    so this character is also ommitted here for consistency. Note that, because
    a3m is just an a2m file format with all "." symbols removed (see page 26 of 
    the HHSuite docs: 
    http://sysbio.rnet.missouri.edu/bdm_download/DeepRank_db_tools/tools/DNCON2/hhsuite-2.0.16-linux-x86_64/hhsuite-userguide.pdf
    this conversion should handle both a2m and a3m files and convert them to the
    same output. This file 
    
    Parameters
    ----------
    unprocessed_alignment: list of lists: An unprocessed a2m or a3m alignment
        file formatted such that each entry is (description, sequence).
    deketekeys: dict: The keys to delete from all sequences in the unprocessed
        alignment. This includes all lowercase characters, ".", and "*". The
        format is {character: None} for each character to delete.
            
    Returns
    -------
    processed_alignment: list of lists: An a2m or a3m alignment file with all
        unaligned columns and duplicate sequences removed.
    """ 
    # Create the translation table
    translation = str.maketrans(deletekeys)
    
    # Loop over elements of the unprocessed alignment
    processed_alignment = []
    observed_seqs = []
    for desc, seq in unprocessed_alignment:

        # Translate and add to the processed alignment if it has
        # not previously been observed
        processed_seq = seq.translate(translation)
        if processed_seq not in observed_seqs:
            observed_seqs.append(processed_seq)
            processed_alignment.append((desc, processed_seq))
            
    # Confirm that all sequences are the same length
    testlen = len(processed_alignment[0][1])
    assert all(len(seq) == testlen for _, seq in processed_alignment)
    
    return processed_alignment
