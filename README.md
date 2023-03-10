# GVP-MSA
Learning protein fitness landscapes with deep mutational scanning data from multiple sources.

## Overview
GVP-MSA is a deep learning model to learn the fitness landscapes, in which a 3D equivariant graph neural network GVP was used to extract features from protein structure, and a pre-trained model MSA Transformer was applied to embed MSA constraints. We describe a multi-protein training scheme to leverage the existing deep mutational scanning data from different proteins to help reason about the fitness landscape of a novel protein. Proof-of-concept trials are designed to validate this training scheme in three aspects: random and positional extrapolation for single variant effects, making zero-shot fitness predictions for novel proteins, and predicting higher-order variant effects from single variant effects. Our study also identified previously overlooked strong baselines, including a position specific averaging model for predicting single variant effects, and an additive model for predicting higher-order effects. 

## Requirements
You need to have PyTorch installed to use this repository.
Also, esm is required to be installed from its github repository: https://github.com/facebookresearch/esm

## Data
Processed protein fitness data and its relevant MSA and protein structure information are available.
Only one example dataset TEM1 is provided in this github repository due to the size constraints. More datasets can be download from https://www.dropbox.com/s/yrutk32l21y6dhr/directed_evolution_input_datasets.tar.gz?dl=0. To test with these datasets, you need to uncompress the downloaded data and move them into the "input_data" directory.

## Running examples

### Simple averaging or additive models

    python ./simple_models/addition.py --dataset_name TEM1
    python ./simple_models/average.py --dataset_name TEM1

### Train GVP-MSA of a single protein

    # split randomly
    python train_single_protein_randomly.py --train_dataset_names TEM1 --n_ensembles 3  

    # split based on mutation position
    python train_single_protein_split_basedon_position.py --train_dataset_names TEM1 --n_ensembles 3  

### Investigate if the performance of protein-specific modeling can be improved by incorporating DMS data from other proteins

First, train GVP-MSA in multi-task framework as a reference model. In the multi-task model, the parameters of the bottom model are shared and the parameters of the top model are specific among different proteins.

    python train_multi_protein_refmodel.py --train_dataset_names 'B3VI55_LIPSTSTABLE' 'BG_STRSQ' 'PTEN' 'AMIE_acet' 'HSP90' 'KKA2_KLEPN_KAN18'  --n_ensembles 1 --multi_model True 

Then, model was finetuned by their own fitness data based on the reference multi-protein model.

    # split randomly
    python train_finetune_single_protein_randomly.py --train_dataset_names TEM1 --n_ensembles 3  --device "cuda:0" --load_model_path results/multi_protein_refmodel/B3VI55_LIPSTSTABLE~BG_STRSQ~PTEN~AMIE_acet~HSP90~KKA2_KLEPN_KAN18/model_fold0_ensemble0.pt --multi_model False 

    # split based on mutation position
    python train_finetune_single_protein_split_basedon_position.py --train_dataset_names TEM1 --n_ensembles 3 --device "cuda:0" --load_model_path results/multi_protein_refmodel/B3VI55_LIPSTSTABLE~BG_STRSQ~PTEN~AMIE_acet~HSP90~KKA2_KLEPN_KAN18/model_fold0_ensemble0.pt --multi_model False

### Performing zero-shot fitness prediction of novel proteins

    python train_zeroshot.py --train_dataset_names 'B3VI55_LIPSTSTABLE' 'BG_STRSQ' 'PTEN' 'AMIE_acet' 'HSP90' 'KKA2_KLEPN_KAN18' 'GB1_2combo' 'YAP1_WW1' 'AVGFP' 'FOS_JUN' --test_dataset_name 'TEM1'

### Predicting higher-order variant effects from single variant effects by training with other DMS datasets with higher-order variant effects.

    python train_single2multi.py --train_dataset_names 'GB1_2combo' 'FOS_JUN' 'YAP1_WW1' 'AVGFP' --test_dataset_name 'TEM1'
