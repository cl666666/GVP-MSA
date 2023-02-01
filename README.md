# GVP-MSA
Learning protein fitness landscapes with deep mutational scanning data from multiple sources.

## Overview
GVP-MSA is a deep learning model to learn the fitness landscapes, in which a 3D equivariant graph neural network GVP was used to extract features from protein structure, and a pre-trained model MSA Transformer was applied to embed MSA constraints. We describe a multi-protein training scheme to leverage the existing deep mutational scanning data from different proteins to help reason about the fitness landscape of a novel protein. Proof-of-concept trials are designed to validate this training scheme in three aspects: random and positional extrapolation for single variant effects, making zero-shot fitness predictions for novel proteins, and predicting higher-order variant effects from single variant effects. Our study also identified previously overlooked strong baselines, including a position specific averaging model for predicting single variant effects, and an additive model for predicting higher-order effects. 

## train single protein

    ```python
    # split randomly
    python train_single_protein_randomly.py --train_dataset_names TEM1 --n_ensembles 3  

    # split based on mutation position
    python train_single_protein_split_basedon_position.py --train_dataset_names KKA2_KLEPN_KAN18   --n_ensembles 3  
    ```
## train multi-task models based on different DMS datasets.

    python train_multi_protein_refmodel.py --train_dataset_names 'B3VI55_LIPSTSTABLE' 'BG_STRSQ' 'PTEN' 'AMIE_acet' 'HSP90' 'KKA2_KLEPN_KAN18'  --n_ensembles 1 --multi_model True 
    ```

## train finetune single protein
    ```python
    # split randomly
    python train_finetune_single_protein_randomly.py --train_dataset_names TEM1 --n_ensembles 3  --device "cuda:0" --load_model_path results/multi_protein_refmodel/B3VI55_LIPSTSTABLE~BG_STRSQ~PTEN~AMIE_acet~HSP90~KKA2_KLEPN_KAN18/model_fold0_ensemble0.pt --multi_model False 

    # split based on mutation position
    python train_finetune_single_protein_split_basedon_position.py --train_dataset_names avGFP --n_ensembles 3 --device "cuda:0" --load_model_path results/multi_protein_refmodel/B3VI55_LIPSTSTABLE~BG_STRSQ~PTEN~AMIE_acet~HSP90~KKA2_KLEPN_KAN18/ model_fold0_ensemble0.pt --multi_model False
    ```

## train zeroshot

    ```python
    python train_zeroshot.py --train_dataset_names 'KKA2_KLEPN_KAN18' 'BG_STRSQ' 'YAP1_WW1' --test_dataset_name 'TEM1' 
    ```

## Predicting higher-order variant effects from single variant effects by training with other DMS datasets with higher-order variant effects.

    ```python
    python train_single2multi.py --train_dataset_names 'GB1_2combo' 'TEM1' 'YAP1_WW1' 'AVGFP' --test_dataset_name 'FOS_JUN' 
    ```

