# GVP-MSA
Learning protein fitness landscapes with deep mutational scanning data from multiple sources.


## train single protein

 
    # split randomly
    python train_single_protein_randomly.py --train_dataset_names KKA2_KLEPN_KAN18 --n_ensembles 3  

    # split based on mutation position
    python train_single_protein_split_basedon_position.py --train_dataset_names KKA2_KLEPN_KAN18   --n_ensembles 3  

## train multi-task models based on different DMS datasets.

    python train_multi_protein_refmodel.py --train_dataset_names 'B3VI55_LIPSTSTABLE' 'BG_STRSQ' 'PTEN' 'AMIE_acet' 'HSP90' 'KKA2_KLEPN_KAN18'  --n_ensembles 1 --multi_model True 

## train finetune single protein
    # split randomly
    python train_finetune_single_protein_randomly.py --train_dataset_names avGFP --n_ensembles3  --device "cuda:2" --load_model_path /home/chenlin/directed_evolution/gvp/src/submit/ multi_protein_refmodel/B3VI55_LIPSTSTABLE~BG_STRSQ~PTEN~AMIE_acet~HSP90~KKA2_KLEPN_KAN18/  model_fold0_ensemble0.pt --multi_model False --epochs 100

    # split based on mutation position
    python train_finetune_single_protein_split_basedon_position.py --train_dataset_names avGFP --n_ensembles 3 --device "cuda:0" --load_model_path results/multi_protein_refmodel/B3VI55_LIPSTSTABLE~BG_STRSQ~PTEN~AMIE_acet~HSP90~KKA2_KLEPN_KAN18/ model_fold0_ensemble0.pt --multi_model False

## train zeroshot

    python train_zeroshot.py --train_dataset_names 'KKA2_KLEPN_KAN18' 'BG_STRSQ' 'YAP1_WW1' --test_dataset_name 'TEM1' 

## train single2multi 

    python train_single2multi.py --train_dataset_names 'GB1_2combo' 'TEM1' 'YAP1_WW1' 'AVGFP' --test_dataset_name 'FOS_JUN' 

