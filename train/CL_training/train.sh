# python trainer.py \
#   --data_file ./train_data_to50_augmented_v2.csv \
#   --protein_feature_dir ./protein_data \
#   --molecule_feature_dir ./molecule_data \
#   --epochs 100 \
#   --batch_size 512 \
#   --lr 1e-4 \
#   --temperature 0.1 \
#   --lambda_contrastive 1.0

python trainer.py \
    --data_file ./train_data_to50_augmented_v2.csv \
    --protein_feature_dir ./protein_data \
    --molecule_feature_dir ./molecule_data \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-4 \
    --temperature 0.1 \
    --k_folds 10 \
    --save_dir contrastive_kfold_summary 