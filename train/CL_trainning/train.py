python protein_feature.py

python molecule_feature.py

python trainer.py \
  --data_file ../../data/train_set.csv \
  --protein_feature_dir ../../data/protein_data \
  --molecule_feature_dir ../../data/molecule_data \
  --batch_size 128 \
  --patience 5 \
  --evaluate