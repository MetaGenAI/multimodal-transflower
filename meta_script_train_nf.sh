exp=transflower1
./script_train_new.sh $exp --hparams_file=training/hparams/edf_gestures/${exp}.yaml --experiment_name ${exp}_edf_gestures3 --num_nodes 1 --max_epochs 50 --data_dir=./data/edf_extracted_data_rel/ --base_filenames_file base_filenames.txt $@
#./script_train_diffu.sh $exp --hparams_file=training/hparams/edf_gestures/${exp}.yaml --experiment_name ${exp}_edf_gestures2 --num_nodes 1 --max_epochs 80 --data_dir=./data/edf_extracted_data2/ --base_filenames_file base_filenames.txt $@
