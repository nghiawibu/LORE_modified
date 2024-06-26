nohup python main.py ctdet_mid \
	--dataset table_mid \
	--exp_id training_wireless \
	--dataset_name PTN \
	--image_dir /dev2/data/LORE/train \
	--wiz_2dpe \
	--wiz_stacking \
	--tsfm_layers 4 \
	--stacking_layers 4 \
	--batch_size 4 \
	--master_batch 4 \
	--arch dla_34 \
	--lr 1e-4 \
	--K 500 \
	--MK 1000 \
	--num_epochs 200 \
	--lr_step '100, 160' \
	--gpus 0 \
	--num_workers 8 \
	--val_intervals 10 \
	--resume \
    --load_model ../exp/ctdet_mid/training_wireless/model_last.pth \
    --load_processor ../exp/ctdet_mid/training_wireless/processor_last.pth > log.txt