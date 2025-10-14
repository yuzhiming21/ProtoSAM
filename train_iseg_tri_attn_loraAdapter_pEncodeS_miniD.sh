#0826
method_name="tri_attn_loraAdapter_pEncodeS_miniDe"

#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic']"
dataset_name="['total_spleen','lung_hospital','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']"

#load_weight="original"
load_weight="medsam"

crop_size=128
input_image_size=256
batch_size=1
learning_rate=0.001
epoch=30

python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0826.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight

#python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0826.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --pretrained
