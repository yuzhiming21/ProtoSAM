#0826
#method_name="sam"
#method_name="sam3d"
#method_name="weaksam"
#method_name="cuhk"
#method_name="lora"
#method_name="baidu"
#method_name="medsam2"
method_name="tri_attn_loraAdapter_pEncodeS_miniDe"

#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic','liver','spleen','colon']"

#dataset_name="['liver','spleen','hippo','colon']"
#dataset_name="['colon']"
#dataset_name="['liver']"
#dataset_name="['spleen']"
#dataset_name="['hippo']"

#dataset_name="['total_spleen','total_pancreas','total_lung_upper_lobe_right','total_kidney_right']"
#dataset_name="['total_spleen']"
#dataset_name="['total_pancreas']"
#dataset_name="['total_kidney_left','total_kidney_right']"
#dataset_name="['total_kidney_left']"
#dataset_name="['total_kidney_right']"
#dataset_name="['total_lung_lower_lobe_left','total_lung_lower_lobe_right','total_lung_middle_lobe_right','total_lung_upper_lobe_left','total_lung_upper_lobe_right']"
#dataset_name="['total_lung_lower_lobe_left']"
#dataset_name="['total_lung_lower_lobe_right']"
#dataset_name="['total_lung_middle_lobe_right']"
#dataset_name="['total_lung_upper_lobe_left']"
#dataset_name="['total_lung_upper_lobe_right']"
#dataset_name="['total_spleen','total_pancreas','total_kidney_left','total_kidney_right','total_lung_lower_lobe_left','total_lung_lower_lobe_right','total_lung_middle_lobe_right','total_lung_upper_lobe_left','total_lung_upper_lobe_right']"
#dataset_name="['AbdomenCT_1K1']"
#dataset_name="['AbdomenCT_1K2']"
#dataset_name="['AbdomenCT_1K3']"
#dataset_name="['AbdomenCT_1K4']"

#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
#dataset_name="['lung','lung2','Lung421']"
#dataset_name="['pancreas']"
#dataset_name="['kits23']"
#dataset_name="['hepatic']"
#dataset_name="['lung_hospital']"
#dataset_name="['brain_torch']"
#dataset_name="['total_spleen','total_pancreas','total_lung_upper_lobe_right','total_kidney_right']"
#dataset_name="['total_spleen']"
#dataset_name="['total_pancreas']"
#dataset_name="['total_kidney_right']"
#dataset_name="['total_lung_upper_lobe_right']"

#dataset_name="['total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right','lung_hospital']"
#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','lung_hospital']"

#dataset_name="['total_spleen','lung','lung2','Lung421','lung_hospital']"
#dataset_name="['lung_hospital']"

#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic','total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']"
#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']"

#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic']"
#dataset_name="['total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right','lung_hospital']"
dataset_name="['total_spleen','lung_hospital','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']"

#dataset_name="['lung','lung2','Lung421','lung_hospital']"

#load_weight="original"
load_weight="medsam"
#load_weight="1"

crop_size=128
#crop_size=96

input_image_size=256
batch_size=1
learning_rate=0.001
#epoch=50
epoch=10

#tree_config=dataset_tree.json
#tree_switch_epoch=10

# exp1 SAM prompt encoder

#python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0429.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight

#python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0429.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --pretrained

#python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0714.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight

python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0826.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight
#python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_0826_2.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight

#python train_iseg_tri_attn_lora_adapter_pEncodeS_miniDe_tree_0923.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --tree_training --tree_config $tree_config --tree_switch_epoch $tree_switch_epoch

