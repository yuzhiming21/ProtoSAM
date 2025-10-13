#method_name="sam"
#method_name="baidu"
#method_name="sam3d"
#method_name="cuhk"
#method_name="lora"
method_name="tri_attn_loraAdapter_pEncodeS_miniDe"
crop_size=128

#dataset_name="['hippo']"

#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
#dataset_name="['lung']"
#dataset_name="['lung2','Lung421']"
#dataset_name="['lung_hospital']"
#dataset_name="['liver']"

#dataset_name="['lung','lung2','Lung421']"
#dataset_name="['lung_center','lung2_center','Lung421_center']"
#dataset_name="['Lung421']"
#dataset_name="['lung_center']"
#dataset_name="['lung2_center']"
#dataset_name="['Lung421_center']"

#dataset_name="['colon']"
#dataset_name="['liver']"
#dataset_name="['spleen']"
#dataset_name="['brain_torch']"

#dataset_name="['total_spleen','total_pancreas','total_lung_upper_lobe_right','total_kidney_right']"
#dataset_name="['AbdomenCT_1K1']"
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
#dataset_name="['AbdomenCT_1K1','AbdomenCT_1K2','AbdomenCT_1K3','AbdomenCT_1K4']"
#dataset_name="['AbdomenCT_1K1']"
#dataset_name="['AbdomenCT_1K2']"

#dataset_name="['lung','lung2','Lung421','pancreas','kits23','hepatic']"
#dataset_name="['lung','lung2','Lung421']"
#dataset_name="['pancreas']"
#dataset_name="['kits23']"
#dataset_name="['hepatic']"
#dataset_name="['brain_torch']"
#dataset_name="['lung_hospital']"
#dataset_name="['total_spleen','total_pancreas','total_lung_upper_lobe_right','total_kidney_right']"
#dataset_name="['total_spleen']"
#dataset_name="['total_pancreas']"
#dataset_name="['total_kidney_right']"
#dataset_name="['total_lung_upper_lobe_right']"

#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic']"
#dataset_name="['lung_hospital','total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']"
dataset_name="['total_spleen','lung_hospital','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']"

#dataset_name="['total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right','lung_hospital']"
#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','lung_hospital']"

#load_weight="original"
load_weight="medsam"

input_image_size=256
batch_size=1
learning_rate=0.001
epoch=10
#num_click=50
save='no'
use_ft='yes'
#use_ft='no'

#python _iseg_test_by_given_clicks.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --save_result $save --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 50 --pretrained  

#python _iseg_simulative_click_test.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --save_result $save --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 8 --pretrained

#python _iseg_simulative_click_test.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 10 --pretrained

#python _iseg_simulative_click_test_0429.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 10 --pretrained

#python _iseg_simulative_click_test_0714.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 10 --pretrained

python _iseg_simulative_click_test_0826.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 10 --pretrained
