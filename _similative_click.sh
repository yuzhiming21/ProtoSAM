method_name="tri_attn_loraAdapter_pEncodeS_miniDe"

dataset_name="['total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right','lung_hospital']"
#dataset_name="['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','lung_hospital']"

#load_weight="original"
load_weight="medsam"

crop_size=128
input_image_size=256
batch_size=1
learning_rate=0.001
epoch=10
save='no'
use_ft='yes'
#use_ft='no'

python _iseg_simulative_click_test_0826.py --method $method_name --data $dataset_name --snapshot_path exps/${method_name}_${crop_size}_bs${batch_size}_10Click_simpleClick_${load_weight}-weight_norm/ --use_ft_weight $use_ft --rand_crop_size $crop_size --max_epoch $epoch --input_image_size $input_image_size --batch_size $batch_size --lr $learning_rate --load_weight $load_weight --num_click 10 --pretrained
