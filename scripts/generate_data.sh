now=$(date +"%Y%m%d_%H%M%S")
exp_name="traffic_gen_$now"
exp_name="traffic_gen_20250910_175219"


# num_of_each_scene=1
# bash ./SceneController/scripts/generate_traffic_flow.sh $exp_name $num_of_each_scene
# sleep 1
# python ./data_utils/traffic2render.py --exp_name $exp_name --all_scene
# sleep 1
cd SceneRenderer/street-gaussian
bash ./scripts/auto_render.sh $exp_name
sleep 1
cd ../..
python ./scripts/tasks/generate_convert_tasks.py --exp_name $exp_name
sleep 1
bash ./scripts/convert_dataset.sh $exp_name


