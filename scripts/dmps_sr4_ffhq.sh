
#!/bin/bash

python3 dmps.py --gpu 0  --task_config  'configs/sr4_config.yaml' --model_config 'configs/model_config.yaml'  --save_dir './saved_results'

# for((i=0; i<1; i++)); do {
# 	python3 main.py --gpu 1  --task_config  'configs/sr4_config.yaml' --model_config 'configs/model_config.yaml'  --save_dir './saved_results'
#   echo "DONE!"
# } & done
# wait
