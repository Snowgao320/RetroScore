
# RetroScore：A comprehensive scoring system for synthetic accessibility guided by retrosynthetic planning
Molecular generation is a critical method in drug design, but its practical application is often limited due to the difficulty in synthesizing the generated molecules. To solve this problem, we present RetroScore, a comprehensive synthetic accessibility evaluation framework guided by multi-step retrosynthetic planning. For the molecular generation task, RetroScore outperformed six of seven synthetic accessibility metrics, yielding molecules with enhanced synthetic accessibility profiles across heterogeneous evaluation frameworks. 
## Environment Requirements  
Create a virtual environment to run the code of RetroScore.
Install pytorch with the cuda version that fits your device.
```
conda create -n retroscore python=3.9 \
conda activate retroscore \
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia \
pip install -r requirements.txt
```
## data
Click here: [[retro_data.zip (dropbox.com)](https://www.dropbox.com/scl/fi/cchn0wjz8j0dqxhr0qrom/retro_data.zip?rlkey=kqz60ec7vx7087vg1o63nucyo&e=1&dl=0)] to download and unzip files. Please put all the folders (`dataset/` and `saved_models/`) under the `data/multi_step/retro_data` directory.

## Single step model
Single step model has been trained according to Graph2edits, if you want to train your own model, please refer to Graph2edits: [enter link description here](https://github.com/Jamson-Zhong/Graph2Edits). We provide two trained checkpoint to use, which is placed in folder "experiments".

1) A single step model trained on uspto 50k is:  experiments/uspto_50k/epoch_123.pt   
2) A single step model trained on uspto full is:  experiments/uspto_full/epoch_65.pt


## Single-step retrosynthesis prediction
Go to the script folder and run the following to predict one step precursor for compounds (default use epoch_65.pt)

1) prediction for one compound
```
python single_step_predict.py --smi "CON(C)C(=O)CC1COCCN1C(=O)OC(C)(C)C" --save_dir ../pred_results --save_name one_mol_single
```
2) prediction for batch compounds
```
python single_step_predict.py --fpath FILE_PATH --save_dir ../pred_results --save_name batch_mol_single
```
FILE_PATH is your csv file with target mol smiles strings as one column and the header named SMILES, the prediction results will be saved at pred_results/xx_mol_single.csv


## Multi-step retrosynthesis planning

Go to the script folder and run the following to plan multi-step routes for compounds (default use epoch_65.pt)

1) prediction for one compound
```
python run_multistep_pre_one.py --smi "CON(C)C(=O)CC1COCCN1C(=O)OC(C)(C)C" --save_dir ../pred_results --save_name one_mol_multi
```
2) prediction for batch compounds
```
python run_multistep_pre_more.py --pred_fpath FILE_PATH --save_dir ../pred_results --save_name batch_mol_multi
```
FILE_PATH is your csv file with target mol smiles strings as one column and the header named SMILES, the planning results will be saved at pred_results/xx_mol_multi.csv and pred_results/xx_mol_multi_all.csv
**pred_results/xx_mol_multi.csv**:  Recommended synthesis route, as length best; sum confidence score best and diff best.
**pred_results/xx_mol_multi_all.csv**:  All found synthesis routes information.

### visualization of multi-step routes
If you want to visualize the multi-step reaction route, we provide a reference script:  **draw_rxn_routes.py**. By default, the script visualizes the head 10 routes of the "col_name" column in a csv file.


## Calculate RetroScore for compounds
To calculate RetroScore for compounds, please prepare a csv file with target mol smiles strings as one column and the header named SMILES. First, run multi-step retrosynthesis planning, then calculate retroscore based on the planning results. You can run the following steps.

1) cd script
2) run multi-step retrosynthesis planning
```
python run_multistep_pre.py --dataset FILE_PATH
```
3) calculate RetroScore based on the planning results
```
python compute_retro_score.py --weight1 0.25
```

**--weight1**:   the weight of RNScore, default the weight of RLScore is **1. - weight1**
FILE_PATH is your csv file with target mol smiles strings as one column and the header named SMILES, the results will be saved at retroscore_results/mol_rs_results.csv
