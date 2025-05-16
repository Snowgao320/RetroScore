import pandas as pd
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='calculate RetroScore')
parser.add_argument('--weight1', type=float, default=0.25, help='weight of RNScore')
args = parser.parse_args()

# read files
dfA = pd.read_csv('../retroscore_results/multi_plan_results.csv')
total = len(dfA)
print('total dataset:', total)

# 计算无路线的分子数
dfa = dfA[~(dfA['routes_num'] == 0)]
find_routes_num = len(dfa)
print('found routes mol num:', find_routes_num)
# 在有路线的里面再刨除掉target in mol
dfa = dfa[~dfa['best_diff_len'].isna()]
target_in_mol_no = len(dfa)
print('target in mol:', find_routes_num - target_in_mol_no)

# 提取三个文件中的分子
smi_list = []
routes_num_list = []
length_list = []
for df in [dfA]:
    smi_list += df['target_mol'].to_list()
    routes_num_list += df['routes_num'].to_list()
    length_list += df['best_diff_len'].to_list()

print('total', len(smi_list))

retro_scores_list = []
label_lst = []
for i, smi in enumerate(smi_list):
    nums = routes_num_list[i]
    length = length_list[i]

    # 计算逆合成分数
    if nums == 0:
        retro_scores_list.append(0)
        label_lst.append("HS")
    elif nums == 1 and math.isnan(length):
        retro_scores_list.append(9)
        label_lst.append("ES")
    else:
        if nums >= 10:
            nums = 10

        s =args.weight1*9*(nums/10) + (1-args.weight1)*9*(1-np.log10(length))
        s = np.clip(s, 0, 9)
        retro_scores_list.append(s)
        if s > 4.5:
            label_lst.append("ES")
        else:
            label_lst.append("HS")


# 存分数
new_df = pd.DataFrame({'smiles':smi_list,
                       'RetroScore':retro_scores_list,
                       'RetroScore_label':label_lst})
new_df.to_csv('../retroscore_results/mol_rs_results.csv', index=False)




