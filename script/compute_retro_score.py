import os
import pandas as pd
import numpy as np
import argparse


class RetroScore:
    def __init__(self, w: float, n: int):
        self.w = w   # RNS weight
        self.n = n   # max route num

    def compute_score(self, route_num, route_len):
        if route_num == -1:  # target in stock
            return np.float64(1)
        elif route_num == 0:
            return np.float64(0)
        else:
            rns = np.clip(route_num, 0, self.n) / self.n
            route_len = np.clip(route_len, 1, 100)
            rls = 1 - np.log10(route_len)
            score = self.w * rns + (1 - self.w) * rls
            score = np.clip(score, 0, 1) * 9
            return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate RetroScore')
    parser.add_argument('--w', type=float, default=0.50, help='weight of RNScore')
    parser.add_argument('--N', type=int, default=10, help='Max route num')
    parser.add_argument('--input_fpath', type=str, default="../pred_results/routes_pred.csv", help='input file path')
    parser.add_argument('--output_dir', type=str, default="../pred_results", help='output file dir')
    args = parser.parse_args()

    # read files
    df = pd.read_csv(args.input_fpath)
    print('data len:', len(df))

    # 实例化分数类
    scorer = RetroScore(args.w, args.N)
    pred_scores = []
    for route_num, route_len in zip(df['routes_num'], df['multi_stage_best_length']):
        pred_scores.append(scorer.compute_score(route_num, route_len))

    # 存分数
    os.makedirs(args.output_dir, exist_ok=True)
    output_fpath = os.path.join(args.output_dir, f'pred_RetroScore.csv')
    df['RetroScore'] = pred_scores
    df.to_csv(output_fpath, index=False)
    print(f"pred file saved to {output_fpath}")




