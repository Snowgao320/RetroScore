import os
import argparse
import multiprocessing as mp


def get_file_list(file_folder):
    file_list = os.listdir(file_folder)
    return file_list


def process(input_data, output_name):
    command = (f'python {args.code_path} '
               f'--dataset {input_data} --save_dir {args.save_dir} --save_name {output_name} '
               f'--cost_weight {args.cost_weight} --filter_ratio {args.filter_ratio} --coef {args.coef} '
               f'--mode sure_param')
    os.system(command)


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input data directory located", type=str,
                    default="../data/NP_gen_data/chunk_split")
parser.add_argument("--save_dir", help="output file directory located", type=str,
                    default="../data/NP_gen_data/pred_split_chunks")
parser.add_argument("--code_path", help="code to run", type=str,
                    default="./run_multistep_pre.py")
parser.add_argument("--max_cpus", help="used cpu nums", type=int,
                    default=12)
parser.add_argument('--cost_weight', type=float, default=0.1,
                        help='The weight of confidence score for expanding function')
parser.add_argument('--filter_ratio', type=float, default=0.9,
                        help='filter ratio with average confidence score of multi-stage screening')
parser.add_argument('--coef', type=float, default=0.3,
                        help='Coef with end dist of multi-stage screening')
args = parser.parse_args()

files_list = get_file_list(args.input_dir)
num = len(files_list)
pool = mp.Pool(processes=min(args.max_cpus, mp.cpu_count()))

for i in range(num):
    name = files_list[i].split('.')[0]
    input_path = args.input_dir + '/' + files_list[i]
    out_name = name + "_res"

    pool.apply_async(process, args=(input_path, out_name))

pool.close()
pool.join()


