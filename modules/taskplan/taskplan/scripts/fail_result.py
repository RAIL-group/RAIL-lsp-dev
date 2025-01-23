import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import taskplan

pd.set_option('display.max_columns', None)


def get_target_seeds(file_name):
    values = []
    with open(file_name, 'r') as file:
        for line in file:
            match = re.search(r's: (\d{4})', line)
            if match:
                values.append(int(match.group(1)))
    return sorted(values)[:100]


def filter(d_dict, target_seeds):
    return {k: v for k, v in d_dict.items() if k in target_seeds}


def get_success_rate(val):
    return (100 - val) / 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure (and write to file) for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--df_opt_greedy', type=str, required=False, default=None)
    parser.add_argument('--df_pes_greedy', type=str, required=False, default=None)
    parser.add_argument('--df_opt_lsp', type=str, required=False, default=None)
    parser.add_argument('--df_pes_lsp', type=str, required=False, default=None)
    parser.add_argument('--df_learned', type=str, required=False, default=None)
    parser.add_argument('--df_opt_oracle', type=str, required=False, default=None)
    parser.add_argument('--df_pes_oracle', type=str, required=False, default=None)
    parser.add_argument('--df_oracle', type=str, required=False, default=None)
    parser.add_argument('--save_dir', type=str, required=False, default=None)
    parser.add_argument('--output_image_file', type=str,
                        required=False, default=None)
    args = parser.parse_args()

    target_maps = get_target_seeds(args.df_oracle)

    args.data_file = args.df_learned
    data = taskplan.utilities.result.process_learned_data(args)
    learned_dict = data.set_index('seed')['LEARNED_LSP'].to_dict()
    learned_dict = filter(learned_dict, target_maps)

    look_up_str = 'ORACLE'
    args.data_file = args.df_oracle
    data = taskplan.utilities.result.process_oracle_data(args)
    other_dict = data.set_index('seed')[look_up_str].to_dict()
    other_dict = filter(other_dict, target_maps)

    dicts = [other_dict]
    look_up_strs = [look_up_str]

    if os.path.exists(args.df_opt_greedy):
        look_up_str = 'OPTIMISTIC_GREEDY'
        args.data_file = args.df_opt_greedy
        data = taskplan.utilities.result.process_optimistic_greedy_data(args)
        other_dict = data.set_index('seed')[look_up_str].to_dict()
        other_dict = filter(other_dict, target_maps)
        dicts.append(other_dict)
        look_up_strs.append(look_up_str)
        plt.clf()
        taskplan.plotting.make_scatter_with_box(other_dict, learned_dict)
        plt.savefig(f'{args.save_dir}/learned_vs_{look_up_str.lower()}.png', dpi=600)
        plt.close()

    if os.path.exists(args.df_pes_greedy):
        look_up_str = 'PESSIMISTIC_GREEDY'
        args.data_file = args.df_pes_greedy
        data = taskplan.utilities.result.process_pessimistic_greedy_data(args)
        other_dict = data.set_index('seed')[look_up_str].to_dict()
        other_dict = filter(other_dict, target_maps)
        dicts.append(other_dict)
        look_up_strs.append(look_up_str)
        plt.clf()
        taskplan.plotting.make_scatter_with_box(other_dict, learned_dict)
        plt.savefig(f'{args.save_dir}/learned_vs_{look_up_str.lower()}.png', dpi=600)
        plt.close()

    if os.path.exists(args.df_opt_lsp):
        look_up_str = 'OPTIMISTIC_LSP'
        args.data_file = args.df_opt_lsp
        data = taskplan.utilities.result.process_optimistic_lsp_data(args)
        other_dict = data.set_index('seed')[look_up_str].to_dict()
        other_dict = filter(other_dict, target_maps)
        dicts.append(other_dict)
        look_up_strs.append(look_up_str)
        plt.clf()
        taskplan.plotting.make_scatter_with_box(other_dict, learned_dict)
        plt.savefig(f'{args.save_dir}/learned_vs_{look_up_str.lower()}.png', dpi=600)
        plt.close()

    if os.path.exists(args.df_pes_lsp):
        look_up_str = 'PESSIMISTIC_LSP'
        args.data_file = args.df_pes_lsp
        data = taskplan.utilities.result.process_pessimistic_lsp_data(args)
        other_dict = data.set_index('seed')[look_up_str].to_dict()
        other_dict = filter(other_dict, target_maps)
        dicts.append(other_dict)
        look_up_strs.append(look_up_str)
        plt.clf()
        taskplan.plotting.make_scatter_with_box(other_dict, learned_dict)
        plt.savefig(f'{args.save_dir}/learned_vs_{look_up_str.lower()}.png', dpi=600)
        plt.close()

    dicts.append(learned_dict)
    look_up_strs.append('LEARNED_LSP')

    # Create a DataFrame for each dictionary and merge them
    dfs = [pd.DataFrame(list(d.items()), columns=['seed', look_up_strs[i]])
           for i, d in enumerate(dicts)]

    # Merge all DataFrames on 'seed' with an outer join
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='seed', how='outer')

    fc_opt_grdy = merged_df['OPTIMISTIC_GREEDY'].isna().sum()
    fc_pes_grdy = merged_df['PESSIMISTIC_GREEDY'].isna().sum()
    fc_opt_lsp = merged_df['OPTIMISTIC_LSP'].isna().sum()
    fc_pes_lsp = merged_df['PESSIMISTIC_LSP'].isna().sum()
    fc_learned = merged_df['LEARNED_LSP'].isna().sum()
    print(f"Succes rate for OPTIMISTIC_GREEDY: {get_success_rate(fc_opt_grdy)}")
    print(f"Succes rate for PESSIMISTIC_GREEDY: {get_success_rate(fc_pes_grdy)}")
    print(f"Succes rate for OPTIMISTIC_LSP: {get_success_rate(fc_opt_lsp)}")
    print(f"Succes rate for PESSIMISTIC_LSP: {get_success_rate(fc_pes_lsp)}")
    print(f"Succes rate for LEARNED_LSP: {get_success_rate(fc_learned)}")

    # Fill missing values with 5000
    merged_df.fillna(8000, inplace=True)
    print(merged_df.describe())
