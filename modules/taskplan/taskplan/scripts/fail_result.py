import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import taskplan

pd.set_option('display.max_columns', None)


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

    args.data_file = args.df_learned
    data = taskplan.utilities.result.process_learned_data(args)
    learned_dict = data.set_index('seed')['LEARNED_LSP'].to_dict()
    dicts = []
    look_up_strs = []

    if os.path.exists(args.df_opt_greedy):
        look_up_str = 'OPTIMISTIC_GREEDY'
        args.data_file = args.df_opt_greedy
        data = taskplan.utilities.result.process_optimistic_greedy_data(args)
        other_dict = data.set_index('seed')[look_up_str].to_dict()
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

    # Fill missing values with 5000
    merged_df.fillna(5000, inplace=True)
    print(merged_df.describe())
