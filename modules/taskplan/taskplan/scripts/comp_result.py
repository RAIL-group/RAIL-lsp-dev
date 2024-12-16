import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

import taskplan

pd.set_option('display.max_columns', None)


def load_logfiles(args):
    combined_df = []
    args.data_file = args.df_opt_greedy
    combined_df.append(taskplan.utilities.result.process_optimistic_greedy_data(args))
    args.data_file = args.df_pes_greedy
    combined_df.append(taskplan.utilities.result.process_pessimistic_greedy_data(args))
    args.data_file = args.df_opt_lsp
    combined_df.append(taskplan.utilities.result.process_optimistic_lsp_data(args))
    args.data_file = args.df_pes_lsp
    combined_df.append(taskplan.utilities.result.process_pessimistic_lsp_data(args))
    args.data_file = args.df_learned
    combined_df.append(taskplan.utilities.result.process_learned_data(args))
    args.data_file = args.df_opt_oracle
    combined_df.append(taskplan.utilities.result.process_optimistic_oracle_data(args))
    args.data_file = args.df_pes_oracle
    combined_df.append(taskplan.utilities.result.process_pessimistic_oracle_data(args))
    args.data_file = args.df_oracle
    combined_df.append(taskplan.utilities.result.process_oracle_data(args))

    return combined_df


def get_common_df(combined_data):
    # Only extract results for the first 100 seeds common to all planners
    common_seeds = set(combined_data[0]['seed'])
    for table in combined_data[1:]:
        common_seeds &= set(table['seed'])

    common_seeds = pd.Series(sorted(common_seeds))  # Convert to sorted Series for consistency

    # Step 2: Select up to 100 common seeds
    if len(common_seeds) < 100:
        print(f"Warning: Only {len(common_seeds)} common seeds found, selecting all of them.")
    selected_seeds = common_seeds.iloc[:100]

    # write the seeds to a file
    selected_seeds.to_csv('selected_seeds.csv', index=False)
    with open('/data/test_logs/selected_seeds.csv', 'w') as f:
        f.write(selected_seeds.to_string(index=False))

    # Step 3: Filter each table for the selected seeds
    filtered_tables = [
        table[table['seed'].isin(selected_seeds)].reset_index(drop=True)
        for table in combined_data
    ]

    # Step 4: Merge all filtered tables on 'seed'
    merged_df = filtered_tables[0]
    for table in filtered_tables[1:]:
        merged_df = pd.merge(merged_df, table, on='seed', how='inner')

    return merged_df


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

    # Load data and create the pd.dataframe for each planner with its seed
    # and corresponding cost
    combined_data = load_logfiles(args)
    common_df = get_common_df(combined_data)
    print(common_df.describe())
    # raise NotImplementedError
