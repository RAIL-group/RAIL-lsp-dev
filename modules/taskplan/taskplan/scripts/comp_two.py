import argparse
import pandas as pd
import matplotlib.pyplot as plt

import taskplan

pd.set_option('display.max_columns', None)


def load_logfiles(args):
    combined_df = []
    if args.df_opt_lsp:
        args.data_file = args.df_opt_lsp
        combined_df.append(taskplan.utilities.result.process_optimistic_lsp_data(args))
    elif args.df_pes_lsp:
        args.data_file = args.df_pes_lsp
        combined_df.append(taskplan.utilities.result.process_pessimistic_lsp_data(args))
    args.data_file = args.df_learned
    combined_df.append(taskplan.utilities.result.process_learned_data(args))

    return combined_df


def get_common_df(combined_data):
    # Only extract results for the first 100 seeds common to all planners
    common_seeds = set(combined_data[0]['seed'])
    for table in combined_data[1:]:
        common_seeds &= set(table['seed'])

    common_seeds = pd.Series(sorted(common_seeds))  # Convert to sorted Series for consistency

    # Select up to 100 common seeds
    if len(common_seeds) < 100:
        print(f"Warning: Only {len(common_seeds)} common seeds found, selecting all of them.")
    selected_seeds = common_seeds.iloc[:]

    # Filter each table for the selected seeds
    filtered_tables = [
        table[table['seed'].isin(selected_seeds)].reset_index(drop=True)
        for table in combined_data
    ]

    # Merge all filtered tables on 'seed'
    merged_df = filtered_tables[0]
    for table in filtered_tables[1:]:
        merged_df = pd.merge(merged_df, table, on='seed', how='inner')

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure (and write to file) for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--df_pes_lsp', type=str, required=False, default=None)
    parser.add_argument('--df_opt_lsp', type=str, required=False, default=None)
    parser.add_argument('--df_learned', type=str, required=False, default=None)
    parser.add_argument('--save_dir', type=str, required=False, default=None)
    args = parser.parse_args()

    # Load data and create the pd.dataframe for each planner with its seed
    # and corresponding cost
    combined_data = load_logfiles(args)
    common_df = get_common_df(combined_data)
    print(common_df.describe())

    result_dict = common_df.set_index('seed').T.to_dict()
    Learned_dict = {k: result_dict[k]['LEARNED_LSP'] for k in result_dict}

    if args.df_opt_lsp:
        Other_strs = ['OPTIMISTIC_LSP']
    elif args.df_pes_lsp:
        Other_strs = ['PESSIMISTIC_LSP']

    for seed in Learned_dict:
        # save the costs if learned_lsp cost is lower than or equal to
        # 'OPTIMISTIC_GREEDY', 'PESSIMISTIC_GREEDY', 'OPTIMISTIC_LSP', 'PESSIMISTIC_LSP'
        learned_cost = Learned_dict[seed]
        other_costs = []
        for look_up_str in Other_strs[:4]:
            other_costs.append(result_dict[seed][look_up_str])

        if learned_cost <= min(other_costs):
            with open(f'{args.save_dir}/better-seeds-costs.csv', 'a') as f:
                f.write(f'{seed} {learned_cost} {other_costs}\n')

        if learned_cost < min(other_costs):
            with open(f'{args.save_dir}/strictly-better-seeds.csv', 'a') as f:
                f.write(f'{seed}\n')

    for look_up_str in Other_strs:
        Other_dict = {k: result_dict[k][look_up_str] for k in result_dict}
        plt.clf()
        taskplan.plotting.make_scatter_with_box(Other_dict, Learned_dict)
        plt.savefig(f'{args.save_dir}/learned_vs_{look_up_str.lower()}.png', dpi=600)
        plt.close()
    exit()
