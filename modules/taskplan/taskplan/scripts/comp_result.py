import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

    result_dict = common_df.set_index('seed').T.to_dict()
    Learned_dict = {k: result_dict[k]['LEARNED_LSP'] for k in result_dict}

    Other_strs = ['OPTIMISTIC_GREEDY', 'PESSIMISTIC_GREEDY',
                  'OPTIMISTIC_LSP', 'PESSIMISTIC_LSP',
                  'OPTIMISTIC_ORACLE', 'PESSIMISTIC_ORACLE', 'ORACLE']

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

    opt_seeds = []
    pes_seeds = []
    for seed in Learned_dict:
        oracle_cost = result_dict[seed]['ORACLE']
        opt_oracle_cost = result_dict[seed]['OPTIMISTIC_ORACLE']
        pes_oracle_cost = result_dict[seed]['PESSIMISTIC_ORACLE']
        if opt_oracle_cost < pes_oracle_cost and \
           oracle_cost <= opt_oracle_cost:
            opt_seeds.append(seed)
        elif opt_oracle_cost > pes_oracle_cost and \
           oracle_cost <= pes_oracle_cost:
            pes_seeds.append(seed)

    limit = min(len(opt_seeds), len(pes_seeds))
    limit = min(limit, 50)
    opt_seeds = opt_seeds[:limit]
    pes_seeds = pes_seeds[:limit]
    opt_dict = {seed: result_dict[seed] for seed in opt_seeds}
    pes_dict = {seed: result_dict[seed] for seed in pes_seeds}
    comb_dict = {**opt_dict, **pes_dict}
    # change result_dict back to original dataframe format of common_df
    comb_df = pd.DataFrame.from_dict(comb_dict, orient='index')
    print(comb_df.describe())
    opt_df = pd.DataFrame.from_dict(opt_dict, orient='index')
    pes_df = pd.DataFrame.from_dict(pes_dict, orient='index')
    print('Optimistic favoring maps')
    print(opt_df.describe())
    print('Pessimistic favoring maps')
    print(pes_df.describe())

    for look_up_str in Other_strs:
        opt_other_dict = {k: opt_dict[k][look_up_str] for k in opt_dict}
        opt_learned_dict = {k: Learned_dict[k] for k in opt_dict}
        pes_other_dict = {k: pes_dict[k][look_up_str] for k in pes_dict}
        pes_learned_dict = {k: Learned_dict[k] for k in pes_dict}
        plt.clf()
        taskplan.plotting.make_scatter_compare(opt_other_dict, opt_learned_dict,
                                               pes_other_dict, pes_learned_dict)
        plt.savefig(f'{args.save_dir}/learned_vs_{look_up_str.lower()}.png', dpi=600)
        plt.close()
