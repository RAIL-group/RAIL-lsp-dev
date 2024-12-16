import argparse

import taskplan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure (and write to file) for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str,
                        required=False, default=None)
    parser.add_argument('--data_file2', type=str,
                        required=False, default=None)
    parser.add_argument('--output_image_file', type=str,
                        required=False, default=None)
    parser.add_argument('--learned', action='store_true')
    parser.add_argument('--optimistic_greedy', action='store_true')
    parser.add_argument('--pessimistic_greedy', action='store_true')
    parser.add_argument('--optimistic_lsp', action='store_true')
    parser.add_argument('--pessimistic_lsp', action='store_true')
    parser.add_argument('--optimistic_oracle', action='store_true')
    parser.add_argument('--pessimistic_oracle', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args()

    if args.learned:
        data = taskplan.utilities.result.process_learned_data(args)
        print(data.describe())
    elif args.optimistic_greedy:
        data = taskplan.utilities.result.process_optimistic_greedy_data(args)
        print(data.describe())
    elif args.pessimistic_greedy:
        data = taskplan.utilities.result.process_pessimistic_greedy_data(args)
        print(data.describe())
    elif args.optimistic_lsp:
        data = taskplan.utilities.result.process_optimistic_lsp_data(args)
        print(data.describe())
    elif args.pessimistic_lsp:
        data = taskplan.utilities.result.process_pessimistic_lsp_data(args)
        print(data.describe())
    elif args.optimistic_oracle:
        data = taskplan.utilities.result.process_optimistic_oracle_data(args)
        print(data.describe())
    elif args.pessimistic_oracle:
        data = taskplan.utilities.result.process_pessimistic_oracle_data(args)
        print(data.describe())
    elif args.oracle:
        data = taskplan.utilities.result.process_oracle_data(args)
        print(data.describe())
