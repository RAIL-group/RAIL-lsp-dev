import re
import pandas as pd


def process_learned_data(args):
    """Preprocessing function for learned lsp"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . learned: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'LEARNED_LSP']
    ).groupby('seed', group_keys=False).tail(1)


def process_optimistic_greedy_data(args):
    """Preprocessing function for naive (closest action planner) lsp"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . optimistic_greedy: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'OPTIMISTIC_GREEDY']
    ).groupby('seed', group_keys=False).tail(1)


def process_pessimistic_greedy_data(args):
    """Preprocessing function for naive (closest action planner) lsp"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . pessimistic_greedy: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'PESSIMISTIC_GREEDY']
    ).groupby('seed', group_keys=False).tail(1)


def process_optimistic_lsp_data(args):
    """Preprocessing function for planner with Learned Search Policy"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . optimistic_lsp: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'OPTIMISTIC_LSP']
    ).groupby('seed', group_keys=False).tail(1)


def process_pessimistic_lsp_data(args):
    """Preprocessing function for planner with Learned Search Policy"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . pessimistic_lsp: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'PESSIMISTIC_LSP']
    ).groupby('seed', group_keys=False).tail(1)


def process_optimistic_oracle_data(args):
    """Preprocessing function for optimistic oracle"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . optimistic_oracle: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'OPTIMISTIC_ORACLE']
    ).groupby('seed', group_keys=False).tail(1)


def process_pessimistic_oracle_data(args):
    """Preprocessing function for pessimistic oracle"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . pessimistic_oracle: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'PESSIMISTIC_ORACLE']
    ).groupby('seed', group_keys=False).tail(1)


def process_oracle_data(args):
    """Preprocessing function for pessimistic oracle"""
    data = []

    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . oracle: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'ORACLE']
    ).groupby('seed', group_keys=False).tail(1)
