""" Plot Transformer representations
"""

import torch
import yaml
import os
from tqdm import tqdm
import argparse
from argparse import ArgumentParser
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE
import gc

import wandb
from dyck import get_vocab_size, RandomWalkDyck
from lm import get_transformer, nn_generate, nn_log_probs, nn_next_token_probs, nn_intermediates
from error_analysis import ErrorType, analyze_and_save_errors, analyze_and_save_ood_errors, read_ood_eval_results


def plot_prefix_rep(model, sequences, dyck, ds_tokens, save_dir, ood_results=None):
    os.makedirs(save_dir, exist_ok=True)

    # print('sequences', sequences)  # debug
    prefixes = []
    for i, seq in enumerate(sequences):
        for j in range(len(seq['tokens'])):
            prefixes.append({
                'tokens': seq['tokens'][:j + 1],
                'log_prob': dyck.compute_log_prob(seq['tokens'][:j + 1]),
                'len': j + 1,
            })
    print('len(prefixes)', len(prefixes))
    gc.collect()
    torch.cuda.empty_cache()

    # print('ds_tokens', ds_tokens)  # debug
    all_intermediates = nn_intermediates(
        model,
        ds_tokens,
        batch_size=128,
        positions='all',
    )
    print('shape of all_intermediates:', len(all_intermediates), len(all_intermediates[0]),
          len(all_intermediates[0][0]))
    all_intermediates_concat = []
    for rep in all_intermediates:
        all_intermediates_concat += rep
    print('shape of all_intermediates_concat:', len(all_intermediates_concat), len(all_intermediates_concat[0]))

    # TODO: detokenize ood_results, convert to batch tensor, send to `nn_intermediates`, extract rep again
    if ood_results is not None:
        seq_to_color = {}
        for prefix_length in ood_results:
            for row_idx, prefix in enumerate(ood_results[prefix_length]['prefix']):
                error_type = ood_results[prefix_length]['error'][row_idx]
                seq_to_color[prefix.replace('<bos>', 'B').replace('<eos>', 'E')] = ErrorType[error_type].value

        error_prefixes_tokenized = [
            {'tokens': dyck.tokenize(prefix)}
            for prefix in seq_to_color
        ]
        ds_error_prefixes = datasets.Dataset.from_pandas(pd.DataFrame(error_prefixes_tokenized))
        ds_error_prefixes.set_format(type='torch')
        # print("ds_error_prefixes['tokens']", ds_error_prefixes['tokens'])  # debug
        # ds_error_prefixes['tokens'] = torch.stack(ds_error_prefixes['tokens'])
        error_prefixes_intermediates = nn_intermediates(
            model,
            ds_error_prefixes['tokens'],
            batch_size=1,  # necessary because different prefixes are of different sizes
            positions='last',
        )
        print('shape of error_prefixes_intermediates:', len(error_prefixes_intermediates), len(error_prefixes_intermediates[0]))
    else:
        error_prefixes_intermediates = []
    # error_prefixes_intermediates_concat = []
    # for rep in error_prefixes_intermediates:
    #     error_prefixes_intermediates_concat += rep
    # print('len(error_prefixes_intermediates_concat):', len(error_prefixes_intermediates_concat))
    # print('len(error_prefixes_intermediates_concat[0])', len(error_prefixes_intermediates_concat[0]))
    # print('shape of error_prefixes_intermediates_concat:', len(error_prefixes_intermediates_concat), len(error_prefixes_intermediates_concat[0]))

    for random_state in range(2):
        X_embedded = TSNE(perplexity=30.0, random_state=random_state).fit_transform(np.array(all_intermediates_concat))
        print('X_embedded.shape', X_embedded.shape)

        ## representations of deep vs. shallow prefixes
        rep_prefix_depth_save_path = os.path.join(save_dir, f'rep_prefix_depth{random_state}.png')
        plt.figure(figsize=(12, 12))
        plt.scatter(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            c=[dyck.get_final_depth(seq['tokens']) for seq in prefixes[:len(all_intermediates_concat)]],
            cmap='viridis',
            alpha=0.5,
        )
        plt.savefig(rep_prefix_depth_save_path)
        plt.show()

        ## representations of long vs. short prefixes
        rep_prefix_len_save_path = os.path.join(save_dir, f'rep_prefix_len{random_state}.png')
        plt.figure(figsize=(12, 12))
        plt.scatter(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            c=[seq['len'] for seq in prefixes[:len(all_intermediates_concat)]],
            cmap='viridis',
            alpha=0.5,
        )
        plt.savefig(rep_prefix_len_save_path)
        plt.show()

        ## representations of prefixes that end in different tokens
        rep_prefix_end_save_path = os.path.join(save_dir, f'rep_prefix_end{random_state}.png')
        plt.figure(figsize=(12, 12))
        plt.scatter(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            c=[seq['tokens'][-1] for seq in prefixes[:len(all_intermediates_concat)]],
            cmap='viridis',
            alpha=0.5,
        )
        plt.savefig(rep_prefix_end_save_path)
        plt.show()

        ## representations of prefixes that end in different bracket types
        def assign_token_value(token_id):
            if dyck.tokens[token_id] in {'(', '['}:
                return 0
            if dyck.tokens[token_id] in {')', ']'}:
                return 1
            return -1

        rep_prefix_end_type_save_path = os.path.join(save_dir, f'rep_prefix_end_type{random_state}.png')
        plt.figure(figsize=(12, 12))
        plt.scatter(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            c=[assign_token_value(seq['tokens'][-1]) for seq in prefixes[:len(all_intermediates_concat)]],
            cmap='viridis',
            alpha=0.5,
        )
        plt.savefig(rep_prefix_end_type_save_path)
        plt.show()

        ## representations of prefixes that end in different opening bracket types
        def assign_token_value(token_id):
            if dyck.tokens[token_id] in {'('}:
                return 0
            if dyck.tokens[token_id] in {'['}:
                return 1
            return -1

        rep_prefix_end_open_type_save_path = os.path.join(save_dir, f'rep_prefix_end_open_type{random_state}.png')
        plt.figure(figsize=(12, 12))
        plt.scatter(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            c=[assign_token_value(seq['tokens'][-1]) for seq in prefixes[:len(all_intermediates_concat)]],
            cmap='viridis',
            alpha=0.5,
        )
        plt.savefig(rep_prefix_end_open_type_save_path)
        plt.show()

        ## representations of prefixes colored by error types
        X_embedded = TSNE(perplexity=30.0, random_state=random_state).fit_transform(np.array(
            all_intermediates_concat + error_prefixes_intermediates,
        ))
        prefixes = prefixes[:len(all_intermediates_concat)] + error_prefixes_tokenized
        print('number of prefixes, adding error prefixes:', len(prefixes))

        def assign_seq_color(dyck, seq):
            seq_str = dyck.detokenize(seq)
            if seq_str in seq_to_color:
                return seq_to_color[seq_str]
            return 0

        rep_prefix_error_type_save_path = os.path.join(save_dir, f'rep_prefix_error_type{random_state}.png')
        plt.figure(figsize=(12, 12))
        plt.scatter(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            c=[assign_seq_color(dyck, seq['tokens']) for seq in prefixes],
            cmap='viridis',
            alpha=0.5,
        )
        plt.savefig(rep_prefix_error_type_save_path)
        plt.show()
