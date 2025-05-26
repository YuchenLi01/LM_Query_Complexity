import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
import os
import collections
import matplotlib.pyplot as plt
from lm import conditional_nn_generate

plt.style.use('tableau-colorblind10')

ErrorType = Enum('ErrorType', ['START_INCORRECT', 
                               'END_INCORRECT', 
                               'MAX_STACK_DEPTH_EXCEEDED', 
                               'STACK_DEPTH_EXCEEDS_REMAINING_LENGTH', 
                               'STACK_UNDERFLOW',
                               'INCORRECT_CLOSING_BRACKET' ])

# analyze an invalid sequence
def analyze_error(dyck, model, seq, backtrack_quota=None, 
                  num_excluded_token=None, excluded_all_token=None):
    stack = []
    has_seen_bos = False

    error = None

    preds = model(torch.tensor([seq]).cuda()).logits
    preds = torch.softmax(preds, dim=-1)

    for i, x in enumerate(seq):
        if x == dyck.bos:
            if not has_seen_bos:
                has_seen_bos = True
                continue
            else:
                error = ErrorType.START_INCORRECT
                break

        elif x == dyck.eos:
            # length is the length of the sequence without the <bos>='B' and <eos>='E' tokens
            if i != dyck.length + 1:
                error = ErrorType.END_INCORRECT
                break

        elif x < dyck.num_types:
            if len(stack) + 1 > dyck.max_depth:
                error = ErrorType.MAX_STACK_DEPTH_EXCEEDED
                break

            elif len(stack) + 1 > dyck.length - i:
                error = ErrorType.STACK_DEPTH_EXCEEDS_REMAINING_LENGTH
                break

            stack.append(x)

        else:
            if len(stack) == 0:
                error = ErrorType.STACK_UNDERFLOW
                break

            # Note that stack.pop() already removes a bracket
            elif stack.pop() + dyck.num_types != x:
                error = ErrorType.INCORRECT_CLOSING_BRACKET
                break

        
    if error:
        most_prob = torch.argmax(preds[0, i - 1, :]).item()
        return {
            'error': error.name,
            'prefix': dyck.detokenize(seq[:i]),
            'generated_token': dyck.tokens[x],
            'generated_token_prob': preds[0, i - 1, x].item(),
            'most_probable_token': dyck.tokens[most_prob],
            'most_probable_token_prob': preds[0, i - 1, most_prob].item(),
            'backtrack_quota': backtrack_quota,
            'num_excluded_token': num_excluded_token,
            'excluded_all_token': excluded_all_token,
        }
    
    return None


def analyze_and_save_errors(dyck, model, seqs, output_file, backtrack_quotas=None,
                            num_excluded_tokens=None, excluded_all_tokens=None):
    if os.path.exists(output_file):
        print(f"Warning: error analysis already exists at: {output_file}, skipped")
        return
    if not backtrack_quotas:
        backtrack_quotas = [None] * len(seqs)
    if not num_excluded_tokens:
        num_excluded_tokens = [None] * len(seqs)
    if not excluded_all_tokens:
        excluded_all_tokens = [None] * len(seqs)
    if type(backtrack_quotas) is list:
        assert len(backtrack_quotas) == len(seqs)
    if type(num_excluded_tokens) is list:
        assert len(num_excluded_tokens) == len(seqs)
    if type(excluded_all_tokens) is list:
        assert len(excluded_all_tokens) == len(seqs)
    errors = []
    for seq, backtrack_quota, num_excluded_token, excluded_all_token in zip(seqs, backtrack_quotas, num_excluded_tokens, excluded_all_tokens):
        error = analyze_error(dyck, model, seq, backtrack_quota, num_excluded_token, excluded_all_token)
        if error:
            errors.append(error)
    if len(errors) > 0:
        df = pd.DataFrame(errors)
        df.to_csv(output_file, index=False)
        return df

    return None


def analyze_and_save_ood_errors(
        ood_sequences,
        dyck,
        model,
        batch_size,
        top_p,
        temperature,
        eval_dir,
        argmax_length=0,
        min_prefix_len=0,
        ood_generation_errors_dir='ood_generation_errors',
        backtrack_quota=0,
        backtrack_stride=None,
        max_allowed_prefix_truncation=0,
        backtracked_prefixes=None,
        model_err_pred=None,
        allow_2nd_backtrack_at_same_prefix=False,
        use_groundtruth_error_predictor=False,
):
    def plot_stacked_bars(stack_dict, x, y, title, filename):
        width = 0.5
        fig, ax = plt.subplots()
        bottom = np.zeros(len(x))

        for error_name, error_count in stack_dict.items():
            ax.bar(x, error_count, width = width, bottom=bottom, label=error_name)
            bottom += error_count
        
        ax.set_ylabel('Fraction of generated sequences')
        ax.set_xlabel('Prefix length')

        ax.set_ylim(0, 0.2)
        ax.set_xlim(0, dyck.length + 2)

        ax.set_title(title)
        ax.legend()
        ax.plot(x, y)
        fig.savefig(filename, bbox_inches='tight', dpi = 300)

    ood_dir = os.path.join(eval_dir, f'{ood_generation_errors_dir}/argmax_length_{argmax_length}_p{top_p}_t{temperature}')

    if not os.path.exists(ood_dir):
        os.makedirs(ood_dir)

    x = []
    y = []

    # To plot error counts by type
    error_count_dict = collections.defaultdict(list)

    # To plot unavoidable errors
    unavoidable_count_dict = collections.defaultdict(list)

    num_ood_sequences = len(ood_sequences)

    for i in range(min_prefix_len, dyck.length + 2):
        if i + argmax_length > dyck.length + 1:
            continue

        output_file = os.path.join(ood_dir, f'prefix_length_{i}.csv')
        
        if os.path.exists(output_file):
            print(f"Warning: error analysis already exists at: {output_file}, skipped")
            df = pd.read_csv(output_file)

        else:
            ood_generations = conditional_nn_generate(
                dyck,
                model,
                prefix_sequences=[seq['tokens'] for seq in ood_sequences],
                batch_size=batch_size,
                top_p=top_p,
                temperature=temperature,
                prefix_length=i,
                argmax_length=argmax_length,
                backtrack_quota=backtrack_quota,
                backtrack_stride=backtrack_stride,
                max_allowed_prefix_truncation=max_allowed_prefix_truncation,
                backtracked_prefixes=backtracked_prefixes,
                model_err_pred=model_err_pred,
                allow_2nd_backtrack_at_same_prefix=allow_2nd_backtrack_at_same_prefix,
                use_groundtruth_error_predictor=use_groundtruth_error_predictor,
            )

            illegal_ood_seqs = []
            backtrack_quotas = []
            num_excluded_tokens = []
            excluded_all_tokens = []
            for _, seq in enumerate(ood_generations):
                if not dyck.accept(seq['tokens']):
                    illegal_ood_seqs.append(seq['tokens'])
                    # print('remaining_backtrack_quota', seq['remaining_backtrack_quota'])  # debug
                    backtrack_quotas.append(seq['remaining_backtrack_quota'])
                    num_excluded_tokens.append(seq['num_excluded_tokens'])
                    excluded_all_tokens.append(seq['excluded_all_tokens'])

            # analyze and save errors
            df = analyze_and_save_errors(dyck, model, illegal_ood_seqs, output_file, backtrack_quotas,
                                         num_excluded_tokens, excluded_all_tokens)

        if df is not None:
            num_illegal_ood_seqs = len(df['error'])
        else:
            num_illegal_ood_seqs = 0

        x.append(i)
        y.append(num_illegal_ood_seqs/num_ood_sequences)

        if df is not None:
            # Storing error counts by type
            error_counts = df['error'].value_counts()
            for error_type in ErrorType:
                error_count_dict[error_type.name].append(error_counts.get(error_type.name, 0)/num_ood_sequences)

            # Storing unavoidable errors
            is_unavoidable = df['generated_token'] == df['most_probable_token']
            unavoidable_sum = is_unavoidable.sum()
            unavoidable_count_dict['unavoidable'].append(unavoidable_sum/num_ood_sequences)
            unavoidable_count_dict['avoidable'].append((num_illegal_ood_seqs - unavoidable_sum)/num_ood_sequences)
        
        else: 
            for error_type in ErrorType:
                error_count_dict[error_type.name].append(0)

            unavoidable_count_dict['unavoidable'].append(0)
            unavoidable_count_dict['avoidable'].append(0)


    # Plot error counts by type
    plot_stacked_bars(error_count_dict, x, y, 
                      f'Error types vs prefix length (argmax_length = {argmax_length})', 
                      os.path.join(ood_dir, f'error_types_vs_ood_prefix_length_argmax_length_{argmax_length}_p{top_p}_t{temperature}.png'))

    # Plot unavoidable errors
    plot_stacked_bars(unavoidable_count_dict, x, y, 
                      f'Unavoidable errors vs prefix length (argmax_length = {argmax_length})', 
                      os.path.join(ood_dir, f'unavoidable_errors_vs_ood_prefix_length_argmax_length_{argmax_length}_p{top_p}_t{temperature}.png'))


def read_ood_eval_results(eval_dir, ood_eval_dir='ood_generation_errors'):
    ood_results = {}

    ood_dir = os.path.join(eval_dir, ood_eval_dir)
    for filename in os.listdir(ood_dir):
        if filename.startswith('prefix_length') and filename.endswith('.csv'):
            prefix_length = int(filename.split('_')[2].split('.')[0])
            df = pd.read_csv(os.path.join(ood_dir, filename))
            ood_results[prefix_length] = df
            # print(f'read {filename}')  # debug

    return ood_results
