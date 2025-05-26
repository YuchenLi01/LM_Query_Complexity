""" Trains/runs a language model.
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
from error_analysis import analyze_and_save_errors, analyze_and_save_ood_errors, read_ood_eval_results
from plot_rep import plot_prefix_rep


def create_args(config_file):
    args = yaml.safe_load(open(config_file))
    # Determine whether CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    return args


def load_dataset(config, save_path, sample_count, batch_size):
    dyck = RandomWalkDyck(config)
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as f:
            print(f"Loading ds from {save_path}")
            ds = pickle.load(f)
    else:
        print('Generating data. Samples:')
        for _ in range(10):
            seq = dyck.generate(seq=[])
            assert seq['log_prob'] == dyck.compute_log_prob(seq['tokens'])
            print(dyck.detokenize(seq['tokens']), seq['log_prob'])
        ds = dyck.make_split(
            num_examples=sample_count,
            num_proc=16,
        )
        ds.set_format(type='torch')
        corpus_dir = os.path.dirname(save_path)
        os.makedirs(corpus_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            print(f"Saving ds to {save_path}")
            pickle.dump(ds, f)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
    )
    return loader, dyck, ds


def init_lm(args):
    cfg = {
        'vocab_size': get_vocab_size(args['language']['bracket_types'], special_tokens=args['language']['special_tokens']),
        'dim': args['lm']['dim'],
        'depth': args['lm']['num_layers'],
        'heads': args['lm']['num_heads'],
        'bos_token_id': args['lm']['bos_token_id'],
        'eos_token_id': args['lm']['eos_token_id'],
        'warmup': args['training']['warmup'],
    }
    return get_transformer(cfg)


def train(
    model,
    args,
    train_loader,
    eval_loader,
    dyck_eval,
    model_save_dir,
):
    if type(args['training']['num_iters']) is not int or args['training']['num_iters'] <= 0:
        num_iters = len(train_loader)
    else:
        num_iters = args['training']['num_iters']

    if args['training']['optimizer'] == 'Adam':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args['training']['learning_rate'],
            weight_decay=args['training']['weight_decay'],
        )
    else:
        raise NotImplementedError(f"optimizer {args['training']['optimizer']}")

    # linear warmup, linear decay to 0
    def lr_func(t):
        if t < args['training']['warmup']:
            return t / args['training']['warmup']
        else:
            return max((num_iters - t) / (num_iters - args['training']['warmup']), 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # EMA schedule
    def ema_decay_schedule(t):
        assert t >= 0
        if t < args['training']['warmup']:
            return 0.999
        else:
            return 1 - max((t+1) ** (-0.67), 0.001)

    # training loop
    os.makedirs(model_save_dir, exist_ok=True)
    eval_acc = 0.
    best_eval_acc = 0.
    ema_eval_acc = 0.
    best_ema_eval_acc = 0.
    bar = tqdm(train_loader, total=num_iters)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )
    for i, batch in enumerate(bar):
        if i == num_iters:
            break
        tokens = batch['tokens'].cuda()
        preds = model(tokens[:, :-1]).logits
        # print('preds', preds.__dict__.keys())  # debug
        # print('preds', preds.shape)  # debug
        labels = tokens[:, 1:].contiguous().view(-1)

        loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.shape[-1]), labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        ema_model.multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(
            ema_decay_schedule(i),
        )
        ema_model.update_parameters(model)
        scheduler.step()

        if (i + 1) % args['reporting']['steps_between_evals'] == 0 or i == num_iters - 1:
            eval_acc = dyck_eval.eval_acc(model, eval_loader)
            bar.set_description(f'loss: {loss.item():.4f}, eval_acc: {eval_acc:.6f}')
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                with open(os.path.join(model_save_dir, 'best.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                with open(os.path.join(model_save_dir, 'best_metrics.txt'), 'wt') as f:
                    f.write(f"step: {i}, eval_acc: {eval_acc}, train_loss: {loss}")

            ema_eval_acc = dyck_eval.eval_acc(ema_model, eval_loader)
            bar.set_description(f'loss: {loss.item():.4f}, ema_eval_acc: {ema_eval_acc:.6f}')
            if ema_eval_acc > best_ema_eval_acc:
                best_ema_eval_acc = ema_eval_acc
                ema_model.multi_avg_fn = None  # cannot pickle dump local object
                with open(os.path.join(model_save_dir, 'best_ema.pkl'), 'wb') as f:
                    pickle.dump(ema_model, f)
                with open(os.path.join(model_save_dir, 'best_ema_metrics.txt'), 'wt') as f:
                    f.write(f"step: {i}, eval_acc: {ema_eval_acc}, non-ema train_loss: {loss}")

        results = {
            'train_loss': loss,
            'eval_acc': eval_acc,
            'ema_eval_acc': ema_eval_acc,
        }
        wandb.log(results)

    with open(os.path.join(model_save_dir, 'final.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(model_save_dir, 'final_metrics.txt'), 'wt') as f:
        f.write(f"step: {num_iters}, eval_acc: {eval_acc}, train_loss: {loss}")

    # Update bn statistics for the ema_model at the end
    torch.optim.swa_utils.update_bn(train_loader, ema_model)
    ema_model.multi_avg_fn = None  # cannot pickle dump local object
    with open(os.path.join(model_save_dir, 'final_ema.pkl'), 'wb') as f:
        pickle.dump(ema_model, f)
    with open(os.path.join(model_save_dir, 'final_ema_metrics.txt'), 'wt') as f:
        f.write(f"step: {num_iters}, eval_acc: {ema_eval_acc}, train_loss: {loss}")


def generate_sequences(
        generated_data_path, 
        dyck, 
        model, 
        top_p,
        temperature,
        args=None,
        redo_if_exist=False, 
        batch_size=None,
    ):
    os.makedirs(
        os.path.dirname(generated_data_path),
        exist_ok=True,
    )

    if os.path.isfile(generated_data_path) and not redo_if_exist:
        with open(generated_data_path, 'rb') as f:
            print(f"Loading ds_generated from {generated_data_path}")
            ds_generated = pickle.load(f)
            sequences = ds_generated.to_list()
            print('len(sequences)', len(sequences))
            print('sequences[0]', sequences[0])
    else:
        if args is None and batch_size is None:
            batch_size = 1024
        if args is None:
            num_sequences = 10000
        else:
            num_sequences = args['language']['dev_sample_count']
        sequences = nn_generate(
            dyck,
            model,
            batch_size=batch_size or args['training']['batch_size'],
            num_sequences=num_sequences,
            top_p=top_p,  # nucleus sampling
            temperature=temperature,
        )
        ds_generated = datasets.Dataset.from_pandas(pd.DataFrame(sequences))
        ds_generated.set_format(type='torch')
        with open(generated_data_path, 'wb') as f:
            print(f"Saving ds_generated to {generated_data_path}")
            pickle.dump(ds_generated, f)
    return ds_generated, sequences


def calibration_plot(truth_logprobs, model_logprobs):
    plt.scatter(truth_logprobs, model_logprobs, alpha=0.5, marker='x', s=10, lw=1)
    plt.xlabel('ground truth log-probability')
    plt.ylabel('model log-probability')
    plt.grid()
    plt.axis('equal')


def calibration_metrics(groundtruth, pred):
    assert len(groundtruth) == len(pred)
    eps = 10 ** (-100)
    eces = []
    fraction_errors = []
    for i in range(len(groundtruth)):
        eces.append(abs(groundtruth[i] - pred[i][0]))
        fraction_errors.append(abs(groundtruth[i] - pred[i][0]) / (groundtruth[i] + eps))
    return {
        'eces': eces,
        'fraction_errors': fraction_errors,
    }


def eval_calibration(model, dyck, args, ds_train, ds_eval, ds_generated, calibration_save_path, ds_ood_eval=None):
    num_subplots = 3
    if ds_ood_eval is not None:
        num_subplots += 1

    plt.figure(figsize=(5 * num_subplots, 5))
    plt.suptitle(
        f"Dyck (length {dyck.length}, max depth {dyck.max_depth}), Transformer ({args['lm']['num_layers']}L {args['lm']['num_heads']}H, {args['lm']['dim']}d)",
        fontsize=12,
    )

    metrics_str = ""

    # check calibration for first 10k training examples
    groundtruth = ds_train['log_prob'][:10000].tolist()
    pred = nn_log_probs(model, ds_train['tokens'][:10000], batch_size=128)
    plt.subplot(1, num_subplots, 1)
    plt.title('training set')
    calibration_plot(
        groundtruth,
        pred,
    )
    metrics = calibration_metrics(groundtruth, pred)
    metrics_str += f"train ece: mean {np.mean(metrics['eces'])}, std {np.std(metrics['eces'])}\n"
    metrics_str += f"train fraction_error (diff / true): mean {np.mean(metrics['fraction_errors'])}, std {np.std(metrics['fraction_errors'])}\n\n"

    # do the same for eval set
    groundtruth = ds_eval['log_prob'].tolist()
    pred = nn_log_probs(model, ds_eval['tokens'], batch_size=128)
    plt.subplot(1, num_subplots, 2)
    plt.title('eval set')
    calibration_plot(
        groundtruth,
        pred,
    )
    metrics = calibration_metrics(groundtruth, pred)
    metrics_str += f"eval ece: mean {np.mean(metrics['eces'])}, std {np.std(metrics['eces'])}\n"
    metrics_str += f"eval fraction_error (diff / true): mean {np.mean(metrics['fraction_errors'])}, std {np.std(metrics['fraction_errors'])}\n\n"

    # check calibration for generated samples (ds_generated)
    groundtruth = [dyck.compute_log_prob(seq['tokens']) for seq in ds_generated.to_list()]
    pred = nn_log_probs(model, ds_generated['tokens'], batch_size=128)
    plt.subplot(1, num_subplots, 3)
    plt.title('model samples')
    calibration_plot(
        groundtruth,
        pred,
    )
    metrics = calibration_metrics(groundtruth, pred)
    metrics_str += f"generated ece: mean {np.mean(metrics['eces'])}, std {np.std(metrics['eces'])}\n"
    metrics_str += f"generated fraction_error (diff / true): mean {np.mean(metrics['fraction_errors'])}, std {np.std(metrics['fraction_errors'])}\n\n"

    if ds_ood_eval is not None:
        # check calibration for OOD samples (ds_ood_eval)
        groundtruth = [dyck.compute_log_prob(seq['tokens']) for seq in ds_ood_eval.to_list()]
        pred = nn_log_probs(model, ds_ood_eval['tokens'], batch_size=128)
        plt.subplot(1, num_subplots, 4)
        plt.title('OOD samples')
        calibration_plot(
            groundtruth,
            pred,
        )
        metrics = calibration_metrics(groundtruth, pred)
        metrics_str += f"ood ece: mean {np.mean(metrics['eces'])}, std {np.std(metrics['eces'])}\n"
        metrics_str += f"ood fraction_error (diff / true): mean {np.mean(metrics['fraction_errors'])}, std {np.std(metrics['fraction_errors'])}\n\n"

    plt.savefig(calibration_save_path)
    plt.show()

    calibration_metrics_path = calibration_save_path.replace('png', 'txt')
    with open(calibration_metrics_path, 'wt') as f:
        f.write(metrics_str)


def eval_entropy(model, dyck, args, ds_train, ds_eval, ds_generated, entropy_save_path, ds_ood_eval=None):
    num_subplots = 3
    if ds_ood_eval is not None:
        num_subplots += 1

    plt.figure(figsize=(5 * num_subplots, 5))
    plt.suptitle(
        f"Dyck (length {dyck.length}, max depth {dyck.max_depth}), Transformer ({args['lm']['num_layers']}L {args['lm']['num_heads']}H, {args['lm']['dim']}d)",
        fontsize=12,
    )

    # check entropy for first 10k training examples
    probs = nn_next_token_probs(model, ds_train['tokens'][:10000], batch_size=128)
    entropies = torch.flatten(
        probs.entropy()
    ).tolist()

    plt.subplot(1, num_subplots, 1)
    plt.title('training set')
    plt.hist(entropies, bins=100)

    # do the same for eval set
    probs = nn_next_token_probs(model, ds_eval['tokens'][:10000], batch_size=128)
    entropies = torch.flatten(
        probs.entropy()
    ).tolist()

    plt.subplot(1, num_subplots, 2)
    plt.title('eval set')
    plt.hist(entropies, bins=100)

    # do the same for generated samples (ds_generated)
    probs = nn_next_token_probs(model, ds_generated['tokens'][:10000], batch_size=128)
    entropies = torch.flatten(
        probs.entropy()
    ).tolist()

    plt.subplot(1, num_subplots, 3)
    plt.title('model samples')
    plt.hist(entropies, bins=100)

    if ds_ood_eval is not None:
        # do the same for OOD samples (ds_ood_eval)
        probs = nn_next_token_probs(model, ds_ood_eval['tokens'][:10000], batch_size=128)
        entropies = torch.flatten(
            probs.entropy()
        ).tolist()

        plt.subplot(1, num_subplots, 4)
        plt.title('OOD samples')
        plt.hist(entropies, bins=100)

    plt.savefig(entropy_save_path)
    plt.show()


def eval_top12_margin(model, dyck, args, ds_train, ds_eval, ds_generated, top12_margin_save_path, ds_ood_eval=None):
    """
    At each position, calc the margin between largest next-token prob and the second largest.
    """
    num_subplots = 3
    if ds_ood_eval is not None:
        num_subplots += 1

    plt.figure(figsize=(5 * num_subplots, 5))
    plt.suptitle(
        f"Dyck (length {dyck.length}, max depth {dyck.max_depth}), Transformer ({args['lm']['num_layers']}L {args['lm']['num_heads']}H, {args['lm']['dim']}d)",
        fontsize=12,
    )

    # check predicted prob margin (top 1 - top 2) for first 10k training examples
    probs = nn_next_token_probs(model, ds_train['tokens'][:10000], batch_size=128)  # (10000, seq_len+2, vocab_size)
    topk_values, _ = torch.topk(probs.probs, k=2, dim=-1)  # (10000, seq_len+2, 2)
    margins = torch.flatten(
        topk_values[:, :, 0] - topk_values[:, :, 1]
    ).tolist()

    plt.subplot(1, num_subplots, 1)
    plt.title('training set')
    plt.hist(margins, bins=100)

    # do the same for eval set
    probs = nn_next_token_probs(model, ds_eval['tokens'][:10000], batch_size=128)
    topk_values, _ = torch.topk(probs.probs, k=2, dim=-1)
    margins = torch.flatten(
        topk_values[:, :, 0] - topk_values[:, :, 1]
    ).tolist()

    plt.subplot(1, num_subplots, 2)
    plt.title('eval set')
    plt.hist(margins, bins=100)

    # do the same for generated samples (ds_generated)
    probs = nn_next_token_probs(model, ds_generated['tokens'][:10000], batch_size=128)
    topk_values, _ = torch.topk(probs.probs, k=2, dim=-1)
    margins = torch.flatten(
        topk_values[:, :, 0] - topk_values[:, :, 1]
    ).tolist()

    plt.subplot(1, num_subplots, 3)
    plt.title('model samples')
    plt.hist(margins, bins=100)

    if ds_ood_eval is not None:
        # do the same for OOD samples (ds_ood_eval)
        probs = nn_next_token_probs(model, ds_ood_eval['tokens'][:10000], batch_size=128)
        topk_values, _ = torch.topk(probs.probs, k=2, dim=-1)
        margins = torch.flatten(
            topk_values[:, :, 0] - topk_values[:, :, 1]
        ).tolist()

        plt.subplot(1, num_subplots, 4)
        plt.title('OOD samples')
        plt.hist(margins, bins=100)

    plt.savefig(top12_margin_save_path)
    plt.show()


def plot_stack_depth(seq, dyck):
    depth_seq = [0]
    for token in seq:
        if token == dyck.bos or token == dyck.eos:
            continue
        if token < dyck.num_types:
            depth_seq.append(depth_seq[-1] + 1)
        else:
            depth_seq.append(depth_seq[-1] - 1)

    plt.plot(depth_seq, alpha=0.2, c='k')


if __name__ == '__main__':
    # Load default arguments from a config file
    argp = ArgumentParser(add_help=False)
    argp.add_argument('--config', type=str)
    argp.add_argument('--train', action='store_true')
    argp.add_argument('--eval', action='store_true')
    argp.add_argument('--random_model', action='store_true')
    argp.add_argument('--best_or_final_checkpoint', type=str, default='best')
    argp.add_argument('--ood_eval', default=False)
    argp.add_argument('--ood_argmax_sampling_length', type=int, default=0)
    argp.add_argument('--ood_min_prefix_len', type=int, default=0)
    argp.add_argument('--top_p', type=float, default=0.0)
    argp.add_argument('--temperature', type=float, default=1.0)
    argp.add_argument('--plot_representations', action='store_true')
    cmd_args, rest = argp.parse_known_args()
    print('cmd_args', cmd_args)
    args = create_args(cmd_args.config)
    print("=== Running with arguments: ===")
    print(yaml.dump(args, default_flow_style=False))

    if 'type_probs' in args['language']:
        type_probs = [float(p) for p in args['language']['type_probs'].split('_')]
    else:
        type_probs = None

    train_loader, dyck_train, ds_train = load_dataset(
        {
            'length': args['language']['train_length'],
            'max_depth': args['language']['train_max_stack_depth'],
            'num_types': args['language']['bracket_types'],
            'type_probs': type_probs,
        },
        args['corpus']['train_corpus_loc'],
        args['language']['train_sample_count'],
        args['training']['batch_size'],
    )

    eval_loader, dyck_eval, ds_eval = load_dataset(
        {
            'length': args['language']['dev_length'],
            'max_depth': args['language']['dev_max_stack_depth'],
            'num_types': args['language']['bracket_types'],
            'type_probs': type_probs,
        },
        args['corpus']['dev_corpus_loc'],
        args['language']['dev_sample_count'],
        args['training']['batch_size'],
    )

    if cmd_args.train:
        for experiment_index in range(args['experiment']['repeat']):
            if args['name'] in {
                # list of runs to skip e.g. because they have been completed previously
            }:
                print(f"Skipped {args['name']} based on {__file__}")
                continue

            model_save_dir = args['reporting']['reporting_loc'] + str(experiment_index)
            model_save_path = os.path.join(model_save_dir, 'final.pkl')
            if os.path.exists(model_save_path):
                print(f"Skipped {args['name']} since {model_save_path} already exists")
                continue

            # Construct the language model
            print('Construct the language model with args', args)
            model = init_lm(args)

            wandb.init(
                project="gpt_dyck",
                name=args['name'] + str(experiment_index),
                reinit=True,
            )
            wandb.config = args

            train(
                model,
                args,
                train_loader,
                eval_loader,
                dyck_eval,
                model_save_dir,
            )
    else:
        print('Training skipped')

    if cmd_args.eval:
        for experiment_index in range(args['experiment']['repeat']):
            model_save_dir = args['reporting']['reporting_loc'] + str(experiment_index)

            if cmd_args.random_model:
                model = init_lm(args)
            else:
                print('Checkpoint to eval:', cmd_args.best_or_final_checkpoint)
                model_save_path = os.path.join(model_save_dir, f'{cmd_args.best_or_final_checkpoint}.pkl')
                if not os.path.exists(model_save_path):
                    print(f"Skipped {args['name']} since {model_save_path} does not exist")
                    continue
                with open(model_save_path, 'rb') as f:
                    model = pickle.load(f)

            # generate some sequences
            if cmd_args.random_model:
                eval_dir = os.path.join(model_save_dir, f'generated_random_p{cmd_args.top_p}_t{cmd_args.temperature}/')
            else:
                eval_dir = os.path.join(model_save_dir, f'generated_{cmd_args.best_or_final_checkpoint}_p{cmd_args.top_p}_t{cmd_args.temperature}/')
            os.makedirs(eval_dir, exist_ok=True)
            generated_data_path = os.path.join(eval_dir, f'generated_p{cmd_args.top_p}_t{cmd_args.temperature}.pkl')
            ds_generated, sequences = generate_sequences(
                generated_data_path,
                dyck_train,
                model,
                cmd_args.top_p,
                cmd_args.temperature,
                args=args,
            )

            illegal_seqs = []

            for i, seq in enumerate(sequences):
                if not dyck_eval.accept(seq['tokens']):
                    # print(
                    #     dyck_eval.detokenize(seq['tokens']),
                    #     nn_log_probs(model, [seq['tokens']], batch_size=1),
                    #     dyck_eval.compute_log_prob(seq['tokens']),
                    # )
                    illegal_seqs.append(seq['tokens'])
            print('Number of invalid sequences: ', len(illegal_seqs))

            # analyze and save errors
            analyze_and_save_errors(dyck_eval, model, illegal_seqs, f'{eval_dir}/errors.csv')

            if cmd_args.ood_eval:
                ood_eval_loader, dyck_ood_eval, ds_ood_eval = load_dataset(
                    {
                        'length': args['language']['ood_eval_length'],
                        'max_depth': args['language']['ood_eval_max_stack_depth'],
                        'num_types': args['language']['bracket_types'],
                        'type_probs': [float(p) for p in args['language']['ood_eval_type_probs'].split('_')],
                    },
                    args['corpus']['ood_eval_corpus_loc'],
                    args['language']['ood_eval_sample_count'],
                    args['training']['batch_size'],
                )

                ood_sequences = ds_ood_eval.to_list()
                # print('ood_sequences', ood_sequences)  # debug

                # Error analysis for OOD sequences
                analyze_and_save_ood_errors(
                    ood_sequences,
                    dyck_ood_eval,
                    model,
                    args['training']['batch_size'],
                    cmd_args.top_p,
                    eval_dir,
                    argmax_length=cmd_args.ood_argmax_sampling_length,
                    min_prefix_len=cmd_args.ood_min_prefix_len,
                )
                ood_results = read_ood_eval_results(eval_dir)
                
                calibration_figure_name = f"calibration_ood_{args['language']['ood_eval_type_probs']}.png"
            else:
                ds_ood_eval = None
                ood_results = None
                calibration_figure_name = "calibration.png"

            # # check calibration
            # calibration_save_path = os.path.join(eval_dir, calibration_figure_name)
            # eval_calibration(model, dyck_train, args, ds_train, ds_eval, ds_generated, calibration_save_path, ds_ood_eval=ds_ood_eval)
            # entropy_save_path = calibration_save_path.replace('calibration', 'entropy')
            # eval_entropy(model, dyck_train, args, ds_train, ds_eval, ds_generated, entropy_save_path, ds_ood_eval=ds_ood_eval)
            # margin_save_path = calibration_save_path.replace('calibration', 'margin')
            # eval_top12_margin(model, dyck_train, args, ds_train, ds_eval, ds_generated, margin_save_path, ds_ood_eval=ds_ood_eval)
            #
            # ## generated
            # stack_depth_generated_save_path = os.path.join(eval_dir, 'stack_depth_generated.png')
            # plt.figure()
            # for i in range(10000):
            #     plot_stack_depth(sequences[i]['tokens'], dyck_eval)
            # plt.savefig(stack_depth_generated_save_path)
            # plt.show()
            #
            # ## groundtruth
            # train_tokens = ds_train['tokens'].numpy().tolist()
            # stack_depth_groundtruth_save_path = os.path.join(eval_dir, 'stack_depth_groundtruth.png')
            # plt.figure()
            # for i in range(10000):
            #     plot_stack_depth(train_tokens[i], dyck_eval)
            # plt.savefig(stack_depth_groundtruth_save_path)
            # plt.show()
    else:
        print('Eval skipped')

    if cmd_args.plot_representations:
        for experiment_index in range(args['experiment']['repeat']):
            model_save_dir = args['reporting']['reporting_loc'] + str(experiment_index)

            if cmd_args.random_model:
                model = init_lm(args)
            else:
                print('Checkpoint to eval:', cmd_args.best_or_final_checkpoint)
                model_save_path = os.path.join(model_save_dir, f'{cmd_args.best_or_final_checkpoint}.pkl')
                if not os.path.exists(model_save_path):
                    print(f"Skipped {args['name']} since {model_save_path} does not exist")
                    continue
                with open(model_save_path, 'rb') as f:
                    model = pickle.load(f)

            # generated sequences
            if cmd_args.random_model:
                eval_dir = os.path.join(model_save_dir, f'generated_random_p{cmd_args.top_p}/')
            else:
                eval_dir = os.path.join(model_save_dir, f'generated_{cmd_args.best_or_final_checkpoint}_p{cmd_args.top_p}/')
            os.makedirs(eval_dir, exist_ok=True)
            generated_data_path = os.path.join(eval_dir, f'generated_p{cmd_args.top_p}.pkl')
            ds_generated, sequences = generate_sequences(
                generated_data_path,
                dyck_train,
                model,
                cmd_args.top_p,
                args=args,
            )

            # sequence TSNE plot
            all_intermediates = nn_intermediates(
                model,
                ds_generated['tokens'][:10000],
                batch_size=args['training']['batch_size'],
                positions='last',
            )
            print('len(all_intermediates)', len(all_intermediates))
            print('len(all_intermediates[0])', len(all_intermediates[0]))

            for random_state in range(2):
                X_embedded = TSNE(perplexity=30.0, random_state=random_state).fit_transform(np.array(all_intermediates))
                print('X_embedded.shape', X_embedded.shape)
                rep_seq_save_dir = os.path.join(eval_dir, 'rep_seq/')
                os.makedirs(rep_seq_save_dir, exist_ok=True)

                ## representations of high predicted-prob sequences vs. low ones
                rep_pred_seq_prob_save_path = os.path.join(rep_seq_save_dir, f'rep_pred_seq_prob{random_state}.png')
                plt.figure(figsize=(12, 12))
                plt.scatter(
                    x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    c=nn_log_probs(model, ds_generated['tokens'], batch_size=128),
                    cmap='viridis',
                    alpha=0.5,
                )
                plt.savefig(rep_pred_seq_prob_save_path)
                plt.show()

                ## representations of correct sequences vs. incorrect ones
                rep_seq_correctness_save_path = os.path.join(rep_seq_save_dir, f'rep_seq_correctness{random_state}.png')
                plt.figure(figsize=(12, 12))
                plt.scatter(
                    x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    c=[1 if dyck_train.accept(seq['tokens']) else 0 for seq in sequences],
                    cmap='viridis',
                    alpha=0.5,
                )
                plt.savefig(rep_seq_correctness_save_path)
                plt.show()

                ## representations of deep sequences vs. shallow ones
                rep_seq_depth_save_path = os.path.join(rep_seq_save_dir, f'rep_seq_depth{random_state}.png')
                plt.figure(figsize=(12, 12))
                plt.scatter(
                    x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    c=[dyck_train.get_max_depth(seq['tokens']) for seq in sequences],
                    cmap='viridis',
                    alpha=0.5,
                )
                plt.savefig(rep_seq_depth_save_path)
                plt.show()

            # prefix TSNE plot
            plot_prefix_rep(
                model,
                sequences[:100],
                dyck_train,
                ds_generated['tokens'][:100],
                os.path.join(eval_dir, 'rep_prefix/'),
            )

            # OOD prefix TSNE plot
            if cmd_args.ood_eval:
                plot_prefix_rep(
                    model,
                    ood_sequences[:200],
                    dyck_train,
                    ds_ood_eval['tokens'][:200],
                    os.path.join(eval_dir, 'rep_prefix_ood/'),
                    ood_results=ood_results,
                )
    else:
        print('Plot representations skipped')
