import hydra
from omegaconf import OmegaConf

import collections
from collections.abc import Iterable
import copy
# import ctranslate2
import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
import sys
import torch
import time
from transformers import LlamaForCausalLM, AutoTokenizer
import wandb

from code_lm_benchmark import (
    GenerateTestCaseTask,
    GenerateTestCaseTaskBenchmark,
    GenerateTestCaseTaskBenchmarkParallel,
    ErrorType,
)
from lm import (
    conditional_nn_generate,
    predict_error,
    sample_top_p,
)
from train_err_pred import MLP


@hydra.main(config_path='../hydra_configs', config_name='main_codellama', version_base=None)
def main(cfg):
    # add runtime info to cfg
    OmegaConf.set_struct(cfg, False)
    cfg.meta = OmegaConf.create({})
    cfg.meta.original_dir = hydra.utils.get_original_cwd()
    cfg.meta.run_dir = os.getcwd()
    cfg.meta.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    err_pred_mode = 'Trained'
    if cfg.error_predictor.use_groundtruth:
        err_pred_mode = 'GT'
    elif cfg.error_predictor.random:
        err_pred_mode = f'Random_th{cfg.generation_configs.err_pred_threshold}'
    else:
        err_pred_mode += f'mlp{cfg.error_predictor.num_mlp_layer}_th{cfg.generation_configs.err_pred_threshold}_rep{cfg.error_predictor.layer}'
    if cfg.generation_configs.block_err_pred:
        err_pred_mode += 'Block'
    group_name = f"quota{cfg.generation_configs.backtrack_quota}"
    if cfg.generation_configs.backtrack_quota > 0:
        group_name += f"_stride{cfg.generation_configs.backtrack_stride}"
    if cfg.generation_configs.block_best_of_n > 1:
        group_name += f"_bon{cfg.generation_configs.block_best_of_n}"
    group_name += f"_p{cfg.generation_configs.top_p}_t{cfg.generation_configs.temperature}"
    if not cfg.generation_configs.redo_backtrack_with_argmax:
        group_name += "_noargmax"
    group_name += f"_errPred{err_pred_mode}_no2back"[-127:]
    if 'Instruct' in cfg.model.name:
        group_name = 'instruct_' + group_name
    if type(cfg.fs.prompts_and_generations_local_path) != str:
        group_name = 'ood_' + group_name
    run_name = f"{group_name}_{cfg.seed}"
    print(f"Reporting to wandb {run_name}")
    run_id = (run_name+'_try5')[-127:]  # TODO update
    print('run_id:', run_id)
    wandb.init(
        project=f"codellama_append_v5",
        group=group_name,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        resume="auto",
        reinit=True,
        id=run_id,
    )

    model = LlamaForCausalLM.from_pretrained(cfg.model.name).to(cfg.meta.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    if cfg.error_predictor.random:
        model_err_pred = 'random'
    else:
        model_err_pred_fn = f'data/model_err_pred_layer{cfg.error_predictor.layer}.pkl'
        if cfg.error_predictor.num_mlp_layer > 1:
            model_err_pred_fn = model_err_pred_fn.replace(
                'err_pred',
                f'{cfg.error_predictor.num_mlp_layer}layer_err_pred',
            )
        print('class of error predictor:', MLP)
        if cfg.generation_configs.block_err_pred:
            model_err_pred_fn = model_err_pred_fn.replace(
                'err_pred',
                'block_err_pred',
            )
        model_err_pred = MLP().to(cfg.meta.device)
        model_err_pred_save_path = os.path.join(
            cfg.meta.original_dir,
            model_err_pred_fn,
        )
        with open(model_err_pred_save_path, 'rb') as f:
            model_err_pred = pickle.load(f)  # requires class `MLP` be defined
        model_err_pred.eval()

    print(
        'cfg.fs.prompts_and_generations_local_path',
        type(cfg.fs.prompts_and_generations_local_path),
        f"value = {cfg.fs.prompts_and_generations_local_path}",
    )
    if type(cfg.fs.prompts_and_generations_local_path) != str:
        # generate new tasks with specified function names
        print(f'generating new prompts with function names in {cfg.fs.prompts_and_generations_local_path}')
        function_names_in_train = {'ovs', 'cyk', 'mcl', 'heh', 'fgu', 'knk', 'zmf'}
        prompts_and_generations = {}
        for function_name in cfg.fs.prompts_and_generations_local_path:
            task = GenerateTestCaseTask(
                1,  # num_demonstrations
                8,  # num_test_cases_per_function
                function_name_length=3,
                function_name=function_name,
            )
            if task.function_name in function_names_in_train:
                raise ValueError(f"function name exists in training set: {task.function_name}")
            prompts_and_generations[task.prompt] = {
                'top_p0.8': [None] * 10,
            }
    elif cfg.fs.prompts_and_generations_local_path:
        print(f'loading prompts from {cfg.fs.prompts_and_generations_local_path}')
        prompts_and_generations_path = os.path.join(
            cfg.meta.original_dir,
            cfg.fs.prompts_and_generations_local_path,
        )
        with open(prompts_and_generations_path, 'rb') as f:
            prompts_and_generations = pickle.load(f)
    else:
        # generate new tasks
        print('generating new prompts')
        function_names_in_train = {'ovs', 'cyk', 'mcl', 'heh', 'fgu', 'knk', 'zmf'}
        prompts_and_generations = {}
        for _ in range(10):
            task = GenerateTestCaseTask(
                1,  # num_demonstrations
                8,  # num_test_cases_per_function
                function_name_length=3,
                function_name=None,
            )
            if task.function_name in function_names_in_train:
                raise ValueError(f"function name exists in training set: {task.function_name}")
            prompts_and_generations[task.prompt] = {
                'top_p0.8': [None] * 10,
            }

    prompts_and_generations_backtrack = []
    backtracked_prefixes = []
    error_parsing_corner_cases = []
    # aggregate_metrics = collections.defaultdict(list)
    aggregate_metrics_main = collections.defaultdict(float)
    num_test_cases_expected = 0

    for prompt in prompts_and_generations:
        print('prompt:', prompt)  # debug
        token_ids_prompt = tokenizer(prompt)['input_ids']
        print('token_ids_prompt', token_ids_prompt)  # debug
        max_length = cfg.generation_configs.max_new_tokens + len(token_ids_prompt)
        function_name = prompt.split('above function ')[-1].replace(', one in each line:', '')
        assert len(function_name) == 3
        num_demonstrations = 1
        num_test_cases_per_function = int(prompt.split('\n')[-1].split(' test cases')[0].replace('List ', ''))
        assert num_test_cases_per_function == 8
        task = GenerateTestCaseTask(
            num_demonstrations,
            num_test_cases_per_function,
            function_name_length=len(function_name),
            function_name=function_name,
        )
        test_cases_repetitions = []
        prev_num_different_correct = 0

        for generation_config_name in ['top_p0.8']:
            print('num repetitions:', len(prompts_and_generations[prompt][generation_config_name]))
            for i, generation in enumerate(prompts_and_generations[prompt][generation_config_name]):
                print('\n' + '#' * 16 + '\n')
                num_test_cases_expected += task.num_test_cases_per_function
                if cfg.error_predictor.use_groundtruth:
                    total_backtrack_strides = 0
                    result = model.generate(
                            torch.tensor(
                                [token_ids_prompt],
                                device=cfg.meta.device,
                            ),
                            max_length=max_length,
                            max_new_tokens=None,
                            do_sample=True,
                            top_p=cfg.generation_configs.top_p,
                            temperature=cfg.generation_configs.temperature,
                        ).detach().tolist()[0]

                    assert result[0] == tokenizer.bos_token_id
                    assert result[:len(token_ids_prompt)] == token_ids_prompt

                    remaining_backtrack_quota = cfg.generation_configs.backtrack_quota
                    assert type(remaining_backtrack_quota) is int
                    assert remaining_backtrack_quota >= 0
                    while remaining_backtrack_quota > 0:
                        prompt_and_generation_str = tokenizer.convert_tokens_to_string(
                            tokenizer.convert_ids_to_tokens(result),
                        )
                        token_ids_generation = result[len(token_ids_prompt):]
                        try:
                            error_token_pos_generation, err_type = task.get_first_error_token(
                                token_ids_generation, 
                                tokenizer, 
                                verbose=1,
                            )
                        except:
                            error_parsing_corner_cases.append(
                                result,
                            )
                            print(f'corner case: {prompt_and_generation_str}')
                            break

                        if error_token_pos_generation is None or err_type == ErrorType.EXTRA_CONTENT_AFTER_TASK_COMPLETION:
                            print('correct according to groundtruth predictor')
                            break
                        
                        print(f'error found by groundtruth predictor:\n\n {prompt_and_generation_str}')
                        if error_token_pos_generation == len(token_ids_generation):
                            print('error token at end')
                            error_token_pos_generation -= 1
                        else:
                            print('error token:', tokenizer.decode(token_ids_generation[error_token_pos_generation]))
                        keep_until = max(1, error_token_pos_generation - cfg.generation_configs.backtrack_stride + 1)
                        total_backtrack_strides += len(token_ids_generation) - keep_until + 1
                        prompt_and_partial_generation_ids = token_ids_prompt + token_ids_generation[:keep_until]

                        # argmax portion
                        result_argmax = model.generate(
                            torch.tensor(
                                [prompt_and_partial_generation_ids],
                                device=cfg.meta.device,
                            ),
                            max_tokens=None,
                            max_new_tokens=cfg.generation_configs.backtrack_stride,
                            top_k=1,  # argmax
                            do_sample=False,  # argmax
                        )

                        # sampling portion
                        if len(result_argmax[0]) < max_length:
                            result = model.generate(
                                result_argmax,
                                max_length=max_length,
                                max_new_tokens=None,
                                do_sample=True,
                                top_p=cfg.generation_configs.top_p,
                                temperature=cfg.generation_configs.temperature,
                            ).detach().tolist()[0]
                        else:
                            result = result_argmax.detach().tolist()[0]
                        assert result[0] == tokenizer.bos_token_id
                        assert result[:len(prompt_and_partial_generation_ids)] == prompt_and_partial_generation_ids
                        remaining_backtrack_quota -= 1
                        print('\n' + '## backtracked with groundtruth predictor ##' + '\n')
                
                elif cfg.generation_configs.block_best_of_n > 1:
                    assert cfg.generation_configs.block_err_pred
                    assert model_err_pred is not None
                    assert tokenizer is not None

                    # prepare special symbols
                    new_line_token_ids = []
                    for token in tokenizer.vocab:
                        if '\n' in token or '<0x0A>' in token or '<0x0D>' in token:
                            new_line_token_ids.append(tokenizer.vocab[token])
                    assert len(new_line_token_ids) > 0, \
                        f"new_line_token_ids = {new_line_token_ids}. Did not specify where to break into blocks."

                    with torch.no_grad():
                        prompt_and_partial_generation_ids = token_ids_prompt
                        for line_idx in range(task.num_test_cases_per_function):
                            step = len(prompt_and_partial_generation_ids)
                            scores2candidate = {}
                            for candidate_idx in range(cfg.generation_configs.block_best_of_n):
                                candidate_step = step
                                candidate_prompt_and_partial_generation_ids = copy.deepcopy(prompt_and_partial_generation_ids)
                                next_tokens = torch.tensor([0.5])  # will be updated in the loop
                                while candidate_step < max_length:
                                    reached_eos = torch.all(next_tokens == tokenizer.eos_token_id)
                                    candidate_is_complete = candidate_step >= step + 3 \
                                        and candidate_prompt_and_partial_generation_ids[-1] in new_line_token_ids
                                    if reached_eos or candidate_is_complete:
                                        err_pred_prob = predict_error(
                                                None,  # dyck
                                                model,
                                                [candidate_prompt_and_partial_generation_ids],
                                                model_err_pred,
                                                use_groundtruth_error_predictor=cfg.error_predictor.use_groundtruth,
                                                return_probs=True,
                                        )[0][1].item()
                                        assert type(err_pred_prob) is float, \
                                            f"err_pred_prob={err_pred_prob}, type={type(err_pred_prob)}, expect float"
                                        scores2candidate[err_pred_prob] = candidate_prompt_and_partial_generation_ids
                                        break

                                    # predict next token
                                    batch_tensor = torch.tensor(
                                        [candidate_prompt_and_partial_generation_ids],
                                        dtype=torch.int64,
                                    ).cuda()
                                    preds = model(batch_tensor).logits
                                    next_tokens, log_probs = sample_top_p(
                                        preds,
                                        cfg.generation_configs.top_p,
                                        candidate_step,
                                        temperature=cfg.generation_configs.temperature,
                                    )
                                    candidate_prompt_and_partial_generation_ids.append(next_tokens[0].item())
                                    candidate_step += 1
                                    del preds
                                    del batch_tensor

                            # select the best candidate
                            if len(scores2candidate) > 0:
                                prompt_and_partial_generation_ids = scores2candidate[min(scores2candidate.keys())]

                    # Gather final output
                    result = prompt_and_partial_generation_ids
                    total_backtrack_strides = 0

                else:
                    output = conditional_nn_generate(
                        None,  # dyck
                        model,
                        [token_ids_prompt],
                        batch_size=1,
                        top_p=cfg.generation_configs.top_p,
                        temperature=cfg.generation_configs.temperature,
                        argmax_length=0,
                        backtrack_quota=cfg.generation_configs.backtrack_quota,
                        backtrack_stride=cfg.generation_configs.backtrack_stride,
                        max_allowed_prefix_truncation=0,
                        backtracked_prefixes=backtracked_prefixes,
                        model_err_pred=model_err_pred,
                        allow_2nd_backtrack_at_same_prefix=False,  # TODO: try to avoid running into same prefix 2nd time
                        use_groundtruth_error_predictor=cfg.error_predictor.use_groundtruth,
                        tokenwise_dfs=False,
                        block_err_pred=cfg.generation_configs.block_err_pred,
                        max_new_tokens=cfg.generation_configs.max_new_tokens,
                        err_pred_threshold=cfg.generation_configs.err_pred_threshold,
                        tokenizer=tokenizer,
                        redo_backtrack_with_argmax=cfg.generation_configs.redo_backtrack_with_argmax,
                        verbose=0,
                    )
                    print('output', output)  # debug
                    result = output[0]['tokens']
                    total_backtrack_strides = output[0]['total_backtrack_strides']

                # metrics
                prompt_and_generation_str = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(result),
                )
                print('prompt_and_generation:\n\n', prompt_and_generation_str)  # debug
                print('\n\n')  # debug
                generation_str = prompt_and_generation_str[len(prompt)+5:]  # +6 because of ' <s> '
                print('generation:\n\n', generation_str)  # debug

                test_cases_single = task.parse(
                    generation_str, 
                    test_cases=[],
                    verbose=1,
                )
                if len(test_cases_single) > num_test_cases_per_function:
                    print(f"Warning: generated {len(test_cases_single)} test cases although instructed to generate only {num_test_cases_per_function}")
                    test_cases_single = test_cases_single[:num_test_cases_per_function]
                metrics_single = task.check(test_cases_single, verbose=1)
                for test_case in test_cases_single:
                    test_cases_repetitions.append(test_case)
                metrics_repetitions = task.check(test_cases_repetitions, verbose=1)

                prompts_and_generations_backtrack.append(result)
                
                # for metric_name in metrics_repetitions:
                #     metrics_single[f'{metric_name}_repetitions'] = metrics_repetitions[metric_name]
                # wandb.log(metrics_single)

                # only keep main metrics
                aggregate_metrics_main['requested'] = num_test_cases_expected
                aggregate_metrics_main['correct'] += metrics_single['num_correct']
                aggregate_metrics_main['diff_correct'] += metrics_repetitions['num_different_correct'] - prev_num_different_correct
                prev_num_different_correct = metrics_repetitions['num_different_correct']
                aggregate_metrics_main['generated'] += metrics_single['num_total']
                aggregate_metrics_main['completion_rate'] = aggregate_metrics_main['correct'] / num_test_cases_expected
                aggregate_metrics_main['diverse_completion_rate'] = aggregate_metrics_main['diff_correct'] / num_test_cases_expected
                assert result[:len(token_ids_prompt)] == token_ids_prompt
                token_ids_generation = result[len(token_ids_prompt):]
                aggregate_metrics_main['queries'] += len(token_ids_generation) + total_backtrack_strides
                wandb.log(aggregate_metrics_main)
                print(f"requested {num_test_cases_expected}, generated {aggregate_metrics_main['generated']}, diverse_completion_rate %.3f" % aggregate_metrics_main['diverse_completion_rate'])

        # aggregate_metrics_stat = {}
        # for metric_name in metrics_repetitions:
        #     aggregate_metrics[metric_name].append(metrics_repetitions[metric_name])
        #     aggregate_metrics_stat[f'{metric_name}_all'] = np.mean(aggregate_metrics[metric_name])
        # aggregate_metrics_stat['generated_all'] = sum(aggregate_metrics['num_total'])
        # aggregate_metrics_stat['correct_rate_all'] = sum(aggregate_metrics['num_correct']) / aggregate_metrics_stat['generated_all']
        # aggregate_metrics_stat['num_requested_all'] = num_test_cases_expected
        # aggregate_metrics_stat['completion_rate_all'] = sum(aggregate_metrics['num_correct']) / num_test_cases_expected
        # aggregate_metrics_stat['diff_completion_rate_all'] = sum(aggregate_metrics['num_different_correct']) / num_test_cases_expected
        # wandb.log(aggregate_metrics_stat)
        # print(f"expected {num_test_cases_expected}, generated {aggregate_metrics_stat['generated_all']}, diverse acc %.3f" % aggregate_metrics_stat['diff_completion_rate_all'])
    
    # save results
    with open(f'prompts_and_generations_backtrack_{run_name}.pkl', 'wb') as f:
        pickle.dump(prompts_and_generations_backtrack, f)
    with open(f'backtracked_prefixes_{run_name}.pkl', 'wb') as f:
        pickle.dump(backtracked_prefixes, f)
    with open(f'error_parsing_corner_cases_{run_name}.pkl', 'wb') as f:
        pickle.dump(error_parsing_corner_cases, f)
    
    wandb.finish()


if __name__ == '__main__':
    main()
