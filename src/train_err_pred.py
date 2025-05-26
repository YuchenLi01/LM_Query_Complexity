import argparse
import copy
import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaModel, GPT2Model
from tqdm import tqdm

from lm import nn_intermediates


class SingleLabelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        x = self.data[ind][1:]
        y = self.data[ind][0]
        return x, y


class MLP(nn.Module):
    def __init__(
            self,
            input_size=4096,
            hidden_size=256,
            output_size=2,
            num_layers=1,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        # Define a sequential container for the layers
        layers = []
        if num_layers == 1:
            # Single-layer MLP
            layers.append(nn.Linear(input_size, output_size))
        else:
            # Input layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(hidden_size, output_size))

        # Register the layers in the module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def test(model_err_pred, test_loader, criterion, device):
    model_err_pred.eval()
    all_ys = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)

    losses = []
    with torch.no_grad():
        for batch_num, input_data in enumerate(test_loader):
            x, y = input_data
            x = x.to(device).float()
            y = y.type(torch.LongTensor).to(device)

            output = model_err_pred(x)
            loss = criterion(output, y)
            losses.append(loss.item())

            pred = output.argmax(dim=-1)

            all_ys = torch.concat([all_ys, y.detach()])
            all_preds = torch.concat([all_preds, pred.detach()])

    test_acc = sum(all_preds == all_ys) / len(all_ys)
    print('Test Loss %6.2f | test acc %.2f' % (sum(losses)/len(losses), test_acc))
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train error predictor for CodeLlama")
    parser.add_argument("--model", default="codellama/CodeLlama-7b-hf", type=str)
    parser.add_argument("--dataset", default="data/prompts_and_generations_codellama_append_v4.pkl", type=str)
    parser.add_argument("--layer", default=31, type=int)
    parser.add_argument("--errors_hiddens_path", default='', type=str)
    parser.add_argument("--correct_hiddens_path", default='', type=str)
    parser.add_argument("--train_err_pred_batch_size", default=2048, type=int)
    parser.add_argument("--best_model_err_pred_save_path", default='', type=str)
    parser.add_argument("--block", default=False, type=bool)
    parser.add_argument("--err_pred_num_layers", default=1, type=int)
    args = parser.parse_args()
    print(f"args:\n\n {args}")

    # format of the generated results (a Python dict)
    # prompt: [generations0, generations1, generations2, ...]
    with open(args.dataset, 'rb') as f:
        prompts_and_generations = pickle.load(f)

    model_name = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LlamaModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"loaded model {model_name} on device {model.device}")
    newline_token_ids = set()
    for token in tokenizer.vocab:
        if '\n' in token or '<0x0A>' in token or '<0x0D>' in token:
            newline_token_ids.add(
                tokenizer.convert_tokens_to_ids(token)
            )
    print('newline_token_ids', newline_token_ids)

    # extract hidden rep
    if not args.errors_hiddens_path:
        args.errors_hiddens_path = f'data/errors_hiddens_layer{args.layer}.pkl'
    if not args.correct_hiddens_path:
        args.correct_hiddens_path = f'data/correct_hiddens_layer{args.layer}.pkl'
    if os.path.exists(args.errors_hiddens_path) and os.path.exists(args.correct_hiddens_path):
        print('loading reps')
        with open(args.errors_hiddens_path, 'rb') as f:
            errors_hiddens = pickle.load(f)
        with open(args.correct_hiddens_path, 'rb') as f:
            correct_hiddens = pickle.load(f)
        print('loaded reps')

    else:
        print('extracting reps from scratch')
        errors_hiddens = []
        correct_hiddens = []

        from code_lm_benchmark import (
            GenerateTestCaseTask,
            ErrorType,
        )

        error_parsing_corner_cases = []
        num_reps_extracted = 0
        num_generations = 0

        for prompt in prompts_and_generations:
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

            for i, generation in enumerate(prompts_and_generations[prompt]):
                # print(prompt + generation)
                num_generations += 1
                token_ids_prompt = tokenizer(prompt)['input_ids']
                token_ids_prompt_and_generation = tokenizer(prompt + generation)['input_ids']
                assert token_ids_prompt_and_generation[-1] != tokenizer.eos_token_id  # may need to be updated if generate eos = True
                # token_ids_prompt_and_generation.append(tokenizer.eos_token_id)  # may need to be updated if generate eos = True
                token_ids_generation = token_ids_prompt_and_generation[len(token_ids_prompt):]

                try:
                    error_token_pos_generation, error_type = task.get_first_error_token(
                        token_ids_generation,
                        tokenizer,
                        verbose=1,
                    )
                except:
                    error_parsing_corner_cases.append(
                        (prompt, i),
                    )
                    continue

                hidden_rep = nn_intermediates(
                        model,
                        [token_ids_prompt_and_generation],
                        batch_size=1,
                        positions='all',
                        layer=args.layer,
                    )

                if error_token_pos_generation is not None:
                    # print('error_type', error_type)
                    if args.block:
                        break_after_idx = error_token_pos_generation
                        while break_after_idx < len(token_ids_generation) \
                                and token_ids_generation[break_after_idx] not in newline_token_ids:
                            break_after_idx += 1
                        break_before_idx = error_token_pos_generation - 1
                        while break_before_idx >= 0 \
                                and token_ids_generation[break_before_idx] not in newline_token_ids:
                            break_before_idx -= 1

                        error_token_pos_all = break_after_idx + len(token_ids_prompt)
                        if error_token_pos_generation == len(token_ids_generation):  # may need to be updated if generate eos = True
                            error_token_pos_all -= 1  # premature ending
                        errors_hiddens.append(
                            hidden_rep[0][error_token_pos_all],
                        )
                        num_reps_extracted += 1

                        correct_pos = break_before_idx + len(token_ids_prompt)
                        correct_hiddens.append(hidden_rep[0][correct_pos])
                        num_reps_extracted += 1
                    else:
                        if error_type != ErrorType.EXTRA_CONTENT_AFTER_TASK_COMPLETION:
                            # print('generation', generation)
                            error_token_pos_all = error_token_pos_generation + len(token_ids_prompt)
                            if error_token_pos_generation == len(token_ids_generation):  # may need to be updated if generate eos = True
                                error_token_pos_all -= 1  # premature ending

                            errors_hiddens.append(
                                hidden_rep[0][error_token_pos_all],
                            )
                            num_reps_extracted += 1
                            # for correct_pos in range(len(token_ids_prompt), error_token_pos_all):
                            if error_token_pos_all > len(token_ids_prompt):
                                correct_pos = random.choice(range(len(token_ids_prompt), error_token_pos_all))
                                correct_hiddens.append(hidden_rep[0][correct_pos])
                                num_reps_extracted += 1
                        else:
                            # TODO: may truncate generation until `error_token_pos_generation` and treat as correct
                            pass
                else:
                    if args.block:
                        if len(token_ids_prompt_and_generation) > len(token_ids_prompt):
                            correct_hiddens.append(hidden_rep[0][len(token_ids_prompt_and_generation)-1])
                            num_reps_extracted += 1
                    else:
                        # for correct_pos in range(len(token_ids_prompt), len(token_ids_prompt_and_generation)):
                        if len(token_ids_prompt_and_generation) > len(token_ids_prompt):
                            correct_pos = random.choice(range(len(token_ids_prompt), len(token_ids_prompt_and_generation)))
                            correct_hiddens.append(hidden_rep[0][correct_pos])
                            num_reps_extracted += 1
                print('\n\n')

            with open(args.errors_hiddens_path, 'wb') as f:
                pickle.dump(errors_hiddens, f)

            with open(args.correct_hiddens_path, 'wb') as f:
                pickle.dump(correct_hiddens, f)

            print('num_generations', num_generations)
            print('num_reps_extracted', num_reps_extracted)

    # features for training error predictor
    train_proportion = 0.9
    train_idx_ends_positive = int(len(errors_hiddens) * train_proportion)
    train_idx_ends_negative = int(len(correct_hiddens) * train_proportion)
    print('positive to negative ratio:', len(correct_hiddens) / len(errors_hiddens))
    train_data = [
             [1] + rep
             for rep in errors_hiddens[:train_idx_ends_positive]
         ] + [
             [0] + rep
             for rep in correct_hiddens[:train_idx_ends_negative]
         ]

    test_data = [
            [1] + rep
            for rep in errors_hiddens[train_idx_ends_positive:]
        ] + [
            [0] + rep
            for rep in correct_hiddens[train_idx_ends_negative:]
        ]
    print('train data size:', len(train_data), 'test data size:', len(test_data))
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)
    print('train features shape:', train_data.shape, 'test features shape:', test_data.shape)
    train_set = SingleLabelDataset(train_data)
    test_set = SingleLabelDataset(test_data)
    train_loader = DataLoader(train_set, batch_size=args.train_err_pred_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.train_err_pred_batch_size, shuffle=False)

    # train error predictor
    criterion = nn.CrossEntropyLoss()
    if not args.best_model_err_pred_save_path:
        args.best_model_err_pred_save_path = f'data/model_err_pred_layer{args.layer}.pkl'

    cur_best_test_acc = 0.0
    if os.path.exists(args.best_model_err_pred_save_path):
        with open(args.best_model_err_pred_save_path, 'rb') as f:
            model_err_pred = pickle.load(f)
        print(f"Loaded existing err pred at {args.best_model_err_pred_save_path}. Skipped training.")
    else:
        print('Training err pred from scratch:')
        model_err_pred = MLP(
            num_layers=args.err_pred_num_layers,
        ).to(device)

        # training loop
        optimizer = torch.optim.AdamW(
            model_err_pred.parameters(),
            lr=0.003,
            weight_decay=0.1,
        )
        epochs = 10

        for epoch in range(epochs):
            losses = []
            for batch_num, input_data in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = input_data
                x = x.to(device).float()
                y = y.type(torch.LongTensor).to(device)

                output = model_err_pred(x)
                loss = criterion(output, y)
                loss.backward()
                losses.append(loss.item())

                optimizer.step()

                if batch_num % 10 == 0:
                    pred = output.argmax(dim=-1)
                    print('\tEpoch %d | Batch %d | Loss %6.2f | train acc %.2f' % (
                    epoch, batch_num, loss.item(), sum(pred == y) / len(y)))
                # break

            pred = output.argmax(dim=-1)
            print(
                'Epoch %d | Loss %6.2f | train acc %.2f' % (epoch, sum(losses) / len(losses), sum(pred == y) / len(y)))
            test_acc = test(model_err_pred, test_loader, criterion, device)
            if test_acc >= cur_best_test_acc:
                with open(args.best_model_err_pred_save_path, 'wb') as f:
                    pickle.dump(model_err_pred, f)
                cur_best_test_acc = test_acc

            model_err_pred.train()

        print('final best_model_err_pred:')
        with open(args.best_model_err_pred_save_path, 'rb') as f:
            cur_best_model_err_pred = pickle.load(f)
        test(cur_best_model_err_pred, test_loader, criterion, device)
