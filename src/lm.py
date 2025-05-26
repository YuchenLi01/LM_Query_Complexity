# Language model learning and sampling

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import datasets
import random

from dyck import get_vocab_size


def test_err_pred(model_err_pred, test_loader, criterion, device):
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


def sample_top_p(
        preds,
        top_p,
        step,
        excluded_tokens=None,
        temperature=1.0,
):
    if excluded_tokens is not None:
        preds[:, step - 1, excluded_tokens] = float('-inf')

    assert temperature >= 0
    if temperature == 0.0:
        # will treat as argmax; probs calculated pretending temperature = 1.0
        top_p = 0.0
    else:
        preds = preds / temperature

    if top_p is None or top_p == 0.0:
        # argmax sampling
        next_tokens = torch.argmax(preds[:, step - 1, :], dim=-1)
        log_probs = torch.gather(torch.log_softmax(preds[:, step - 1, :], dim=-1), 1, next_tokens.unsqueeze(-1)).squeeze(-1)
    elif top_p == 1.0:
        # no truncation
        next_tokens = torch.multinomial(torch.softmax(preds[:, step - 1, :], dim=-1), num_samples=1).squeeze(-1)
        log_probs = torch.gather(torch.log_softmax(preds[:, step - 1, :], dim=-1), 1, next_tokens.unsqueeze(-1)).squeeze(-1)
    else:
        sorted_logits, sorted_indices = torch.sort(preds[:, step - 1, :], descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set removed token logits to negative infinity
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        unsorted_logits = torch.zeros_like(sorted_logits).scatter_(-1, sorted_indices, sorted_logits)

        # Sample from the logits
        next_tokens = torch.multinomial(torch.softmax(unsorted_logits, dim=-1), num_samples=1).squeeze(-1)
        log_probs = torch.gather(torch.log_softmax(unsorted_logits, dim=-1), 1, next_tokens.unsqueeze(-1)).squeeze(-1)
    
    return next_tokens, log_probs


# batch generation from trained NN
def nn_generate(
        dyck,
        model,
        batch_size=32,
        num_sequences=1000,
        top_p=None,
        temperature=1.0,
        use_template_implementation=True,
        verbose=0,
):
    model.eval()
    output = []

    seq_length = dyck.length + 2  # <bos>='B' + length + <eos>='E'

    # Calculate the number of chunks we'll need
    num_chunks = (num_sequences + batch_size - 1) // batch_size
    print('nn_generate num_chunks:', num_chunks)
    report_progress_interval = num_chunks // 10

    with torch.no_grad():
        if verbose >= 1:
            loop = tqdm(range(num_chunks), desc="Generating sequences")
        else:
            loop = range(num_chunks)
        for loop_idx in loop:
            current_batch_size = min(batch_size, num_sequences - len(output))
            batch_seqs = [[dyck.bos] for _ in range(current_batch_size)]
            batch_log_probs = [0.] * current_batch_size

            if use_template_implementation:
                batch_seqs = model.generate(
                    torch.tensor(
                        [[dyck.bos]],
                        device='cuda',
                    ),
                    max_length=seq_length,
                    max_new_tokens=None,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=current_batch_size,
                ).detach().tolist()
                batch_log_probs = None
            else:
                for step in range(1, seq_length):
                    preds = model(torch.tensor(batch_seqs).cuda()).logits

                    # Sample
                    next_tokens, log_probs = sample_top_p(
                        preds,
                        top_p,
                        step,
                        temperature=temperature,
                    )

                    # Update sequences and log probs
                    for i in range(current_batch_size):
                        batch_seqs[i].append(next_tokens[i].item())
                        batch_log_probs[i] += log_probs[i].item()

            # Add to the final output
            for i in range(current_batch_size):
                single_seq_output = {
                    'tokens': batch_seqs[i],
                }
                if batch_log_probs is not None:
                    single_seq_output['log_prob_truncated'] = batch_log_probs[i]
                output.append(single_seq_output)

            if (loop_idx + 1) % report_progress_interval == 0:
                print(f"generated {loop_idx + 1} batches out of {num_chunks}. recent batch size {current_batch_size}")

    return output


# for each prefix, returns 0 if correct, 1 if wrong
def predict_error(
        dyck,
        model,
        batch_seqs,
        model_err_pred,
        use_groundtruth_error_predictor=False,
        unbounded_depth=False,
        return_probs=False,
        threshold=0.5,
):
    assert 0 <= threshold <= 1, f"expect threshold to be in [0, 1]; received {threshold}"
    batch_size = len(batch_seqs)
    if use_groundtruth_error_predictor:
        if dyck is not None:
            for batch_seq in batch_seqs:
                if not dyck.accept_prefix(batch_seq, unbounded_depth=unbounded_depth):
                    return 1
        return 0
    else:
        if model_err_pred == 'random':
            err_pred_probs = [[None, random.random()]]
        else:
            all_intermediates = nn_intermediates(model, batch_seqs, batch_size=batch_size, positions='last')
            err_pred_output = model_err_pred(torch.tensor(all_intermediates).to('cuda:0'))
            err_pred_probs = torch.softmax(err_pred_output, dim=-1)

        err_criterion = err_pred_probs[0][1] >= threshold
        if err_criterion:
            print(
                'err_pred_probs',
                err_pred_probs,
            )

        if return_probs:
            return err_pred_probs
        
        return err_criterion


# conditional generation from trained NN
def conditional_nn_generate(
        dyck,
        model,
        prefix_sequences,
        batch_size=32,
        top_p=None,
        temperature=1.0,
        prefix_length=None,  # exclude bos, assume all prefix_sequences in batch have same len
        argmax_length=0,
        backtrack_quota=0,
        backtrack_stride=None,
        max_allowed_prefix_truncation=0,
        backtracked_prefixes=None,
        model_err_pred=None,
        allow_2nd_backtrack_at_same_prefix=True,  # TODO: try to avoid running into same prefix 2nd time
        use_groundtruth_error_predictor=False,
        tokenwise_dfs=False,
        block_err_pred=False,
        max_new_tokens=256,
        err_pred_threshold=0.5,
        tokenizer=None,
        redo_backtrack_with_argmax=True,
        verbose=0,
):
    model.eval()
    output = []
    predicted_mistake_prefixes = set()

    num_sequences = len(prefix_sequences)
    prefix_length = prefix_length or len(prefix_sequences[0]) - 1  # TODO: assumes all prefixes have same len

    if dyck is not None:
        seq_length = dyck.length + 2  # <bos>='B' + length + <eos>='E'
    else:
        seq_length = max_new_tokens + prefix_length

    # Calculate the number of chunks we'll need
    num_chunks = (num_sequences + batch_size - 1) // batch_size

    # prepare special symbols
    new_line_token_ids = []
    if tokenizer is not None:
        for token in tokenizer.vocab:
            if '\n' in token or '<0x0A>' in token or '<0x0D>' in token:
                new_line_token_ids.append(tokenizer.vocab[token])

    i = 0

    with torch.no_grad():
        if verbose >= 1:
            loop = tqdm(range(num_chunks), desc="Generating sequences")
        else:
            loop = range(num_chunks)
        for _ in loop:
            current_batch_size = min(batch_size, num_sequences - len(output))
            batch_seqs = []

            for _ in range(current_batch_size):
                batch_seqs.append(prefix_sequences[i][:prefix_length + 1])
                i += 1

            batch_tensor = torch.tensor(batch_seqs, dtype=torch.int64).cuda()
            preds = model(batch_tensor).logits
            # Calculate log probabilities for the prefix
            batch_log_probs = torch.log_softmax(preds, dim=-1).gather(2, batch_tensor[:, 1:].unsqueeze(-1)).sum(dim=1).reshape(-1)
            batch_log_probs = batch_log_probs.tolist()
            num_excluded_tokens = 0
            excluded_all_tokens = 0

            del batch_tensor

            gen_idx = 0
            step = 1 + prefix_length
            total_backtrack_strides = 0
            while step < seq_length:
                if backtrack_quota > 0:
                    assert batch_size == 1  # TODO: support batch_size > 1, but need padding since diff rows in a tensor must have same len
                    assert backtrack_stride == 'NA' or (type(backtrack_stride) is int and backtrack_stride >= 1)
                    assert model_err_pred is not None
                    max_allowed_backtrack_stride = min(
                        gen_idx + max_allowed_prefix_truncation,  # cannot truncate prefix more than allowed
                        step - 1,  # must keep bos token
                    )

                    allow_backtrack = True
                    if max_allowed_backtrack_stride <= 0:
                        allow_backtrack = False
                    if backtracked_prefixes is not None:
                        assert type(backtracked_prefixes) is list
                        if batch_seqs in backtracked_prefixes:  # TODO: faster membership lookup
                            if not allow_2nd_backtrack_at_same_prefix:
                                allow_backtrack = False
                    if block_err_pred:
                        assert len(new_line_token_ids) > 0, \
                            f"new_line_token_ids = {new_line_token_ids}. Did not specify where to break into blocks."
                        if len(batch_seqs[0]) == 0 or batch_seqs[0][-1] not in new_line_token_ids:
                            allow_backtrack = False

                    if allow_backtrack:
                        if predict_error(
                            dyck,
                            model,
                            batch_seqs,
                            model_err_pred,
                            use_groundtruth_error_predictor=use_groundtruth_error_predictor,
                            threshold=err_pred_threshold,
                        ):
                            # backtrack
                            if backtracked_prefixes is not None:
                                backtracked_prefixes.append(batch_seqs[0])
                            for batch_seq in batch_seqs:
                                predicted_mistake_prefixes.add(tuple(batch_seq))
                            if block_err_pred and backtrack_stride == 'NA':
                                this_backtrack_stride = 1
                                while len(batch_seqs[0]) - 1 - this_backtrack_stride >= 0 \
                                        and batch_seqs[0][len(batch_seqs[0])-1-this_backtrack_stride] not in new_line_token_ids:
                                    this_backtrack_stride += 1
                            else:
                                this_backtrack_stride = min(backtrack_stride, max_allowed_backtrack_stride)
                            batch_seqs = [
                                batch_seq[:-this_backtrack_stride]
                                for batch_seq in batch_seqs
                            ]
                            gen_idx -= this_backtrack_stride
                            step -= this_backtrack_stride
                            if redo_backtrack_with_argmax:
                                argmax_length += this_backtrack_stride
                            backtrack_quota -= 1
                            if verbose >= 1:
                                print(f'backtracked {this_backtrack_stride} steps')
                            total_backtrack_strides += this_backtrack_stride

                batch_tensor = torch.tensor(batch_seqs, dtype=torch.int64).cuda()
                preds = model(batch_tensor).logits

                p = 0.0 if gen_idx < argmax_length else top_p
                next_tokens, log_probs = sample_top_p(
                    preds,
                    p,
                    step,
                    temperature=temperature,
                )
                if tokenwise_dfs:
                    raise NotImplementedError('did not work, deprecated')
                    # print('batch_seqs[0]', batch_seqs[0])  # debug
                    # print('next_tokens[0].item()', next_tokens[0].item())  # debug
                    assert batch_size == 1
                    excluded_tokens = []
                    while predict_error(
                        dyck,
                        model,
                        [batch_seqs[0] + [next_tokens[0].item()]],
                        model_err_pred,
                        use_groundtruth_error_predictor=use_groundtruth_error_predictor,
                    ) or tuple(batch_seqs[0] + [next_tokens[0].item()]) in predicted_mistake_prefixes:
                        predicted_mistake_prefixes.add(tuple(batch_seqs[0] + [next_tokens[0].item()]))
                        if len(excluded_tokens) < dyck.vocab_size:
                            excluded_tokens.append(next_tokens[0].item())
                            num_excluded_tokens += 1
                            print(f'batch_seqs = {dyck.detokenize(batch_seqs[0])}, excluded_tokens = {dyck.detokenize(excluded_tokens)}')  # debug
                            next_tokens, log_probs = sample_top_p(preds, p, step, excluded_tokens=excluded_tokens)
                            # else:
                            #     print(f'batch_seqs = {dyck.detokenize(batch_seqs[0])}, passed = {dyck.detokenize([next_tokens[0].item()])}')  # debug
                        else:
                            excluded_all_tokens += 1
                            next_tokens, log_probs = sample_top_p(preds, p, step, excluded_tokens=None)  # TODO: backtracking more positions 
                            break

                # Update sequences and log probs
                for j in range(current_batch_size):
                    batch_seqs[j].append(next_tokens[j].item())
                    batch_log_probs[j] += log_probs[j].item()

                del preds
                del batch_tensor

                gen_idx += 1
                step += 1

                if verbose >= 1 and tokenizer is not None:
                    print('next_tokens', next_tokens, tokenizer.convert_ids_to_tokens(next_tokens))
                if tokenizer is not None and torch.all(next_tokens == tokenizer.eos_token_id):
                    break

            # Add to the final output
            for j in range(current_batch_size):
                output.append({
                    'tokens': batch_seqs[j],
                    'log_prob_truncated': batch_log_probs[j],
                    'remaining_backtrack_quota': backtrack_quota,
                    'num_excluded_tokens': num_excluded_tokens,
                    'excluded_all_tokens': excluded_all_tokens,
                    'total_backtrack_strides': total_backtrack_strides,
                })

        return output  # `backtracked_prefixes` may be updated


def nn_log_probs(model, sequences, batch_size=32):
    model.eval()
    
    # Initialize an empty list to store the log probabilities for each batch
    all_log_probs = []

    with torch.no_grad():
        # Iterate over sequences in chunks (batches)
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]

            # Convert the batch sequences to a tensor and send to the same device as the model
            batch_tensor = torch.tensor(batch_seqs, dtype=torch.int64).cuda()

            # Predict using the model
            preds = model(batch_tensor).logits

            # Calculate log probabilities
            log_probs = torch.log_softmax(preds, dim=-1).gather(2, batch_tensor[:, 1:].unsqueeze(-1)).sum(dim=1)

            all_log_probs.extend(log_probs.tolist())
    
    return all_log_probs


def nn_next_token_probs(model, sequences, batch_size=32):
    model.eval()

    with torch.no_grad():
        # Initialize an empty list to store the probabilities for each batch
        all_preds = []

        # Iterate over sequences in chunks (batches)
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # Convert the batch sequences to a tensor and send to the same device as the model
            batch_tensor = torch.tensor(batch_seqs, dtype=torch.int64).cuda()

            # Predict using the model
            preds = model(batch_tensor).logits
            all_preds.extend(preds.tolist())

        # Calculate probabilities
        probs = torch.distributions.Categorical(
            logits=torch.tensor(all_preds),
        )

    return probs


def nn_intermediates(
        model,
        sequences,
        batch_size=32,
        positions='last',
        layer=-1,
):
    """

    Args:
        model:
        sequences: each sequence must have the same length
        batch_size:
        positions: 'last' or 'all'
        layer:

    Returns:

    """
    model.eval()

    # Initialize an empty list to store the intermediates for each batch
    all_intermediates = []

    with torch.no_grad():
        # Iterate over sequences in chunks (batches)
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]

            # Convert the batch sequences to a tensor and send to the same device as the model
            if batch_size > 1:
                batch_tensor = torch.tensor(batch_seqs, dtype=torch.int64).cuda()
            else:
                assert batch_size == 1
                batch_tensor = torch.unsqueeze(torch.tensor(batch_seqs[0]), dim=0).cuda()

            # Predict using the model
            output = model(batch_tensor, output_hidden_states=True)
            intermediate = output.hidden_states

            if positions == 'last':
                rep = intermediate[layer][:, -1]
            elif positions == 'all':
                rep = intermediate[layer]
            else:
                raise NotImplementedError('positions')

            all_intermediates.extend(rep.tolist())

    return all_intermediates


def get_transformer(cfg):
    from transformers import LlamaForCausalLM, LlamaConfig
    print("Initializing Llama with config:")
    print(cfg)
    try:
        config = LlamaConfig(
            vocab_size=cfg['vocab_size'],
            hidden_size=cfg['dim'],
            intermediate_size=cfg['dim'] * 2,
            num_hidden_layers=cfg['depth'],
            num_attention_heads=cfg['heads'],
            max_position_embeddings =cfg.get('max_seq_len') or 1024,
            pad_token_id=None,
            bos_token_id=cfg['bos_token_id'],
            eos_token_id=cfg['eos_token_id'],
        )
    except:
        config = LlamaConfig(
            vocab_size=get_vocab_size(cfg.language.bracket_types),
            hidden_size=cfg.lm.dim,
            intermediate_size=cfg.lm.dim * 2,
            num_hidden_layers=cfg.lm.num_layers,
            num_attention_heads=cfg.lm.num_heads,
            max_position_embeddings =cfg.get('max_seq_len') or 1024,
            pad_token_id=None,
            bos_token_id=cfg.language.bos_token_id,
            eos_token_id=cfg.language.eos_token_id,
        )
    model = LlamaForCausalLM(config).cuda()
    return model
