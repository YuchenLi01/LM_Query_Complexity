# Dyck distribution learning and sampling

import copy
import numpy as np
import torch
from tqdm import tqdm
import datasets


LARGE_NEGATIVE_FOR_LOG_PROB = -100.0


def get_vocab_size(num_types, special_tokens):
    return 2 * num_types + len(special_tokens)


class RandomWalkDyck: # Dyck language data generator with likelihoods
    def __init__(self, config={}):
        self.length = config['length']
        self.num_types = config.get('num_types') or 2
        self.max_depth = config.get('max_depth')
        self.np_rng = np.random.default_rng(config.get('seed')) # determinism
        # Allow for unequal opening bracket probabilities, default to equal
        self.type_probs = config.get('type_probs', np.ones(self.num_types)/self.num_types)
        if self.type_probs is None:
            self.type_probs = np.ones(self.num_types) / self.num_types
        special_tokens = ['B', 'E', 'P', 'S']
        self.vocab_size = get_vocab_size(self.num_types, special_tokens)
        self.bos = 2*self.num_types
        self.eos = 2*self.num_types+1
        if self.num_types <= 4:
            self.tokens = list('([{<'[:self.num_types]) + list(')]}>'[:self.num_types]) + special_tokens
        else:
            self.tokens = [str(token) for token in range(self.num_types * 2)] + special_tokens
        self.vocab = dict(zip(self.tokens, range(self.vocab_size)))
        self.pop_prob = 0.5
        
    def is_open_bracket(self, x):
        assert type(x) is int
        return 0 <= x <= self.num_types - 1
    
    def is_close_bracket(self, x):
        assert type(x) is int
        return self.num_types <= x <= 2 * self.num_types - 1
    
    def parse(self, seq):
        assert self.accept_prefix(seq, unbounded_depth=False), f'seq is invalid prefix: {seq}'
        stack = []
        for x in seq:
            if self.is_open_bracket(x):
                stack.append(x)
            elif self.is_close_bracket(x):
                stack.pop()
        return stack

    
    def generate(self, seq=[]):
        seq = copy.deepcopy(seq)
        stack = self.parse(seq)
        prefix_log_probs = []
        log_prob = 0.
        
        num_remaining_pos = self.length
        if len(seq) >= 1:
            assert len(seq) < self.length + 2, f'seq length reached max: {seq}'
            assert seq[-1] != self.eos, f'seq ends with eos: {seq}'
            seq = seq[1:]
            num_remaining_pos -= len(seq)
        for _ in range(num_remaining_pos):
            if len(stack) == 0:
                seq.append(self.np_rng.choice(self.num_types, p = self.type_probs))
                stack.append(seq[-1])
                log_prob += np.log(self.type_probs[seq[-1]])
                  
            else:
                current_pop_prob = self.pop_prob
                
                if self.length - len(seq) == len(stack) or (self.max_depth is not None and len(stack) == self.max_depth):
                    current_pop_prob = 1.
                
                if self.np_rng.random() < current_pop_prob:
                    seq.append(stack[-1] + self.num_types)
                    stack.pop()
                    log_prob += np.log(current_pop_prob)
                else:
                    seq.append(self.np_rng.choice(self.num_types, p = self.type_probs))
                    stack.append(seq[-1])
                    log_prob += np.log(1-current_pop_prob) + np.log(self.type_probs[seq[-1]])

            prefix_log_probs.append(log_prob)

        return {'tokens': [self.bos] + seq + [self.eos],
                'log_prob': log_prob,
                'prefix_log_probs': prefix_log_probs}

    def compute_log_prob(self, seq, use_finite_invalid_prob=False):
        """
        This is for full sequences. For prefixes, use `compute_prefix_log_prob`.
        """
        invalid_prob = -np.inf
        if use_finite_invalid_prob:
            invalid_prob = LARGE_NEGATIVE_FOR_LOG_PROB

        if seq[0] != self.bos or seq[-1] != self.eos:
            return invalid_prob

        log_prob = 0.
        stack = []

        for idx, x in enumerate(seq[1:-1]):
            remaining_length = len(seq) - 2 - idx  # Adjusted to seq's length
            
            if x == self.bos or x == self.eos:
                return invalid_prob

            if len(stack) == 0:
                # Must push when stack is empty
                if x >= self.num_types:
                    return invalid_prob
                log_prob += np.log(self.type_probs[x])
                
            else:
                current_pop_prob = self.pop_prob
                if remaining_length == len(stack) or (self.max_depth is not None and len(stack) == self.max_depth):
                    current_pop_prob = 1.0
                
                if x < self.num_types:
                    # Push action
                    log_prob += np.log(1 - current_pop_prob) + np.log(self.type_probs[x])
                else:
                    # Pop action
                    log_prob += np.log(current_pop_prob)

            if x < self.num_types:
                stack.append(x)
            else:
                # Adjusted to consider different types of brackets
                if not stack or x != stack[-1] + self.num_types:
                    return invalid_prob
                stack.pop()

        return log_prob

    def compute_prefix_log_prob(self, seq, use_finite_invalid_prob=False):
        if seq[-1] == self.eos:
            return self.compute_log_prob(seq, use_finite_invalid_prob=use_finite_invalid_prob)

        invalid_prob = -np.inf
        if use_finite_invalid_prob:
            invalid_prob = LARGE_NEGATIVE_FOR_LOG_PROB

        if seq[0] != self.bos:
            return invalid_prob

        log_prob = 0.
        stack = []

        for idx, x in enumerate(seq[1:]):
            remaining_length = self.length - idx

            if x == self.bos or x == self.eos:
                return invalid_prob

            if len(stack) == 0:
                # Must push when stack is empty
                if x >= self.num_types:
                    return invalid_prob
                log_prob += np.log(self.type_probs[x])

            else:
                current_pop_prob = self.pop_prob
                if remaining_length == len(stack) or (self.max_depth is not None and len(stack) == self.max_depth):
                    current_pop_prob = 1.0

                if x < self.num_types:
                    # Push action
                    log_prob += np.log(1 - current_pop_prob) + np.log(self.type_probs[x])
                else:
                    # Pop action
                    log_prob += np.log(current_pop_prob)

            if x < self.num_types:
                stack.append(x)
            else:
                # Adjusted to consider different types of brackets
                if not stack or x != stack[-1] + self.num_types:
                    return invalid_prob
                stack.pop()

        return log_prob

    def tokenize(self, seq):  # bracket string -> int sequence
        if self.num_types > 4:
            raise ValueError('too lazy to implement')
        # print('seq', seq)  # debug
        seq = seq.replace('<bos>', 'B').replace('<eos>', 'E')
        return [self.vocab[x] for x in seq]

    def detokenize(self, seq): # int sequence -> bracket string
        if self.num_types > 4:
            # raise ValueError('too lazy to implement')
            return str(seq)
        return ''.join(map(lambda x: self.tokens[x], seq))
    
    def make_split(self, num_examples, seed=None, num_proc=None): # create HuggingFace dataset
        ds = datasets.Dataset.from_dict({'id': list(range(num_examples))})
        if seed is None:
            seed = self.np_rng.integers(1000000)
        has_seeded = False # per-process flag to seed RNG once
        def f(x):
            nonlocal has_seeded
            if not has_seeded:
                self.np_rng = np.random.default_rng(x['id'] + seed)
                has_seeded = True
            return self.generate(seq=[])
        ds = ds.map(f, num_proc=num_proc, remove_columns=['id'])
        return ds
    
    def accept(self, seq):
        stack = []
        has_seen_bos = False
        for i, x in enumerate(seq):
            if x == self.bos:
                if not has_seen_bos:
                    has_seen_bos = True
                    continue
                else:
                    return False
            elif x == self.eos:
                return len(stack) == 0 and i == self.length + 1
            elif x < self.num_types:
                stack.append(x)
            else:
                if len(stack) == 0:
                    return False
                if stack.pop() + self.num_types != x:
                    return False
        return len(stack) == 0

    def accept_prefix(self, seq, unbounded_depth=False):
        return self.return_1st_err_pos(seq, unbounded_depth=unbounded_depth) == -1

    def return_1st_err_pos(self, seq, unbounded_depth=False):
        stack = []
        if len(seq) >= 1 and seq[0] != self.bos:
            return 0
        has_seen_bos = False
        for i, x in enumerate(seq):
            if x == self.bos:
                if not has_seen_bos:
                    has_seen_bos = True
                    continue
                else:
                    return i
            elif x == self.eos:
                if len(stack) == 0 and i == self.length + 1:
                    return -1
                return i
            elif x < self.num_types:
                if not unbounded_depth and self.length - i < len(stack):
                    return i
                stack.append(x)
            else:
                if len(stack) == 0:
                    return i
                if stack.pop() + self.num_types != x:
                    return i
        return -1

    def get_max_depth(self, seq):
        """
        For invalid sequences, proceed as if ( and ] could cancel out; allow negative depths in ( ) )
        """
        cur_depth = 0
        max_depth = 0
        for x in seq:
            if x == self.bos:
                continue
            elif x == self.eos:
                return max_depth
            elif x < self.num_types:
                cur_depth += 1
                max_depth = max(max_depth, cur_depth)
            else:
                cur_depth -= 1
        return max_depth

    def get_final_depth(self, seq):
        """
        For invalid sequences, proceed as if ( and ] could cancel out; allow negative depths in ( ) )
        """
        cur_depth = 0
        for x in seq:
            if x == self.bos:
                continue
            elif x == self.eos:
                return cur_depth
            elif x < self.num_types:
                cur_depth += 1
            else:
                cur_depth -= 1
        return cur_depth
    
    def eval_acc(self, model, eval_loader):
        # evaluate Dyck accuracy
        model.eval()
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for eval_batch in eval_loader:
                tokens = eval_batch['tokens'].cuda()
                preds = model(tokens[:,:-1]).logits
                labels = tokens[:,1:].contiguous()
                # only score close bracket tokens (between num_types and 2*num_types-1)
                close_pos = (labels >= self.num_types) & (labels < 2*self.num_types)
                num_correct += (torch.argmax(preds, dim=-1)[close_pos] == labels[close_pos]).sum().item()
                num_total += close_pos.sum().item()
        return num_correct / num_total
