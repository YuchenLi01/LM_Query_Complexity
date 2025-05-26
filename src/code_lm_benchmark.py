import ast
from collections import namedtuple
import copy
from enum import Enum
import inspect
import itertools
import pickle
import random
import re
import string


# TestCase = namedtuple('TestCase', ['argument_l', 'argument_item', 'results'])

# Enum for error types in the generated code
ErrorType = Enum('ErrorType', [
    'LINE_WITHOUT_ASSERT', # A non-empty line did not start with assert
    'EXTRA_CONTENT_AFTER_TASK_COMPLETION', # Model generated more content after the task was completed
    'FUNCTION_NOT_TESTED', # The function that was supposed to be tested was not tested (asserted)
    'FUNCTION_CALL_NO_CLOSING_PARENTHESIS', # Unable to parse the function call due to missing closing parenthesis
    'FUNCTION_CALL_SYNTAX_ERROR', # Syntax error in the function call
    'FUNCTION_CALL_VALUE_ERROR', # Value error in the function call
    'FUNCTION_CALL_UNKNOWN_VARIABLE', # The function call has an undefined variable
    'FIRST_ARGUMENT_TYPE_ERROR', # The first argument is not a list
    'ONLY_ONE_ARGUMENT', # Only one argument was passed to the function
    'MORE_THAN_TWO_ARGUMENTS', # More than two arguments were passed to the function
    'NO_ASSERTION_EQUALITY', # No == found in the assertion
    'EXPECTED_OUTPUT_NO_CLOSING_BRACKET', # Unable to parse the expected output due to missing closing bracket
    'EXPECTED_OUTPUT_SYNTAX_ERROR', # Syntax error in the expected output
    'EXPECTED_OUTPUT_VALUE_ERROR', # Value error in the expected output
    'EXPECTED_OUTPUT_TYPE_ERROR', # The expected output is not a list
    'EXPECTED_OUTPUT_UNDEFINED_VARIABLE', # The expected output has an undefined variable
    'ASSERTION_FAILURE', # The assertion failed, i.e. the expected output != the actual output
    'FEWER_TEST_CASES_THAN_EXPECTED', # Fewer test cases than the model was instructed to generate
])



class GenerateTestCaseTask:
    def __init__(
        self,
        num_demonstrations,
        num_test_cases_per_function,
        function_name_length: int = 3,
        function_name = None,
    ):
        self.num_test_cases_per_function = num_test_cases_per_function
        self.function_name_length = function_name_length

        if function_name is None:
            self.function_name = ''.join(random.choice(string.ascii_lowercase) for i in range(function_name_length))
        else:
            assert len(function_name) == function_name_length
            self.function_name = function_name

        if num_demonstrations > 0:
            demo_f = f"def f(a, b):\n{' '*4}return a + b"
            demo_soln_str = ""
            for _ in range(num_test_cases_per_function):
                i = random.randint(1, 9)
                j = random.randint(1, 9)
                demo_soln_str += f"\nassert f({i}, {j}) == {i+j}"

            if num_demonstrations > 1:
                raise NotImplementedError

        self.prompt = self.write_demonstration(
            demo_f=self.write_function_to_test(),
            demo_soln_str='',
        )
        for _ in range(num_demonstrations):
            demo_str = self.write_demonstration(
                demo_f=demo_f,
                demo_f_name='f',
                demo_soln_str=demo_soln_str,
              )
            self.prompt = demo_str + '\n\n' + self.prompt

    def function_to_test(self, l, item):
        assert type(l) is list
        l.append(item)
        return l

    def write_function_to_test(self):
        content_str = inspect.getsource(self.function_to_test)
        content_str = content_str.replace('function_to_test(self, l, item)', f"{self.function_name}(l, item)")
        content_str = '\n'.join([
            line[4:] for line in content_str.split('\n')
        ])
        return content_str

    def write_demonstration(self, demo_f='', demo_f_name='', demo_soln_str=''):
        if demo_f_name == '':
            demo_f_name = self.function_name
        return f'{demo_f}\n\nList {self.num_test_cases_per_function} test cases of the above function {demo_f_name}, one in each line:{demo_soln_str}'

    def print_test_case(self, include_prompts=True, include_solns=True):
        print('\n' * 2)

        if include_prompts:
            print('prompt:')
            print('-' * 20)
            print(self.prompt)
            print('-' * 20)
            print('\n' * 2)

    def parse(self, generated, test_cases=[], verbose=1):
        extra_content_lines = 0
        for line in generated.split('\n'):
            line = line.replace(';', '').lstrip()
            if len(line) < 6 or line[:6] != 'assert' or self.function_name not in line or line[:10] != f'assert {self.function_name}':
                if line:
                    if len(test_cases) > 0:
                        extra_content_lines += 1
                        if extra_content_lines >= 3:
                            if verbose >= 2:
                                print('igoring remaining content')
                            break
                    if verbose >= 1:
                        print('cannot parse assert or function name:', line)
                continue

            line = line.replace(f'assert {self.function_name}', '')
            line_sp = line.split('==')
            if len(line_sp) < 2:
                if verbose >= 1:
                    print('cannot separate arguments and results', line)
                break

            arguments = line_sp[0].strip()
            results = line_sp[1].strip()
            if verbose >= 2:
                print(f"arguments: `{arguments}`, results: `{results}`")

            try:
                results = ast.literal_eval(results)
            except:
                if verbose >= 1:
                    print('cannot parse results:', results)
                continue

            if arguments[0] != '(' or arguments[-1] != ')' or arguments[1] != '[':
                if verbose >= 1:
                    print('cannot parse arguments:', arguments)
                continue

            last_close_square_braket = arguments.rfind(']')
            # print('last_close_square_braket:', last_close_square_braket, arguments[1:last_close_square_braket+1])

            try:
                argument_l = ast.literal_eval(arguments[1:last_close_square_braket+1])
                # print('argument_l:', argument_l, type(argument_l))
            except:
                if verbose >= 1:
                    print('cannot parse argument_l:', arguments[1:last_close_square_braket+1])
                continue

            if type(argument_l) is not list:
                if verbose >= 1:
                    print('argument_l is not list:', argument_l, 'parsed from text:', arguments[1:last_close_square_braket+1])
                continue

            last_close_round_braket = arguments.rfind(')')
            arguments_remaining = arguments[last_close_square_braket+2:last_close_round_braket]

            try:
                argument_item = eval(arguments_remaining)
            except:
                if verbose >= 1:
                    print('cannot parse argument_item:', arguments_remaining)
                continue
            # print('argument_idx:', argument_idx, type(argument_idx))

            test_case = {
                'argument_l': argument_l,
                'argument_item': argument_item,
                'results': results,
            }
            if test_case not in test_cases:
                test_cases.append(test_case)

        return test_cases

    @staticmethod
    def equal(test_case1, test_case2):
        if test_case1['argument_l'] != test_case2['argument_l']:
            return False
        if test_case1['argument_item'] != test_case2['argument_item']:
            return False
        if test_case1['results'] != test_case2['results']:
            return False
        return True

    @staticmethod
    def all_different(test_case, test_cases_list, equal):
        for test_case2 in test_cases_list:
            if equal(test_case, test_case2):
                return False
        return True

    def _get_first_error_index(self, generated, tokenizer):

        def remove_leading_whitespace_and_update_position(line, char_idx):
            count = len(line) - len(line.lstrip())
            line = line.lstrip()
            char_idx += count
            return (line, char_idx)

        def find_unknown_variable_in_list(line, char_idx):
            line = line[1:] # Skip "["
            char_idx += 1
            bracket_depth = 0
            curr_element_start_idx = 0
            for i, ch in enumerate(line):
                if ch == '[':
                    bracket_depth += 1
                elif ch == ']':
                    bracket_depth -= 1
                elif ch == ',' and bracket_depth == 0:
                    try:
                        eval(line[curr_element_start_idx: i])
                    except NameError:
                        # Doesn't handle nested lists
                        line = line[curr_element_start_idx: ]
                        char_idx += curr_element_start_idx
                        line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)
                        return (line, char_idx)
                    curr_element_start_idx = i + 1
            return None, None

        # Remove eos token if exists
        generated = generated.removesuffix(tokenizer.eos_token)

        num_test_cases = 0

        for line_idx, line_content in enumerate(generated.split('\n')):

            char_idx = 0
            line = line_content

            # Skip empty lines
            if not line:
                continue

            if num_test_cases == self.num_test_cases_per_function: # Task completed
                # Error is at the first position of the current line
                return (line_idx, line_content, char_idx, ErrorType.EXTRA_CONTENT_AFTER_TASK_COMPLETION)
            
            elif len(line) < 6 or line[:6] != 'assert':
                # Error is at the first position of the current line
                return (line_idx, line_content, char_idx, ErrorType.LINE_WITHOUT_ASSERT)

            # Move forward from the assert
            line = line[6:]
            char_idx += 6

            # Remove whitespaces between assert and the function name
            line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)

            # Check if the assert is for the function we are testing
            if not line.startswith(self.function_name):
                return (line_idx, line_content, char_idx, ErrorType.FUNCTION_NOT_TESTED)

            # Move forward from the function name
            line = line[len(self.function_name):]
            char_idx += len(self.function_name)

            # To find at what character do the arguments end
            eq_idx = line.find("==")

            args_end = eq_idx - 1 if eq_idx != -1 else len(line) - 1

            while args_end != -1 and line[args_end] != ')':
                args_end -= 1
            
            # If the closing parenthesis is not found
            if args_end == -1:
                char_idx += len(line)
                return (line_idx, line_content, char_idx, ErrorType.FUNCTION_CALL_NO_CLOSING_PARENTHESIS)

            # Might not be the best way of finding the position of the syntax error
            try:
                args_tuple = eval(line[:args_end + 1])
                assert type(args_tuple) is tuple
            except SyntaxError as e:
                char_idx += e.offset
                return (line_idx, line_content, char_idx, ErrorType.FUNCTION_CALL_SYNTAX_ERROR)
            except ValueError as e:
                return (line_idx, line_content, char_idx, ErrorType.FUNCTION_CALL_VALUE_ERROR)
            except NameError as e: # Raised when a variable is not defined
                # Parse tuple manually
                line = line[1:] # Skip "("
                char_idx += 1
                # Find the end of the first argument
                bracket_depth = 0
                for i, ch in enumerate(line):
                    if ch == '[':
                        bracket_depth += 1
                    elif ch == ']':
                        bracket_depth -= 1
                    elif ch == ',' and bracket_depth == 0:
                        break

                try: 
                    eval(line[: i])
                except NameError:
                    # Assumes that the first argument is a list
                    line, char_idx = find_unknown_variable_in_list(line, char_idx)
                    return (line_idx, line_content, char_idx, ErrorType.FUNCTION_CALL_UNKNOWN_VARIABLE)
                
                line = line[i + 1: ]
                char_idx += i + 1
                # Assumes that the variable name is unique, i.e. only instances of that string in (the rest of) the line would be as the variable name
                # Also assumes that there are only two arguments (otherwise the , would be the error token if it occurs before the unknown variable) 
                unknown_var_start = line.find(e.name)
                char_idx += unknown_var_start
                return (line_idx, line_content, char_idx, ErrorType.FUNCTION_CALL_UNKNOWN_VARIABLE)



            # Specific to the list append task
            if type(args_tuple[0]) is not list:
                # Move forward from the opening parenthesis
                char_idx += 1
                return (line_idx, line_content, char_idx, ErrorType.FIRST_ARGUMENT_TYPE_ERROR)

            # Only one argument
            if len(args_tuple) == 1:
                # The closing parenthesis is the error position in this case
                char_idx += args_end
                return (line_idx, line_content, char_idx, ErrorType.ONLY_ONE_ARGUMENT)

            # More than 2 arguments
            if len(args_tuple) >= 3:
                # the first character after the 2nd argument is the error position
                # i.e. , instead of )
                char_idx += len(str(args_tuple[:2]))
                return (line_idx, line_content, char_idx, ErrorType.MORE_THAN_TWO_ARGUMENTS)


            line = line[args_end + 1: ]
            char_idx += args_end + 1
            line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)

            # If the line ends here, then the error is that the assertion is incomplete (no == followed by the expected output)
            if eq_idx == -1:
                return (line_idx, line_content, char_idx, ErrorType.NO_ASSERTION_EQUALITY)


            # Check for the actual assertion equality
            expected_out = self.function_to_test(args_tuple[0], args_tuple[1])

            line = line[2:] # == is two chars
            char_idx += 2
            line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)

            try:
                asserted_out = eval(line)
            except SyntaxError as e:
                if "was never closed" in e.msg: # List was never closed
                    char_idx += len(line)
                    return (line_idx, line_content, char_idx, ErrorType.EXPECTED_OUTPUT_NO_CLOSING_BRACKET)
                print(e.msg)
                char_idx += e.offset
                return (line_idx, line_content, char_idx, ErrorType.EXPECTED_OUTPUT_SYNTAX_ERROR)
            except ValueError as e:
                return (line_idx, line_content, char_idx, ErrorType.EXPECTED_OUTPUT_VALUE_ERROR)
            except NameError: # Raised when a variable is not defined
                # The error is at the first character of the variable name
                # We assume that we are only dealing with lists
                line, char_idx = find_unknown_variable_in_list(line, char_idx)
                return (line_idx, line_content, char_idx, ErrorType.EXPECTED_OUTPUT_UNDEFINED_VARIABLE)


            if type(asserted_out) is not list:
                return (line_idx, line_content, char_idx, ErrorType.EXPECTED_OUTPUT_TYPE_ERROR)

            if expected_out == asserted_out:
                num_test_cases += 1
                continue

            i = 0
            while i < len(asserted_out) and i < len(expected_out):

                # If the current element differs from expected
                if asserted_out[i] != expected_out[i]:

                    # Edge case - i = 0 => first element is the error
                    # len(asserted_out[:0]) will be 2 not 0 so we need to handle this case separately
                    if i == 0:
                        char_idx += 1
                        line = line[1:]
                        line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)
                        return (line_idx, line_content, char_idx, ErrorType.ASSERTION_FAILURE)

                    # Adding length of elements so far
                    curr_element_idx = 0 # Track element idx we are at
                    bracket_depth = 0 # To handle nested lists
                    for ind, ch in enumerate(line):
                        if ch == "[":
                            bracket_depth += 1
                        elif ch == "]":
                            bracket_depth -= 1
                        elif ch == ',' and bracket_depth == 1:
                            curr_element_idx += 1
                            if curr_element_idx == i:
                                break


                    # Start index of the first incorrect element
                    line = line[ind+1:]
                    char_idx += ind+1
                    line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)


                    # if the expected element is a string, we need to add quotes when comparing elements character by character
                    if type(expected_out[i]) is str:
                        expected_element = f'"{expected_out[i]}"'
                    else:
                        expected_element = str(expected_out[i])


                    j = 0
                    while j < len(expected_element) and j < len(line):

                        # Assumes that the incorrect element is not a list
                        if expected_element[j] != line[j]:

                            # ' and " are interchangeable
                            if expected_element[j] == '"' and line[j] == "'":
                                j += 1
                                continue

                            char_idx += j
                            return (line_idx, line_content, char_idx, ErrorType.ASSERTION_FAILURE)

                        j += 1

                i += 1

            # The fact the we've reached here means that the two lists have
            # different lengths

            # Edge case - i = 0 => empty list
            # This can only occur if asserted_out is []
            # expected_out cannot be []
            # So here ] is the error token
            if i == 0:
                char_idx += 1
                line = line[1: ]
                line, char_idx = remove_leading_whitespace_and_update_position(line, char_idx)
                return (line_idx, line_content, char_idx, ErrorType.ASSERTION_FAILURE)

            # If not the edge case then that means
            # the error is that asserted out has ] instead of , or vice-versa
            # in either case, the error location is the same
            char_idx += len(str(asserted_out[:i])) - 1
            return (line_idx, line_content, char_idx, ErrorType.ASSERTION_FAILURE)

        if num_test_cases < self.num_test_cases_per_function:
            line_idx = len(generated.split('\n')) - 1
            line_content = generated.split('\n')[-1]
            return (line_idx, line_content, len(line_content), ErrorType.FEWER_TEST_CASES_THAN_EXPECTED)

        return None


    def get_first_error_token(self, token_ids, tokenizer, verbose=1):
        """
        Given the token_ids of the generated code, this function returns the index of the first error token

        Ideally, the eos_token is included in the token_ids, but if it is not, the function will still work.
        Only that we won't be able to distinguish between premature eos generation and reaching the max_tokens limit.

        Returns: index of the first error token
        If no error is found, returns None
        If the "error" is that max_tokens limit was reached, returns len(token_ids)

        Returns: error type
        """
        generated = tokenizer.decode(token_ids)

        error_info = self._get_first_error_index(generated, tokenizer)

        if error_info is None:
            if verbose >= 1:
                print("No Errors")
            return None, None

        line_idx, line, char_idx, error_type = error_info

        if verbose >= 1:
            print("Error Line:")
            print(line)

        i = 0
        j = 0

        # get to the correct line
        while i < line_idx:
            if tokenizer.decode(token_ids[j]) == '\n':
                 i += 1
            j += 1

        k = j + 1
        while k < len(token_ids) + 1:
            # Find the error token by checking if adding the current token the
            # line output so far makes the length exceed the error index
            if len(tokenizer.decode(token_ids[j: k])) > char_idx:
                break
            k += 1

        if verbose >= 1:
            # Error is that we stopped generating
            if k == len(token_ids) + 1:
                print("Error:", "End of generated code, increase max_tokens?")

            # The error token
            else:
                print("Error Token:", repr(tokenizer.decode(token_ids[k - 1])))

            print("Error type:", error_type)

        # return index of error token and the error type
        return (k - 1, error_type)

    @staticmethod
    def count_distinct(iterable):
        distinct = []
        for item in iterable:
            if GenerateTestCaseTask.all_different(
                item,
                distinct,
                lambda x, y: x == y,
            ):
                distinct.append(item)
        return len(distinct)

    def check(self, test_cases, verbose=1):
        num_correct = 0
        num_different_correct = 0
        different_correct_test_cases = []
        list_lengths = set()
        list_elements = []
        indices = []
        for test_case in test_cases:
            expected = self.function_to_test(
                copy.deepcopy(test_case['argument_l']),
                copy.deepcopy(test_case['argument_item']),
            )
            if expected == test_case['results']:
                num_correct += 1
                if GenerateTestCaseTask.all_different(
                        test_case,
                        different_correct_test_cases,
                        GenerateTestCaseTask.equal,
                ):
                    num_different_correct += 1
                    different_correct_test_cases.append(test_case)
            else:
                if verbose >= 2:
                    print(f"incorrect test case: expected: {expected}; got: {test_case['results']}")
            try:
                list_elements += test_case['argument_l']
            except:
                print(f"Error: argument_l == {test_case['argument_l']}")
            try:
                list_lengths.add(len(test_case['argument_l']))
            except:
                print(f"Error: argument_l == {test_case['argument_l']}")
                pass
            try:
                indices.append(test_case['argument_item'])
            except:
                print(f"Error: argument_item == {test_case['argument_item']}")
        return {
            'num_correct': num_correct,
            'num_different_correct': num_different_correct,
            'num_total': len(test_cases),
            'diversity_list_elements': GenerateTestCaseTask.count_distinct(list_elements),
            'diversity_list_lengths': len(list_lengths),
            'diversity_items': GenerateTestCaseTask.count_distinct(indices),
        }


class GenerateTestCaseTaskBenchmark:
    def __init__(
        self,
        num_tasks,
        num_demonstrations,
        num_test_cases_per_function,
        function_name_length: int = 3,
        function_name = None,
        verbose: int = 1,
    ):
        self.num_tasks = num_tasks
        self.num_demonstrations = num_demonstrations
        self.num_test_cases_per_function = num_test_cases_per_function
        self.function_name_length = function_name_length
        self.function_name = function_name
        self.verbose = verbose

    def run_single(
            self,
            engine,
            generation_config_name,
            args,
            task,
            metric_names,
            all_metrics,
        ):
        print('function name:', task.function_name)
        if self.verbose >= 2:
            print(task.prompt)
        result = engine.generate(
                task.prompt,
                max_tokens=256,
                **args,
            )
        if self.verbose >= 2:
            print(result)
        test_cases = task.parse(result)
        metrics = task.check(test_cases, verbose=self.verbose)
        for metric_name in metric_names:
            all_metrics[metric_name][generation_config_name].append(
                metrics[metric_name]
            )

    def run(self, engine):
        generation_config_names = [
            'top_p0.8',
            'argmax',
            'temperature0.2',
            'beam8_length_penalty-2',
            'beam8_length_penalty0',
            'beam8_length_penalty-10',
            'beam8_length_penalty10',
        ]
        args = [
            {'top_p': 0.8},
            {'top_k': 1},
            {'temperature': 0.2},
            {'beam_size': 8},
            {'beam_size': 8, 'length_penalty': 0.0},
            {'beam_size': 8, 'length_penalty': -10.0},
            {'beam_size': 8, 'length_penalty': 10.0},
        ]
        assert len(generation_config_names) == len(args)

        metric_names = [
            'num_correct',
            'num_different_correct',
            'num_total',
            'diversity_list_elements',
            'diversity_list_lengths',
            'diversity_items',
        ]
        all_metrics = {
            metric_name: {
                generation_config_name: []
                for generation_config_name in generation_config_names
            }
            for metric_name in metric_names
        }

        for num_tasks in range(self.num_tasks):
            task = GenerateTestCaseTask(
                self.num_demonstrations,
                self.num_test_cases_per_function,
                function_name_length=self.function_name_length,
                function_name=self.function_name,
            )

            for i in range(len(generation_config_names)):
                self.run_single(
                    engine,
                    generation_config_names[i],
                    args[i],
                    task,
                    metric_names,
                    all_metrics,
                )
            print(f"\n\n Checked {num_tasks} tasks")
            print(all_metrics)
        return all_metrics


class GenerateTestCaseTaskBenchmarkParallel:
    def __init__(
        self,
        num_tasks,
        num_demonstrations,
        num_test_cases_per_function,
        num_repetitions_per_prompt,
        function_name_length: int = 3,
        function_name = None,
        verbose: int = 1,
        save_results_path_starting_idx: int = 0,
    ):
        self.num_tasks = num_tasks
        self.num_demonstrations = num_demonstrations
        self.num_test_cases_per_function = num_test_cases_per_function
        self.num_repetitions_per_prompt = num_repetitions_per_prompt
        self.function_name_length = function_name_length
        self.function_name = function_name
        self.verbose = verbose
        self.save_results_path_starting_idx = save_results_path_starting_idx

    def run_single(
            self,
            engine,
            generation_config_name,
            args,
            task,
            metric_names,
            all_metrics,
            prompts_and_generations,
        ):
        print('function name:', task.function_name)
        if self.verbose >= 2:
            print(task.prompt)
        for _ in range(self.num_repetitions_per_prompt):
            result = engine.generate(
                    task.prompt,
                    max_tokens=256,
                    **args,
                )
            prompts_and_generations[task.prompt][generation_config_name].append(result)
            if self.verbose >= 2:
                print(result)
            test_cases = task.parse(
                result,
                test_cases=[],
                verbose=self.verbose,
            )
            if len(test_cases) > self.num_test_cases_per_function:
                print(f"Warning: generated {len(test_cases)} test cases although instructed to generate only {self.num_test_cases_per_function}. result = {result}")
                test_cases = test_cases[:self.num_test_cases_per_function]
            if generation_config_name == 'argmax':
                break  # model will generate the same result by repeating argmax

            metrics = task.check(test_cases, verbose=self.verbose)
            if metrics['num_different_correct'] > self.num_test_cases_per_function:
                raise ValueError(f"num_different_correct = {metrics['num_different_correct']}, metrics = {metrics}, result = {result}")
            for metric_name in metric_names:
                all_metrics[metric_name][generation_config_name].append(
                    metrics[metric_name]
                )

    def run(self, engine):
        generation_config_names = [
            'top_p0.8',
            'argmax',
            'temperature0.2',
            'beam8_length_penalty-0.5',
            'beam8_length_penalty0',
            'beam8_length_penalty0.5',
        ]
        args = [
            {'top_p': 0.8},
            {'top_k': 1},
            {'temperature': 0.2},
            {'beam_size': 8, 'length_penalty': -0.5},
            {'beam_size': 8, 'length_penalty': 0.0},
            {'beam_size': 8, 'length_penalty': 0.5},
        ]
        assert len(generation_config_names) == len(args)

        metric_names = [
            'num_correct',
            'num_different_correct',
            'num_total',
            'diversity_list_elements',
            'diversity_list_lengths',
            'diversity_items',
        ]
        all_metrics = {
            metric_name: {
                generation_config_name: []
                for generation_config_name in generation_config_names
            }
            for metric_name in metric_names
        }
        prompts_and_generations = {}

        for num_tasks in range(self.num_tasks):
            task = GenerateTestCaseTask(
                self.num_demonstrations,
                self.num_test_cases_per_function,
                function_name_length=self.function_name_length,
                function_name=self.function_name,
            )
            if task.prompt not in prompts_and_generations:
                prompts_and_generations[task.prompt] = {
                    generation_config_name: []
                    for generation_config_name in generation_config_names
                }

            for i in range(len(generation_config_names)):
                self.run_single(
                    engine,
                    generation_config_names[i],
                    args[i],
                    task,
                    metric_names,
                    all_metrics,
                    prompts_and_generations,
                )
            print(f"\n\n Checked {num_tasks} tasks")
            print(all_metrics)

            with open(f'all_metrics_task10_repeat10_{num_tasks+self.save_results_path_starting_idx}.pkl', 'wb') as f:
                pickle.dump(all_metrics, f )

            with open(f'prompts_and_generations_task10_repeat10_{num_tasks+self.save_results_path_starting_idx}.pkl', 'wb') as f:
                pickle.dump(prompts_and_generations, f )

        return all_metrics, prompts_and_generations


