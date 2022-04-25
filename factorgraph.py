import itertools
import pprint
import re
import time
from typing import List
import sys

import numpy as np

pp = pprint.PrettyPrinter(indent=4)


class VariableNode:
    def __init__(self, name) -> None:
        self.name = name
        self.val = None
        self.prob = None  # marginal probability
        self._fix = False

    def is_assigned(self):
        return self.val is not None

    def reset(self):
        if self._fix:
            raise ResetFixedVariableError()
        self.val = None
        self.prob = None

    def fix(self):
        self.prob = None
        self._fix = True

    def unfix(self):
        self._fix = False

    @property
    def fixed(self):
        return self._fix

    def __repr__(self):
        return self.name

    def __str__(self) -> str:
        return f"VariableNode(name={self.name}, val={self.val}, fixed={self.fixed}, prob={self.prob})"


class NoVariableError(Exception):
    pass


class ResetFixedVariableError(Exception):
    pass


class FactorNode:
    """
    formula example:
    "0.95 if V[1] else 0.05"
    """

    var_regex = "V\[\w+]"

    def __init__(self, name, formula) -> None:
        self.name = name
        self.formula = formula
        self.prob = None

    def calc_prob(self, variables, scope) -> float:
        var_names = self.get_neighbors()
        conv_formula = self.formula
        for var_name in var_names:
            val = FactorNode.var_to_val(var_name, variables)
            conv_formula = conv_formula.replace(var_name, str(val))
        augmented_locals = {**scope, **locals()} if scope else locals()
        return eval(conv_formula, globals(), augmented_locals)

    def reset(self):
        self.prob = None

    def is_assigned(self):
        return self.prob is not None

    def get_neighbors(self) -> List[str]:
        return list(set(re.findall(self.var_regex, self.formula)))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"FactorNode(name={self.name}, formula={self.formula}, prob={self.prob})"

    @staticmethod
    def var_to_val(var_name, variables):
        var_name = var_name[2:-1]
        for var in variables:
            if var.name == var_name:
                return var.val
        raise NoVariableError(f"Variable not found: {var_name}")


class FactorGraph:
    def __init__(
        self, variables: List[VariableNode], factors: List[FactorNode]
    ):
        self.variables = variables
        self.factors = factors

    @property
    def size(self):
        return len(self.variables)

    @property
    def size_factor(self):
        return len(self.factors)

    def add_variable(self, variable_node: VariableNode):
        self.variables.append(variable_node)

    def get_variable(self, name):
        for var in self.variables:
            if var.name == name:
                return var
        raise NoVariableError(f"Variable not found: {name}")

    def add_factor(self, factor_node: FactorNode):
        self.factors.append(factor_node)

    def inference_exhaustive(self, scope=None):
        cnt_unfixed_var = len([var for var in self.variables if not var.fixed])
        best_prob = 0
        best_val_vec = None
        print("inference_exhaustive")
        num_cases = 2 ** cnt_unfixed_var
        print("num cases:", num_cases)
        batch_size = 1000
        time_previous = []
        for idx, val_vec in enumerate(
            itertools.product([0, 1], repeat=cnt_unfixed_var)
        ):
            start_time = time.time()
            self.assign_vals(val_vec)
            prob = self.calc_prob(scope)
            if prob > best_prob:
                best_prob = prob
                best_val_vec = val_vec
            time_taken = time.time() - start_time
            time_previous.append(time_taken)
            if idx % batch_size == 0 and idx:
                avg_time = np.mean(time_previous)
                print(
                    f"{idx} cases done ({(idx + 1)/ num_cases * 100}%), avg time: {avg_time}",
                    end="\r",
                    flush=True,
                )
                time_previous = []
        self.assign_vals(best_val_vec)
        self.assign_prob(scope)
        self.best_prob = best_prob

    def marginal_inference_exhaustive(self, scope=None):
        unfixed_vars = [var for var in self.variables if not var.fixed]
        cnt_unfixed_var = len(unfixed_vars)
        prob_mat = []
        for val_vec in itertools.product([0, 1], repeat=cnt_unfixed_var):
            self.assign_vals(val_vec)
            prob = self.calc_prob(scope)
            prob_mat.append(list(val_vec) + [prob])
        self.reset()
        prob_mat = np.array(prob_mat)
        sum_prob = np.sum(prob_mat[:, -1])
        for idx, var in enumerate(unfixed_vars):
            marginal_prob = (
                np.sum(prob_mat[:, -1] * prob_mat[:, idx]) / sum_prob
            )
            var.prob = marginal_prob

    def marginal_sum_product(self, scope=None, debug=False):
        max_iter = 100
        if debug:
            start_time = time.time()
        message_dict = self.get_message_dict()
        if debug:
            print(
                f"Initialize meesage dict (time : {time.time() - start_time}, memory: {sys.getsizeof(message_dict)} bytes)"
            )
            start_time = time.time()

        converge_cnt = 0
        marginal_probs = {var: -1 for var in self.variables}
        if debug:
            avg_iter_time = []
        for iter in range(max_iter):
            # if debug:  # print iteration
            print(f"iteration {iter + 1} starts")
            if debug:
                iter_start_time = time.time()
            message_dict = self.update_message_dict(message_dict, scope, debug)
            if debug:
                print(
                    f"update message dict (time : {time.time() - iter_start_time}"
                )
                probcalc_start_time = time.time()
            new_marginal_probs = self.calc_marginal_probs(message_dict)
            if debug:
                print(
                    f"calc marginal probs (time : {time.time() - probcalc_start_time}"
                )
                print(f"iter took {time.time() - iter_start_time} seconds")
                avg_iter_time.append(time.time() - iter_start_time)
            if marginal_probs == new_marginal_probs:
                converge_cnt += 1
                if converge_cnt >= 2:
                    break
            marginal_probs = new_marginal_probs
        if debug:
            print(f"avg iter time: {np.mean(avg_iter_time)}")
        if converge_cnt < 2:
            print(f"Warning: max iteration reached: {max_iter}")
        else:
            print(f"Converge after {iter} iterations")
        for var in self.variables:
            if not var.fixed:
                var.prob = marginal_probs[var][1]

    def get_message_dict(self, message_dict=None):
        if message_dict:
            new_message_dict = {"v2f": {}, "f2v": {}}
            for var in message_dict["v2f"]:
                new_message_dict["v2f"][var] = {
                    factor: (0.5, 0.5) for factor in message_dict["v2f"][var]
                }
            for factor in message_dict["f2v"]:
                new_message_dict["f2v"][factor] = {
                    var: (0.5, 0.5) for var in message_dict["f2v"][factor]
                }
            return new_message_dict
        else:
            v2f_message, f2v_message = {}, {}
            for factor in self.factors:
                neighbor_names = factor.get_neighbors()
                neighbors = [
                    self.get_variable(name[2:-1]) for name in neighbor_names
                ]
                neighbors_unfixed = [var for var in neighbors if not var.fixed]
                f2v_message[factor] = {
                    var: (0.5, 0.5) for var in neighbors_unfixed
                }
                for var in neighbors_unfixed:
                    if var not in v2f_message:
                        v2f_message[var] = {}
                    v2f_message[var][factor] = (0.5, 0.5)
            return {"v2f": v2f_message, "f2v": f2v_message}

    def update_message_dict(self, message_dict, scope, debug):
        new_message_dict = self.get_message_dict()
        old_v2f_message = message_dict["v2f"]
        old_f2v_message = message_dict["f2v"]
        if debug:
            start_time = time.time()
            print("v2f message start...")
        for idx, v in enumerate(old_v2f_message, start=1):
            vdict = old_v2f_message[v]
            for f in vdict:
                if fstars := set(vdict.keys()) - {f}:
                    message = []
                    for vval in [0, 1]:
                        prob = 1
                        for fstar in fstars:
                            prob *= old_f2v_message[fstar][v][vval]
                        message.append(prob)
                    sum_message = sum(message)
                    new_message_dict["v2f"][v][f] = tuple(
                        p / sum_message for p in message
                    )
                else:
                    # uniform distribution
                    new_message_dict["v2f"][v][f] = (0.5, 0.5)
            if debug:
                print(
                    f"{idx} var2facs done ({idx/ len(old_v2f_message) * 100:.2f}%)",
                    end="\r",
                    flush=True,
                )
        if debug:
            print(f"v2f message done (time : {time.time() - start_time})")
            start_time = time.time()
            print("f2v message start...")
        for idx, f in enumerate(old_f2v_message, start=1):
            fdict = old_f2v_message[f]
            for v in fdict:
                # if is_write_iter: print(f"{(f, v)}", end="\r", flush=True)
                if not (vstars := set(fdict.keys()) - {v}):
                    if debug:
                        print(
                            f"{idx - 1} fac2vars done... ({(idx - 1)/ len(old_f2v_message) * 100}%)\n{(f, v)} ing... no vstar",
                            end="\r",
                            flush=True,
                        )
                    v.val = 0
                    prob_0 = f.calc_prob(self.variables, scope)
                    v.val = 1
                    prob_1 = f.calc_prob(self.variables, scope)
                    new_message_dict["f2v"][f][v] = (prob_0, prob_1)
                else:
                    message = []
                    vstars_unfixed = [
                        vstar for vstar in vstars if not vstar.fixed
                    ]
                    if debug:
                        print(
                            f"{idx - 1} fac2vars done... ({(idx - 1)/ len(old_f2v_message) * 100}%)\n{(f, v)} ing... star cases = {2 ** len(vstars_unfixed)}",
                            end="\r",
                            flush=True,
                        )
                    for vval in [0, 1]:
                        v.val, prob = vval, 0
                        for val_vec in itertools.product(
                            [0, 1], repeat=len(vstars_unfixed)
                        ):
                            self.assign_vals(dict(zip(vstars_unfixed, val_vec)))
                            prob_vstar = f.calc_prob(self.variables, scope)
                            for vstar in vstars:
                                prob_vstar *= old_v2f_message[vstar][f][
                                    vstar.val
                                ]
                            prob += prob_vstar
                        message.append(prob)
                    sum_message = sum(message)
                    new_message_dict["f2v"][f][v] = tuple(
                        p / sum_message for p in message
                    )
            # if is_write_iter:
            #     print(
            #         f"{idx - 1} fac2vars done... ({(idx - 1)/ len(old_f2v_message) * 100}%)\n{(f, v)} ing... star cases = {2 ** len(vstars_unfixed)}",
            #         end="\r",
            #         flush=True,
            #     )
        if debug:
            print(f"f2v message done (time : {time.time() - start_time})")
        return new_message_dict

    def calc_marginal_probs(self, message_dict):
        marginal_probs = {}
        for v in message_dict["v2f"]:
            prob_0, prob_1 = 1, 1
            for f in message_dict["v2f"][v]:
                prob_0 *= message_dict["f2v"][f][v][0]
                prob_1 *= message_dict["f2v"][f][v][1]
            prob_sum = prob_0 + prob_1
            prob_0 = prob_0 / prob_sum
            prob_1 = prob_1 / prob_sum
            marginal_probs[v] = (prob_0, prob_1)
        return marginal_probs

    def assign_vals(self, val_vec):
        if not isinstance(val_vec, dict):
            unfixed_vars = [var for var in self.variables if not var.fixed]
            val_vec = dict(zip(unfixed_vars, val_vec))
        for var, val in val_vec.items():
            var.val = val

    def calc_prob(self, scope):
        prob = 1
        for factor in self.factors:
            prob *= factor.calc_prob(self.variables, scope)
        return prob

    def assign_prob(self, scope):
        for factor in self.factors:
            factor.prob = factor.calc_prob(self.variables, scope)

    def reset(self):
        unfixed_vars = [var for var in self.variables if not var.fixed]
        for var in unfixed_vars:
            var.reset()
        for factor in self.factors:
            factor.reset()

    def __str__(self) -> str:
        ret = "".join(f"{variable}\n" for variable in self.variables)
        for factor in self.factors:
            ret += f"{factor}\n"
        return ret
