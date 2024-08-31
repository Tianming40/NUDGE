import numpy as np
import torch.nn as nn
import torch
from .fol.logic import *
from .fol.language import DataType
from nsfr.utils.logic import get_index_by_predname_meta,get_index_for_tree
import copy
from nsfr.utils.torch import softor, weight_sum
import torch.nn.functional as F


class MetaNSFReasoner(nn.Module):
    """The Neuro-Symbolic Forward Reasoner.

    Args:
        perception_model (nn.Module): The perception model.
        facts_converter (nn.Module): The facts converter module.
        infer_module (nn.Module): The differentiable forward-chaining inference module.
        atoms (list(atom)): The set of ground atoms (facts).
    """

    def __init__(self, facts_converter, infer_module, atoms, bk, meta_interpreter, clauses, train=False):
        super().__init__()
        self.fc = facts_converter
        self.im = infer_module
        self.atoms = atoms
        self.bk = bk
        self.obj_clauses = clauses
        self.clauses = meta_interpreter
        self._train = train
        self.prednames = self.get_prednames()
        self.V_0 = []
        self.V_T = []

    def get_params(self):
        return self.im.get_params()  # + self.fc.get_params()

    def get_prednames(self):
        prednames = []
        for clause in self.obj_clauses:
            if clause.head.pred.name not in prednames:
                prednames.append(clause.head.pred.name)
        return prednames

    def forward(self, x):
        zs = x
        # convert to the valuation tensor
        self.V_0 = self.fc(zs, self.atoms, self.bk)
        # perform T-step forward-chaining reasoning
        self.V_T = self.im(self.V_0)
        # only return probs of actions
        actions = self.get_predictions(self.V_T, prednames=self.prednames)
        return actions

    def get_probs(self):
        probs = {}
        for i, atom in enumerate(self.atoms):
            probs[atom] = round(self.V_T[0][i].item(), 3)
        return probs
    def get_predictions(self, V_T, prednames):
        predicts = self.predict_multi(v=V_T, prednames=prednames)
        # print(predicts)
        return predicts

    def predict(self, v, predname):
        """Extracting a value from the valuation tensor using a given predicate.
        """
        # v: batch * |atoms|
        target_index_lst = get_index_by_predname_meta(pred_str=predname, metaatoms=self.atoms)

        max_sum = float('-inf')
        target_index = target_index_lst[0]

        for index in target_index_lst:
            current_sum = v[:, index].sum()
            if current_sum > max_sum:
                max_sum = current_sum
                target_index = index

        # print('+++++++++++++++')
        # print(v)
        # print(v.shape)
        # print(target_index)
        leaves = proof.find_leaf_values(self.atoms[target_index].terms[1].value)
        updated_leaves = []
        for atoms, value in leaves:
            new_value = v[:, get_index_for_tree(atoms, self.atoms)].item()
            updated_leaves.append((atoms, new_value))
        new_tree = proof.reconstruct_proof(updated_leaves)
        gotten_atom = copy.deepcopy(self.atoms[target_index])
        gotten_atom.terms[1].value = MetaConst(new_tree, dtype=DataType('proof'))
        print(str(gotten_atom) + ' + ' + str(v[:, target_index].item()))
        return v[:, target_index]
    def predict_multi(self, v, prednames):
        """Extracting values from the valuation tensor using given predicates.

        prednames = ['kp1', 'kp2', 'kp3']
        """
        # v: batch * |atoms|
        target_indices = []
        B = v.size(0)  # 获取批量大小
        results = []
        for predname in prednames:
            target_index_lst = get_index_by_predname_meta(
                pred_str=predname, metaatoms=self.atoms)
            values_to_or = [v[:, index] for index in target_index_lst]
            soft_or_result = softor(values_to_or, dim=0, gamma=0.01)
            results.append(soft_or_result.unsqueeze(-1))

        prob = torch.cat(results, dim=1)

        N = len(prednames)
        assert prob.size(0) == B and prob.size(1) == N, 'Invalid shape in the prediction.'
        return prob




        B = v.size(0)
        N = len(prednames)
        assert prob.size(0) == B and prob.size(
            1) == N, 'Invalid shape in the prediction.'
        return prob


    def get_top_atoms(self, v):
        top_atoms = []
        for i, atom in enumerate(self.atoms):
            if v[i] > 0.7:
                top_atoms.append(atom)
        return top_atoms

    def atoms_to_text(self, atoms):
        text = ''
        for atom in atoms:
            text += str(atom) + ', '
        return text

    def get_predicate_valuation(self, predname: str, initial_valuation: bool = True):
        valuation = self.V_0 if initial_valuation else self.V_T
        target_index_lst = get_index_by_predname_meta(pred_str=predname, metaatoms=self.atoms)

        max_sum = float('-inf')
        target_index = target_index_lst[0]

        for index in target_index_lst:
            current_sum = valuation[:, index].sum()
            if current_sum > max_sum:
                max_sum = current_sum
                target_index = index
        value = valuation[:, target_index].item()

        gotten_atom = self.atoms[target_index]
        leaves = proof.find_leaf_values(self.atoms[target_index].terms[1].value)
        updated_leaves = []
        for atoms, old_value in leaves:
            index = get_index_for_tree(atoms, self.atoms)
            if index is not None:
                new_value = np.round( valuation[:, get_index_for_tree(atoms, self.atoms)].item(),2)
                updated_leaves.append((atoms, new_value))
            else:
                updated_leaves.append((atoms, old_value))
        new_tree = proof.reconstruct_proof(updated_leaves)
        gotten_atom = copy.deepcopy(self.atoms[target_index])
        gotten_atom.terms[1].value = MetaConst(new_tree, dtype=DataType('proof'))
        # print(str(gotten_atom) + ' + ' + str(valuation[:, target_index].item()))
        return value, gotten_atom