import torch
import torch.nn as nn
from .fol.logic import NeuralPredicate,Clause,Atom
from tqdm import tqdm
from .fol.logic import *
from .fol.language import DataType, MetaLanguage
from .fol.logic_ops import unify,subs_list,meta_clause_unify

p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])
class MetaFactsConverter(nn.Module):
    """
    FactsConverter converts the output fromt the perception module to the valuation vector.
    """



    def __init__(self, lang, valuation_module, clause_weight,device=None):
        super(MetaFactsConverter, self).__init__()
        # self.e = perception_module.e
        self.e = 0
        # self.d = perception_module.d
        self.d = 0
        self.lang = lang
        self.clause_weight = clause_weight
        self.vm = valuation_module  # valuation functions
        self.device = device

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def forward(self, Z, G, B):
        return self.convert(Z, G, B)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype(self):
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def convert(self, Z, G, B):
        # FIXME G meta_atoms B Meta_bk Z predict from
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
        V = torch.zeros((batch_size, len(G))).to(
            torch.float32).to(self.device)

        for i, meta_atom in enumerate(G):
            # TODO modify here atom is metaatom
            if meta_atom.pred.name == 'solve*' and type(meta_atom.terms[0].value) == list:
                if len(meta_atom.terms[0].value) == 1 and type(meta_atom.terms[0].value[0].pred) == NeuralPredicate:
                    # TODO vm must be modified
                    V[:, i] = self.vm(Z, meta_atom.terms[0].value[0])
                    # print(V[:, i])
                if meta_atom in B:
                # V[:, i] += 1.0
                    V[:, i] += torch.ones((batch_size, )).to(torch.float32).to(self.device)
            elif meta_atom.pred.name == 'clause' and type(meta_atom.terms[1].value) == list and meta_atom in B:
                for clause, weight in self.clause_weight.items():
                    if meta_clause_unify(clause, meta_atom):
                        V[:, i] += torch.full((batch_size,), weight).to(torch.float32).to(self.device)
                    else:
                        V[:, i] += torch.ones((batch_size,)).to(torch.float32).to(self.device)
        # metasolveture = MetaAtom(self.lang.get_meta_pred_by_name('solve'), [MetaConst([true], dtype='atoms')])
        # index = G.index(metasolveture)
        # # print('ppppppppppp',index)
        # # print('ssssssssssss',G[index])
        V[:, 1] = torch.ones((batch_size, )).to(torch.float32).to(self.device)
        return V

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, meta_atom in enumerate(G):
            if meta_atom.meta_predicate.name == 'solve*' and type(meta_atom.terms[0].value) == list:
                if len(meta_atom.terms[0].value) == 1 and type(meta_atom.meta_terms[0].value[0].pred) == NeuralPredicate:
                    v[i] = self.vm.eval(meta_atom.terms[0].value[0], zs)
        return v

    def call(self, pred):
        return pred
