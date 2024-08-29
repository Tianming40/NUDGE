import os.path

from lark import Lark
from .exp_parser import ExpTree
from .language import Language, DataType, MetaLanguage
from .logic_ops import unify,subs_list
from .logic import Predicate, NeuralPredicate, FuncSymbol, Const, MetaPredicate,MetaRule,MetaAtom, Atom,MetaConst,proof
import itertools

class DataUtils(object):
    """Utilities about logic.

    A class of utilities about first-order logic.

    Args:
        dataset_type (str): A dataset type (kandinsky or clevr).
        dataset (str): A dataset to be used.

    Attrs:
        base_path: The base path of the dataset.
    """

    def __init__(self, lark_path, lang_base_path, dataset: str):
        self.base_path = lang_base_path + dataset + '/'
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_metaRule = Lark(grammar.read(), start="metarule")

    p_ = Predicate('.', 1, [DataType('spec')])
    false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
    true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])

    def load_clauses(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')
                tree = self.lp_clause.parse(line)
                clause = ExpTree(lang).transform(tree)
                clauses.append(clause)
        return clauses

    def load_atoms(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        atoms = []

        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    tree = self.lp_atom.parse(line[:-2])
                    atom = ExpTree(lang).transform(tree)
                    atoms.append(atom)
        return atoms

    def load_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_pred(line) for line in lines]
        return preds

    def load_neural_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_neural_pred(line) for line in lines]
        return preds

    def load_consts(self, path):
        f = open(path)
        lines = f.readlines()
        consts = []
        for line in lines:
            line = line.replace('\n', '')
            if len(line) > 0:
                consts.extend(self.parse_const(line))
        return consts

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return Predicate(pred, int(arity), dtypes)

    def parse_neural_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return NeuralPredicate(pred, int(arity), dtypes)

    def parse_funcs(self, line):
        """Parse string to function symbols.
        """
        funcs = []
        for func_arity in line.split(','):
            func, arity = func_arity.split(':')
            funcs.append(FuncSymbol(func, int(arity)))
        return funcs

    def parse_const(self, line):
        """Parse string to function symbols.
        """
        dtype_name, const_names_str = line.split(':')
        dtype = DataType(dtype_name)
        const_names = const_names_str.split(',')
        return [Const(const_name, dtype) for const_name in const_names]

    def parse_clause(self, clause_str, lang):
        tree = self.lp_clause.parse(clause_str)
        return ExpTree(lang).transform(tree)

    def get_clauses(self, lang):
        return self.load_clauses(self.base_path + 'clauses.txt', lang)

    def get_bk(self, lang):
        return self.load_atoms(self.base_path + 'bk.txt', lang)

    def load_language(self):
        """Load language, background knowledge, and clauses from files.
        """
        preds = self.load_preds(self.base_path + 'preds.txt') + \
                self.load_neural_preds(self.base_path + 'neural_preds.txt')
        consts = self.load_consts(self.base_path + 'consts.txt')
        lang = Language(preds, [], consts)
        return lang

    def load_meta_clauses(self, path, lang, facts):
        meta_clauses = []
        clauses = []
        meta_bk_clause_true = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]
                    tree = self.lp_clause.parse(line)
                    # print(tree)
                    clause = ExpTree(lang).transform(tree)
                    clauses.append(clause)
        head_unifier_dic = self.build_head_unifier_dic(clauses, facts)
        for clause in clauses:
            for fi, fact in enumerate(facts):
                if (clause.head, fact) in head_unifier_dic:
                    theta = head_unifier_dic[(clause.head, fact)]
                    clause_ = subs_list(clause, theta)
                    body = clause_.body
                    theta_list = self.generate_subs(lang,body)
                    if len(theta_list) > 0:
                        for the in theta_list:
                            _clause_ = subs_list(clause_, the)
                            clause_pred = lang.get_meta_pred_by_name('clause')
                            head_cons = MetaConst(_clause_.head, dtype=DataType('atom'))
                            body_cons = MetaConst(_clause_.body, dtype=DataType('atoms'))
                            meta_clause = MetaAtom(clause_pred, [head_cons, body_cons])
                            meta_bk_clause_true.append(meta_clause)
                    else:
                        clause_pred = lang.get_meta_pred_by_name('clause')
                        head_cons = MetaConst(clause_.head, dtype=DataType('atom'))
                        body_cons = MetaConst([self.true], dtype=DataType('atoms'))
                        meta_clause = MetaAtom(clause_pred, [head_cons, body_cons])
                        meta_bk_clause_true.append(meta_clause)



        return meta_bk_clause_true

    def build_head_unifier_dic(self, clauses, facts):
        """Build dictionary {(head, fact) -> unifier}.

        Returns:
            dic ({(atom,atom) -> subtitution}): A dictionary to map the pair of ground atoms to their unifier.
        """
        dic = {}
        heads = set([c.head for c in clauses])
        for head in heads:
            for fi, fact in enumerate(facts):
                unify_flag, theta_list = unify([head, fact])
                if unify_flag:
                    dic[(head, fact)] = theta_list
        return dic

    def generate_subs(self, lang, body):
        """Generate substitutions from given body atoms.

        Generate the possible substitutions from given list of atoms. If the body contains any variables,
        then generate the substitutions by enumerating constants that matches the data type.
        Args:
            body (list(atom)): The body atoms which may contain existentially quantified variables.

        Returns:
            theta_list (list(substitution)): The list of substitutions of the given body atoms.
        """
        # example: body = [on_left(O2,O1)]
        # extract all variables and corresponding data types from given body atoms
        var_dtype_list = []
        dtypes = []
        vars = []
        for atom in body:
            terms = atom.terms
            for i, term in enumerate(terms):
                if term.is_var():
                    v = term
                    dtype = atom.pred.dtypes[i]
                    var_dtype_list.append((v, dtype))
                    dtypes.append(dtype)
                    vars.append(v)
        # in case there is no variables in the body
        if len(list(set(dtypes))) == 0:
            return []

        # {O2: [obj2, obj3, obj4, obj5, obj6, obj7, obj8, obj9, obj10, obj11], O1: [obj1]}
        var_to_consts_dic = {}
        for v, dtype in var_dtype_list:
            if not v in var_to_consts_dic:
                var_to_consts_dic[v] = lang.get_by_dtype(dtype)

        # [[obj2, obj3, obj4, obj5, obj6, obj7, obj8, obj9, obj10, obj11], [obj1]]
        subs_consts_list = list(var_to_consts_dic.values())
        # for v in vars:
        #     subs_consts_list.append(var_to_consts_dic[v])

        # [(obj2, obj1), (obj3, obj1), (obj4, obj1), (obj5, obj1), (obj6, obj1), (obj7, obj1), (obj8, obj1), (obj9, obj1), (obj10, obj1), (obj11, obj1)]
        subs_consts_list_by_product = list(itertools.product(*subs_consts_list))

        # [O2, O1]
        subs_vars = list(var_to_consts_dic.keys())

        # [[(O2, obj2), (O1, obj1)], [(O2, obj3), (O1, obj1)], ...]
        theta_list = []
        for subs_consts in subs_consts_list_by_product:
            theta = []

            for i, const in enumerate(subs_consts):
                s = (subs_vars[i], const)
                theta.append(s)
            theta_list.append(theta)
        return theta_list

    def load_meta_atoms(self, path, lang):
        # TODO adjust
        metaatoms = []
        solve_pred = lang.get_meta_pred_by_name('solve*')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-2]
                    else:
                        line = line[:-1]
                    tree = self.lp_atom.parse(line)
                    atom = ExpTree(lang).transform(tree)

                    metaconst = MetaConst([atom],dtype=DataType('atoms'))
                    metasolve = MetaAtom(solve_pred, [metaconst, MetaConst(proof(atoms=metaconst, tree = 1),dtype=DataType('proof'))])
                    metaatoms.append(metasolve)
        true_const = MetaConst([self.true], dtype=DataType('atoms'))
        false_const = MetaConst([self.false], dtype=DataType('atoms'))

        meta_true = MetaAtom(solve_pred, [true_const, MetaConst(proof(atoms=true_const, tree = 1),dtype=DataType('proof'))])
        meta_false = MetaAtom(solve_pred, [false_const, MetaConst(proof(atoms=false_const, tree = 0),dtype=DataType('proof'))])
        metaatoms += [meta_true]
        return metaatoms

    def load_interpreter(self,path,lang):
        metaInterpreter = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-1]
                    tree = self.lp_metaRule.parse(line)
                    # print(tree)
                    rule = ExpTree(lang).transform(tree)
                    # print(rule.metahead)
                                            # print(isinstance(rule.metahead, MetaSolve))
                    metaInterpreter.append(rule)
        return metaInterpreter

    def load_meta_preds(self , path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_meta_pred(line) for line in lines]
        return preds

    def parse_meta_pred(self, line):
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return MetaPredicate(pred, int(arity), dtypes)

    def load_metalanguage(self,metaconsts):
        preds = self.load_preds(self.base_path + 'preds.txt') + \
                self.load_neural_preds(self.base_path + 'neural_preds.txt')
        consts = self.load_consts(self.base_path + 'consts.txt')
        metapreds = self.load_meta_preds(self.base_path + 'meta_preds_proof.txt')
        metalang = MetaLanguage(preds, metapreds, [], consts, metaconsts)
        return metalang