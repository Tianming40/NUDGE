import os.path

from lark import Lark
from .exp_parser import ExpTree
from .language import Language, DataType, MetaLanguage
from .logic_ops import unify,subs_list
from .logic import Predicate, NeuralPredicate, FuncSymbol, Const, MetaPredicate,MetaRule,MetaAtom, Atom,MetaConst
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

    def load_meta_clauses(self, path, lang):
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

        for clause in clauses:
            dtypes = clause.head.pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = list(set(itertools.product(*consts_list)))
            head_atoms = []
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    unify_flag, theta = unify([clause.head, Atom(clause.head.pred, args)])
                    if unify_flag:
                        clause_ = subs_list(clause, theta)
                        body = clause_.body
                        theta_list = self.generate_subs(lang, body)
                        for the in theta_list:
                            _clause_ = subs_list(clause_, the)
                            clause_pred = lang.get_meta_pred_by_name('clause')
                            head_cons = MetaConst(_clause_.head, dtype='atom')
                            body_cons = MetaConst(_clause_.body, dtype='atoms')
                            meta_clause = MetaAtom(clause_pred, [head_cons, body_cons])
                            meta_bk_clause_true.append(meta_clause)
        return meta_bk_clause_true

    def generate_subs(self, lang, body):
        """Generate substitutions from given body atoms.

        Generate the possible substitutions from given list of atoms. If the body contains any variables,
        then generate the substitutions by enumerating constants that matches the data type.
        !!! ASSUMPTION: The body has variables that have the same data type
            e.g. variables O1(object) and Y(color) cannot appear in one clause !!!

        Args:
            body (list(atom)): The body atoms which may contain existentially quantified variables.

        Returns:
            theta_list (list(substitution)): The list of substitutions of the given body atoms.
        """
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
        # check the data type consistency
        assert len(list(set(dtypes))) == 1, "Invalid existentially quantified variables. " + \
                                            str(len(list(set(dtypes)))) + " data types in the body: " + str(
            body) + " dypes: " + str(dtypes)

        vars = list(set(vars))
        n_vars = len(vars)
        consts = lang.get_by_dtype(dtypes[0])

        # e.g. if the data type is shape, then subs_consts_list = [(red,), (yellow,), (blue,)]
        subs_consts_list = itertools.permutations(consts, n_vars)

        theta_list = []
        # generate substitutions by combining variables to the head of subs_consts_list
        for subs_consts in subs_consts_list:
            theta = []
            for i, const in enumerate(subs_consts):
                s = (vars[i], const)
                theta.append(s)
            theta_list.append(theta)
        # e.g. theta_list: [[(Z, red)], [(Z, yellow)], [(Z, blue)]]
        # print("theta_list: ", theta_list)
        return theta_list

    def load_meta_atoms(self, path, lang):
        # TODO adjust
        metaatoms = []
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line[-1] == '\n':
                        line = line[:-2]
                    else:
                        line = line[:-1]
                    tree = self.lp_atom.parse(line)
                    atom = ExpTree(lang).transform(tree)
                    solve_pred=lang.get_meta_pred_by_name('solve*')
                    metaconst = MetaConst([atom],dtype='atoms')
                    metasolve = MetaAtom(solve_pred, [metaconst])
                    metaatoms.append(metasolve)
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
        consts = self.load_consts(self.base_path + 'const.txt')
        metapreds = self.load_meta_preds(self.base_path + 'meta_preds.txt')
        metalang = MetaLanguage(preds, metapreds, [], consts, metaconsts)
        return metalang