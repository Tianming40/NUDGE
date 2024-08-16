from nsfr.infer import InferModule, ClauseInferModule, ClauseBodyInferModule
from nsfr.tensor_encoder import TensorEncoder, MetaTensorEncoder
from nsfr.fol.logic import *
from nsfr.fol.logic_ops import *
from nsfr.fol.data_utils import DataUtils
from nsfr.fol.language import DataType
import copy

p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(lark_path, lang_base_path, dataset):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path, dataset=dataset)
    lang = du.load_language()
    clauses = du.get_clauses(lang)
    bk = du.get_bk(lang)
    atoms = generate_atoms(lang)
    return lang, clauses, bk, atoms



def build_infer_module(clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train)
    return im


def build_meta_infer_module(clauses, atoms, lang, m, infer_step, device, train=False ):
    te = MetaTensorEncoder(lang, atoms, clauses,  device=device)
    I = te.encode()
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train)
    return im


def generate_atoms(lang, meta=False):
    spec_atoms = [false, true]
    atoms = []
    for pred in lang.preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        # args_list = lang.get_args_by_pred(pred)
        args_str_list = []
        # args_mem = []
        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                # if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                # if len(set(args)) == len(args):
                # if not (str(sorted([str(arg) for arg in args])) in args_str_list):
                atoms.append(Atom(pred, args))
                # args_str_list.append(
                #    str(sorted([str(arg) for arg in args])))
                # print('add atom: ', Atom(pred, args))
    if meta:
        return sorted(atoms)
    else:
        return spec_atoms + sorted(atoms)


def build_clause_infer_module(clauses, bk_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    if len(bk_clauses) > 0:
        te_bk = TensorEncoder(lang, atoms, bk_clauses, device=device)
        I_bk = te_bk.encode()
    else:
        te_bk = None
        I_bk = None

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk)
    return im


def build_clause_body_infer_module(clauses, atoms, lang, device, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    # TODO
    im = ClauseBodyInferModule(I, device=device, train=train)
    # im = ClauseInferModule(I, device=device, train=train)
    return im


def get_prednames(clauses):
    prednames = []
    for clause in clauses:
        prednames.append(clause.head.pred.name)
    return prednames


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ['diff_color', 'diff_shape']:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    atoms.append(Atom(pred, args))
    return atoms


def get_index_by_predname(pred_str, atoms):
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
    assert 1, pred_str + ' not found.'

def get_index_by_predname_meta(pred_str, metaatoms):
    for i, metaatom in enumerate(metaatoms):
        if metaatom.pred.name == 'solve' :
            if metaatom.terms[0].value.pred.name == pred_str:
                # print('+++++', metaatom)
                return i
    assert 1, pred_str + ' not found.'


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]


def get_searched_clauses(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.
    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.load_clauses(du.base_path + dataset + '/beam_searched.txt', lang)
    return clauses



def get_metalang(lark_path, lang_base_path, dataset, exhaustion = False, filter=True):

    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path, dataset=dataset)
    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path,  dataset)
    # FIXME n must be choosen
    head = []
    body = []
    head_predicate_names = set(clause.head.pred.name for clause in clauses)
    body_predicate_names = set(body.pred.name for clause in clauses for body in clause.body)
    for atom in atoms:
        if atom.pred.name in body_predicate_names:
            body.append(atom)
        if atom.pred.name in head_predicate_names:
            head.append(atom)

    # filtered_atoms = [atom for atom in atoms if atom not in head and atom not in body]

    patterns = get_patterns(clauses)

    metaconsts = generate_metaconsts(generate_atoms(lang, True), head, lang, patterns)

    metalang = du.load_metalanguage(metaconsts)
    meta_bk_true = du.load_meta_clauses(du.base_path + 'clauses.txt', metalang, atoms)
    meta_bk = du.load_meta_atoms(du.base_path + 'bk.txt', metalang)
    meta_bk += meta_bk_true
    meta_interpreter = du.load_interpreter(du.base_path + 'proof_tree_interpreter.txt', metalang)

    n = 0
    for clause in clauses:
        if len(clause.body) > n:
            n = len(clause.body)
    metalang = metalang_extend_proof(n, metalang, bk, head, body, exhaustion)


    metalang,meta_atoms = generate_metaatoms(metalang, meta_bk, exhaustion )
    if filter:
        meta_atoms = [atom for atom in meta_atoms if not (atom.pred.name == 'clause' and atom not in meta_bk)]
        return metalang, meta_bk, meta_interpreter, meta_atoms
    else:
        return metalang, meta_bk, meta_interpreter, meta_atoms


def get_value(atom,bk):
    if atom in bk:
        prob = 1
    elif type(atom.pred) == NeuralPredicate:
        prob = 0
       # prob = vm
    else:
        prob = 0
    return prob

def metalang_extend_proof(n, metalang, bk, head, body, exhaustion = False):
    if exhaustion:
        consts_proof = []
        atoms_list = []
        for atom in body:
            meta_proof = MetaConst(proof([atom],get_value(atom, bk)),dtype=DataType('proof'))
            consts_proof.append(meta_proof)
        for const in metalang.metaconsts:
            if const.dtype == DataType('atoms'):
                atoms_list.append(const)
            # if const.dtype == 'atom':
            #     atoms_list.append(MetaConst([const.value],dtype = 'atoms'))

        for i in range(n):
            new_proofs = set()

            current_combinations = list(itertools.product(consts_proof, repeat=2))
            filtered_combinations = [combo for combo in current_combinations if combo[0] != combo[1]]

            for combination in filtered_combinations:
                for atom in atoms_list:
                    new_meta_proof = MetaConst(proof(atom, [combination]), dtype=DataType('proof'))
                    new_proofs.add(new_meta_proof)

            consts_proof.extend(new_proofs)
            consts_proof = list(set(consts_proof))

        metalang.metaconsts += consts_proof
    else:
        metalang.metaconsts.append(MetaConst(value=proof(atoms=None, tree=None), dtype=DataType('proof')))

    return metalang

def get_patterns(clauses):
    all_patterns = []

    for clause in clauses:
        bodys = clause.body.copy()

        while len(bodys) > 0:
            all_patterns.append(bodys.copy())

            if len(bodys) > 1:
                popped_atom = bodys.pop(0)
                all_patterns.append([popped_atom])
            else:
                popped_atom = bodys.pop(0)

    return all_patterns



def generate_metaconsts(atoms, head, lang,patterns):
    # FIxme modify
    metaconsts = []
    head_atoms = []
    ite_body_atoms = []
    for atom in atoms:
        if atom in head:
            meta_atom = MetaConst(atom,  dtype=DataType('atom'))
            head_atoms.append(meta_atom)

    # for i in range(1, n+1):
    #     for combo in itertools.product(ite_body_atoms, repeat=i):  # Cartesian Product with len=1
    #         if len(set(combo)) == len(combo):
    #             combo = list(combo)
    #             if ispattern(combo, pattern):
    #                 metaconst_atoms = MetaConst(combo, dtype='atoms')
    #                 metaconsts.append(metaconst_atoms)
    for pattern in patterns:
        theta_list = generate_subs(lang, pattern)
        for the in theta_list:
            body_cons = [subs_list(bi, the) for bi in pattern]

            body_consts = MetaConst(body_cons, dtype=DataType('atoms'))
            metaconsts.append(body_consts)

    metaconsts += head_atoms
    return metaconsts


def generate_subs( lang, body):
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



def ispattern(atoms, pattern):
    atomspattern = []
    for atom in atoms:
        atomspattern.append(atom.pred)
    if atomspattern in pattern:
        return True
    else: return False



def generate_metaatoms(lang, bk, exhaustion = False):
    metaatoms = []
    consts_proof =[]
    for pred in lang.metapreds:
        dtypes = pred.dtypes
        consts_list = [lang.get_meta_by_dtype(dtype) for dtype in dtypes]
        # print(pred,'++++++++++++++++++++************************************************************\n',consts_list)
        args_list = list(set(itertools.product(*consts_list)))
        # print(pred,'++++++++++++++++++++************************************************************\n',args_list)
        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                args = list(args)
                metaatoms.append(MetaAtom(pred, args))

    if not exhaustion:
        metaatoms = get_proof_tree(metaatoms, bk, lang)
        for atom in metaatoms:
            if atom.pred.has_proof:
                consts_proof.append(atom.terms[atom.pred.proof_index[0]])

        lang.metaconsts += consts_proof
        lang.metaconsts.append(MetaConst(value=proof(atoms=MetaConst(true, dtype=DataType('atom')), tree=1),dtype=DataType('proof')))
        lang.metaconsts.append(MetaConst(value=proof(atoms=MetaConst(false, dtype=DataType('atom')), tree=0),dtype=DataType('proof')))
        lang.metaconsts.remove(MetaConst(value=proof(atoms=None, tree=None), dtype=DataType('proof')))

    metasolveture = MetaAtom(lang.get_meta_pred_by_name('solve'), [MetaConst(true, dtype=DataType('atom')),
                                                                   MetaConst(value=proof(
                                                                       atoms=MetaConst(true, dtype=DataType('atom')), tree=1),
                                                                       dtype=DataType('proof'))])
    metasolvefalse = MetaAtom(lang.get_meta_pred_by_name('solve'), [MetaConst(false, dtype=DataType('atom')),
                                                                    MetaConst(value=proof(
                                                                        atoms=MetaConst(false, dtype=DataType('atom')),
                                                                        tree=0), dtype=DataType('proof'))])
    spec_meta_atom = [metasolvefalse, metasolveture]
    return lang,spec_meta_atom + sorted(metaatoms)


def get_proof_tree(atoms, bk, metalang):
    atoms_with_proof = []
    for atom in atoms:
        if atom.pred.has_proof:
            if atom.pred.name == 'solve*' and len(atom.terms[0].value)>1 :
                atom_with_proof = get_proof_for_many_Atoms(atom, bk, metalang)
            else:
                atom_with_proof = get_proof_for_single_Atom(atom, bk, metalang)
            atoms_with_proof+=atom_with_proof
        else:
            atoms_with_proof.append(atom)
    return atoms_with_proof

def get_bk_clauses_group(bk):
    clause_groups = {}
    for atom in bk:
        if atom.pred.name == 'clause':
            term_0 = atom.terms[0].value
            if term_0 not in clause_groups:
                clause_groups[term_0] = []
            clause_groups[term_0].append(atom)
    return clause_groups


def get_proof_for_single_Atom(atom,bk,metalang):
    solve_pred = metalang.get_meta_pred_by_name('solve*')
    Atom_with_proof = []
    bk_without_meta = []
    for bkatom in bk:
        if bkatom.pred.name == 'solve*':
            bk_without_meta.append(bkatom.terms[0].value[0])
    bk_without_meta.append(true)
    bk_atom_has_body = get_bk_clauses_group(bk)


    if atom.pred.name == 'solve*' :
        atom_with_proof = copy.deepcopy(atom)
        atom_with_proof.terms[1].value.atoms = atom.terms[0]
        atom_with_proof.terms[1].value.tree = get_value(atom.terms[0].value[0], bk_without_meta)
        Atom_with_proof.append(atom_with_proof)
    elif atom.pred.name == 'solve':
        atom_with_proof = copy.deepcopy(atom)
        atom_with_proof.terms[1].value.atoms = atom.terms[0]
        subs_list = bk_atom_has_body[atom.terms[0].value]
        for i, subs in enumerate(subs_list):
            bodyatom = MetaAtom(solve_pred, [subs.terms[1], MetaConst(value=proof(atoms=None, tree=None), dtype=DataType('proof'))])
            atom_with_proof_body = get_proof_tree([bodyatom],bk,metalang)
            for _atom in atom_with_proof_body:
                new_atom = copy.deepcopy(atom_with_proof)
                new_atom.terms[1].value.atoms = subs.terms[1]
                new_atom.terms[1].value.tree = _atom.terms[1].value.tree
                Atom_with_proof.append(new_atom)
    return flatten(Atom_with_proof)

def get_proof_for_many_Atoms(atom,bk,metalang):
    # 返回list！！！！
    solve_pred = metalang.get_meta_pred_by_name('solve*')
    Atom_with_proof=[]
    atom_with_proof = copy.deepcopy(atom)
    atom_with_proof.terms[1].value.atoms = atom.terms[0]
    atom_with_proof.terms[1].value.tree = []
    first_atom = MetaAtom(solve_pred, [MetaConst(value=[atom_with_proof.terms[0].value[0]], dtype=DataType('atoms')),
                                       MetaConst(value=proof(atoms=None, tree=None), dtype=DataType('proof'))])
    proof_list = get_proof_for_single_Atom(first_atom, bk, metalang)
    if len(atom.terms[0].value) == 2:
        second_atom = MetaAtom(solve_pred, [MetaConst(value=[atom_with_proof.terms[0].value[1]], dtype=DataType('atoms')),
                                       MetaConst(value=proof(atoms=None, tree=None), dtype=DataType('proof'))])
        second_proof_list = get_proof_for_single_Atom(second_atom, bk, metalang)
        for firstproof in proof_list:
            in_proof = copy.deepcopy(atom_with_proof)
            in_proof.terms[1].value.tree.append(firstproof.terms[1].value)
            for secondproof in second_proof_list:
                new_atom_with_proof = copy.deepcopy(in_proof)
                new_atom_with_proof.terms[1].value.tree.append(secondproof.terms[1].value)
                Atom_with_proof.append(new_atom_with_proof)

    else:
        second_atom = MetaAtom(solve_pred, [MetaConst(value=atom_with_proof.terms[0].value[1:], dtype=DataType('atoms')),
                                            MetaConst(value=proof(atoms=None, tree=None), dtype=DataType('proof'))])
        second_proof_list = get_proof_for_many_Atoms(second_atom, bk, metalang)

        for firstproof in proof_list:
            in_proof = copy.deepcopy(atom_with_proof)
            in_proof.terms[1].value.tree.append(firstproof.terms[1].value)
            for secondproof in second_proof_list:
                new_atom_with_proof = copy.deepcopy(in_proof)
                new_atom_with_proof.terms[1].value.tree.append(secondproof.terms[1].value)
                Atom_with_proof.append(new_atom_with_proof)
    return flatten(Atom_with_proof)

