from nsfr.infer import InferModule, ClauseInferModule, ClauseBodyInferModule
from nsfr.tensor_encoder import TensorEncoder, MetaTensorEncoder
from nsfr.fol.logic import *
from nsfr.fol.data_utils import DataUtils
from nsfr.fol.language import DataType

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



def get_metalang(lark_path, lang_base_path, dataset, n, exhaustion = False, filter=True):

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

    filtered_atoms = [atom for atom in atoms if atom not in head and atom not in body]

    pattern = get_pattern(clauses)

    metaconsts = generate_metaconsts(generate_atoms(lang, True), n, head, body, pattern)

    metalang = du.load_metalanguage(metaconsts)
    meta_bk_true = du.load_meta_clauses(du.base_path + 'clauses.txt', metalang)
    meta_bk = du.load_meta_atoms(du.base_path + 'bk.txt', metalang)
    meta_bk += meta_bk_true
    meta_interpreter = du.load_interpreter(du.base_path + 'naive_meta_interpreter.txt', metalang)
    meta_atoms = generate_metaatoms(metalang, bk, exhaustion )
    if filter:
        meta_atoms = [atom for atom in meta_atoms if not (atom.pred.name == 'clause' and atom not in meta_bk)]
        return metalang, meta_bk, meta_interpreter, meta_atoms
    else:
        return metalang, meta_bk, meta_interpreter, meta_atoms


def get_pattern(clauses):
    all_patterns = []

    for clause in clauses:
        body_preds = [body.pred for body in clause.body]

        while len(body_preds) > 0:
            all_patterns.append(body_preds.copy())

            if len(body_preds) > 1:
                popped_pred = body_preds.pop(0)
                all_patterns.append([popped_pred])
            else:
                popped_pred = body_preds.pop(0)

    return all_patterns



def generate_metaconsts(atoms, n, head, body,pattern):
    # FIxme modify
    metaconsts = []
    head_atoms = []
    ite_body_atoms = []
    for atom in atoms:
        if atom in body:
            ite_body_atoms.append(atom)
        if atom in head:
            meta_atom = MetaConst(atom,  dtype='atom')
            head_atoms.append(meta_atom)

    for i in range(1, n+1):
        for combo in itertools.product(ite_body_atoms, repeat=i):  # Cartesian Product with len=1
            if len(set(combo)) == len(combo):
                combo = list(combo)
                if ispattern(combo, pattern):
                    metaconst_atoms = MetaConst(combo, dtype='atoms')
                    metaconsts.append(metaconst_atoms)
    metaconsts += head_atoms
    return metaconsts

def ispattern(atoms, pattern):
    atomspattern = []
    for atom in atoms:
        atomspattern.append(atom.pred)
    if atomspattern in pattern:
        return True
    else: return False



def generate_metaatoms(lang, bk, exhaustion = False):
    metaatoms = []
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

    metasolveture = MetaAtom(lang.get_meta_pred_by_name('solve'), [MetaConst(true, dtype='atom')])
    metasolvefalse = MetaAtom(lang.get_meta_pred_by_name('solve'), [MetaConst(false, dtype='atom')])
    spec_meta_atom = [metasolvefalse, metasolveture]
    return spec_meta_atom + sorted(metaatoms)



