import os

from nsfr.facts_converter import FactsConverter
from nsfr.meta_facts_converter import MetaFactsConverter
from nsfr.utils.logic import get_lang, build_infer_module,get_metalang,build_meta_infer_module
from nsfr.nsfr import NSFReasoner
from nsfr.metansfr import MetaNSFReasoner
from nsfr.valuation import ValuationModule


def get_nsfr_model(env_name: str, rules: str, device: str, train=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return NSFR


def get_meta_nsfr_model(env_name: str, rules: str, device: str,clause_weight:dict, train=False):

    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = f"in/envs/{env_name}/logic/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, rules)
    n = 0
    for clause in clauses:
        if len(clause.body) > n:
            n = len(clause.body)
    metalang, meta_bk, meta_interpreter, meta_atoms = get_metalang(lark_path, lang_base_path, rules,  exhaustion = False, filter=True)

    current_directory = os.getcwd()
    long_text_list = [str(meta_atom) for meta_atom in meta_atoms]
    long_text = "\n".join(long_text_list)
    file_name = f"meta_atom_output_{env_name}.txt"
    file_path = os.path.join(current_directory, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(long_text)
    print(f"the meta_atoms file has saved in {file_path}~~~~~~~~~~~~")

    val_fn_path = f"in/envs/{env_name}/valuation.py"
    val_module = ValuationModule(val_fn_path, metalang, device)

    FC = MetaFactsConverter(lang=metalang, valuation_module=val_module, clause_weight=clause_weight,device=device)
    # prednames = []
    # for clause in clauses:
    #     if clause.head.pred.name not in prednames:
    #         prednames.append(clause.head.pred.name)
    # m = len(prednames)
    # m = 5
    IM = build_meta_infer_module(meta_interpreter, meta_atoms, metalang, m=len(meta_interpreter), infer_step=n,
                                 train=train, device=device)
    # Neuro-Symbolic Forward Reasoner

    MetaNSFR = MetaNSFReasoner(facts_converter=FC, infer_module=IM, atoms=meta_atoms, bk=meta_bk, meta_interpreter=meta_interpreter, clauses = clauses,train=train)
    return MetaNSFR
