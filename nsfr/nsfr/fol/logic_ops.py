from .logic import Clause, Atom, FuncTerm, Const, Var,MetaVar, Term, MetaConst,MetaAtom,MetaRule
from .language import DataType


def subs(exp, target_var, const):
    """
    Substitute var = const

    Inputs
    ------
    exp : .logic.CLause .logic.Atom .logic.FuncTerm .logic.Const .logic.Var
        logical expression
        atom, clause, or term
    target_var : .logic.Var
        target variable of the substitution
    const : .logic.Const
        constant to be substituted

    Returns
    -------
    exp : .logic.CLause .logic.Atom .logic.FuncTerm .logic.Const .logic.Var
        result of the substitution
        logical expression
        atom, clause, or term
    """
    if type(exp) == Clause:
        head = subs(exp.head, target_var, const)
        body = [subs(bi, target_var, const) for bi in exp.body]
        return Clause(head, body)
    elif type(exp) == Atom:
        terms = [subs(term, target_var, const) for term in exp.terms]
        return Atom(exp.pred, terms)
    elif type(exp) == FuncTerm:
        args = [subs(arg, target_var, const) for arg in exp.env]
        return FuncTerm(exp.func_symbol, args)
    elif type(exp) == Var:
        if exp.name == target_var.name:
            return const
        else:
            return exp
    elif type(exp) == Const:
        return exp
    else:
        assert 1 == 0, 'Unknown type in substitution: ' + str(exp)

def meta_subs(exp, target_var, const):
    if type(exp) == Clause:
        head = subs(exp.head, target_var, const)
        body = [subs(bi, target_var, const) for bi in exp.body]
        return Clause(head, body)
    elif type(exp) == Atom:
        terms = [subs(term, target_var, const) for term in exp.terms]
        return Atom(exp.pred, terms)
    elif type(exp) == FuncTerm:
        args = [subs(arg, target_var, const) for arg in exp.args]
        return FuncTerm(exp.func_symbol, args)
    elif type(exp) == Var:
        if exp.name == target_var.name:
            return const
        else:
            return exp
    elif type(exp) == Const:
        return exp
    elif type(exp) == MetaVar:
        if exp.name == target_var.name:
            return const
        else:
            return exp
    elif type(exp) == MetaConst:
        return exp
    elif type(exp) == MetaAtom:
        terms = [meta_subs(term, target_var, const) for term in exp.terms]
        return MetaAtom(exp.pred, terms)
    else:
        assert 1 == 0, 'Unknown type in substitution: ' + str(exp)



def subs_list(clause, theta_list):
    """
    perform list of substitutions

    Inputs
    ------
    clause : .logic.Clause
        target clause
    theta_list : List[(.logic.Var, .logic.Const)]
        list of substitute operations to be performed
    """
    result = clause
    for theta in theta_list:
        result = subs(result, theta[0], theta[1])
    return result

def meta_subs_list(exp, theta_list):
    if type(exp) == MetaRule:
        head = exp.head
        body = exp.body
        for target_var, const in theta_list:
            if const.dtype==DataType('atoms'):
                const_value_length = len(const.value)
            else:
                const_value_length = 0

            if '_' in target_var.name and const_value_length >1:
                assert target_var.name.count('_') == 1, (f"InvalidUnderscoreCountError: The string '{target_var.name}' must contain exactly one underscore, "
                                                         f"but it contains {target_var.name.count('_')}.")
                head = meta_subs(head, target_var, const)
                body_var1_name,body_var2_name  = target_var.name.split('_')
                body_var1 = MetaVar(body_var1_name)
                body_var2 = MetaVar(body_var2_name)
                const1 = MetaConst(const.value[:1],dtype=DataType('atoms'))
                const2 = MetaConst(const.value[1:],dtype=DataType('atoms'))
                body = [meta_subs(bi, body_var1, const1) for bi in body]
                body = [meta_subs(bi, body_var2, const2) for bi in body]
            elif '_' in target_var.name and const.dtype == DataType('proof'):
                head = meta_subs(head, target_var, const)
                body_var1_name, body_var2_name = target_var.name.split('_')
                body_var1 = MetaVar(body_var1_name)
                body_var2 = MetaVar(body_var2_name)
                const1 = MetaConst(const.value.tree[0], dtype=DataType('proof'))
                const2 = MetaConst(const.value.tree[1], dtype=DataType('proof'))
                body = [meta_subs(bi, body_var1, const1) for bi in body]
                body = [meta_subs(bi, body_var2, const2) for bi in body]
            else:
                head = meta_subs(head, target_var, const)
                body = [meta_subs(bi, target_var, const) for bi in body]
        return MetaRule(head, body)
    elif type(exp) == MetaAtom:
        terms = exp.terms
        for target_var, const in theta_list:
            terms = [meta_subs(term, target_var, const) for term in terms]
        return MetaAtom(exp.pred, terms)
    #elif type(exp) == FuncTerm:
    #    for target_var, const in theta_list:
    #        args = [subs(arg, target_var, const) for arg in exp.args]
    #    return FuncTerm(exp.func_symbol, args)
    #elif type(exp) == Var:
    #    if exp.name == target_var.name:
    #        return const
    #    else:
    #        return exp
    #elif type(exp) == Const:
    #    return exp
    else:
        assert 1 == 0, 'Unknown type in substitution: ' + str(exp)


def unify(atoms):
    """
    Unification of first-order logic expressions
    details in [Foundations of Inductive Logic Programming. Nienhuys-Cheng, S.-H. et.al. 1997.]

    Inputs
    ------
        atoms : List[.logic.Atom]
    Returns 
    -------
        flag : bool
            unifiable or not
        unifier : List[(.logic.Var, .logic.Const)]
            unifiable - unifier (list of substitutions)
            not unifiable - empty list
    """
    # empty set
    if len(atoms) == 0:
        return (1, [])
    # check predicates
    for i in range(len(atoms)-1):
        if atoms[i].pred != atoms[i+1].pred:
            return (0, [])

    # check all the same
    all_same_flag = True
    for i in range(len(atoms)-1):
        all_same_flag = all_same_flag and (atoms[i] == atoms[i+1])
    if all_same_flag:
        return (1, [])

    k = 0
    theta_list = []

    atoms_ = atoms
    while(True):
        # check terms from left
        for i in range(atoms_[0].pred.arity):
            # atom_1(term_1, ..., term_i, ...), ..., atom_j(term_1, ..., term_i, ...), ...
            terms_i = [atoms_[j].terms[i] for j in range(len(atoms_))]
            disagree_flag, disagree_set = get_disagreements(terms_i)
            if not disagree_flag:
                continue
            var_list = [x for x in disagree_set if type(x) == Var]
            if len(var_list) == 0:
                return (0, [])
            else:
                # substitute
                subs_var = var_list[0]
                # find term where the var does not occur
                subs_flag, subs_term = find_subs_term(
                    subs_var, disagree_set)
                if subs_flag:
                    k += 1
                    theta_list.append((subs_var, subs_term))
                    subs_flag = True
                    # UNIFICATION SUCCESS
                    atoms_ = [subs(atom, subs_var, subs_term)
                              for atom in atoms_]
                    if is_singleton(atoms_):
                        return (1, theta_list)
                else:
                    # UNIFICATION FAILED
                    return (0, [])


def meta_unify(meta_atoms):
    # empty set
    if len(meta_atoms) == 0:
        return (1, [])
    # check predicates
    for i in range(len(meta_atoms) - 1):
        if meta_atoms[i].pred != meta_atoms[i + 1].pred:
            return (0, [])

    # check all the same
    all_same_flag = True
    for i in range(len(meta_atoms) - 1):
        all_same_flag = all_same_flag and (meta_atoms[i] == meta_atoms[i + 1])
    if all_same_flag:
        return (1, [])

    k = 0
    theta_list = []

    atoms_ = meta_atoms
    while (True):
        # check terms from left
        for i in range(atoms_[0].pred.arity):
            # atom_1(term_1, ..., term_i, ...), ..., atom_j(term_1, ..., term_i, ...), ...
            terms_i = [atoms_[j].terms[i] for j in range(len(atoms_))]
            disagree_flag, disagree_set = get_disagreements(terms_i)
            if not disagree_flag:
                continue
            var_list = [x for x in disagree_set if type(x) == MetaVar]
            const_list = [y for y in disagree_set if type(y) == MetaConst]
            var_name = var_list[0].name
            if type(const_list[0].value) == list:
                const_value_length = len(const_list[0].value)
            else:
                const_value_length = 0

            if len(var_list) == 0:
                return (0, [])
            elif '_' in var_name and const_value_length == 1: return (0, [])
            elif '_' not in var_name and const_value_length > 1: return (0, [])
            else:
                # substitute
                subs_var = var_list[0]
                # find term where the var does not occur
                subs_flag, subs_term = find_subs_term(
                    subs_var, disagree_set)
                if subs_flag:
                    k += 1
                    theta_list.append((subs_var, subs_term))
                    subs_flag = True
                    # UNIFICATION SUCCESS
                    atoms_ = [meta_subs(atom, subs_var, subs_term)
                              for atom in atoms_]
                    if is_singleton(atoms_):
                        return (1, theta_list)
                else:
                    # UNIFICATION FAILED
                    return (0, [])


def get_disagreements(terms):
    """
    get desagreements in the unification algorithm
    details in [Foundations of Inductive Logic Programming. Nienhuys-Cheng, S.-H. et.al. 1997.]

    Inputs
    ------
    temrs : List[Term]
        Term : .logic.FuncTerm .logic.Const .logic.Var
        list of terms        

    Returns
    -------
    disagree_flag : bool
        flag of disagreement
    disagree_terms : List[Term]
        Term : .logic.FuncTerm .logic.Const .logic.Var    
        terms of disagreement
    """
    disagree_flag, disagree_index = get_disagree_index(terms)
    if disagree_flag:
        disagree_terms = [term.get_ith_term(
            disagree_index) for term in terms]
        return disagree_flag, disagree_terms
    else:
        return disagree_flag, []


def get_disagree_index(terms):
    """
    get the desagreement index in the unification algorithm
    details in [Foundations of Inductive Logic Programming. Nienhuys-Cheng, S.-H. et.al. 1997.]

    Inputs
    ------
    terms : List[Term]
        Term : .logic.FuncTerm .logic.Const .logic.Var
        list of terms        

    Returns
    -------
    disagree_flag : bool
        flag of disagreement
    disagree_index : int
        index of the disagreement term in the args of predicates
    """
    symbols_list = [term.to_list() for term in terms]
    n = min([len(symbols) for symbols in symbols_list])
    for i in range(n):
        ith_symbols = [symbols[i] for symbols in symbols_list]
        for j in range(len(ith_symbols)-1):
            if ith_symbols[j] != ith_symbols[j+1]:
                return (True, i)
    # all the same terms
    return (False, 0)


def occur_check(variable, term):
    """
    occur check function
    details in [Foundations of Inductive Logic Programming. Nienhuys-Cheng, S.-H. et.al. 1997.]

    Inputs
    ------
    variable : .logic.Var
    term : Term
        Term : .logic.FuncTerm .logic.Const .logic.Var

    Returns
    -------
    occur_flag : bool
        flag ofthe occurance of the variable
    """
    if type(term) == Const:
        return False
    elif type(term) == Var:
        return variable.name == term.name
    elif type(term) == MetaConst:
        return False
    elif type(term) == MetaVar:
        return variable.name == term.name
    else:
        # func term case
        for arg in term.env:
            if occur_check(variable, arg):
                return True
        return False


def find_subs_term(subs_var, disagree_set):
    """
    Find term where the var does not occur

    Inputs
    ------
    subs_var : .logic.Var
    disagree_set : List[.logic.Term]

    Returns
    -------
    flag : bool
    term : .logic.Term
    """
    for term in disagree_set:
        if not occur_check(subs_var, term):
            return True, term
    return False, Term()


def is_singleton(atoms):
    """
    returns whether all the input atoms are the same or not

    Inputs
    ------
     atoms: List[.logic.Atom]
        [a_1, a_2, ..., a_n]

    Returns
    -------
    flag : bool
        a_1 == a_2 == ... == a_n
    """
    result = True
    for i in range(len(atoms)-1):
        result = result and (atoms[i] == atoms[i+1])
    return result


def is_entailed(e, clause, facts, n):
    """
    decision function of ground atom is entailed by a clause and facts by n-step inference

    Inputs
    ------
     e : .logic.Atom
        ground atom
    clause : .logic.Clause
        clause
    facts : List[.logic.Atom]
        set of facts
    n : int
        infer step 

    Returns
    -------
    flag : bool
        ${clause} \cup facts \models e$
    """
    if len(clause.body) == 0:
        flag, thetas = unify([e, clause.head])
        return flag
    if len(clause.body) == 1:
        return e in t_p_n(clause, facts, n)


def t_p_n(clause, facts, n):
    """
    applying the T_p operator n-times taking union of results 

    Inputs
    ------
    clause : .logic.Clause
        clause
    facts : List[.logic.Atom]
        set of facts
    n : int
        infer step 

    Returns
    -------
    G : Set[.logic.Atom]
        set of ground atoms entailed by ${clause} \cup facts$
    """
    G = set(facts)
    for i in range(n):
        G = G.union(t_p(clause, G))
    return G


def t_p(clause, facts):
    """
    T_p operator
    limited to clauses with one body atom

    Inputs
    ------
    clause : .logic.Clause
        clause
    facts : List[.logic.Atom]
        set of facts

    Returns
    -------
    S : List[.logic.Atom]
        set of ground atoms entailed by one step forward-chaining inference
    """
    # |body| == 1
    S = []
    unify_dic = {}
    for fact in facts:
        flag, thetas = unify([clause.body[0], fact])
        if flag:
            head_fact = subs_list(clause.head, thetas)
            S = S + [head_fact]
    return list(set(S))
