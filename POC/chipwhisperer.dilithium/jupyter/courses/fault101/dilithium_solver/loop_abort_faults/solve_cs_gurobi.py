
import gurobipy as gp
from gurobipy import GRB

possible_values = [-2, -1, 0, 1, 2]

def distance_to_zero_dot_five(x):
    return abs(0.5-abs((x-int(x))))

DISTANCE_FIX = 0.49
INDICATOR_CONSTRAINT = False
INDICATOR_VARIABLES = False

def solve_min_number_rows_indicator_variables(LHS_matrix, b_matrix, start_s = None, shat_not_rounded = None, actual_s = None):
    A = LHS_matrix
    M = len(A)
    N = len(A[0])
    # Create a new model
    model = gp.Model("mip1")
    model.setParam("FeasibilityTol", 1e-2)
    #model.setParam("NonConvex", 2)
    x_variables = []
    s_variables = []
    sum_cs_variables  = []
    aux_s_variables = []

    x_variables = model.addVars(M, vtype=GRB.BINARY, name=f"x_variables")
    model.update()

    for m in range(M):
        #x_variables.append(model.addVar(vtype=GRB.BINARY, name=f"x_{m}"))
        pass#sum_cs_variables.append(model.addVar(vtype=GRB.INTEGER, name=f"sum_cs_{m}", lb=-500,ub=500))
    if INDICATOR_VARIABLES:
        for n in range(N):
            aux_s_variables.append(model.addVars(possible_values, vtype=GRB.BINARY, name=f"s_variables_{n}"))
    else:
        s_variables = model.addVars(N, vtype=GRB.INTEGER, name=f"s_variables", lb=-2, ub=2)
    model.update()
    obj = gp.LinExpr()
    for m in range(M):
        obj += x_variables[m]
    model.setObjective(obj, GRB.MAXIMIZE)
    if INDICATOR_VARIABLES:
        for n in range(N):
            model.addConstr(aux_s_variables[n].sum() == 1)
    fixed_s_values = set()
    if shat_not_rounded is not None:
        for n in range(len(shat_not_rounded)):
            if distance_to_zero_dot_five(shat_not_rounded[n])>=DISTANCE_FIX:
                fixed_s_values.add(n)
                if actual_s:
                    print("actual s len", len(actual_s))
                    print("start_s len", len(start_s))
                    if start_s[n] != actual_s[n]:
                        raise Exception("Start s != actual_s although DISTANCE_FIX")
    for m in range(M):
        constraint_lhs = gp.LinExpr()
        if INDICATOR_VARIABLES:
            constraint_rhs = gp.LinExpr()
            for n in range(N):
                if n not in fixed_s_values:
                    for j in range(len(possible_values)):
                        constraint_lhs += aux_s_variables[n][possible_values[j]] * ((possible_values[j]*A[m][n]))
                else:
                    constraint_lhs += start_s[n]*A[m][n]
        else:
            for n in range(N):
                if n not in fixed_s_values:
                    constraint_lhs += s_variables[n] * A[m][n]
                else:
                    constraint_lhs += start_s[n]*A[m][n]
        #constraint_rhs += sum_cs_variables[m]
        #model.addConstr(constraint_lhs, sense=GRB.EQUAL,rhs=constraint_rhs)
        if INDICATOR_CONSTRAINT:
            model.addGenConstrIndicator(x_variables[m], 1, constraint_lhs, GRB.EQUAL, b_matrix[m])
        else:
            #Use big-M
            #b-c_i s \leq M*(1-x)
            #b-c_i s \geq -M*(1-x)
            M = 200
            #print("b_matrix[m]", b_matrix[m])
            model.addConstr(b_matrix[m]-constraint_lhs <= M*(1-x_variables[m]))
            model.addConstr(b_matrix[m]-constraint_lhs >= -M*(1-x_variables[m]))
    #for m in range(M):
       # model.addQConstr(((1-x_variables[m])*(b_matrix[m]-sum_cs_variables[m])), sense=GRB.EQUAL, rhs=0,name= f"qc_{m}")
        #model.addQConstr(((x_variables[m])*(b_matrix[m]-sum_cs_variables[m])), sense=GRB.EQUAL, rhs=0,name= f"qc_{m}")
        #x_variables[m] == 1 -> sum_cs_variables[m] == b_matrix[m]
        #print("b_matrix[m]",b_matrix[m])
    #    model.addGenConstrIndicator(x_variables[m], 1, sum_cs_variables[m], GRB.EQUAL, b_matrix[m]
    #m_sum_limit = gp.LinExpr()
    #for m in range(M):
    #    m_sum_limit += x_variables[m]
    #model.addConstr(m_sum_limit >= int(len(x_variables)*0.8))
    #m_sum_limit = gp.LinExpr()
    #for m in range(M):
    #    m_sum_limit += x_variables[m]
    #model.addConstr(m_sum_limit <= int(len(x_variables)*0.1))
    #print(model)
    model.params.BestObjStop = int(len(x_variables)*0.76)
    if start_s is not None: 
        for n in range(len(start_s)):
            if INDICATOR_VARIABLES:
                for j in range(len(possible_values)):
                    if start_s[n] == possible_values[j]:
                        aux_s_variables[n][possible_values[j]].start = 1
                    else:
                        aux_s_variables[n][possible_values[j]].start = 0
            else:
                s_variables[n].start = start_s[n]
        for m in range(len(A)):
            res = 0
            for n in range(len(A[0])):
                res += A[m][n]*start_s[n]
            #sum_cs_variables[m].start = res
            if res != b_matrix[m]:
                x_variables[m].start = 0
            else:
                x_variables[m].start = 1
            #print("x", m, " ",res == b_matrix[m])
    if shat_not_rounded is not None:
        for n in range(len(shat_not_rounded)):
            if distance_to_zero_dot_five(shat_not_rounded[n])>=DISTANCE_FIX:
                print("Fix ",n," to be ", start_s[n], "because non rounded", shat_not_rounded[n])
                if INDICATOR_VARIABLES:
                    for j in range(len(possible_values)):
                        if possible_values[j] == start_s[n]:
                            aux_s_variables[n][possible_values[j]].lb = 1
                            aux_s_variables[n][possible_values[j]].ub = 1
                        else:
                            aux_s_variables[n][possible_values[j]].lb = 0
                            aux_s_variables[n][possible_values[j]].ub = 0
                else:
                    s_variables[n].lb = start_s[n]
                    s_variables[n].ub = start_s[n]
            else:
                if INDICATOR_VARIABLES:
                    for j in range(len(possible_values)):
                        #TODO: This is wrong, because it does not account for negative values (-1.03 should be -1 or -2), fix that!
                        if possible_values[j] == int(shat_not_rounded[n]) or possible_values[j] == int(shat_not_rounded[n])+1:
                            aux_s_variables[n][possible_values[j]].lb = 1
                            aux_s_variables[n][possible_values[j]].ub = 0
                        else:
                            aux_s_variables[n][possible_values[j]].lb = 0
                            aux_s_variables[n][possible_values[j]].ub = 0
                else:
                    if shat_not_rounded[n] > 0:
                        s_variables[n].lb = int(shat_not_rounded[n])
                        s_variables[n].ub = int(shat_not_rounded[n])+1
                    else:
                        s_variables[n].lb = int(shat_not_rounded[n])-1
                        s_variables[n].ub = int(shat_not_rounded[n])

    model.update()
    #print (model.display())
    model.optimize()
    #print (model.display())
    print(obj.getValue())
    #print(model.getGenConstrIndicator())
    #for v in model.getVars():
    #    print('%s %g' % (v.varName, v.x))
    #print("b_matrix", b_matrix)
    #print("A matrix", LHS_matrix)
    ret_s = []
    for n in range(N):
        if INDICATOR_VARIABLES:
            for j in range(len(possible_values)):
                if aux_s_variables[n][possible_values[j]].x > 0.5:
                    #print(f"s{n}", possible_values[j])
                    ret_s.append(possible_values[j])
        else:
            ret_s.append(s_variables[n].x)
    return ret_s


def solve_max_rows_simplified(LHS_matrix, b_matrix,  actual_s = None):
    A = LHS_matrix
    M = len(A)
    N = len(A[0])
    # Create a new model
    model = gp.Model("mip1")
    model.setParam("FeasibilityTol", 1e-2)
    #model.setParam("NonConvex", 2)
    x_variables = []
    s_variables = []
    sum_cs_variables  = []
    aux_s_variables = []

    x_variables = model.addVars(M, vtype=GRB.BINARY, name=f"x_variables")
    model.update()

    for m in range(M):
        #x_variables.append(model.addVar(vtype=GRB.BINARY, name=f"x_{m}"))
        pass#sum_cs_variables.append(model.addVar(vtype=GRB.INTEGER, name=f"sum_cs_{m}", lb=-500,ub=500))
    s_variables = model.addVars(N, vtype=GRB.INTEGER, name=f"s_variables", lb=-2, ub=2)
    model.update()
    obj = gp.LinExpr()
    for m in range(M):
        obj += x_variables[m]
    model.setObjective(obj, GRB.MAXIMIZE)
    for m in range(M):
        constraint_lhs = gp.LinExpr()
        for n in range(N):
            constraint_lhs += s_variables[n]*A[m][n]

        #Use big-M
        #b-c_i s \leq M*(1-x)
        #b-c_i s \geq -M*(1-x)
        M = 200
        #print("b_matrix[m]", b_matrix[m])
        model.addConstr(b_matrix[m]-constraint_lhs <= M*(1-x_variables[m]))
        model.addConstr(b_matrix[m]-constraint_lhs >= -M*(1-x_variables[m]))
    model.params.BestObjStop = int(len(x_variables)*0.76)
    
    model.update()
    #print (model.display())
    model.optimize()
    #print (model.display())
    print(obj.getValue())
    #print(model.getGenConstrIndicator())
    #for v in model.getVars():
    #    print('%s %g' % (v.varName, v.x))
    #print("b_matrix", b_matrix)
    #print("A matrix", LHS_matrix)
    ret_s = []
    for n in range(N):
       
        ret_s.append(s_variables[n].x)
    return ret_s
