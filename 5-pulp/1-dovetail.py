# Import PuLP modeler functions
import pulp

# max z = 3x1 + 2x2
# st
#      x1 + x2 <=  9
#     3x1 + x2 <= 18
#      x1      <=  7
#           x2 <=  6
# x1, x2 >= 0

# -------------
def solver_1():
    # A LP problem
    prob = pulp.LpProblem("dovetail", pulp.LpMaximize)

    # Variables
    # 0 <= x1 <= 7
    x1 = pulp.LpVariable("x1", lowBound=0, upBound=7,
                cat=pulp.const.LpContinuous)
    # 0 <= x2 <= 6
    x2 = pulp.LpVariable("x2", 0, 6)

    # Use None for +/- Infinity,
    # i.e. z <= 0 -> LpVariable("z", None, 0)

    # Objective
    prob += 3*x1 + 2*x2, "obj"
    # (the name at the end is facultative)

    # Constraints
    prob += x1 + x2 <= 9,  "c1"
    prob += 3*x1 + x2 <= 18, "c2"
    # (the names at the end are facultative)

    # Write the problem as an LP or MPS file
    # prob.writeLP('dovetail.lp')
    # prob.writeLP('dovetail.mps')

    # see the full definition of this model
    # print(prob)

    # solve the problem using the default solver
    prob.solve()

    # print the status of the solved LP
    print("Status:", pulp.LpStatus[prob.status])

    # print the value of the objective
    print("objective =", pulp.value(prob.objective))

    # print the value of the variables at the optimum
    for v in prob.variables():
        print(f'{v.name} = {v.varValue:5.2f}')

    print()
    print("Sensitivity Analysis\nConstraint\t\tShadow Price\tSlack")
    for name, c in prob.constraints.items():
        print(f'{name} : {c} \t{c.pi} \t\t{c.slack}')

# ------------
def solver_2():
    c = (3, 2)

    A = [   ( 1, 1),
            ( 3, 1),
            ( 1, 0),
            ( 0, 1) ]

    b = (9, 18, 7, 6)

    m, n = len(A), len(A[0])

    # A LP problem
    prob = pulp.LpProblem("dovetail", pulp.LpMaximize)

    # Variables
    x =  pulp.LpVariable.dicts('x', range(n))
    # x =  pulp.LpVariable.dicts('x', range(n), cat=pulp.LpInteger)

    # Objective
    prob += pulp.lpSum([c[i]* x[i] for i in range(n)]), 'obj'
    # (the name at the end is facultative)

    # Constraints
    for i in range(m):
        prob += pulp.lpSum([A[i][j] * x[j] for j in range(n)]) <= b[i], f'constraint_{i}'

    # solve the problem using the default solver
    prob.solve()

    # print the status of the solved LP
    print("Status:", pulp.LpStatus[prob.status])

    # print the value of the objective
    print("objective =", pulp.value(prob.objective))

    # print the value of the variables at the optimum
    for v in prob.variables():
        print(f'{v.name} = {v.varValue:5.2f}')


if __name__=='__main__':
    solver_1()
    # solver_2()
