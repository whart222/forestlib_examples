import pyomo.environ as pyo
from forestlib.sp import stochastic_program


#
# Data for a simple newsvendor example
#
model_data = {
    "data": {"c": 1.0, "b": 1.5, "h": 0.1},
    "scenarios": [
        {"ID": 1, "d": 15},
        {"ID": 2, "d": 60},
        {"ID": 3, "d": 72},
        {"ID": 4, "d": 78},
        {"ID": 5, "d": 82},
    ],
}


#
# Function that constructs a newsvendor model
# including a single second stage
#
def model_builder(data, args):
    b = data["b"]
    c = data["c"]
    h = data["h"]
    d = data["d"]

    M = pyo.ConcreteModel(data["ID"])

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.o = pyo.Objective(expr=M.y)

    return M


#
# Function that constructs the first stage of a
# newsvendor model
#
def first_stage(M, data, args):
    M.x = pyo.Var(within=pyo.NonNegativeReals)


#
# Function that constructs the second stage of a
# newsvendor model
#
def second_stage(M, S, data, args):
    b = data["b"]
    c = data["c"]
    h = data["h"]
    d = data["d"]

    S.y = pyo.Var()

    S.o = pyo.Objective(expr=S.y)
    S.greater = pyo.Constraint(expr=S.y >= (c - b) * M.x + b * d)
    S.less = pyo.Constraint(expr=S.y >= (c + h) * M.x - h * d)


def newsvendor_sp():
    """
    A simple news vendor application

    See https://stoprog.org/sites/default/files/SPTutorial/TutorialSP.pdf
    """
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_model(model_data=model_data, model_builder=model_builder)
    return sp
