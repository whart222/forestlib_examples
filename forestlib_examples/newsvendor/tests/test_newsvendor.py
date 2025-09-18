import pytest
import pyomo.environ as pyo
from forestlib.sp import stochastic_program
import forestlib_examples.newsvendor as newsvendor


class TestNewsVendor:
    """
    Test the news vendor application

    See https://stoprog.org/sites/default/files/SPTutorial/TutorialSP.pdf
    """

    def test_single_builder(self):
        sp = newsvendor.newsvendor_sp()

        assert sp.get_objective_coef(0) == 0

        assert set(sp.bundles.keys()) == {"1", "2", "3", "4", "5"}
        assert sp.bundles["1"].probability == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"1", "2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        sp.solve(M1, solver="glpk")
        assert pyo.value(M1.s[None, 1].x) == 15.0

        sp.solve(M2, solver="glpk")
        assert pyo.value(M2.s[None, 2].x) == 60.0
