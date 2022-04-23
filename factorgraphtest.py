import unittest
from factorgraph import VariableNode, FactorNode, FactorGraph


class TestVariableNode(unittest.TestCase):
    def test_str_unfixed(self):
        var = VariableNode("test")
        var.val = 0
        self.assertEqual(
            str(var), "VariableNode(name=test, val=0, fixed=False, prob=None)"
        )

    def test_str_fixed(self):
        var = VariableNode("test")
        var.val = 0
        var.fix()
        self.assertEqual(
            str(var), "VariableNode(name=test, val=0, fixed=True, prob=None)"
        )


class TestFactorNode(unittest.TestCase):
    def test_formula_simple(self):
        var_a = VariableNode("a")
        fac = FactorNode("f", "0.95 if V[a] else 0.05")
        var_a.val = True
        self.assertEqual(fac.calc_prob([var_a], None), 0.95)

    def test_formula_multiple(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        fac = FactorNode("f", "0.95 if (V[a] and V[a]) and V[b] else 0.05")
        var_a.val = True
        var_b.val = False
        self.assertEqual(fac.calc_prob([var_a, var_b], None), 0.05)


class TestFactorGraph(unittest.TestCase):
    def test_init(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        fac = FactorNode("f", "0.95 if V[a] and V[b] else 0.05")
        fg = FactorGraph([var_a, var_b], [fac])
        self.assertTrue(True)

    def test_exhaustive_inference(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        fac = FactorNode("f", "0.95 if V[a] and not V[b] else 0.05")
        fg = FactorGraph([var_a, var_b], [fac])
        fg.inference_exhaustive()
        self.assertEqual(var_a.val, True)
        self.assertEqual(var_b.val, False)
        self.assertEqual(fg.best_prob, 0.95)

    def test_varfix(self):
        var_a = VariableNode("a")
        var_a.val = False
        var_a.fix()
        fac = FactorNode("f", "0.95 if V[a] else 0.05")
        fg = FactorGraph([var_a], [fac])
        fg.inference_exhaustive()
        self.assertEqual(var_a.val, False)
        self.assertEqual(fg.best_prob, 0.05)

    def test_marginal_inference_exhaustive(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        fac = FactorNode("f", "0.95 if V[a] and not V[b] else 0.05")
        fg = FactorGraph([var_a, var_b], [fac])
        fg.marginal_inference_exhaustive()
        self.assertEqual(
            var_a.prob, (0.95 + 0.05) / (0.95 + 0.05 + 0.05 + 0.05)
        )
        self.assertEqual(
            var_b.prob, (0.05 + 0.05) / (0.95 + 0.05 + 0.05 + 0.05)
        )

    def test_marginal_sum_product(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        fac = FactorNode("f", "0.95 if V[a] and not V[b] else 0.05")
        fg = FactorGraph([var_a, var_b], [fac])
        fg.marginal_sum_product()
        self.assertEqual(
            var_a.prob, (0.95 + 0.05) / (0.95 + 0.05 + 0.05 + 0.05)
        )
        self.assertEqual(
            var_b.prob, (0.05 + 0.05) / (0.95 + 0.05 + 0.05 + 0.05)
        )

    def test_marginal_sum_product(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        fac = FactorNode("f", "0.95 if V[a] and not V[b] else 0.05")
        fg = FactorGraph([var_a, var_b], [fac])
        var_a.val = 0
        var_a.fix()
        fg.marginal_sum_product()
        self.assertEqual(var_b.prob, 0.5)


    def test_marginal_sum_product_2(self):
        var_a = VariableNode("a")
        var_b = VariableNode("b")
        var_c = VariableNode("c")
        fac1 = FactorNode("f1", "0.95 if V[a] or V[b] else 0.05")
        fac2 = FactorNode("f2", "0.95 if V[b] or V[c] else 0.05")
        fac3 = FactorNode("f3", "1 / (V[a] + V[b] + V[c] + 1)")
        fg = FactorGraph([var_a, var_b, var_c], [fac1, fac2, fac3])
        fg.marginal_sum_product()
        self.assertEqual(var_a.prob, 0.5)


if __name__ == "__main__":
    unittest.main()
