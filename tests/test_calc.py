import unittest

from src.agent.calc import compute, format_number


class ComputeTests(unittest.TestCase):
    def test_basic_operations(self):
        self.assertEqual(compute(2, 3, "add"), 5)
        self.assertEqual(compute(10, 4, "subtract"), 6)
        self.assertEqual(compute(6, 7, "multiply"), 42)
        self.assertEqual(compute(9, 3, "divide"), 3)
        self.assertEqual(compute(2, 10, "power"), 1024)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            compute(1, 0, "divide")

    def test_unknown_operation(self):
        with self.assertRaises(ValueError) as ctx:
            compute(1, 2, "modulo")
        self.assertIn("modulo", str(ctx.exception))

    def test_power_guards(self):
        with self.assertRaises(ValueError):
            compute(2, 5000, "power")
        with self.assertRaises(ValueError):
            compute(0, -1, "power")  # zero to a negative power
        with self.assertRaises(ValueError):
            compute(-8, 0.5, "power")  # would be a complex number


class FormatNumberTests(unittest.TestCase):
    def test_whole_numbers_have_no_decimal(self):
        self.assertEqual(format_number(7006652.0), "7006652")

    def test_decimals_are_trimmed(self):
        self.assertEqual(format_number(1 / 3), "0.3333333333")

    def test_negative_and_zero(self):
        self.assertEqual(format_number(-4.0), "-4")
        self.assertEqual(format_number(0.0), "0")


if __name__ == "__main__":
    unittest.main()
