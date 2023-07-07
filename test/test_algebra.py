import cmath

import pytest

from pauli_algebra.algebra import (
    CompositePauliElement,
    add_pauli_algebra,
    make_pauli_algebra_element,
    multiply_pauli_algebra,
    multiply_pauli_algebra_by_scalar,
    pauli_algebra_to_string,
)
from pauli_algebra.matrices import Pauli_I, Pauli_X, Pauli_Y, Pauli_Z
from pauli_algebra.utils import (
    generate_pauli_element_from_strings_and_scalars,
    generate_pauli_element_from_tuples,
)


class TestCompositePauliElement:
    pauli_types = {"I": Pauli_I, "X": Pauli_X, "Y": Pauli_Y, "Z": Pauli_Z}

    def test_composite_pauli_element_constructor(self):
        """Test constructor from list of Pauli matrices."""
        ref_string = "XYZZI"
        element = CompositePauliElement(
            [Pauli_X(), Pauli_Y(), Pauli_Z(), Pauli_Z(), Pauli_I()]
        )
        assert element.num_qubits == len(ref_string)
        for symbol, pauli in zip(ref_string, element.pauli_matrices):
            assert isinstance(pauli, self.pauli_types[symbol])
        assert str(element) == ref_string

    @pytest.mark.parametrize("string", ["XYZZI", "XYZI", "X", "ZY"])
    def test_composite_pauli_element_from_string(self, string: str):
        """Test pauli element generation from strings."""
        element = CompositePauliElement.from_string(string)
        assert element.num_qubits == len(string)
        for symbol, pauli in zip(string, element.pauli_matrices):
            assert isinstance(pauli, self.pauli_types[symbol])
        assert str(element) == string

    @pytest.mark.parametrize("string", ["XYZZI", "XYZI", "X", "ZY"])
    def test_composite_pauli_element_equality(self, string):
        """Test equality for distinct composite pauli elements."""
        element1 = CompositePauliElement.from_string(string)
        element2 = CompositePauliElement.from_string(string)
        # distinct object instances
        assert element1 is not element2
        assert element1 == element2

    def test_composite_pauli_element_equality_from_constructor_and_string(self):
        """Test equality for distinct composite pauli elements."""
        element_from_constructor = CompositePauliElement(
            [Pauli_X(), Pauli_Y(), Pauli_Z(), Pauli_Z(), Pauli_I()]
        )
        element_from_string = CompositePauliElement.from_string("XYZZI")
        # distinct object instances
        assert element_from_constructor is not element_from_string
        assert element_from_constructor == element_from_string

    def test_composite_pauli_element_hashing(self):
        """Test equality when distinct composite pauli elements are hashed."""
        element_from_constructor = CompositePauliElement(
            [Pauli_X(), Pauli_Y(), Pauli_Z(), Pauli_Z(), Pauli_I()]
        )
        element_from_string = CompositePauliElement.from_string("XYZZI")
        # distinct object instances
        assert element_from_constructor is not element_from_string
        assert hash(element_from_constructor) == hash(element_from_string)
        assert len(set([element_from_constructor, element_from_string])) == 1


class TestPauliAlgebra:
    def test_make_pauli_algebra_element(self):
        string = "XYZZI"
        pauli_element = make_pauli_algebra_element(string)
        assert str(pauli_element) == string
        assert pauli_element.num_qubits == len(string)
        # ensure that the element is scaled by 1 by default
        assert (
            pauli_element._elements_to_scalars[
                CompositePauliElement.from_string(string)
            ]
            == 1
        )

    def test_make_pauli_algebra_element_with_scalar(self):
        string = "XYZZI"
        scalar = 1 + 4j
        pauli_element = make_pauli_algebra_element(string, scalar=scalar)
        assert str(pauli_element) == "(1+4i)" + string
        assert pauli_element.num_qubits == len(string)
        # ensure that the element is scaled by 1 by default
        assert cmath.isclose(
            pauli_element._elements_to_scalars[
                CompositePauliElement.from_string(string)
            ],
            scalar,
        )

    @pytest.mark.parametrize(
        "strings, scalars",
        [
            (["XYZZI", "IZZYX"], [1, -1]),
            (["XYZI", "IZYX"], [-1, 4j]),
            (["ZI", "YX"], [1 - 3j, -1 + 1j]),
            (["I", "X"], [-3, -8 + 3j]),
        ],
    )
    def test_pauli_algebra_with_multiple_elements(self, strings, scalars):
        strings_to_scalars = {
            string: scalar for string, scalar in zip(strings, scalars)
        }
        pauli_element = generate_pauli_element_from_strings_and_scalars(
            strings, scalars
        )

        # check dimension is correct
        assert pauli_element.num_qubits == len(strings[0])
        # check same number of terms
        assert len(pauli_element._elements_to_scalars) == len(strings)
        # check terms have same complex coefficient
        for composite_pauli in pauli_element._elements_to_scalars:
            assert cmath.isclose(
                pauli_element._elements_to_scalars[composite_pauli],
                strings_to_scalars[str(composite_pauli)],
            )

    @pytest.mark.parametrize(
        "strings, scalars, ref_string",
        [
            (["XYZZI", "IZZYX"], [1, -1], "XYZZI + -IZZYX"),
            (["XYZI", "IZYX"], [-1, 4j], "-XYZI + 4iIZYX"),
            (["ZI", "YX"], [1 - 3j, -1 + 1j], "(1-3i)ZI + (-1+1i)YX"),
            (["I", "X"], [-3, -8 + 3j], "-3I + (-8+3i)X"),
        ],
    )
    def test_pauli_algebra_strings(self, strings, scalars, ref_string):
        """
        Note that this test may fail because of potential precision issues
        with floats, e.g., 2i == (0+2i) but would fail a string comparison.
        I could hammer this out with more time but I will not do it here.
        """
        pauli_element = generate_pauli_element_from_strings_and_scalars(
            strings, scalars
        )
        # check dimension is correct
        assert pauli_element.num_qubits == len(strings[0])
        # check string
        assert pauli_algebra_to_string(pauli_element) == ref_string

    @pytest.mark.parametrize(
        "strings, scalars, ref",
        [
            (["XYZZI", "XYZZI"], [1, 1], {"XYZZI": 2}),
            (["XYZZI", "XYZZI"], [1, -2], {"XYZZI": -1}),
            (["XYZZI", "XYZZI"], [-1, 1j], {"XYZZI": (-1 + 1j)}),
            (["XYZZI", "XYZZI"], [1, -1j], {"XYZZI": (1 - 1j)}),
            (["XYZZI", "XYZZI"], [-2j, 1j], {"XYZZI": -1j}),
            (["XYZZI", "XYZZI", "XYYZI"], [-2j, 1j, 1], {"XYZZI": -1j, "XYYZI": 1}),
            (
                ["XYZZI", "XYZYI", "XYYZI"],
                [-2j, 1j, 1],
                {"XYZZI": -2j, "XYZYI": 1j, "XYYZI": 1},
            ),
            (["XYZZI", "XYZZI"], [1, -1], {}),  # edge case
            (["XYZZI", "XYZZI", "XYZZZ"], [1, -1, 1], {"XYZZZ": 1}),  # edge case
        ],
    )
    def test_simplify(self, strings, scalars, ref):
        pauli_element = generate_pauli_element_from_strings_and_scalars(
            strings, scalars
        )
        # check dimension is correct
        assert pauli_element.num_qubits == len(strings[0])
        # check same number of terms
        assert len(pauli_element.scaled_elements) == len(ref)
        # check terms have same complex coefficient
        for composite_pauli in pauli_element._elements_to_scalars:
            assert cmath.isclose(
                pauli_element._elements_to_scalars[composite_pauli],
                ref[str(composite_pauli)],
            )

    @pytest.mark.parametrize(
        "strings, scalars, multiply_by",
        [
            (["XYZZI", "IZZYX"], [1, -1], 2),
            (["XYZI", "IZYX"], [-1, 4j], 1 - 4j),
            (["XYYY"], [-8 + 3j], -8 - 3j),
        ],
    )
    def test_multiply_algebra_by_scalar(self, strings, scalars, multiply_by):
        """
        Test that multiplication by a scalar results in the
        expected scaled coefficients.
        """
        pauli_element = generate_pauli_element_from_strings_and_scalars(
            strings, scalars
        )
        ref_final_scalars = {
            string: scalar * multiply_by for string, scalar in zip(strings, scalars)
        }
        scaled_pauli_element = multiply_pauli_algebra_by_scalar(
            multiply_by,
            pauli_element,
        )
        # check dimension is correct
        assert scaled_pauli_element.num_qubits == len(strings[0])
        # check same number of terms
        assert len(scaled_pauli_element._elements_to_scalars) == len(strings)
        # check terms have same complex coefficient
        for composite_pauli in scaled_pauli_element._elements_to_scalars:
            assert cmath.isclose(
                scaled_pauli_element._elements_to_scalars[composite_pauli],
                ref_final_scalars[str(composite_pauli)],
            )

    @pytest.mark.parametrize(
        "params1, params2, ref_final_scalars",
        [
            ([("XX", 1)], [("XY", -2j)], {"XX": 1, "XY": -2j}),  # from prompt
            ([("XYZZI", 1)], [("IZZYX", -1)], {"XYZZI": 1, "IZZYX": -1}),
            ([("XYZZI", 1)], [("IZZYX", -1j)], {"XYZZI": 1, "IZZYX": -1j}),
            ([("XZI", 1)], [("ZYX", (1 + 2j))], {"XZI": 1, "ZYX": (1 + 2j)}),
            ([("XZI", 1j)], [("ZYX", -2j)], {"XZI": 1j, "ZYX": -2j}),
            (
                [("XZI", 1j), ("XYZ", -9)],
                [("ZYX", -2j)],
                {"XZI": 1j, "XYZ": -9, "ZYX": -2j},
            ),
        ],
    )
    def test_add_pauli_algebra(self, params1, params2, ref_final_scalars):
        """
        Test that addition is properly represented for
        expressions that will not require simplification.
        """
        pauli1 = generate_pauli_element_from_tuples(params1)
        pauli2 = generate_pauli_element_from_tuples(params2)
        pauli_sum = add_pauli_algebra(pauli1, pauli2)

        # ensure the number of terms matches
        assert len(pauli_sum.scaled_elements) == len(ref_final_scalars)

        # verify coefficients
        for composite_pauli in pauli_sum._elements_to_scalars:
            assert cmath.isclose(
                pauli_sum._elements_to_scalars[composite_pauli],
                ref_final_scalars[str(composite_pauli)],
            )

    @pytest.mark.parametrize(
        "params1, params2, ref_final_scalars",
        [
            ([("XZ", 5j)], [("XZ", -3j)], {"XZ": 2j}),  # from prompt
            ([("XYZZI", 1)], [("XYZZI", 2)], {"XYZZI": 3}),
            ([("XYZZI", 1)], [("XYZZI", 2j)], {"XYZZI": (1 + 2j)}),
            ([("XYZZI", 1j)], [("XYZZI", -2j)], {"XYZZI": -1j}),
            (
                [("XYZZI", 1j), ("XYZYX", -11)],
                [("XYZZI", -2j)],
                {"XYZZI": -1j, "XYZYX": -11},
            ),
            (
                [("XYZZI", 1j), ("XYZYX", -11)],
                [("XYZZI", -1j)],
                {"XYZYX": -11},
            ),  # edge case
        ],
    )
    def test_add_pauli_algebra_with_simpliciation(
        self, params1, params2, ref_final_scalars
    ):
        """
        Test that addition is properly represented for
        expressions that are simplified automatically.
        """
        pauli1 = generate_pauli_element_from_tuples(params1)
        pauli2 = generate_pauli_element_from_tuples(params2)
        pauli_sum = add_pauli_algebra(pauli1, pauli2)

        # ensure the number of terms matches
        assert len(pauli_sum.scaled_elements) == len(ref_final_scalars)

        # verify coefficients
        for composite_pauli in pauli_sum._elements_to_scalars:
            assert cmath.isclose(
                pauli_sum._elements_to_scalars[composite_pauli],
                ref_final_scalars[str(composite_pauli)],
            )

    @pytest.mark.parametrize(
        "params1, params2, ref_final_scalars",
        [
            ([("ZX", 1j)], [("ZY", -1)], {"IZ": 1}),  # from prompt
            ([("XYZZI", 1)], [("XYZZI", 2)], {"IIIII": 2}),
            ([("XY", 1j)], [("ZZ", -2)], {"YX": -2j}),
            ([("ZZ", 1j)], [("XY", -2)], {"YX": -2j}),
            ([("XYZ", 1)], [("YZX", -1)], {"ZXY": 1j}),
            ([("IXYZ", 11)], [("YZIX", -2)], {"YYYY": -22}),
        ],
    )
    def test_multiply_pauli_algebra(self, params1, params2, ref_final_scalars):
        """
        Test that multiplication is performed correctly
        for a few test cases.
        """
        pauli1 = generate_pauli_element_from_tuples(params1)
        pauli2 = generate_pauli_element_from_tuples(params2)
        pauli_product = multiply_pauli_algebra(pauli1, pauli2)

        # ensure the number of terms matches
        assert len(pauli_product.scaled_elements) == len(ref_final_scalars)

        # verify coefficients
        for composite_pauli in pauli_product._elements_to_scalars:
            assert cmath.isclose(
                pauli_product._elements_to_scalars[composite_pauli],
                ref_final_scalars[str(composite_pauli)],
            )
