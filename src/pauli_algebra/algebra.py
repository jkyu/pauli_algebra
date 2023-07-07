from typing import Dict, List, NamedTuple, Optional

import numpy as np

from pauli_algebra.matrices import (
    PauliMatrix,
    multiply_pauli,
    pauli_from_symbol,
    pauli_to_string,
)


class CompositePauliElement:
    """
    Represents a composite Pauli element for n-qubits as
    an n-vector of Pauli operators. This composite element
    does not account for a scalar factor or phase.
    This symbolically represents the direct product of
    num_qubits Pauli matrices.

    The rationale behind this choice is that it will be confusing
    if composite elements hash to the same value despite
    being characterized by different scalars.
    (see ScaledCompositePauliElement named tuple, which pairs
    a composite pauli element with a scalar.)
    """

    def __init__(
        self, pauli_matrices: List[PauliMatrix], element_string: Optional[str] = None
    ):
        self.num_qubits = len(pauli_matrices)
        self.pauli_matrices = pauli_matrices
        if element_string is None:
            element_string = self._pauli_matrices_to_string()
        self._element_string = element_string

    def __eq__(self, other: "CompositePauliElement") -> bool:
        if self.num_qubits != other.num_qubits:
            return False
        for pauli1, pauli2 in zip(self.pauli_matrices, other.pauli_matrices):
            if pauli1.symbol != pauli2.symbol:
                return False
        return True

    def __hash__(self):
        """
        Enable hashing of CompositePauliElement instances so that instances with
        the same element string are identical under the hash function.
        This enables use of a CompositePauliElement instance as a hash key.
        """
        return hash(self._element_string)

    def __repr__(self):
        pauli_matrix_strings = []
        for pauli in self.pauli_matrices:
            pauli_matrix_strings.append(pauli_to_string(pauli))
        joined_pauli_matrix_strings = ", ".join(pauli_matrix_strings)
        return f"CompositePauliElement(pauli_matrices=[{joined_pauli_matrix_strings}], element_string={self._element_string})"

    def __str__(self):
        return self._element_string

    def _pauli_matrices_to_string(self):
        string_components = []
        for pauli in self.pauli_matrices:
            string_components.append(pauli.symbol)
        return "".join(string_components)

    @classmethod
    def from_string(cls, element_string: str) -> "CompositePauliElement":
        """
        Generate a composite Pauli matrix from a string of symbols,
        e.g., XYZ.
        """
        # composite string assumed to not start with "-" or "i"
        pauli_matrices = []
        for symbol in element_string:
            pauli_matrices.append(pauli_from_symbol(symbol))
        return CompositePauliElement(pauli_matrices, element_string=element_string)


class ScaledCompositePauliElement(NamedTuple):
    """Named tuple for pairing a complex scalar with a CompositePauliElement"""

    element: CompositePauliElement
    scalar: complex


class PauliAlgebra:
    """
    This class represents a linear combination of composite Pauli matrices
    of the same dimension. While CompositePauliElement would represent, e.g.,
    'XYZ', this class can represent, e.g., 2XYZ + 3iYZX.

    Every operation, i.e., addition, multiplication or scalar multiplication,
    on a PauliAlgebra object generates a new PauliAlgebra object. Simplication
    is performed on instantiation, so that CompositePauliElements of the same
    kind (represented by the same symbols in the same order) have their
    scalar coeffients combined.

    There is minimal error handling/checking for dimensionality mismatches,
    which would cause addition or multiplication to fail.
    This is important for a robust program, but I only included very basic
    validation here.
    """

    def __init__(
        self, scaled_elements: List[ScaledCompositePauliElement], num_qubits: int = None
    ):
        if num_qubits is None:
            if len(scaled_elements) == 0:
                raise ValueError(
                    "Cannot create a Pauli algebra element without either providing a composite Pauli operator or its dimension."
                )
            num_qubits = scaled_elements[0].element.num_qubits
        self.num_qubits = num_qubits

        # simplify the Pauli algebra expression on instantiation
        self._elements_to_scalars = self._simplify_by_pauli_element_strings(
            scaled_elements
        )

        # the publicly accessible terms in the Pauli expression
        # are collected after simplification
        self.scaled_elements = [
            ScaledCompositePauliElement(element, scalar)
            for element, scalar in self._elements_to_scalars.items()
        ]
        self._string = self._to_string()

    def _simplify_by_pauli_element_strings(
        self, scaled_elements: List[ScaledCompositePauliElement]
    ) -> Dict[CompositePauliElement, complex]:
        """
        This takes a list of ScaledCompositePauliElements containing information
        about the CompositePauliElements in the Pauli expression and their scalar
        coefficients.

        CompositePauliElements that are the same have their scalars combined.
        Hashing is used to identify CompositePauliElements as equivalent for this
        purpose.

        A dictionary that maps the CompositePauliElement to its complex coefficient
        is returned. In the case of simplification, the number of terms in the
        Pauli expression decreases.
        """
        elements_to_scalars = {}
        for scaled_element in scaled_elements:
            try:
                elements_to_scalars[scaled_element.element] += scaled_element.scalar
            except:
                elements_to_scalars[scaled_element.element] = scaled_element.scalar
            # remove element if its scalar has been reduced to zero
            if elements_to_scalars[scaled_element.element] == 0:
                elements_to_scalars.pop(scaled_element.element)
        return elements_to_scalars

    def __str__(self):
        return self._string

    @classmethod
    def make_pauli_algebra_element(
        cls, element_string: str, scalar: complex = 1
    ) -> "PauliAlgebra":
        """
        Initialize a PauliAlgebra element from a Pauli group element
        string assumed to have unit phase (per the prompt) unless
        a scalar is provided.
        """
        composite_pauli_elements = [
            ScaledCompositePauliElement(
                CompositePauliElement.from_string(element_string), scalar
            )
        ]
        return PauliAlgebra(composite_pauli_elements)

    def multiply_pauli_algebra_by_scalar(self, scalar: complex) -> "PauliAlgebra":
        """
        Multiply the entire Pauli expression by a complex scalar
        and return a new PauliAlgebra object containing the result.

        Scalar multiplication is linear in composite Pauli vector spaces.
        """
        new_scaled_elements = []
        for scaled_element in self.scaled_elements:
            new_scaled_elements.append(
                ScaledCompositePauliElement(
                    scaled_element.element, scaled_element.scalar * scalar
                )
            )
        return PauliAlgebra(new_scaled_elements)

    def add_pauli_algebra(self, other: "PauliAlgebra") -> "PauliAlgebra":
        """
        Concatenate list of scaled elements to perform addition and return a new
        PauliAlgebra object with the result. Simplification happens on instantiation.
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                "Two Pauli algebra elements must belong to the same dimensional space to be added."
            )
        return PauliAlgebra(self.scaled_elements + other.scaled_elements)

    def multiply_pauli_algebra(self, other: "PauliAlgebra") -> "PauliAlgebra":
        """
        Perform multiplication by distributing the left pauli element
        across the right pauli element. The result is returned in a new
        PauliAlgebra object.

        Multiplication is linear and therefore distributive
        (see _multiply_composite_pauli_elements())
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                "Two Pauli algebra elements must belong to the same dimensional space to be multiplied."
            )
        new_scaled_elements = []
        for scaled_element in self.scaled_elements:
            for other_scaled_element in other.scaled_elements:
                new_scaled_element = self._multiply_composite_pauli_elements(
                    scaled_element, other_scaled_element
                )
                new_scaled_elements.append(new_scaled_element)
        return PauliAlgebra(new_scaled_elements)

    def _multiply_composite_pauli_elements(
        self,
        scaled_element1: ScaledCompositePauliElement,
        scaled_element2: ScaledCompositePauliElement,
    ) -> ScaledCompositePauliElement:
        """
        Perform element-wise pauli matrix multiplication, where an element refers to
        a whole pauli matrix in Pauli matrix direct product. Each product yields a
        a single Pauli matrix. A new composite Pauli operator is generated just
        from the Pauli symbols. The phases are collected separately and multiplied into
        the scalar.

        This multiplication is a bit cumbersome because of the design decision to
        keep the CompositePauliElement separate from an attached scalar coefficient.
        This is the trade-off made for a hashable CompositePauliElement.
        """
        element1 = scaled_element1.element
        element2 = scaled_element2.element
        scalar = scaled_element1.scalar * scaled_element2.scalar
        new_element_string = []
        for pauli1, pauli2 in zip(element1.pauli_matrices, element2.pauli_matrices):
            new_pauli = multiply_pauli(pauli1, pauli2)
            new_element_string.append(new_pauli.symbol)
            scalar *= new_pauli.phase
        new_composite_element = CompositePauliElement.from_string(
            "".join(new_element_string)
        )
        return ScaledCompositePauliElement(new_composite_element, scalar)

    def _to_string(self) -> str:
        """
        Could make this prettier by replacing + with - where applicable
        but not worth the time right now.
        """
        connector = " + "
        string_components = []
        for scaled_element in self.scaled_elements:
            if scaled_element.scalar == 0:
                continue
            string_components.append(
                f"{self._simplify_scalar(scaled_element.scalar)}{str(scaled_element.element)}"
            )
        return connector.join(string_components)

    def _simplify_scalar(self, scalar: complex) -> complex:
        """
        Logic for making the scalar look pretty.
        """
        if np.real(scalar) == 0:
            # pure imaginary
            if scalar == 1j:
                simple_scalar = "i"
            elif scalar == -1j:
                simple_scalar = "-i"
            else:
                simple_scalar = str(scalar).replace("j", "i")
        elif np.imag(scalar) == 0:
            # pure real
            if scalar == 1:
                simple_scalar = ""
            elif scalar == -1:
                simple_scalar = "-"
            else:
                simple_scalar = str(scalar)
        else:
            simple_scalar = str(scalar).replace("j", "i")
        return simple_scalar


def make_pauli_algebra_element(string: str, scalar: complex = 1) -> PauliAlgebra:
    """
    Wrapper to generate a PauliAlgebra expression from an initial
    string that details only a single composite Pauli matrix term.
    By default, the scalar factor is set to 1. This satisfies the
    prompt, but I added the scalar as a keyword argument to allow
    for a little bit more flexibility.
    """
    return PauliAlgebra.make_pauli_algebra_element(string, scalar)


def pauli_algebra_to_string(pauli_element: PauliAlgebra) -> str:
    """
    Returns the string form of the Pauli expression (potentially
    a linear combination of composite Pauli matrices).
    """
    return str(pauli_element)


def simplify(pauli_element: PauliAlgebra) -> PauliAlgebra:
    """
    Simplification is performed automatically on PauliAlgebra
    instantiation. This is an idempotent function included to
    satisfy the prompt.
    """
    return pauli_element


def add_pauli_algebra(
    pauli_element1: PauliAlgebra, pauli_element2: PauliAlgebra
) -> PauliAlgebra:
    """
    Add two Pauli expressions and return a new PauliAlgebra
    object with the result.
    """
    return pauli_element1.add_pauli_algebra(pauli_element2)


def multiply_pauli_algebra(
    pauli_element1: PauliAlgebra, pauli_element2: PauliAlgebra
) -> PauliAlgebra:
    """
    Perform left multiplication on pauli_element2 by pauli_element1.
    Return the result in a new PauliAlgebra instance.
    """
    return pauli_element1.multiply_pauli_algebra(pauli_element2)


def multiply_pauli_algebra_by_scalar(
    scalar: complex, pauli_element: PauliAlgebra
) -> PauliAlgebra:
    """
    Perform scalar multiplication on pauli_element by scalar.
    Return the result in a new PauliAlgebra instance.
    """
    return pauli_element.multiply_pauli_algebra_by_scalar(scalar)
