from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from pauli_algebra.algebra import (
    CompositePauliElement,
    PauliAlgebra,
    ScaledCompositePauliElement,
)
from pauli_algebra.matrices import (
    PAULI_DIM,
    PAULI_MATRIX_SHAPE,
    PAULI_NORMALIZATION_FACTOR,
    PAULI_VECTOR_SHAPE,
    Pauli_I,
    Pauli_X,
    Pauli_Y,
    Pauli_Z,
)


def generate_random_hermitian(bound: float, size: int = PAULI_DIM):
    """
    Generates a random hermitian matrix with the following steps:
    1. generate two vectors of length size**2 with elements sampled uniformly
       in [-bound, bound). Each vector lives in R^(size**2)
    2. the second vector is multiplied by i to make it imaginary.
       adding the vectors elementwise yields a complex vector in C^(size**2)
    3. the complex vector is resized as (size, size) and symmetrized.

    Notes:
    - working with vectors and reshaping is fine because C^(size**2)
      is isomorphic to C^(size x size)
    - I would have liked to sample in (-bound, bound) to match the exact
      ask by the prompt, but np.random.uniform() includes the lower bound.
      There is 0 probability of sampling -bound, so maybe this is close
      enough for now.
    """
    # require bound to be positive
    if bound <= 0:
        raise ValueError(f"{bound} must be a positive parameter.")

    length = size**2

    real_vector = np.random.uniform(low=-bound, high=bound, size=length)
    imag_vector = np.random.uniform(low=-bound, high=bound, size=length)

    complex_vector = real_vector + 1j * imag_vector

    complex_matrix = np.reshape(complex_vector, (size, size))
    symmetrized_matrix = complex_matrix + np.conjugate(np.transpose(complex_matrix))

    return symmetrized_matrix


class PauliExpansion:
    """Dataclass for expansion components."""

    def __init__(self, rI: complex, rX: complex, rY: complex, rZ: complex):
        """
        The expansion coefficients are expected to be purely real for
        Hermitian matrices. This is assumed to be true here and there
        is no validation performed.
        """
        self.rI = np.real(rI)
        self.rX = np.real(rX)
        self.rY = np.real(rY)
        self.rZ = np.real(rZ)


def get_pauli_projections(vector: ArrayLike) -> Tuple[float]:
    """Separate function to do the projection to make it testable."""

    conjugate_vector = np.conjugate(vector)

    # the pauli matrices as given are not normalized so we divide
    # by the squared magnitude of the pauli matrix when doing the
    # the projection
    rI = np.dot(conjugate_vector, Pauli_I.vector) / PAULI_NORMALIZATION_FACTOR**2
    rX = np.dot(conjugate_vector, Pauli_X.vector) / PAULI_NORMALIZATION_FACTOR**2
    rY = np.dot(conjugate_vector, Pauli_Y.vector) / PAULI_NORMALIZATION_FACTOR**2
    rZ = np.dot(conjugate_vector, Pauli_Z.vector) / PAULI_NORMALIZATION_FACTOR**2

    return (rI, rX, rY, rZ)


def get_pauli_expansion_coefficients(matrix: ArrayLike) -> PauliExpansion:
    """
    Return expansion coefficients in Pauli basis.
    The expansion coefficients can be accessed as data class elements.
    See the PauliExpansion class.
    """
    if np.shape(matrix) != PAULI_MATRIX_SHAPE:
        raise ValueError(
            "Hermitian matrix must be of size (2,2) to be represented in the basis of Pauli matrices."
        )

    vector = np.reshape(matrix, PAULI_VECTOR_SHAPE)
    rI, rX, rY, rZ = get_pauli_projections(vector)

    return PauliExpansion(rI, rX, rY, rZ)


def generate_pauli_element_from_strings_and_scalars(
    strings: List[str], scalars: List[complex]
) -> PauliAlgebra:
    """
    A test helper to generate a pauli group element given a list
    of strings and scalars. This can generate a linear combination
    of composite Pauli matrices.
    """
    composite_pauli_elements = [
        CompositePauliElement.from_string(string) for string in strings
    ]
    scaled_elements = [
        ScaledCompositePauliElement(composite_pauli, scalar)
        for (composite_pauli, scalar) in zip(composite_pauli_elements, scalars)
    ]
    return PauliAlgebra(scaled_elements)


def generate_pauli_element_from_tuples(
    params: List[Tuple[str, complex]]
) -> PauliAlgebra:
    """
    A test helper to generate a pauli group element given a list of
    tuples, in which the first tuple element is a string and the second
    is a scalar, e.g., ("XYZ", 1). This can generate a linear combination
    of composite Pauli matrices.
    """
    pauli = generate_pauli_element_from_strings_and_scalars(
        [pair[0] for pair in params], [pair[1] for pair in params]
    )
    return pauli
