import numpy as np

from pauli_algebra.matrices import (
    PAULI_DIM,
    PAULI_MATRIX_SHAPE,
    PAULI_VECTOR_SHAPE,
    Pauli_I,
    Pauli_X,
    Pauli_Y,
    Pauli_Z,
)
from pauli_algebra.utils import (
    generate_random_hermitian,
    get_pauli_expansion_coefficients,
    get_pauli_projections,
)


class TestPauliUtils:
    EPSILON = 1e-8
    TEST_BOUND = 10

    def test_generate_random_hermitian(self):
        matrix = generate_random_hermitian(self.TEST_BOUND, size=PAULI_DIM)

        # verify that the conjugate transpose of the matrix is itself
        difference = matrix - np.conjugate(np.transpose(matrix))
        # infinity norm within epsilon of zero
        assert np.max(np.abs(difference)) < self.EPSILON

    def test_get_pauli_expansion_coefficients(self):
        # reconstruct the matrix from its expansion coeffs
        matrix = generate_random_hermitian(self.TEST_BOUND, size=PAULI_DIM)
        expansion_coefficients = get_pauli_expansion_coefficients(matrix)
        reconstructed_matrix = (
            expansion_coefficients.rI * Pauli_I.vector
            + expansion_coefficients.rX * Pauli_X.vector
            + expansion_coefficients.rY * Pauli_Y.vector
            + expansion_coefficients.rZ * Pauli_Z.vector
        )
        reconstructed_matrix = np.reshape(reconstructed_matrix, PAULI_MATRIX_SHAPE)

        difference = matrix - reconstructed_matrix
        assert np.max(np.abs(difference)) < self.EPSILON

    def test_get_pauli_projections(self):
        # test all projection components are real
        matrix = generate_random_hermitian(self.TEST_BOUND, size=PAULI_DIM)
        vector = np.reshape(matrix, PAULI_VECTOR_SHAPE)
        projection_components = get_pauli_projections(vector)
        for component in projection_components:
            assert np.real(component) != 0 and np.imag(component) == 0
