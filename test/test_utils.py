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
        """
        Verify that the random matrix generated is in fact Hermitian.
        """
        matrix = generate_random_hermitian(self.TEST_BOUND, size=PAULI_DIM)

        # verify that the conjugate transpose of the matrix is itself
        difference = matrix - np.conjugate(np.transpose(matrix))
        # infinity norm is within epsilon of zero
        assert np.max(np.abs(difference)) < self.EPSILON

    def test_get_pauli_expansion_coefficients(self):
        """
        Tests that the infinity norm of the difference between
        the original 2x2 Hermitian matrix and its reconstruction
        from the Pauli matrix decomposition is numerically zero.
        """
        matrix = generate_random_hermitian(self.TEST_BOUND, size=PAULI_DIM)

        # perform decomposition in the basis of the Pauli matrices
        expansion_coefficients = get_pauli_expansion_coefficients(matrix)
        # reconstruct the matrix from its decomposition
        reconstructed_matrix = (
            expansion_coefficients.rI * Pauli_I.vector
            + expansion_coefficients.rX * Pauli_X.vector
            + expansion_coefficients.rY * Pauli_Y.vector
            + expansion_coefficients.rZ * Pauli_Z.vector
        )
        reconstructed_matrix = np.reshape(reconstructed_matrix, PAULI_MATRIX_SHAPE)

        # verify the original and reconstructed matrices are
        # the same using the infinity norm of the difference
        difference = matrix - reconstructed_matrix
        assert np.max(np.abs(difference)) < self.EPSILON

    def test_get_pauli_projections(self):
        """
        Verify that all coefficients obtained from projecting
        a Hermitian matrix onto the Pauli basis are real.
        """
        matrix = generate_random_hermitian(self.TEST_BOUND, size=PAULI_DIM)
        vector = np.reshape(matrix, PAULI_VECTOR_SHAPE)
        projection_components = get_pauli_projections(vector)
        for component in projection_components:
            assert np.real(component) != 0 and np.imag(component) == 0
