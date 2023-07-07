import pytest

from pauli_algebra.matrices import (
    Pauli_I,
    Pauli_X,
    Pauli_Y,
    Pauli_Z,
    multiply_pauli,
    multiply_phase,
)


class TestPauliMatrices:
    @pytest.mark.parametrize("phase", [1, -1, 1j, -1j])
    def test_pauli_I_multiplication(self, phase: complex):
        I = Pauli_I(phase)
        assert I.phase == phase

        II = I.multiply_pauli(Pauli_I())
        assert II.symbol == "I"
        assert II.phase == phase

        IX = I.multiply_pauli(Pauli_X())
        assert IX.symbol == "X"
        assert IX.phase == phase

        IY = I.multiply_pauli(Pauli_Y())
        assert IY.symbol == "Y"
        assert IY.phase == phase

        IZ = I.multiply_pauli(Pauli_Z())
        assert IZ.symbol == "Z"
        assert IZ.phase == phase

    @pytest.mark.parametrize("phase", [1, -1, 1j, -1j])
    def test_pauli_X_multiplication(self, phase: complex):
        X = Pauli_X(phase)
        assert X.phase == phase

        XI = X.multiply_pauli(Pauli_I())
        assert XI.symbol == "X"
        assert XI.phase == phase

        XX = X.multiply_pauli(Pauli_X())
        assert XX.symbol == "I"
        assert XX.phase == phase

        XY = X.multiply_pauli(Pauli_Y())
        assert XY.symbol == "Z"
        assert XY.phase == phase * 1j

        XZ = X.multiply_pauli(Pauli_Z())
        assert XZ.symbol == "Y"
        assert XZ.phase == phase * -1j

    @pytest.mark.parametrize("phase", [1, -1, 1j, -1j])
    def test_pauli_Y_multiplication(self, phase: complex):
        Y = Pauli_Y(phase)
        assert Y.phase == phase

        YI = Y.multiply_pauli(Pauli_I())
        assert YI.symbol == "Y"
        assert YI.phase == phase

        YX = Y.multiply_pauli(Pauli_X())
        assert YX.symbol == "Z"
        assert YX.phase == phase * -1j

        YY = Y.multiply_pauli(Pauli_Y())
        assert YY.symbol == "I"
        assert YY.phase == phase

        YZ = Y.multiply_pauli(Pauli_Z())
        assert YZ.symbol == "X"
        assert YZ.phase == phase * 1j

    @pytest.mark.parametrize("phase", [1, -1, 1j, -1j])
    def test_pauli_Z_multiplication(self, phase: complex):
        Z = Pauli_Z(phase)
        assert Z.phase == phase

        ZI = Z.multiply_pauli(Pauli_I())
        assert ZI.symbol == "Z"
        assert ZI.phase == phase

        ZX = Z.multiply_pauli(Pauli_X())
        assert ZX.symbol == "Y"
        assert ZX.phase == phase * 1j

        ZY = Z.multiply_pauli(Pauli_Y())
        assert ZY.symbol == "X"
        assert ZY.phase == phase * -1j

        ZZ = Z.multiply_pauli(Pauli_Z())
        assert ZZ.symbol == "I"
        assert ZZ.phase == phase

    def test_multiply_pauli(self):
        """Ensure correct ordering."""
        Z = Pauli_Z(1)
        Y = Pauli_Y(1)
        ZY = multiply_pauli(Z, Y)
        assert ZY.phase == -1j
        assert ZY.symbol == "X"

    @pytest.mark.parametrize(
        "input_phase, final_phase", [(-1j, -1), (1j, 1), (-1, 1j), (1, -1j)]
    )
    def test_multiply_phase(self, input_phase: complex, final_phase: complex):
        """Ensure disctinct PauliMatrix object is created with new phase."""
        I = Pauli_I(phase=-1j)
        newI = multiply_phase(input_phase, I)
        assert I is not newI
        assert newI.phase == final_phase

    @pytest.mark.parametrize(
        "phase, ref_string", [(1, "Z"), (-1, "-Z"), (1j, "iZ"), (-1j, "-iZ")]
    )
    def test_pauli_to_string(self, phase: complex, ref_string: str):
        Z = Pauli_Z(phase)
        assert str(Z) == ref_string

    def test_pauli_to_string_after_multiplication(self):
        ref_string = "-iX"
        Z = Pauli_Z(1)
        Y = Pauli_Y(1)
        ZY = multiply_pauli(Z, Y)
        assert str(ZY) == ref_string
