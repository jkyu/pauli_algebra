from abc import ABC, abstractmethod

import numpy as np

PAULI_NORMALIZATION_FACTOR = np.sqrt(2)
PAULI_DIM = 2
PAULI_MATRIX_SHAPE = (2, 2)
PAULI_VECTOR_SHAPE = (4,)
VALID_PHASES = {1, -1, 1j, -1j}


class PauliMatrix(ABC):
    """
    Abstract base class for the four 2x2 Pauli matrices.
    The Pauli matrices are represented by a symbol (I, X, Y, or Z),
    and are used here with their vector representations because it is
    easier for me to think about the basis as four vectors
    (C^(2x2) is isomorphic to C^4).

    An instance a Pauli matrix also records the phase.
    A memory optimization would be to make 16 singletons, one for each
    phase and symbol pair, but I will not do that as part of this exercise.
    """

    def __init__(self, phase: complex = 1):
        self.phase = phase
        self.string = self._to_string()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(phase={self.phase})"

    def __str__(self) -> str:
        return self.string

    @property
    @abstractmethod
    def symbol(self):
        pass

    def multiply_phase(self, phi: complex) -> "PauliMatrix":
        """
        Multiply the current phase by phi and return a new
        PauliMatrix instance with the result.
        """
        if phi not in VALID_PHASES:
            raise ValueError("phi must be in {1, -1, i, -i}.")
        new_phase = self.phase * phi
        return self.__class__(new_phase)

    def _to_string(self) -> str:
        """
        Create string representation of the Pauli matrix
        with its phase.
        """
        string_components = []
        # write minus sign if -1 or -i
        if np.sign(self.phase) < 0:
            string_components.append("-")
        # write i if imaginary phase
        if np.imag(self.phase) != 0:
            string_components.append("i")
        string_components.append(self.symbol)
        return "".join(string_components)

    @abstractmethod
    def multiply_pauli(self, other_pauli: "PauliMatrix") -> "PauliMatrix":
        """
        This pauli matrix operates on other_pauli from the left.
        The multiplication is performed and a new PauliMatrix
        instance with the result is returned.
        """
        pass


class Pauli_I(PauliMatrix):
    symbol = "I"
    vector = np.array([1, 0, 0, 1])

    def multiply_pauli(self, other_pauli: PauliMatrix) -> PauliMatrix:
        # II = I
        # IX = X
        # IY = Y
        # IZ = Z
        return other_pauli.__class__(self.phase * other_pauli.phase)


class Pauli_X(PauliMatrix):
    symbol = "X"
    vector = np.array([0, 1, 1, 0])

    def multiply_pauli(self, other_pauli: PauliMatrix) -> PauliMatrix:
        match other_pauli.symbol:
            case "I":
                # XI = X
                return self
            case "X":
                # XX = I
                return Pauli_I(self.phase * other_pauli.phase)
            case "Y":
                # XY = iZ
                return Pauli_Z(self.phase * 1j)
            case "Z":
                # XZ = -iY
                return Pauli_Y(self.phase * -1j)


class Pauli_Y(PauliMatrix):
    symbol = "Y"
    vector = np.array([0, -1j, 1j, 0])

    def multiply_pauli(self, other_pauli: PauliMatrix) -> PauliMatrix:
        match other_pauli.symbol:
            case "I":
                # YI = Y
                return self
            case "X":
                # YX = -iZ
                return Pauli_Z(self.phase * -1j)
            case "Y":
                # YY = I
                return Pauli_I(self.phase * other_pauli.phase)
            case "Z":
                # YZ = iX
                return Pauli_X(self.phase * 1j)


class Pauli_Z(PauliMatrix):
    symbol = "Z"
    vector = np.array([1, 0, 0, -1])

    def multiply_pauli(self, other_pauli: PauliMatrix) -> PauliMatrix:
        match other_pauli.symbol:
            case "I":
                # ZI = Z
                return self
            case "X":
                # ZX = iY
                return Pauli_Y(self.phase * 1j)
            case "Y":
                # ZY = -iX
                return Pauli_X(self.phase * -1j)
            case "Z":
                # ZZ = I
                return Pauli_I(self.phase * other_pauli.phase)


def multiply_phase(phi: complex, pauli: PauliMatrix) -> PauliMatrix:
    """
    Multiply the pauli matrix by a phase and create a new pauli matrix
    object instance with the phase product.
    """
    return pauli.multiply_phase(phi)


def multiply_pauli(pauli1: PauliMatrix, pauli2: PauliMatrix) -> PauliMatrix:
    """
    Multiply two pauli matrices. This will perform the following operation
    for P1 = pauli1 and P2 = pauli2:
        product = P1^t P2
    where P1^t is the complex conjugate of P1. This function returns
    a new pauli matrix object instance with the product.
    """
    return pauli1.multiply_pauli(pauli2)


def pauli_to_string(pauli: PauliMatrix) -> str:
    """
    Convert the pauli matrix to string representation.
    """
    return str(pauli)


def pauli_from_symbol(symbol: str) -> PauliMatrix:
    """
    This generates an instance of the PauliMatrix class of the
    correct type given one of the Pauli matrix symbols: I, X, Y, or Z.
    """
    match symbol:
        case "I":
            return Pauli_I()
        case "X":
            return Pauli_X()
        case "Y":
            return Pauli_Y()
        case "Z":
            return Pauli_Z()
        case _:
            raise ValueError(f"Symbol {symbol} does not represent a Pauli matrix.")
