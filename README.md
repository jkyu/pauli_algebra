# Pauli Computer Algebra (Jimmy Yu)

## Install the package
Running the Pauli computer algebra package requires installing it as a python package.
I recommend installing it in a fresh virtual environment with the `pytest` and `NumPy`.
Because I used the new `switch/case` feature, `Python >= 3.10` is required. 
In the same directory as this README, run:
```
pip install .
```

Once installed, the modules in this package can be imported.
For example,
```
from pauli_algebra import algebra, matrices, utils
```

A test suite is included and can be executed by running the following command in the same directory as this README.
```
pytest .
```

## Implementation discussion

Before starting a discussion of how I approached each goal (and where to find the code corresponding to each goal), I want to point out that most of the following information is redundant with documentation in the code itself.
Some of the discussion may be extraneous, so feel free to dig into the code instead of reading the following write-up if you prefer.
While I did write docstrings for the important functions, they are not formatted to any particular standard. 

### Goal 1 Implementation
Goal 1 is implemented in `pauli_algebra.utils.generate_random_hermitian()`. 
This function supports generation of a Hermitian matrix of arbitrary dimension and with complex elements.
`test_utils.TestPauliUtils.test_generate_random_hermitian()` verifies that this function generates Hermitian matrices.

Note: a deviation is made from the prompt here. Samples for the real and imaginary components of the elements of the Hermitian matrix are drawn from $Uniform[-b,b)$ rather than $Uniform(-b,b)$ because `numpy.random.uniform()` uses an inclusive lower bound. I don't think this is a big deal for this application.

### Goal 2 Implementation
Goal 2 is implemented across `pauli_algebra.utils.get_pauli_expansion_coefficients()` and `pauli_algebra.matrices`.
For the purposes of Goal 2, `pauli_algebra.matrices` includes vector representations of four $2\times 2$ Pauli matrices.
These are basis vectors for the Pauli matrix decomposition: $$r_II + r_XX + r_YY+r_ZZ$$

The scalar factors $r_I, r_X, r_Y,$ and $r_Z$ are obtained by projection of the Hermitian matrix on each of the basis vectors $I, X, Y,$ and $Z$.
`test_utils.TestPauliUtils.test_get_pauli_expansion_coefficients()` verifies the correctness of the expansion and `test_utils.TestPauliUtils.test_get_pauli_projections()` verifies that $r_I, r_X, r_Y$ and $r_Z$ are real.

### Goal 3 Implementation
Goal 3 is implemented in the `pauli_algebra.matrices` module. 
The Pauli matrices $I, X, Y,$ and $Z$ are instantiated by calling the constructors `Pauli_I()`, `Pauli_X()`, `Pauli_Y()`, and `Pauli_Z()`, respectively.
An optional argument can be used to specify the phase, which defaults to 1 otherwise.
The Pauli matrix products are enumerated so that the Pauli Computer Algebra can perform symbolic operations without manipulating any matrices.
`multiply_phase()`, `multiply_pauli()`, and `pauli_to_string()` are all implemented, as requested by the exercise prompt.
They are just wrappers on functions belonging to the `PauliMatrix` classes but expose a "public" function API.

I made the design choice to have instances of the Pauli matrices emulate primatives. 
Operations on the matrices will return the result in a new instance rather than modify the properties of an existing `PauliMatrix` object. 
This allows implementing four Pauli matrices as separate classes and also allows for a very clear picture of "before" and "after."
For example, two Pauli matrices involved in multiplication are preserved when generating a product.
This approach can be optimized for memory usage (in the case that we have many qubits) by creating singleton instances of the Pauli matrices for each phase and symbol combination (e.g., iX).
There are only 16 such combinations, so this would work to avoid making a new object instance for every usage of a Pauli matrix.

Please see `test/test_matrices.py` for verification that the Pauli matrix operations behave correctly.

### Goal 4 Implementation
Goal 4 is primarily implemented in the `pauli_algebra.algebra` module, although it builds on the code written for the previous goals.
This follows the same principle of treating linear combinations of composite Pauli operators as primatives.
I reference these linear combinations as "Pauli expressions" throughout comments in the code, which I hope does not overload terminology in this domain, but the usage is consistent here.
- The `CompositePauliElement` class is implemented to facilitate the matching of composite Pauli matrices during simplification. Simplification is performed by hashing the string of symbols and combining scalar factors that share the same hash.
- This results in a trade-off where `CompositePauliElement` does not also contain information about its scalar factor. This is a design decision made to prevent any confusion over the "equality" (according to the hash function) of composite Pauli matrices that have different scalar factors. A NamedTuple, `ScaledCompositePauliElement` composes a complex scalar and the `CompositePauliElement` to satisfy the need for association between a scalar factor and the composite Pauli matrix that it scales.
- `PauliAlgebra` then implements the representation of a linear combination of composite Pauli operators, e.g., $iXYZ - 2YZX$ (in contrast to `CompositePauliElement`, which represents only one term). I made the choice to simplify the Pauli expression upon instantiation of the `PauliAlgebra` object.

The following functions are all implemented (and like in part 3, they all wrap functions belonging to the `PauliAlgebra` class):
- `pauli_algebra.algebra.make_pauli_algebra_element()`
- `pauli_algebra.algebra.multiply_pauli_algebra_by_scalar()`
- `pauli_algebra.algebra.add_pauli_algebra()`
- `pauli_algebra.algebra.multiply_pauli_algebra()`
- `pauli_algebra.algebra.simplify()`
- `pauli_algebra.algebra.pauli_algebra_to_string()`

Please see `test/test_algebra.py` for validation and also examples for performing operations on Pauli expressions with this module.

Notes:
- I did not implement error handling for what I would expect to be common errors involving dimensionality mismatches, e.g., attempting to add Pauli algebra elements for 4 qubit and 5 qubit systems.
- I left the `pauli_algebra_to_string()` results a little bit ugly. You'll see `"XYZ + -YZX`, for example. This shouldn't be too hard to fix, but I don't think it adds too much to the project in its current state, although it might be helpful to have prettier strings if we want to generate full Pauli expressions with many terms from a string.
- Some of the tests on the `*_to_string()` are a bit tricky because of the complex numbers, e.g., $2i$ and $(0+2i)$ being valid string representations of the same number. This could be ironed out with some more care. While the tests all pass on my computer, I could see errors coming up elsewhere, so don't be alarmed if the `*_to_string()` tests are the only ones that fail.