module SIMDArrays

using jBLAS, LinearAlgebra, SIMD, Random, Base.Cartesian
using UnsafeArrays
using ForwardDiff
# Base.mightalias(A::UnsafeArray, B::UnsafeArray) = false

import jBLAS: REGISTER_SIZE, CACHELINE_SIZE, Kernel, initkernel!, kernel!
# or whatever unsafe views are called
# needed for operations so we can ignore excess elemenents when necessary.

export  SizedSIMDVector,
        SizedSIMDMatrix,
        SizedSIMDArray,
        randsimd,
        randnsimd,
        SymmetricMatrix,
        full_length, # not really supported yet.
        inv_U_triangle!,
        cholesky_U!,
        inv_L_triangle!,
        safecholesky!,
        choldet!,
        safecholdet!,
        invchol!,
        invcholdet!,
        safeinvcholdet!,
        safeinvchol!
        # vsub!,
        # vadd!,
        # reflect!

# Would sorta make sense for this to depend on SIMD?
# Or would I rather, at least for now, just wrap functions in BLAS calls?


include("simd_arrays.jl")
include("blas.jl")
include("random.jl")
# include("DualArrays.jl")
# include("instantiation.jl")
include("miscellaneous.jl")
include("naive_linalg.jl")

end # module
