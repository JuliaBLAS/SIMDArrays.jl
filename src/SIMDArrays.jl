module SIMDArrays

using jBLAS, LinearAlgebra, Random, Base.Cartesian, VectorizedRNG, SIMDPirates, VectorizationBase
using UnsafeArrays
using ForwardDiff
# Base.mightalias(A::UnsafeArray, B::UnsafeArray) = false

import StaticArrays
import  jBLAS: REGISTER_SIZE, CACHELINE_SIZE,
        Kernel, initkernel!, kernel!,
        PrefetchA, PrefetchX, PrefetchAX

using SIMDPirates: Vec, evmul, vadd, vsub, vbroadcast, vload, vstore, vfma
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
        safeinvchol!,
        @Static, @Sized
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
