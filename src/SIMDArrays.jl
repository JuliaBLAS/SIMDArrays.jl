module SIMDArrays

using jBLAS, LinearAlgebra, SIMD, Random, Base.Cartesian
using UnsafeArrays
# Base.mightalias(A::UnsafeArray, B::UnsafeArray) = false

import jBLAS: REGISTER_SIZE, CACHELINE_SIZE, Kernel, initkernel!, kernel!
# or whatever unsafe views are called
# needed for operations so we can ignore excess elemenents when necessary.

export  SizedSIMDVector,
        SizedSIMDMatrix,
        SizedSIMDArray,
        randsimd,
        randnsimd,
        SymmetricMatrix # not really supported yet.

# Would sorta make sense for this to depend on SIMD?
# Or would I rather, at least for now, just wrap functions in BLAS calls?


include("simd_arrays.jl")
include("blas.jl")
include("random.jl")
# include("instantiation.jl")
include("miscellaneous.jl")

end # module
