module SIMDArrays

using CpuId, LinearAlgebra, UnsafeArrays
# or whatever unsafe views are called
# needed for operations so we can ignore excess elemenents when necessary.

Base.mightalias(A::UnsafeArray, B::UnsafeArray) = false

# Would sorta make sense for this to depend on SIMD?
# Or would I rather, at least for now, just wrap functions in BLAS calls?

const REGISTER_SIZE = CpuId.simdbytes()

abstract type AbstractSIMDArray{T,N} <: AbstractArray{T,N} end


struct SIMDArray{T,N} <: AbstractSIMDArray{T,N}
    data::Array{T,N}
    nrows::Int
    @generated function SIMDArray{T}(::UndefInitializer, S::NTuple{N}) where N
        quote
            R, L = calculate_L_from_size(S) # R means number of rows.
            nrows = S[1]
            data = Array{T,N}(undef, setfirstindex(S, R))
            # We need to zero out the excess, so it doesn't interfere
            # with operations like summing the columns or matrix mul.
            Base.Cartesian.@nloops $(N-1) i j -> 1:S[j+1] begin
                for i_0 = L-R+1:L
                    data[(Base.Cartesian.@ntuple $N j -> i_{j-1})...] = zero($T)
                end
            end
            new{T,N}(data, nrows)
        end
    end
end

"""
Parameters are
S, the size tuple
T, the element type
N, the number of axis
R, the actual number of rows, where the excess R > S[1] are zero.
L, number of elements (including buffer zeros).
"""
mutable struct SizedSIMDArray{S,T,N,R,L} <: AbstractSIMDArray{T,N}
    data::NTuple{L,T}
    function SizedSIMDArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        #check_array_parameters(S, T, Val{N}, Val{L})
        new()
    end
    @generated function SizedSIMDArray{S,T}(::UndefInitializer) where {S,T}
        #check_array_parameters(S, T, Val{N}, Val{L})
        N = length(S)

        R, L = calculate_L_from_size(S)
        # @show S, T, N, L
        :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
end
const StaticSIMDVector{N,T,L} = SizedSIMDArray{Tuple{N},T,L}
const StaticSIMDMatrix{M,N,T,L} = SizedSIMDArray{Tuple{M,N},T,L}

@generated function Base.size(A::SIMDArray{T,N}) where {T,N}
    quote
        s = size(A.data)
        Base.Cartesian.@nextract $N s s
        s_1 = A.nrows
        Base.Cartesian.@ntuple $N s
    end
end
Base.size(::SizedSIMDArray{S}) where S = S
# Do we want this, or L?
@generated Base.length(::SizedSIMDArray{S}) where S = prod(S)

function Base.getindex(A::SIMDArray{T,N}, i) where {T,N}
    @boundscheck
    A.data[i]
end

round_x_up_to_nearest_y(x::Integer, y::Integer) = cld(x, y) * y
function calculate_L_from_size(S::NTuple{N}, ::Type{T} = Float64) where {N,T}
    num_rows = S[1]
    # Use number of simdbytes, or cacheline_length?
    # cacheline_length = 64 ÷ sizeof(T)
    entries_per_register = REGISTER_SIZE ÷ sizeof(T)
    if num_rows >= entries_per_register
        L = R = round_x_up_to_nearest_y(num_rows, entries_per_register)
    elseif 2num_rows >= entries_per_register
        L = R = round_x_up_to_nearest_y(num_rows, entries_per_register ÷ 2)
    elseif 4num_rows >= entries_per_register
        L = R = round_x_up_to_nearest_y(num_rows, entries_per_register ÷ 4)
    elseif 8num_rows >= entries_per_register
        L = R = round_x_up_to_nearest_y(num_rows, entries_per_register ÷ 8))
    else
        L = R = round_x_up_to_nearest_y(num_rows, max(entries_per_register ÷ 16,1))
    end
    for n ∈ 2:N
        L *= S[n]
    end
    R, L
end



# Both versions produce the same assembly.
# The non-generated version seems simpler. Only reason to prefer the generated
# would be concern that it would less likely to regress, but I don't think that's a realistic concern
# not one worth writing hard-to-read code over.
setfirstindex(tup::NTuple{N,T}, val::T) where {N,T}  = ntuple(j -> j == 1 ? val : tup[j], Val(N))
# @generated function setfirstindex(tup::NTuple{N,T}, val::T) where {N,T}
#     Expr(:tuple, [n == 1 ? :val : :(tup[$n]) for n ∈ 1:N]...)
# end


end # module
