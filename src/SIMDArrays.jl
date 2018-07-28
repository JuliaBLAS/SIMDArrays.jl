module SIMDArrays

using CpuId

const REGISTER_SIZE = CpuId.simdbytes()


abstract type AbstractSIMDArray{T,N} <: AbstractArray{T,N} end

struct SIMDArray{T,N} <: AbstractSIMDArray{T,N}
    data::Array{T,N}
    nrows::Int
    function SIMDArray{T}(::UndefInitializer, S::NTuple{N}) where N
        R, L = calculate_L_from_size(S)
        nrows = S[1]
        data = Array{T,N}(undef, setfirstindex(S, R))
        new{T,N}()
    end
end
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
        q = :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))

        q
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
function calculate_L_from_size(S::NTuple{N}) where N
    num_rows = S[1]
    # Use number of simdbytes, or cacheline_length?
    # cacheline_length = 64 ÷ sizeof(T)
    if num_rows > REGISTER_SIZE
        L = R = round_x_up_to_nearest_y(num_rows, REGISTER_SIZE)
    elseif 2num_rows > REGISTER_SIZE
        L = R = round_x_up_to_nearest_y(num_rows, REGISTER_SIZE ÷ 2)
    elseif 4num_rows > REGISTER_SIZE
        L = R = round_x_up_to_nearest_y(num_rows, REGISTER_SIZE ÷ 4)
    else
        L = R = round_x_up_to_nearest_y(num_rows, REGISTER_SIZE ÷ 8)
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
