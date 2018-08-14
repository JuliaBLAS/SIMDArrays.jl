
abstract type AbstractSIMDArray{T,N} <: AbstractArray{T,N} end


struct SIMDArray{T,N} <: AbstractSIMDArray{T,N}
    data::Array{T,N}
    nrows::Int
    function SIMDArray{T}(::UndefInitializer, S::NTuple{N}) where {T,N}
        R, L = calculate_L_from_size(S) # R means number of rows.
        nrows = S[1]
        data = Array{T,N}(undef, setfirstindex(S, R))
        # We need to zero out the excess, so it doesn't interfere
        # with operations like summing the columns or matrix mul.
        # add @generated and quote the expression if you add this back.
        # Base.Cartesian.@nloops $(N-1) i j -> 1:S[j+1] begin
        #     for i_0 = L-R+1:L
        #         data[(Base.Cartesian.@ntuple $N j -> i_{j-1})...] = zero($T)
        #     end
        # end
        new{T,N}(data, nrows)
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
mutable struct SizedSIMDArray{S<:Tuple,T,N,R,L} <: AbstractSIMDArray{T,N}
    data::NTuple{L,T}
    function SizedSIMDArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        #check_array_parameters(S, T, Val{N}, Val{L})
        new()
    end
    @generated function SizedSIMDArray{S,T}(::UndefInitializer) where {S,T}
        SV = S.parameters
        N = length(SV)
        R, L = calculate_L_from_size(SV)
        # quote
        #     out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
        #     Base.Cartesian.@nloops $(N-1) i j -> 1:$SV[j+1] begin
        #         for i_0 = $(SV[1]+1:R)
        #             out[(Base.Cartesian.@ntuple $N j -> i_{j-1})...] = $(zero(T))
        #         end
        #     end
        #     out
        # end
        :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
end

@generated function SizedSIMDArray(::UndefInitializer, ::Val{S}, ::Type{T}=Float64) where {S,T}
    N = length(S)
    R, L = calculate_L_from_size(S)
    SD = Expr(:curly, :Tuple, S...)
    # quote
    #     out = SizedSIMDArray{$SD,$T,$N,$R,$L}(undef)
    #     Base.Cartesian.@nloops $(N-1) i j -> 1:$S[j+1] begin
    #         @inbounds for i_0 = $(S[1]+1:R)
    #             out[(Base.Cartesian.@ntuple $N j -> i_{j-1})...] = $(zero(T))
    #         end
    #     end
    #     out
    # end
    :(SizedSIMDArray{$SD,$T,$N,$R,$L}(undef))
end

const SizedSIMDVector{N,T,R,L} = SizedSIMDArray{Tuple{N},T,1,R,L}
const SizedSIMDMatrix{M,N,T,R,L} = SizedSIMDArray{Tuple{M,N},T,2,R,L}
struct StaticSIMDArray{S<:Tuple,T,N,R,L} <: AbstractSIMDArray{T,N}
    data::NTuple{L,T}
    function StaticSIMDArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        #check_array_parameters(S, T, Val{N}, Val{L})
        new()
    end
    @generated function StaticSIMDArray{S,T}(::UndefInitializer) where {S,T}
        #check_array_parameters(S, T, Val{N}, Val{L})
        N = length(S)

        R, L = calculate_L_from_size(S)
        # @show S, T, N, L
        :(StaticSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
end
const StaticSIMDVector{N,T,R,L} = StaticSIMDArray{Tuple{N},T,1,R,L}
const StaticSIMDMatrix{M,N,T,R,L} = StaticSIMDArray{Tuple{M,N},T,2,R,L}

@inline Base.pointer(A::SIMDArray) = pointer(A.data)
@inline Base.pointer(A::SizedSIMDArray{S,T}) where {S,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))

@generated function Base.size(A::SIMDArray{T,N}) where {T,N}
    quote
        s = size(A.data)
        Base.Cartesian.@nextract $N s s
        s_1 = A.nrows
        Base.Cartesian.@ntuple $N s
    end
end
Base.size(::SizedSIMDArray{S}) where S = tuple(S.parameters...)

to_tuple(S) = tuple(S.parameters...)

# Do we want this, or L?
@generated Base.length(::SizedSIMDArray{S}) where S = prod(to_tuple(S))

@inline Base.getindex(A::SIMDArray, i...) = A.data[i...]
@inline Base.setindex!(A::SIMDArray, v, i...) = A.data[i...] = v
@inline function Base.getindex(A::SizedSIMDArray{S,T,N,R,L}, i::Int) where {S,T,N,R,L}
    @boundscheck i < L
    A.data[i]
end
"""
Returns zero based index. Don't forget to add one when using with arrays instead of pointers.
"""
function sub2ind_expr(S::NTuple{N}, R) where N
    ex = :(i[$N] - 1)
    for i ∈ (N - 1):-1:2
        ex = :(i[$i] - 1 + $(S[i]) * $ex)
    end
    :(i[1] - 1 + $R * $ex)
end
@generated function Base.getindex(A::SizedSIMDArray{S,T,N,R,L}, i::Vararg{Int,N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        unsafe_load(pointer(A), $ex+1)
    end
end
@generated function Base.getindex(A::SizedSIMDArray{S,T,N,R,L}, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        unsafe_load(pointer(A), $ex+1)
    end
end
@inline function Base.setindex!(A::SizedSIMDArray{S,T,N,R,L}, v, i) where {S,T,N,R,L}
    @boundscheck i < L
    unsafe_store!(pointer(A), v, i)
end
@generated function Base.setindex!(A::SizedSIMDArray{S,T,N,R,L}, v, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        unsafe_store!(pointer(A), v, $ex+1)
    end
end
@generated function Base.setindex!(A::SizedSIMDArray{S,T,N,R,L}, v, i::Vararg{Int,N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        unsafe_store!(pointer(A), v, $ex+1)
    end
end

round_x_up_to_nearest_y(x::Integer, y::Integer) = cld(x, y) * y
function calculate_L_from_size(S, ::Type{T} = Float64) where T
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
        L = R = round_x_up_to_nearest_y(num_rows, entries_per_register ÷ 8)
    else
        L = R = round_x_up_to_nearest_y(num_rows, max(entries_per_register ÷ 16,1))
    end
    for n ∈ 2:length(S)
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

Base.IndexStyle(::Type{<:AbstractSIMDArray}) = IndexLinear()
Base.IndexStyle(::AbstractSIMDArray, ::AbstractArray) = IndexCartesian()
Base.IndexStyle(::AbstractArray, ::AbstractSIMDArray) = IndexCartesian()
Base.IndexStyle(::AbstractSIMDArray, ::AbstractArray, ::AbstractArray) = IndexCartesian()
Base.IndexStyle(::AbstractArray, ::AbstractArray, ::AbstractSIMDArray) = IndexCartesian()
Base.IndexStyle(::AbstractArray, ::AbstractSIMDArray, ::AbstractArray) = IndexCartesian()
Base.IndexStyle(::AbstractSIMDArray, ::AbstractSIMDArray, ::AbstractArray) = IndexCartesian()
Base.IndexStyle(::AbstractSIMDArray, ::AbstractArray, ::AbstractSIMDArray) = IndexCartesian()
Base.IndexStyle(::AbstractArray, ::AbstractSIMDArray, ::AbstractSIMDArray) = IndexCartesian()

# function Base.similar()

"""
This is only meant to make recursive algorithms easiesr to implement.
Wraps a pointer, while passing info on the size of the block and stride.
"""
struct PtrMatrix{M,N,T,Stride}
    ptr::Ptr{T}
end
@inline Base.pointer(ptr::PtrMatrix) = ptr.ptr
# const SIMDMat{M,N,T,Stride} = Union{SizedSIMDMatrix{M,N,T,Stride},PtrMatrix{M,N,T,Stride}}
