
abstract type AbstractSIMDArray{T,N} <: AbstractArray{T,N} end
abstract type AbstractSizedSIMDArray{S<:Tuple,T,N,R,L} <: AbstractSIMDArray{T,N} end
const AbstractSizedSIMDVector{N,T,R,L} = AbstractSizedSIMDArray{Tuple{N},T,1,R,L}
const AbstractSizedSIMDMatrix{M,N,T,R,L} = AbstractSizedSIMDArray{Tuple{M,N},T,2,R,L}



struct SIMDArray{T,N} <: AbstractSIMDArray{T,N}
    data::Array{T,N}
    nrows::Int
    @generated function SIMDArray{T}(::UndefInitializer, S::NTuple{N}) where {T,N}
        quote
            R, L = calculate_L_from_size(S) # R means number of rows.
            nrows = S[1]
            data = Array{T,N}(undef, setfirstindex(S, R))
            # We need to zero out the excess, so it doesn't interfere
            # with operations like summing the columns or matrix mul.
            # add @generated and quote the expression if you add this back.
            Base.Cartesian.@nloops $(N-1) i j -> 1:S[j+1] begin
                for i_0 = L-R+1:L
                    ( Base.Cartesian.@nref $N out n -> i_{n-1} ) = $(zero(T))
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
mutable struct SizedSIMDArray{S<:Tuple,T,N,R,L} <: AbstractSizedSIMDArray{S,T,N,R,L}
    data::NTuple{L,T}
    function SizedSIMDArray{S,T,N,R,L}(::UndefInitializer) where {S,T,N,R,L}
        new()
    end
    @generated function SizedSIMDArray{S,T}(::UndefInitializer) where {S,T}
        SV = S.parameters
        N = length(SV)
        Stup = ntuple(n -> SV[n], N)#unstable
        R, L = calculate_L_from_size(SV)
        quote
            out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
            Base.Cartesian.@nloops $(N-1) i n -> 1:$Stup[n+1] begin
                @inbounds for i_0 = $(SV[1]+1:R)
                    ( Base.Cartesian.@nref $N out n -> i_{n-1} ) = zero($T)
                end
            end
            out
        end
        # :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
    @generated function SizedSIMDArray{S,T,N}(::UndefInitializer) where {S,T,N}
        SV = S.parameters
        # N = length(SV)
        Stup = ntuple(n -> SV[n], Val(N))
        R, L = calculate_L_from_size(SV)
        quote
            out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
            Base.Cartesian.@nloops $(N-1) i n -> 1:$Stup[n+1] begin
                @inbounds for i_0 = $(SV[1]+1:R)
                    ( Base.Cartesian.@nref $N out n -> i_{n-1} ) = zero($T)
                end
            end
            out
        end
        # :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
end

@generated function SizedSIMDArray(::UndefInitializer, ::Val{S}, ::Type{T}=Float64) where {S,T}
    N = length(S)
    R, L = calculate_L_from_size(S)
    # SD = Expr(:curly, :Tuple, S...)
    SD = Tuple{S...}
    quote
        out = SizedSIMDArray{$SD,$T,$N,$R,$L}(undef)
        Base.Cartesian.@nloops $(N-1) i n -> 1:$S[n+1] begin
            @inbounds for i_0 = $(S[1]+1:R)
                ( Base.Cartesian.@nref $N out n -> i_{n-1} ) = $(zero(T))
            end
        end
        out
    end
    # :(SizedSIMDArray{$SD,$T,$N,$R,$L}(undef))
end

# Not type stable!
function SizedSIMDArray(A::AbstractArray{T,N}) where {T,N}
    out = SizedSIMDArray{Tuple{size(A)...},T,N}(undef)
    copyto!(out, A)
    out
end

@inline full_length(::AbstractSizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L} = L
@inline full_length(x) = length(x)
const SizedSIMDVector{N,T,R,L} = SizedSIMDArray{Tuple{N},T,1,R,L} # R and L will always be the same...
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

@inline Base.unsafe_convert(::Type{Ptr{T}}, A::SizedSIMDArray) where T = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@generated function strides(A::AbstractSizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
    SV = S.parameters
    N = length(SV)
    N == 1 && return (1,)
    last = R
    q = Expr(:tuple, 1, last)
    for n ∈ 3:N
        last *= SV[n-1]
        push!(q.args, last)
    end
    q
end

@generated function Base.size(A::SIMDArray{T,N}) where {T,N}
    quote
        s = size(A.data)
        Base.Cartesian.@nextract $N s s
        s_1 = A.nrows
        Base.Cartesian.@ntuple $N s
    end
end
to_tuple(S) = tuple(S.parameters...)
@generated Base.size(::AbstractSizedSIMDArray{S}) where S = to_tuple(S)


# Do we want this, or L?
@generated Base.length(::AbstractSizedSIMDArray{S}) where S = prod(to_tuple(S))

@inline Base.getindex(A::SIMDArray, i...) = A.data[i...]
@inline Base.setindex!(A::SIMDArray, v, i...) = A.data[i...] = v
@inline function Base.getindex(A::SizedSIMDArray{S,T,1,L,L}, i::Int) where {S,T,L}
    @boundscheck i <= L || throw(BoundsError())
    unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), i)
end
@inline function Base.getindex(A::SizedSIMDArray, i::Int)
    @boundscheck i <= full_length(A) || throw(BoundsError())
    T = eltype(A)
    unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), i)
end
"""
Returns zero based index. Don't forget to add one when using with arrays instead of pointers.
"""
function sub2ind_expr(S, R)
    N = length(S)
    N == 1 && return :(i[1] - 1)
    ex = :(i[$N] - 1)
    for i ∈ (N - 1):-1:2
        ex = :(i[$i] - 1 + $(S[i]) * $ex)
    end
    :(i[1] + $R * $ex)
end

@generated function Base.getindex(A::SizedSIMDArray{S,T,N,R,L}, i::Vararg{Int,N}) where {S,T,N,R,L}
    # dims = ntuple(j -> S.parameters[j], Val(N))
    sv = S.parameters
    ex = sub2ind_expr(sv, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $sv[d]) d->throw(BoundsError()) d -> nothing
        end
        T = eltype(A)
        # unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A) + $(sizeof(T))*($ex) ))
        # A.data[$ex+1]
        unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), $ex)
    end
end
@generated function Base.getindex(A::SizedSIMDArray{S,T,N,R,L}, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(d == 1 ? i[d] > $R : i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), $ex)
    end
end
@inline function Base.setindex!(A::SizedSIMDArray, v, i::Int)
    @boundscheck i <= full_length(A) || throw(BoundsError())
    T = eltype(A)
    unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), i)
    v
end
@inline function Base.setindex!(A::SizedSIMDVector, v, i::Int)
    @boundscheck i <= full_length(A) || throw(BoundsError())
    T = eltype(A)
    unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), i)
    v
end


@generated function Base.setindex!(A::SizedSIMDArray{S,T,N,R,L}, v, i::CartesianIndex{N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d->throw(BoundsError("Dimension $d out of bounds")) d -> nothing
        end
        T = eltype(A)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), $ex)
        v
    end
end
@generated function Base.setindex!(A::SizedSIMDArray{S,T,N,R,L}, v, i::Vararg{Integer,N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->( d == 1 ? i[d] > $R : i[d] > $dims[d]) d->throw(BoundsError("Dimension $d out of bounds $(i[d]) > $R")) d -> nothing
        end
        T = eltype(A)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), $ex)
        v
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

# defauling to IndexCartesian() at the moment
# I want @eachindex when we only have two SIMDArrays to bring us to all elements.
Base.IndexStyle(::Type{<:AbstractSIMDArray}, ::Type{<:AbstractSIMDArray}) = IndexLinear()
# Base.IndexStyle(::AbstractSIMDArray, ::AbstractArray) = IndexCartesian()
# Base.IndexStyle(::AbstractArray, ::AbstractSIMDArray) = IndexCartesian()
# Base.IndexStyle(::AbstractSIMDArray, ::AbstractArray, ::AbstractArray) = IndexCartesian()
# Base.IndexStyle(::AbstractArray, ::AbstractArray, ::AbstractSIMDArray) = IndexCartesian()
# Base.IndexStyle(::AbstractArray, ::AbstractSIMDArray, ::AbstractArray) = IndexCartesian()
# Base.IndexStyle(::AbstractSIMDArray, ::AbstractSIMDArray, ::AbstractArray) = IndexCartesian()
# Base.IndexStyle(::AbstractSIMDArray, ::AbstractArray, ::AbstractSIMDArray) = IndexCartesian()
# Base.IndexStyle(::AbstractArray, ::AbstractSIMDArray, ::AbstractSIMDArray) = IndexCartesian()

# function Base.similar()

"""
This is only meant to make recursive algorithms easiesr to implement.
Wraps a pointer, while passing info on the size of the block and stride.
"""
struct PtrArray{S,T,N,R,L} <: AbstractSizedSIMDArray{S,T,N,R,L}
    ptr::Ptr{T}
    @generated function PtrArray{S,T,N,R}(ptr::Ptr{T}) where {S,T,N,R}
        L = R
        for i ∈ 2:N
            L *= S.parameters[i]
        end
        :(PtrArray{$S,$T,$N,$R,$L}(ptr))
    end
    @generated function PtrArray{S,T}(ptr::Ptr{T}) where {S,T}
        SV = S.parameters
        N = length(SV)
        R, L = calculate_L_from_size(SV)
        :(PtrArray{$S,$T,$N,$R,$L}(ptr))
    end
end
const PtrVector{N,T,R,L} = PtrArray{Tuple{N},T,1,R,L} # R and L will always be the same...
const PtrMatrix{M,N,T,R,L} = PtrArray{Tuple{M,N},T,2,R,L}
# struct PtrMatrix{M,N,T,Stride}
#     ptr::Ptr{T}
# end
@inline Base.pointer(ptr::PtrMatrix) = ptr.ptr
# const SIMDMat{M,N,T,Stride} = Union{SizedSIMDMatrix{M,N,T,Stride},PtrMatrix{M,N,T,Stride}}

@generated function jBLAS.prefetch(A::PtrMatrix{M,P,T,stride}, ::Val{RoW}) where {M,P,T,stride, RoW}
    T_size = sizeof(T)
    T_stride = stride * T_size
    CL = CACHELINE_SIZE ÷ T_size
    Q, r = divrem(M, CL) #Assuming L2M is a multiple of W
    quote
        # for p = 0:$(P-1), q = 0:$(Q-1)
        #     prefetch(A.ptr + q * $CACHELINE_SIZE + $T_stride * p, Val{2}(), Val{$RoW}())
        # end
        for p = 0:$(P-1)
            prefetch(A.ptr + $T_stride * p, Val{2}(), Val{$RoW}())
        end
    end
end



# @generated function Base.similar(::SizedSIMDArray{S},::Type{T}) where {S,T}
#     SV = S.parameters
#     N = length(SV)
#     Stup = ntuple(n -> SV[n], N)#unstable
#     R, L = calculate_L_from_size(SV)
#     quote
#         out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
#         Base.Cartesian.@nloops $(N-1) i n -> 1:$Stup[n+1] begin
#             @inbounds for i_0 = $(SV[1]+1:R)
#                 ( Base.Cartesian.@nref $N out n -> i_{n-1} ) = $(zero(T))
#             end
#         end
#         out
#     end
#     # :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
# end
