


mutable struct DualArray{S<:Tuple,dT,dN,dTag,D <: ForwardDiff.Dual{dTag,dT,dN},N,R,L,L2} <: AbstractSizedSIMDArray{S,D,N,R,L}
    value::NTuple{L,dT}
    partials::NTuple{L2,dT} #L2 = dN * L; dN copies of value.
end
@inline Base.pointer(A::DualArray{S,dT}) = Base.unsafe_convert(Ptr{dT}, pointer_from_objref(A))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::SizedSIMDArray) = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))


@inline full_length(::DualArray{S,T,N,R,L}) where {S,T,N,R,L} = L
@inline full_length(x) = length(x)
const SizedSIMDVector{N,T,L} = SizedSIMDArray{Tuple{N},T,1,L,L}
const SizedSIMDMatrix{M,N,T,R,L} = SizedSIMDArray{Tuple{M,N},T,2,R,L}



### I think the compiler might not like inlining uses of pointer.
### Therefore

@generated function Base.getindex(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, i::Integer) where {S,dT,dN,dTag,D,N,R,L,L2}
    quote
        $(Expr(:meta, :inline))
        @boundscheck full_length(A) < $L
        ptr_A = Base.unsafe_convert(Ptr{$dT} pointer_from_objref(A))
        $D(
            unsafe_load(ptr_A, i),
            $(Expr(:tuple,
                [
                    :(unsafe_load(ptr_A, i + $(L*n)))
                    for n in 1:dN # n = 0 access value; n = 1:N accesses partials.
                ]
            ))
        )
    end
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
        i = $ex + 1
        ptr_A = Base.unsafe_convert(Ptr{$dT} pointer_from_objref(A))
        $D(
            unsafe_load(ptr_A, i),
            $(Expr(:tuple,
                [
                    :(unsafe_load(ptr_A, i + $(L*n)))
                    for n in 1:dN # n = 0 access value; n = 1:N accesses partials.
                ]
            ))
        )
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
        i = $ex + 1
        ptr_A = Base.unsafe_convert(Ptr{$dT} pointer_from_objref(A))
        $D(
            unsafe_load(ptr_A, i),
            $(Expr(:tuple,
                [
                    :(unsafe_load(ptr_A, i + $(L*n)))
                    for n in 1:dN # n = 0 access value; n = 1:N accesses partials.
                ]
            ))
        )
    end
end
@inline function Base.setindex!(A::SizedSIMDArray, v, i::Int)
    @boundscheck i < full_length(A)
    T = eltype(A)
    unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), i)
    v
end
@inline function Base.setindex!(A::SizedSIMDVector, v, i::Int)
    @boundscheck i < full_length(A)
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
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        T = eltype(A)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), $ex+1)
        v
    end
end
@generated function Base.setindex!(A::SizedSIMDArray{S,T,N,R,L}, v, i::Vararg{Integer,N}) where {S,T,N,R,L}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        T = eltype(A)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), $ex+1)
        v
    end
end