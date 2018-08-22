@generated function Base.zero(::Type{<:SizedSIMDArray{S,T}}) where {S,T}
    SV = S.parameters
    N = length(SV)
    Stup = ntuple(n -> SV[n], N)#unstable
    R, L = calculate_L_from_size(SV)
    quote
        out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
        for i ∈ 1:$L
            out[i] = zero($T)
        end
        out
    end
end

@generated function Base.similar(::SizedSIMDArray{S,T}, v::Vararg{Int,N}) where {S,T,N}
    :(SizedSIMDArray(undef, Val(v), $T))
end
Base.similar(::SizedSIMDArray{S,T}) where {S,T} = SizedSIMDArray{S,T}(undef)
Base.similar(::SizedSIMDArray{S},::Type{T}) where {S,T} = SizedSIMDArray{S,T}(undef)
@generated function Base.copyto!(out::SizedSIMDArray{S,T,N,R,L}, A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
    T_size = sizeof(T)
    VL = min(REGISTER_SIZE ÷ T_size,L)
    V = Vec{VL,T}
    num_reps = L ÷ VL
    LT = L * T_size
    VLT = VL * T_size
    quote
        ptr_out, ptr_A = pointer(out), pointer(A)
        @inbounds for i ∈ 0:$VLT:$(LT-VLT)
            vstore(vload($V, ptr_A + i), ptr_out + i)
        end
        out
    end
end
const SymmetricMatrix{P,T} = Symmetric{T, <: SizedSIMDMatrix{P,P,T}}
function Base.setindex!(S::SymmetricMatrix{P,T}, v, i::Integer, j::Integer) where {P,T}
    @boundscheck begin
        i, j = minmax(i, j)
        (j > P || i < 1) && throw(BoundsError())
    end
    S.data[i,j] = v
end

Base.copy(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L} = copyto!(SizedSIMDArray{S,T,N,R,L}(undef), A)

@generated function scale!(out::SizedSIMDArray{S,T,N,R,L}, val::T) where {S,T,N,R,L}
    T_size = sizeof(T)
    VL = min(REGISTER_SIZE ÷ T_size,L)
    V = Vec{VL,T}
    num_reps = L ÷ VL
    LT = L * T_size
    VLT = VL * T_size
    quote
        ptr_out = pointer(out)
        v = $V(val)
        @inbounds for i ∈ 0:$VLT:$(LT-VLT)
            vstore(vload($V, ptr_out + i) * v, ptr_out + i)
        end
        out
    end
end

function LinearAlgebra.dot(A::SizedSIMDArray{S,T,N,R,L}, B::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
    out = zero(T)
    @inbounds @simd for i ∈ 1:L
        out += x[i] * y[i]
    end
    out
end
