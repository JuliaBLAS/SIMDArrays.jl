
Base.@pure randsimd(S...) = randsimd(Float64, Val{S}())
Base.@pure randsimd(::Type{T}, S...) where T = randsimd(T, Val{S}())
randsimd(::Type{T}, ::Val{S}) where {S,T} = rand!(SizedSIMDArray(undef, Val(S), T))
# function Random.rand!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
#     @inbounds for i ∈ 1:L
#         A[i] = rand(T)
#     end
#     A
# end
@generated function Random.rand!(A::SizedSIMDArray{S,T,N}) where {S,T,N}
    quote
        @inbounds Base.Cartesian.@nloops $N i n -> 1:size(A,n) (Base.Cartesian.@nref $N A i) = rand(T)
        A
    end
end


Base.@pure randnsimd(S...) = randnsimd(Float64, Val{S}())
Base.@pure randnsimd(::Type{T}, S...) where T  = randnsimd(T, Val{S}())
randnsimd(::Type{T}, ::Val{S}) where {S,T} = randn!(SizedSIMDArray(undef, Val(S), T))
@generated function Random.randn!(A::SizedSIMDArray{S,T,N}) where {S,T,N}
    quote
        @inbounds Base.Cartesian.@nloops $N i n -> 1:size(A,n) (Base.Cartesian.@nref $N A i) = randn(T)
        A
    end
end
# function Random.randn!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
#     @inbounds for i ∈ 1:L
#         A[i] = randn(T)
#     end
#     A
# end
