
Base.@pure randsimd(S...) = randsimd(Float64, Val{S}())
Base.@pure randsimd(::Type{T}, S...) where T = randsimd(T, Val{S}())
randsimd(::Type{T}, ::Val{S}) where {S,T} = rand!(SizedSIMDArray(undef, Val(S), T))
# function Random.rand!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
#     @inbounds for i ∈ 1:L
#         A[i] = rand(T)
#     end
#     A
# end
@generated function Random.rand!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
    quote
        @inbounds Base.Cartesian.@nloops $N i n -> 1:size(A,n) (Base.Cartesian.@nref $N A i) = rand(T)
        A
    end
end
@generated function Random.rand!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T<:Union{Float32,Float64},N,R,L}
    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(rand(VectorizedRNG.GLOBAL_vPCG, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    if nrep > 0
        q = quote
            ptr_A = pointer(A)
            for i ∈ 1:$(4W):$(nrep * 4W)
                u = $float_q
                $store_expr
            end
        end
    else
        q = quote ptr_A = pointer(A) end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        push!(q.args, :($u_sym = rand(VectorizedRNG.GLOBAL_vPCG, Vec{$(W*rrep),$T}) ))
        for n ∈ 0:rrep - 1
            push!(q.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(u_sym, W, n*W)), $(L - r + 1 + n*W))))
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = rand(VectorizedRNG.GLOBAL_vPCG, Vec{$rrem,$T}) ))
            push!(q.args, :(vstore!(ptr_A, $u_sym, $(L - rrem + 1))))
        end
    end
    push!(q.args, :A)
    q
end
@generated function Random.rand(::Type{<:StaticSIMDArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    N = length(S.parameters)

    R, L = calculate_L_from_size(S.parameters)

    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(rand(VectorizedRNG.GLOBAL_vPCG, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    u_exprs = quote end
    out_exprs = Expr(:tuple,)
    for i ∈ 1:nrep
        u_sym = Symbol(:u_, i)
        push!(u_exprs.args, :($u_sym = $float_q ))
        for j ∈ 1:4W
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        push!(u_exprs.args, :($u_sym = rand(VectorizedRNG.GLOBAL_vPCG, Vec{$(W*rrep),$T}) ))
        for j ∈ 1:W*rrep
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = rand(VectorizedRNG.GLOBAL_vPCG, Vec{$rrem,$T}) ))
            for j ∈ 1:rrem
                push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
            end
        end
    end
    push!(u_exprs.args, :(StaticSIMDArray{$S,$T,$N,$R,$L}($out_exprs)))
    u_exprs
end

function Random.rand(::Type{<: SizedSIMDArray{S,T}}) where {S,T}
    rand!(SizedSIMDArray{S,T}(undef))
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
function Random.randn(::Type{<: SizedSIMDArray{S,T}}) where {S,T}
    randn!(SizedSIMDArray{S,T}(undef))
end

@generated function Random.randn!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T<:Union{Float32,Float64},N,R,L}
    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(randn(VectorizedRNG.GLOBAL_vPCG, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    if nrep > 0
        q = quote
            ptr_A = pointer(A)
            for i ∈ 1:$(4W):$(nrep * 4W)
                u = $float_q
                $store_expr
            end
        end
    else
        q = quote ptr_A = pointer(A) end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        push!(q.args, :($u_sym = randn(VectorizedRNG.GLOBAL_vPCG, Vec{$(W*rrep),$T}) ))
        for n ∈ 0:rrep - 1
            push!(q.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(u_sym, W, n*W)), $(L - r + 1 + n*W))))
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = randn(VectorizedRNG.GLOBAL_vPCG, Vec{$rrem,$T}) ))
            push!(q.args, :(vstore!(ptr_A, $u_sym, $(L - rrem + 1))))
        end
    end
    push!(q.args, :A)
    q
end


@generated function Random.randn(::Type{<:StaticSIMDArray{S,T}}) where {S,T<:Union{Float32,Float64}}
    N = length(S.parameters)

    R, L = calculate_L_from_size(S.parameters)

    size_T = sizeof(T)
    W = VectorizationBase.pick_vector_width(L, T)
    nrep, r = divrem(L, 4W)
    float_q = :(randn(VectorizedRNG.GLOBAL_vPCG, Vec{$(4W),$T}, VectorizedRNG.RXS_M_XS))
    store_expr = quote end
    for n ∈ 0:3
        push!(store_expr.args, :(vstore!(ptr_A, $(VectorizedRNG.subset_vec(:u, W, n*W)), i + $(n*W))))
    end
    u_exprs = quote end
    out_exprs = Expr(:tuple,)
    for i ∈ 1:nrep
        u_sym = gensym(Symbol(:u_, i))
        push!(u_exprs.args, :($u_sym = $float_q ))
        for j ∈ 1:4W
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
    end
    if r > 0
        rrep, rrem = divrem(r, W)
        u_sym = gensym(Symbol(:u_remw))
        push!(u_exprs.args, :($u_sym = randn(VectorizedRNG.GLOBAL_vPCG, Vec{$(W*rrep),$T}) ))
        for j ∈ 1:W*rrep
            push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
        end
        if rrem > 0
            u_sym = gensym(:u_rem)
            push!(u_exprs.args, :($u_sym = randn(VectorizedRNG.GLOBAL_vPCG, Vec{$rrem,$T}) ))
            for j ∈ 1:rrem
                push!(out_exprs.args, :( @inbounds $u_sym[$j].value ) )
            end
        end
    end
    push!(u_exprs.args, :(StaticSIMDArray{$S,$T,$N,$R,$L}($out_exprs)))
    u_exprs
end
# function Random.randn!(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}
#     @inbounds for i ∈ 1:L
#         A[i] = randn(T)
#     end
#     A
# end

function rand_expr(expr, R = :StaticSIMDArray)
    N = length(expr.args)
    n = 2
    if isa(expr.args[2], Int)
        T = Float64
    else
        T = expr.args[2]
        n += 1
    end
    S = Tuple{expr.args[n:end]...}
    return :( $(expr.args[1])( $(R){$S, $T}  )  )
end

macro Static(expr)
    rand_expr(expr, :StaticSIMDArray)
end
macro Sized(expr)
    rand_expr(expr, :SizedSIMDArray)
end
