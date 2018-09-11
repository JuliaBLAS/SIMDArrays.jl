


mutable struct DualArray{S<:Tuple,dT,dN,dTag,D <: ForwardDiff.Dual{dTag,dT,dN},N,R,L,L2} <: AbstractSizedSIMDArray{S,D,N,R,L}
    value::NTuple{L,dT}
    partials::NTuple{L2,dT} #L2 = dN * L; dN copies of value.
    function DualArray{S,dT,dN,dTag,D,N,R,L,L2}(::UndefInitializer) where {S,dT,dN,dTag,D,N,R,L,L2}
        new()
    end
    @generated function DualArray{S,T,D}(::UndefInitializer) where {S,T,D <: ForwardDiff.Dual}
        SV = S.parameters
        N = length(SV)
        Stup = ntuple(n -> SV[n], N)#unstable
        R, L = calculate_L_from_size(SV)
        quote
            out = DualArray{S,T,D}(undef)
            Base.Cartesian.@nloops $(N-1) i n -> 1:$Stup[n+1] begin
                @inbounds for i_0 = $(SV[1]+1:R)
                    ( Base.Cartesian.@nref $N out n -> i_{n-1} ) = zero($T)
                end
            end
            out
        end
        # :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
    @generated function DualArray{S,T,N}(::UndefInitializer) where {S,T,N}
        SV = S.parameters
        N = length(SV)
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
@inline Base.pointer(A::DualArray{S,dT}) where {S,dT} = Base.unsafe_convert(Ptr{dT}, pointer_from_objref(A))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::DualArray) where T = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))

struct GradientConfig{T,V,N,D} <: ForwardDiff.AbstractConfig{N}
    seeds::NTuple{N,Partials{N,V}}
    duals::D
end

@inline full_length(::DualArray{S,T,N,R,L}) where {S,T,N,R,L} = L
const DualVector{N,T,L} = DualArray{Tuple{N},T,1,L,L}
const DualMatrix{M,N,T,R,L} = DualArray{Tuple{M,N},T,2,R,L}



### I think the compiler might not like inlining uses of pointer.
### Therefore

@generated function Base.getindex(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, i::Integer) where {S,dT,dN,dTag,D,N,R,L,L2}
    quote
        $(Expr(:meta, :inline))
        @boundscheck full_length(A) < $L
        ptr_A = Base.unsafe_convert(Ptr{$dT}, pointer_from_objref(A))
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



@generated function Base.getindex(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, i::Vararg{Int,N}) where {S,dT,dN,dTag,D,N,R,L,L2}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        i = $ex + 1
        ptr_A = Base.unsafe_convert(Ptr{$dT}, pointer_from_objref(A))
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
@generated function Base.getindex(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, i::CartesianIndex{N}) where {S,dT,dN,dTag,D,N,R,L,L2}
    dims = ntuple(j -> S.parameters[j], Val(N))
    ex = sub2ind_expr(dims, R)
    quote
        $(Expr(:meta, :inline))
        # @show i, S
        @boundscheck begin
            Base.Cartesian.@nif $(N+1) d->(i[d] > $dims[d]) d->throw(BoundsError()) d -> nothing
        end
        i = $ex + 1
        ptr_A = Base.unsafe_convert(Ptr{$dT}, pointer_from_objref(A))
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
@generated function Base.setindex!(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, v::D, i::Int) where {S,dT,dN,dTag,D,N,R,L,L2}
    quote
        @boundscheck i < full_length(A)
        T = eltype(A)

        ptr_A = Base.unsafe_convert(Ptr{$dT}, pointer_from_objref(A))
        unsafe_store!(ptr_A, v.value, i)
        @nexprs $dN n -> begin
            unsafe_store!(ptr_A, v.partials[n], i + $(L*n))
        end
        v
    end
end
@generated function Base.setindex!(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, v, i::Int) where {S,dT,dN,dTag,D,N,R,L,L2}
    @boundscheck i < full_length(A)
    T = eltype(A)
    unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), convert(T,v), i)
    v
end


@generated function Base.setindex!(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, v, i::CartesianIndex{N}) where {S,dT,dN,dTag,D,N,R,L,L2}
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
@generated function Base.setindex!(A::DualArray{S,dT,dN,dTag,D,N,R,L,L2}, v, i::Vararg{Integer,N}) where {S,dT,dN,dTag,D,N,R,L,L2}
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










mutable struct GradientDiffResult{V,G} <: DiffResults.DiffResult{1,V,Tuple{G}}
    value::V
    grad::G
end
mutable struct HessianDiffResult{V,G,H} <: DiffResults.DiffResult{2,V,Tuple{G,H}}
    value::V
    grad::G
    hess::H
end
const GradResult{V,G} = Union{GradientDiffResult{V,G},HessianDiffResult{V,G}}
@inline DiffResults.derivative(result::GradResult) = result.grad
@inline DiffResults.derivative(result::GradResult, ::Type{Val{1}}) = result.grad
@inline DiffResults.derivative(result::HessianDiffResult, ::Type{Val{2}}) = result.hess
@inline DiffResults.gradient(result::GradResult) = result.grad
@inline DiffResults.hessian(result::HessianDiffResult) = result.hess

@inline DiffResults.value!(r::GradResult, x::Number) = (r.value = x; return r)
# @inline DiffResults.value!(r::GradResult, x::AbstractArray) = (copyto!(value(r), x); return r)
@inline DiffResults.value!(f, r::GradResult, x::Number) = (r.value = f(x); return r)
# @inline DiffResults.value!(f, r::GradResult, x::AbstractArray) = (map!(f, value(r), x); return r)

function DiffResults.derivative!(r::GradResult, x::Number)
    r.derivs = tuple_setindex(r.derivs, x, Val{i})
    return r
end
function DiffResults.derivative!(r::GradResult, x::AbstractArray)
    copyto!(r.grad, x)
    return r
end
function DiffResults.derivative!(r::GradResult, x::AbstractArray, ::Type{Val{1}})
    copyto!(r.grad, x)
    return r
end
function DiffResults.derivative!(r::HessianDiffResult, x::AbstractArray, ::Type{Val{2}})
    copyto!(r.hess, x)
    return r
end
function DiffResults.derivative!(f, r::GradResult, x::Number)
    r.grad = f(x)
    return r
end
function DiffResults.derivative!(f, r::GradResult, x::AbstractArray)
    map!(f, r.grad, x)
    return r
end


function GradientDiffResult(x::SizedVector{P,T}) where {T,P}
    GradientDiffResult(zero(T), similar(x))
end
function HessianDiffResult(x::SizedSIMDVector{P,T}) where {T,P}
    HessianDiffResult(zero(T), similar(x), Symmetric(SizedSIMDMatrix{P,P,T}(undef)))
end
function HessianDiffResult(x::MVector{P,T}) where {T,P}
    HessianDiffResult(zero(T), similar(x), Symmetric(MMatrix{P,P,T}(undef)))
end

abstract type AutoDiffDifferentiable{P,T,F} <: DifferentiableObject{P} end
abstract type Configuration{P,V,F} end

struct GradientConfiguration{P,V,F,T,ND,DG,SV_V <: SizedVector{P,V}} <: Configuration{P,V,F}
    f::F
    result::GradientDiffResult{V,SV_V}
    gconfig::ForwardDiff.GradientConfig{T,V,ND,DG}
end

struct HessianConfiguration{P,V,F,T,T2,ND,DJ,DG,DG2,P2, SV_V <: SizedVector{P,V}, SQM <: SizedSquareMatrix{P,V},
                                        SV_D <: SizedVector{P,ForwardDiff.Dual{T,V,ND}} } <: Configuration{P,V,F}
    f::F
    result::HessianDiffResult{V,SV_V,Symmetric{V,SQM}}
    inner_result::GradientDiffResult{ForwardDiff.Dual{T,V,ND},SV_D}
    jacobian_config::ForwardDiff.JacobianConfig{T,V,ND,DJ}
    gradient_config::ForwardDiff.GradientConfig{T,ForwardDiff.Dual{T,V,ND},ND,DG}
    gconfig::ForwardDiff.GradientConfig{T2,V,ND,DG2}
end
struct TwiceDifferentiable{P,T,F,C<:Configuration{P,T,F}, SV_V <: SizedVector{P,T}} <: AutoDiffDifferentiable{P,T,F}
    x_f::SV_V # x used to evaluate f (stored in F)
    x_df::SV_V # x used to evaluate df (stored in DF)
    x_h::SV_V #??
    config::C
end
struct OnceDifferentiable{P,T,F,C<:GradientConfiguration{P,T,F}, SV_V <: SizedVector{P,T}} <: AutoDiffDifferentiable{P,T,F}
    x_f::SV_V# x used to evaluate f (stored in F)
    x_df::SV_V # x used to evaluate df (stored in DF)
    config::C
end


function GradientConfiguration(f::F, x::SizedVector{P,T}) where {F,T,P}
    result = GradientDiffResult(x)
    chunk = Chunk(Val{P}())
    tag = ForwardDiff.Tag(f, T)
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)

    GradientConfiguration(f, result, gconfig)
end
function HessianConfiguration(f::F, x::SizedVector{P,T}) where {F,T,P}
    result = HessianDiffResult(x)
    chunk = Chunk(Val{P}())
    tag = ForwardDiff.Tag(f, T)

    jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    inner_result = GradientDiffResult(jacobian_config.duals[2])
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)

    HessianConfiguration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
end
TwiceDifferentiable(f::F, ::Val{P}) where {F,P} = TwiceDifferentiable(f, SizedSIMDVector{P,Float64}(undef))
function TwiceDifferentiable(f::F, x::SV_V) where {F,T,P,SV_V <: SizedVector{P,T}}
    TwiceDifferentiable(x, similar(x), similar(x), HessianConfiguration(f, x))
end
function TwiceDifferentiable(x_f::SV_V,x_df::SV_V,x_h::SV_V,config::C) where {T,F,C<:Configuration{T,F},P,SV_V <: SizedVector{P,T}}
    TwiceDifferentiable{P,T,F,C}(x_f, x_df, x_h, config)
end



function OnceDifferentiable(f, x_f::SizedVector{P,T}) where {T,P}
    OnceDifferentiable(x_f, similar(x_f), GradientConfiguration(f, x_f))
end
function OnceDifferentiable(x_f::SV_V, config::C) where {T,F,C<:GradientConfiguration{T,F},P,SV_V <: SizedVector{P,T}}
    OnceDifferentiable{P,T,F,C}(x_f, similar(x_f), config)
end

# ProfileDifferentiable(f::F, ::Val{N}) where {F,N} = ProfileDifferentiable(f, Vector{Float64}(undef, N), Val{N}())
# function ProfileDifferentiable(f::F, x::AbstractArray{T}, ::Val{N}) where {F,T,N}
#     x_f = Vector{T}(undef, N-1)
#     ProfileDifferentiable(x_f, similar(x_f), similar(x_f), Configuration(f, x_f, ValM1(Val{N}())), x, Ref(N), Ref{T}(),::Val{N})
# end
# function ProfileDifferentiable(x_f::Vector{T},x_df::Vector{T},x_h::Vector{T}, config::C, x,::A i::RefValue{Int}, v::RefValue{T},::Val{N}) where {T,A<:AbstractArray{T},C,N}
#     ProfileDifferentiable{N,T,A,C}(x_f, x_df, x_h, config, x, i, v)
# end


@inline DiffResults.value!(obj::AutoDiffDifferentiable, x::Real) = DiffResults.value!(obj.config.result, x)

@inline value(obj::AutoDiffDifferentiable) = obj.config.result.value
@inline gradient(obj::AutoDiffDifferentiable) = obj.config.result.grad
# @inline gradient(obj::AutoDiffDifferentiable, i::Integer) = DiffResults.gradient(obj.config.result)[i]
@inline hessian(obj::TwiceDifferentiable) = obj.config.result.hess


@inline f(obj::AutoDiffDifferentiable, x) = obj.config.f(x)

"""
For DynamicHMC support. Copies data.
"""
@inline function (obj::OnceDifferentiable)(x)
    fdf(obj, x)
    DiffResults.ImmutableDiffResult(-obj.config.result.value, (-1 .* obj.config.result.derivs[1],))
end
@inline function (obj::TwiceDifferentiable)(x)
    fdf(obj, x)
    DiffResults.ImmutableDiffResult(-obj.config.result.value, (-1 .* obj.config.result.derivs[1],))
end

@inline function df(obj::AutoDiffDifferentiable, x)
    ForwardDiff.gradient!(gradient(obj), obj.config.f, x, obj.config.gconfig, Val{false}())
end
# @noinline set_gradient!(res, ∇) = res.derivs = ∇ #obj.config.result.derivs = (gradient(obj),)
# @noinline set_hessian!(res, ∇Λ) = res.derivs = ∇Λ # obj.config.result.derivs = (gradient(obj), hessian(obj))
@inline function fdf(obj::TwiceDifferentiable, x)
    obj.config.result.grad = gradient(obj)
    obj.config.result.hess = hessian(obj)
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    # DiffResults.value(obj.config.result)
    obj.config.result.value
end
@inline function fdf(obj::OnceDifferentiable, x)
    obj.config.result.grad = gradient(obj)
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    # DiffResults.value(obj.config.result)
    obj.config.result.value
end

# function Configuration(f::F, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)) where {F,V}
#     result = DiffResults.HessianResult(x)
#     jacobian_config = ForwardDiff.JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
#     gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
#     inner_result = DiffResults.DiffResult(zero(eltype(jacobian_config.duals[2])), jacobian_config.duals[2])
#     gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)
#     Configuration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
# end

@inline function (c::HessianConfiguration)(y, z)
    c.inner_result.derivs = (y,) #Already true?
    ForwardDiff.gradient!(c.inner_result, c.f, z, c.gradient_config, Val{false}())
    DiffResults.value!(c.result, ForwardDiff.value(DiffResults.value(c.inner_result)))
    y
end
@inline function hessian!(c::HessianConfiguration, x::AbstractArray)
    ForwardDiff.jacobian!(DiffResults.hessian(c.result), c, DiffResults.gradient(c.result), x, c.jacobian_config, Val{false}())
    # DiffResults.hessian(c.result)
    c.result.hess
end

"""
Force (re-)evaluation of the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
@inline function value!!(obj::AutoDiffDifferentiable, x)
    # obj.f_calls .+= 1
    # copyto!(obj.x_f, x)
    DiffResults.value!(obj, f(obj, x) )
    value(obj)
end
# """
# Evaluates the objective value at `x`.
#
# Returns `f(x)`, but does *not* store the value in `obj.F`
# """
# @inline function value(obj::AutoDiffDifferentiable, x)
#     if x != obj.x_f
#         # obj.f_calls .+= 1
#         value!!(obj, x)
#     end
#     value(obj)
# end
"""
Evaluates the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
@inline function value!(obj::AutoDiffDifferentiable, x)
    # if x != obj.x_f
        value!!(obj, x)
    # end
    value(obj)
end

"""
Evaluates the gradient value at `x`

This does *not* update `obj.DF`.
"""
@inline function gradient(obj::AutoDiffDifferentiable, x)
    DF = gradient(obj)
    # if x != obj.x_df
        tmp = copy(DF)
        gradient!!(obj, x)
        @inbounds for i ∈ eachindex(tmp)
            tmp[i], DF[i] = DF[i], tmp[i]
        end
        return tmp
    # end
    DF
end
"""
Evaluates the gradient value at `x`.

Stores the value in `obj.DF`.
"""
@inline function gradient!(obj::AutoDiffDifferentiable, x)
    # if x != obj.x_df
        gradient!!(obj, x)
    # end
    gradient(obj)
end
"""
Force (re-)evaluation of the gradient value at `x`.

Stores the value in `obj.DF`.
"""
@inline function gradient!!(obj::AutoDiffDifferentiable, x)
    # obj.df_calls .+= 1
    copyto!(obj.x_df, x)
    df(obj, x)
end

@inline function value_gradient!(obj::AutoDiffDifferentiable, x)
    # if x != obj.x_f && x != obj.x_df
        value_gradient!!(obj, x)
    # elseif x != obj.x_f
    #     value!!(obj, x)
    # elseif x != obj.x_df
    #     gradient!!(obj, x)
    # end
    value(obj)
end
@inline function value_gradient!!(obj::AutoDiffDifferentiable, x)
    # obj.f_calls .+= 1
    # obj.df_calls .+= 1
    # copyto!(obj.x_f, x)
    # copyto!(obj.x_df, x)
    DiffResults.value!(obj, fdf(obj, x))
end

@inline function hessian!(obj::AutoDiffDifferentiable, x)
    # if x != obj.x_h
        hessian!!(obj, x)
    # end
    nothing
end
@inline function hessian!!(obj::AutoDiffDifferentiable, x)
    # obj.h_calls .+= 1
    # copyto!(obj.x_h, x)
    hessian!(obj.config, x)
end



# function ForwardDiff.vector_mode_dual_eval(f::F, x::SizedSIMDVector, cfg::Union{JacobianConfig,GradientConfig}) where F
#     xdual = cfg.duals
#     seed!(xdual, x, cfg.seeds)
#     return f(xdual)
# end

@inline function ForwardDiff.seed!(duals::SizedSIMDVector{P,ForwardDiff.Dual{T,V,N}}, x,
                                    seeds::NTuple{N,ForwardDiff.Partials{N,V}}) where {T,V,N,P}
    @inbounds for i in 1:N
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
    return duals
end

@inline function ForwardDiff.seed!(duals::SizedSIMDVector{P,ForwardDiff.Dual{T,V,N}}, x, index,
                                    seed::ForwardDiff.Partials{N,V} = zero(Partials{N,V})) where {T,V,N,P}
    offset = index - 1
    @inbounds for i in 1:N
        j = i + offset
        duals[j] = ForwardDiff.Dual{T,V,N}(x[j], seed)
    end
    return duals
end
