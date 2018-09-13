#
# These algorithms are naive
# do not yet expect reasonable performance.
# I implemented them because I actively needed them.
#
# These algorithms are currently NOT included. That is, this code is NOT run.
#
#

sym(A, i, j) = Symbol(A, :_, i, :_, j)
symi(A, i, j) = Symbol(A, :i_, i, :_, j)

function load_U_quote!(qa, P, symbol_name = :U, extract_from = :U)
    for c ∈ 1:P, r ∈ 1:c
        push!(qa, :($(sym(symbol_name, r, c)) = $extract_from[$r, $c]) )
    end
end
function invert_U_load_quote!(qa, P, output = :U, input = :U, ::Type{T} = Float64) where T
    for c ∈ 1:P
        for r ∈ 1:c-1
            push!(qa, :($(sym(input, r, c)) = $input[$r, $c]) )
        end
        push!(qa, :($(sym(output, c, c)) = $(one(T))/$input[$c, $c] ) )
    end
end
function invert_U_core_quote!(qa, P, output = :U, input = :U) # N x N block, with stride S.
    for c ∈ 1:P
        for r ∈ c-1:-1:1
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, r+1)) * $(sym(output, r+1, c)) ))
            for rc ∈ r+2:c
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, rc)) * $(sym(output, rc, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  $(sym(output, r, r)) ) )
        end
    end
end
function store_U_quote!(qa, P, destination = :U, insert = :U)
    for c ∈ 1:P
        for r ∈ 1:c-1
            push!(qa, :($destination[$r, $c] = $(sym(insert, r, c))))
        end
        push!(qa, :($destination[$c, $c] = $(sym(insert, c, c))) )
    end
    push!(qa, :(nothing))
end

@generated function inv_U_triangle!(Ui::AbstractMatrix{T}, U::AbstractMatrix{T}, ::Val{P}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    invert_U_load_quote!(qa, P, :Ui, :U, T)
    invert_U_core_quote!(qa, P, :Ui, :U)
    store_U_quote!(qa, P, :Ui, :Ui)
    q
end

# function chol_U_core_quote!(qa, P, output = :U, ::Type{T} = Float64) where T
#     for c ∈ 1:P
#         for r ∈ 1:c-1
#
#         end
#         for cr ∈ 1:c-1, r ∈ c:P
#             push!(qa, :( $(sym(output, r, c)) -= $(sym(output, r, cr)) * $(sym(output, c, cr))  ) )
#         end
#         push!(qa, :( $(sym(output, c, c)) = sqrt( $(sym(output, c, c)) ) ) )
#         push!(qa, :( $(symi(output, c, c)) = $(one(T)) / $(sym(output, c, c)) ))
#         for r ∈ c+1:P
#             push!(qa, :( $(sym(output, r, c)) *= $(symi(output, c, c))  ) )
#         end
#     end
# end
function chol_U_core_quote!(qa, P, output = :U, ::Type{T} = Float64) where T
    @inbounds begin
        for r = 1:P
            for c = 1:r - 1
                push!(qa, :($(sym(output, r, r)) -= $(sym(output, c, r)) * $(sym(output, c, r))) )
            end
            push!(qa, :($(sym(output, r, r)) = sqrt($(sym(output, r, r)))) )
            push!(qa, :($(symi(output, r, r)) = $(one(T))/$(sym(output, r, r))))
            for c = r + 1:P
                for i = 1:r - 1
                    push!(qa, :($(sym(output, r, c)) -= $(sym(output, i, r)) * $(sym(output, i, c))) )
                end
                push!(qa, :($(sym(output, r, c)) *= $(symi(output, r, r))) )
            end
        end
    end
end
@generated function cholesky_U!(U::AbstractMatrix{T}, S::AbstractMatrix{T}, ::Val{P}) where {T,P}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_U_quote!(qa, P, :U, :S)
    chol_U_core_quote!(qa, P, :U, T)
    store_U_quote!(qa, P, :U, :U)
    q
end



function load_L_quote!(qa, P, symbol_name = :L, extract_from = :L)
    for c ∈ 1:P, r ∈ c:P
        push!(qa, :($(sym(symbol_name, r, c)) = $extract_from[$r, $c]) )
    end
end
function invert_diag_quote!(qa, P, output = :Li, input = :L, ::Type{T} = Float64) where T
    for p ∈ 1:P
        push!(qa, :($(sym(output,p,p)) =  $(one(T)) / $(sym(input,p,p))))
    end
end
## Lower triangular seems faster than upper triangular.
function invert_L_load_quote!(qa, P, output = :Li, input = :L, ::Type{T} = Float64) where T # N x N block, with stride S.
    for c ∈ 1:P
        push!(qa, :($(sym(output, c, c)) = $(one(T)) / $input[$c, $c] ) )
        for r ∈ c+1:P
            push!(qa, :($(sym(input, r, c)) = $input[$r, $c]) )
        end
    end
end
function invert_L_core_quote!(qa, P, output = :Li, input = :L) # N x N block, with stride S.
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
function store_L_quote!(qa, P, output = :Li) # N x N block, with stride S.
    for c ∈ 1:P
        push!(qa, :($output[$c, $c] = $(sym(output, c, c))) )
        for r ∈ c+1:P
            push!(qa, :($output[$r, $c] = $(sym(output, r, c))))
        end
    end
    push!(qa, :(nothing))
end
function store_L_prepad_quote!(qa, P, pad, output = :Li) # N x N block, with stride S.
    for c ∈ 1:P
        push!(qa, :($output[$c, $c] = $(sym(output, c, c))) )
        for r ∈ c+1:P
            push!(qa, :($output[$(r+pad), $c] = $(sym(output, r, c))))
        end
    end
    push!(qa, :(nothing))
end
@generated function inv_L_triangle!(Li::SizedSIMDMatrix{P,P,T}, L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    invert_L_load_quote!(qa, P, :Li, :L, T)
    invert_L_core_quote!(qa, P, :Li, :L)
    store_L_quote!(qa, P, :Li)
    q
end




function chol_L_core_quote!(qa, P, output = :L, ::Type{T} = Float64) where T
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(output, r, c)) -= $(sym(output, r, cr)) * $(sym(output, c, cr))  ) )
        end
        push!(qa, :( $(sym(output, c, c)) = sqrt( $(sym(output, c, c)) ) ) )
        push!(qa, :( $(symi(output, c, c)) = $(one(T)) / $(sym(output, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) *= $(symi(output, c, c))  ) )
        end
    end
end
@generated function LinearAlgebra.cholesky!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :S)
    chol_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    q
end
@generated function LinearAlgebra.cholesky!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    chol_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    q
end
@generated function LinearAlgebra.cholesky!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    chol_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    q
end
function safechol_L_core_quote!(qa, P, output = :L, ::Type{T} = Float64) where T
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(output, r, c)) -= $(sym(output, r, cr)) * $(sym(output, c, cr))  ) )
        end
        push!(qa, :( $(sym(output, c, c)) > $(zero(T)) || return false ))
        push!(qa, :( $(sym(output, c, c)) = sqrt( $(sym(output, c, c)) ) ) )
        push!(qa, :( $(symi(output, c, c)) = $(one(T)) / $(sym(output, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) *= $(symi(output, c, c))  ) )
        end
    end
end
@generated function safecholesky!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :S)
    safechol_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(true))
    q
end
@generated function safecholesky!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safechol_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(true))
    q
end
@generated function safecholesky!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safechol_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(true))
    q
end

function choldet_L_core_quote!(qa, P, output = :L, ::Type{T} = Float64) where T
    push!(qa, :(cdet = $(one(T))))
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(output, r, c)) -= $(sym(output, r, cr)) * $(sym(output, c, cr))  ) )
        end
        push!(qa, :( $(sym(output, c, c)) = sqrt( $(sym(output, c, c)) ) ) )
        push!(qa, :(cdet *= $(sym(output, c, c)) ))
        push!(qa, :( $(symi(output, c, c)) = $(one(T)) / $(sym(output, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) *= $(symi(output, c, c))  ) )
        end
    end
end
@generated function choldet!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :S)
    choldet_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet))
    q
end
@generated function choldet!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    choldet_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet))
    q
end
@generated function choldet!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    choldet_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet))
    q
end
function safecholdet_L_core_quote!(qa, P, output = :L, ::Type{T} = Float64) where T
    push!(qa, :(cdet = $(one(T))))
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(output, r, c)) -= $(sym(output, r, cr)) * $(sym(output, c, cr))  ) )
        end
        push!(qa, :( $(sym(output, c, c)) > $(zero(T)) || return ($(-T(Inf)), false) ))
        push!(qa, :( $(sym(output, c, c)) = sqrt( $(sym(output, c, c)) ) ) )
        push!(qa, :(cdet *= $(sym(output, c, c)) ))
        push!(qa, :( $(symi(output, c, c)) = $(one(T)) / $(sym(output, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) *= $(symi(output, c, c))  ) )
        end
    end
end
@generated function safecholdet!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :S)
    safecholdet_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet, true))
    q
end
@generated function safecholdet!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safecholdet_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet, true))
    q
end
@generated function safecholdet!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safecholdet_L_core_quote!(qa, P, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet, true))
    q
end



function invchol_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $(sym(output, c, c))  ) )
        end
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
@generated function invchol!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :S, :S)
    invchol_L_core_quote!(qa, P, :L, :S, T)
    store_L_quote!(qa, P, :L)
    q
end
@generated function invchol!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    invchol_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    q
end
@generated function invchol!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    invchol_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    q
end
function invcholdet_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    push!(qa, :(cdet = $(one(T))))
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( cdet *=  $(sym(input, c, c)) ))
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $(sym(output, c, c))  ) )
        end
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
# @generated function invcholdet!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
#     @assert P <= M
#     q = quote @fastmath @inbounds begin end end
#     qa = q.args[2].args[3].args[3].args
#     load_L_quote!(qa, P, :S, :S)
#     invcholdet_L_core_quote!(qa, P, :L, :S, T)
#     store_L_quote!(qa, P, :L)
#     push!(qa, :(cdet))
#     q
# end
@generated function invcholdet!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    invcholdet_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet))
    q
end
@generated function invcholdet!(L::SizedSIMDMatrix{MpPad,M,T}, S::SizedSIMDMatrix{N,N,T}) where {MpPad,M,N,T}
    P = min(M,N)
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :S, :S)
    invcholdet_L_core_quote!(qa, P, :L, :S, T)
    if M == MpPad
        store_L_quote!(qa, P, :L)
    else
        store_L_prepad_quote!(qa, P, MpPad - P, :L)
    end
    push!(qa, :(cdet))
    q
end
# @generated function invcholdet!(L::SizedSIMDMatrix{M,M,T}, S::SizedSIMDMatrix{N,N,T}) where {M,N,T}
#     P = min(M,N)
#     q = quote @fastmath @inbounds begin end end
#     qa = q.args[2].args[3].args[3].args
#     load_L_quote!(qa, P, :S, :S)
#     invcholdet_L_core_quote!(qa, P, :L, :S, T)
#     store_L_quote!(qa, P, :L)
#     push!(qa, :(cdet))
#     q
# end
@generated function invcholdet!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    invcholdet_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet))
    q
end
function safeinvcholdet_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    push!(qa, :(cdet = $(one(T))))
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) > $(zero(T)) || return ($(-T(Inf)), false) ))
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( cdet *=  $(sym(input, c, c)) ))
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $(sym(output, c, c))  ) )
        end
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
@generated function safeinvcholdet!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :S, :S)
    safeinvcholdet_L_core_quote!(qa, P, :L, :S, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet, true))
    q
end
@generated function safeinvcholdet!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safeinvcholdet_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet, true))
    q
end
@generated function safeinvcholdet!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safeinvcholdet_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(cdet, true))
    q
end
function safeinvchol_L_core_quote!(qa, P, output = :L, input = :S, ::Type{T} = Float64) where T
    for c ∈ 1:P
        for cr ∈ 1:c-1, r ∈ c:P
            push!(qa, :( $(sym(input, r, c)) -= $(sym(input, r, cr)) * $(sym(input, c, cr))  ) )
        end
        push!(qa, :( $(sym(input, c, c)) > $(zero(T)) || return false ))
        push!(qa, :( $(sym(input, c, c)) = sqrt( $(sym(input, c, c)) ) ) )
        push!(qa, :( $(sym(output, c, c)) = $(one(T)) / $(sym(input, c, c)) ))
        for r ∈ c+1:P
            push!(qa, :( $(sym(input, r, c)) *= $(sym(output, c, c))  ) )
        end
    end
    for c ∈ 1:P
        for r ∈ c+1:P
            push!(qa, :( $(sym(output, r, c)) = $(sym(input, r, c)) * $(sym(output, c, c)) ))
            for cr ∈ c+1:r-1
                push!(qa, :( $(sym(output, r, c)) += $(sym(input, r, cr)) * $(sym(output, cr, c)) ))
            end
            push!(qa, :( $(sym(output, r, c)) *=  -$(sym(output, r, r)) ) )
        end
    end
end
@generated function safeinvchol!(L::SizedSIMDMatrix{P,P,T}, S::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :S, :S)
    safeinvchol_L_core_quote!(qa, P, :L, :S, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(true))
    q
end
@generated function safeinvchol!(L::SizedSIMDMatrix{P,P,T}) where {P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safeinvchol_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(true))
    q
end
@generated function safeinvchol!(L::SizedSIMDMatrix{M,M,T}, ::Val{P}) where {M,P,T}
    q = quote @fastmath @inbounds begin end end
    qa = q.args[2].args[3].args[3].args
    load_L_quote!(qa, P, :L, :L)
    safeinvchol_L_core_quote!(qa, P, :L, :L, T)
    store_L_quote!(qa, P, :L)
    push!(qa, :(true))
    q
end
