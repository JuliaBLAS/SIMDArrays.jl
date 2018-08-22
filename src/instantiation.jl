


@generated function Base.zeros(::SizedSIMDArray{S,T})
    SV = S.parameters
    N = length(SV)
    R, L = calculate_L_from_size(SV)
    quote
        out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
        @inbounds for i ∈ 1:$L
            out[i] = 0
        end
        out
    end
end
