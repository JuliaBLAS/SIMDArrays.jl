


@generated function Base.zeros(::Type{SizedSIMDArray{S,T}}) where {S,T}
    SV = S.parameters
    N = length(SV)
    R, L = calculate_L_from_size(SV)
    quote
        out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
        @inbounds for i ∈ 1:$L
            out[i] = zero($T)
        end
        out
    end
end

@generated function Base.fill(::Type{SizedSIMDArray{S,T}}, v::T) where {S,T}
    SV = S.parameters
    N = length(SV)
    R, L = calculate_L_from_size(SV)
    quote
        out = SizedSIMDArray{$S,$T,$N,$R,$L}(undef)
        @inbounds for i ∈ 1:$L #Here, we accept the risk that the buffer becomes subnormal?
            out[i] = v
        end
        out
    end
end
