
@inline function LinearAlgebra.mul!(C, A, B::SIMDArray)
    @uviews B.data mul!(C, A, @view(B.data[1:B.nrow,:]))
end

@inline function LinearAlgebra.mul!(C, A::SIMDArray, B::SIMDArray)
    @uviews B.data mul!(C, A.data, @view(B.data[1:B.nrow,:]))
end
@inline function LinearAlgebra.mul!(C::SIMDArray, A, B::SIMDArray)
    @uviews B.data mul!(C.data, A, @view(B.data[1:B.nrow,:]))
end
@inline function LinearAlgebra.mul!(C::SIMDArray, A::SIMDArray, B::SIMDArray)
    @uviews B.data mul!(C.data, A.data, @view(B.data[1:B.nrow,:]))
end




# """
# This has been implemented with a generated function.
# There has got to be a better, more elegant, way. But often the easiest
# approach to describe general behavior of a function succinctly is to just be explicit.
# """
# @generated function LinearAlgebra.mul!(C::Union{<:SIMDArray, A::SIMDArray, B::SIMDArray)


# end
