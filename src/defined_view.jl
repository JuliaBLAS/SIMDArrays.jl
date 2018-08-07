### Defined views use safe `@views`, with the intention
### that they are replaced with "UnsafeArrays" via the
### @uviews macro in a safe way.

function defined(A::SIMDArray{T,N}) where {T,N}
    
end
function defined(A::SizedSIMDArray{S,T,N,R,L}) where {S,T,N,R,L}

end

