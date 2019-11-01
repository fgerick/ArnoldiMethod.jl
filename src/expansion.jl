using Random
using LinearAlgebra
# using LinearAlgebra.BLAS: gemv!
import LinearAlgebra.BLAS.gemv!
function gemv!(trans::AbstractChar, alpha::Union{(T), Bool},
               A::AbstractVecOrMat{T}, X::AbstractVector{T},
               beta::Union{(T), Bool}, Y::AbstractVector{T}) where T <: Number
    # Base.require_one_based_indexing(A, X, Y)
    m,n = size(A,1),size(A,2)
    if trans == 'N' && (length(X) != n || length(Y) != m)
        throw(DimensionMismatch("A has dimensions $(size(A)), X has length $(length(X)) and Y has length $(length(Y))"))
    elseif trans == 'C' && (length(X) != m || length(Y) != n)
        throw(DimensionMismatch("the adjoint of A has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
    elseif trans == 'T' && (length(X) != m || length(Y) != n)
        throw(DimensionMismatch("the transpose of A has dimensions $n, $m, X has length $(length(X)) and Y has length $(length(Y))"))
    end
    LinearAlgebra.chkstride1(A)
     _gemv!(trans, size(A,1), size(A,2), alpha,
             A, max(1,stride(A,2)), X, stride(X,1),
             beta, Y, stride(Y,1))
    Y
end

Base.@inline function _gemv!(TRANS::Char, M::Int, N::Int, ALPHA::T, A::AbstractVecOrMat{T},
                LDA::Int, X::AbstractVector{T}, INCX::Int, BETA::T,
                Y::AbstractVector{T}, INCY::Int) where {T <: Number}

    INFO = 0
    if (TRANS != 'N') && (TRANS != 'T') && (TRANS != 'C')
        INFO = 1
    elseif (M<0)
        INFO = 2
    elseif (N<0)
        INFO = 3
    elseif (LDA < max(1,M))
        INFO = 6
    elseif (INCX==0)
        INFO = 8
    elseif (INCY==0)
        INFO = 11
    end
    @assert INFO==0

    if ((M == 0) || (N == 0) || ((ALPHA==zero(T)) && (BETA == one(T))))
        return
    end


    if (TRANS=='N')
        LENX = N
        LENY = M
    else
        LENX = M
        LENY = N
    end
    if (INCX > 0)
        KX = 1
    else
        KX = 1 - (LENX-1)*INCX
    end
    if (INCY > 0)
        KY = 1
    else
        KY = 1 - (LENY-1)*INCY
    end

    @inbounds begin
    if (BETA!=one(T))
        if (INCY==1)
            if (BETA==zero(T))
                for I = 1:LENY
                    Y[I] = zero(T)
                end
            else
                for I = 1:LENY
                    Y[I] = BETA*Y[I]
                end
            end
        else
            IY = KY
            if (BETA==zero(T))
                for I = 1:LENY
                    Y[IY] = zero(T)
                    IY = IY + INCY
                end
            else
                for I = 1:LENY
                    Y[IY] = BETA*Y[IY]
                    IY = IY + INCY
                end
            end
        end
    end
    if (ALPHA==zero(T))
        return
    end
    if (TRANS == 'N')

        JX = KX
        if (INCY==1)
            for J = 1:N
                TEMP = ALPHA*X[JX]
                for I = 1:M
                    Y[I] = Y[I] + TEMP*A[I,J]
                end
                JX = JX + INCX
            end
        else
            for J = 1:N
                TEMP = ALPHA*X[JX]
                IY = KY
                for I = 1:M
                    Y[IY] = Y[IY] + TEMP*A[I,J]
                    IY = IY + INCY
                end
                JX = JX + INCX
            end
        end
    else

        JY = KY
        if (INCX==1)
            for J = 1:N
                TEMP = zero(T)
               for I = 1:M
                    TEMP = TEMP + A[I,J]*X[I]
                end
                Y[JY] = Y[JY] + ALPHA*TEMP
                JY = JY + INCY
            end
        else
            for J = 1:N
                TEMP = zero(T)
                IX = KX
                for I = 1:M
                    TEMP = TEMP + A[I,J]*X[IX]
                    IX = IX + INCX
                end
                Y[JY] = Y[JY] + ALPHA*TEMP
                JY = JY + INCY
            end
        end
    end
    end
    return
end



"""
    reinitialize!(a::Arnoldi, j::Int = 0) → a

Generate a random `j+1`th column orthonormal against V[:,1:j]

Returns true if the column is a valid new basis vector.
Returns false if the column is numerically in the span of the previous vectors.
"""
function reinitialize!(arnoldi::Arnoldi{T}, j::Int = 0) where {T}
    V = arnoldi.V
    v = view(V, :, j+1)

    # Generate a new random column
    rand!(v)

    # Norm before orthogonalization
    rnorm = norm(v)

    # Just normalize, don't orthogonalize
    if j == 0
        v ./= rnorm
        return true
    end

    # Constant used by ARPACK.
    η = √2 / 2
    Vprev = view(V, :, 1:j)

    # Orthogonalize: h = Vprev' * v, v ← v - Vprev * Vprev' * v = v - Vprev * h
    h = Vprev' * v
    gemv!('N', -one(T), Vprev, h, one(T), v)

    # Norm after orthogonalization
    wnorm = norm(v)

    # Reorthogonalize once
    if wnorm < η * rnorm
        rnorm = wnorm
        mul!(h, Vprev', v)
        gemv!('N', -one(T), Vprev, h, one(T), v)
        wnorm = norm(v)
    end

    if wnorm ≤ η * rnorm
        # If we have to reorthogonalize thrice, then we're just numerically in the span
        return false
    else
        # Otherwise we just normalize this new basis vector
        v ./= wnorm
        return true
    end
end

"""
    orthogonalize!(arnoldi, j) → Bool

Orthogonalize arnoldi.V[:, j+1] against arnoldi.V[:, 1:j].

Returns true if the column is a valid new basis vector.
Returns false if the column is numerically in the span of the previous vectors.
"""
function orthogonalize!(arnoldi::Arnoldi{T}, j::Integer) where {T}
    V = arnoldi.V
    H = arnoldi.H

    # Constant used by ARPACK.
    η = √2 / 2

    Vprev = view(V, :, 1:j)
    v = view(V, :, j+1)
    h = view(H, 1:j, j)

    # Norm before orthogonalization
    rnorm = norm(v)

    # Orthogonalize: h = Vprev' * v, v ← v - Vprev * Vprev' * v = v - Vprev * h
    mul!(h, Vprev', v)
    gemv!('N', -one(T), Vprev, h, one(T), v)

    # Norm after orthogonalization
    wnorm = norm(v)

    # Reorthogonalize once
    if wnorm < η * rnorm
        rnorm = wnorm
        correction = Vprev' * v
        gemv!('N', -one(T), Vprev, correction, one(T), v)
        h .+= correction
        wnorm = norm(v)
    end

    if wnorm ≤ η * rnorm
        # If we have to reorthogonalize thrice, then we're just numerically in the span
        H[j+1,j] = zero(T)
        return false
    else
        # Otherwise we just normalize this new basis vector
        H[j+1,j] = wnorm
        v ./= wnorm
        return true
    end
end

"""
    iterate_arnoldi!(A, arnoldi, from:to) → arnoldi

Perform Arnoldi from `from` to `to`.
"""
function iterate_arnoldi!(A, arnoldi::Arnoldi{T}, range::UnitRange{Int}) where {T}
    V, H = arnoldi.V, arnoldi.H

    for j = range
        # Generate a new column of the Krylov subspace
        mul!(view(V, :, j+1), A, view(V, :,j))

        # Orthogonalize it against the other columns
        # If V[:,j+1] is in the span of V[:,1:j], then we generate a new
        # vector. If j == n, then obviously we cannot find a new orthogonal
        # column V[:,j+1].
        if orthogonalize!(arnoldi, j) === false && j != size(V, 1)
            reinitialize!(arnoldi, j)
        end
    end

    return arnoldi
end
