using SIMD
import Base.Cartesian: @nexprs

struct Block{T1,T2,T3,T4,G}
    Ac::T1
    Bc::T2
    AB::T3
    C::T4
    mc::UInt
    kc::UInt
    nc::UInt
    mr::UInt
    nr::UInt
    inc1A::UInt
    inc2A::UInt
    inc1B::UInt
    inc2B::UInt
    inc1C::UInt
    inc2C::UInt
end

const Ac = Vector{UInt8}(undef, 110592)
const Bc = Vector{UInt8}(undef, 6266880)
const AB = Vector{UInt8}(undef, 12*4*8)

function Block(A::X, B::W, C::Z, generic) where {X, W, Z}
    global Ac, Bc, AB
    mr=UInt(12); nr=UInt(4)
    m, n = size(C)
    mc = UInt(72)
    kc = UInt(192)
    nc = UInt(4080)
    T = promote_type(eltype(X), eltype(W), eltype(Z))
    siz = sizeof(T)
    _Ac = unsafe_wrap(Array, Ptr{T}(pointer(Ac)), length(Ac)÷siz)
    _Bc = unsafe_wrap(Array, Ptr{T}(pointer(Bc)), length(Bc)÷siz)
    _AB = unsafe_wrap(Array, Ptr{T}(pointer(AB)), length(AB)÷siz)
    Block{typeof(_Ac),typeof(_Bc),typeof(_AB),typeof(C),generic}(_Ac, _Bc, _AB, C, mc, kc, nc, mr, nr,
                                                      convert(NTuple{2,UInt}, strides(A))..., convert(NTuple{2,UInt}, strides(B))..., convert(NTuple{2,UInt}, strides(C))...,)
end

"""
    addmul!(C, A, B, blk::Block{T1,T2,T3,T4,G}=Block(A, B, C, false)) -> C

`addmul!` computs ``C = AB + C``, where ``A``, ``B``, and ``C`` are matrices.
"""
function addmul!(C, A, B, blk::Block{T1,T2,T3,T4,G}=Block(A, B, C, false)) where {T1,T2,T3,T4,G}
    m = UInt(size(A)[1])
    k = UInt(size(A)[2])
    _k = UInt(size(B)[1])
    n = UInt(size(B)[2])
    @assert k == _k
    _m = UInt(size(C)[1])
    _n = UInt(size(C)[2])
    @assert m == _m && n == _n
    mb, _mc = cld(m, blk.mc), UInt(m % blk.mc)
    nb, _nc = cld(n, blk.nc), UInt(n % blk.nc)
    kb, _kc = cld(k, blk.kc), UInt(k % blk.kc)
    for j in UnitRange{UInt}(one(UInt),nb) # Loop 5
        nc = (j!=nb || iszero(_nc)) ? blk.nc : _nc
        for l in UnitRange{UInt}(one(UInt),kb) # Loop 4
            kc = (l!=kb || iszero(_kc)) ? blk.kc : _kc
            #_β = l==1 ? β : 1.0
            offsetB = blk.kc*(l-0x01)*blk.inc1B + blk.nc*(j-0x01)*blk.inc2B
            pack_B!(blk, B, kc, nc, offsetB)
            for i in UnitRange{UInt}(one(UInt),mb) # Loop 3
                mc = (i!=mb || iszero(_mc)) ? blk.mc : _mc
                offsetA = blk.mc*(i-one(UInt))*blk.inc1A + blk.kc*(l-one(UInt))*blk.inc2A
                offsetC = blk.mc*(i-one(UInt))*blk.inc1C + blk.nc*(j-one(UInt))*blk.inc2C
                pack_A!(blk, A, mc, kc, offsetA)
                macro_ker!(blk, C, mc, nc, kc, offsetC)
            end # Loop 3
        end # Loop 4
    end # Loop 5
    C
end

@inline function pack_MRxK!(blk::Block{T1,T2,T3,T4,G}, A, k::Integer,
                            offsetA::Integer, offsetAc::Integer) where {T1,T2,T3,T4,G}
    @inbounds for j in UnitRange{UInt}(one(UInt),k)
        for i in UnitRange{UInt}(one(UInt), blk.mr)
            blk.Ac[offsetAc+i] = A[offsetA + (i-one(UInt))*blk.inc1A + one(UInt)]
        end
        offsetAc += blk.mr
        offsetA  += blk.inc2A
    end
    return nothing
end

function pack_A!(blk::Block{T1,T2,T3,T4,G}, A, mc::Integer, kc::Integer,
                 offsetA::Integer) where {T1,T2,T3,T4,G}
    mp, _mr = divrem(mc, blk.mr)
    offsetAc = 0x00
    for i in one(UInt8):UInt8(mp)
        pack_MRxK!(blk, A, kc, offsetA, offsetAc)
        offsetAc += kc*blk.mr
        offsetA  += blk.mr*blk.inc1A
    end
    if _mr > 0
        @inbounds for j in 1:kc
            for i in UnitRange{UInt}(one(UInt), _mr)
                blk.Ac[offsetAc+i] = A[offsetA + (i-0x01)*blk.inc1A + 0x01]
            end
            for i in UnitRange{UInt}(_mr+one(UInt), blk.mr)
                blk.Ac[offsetAc+i] = zero(eltype(A))
            end
            offsetAc += blk.mr
            offsetA  += blk.inc2A
        end
    end
    return nothing
end

@inline function pack_KxNR!(blk::Block{T1,T2,T3,T4,G}, B, k::Integer,
                            offsetB::Integer, offsetBc::Integer) where {T1,T2,T3,T4,G}
    @inbounds for i = 1:k
        for j = 1:blk.nr
            blk.Bc[offsetBc+j] = B[offsetB + (j-0x01)*blk.inc2B + 0x01]
        end
        offsetBc += blk.nr
        offsetB  += blk.inc1B
    end
    return nothing
end

function pack_B!(blk::Block{T1,T2,T3,T4,G}, B,
                 kc::Integer, nc::Integer, offsetB::Integer) where {T1,T2,T3,T4,G}
    np, _nr = divrem(nc, blk.nr)
    offsetBc = 0
    for j in UnitRange{UInt}(one(UInt),np)
        pack_KxNR!(blk, B, kc, offsetB, offsetBc)
        offsetBc += kc*blk.nr
        offsetB  += blk.nr*blk.inc2B
    end
    if _nr > 0
        @inbounds for i in 1:kc
            for j in UnitRange{UInt}(one(UInt), _nr)
                blk.Bc[offsetBc+j] = B[offsetB + (j-one(UInt))*blk.inc2B + 0x01]
            end
            for j in UnitRange{UInt}(_nr+one(UInt), blk.nr)
                blk.Bc[offsetBc+j] = zero(eltype(B))
            end
            offsetBc += blk.nr
            offsetB  += blk.inc1B
        end
    end
    return nothing
end

@inline function macro_ker!(blk::Block{T1,T2,T3,T4,G}, C, mc::Integer, nc::Integer, kc::Integer,
                            offsetC::Integer) where {T1,T2,T3,T4,G}
    mp, _mr = cld(mc, blk.mr), UInt(mc % blk.mr)
    np, _nr = cld(nc, blk.nr), UInt(nc % blk.nr)
    for j in UnitRange{UInt}(one(UInt),np)
        nr = (j!=np || iszero(_nr)) ? blk.nr : _nr
        for i in UnitRange{UInt}(one(UInt),mp)
            mr = (i!=mp || iszero(_mr)) ? blk.mr : _mr
            offsetA = (i-0x01)*kc*blk.mr
            offsetB = (j-0x01)*kc*blk.nr
            if mr == blk.mr && nr==blk.nr
                micro_ker!(blk, kc, offsetA, offsetB, offsetC+(i-0x01)*blk.mr*blk.inc1C + (j-0x01)*blk.nr*blk.inc2C, Val(true))
            else
                micro_ker!(blk, kc, offsetA, offsetB, 0x00, Val(false))
                _axpy!(C, one(UInt), blk.AB, mr, nr, offsetC+(i-0x01)*blk.mr*blk.inc1C + (j-0x01)*blk.nr*blk.inc2C+0x01,
                       0x01, 0x01, blk.mr)
            end
        end
    end
    return nothing
end

@inline function micro_ker!(blk::Block{T1,T2,T3,T4,G}, kc::Integer,
                                     offsetA::Integer, offsetB::Integer, offsetC::Integer,
                                     ::Val{loadC}) where {T1,T2,T3,T4,G,loadC}
    #expr = kernel_quote(T1, 8, 6, loadC)
    #quote
    #    $(Expr(:meta, :inline))
    #    @assert blk.mr == 8 && blk.nr == 6
    #    $expr
    #end
    @inbounds begin
        if !G
            pA, pAB = pointer(blk.Ac), pointer(blk.AB)
            T = eltype(T1)
            siz = 0x08
            VT = Vec{4, T}
            if loadC
                pC = pointer(blk.C)
                @nexprs 4 i -> begin
                    ab_i_1 = vload(VT, pC + (offsetC+(i-0x01)*blk.inc2C  )siz)
                    ab_i_2 = vload(VT, pC + (offsetC+(i-0x01)*blk.inc2C+0x04)siz)
                    ab_i_3 = vload(VT, pC + (offsetC+(i-0x01)*blk.inc2C+0x08)siz)
                end
            else
                @nexprs 4 i -> begin
                    ab_i_1 = zero(VT)
                    ab_i_2 = zero(VT)
                    ab_i_3 = zero(VT)
                end
            end
            for k in UnitRange{UInt}(zero(UInt),(kc-0x01))
                a1 = vload(VT, pA + (offsetA+blk.mr*k)siz)
                a2 = vload(VT, pA + (offsetA+blk.mr*k+0x04)siz)
                a3 = vload(VT, pA + (offsetA+blk.mr*k+0x08)siz)
                @nexprs 4 i -> begin
                    b_i = VT(blk.Bc[offsetB+k*blk.nr+i])
                    ab_i_1 = muladd(a1, b_i, ab_i_1)
                    ab_i_2 = muladd(a2, b_i, ab_i_2)
                    ab_i_3 = muladd(a3, b_i, ab_i_3)
                end
            end
            if loadC
                @nexprs 4 i -> begin
                    vstore(ab_i_1, pC + (offsetC+(i-0x01)*blk.inc2C)siz)
                    vstore(ab_i_2, pC + (offsetC+(i-0x01)*blk.inc2C+0x04)siz)
                    vstore(ab_i_3, pC + (offsetC+(i-0x01)*blk.inc2C+0x08)siz)
                end
            else
                @nexprs 4 i -> begin
                    vstore(ab_i_1, pAB + (i-0x01)blk.mr*siz)
                    vstore(ab_i_2, pAB + ((i-0x01)blk.mr+0x04)*siz)
                    vstore(ab_i_3, pAB + ((i-0x01)blk.mr+0x08)*siz)
                end
            end
        else
            fill!(blk.AB, zero(eltype(blk.AB)))
            for k in UnitRange{UInt}(one(UInt),kc)
                for j in UnitRange{UInt}(one(UInt),blk.nr), i in UnitRange{UInt}(one(UInt),blk.mr)
                    blk.AB[i + (j-0x01)*blk.mr] += blk.Ac[offsetA+i] * blk.Bc[offsetB+j]
                end
                offsetA += blk.mr
                offsetB += blk.nr
            end
            if loadC
                for j in UnitRange{UInt}(zero(UInt),(blk.nr-one(UInt))), i in UnitRange{UInt}(zero(UInt),blk.mr-one(UInt))
                    blk.C[offsetC+i*blk.inc1C+j*blk.inc2C+0x01] += blk.AB[i + j*blk.mr + 0x01]
                end
            end
        end
        return nothing
    end
end

@inline function _axpy!(Y, α, X, m::Integer, n::Integer,
                        offsetY::Integer, offsetX::Integer, inc1X::Integer, inc2X::Integer)
    IT = UInt
    inc1Y, inc2Y = IT(stride(Y, 1)), IT(stride(Y, 2))
    @inbounds for j in UnitRange{UInt}(zero(IT),n-one(IT)), i in UnitRange{UInt8}(zero(IT),m-one(IT))
        Y[offsetY+i*inc1Y+j*inc2Y] += α*X[offsetX+i*inc1X+j*inc2X]
    end
    return nothing
end
