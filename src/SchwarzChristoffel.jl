module SchwarzChristoffel

export SchwarzChristoffelDerivate, SchwarzChristoffelMap, segment, integral

using StaticArrays
using FixedPointNumbers
using FastGaussQuadrature
using Parameters: @unpack
using Bernstein

const NQUAD = 4

include("fargextremum.jl")

######################
# Low-level functions

"""
    schwarzchristoffelderivative(x,β,xv; skip = ())

Evluate the function `prod((xv.-x).^.-β)`.

Indices listed in skip are omitted from the product.
"""
function schwarzchristoffelderivative(x,β,xv; skip=())
    f = one(ComplexF64)
    for (i,(xk,βk)) in enumerate(zip(x,β))
        i in skip && continue
        f *= complex(xv-xk)^-βk
    end
    return f
end

function assemble_quadcache(β)
    qjac = map(βi->(gaussjacobi(NQUAD,0,-βi), gaussjacobi(2NQUAD,0,-βi)), β)
    qleg = (gausslegendre(NQUAD),gausslegendre(2NQUAD))
    return (
        jacobi_quadrules=qjac,
        legendre_quadrule=qleg
    )
end


#############################
# SchwarzChristoffelDerivate

struct SchwarzChristoffelDerivate{X,B,QC}
    x::X
    β::B
    tol::Float64
    quadcache::QC
end

function SchwarzChristoffelDerivate(x,β; tol = eps())
    idx = sortperm(x)
    x = x[idx]
    β = β[idx]
    SchwarzChristoffelDerivate(x,β,tol,assemble_quadcache(β))
end

"""
    (f::SchwarzChristoffelDerivate)(xv; skip = ())

Evluate the function `prod((xv.-x).^.-β)`.

Indices listed in skip are omitted from the product.
"""
(f::SchwarzChristoffelDerivate)(xv; skip=()) = schwarzchristoffelderivative(f.x,f.β,xv; skip=skip)

"""
    segment(f::SchwarzChristoffelDerivate, k)

Integral of `f` from `x[k]` to `x[k+1]`.
"""
function segment(f::SchwarzChristoffelDerivate,k)
    @unpack x = f
    @assert(k in 0:length(x))

    if k == 0

    end

    xm = (x[k]+x[k+1])/2
    return integral(f,k,xm) - integral(f,k+1,xm)
end

"""
    integral(f::SchwarzChristoffelDerivate, k, xv)

Integral of `f` from `x[k]` to `xv`.
"""
function integral(f::SchwarzChristoffelDerivate,k,xv)
    @unpack x,β,tol,quadcache = f
    qjac = quadcache.jacobi_quadrules
    qleg = quadcache.legendre_quadrule
    @assert k in 1:length(x)

    Δx = xv - x[k]
    Δx == 0 && return zero(ComplexF64)

    t = zero(Q0f63)
    I = zero(ComplexF64)

    Δt,ΔI = shrink_integral(
        Δt -> begin
            Iq = ntuple(i->
                semijacobi_integral(
                    x->f(x,skip=(k,)),
                    x[k],β[k],Δx*-float(Δt),
                    qjac[k][i]
                ),
                Val(2)
            )
            return Iq[2],abs(Iq[1]-Iq[2])
        end,
        Q0f63(-1), tol
    )
    t += Δt; I += ΔI
    while t > -1
        Δt,ΔI = shrink_integral(
            Δt -> begin
                Iq = ntuple(i->
                    legendre_integral(
                        f, x[k]+Δx*-float(t), Δx*-float(Δt),
                        qleg[i]
                    ),
                    Val(2)
                )
                return Iq[2],abs(Iq[1]-Iq[2])
            end,
            max(-1-t, Δt+Δt), tol
        )
        t += Δt; I += ΔI
    end
    return I::ComplexF64
end

legendre_integral(f,x,Δx, (xq,wq)) =
    Δx/2 * mapreduce(
        ((xqk,wqk),) -> f(x+Δx*(xqk+1)/2)*wqk,
        +, zip(xq,wq)
    )
semijacobi_integral(f,x,β,Δx, (xq,wq)) =
    complex(Δx/2)^(-β) * legendre_integral(f,x,Δx, (xq,wq))

function shrink_integral(IEfun,Δt, tol)
    I,E = IEfun(Δt)
    while E > sqrt(tol)*abs(I)
        Δt *= Q0f63(0.5)
        Δt == 0 && error("Could not converge Schwarz-Christoffel integral to desired accuracy")
        I,E = IEfun(Δt)
    end
    return Δt,I
end


########################
# SchwarzChristoffelMap

struct SchwarzChristoffelMap{F<:SchwarzChristoffelDerivate,Z}
    f::F
    z::Z
end

function SchwarzChristoffelMap(f::SchwarzChristoffelDerivate)
    x = f.x
    z = similar(x,ComplexF64)
    z[end] = 0
    for k = length(x)-1:-1:1
        z[k] = z[k+1] - segment(f,k)
    end
    return SchwarzChristoffelMap(f,z)
end

SchwarzChristoffelMap(x,β; kwargs...) = SchwarzChristoffelMap(SchwarzChristoffelDerivate(x,β; kwargs...))

segment(F:SchwarzChristoffelMap,args...) = segment(F.f,args...)

(F::SchwarzChristoffelMap)(xv::Real) = F(complex(xv))
function (F::SchwarzChristoffelMap)(xv::Complex)
    # signbit(imag(xv)) && conj(F(conj(xv)))
    @unpack f,z = F
    @unpack x = f
    k = fargmax(1:length(x)) do k
        xv == x[k] && return Inf

        # Map from x[k]..xv to -1..1
        ϕ = xx->(xx-x[k])/(xv-x[k]) - (xx-xv)/(x[k]-xv)

        k == 1 && return Bernstein.radius(ϕ(x[2]))
        k == length(x) && return Bernstein.radius(ϕ(x[end-1]))
        return min(
            Bernstein.radius(ϕ(x[k-1])),
            Bernstein.radius(ϕ(x[k+1]))
        )
    end
    return z[k] + integral(f,k,xv)
end
