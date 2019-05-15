module SchwarzChristoffel

export SchwarzChristoffelDerivate, SchwarzChristoffelMap, segment, integral

using StaticArrays
using FixedPointNumbers
using FastGaussQuadrature
using Parameters: @unpack
using Bernstein
using DifferentialEquations
using Roots

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
    qjac = map(βi->(SVector{NQUAD}.(gaussjacobi(NQUAD,0,-βi)), SVector{2NQUAD}.(gaussjacobi(2NQUAD,0,-βi))), β)
    qleg = (SVector{NQUAD}.(gausslegendre(NQUAD)), SVector{2NQUAD}.(gausslegendre(2NQUAD)))
    qinf = sum(β) > 1 ?
        (SVector{NQUAD}.(gaussjacobi(NQUAD,0,sum(β)-2)), SVector{2NQUAD}.(gaussjacobi(2NQUAD,0,sum(β)-2))) :
        nothing
    return (
        jacobi_quadrules=qjac,
        legendre_quadrule=qleg,
        inf_quadrule=qinf
    )
end

function adaptive_semijacobi_integral(f,x,β,Δx,tol,(qjac,qleg))
    Δx == 0 && return zero(ComplexF64)

    t = zero(Q0f63)
    I = zero(ComplexF64)

    Δt,ΔI = shrink_integral(
        Δt -> begin
            Iq = ntuple(i->
                semijacobi_integral(
                    f, x,β,Δx*-float(Δt),
                    qjac[i]
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
                        xv->f(xv)*complex(xv-x)^-β,
                        x+Δx*-float(t), Δx*-float(Δt),
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

struct InfSegmentMap
    x0::Float64
    b::Float64
end

function InfSegmentMap(xv,x1,x2)
    s = xv < x1 ? -1 : 1
    InfSegmentMap(xv,s*sqrt(xv*(xv-x1-x2) + x1*x2))
end

(csm::InfSegmentMap)(y) = csm.b/y-csm.b+csm.x0
grad(csm::InfSegmentMap,y) = -csm.b/y^2
finv(csm::InfSegmentMap,x) = csm.b/(x - csm.x0 + csm.b)


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

hasfiniteintegral(f::SchwarzChristoffelDerivate) = sum(f.β) > 1
nsegments(f::SchwarzChristoffelDerivate) = length(f.x)+1

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

    if k in 1:length(x)-1
        xm = (x[k]+x[k+1])/2
    elseif k == 0
        xm = InfSegmentMap(x[1],x[2],x[end])(0.5)
    elseif k == length(x)
        xm = InfSegmentMap(x[end],x[1],x[end-1])(0.5)
    else
        throw(ArgumentError("attempt to compute segment $k of $(length(x)+1)-segment SchwarzChristoffelDerivate"))
    end
    return integral(f,k,xm) - integral(f,k+1,xm)
end

"""
    integral(f::SchwarzChristoffelDerivate, k, xv)

Integral of `f` from `x[k]` to `xv`.
"""
function integral(f::SchwarzChristoffelDerivate,k,xv)
    @unpack x,β,tol,quadcache = f
    @assert k in 0:length(x)+1
    qleg = quadcache.legendre_quadrule

    if k in 1:length(x)
        qjac = quadcache.jacobi_quadrules[k]
        return adaptive_semijacobi_integral(xx->f(xx,skip=(k,)),x[k],β[k],xv-x[k],tol,(qjac,qleg))
    else
        @assert isreal(xv)
        qinf = quadcache.inf_quadrule
        m = InfSegmentMap(xv,x[1],x[end])
        return adaptive_semijacobi_integral(
            y->f(m(y))*grad(m,y),
            0, 2-sum(β), 1,
            tol,(qinf,qleg)
        )
    end
end


########################
# SchwarzChristoffelMap

struct SchwarzChristoffelMap{F<:SchwarzChristoffelDerivate,Z,Zinf}
    f::F
    z::Z
    zinf::Zinf
end

function SchwarzChristoffelMap(f::SchwarzChristoffelDerivate)
    @unpack x = f
    z = similar(x,ComplexF64,)
    z[end] = 0
    for k = length(x)-1:-1:1
        z[k] = z[k+1] - segment(f,k)
    end
    zinf = hasfiniteintegral(f) ? segment(f,length(x)) : Inf
    return SchwarzChristoffelMap(f,z,zinf)
end

SchwarzChristoffelMap(x,β; kwargs...) = SchwarzChristoffelMap(SchwarzChristoffelDerivate(x,β; kwargs...))

segment(F::SchwarzChristoffelMap,args...) = segment(F.f,args...)
nsegments(f::SchwarzChristoffelMap) = nsegments(F.f)
grad(F::SchwarzChristoffelMap,x) = F.f(x)

(F::SchwarzChristoffelMap)(xv::Real) = F(complex(xv))
function (F::SchwarzChristoffelMap)(xv::Complex)
    # signbit(imag(xv)) && conj(F(conj(xv)))
    @unpack f,z,zinf = F
    @unpack x = f

    abs(xv) == Inf && return zinf

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


###############################
# InverseSchwarzChristoffelMap

struct InverseSchwarzChristoffelMap{SCM,X,D,Z}
    F::SCM
    x::X
    d::D
    z::Z
end

finv(F::SchwarzChristoffelMap, x = default_starting_points(F)) =
    InverseSchwarzChristoffelMap(F,x,conj.(sign.(F.f.(x))),F.(x))

const NSAMPLES = 3

function default_starting_points(F::SchwarzChristoffelMap)
    @unpack x = F.f
    n = length(x)
    midpoints = LinRange(0,1, 2NSAMPLES+1)[2:2:end]
    xs = Vector{Float64}(undef, NSAMPLES*(n+1))
    for k = 1:n-1
        xs[NSAMPLES*(k-1) .+ (1:NSAMPLES)] .= x[k] .+ (x[k+1]-x[k]).*midpoints
    end
    xs[NSAMPLES*(n-1) .+ (1:NSAMPLES)] .= InfSegmentMap(x[1],x[2],x[end]).(midpoints)
    xs[NSAMPLES*n .+ (1:NSAMPLES)] .= InfSegmentMap(x[end],x[1],x[end-1]).(midpoints)
    return xs
end

function find_starting_point(x,d,z,ze)
    @assert length(x) == length(d) == length(z)
    k = fargmin(1:length(x)) do k
        imag(d[k]*(ze-z[k])) < 0 && return Inf
        return abs(ze - z[k])
    end
    return x[k],z[k]
end

function (iF::InverseSchwarzChristoffelMap)(ze)
    @unpack F,x,d,z = iF
    xs,zs = find_starting_point(x,d,z,ze)
    x̃e = finv_ode(F,xs,zs,ze)
    return finv_newton(F,x̃e,ze)
end

function finv_ode(F::SchwarzChristoffelMap,xs,zs,ze)
    @unpack f = F
    prob = ODEProblem(
        (x,_,z)->(ze-zs)/f(x),
        ComplexF64(xs), (0.0,1.0)
    )
    sol = solve(prob, reltol=1e-2, save_everystep=false)
    return sol(1)
end

function finv_newton(F::SchwarzChristoffelMap,xs,ze)
    @unpack f = F
    fd = (x->F(x) - ze), f
    return find_zero(
        fd,
        complex(xs),
        Roots.Newton(),
        rtol=1e-3
    )
end

end
