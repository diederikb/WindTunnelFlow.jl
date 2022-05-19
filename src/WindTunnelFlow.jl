module WindTunnelFlow

#using DocStringExtensions
using Reexport
using UnPack
using LinearAlgebra
@reexport using ViscousFlow
@reexport using GridPotentialFlow

import ViscousFlow: viscousflow_vorticity_bc_rhs!, viscousflow_vorticity_ode_rhs!, velocity!

export create_windtunnelwalls, WindTunnelProblem


include("NeumannPoisson.jl")

@ilmproblem(WindTunnel,vector)

struct WindTunnelCache{CDT,FRT,DVT,VFT,VORT,DILT,VELT,FCT,WTST,WTBT,WTVT} <: AbstractExtraILMCache
    # ViscousIncompressibleFlow
    cdcache :: CDT
    fcache :: FRT
    dvb :: DVT
    vb_tmp :: DVT
    v_tmp :: VFT
    dv :: VFT
    dv_tmp :: VFT
    w_tmp :: VORT
    divv_tmp :: DILT
    velcache :: VELT
    f :: FCT
    # Wind tunnel correction problem with caches
    wt_sys :: WTST
    wt_bool :: WTBT
    wt_vel :: WTVT
end

function ImmersedLayers.prob_cache(prob::WindTunnelProblem,
                                   base_cache::BasicILMCache{N,scaling}) where {N,scaling}
    @unpack phys_params, forcing = prob
    @unpack gdata_cache, gcurl_cache, g = base_cache

    #============ ViscousFlow cache ============#
    dvb = zeros_surface(base_cache)
    vb_tmp = zeros_surface(base_cache)

    v_tmp = zeros_grid(base_cache)
    dv = zeros_grid(base_cache)
    dv_tmp = zeros_grid(base_cache)
    w_tmp = zeros_gridcurl(base_cache)
    divv_tmp = zeros_griddiv(base_cache)

    velcache = VectorFieldCache(base_cache)

    # Construct a Lapacian outfitted with the viscosity
    Re = ViscousFlow.get_Reynolds_number(phys_params)
    viscous_L = Laplacian(base_cache,gcurl_cache,1.0/Re)

    # Create cache for the convective derivative
    cdcache = ConvectiveDerivativeCache(base_cache)

    fcache = nothing

    # Create cache for the forcing regions
    fmods = ViscousFlow.get_forcing_models(forcing)
    fcache = ForcingModelAndRegion(fmods,base_cache)

    # The state here is vorticity, the constraint is the surface traction
    f = ViscousFlow._get_ode_function_list(viscous_L,base_cache)


    #============ Wind tunnel correction problem ============#
    wt_walls = create_windtunnelwalls(g,phys_params)
    wt_prob = PotentialFlowProblem(g,wt_walls,scaling=GridScaling,phys_params=phys_params)
    wt_sys = construct_system(wt_prob);

    wt_bool_surface = ScalarData(points(wt_walls))

    regop = ImmersedLayers._get_regularization(points(wt_walls),areas(wt_walls),base_cache.g, CartesianGrids.Goza,GridScaling)
    Rcurl = ImmersedLayers._regularization_matrix(regop, wt_bool_surface, zeros_gridcurl(base_cache))

    wt_bool = zeros_gridcurl(base_cache)
    wt_bool_surface .= 1
    mul!(wt_bool,Rcurl,wt_bool_surface)
    wt_bool .= (wt_bool .!= 0.0);

    wt_vel = zeros_grid(base_cache)

    WindTunnelCache(cdcache,fcache,dvb,vb_tmp,v_tmp,dv,dv_tmp,w_tmp,divv_tmp,velcache,f,wt_sys,wt_bool,wt_vel)
end

function ViscousFlow.viscousflow_vorticity_bc_rhs!(vb,sys::ILMSystem{true,T},t) where {T<:WindTunnelProblem}
    @unpack bc, forcing,extra_cache, base_cache, phys_params = sys
    @unpack dvb, vb_tmp, v_tmp, velcache, divv_tmp = extra_cache
    @unpack dcache, ϕtemp = velcache

    ViscousFlow.viscousflow_velocity_bc_rhs!(vb,sys,t)

    # Subtract influence of scalar potential field
    fill!(divv_tmp,0.0)
    prescribed_surface_jump!(dvb,t,sys)
    scalarpotential_from_masked_divv!(ϕtemp,divv_tmp,dvb,base_cache,dcache)
    vecfield_from_scalarpotential!(v_tmp,ϕtemp,base_cache)
    interpolate!(vb_tmp,v_tmp,base_cache)
    vb .-= vb_tmp

    # Subtract influence of free stream
    freestream_func = ViscousFlow.get_freestream_func(forcing)
    Uinf, Vinf = freestream_func(t,phys_params)
    vb.u .-= Uinf
    vb.v .-= Vinf

    # Subtract influence of wind tunnel correction
    interpolate!(vb_tmp,extra_cache.wt_vel,base_cache)
    vb .-= vb_tmp

    return vb
end

function ViscousFlow.viscousflow_vorticity_ode_rhs!(dw,w,sys::ILMSystem{true,T},t) where {T<:WindTunnelProblem}
    @unpack extra_cache, base_cache = sys
    @unpack v_tmp, dv, wt_bool = extra_cache

    ViscousFlow.velocity!(v_tmp,w,sys,t)
    ViscousFlow.viscousflow_velocity_ode_rhs!(dv,v_tmp,sys,t)
    curl!(dw,dv,base_cache)

    dw .-= dw.*wt_bool

    return dw
end

function ViscousFlow.velocity!(v::Edges{Primal},w::Nodes{Dual},sys::ILMSystem{true,T},t) where {T<:WindTunnelProblem}
    @unpack forcing, phys_params, extra_cache, base_cache = sys
    @unpack dvb, velcache, divv_tmp, w_tmp, wt_vel = extra_cache

    prescribed_surface_jump!(dvb,t,sys)

    freestream_func = ViscousFlow.get_freestream_func(forcing)
    Vinf = freestream_func(t,phys_params)
    vecfield_uniformvecfield!(base_cache.gdata_cache,Vinf[1],Vinf[2],base_cache)

    fill!(wt_vel,0.0)
    ViscousFlow.velocity!(wt_vel,w,divv_tmp,dvb,base_cache.gdata_cache,base_cache,velcache,w_tmp)

    wt_vn = zeros_surfacescalar(extra_cache.wt_sys)
    wt_dvn = zeros_surfacescalar(extra_cache.wt_sys)
    f = zeros_griddiv(extra_cache.wt_sys)
    df = zeros_surfacescalar(extra_cache.wt_sys)
    v_df = zeros_grid(sys.extra_cache.wt_sys)

    normal_interpolate!(wt_vn,wt_vel,extra_cache.wt_sys)
    wt_vn .*= -1

    Δs = surface_point_spacing(base_cache.g,phys_params)
    vn_pts = points(extra_cache.wt_sys)
    add_sink!(wt_vn,vn_pts,Δs,phys_params,sys,t)

    f, df, s, ds = GridPotentialFlow.scalarpotential!(f,df,wt_vn,wt_dvn,extra_cache.wt_sys,t)

    # Compute Gϕ̄
    grad!(wt_vel,f,sys.extra_cache.wt_sys);

    # Eldredge JCP 2022 Eq 39: Ḡϕ̄ = Gϕ̄ - I(ϕ⁺-ϕ⁻)∘Rn
    regularize_normal!(v_df,df,sys.extra_cache.wt_sys)
    wt_vel .-= v_df

    # Add potential flow velocity field to freestream velocity field
    base_cache.gdata_cache .+= wt_vel

    fill!(divv_tmp,0.0)
    ViscousFlow.velocity!(v,w,divv_tmp,dvb,base_cache.gdata_cache,base_cache,velcache,w_tmp)
end

function create_windtunnelwalls(g,phys_params)
    Δs = surface_point_spacing(g,phys_params)
    haskey(phys_params,"wind tunnel length") || error("No wind tunnel length set")
    haskey(phys_params,"wind tunnel height") || error("No wind tunnel height set")

    L = phys_params["wind tunnel length"]
    H = phys_params["wind tunnel height"]
    cent = get(phys_params,"wind tunnel center",(0.0,0.0))

    plate_tb = Plate(L,Δs);
    bl = BodyList([deepcopy(plate_tb),deepcopy(plate_tb)])
    t1! = RigidTransform((cent[1],cent[2]+H/2),π)
    t2! = RigidTransform((cent[1],cent[2]-H/2),0.0)
    tl! = RigidTransformList([t1!,t2!])
    tl!(bl)
    return bl
end

function add_sink!(vn,vn_pts,ds,phys_params,sys,t)
    # tstart = get(phys_params,"sink start time",-Inf)
    # tend = get(phys_params,"sink end time",-Inf)
    # trise = get(phys_params,"sink rise time",0.0)
    # tfall = get(phys_params,"sink fall time",0.0)
    t0 = get(phys_params,"sink time",0.0)
    σ = get(phys_params,"sink sigma",0.0)
    Q = get(phys_params,"sink strength",0.0)

    g = Gaussian(σ,sqrt(π*σ^2)) >> t0

    if g(t) > 0.0
        # println(t)
        x_c = get(phys_params,"sink position",0.0)
        L = get(phys_params,"sink width",0.0)
        N = get(phys_params,"sink points",10)
        H = phys_params["wind tunnel height"]
        cent = get(phys_params,"wind tunnel center",(0.0,0.0))
        y_c = cent[2]+H/2

        dS = L/N
        sink = Plate(L,dS) # assume sink distribution is horizontal
        T = RigidTransform((x_c,y_c),0.0)
        T(sink)

        q_pts = points(sink)
        q = ScalarData(q_pts)

        q .= g(t)*Q/L

        line_regularize!(vn,vn_pts,q,q_pts,ds,dS)
    end
end

function line_regularize!(vn::ScalarData,vn_pts::VectorData,q::ScalarData,q_pts::VectorData,ds,dS;ddf_radius=1.0,ddf=CartesianGrids.ddf_witchhat)
    for i in 1:length(q)
        for j in 1:length(vn)
            dx = vn_pts.u[j] - q_pts.u[i]
            dy = vn_pts.v[j] - q_pts.v[i]
            if abs(dx)/ds < ddf_radius && abs(dy)/ds < ddf_radius
                dist = sqrt(dx^2 + dy^2)/ds
                vn[j] += q[i] * ddf(dist) * dS/ds
            end
        end
    end
end

end # module
