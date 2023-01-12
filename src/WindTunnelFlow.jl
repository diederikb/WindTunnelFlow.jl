module WindTunnelFlow

#using DocStringExtensions
using Reexport
using UnPack
using LinearAlgebra
@reexport using ViscousFlow
@reexport using GridPotentialFlow

import ImmersedLayers: ConstrainedODEFunction
import ViscousFlow: viscousflow_vorticity_bc_rhs!, viscousflow_vorticity_ode_rhs!, velocity!, streamfunction!

export create_windtunnel_boundaries, WindTunnelProblem, corrected_streamfunction!, UniformFlowThrough

@ilmproblem(WindTunnel,vector)

"""
When creating the time marching for a `WindTunnelProblem`, include a velocity update in each time-stepping stage such that viscousflow_vorticity_bc_rhs! uses an updated value extra_cache.wt_body_bc. The way it is implemented now is not very clean and should be improved in the future. Its effects should also be analyzed in more detail. Ideally, the coupling between the potential and viscous flow models would be monolithical and this update would not be necessary.
"""
function ImmersedLayers.ConstrainedODEFunction(sys::ILMSystem{true,T}) where {T<:WindTunnelProblem}
    @unpack extra_cache = sys
    @unpack f = extra_cache

    ImmersedLayers._constrained_ode_function(f.lin_op,
                                             f.ode_rhs,
                                             f.bc_rhs,
                                             f.constraint_force,
                                             f.bc_op;
                                             _func_cache=zeros_sol(sys),
                                             param_update_func=WindTunnelFlow.update_system!,
                                             force_static=true
                                             )
end

function update_system!(sys::ILMSystem,u,sysold::ILMSystem,t)

    # These next sys. assignments should not be necessary. Check if they can be removed.
    sys.phys_params = sysold.phys_params
    sys.bc = sysold.bc
    sys.forcing = sysold.forcing
    sys.timestep_func = sysold.timestep_func
    sys.motions = sysold.motions
    sys.base_cache = sysold.base_cache
    sys.extra_cache = sysold.extra_cache

    velocity!(sys.extra_cache.v_tmp,state(u),sys,t)

    return sys
end

struct WindTunnelCache{CDT,FRT,DVT,VFT,VORT,DILT,VELT,FCT,WTST,WTBT,WTVT,WTSCT} <: AbstractExtraILMCache
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
    wt_body_bc :: DVT
    wt_tmp_boundaries:: WTSCT
    wt_vn_boundaries:: WTSCT
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
    viscous_L = Laplacian(base_cache,1.0/Re)

    # Create cache for the convective derivative
    cdcache = ConvectiveDerivativeCache(base_cache)

    fcache = nothing

    # Create cache for the forcing regions
    fmods = ViscousFlow.get_forcing_models(forcing)
    fcache = ForcingModelAndRegion(fmods,base_cache)

    # The state here is vorticity, the constraint is the surface traction
    f = ViscousFlow._get_ode_function_list(viscous_L,base_cache)


    #============ Wind tunnel correction problem ============#
    hasinlet = false
    if haskey(phys_params, "inlets")
        if !isempty(phys_params["inlets"])
            hasinlet = true
        end
    end

    wt_boundaries = create_windtunnel_boundaries(g,phys_params;withinlet=hasinlet)
    wt_tmp_boundaries = [ScalarData(points(wt_boundary)) for wt_boundary in wt_boundaries]
    wt_vn_boundaries = [ScalarData(points(wt_boundary)) for wt_boundary in wt_boundaries]

    wt_prob = PotentialFlowProblem(g,wt_boundaries,scaling=GridScaling,phys_params=phys_params)
    wt_sys = construct_system(wt_prob);

    wt_bool_surface = ScalarData(points(wt_boundaries))

    regop = ImmersedLayers._get_regularization(points(wt_boundaries),areas(wt_boundaries),base_cache.g, CartesianGrids.Goza,GridScaling)
    Rcurl = ImmersedLayers._regularization_matrix(regop, wt_bool_surface, zeros_gridcurl(base_cache))

    wt_bool = zeros_gridcurl(base_cache)
    wt_bool_surface .= 1
    mul!(wt_bool,Rcurl,wt_bool_surface)
    wt_bool .= (wt_bool .!= 0.0);

    wt_vel = zeros_grid(base_cache)
    wt_body_bc = zeros_surface(base_cache)

    WindTunnelCache(cdcache,
                    fcache,
                    dvb,
                    vb_tmp,
                    v_tmp,
                    dv,
                    dv_tmp,
                    w_tmp,
                    divv_tmp,
                    velcache,
                    f,
                    wt_sys,
                    wt_bool,
                    wt_vel,
                    wt_body_bc,
                    wt_tmp_boundaries,
                    wt_vn_boundaries)
end

function ViscousFlow.viscousflow_vorticity_bc_rhs!(vb,sys::ILMSystem{true,T},t) where {T<:WindTunnelProblem}
    @unpack bc, forcing,extra_cache, base_cache, phys_params = sys
    @unpack dvb, vb_tmp, v_tmp, velcache, divv_tmp = extra_cache
    @unpack dcache, ϕtemp = velcache

    ViscousFlow.viscousflow_velocity_bc_rhs!(vb,sys,t)

    # Subtract influence of non-wind-tunnel scalar potential field (coming from the jump in normal velocities on the body)
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
    vb .-= extra_cache.wt_body_bc

    return vb
end

function ViscousFlow.viscousflow_vorticity_ode_rhs!(dw,w,sys::ILMSystem{true,T},t) where {T<:WindTunnelProblem}
    @unpack extra_cache, base_cache = sys
    @unpack v_tmp, dv, wt_bool = extra_cache

    ViscousFlow.velocity!(v_tmp,w,sys,t)
    ViscousFlow.viscousflow_velocity_ode_rhs!(dv,v_tmp,sys,t)
    curl!(dw,dv,base_cache)

    # Remove vorticity coming from wind tunnel walls numerical artifacts (such as at the corners)
    dw .-= dw.*wt_bool

    return dw
end

"""
Note: while most `ViscousFlow` methods are reused, this method differs from the non-parametric `velocity!` method to account for the wind tunnel walls and sinks.
"""
function ViscousFlow.velocity!(v::Edges{Primal},w::Nodes{Dual},sys::ILMSystem{true,T},t) where {T<:WindTunnelProblem}
    @unpack forcing, phys_params, extra_cache, base_cache = sys
    @unpack dvb, velcache, divv_tmp, w_tmp, wt_vel, vb_tmp, wt_tmp_boundaries, wt_vn_boundaries = extra_cache

    # Compute the freestream velocity field and store it in gdata_cache
    freestream_func = ViscousFlow.get_freestream_func(forcing)
    Vinf = freestream_func(t,phys_params)
    vecfield_uniformvecfield!(base_cache.gdata_cache,Vinf[1],Vinf[2],base_cache)

    # Compute the uncorrected velocity field from the freestream velocity and vorticity `w` and store it in wt_vel
    prescribed_surface_jump!(dvb,t,sys)
    fill!(wt_vel,0.0)
    ViscousFlow.velocity!(wt_vel,w,divv_tmp,dvb,base_cache.gdata_cache,base_cache,velcache,w_tmp)

    # Interpolate the velocity onto the wind-tunnel boundaries and flip its sign
    wt_vn = extra_cache.wt_sys.extra_cache.helmcache.dcache.vn
    wt_dvn = extra_cache.wt_sys.extra_cache.helmcache.dcache.dvn
    normal_interpolate!(wt_vn,wt_vel,extra_cache.wt_sys)
    wt_vn .*= -1

    if haskey(phys_params, "inlets")
        for inlet in phys_params["inlets"]
            pts_inlet = points(extra_cache.wt_sys.base_cache.bl[inlet.wt_body])
            wt_vn_inlet = wt_vn_boundaries[inlet.wt_body]
            wt_vn_inlet .= 0.0
            tmp = wt_tmp_boundaries[inlet.wt_body]
            add_flow_through!(wt_vn_inlet,pts_inlet,tmp,inlet,sys,t)
            wt_vn[getrange(extra_cache.wt_sys.base_cache.bl,inlet.wt_body)] .+= wt_vn_inlet
        end
    end

    if haskey(phys_params, "outlets")
        for outlet in phys_params["outlets"]
            pts_outlet = points(extra_cache.wt_sys.base_cache.bl[outlet.wt_body])
            wt_vn_outlet = wt_vn_boundaries[outlet.wt_body]
            wt_vn_outlet .= 0.0
            tmp = wt_tmp_boundaries[outlet.wt_body]
            add_flow_through!(wt_vn_outlet,pts_outlet,tmp,outlet,sys,t)
            wt_vn[getrange(extra_cache.wt_sys.base_cache.bl,outlet.wt_body)] .+= wt_vn_outlet
            # wt_dvn[getrange(extra_cache.wt_sys.base_cache.bl,outlet.wt_body)] .+= wt_vn_outlet
        end
    end

    # Solve Neumann problem for ϕ and dϕ (double layer)
    dϕtemp = sys.extra_cache.wt_sys.extra_cache.dϕtemp
    GridPotentialFlow.solve!(velcache.ϕtemp,dϕtemp,wt_vn,wt_dvn,nothing,extra_cache.wt_sys,t)

    # Compute Gϕ̄ and overwrite wt_vel with the result
    grad!(wt_vel,velcache.ϕtemp,sys.extra_cache.wt_sys);

    # Eldredge JCP 2022 Eq 38: Ḡϕ̄ = Gϕ̄ - I(ϕ⁺-ϕ⁻)∘Rn
    # Double check if this is necessary
    regularize_normal!(velcache.vϕ,dϕtemp,sys.extra_cache.wt_sys)
    wt_vel .-= velcache.vϕ

    # Interpolate the potential flow velocity onto the body for use in viscousflow_vorticity_bc_rhs!
    interpolate!(extra_cache.wt_body_bc,wt_vel,base_cache)

    # Add the potential flow velocity field to the freestream velocity field
    base_cache.gdata_cache .+= wt_vel

    # Compute the corrected velocity field from the potential flow velocity, freestream velocity and vorticity `w` and store it in wt_vel. This velocity field will violate the boundary conditions on the body.
    fill!(divv_tmp,0.0)
    ViscousFlow.velocity!(v,w,divv_tmp,dvb,base_cache.gdata_cache,base_cache,velcache,w_tmp)
    return wt_vn
end

"""
Computes in-place the streamfunction field that accounts for the presence of the wind-tunnel walls.
"""
function ViscousFlow.streamfunction!(ψ::Nodes{Dual},w::Nodes{Dual},sys::ILMSystem{true,T},t;withcorrection=true) where {T<:WindTunnelProblem}
    @unpack phys_params, forcing, extra_cache, base_cache = sys
    @unpack dvb, velcache, divv_tmp, w_tmp, wt_vel, velcache, wt_tmp_boundaries, wt_vn_boundaries = extra_cache
    @unpack wcache = velcache

    # Same code as ViscousFlow.streamfunction!
    freestream_func = ViscousFlow.get_freestream_func(forcing)
    Vinf = freestream_func(t,phys_params)

    ViscousFlow.streamfunction!(ψ,w,Vinf,base_cache,wcache)

    if withcorrection
        # Compute uncorrected velocity field
        prescribed_surface_jump!(dvb,t,sys)
        ViscousFlow.velocity!(wt_vel,w,divv_tmp,dvb,Vinf,base_cache,velcache,w_tmp)

        # Compute boundary condition for correcting scalar potential
        wt_vn = extra_cache.wt_sys.extra_cache.helmcache.dcache.vn
        wt_dvn = extra_cache.wt_sys.extra_cache.helmcache.dcache.dvn
        normal_interpolate!(wt_vn,wt_vel,extra_cache.wt_sys)
        wt_vn .*= -1

        if haskey(phys_params, "inlets")
            for inlet in phys_params["inlets"]
                pts_inlet = points(extra_cache.wt_sys.base_cache.bl[inlet.wt_body])
                wt_vn_inlet = wt_vn_boundaries[inlet.wt_body]
                wt_vn_inlet .= 0.0
                tmp = wt_tmp_boundaries[inlet.wt_body]
                add_flow_through!(wt_vn_inlet,pts_inlet,tmp,inlet,sys,t)
                wt_vn[getrange(extra_cache.wt_sys.base_cache.bl,inlet.wt_body)] .+= wt_vn_inlet
            end
        end

        if haskey(phys_params, "outlets")
            for outlet in phys_params["outlets"]
                pts_outlet = points(extra_cache.wt_sys.base_cache.bl[outlet.wt_body])
                wt_vn_outlet = wt_vn_boundaries[outlet.wt_body]
                wt_vn_outlet .= 0.0
                tmp = wt_tmp_boundaries[outlet.wt_body]
                add_flow_through!(wt_vn_outlet,pts_outlet,tmp,outlet,sys,t)
                wt_vn[getrange(extra_cache.wt_sys.base_cache.bl,outlet.wt_body)] .+= wt_vn_outlet
                # wt_dvn[getrange(extra_cache.wt_sys.base_cache.bl,outlet.wt_body)] .+= wt_vn_outlet
            end
        end

        # Compute scalar potential and its jump on the surface
        dϕtemp = sys.extra_cache.wt_sys.extra_cache.dϕtemp
        dψtemp = sys.extra_cache.wt_sys.extra_cache.γtemp
        GridPotentialFlow.solve!(velcache.ϕtemp,dϕtemp,wt_vn,wt_dvn,nothing,extra_cache.wt_sys,t)

        # Compute the streamfunction from the jump in scalar potential and normal velocity
        GridPotentialFlow.solve!(velcache.ψtemp,dψtemp,dϕtemp,wt_dvn,sys.extra_cache.wt_sys,t);

        ψ .+= velcache.ψtemp
    end
end

"""
Creates the boundaries of the wind tunnel where boundary conditions will be imposed for the correcting potential flow.

Returns a `BodyList` with elements:

    1. top boundary
    2. bottom boundary
    3. inlet boundary
"""
function create_windtunnel_boundaries(g,phys_params;withinlet=true)
    Δs = surface_point_spacing(g,phys_params)
    haskey(phys_params,"wind tunnel length") || error("No wind tunnel length set")
    haskey(phys_params,"wind tunnel height") || error("No wind tunnel height set")

    L = phys_params["wind tunnel length"]
    H = phys_params["wind tunnel height"]
    cent = get(phys_params,"wind tunnel center",(0.0,0.0))

    plate_tb = Plate(L,Δs);
    bl = BodyList([deepcopy(plate_tb),deepcopy(plate_tb)])
    t1! = RigidTransform((cent[1],cent[2]+H/2),0.0)
    t2! = RigidTransform((cent[1],cent[2]-H/2),π)
    tl! = RigidTransformList([t1!,t2!])
    tl!(bl)

    if withinlet
        inlet = Plate(H,Δs)
        RigidTransform((cent[1]-L/2,cent[2]),π/2)(inlet)
        push!(bl,inlet)
    end

    return bl
end

struct UniformFlowThrough
    boundary :: Body
    set_velocity! :: Function # per unit length. If the normals of the boundary point to the interior of the wind tunnel, a positive strength gives inflow and a negative strength gives outflow
    wt_body :: Int # body of the wind tunnel walls on which the uniform flow through has to be applied exclusively
    pts :: VectorData
    scalar_cache :: ScalarData
end

UniformFlowThrough(boundary, set_velocity!, wt_body) = UniformFlowThrough(boundary, set_velocity!, wt_body, points(boundary), ScalarData(points(boundary)))

function add_flow_through!(vn, vn_pts, tmp, flow, sys, t)
    @unpack phys_params, extra_cache, base_cache = sys
    ds = surface_point_spacing(base_cache.g,phys_params)

    dS = dlength(flow.boundary)[1] # Should change to using the full ds vector
    flow.set_velocity!(flow.scalar_cache,flow.pts,t,phys_params)

    line_regularize!(vn,vn_pts,flow.scalar_cache,flow.pts,tmp,ds,dS)
end

function line_regularize!(vn::ScalarData, vn_pts::VectorData, q::ScalarData, q_pts::VectorData, tmp::ScalarData, ds, dS; ddf_radius=1.0, ddf=CartesianGrids.ddf_witchhat)
    for i in 1:length(q)
        @inbounds tmp .= sqrt.((vn_pts.u .- q_pts.u[i]) .^2 .+ (vn_pts.v .- q_pts.v[i]) .^ 2)
        @inbounds vn .+= q[i] .* ddf.(tmp ./ ds) .* dS ./ ds
    end
end

end # module
