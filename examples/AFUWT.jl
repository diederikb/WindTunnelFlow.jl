using WindTunnelFlow
using JSON
using DelimitedFiles
using Plots
using Measures
using LinearAlgebra
using BenchmarkTools
using Statistics
using Interpolations

ENV["GKSwstype"]="nul"

# Parse input file
files = readdir()
json_file_id = findall(f->occursin(r"AFUWT.*\.json",f),files)[1]
json_file = files[json_file_id]
inputs = JSON.parsefile(json_file)

# Create the wind tunnel problem
include("AFUWT_create_sys.jl")

# Write grid parameters
grid_dict = Dict(key=>getfield(g, key) for key ∈ fieldnames(PhysicalGrid))
open("$(case)_grid.json", "w") do io
    JSON.print(io, grid_dict)
end

# Define a start and end time during which we want to save the solution for all timesteps
Δt = prob.timestep_func(sys) # simulated time per time step
save_skip = Int(ceil((5/params["grid Re"])^3))

if occursin("step_opening_closing",lowercase(gust_type))
    t_gust_start = t_open
    t_gust_end = t_close + tau_close + 2 * V_in_star / c_star
elseif occursin("gust_from_file",lowercase(gust_type))
    t_gust_start = inputs["t_gust_start"]
    t_gust_end = inputs["t_gust_end"]
else
    t_gust_start = t_suction - 4 * sigma_suction
    t_gust_end = t_suction + 4 * sigma_suction + 0.5 * V_in_star / c_star
end
# Construct an array with the times we want to save the solution
save_times = cat(
    0:save_skip*save_skip*Δt:t_gust_start-save_skip*Δt,
    t_gust_start:Δt:t_gust_end,
    t_gust_end+save_skip*Δt:save_skip*Δt:t_final,
    dims=1)

# Initialize the solution and integrator
print("Initializing solution... ")
u0 = init_sol(sys)
print("done\n")
flush(stdout)
tspan = (0.0,t_final)
print("Initializing integrator... ")
flush(stdout)
integrator = init(u0,tspan,sys,alg=ConstrainedSystems.LiskaIFHERK(maxiter=1),saveat=save_times);
print("done\n")
flush(stdout)

# Run (with a benchmark test at the beginning)
print("Running solver... ")
flush(stdout)
b = @benchmark step!($integrator)
io = IOBuffer()
show(io, "text/plain", b)
s = String(take!(io))
println(s)

for (u,t) in tuples(integrator)
    println(t)
    flush(stdout)
end
print("Solver finished\n")
flush(stdout)

# Compute force
sol = integrator.sol;
fx_wt, fy_wt = force(sol,sys,1)
m_wt = moment(sol,sys,1)

# Compute suction ratio history
pts = points(suction.boundary)
vel = ScalarData(pts)
Q_suction = []
for i in 1:length(sol.t)
    suction_velocity!(vel,suction.boundary,sol.t[i],params)
    Q_suction_i = -integrate(vel,ScalarData(dlength(suction.boundary))) * W_TS_star
    push!(Q_suction,Q_suction_i)
end

# Write solution output during gust
print("Writing solution output during gust... ")
flush(stdout)
idx = findall(t_gust_start .<= sol.t .<= t_gust_end)[1:4:end]
# Ensure that data for snapshots is written to files
for snapshot_time in snapshot_times
    push!(idx,findfirst(isapprox.(sol.t,snapshot_time,rtol=1e-6)))
end
for i in idx
    open("$(case)_snapshot_$(i)_vorticity_wind_tunnel.txt", "w") do io
        writedlm(io, sol.u[i].x[1])
    end
    open("$(case)_snapshot_$(i)_time_wind_tunnel.txt", "w") do io
        writedlm(io, sol.t[i])
    end
end
print("done\n")
flush(stdout)

print("Writing force and moment history of the wind tunnel flow...")
flush(stdout)
# Write force output
open("$(case)_force_wind_tunnel.txt", "w") do io
    writedlm(io, [sol.t fx_wt fy_wt])
end
# Write moment output
open("$(case)_moment_wind_tunnel.txt", "w") do io
    writedlm(io, [sol.t m_wt])
end
print("done\n")
flush(stdout)

# Probe the velocity history at LE, center and TE of the body when the body is not present to use as freestream for a ViscousFlow.jl and Wagner simulation
print("Creating probe WindTunnelProblem... ")
flush(stdout)
probe_prob = WindTunnelProblem(g,phys_params=params;timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
                                   bc=ViscousFlow.get_bc_func(nothing))
probe_sys = construct_system(probe_prob);
print("done\n")
flush(stdout)

flat_plate_probe = Plate(c_star,Δs)
T = RigidTransform((L_TS_star / 2 + x̃_O_WT_star, H_TS_star / 2 + ỹ_O_WT_star), -α*π/180)
T(flat_plate_probe) # transform the body to the current configuration

center = (L_TS_star / 2 + x_O_WT_star, H_TS_star / 2 + y_O_WT_star)
LE = (flat_plate_probe.x[1],flat_plate_probe.y[1])
TE = (flat_plate_probe.x[end],flat_plate_probe.y[end])
center = ((LE[1]+TE[1])/2,(LE[2]+TE[2])/2)

Umid_hist = Vector()
ULE_hist = Vector()
UTE_hist = Vector()
Umean_hist = Vector()

Vmid_hist = Vector()
VLE_hist = Vector()
VTE_hist = Vector()
Vmean_hist = Vector()

wt_vel = zeros_grid(sys);

print("Probing velocity...")
flush(stdout)
for i in 1:length(sol.t)
    ViscousFlow.velocity!(wt_vel, zeros_gridcurl(sys), sys, sol.t[i]);
    vel_fcn = interpolatable_field(wt_vel,g);

    push!(Umid_hist,vel_fcn[1](center[1],center[2]))
    push!(ULE_hist,vel_fcn[1](LE[1],LE[2]))
    push!(UTE_hist,vel_fcn[1](TE[1],TE[2]))
    push!(Umean_hist,mean(vel_fcn[1].(plate.x,plate.y)))

    push!(Vmid_hist,vel_fcn[2](center[1],center[2]))
    push!(VLE_hist,vel_fcn[2](LE[1],LE[2]))
    push!(VTE_hist,vel_fcn[2](TE[1],TE[2]))
    push!(Vmean_hist,mean(vel_fcn[2].(plate.x,plate.y)))
end
print("done\n")
flush(stdout)

open("$(case)_Q_and_V_probe.txt", "w") do io
    writedlm(io, [sol.t Q_suction/Q_in_star ULE_hist Umid_hist UTE_hist Umean_hist VLE_hist Vmid_hist VTE_hist Vmean_hist])
end

function interpolate_freestream(t,phys_params)
    U_interp  = phys_params["Umean_interp"]
    V_interp  = phys_params["Vmean_interp"]
    return U_interp(t), V_interp(t)
end

params["Umean_interp"] = LinearInterpolation(sol.t,Umean_hist)
params["Vmean_interp"] = LinearInterpolation(sol.t,Vmean_hist)
params["freestream"] = interpolate_freestream

# ViscousFlow.jl simulation
print("Creating ViscousIncompressibleFlowProblem... ")
flush(stdout)
viscous_prob = ViscousIncompressibleFlowProblem(
    g,
    airfoil,
    phys_params=params;
    timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
    bc=ViscousFlow.get_bc_func(nothing))
viscous_sys = construct_system(viscous_prob);
print("done\n")
flush(stdout)

# Initialize the solution and integrator
print("Initializing solution... ")
flush(stdout)
u0 = init_sol(viscous_sys)
print("done\n")
flush(stdout)
tspan = (0.0,t_final)
print("Initializing integrator... ")
flush(stdout)
integrator = init(u0,tspan,viscous_sys;saveat=save_times);
print("done\n")
flush(stdout)

# Run
print("Running solver...\n")
flush(stdout)
for (u,t) in tuples(integrator)
    println(t)
    flush(stdout)
end
print("Solver finished\n")
flush(stdout)

# Compute force
sol = integrator.sol;
fx_viscous, fy_viscous = force(sol,viscous_sys,1)
m_viscous = moment(sol,viscous_sys,1)

# Write solution output during gust
print("Writing solution output during gust... ")
flush(stdout)
for i in idx
    open("$(case)_snapshot_$(i)_vorticity_viscous_flow.txt", "w") do io
        writedlm(io, sol.u[i].x[1])
    end
    open("$(case)_snapshot_$(i)_time_viscous_flow.txt", "w") do io
        writedlm(io, sol.t[i])
    end
end
print("done\n")
flush(stdout)

print("Writing force and moment history of the viscous flow...")
flush(stdout)
# Write force output
open("$(case)_force_viscous_flow.txt", "w") do io
    writedlm(io, [sol.t fx_viscous fy_viscous])
end

# Write moment output
open("$(case)_moment_viscous_flow.txt", "w") do io
    writedlm(io, [sol.t m_viscous])
end
print("done\n")
flush(stdout)

plot(sol.t,fx_wt,label="Viscous flow (in wind tunnel)",legend=:topleft,xlabel="convective time",ylabel="C_D")
plot!(sol.t,fx_viscous,label="Viscous flow")
savefig("$(case)_CD.pdf")

plot(sol.t,fy_wt,label="Viscous flow (in wind tunnel)",legend=:topleft,xlabel="convective time",ylabel="C_L")
plot!(sol.t,fy_viscous,label="Viscous flow")
savefig("$(case)_CL.pdf")

plot(sol.t,Umid_hist,label="U mid-chord",xlabel="convective time")
plot!(sol.t,ULE_hist,label="U LE")
plot!(sol.t,UTE_hist,label="U TE")
savefig("$(case)_U_probe_history.pdf")

plot(sol.t,Vmid_hist,label="V mid-chord",xlabel="convective time")
plot!(sol.t,VLE_hist,label="V LE")
plot!(sol.t,VTE_hist,label="V TE")
savefig("$(case)_V_probe_history.pdf")
