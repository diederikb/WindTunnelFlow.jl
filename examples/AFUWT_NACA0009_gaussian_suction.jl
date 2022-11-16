using WindTunnelFlow
using JSON
using DelimitedFiles
using Plots

#println("Number of Julia threads: $(Threads.threads())")

ENV["GKSwstype"]="nul"

function suction_velocity!(vel,pts,t,phys_params)
    V_out = phys_params["V_SD"]
    σ = phys_params["sigma_suction"]
    t_0 = phys_params["t_suction"]
    g = Gaussian(σ,sqrt(π*σ^2)) >> t_0
    vel .= -V_out*g(t)
end

function inflow_velocity!(vel,pts,t,phys_params)
    V_in = phys_params["V_in"]
    vel .= V_in
end

# Parse input file
parsed_inputs = JSON.parsefile("AFUWT_NACA0009_gaussian_suction.json")

case = parsed_inputs["case"]
c = parsed_inputs["c"] # m
α = parsed_inputs["alpha"] # degrees
Re = parsed_inputs["Re"]
t_final = parsed_inputs["t_final"]
H_TS = parsed_inputs["H_TS"] # m
W_TS = parsed_inputs["W_TS"] # m
L_TS = parsed_inputs["L_TS"] # m
Q_SD_over_Q_in = parsed_inputs["Q_SD_over_Q_in"]
x_SD_lo_over_L_TS = parsed_inputs["x_SD_lo_over_L_TS"]
x_SD_hi_over_L_TS = parsed_inputs["x_SD_hi_over_L_TS"]
sigma_suction = parsed_inputs["sigma_suction"]
t_suction = parsed_inputs["t_suction"]
grid_Re = get(parsed_inputs,"grid_Re",2.0)

# Rescale every length by the chord length
c_star = c / c
H_TS_star = H_TS / c
W_TS_star = W_TS / c
L_TS_star = L_TS / c

# Set V_in_star to 1
V_in = 1.0 * c # m/s
V_in_star = V_in / c

# Compute other wind tunnel parameters in scaled form
A_TS_star = H_TS_star * W_TS_star # Test section area
Q_in_star = V_in_star * A_TS_star # Inlet flow rate
Q_SD_star = Q_SD_over_Q_in * Q_in_star # Suction duct flow reate
x_SD_lo_star = x_SD_lo_over_L_TS * L_TS_star # Lowest x-coordinate of the suction opening
x_SD_hi_star = x_SD_hi_over_L_TS * L_TS_star # Highest x-coordinate of the suction opening
L_SD_star = x_SD_hi_star - x_SD_lo_star # Length of the suction opening
A_SD_star = L_SD_star * W_TS_star # Area of the suction opening
V_SD_star = Q_SD_star / A_SD_star # Flow velocity through the suction opening
x_O_WT_star = -L_TS_star/2 # x-coordinate of the wind tunnel frame origin using the center of the body as the origin
y_O_WT_star = -H_TS_star/2 # y-coordinate of the wind tunnel frame origin using the center of the body as the origin

params = Dict()
params["Re"] = Re
params["grid Re"] = grid_Re
params["wind tunnel length"] = L_TS_star
params["wind tunnel height"] = H_TS_star
params["wind tunnel center"] = (L_TS_star / 2 + x_O_WT_star, H_TS_star / 2 + y_O_WT_star)
params["freestream speed"] = V_in_star
params["freestream angle"] = 0.0
params["sigma_suction"] = sigma_suction
params["t_suction"] = t_suction
params["V_in"] = V_in_star
params["V_SD"] = V_SD_star
xlim = (-0.05 * L_TS_star + x_O_WT_star, 1.05 * L_TS_star + x_O_WT_star)
ylim = (-0.05 * H_TS_star + y_O_WT_star, 1.05 * H_TS_star + y_O_WT_star)
g = setup_grid(xlim, ylim, params)

println(g)
flush(stdout)

# Airfoil in the test section
Δs = surface_point_spacing(g,params)
airfoil = NACA4(0.0, 0.0, 0.09, 300, len=c_star)
airfoil = SplinedBody(airfoil.x, airfoil.y, Δs)
T = RigidTransform((L_TS_star / 2 + x_O_WT_star, H_TS_star / 2 + y_O_WT_star), -α*π/180)
T(airfoil) # transform the body to the current configuration

# Create the inflow
N = ceil(Int, H_TS_star / surface_point_spacing(g,params))
inflow_boundary = BasicBody(
    ones(N) * x_O_WT_star,
    collect(range(0, H_TS_star, N)) .+ y_O_WT_star,
    closuretype=RigidBodyTools.OpenBody)
inflow = UniformFlowThrough(inflow_boundary,inflow_velocity!,3)

params["inlets"] = [inflow]

# Create the suction at the top of the wind tunnel
N = ceil(Int, L_SD_star / surface_point_spacing(g,params))
suction_boundary = BasicBody(
    collect(range(x_SD_lo_star, x_SD_hi_star, N)) .+ x_O_WT_star,
    H_TS_star * ones(N) .+ y_O_WT_star;
    closuretype=RigidBodyTools.OpenBody)
suction = UniformFlowThrough(suction_boundary,suction_velocity!,1)

params["outlets"] = [suction]

# Create the wind tunnel problem
print("Creating WindTunnelProblem... ")
flush(stdout)
prob = WindTunnelProblem(g,airfoil,phys_params=params;timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
                                   bc=ViscousFlow.get_bc_func(nothing))
sys = construct_system(prob);
print("done\n")
flush(stdout)

# Initialize the solution and integrator
print("Initializing solution... ")
u0 = init_sol(sys)
print("done\n")
flush(stdout)
tspan = (0.0,t_final)
print("Initializing integrator... ")
flush(stdout)
integrator = init(u0,tspan,sys,alg=ConstrainedSystems.LiskaIFHERK(maxiter=1));
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
fx_wt, fy_wt = force(sol,sys,1)

# Compute suction ratio history
pts = points(suction.boundary)
vel = ScalarData(pts)
Q_suction = []
for i in 1:length(sol.t)
    suction_velocity!(vel,suction.boundary,sol.t[i],params)
    Q_suction_i = -integrate(vel,ScalarData(dlength(suction.boundary))) * W_TS_star
    push!(Q_suction,Q_suction_i)
end

# Write output
open("$(case)_force_wind_tunnel.txt", "w") do io
    writedlm(io, [sol.t fx_wt fy_wt])
end

anim_sample_freq = 5 # samples per simulated time unit
anim_fps = 15 # frames per second of real time
Δt = prob.timestep_func(sys) # simulated time per time step
anim_sample_step = ceil(Int,1/(Δt*anim_sample_freq)) # time steps per sample

# Make animation
wt_walls = create_windtunnel_boundaries(g,params,withinlet=false)
ψ = zeros_gridcurl(sys)
ViscousFlow.streamfunction!(ψ,sol.u[end].x[1],sys,sol.t[end])
y_probe = (0:0.1*H_TS_star:H_TS_star) .+ y_O_WT_star

anim = @animate for i in 1:anim_sample_step:length(sol.t)
#     l = @layout [a{0.6w} [Plots.grid(2,1)]]
    ViscousFlow.streamfunction!(ψ,sol.u[i].x[1],sys,sol.t[i])
    ψ_fcn = interpolatable_field(ψ,g)
    ψ_probe = ψ_fcn.(x_O_WT_star,y_probe)
    p1=plot(ψ,sys,c=:gray,levels=ψ_probe,title="t = $(round(integrator.sol.t[i]; digits=1))",xlabel="\$x/c\$",ylabel="\$y/c\$",clim=(-10,10))
    plot!(sol.u[i].x[1],sys,clim=(-15,15),color=cgrad(:RdBu, rev = true),levels=range(-15,15,length=30))
    plot!(wt_walls,xlim=xlim,ylim=ylim,lc=:black,lw=2)
    plot!(suction.boundary,lc=:red,lw=2)
    plot!(airfoil,fc=:white,lc=:black)
    p2=plot(sol.t[1:i],Q_suction[1:i]/Q_in_star,xlim=(0.0,sol.t[end]),ylim=(0,1),ylabel="\$Q_{suction}/Q_{in}\$",legend=false)
    p3=plot(sol.t[1:i],fy_wt[1:i],xlim=(0.0,integrator.sol.t[end]),ylim=(-1,1),xlabel="\$tU/c\$",ylabel="\$C_L\$",legend=false)
#     plot(p1,p2,p3,layout = l,size=(1000,300),margin=4mm)
    plot(p1,p2,p3, layout=Plots.grid(3, 1, heights=[0.4 ,0.15, 0.45]),size=(600,670))
end
gif(anim, "$(case)_vorticity_Qratio_CL.gif", fps=anim_fps)

anim = @animate for i in 1:anim_sample_step:length(sol.t)
#     l = @layout [a{0.6w} [Plots.grid(2,1)]]
    ViscousFlow.streamfunction!(ψ,sol.u[i].x[1],sys,sol.t[i])
    ψ_fcn = interpolatable_field(ψ,g)
    ψ_probe = ψ_fcn.(x_O_WT_star,y_probe)
    plot(ψ,sys,c=:gray,levels=ψ_probe,title="t = $(round(integrator.sol.t[i]; digits=1))",xlabel="\$x/c\$",ylabel="\$y/c\$",clim=(-10,10))
    plot!(sol.u[i].x[1],sys,clim=(-15,15),color=cgrad(:RdBu, rev = true),levels=range(-15,15,length=30))
    plot!(wt_walls,xlim=xlim,ylim=ylim,lc=:black,lw=2)
    plot!(suction.boundary,lc=:red,lw=2)
    plot!(airfoil,fc=:white,lc=:black)
end
gif(anim, "$(case)_vorticity.gif", fps=anim_fps)

# Probe the velocity history at LE, center and TE of the body when the body is not present to use as freestream for a ViscousFlow.jl and Wagner simulation
print("Creating probe WindTunnelProblem... ")
flush(stdout)
probe_prob = WindTunnelProblem(g,phys_params=params;timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
                                   bc=ViscousFlow.get_bc_func(nothing))
probe_sys = construct_system(probe_prob);
print("done\n")
flush(stdout)

center = (L_TS_star / 2 + x_O_WT_star, H_TS_star / 2 + y_O_WT_star)
LE = (center[1] - 1/2*cos(α*π/180), center[2] + 1/2*sin(α*π/180))
TE = (center[1] + 1/2*cos(α*π/180), center[2] - 1/2*sin(α*π/180))

Umid_hist = Vector()
ULE_hist = Vector()
UTE_hist = Vector()

Vmid_hist = Vector()
VLE_hist = Vector()
VTE_hist = Vector()

wt_vel = zeros_grid(sys);

print("Probing velocity...")
flush(stdout)
for i in 1:length(sol.t)
    ViscousFlow.velocity!(wt_vel, zeros_gridcurl(sys), sys, sol.t[i]);
    vel_fcn = interpolatable_field(wt_vel,g);

    push!(Umid_hist,vel_fcn[1](center[1],center[2]))
    push!(ULE_hist,vel_fcn[1](LE[1],LE[2]))
    push!(UTE_hist,vel_fcn[1](TE[1],TE[2]))

    push!(Vmid_hist,vel_fcn[2](center[1],center[2]))
    push!(VLE_hist,vel_fcn[2](LE[1],LE[2]))
    push!(VTE_hist,vel_fcn[2](TE[1],TE[2]))
end
print("done\n")
flush(stdout)

open("$(case)_Q_and_V_probe.txt", "w") do io
    writedlm(io, [sol.t Q_suction/Q_in_star ULE_hist Umid_hist UTE_hist VLE_hist Vmid_hist VTE_hist])
end

plot(integrator.sol.t,Umid_hist,label="U mid-chord",xlabel="convective time")
plot!(integrator.sol.t,ULE_hist,label="U LE")
plot!(integrator.sol.t,UTE_hist,label="U TE")
savefig("$(case)_U_probe_history.pdf")

plot(integrator.sol.t,Vmid_hist,label="V mid-chord",xlabel="convective time")
plot!(integrator.sol.t,VLE_hist,label="V LE")
plot!(integrator.sol.t,VTE_hist,label="V TE")
savefig("$(case)_V_probe_history.pdf")

# Clear some memory
probe_sys = nothing
integrator = nothing

function gaussian_freestream(t,phys_params)
    Uinf = get(phys_params,"freestream speed",0.0)
    U_mid = get(phys_params,"U_mid",0.0)
    V_mid = get(phys_params,"V_mid",0.0)
    σ = phys_params["sigma_suction"]
    t_0 = phys_params["t_suction"]
    g = Gaussian(σ,sqrt(π*σ^2)) >> t_0
    return Uinf - (Uinf - U_mid)*g(t), V_mid*g(t)
end

params["U_mid"] = minimum(Umid_hist)
params["V_mid"] = maximum(Vmid_hist)
forcing_dict = Dict("freestream" => gaussian_freestream)

# ViscousFlow.jl simulation
print("Creating ViscousIncompressibleFlowProblem... ")
flush(stdout)
viscous_prob = ViscousIncompressibleFlowProblem(g,airfoil,phys_params=params;timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
                                   bc=ViscousFlow.get_bc_func(nothing),forcing=forcing_dict)
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
integrator = init(u0,tspan,viscous_sys);
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

# Write output
open("$(case)_force_no_wind_tunnel.txt", "w") do io
    writedlm(io, [sol.t fx_viscous fy_viscous])
end

# Wagner response
Φ(t) = t ≥ 0 ? 1.0 - 0.165*exp(-0.091*t) - 0.335*exp(-0.6*t) : 0.0

function duhamelintegral(f_array,t_array,ind_fun)
    s = 0.0
    Δt = diff(t_array)
    for i in 2:length(t_array)
        s += f_array[i]*ind_fun(t_array[end]-t_array[i])*Δt[i-1]
    end
    return s
end

ḣ_old = 0.0
Γb_old = 0.0
fy_wagner = Vector()
Γ̇b_hist = Vector()

Δt_hist = diff(sol.t)

Γ_b0 = -π*c_star*V_in_star*α*pi/180

for i in 1:length(integrator.sol.t)-1
    ḣ = -Vmid_hist[i+1]
    ḧ = (ḣ-ḣ_old)/Δt_hist[i]
    ḣ_old = ḣ
    Γb = π*c_star*ḣ
    Γ̇b = (Γb-Γb_old)/Δt_hist[i]
    Γb_old = Γb

    push!(Γ̇b_hist,Γ̇b)

    fy_wagner_added_mass_i = -π/4*c_star^2*ḧ
    fy_wagner_i = fy_wagner_added_mass_i - Γ_b0 * Φ(sol.t[i]) - duhamelintegral(Γ̇b_hist,sol.t[2:i],Φ)

    push!(fy_wagner,fy_wagner_i)
end

# Write output
open("$(case)_force_wagner.txt", "w") do io
    writedlm(io, [sol.t[1:end-1] fx_wagner fy_wagner])
end

plot(sol.t,fx_wt,label="Viscous flow (in wind tunnel)",legend=:topleft,xlabel="convective time",ylabel="C_D")
plot!(sol.t,fx_viscous,label="Viscous flow")
plot!(sol.t[1:end-1],fx_wagner,label="Wagner")
savefig("$(case)_CD.pdf")

plot(sol.t,fy_wt,label="Viscous flow (in wind tunnel)",legend=:topleft,xlabel="convective time",ylabel="C_L")
plot!(sol.t,fy_viscous,label="Viscous flow")
plot!(sol.t[1:end-1],fy_wagner,label="Wagner")
savefig("$(case)_CL.pdf")
