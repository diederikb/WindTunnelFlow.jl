using WindTunnelFlow
using JSON
using DelimitedFiles
using Plots

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
V_in_star = V_in / c # m/s

# Compute other wind tunnel parameters
A_TS_star = H_TS_star * W_TS_star
Q_in_star = V_in_star * A_TS_star
Q_SD_star = Q_SD_over_Q_in * Q_in_star
x_SD_lo_star = x_SD_lo_over_L_TS * L_TS_star
x_SD_hi_star = x_SD_hi_over_L_TS * L_TS_star
L_SD_star = x_SD_hi_star - x_SD_lo_star
A_SD_star = L_SD_star * W_TS_star
V_SD_star = Q_SD_star / A_SD_star

params = Dict()
params["Re"] = Re
params["grid Re"] = grid_Re
params["wind tunnel length"] = L_TS_star
params["wind tunnel height"] = H_TS_star
params["wind tunnel center"] = (L_TS_star / 2, H_TS_star / 2)
params["freestream speed"] = V_in_star
params["freestream angle"] = 0.0
params["V_in"] = V_in_star
params["V_SD"] = V_SD_star
params["sigma_suction"] = sigma_suction
params["t_suction"] = t_suction
xlim = (-0.05 * L_TS_star, 1.05 * L_TS_star)
ylim = (-0.05 * H_TS_star, 1.05 * H_TS_star)
g = setup_grid(xlim, ylim, params)

# Airfoil in the test section
Δs = surface_point_spacing(g,params)
airfoil = NACA4(0.0, 0.0, 0.09, 300, len=c_star)
airfoil = SplinedBody(airfoil.x, airfoil.y, Δs)
T = RigidTransform((L_TS_star / 2, H_TS_star / 2), -α*π/180)
T(airfoil) # transform the body to the current configuration

# Create the inflow
N = ceil(Int, H_TS_star / surface_point_spacing(g,params))
inflow_boundary = BasicBody(
    zeros(N),
    collect(range(0, H_TS_star, N)),
    closuretype=RigidBodyTools.OpenBody)
inflow = UniformFlowThrough(inflow_boundary,inflow_velocity!,3)

params["inlets"] = [inflow]

# Create the suction at the top of the wind tunnel
N = ceil(Int, L_SD_star / surface_point_spacing(g,params))
suction_boundary = BasicBody(
    collect(range(x_SD_lo_star, x_SD_hi_star, N)),
    H_TS_star * ones(N);
    closuretype=RigidBodyTools.OpenBody)
suction = UniformFlowThrough(suction_boundary,suction_velocity!,1)

params["outlets"] = [suction]

# Create the wind tunnel problem
prob = WindTunnelProblem(g,airfoil,phys_params=params;timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
                                   bc=ViscousFlow.get_bc_func(nothing))
sys = construct_system(prob);

# Initialize the solution and integrator
u0 = init_sol(sys)
tspan = (0.0,t_final)
integrator = init(u0,tspan,sys);

# Run
step!(integrator,t_final)

# Compute force
sol = integrator.sol;
fx, fy = force(sol,sys,1)

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
open("$(case)_output.txt", "w") do io
    writedlm(io, [sol.t Q_suction/Q_in_star fx fy])
end

anim_sample_freq = 5 # samples per simulated time unit
anim_fps = 15 # frames per second of real time
Δt = prob.timestep_func(sys) # simulated time per time step
anim_sample_step = ceil(Int,1/(Δt*anim_sample_freq)) # time steps per sample

# Make animation
wt_walls = create_windtunnel_boundaries(g,params,withinlet=false)
ψ = zeros_gridcurl(sys)
ViscousFlow.streamfunction!(ψ,sol.u[end].x[1],sys,sol.t[end])
y_probe = 0:0.1*H_TS_star:H_TS_star
ψ_fcn = interpolatable_field(ψ,g)
ψ_probe = ψ_fcn.(0.0,y_probe)

anim = @animate for i in 1:anim_sample_step:length(sol.t)
#     l = @layout [a{0.6w} [Plots.grid(2,1)]]
    ViscousFlow.streamfunction!(ψ,sol.u[i].x[1],sys,sol.t[i])
    p1=plot(ψ,sys,c=:gray,levels=ψ_probe,title="t = $(round(integrator.sol.t[i]; digits=1))",xlabel="\$x/c\$",ylabel="\$y/c\$",clim=(-10,10))
    plot!(sol.u[i].x[1],sys,clim=(-15,15),color=cgrad(:RdBu, rev = true),levels=range(-15,15,length=30))
    plot!(wt_walls,xlim=xlim,ylim=ylim,lc=:black,lw=2)
    plot!(suction.boundary,lc=:red,lw=2)
    plot!(airfoil,fc=:white,lc=:black)
    p2=plot(sol.t[1:i],Q_suction[1:i]/Q_in_star,xlim=(0.0,sol.t[end]),ylim=(0,1),ylabel="\$Q_{suction}/Q_{in}\$",legend=false)
    p3=plot(sol.t[1:i],fy[1:i],xlim=(0.0,integrator.sol.t[end]),ylim=(-3,3),xlabel="\$tU/c\$",ylabel="\$C_L\$",legend=false)
#     plot(p1,p2,p3,layout = l,size=(1000,300),margin=4mm)
    plot(p1,p2,p3, layout=Plots.grid(3, 1, heights=[0.4 ,0.15, 0.45]),size=(600,670))
end
gif(anim, "$(case).gif", fps=anim_fps)
