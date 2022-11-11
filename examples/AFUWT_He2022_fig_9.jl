using WindTunnelFlow
using JSON
using Plots

function outflow_velocity(vel,pts,t,phys_params)
    V_out = phys_params["V_out"]
    vel .= -V_out
end

function inflow_velocity(vel,pts,t,phys_params)
    V_in = phys_params["V_in"]
    vel .= V_in
end

# Parse input file
parsed_inputs = JSON.parsefile("AFUWT_He2022_fig_9.json")

case = parsed_inputs["case"]
H_TS = parsed_inputs["H_TS"] # m
W_TS = parsed_inputs["W_TS"] # m
L_TS = parsed_inputs["L_TS"] # m
u_0 = parsed_inputs["u_0"] # m/s
V_in_over_u_0 = parsed_inputs["V_in_over_u_0"] # He 2022, figure 9
V_TS_over_u_0 = parsed_inputs["V_TS_over_u_0"] # He 2022, figure 9
x_SD_lo_over_L_TS = parsed_inputs["x_SD_lo_over_L_TS"]
x_SD_hi_over_L_TS = parsed_inputs["x_SD_hi_over_L_TS"]

# Compute other wind tunnel parameters
V_in = V_in_over_u_0 * u_0
V_TS = V_TS_over_u_0 * u_0

A_TS = H_TS * W_TS # m^2
Q_TS = V_TS_over_u_0 * A_TS # m^3/s
Q_in = V_in_over_u_0 * A_TS # m^3/s
Q_SD = Q_in - Q_TS

x_SD_lo = x_SD_lo_over_L_TS * L_TS
x_SD_hi = x_SD_hi_over_L_TS * L_TS
L_SD = x_SD_hi - x_SD_lo
A_SD = L_SD * W_TS
V_out = Q_SD / A_SD

params = Dict()
params["Re"] = 200
params["grid Re"] = 2.0
params["wind tunnel length"] = L_TS
params["wind tunnel height"] = H_TS
params["wind tunnel center"] = (L_TS / 2, H_TS / 2)
params["freestream speed"] = V_in
params["freestream angle"] = 0.0
params["V_in"] = V_in
params["V_out"] = V_out
params["V_TS"] = V_TS
t_final = 5.0
xlim = (-0.1 * L_TS, 1.1 * L_TS)
ylim = (-0.1 * H_TS, 1.1 * H_TS)
g = setup_grid(xlim, ylim, params)

# Create the inflow
N = ceil(Int, H_TS / surface_point_spacing(g,params))
inflow_boundary = BasicBody(
    zeros(N),
    collect(range(0, H_TS, N)),
    closuretype=RigidBodyTools.OpenBody)
inflow = UniformFlowThrough(inflow_boundary,inflow_velocity,3)

params["inlets"] = [inflow]

# Create the suction at the top of the wind tunnel
N = ceil(Int, L_SD / surface_point_spacing(g,params))
outflow_boundary = BasicBody(
    collect(range(x_SD_lo, x_SD_hi, N)),
    H_TS * ones(N);
    closuretype=RigidBodyTools.OpenBody)
outflow = UniformFlowThrough(outflow_boundary,outflow_velocity,1)

params["outlets"] = [outflow]

# Create the wind tunnel problem
prob = WindTunnelProblem(g,phys_params=params;timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
                                   bc=ViscousFlow.get_bc_func(nothing))
sys = construct_system(prob);

# Initialize the solution and integrator
u0 = init_sol(sys)
tspan = (0.0,t_final)
integrator = init(u0,tspan,sys);

# Run for one step
step!(integrator)

# Plot streamfunction
wt_walls = create_windtunnel_boundaries(g,params,withinlet=false)
ψ = zeros_gridcurl(sys)
ViscousFlow.streamfunction!(ψ,integrator.sol.u[end].x[1],sys,0.0)
plot(ψ,g,title="streamfunction",xlims=xlim,ylims=ylim,xlabel="x",ylabel="y")
plot!(wt_walls)
savefig("$(case)_streamfunction.pdf")

# Plot velocity and aoa on the centerline
wt_vel = zeros_grid(sys);
freestream_func = ViscousFlow.get_freestream_func(sys.forcing)
Vinf = freestream_func(0.0,params)
wt_vn, wt_dvn = ViscousFlow.velocity!(wt_vel, zeros_gridcurl(sys), sys, 0.1);
vel_fcn = interpolatable_field(wt_vel,g);

x_centerline = -0.1*L_TS:0.01:1.1*L_TS
y_centerline = H_TS / 2 * ones(length(x_centerline))
u_centerline = vel_fcn[1].(x_centerline, y_centerline)
v_centerline = vel_fcn[2].(x_centerline, y_centerline)
aoa_centerline = 180 / π * atan.(v_centerline,u_centerline) # degrees

plot(x_centerline,u_centerline./ u_0, label="u/u0",xlabel="x");
plot!(x_centerline,v_centerline ./ u_0, label="v/u0");
savefig("$(case)_centerline_velocity.pdf")

plot(x_centerline,aoa_centerline,xlim=[0,L_TS],xlabel="x",ylabel="aoa [degrees]",legend=false);
savefig("$(case)_centerline_aoa.pdf")
