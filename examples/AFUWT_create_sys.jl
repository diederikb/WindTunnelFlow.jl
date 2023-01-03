function gaussian_suction_velocity!(vel,pts,t,phys_params)
    V_out = phys_params["V_SD"]
    σ = phys_params["sigma_suction"]
    t_0 = phys_params["t_suction"]
    g = Gaussian(σ,sqrt(π*σ^2)) >> t_0
    vel .= -V_out*g(t)
end

function step_suction_velocity!(vel,pts,t,phys_params)
    V_out = phys_params["V_SD"]
    t_open = phys_params["t_open"]
    t_close = phys_params["t_close"]
    tau_open = phys_params["tau_open"]
    tau_close = phys_params["tau_close"]
    g(t,t_open,t_close,tau_open,tau_close) = (t_open <= t < t_close ? (1 - exp(-(t-t_open)/tau_open)) : 0.0) + (t_close <= t ? (1 - exp(-(t_close-t_open)/tau_open)) * exp(-(t-t_close)/tau_close) : 0.0)

    vel .= -V_out*g(t,t_open,t_close,tau_open,tau_close)
end

function inflow_velocity!(vel,pts,t,phys_params)
    V_in = phys_params["V_in"]
    vel .= V_in
end

case = inputs["case"]
airfoil_name = inputs["airfoil"]
gust_type = inputs["gust_type"]
c = inputs["c"] # m
α = inputs["alpha"] # degrees
Re = inputs["Re"]
t_final = inputs["t_final"]
H_TS = inputs["H_TS"] # m
W_TS = inputs["W_TS"] # m
L_TS = inputs["L_TS"] # m
x_O_over_L_TS = inputs["x_O_over_L_TS"]
y_O_over_H_TS = inputs["y_O_over_H_TS"]
Q_SD_over_Q_in = inputs["Q_SD_over_Q_in"]
x_SD_lo_over_L_TS = inputs["x_SD_lo_over_L_TS"]
x_SD_hi_over_L_TS = inputs["x_SD_hi_over_L_TS"]
haskey(inputs,"t_open") && (t_open = inputs["t_open"])
haskey(inputs,"t_close") && (t_close = inputs["t_close"])
haskey(inputs,"tau_open") && (tau_open = inputs["tau_open"])
haskey(inputs,"tau_close") && (tau_close = inputs["tau_close"])
haskey(inputs,"sigma_suction") && (sigma_suction = inputs["sigma_suction"])
haskey(inputs,"t_suction") && (t_suction = inputs["t_suction"])
grid_Re = get(inputs,"grid_Re",2.0)

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
x_O_WT_star = -x_O_over_L_TS * L_TS_star # x-coordinate of the wind tunnel frame origin using the center of the body as the origin
y_O_WT_star = -y_O_over_H_TS * H_TS_star # y-coordinate of the wind tunnel frame origin using the center of the body as the origin

params = Dict()
params["Re"] = Re
params["grid Re"] = grid_Re
params["wind tunnel length"] = L_TS_star
params["wind tunnel height"] = H_TS_star
params["wind tunnel center"] = (L_TS_star / 2 + x_O_WT_star, H_TS_star / 2 + y_O_WT_star)
params["freestream speed"] = V_in_star
params["freestream angle"] = 0.0
haskey(inputs,"t_open") && (params["t_open"] = t_open)
haskey(inputs,"t_close") && (params["t_close"] = t_close)
haskey(inputs,"tau_open") && (params["tau_open"] = tau_open)
haskey(inputs,"tau_close") && (params["tau_close"] = tau_close)
haskey(inputs,"sigma_suction") && (params["sigma_suction"] = sigma_suction)
haskey(inputs,"t_suction") && (params["t_suction"] = t_suction)
params["V_in"] = V_in_star
params["V_SD"] = V_SD_star
xlim = (-0.05 * L_TS_star + x_O_WT_star, 1.05 * L_TS_star + x_O_WT_star)
ylim = (-0.05 * H_TS_star + y_O_WT_star, 1.05 * H_TS_star + y_O_WT_star)
files = readdir()
gridfile_idx = findall(f->occursin(r".*grid\.txt",f),files)
if length(gridfile_idx) > 0
    grid_info = readdlm(files[gridfile_idx[end]]) # should do this as a json file
    g = PhysicalGrid(
        (grid_info[1,1],grid_info[2,1]),
        (grid_info[3,1],grid_info[4,1]),
        grid_info[5,1],
        ((grid_info[6,1],grid_info[6,2]),(grid_info[7,1],grid_info[7,2])),
        1)
else
    g = setup_grid(xlim, ylim, params)
end

println(g)
flush(stdout)

# Airfoil in the test section
Δs = surface_point_spacing(g,params)
if occursin("naca0009",lowercase(airfoil_name))
    airfoil = NACA4(0.0, 0.0, 0.09, Δs, len=c_star)
    println("creating NACA airfoil")
else
    if !occursin("flat_plate",lowercase(airfoil_name))
        println("airfoil not recognized, using flat plate")
    else
        println("creating flat plate airfoil")
    end
    airfoil = Plate(c_star,Δs)
end

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
if occursin("step_opening_closing",lowercase(gust_type))
    suction_velocity! = step_suction_velocity!
    println("creating step gust")
else
    if !occursin("gaussian_suction",lowercase(gust_type))
        println("gust not recognized, using gaussian_suction")
    else
        println("creating gaussian gust")
    end
    suction_velocity! = gaussian_suction_velocity!
end
suction = UniformFlowThrough(suction_boundary,suction_velocity!,1)

params["outlets"] = [suction]

# Create the wind tunnel problem
print("Creating WindTunnelProblem... ")
flush(stdout)
prob = WindTunnelProblem(
    g,
    airfoil,
    phys_params=params;
    timestep_func=ViscousFlow.DEFAULT_TIMESTEP_FUNC,
    bc=ViscousFlow.get_bc_func(nothing))
sys = construct_system(prob);
print("done\n")
flush(stdout)
