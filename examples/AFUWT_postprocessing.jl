using Plots
using DelimitedFiles
using WindTunnelFlow
using JSON
using Statistics

ENV["GKSwstype"]="nul"

function read_vorticity_relative_index!(w,i)
    files = readdir()
    f_idx = findall(f->occursin(r"snapshot.*vorticity_wind_tunnel",f),files)
    println("$i out of $(length(f_idx))")
    if i <= length(f_idx)
        f = files[f_idx[i]]
        w_data = readdlm(f, Float64)
        println(size(w_data))
        w.data .= w_data
    else
        println("i is out of bounds")
    end
    return w
end

function read_vorticity!(w,i)
    files = readdir()
    f_idx = findall(f->occursin(r"snapshot_$(i).*vorticity_wind_tunnel",f),files)
    if length(f_idx) == 1
        f = files[f_idx[1]]
        w_data = readdlm(f, Float64)
        println(size(w_data))
        w.data .= w_data
    else
        println("i is out of bounds")
    end
    return w
end

function read_timestamp(i)
    files = readdir()
    f_idx = findall(f->occursin(r"snapshot_$(i).*time_wind_tunnel",f),files)
    snapshots_time = Float64[]
    f = files[f_idx[1]]
    t = readdlm(f, Float64)[1]
    return t
end

function q_criterion!(Q::Nodes{Dual},vel_grad,sys,nodes_primal_tmp::Nodes{Primal})
    Q .= 0.5 .* (vel_grad.dudy .+ vel_grad.dvdx) .^ 2
    grid_interpolate!(nodes_primal_tmp,Q)
    nodes_primal_tmp .+= vel_grad.dudx .+ vel_grad.dvdy
    nodes_primal_tmp .= .- abs.(nodes_primal_tmp) # -|S|²
    grid_interpolate!(Q,nodes_primal_tmp)
    Q .+= abs.(0.5 .* (vel_grad.dudy .- vel_grad.dvdx) .^ 2) # |Ω|² - |S|²
    Q .*= 0.5
    return Q
end

files = readdir()
json_file = files[findall(f->occursin(r".*\.json",f),files)[1]]
Q_and_V_probe_file = files[findall(f->occursin(r".*Q_and_V_probe\.txt",f),files)[1]]
force_wind_tunnel_file = files[findall(f->occursin(r".*force_wind_tunnel\.txt",f),files)[1]]

inputs = JSON.parsefile(json_file)
t_suction = inputs["t_suction"]
σ_suction = inputs["sigma_suction"]

force_wind_tunnel = readdlm(force_wind_tunnel_file, '\t', Float64, '\n');
Q_and_V_probe = readdlm(Q_and_V_probe_file, '\t', Float64, '\n');
t = force_wind_tunnel[:,1];
C_D = force_wind_tunnel[:,2];
C_L = force_wind_tunnel[:,3];
Q_Q_in = Q_and_V_probe[:,2];

# Create the wind tunnel problem and sys
include("AFUWT_create_sys.jl")
wt_walls = create_windtunnel_boundaries(g,params,withinlet=true);

# Find frames
t_suction_start = t_suction - 3 * σ_suction
t_suction_end = t_suction + 3 * σ_suction
suction_idxs = findall(t->t_suction_start <= t <= t_suction_end,t)
C_L_max_idx = findmax(C_L[suction_idxs])[2] + suction_idxs[1] - 1
C_L_min_idx = findmin(C_L[suction_idxs])[2] + suction_idxs[1] - 1

frames_times = [
    t_suction_start,
    t[C_L_max_idx],
    t_suction,
    t[C_L_min_idx],
    t_suction_end,
    t_suction_end + 1 * V_in_star / c_star]
frames_idx = [findmin(abs.(frames_times[i] .- t))[2] for i in 1:length(frames_times)];

# Create force figure
plot(t,force_wind_tunnel[:,2],label="C_D")
plot!(t,force_wind_tunnel[:,3],label="C_L")
plot!(t,Q_and_V_probe[:,2],label="Q/Q_in")
# vline!(t[frames_idx],c=:black,ls=:dash)
scatter!(t[frames_idx],force_wind_tunnel[frames_idx,3],c=:black,ls=:dash)
plot!(xlabel="U*t/c",xlim=[9,12])
savefig("$(case)_forces.png")

# Create 6-snapshot figure
fig_xlim = [-0.75,0.75]
fig_ylim = [-0.75,0.75]
y_probe_ψ = range(ylim[1],ylim[2],50)
x_probe_ψ = zeros(length(y_probe_ψ)) .+ 0.0

plot_list = []

v = zeros_grid(sys)
w = zeros_gridcurl(sys)
ψ = zeros_gridcurl(sys)
Q = Nodes(Dual,size(g))
nodes_primal_tmp = Nodes(Primal,size(g))
∇v = EdgeGradient(Primal,Dual,size(g));

read_vorticity!(w,frames_idx[1]);
ViscousFlow.velocity!(v, w, sys, frames_idx[1]);
grad!(∇v,v)
∇v ./= cellsize(sys)
q_criterion!(Q,∇v,sys,nodes_primal_tmp);

n_parallel_points = 10
offset = 0.15
airfoil_pts = length(airfoil)
x_probe_Q = range(airfoil.x[1] + offset * sin(α*pi/180),airfoil.x[end] + offset * sin(α*pi/180),n_parallel_points)
y_probe_Q = range(airfoil.y[1] + offset * cos(α*pi/180),airfoil.y[end] + offset * cos(α*pi/180),n_parallel_points)

Q_fcn = interpolatable_field(Q,g)
Q_probe = Q_fcn.(x_probe_Q,y_probe_Q)
Q_mean = mean(Q_probe)
Q_std = std(Q_probe)
Q_levels = range(Q_mean,Q_mean+Q_std,10)

for i in frames_idx
    read_vorticity!(w,i);
    t_i = read_timestamp(i);
    ViscousFlow.streamfunction!(ψ,w,sys,t_i);

    ψ_fcn = interpolatable_field(ψ,g)
    ψ_probe = ψ_fcn.(x_probe_ψ,y_probe_ψ)

    ViscousFlow.velocity!(v, w, sys, t_i);
    grad!(∇v,v)
    ∇v ./= cellsize(sys)
    q_criterion!(Q,∇v,sys,nodes_primal_tmp);

    p = plot(ψ,g,c=:gray,xlim=fig_xlim,ylim=fig_ylim,levels=ψ_probe,title="time = $(t[frames_idx])")
    plot!(Q,g,xlim=[-0.75,0.75],ylim=[-0.75,0.75],levels=Q_levels,c=:black)
    plot!(airfoil,fc=:white,lc=:black)
    plot!(wt_walls,c=:black)

    push!(plot_list,p)
end

plot(plot_list..., layout = (2, 3), size=(800,600))
savefig("$(case)_snapshots.pdf")
