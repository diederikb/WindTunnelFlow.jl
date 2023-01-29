using DelimitedFiles
using WindTunnelFlow
using JSON
using Statistics
using Interpolations
using Contour

ENV["GKSwstype"]="nul"

"""
Create an array of the columns in the matrix `M`.
"""
columns(M) = [view(M, :, i) for i in 1:size(M, 2)]

"""
Returns the file path of the first file in `dir` that has `filename_part` in its name.
"""
function find_file(dir,filename_part)
    files = readdir(dir)
    idx = findall(f -> occursin(filename_part,f) && !occursin(".swp",f),files)
    file = files[idx[1]]
    file = joinpath(dir,file)
    return file
end

"""
Returns an array of the indices that appear after `id_prefix` in the filenames in `dir` that contain the strings in `patterns`.
"""
function id_list_with_patterns(dir,id_prefix,patterns::AbstractVector)
    files = readdir(dir)
    file_ids_with_patterns = findall(f -> all(occursin.(patterns,f)),files)
    match_string = "(?<=$(id_prefix))(\\d+)"
    my_matches = match.(Regex(match_string), files[file_ids_with_pattern])
    return [parse(Int,my_match.match) for my_match in my_matches]
end

"""
Reads and returns the data in the file in `dir` whose filename contains the strings in `patterns`.
"""
function read_file_with_patterns(dir,patterns::AbstractVector)
    files = readdir(dir)
    file_ids_with_patterns = findall(f -> all(occursin.(patterns,f)),files)
    f = files[file_ids_with_patterns[1]]
    f = joinpath(dir,f)
    data = readdlm(f, Float64)
    return data
end

"""
Writes the coordinates of the contour lines of `f`(`x`,`y`) to `filename` for the values in `lvls` limited to `xlim` and `ylim`. If `closed` is true, the first coordinate will be added again at the end of the file.
"""
function write_contours(filename,x,y,f,lvls,xlim,ylim; closed=false, delimiter=' ', coordinatestep=4)
    open(filename, "w") do io
        writedlm(io,["x" "y"],delimiter)
        for cl in levels(contours(x, y, f, lvls))
            lvl = level(cl) # the z-value of this contour level
            for line in lines(cl)
                x_array = Float64[]
                y_array = Float64[]
                xs, ys = Contour.coordinates(line) # coordinates of this line segment
                for i in 1:coordinatestep:length(xs)
                    if xlim[1] <= xs[i] && xs[i] <= xlim[2] && ylim[1] <= ys[i] && ys[i] <= ylim[2]
                        push!(x_array, xs[i])
                        push!(y_array, ys[i])
                    end
                end
                if length(x_array) > 0
                    if closed
                        push!(x_array, x_array[1])
                        push!(y_array, y_array[1])
                    end
                    writedlm(io,zip(x_array,y_array),delimiter)
                    print(io,"\n")
                end
            end
        end
    end
end

"""
Writes the coordinates of the `body` to `filename`.
"""
function write_body(filename::String, body::Body; delimiter=' ')
    open(filename, "w") do io
        writedlm(io,["x" "y"],delimiter)
        writedlm(io, zip(body.x,body.y), delimiter)
    end
end

"""
Writes the time `t` and force `f` history to `filename`.
"""
function write_force(filename::String, t, f; delimiter=' ')
    open(filename, "w") do io
        writedlm(io,["t" "f"],delimiter)
        writedlm(io, zip(t, f), delimiter)
    end
end

"""
Computes the Q criterion in place from the velocity gradient `vel_grad`.
"""
function q_criterion!(Q::Nodes{Dual},Ssq::Nodes{Primal},vel_grad)
    Q .= 0.5 .* (vel_grad.dudy .+ vel_grad.dvdx) .^ 2
    grid_interpolate!(Ssq,Q)
    Ssq .+= vel_grad.dudx .^2 .+ vel_grad.dvdy .^2
    Ssq .= .- Ssq # -|S|²
    grid_interpolate!(Q,Ssq)
    Q .+= 0.5 .* (vel_grad.dudy .- vel_grad.dvdx) .^ 2 # |Ω|² - |S|²
    Q .*= 0.5
    Ssq .*= -1
    return Q
end

"""
Wagner function
"""
Φ(t) = t ≥ 0 ? 1.0 - 0.165*exp(-0.091*t) - 0.335*exp(-0.6*t) : 0.0

"""
Returns the result of the Duhamel integral.
"""
function duhamelintegral(df_array,t_array,ind_fun)
    s = 0.0
    for i in 1:length(t_array)
        s += df_array[i]*ind_fun(t_array[end]-t_array[i])
    end
    return s
end

"""
Computes the vertical force response for a flat plate at a small angle of attack `α`, moving at a velocity (`U`,`V`), and with its wake along the x-axis of the body reference frame.
α in degrees.

Some of the differencing might be inconsistent with each other.
"""
function wagner_lift_response(t::AbstractVector, U, V, α; c=1, steadystart=false)
    length(U) == 1 && (U = U * ones(length(t)))
    length(V) == 1 && (V = V * ones(length(t)))

    Γb = π .* c .* (V .+ U .* sin.(α*π/180))
    dΓb = diff(Γb)
    Umidpoint = 0.5 * (U[1:end-1] + U[2:end])
    tconv = zeros(length(t))
    tconv = [sum(diff(t)[1:i] .* Umidpoint[1:i]) for i in 1:length(t) - 1]

    dUdt = backward_difference(t,U)
    dVdt = backward_difference(t,V)

    added_mass = -π/4 .* c^2 .* (dVdt .* cos(α*π/180)^2 .+ dUdt .* sin.(α*π/180) .* cos.(α*π/180))

    circulatory_lift = zeros(length(t))
    circulatory_lift[2:end] = .-U[2:end] .* [duhamelintegral(dΓb[1:i], tconv[2:i], Φ) for i in 1:length(t)-1]
    if(!steadystart)
        circulatory_lift .-= U .* Γb[0] .* Φ.(t)
    end
    lift = added_mass .+ circulatory_lift

    return lift, added_mass, circulatory_lift
end

function central_difference(t,f)
    dfdt = zeros(length(t))
    dfdt[1] = (f[2] - f[1]) / (t[2] - t[1])
    dfdt[end] = (f[end] - f[end-1]) / (t[end] - t[end-1])
    dfdt[2:end-1] = (f[3:end] .- f[1:end-2]) ./ (t[3:end] .- t[1:end-2])
    return dfdt
end

function backward_difference(t,f)
    dfdt = zeros(length(t))
    dfdt[1] = (f[2] - f[1]) / (t[2] - t[1])
    dfdt[2:end] = (f[2:end] .- f[1:end-1]) ./ (t[2:end] .- t[1:end-1])
    return dfdt
end


#files = readdir()
#json_file = files[findall(f->occursin(r".*\.json",f),files)[1]]
#Q_and_V_probe_file = files[findall(f->occursin(r".*Q_and_V_probe\.txt",f),files)[1]]
#force_wind_tunnel_file = files[findall(f->occursin(r".*force_wind_tunnel\.txt",f),files)[1]]
#
#inputs = JSON.parsefile(json_file)
#t_suction = inputs["t_suction"]
#σ_suction = inputs["sigma_suction"]
#
#force_wind_tunnel = readdlm(force_wind_tunnel_file, '\t', Float64, '\n');
#Q_and_V_probe = readdlm(Q_and_V_probe_file, '\t', Float64, '\n');
#t = force_wind_tunnel[:,1];
#C_D = force_wind_tunnel[:,2];
#C_L = force_wind_tunnel[:,3];
#Q_Q_in = Q_and_V_probe[:,2];
#
## Create the wind tunnel problem and sys
#include("AFUWT_create_sys.jl")
#wt_walls = create_windtunnel_boundaries(g,params,withinlet=true);
#
## Find frames
#t_suction_start = t_suction - 3 * σ_suction
#t_suction_end = t_suction + 3 * σ_suction
#suction_idxs = findall(t->t_suction_start <= t <= t_suction_end,t)
#C_L_max_idx = findmax(C_L[suction_idxs])[2] + suction_idxs[1] - 1
#C_L_min_idx = findmin(C_L[suction_idxs])[2] + suction_idxs[1] - 1
#
#frames_times = [
#    t_suction_start,
#    t[C_L_max_idx],
#    t_suction,
#    t[C_L_min_idx],
#    t_suction_end,
#    t_suction_end + 1 * V_in_star / c_star]
#frames_idx = [findmin(abs.(frames_times[i] .- t))[2] for i in 1:length(frames_times)];
#
## Create force figure
#plot(t,force_wind_tunnel[:,2],label="C_D")
#plot!(t,force_wind_tunnel[:,3],label="C_L")
#plot!(t,Q_and_V_probe[:,2],label="Q/Q_in")
## vline!(t[frames_idx],c=:black,ls=:dash)
#scatter!(t[frames_idx],force_wind_tunnel[frames_idx,3],c=:black,ls=:dash)
#plot!(xlabel="U*t/c",xlim=[9,12])
#savefig("$(case)_forces.png")

## Create 6-snapshot figure
#fig_xlim = [-0.75,0.75]
#fig_ylim = [-0.75,0.75]
#y_probe_ψ = range(ylim[1],ylim[2],50)
#x_probe_ψ = zeros(length(y_probe_ψ)) .+ 0.0
#
#plot_list = []
#
#v = zeros_grid(sys)
#w = zeros_gridcurl(sys)
#ψ = zeros_gridcurl(sys)
#Q = Nodes(Dual,size(g))
#nodes_primal_tmp = Nodes(Primal,size(g))
#∇v = EdgeGradient(Primal,Dual,size(g));
#
#read_vorticity!(w,frames_idx[1]);
#ViscousFlow.velocity!(v, w, sys, frames_idx[1]);
#grad!(∇v,v)
#∇v ./= cellsize(sys)
#q_criterion!(Q,∇v,sys,nodes_primal_tmp);
#
#n_parallel_points = 10
#offset = 0.15
#airfoil_pts = length(airfoil)
#x_probe_Q = range(airfoil.x[1] + offset * sin(α*pi/180),airfoil.x[end] + offset * sin(α*pi/180),n_parallel_points)
#y_probe_Q = range(airfoil.y[1] + offset * cos(α*pi/180),airfoil.y[end] + offset * cos(α*pi/180),n_parallel_points)
#
#Q_fcn = interpolatable_field(Q,g)
#Q_probe = Q_fcn.(x_probe_Q,y_probe_Q)
#Q_mean = mean(Q_probe)
#Q_std = std(Q_probe)
#Q_levels = range(Q_mean,Q_mean+Q_std,10)
#
#for i in frames_idx
#    read_vorticity!(w,i);
#    t_i = read_timestamp(i);
#    ViscousFlow.streamfunction!(ψ,w,sys,t_i);
#
#    ψ_fcn = interpolatable_field(ψ,g)
#    ψ_probe = ψ_fcn.(x_probe_ψ,y_probe_ψ)
#
#    ViscousFlow.velocity!(v, w, sys, t_i);
#    grad!(∇v,v)
#    ∇v ./= cellsize(sys)
#    q_criterion!(Q,∇v,sys,nodes_primal_tmp);
#
#    p = plot(ψ,g,c=:gray,xlim=fig_xlim,ylim=fig_ylim,levels=ψ_probe,title="time = $(t[frames_idx])")
#    plot!(Q,g,xlim=[-0.75,0.75],ylim=[-0.75,0.75],levels=Q_levels,c=:black)
#    plot!(airfoil,fc=:white,lc=:black)
#    plot!(wt_walls,c=:black)
#
#    push!(plot_list,p)
#end
#
#plot(plot_list..., layout = (2, 3), size=(800,600))
#savefig("$(case)_snapshots.pdf")
