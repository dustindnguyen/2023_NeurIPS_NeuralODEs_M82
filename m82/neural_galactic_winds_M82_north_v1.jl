begin
    using Plots, LinearAlgebra, DifferentialEquations, LaTeXStrings, FileIO, JLD2, SparseArrays, Distances, Statistics, Measures, CSV, DataFrames
    using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationPolyalgorithms, Zygote, TruncatedStacktraces, Flux, Optim
    TruncatedStacktraces.VERBOSE[] = true
    workingdirectory = "/m82/"
end 


# Define constants in cgs units unless specified otherwise 
begin
    const γ     = 5.0/3.0             # Equation of State, Mono-atomic Ideal Gas
    const mu    = 0.6                 # Solar metallicity 
    const kb    = 1.380649e-16        # Boltzmann constant [erg/K]
    const kbKeV = 8.617333e-8         # Boltzmann constant [KeV/K]
    const Msun  = 1.9884754e33        # Solar mass [g]
    const yr    = 3.155692e7          # Year [s] 
    const kyr   = 3.155692e10         # Kiloyear [s] 
    const kpc   = 3.0856775e21        # Kiloparsec [cm]
    const pc    = 3.0856775e18        # Parsec [cm]
    const mp    = 1.6726219e-24       # Proton mass
    const kev_to_erg = 1.60219e-09    # Conversion of KeV to erg 

    # Even though we're using float64 and can handle cgs units (1/kpc^3), we'll still
    # convert to so-called code units in the system of ODEs. This helps the numerical solver. 

    const LENGTH_UNIT   = kpc 
    const TIME_UNIT     = kyr 
    const MASS_UNIT     = Msun 
    const VELOCITY_UNIT = LENGTH_UNIT / TIME_UNIT
    const DENSITY_UNIT  = MASS_UNIT / LENGTH_UNIT^3
    const PRESSURE_UNIT = MASS_UNIT * VELOCITY_UNIT^2 / LENGTH_UNIT^3 
    const ENERGY_UNIT   = MASS_UNIT * VELOCITY_UNIT^2 
end 

# Load in processed data 
begin 
    df = CSV.File(workingdirectory*"data/north_data.csv") |> DataFrame

    # Extract columns as arrays
    x_data = df[!, :z_N] |> collect
    T_data = df[!, :T_N] |> collect
    n_data = df[!, :n_N] |> collect
    dlogAdx_data = df[!, :dLA_N] |> collect
    K_data = T_data ./ n_data.^(2.0/3.0)
end 

# Define initial conditions, these are pre-calculated from the Python jupyter notebook 
begin 
    # These x-values are the same as the data's. 
    const x_i      = x_data[1]               # start
    const x_f      = x_data[end]             # stop
    const nx       = length(x_data)          # number steps between start and stop 
    const dx       = (x_f - x_i) / (nx-1)    # dx required  
    const xspan    = [x_i,x_f]
    const x_range  = range(start=x_i, stop=x_f, step=dx)
    const x_values = collect(x_range); 

    # This is the range and span we evaluate over for our models. 
    # This additional resolution is required to resolve cooling. 
    # fdx is the factor of model resolution relative to data resolution. 

    const fdx       = 20.0
    const dx2       = dx / fdx  
    const x_range2  = range(start=x_i,stop=x_f, step=dx2)
    const x_values2 = collect(x_range2)

    # indexes to compare the higher reslution to the data 
    indexes = 1:Int64(fdx):length(x_values2)

    println("Length of data  array: $(length(x_data))")
    println("Length of model array: $(length(x_range2))")

    # initial condition 
    # cgs 
    vInf = 1835e5   # v_Inf = 1853 km/s 

    v0 = vInf 
    n0 = n_data[1] 
    T0 = T_data[1] 
    ρ0 = n0 * mu * mp 
    P0 = n0 * kb * T0 
    u0 =  [v0,ρ0,P0]
    u0_CU = [v0/VELOCITY_UNIT,ρ0/DENSITY_UNIT,P0/PRESSURE_UNIT]
    xspan_CU = xspan / LENGTH_UNIT
    dx_CU = dx / LENGTH_UNIT 

    # Plot the resolutions for illustrative purposes. 
    # black is data resolution, blue is model resolution  
    plot(x_data,x_data ./ x_data,seriestype=:scatter,marker=:circle,markeralpha=0.5,xlims=(0.5,0.6),ylims=(0.5,1.5),c=:black,label="data")
    plot!(x_values2,x_values2 ./ x_values2,seriestype=:scatter,marker=:cross,c=:green,markeralpha=0.5,xlabel="kpc",label="model")
    plot!(x_values2[indexes], x_values2[indexes]./x_values2[indexes] .* 1.125,seriestype=:scatter,marker=:circle,c=:green,markeralpha=0.75,label="indexed model")
end 



begin
    function gravity_function(r)
        σ = 200e5       # cm/s 
        return σ^2 / r  # cm/s^2 
    end 
    plot(gravity_function.(x_range*LENGTH_UNIT)./(LENGTH_UNIT/TIME_UNIT^2))
end


begin
    function Λ_function(T_cgs)
        # This if-statement is required for some random solutions that predict negative T. 
        # I've never seen negative T in solutions that didn't fail, this must be something to do with the error estimation of DifferentialEquations.jl
        if T_cgs < 0
            Λ = 0.0 
            return Λ
        else
            logT = log10(T_cgs)
            if logT <= 4.0
            Λ = 0.0 
            elseif 4.0 < logT && logT <= 5.9
                Λ = 10^(-1.3 * (logT-5.25)^2 - 21.25)
            elseif 5.9 < logT && logT <= 7.4
                Λ = 10^(0.7 * (logT-7.1)^2 - 22.8)
            else
                Λ = 10^(0.45 * logT - 26.065)
            end
            return Λ
        end 
    end   

    T_range = range(start=1e4, stop=1e8, length=Int64(1e5))
    Λ_values = Λ_function.(T_range)  
    plot(log10.(T_range),log10.(Λ_values),xlabel=L"$ \log_{10}( T \ \mathrm{[K]})$",ylabel=L"$ \log_{10}( Λ \ \mathrm{[ergs \, cm^3 \, s^{-1}]})$")
end


begin
    function dlogAdx_interpolation(x, x_values, dlogAdx)
        idx = searchsortedlast(x_values, x)

        # If x is outside the range of x_values, return the edge values of dlogAdx
        if idx == 0
            return dlogAdx[1]
        elseif idx == length(x_values)
            return dlogAdx[end]
        end

        # Calculate the slope between the two surrounding points
        slope = (dlogAdx[idx+1] - dlogAdx[idx]) / (x_values[idx+1] - x_values[idx])

        # Return the interpolated value
        return dlogAdx[idx] + slope * (x - x_values[idx])
    end
     # Plot the surface area expansion rate for the higher resolution spatial points 
     interp_values = [dlogAdx_interpolation(x, x_data, dlogAdx_data) for x in x_range2]
    
     plot(x_data,dlogAdx_data/(1/LENGTH_UNIT),seriestype=:scatter,market=:circle,markeralpha=0.5,c=:black,xlabel=L"$x\, \mathrm{[kpc]}$",ylabel=L"$d\log A / dx \, \mathrm{[kpc^{-1}]}$")
     plot!(x_range2,interp_values/(1/LENGTH_UNIT),seriestype=:scatter,marker=:cross,markeralpha=0.5)
end

function extract_cgs_solution(solution)
    v = solution[1,:] * VELOCITY_UNIT
    ρ = solution[2,:] * DENSITY_UNIT
    P = solution[3,:] * PRESSURE_UNIT
    n = ρ / (mu*mp)
    T = P ./ (n*kb)
    c = sqrt.(γ * P ./ ρ)
    K = T ./ n.^(2/3)
    M = v ./ c 
    return v, ρ, P, n, T, c, K, M
end 

function sph_wind!(du,u,p,r)
    v, ρ, P = u

    n = ρ / (mu*(mp/MASS_UNIT))

    r_cgs = r * LENGTH_UNIT
    P_cgs = P * PRESSURE_UNIT
    n_cgs = ρ * DENSITY_UNIT / (mu*mp)
    T_cgs = P_cgs ./ (n_cgs * kb)

    # Additional physics 
    Λ       = Λ_function(T_cgs) / (ENERGY_UNIT * LENGTH_UNIT^3 / TIME_UNIT) 
    dlogAdr = 2 / r 
    G       = gravity_function(r_cgs) / (LENGTH_UNIT / TIME_UNIT^2)
    μ       = 0.0   
    
    du[1] = dvdr = ((γ*P*v/(ρ))*dlogAdr + ((γ-1.0)*Λ*n^2.0/ρ) - ((γ+1.0)*μ*v^2.0/(2.0*ρ)) - (G*v))/(v^2.0-γ*P/ρ)
    du[2] = dρdr = (μ/v) - (ρ)*(dlogAdr) - (ρ/v)*(dvdr) 

    if T_cgs <= 1e4   # Temperature floor 1e4K -> switch to isothermal wind equation of state 
        return du[3] = dPdr = (P/ρ)*dρdr  
    else          
        return du[3] = dPdr = (γ*P/ρ)*dρdr - ((γ-1.0)*n^2.0*Λ/v) + (γ-1.0)*(μ/v)*(v^2.0/2.0 - (γ/(γ-1.0))*(P/ρ))
    end 
end

function nonsph_wind!(du,u,p,r)
    v, ρ, P = u

    n = ρ / (mu*(mp/MASS_UNIT))

    r_cgs = r * LENGTH_UNIT 
    P_cgs = P * PRESSURE_UNIT
    n_cgs = ρ * DENSITY_UNIT / (mu*mp)
    T_cgs = P_cgs ./ (n_cgs * kb)

    # Additional physics 
    Λ       = Λ_function(T_cgs) / (ENERGY_UNIT * LENGTH_UNIT^3 / TIME_UNIT) 
    dlogAdr = dlogAdx_interpolation(r, x_data, dlogAdx_data) / (1/LENGTH_UNIT)
    G       = gravity_function(r_cgs) / (LENGTH_UNIT / TIME_UNIT^2)
    μ       = 0.0   
    
    du[1] = dvdr = ((γ*P*v/(ρ))*dlogAdr + ((γ-1.0)*Λ*n^2.0/ρ) - ((γ+1.0)*μ*v^2.0/(2.0*ρ)) - (G*v))/(v^2.0-γ*P/ρ)
    du[2] = dρdr = (μ/v) - (ρ)*(dlogAdr) - (ρ/v)*(dvdr) 

    if T_cgs <= 1e4   # Temperature floor 1e4K -> switch to isothermal wind equation of state 
        return du[3] = dPdr = (P/ρ)*dρdr  
    else          
        return du[3] = dPdr = (γ*P/ρ)*dρdr - ((γ-1.0)*n^2.0*Λ/v) + (γ-1.0)*(μ/v)*(v^2.0/2.0 - (γ/(γ-1.0))*(P/ρ))
    end 
end


begin
    sph_prob    = ODEProblem(sph_wind!,u0_CU,xspan)
    nonsph_prob = ODEProblem(nonsph_wind!,u0_CU,xspan)

    sph_sol = solve(sph_prob, Tsit5(), dt = dx2, saveat=dx2, abstol=1f-10, reltol=1f-10)
    nonsph_sol = solve(nonsph_prob, Tsit5(), dt = dx2, saveat=dx2, abstol=1f-10, reltol=1f-10)

    # save sph_sol for later comparison
    save_object(workingdirectory*"output/north/v1/PRED_sph.jld2",sph_sol)

    v_sph, ρ_sph, P_sph, n_sph, T_sph, c_sph, K_sph, M_sph = extract_cgs_solution(sph_sol)
    v_nonsph, ρ_nonsph, P_nonsph, n_nonsph, T_nonsph, c_nonsph, K_nonsph, M_nonsph = extract_cgs_solution(nonsph_sol)
    

    label1 = L"$\mathcal{M}$"
    label2 = L"$\log_{10}(n \ \mathrm{[cm^{-3}]})$"
    label3 = L"$\log_{10}(T \ \mathrm{[K]})$"
    label4 = L"$v \ \mathrm{[km\,s^{-1}]}$"
    label5 = L"$\log_{10}(K/k_b \ \mathrm{[K\,cm^2]}$)"

    xlabel = L"$x \ \mathrm{[kpc]}$"


    M_min = 1
    M_max = 15
    M_bounds = (M_min,M_max)

    nlog_min = -2
    nlog_max = 1
    nlog_bounds = (nlog_min,nlog_max)

    Tlog_min = 3.5
    Tlog_max = 7.5
    Tlog_bounds = (Tlog_min,Tlog_max)

    v_min = 1500
    v_max = 2000
    v_bounds =  (v_min,v_max)

    Klog_min = 3.5
    Klog_max = 7.5
    Klog_bounds = (Klog_min,Klog_max)

    color_palette = :okabe_ito
    
    data_color = :black 
    sph_color = 1 
    nonsph_color = 2 


  
    p1 = plot(x_data,log10.(n_data),ylabel=label2,xlabel=xlabel,ylim=nlog_bounds,margin=5mm,label="data",color=data_color,marker=:cross,linealpha=0.5)
    plot!(p1,x_range2,log10.(n_sph),margin=5mm,label="spherical",palette=color_palette,color=sph_color)
    plot!(p1,x_range2,log10.(n_nonsph),margin=5mm,label="non-spherical",palette=color_palette,color=nonsph_color)
    
    p2 = plot(x_data,log10.(T_data),ylabel=label3,xlabel=xlabel,ylim=Tlog_bounds,margin=5mm,label="",color=data_color,marker=:cross,linealpha=0.5)
    plot!(p2,x_range2,log10.(T_sph),margin=5mm,label="",palette=color_palette,color=sph_color)
    plot!(p2,x_range2,log10.(T_nonsph),margin=5mm,label="",palette=color_palette,color=nonsph_color)
    
    p3 = plot(x_range2,v_sph/1e5,ylabel=label4,xlabel=xlabel,ylim=v_bounds,margin=5mm,label="",palette=color_palette,color=sph_color)
    plot!(p3,x_range2,v_nonsph/1e5,margin=5mm,label="",palette=color_palette,color=nonsph_color)
    
    combined_plot = plot(p1, p2, p3, layout=(1, 3), legend=:topleft,size=(1200,300))
end


begin
    Ξ = 64
    ann = Chain(  
        Dense(1, Ξ, swish),
        SkipConnection(Dense(Ξ, Ξ, swish), +),
        SkipConnection(Dense(Ξ, Ξ, swish), +),
        SkipConnection(Dense(Ξ, Ξ, swish), +),
        Dense(Ξ, 1, relu)  
    )
    p_init, re = Flux.destructure(ann)
end

function neural_wind!(du,u,p,r)
    v, ρ, P = u

    n = ρ / (mu*(mp/MASS_UNIT))

    r_cgs = r * LENGTH_UNIT
    P_cgs = P * PRESSURE_UNIT
    n_cgs = ρ * DENSITY_UNIT / (mu*mp)
    T_cgs = P_cgs ./ (n_cgs * kb)

    # Restructure the neural network using (updated) parameters p 
    Φ = re(p)
    input = [r]

    # Additional physics 
    Λ       = Λ_function(T_cgs) / (ENERGY_UNIT * LENGTH_UNIT^3 / TIME_UNIT)  
    dlogAdr = dlogAdx_interpolation(r, x_data, dlogAdx_data) / (1 / LENGTH_UNIT)
    G       = gravity_function(r_cgs) / (LENGTH_UNIT / TIME_UNIT^2)
    μ       = Φ(input)[1] 
    
    du[1] = dvdr = ((γ*P*v/(ρ))*dlogAdr + ((γ-1.0)*Λ*n^2.0/ρ) - ((γ+1.0)*μ*v^2.0/(2.0*ρ)) - (G*v))/(v^2.0-γ*P/ρ)
    du[2] = dρdr = (μ/v) - (ρ)*(dlogAdr) - (ρ/v)*(dvdr) 

    if T_cgs <= 1e4   # Temperature floor 1e4K -> switch to isothermal wind equation of state 
        return du[3] = dPdr = (P/ρ)*dρdr  
    else          
        return du[3] = dPdr = (γ*P/ρ)*dρdr - ((γ-1.0)*n^2.0*Λ/v) + (γ-1.0)*(μ/v)*(v^2.0/2.0 - (γ/(γ-1.0))*(P/ρ))
    end 
end

neural_prob=ODEProblem(neural_wind!,u0_CU,xspan)

sol_pred = solve(neural_prob, p=p_init, Tsit5(), dt = dx2, saveat=dx2, abstol=1f-8, reltol=1f-8)

v_pred, ρ_pred, P_pred, n_pred, T_pred, c_pred, K_pred, M_pred = extract_cgs_solution(sol_pred)

begin 
    nlog_min = -2
    nlog_max = 1
    nlog_bounds = (nlog_min,nlog_max)

    Tlog_min = 3.5
    Tlog_max = 7.5
    Tlog_bounds = (Tlog_min,Tlog_max)

    v_min = 1500
    v_max = 2000
    v_bounds =  (v_min,v_max)

    color_palette = :okabe_ito
    
    data_color = :black 
    sph_color = 1 
    nonsph_color = 2 
    untrained_color = 3  

  
    p1 = plot(x_data,log10.(n_data),ylabel=label2,xlabel=xlabel,ylim=nlog_bounds,margin=5mm,label="data",color=data_color,marker=:cross,linealpha=0.5)
    plot!(p1,x_range2,log10.(n_sph),margin=5mm,label="spherical",palette=color_palette,color=sph_color)
    plot!(p1,x_range2,log10.(n_nonsph),margin=5mm,label="non-spherical",palette=color_palette,color=nonsph_color)
    plot!(p1,x_range2, log10.(n_pred),margin=5mm,label="untrained pred",palette=color_palette,color=untrained_color)
    
    p2 = plot(x_data,log10.(T_data),ylabel=label3,xlabel=xlabel,ylim=Tlog_bounds,margin=5mm,label="",color=data_color,marker=:cross,linealpha=0.5)
    plot!(p2,x_range2,log10.(T_sph),margin=5mm,label="",palette=color_palette,color=sph_color)
    plot!(p2,x_range2,log10.(T_nonsph),margin=5mm,label="",palette=color_palette,color=nonsph_color)
    plot!(p2,x_range2, log10.(T_pred),margin=5mm,label="",palette=color_palette,color=untrained_color)

    p3 = plot(x_range2,v_sph/1e5,ylabel=label4,xlabel=xlabel,ylim=v_bounds,margin=5mm,label="",palette=color_palette,color=sph_color)
    plot!(p3,x_range2,v_nonsph/1e5,margin=5mm,label="",palette=color_palette,color=nonsph_color)
    plot!(p3,x_range2,v_pred/1e5,margin=5mm,label="",palette=color_palette,color=untrained_color)

    combined_plot = plot(p1, p2, p3, layout=(1, 3), legend=:topleft,size=(1200,300))
end 

# function for L2 regularization 
function l2_regularization(p, λ)
    return λ * sum(x -> x^2, p)
end

# Prediction function 
function predict(θ)
    return solve(neural_prob,Tsit5(),p=θ,dt=dx2,saveat=dx2,abstol=1f-8,reltol=1f-8)
end

function mse_loss(θ)
    sol_pred = predict(θ)

    # We'll need all of the quantities. 
    # Mach number for penalty bounds 
    # Entropy for optimization 

    v_pred, ρ_pred, P_pred, n_pred, T_pred, c_pred, K_pred, M_pred = extract_cgs_solution(sol_pred)

    if sol_pred.retcode == ReturnCode.Success
        sol_pred = Array(sol_pred)

        l = (K_data[:] - K_pred[indexes])
        
        # Create a linear weight vector that decreases from 1 to some small positive value
        weights = LinRange(1, 0.01, length(l))  # The smallest weight is 0.01

        # Calculate mse with weights
        mse = mean(abs2.(l .* weights))
        
        # Initialize penalty term
        penalty_term = 0.0
        penalty_weight = 0.1

        # Define Mach bounds
        lower_bound = 1.0
        upper_bound = 1.5

        # Calculate the penalty term without for loop using broadcasting and element-wise operations
        penalty_mask = (lower_bound .<= M_pred) .& (M_pred .<= upper_bound)
        penalty_term = penalty_weight * sum((1.0 .- abs.(M_pred[penalty_mask] .- 1.0)).^2)

        # Make the penalty term a function of mse_total
        penalty_term *= mse

        # Add the penalty term to the loss function
        loss_with_penalty = mse + penalty_term

        return loss_with_penalty, sol_pred
    else
        return Inf, sol_pred
    end
end

# Test loss calculation on initial ANN parameters 
mse_loss(p_init)

begin
    LOSS_ADAM = []
    PRED_ADAM = []
    PARS_ADAM = []

    callback = function (θ,l,pred)
        global iters += 1 
        println("Iteration: $iters || Loss: $l")
        append!(PRED_ADAM, [pred])
        append!(LOSS_ADAM, l)
        append!(PARS_ADAM, [θ])
        false
    end 
end 

# Optimization Round 1: ADAM 
begin
    
    #adtype = Optimization.AutoZygote() 
    adtype = Optimization.AutoForwardDiff()
    optf  = Optimization.OptimizationFunction((x,p)->mse_loss(x),adtype) 
    optprob  = Optimization.OptimizationProblem(optf,p_init)
    
    n_iters = 150
    decay_rate = 0.98

    println("Optimization with Adam optimizer")
    global iters = 0; 
    p_ADAM  = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.075,(decay_rate,0.999)), callback = callback, maxiters = n_iters)
    #save_object(workingdirectory*"output/p_ADAM.jld2",PARS_ADAM)
    #save_object(workingdirectory*"output/l_ADAM.jld2",LOSS_ADAM)
    #save_object(workingdirectory*"output/sol_ADAM.jld2",PRED_ADAM)
end



function extract_optimized_solutions(predictions,string)
    if string == "start"
        v = predictions[1][1,:] * VELOCITY_UNIT
        ρ = predictions[1][2,:] * DENSITY_UNIT
        P = predictions[1][3,:] * PRESSURE_UNIT
        n = ρ ./ (mu * mp)
        T = P ./ (n * kb)
        c = sqrt.(γ * P ./ ρ)
        K = T ./ n.^(2/3)
        M = v ./ c 
    elseif string == "end"
        v = predictions[end][1,:] * VELOCITY_UNIT
        ρ = predictions[end][2,:] * DENSITY_UNIT
        P = predictions[end][3,:] * PRESSURE_UNIT
        n = ρ ./ (mu * mp)
        T = P ./ (n * kb)
        c = sqrt.(γ * P ./ ρ)
        K = T ./ n.^(2/3)
        M = v ./ c 
    end
    return v, ρ, P, n, T, c, K, M
end 

v_adam_start, ρ_adam_start, P_adam_start, n_adam_start, T_adam_start, c_adam_start, K_adam_start, M_adam_start = extract_optimized_solutions(PRED_ADAM,"start")
v_adam_end, ρ_adam_end, P_adam_end, n_adam_end, T_adam_end, c_adam_end, K_adam_end, M_adam_end = extract_optimized_solutions(PRED_ADAM,"end")

begin 
    p1 = plot(x_data,log10.(K_data),c=:black)
    plot!(x_values2,log10.(K_adam_start))
    plot!(x_values2,log10.(K_adam_end))

    p2 = plot(x_data,log10.(n_data),c=:black)
    plot!(x_values2,log10.(n_adam_start))
    plot!(x_values2,log10.(n_adam_end))

    p3 = plot(x_data,log10.(T_data),c=:black)
    plot!(x_values2,log10.(T_adam_start))
    plot!(x_values2,log10.(T_adam_end))

    p4 = plot(x_values2,v_adam_start/1e5)
    plot!(x_values2,v_adam_end/1e5)

    combined_plot = plot(p1, p2, p3, p4, layout=(1, 4), legend=:topleft,size=(1200,300))
end 


#######################
#######################
#######################
######## BFGS #########
#######################
#######################

begin
    LOSS_ADAM = load_object(workingdirectory*"/output/l_ADAM.jld2")
    PRED_ADAM = load_object(workingdirectory*"/output/sol_ADAM.jld2")
    PARS_ADAM = load_object(workingdirectory*"/output/p_ADAM.jld2")

    v_end = PRED_ADAM[end][1,:] * VELOCITY_UNIT
    ρ_end = PRED_ADAM[end][2,:] * DENSITY_UNIT
    P_end = PRED_ADAM[end][3,:] * PRESSURE_UNIT
    n_end = ρ_end ./ (mu * mp)
    T_end = P_end ./ (n_end * kb) 
    M_end = v_end ./ (γ .* P_end ./ ρ_end).^(0.5)

    v_start = PRED_ADAM[1][1,:] * VELOCITY_UNIT
    ρ_start = PRED_ADAM[1][2,:] * DENSITY_UNIT
    P_start = PRED_ADAM[1][3,:] * PRESSURE_UNIT
    n_start = ρ_start ./ (mu * mp)
    T_start = P_start ./ (n_start * kb) 
    M_start = v_start ./ (γ .* P_start ./ ρ_start).^(0.5);

    label1 = L"$\mathcal{M}$"
    label2 = L"$\log_{10}(n \ \mathrm{[cm^{-3}]})$"
    label3 = L"$\log_{10}(T \ \mathrm{[K]})$"
    label4 = L"$v \times 1000 \,\mathrm{[km\,s^{-1}]}$"

    xlabel = L"$r \ \mathrm{[kpc]}$"


    M_min = 1
    M_max = 7
    M_bounds = (M_min,M_max)

    nlog_min = -1
    nlog_max = 1.5
    nlog_bounds = (nlog_min,nlog_max)

    Tlog_min = 6
    Tlog_max = 7.5
    Tlog_bounds = (Tlog_min,Tlog_max)

    v_min = 500
    v_max = 1750
    v_bounds =  (v_min,v_max)


    p1 = plot(x_range/kpc,M_vals_truth,ylabel=label1,xlabel=xlabel,ylim=M_bounds,margin=5mm,label="truth",c=:black)
    plot!(x_range/kpc,M_start,label="untrained prediction")
    plot!(x_range/kpc,M_end,label="trained prediction")

    p2 = plot(x_range/kpc,log10.(n_vals_truth),ylabel=label2,xlabel=xlabel,ylim=nlog_bounds,margin=5mm,label="",c=:black)
    plot!(x_range/kpc,log10.(n_start))
    plot!(x_range/kpc,log10.(n_end))

    p3 = plot(x_range/kpc,log10.(T_vals_truth),ylabel=label3,xlabel=xlabel,ylim=Tlog_bounds,margin=5mm,label="",c=:black)
    plot!(x_range/kpc,log10.(T_start))
    plot!(x_range/kpc,log10.(T_end))

    p4 = plot(x_range/kpc,v_vals_truth/1e5,ylabel=label4,xlabel=xlabel,ylim=v_bounds,margin=5mm,label="",c=:black)
    plot!(x_range/kpc,v_start/1e5)
    plot!(x_range/kpc,v_end/1e5)


    combined_plot = plot(p1, p2, p3, p4, layout=(1, 4), legend=:topleft,size=(1200,300))
end 

# Optimization Round 2: BFGS

# redefine MSE loss, needs more sensitivity on low Mach numbers as we converge. 

begin
    LOSS_BFGS = []
    PRED_BFGS = []
    PARS_BFGS = []

    callback = function (θ,l,pred)
        global iters += 1 
        println("Iteration: $iters || Loss: $l")
        append!(PRED_BFGS, [pred])
        append!(LOSS_BFGS, l)
        append!(PARS_BFGS, [θ])
        false
    end 
end 



begin
    
    #adtype = Optimization.AutoZygote() 
    adtype = Optimization.AutoForwardDiff()
    optf  = Optimization.OptimizationFunction((x,p)->mse_loss(x),adtype) 
    optprob  = Optimization.OptimizationProblem(optf,Array(PARS_ADAM[end]))
    
    
    println("Optimization with BFGS")
    global iters = 0; 
    
    n_iters = 150; 
    
    p_pred_bfgs  = Optimization.solve(optprob, BFGS(initial_stepnorm=0.005), g_tol=1e-9, maxiters=n_iters, callback = callback)
    save_object(workingdirectory*"output/p_BFGS.jld2",PARS_BFGS)
    save_object(workingdirectory*"output/l_BFGS.jld2",LOSS_BFGS)
    save_object(workingdirectory*"output/sol_BFGS.jld2",PRED_BFGS)
end


v_bfgs_end, ρ_bfgs_end, P_bfgs_end, n_bfgs_end, T_bfgs_end, c_bfgs_end, K_bfgs_end, M_bfgs_end = extract_optimized_solutions(PRED_BFGS,"end")


begin 
    p1 = plot(x_data,log10.(K_data),c=:black)
    plot!(x_values2,log10.(K_adam_start))
    plot!(x_values2,log10.(K_adam_end))
    plot!(x_values2,log10.(K_bfgs_end))

    p2 = plot(x_data,log10.(n_data),c=:black)
    plot!(x_values2,log10.(n_adam_start))
    plot!(x_values2,log10.(n_adam_end))
    plot!(x_values2,log10.(n_bfgs_end))

    p3 = plot(x_data,log10.(T_data),c=:black)
    plot!(x_values2,log10.(T_adam_start))
    plot!(x_values2,log10.(T_adam_end))
    plot!(x_values2,log10.(T_bfgs_end))

    p4 = plot(x_values2,v_adam_start/1e5)
    plot!(x_values2,v_adam_end/1e5)
    plot!(x_values2,v_bfgs_end/1e5)

    combined_plot = plot(p1, p2, p3, p4, layout=(1, 4), legend=:topleft,size=(1200,300))
end 


