@ilmproblem(NeumannPoisson,scalar)

struct NeumannPoissonCache{SMT,DVT,VNT,FT,ST} <: AbstractExtraILMCache
    S :: SMT
    dvn :: DVT
    vn :: VNT
    vnplus :: VNT
    vnminus :: VNT
    fstar :: FT
    sstar :: ST
end

function ImmersedLayers.prob_cache(prob::NeumannPoissonProblem,base_cache::BasicILMCache)
    S = create_CLinvCT(base_cache)
    dvn = zeros_surface(base_cache)
    vn = zeros_surface(base_cache)
    vnplus = zeros_surface(base_cache)
    vnminus = zeros_surface(base_cache)
    vcorr = zeros_gridgrad(base_cache)
    fstar = zeros_grid(base_cache)
    sstar = zeros_gridcurl(base_cache)

   NeumannPoissonCache(S,dvn,vn,vnplus,vnminus,fstar,sstar)
end

function ImmersedLayers.solve(vnplus,vnminus,sys::ILMSystem)
    @unpack extra_cache, base_cache, bc, phys_params = sys
    @unpack S, dvn, vn, fstar, sstar = extra_cache

    fill!(fstar,0.0)

    f = zeros_grid(base_cache)
    s = zeros_gridcurl(base_cache)
    df = zeros_surface(base_cache)
    ds = zeros_surface(base_cache)

    # Get the precribed jump and average of the surface normal derivatives
    dvn .= vnplus .- vnminus
    vn .= 0.5*vnplus .+ 0.5*vnminus

    # Find the potential
    regularize!(fstar,dvn,base_cache)
    inverse_laplacian!(fstar,base_cache)

    surface_grad!(df,fstar,base_cache)
    df .= vn - df
    df .= -(S\df);

    surface_divergence!(f,df,base_cache)
    inverse_laplacian!(f,base_cache)
    f .+= fstar

    # Find the streamfunction
    surface_curl!(sstar,df,base_cache)

    surface_grad_cross!(ds,fstar,base_cache)
    ds .= S\ds

    surface_curl_cross!(s,ds,base_cache)
    s .-= sstar
    s .*= -1.0

    inverse_laplacian!(s,base_cache)

    return f, df, s, ds
end
