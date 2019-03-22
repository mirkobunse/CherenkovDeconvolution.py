if VERSION < v"0.6" || VERSION >= v"0.7"
  error("This setup requires Julia 0.6")
end

# add registered packages
Pkg.add("PyCall")

# clone un-registered packages
Pkg.clone("https://github.com/mirkobunse/CherenkovDeconvolution.jl.git")
Pkg.checkout("CherenkovDeconvolution", "julia-0.6", pull=false)

using CherenkovDeconvolution # trigger precompilation
