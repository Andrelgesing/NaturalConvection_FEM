
#%% -- Loading necessary libraries and setting options for FEniCS
import sys

from dolfin import *
from loadParam import *


import matplotlib.pyplot as plt

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True
parameters["allow_extrapolation"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"


# import your code
from myDriver import *
# Test


param = Param()
param.Prandtl = 0.015

for order in [2,3,4,5]:
    mesh_ =  generate_mesh(param)
    #plot(mesh_)
    #plt.show(block=False)

    print('- order = %d'%order)
    param.order = order
    
    V  = VectorElement("CG", mesh_.ufl_cell(), param.order)
    Q  = FiniteElement("CG", mesh_.ufl_cell(), param.order-1)
    VT = FiniteElement("CG", mesh_.ufl_cell(), param.order)
    W  = FunctionSpace(mesh_, MixedElement([V, Q,VT]))
   
    # just used for plotting
    Wt= FunctionSpace(mesh_, VT)
    q0 = Function(W)

    # apply the boundary condition of the problem: the problem at the first Rayleigh number
    # has the same boundary condition as the target Rayleigh. 
    # it is necessary because the linearized problem has only homogeneous boundary condition
    bcs = param.define_boundary_conditions(W)
    for bc in bcs:
        bc.apply(q0.vector())

    # one needs to go progessively to high Rayleigh number
    for Ra in [1e4,2e4,3e4]:
        param.Rayleigh = Ra
        print('---- Ra = %f --'%param.Rayleigh)
        solve_newton(mesh_,param,q0,W)
        
    for Ra in [35000,40000,45000,50000 ]: # you may change the numbers :)
        param.Rayleigh = Ra
        print('---- Ra = %f --'%param.Rayleigh)
        solve_newton(mesh_,param,q0,W)
        eigenvalues, egv_real_part, egv_imag_part  = solveEigenvalueProblem(mesh_,param,q0,W)
        
        plot_streamlines_and_isotemperature(param,q0,Wt,filename='baseflow.png')
        plot_streamlines_and_isotemperature(param,egv_real_part,Wt,filename='real_part.png')
        plot_streamlines_and_isotemperature(param,egv_imag_part,Wt,filename='imag_part.png')        

# plot temperature
plot(q0[3])
plt.show(block=True)

