from __future__ import print_function
import matplotlib.pyplot as plt
from dolfin import *
import mshr
import numpy as np

"""
   In this file are defined the case parameters, geometry and boundary conditions
   at the end, the parameters of the solver: physical properties, polynomial order ...

"""
class Walls(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary

class Param():
    def __init__(self):
        # physical parameters
        #self.Rayleigh = 2.108e6
        self.Rayleigh = 2.0e6
        self.Prandtl  = 0.71

        #self.Rayleigh = 45000
        #self.Prandtl  = 0.015
        

        # number of triangles per side, before refinement
        self.N = 20
        self.order= 3

        # --- parameters for the Newton method ---
        # relaxation parameter
        self.alpha      =  1       # =1 no relaxation, between 0 and 1: relaxation
        self.tolerance  =  1e-6    # absolute tolerance

        # initial value that will be replaced in the function below
        self.bc = None


    def define_boundary_conditions_LNS(self,W):

        # for the velocity
        self.bc0 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), Walls())

        # this is the temperature
        self.bc1 = DirichletBC(W.sub(2), Constant(0.0), Walls())

        self.bc = [self.bc0, self.bc1]

        # raise ValueError('LNS boundary conditions not implemented')

    def define_boundary_conditions(self,W):
        # for the velocity
        bc0 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), Walls())
        # this is the temperature
        bc1 = DirichletBC(W.sub(2),  Expression('x[0]-0.5', degree=1), Walls())
        bc = [bc0, bc1]
        # raise ValueError('NS boundary conditions not implemented')
        return bc
