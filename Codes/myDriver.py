from __future__ import print_function
from dolfin import *
from loadParam import *
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import scipy.sparse as sp                                                                      
import numpy as np  
import timeit
import mshr


class Walls(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary 

def refine_mesh_close_to_borders(mesh):
   
    alpha_v = [ 0.05, 0.01, 0.005 ]

    for alpha in alpha_v:
        cell_markers = MeshFunction("bool", mesh, 2)
        cell_markers.set_all(False)

        for cell in cells(mesh):
                x = cell.midpoint()[0]
                y = cell.midpoint()[1]
                # cond = near(x,0.0001, alpha) or near(x,0.9999, alpha) or near(y,0.0001, alpha) or near(y,0.9999, alpha)
                cond = (x < alpha) or (y < alpha) or (x > 0.9999-alpha) or (y > 0.9999-alpha)
                if cond :
                    cell_markers[cell] = True

        print("refining mesh: former number of vertices: ",mesh.coordinates().size)
        mesh = refine(mesh, cell_markers , redistribute = False)
        print("               number of vertices: ",mesh.coordinates().size)
        
    return mesh

def generate_mesh(param):
    N = param.N
    mesh=UnitSquareMesh(N,N)
    mesh = refine_mesh_close_to_borders(mesh)
    # store the mesh in some folder
    File("./Mesh/mesh.xml") << mesh
    return mesh

def load_mesh(param):
    mesh_ = Mesh("./Mesh/mesh.xml")
    return mesh_

def Jacobian_variational_formulation(param, 
                                     u0, p0, T0 ,
                                     uhat, phat, That,
                                     w   , q   , theta  ):
    # this is usefull to avoid compilation every time one changes the values ... 
    Prandtl         =  Constant(param.Prandtl)
    Rayleigh        =  Constant(param.Rayleigh)    
    one_over_sqrtRA =  Constant(1.0/np.sqrt(param.Rayleigh))  
    ey              = as_vector([0,1])
    dV = dx
    LNS  = inner(grad(uhat)*u0, w) * dV
    LNS += inner(grad(u0)*uhat, w) * dV
    LNS += Prandtl*one_over_sqrtRA* inner(grad(uhat),grad(w))*dV
    LNS -= Prandtl*That*inner(ey,w)*dV
    LNS -= Prandtl*phat*div(w)*dV
    LNS -= Prandtl*q*div(u0+uhat)*dV
    LNS += inner(u0,grad(That))*theta*dV
    LNS += inner(uhat,grad(T0))*theta*dV
    LNS += one_over_sqrtRA*inner(grad(That),grad(theta))*dV
    
    # raise ValueError("LNS form has not been implemented")
    return LNS
    
def NavierStokes(param, u0,p0,T0,
                        w,q,theta):  
    Prandtl         =  Constant(param.Prandtl)
    Rayleigh        =  Constant(param.Rayleigh)    
    one_over_sqrtRA =  Constant(1.0/np.sqrt(param.Rayleigh))  
    ey              =  as_vector([0,1])
    dV = dx
    NS  = inner(grad(u0)*u0, w) * dV
    NS += Prandtl*one_over_sqrtRA* inner(grad(u0),grad(w))*dV
    NS -= Prandtl*T0*inner(ey,w)*dV
    NS -= Prandtl*p0*div(w)*dV
    NS -= Prandtl*q*div(u0)*dV
    NS += inner(grad(T0),u0)*theta*dV
    NS += one_over_sqrtRA*inner(grad(T0),grad(theta))*dV
    
    # raise ValueError("NavierStokes variational formulation has not been implemented ...")
    return NS                           
                       


def solve_newton_step(mymesh,param,q0,W):    
    eigenmode_real, eigenmode_imag = None,None
    egv_real_part      = Function(W)
    egv_imag_part      = Function(W)

    param.define_boundary_conditions_LNS(W)
              
    # Define unknown and test function(s)
    (uhat, phat, That) = TrialFunctions(W)
    (w, q, theta)      = TestFunctions(W)
    
    (u0, p0 , T0) = (as_vector((q0[0], q0[1])), q0[2],q0[3])
    
    dq =  Function(W)
    #print("number of degrees of freedom: %d"%q0.vector().get_local().shape)    
    
   
    # Define variational forms
    LNS = Jacobian_variational_formulation( param, u0, p0, T0 ,uhat, phat, That, w , q, theta  )
    NS  = NavierStokes(param, u0, p0, T0, w, q, theta)
    
    # Assemble matrix, vector
    lhs  = assemble(LNS)
    rhs  = assemble(NS)
    
    # apply LNS boundary conditions (see loadParam.py)...
    bcs = param.bc
    # bc.apply ...
    
    # solve the linear system of equations
    # solve() ...
    solve(LNS == NS, w, bcs, solver_parameters={
        "newton_solver": {"relaxation_parameter": 1e-0, "maximum_iterations": 1000, "linear_solver": 'mumps'}})
    # increment the solution vector with a potential relaxation factor     
    # q0.vector()[:] =  q0.vector().get_local() + ... 
    
    # raise ValueError("solve_newton_step not implemented")
                 
    return q0,dq
    


def solve_newton(mymesh,param,q0,W):
    solve_newton_step(mymesh, param, q0, W)
    # epsilon_N = 2*param.tolerance
    # i= 0
    # while epsilon_N > param.tolerance:
    #     # - newton step
    #
    #     # evaluation of the residual epsilon_N = L2 norm of dq ...
    #
    #     raise ValueError("solve_newton not implemented")
    #     print("      step %d\t,eps = %g"%(i,epsilon_N))
    #     i+=1
        
        
def solveEigenvalueProblem(mymesh,param,q0,W):
    """
       In this function 
       -> matrices are assembled
       -> eigenvalue problem is solved using scipy's eigs
       
       -> you have to create 
         1- the jacobian matrix
         2- the mass matrix
          
         the rest has already been done C-c C-v from codes seen in the course.
    """
    # Define unknown and test function(s)
    (uhat, phat, That) = TrialFunctions(W) # perturbation
    (w, q, theta)      = TestFunctions(W)
    (u0, p0 , T0) = (as_vector((q0[0], q0[1])), q0[2],q0[3])
                 
    # Define variational forms
    LNS =  Jacobian_variational_formulation( param, u0, p0, T0 ,uhat, phat, That, w , q, theta  )
    # m   =  ...
    
    raise ValueError("Variational forms form the eigenvalue problem are not implemented yet")
    
    #
    # ---  don't touch the rest of that function  --- 
    #
    
    A, M = PETScMatrix(),  PETScMatrix()

    # assemble the matrices
    assemble(-LNS, tensor=A)
    assemble(m, tensor=M)
    
    
    # applying the dirichlet boundary conditions on all the tensors
    for bc_ in param.bc:
        bc_.apply(A)
        bc_.apply(M)
    
    # This just converts PETSc to CSR
    A    = sp.csr_matrix(A.mat().getValuesCSR()[::-1])
    M    = sp.csr_matrix(M.mat().getValuesCSR()[::-1])    
    
    # get indices of the dirichlet boundary conditions
    bcinds = []
    for bc_ in param.bc:
        bcdict = bc_.get_boundary_values()
        bcinds.extend(bcdict.keys())
    # Create shift matrix  -> take care of the dirichlet boundary conditions
    # otherwise one gets ficticious eigenvalues equal to 1 ...
    shift = -1.2345e2*np.ones(len(bcinds))
    S = sp.csr_matrix((shift, (bcinds, bcinds)), shape=A.shape)
    
    num_eig =80

    print("    Solving EVP ")
    #-------- Eigenvalue Solver ------------------------------------------------------
    tic = timeit.default_timer()
    v, V = eigs(A+S, num_eig, M, sigma=0.5, ncv=400)
    toc = timeit.default_timer()
    mins = int((toc-tic)/60)
    print ("    done in %3d min %4.2f s" % (mins, (toc-tic) -mins*60))
    
    # --------------------------------------------------------------------------------    
    # output of the eigenvalue solver not necessarily ordered.
    idxs = np.argsort(-np.real(v)) 
    eigenvalues = v[idxs]
    print(eigenvalues[:10])
    # defining the output of the function
    eigenmode_real, eigenmode_imag = None,None
    egv_real_part      = Function(W)
    egv_imag_part      = Function(W)
    
    # storing the most dangerous mode (highest real part)
    egv_real_part.vector().set_local( np.real(V[:,idxs[0]])  )
    if np.iscomplex(eigenvalues[0]):
        egv_imag_part.vector().set_local( np.imag(V[:,idxs[1]])  )
    
    # saving the values in files
    (ur,pr,thetar) = egv_real_part.split(deepcopy=True)
    (ui,pi,thetai) = egv_imag_part.split(deepcopy=True)
    
    folder = 'Pr_%3.1f/Ra_%f/'%(param.Prandtl,param.Rayleigh)
    File(folder+"realPart_Vel.pvd") <<  ur   
    File(folder+"imagPart_Vel.pvd") <<  ui   
    File(folder+"realPart_theta.pvd") <<  thetar  
    File(folder+"imagPart_theta.pvd") <<  thetai
    
    np.save(folder+"eigenvalues",eigenvalues)
    np.save(folder+"realPart",egv_real_part.vector().get_local())
    np.save(folder+"imagPart",egv_imag_part.vector().get_local())
    
    
    return eigenvalues, egv_real_part, egv_imag_part




def plot_streamlines_and_isotemperature(param,q0,Wt,filename='defaultname.png'):
    # feel free to change the plotting function :)

    folder = 'Pr_%3.1f/Ra_%f/'%(param.Prandtl,param.Rayleigh)
    
    uv,p,t = q0.split(deepcopy=True)
    psi  = TrialFunction(Wt)
    psiv = TestFunction(Wt)
    
    a = inner(grad(psi),grad(psiv))*dx 
    L = (uv[0].dx(1) - uv[1].dx(0))*psiv*dx 
    
    bc =  DirichletBC(Wt, Constant(0.0), Walls())
    
    LHS = assemble(a)
    RHS = assemble(L)
    
    bc.apply(LHS)
    bc.apply(RHS)
    
    psi = Function(Wt) 
    solve(LHS,psi.vector(),RHS)
    
    Nplot = 200
    X = np.linspace(0,1,Nplot)
    Y = np.linspace(0,1,Nplot)
    T   = np.zeros((Nplot,Nplot))
    PSI = np.zeros((Nplot,Nplot))
    # interpolating the function at specific points
    for ix,x in enumerate(X):
        for iy,y in enumerate(Y):
            T[iy,ix] = t(x,y)
            PSI[iy,ix] = psi(x,y)
            
    
    fig,ax = plt.subplots(figsize=(6,6))
    ax.contour(X,Y,PSI,12,colors='k',linewidths=1)
    ax.contourf(X,Y,T,20)
    plt.savefig(folder+filename,dpi=100,bb_inches='tight')       
    
    plt.close()
    
    
    
    
    
