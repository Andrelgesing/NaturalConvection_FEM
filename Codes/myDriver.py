from __future__ import print_function
from dolfin import *
from loadParam import *
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import timeit
import os
import mshr


class Walls(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary

def refine_mesh_close_to_borders(mesh):

    alpha_v = [ 0.05, 0.01, 0.005 ]

    for alpha in alpha_v:
        cell_markers = MeshFunction("bool", mesh, dim=2)
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
    ey              =  as_vector((0,1))
    LNS  = inner(grad(uhat)*u0, w) * dx
    LNS += inner(grad(u0)*uhat, w) * dx
    LNS += Prandtl*one_over_sqrtRA* inner(grad(uhat),grad(w))*dx
    LNS -= Prandtl*That*inner(ey,w)*dx
    LNS -= Prandtl*phat*div(w)*dx
    LNS -= Prandtl*q*div(uhat)*dx
    LNS += inner(u0,grad(That))*theta*dx
    LNS += inner(uhat,grad(T0))*theta*dx
    LNS += one_over_sqrtRA*inner(grad(That),grad(theta))*dx

    # raise ValueError("LNS form has not been implemented")
    return LNS

def NavierStokes(param, u0,p0,T0,
                        w,q,theta):
    Prandtl         =  Constant(param.Prandtl)
    Rayleigh        =  Constant(param.Rayleigh)
    one_over_sqrtRA =  Constant(1.0/np.sqrt(param.Rayleigh))
    ey              =  as_vector((0,1))
    NS  = inner(grad(u0)*u0, w)*dx
    NS += Prandtl*one_over_sqrtRA* inner(grad(u0),grad(w))*dx
    NS -= Prandtl*T0*inner(ey,w)*dx
    NS -= Prandtl*p0*div(w)*dx
    NS -= Prandtl*q*div(u0)*dx
    NS += inner(u0, grad(T0))*theta*dx
    NS += one_over_sqrtRA*inner(grad(T0),grad(theta))*dx

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
    # print("number of degrees of freedom: %d"%q0.vector().get_local().shape)


    # Define variational forms
    LNS = Jacobian_variational_formulation( param, u0, p0, T0 ,uhat, phat, That, w , q, theta  )
    NS  = NavierStokes(param, u0, p0, T0, w, q, theta)

    # Assemble matrix, vector
    lhs = assemble(LNS)
    rhs = assemble(-NS)

    # apply LNS boundary conditions (see loadParam.py)...
    # bcs = param.define_boundary_conditions(W)
    # [bc.apply(rhs) for bc in bcs]
    # [bc.apply(lhs) for bc in param.bc]
    [bc.apply(lhs, rhs) for bc in param.bc]
    [bc.apply(dq.vector()) for bc in param.bc]

    # solve the linear system of equations
    solve(lhs, dq.vector(), rhs)
    
    # increment the solution vector with a potential relaxation factor
    q0.vector()[:] = q0.vector().get_local() + param.alpha*dq.vector().get_local()

    return q0, dq


def solve_newton(mymesh,param,q0,W):
    epsilon_N = 10*param.tolerance
    i = 0
    while epsilon_N > param.tolerance:
        
        # - newton step
        q0, dq = solve_newton_step(mymesh, param, q0, W)
        
        # evaluation of the residual epsilon_N = L2 norm of dq ...
        epsilon_N = sqrt(assemble((dq[0]*dq[0]  + dq[1]*dq[1] + dq[3]*dq[3])* dx ))
        # epsilon_N = norm(dq, norm_type="l2", mesh=mymesh)
        # if i % 10 == 0:
        print("      step %d\t,eps = %g"%(i,epsilon_N))
        i+=1


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
    LNS = Jacobian_variational_formulation( param, u0, p0, T0 ,uhat, phat, That, w , q, theta  )
    m   =  - inner(uhat, w)*dx - That*theta*dx


    #
    # ---  don't touch the rest of that function  ---
    #

    A, M = PETScMatrix(),  PETScMatrix()

    # assemble the matrices
    assemble(LNS, tensor=A)
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

    folder = 'Data/Pr_%1.3f/Ra_%d/Order_%d/' % (param.Prandtl, int(param.Rayleigh), param.order)
    # folder = 'Pr_%3.1f/Ra_%f/'%(param.Prandtl,param.Rayleigh)
    # File(folder+"realPart_Vel.pvd") <<  ur
    # File(folder+"imagPart_Vel.pvd") <<  ui
    # File(folder+"realPart_theta.pvd") <<  thetar
    # File(folder+"imagPart_theta.pvd") <<  thetai
    name = folder+"eigenvalues.csv"
    os.makedirs(os.path.dirname(name), exist_ok=True)
    np.savetxt(name,eigenvalues, delimiter=',')
    np.savetxt(folder+"realPart.csv",egv_real_part.vector().get_local(), delimiter=',')
    np.savetxt(folder+"imagPart.csv",egv_imag_part.vector().get_local(), delimiter=',')

    # np.savetxt(folder+"eigenvalues.csv",eigenvalues, delimeter=',')
    # np.save(folder+"realPart.csv",egv_real_part.vector().get_local())
    # np.save(folder+"imagPart.csv",egv_imag_part.vector().get_local())

    return eigenvalues, egv_real_part, egv_imag_part




def plot_streamlines_and_isotemperature(param,q0,Wt,filename='defaultname.png'):
    # feel free to change the plotting function :)

    folder = 'Figures/Pr_%1.3f/Ra_%d/Order_%d/'%(param.Prandtl,int(param.Rayleigh), param.order)

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
    p = ax.contourf(X,Y,T,20)
    plt.colorbar(p)
    name = folder+filename
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name,dpi=100,bb_inches='tight')

    plt.close()
