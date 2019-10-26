# Firedrake geometric multigrid solver
# arising from the steady state of a parabolic ice sheet problem
# uses Firedrake fem library with a PETSc interface
# docker run -ti -v $(pwd):/home/firedrake/shared/ firedrakeproject/firedrake
# curl -O  https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
# source firedrake/bin/activate

from firedrake import *
import ufl
import numpy as np
from scipy.sparse import csr_matrix
import scipy
import matplotlib.pyplot as plt
from time import time

class level:

    '''
    Stores one level of a multigrid hierarchy.

    Attributes:
        presmooth         (str): PETSc smoother to be called on downsweeps
        postsmooth        (str): PETSc smoother to be called on upsweeps
        mesh             (mesh): Firedrake mesh object associated with the current level
        fspace (function space): Firedrake FunctionSpace object associated with the current level
        op       (ufl operator): unassembled LinearOperator associated with current level
        bcs (boundary conditions): boundary conditions to be applied on the current level
    '''

    def __init__(self, presmooth, postsmooth, mesh, fspace, op, A, bcs):
      
        self.presmooth  = presmooth #on the coarsest grid, this should be a direct solver
        self.postsmooth = postsmooth
        self.mesh       = mesh
        self.fspace     = fspace
        self.op         = op
        self.A          = A
        self.bcs        = bcs

class gmg_solver:

  '''
  A classic GMG solver.
  
  Methods:       lvl_solve: called within the 'solve' method to recursively solve                                           the variational problem
                 solve:     calls lvl_solve to recursively solve the variational
                            problem
              
  Attributes:    levels   (list): a list of level objects ordered from coarse to fine
                 preiters  (int): number of smoothing iterations on down-sweeps
                 postiters (int): number of smoothing iterations on up-sweeps
  '''

  def __init__(self, levels, preiters, postiters, resid_norm=norm):
    
    self.levels    = levels
    self.level     = self.levels[0]
    self.residuals = []
    self.preiters  = preiters
    self.postiters = postiters

  def lvl_solve(self, lvl, u, b, cycle):

      self.level = self.levels[lvl]
      solve(self.level.A, u, b, solver_parameters = {'ksp_type': 'richardson',
                                          'ksp_max_it': self.preiters,
                                          'ksp_convergence_test': 'skip',
                                          'ksp_initial_guess_nonzero': True,
                                          'ksp_richardson_scale': 1,
                                          'pc_type': self.level.presmooth}) #what does ksp_type richardson, pc_type preonly, do, exactly?
      
      uc = Function(self.levels[lvl + 1].fspace)
      bc = Function(self.levels[lvl + 1].fspace)
      r = Function(self.level.fspace)
      v = TestFunction(self.level.fspace)
      r.assign(b - assemble(action(self.level.op, u), bcs=self.level.bcs))
      restrict(r, bc)
      
      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, bc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, bc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, bc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, bc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, 'W')

      else:
          solve(self.levels[lvl + 1].A, uc, bc, solver_parameters = {'pc_type': self.levels[lvl + 1].presmooth, 'ksp_type': 'preonly'})
  
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      solve(self.level.A, u, b, solver_parameters = {'ksp_type': 'richardson',
                                          'ksp_max_it': self.postiters,
                                          'pc_type': self.level.postsmooth,
                                          'ksp_convergence_test' : 'skip',
                                          'ksp_richardson_scale': 1,
                                          'ksp_initial_guess_nonzero': True})
      
  
  def solve(self, u0, b, resid_norm, cycle='V', rtol=1e-12, maxiters=50):
    
    self.level = self.levels[0]
    self.residuals.append( norm(b - assemble(action(self.level.op, u0), bcs=self.level.bcs)) )
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    u = u0
    z = 0
    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, cycle)
        self.residuals.append( norm(b - assemble(action(self.level.op, u), bcs=self.level.bcs)) )
        z += 1

    if z == maxiters:
      print('maxiters exceeded')
      print('gmg final residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
      print('convergence factor gmg: ' + str((self.residuals[-1] / self.residuals[0]) ** (1.0 / len(self.residuals))) + '\n')

    else:
      print('gmg coverged within rtol')
      print('gmg final residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
      print('convergence factor gmg: ' + str((self.residuals[-1] / self.residuals[0]) ** (1.0 / len(self.residuals))) + '\n')
    
    return u

class licp_gmg_solver:

  '''
  A linear implicit complentarity problem (LICP) solver based on the classic
  monotone multigrid method for variational inequalities. Only differs from the
  classic GMG solver in the smoother, which enforces an inequality constraint
  by projection, and the addition of a special restriction operator for the
  constraint.
  
  Methods:       lvl_solve: called within the 'solve' method to recursively solve                                           the variational problem
                 solve:     calls lvl_solve to recursively solve the variational
                            problem
              
  Attributes:    levels   (list): a list of level objects ordered from coarse to fine
                 preiters  (int): number of smoothing iterations on down-sweeps
                 postiters (int): number of smoothing iterations on up-sweeps
  '''


  def __init__(self, levels, preiters, postiters, resid_norm=norm):
    
    self.levels    = levels
    self.level     = self.levels[0]
    self.residuals = []
    self.preiters  = preiters
    self.postiters = postiters

  def lvl_solve(self, lvl, u, b, cycle):

      self.level = self.levels[lvl]
      solve(self.level.A, u, b, solver_parameters = {'ksp_type': 'richardson',
                                          'ksp_max_it': self.preiters,
                                          'ksp_convergence_test': 'skip',
                                          'ksp_initial_guess_nonzero': True,
                                          'ksp_richardson_scale': 1,
                                          'pc_type': self.level.presmooth}) #what does ksp_type richardson, pc_type preonly, do, exactly?
      
      uc = Function(self.levels[lvl + 1].fspace)
      bc = Function(self.levels[lvl + 1].fspace)
      r = Function(self.level.fspace)
      v = TestFunction(self.level.fspace)
      r.assign(b - assemble(action(self.level.op, u), bcs=self.level.bcs))
      restrict(r, bc)
      
      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, bc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, bc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, bc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, bc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, 'W')

      else:
          solve(self.levels[lvl + 1].A, uc, bc, solver_parameters = {'pc_type': self.levels[lvl + 1].presmooth, 'ksp_type': 'preonly'})
  
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      solve(self.level.A, u, b, solver_parameters = {'ksp_type': 'richardson',
                                          'ksp_max_it': self.postiters,
                                          'pc_type': self.level.postsmooth,
                                          'ksp_convergence_test' : 'skip',
                                          'ksp_richardson_scale': 1,
                                          'ksp_initial_guess_nonzero': True})
  
  
  def solve(self, u0, b, resid_norm, cycle='V', rtol=1e-12, maxiters=50):
    
    self.level = self.levels[0]
    self.residuals.append( norm(b - assemble(action(self.level.op, u0), bcs=self.level.bcs)) )
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    u = u0
    z = 0
    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, cycle)
        self.residuals.append( norm(b - assemble(action(self.level.op, u), bcs=self.level.bcs)) )
        z += 1

    if z == maxiters:
      print('maxiters exceeded')
      print('gmg final residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
      print('convergence factor gmg: ' + str((self.residuals[-1] / self.residuals[0]) ** (1.0 / len(self.residuals))) + '\n')

    else:
      print('gmg coverged within rtol')
      print('gmg final residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
      print('convergence factor gmg: ' + str((self.residuals[-1] / self.residuals[0]) ** (1.0 / len(self.residuals))) + '\n')
    
    return u
