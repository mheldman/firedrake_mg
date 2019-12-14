# Firedrake geometric multigrid solver
# arising from the steady state of a parabolic ice sheet problem
# uses Firedrake fem library with a PETSc interface
# docker run -ti -v $(pwd):/home/firedrake/shared/ firedrakeproject/firedrake
# curl -O  https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
# source firedrake/bin/activate

from firedrake import *
import numpy as np
from time import time
from relaxation import projected_gauss_seidel, symmetric_pgs
from gridtransfers import mrestrict, inject1
import matplotlib.pyplot as plt
import ufl


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
      r.assign(b - assemble(action(self.level.op, u)))
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

class licplevel:

    '''
    Stores one level of a multigrid hierarchy.

    Attributes:
        presmooth         (str): PETSc smoother to be called on downsweeps
        postsmooth        (str): PETSc smoother to be called on upsweeps
        mesh             (mesh): Firedrake mesh object associated with the current level
        fspace (function space): Firedrake FunctionSpace object associated with the current level
        a, b    (ufl operators): unassembled NonlinearOperators associated with current level
        A, B    (assmebled ufl operators): assembled versions of A and B
        M       (assembled ufl operator): assembled mass matrix
        H       (Function or TestFunction): Function or TestFunction for the forms a and b
        bcs (boundary conditions): boundary conditions to be applied on the current level
        
    '''

    def __init__(self, mesh, fspace, a, b, A, B, bcs, H, M=None, findices=None, bindices=None, bvals=None):
      
        self.mesh       = mesh
        self.fspace     = fspace
        self.a          = a
        self.A          = A
        self.b          = b
        self.B          = B
        self.M          = M
        self.findices   = findices
        self.bcs        = bcs
        self.H          = H
        self.Ja         = None
        self.Jb         = None
        if bindices is None:
          w = Function(fspace)
          bindices = np.zeros(len(w.vector().array()))
        if bvals is None:
          bvals = []
        self.bindices   = bindices
        self.bvals      = bvals


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

  def lvl_solve(self, lvl, u, b, c, cycle):

      self.level = self.levels[lvl]
      ai, aj, av = self.level.A.M.handle.getValuesCSR()
      bi, bj, bv = self.level.B.M.handle.getValuesCSR()
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array())
      uc = Function(self.levels[lvl + 1].fspace)
      bc = Function(self.levels[lvl + 1].fspace)
      cc = Function(self.levels[lvl + 1].fspace)
      ra = Function(self.level.fspace)
      rb = Function(self.level.fspace)
      
      ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs))
      restrict(ra, bc)
      rb.assign(c - assemble(action(self.level.b, u), bcs=self.level.bcs))
      mi, mj, mv = self.level.M.M.handle.getValuesCSR()
      '''
      w = Function(self.level.fspace)
      solve(self.level.M, w, rb, solver_parameters = {'ksp_type': 'preonly',
                                          'ksp_max_it': 1,
                                          'ksp_convergence_test': 'skip',
                                          'ksp_initial_guess_nonzero': True,
                                          'pc_type': 'jacobi'}) #get function space representation of residual
      '''
      mrestrict(mi, mj, mv, self.levels[lvl + 1].findices, rb.dat.data, cc.dat.data)
      
      v = TestFunction(self.levels[lvl + 1].fspace)
      cc = assemble(cc*v*dx)
      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'W')

      else:
          aic, ajc, avc = self.levels[-1].A.M.handle.getValuesCSR()
          bic, bjc, bvc = self.levels[-1].B.M.handle.getValuesCSR()
          uold = cc.copy()
          while norm(uold - uc) > 1e-10:
            uold = uc.copy()
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array())
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array())

  
  
  def solve(self, u0, b, c, cycle='V', rtol=1e-12, maxiters=50):
    
    self.level = self.levels[0]
    u = u0
    ra, rb = Function(self.level.fspace), Function(self.level.fspace)
    ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs)), rb.assign(c - assemble(action(self.level.b, u), bcs=self.level.bcs))
    self.residuals.append(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()), np.inf))
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    
    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, c, cycle)
        ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs)), rb.assign(c - assemble(action(self.level.b, u), bcs=self.level.bcs))
        self.residuals.append(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()), np.inf))
        z += 1

    if z == maxiters:
      print('maxiters exceeded')
      print('\n' + 'convergence summary')
      print('-------------------')
      residuals = self.residuals
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    else:
      print('gmg coverged within rtol')
      residuals = self.residuals
      print('\n' + 'convergence summary')
      print('-------------------')
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    return u

class obstacle_gmg_solver:

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

  def lvl_solve(self, lvl, u, b, c, cycle):

      self.level = self.levels[lvl]
      ai, aj, av = self.level.A.M.handle.getValuesCSR()
      bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
      uc = Function(self.levels[lvl + 1].fspace)
      bc = Function(self.levels[lvl + 1].fspace)
      cc = Function(self.levels[lvl + 1].fspace)
      ra = Function(self.level.fspace)
      rb = Function(self.level.fspace)
      
      ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs))
      restrict(ra, bc)
      rb.assign(c - u)
      mi, mj, mv = self.level.M.M.handle.getValuesCSR()
      '''
      w = Function(self.level.fspace)
      solve(self.level.M, w, rb, solver_parameters = {'ksp_type': 'preonly',
                                          'ksp_max_it': 1,
                                          'ksp_convergence_test': 'skip',
                                          'ksp_initial_guess_nonzero': True,
                                          'pc_type': 'jacobi'}) #get function space representation of residual
      '''
      mrestrict(mi, mj, mv, self.levels[lvl + 1].findices, rb.dat.data, cc.dat.data)

      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'W')

      else:
          aic, ajc, avc = self.levels[-1].A.M.handle.getValuesCSR()
          bic, bjc, bvc = np.arange(aic.shape[0],dtype='int32'), np.arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          uold = cc.copy(deepcopy=True)
          while norm(uold - uc) > 1e-10:
            uold = uc.copy(deepcopy=True)
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])

  
  
  def solve(self, u0, b, c, cycle='V', rtol=1e-12, maxiters=50):
    
    self.level = self.levels[0]
    u = u0
    ra, rb = Function(self.level.fspace), Function(self.level.fspace)
    ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs)), rb.assign(c - u)
    self.residuals.append(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()), np.inf))
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    
    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, c, cycle)
        ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs)), rb.assign(c - u)
        self.residuals.append(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()), np.inf))
        z += 1

    if z == maxiters:
      print('maxiters exceeded')
      print('\n' + 'convergence summary')
      print('-------------------')
      residuals = self.residuals
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    else:
      print('gmg coverged within rtol')
      residuals = self.residuals
      print('\n' + 'convergence summary')
      print('-------------------')
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    return u

class obstacle_gmg_solver:

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

  def lvl_solve(self, lvl, u, b, c, cycle):

      self.level = self.levels[lvl]
      ai, aj, av = self.level.A.M.handle.getValuesCSR()
      bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
      uc = Function(self.levels[lvl + 1].fspace)
      bc = Function(self.levels[lvl + 1].fspace)
      cc = Function(self.levels[lvl + 1].fspace)
      ra = Function(self.level.fspace)
      rb = Function(self.level.fspace)
      
      ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs))
      restrict(ra, bc)
      rb.assign(c - u)
      mi, mj, mv = self.level.M.M.handle.getValuesCSR()
      '''
      w = Function(self.level.fspace)
      solve(self.level.M, w, rb, solver_parameters = {'ksp_type': 'preonly',
                                          'ksp_max_it': 1,
                                          'ksp_convergence_test': 'skip',
                                          'ksp_initial_guess_nonzero': True,
                                          'pc_type': 'jacobi'}) #get function space representation of residual
      '''
      mrestrict(mi, mj, mv, self.levels[lvl + 1].findices, rb.dat.data, cc.dat.data)

      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'W')

      else:
          aic, ajc, avc = self.levels[-1].A.M.handle.getValuesCSR()
          bic, bjc, bvc = np.arange(aic.shape[0],dtype='int32'), np.arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          uold = cc.copy(deepcopy=True)
          while norm(uold - uc) > 1e-12:
            uold = uc.copy(deepcopy=True)
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), np.arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].bindices])
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])

  
  
  def solve(self, u0, b, c, cycle='V', rtol=1e-12, maxiters=50):
    
    self.level = self.levels[0]
    u = u0
    ra, rb = Function(self.level.fspace), Function(self.level.fspace)
    ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs)), rb.assign(c - u)
    self.residuals.append(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()), np.inf))
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    
    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, c, cycle)
        ra.assign(b - assemble(action(self.level.a, u), bcs=self.level.bcs)), rb.assign(assemble((c - u)*v*dx))
        self.residuals.append(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()), np.inf))
        z += 1

    if z == maxiters:
      print('maxiters exceeded')
      print('\n' + 'convergence summary')
      print('-------------------')
      residuals = self.residuals
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    else:
      print('gmg coverged within rtol')
      residuals = self.residuals
      print('\n' + 'convergence summary')
      print('-------------------')
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    return u


class nlobstacle_gmg_solver:

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

  def lvl_solve(self, lvl, u, b, c, cycle):

      self.level = self.levels[lvl]
      ai, aj, av = self.level.A.M.handle.getValuesCSR()
      bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
        #projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.flip(np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]))
      uc = Function(self.levels[lvl + 1].fspace)
      bc = Function(self.levels[lvl + 1].fspace)
      cc = Function(self.levels[lvl + 1].fspace)
      ra = Function(self.level.fspace)
      rb = Function(self.level.fspace)
      
      ra.assign(b - assemble(action(self.level.Ja, u)))
      restrict(ra, bc)
      rb.assign(c - u)
      mi, mj, mv = self.level.M.M.handle.getValuesCSR()
      mrestrict(mi, mj, mv, self.levels[lvl + 1].findices, rb.dat.data, cc.dat.data)

      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, bc, cc, cycle)
              self.lvl_solve(lvl + 1, uc, bc, cc, 'W')

      else:
          aic, ajc, avc = self.levels[-1].A.M.handle.getValuesCSR()
          bic, bjc, bvc = np.arange(aic.shape[0],dtype='int32'), np.arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          uold = uc.copy(deepcopy = True)
          uold.assign(uc + 1.0)
          z = 0
          maxz = len(uc.vector()[:])
          while (sqrt(len(uc.vector()[:])) + 1)*norm(uold - uc) > 1e-14 and z < maxz:
            uold = uc.copy(deepcopy=True)
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), np.arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].bindices])
            z += 1
          ra = Function(self.levels[-1].fspace)
          ra.assign(assemble(action(self.levels[-1].Ja, uc)) - bc)
          if z < maxz:
            print('coarse solve converged ' + str(z) + ' its. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - cc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
          else:
            print('coarse solve diverged. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - cc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      duf.vector()[self.level.bindices] = 0.0
      u += duf
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
        #projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.flip(np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]))
        
  def fmgsolve(self, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, innerits=1, j=2, u0=None):
    #f, g, and bc are the rhs, obstacle, and boundary condition: expressions to be interpolated
    mult = 1.0
    u, v = Function(self.levels[-j].fspace), TestFunction(self.levels[-j].fspace)
    if u0 is not None:
      u = u0
    u.vector()[self.levels[-j].bindices] = self.levels[-j].bvals
    psi = Function(self.levels[-j].fspace)
    psi.interpolate(g)
    for i in range(len(self.levels) - j, -1, -1):
      print('fmg solving level number', len(self.levels) - i)
      gmgsolver = nlobstacle_gmg_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
      gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol*mult, atol=atol, maxiters=maxiters, innerits=innerits)
      if i == len(self.levels) - j:
        mult = gmgsolver.residuals[0]
      if i > 0:
        uf = Function(self.levels[i - len(self.levels) - 1].fspace)
        prolong(u, uf)
        u = uf.copy()
        u.vector()[self.levels[i - len(self.levels) - 1].bindices] = self.levels[i - len(self.levels) - 1].bvals
        v = TestFunction(self.levels[i - len(self.levels) - 1].fspace)
        psi = Function(self.levels[i - len(self.levels) - 1].fspace)
        psi.interpolate(g)
    return u

  
  
  def solve(self, u0, c, cycle='V', atol=1e-15, rtol=1e-12, maxiters=50, innerits=1):
    
    self.level = self.levels[0]
    u = u0
    u.vector()[self.level.bindices] = self.level.bvals
    Fu = replace(self.level.a, { self.level.H : u })
    ra, rb = Function(self.level.fspace), Function(self.level.fspace)
    ra.assign(assemble(Fu)), rb.assign(u - c)
    r = np.minimum(ra.vector().array(), rb.vector().array())
    self.residuals.append(np.linalg.norm(r[~self.level.bindices]))
    b = Function(self.level.fspace)
    J = derivative(Fu, u)
    b.assign(assemble(action(J, u) - Fu))
    uf = u.copy()
    lvl = 0
    for level in self.levels:
      Fu = replace(level.a, {level.H : uf})
      level.Ja = derivative(Fu, uf)
      level.A = assemble(level.Ja)
      if lvl < len(self.levels) - 1:
        uc = Function(self.levels[lvl + 1].fspace)
        inject(uf, uc)
        uf = uc.copy()
        lvl += 1
    
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    innits = 0
    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters and self.residuals[-1] > atol:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, c, cycle)
        innits+=1
        Fu = replace(self.level.a, { self.level.H : u })
        ra, rb = Function(self.level.fspace), Function(self.level.fspace)
        ra.assign(assemble(Fu)), rb.assign(u - c)
        r = np.minimum(ra.vector().array(), rb.vector().array())
        self.residuals.append(np.linalg.norm(r[~self.level.bindices]))
        if innits == innerits:
          b = Function(self.level.fspace)
          b.assign(assemble(action(J, u) - Fu))
          uf = u.copy()
          lvl = 0
          for level in self.levels:
            Fu = replace(level.a, {level.H : uf})
            level.Ja = derivative(Fu, uf)
            level.A = assemble(level.Ja)
            if lvl < len(self.levels) - 1:
              uc = Function(self.levels[lvl + 1].fspace)
              #restrict(uf, uc)
              inject(uf, uc)
              uf = uc.copy()
              lvl += 1
            '''
            if lvl == 0:
              ai, aj, av = self.levels[0].A.M.handle.getValuesCSR()
              for i in range(len(uf.vector()[:])):
                row = []
                start, end = ai[i], ai[i + 1]
                diag = 0.0
                for j in range(start, end):
                  row.append(av[j])
                  if i == aj[j]:
                    diag = av[j]
                print('diag:' diag)
                print(row)
                print(b.vector()[i])
            '''
          
          innits = 0
          
        z+=1

    if z == maxiters:
      print('maxiters exceeded')
      print('\n' + 'convergence summary')
      print('-------------------')
      residuals = self.residuals
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    else:
      print('gmg coverged within rtol')
      residuals = self.residuals
      print('\n' + 'convergence summary')
      print('-------------------')
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))



      print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
      print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    return u



class nlobstacle_pfas_solver:

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
  
  def computeABfg(self, u, lvl):
  
    level = self.levels[lvl]
    Au, Bu = replace(level.a, {level.H : u}), replace(level.b, {level.H : u})
    Ja, Jb = derivative(Au, u), derivative(Bu, u)
    A, B   = assemble(Ja), assemble(Jb, bcs=level.bcs)
    Au     = replace(level.a, {level.H : u})
    Ja     = derivative(Au, u)
    return A, B, Au, Bu, Ja, Jb
  
  def lvl_solve(self, lvl, u, f, g, cycle):

      self.level = self.levels[lvl]
      
      A, B, Au, Bu, Ja, Jb = self.computeABfg(u, lvl)
      ai, aj, av = A.M.handle.getValuesCSR()
      bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      rhs = Function(self.level.fspace)
      rhs.assign(f + assemble(action(Ja, u) - Au, bcs=self.level.bcs)) #compute the right hand side to the linearized problem; we use the linear approximation near u to get A(v) ~ A(u) + JA(u)(v - u), so the nonlinear equation A(v) = f becomes JA(u)v = f + JA(u)u - A(u)
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, rhs.vector().array(), g.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
        '''
        solve(A, u, f, solver_parameters = {'ksp_type':'richardson',
                                                    'ksp_max_it':1,
                                                    'pc_type':'jacobi',
                                                    'ksp_convergence_test' : 'skip',
                                                    'ksp_richardson_scale':2/3,
                                                    'ksp_initial_guess_nonzero':True})
        u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:])
        '''
        A, B, Au, Bu, Ja, Jb = self.computeABfg(u, lvl)
        rhs.assign(f + assemble(action(Ja, u) - Au, bcs=self.level.bcs))
        ai, aj, av = A.M.handle.getValuesCSR()
        bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      
      uc, fc, gc = Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace)
      r = Function(self.level.fspace)
      r.assign(f - assemble(Au, bcs=self.level.bcs)) #compute nonlinear residual
      inject(r, fc), inject(g, gc), inject(u, uc)
      #inject1(r.dat.data, fc.dat.data, self.levels[lvl + 1].findices), inject1(g.dat.data, gc.dat.data, self.levels[lvl + 1].findices), inject1(u.dat.data, uc.dat.data, self.levels[lvl + 1].findices) #inject the residual, the obstacle, and the current iterate to the coarse grid. inject1 is a (matrix free) injection operator I wrote myself.. performance with firedrake injection operator is about the same although mine is slightly more accurate

      Auc = replace(self.levels[lvl + 1].a, {self.levels[lvl + 1].H : uc})
      Auc = assemble(Auc, bcs=self.levels[lvl + 1].bcs)
      fc += Auc #correct nonlinear residual R(f - A(u)) + A(Ru) to obtain the full approximation. The new problem is A(v) = R(f - A(u)) + A(Ru), which for the linear problem is the same as A(v - Ru) = R(f - A(u)). The coarse grid solution v - Ru to the error equation will be the correction
      uold = uc.copy(deepcopy=True)
      if lvl < len(self.levels) - 2:

          if cycle == 'W':
              self.lvl_solve(lvl + 1, uc, fc, gc, cycle)
              self.lvl_solve(lvl + 1, uc, fc, gc, cycle)

          elif cycle == 'V':
              self.lvl_solve(lvl + 1, uc, fc, gc, cycle)

          elif cycle == 'FV':
              self.lvl_solve(lvl + 1, uc, fc, gc, cycle)
              self.lvl_solve(lvl + 1, uc, fc, gc, 'V')

          elif cycle == 'FW':
              self.lvl_solve(lvl + 1, uc, fc, gc, cycle)
              self.lvl_solve(lvl + 1, uc, fc, gc, 'W')

      else:
          Ac, Bc, Auc, Buc, Jac, Jbc = self.computeABfg(uc, lvl + 1)
          rhsc = Function(self.levels[lvl+1].fspace)
          rhsc.assign(fc + assemble(action(Jac, uc) - Auc, bcs=self.levels[-1].bcs))
          aic, ajc, avc  = Ac.M.handle.getValuesCSR()
          bic, bjc, bvc = np.arange(aic.shape[0],dtype='int32'), np.arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          ucc = uc.copy(deepcopy=True)
          ucc.assign(uc + Constant(1.0))
          z = 0
          maxz = 10000
          while (sqrt(len(uc.vector()[:])) + 1)*norm(ucc - uc) > 1e-8 and z < maxz:
            
            ucc = uc.copy(deepcopy=True)
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, rhsc.vector().array(), gc.vector().array(), np.arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].bindices])
            '''
            solve(Ac, uc, fc, solver_parameters = {'ksp_type':'richardson',
                                                    'ksp_max_it':1,
                                                    'pc_type':'jacobi',
                                                    'ksp_convergence_test' : 'skip',
                                                    'ksp_richardson_scale':2/3,
                                                    'ksp_initial_guess_nonzero':True})
            #uc.vector()[:] = np.maximum(uc.vector()[:], gc.vector()[:])
            '''
            if z % 1000 == 0: #this could be a problem if the solution to the coarse grid problem is not trivial
              Ac, Bc, Auc, Buc, Jac, Jbc = self.computeABfg(uc, lvl + 1)
              rhsc.assign(fc + assemble(action(Jac, uc) - Auc, bcs=self.levels[-1].bcs))
              aic, ajc, avc = Ac.M.handle.getValuesCSR()
              bic, bjc, bvc = np.arange(aic.shape[0],dtype='int32'), np.arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
            z += 1
          ra = Function(self.levels[-1].fspace)
          if z < maxz:
            ra.assign(fc - assemble(action(self.levels[-1].a, uc)))
            print('coarse solve converged ' + str(z) + ' its. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
          else:
            ra.assign(assemble(fc - action(self.levels[-1].a, uc)))
            print('coarse solve diverged. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
      
      
      duf = Function(self.level.fspace)
      uc.assign(4.*(uc - uold)) #compute the correction v - Ru, where v is the smoothed coarse grid function (why do i need 4*? some scaling issue)
      prolong(uc, duf) #prolong the correction...
      u += duf #..and correct the fine grid function
      u.vector()[u.vector()[:] < g.vector()[:]] = g.vector()[u.vector()[:] < g.vector()[:]] #optional projection onto the feasible set
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, rhs.vector().array(), g.vector().array(), np.arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
        if i < self.postiters - 1:
          A, B, Au, Bu, Ja, Jb = self.computeABfg(u, lvl)
          ai, aj, av = A.M.handle.getValuesCSR()
          #bi, bj, bv = B.M.handle.getValuesCSR()
          bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
          A, B, Au, Bu, Ja, Jb = self.computeABfg(u, lvl)
          rhs.assign(f + assemble(action(Ja, u) - Au, bcs=self.level.bcs))
        '''
        solve(A, u, f, solver_parameters = {'ksp_type':'richardson',
                                                    'ksp_max_it':1,
                                                    'pc_type':'jacobi',
                                                    'ksp_convergence_test' : 'skip',
                                                    'ksp_richardson_scale':2/3,
                                                    'ksp_initial_guess_nonzero':True})
        u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:])
        '''

      
      if lvl > 0:
        self.level = self.levels[lvl - 1]

  def fmgsolve(self, f, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, inner_its=1, j=2, u0=None):
    #f, g, and bc are the rhs, obstacle, and boundary condition: expressions to be interpolated
    mult = 1.0
    u, v = Function(self.levels[-j].fspace), TestFunction(self.levels[-j].fspace)
    if u0 is not None:
      u = u0
    bc = Function(self.levels[-j].fspace)
    bc.vector()[self.levels[-j].bindices] = self.levels[-j].bvals
    u = assemble(u*v*dx, bcs=DirichletBC(self.levels[-j].fspace, bc, 'on_boundary'))
    psi = Function(self.levels[-j].fspace)
    psi.interpolate(g)
    ff = Function(self.levels[-j].fspace)
    ff.interpolate(f)
    for i in range(len(self.levels) - j, -1, -1):
      print('fmg solving level number', len(self.levels) - i)
      gmgsolver = nlobstacle_pfas_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
      gmgsolver.solve(u, assemble(ff*v*dx), psi, cycle=cycle, rtol=rtol*mult, atol=atol*mult, maxiters=maxiters, inner_its=inner_its)
      if i == len(self.levels) - j:
        mult *= gmgsolver.residuals[0]
      if i > 0:
        uf = Function(self.levels[i - len(self.levels) - 1].fspace)
        prolong(u, uf)
        u = uf.copy()
        u.vector()[self.levels[i - len(self.levels) - 1].bindices] = self.levels[i - len(self.levels) - 1].bvals
        v = TestFunction(self.levels[i - len(self.levels) - 1].fspace)
        psi = Function(self.levels[i - len(self.levels) - 1].fspace)
        psi.interpolate(g)
        ff = Function(self.levels[i - len(self.levels) - 1].fspace)
        ff.interpolate(f)
    
    return u
      
  
  

  def solve(self, u0, f, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, inner_its=1):
    
    self.level = self.levels[0]
    u = u0
    u.vector()[self.level.bindices] = self.level.bvals
    A, B, Au, Bu, Ja, Jb = self.computeABfg(u, 0)
    ra, rb = Function(self.level.fspace), Function(self.level.fspace)
    ra.assign(assemble(Au, bcs=self.level.bcs) - f)
    rb.assign(u - g)
    self.residuals.append(np.linalg.norm(np.minimum(ra.vector().array()[~self.level.bindices], rb.vector().array()[~self.level.bindices])))
    
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    while self.residuals[-1] / self.residuals[0] > rtol and self.residuals[-1] > atol and z < maxiters:
        print('\nresidual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, f, g, cycle)
        A, B, Au, Bu, Ja, Jb = self.computeABfg(u, 0)
        ra.assign(assemble(Au, bcs=self.level.bcs) - f)
        rb.assign(u - g)
        self.residuals.append(np.linalg.norm(np.minimum(ra.vector().array()[~self.level.bindices], rb.vector().array()[~self.level.bindices])))
        z += 1
    if z == maxiters:
      print('maxiters exceeded')
      print('\n' + 'convergence summary')
      print('-------------------')
      residuals = self.residuals
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))


      if len(residuals) > 1:
        print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
        print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    else:
      print('gmg coverged')
      residuals = self.residuals
      print('\n' + 'convergence summary')
      print('-------------------')
      for i in range(len(residuals)):
          if i == 0:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(
                  str(i))) + 'convergence factor 0: ---------------')
          else:
              print('residual ' + str(i) + ': ' + str(residuals[i]) + ' ' * (
              22 + len(str(maxiters)) - len(str(residuals[i])) - len(str(i))) \
                    + 'convergence factor ' + str(i) + ': ' + str(residuals[i] / residuals[i - 1]))


      if len(residuals) > 1:
        print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
        print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    return u
