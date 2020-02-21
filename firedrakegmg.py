# Firedrake geometric multigrid solver
# arising from the steady state of a parabolic ice sheet problem
# uses Firedrake fem library with a PETSc interface
# docker run -ti -v $(pwd):/home/firedrake/shared/ firedrakeproject/firedrake
# curl -O  https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
# source firedrake/bin/activate

from firedrake import *
import numpy as np
from time import time
from relaxation import projected_gauss_seidel, symmetric_pgs, gauss_seidel, rs_gauss_seidel, rsc_gauss_seidel
from pyrelaxation import nlpgs
from gridtransfers import mrestrict, inject1, restrict_cp
import matplotlib.pyplot as plt
import ufl
from numpy import arange


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

    def __init__(self, mesh, fspace, a, b, A, B, bcs, H, Ja=None, w=None, M=None, findices=None, bindices=None, bvals=None, smoother='sympgs', track_errs=False):

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
        self.w          = w
        if Ja is None:
            self.Ja         = derivative(a, H)
        else:
            self.Ja = Ja
        self.Jb         = None
        self.smoother   = smoother
        if bindices is None:
          w = Function(fspace)
          bindices = np.zeros(len(w.vector().array()))
        if bvals is None:
          bvals = []
        self.bindices   = bindices #boolean vector containing boundary indices on each levels
        self.bvals      = bvals
        self.inactive_indices = ~bindices #this boolean vector will contain the inactive indices on each level
        self.track_errs=track_errs


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
      bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
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
          bic, bjc, bvc = arange(aic.shape[0],dtype='int32'), arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          uold = cc.copy(deepcopy=True)
          while norm(uold - uc) > 1e-10:
            uold = uc.copy(deepcopy=True)
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])



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
      bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
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
          bic, bjc, bvc = arange(aic.shape[0],dtype='int32'), arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          uold = cc.copy(deepcopy=True)
          while norm(uold - uc) > 1e-12:
            uold = uc.copy(deepcopy=True)
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].bindices])
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      prolong(uc, duf)
      u += duf
      for i in range(self.postiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])



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

def compute_active(ra, rb):
  return arange(len(ra.vector()[:]))[ra.vector()[:] > 0.0 & rb.vector()[:] < 1e-15]

def zero_rows_cols(Ap, Aj, Ax, indices):
  for i in indices:
    start = Ap[i]
    end   = Ap[i + 1]
    Ax[start:end] = 0.0


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



  def inactive_set(self, u, c, Au): #compute the ative set on each level
    ra = assemble(Au).vector()[:] #compute the residual
    rb = Function(self.level.fspace)
    rb.assign(u - c)
    indc = np.zeros(len(ra))
    indc[(rb.vector()[:] > 1e-15)] = 1
    lvl = 0
    #print(ra[~self.level.bindices])
    for level in self.levels:
      level.inactive_indices = (~level.bindices) & (np.rint(indc).astype('bool')) #on the fine grid, we want to do projected gauss-seidel on the whole space. on coarse grids, we only smooth on the active nodes
      #print(level.inactive_indices)
      #print(~level.bindices)
      if lvl < len(self.levels) - 1:
        ias_f = Function(level.fspace)
        ias_fc = Function(self.levels[lvl + 1].fspace)
        ias_f.vector()[:] = indc
        inject(ias_f, ias_fc)
        indc = ias_fc.vector()[:]
      lvl += 1


    def line_search(self, u, old, duf, f, g, lvl):
        alpha = .5
        sigma = 1e-4
        alphac = 1e-2
        Auold = replace(self.level.a, {self.level.H : old})
        Au    = replace(self.level.a, {self.level.H : u})
        Ja    = replace(self.level.Ja, {self.level.H : u})
        resid = self.compute_residual(old, g, Auold, f, lvl) #compute the old residual
        while self.compute_residual(u, g, Au, f, lvl) > (1 - sigma*alpha)*resid and alpha > alphac: #if the decrease is not sufficient
          u.assign(old + alpha*duf) #move back to u + alpha*duf. In the worst case we accept the smoothed iterate
          rhs = self.update_jacobian_rhs(u, f)
          ai, aj, av = self.level.A.M.handle.getValuesCSR()
          #bi, bj, bv = B.M.handle.getValuesCSR()
          bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
          rhs.assign(f + assemble(action(Ja, u) - Au))
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, rhs.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]) #we have to smooth the new iterate to avoid unnecessarily rejecting an iterate with only high frequency errors
          alpha*=.5
        if alpha < alphac: #if the step size is too small, rejecet the correction
          u = old
          alpha = 0.0
        print('step size =', 2.*alpha)

  def update_jacobian_rhs(self, u, b):

    Fu = replace(self.level.a, { self.level.H : u, self.level.w : u})
    Ja = replace(self.level.Ja, {self.level.H : u, self.level.w : u})
    self.level.A = assemble(Ja)
    b.assign(assemble(action(Ja, u) - Fu))

  def lvl_solve(self, lvl, u, b, c, cycle):

      self.level = self.levels[lvl]
      Au = replace(self.level.a, { self.level.H : u, self.level.w : u})
      Ja = replace(self.level.Ja, {self.level.H : u, self.level.w : u})

      for i in range(self.preiters):
        if lvl == 0:
          self.update_jacobian_rhs(u, b)
          self.apply_pgs(u, b, c)
        else:
          self.apply_smoother(u, b, c)

      if lvl == 0:
        old = u.copy(deepcopy=True)
        self.update_jacobian_levels(u, b)
        #self.inactive_set(u, c, Au)

      uc, bc, cc = Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace)
      ra, rb = Function(self.level.fspace), Function(self.level.fspace)
      ra.assign(b - assemble(action(Ja, u)))
      ra.vector()[~self.level.inactive_indices] = 0.0
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
          bic, bjc, bvc = arange(aic.shape[0],dtype='int32'), arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          uold = uc.copy(deepcopy = True)
          uold.assign(uc + 1.0)
          z = 0
          maxz = 500
          while (sqrt(len(uc.vector()[:])) + 1)*norm(uold - uc) > 1e-14 and z < maxz:
            uold = uc.copy(deepcopy=True)
            if self.level.smoother == 'rsgs' and lvl != 0:
              uc.vector()[~self.levels[-1].inactive_indices] = 0.0
              bc.vector()[~self.levels[-1].inactive_indices] = 0.0
              zero_rows_cols(aic, ajc, avc, arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].inactive_indices])
              rs_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[self.levels[-1].inactive_indices]) #do gauss-seidel on the active indices and then project

            else:
              projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, bc.vector().array(), cc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].bindices])
            z += 1
          ra = Function(self.levels[-1].fspace)
          Jac = replace(self.levels[-1].Ja, {self.levels[-1].H : uc})
          ra.assign(assemble(action(Jac, uc)) - bc)
          if z < maxz:
            print('coarse solve converged ' + str(z) + ' its. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - cc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
            #print(uc.vector()[~self.levels[-1].bindices], avc)
          else:
            print('coarse solve diverged. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - cc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
      self.level = self.levels[lvl]
      duf = Function(self.level.fspace)
      uc.vector()[~self.levels[lvl + 1].inactive_indices] = 0.0
      prolong(uc, duf)

      if lvl == 0:
        duf.vector()[~self.level.inactive_indices] = 0.0
        u += duf
      else:
        u += duf

      for i in range(self.postiters):

        if lvl == 0:
          self.update_jacobian_rhs(u, b)
          self.apply_pgs(u, b, c)
          if i == 1:
            self.line_search(u, old, duf, c)
        else:
          self.apply_smoother(u, b, c)

      if lvl > 0:
        self.level = self.levels[lvl - 1]

  def compute_obst(self, g, lvl):
    psi = Function(self.levels[lvl].fspace)
    x, y = SpatialCoordinate(self.levels[lvl].mesh)
    psi.interpolate(g(x, y))
    return psi

  def prolong_u(self, u, lvl):
    uf = Function(self.levels[lvl].fspace)
    prolong(u, uf)
    uf.vector()[self.levels[lvl].bindices] = self.levels[lvl].bvals
    return uf


  def fmgsolve(self, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, innerits=1, j=2, u0=None, constrain=False):
    #f, g, and bc are the rhs, obstacle, and boundary condition: expressions to be interpolated

    mult = 1.0

    u, rhs, uold = Function(self.levels[-1].fspace), Function(self.levels[-j + 1].fspace), Function(self.levels[-j + 1].fspace)
    psi = self.compute_obst(g, -1)
    self.level = self.levels[-1]
    Au = replace(self.level.a, {self.level.H : u, self.level.w : u.copy()})
    Ja = replace(self.level.Ja, {self.level.H : u, self.level.w : u.copy()})

    if u0 is not None:
      u = u0

    for i in range(len(self.levels) - 1, -1, -1):

      if i > len(self.levels) - j:
        print('\nfmg solving level number', len(self.levels) - i, 'using one grid solver')
        gmgsolver = ngs_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
        gmgsolver.solve(u, psi, cycle=cycle, atol=1e-15, rtol=1e-12, maxiters=50)
      else:
        print('\nfmg solving level number', len(self.levels) - i, 'using multigrid solver')
        gmgsolver = nlobstacle_gmg_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
        if i > 0:
          gmgsolver.solve(u, psi, cycle=cycle, atol=1e-15, rtol=1e-12, maxiters=20, innerits=1, constrain=True)
        else:
          gmgsolver.solve(u, psi, cycle=cycle, atol=1e-15, rtol=1e-12, maxiters=maxiters, innerits=1, constrain=True)



      if i > 0:
        u = self.prolong_u(u, i - 1)
        psi = self.compute_obst(g, i - 1)
        u.vector()[:] = np.maximum(u.vector()[:], psi.vector()[:])
        u.vector()[self.levels[i-1].bindices] = self.levels[i-1].bvals
        self.level = self.levels[i-1]
      self.residuals.append(gmgsolver.residuals)

    return u

  def solve(self, u0, c, cycle='V', atol=1e-15, rtol=1e-12, maxiters=50, innerits=1, constrain=True):

    self.level = self.levels[0]
    u, b = u0, Function(self.level.fspace)
    u.vector()[:] = np.maximum(u.vector()[:], c.vector()[:])
    u.vector()[self.level.bindices] = self.level.bvals
    Au = replace(self.level.a, { self.level.H : u, self.level.w : u})
    self.residuals.append(self.compute_residual(u, c, Au))

    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    innits = 0

    while self.residuals[-1] / self.residuals[0] > rtol and z < maxiters and self.residuals[-1] > atol:

        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, b, c, cycle)
        u.vector()[:] = np.maximum(u.vector()[:], c.vector()[:])
        self.residuals.append(self.compute_residual(u, c, Au))
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



      if len(residuals) > 1:
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



      if len(residuals) > 1:
        print('aggregate convergence factor: ' + str((residuals[-1] / residuals[0]) ** (1.0 / (len(residuals) - 1.0))))
        print('residual reduction: ' + str(residuals[-1] / residuals[0]) + '\n')

    return u

  def update_jacobian_levels(self, u, b):

    lvl = 0
    uf = u.copy(deepcopy=True)
    for level in self.levels:
        Fu = replace(level.a, { level.H : uf, level.w : uf})
        Ja = replace(level.Ja, {level.H : uf, level.w : uf})
        level.A = assemble(Ja)
        if lvl == 0:
          b = Function(self.level.fspace)
          b.assign(assemble(action(Ja, uf) - Fu))
        if lvl < len(self.levels) - 1:
          uc = Function(self.levels[lvl + 1].fspace)
          inject(uf, uc)
          uf = uc.copy()
          lvl += 1

  def compute_linear_residual(self, u, c, b):

    level = self.levels[0]
    ra, rb, r = Function(level.fspace), Function(level.fspace), Function(level.fspace)
    ra.assign(assemble(action(replace(self.level.Ja, {self.level.H : u}), u)) - b), rb.assign(u - c)
    ra, rb = ra.vector()[:], rb.vector()[:]
    inactive = (rb > 0.0)
    r.vector()[inactive] = ra[inactive] #if the point is in the inactive set, then the PDE should be satisfied
    r.vector()[~inactive] = np.minimum(ra[~inactive], 0.0) #if the point is in the active set, then the PDE residual must be nonnegative
    r.vector()[level.bindices] = 0.0
    return norm(r)



  def apply_smoother(self, u, b, c):

        ai, aj, av = self.level.A.M.handle.getValuesCSR()
        bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])

        if self.level.smoother == 'pgs':
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])

        elif self.level.smoother == 'sympgs':
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]))

        elif self.level.smoother == 'rsgs':
          #ra = assemble(replace(self.level.a, {self.level.H : u})).vector()[:] #compute the residual
          #rb = Function(self.level.fspace)
          #rb.assign(u - g)
          #indc = np.zeros(len(ra), dtype='bool')
          #indc[(ra <= 0.0) | (rb.vector()[:] > 1e-15)] = 1#if F_i(u) < 0 or u_i > 0 then index i is not active
          u.vector()[~self.level.inactive_indices] = 0.0
          b.vector()[~self.level.inactive_indices] = 0.0
          rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector()[:], c.vector()[:], arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]) #do gauss-seidel on the active indices and then project
        elif self.level.smoother == 'symrsgs':
          #ra = assemble(replace(self.level.a, {self.level.H : u})).vector()[:] #compute the residual
          #rb = Function(self.level.fspace)
          #rb.assign(u - g)
          #indc = np.zeros(len(ra), dtype='bool')
          #indc[(ra <= 0.0) | (rb.vector()[:] > 1e-15)] = 1#if F_i(u) < 0 or u_i > 0 then index i is not active
          rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]) #do gauss-seidel on the active indices and then project
          rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]))
        elif self.level.smoother == 'rscgs':
          rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
        elif self.level.smoother == 'symrscgs':
          rsc_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
          rsc_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]))
        else:
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])

  def apply_pgs(self, u, b, c):

      ai, aj, av = self.level.A.M.handle.getValuesCSR()
      bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
      projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]))

  def compute_residual(self, u, c, Au):
    level = self.levels[0]
    ra, rb, r = Function(level.fspace), Function(level.fspace), Function(level.fspace)
    ra.assign(assemble(Au)), rb.assign(u - c)
    ra, rb = ra.vector()[:], rb.vector()[:]
    inactive = (rb > 0.0) | (ra <= 0)
    r.vector()[inactive] = ra[inactive] #if the point is in the inactive set, then the PDE should be satisfied
    r.vector()[~inactive] = np.minimum(ra[~inactive], 0.0) #if the point is in the active set, then the PDE residual must be nonnegative
    r.vector()[level.bindices] = 0.0
    return norm(r)


  def line_search(self, u, old, duf, c):

        alpha = .5
        sigma = 1e-4
        alphac = 1e-5
        Auold = replace(self.level.a, {self.level.H : old})
        Au    = replace(self.level.a, {self.level.H : u})
        Ja    = replace(self.level.Ja, {self.level.H : u})
        b     = Function(self.level.fspace)
        resid = self.compute_residual(old, c, Auold) #compute the old residual
        print(resid)
        while self.compute_residual(u, c, Au) > (1 - sigma*alpha)*resid and alpha > alphac: #if the decrease is not sufficient
          u.assign(old + alpha*duf) #move back to u + alpha*duf. In the worst case we accept the smoothed iterate
          self.update_jacobian_rhs(u, b)
          ai, aj, av = self.level.A.M.handle.getValuesCSR()
          bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]) #we have to smooth the new iterate to avoid unnecessarily rejecting an iterate with only high frequency errors
          alpha*=.5
        if alpha <= alphac: #if the step size is too small, rejecet the correction
          u.assign(old)
          alpha = 0.0
        print('step size =', 2.*alpha)





class nlobstacle_pfas_solver:

  '''
  A linear implicit complementarity problem (LICP) solver based on the classic
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


  def __init__(self, levels, preiters, postiters, resid_norm=norm, uexact=None):

    self.levels    = levels
    self.level     = self.levels[0]
    self.residuals = []
    self.preiters  = preiters
    self.postiters = postiters
    self.uexact = uexact

  def compute_residual(self, u, g, Au, f, lvl):
    level = self.levels[lvl]
    ra, rb, r = Function(level.fspace), Function(level.fspace), Function(level.fspace)
    ra.assign(assemble(Au) - f), rb.assign(u - g)
    ra, rb = ra.vector()[:], rb.vector()[:]
    inactive = (rb > 0.0)
    r.vector()[inactive] = ra[inactive] #if the point is in the inactive set, then the PDE should be satisfied
    r.vector()[~inactive] = np.minimum(ra[~inactive], 0.0) #if the point is in the active set, then the PDE residual must be nonnegative
    r.vector()[level.bindices] = 0.0
    return norm(r)

  def print_L2_residual(self, u, g, Au, f, lvl):
    print(lvl*' ' + 'L2 residual norm:', self.compute_residual(u, g, Au, f, lvl))

  def inactive_set(self, u, c, Au): #compute the ative set on each level
    ra = assemble(Au).vector()[:] #compute the residual
    rb = Function(self.level.fspace)
    rb.assign(u - c)
    indc = np.zeros(len(ra))
    indc[(ra <= 0.0) | (rb.vector()[:] > 1e-15)] = 1
    #indc[(rb.vector()[:] > 1e-15)] = 1
    lvl = 0
    for level in self.levels:
      if lvl > 0:
        level.inactive_indices = (~level.bindices) & (np.rint(indc).astype('bool')) #on the fine grid, we want to do projected gauss-seidel on the whole space. on coarse grids, we only smooth on the active nodes
      else:
        level.inactive_indices = (~level.bindices)
      if lvl < len(self.levels) - 1:
        ias_f = Function(level.fspace)
        ias_fc = Function(self.levels[lvl + 1].fspace)
        ias_f.vector()[:] = indc
        inject(ias_f, ias_fc)
        indc = ias_fc.vector()[:]
      lvl += 1


  def lvl_solve(self, lvl, u, f, g, cycle): #use injection for defect obstacles. less robust but can have better asymptotic convergence

      self.level = self.levels[lvl]
      Au = replace(self.level.a, {self.level.H : u})
      Ja = replace(self.level.Ja, {self.level.H : u , self.level.w : u})
      if self.level.track_errs:
        print(lvl*' ' + 'presmoothing sweep 0 on level ', lvl)
        if lvl == 0 and self.uexact is not None:
          self.print_error_stats(u)
        self.print_L2_residual(u, g, Au, f, lvl)

      for i in range(self.preiters):

        rhs = self.update_jacobian_rhs(u, f)
        #ai, aj, av = self.level.A.M.handle.getValuesCSR()
        #bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
        self.apply_smoother(Au, rhs, u, g)

        if self.level.track_errs:
          print(lvl*' ' + 'presmoothing sweep', i + 1, 'on level', lvl)
          if lvl == 0 and self.uexact is not None:
            self.print_error_stats(u)
          self.print_L2_residual(u, g, Au, f, lvl)


      uc, fc, gc = Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace)
      r = Function(self.level.fspace)
      r.assign(f - assemble(Au)) #compute nonlinear residual
      r.vector()[self.levels[lvl].bindices] = 0.0
      ru = Function(self.level.fspace)
      ru.assign(g - u)
      mi, mj, mv = self.level.M.M.handle.getValuesCSR()
      restrict(r, fc), mrestrict(mi, mj, mv, self.levels[lvl + 1].findices, ru.dat.data, gc.dat.data), inject(u, uc)
      #print(gc.vector()[:]) #rescale 'injected' coarse grid obstacle
      fc *= .25
      #uc *= .25
      gc += uc.copy(deepcopy=True)

      #inject1(r.dat.data, fc.dat.data, self.levels[lvl + 1].findices), inject1(g.dat.data, gc.dat.data, self.levels[lvl + 1].findices), inject1(u.dat.data, uc.dat.data, self.levels[lvl + 1].findices) #inject the residual, the obstacle, and the current iterate to the coarse grid. inject1 is a (matrix free) injection operator I wrote myself.. performance with firedrake injection operator is about the same although mine is slightly more accurate
      #fc.vector()[self.levels[lvl + 1].bindices] = 0.0
      #restrict_cp(u.vector()[:], uc.dat.data, g.vector()[:], gc.vector()[:], r.vector()[:], fc.dat.data, self.levels[lvl + 1].findices, np.arange(len(uc.vector()[:]), dtype='int32')[~self.levels[lvl + 1].bindices])
      Auc = replace(self.levels[lvl + 1].a, {self.levels[lvl + 1].H : uc})
      fc += assemble(Auc) #correct nonlinear residual R(f - A(u)) + A(Ru) to obtain the full approximation. The new problem is A(v) = R(f - A(u)) + A(Ru), which for the linear problem is the same as A(v - Ru) = R(f - A(u)). The coarse grid solution v - Ru to the error equation will be the correction
      #print(fc.vector()[:])
      #uc.vector()[self.levels[lvl + 1].bindices] = self.levels[lvl + 1].bvals #bcs should be just the restricted bcs from the fine grid
      #print(lvl + 1, np.max(abs(uc.vector()[:])))
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
          if self.levels[-1].track_errs:
            print((lvl + 1)*' ' + 'coarse grid sweep 0')
            self.print_L2_residual(uc, gc, Auc, fc, lvl + 1)

          self.level = self.levels[-1]
          ucc = uc.copy(deepcopy=True)
          ucc.assign(uc + Constant(1.0))
          z = 0
          maxz = 20
          while (sqrt(len(uc.vector()[:])) + 1)*norm(ucc - uc) > 1e-12 and z < maxz:
            ucc = uc.copy(deepcopy=True)
            rhsc = self.update_jacobian_rhs(uc, fc)
            aic, ajc, avc = self.level.A.M.handle.getValuesCSR()
            bic, bjc, bvc = arange(aic.shape[0],dtype='int32'), arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
            projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, rhsc.vector().array(), gc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[~self.levels[-1].bindices])

            if self.level.track_errs:
              print((lvl + 1)*' ' + 'coarse grid sweep 0')
              self.print_L2_residual(uc, gc, Auc, fc, lvl + 1)
            z += 1
          ra = Function(self.levels[-1].fspace)
          uc.vector()[uc.vector()[:] < gc.vector()[:]] = gc.vector()[uc.vector()[:] < gc.vector()[:]]
          ra.assign(assemble(Auc) - fc)
          '''
          if z < maxz:
            print((lvl + 1)*' ' + 'coarse solve converged ' + str(z) + ' its. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
          else:
            print((lvl + 1)*' ' + 'coarse solve diverged. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[~self.levels[-1].bindices], ra.vector()[~self.levels[-1].bindices])))
          '''
          self.level = self.levels[-2]

      duf = Function(self.level.fspace)
      uc.assign(4.*(uc - uold)) #compute the correction v - Ru, where v is the smoothed coarse grid function (why do i need 4*? some scaling issue)
      prolong(uc, duf) #prolong the correction...
      if lvl == 0:
        duf.vector()[self.level.bindices] = 0.0
        old = u.copy(deepcopy=True)
        u += duf #..and correct the fine grid function. how to determine if the correction is good? checking for descent in the residual is not good enough because the introduced errors might be high frequency
        #u.vector()[u.vector()[:] < g.vector()[:]] = g.vector()[u.vector()[:] < g.vector()[:]] #optional projection onto the feasible set
      else:
        duf.vector()[self.level.bindices] = 0.0
        old = u.copy(deepcopy=True)
        u += duf


      #u.vector()[self.level.bindices] = self.level.bvals

      if self.level.track_errs:
        print(lvl*' ' + 'postsmoothing sweep', 0, 'on level', lvl)
        if lvl == 0 and self.uexact is not None:
          self.print_error_stats(u)
        self.print_L2_residual(u, g, Au, f, lvl)

      for i in range(self.postiters):
        #self.apply_smoother(Au, rhs, u, g)
        rhs = self.update_jacobian_rhs(u, f)

        if i == 1: #smooth once, then apply linesearch
          #self.line_search(u, old, duf, f, g, lvl)
          rhs = self.update_jacobian_rhs(u, f)
        self.apply_smoother(Au, rhs, u, g)

        if self.level.track_errs:
          print(lvl*' ' + 'postsmoothing sweep', i + 1, 'on level', lvl)
          if lvl == 0 and self.uexact is not None:
            self.print_error_stats(u)
          self.print_L2_residual(u, g, Au, f, lvl)

      if lvl > 0:
        self.level = self.levels[lvl - 1]

  def pfas_lvl_solve(self, lvl, u, f, g, cycle):


    self.level = self.levels[lvl]
    Au = replace(self.level.a, {self.level.H : u})
    Ja = replace(self.level.Ja, {self.level.H : u})

    if self.level.track_errs:
      print(lvl*' ' + 'presmoothing sweep 0 on level ', lvl)
      if lvl == 0 and self.uexact is not None:
        self.print_error_stats(u)
      self.print_L2_residual(u, g, Au, f, lvl)
    trunc = False
    if lvl == 0:
      trunc = False
    for i in range(self.preiters):

      rhs = self.update_jacobian_rhs(u, f)
      #ai, aj, av = self.level.A.M.handle.getValuesCSR()
      #bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      self.apply_smoother(Au, rhs, u, g, tr=trunc)

      if self.level.track_errs:
        print(lvl*' ' + 'presmoothing sweep', i + 1, 'on level', lvl)
        if lvl == 0 and self.uexact is not None:
          self.print_error_stats(u)
        self.print_L2_residual(u, g, Au, f, lvl)


    uc, fc, gc = Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace)
    r = Function(self.level.fspace)
    r.assign(f - assemble(Au)) #compute nonlinear residual
    r.vector()[self.levels[lvl].bindices] = 0.0
    ru = Function(self.level.fspace)
    ru.assign(g - u)
    mi, mj, mv = self.level.M.M.handle.getValuesCSR()
    inject(r, fc), inject(ru, gc), inject(u, uc)
    #print(gc.vector()[:]) #rescale 'injected' coarse grid obstacle
    fc *= .25
    #uc *= .25
    gc += uc.copy(deepcopy=True)

    if lvl == 0:
      self.inactive_set(u, g, Au)

    #inject1(r.dat.data, fc.dat.data, self.levels[lvl + 1].findices), inject1(g.dat.data, gc.dat.data, self.levels[lvl + 1].findices), inject1(u.dat.data, uc.dat.data, self.levels[lvl + 1].findices) #inject the residual, the obstacle, and the current iterate to the coarse grid. inject1 is a (matrix free) injection operator I wrote myself.. performance with firedrake injection operator is about the same although mine is slightly more accurate
    #fc.vector()[self.levels[lvl + 1].bindices] = 0.0
    #restrict_cp(u.vector()[:], uc.dat.data, g.vector()[:], gc.vector()[:], r.vector()[:], fc.dat.data, self.levels[lvl + 1].findices, np.arange(len(uc.vector()[:]), dtype='int32')[~self.levels[lvl + 1].bindices])
    Auc = replace(self.levels[lvl + 1].a, {self.levels[lvl + 1].H : uc})
    fc += assemble(Auc).copy(deepcopy=True) #correct nonlinear residual R(f - A(u)) + A(Ru) to obtain the full approximation. The new problem is A(v) = R(f - A(u)) + A(Ru), which for the linear problem is the same as A(v - Ru) = R(f - A(u)). The coarse grid solution v - Ru to the error equation will be the correction
    #print(fc.vector()[:])
    #uc.vector()[self.levels[lvl + 1].bindices] = self.levels[lvl + 1].bvals #bcs should be just the restricted bcs from the fine grid
    #print(lvl + 1, np.max(abs(uc.vector()[:])))
    uold = uc.copy(deepcopy=True)
    if lvl < len(self.levels) - 2:

        if cycle == 'W':
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, cycle)
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, cycle)

        elif cycle == 'V':
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, cycle)

        elif cycle == 'FV':
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, cycle)
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, 'V')

        elif cycle == 'FW':
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, cycle)
            self.pfas_lvl_solve(lvl + 1, uc, fc, gc, 'W')

    else:
        if self.levels[-1].track_errs:
          print((lvl + 1)*' ' + 'coarse grid sweep 0')
          self.print_L2_residual(uc, gc, Auc, fc, lvl + 1)

        self.level = self.levels[-1]
        ucc = uc.copy(deepcopy=True)
        ucc.assign(uc + Constant(1.0))
        z = 0
        maxz = 20
        while (sqrt(len(uc.vector()[:])) + 1)*norm(ucc - uc) > 1e-12 and z < maxz:
          ucc = uc.copy(deepcopy=True)
          rhsc = self.update_jacobian_rhs(uc, fc)
          aic, ajc, avc = self.level.A.M.handle.getValuesCSR()
          bic, bjc, bvc = arange(aic.shape[0],dtype='int32'), arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          #rs_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, rhsc.vector().array(), gc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[self.levels[-1].inactive_indices])
          projected_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, rhsc.vector().array(), gc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[self.levels[-1].inactive_indices])

          if self.level.track_errs:
            print((lvl + 1)*' ' + 'coarse grid sweep 0')
            self.print_L2_residual(uc, gc, Auc, fc, lvl + 1)
          z += 1
        ra = Function(self.levels[-1].fspace)
        #uc.vector()[uc.vector()[:] < gc.vector()[:]] = gc.vector()[uc.vector()[:] < gc.vector()[:]]
        ra.assign(assemble(Auc) - fc)
        if z < maxz:
          print((lvl + 1)*' ' + 'coarse solve converged ' + str(z) + ' its. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[self.levels[-1].inactive_indices], ra.vector()[self.levels[-1].inactive_indices])))
        else:
          print((lvl + 1)*' ' + 'coarse solve diverged. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[self.levels[-1].inactive_indices], ra.vector()[self.levels[-1].inactive_indices])))
        self.level = self.levels[-2]

    duf = Function(self.level.fspace)
    uc.assign(4.*(uc - uold)) #compute the correction v - Ru, where v is the smoothed coarse grid function (why do i need 4*? some scaling issue)
    prolong(uc, duf) #prolong the correction...
    if lvl == 0:
      duf.vector()[~self.level.inactive_indices] = 0.0
      old = u.copy(deepcopy=True)
      u += duf #..and correct the fine grid function. how to determine if the correction is good? checking for descent in the residual is not good enough because the introduced errors might be high frequency
      #u.vector()[u.vector()[:] < g.vector()[:]] = g.vector()[u.vector()[:] < g.vector()[:]] #optional projection onto the feasible set
    else:
      duf.vector()[~self.level.inactive_indices] = 0.0
      old = u.copy(deepcopy=True)
      u += duf



    #u.vector()[self.level.bindices] = self.level.bvals

    if self.level.track_errs:
      print(lvl*' ' + 'postsmoothing sweep', 0, 'on level', lvl)
      if lvl == 0 and self.uexact is not None:
        self.print_error_stats(u)
      self.print_L2_residual(u, g, Au, f, lvl)

    for i in range(self.postiters):
      #self.apply_smoother(Au, rhs, u, g)
      rhs = self.update_jacobian_rhs(u, f)

      if i == 1: #smooth once, then apply linesearch
        self.line_search(u, old, duf, f, g, lvl)
        rhs = self.update_jacobian_rhs(u, f)
      self.apply_smoother(Au, rhs, u, g, tr=trunc)

      if self.level.track_errs:
        print(lvl*' ' + 'postsmoothing sweep', i + 1, 'on level', lvl)
        if lvl == 0 and self.uexact is not None:
          self.print_error_stats(u)
        self.print_L2_residual(u, g, Au, f, lvl)

    if lvl > 0:
      self.level = self.levels[lvl - 1]
      self.level.inactive_indices = ~self.level.bindices

  def trunc_lvl_solve(self, lvl, u, f, g, cycle): #truncate to the inactive set and ignore the coarse grid obstacles


    self.level = self.levels[lvl]
    Au = replace(self.level.a, {self.level.H : u})
    Ja = replace(self.level.Ja, {self.level.H : u})

    if self.level.track_errs:
      print(lvl*' ' + 'presmoothing sweep 0 on level ', lvl)
      if lvl == 0 and self.uexact is not None:
        self.print_error_stats(u)
      self.print_L2_residual(u, g, Au, f, lvl)
    trunc = True
    if lvl == 0:
      trunc = False
    for i in range(self.preiters):

      rhs = self.update_jacobian_rhs(u, f)
      #ai, aj, av = self.level.A.M.handle.getValuesCSR()
      #bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      self.apply_smoother(Au, rhs, u, g, tr=trunc)

      if self.level.track_errs:
        print(lvl*' ' + 'presmoothing sweep', i + 1, 'on level', lvl)
        if lvl == 0 and self.uexact is not None:
          self.print_error_stats(u)
        self.print_L2_residual(u, g, Au, f, lvl)


    uc, fc, gc = Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace), Function(self.levels[lvl + 1].fspace)
    r = Function(self.level.fspace)
    r.assign(f - assemble(Au)) #compute nonlinear residual
    r.vector()[self.levels[lvl].bindices] = 0.0
    ru = Function(self.level.fspace)
    ru.assign(g - u)
    mi, mj, mv = self.level.M.M.handle.getValuesCSR()
    restrict(r, fc), inject(ru, gc), inject(u, uc)
    #print(gc.vector()[:]) #rescale 'injected' coarse grid obstacle
    fc *= .25
    #uc *= .25
    gc += uc.copy(deepcopy=True)

    if lvl == 0:
      self.inactive_set(u, g, Au)

    #inject1(r.dat.data, fc.dat.data, self.levels[lvl + 1].findices), inject1(g.dat.data, gc.dat.data, self.levels[lvl + 1].findices), inject1(u.dat.data, uc.dat.data, self.levels[lvl + 1].findices) #inject the residual, the obstacle, and the current iterate to the coarse grid. inject1 is a (matrix free) injection operator I wrote myself.. performance with firedrake injection operator is about the same although mine is slightly more accurate
    #fc.vector()[self.levels[lvl + 1].bindices] = 0.0
    #restrict_cp(u.vector()[:], uc.dat.data, g.vector()[:], gc.vector()[:], r.vector()[:], fc.dat.data, self.levels[lvl + 1].findices, np.arange(len(uc.vector()[:]), dtype='int32')[~self.levels[lvl + 1].bindices])
    Auc = replace(self.levels[lvl + 1].a, {self.levels[lvl + 1].H : uc})
    fc += assemble(Auc).copy(deepcopy=True) #correct nonlinear residual R(f - A(u)) + A(Ru) to obtain the full approximation. The new problem is A(v) = R(f - A(u)) + A(Ru), which for the linear problem is the same as A(v - Ru) = R(f - A(u)). The coarse grid solution v - Ru to the error equation will be the correction
    #print(fc.vector()[:])
    #uc.vector()[self.levels[lvl + 1].bindices] = self.levels[lvl + 1].bvals #bcs should be just the restricted bcs from the fine grid
    #print(lvl + 1, np.max(abs(uc.vector()[:])))
    uold = uc.copy(deepcopy=True)
    if lvl < len(self.levels) - 2:

        if cycle == 'W':
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, cycle)
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, cycle)

        elif cycle == 'V':
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, cycle)

        elif cycle == 'FV':
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, cycle)
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, 'V')

        elif cycle == 'FW':
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, cycle)
            self.trunc_lvl_solve(lvl + 1, uc, fc, gc, 'W')

    else:
        if self.levels[-1].track_errs:
          print((lvl + 1)*' ' + 'coarse grid sweep 0')
          self.print_L2_residual(uc, gc, Auc, fc, lvl + 1)

        self.level = self.levels[-1]
        ucc = uc.copy(deepcopy=True)
        ucc.assign(uc + Constant(1.0))
        z = 0
        maxz = 20
        while (sqrt(len(uc.vector()[:])) + 1)*norm(ucc - uc) > 1e-12 and z < maxz:
          ucc = uc.copy(deepcopy=True)
          rhsc = self.update_jacobian_rhs(uc, fc)
          aic, ajc, avc = self.level.A.M.handle.getValuesCSR()
          bic, bjc, bvc = arange(aic.shape[0],dtype='int32'), arange(aic.shape[0] - 1,dtype='int32'), np.ones(aic.shape[0])
          #rs_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, rhsc.vector().array(), gc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[self.levels[-1].inactive_indices])
          rs_gauss_seidel(aic, ajc, avc, bic, bjc, bvc, uc.dat.data, rhsc.vector().array(), gc.vector().array(), arange(len(uc.vector().array()), dtype='int32')[self.levels[-1].inactive_indices])

          if self.level.track_errs:
            print((lvl + 1)*' ' + 'coarse grid sweep 0')
            self.print_L2_residual(uc, gc, Auc, fc, lvl + 1)
          z += 1
        ra = Function(self.levels[-1].fspace)
        #uc.vector()[uc.vector()[:] < gc.vector()[:]] = gc.vector()[uc.vector()[:] < gc.vector()[:]]
        ra.assign(assemble(Auc) - fc)
        if z < maxz:
          print((lvl + 1)*' ' + 'coarse solve converged ' + str(z) + ' its. resid norm ', np.linalg.norm(np.minimum(uc.vector()[self.levels[-1].inactive_indices] - gc.vector()[self.levels[-1].inactive_indices], ra.vector()[self.levels[-1].inactive_indices])))
        else:
          print((lvl + 1)*' ' + 'coarse solve diverged. resid norm ', np.linalg.norm(np.minimum(uc.vector()[~self.levels[-1].bindices] - gc.vector()[self.levels[-1].inactive_indices], ra.vector()[self.levels[-1].inactive_indices])))
        self.level = self.levels[-2]

    duf = Function(self.level.fspace)
    uc.assign(4.*(uc - uold)) #compute the correction v - Ru, where v is the smoothed coarse grid function (why do i need 4*? some scaling issue)
    prolong(uc, duf) #prolong the correction...
    if lvl == 0:
      duf.vector()[~self.level.inactive_indices] = 0.0
      old = u.copy(deepcopy=True)
      u += duf #..and correct the fine grid function. how to determine if the correction is good? checking for descent in the residual is not good enough because the introduced errors might be high frequency
      #u.vector()[u.vector()[:] < g.vector()[:]] = g.vector()[u.vector()[:] < g.vector()[:]] #optional projection onto the feasible set
    else:
      duf.vector()[~self.level.inactive_indices] = 0.0
      old = u.copy(deepcopy=True)
      u += duf



    #u.vector()[self.level.bindices] = self.level.bvals

    if self.level.track_errs:
      print(lvl*' ' + 'postsmoothing sweep', 0, 'on level', lvl)
      if lvl == 0 and self.uexact is not None:
        self.print_error_stats(u)
      self.print_L2_residual(u, g, Au, f, lvl)

    for i in range(self.postiters):
      #self.apply_smoother(Au, rhs, u, g)
      rhs = self.update_jacobian_rhs(u, f)

      if i == 1: #smooth once, then apply linesearch
        self.line_search(u, old, duf, f, g, lvl)
        rhs = self.update_jacobian_rhs(u, f)
      self.apply_smoother(Au, rhs, u, g, tr=trunc)

      if self.level.track_errs:
        print(lvl*' ' + 'postsmoothing sweep', i + 1, 'on level', lvl)
        if lvl == 0 and self.uexact is not None:
          self.print_error_stats(u)
        self.print_L2_residual(u, g, Au, f, lvl)

    if lvl > 0:
        self.level = self.levels[lvl - 1]

  def line_search(self, u, old, duf, f, g, lvl):
        alpha = 2.
        sigma = 1e-4
        alphac = 1e-2
        Auold = replace(self.level.a, {self.level.H : old})
        Au    = replace(self.level.a, {self.level.H : u})
        Ja    = replace(self.level.Ja, {self.level.H : u})
        resid = self.compute_residual(old, g, Auold, f, lvl) #compute the old residual
        while self.compute_residual(u, g, Au, f, lvl) > (1 - sigma*alpha)*resid and alpha > alphac: #if the decrease is not sufficient
          u.assign(old + alpha*duf) #move back to u + alpha*duf. In the worst case we accept the smoothed iterate
          rhs = self.update_jacobian_rhs(u, f)
          ai, aj, av = self.level.A.M.handle.getValuesCSR()
          #bi, bj, bv = B.M.handle.getValuesCSR()
          bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
          rhs.assign(f + assemble(action(Ja, u) - Au))
          projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, rhs.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]) #we have to smooth the new iterate to avoid unnecessarily rejecting an iterate with only high frequency errors
          alpha*=.5
        if alpha < alphac: #if the step size is too small, rejecet the correction
          u = old
          alpha = 0.0
        print('step size =', 2.*alpha)

  def update_jacobian_rhs(self, u, f):

      rhs = Function(self.level.fspace)
      w, v  = TrialFunction(self.level.fspace), TestFunction(self.level.fspace)
      Ja = replace(self.level.Ja, {self.level.H : u, self.level.w : u})
      Au = replace(self.level.a, {self.level.H : u})
      rhs.assign(f + assemble(action(Ja, u) - Au))
      self.level.A = assemble(Ja)
      return rhs

  def apply_smoother(self, Au, f, u, g, tr=False):

        ai, aj, av = self.level.A.M.handle.getValuesCSR()
        bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
        if self.level.smoother in ['nlpgs', 'symnlpgs']:
          if self.level.smoother == 'nlpgs':
            nlpgs(Au, f, u, g, arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
          else:
            nlpgs(Au, f, u, g, arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
            nlpgs(Au, f, u, g, np.flip(arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]))
        elif tr:
          rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]) #do gauss-seidel on
        else:
          if self.level.smoother == 'pgs':
            projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
            #projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]))

          elif self.level.smoother == 'sympgs':

            projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
            projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]))

          elif self.level.smoother == 'rsgs':
            uold = u.copy(deepcopy=True)
            ra = assemble(replace(self.level.a, {self.level.H : u})).vector()[:] #compute the residual
            rb = Function(self.level.fspace)
            rb.assign(u - g)
            indc = np.zeros(len(ra), dtype='bool')
            indc[(ra <= 0.0) | (rb.vector()[:] > 1e-15)] = 1#if F_i(u) < 0 or u_i > 0 then index i is not active
            self.level.inactive_indices = indc
            #u.vector()[~self.level.inactive_indices] = g.vector()[~self.level.inactive_indices]
            #u.vector()[self.level.bindices] = self.level.bvals
            #ra = assemble(replace(self.level.a, {self.level.H : u})).vector()[:] #compute the residual
            #rb = Function(self.level.fspace)
            #rb.assign(u - g)
            #indc = np.zeros(len(ra), dtype='bool')
            #indc[(ra <= 0.0) | (rb.vector()[:] > 1e-15)] = 1#if F_i(u) < 0 or u_i > 0 then index i is not active
            rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]) #do gauss-seidel on the active indices and then project
            #u.vector()[~self.level.inactive_indices] = uold.vector()[~self.level.inactive_indices]
          elif self.level.smoother == 'symrsgs':
            uold = u.copy(deepcopy=True)
            ra = assemble(replace(self.level.a, {self.level.H : u})).vector()[:] #compute the residual
            rb = Function(self.level.fspace)
            rb.assign(u - g)
            indc = np.zeros(len(ra), dtype='bool')
            indc[(ra <= 0.0) | (rb.vector()[:] > 1e-15)] = 1#if F_i(u) < 0 or u_i > 0 then index i is not active
            self.level.inactive_indices = indc
            u.vector()[self.level.inactive_indices] = 0.0
            rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]) #do gauss-seidel on the active indices and then project
            rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]))
            #u.vector()[~self.level.inactive_indices] = uold.vector()[~self.level.inactive_indices]
          elif self.level.smoother == 'rscgs':
            rs_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
          elif self.level.smoother == 'symrscgs':
            rsc_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])
            rsc_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices]))
          elif self.level.smoother == 'prich':
            solve(A, u, f, solver_parameters = {'ksp_type':'richardson',
                                                      'ksp_max_it':1,
                                                      'ksp_convergence_test' : 'skip',
                                                      'ksp_richardson_scale':1/3,
                                                      'ksp_initial_guess_nonzero':True})
            u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:]) #projection step
            u.vector()[self.level.bindices] = self.level.bvals #update the boundary values
          elif self.level.smoother == 'pjacobi':
            solve(A, u, f, solver_parameters = {'ksp_type':'richardson',
                                                      'ksp_max_it':1,
                                                      'pc_type':'jacobi',
                                                      'ksp_convergence_test' : 'skip',
                                                      'ksp_richardson_scale':1/3,
                                                      'ksp_initial_guess_nonzero':True})
            u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:]) #projection step
            u.vector()[self.level.bindices] = self.level.bvals #update the boundary values
          else:
            projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[self.level.inactive_indices])

  def print_error_stats(self, u):
    x, y = SpatialCoordinate(u)
    uexact = Function(u.function_space())
    uexact.interpolate(self.uexact(x, y))
    #print('L2 error: ', norm(uexact**(3/8) - u))
    print('L2 error: ', norm(uexact - u))
    #print('Linf error: ', np.linalg.norm(uexact.vector()[:]**(3/8) - u.vector()[:], np.inf))
    print('Linf error: ', np.linalg.norm(uexact.vector()[:] - u.vector()[:], np.inf))
    print('Max val u: ', np.linalg.norm(u.vector()[:], np.inf))

  def compute_obst(self, g, lvl):
    psi = Function(self.levels[lvl].fspace)
    x, y = SpatialCoordinate(self.levels[lvl].mesh)
    psi.interpolate(g(x, y))
    return psi

  def prolong_u(self, u, lvl):
    uf = Function(self.levels[lvl].fspace)
    prolong(u, uf)
    uf.vector()[self.levels[lvl].bindices] = self.levels[lvl].bvals
    return uf

  def fmgsolve(self, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, inner_its=1, j=2, u0=None, constrain=False, forms=None):

    #f, g, and bc are the rhs, obstacle, and boundary condition: expressions to be interpolated (change to functions which generate expressions via spatial coordinates)
    mult = 1.0
    u, v = Function(self.levels[-1].fspace), TestFunction(self.levels[-1].fspace)
    if u0 is not None:
      u = u0
    psi = self.compute_obst(g, -1)
    u.vector()[self.levels[-1].bindices] = self.levels[-1].bvals
    #u = assemble(u*v*dx, bcs=DirichletBC(self.levels[-j].fspace, bc, 'on_boundary'))

    #first loop over coarsest levels
    for i in range(len(self.levels) - 1, -1, -1):
      if i > len(self.levels) - j:
        print('\nfmg solving level number', len(self.levels) - i, 'using one grid solver')
        gmgsolver = ngs_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
      else:
        print('\nfmg solving level number', len(self.levels) - i, 'using multigrid solver')
        gmgsolver = nlobstacle_pfas_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm, uexact=self.uexact)

      if forms is not None:
        for level in gmgsolver.levels:
            level.a = forms[i](level.H, TestFunction(level.fspace))

      if i > len(self.levels) - j:
        gmgsolver.solve(u, psi, cycle=cycle, rtol=1e-100, atol=1e-10, maxiters=200, inner_its=inner_its)
      elif i > 0:
        gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=30, inner_its=inner_its)
      else:
        gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, inner_its=inner_its)

      if i > 0:
        u = self.prolong_u(u, i - 1)
        psi = self.compute_obst(g, i - 1)
        u.vector()[:] = np.maximum(u.vector()[:], psi.vector()[:])
        u.vector()[self.levels[i-1].bindices] = self.levels[i-1].bvals

      if self.uexact is not None:
        self.print_error_stats(u)

      self.residuals.append(gmgsolver.residuals)
    return u


  def inactive_set(self, u, g, Au): #compute the ative set on each level
    ra = assemble(Au).vector()[:] #compute the residual
    rb = Function(self.level.fspace)
    rb.assign(u - g)
    indc = np.zeros(len(ra))
    indc[(rb.vector()[:] > 1e-15) | (ra <= 0.0)] = 1
    lvl = 0
    for level in self.levels:
      if lvl > 0:
        level.inactive_indices = (~level.bindices) & (np.rint(indc).astype('bool')) #on the fine grid, we want to do projected gauss-seidel on the whole space. on coarse grids, we only smooth on the active nodes
      if lvl < len(self.levels) - 1:
        ias_f = Function(level.fspace)
        ias_fc = Function(self.levels[lvl + 1].fspace)
        ias_f.vector()[:] = indc
        inject(ias_f, ias_fc)
        indc = ias_fc.vector()[:]
      lvl += 1

  def solve(self, u0, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, inner_its=1, constrain=False, jac=None):

    self.level = self.levels[0]
    u = u0
    u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:])
    u.vector()[self.level.bindices] = self.level.bvals
    f = Function(self.level.fspace)
    Au = replace(self.level.a, {self.level.H : u})
    self.residuals.append(self.compute_residual(u, g, Au, f, 0))
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0

    while self.residuals[-1] / self.residuals[0] > rtol and self.residuals[-1] > atol and z < maxiters:
        print('residual iteration ' + str(z) + ': ' + str(self.residuals[-1]) + ' ratio:', self.residuals[-1] / self.residuals[z-1])
        if z % 2 == 0 or True:
            self.lvl_solve(0, u, f, g, cycle)
            u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:])
            self.residuals.append(self.compute_residual(u, g, Au, f, 0))
        else: #do a pfas cycle every other iteration to alternate between
        # strict and relaxed coarse grid obstacles
            self.trunc_lvl_solve(0, u, f, g, cycle)
            u.vector()[:] = np.maximum(u.vector()[:], g.vector()[:])
            self.residuals.append(self.compute_residual(u, g, Au, f, 0))
            for level in self.levels:
                level.inactive_indices = ~self.level.bindices
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

class ngs_solver:

  '''
  A linear implicit complementarity problem (LICP) solver based on the classic
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
    Au = replace(level.a, {level.H : u})
    Au     = replace(level.a, {level.H : u, level.w : u.copy(deepcopy=True)})
    Ja     = replace(level.Ja, {level.H : u, level.w : u.copy(deepcopy=True)})
    A   = assemble(Ja)
    return A, Au, Ja

  def lvl_solve(self, lvl, u, f, g, cycle):

      self.level = self.levels[lvl]

      A, Au, Ja = self.computeABfg(u, lvl)
      ai, aj, av = A.M.handle.getValuesCSR()
      bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
      rhs = Function(self.level.fspace)
      rhs.assign(f + assemble(action(Ja, u) - Au)) #compute the right hand side to the linearized problem; we use the linear approximation near u to get A(v) ~ A(u) + JA(u)(v - u), so the nonlinear equation A(v) = f becomes JA(u)v = f + JA(u)u - A(u)
      for i in range(self.preiters):
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, rhs.vector().array(), g.vector().array(), arange(len(u.vector().array()), dtype='int32')[~self.level.bindices])
        projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, rhs.vector().array(), g.vector().array(), np.flip(arange(len(u.vector().array()), dtype='int32')[~self.level.bindices]))
        u.vector()[:] = np.maximum( u.vector()[:], 0.0)
        #print(u.vector()[:])
        #print(av)
        if i < self.postiters - 1:
          A, Au, Ja  = self.computeABfg(u, lvl)
          ai, aj, av = A.M.handle.getValuesCSR()
          #bi, bj, bv = B.M.handle.getValuesCSR()
          bi, bj, bv = arange(ai.shape[0],dtype='int32'), arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
          A, Au, Ja  = self.computeABfg(u, lvl)
          rhs.assign(f + assemble(action(Ja, u) - Au))

  def fmgsolve(self, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, inner_its=1, j=2, u0=None):
    #f, g, and bc are the rhs, obstacle, and boundary condition: expressions to be interpolated (change to functions which generate expressions via spatial coordinates)
    mult = 1.0
    u, v = Function(self.levels[-j].fspace), TestFunction(self.levels[-j].fspace)
    if u0 is not None:
      u = u0
    bc = Function(self.levels[-j].fspace)
    bc.vector()[self.levels[-j].bindices] = self.levels[-j].bvals
    #u = assemble(u*v*dx, bcs=DirichletBC(self.levels[-j].fspace, bc, 'on_boundary'))
    psi = Function(self.levels[-j].fspace)
    x, y = SpatialCoordinate(self.levels[-j].mesh)
    psi.interpolate(g(x, y))
    for i in range(len(self.levels) - j, -1, -1):
      print('fmg solving level number', len(self.levels) - i)
      gmgsolver = ngs_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
      if i != len(self.levels) - j:
          gmgsolver.level = gmgsolver.levels[0]
          ra, rb = Function(gmgsolver.level.fspace), Function(gmgsolver.level.fspace)
          A, Au, Ja  = gmgsolver.computeABfg(u, 0)
          ra, rb, r = Function(gmgsolver.level.fspace), Function(gmgsolver.level.fspace), Function(gmgsolver.level.fspace)
          ra.assign(assemble(Au)), rb.assign(u - psi)
          r.vector()[~gmgsolver.level.bindices] = np.minimum(ra.vector()[~gmgsolver.level.bindices], rb.vector()[~gmgsolver.level.bindices])
          r.vector()[gmgsolver.level.bindices] = 0.0 #compute the intial residual to rescale the relative tolerance
          gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol*mult/norm(r), atol=atol, maxiters=maxiters, inner_its=inner_its)
      if i == len(self.levels) - j:
        gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, inner_its=inner_its)
        mult = gmgsolver.residuals[0]

      if i > 0:
        uf = Function(self.levels[i - len(self.levels) - 1].fspace)
        prolong(u, uf)
        u = uf.copy()
        u.vector()[self.levels[i - len(self.levels) - 1].bindices] = self.levels[i - len(self.levels) - 1].bvals
        v = TestFunction(self.levels[i - len(self.levels) - 1].fspace)
        psi = Function(self.levels[i - len(self.levels) - 1].fspace)
        x, y = SpatialCoordinate(self.levels[i - len(self.levels) - 1].mesh)
        psi.interpolate(g(x, y))
        u.vector()[:] = np.maximum(uf.vector()[:], psi.vector()[:])
      self.residuals.append(gmgsolver.residuals)
    return u




  def solve(self, u0, g, cycle='V', rtol=1e-12, atol=1e-15, maxiters=50, inner_its=1, constrain=False):

    self.level = self.levels[0]
    u = u0
    u.vector()[self.level.bindices] = self.level.bvals
    A, Au, Ja  = self.computeABfg(u, 0)
    ra, rb, r = Function(self.level.fspace), Function(self.level.fspace), Function(self.level.fspace)
    ra.assign(assemble(Au)), rb.assign(u - g)
    r.vector()[~self.level.bindices] = np.minimum(ra.vector()[~self.level.bindices], rb.vector()[~self.level.bindices])
    r.vector()[self.level.bindices] = 0.0
    self.residuals.append(norm(r))
    print('Multigrid solver, number of unknowns', len(u0.vector().array()))
    z = 0
    f = Function(self.level.fspace)
    while self.residuals[-1] / self.residuals[0] > rtol and self.residuals[-1] > atol and z < maxiters:
        print('\nresidual iteration ' + str(z) + ': ' + str(self.residuals[-1]))
        self.lvl_solve(0, u, f, g, cycle)
        A, Au, Ja  = self.computeABfg(u, 0)
        ra, rb, r = Function(self.level.fspace), Function(self.level.fspace), Function(self.level.fspace)
        ra.assign(assemble(Au)), rb.assign(u - g)
        r.vector()[~self.level.bindices] = np.minimum(ra.vector()[~self.level.bindices], rb.vector()[~self.level.bindices])
        r.vector()[self.level.bindices] = 0.0
        self.residuals.append(norm(r))
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


    '''
    for i in range(len(self.levels) - j, -1, -1):
      print('fmg solving level number', len(self.levels) - i)
      if i == len(self.levels) - j + 1:
        gmgsolver = ngs_solver(self.levels[i:len(self.levels)], self.preiters, self.postiters, resid_norm=norm)
      else:
        gmgsolver = nlobstacle_pfas_solver(self.levels[i:len(self.levels) - j + 2], self.preiters, self.postiters, resid_norm=norm)

      if i != len(self.levels) - j + 1:
          gmgsolver.level = gmgsolver.levels[0]
          ra, rb = Function(gmgsolver.level.fspace), Function(gmgsolver.level.fspace)
          A, Au, Ja = gmgsolver.computeABfg(u, 0)
          ra, rb, r = Function(gmgsolver.level.fspace), Function(gmgsolver.level.fspace), Function(gmgsolver.level.fspace)
          ra.assign(assemble(Au)), rb.assign(u - psi)
          r.vector()[~gmgsolver.level.bindices] = np.minimum(ra.vector()[~gmgsolver.level.bindices], rb.vector()[~gmgsolver.level.bindices])
          r.vector()[gmgsolver.level.bindices] = 0.0 #compute the intial residual to rescale the relative tolerance
          if forms is not None:
            for level in gmgsolver.levels:
              level.a = forms[i](level.H, TestFunction(level.fspace))
          if i != 0:
            gmgsolver.solve(u, psi, cycle=cycle, rtol=1e-100, atol=atol, maxiters=2, inner_its=inner_its, constrain=constrain)
            #gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol*mult/norm(r), atol=atol, maxiters=2, inner_its=inner_its, constrain=constrain)
          else:
            gmgsolver.solve(u, psi, cycle=cycle, rtol=1e-100, atol=atol, maxiters=2, inner_its=inner_its, constrain=constrain)
            #gmgsolver.solve(u, psi, cycle=cycle, rtol=rtol*mult/norm(r), atol=atol, maxiters=2, inner_its=inner_its, constrain=constrain)
          if self.uexact is not None:
            self.print_error_stats(u)

      if i == len(self.levels) - j + 1:
        gmgsolver.solve(u, psi, cycle=cycle, rtol=1e-100, atol=1e-10, maxiters=500, inner_its=inner_its)
        mult = gmgsolver.residuals[0]

      if i > 0:
        uf = Function(self.levels[i - len(self.levels) - 1].fspace)
        prolong(u, uf)
        u = uf.copy()
        u.vector()[self.levels[i - len(self.levels) - 1].bindices] = self.levels[i - len(self.levels) - 1].bvals
        v = TestFunction(self.levels[i - len(self.levels) - 1].fspace)
        psi = Function(self.levels[i - len(self.levels) - 1].fspace)
        x, y = SpatialCoordinate(self.levels[i - len(self.levels) - 1].mesh)
        psi.interpolate(g(x, y))
        u.vector()[:] = np.maximum(uf.vector()[:], psi.vector()[:])
      '''
