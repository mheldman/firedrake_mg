from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from relaxation import projected_gauss_seidel
from time import time
import matplotlib.pyplot as plt
import sys, getopt
import importlib

'''
A test file for multgrid NCP solvers. Along with the arguments listed below, the user should pass the name of a .py file which contains a ufl expression defining the coarse mesh, along with UFL exprressions defining the boundary condition, obstacle, initial guess, and exact solution (if applicable). If no initial guess is passed then the solver defaults to using a zero initial guess which satisfies the boudary condition.

The arguments have the following functions:

  --outputfolder
  
  The path to an output folder where the results should be saved.

  --plotresiduals
  --plotobstacle
  --plotobstacle3d
  --plotsolution
  --plotsolution3d
  --plotsolnseq
  --resheatmap
  
These tell the program what to output; in addition to the usual commandline output of residual norm on each iteration, we can plot the residual in a semilog plot, plot the obstacle, plot the solution, plot the residuals in a heat map/

  --numlevels n
  
Refine the coarse grid n - 1 times and use these levels in the multigrid. Right now we only allow uniform refinement by bisection, but the option of multiple refinements per level is available in Firedrake and could be added.

  --coarsemx m - 1 --coarsemy n - 1

m and n are the number of unknowns on the coarsest grid in the x and y directions.

  --maxiters n

Maximum number of iterations for the solver (on each level of the multigrid hierarchy in a full multigrid method)

  --cycle <'V', 'W', 'FV'>
  
Multigrid cycle type to use (on each level in full multigrid).

  --fmg --quad

Use full multigrid/quadrilateral P1 elements (instead of triangular).

  --eps r

Regularization factor.

  --rtol r --atol q

relative tolerance for the residual norm (tolerance for ||r_f||/||r_0||) and absolute tolerance (tolerance for ||r_f|| [usually close to machine precision]).

  --mgtype <'pfas', 'gmg'>

use either geometric multigrid or the projected full approximation scheme (add truncated Newton multigrid).

  --p p --q q

If probtype = 'plaplace', set p to be the exponent for the p-Laplace obstacle problem (p=2 is the classical obstacle problem). If probtype = 'ice', set p to be the exponent for the p-Laplace term and q to be the exponent for the porous medium term (q=0 is the p-Laplace problem).

  --preiters a --postiters b

Number of pre- and post-smooths in the multigrid

  --probtype <'obstacle', 'mse', 'plaplace', 'ice', 'custom'>

Set the type of obstacle problem (which diffusion coefficient to use). If it's custom, then the user should provide their own diffusion coefficient as a function D(u) of the solution u [add option for D(u) to be independent of u, e.g., D(u) = a(x)].

Example usage:

>> python test.py -d radial.py -o ./testresults/ncp/radial -p 4 -q 5 --probtype ice --numlevels 6 --maxiters 50 --fmg --fmgc 2
'''

def main(argv):
  output = ''
  plotresiduals = False
  plotobstacle = False
  plotobstacle3d = False
  plotsolution = False
  plotsolution3d = False
  plotsolnseq = False
  resheatmap = False
  plotexact = False
  plotexact3d = False
  # add coarse grid statistics? would have to make some changes to the multigrid file
  numlevels = 6
  coarsemx = 2
  coarsemy = 2
  maxiters = 50
  cycle = 'FV'
  fmg = False
  quad = False
  eps = 0.0
  rtol = 1e-8
  atol = 1e-15
  mgtype = 'pfas'
  p = 4
  q = 5
  fmgc = 2
  preiters = 2
  postiters = 2
  innits = 1
  probtype = "" #should be in obstacle, mse, plaplace (in which p should be specified, default is p=4), ice (in which q should be specified, default is q=5), bratu (in which lambda should be specified)
  
  try:
   opts, args = getopt.getopt(argv,"hi:o:p:q:d:", ["ofolder=", "plotresiduals=", "plotobstacle="
  "plotsolution=", "plotsolution3d=", "plotsolnseq=", "resheatmap=", "numlevels=", "coarsemx=", "coarsemy=", "maxiters=", "cycle=", "fmg=", "rtol=", "atol=", "mgtype=", "p=", "q=", "preiters=", "postiters=", "probtype=", "data=", "innerits=", "fmgc=", "eps=", "plotexact=", "plotexact3d="])
  except getopt.GetoptError:
    print('radial_example.py --show_active_set --show_residuals --verbose --num_grids <int> --coarse_mx <int> --coarse_my --mx <int> --my <int> <int> -o <outputfile> --maxiters <int> --ksp_type <cg> --pc_type <ilu, lu, amg, gmg> --cycle_type <V, F, W, fmg> --tol <float> --plot_solution --solver_type <rsp, pfas, pfas_rsp> --smoother <rsp, pgs>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print('radial_example.py --show_active_set --show_residuals --verbose --num_grids <int> --coarse_mx <int> --coarse_my --mx <int> --my <int> <int> -o <outputfile> --maxiters <int> --ksp_type <cg> --pc_type <ilu, lu, amg, gmg> --cycle_type <V, F, W, fmg> --tol <float> --plot_solution --solver_type <rsp, pfas, pfas_rsp>')
       sys.exit()
    elif opt in ("-o", "--ofolder"):
       output = arg
    elif opt == "--plotresiduals":
       plotresiduals = True
    elif opt == "--plotobstacle":
      plotobstacle = True
    elif opt == "--plotsolution":
      plotsolution = True
    elif opt == "--plotsolution3d":
      plotsolution3d = True
    elif opt == "--plotsolnseq":
      plotsolnseq = True
    elif opt == "--resheatmap":
      resheatmap = True
    elif opt == "--plotexact":
      plotexact = True
    elif opt == "--plotexact3d":
      plotexact3d = True
    elif opt == "--fmg":
      fmg = True
    elif opt == "--numlevels":
      numlevels = int(arg)
    elif opt == "--coarsemx":
      coarsemx = int(arg)
    elif opt == "--coarsemy":
      coarsemy = int(arg)
    elif opt == "--fmgc":
      fmgj = int(arg)
    elif opt == "--maxiters":
      maxiters = int(arg)
    elif opt == "--preiters":
      preiters = int(arg)
    elif opt == "--postiters":
      preiters = int(arg)
    elif opt == "--cycle":
      cycle = arg
    elif opt == "--rtol":
      rtol = float(arg)
    elif opt == "--atol":
      atol = float(arg)
    elif opt == "--eps":
      eps = float(arg)
    elif opt == "--mgtype":
      mgtype = arg
    elif opt == "--probtype":
      probtype = arg
    elif opt == "--innerits":
      innits = int(arg)
    elif opt == "-p":
      p = float(arg)
    elif opt == "-q":
      q = float(arg)
    elif opt in ("-d", "--data"):
      if arg.endswith(".py"):
        arg = arg[0:len(arg) - 3]
      data = importlib.import_module(arg, package=None)
      psi, f, g, D, br, c, trans, cmesh, exact, init = data.psi, data.f, data.g, data.D, data.b, data.c, data.transform, data.cmesh, data.exact, data.init
      #specify (in order) obstacle, righthand side, boundary, coarse mesh, and exact solution
    else:
      print('option ' + opt + ' not recognized')
  if probtype == "iceflat":
    def D(u):
      return u**q*inner(grad(u), grad(u))**((p-2)/2)
  elif probtype == "plaplace":
    def D(u):
      return inner(grad(u), grad(u))**((p-2)/2)
  elif probtype == "mse":
    def D(u):
      return 1./sqrt(1. + inner(u, u))
  elif probtype == "obstacle":
    def D(u):
      return Constant(1.0)
    innits = maxiters + 1
  def build_levels(numlevels):
    levels = []
    coarse = cmesh
    mh = MeshHierarchy(coarse, numlevels - 1)
    z = 0
    
    for mesh in mh:
      V  = FunctionSpace(mesh, "CG", 1)
      x, y = SpatialCoordinate(mesh)
      w = Function(V)
      w.interpolate(g(x, y))
      bcs = DirichletBC(V, w, 'on_boundary')
  
      
      uf  = Function(V)
      uf.vector()[:] = np.arange(len(uf.vector()[:]))
      
      u, v = TrialFunction(V), TestFunction(V)
      bndry = np.rint(assemble(Constant(0.0)*v*dx, bcs=DirichletBC(V, 1.0, 'on_boundary')).vector().array()).astype('bool')
      bvals = w.vector().array()[bndry]
      u = Function(V)
      w = Function(V)
      a    = (D(u, w)*inner(grad(u), grad(v)) + eps*u*v)*dx
      #a = (D(u, w)*inner(grad(u), grad(v)))*dx
      A    = None
      H = TrialFunction(V)
      b    = H*v*dx #+ k*inner(grad(u), grad(v))*dx
      B    = None
      m    = H*v*dx
      M    = assemble(m)
      lvl = licplevel(mesh, V, a, b, A, B, bcs, u, w=w, findices=None, M=M, bindices=bndry, bvals=bvals)
      
      if z > 0:
        inject(uf, uc)
        levels[z - 1].findices = uc.vector()[:]
        levels[z - 1].findices = np.rint(levels[z - 1].findices)
        levels[z - 1].findices = levels[z - 1].findices.astype('int32')
      levels.append(lvl)
      uc    = Function(V)
      z += 1
    levels.reverse()
    return levels
  
  tstart = time()
  print('building multigrid hierarchy...')
  levels = build_levels(numlevels)
  print('time to build hierarchy:', time() - tstart)
  
  x, y = SpatialCoordinate(levels[0].mesh)
  if exact is not None:
    uexact = Function(levels[0].fspace)
    uexact.interpolate(exact(x, y))

  
  w = Function(levels[0].fspace)
  w.interpolate(f(x, y))

  if fmg:
    if mgtype == 'gmg':
      v = TestFunction(levels[-fmgj + 1].fspace)
      xj, yj = SpatialCoordinate(levels[-fmgj + 1].mesh)
      u = Function(levels[-fmgj + 1].fspace)
      u = assemble(init(xj, yj)*v*dx, bcs=levels[-fmgj + 1].bcs)
    else:
      v = TestFunction(levels[-fmgj].fspace)
      xj, yj = SpatialCoordinate(levels[-fmgj].mesh)
      u = Function(levels[-fmgj].fspace)
      u = assemble(init(xj, yj)*v*dx)
      u = assemble(init(xj, yj)*v*dx, bcs=levels[-fmgj].bcs)
  else:
    u = Function(levels[0].fspace)
    u.interpolate(init(x, y))
  
  g = Function(levels[0].fspace)
  g.interpolate(psi(x, y))

  for level in levels:
    v = TestFunction(level.fspace)
    m = Function(level.fspace)
    x, y = SpatialCoordinate(level.mesh)
    m.interpolate(f(x, y))
    #print(assemble(m*v*dx).vector()[:])
    level.a = level.a - m*v*dx

  if probtype == "ice":
    print("solving ice sheet problem: p =", p, "q =", q)
  elif probtype == "plaplace":
    print("solving p-laplace obstacle problem: p =", p)
  elif probtype == "mse":
    print("solving minimal surface obstacle problem")
  elif probtype == "obstacle":
    print("solving classical obstacle problem")
  else:
    print("solving obstacle problem with custom diffusivity")


  tstart = time()
  if mgtype == 'gmg':
    gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
    if fmg:
      u = gmgsolver.fmgsolve(psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits, j=fmgj, u0=u)
    else:
      gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits)
  elif mgtype == 'ngs':
    gmgsolver = ngs_solver(levels, preiters, postiters)
    if fmg:
      u = gmgsolver.fmgsolve(psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, j=fmgj, u0=u)
    else:
      gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
  elif mgtype == 'pfas':
    gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
    if fmg:
      u = gmgsolver.fmgsolve(psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, j=fmgj, u0=u)
    else:
      gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)

  print('time for gmg solve:', time() - tstart)
  if trans is not None:
    u.assign(trans(u))
  if exact is not None:
    if trans is not None:
      uexact.assign(trans(uexact))
    print('L2 error: ', norm(uexact - u))
    print('Linf error: ', np.linalg.norm(uexact.vector()[:] - u.vector()[:], np.inf))
  if fmg:
    n = 0
    for lvl in range(fmgc, len(levels) + 1):
      residuals = gmgsolver.residuals[n]
      with open(output + 'residuals' + str(lvl) + '.txt', 'w') as f:
        z = 0
        s = iter
        f.write("iter residual\n")
        for residual in residuals:
          s = str(z)
          r = "{:.6e}".format(residual/residuals[0])
          f.write(s.rjust(4, ' ') + ' ' + r.rjust(12, ' ') + '\n')
          z += 1
        f.close()
        n += 1
  else:
    residuals = gmgsolver.residuals
    with open(output + 'residuals.txt', 'w') as f:
      z = 0
      s = iter
      f.write("iter residual\n")
      for residual in residuals:
        s = str(z)
        r = "{:.6e}".format(residual/residuals[0])
        f.write(s.rjust(4, ' ') + ' ' + r.rjust(12, ' ') + '\n')
        z += 1
      f.close()

  if plotobstacle:
    print("plotting obstacle..")
    plot(g)
    plt.savefig(output + "obstacle.png") #add file name
  if plotobstacle3d:
    print("plotting 3d obstacle..")
    plot(g, plot3d=True)
    plt.savefig(output + "obstacle3d.png") #add file name
  if plotsolution:
    print("plotting solution..")
    plot(u)
    plt.savefig(output + "solution.png") #add file name
  if plotsolution3d:
    print("plotting 3d solution..")
    plot(u, plot3d=True)
    plt.savefig(output + "solution3d.png") #add file name
  if resheatmap:
      print("plotting residual..")
      r = Function(levels[0].fspace)
      r.vector()[:] = np.minimum(assemble(action(levels[0].a, u)).vector().array(), u.vector().array())
      plot(r)
      plt.savefig(output + "residual.png") #add file name
  if plotexact:
    print("plotting exact solution..")
    plot(uexact, plot3d=True)
    plt.savefig(output + "exact.png") #add file name
  if plotexact3d:
    print("plotting 3d exact solution..")
    plot(uexact, plot3d=True)
    plt.savefig(output + "exact3d.png") #add file name

 

if __name__ == "__main__":
    main(sys.argv[1:])



