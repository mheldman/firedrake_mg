# python test.py -d zznonflat.py -o ./testresults/nonflattilt/ --mgtype pfas --numlevels 5 --rtol 1e-12 --eps 0.0 --maxiters 50 --atol 1e-20 --innerits 1 --fmg true --fmgc 2


#--fmg true --fmgc 2
# python testnonflat.py -d zznonflat.py -o ./testresults/nonflat/ --mgtype gmg --fmg true --fmgc 2 --numlevels 4 --rtol 1e-12 --eps 0.0 --maxiters 30 --atol 1e-20



from firedrake import *

p  = 4
q  = p + 1
H0 = 3 #km
R  = 750 #km
b0 = .5 #km (500 m)
z0 = 1.2
L  = 1000 #km
rho = 910 #kg / m^3
gc = 9.81 #m / s^2
A = 10**(-16) #Pa^-3 a^-1

def b(x, y):
  r = sqrt((x - L)**2 + (y - L)**2)
  return conditional(r > .1, -b0*cos(z0*pi*r/R), -b0)

cmesh = RectangleMesh(2, 2, 2*L, 2*L, quadrilateral=True)

def g(x, y):
  return Constant(0.0)

def f(x, y):
  r = sqrt((x - L)**2 + (y - L)**2)
  #C = -19208566844536797./1.28e22#-1.0#
  return conditional(r < R - 0.1, conditional(r > .1,  -(3/8)**3*(((3*3**0.041666666666666664*
             ((390625*2**0.5416666666666666*3**0.2916666666666667*
                (-(750 - r)**0.6666666666666666 + r**0.6666666666666666))/
                (-((-750 + r)*r))**0.6666666666666666 +
               6*sqrt(5)*pi**2*
               (-7500 + 6**0.6666666666666666*(750 - r)**1.3333333333333333 +
                  40*r - 6**0.6666666666666666*r**1.3333333333333333)**0.625*
                cos((pi*r)/625.) -
               (3125*sqrt(5)*pi*
                 (-30 + 6**0.6666666666666666*(750 - r)**0.3333333333333333 +
                   6**0.6666666666666666*r**0.3333333333333333)*
                  sin((pi*r)/625.))/
                (-7500 + 6**0.6666666666666666*(750 - r)**1.3333333333333333 +
                   40*r - 6**0.6666666666666666*r**1.3333333333333333)**0.375)*
             ((3**0.6666666666666666*
                   (-30 + 30*(1 - r/750.)**0.3333333333333333 +
                     6**0.6666666666666666*r**0.3333333333333333))/1250. -
                (3*2**0.125*3**0.041666666666666664*pi*
                   (-7500 + 6**0.6666666666666666*
                       (750 - r)**1.3333333333333333 + 40*r -
                     6**0.6666666666666666*r**1.3333333333333333)**0.625*
                   sin((pi*r)/625.))/(78125.*sqrt(5)))**2)/
           (2.44140625e8*2**0.875) +
          (-(3**0.6666666666666666*
                  (-30 + 30*(1 - r/750.)**0.3333333333333333 +
                    6**0.6666666666666666*r**0.3333333333333333))/1250. +
              (3*2**0.125*3**0.041666666666666664*pi*
                 (-7500 + 6**0.6666666666666666*(750 - r)**1.3333333333333333 +
                   40*r - 6**0.6666666666666666*r**1.3333333333333333)**0.625*
               sin((pi*r)/625.))/(78125.*sqrt(5)))**3/r)), 57625700533610391*(3/8)**3*(2.5e19/711428401649511)/6103515625000000000000000000), -57625700533610391*(3/8)**3*(2.5e19/711428401649511)/12207031250000000000000000000)
  
  
  
#711428401649511/2.5e19
  
  
  
'''
            C*((3*3**0.041666666666666664*
             ((25*2**0.5416666666666666*3**0.2916666666666667*
                  (-(750 - r)**0.6666666666666666 + r**0.6666666666666666))/
                (-((-750 + r)*r))**0.6666666666666666 +
               216*sqrt(5)*pi**2*
                (-7500 + 6**0.6666666666666666*(750 - r)**1.3333333333333333 +
                   40*r - 6**0.6666666666666666*r**1.3333333333333333)**0.625*
                cos((6*pi*r)/5.) -
               (150*sqrt(5)*pi*
                  (-30 + 6**0.6666666666666666*(750 - r)**0.3333333333333333 +
                    6**0.6666666666666666*r**0.3333333333333333)*
                  sin((6*pi*r)/5.))/
                 (-7500 + 6**0.6666666666666666*(750 - r)**1.3333333333333333 +
                  40*r - 6**0.6666666666666666*r**1.3333333333333333)**0.375)*
             ((3**0.6666666666666666*
                   (-30 + 30*(1 - r/750.)**0.3333333333333333 +
                     6**0.6666666666666666*r**0.3333333333333333))/1250. -
                (72*3**0.6666666666666666*pi*
                   (1 - (3*(1 - (1 - r/750.)**1.3333333333333333 -
                           (2*r)/1125. +
                           r**1.3333333333333333/(3750.*6**0.3333333333333333)))
                        /2.)**0.625*sin((6*pi*r)/5.))/25.)**2)/(15625.*2**0.875)
            + (-(3**0.6666666666666666*
                  (-30 + 30*(1 - r/750.)**0.3333333333333333 +
                    6**0.6666666666666666*r**0.3333333333333333))/1250. +
              (72*3**0.6666666666666666*pi*
                 (1 - (3*(1 - (1 - r/750.)**1.3333333333333333 - (2*r)/1125. +
                         r**1.3333333333333333/(3750.*6**0.3333333333333333)))/
                   2.)**0.625*sin((6*pi*r)/5.))/25.)**3/r), (1555893914407480557/312500000000000000000000000000)), -(1555893914407480557/6250000000000000000000000000000))
  
  
'''
#conditional(r < R, .01, -.02)
#
                    
def psi(x, y):
  return Constant(0.0)

def init(x, y):
  r = sqrt((x - L)**2 + (y - L)**2)
  return 1000.*Max(f(x,y), 0.0)#conditional(r < R, sin(pi*r/R)*sin(pi*r/R), 0.0)#.0001*sin(pi*r/R)*sin(pi*r/R)#

def exact(x, y):
    r = sqrt((x - L)**2 + (y - L)**2)
    return conditional( r < R, H0**((2*p/(p-1)))*(1 - ((p - 1)/(p - 2))*((r/R)**(p/(p - 1)) - (1 - r/R)**(p/(p - 1)) + 1 - (p/(p - 1))*(r/R))), 0.0)**((p-1)/(2*p))

eps = 1e-8
def form(u, v):
   x, y = SpatialCoordinate(u)
   C = 2.*A*(rho*gc)**(p-1)*(2.5e19/711428401649511)#*1.28e22/19208566844536797.
   return C*u**q/(p+1)*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))*inner(grad(u) + grad(b(x, y)), grad(v))*dx
    
   
   
   #C*inner(4*u**3*grad(u) + ((2*p)/(p-1))*u**(5/2)*grad(b(x, y)), 4*u**3*grad(u) +  ((2*p)/(p-1))*u**((p+1)/2)*grad(b(x, y)))*inner(4*u**3*grad(u) + ((2*p)/(p-1))*u**(5/2)*grad(b(x, y)), grad(v))*dx
   
   #C*u**(p+1)/(p+1)*(inner(grad(u) , grad(u) + grad(b(x, y))))*inner(grad(u), grad(v))*dx
   
   '''
Choices for the form:

  Non-transformed models (these should have an exact solution u which exhibits infinite gradients at the ice margin) and the jacobian is zero where u is zero
  
  C = 2.*A*(rho*gc)**(p-1)*1.28e22/19208566844536797.

   - full ice sheet model
   
      C*u**q/(p+1)*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))*inner(grad(u) + grad(b(x, y)), grad(v))*dx
   
   - ice sheet model with convective term removed
   
      C*u**q/(p+1)*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))*inner(grad(u), grad(v))*dx
      
   - ice sheet model with convective term included with power r
      
      C*u**q/(p+1)*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))*inner(grad(u), grad(v))*dx + C*inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y)))*u**r/(p+1)*inner(grad(b(x, y)), grad(v))*dx
      
   - ice sheet model with a different power of q on all the porous medium terms
   
      C*u**q/(p+1)*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))*inner(grad(u) + grad(b(x, y)), grad(v))*dx (just change the q exponent)
    
    - ice sheet model with no power of q (tilted p-Laplace)
      
      C*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))*inner(grad(u) + grad(b(x, y)), grad(v))*dx
      
    Transformed models (these should have exact solution u**(8/3), which does not exhibit infinite gradient at the ice margin) but the jacobian is infinite where u is zero
      
      C = 2.*A*(rho*gc*(p-1)/(2.*p))**(p-1)*1.28e22/19208566844536797.
      
    - full ice sheet model
      
      C*inner(grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)), grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)))*inner(grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)), grad(v))*dx
      
    - ice sheet model with convective term removed
      
      C*inner(grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)), grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)))*inner(grad(u), grad(v))*dx
    
    - ice sheet model with convective term retained but a smaller power of q
      
     C*inner(grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)), grad(u) + (2.*p)/(p-1)*u**((p+1)/(2.*p))*grad(b(x, y)))*inner(grad(u) + (2.*p)/(p-1)*u**q*grad(b(x, y)), grad(v))*dx (change power of q)
    
    - ice sheet model with smaller power of q on everything
      
       C*inner(grad(u) + (2.*p)/(p-1)*u**q*grad(b(x, y)), grad(u) + (2.*p)/(p-1)*u**q*grad(b(x, y)))*inner(grad(u) + (2.*p)/(p-1)*u**q*grad(b(x, y)), grad(v))*dx (change power of q)
      
    - tilted p-Laplace model
        
        C*inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y)))*inner(grad(u) + grad(b(x, y)), grad(v))*dx
'''
'''
#eps = 1e-15
#e = 1.0
#r = 0.0
#(2.*p)/(p-1)*(u + eps)**(e*(p+1)/(2.*p))*
def D(u, w):
  x, y = SpatialCoordinate(u.function_space().mesh())
  C = 2.*A*(rho*gc)**(p-1)*1.28e22/19208566844536797.
  # + (2.*p)/(p-1)*(u + eps)**(0.0*(p+1)/(2.*p))*
  #*(p-1)/(2.*p)
  return C*u**q/(p+1)*(inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y))))
  # + grad(b(x, y))
  
  #C*inner(grad(u) + (2.*p)/(p-1)*w**((p+1)/(2.*p))*grad(b(x, y)), grad(u) + (2.*p)/(p-1)*(w)**((p+1)/(2.*p))*grad(b(x, y)))
  
  #return C*inner(grad(u)+ (2.*p)/(p-1)*(u + eps)**((p+1)/(2.*p))*grad(b(x, y)), grad(u)+ (2.*p)/(p-1)*(u + eps)**((p+1)/(2.*p))*grad(b(x, y)))/(p + 1)
  
  
  #u**((p+1))
  #   inner(grad(u)+ (2.*p)/(p-1)*(u + eps)**((p+1)/(2.*p))*grad(b(x, y)), grad(u)+ (2.*p)/(p-1)*(u + eps)**((p+1)/(2.*p))*grad(b(x, y)))/(p + 1) #transformed problem
  #u**((p+1))/(p+1)*inner(grad(u) + grad(b(x, y)), grad(u) + grad(b(x, y)))
  # + grad(b(x, y))
  #
def c(u):
  x, y = SpatialCoordinate(u)
  return 1.0#(2.*p)/(p-1)*(u)**(e*(p+1)/(2.*p))
'''
def transform(u):
  return u#u**((p-1)/(2*p))
