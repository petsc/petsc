/* Program usage:  mpiexec -n <procs> ex7 [-help] [all PETSc options] */

static char help[] = "Nonlinear PDE in 2d.\n\
We solve the Stokes equation in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\n";

/*T
   Concepts: SNES^parallel Stokes example
   Concepts: DA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    The Stokes equation is given by the partial differential equation
  
        -alpha*Laplacian u - \nabla p = f,  0 < x,y < 1,

          \nabla \cdot u              = 0
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A P2/P1 finite element approximation is used to discretize the boundary
    value problem on the two triangles which make up each rectangle in the DA
    to obtain a nonlinear system of equations.

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include "petscsys.h"
#include "petscbag.h"
#include "petscda.h"
#include "petscdmmg.h"
#include "petscsnes.h"

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  PetscReal alpha;          /* parameter controlling linearity */
  PetscReal lambda;         /* parameter controlling nonlinearity */
} AppCtx;

typedef struct {
  PetscScalar u;
  PetscScalar v;
  PetscScalar p;
} Field;

static PetscScalar Kref[36] = { 0.5,  0.5, -0.5,  0,  0, -0.5,
                                0.5,  0.5, -0.5,  0,  0, -0.5,
                               -0.5, -0.5,  0.5,  0,  0,  0.5,
                                  0,    0,    0,  0,  0,    0,
                                  0,    0,    0,  0,  0,    0,
                               -0.5, -0.5,  0.5,  0,  0,  0.5};

static PetscScalar Identity[9] = {0.0833333, 0.0416667, 0.0416667,
                                  0.0416667, 0.0833333, 0.0416667,
                                  0.0416667, 0.0416667, 0.0833333};

static PetscScalar Gradient[18] = {-0.1666667, -0.1666667, -0.1666667,
                                   -0.1666667, -0.1666667, -0.1666667,
                                    0.1666667,  0.1666667,  0.1666667,
                                    0.0,        0.0,        0.0,
                                    0.0,        0.0,        0.0,
                                    0.1666667,  0.1666667,  0.1666667};

static PetscScalar Divergence[18] = {-0.1666667, 0.1666667, 0.0,
                                     -0.1666667, 0.0,       0.1666667,
                                     -0.1666667, 0.1666667, 0.0,
                                     -0.1666667, 0.0,       0.1666667,
                                     -0.1666667, 0.1666667, 0.0,
                                     -0.1666667, 0.0,       0.1666667};

/* These are */
static PetscScalar quadPoints[8] = {0.17855873, 0.15505103,
                                    0.07503111, 0.64494897,
                                    0.66639025, 0.15505103,
                                    0.28001992, 0.64494897};
static PetscScalar quadWeights[4] = {0.15902069,  0.09097931,  0.15902069,  0.09097931};

/* 
   User-defined routines
*/
extern PetscErrorCode CreateNullSpace(DMMG, Vec*);
extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,Field**,Field**,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DALocalInfo*,Field**,Mat,AppCtx*);
extern PetscErrorCode L_2Error(DA, Vec, double *, AppCtx *);
extern PetscErrorCode PrintVector(DMMG, Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG                  *dmmg;                 /* hierarchy manager */
  DA                     da;
  SNES                   snes;                 /* nonlinear solver */
  AppCtx                *user;                 /* user-defined work context */
  PetscBag               bag;
  PetscInt               its;                  /* iterations for convergence */
  SNESConvergedReason    reason;
  PetscTruth             drawContours;         /* flag for drawing contours */
  PetscErrorCode         ierr;
  PetscReal              lambda_max = 6.81, lambda_min = 0.0, error;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(AppCtx), &bag);CHKERRQ(ierr);
  ierr = PetscBagGetData(bag, (void **) &user);CHKERRQ(ierr);
  ierr = PetscBagSetName(bag, "params", "Parameters for SNES example 4");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &user->alpha, 1.0, "alpha", "Linear coefficient");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &user->lambda, 6.0, "lambda", "Nonlinear coefficient");CHKERRQ(ierr);
  ierr = PetscBagSetFromOptions(bag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&user->alpha,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lambda",&user->lambda,PETSC_NULL);CHKERRQ(ierr);
  if (user->lambda > lambda_max || user->lambda < lambda_min) {
    SETERRQ3(1,"Lambda %G is out of range [%G, %G]", user->lambda, lambda_min, lambda_max);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create multilevel DA data structure (DMMG) to manage hierarchical solvers
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGCreate(PETSC_COMM_WORLD,1,user,&dmmg);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,-3,-3,PETSC_DECIDE,PETSC_DECIDE,
                    3,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DASetFieldName(da, 0, "ooblek");CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg, (DM) da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set the discretization functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGSetSNESLocal(dmmg, FormFunctionLocal, FormJacobianLocal, 0, 0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set the problem null space
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGSetNullSpace(dmmg, PETSC_FALSE, 1, CreateNullSpace);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = DMMGSetInitialGuess(dmmg, FormInitialGuess);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D, %s\n",its,SNESConvergedReasons[reason]);CHKERRQ(ierr);
  ierr = L_2Error(DMMGGetDA(dmmg), DMMGGetx(dmmg), &error, user);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_2 error in the solution: %G\n", error);CHKERRQ(ierr);
  ierr = PrintVector(dmmg[0], DMMGGetx(dmmg));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscBagDestroy(bag);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PrintVector"
PetscErrorCode PrintVector(DMMG dmmg, Vec U)
{
  DA             da = (DA) dmmg->dm;
  Field        **u;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  for(j = ys+ym-1; j >= ys; j--) {
    for(i = xs; i < xs+xm; i++) {
      printf("u[%d][%d] = (%g, %g, %g) ", j, i, u[j][i].u, u[j][i].v, u[j][i].p);
    }
    printf("\n");
  }
  ierr = DAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
PetscErrorCode ExactSolution(PetscReal x, PetscReal y, Field *u)
{
  PetscFunctionBegin;
  (*u).u = x;
  (*u).v = -y;
  (*u).p = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateNullSpace"
PetscErrorCode CreateNullSpace(DMMG dmmg, Vec *nulls)
{
  DA             da = (DA) dmmg->dm;
  Vec            X = nulls[0];
  Field        **x;
  PetscInt       xs,ys,xm,ym,i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCorners(da, &xs, &ys, PETSC_NULL, &xm, &ym, PETSC_NULL);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, X, &x);CHKERRQ(ierr);
  for(j = ys; j < ys+ym; j++) {
    for(i = xs; i < xs+xm; i++) {
      x[j][i].u = 0.0;
      x[j][i].v = 0.0;
      x[j][i].p = 1.0;
    }
  }
  ierr = DAVecRestoreArray(da, X, &x);CHKERRQ(ierr);
  ierr = PrintVector(dmmg, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   dmmg - The DMMG context
   X - vector

   Output Parameter:
   X - vector
*/
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  AppCtx        *user = (AppCtx *) dmmg->user;
  DA             da = (DA) dmmg->dm;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      lambda,temp1,temp,hx,hy;
  Field        **x;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  if (lambda == 0.0) {
    temp1  = 0.0;
  } else {
    temp1  = lambda/(lambda + 1.0);
  }

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,My-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
#define CHECK_SOLUTION
#ifdef CHECK_SOLUTION
        ierr = ExactSolution(i*hx, j*hy, &x[j][i]);CHKERRQ(ierr);
#else
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* Boundary conditions are usually zero Dirichlet */
        ierr = ExactSolution(i*hx, j*hy, &x[j][i]);CHKERRQ(ierr);
      } else {
        x[j][i].u = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
        x[j][i].v = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
        x[j][i].p = 1.0;
      }
#endif
    }
  }

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = PrintVector(dmmg, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "constantResidual"
PetscErrorCode constantResidual(PetscReal lambda, PetscTruth isLower, int i, int j, PetscReal hx, PetscReal hy, Field r[])
{
  Field       rLocal[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  PetscScalar phi[3] = {0.0, 0.0, 0.0};
  PetscReal   xI = i*hx, yI = j*hy, hxhy = hx*hy, x, y;
  Field       res;
  PetscInt    q, k;

  PetscFunctionBegin;
  for(q = 0; q < 4; q++) {
    phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
    phi[1] = quadPoints[q*2];
    phi[2] = quadPoints[q*2+1];
    if (isLower) {
      x    = xI + quadPoints[q*2]*hx;
      y    = yI + quadPoints[q*2+1]*hy;
    } else {
      x    = xI + 1.0 - quadPoints[q*2]*hx;
      y    = yI + 1.0 - quadPoints[q*2+1]*hy;
    }
    res.u  = quadWeights[q]*(0.0);
    res.v  = quadWeights[q]*(0.0);
    res.p  = quadWeights[q]*(0.0);
    for(k = 0; k < 3; k++) {
      rLocal[k].u += phi[k]*res.u;
      rLocal[k].v += phi[k]*res.v;
      rLocal[k].p += phi[k]*res.p;
    }
  }
  for(k = 0; k < 3; k++) {
    printf("  constLocal[%d] = (%g, %g, %g)\n", k, lambda*hxhy*rLocal[k].u, lambda*hxhy*rLocal[k].v, hxhy*rLocal[k].p);
    r[k].u += lambda*hxhy*rLocal[k].u;
    r[k].v += lambda*hxhy*rLocal[k].v;
    r[k].p += hxhy*rLocal[k].p;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode nonlinearResidual(PetscReal lambda, Field u[], Field r[]) {
  Field       rLocal[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  PetscScalar phi[3] = {0.0, 0.0, 0.0};
  Field       res;
  PetscInt q;

  PetscFunctionBegin;
  for(q = 0; q < 4; q++) {
    phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
    phi[1] = quadPoints[q*2];
    phi[2] = quadPoints[q*2+1];
    res.u  = quadWeights[q]*PetscExpScalar(u[0].u*phi[0] + u[1].u*phi[1] + u[2].u*phi[2]);
    res.v  = quadWeights[q]*PetscExpScalar(u[0].v*phi[0] + u[1].v*phi[1] + u[2].v*phi[2]);
    rLocal[0].u += phi[0]*res.u;
    rLocal[0].v += phi[0]*res.v;
    rLocal[1].u += phi[1]*res.u;  
    rLocal[1].v += phi[1]*res.v;  
    rLocal[2].u += phi[2]*res.u;
    rLocal[2].v += phi[2]*res.v;
  }
  r[0].u += lambda*rLocal[0].u;
  r[0].v += lambda*rLocal[0].v;
  r[1].u += lambda*rLocal[1].u;
  r[1].v += lambda*rLocal[1].v;
  r[2].u += lambda*rLocal[2].u;
  r[2].v += lambda*rLocal[2].v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Evaluates nonlinear function, F(x).

       Process adiC(36): FormFunctionLocal

 */
PetscErrorCode FormFunctionLocal(DALocalInfo *info, Field **x, Field **f, AppCtx *user)
{
  Field          uLocal[3];
  Field          rLocal[3];
  PetscScalar    G[4];
  Field          uExact;
  PetscReal      alpha,lambda,hx,hy,hxhy,sc,detJInv;
  PetscInt       i,j,k,l;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Naive Jacobian calculation:

     J = / 1/hx  0   \ J^{-1} = / hx   0 \  1/|J| = hx*hy = |J^{-1}|
         \  0   1/hy /          \  0  hy /
   */
  alpha   = user->alpha;
  lambda  = user->lambda;
  hx      = 1.0/(PetscReal)(info->mx-1);
  hy      = 1.0/(PetscReal)(info->my-1);
  sc      = hx*hy*lambda;
  hxhy    = hx*hy; 
  detJInv = hxhy;
  G[0] = (1.0/(hx*hx)) * detJInv;
  G[1] = 0.0;
  G[2] = G[1];
  G[3] = (1.0/(hy*hy)) * detJInv;
  for(k = 0; k < 4; k++) {
    printf("G[%d] = %g\n", k, G[k]);
  }

  /* Zero the vector */
  ierr = PetscMemzero((void *) &(f[info->xs][info->ys]), info->xm*info->ym*sizeof(Field));CHKERRQ(ierr);
  /* Compute function over the locally owned part of the grid. For each
     vertex (i,j), we consider the element below:

       2 (1)    (0)
     i,j+1 --- i+1,j+1
       |  \      |
       |   \     |
       |    \    |
       |     \   |
       |      \  |
      i,j  --- i+1,j
       0         1 (2)

     and therefore we do not loop over the last vertex in each dimension.
  */
  for(j = info->ys; j < info->ys+info->ym-1; j++) {
    for(i = info->xs; i < info->xs+info->xm-1; i++) {
      /* Lower element */
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i];
      printf("Lower Solution ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  uLocal[%d] = (%g, %g, %g)\n", k, uLocal[k].u, uLocal[k].v, uLocal[k].p);
      }
      for(k = 0; k < 3; k++) {
        rLocal[k].u = 0.0;
        rLocal[k].v = 0.0;
        rLocal[k].p = 0.0;
        for(l = 0; l < 3; l++) {
          rLocal[k].u += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1])*uLocal[l].u;
          rLocal[k].v += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1])*uLocal[l].v;
          /* rLocal[k].u += hxhy*Identity[k*3+l]*uLocal[l].u; */
          /* rLocal[k].p += hxhy*Identity[k*3+l]*uLocal[l].p; */
          /* Gradient */
          rLocal[k].u += hx*Gradient[(k*2+0)*3 + l]*uLocal[l].p;
          rLocal[k].v += hy*Gradient[(k*2+1)*3 + l]*uLocal[l].p;
          /* Divergence */
          rLocal[k].p += hx*Divergence[(k*2+0)*3 + l]*uLocal[l].u + hy*Divergence[(k*2+1)*3 + l]*uLocal[l].v;
        }
      }
      printf("Lower Laplacian ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  rLocal[%d] = (%g, %g, %g)\n", k, rLocal[k].u, rLocal[k].v, rLocal[k].p);
      }
      ierr = constantResidual(1.0, PETSC_TRUE, i, j, hx, hy, rLocal);CHKERRQ(ierr);
      printf("Lower Laplacian+Constant ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  rLocal[%d] = (%g, %g, %g)\n", k, rLocal[k].u, rLocal[k].v, rLocal[k].p);
      }
      ierr = nonlinearResidual(0.0*sc, uLocal, rLocal);CHKERRQ(ierr);
      printf("Lower Full nonlinear ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  rLocal[%d] = (%g, %g, %g)\n", k, rLocal[k].u, rLocal[k].v, rLocal[k].p);
      }
      f[j][i].u   += rLocal[0].u;
      f[j][i].v   += rLocal[0].v;
      f[j][i].p   += rLocal[0].p;
      f[j][i+1].u += rLocal[1].u;
      f[j][i+1].v += rLocal[1].v;
      f[j][i+1].p += rLocal[1].p;
      f[j+1][i].u += rLocal[2].u;
      f[j+1][i].v += rLocal[2].v;
      f[j+1][i].p += rLocal[2].p;
      /* Upper element */
      uLocal[0] = x[j+1][i+1];
      uLocal[1] = x[j+1][i];
      uLocal[2] = x[j][i+1];
      printf("Upper Solution ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  uLocal[%d] = (%g, %g, %g)\n", k, uLocal[k].u, uLocal[k].v, uLocal[k].p);
      }
      for(k = 0; k < 3; k++) {
        rLocal[k].u = 0.0;
        rLocal[k].v = 0.0;
        rLocal[k].p = 0.0;
        for(l = 0; l < 3; l++) {
          rLocal[k].u += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1])*uLocal[l].u;
          rLocal[k].v += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1])*uLocal[l].v;
          /* rLocal[k].p += Identity[k*3+l]*uLocal[l].p; */
          /* Gradient */
          rLocal[k].u += hx*Gradient[(k*2+0)*3 + l]*uLocal[l].p;
          rLocal[k].v += hy*Gradient[(k*2+1)*3 + l]*uLocal[l].p;
          /* Divergence */
          rLocal[k].p += hx*Divergence[(k*2+0)*3 + l]*uLocal[l].u + hy*Divergence[(k*2+1)*3 + l]*uLocal[l].v;
        }
      }
      printf("Upper Laplacian ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  rLocal[%d] = (%g, %g, %g)\n", k, rLocal[k].u, rLocal[k].v, rLocal[k].p);
      }
      ierr = constantResidual(1.0, PETSC_TRUTH, i, j, hx, hy, rLocal);CHKERRQ(ierr);
      printf("Upper Laplacian+Constant ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  rLocal[%d] = (%g, %g, %g)\n", k, rLocal[k].u, rLocal[k].v, rLocal[k].p);
      }
      ierr = nonlinearResidual(0.0*sc, uLocal, rLocal);CHKERRQ(ierr);
      printf("Upper Full nonlinear ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 3; k++) {
        printf("  rLocal[%d] = (%g, %g, %g)\n", k, rLocal[k].u, rLocal[k].v, rLocal[k].p);
      }
      f[j+1][i+1].u += rLocal[0].u;
      f[j+1][i+1].v += rLocal[0].v;
      f[j+1][i+1].p += rLocal[0].p;
      f[j+1][i].u   += rLocal[1].u;
      f[j+1][i].v   += rLocal[1].v;
      f[j+1][i].p   += rLocal[1].p;
      f[j][i+1].u   += rLocal[2].u;
      f[j][i+1].v   += rLocal[2].v;
      f[j][i+1].p   += rLocal[2].p;
      /* Boundary conditions */
      if (i == 0 || j == 0) {
        ierr = ExactSolution(i*hx, j*hy, &uExact);CHKERRQ(ierr);
        f[j][i].u = x[j][i].u - uExact.u;
        f[j][i].v = x[j][i].v - uExact.v;
        f[j][i].p = x[j][i].p - uExact.p;
      }
      if ((i == info->mx-2) || (j == 0)) {
        ierr = ExactSolution((i+1)*hx, j*hy, &uExact);CHKERRQ(ierr);
        f[j][i+1].u = x[j][i+1].u - uExact.u;
        f[j][i+1].v = x[j][i+1].v - uExact.v;
        f[j][i+1].p = x[j][i+1].p - uExact.p;
      }
      if ((i == info->mx-2) || (j == info->my-2)) {
        ierr = ExactSolution((i+1)*hx, (j+1)*hy, &uExact);CHKERRQ(ierr);
        f[j+1][i+1].u = x[j+1][i+1].u - uExact.u;
        f[j+1][i+1].v = x[j+1][i+1].v - uExact.v;
        f[j+1][i+1].p = x[j+1][i+1].p - uExact.p;
      }
      if ((i == 0) || (j == info->my-2)) {
        ierr = ExactSolution(i*hx, (j+1)*hy, &uExact);CHKERRQ(ierr);
        f[j+1][i].u = x[j+1][i].u - uExact.u;
        f[j+1][i].v = x[j+1][i].v - uExact.v;
        f[j+1][i].p = x[j+1][i].p - uExact.p;
      }
    }
  }

  for(j = info->ys+info->ym-1; j >= info->ys; j--) {
    for(i = info->xs; i < info->xs+info->xm; i++) {
      printf("f[%d][%d] = (%g, %g, %g) ", j, i, f[j][i].u, f[j][i].v, f[j][i].p);
    }
    printf("\n");
  }
  ierr = PetscLogFlops(68.0*(info->ym-1)*(info->xm-1));CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

PetscErrorCode nonlinearJacobian(PetscReal lambda, Field u[], PetscScalar J[]) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
   FormJacobianLocal - Evaluates Jacobian matrix.
*/
PetscErrorCode FormJacobianLocal(DALocalInfo *info, Field **x, Mat jac, AppCtx *user)
{
  Field          uLocal[4];
  PetscScalar    JLocal[144];
  MatStencil     rows[4*3], cols[4*3], ident;
  PetscInt       lowerRow[3] = {0, 1, 3};
  PetscInt       upperRow[3] = {2, 3, 1};
  PetscInt       hasLower[3], hasUpper[3], velocityRows[4], pressureRows[4];
  PetscScalar    alpha,lambda,hx,hy,hxhy,detJInv,G[4],sc,one = 1.0;
  PetscInt       i,j,k,l,numRows,dof = info->dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  alpha  = user->alpha;
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = hx*hy*lambda;
  hxhy   = hx*hy; 
  detJInv = hxhy;
  G[0] = (1.0/(hx*hx)) * detJInv;
  G[1] = 0.0;
  G[2] = G[1];
  G[3] = (1.0/(hy*hy)) * detJInv;
  for(k = 0; k < 4; k++) {
    printf("G[%d] = %g\n", k, G[k]);
  }

  ierr = MatZeroEntries(jac);CHKERRQ(ierr);
  /* 
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors. 
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
#define NOT_PRES_BC 1
  for (j=info->ys; j<info->ys+info->ym-1; j++) {
    for (i=info->xs; i<info->xs+info->xm-1; i++) {
      ierr = PetscMemzero(JLocal, 144 * sizeof(PetscScalar));CHKERRQ(ierr);
      numRows = 0;
      /* Lower element */
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i+1];
      uLocal[3] = x[j+1][i];
      /* i,j */
      if (i == 0 || j == 0) {
        hasLower[0] = 0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ident.i = i; ident.j = j; ident.c = 0;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ident.i = i; ident.j = j; ident.c = 1;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#ifdef PRES_BC
        ident.i = i; ident.j = j; ident.c = 2;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#endif
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasLower[0] = 1;
        velocityRows[0] = numRows;
        rows[numRows].i = i; rows[numRows].j = j; rows[numRows].c = 0;
        numRows++;
        rows[numRows].i = i; rows[numRows].j = j; rows[numRows].c = 1;
        numRows++;
#ifdef PRES_BC
        pressureRows[0] = numRows;
        rows[numRows].i = i; rows[numRows].j = j; rows[numRows].c = 2;
        numRows++;
#endif
      }
#ifndef PRES_BC
      pressureRows[0] = numRows;
      rows[numRows].i = i; rows[numRows].j = j; rows[numRows].c = 2;
      numRows++;
#endif
      cols[0*dof+0].i = i; cols[0*dof+0].j = j; cols[0*dof+0].c = 0;
      cols[0*dof+1].i = i; cols[0*dof+1].j = j; cols[0*dof+1].c = 1;
      cols[0*dof+2].i = i; cols[0*dof+2].j = j; cols[0*dof+2].c = 2;
      /* i+1,j */
      if ((i == info->mx-2) || (j == 0)) {
        hasLower[1] = 0;
        hasUpper[2] = 0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ident.i = i+1; ident.j = j; ident.c = 0;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ident.i = i+1; ident.j = j; ident.c = 1;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#ifdef PRES_BC
        ident.i = i+1; ident.j = j; ident.c = 2;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#endif
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasLower[1] = 1;
        hasUpper[2] = 1;
        velocityRows[1] = numRows;
        rows[numRows].i = i+1; rows[numRows].j = j; rows[numRows].c = 0;
        numRows++;
        rows[numRows].i = i+1; rows[numRows].j = j; rows[numRows].c = 1;
        numRows++;
#ifdef PRES_BC
        pressureRows[1] = numRows;
        rows[numRows].i = i+1; rows[numRows].j = j; rows[numRows].c = 2;
        numRows++;
#endif
      }
#ifndef PRES_BC
      pressureRows[1] = numRows;
      rows[numRows].i = i+1; rows[numRows].j = j; rows[numRows].c = 2;
      numRows++;
#endif
      cols[1*dof+0].i = i+1; cols[1*dof+0].j = j; cols[1*dof+0].c = 0;
      cols[1*dof+1].i = i+1; cols[1*dof+1].j = j; cols[1*dof+1].c = 1;
      cols[1*dof+2].i = i+1; cols[1*dof+2].j = j; cols[1*dof+2].c = 2;
      /* i+1,j+1 */
      if ((i == info->mx-2) || (j == info->my-2)) {
        hasUpper[0] = 0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ident.i = i+1; ident.j = j+1; ident.c = 0;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ident.i = i+1; ident.j = j+1; ident.c = 1;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#ifdef PRES_BC
        ident.i = i+1; ident.j = j+1; ident.c = 2;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#endif
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasUpper[0] = 1;
        velocityRows[2] = numRows;
        rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 0;
        numRows++;
        rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 1;
        numRows++;
#ifdef PRES_BC
        pressureRows[2] = numRows;
        rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 2;
        numRows++;
#endif
      }
#ifndef PRES_BC
      pressureRows[2] = numRows;
      rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 2;
      numRows++;
#endif
      cols[2*dof+0].i = i+1; cols[2*dof+0].j = j+1; cols[2*dof+0].c = 0;
      cols[2*dof+1].i = i+1; cols[2*dof+1].j = j+1; cols[2*dof+1].c = 1;
      cols[2*dof+2].i = i+1; cols[2*dof+2].j = j+1; cols[2*dof+2].c = 2;
      /* i,j+1 */
      if ((i == 0) || (j == info->my-2)) {
        hasLower[2] = 0;
        hasUpper[1] = 0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ident.i = i; ident.j = j+1; ident.c = 0;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ident.i = i; ident.j = j+1; ident.c = 1;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#ifdef PRES_BC
        ident.i = i; ident.j = j+1; ident.c = 2;
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
#endif
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasLower[2] = 1;
        hasUpper[1] = 1;
        velocityRows[3] = numRows;
        rows[numRows].i = i; rows[numRows].j = j+1; rows[numRows].c = 0;
        numRows++;
        rows[numRows].i = i; rows[numRows].j = j+1; rows[numRows].c = 1;
        numRows++;
#ifdef PRES_BC
        pressureRows[3] = numRows;
        rows[numRows].i = i; rows[numRows].j = j+1; rows[numRows].c = 2;
        numRows++;
#endif
      }
#ifndef PRES_BC
      pressureRows[3] = numRows;
      rows[numRows].i = i; rows[numRows].j = j+1; rows[numRows].c = 2;
      numRows++;
#endif
      cols[3*dof+0].i = i; cols[3*dof+0].j = j+1; cols[3*dof+0].c = 0;
      cols[3*dof+1].i = i; cols[3*dof+1].j = j+1; cols[3*dof+1].c = 1;
      cols[3*dof+2].i = i; cols[3*dof+2].j = j+1; cols[3*dof+2].c = 2;

      /* Lower Element */
      for(k = 0; k < 3; k++) {
#ifdef PRES_BC
        if (!hasLower[k]) continue;
#endif
        for(l = 0; l < 3; l++) {
          /* Divergence */
          JLocal[pressureRows[lowerRow[k]]*dof*4 + lowerRow[l]*dof+0] += hx*Divergence[(k*2+0)*3 + l];
          JLocal[pressureRows[lowerRow[k]]*dof*4 + lowerRow[l]*dof+1] += hy*Divergence[(k*2+1)*3 + l];
/*        JLocal[pressureRows[lowerRow[k]]*dof*4 + lowerRow[l]*dof+2] += Identity[k*3 + l]; */
        }
        if (!hasLower[k]) continue;
        for(l = 0; l < 3; l++) {
          /* Laplacian */
          JLocal[velocityRows[lowerRow[k]]*dof*4     + lowerRow[l]*dof+0] += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1]);
          JLocal[(velocityRows[lowerRow[k]]+1)*dof*4 + lowerRow[l]*dof+1] += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1]);
/*        JLocal[velocityRows[lowerRow[k]]*dof*4     + lowerRow[l]*dof+0] += Identity[k*3 + l]; */
/*        JLocal[(velocityRows[lowerRow[k]]+1)*dof*4 + lowerRow[l]*dof+1] += Identity[k*3 + l]; */
          /* Gradient */
          JLocal[velocityRows[lowerRow[k]]*dof*4     + lowerRow[l]*dof+2] += hx*Gradient[(k*2+0)*3 + l];
          JLocal[(velocityRows[lowerRow[k]]+1)*dof*4 + lowerRow[l]*dof+2] += hy*Gradient[(k*2+1)*3 + l];
        }
      }
      /* Upper Element */
      for(k = 0; k < 3; k++) {
#ifdef PRES_BC
        if (!hasUpper[k]) continue;
#endif
        for(l = 0; l < 3; l++) {
          /* Divergence */
          JLocal[pressureRows[upperRow[k]]*dof*4 + upperRow[l]*dof+0] += hx*Divergence[(k*2+0)*3 + l];
          JLocal[pressureRows[upperRow[k]]*dof*4 + upperRow[l]*dof+1] += hy*Divergence[(k*2+1)*3 + l];
/*        JLocal[pressureRows[upperRow[k]]*dof*4 + upperRow[l]*dof+2] += Identity[k*3 + l]; */
        }
        if (!hasUpper[k]) continue;
        for(l = 0; l < 3; l++) {
          /* Laplacian */
          JLocal[velocityRows[upperRow[k]]*dof*4     + upperRow[l]*dof+0] += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1]);
          JLocal[(velocityRows[upperRow[k]]+1)*dof*4 + upperRow[l]*dof+1] += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1]);
          /* Gradient */
          JLocal[velocityRows[upperRow[k]]*dof*4     + upperRow[l]*dof+2] += hx*Gradient[(k*2+0)*3 + l];
          JLocal[(velocityRows[upperRow[k]]+1)*dof*4 + upperRow[l]*dof+2] += hy*Gradient[(k*2+1)*3 + l];
        }
      }

      ierr = nonlinearJacobian(-1.0*sc, uLocal, JLocal);CHKERRQ(ierr);
      printf("Element matrix for (%d, %d)\n", i, j);
      printf("   col         ");
      for(l = 0; l < 4*3; l++) {
        printf("(%d, %d, %d) ", cols[l].i, cols[l].j, cols[l].c);
      }
      printf("\n");
      for(k = 0; k < numRows; k++) {
        printf("row (%d, %d, %d): ", rows[k].i, rows[k].j, rows[k].c);
        for(l = 0; l < 4; l++) {
          printf("%9.6g %9.6g %9.6g ", JLocal[k*dof*4 + l*dof+0], JLocal[k*dof*4 + l*dof+1], JLocal[k*dof*4 + l*dof+2]);
        }
        printf("\n");
      }
      ierr = MatSetValuesStencil(jac,numRows,rows,4*dof,cols, JLocal,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "L_2Error"
/*
  L_2Error - Integrate the L_2 error of our solution over each face
*/
PetscErrorCode L_2Error(DA da, Vec fVec, double *error, AppCtx *user)
{
  DALocalInfo info;
  Vec fLocalVec;
  Field **f;
  Field u, uExact, uLocal[4];
  PetscScalar hx, hy, hxhy, x, y, phi[3];
  PetscInt i, j, q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da, &info);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &fLocalVec);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,fVec, INSERT_VALUES, fLocalVec);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,fVec, INSERT_VALUES, fLocalVec);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, fLocalVec, (void **) &f);CHKERRQ(ierr);

  *error = 0.0;
  hx     = 1.0/(PetscReal)(info.mx-1);
  hy     = 1.0/(PetscReal)(info.my-1);
  hxhy   = hx*hy;
  for(j = info.ys; j < info.ys+info.ym-1; j++) {
    for(i = info.xs; i < info.xs+info.xm-1; i++) {
      uLocal[0] = f[j][i];
      uLocal[1] = f[j][i+1];
      uLocal[2] = f[j+1][i+1];
      uLocal[3] = f[j+1][i];
      /* Lower element */
      for(q = 0; q < 4; q++) {
        phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
        phi[1] = quadPoints[q*2];
        phi[2] = quadPoints[q*2+1];
        u.u = uLocal[0].u*phi[0]+ uLocal[1].u*phi[1] + uLocal[3].u*phi[2];
        u.v = uLocal[0].v*phi[0]+ uLocal[1].v*phi[1] + uLocal[3].v*phi[2];
        u.p = uLocal[0].p*phi[0]+ uLocal[1].p*phi[1] + uLocal[3].p*phi[2];
        x = (quadPoints[q*2] + i)*hx;
        y = (quadPoints[q*2+1] + j)*hy;
        ierr = ExactSolution(x, y, &uExact);CHKERRQ(ierr);
        *error += hxhy*quadWeights[q]*((u.u - uExact.u)*(u.u - uExact.u) + (u.v - uExact.v)*(u.v - uExact.v) + (u.p - uExact.p)*(u.p - uExact.p));
      }
      /* Upper element */
      /*
        The affine map from the lower to the upper is

        / x_U \ = / -1  0 \ / x_L \ + / hx \
        \ y_U /   \  0 -1 / \ y_L /   \ hy /
       */
      for(q = 0; q < 4; q++) {
        phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
        phi[1] = quadPoints[q*2];
        phi[2] = quadPoints[q*2+1];
        u.u = uLocal[2].u*phi[0]+ uLocal[3].u*phi[1] + uLocal[1].u*phi[2];
        u.v = uLocal[2].v*phi[0]+ uLocal[3].v*phi[1] + uLocal[1].v*phi[2];
        u.p = uLocal[0].p*phi[0]+ uLocal[1].p*phi[1] + uLocal[3].p*phi[2];
        x = (1.0 - quadPoints[q*2] + i)*hx;
        y = (1.0 - quadPoints[q*2+1] + j)*hy;
        ierr = ExactSolution(x, y, &uExact);CHKERRQ(ierr);
        *error += hxhy*quadWeights[q]*((u.u - uExact.u)*(u.u - uExact.u) + (u.v - uExact.v)*(u.v - uExact.v) + (u.p - uExact.p)*(u.p - uExact.p));
      }
    }
  }

  ierr = DAVecRestoreArray(da, fLocalVec, (void **) &f);CHKERRQ(ierr);
  /* ierr = DALocalToGlobalBegin(da,xLocalVec,xVec);CHKERRQ(ierr); */
  /* ierr = DALocalToGlobalEnd(da,xLocalVec,xVec);CHKERRQ(ierr); */
  ierr = DARestoreLocalVector(da, &fLocalVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
