/* Program usage:  mpirun -np <procs> ex7 [-help] [all PETSc options] */

static char help[] = "Nonlinear PDE in 2d.\n\
We solve the Navier-Stokes equation in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\n";

/*T
   Concepts: SNES^parallel Lane-Emden example
   Concepts: DA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    The Lane-Emden equation is given by the partial differential equation
  
            Laplacian u - grad p = f,  0 < x,y < 1,
            div u                = 0
  
    with boundary conditions
   
            u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A bilinear finite element approximation is used to discretize the boundary
    value problem to obtain a nonlinear system of equations.

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include "petsc.h"
#include "petscbag.h"
#include "petscda.h"
#include "petscdmmg.h"
#include "petscsnes.h"
#include <PetscSimOutput.h>

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
  PetscReal u, v, p;
} Field;

static PetscScalar laplacian[16] = { 0.666667, -0.166667, -0.333333, -0.166667,
                                    -0.166667,  0.666667, -0.166667, -0.333333,
                                    -0.333333, -0.166667,  0.666667, -0.166667,
                                    -0.166667, -0.333333, -0.166667,  0.666667};

static PetscScalar gradient[32] = {-1/6,  1/6,  1/12, -1/12,
                                   -1/6,  1/6,  1/12, -1/12,
                                   -1/12, 1/12, 1/6,  -1/6,
                                   -1/12, 1/12, 1/6,  -1/6,
                                   -1/6,  -1/12, 1/12, 1/6,
                                   -1/12, -1/6, 1/6, 1/12,
                                   -1/12, -1/6, 1/6, 1/12,
                                   -1/6,  -1/12, 1/12, 1/6};

static PetscScalar divergence[32] = {-1/6,  1/6,  1/12, -1/12, -1/6,  -1/12, 1/12, 1/6,
                                     -1/6,  1/6,  1/12, -1/12, -1/12, -1/6,  1/6,  1/12,
                                     -1/12, 1/12, 1/6,  -1/6,  -1/12, -1/6,  1/6,  1/12,
                                     -1/12, 1/12, 1/6,  -1/6,  -1/6,  -1/12, 1/12, 1/6};




/* These are */
static PetscScalar quadPoints[8] = {0.211325, 0.211325,
                                    0.788675, 0.211325,
                                    0.788675, 0.788675,
                                    0.211325, 0.788675};
static PetscScalar quadWeights[4] = {0.25, 0.25, 0.25, 0.25};

/* 
   User-defined routines
*/
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
    SETERRQ3(1,"Lambda %g is out of range [%g, %g]", user->lambda, lambda_min, lambda_max);
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
  ierr = DASetFieldName(da, 0, "ooblek"); CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg, (DM) da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set the discretization functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGSetSNESLocal(dmmg, FormFunctionLocal, FormJacobianLocal, 0, 0);CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_2 error in the solution: %g\n", error);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Visualize the solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  {
    PetscViewer viewer;

    ierr = PetscWriteOutputInitialize(PETSC_COMM_WORLD, "ex4.data", &viewer);CHKERRQ(ierr);
    ierr = PetscWriteOutputBag(viewer, "params", bag);CHKERRQ(ierr);
    ierr = PetscWriteOutputVecDA(viewer, "u", DMMGGetx(dmmg), DMMGGetDA(dmmg));CHKERRQ(ierr);
    ierr = PetscWriteOutputFinalize(viewer);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-contours", &drawContours);CHKERRQ(ierr);
  if (drawContours) {
    ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); 
  }
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
#define __FUNCT__ "ExactSolution"
PetscErrorCode ExactSolution(PetscReal x, PetscReal y, Field *u)
{
  PetscFunctionBegin;
#if 1
  u->u = y + 3;
  u->v = x - 2;
#else
  u->u = x*x*y;
  u->v = -x*y*y;
  u->p = 2.0*x*y;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   dmmg - The DMMG context
   U - vector

   Output Parameter:
   U - vector
*/
PetscErrorCode FormInitialGuess(DMMG dmmg, Vec U)
{
  AppCtx        *user = (AppCtx *) dmmg->user;
  DA             da = (DA) dmmg->dm;
  Field        **u;
  PetscReal      lambda,temp1,temp,hx,hy,x,y;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  ierr = DAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(Mx-1);
  hy   = 1.0/(PetscReal)(My-1);
  lambda = user->lambda;
  if (lambda == 0.0) {
    temp1  = 1.0;
  } else {
    temp1  = lambda/(lambda + 1.0);
  }

  /* Compute initial guess over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,My-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      y = j*hy;
      printf("i: %d j: %d x: %g y: %g\n", i, j, x, y);
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* boundary conditions are all zero Dirichlet */
        u[j][i].u = 0.0; 
        u[j][i].v = 0.0; 
        u[j][i].p = 0.0; 
      } else {
        u[j][i].u = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
        u[j][i].v = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
      }
    }
  }
  ierr = DAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PrintVector(dmmg, U);CHKERRQ(ierr);
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
#define __FUNCT__ "gradientResidual"
PetscErrorCode gradientResidual(Field u[], Field r[]) {
  PetscFunctionBegin;
  r[0].u += (-2.0*u[0].p + 2.0*u[1].p + u[2].p - u[3].p)*0.08333333;
  r[0].v += (-2.0*u[0].p - u[1].p + u[2].p + 2.0*u[3].p)*0.08333333;
  r[1].u += (-2.0*u[0].p + 2.0*u[1].p + u[2].p - u[3].p)*0.08333333;
  r[1].v += (-u[0].p - 2.0*u[1].p + 2.0*u[2].p + u[3].p)*0.08333333;
  r[2].u += (-u[0].p + u[1].p + 2.0*u[2].p - 2.0*u[3].p)*0.08333333;
  r[2].v += (-u[0].p - 2.0*u[1].p + 2.0*u[2].p + u[3].p)*0.08333333;
  r[3].u += (-u[0].p + u[1].p + 2.0*u[2].p - 2.0*u[3].p)*0.08333333;
  r[3].v += (-2.0*u[0].p - u[1].p + u[2].p + 2.0*u[3].p)*0.08333333;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "divergenceResidual"
PetscErrorCode divergenceResidual(Field u[], Field r[]) {
  PetscFunctionBegin;
  r[0].p += (-2.0*u[0].u + 2.0*u[1].u + u[2].u - u[3].u - 2.0*u[0].v - u[1].v + u[2].v + 2.0*u[3].v)*0.08333333;
  r[1].p += (-2.0*u[0].u + 2.0*u[1].u + u[2].u - u[3].u - u[0].v - 2.0*u[1].v + 2.0*u[2].v + u[3].v)*0.08333333;
  r[2].p += (-u[0].u + u[1].u + 2.0*u[2].u - 2.0*u[3].u - u[0].v - 2.0*u[1].v + 2.0*u[2].v + u[3].v)*0.08333333;
  r[3].p += (-u[0].u + u[1].u + 2.0*u[2].u - 2.0*u[3].u - 2.0*u[0].v - u[1].v + u[2].v + 2.0*u[3].v)*0.08333333;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "constantResidual"
PetscErrorCode constantResidual(PetscReal lambda, int i, int j, PetscReal hx, PetscReal hy, Field r[])
{
  Field       rLocal[4] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  PetscScalar phi[4] = {0.0, 0.0, 0.0, 0.0};
  PetscReal   xI = i*hx, yI = j*hy, x, y;
  Field       res;
  PetscInt    q, k;

  PetscFunctionBegin;
  for(q = 0; q < 4; q++) {
    phi[0] = (1.0 - quadPoints[q*2])*(1.0 - quadPoints[q*2+1]);
    phi[1] =  quadPoints[q*2]       *(1.0 - quadPoints[q*2+1]);
    phi[2] =  quadPoints[q*2]       * quadPoints[q*2+1];
    phi[3] = (1.0 - quadPoints[q*2])* quadPoints[q*2+1];
    x      = xI + quadPoints[q*2];
    y      = yI + quadPoints[q*2+1];
    res.u    = lambda*quadWeights[q]*(0.0);
    res.v    = lambda*quadWeights[q]*(0.0);
    res.p    = lambda*quadWeights[q]*(0.0);
    for(k = 0; k < 4; k++) {
      rLocal[k].u += phi[k]*res.u;
      rLocal[k].v += phi[k]*res.v;
      rLocal[k].p += phi[k]*res.p;
    }
  }
  for(k = 0; k < 4; k++) {
    r[k].u += rLocal[k].u;
    r[k].v += rLocal[k].v;
    r[k].p += rLocal[k].p;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "nonlinearResidual"
PetscErrorCode nonlinearResidual(PetscReal lambda, PetscScalar u[], PetscScalar r[]) {
  PetscFunctionBegin;
  r[0] += lambda*(48.0*u[0]*u[0]*u[0] + 12.0*u[1]*u[1]*u[1] + 9.0*u[0]*u[0]*(4.0*u[1] + u[2] + 4.0*u[3]) + u[1]*u[1]*(9.0*u[2] + 6.0*u[3]) + u[1]*(6.0*u[2]*u[2] + 8.0*u[2]*u[3] + 6.0*u[3]*u[3])
           + 3.0*(u[2]*u[2]*u[2] + 2.0*u[2]*u[2]*u[3] + 3.0*u[2]*u[3]*u[3] + 4.0*u[3]*u[3]*u[3])
           + 2.0*u[0]*(12.0*u[1]*u[1] + u[1]*(6.0*u[2] + 9.0*u[3]) + 2.0*(u[2]*u[2] + 3.0*u[2]*u[3] + 6.0*u[3]*u[3])))/1200.0;
  r[1] += lambda*(12.0*u[0]*u[0]*u[0] + 48.0*u[1]*u[1]*u[1] + 9.0*u[1]*u[1]*(4.0*u[2] + u[3]) + 3.0*u[0]*u[0]*(8.0*u[1] + 2.0*u[2] + 3.0*u[3])
           + 4.0*u[1]*(6.0*u[2]*u[2] + 3.0*u[2]*u[3] + u[3]*u[3]) + 3.0*(4.0*u[2]*u[2]*u[2] + 3.0*u[2]*u[2]*u[3] + 2.0*u[2]*u[3]*u[3] + u[3]*u[3]*u[3])
           + 2.0*u[0]*((18.0*u[1]*u[1] + 3.0*u[2]*u[2] + 4.0*u[2]*u[3] + 3.0*u[3]*u[3]) + u[1]*(9.0*u[2] + 6.0*u[3])))/1200.0;
  r[2] += lambda*(3.0*u[0]*u[0]*u[0] + u[0]*u[0]*(6.0*u[1] + 4.0*u[2] + 6.0*u[3]) + u[0]*(9.0*u[1]*u[1] + 9.0*u[2]*u[2] + 12.0*u[2]*u[3] + 9.0*u[3]*u[3] + 4.0*u[1]*(3.0*u[2] + 2.0*u[3]))
           + 6.0*(2.0*u[1]*u[1]*u[1] + u[1]*u[1]*(4.0*u[2] + u[3]) + u[1]*(6.0*u[2]*u[2] + 3.0*u[2]*u[3] + u[3]*u[3]) + 2.0*(4.0*u[2]*u[2]*u[2] + 3.0*u[2]*u[2]*u[3] + 2.0*u[2]*u[3]*u[3] + u[3]*u[3]*u[3])))/1200.0;
  r[3] += lambda*(12.0*u[0]*u[0]*u[0] + 3.0*u[1]*u[1]*u[1] + u[1]*u[1]*(6.0*u[2] + 4.0*u[3]) + 3.0*u[0]*u[0]*(3.0*u[1] + 2.0*u[2] + 8.0*u[3])
           + 3.0*u[1]*(3.0*u[2]*u[2] + 4.0*u[2]*u[3] + 3.0*u[3]*u[3]) + 12.0*(u[2]*u[2]*u[2] + 2.0*u[2]*u[2]*u[3] + 3.0*u[2]*u[3]*u[3] + 4.0*u[3]*u[3]*u[3])
           + 2.0*u[0]*(3.0*u[1]*u[1] + u[1]*(4.0*u[2] + 6.0*u[3]) + 3.0*(u[2]*u[2] + 3.0*u[2]*u[3] + 6.0*u[3]*u[3])))/1200.0;
  PetscFunctionReturn(0);
}

PetscErrorCode nonlinearResidualBratu(PetscReal lambda, PetscScalar u[], PetscScalar r[]) {
  PetscScalar rLocal[4] = {0.0, 0.0, 0.0, 0.0};
  PetscScalar phi[4] = {0.0, 0.0, 0.0, 0.0};
  PetscScalar res;
  PetscInt q;

  PetscFunctionBegin;
  for(q = 0; q < 4; q++) {
    phi[0] = (1.0 - quadPoints[q*2])*(1.0 - quadPoints[q*2+1]);
    phi[1] =  quadPoints[q*2]       *(1.0 - quadPoints[q*2+1]);
    phi[2] =  quadPoints[q*2]       * quadPoints[q*2+1];
    phi[3] = (1.0 - quadPoints[q*2])* quadPoints[q*2+1];
    res    = quadWeights[q]*PetscExpScalar(u[0]*phi[0]+ u[1]*phi[1] + u[2]*phi[2]+ u[3]*phi[3]);
    rLocal[0] += phi[0]*res;
    rLocal[1] += phi[1]*res;  
    rLocal[2] += phi[2]*res;
    rLocal[3] += phi[3]*res;
  }
  r[0] += lambda*rLocal[0];
  r[1] += lambda*rLocal[1];
  r[2] += lambda*rLocal[2];
  r[3] += lambda*rLocal[3];
  PetscFunctionReturn(0);
}

PetscErrorCode nonlinearJacobian(PetscReal lambda, PetscScalar u[], PetscScalar J[]) {
  PetscFunctionBegin;
  J[0]  = lambda*(72.0*u[0]*u[0] + 12.0*u[1]*u[1] + 9.0*u[0]*(4.0*u[1] + u[2] + 4.0*u[3]) + u[1]*(6.0*u[2] + 9.0*u[3]) + 2.0*(u[2]*u[2] + 3.0*u[2]*u[3] + 6.0*u[3]*u[3]))/600.0;
  J[1]  = lambda*(18.0*u[0]*u[0] + 18.0*u[1]*u[1] + 3.0*u[2]*u[2] + 4.0*u[2]*u[3] + 3.0*u[3]*u[3] + 3.0*u[0]*(8.0*u[1] + 2.0*u[2] + 3.0*u[3]) + u[1]*(9.0*u[2] + 6.0*u[3]))/600.0;
  J[2]  = lambda*( 9.0*u[0]*u[0] +  9.0*u[1]*u[1] + 9.0*u[2]*u[2] + 12.0*u[2]*u[3] + 9.0*u[3]*u[3] + 4.0*u[1]*(3.0*u[2] + 2.0*u[3]) + 4.0*u[0]*(3.0*u[1] + 2.0*u[2] + 3.0*u[3]))/1200.0;
  J[3]  = lambda*(18.0*u[0]*u[0] +  3.0*u[1]*u[1] + u[1]*(4.0*u[2] + 6.0*u[3]) + 3.0*u[0]*(3.0*u[1] + 2.0*u[2] + 8.0*u[3]) + 3.0*(u[2]*u[2] + 3.0*u[2]*u[3] + 6.0*u[3]*u[3]))/600.0;

  J[4]  = lambda*(18.0*u[0]*u[0] + 18.0*u[1]*u[1] + 3.0*u[2]*u[2] + 4.0*u[2]*u[3] + 3.0*u[3]*u[3] + 3.0*u[0]*(8.0*u[1] + 2.0*u[2] + 3.0*u[3]) + u[1]*(9.0*u[2] + 6.0*u[3]))/600.0;
  J[5]  = lambda*(12.0*u[0]*u[0] + 72.0*u[1]*u[1] + 9.0*u[1]*(4.0*u[2] + u[3]) + u[0]*(36.0*u[1] + 9.0*u[2] + 6.0*u[3]) + 2.0*(6.0*u[2]*u[2] + 3.0*u[2]*u[3] + u[3]*u[3]))/600.0;
  J[6]  = lambda*( 3.0*u[0]*u[0] + u[0]*(9.0*u[1] + 6.0*u[2] + 4.0*u[3]) + 3.0*(6.0*u[1]*u[1] + 6.0*u[2]*u[2] + 3.0*u[2]*u[3] + u[3]*u[3] + 2.0*u[1]*(4.0*u[2] + u[3])))/600.0;
  J[7]  = lambda*( 9.0*u[0]*u[0] +  9.0*u[1]*u[1] + 9.0*u[2]*u[2] + 12.0*u[2]*u[3] + 9.0*u[3]*u[3] + 4.0*u[1]*(3.0*u[2] + 2.0*u[3]) + 4.0*u[0]*(3.0*u[1] + 2.0*u[2] + 3.0*u[3]))/1200.0;

  J[8]  = lambda*( 9.0*u[0]*u[0] +  9.0*u[1]*u[1] + 9.0*u[2]*u[2] + 12.0*u[2]*u[3] + 9.0*u[3]*u[3] + 4.0*u[1]*(3.0*u[2] + 2.0*u[3]) + 4.0*u[0]*(3.0*u[1] + 2.0*u[2] + 3.0*u[3]))/1200.0;
  J[9]  = lambda*( 3.0*u[0]*u[0] + u[0]*(9.0*u[1] + 6.0*u[2] + 4.0*u[3]) + 3.0*(6.0*u[1]*u[1] + 6.0*u[2]*u[2] + 3.0*u[2]*u[3] + u[3]*u[3] + 2.0*u[1]*(4.0*u[2] + u[3])))/600.0;
  J[10] = lambda*( 2.0*u[0]*u[0] + u[0]*(6.0*u[1] + 9.0*u[2] + 6.0*u[3]) + 3.0*(4.0*u[1]*u[1] + 3.0*u[1]*(4.0*u[2] + u[3]) + 4.0*(6.0*u[2]*u[2] + 3.0*u[2]*u[3] + u[3]*u[3])))/600.0;
  J[11] = lambda*( 3.0*u[0]*u[0] + u[0]*(4.0*u[1] + 6.0*u[2] + 9.0*u[3]) + 3.0*(u[1]*u[1] + 6.0*u[2]*u[2] + 8.0*u[2]*u[3] + 6.0*u[3]*u[3] + u[1]*(3.0*u[2] + 2.0*u[3])))/600.0;

  J[12] = lambda*(18.0*u[0]*u[0] +  3.0*u[1]*u[1] + u[1]*(4.0*u[2] + 6.0*u[3]) + 3.0*u[0]*(3.0*u[1] + 2.0*u[2] + 8.0*u[3]) + 3.0*(u[2]*u[2] + 3.0*u[2]*u[3] + 6.0*u[3]*u[3]))/600.0;
  J[13] = lambda*( 9.0*u[0]*u[0] +  9.0*u[1]*u[1] + 9.0*u[2]*u[2] + 12.0*u[2]*u[3] + 9.0*u[3]*u[3] + 4.0*u[1]*(3.0*u[2] + 2.0*u[3]) + 4.0*u[0]*(3.0*u[1] + 2.0*u[2] + 3.0*u[3]))/1200.0;
  J[14] = lambda*( 3.0*u[0]*u[0] + u[0]*(4.0*u[1] + 6.0*u[2] + 9.0*u[3]) + 3.0*(u[1]*u[1] + 6.0*u[2]*u[2] + 8.0*u[2]*u[3] + 6.0*u[3]*u[3] + u[1]*(3.0*u[2] + 2.0*u[3])))/600.0;
  J[15] = lambda*(12.0*u[0]*u[0] +  2.0*u[1]*u[1] + u[1]*(6.0*u[2] + 9.0*u[3]) + 12.0*(u[2]*u[2] + 3.0*u[2]*u[3] + 6.0*u[3]*u[3]) + u[0]*(6.0*u[1] + 9.0*(u[2] + 4.0*u[3])))/600.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Evaluates nonlinear function, F(x).

       Process adiC(36): FormFunctionLocal

 */
PetscErrorCode FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,AppCtx *user)
{
  Field          uLocal[4];
  Field          rLocal[4];
  Field          uExact;
  PetscReal      alpha,lambda,hx,hy,hxhy,sc;
  PetscInt       i,j,k,l;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  alpha  = user->alpha;
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = hx*hy*lambda;
  hxhy   = hx*hy; 

  /* Zero the vector */
  ierr = PetscMemzero((void *) &(f[info->xs][info->ys]), info->xm*info->ym*sizeof(Field));CHKERRQ(ierr);
  /* Compute function over the locally owned part of the grid. For each
     vertex (i,j), we consider the element below:

       3         2
     i,j+1 --- i+1,j
       |         |
       |         |
      i,j  --- i+1,j
       0         1

     and therefore we do not loop over the last vertex in each dimension.
  */
  for(j = info->ys; j < info->ys+info->ym-1; j++) {
    for(i = info->xs; i < info->xs+info->xm-1; i++) {
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i+1];
      uLocal[3] = x[j+1][i];
      printf("Solution ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 4; k++) {
        printf("  uLocal[%d] = (%g, %g, %g)\n", k, uLocal[k].u, uLocal[k].v, uLocal[k].p);
      }
      for(k = 0; k < 4; k++) {
        rLocal[k].u = 0.0;
        rLocal[k].v = 0.0;
        rLocal[k].p = 0.0;
        for(l = 0; l < 4; l++) {
          rLocal[k].u += laplacian[k*4 + l]*uLocal[l].u;
          rLocal[k].v += laplacian[k*4 + l]*uLocal[l].v;
        }
        rLocal[k].u *= -1.0*hxhy;
        rLocal[k].v *= -1.0*hxhy;
      }
      printf("Laplacian ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 4; k++) {
        printf("  rLocal[%d] = (%g, %g)\n", k, rLocal[k].u, rLocal[k].v);
      }
      /* ierr = gradientResidual(uLocal, rLocal);CHKERRQ(ierr); */
      printf("Gradient+Laplacian ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 4; k++) {
        printf("  rLocal[%d] = (%g, %g)\n", k, rLocal[k].u, rLocal[k].v);
      }
      /* ierr = divergenceResidual(uLocal, rLocal);CHKERRQ(ierr); */
      printf("Divergence ElementVector for (%d, %d)\n", i, j);
      for(k = 0; k < 4; k++) {
        printf("  rLocal[%d] = (%g)\n", k, rLocal[k].p);
      }
      ierr = constantResidual(-1.0, i, j, hx, hy, rLocal);CHKERRQ(ierr);
      /* ierr = nonlinearResidual(-1.0*sc, uLocal, rLocal);CHKERRQ(ierr); */
      f[j][i].u     += rLocal[0].u;
      f[j][i].v     += rLocal[0].v;
      f[j][i].p     += rLocal[0].p;
      f[j][i+1].u   += rLocal[1].u;
      f[j][i+1].v   += rLocal[1].v;
      f[j][i+1].p   += rLocal[1].p;
      f[j+1][i+1].u += rLocal[2].u;
      f[j+1][i+1].v += rLocal[2].v;
      f[j+1][i+1].p += rLocal[2].p;
      f[j+1][i].u   += rLocal[3].u;
      f[j+1][i].v   += rLocal[3].v;
      f[j+1][i].p   += rLocal[3].p;
      if (i == 0 || j == 0) {
        ierr = ExactSolution(i*hx, j*hy, &uExact);CHKERRQ(ierr);
        f[j][i].u = x[j][i].u - uExact.u;
        f[j][i].v = x[j][i].v - uExact.v;
      }
      if ((i == info->mx-2) || (j == 0)) {
        ierr = ExactSolution((i+1)*hx, j*hy, &uExact);CHKERRQ(ierr);
        f[j][i+1].u = x[j][i+1].u - uExact.u;
        f[j][i+1].v = x[j][i+1].v - uExact.v;
      }
      if ((i == info->mx-2) || (j == info->my-2)) {
        ierr = ExactSolution((i+1)*hx, (j+1)*hy, &uExact);CHKERRQ(ierr);
        f[j+1][i+1].u = x[j+1][i+1].u - uExact.u;
        f[j+1][i+1].v = x[j+1][i+1].v - uExact.v;
      }
      if ((i == 0) || (j == info->my-2)) {
        ierr = ExactSolution(i*hx, (j+1)*hy, &uExact);CHKERRQ(ierr);
        f[j+1][i].u = x[j+1][i].u - uExact.u;
        f[j+1][i].v = x[j+1][i].v - uExact.v;
      }
    }
  }

  for(j = info->ys+info->ym-1; j >= info->ys; j--) {
    for(i = info->xs; i < info->xs+info->xm; i++) {
      printf("f[%d][%d] = (%g, %g, %g) ", j, i, f[j][i].u, f[j][i].v, f[j][i].p);
    }
    printf("\n");
  }
  ierr = PetscLogFlops(68*(info->ym-1)*(info->xm-1));CHKERRQ(ierr);
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
  MatStencil     rows[12], cols[12], ident[2];
  PetscInt       rowActive[4];
  PetscInt       localRows[4];
  PetscScalar    alpha,lambda,hx,hy,hxhy,sc;
  PetscInt       i,j,k,l,row,numRows,numLocalRows;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  alpha  = user->alpha;
  lambda = user->lambda;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  sc     = hx*hy*lambda;
  hxhy   = hx*hy; 

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
  for (j=info->ys; j<info->ys+info->ym-1; j++) {
    for (i=info->xs; i<info->xs+info->xm-1; i++) {
      numRows = 0;
      numLocalRows = 0;
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i+1];
      uLocal[3] = x[j+1][i];
      ierr = PetscMemzero(JLocal, 144 * sizeof(PetscScalar));CHKERRQ(ierr);
      /* i,j */
      if (i == 0 || j == 0) {
        ident[0].i = i; ident[0].j = j; ident[0].c = 0;
        ident[1].i = i; ident[1].j = j; ident[1].c = 1;
        JLocal[0] = 1.0; JLocal[1] = 0.0;
        JLocal[2] = 0.0; JLocal[3] = 1.0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,2,ident,2,ident,JLocal,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        rowActive[0] = 0;
      } else {
        rowActive[0] = 1;
        localRows[numLocalRows++] = 0;
      }
      /* i+1,j */
      if ((i == info->mx-2) || (j == 0)) {
        ident[0].i = i+1; ident[0].j = j; ident[0].c = 0;
        ident[1].i = i+1; ident[1].j = j; ident[1].c = 1;
        JLocal[0] = 1.0; JLocal[1] = 0.0;
        JLocal[2] = 0.0; JLocal[3] = 1.0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,2,ident,2,ident,JLocal,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        rowActive[1] = 0;
      } else {
        rowActive[1] = 1;
        localRows[numLocalRows++] = 1;
      }
      /* i+1,j+1 */
      if ((i == info->mx-2) || (j == info->my-2)) {
        ident[0].i = i+1; ident[0].j = j+1; ident[0].c = 0;
        ident[1].i = i+1; ident[1].j = j+1; ident[1].c = 1;
        JLocal[0] = 1.0; JLocal[1] = 0.0;
        JLocal[2] = 0.0; JLocal[3] = 1.0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,2,ident,2,ident,JLocal,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        rowActive[2] = 0;
      } else {
        rowActive[2] = 1;
        localRows[numLocalRows++] = 2;
      }
      /* i,j+1 */
      if ((i == 0) || (j == info->my-2)) {
        ident[0].i = i; ident[0].j = j+1; ident[0].c = 0;
        ident[1].i = i; ident[1].j = j+1; ident[1].c = 1;
        JLocal[0] = 1.0; JLocal[1] = 0.0;
        JLocal[2] = 0.0; JLocal[3] = 1.0;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,2,ident,2,ident,JLocal,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        rowActive[3] = 0;
      } else {
        rowActive[3] = 1;
        localRows[numLocalRows++] = 3;
      }

      if (rowActive[0]) {
        rows[numRows].i = i;   rows[numRows].j = j;   rows[numRows].c = 0;
        numRows++;
      }
      if (rowActive[1]) {
        rows[numRows].i = i+1; rows[numRows].j = j;   rows[numRows].c = 0;
        numRows++;
      }
      if (rowActive[2]) {
        rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 0;
        numRows++;
      }
      if (rowActive[3]) {
        rows[numRows].i = i;   rows[numRows].j = j+1; rows[numRows].c = 0;
        numRows++;
      }
      cols[0].i = i;   cols[0].j = j;   cols[0].c = 0;
      cols[1].i = i+1; cols[1].j = j;   cols[1].c = 0;
      cols[2].i = i+1; cols[2].j = j+1; cols[2].c = 0;
      cols[3].i = i;   cols[3].j = j+1; cols[3].c = 0;
      if (rowActive[0]) {
        rows[numRows].i = i;   rows[numRows].j = j;   rows[numRows].c = 1;
        numRows++;
      }
      if (rowActive[1]) {
        rows[numRows].i = i+1; rows[numRows].j = j;   rows[numRows].c = 1;
        numRows++;
      }
      if (rowActive[2]) {
        rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 1;
        numRows++;
      }
      if (rowActive[3]) {
        rows[numRows].i = i;   rows[numRows].j = j+1; rows[numRows].c = 1;
        numRows++;
      }
      cols[4].i = i;   cols[4].j = j;   cols[4].c = 1;
      cols[5].i = i+1; cols[5].j = j;   cols[5].c = 1;
      cols[6].i = i+1; cols[6].j = j+1; cols[6].c = 1;
      cols[7].i = i;   cols[7].j = j+1; cols[7].c = 1;
      rows[numRows].i = i;   rows[numRows].j = j;   rows[numRows].c = 2;
      numRows++;
      rows[numRows].i = i+1; rows[numRows].j = j;   rows[numRows].c = 2;
      numRows++;
      rows[numRows].i = i+1; rows[numRows].j = j+1; rows[numRows].c = 2;
      numRows++;
      rows[numRows].i = i;   rows[numRows].j = j+1; rows[numRows].c = 2;
      numRows++;
      cols[8].i = i;    cols[8].j = j;    cols[8].c = 2;
      cols[9].i = i+1;  cols[9].j = j;    cols[9].c = 2;
      cols[10].i = i+1; cols[10].j = j+1; cols[10].c = 2;
      cols[11].i = i;   cols[11].j = j+1; cols[11].c = 2;

      row = 0;
      for(k = 0; k < numLocalRows; k++) {
        /* u-u block */
        for(l = 0; l < 4; l++) {
          JLocal[row*12 + l] = -hxhy*laplacian[localRows[k]*4 + l];
        }
#if 0
        /* u-p block */
        for(l = 0; l < 4; l++) {
          JLocal[row*12 + l + 8] = -hxhy*gradient[localRows[k]*4 + l];
        }
#endif
        row++;
      }
      for(k = 0; k < numLocalRows; k++) {
        /* v-v block */
        for(l = 0; l < 4; l++) {
          JLocal[row*12 + l + 4] = -hxhy*laplacian[localRows[k]*4 + l];
        }
#if 0
        /* v-p block */
        for(l = 0; l < 4; l++) {
          JLocal[row*12 + l + 8] = -hxhy*gradient[(localRows[k] + 4)*4 + l];
        }
#endif
        row++;
      }
#if 0
      for(k = 0; k < 4; k++) {
        /* p-(u,v) block */
        for(l = 0; l < 8; l++) {
          JLocal[row*12 + l] = -hxhy*divergence[k*4 + l];
        }
        row++;
      }
#endif
      printf("Element matrix for (%d, %d)\n", i, j);
      printf("   col  ");
      for(l = 0; l < 12; l++) {
        printf("(%d, %d, %d) ", cols[l].i, cols[l].j, cols[l].c);
      }
      printf("\n");
      for(k = 0; k < numRows; k++) {
        printf("row (%d, %d, %d): ", rows[k].i, rows[k].j, rows[k].c);
        for(l = 0; l < 12; l++) {
          printf("%8.6g ", JLocal[k*12 + l]);
        }
        printf("\n");
      }
      ierr = MatSetValuesStencil(jac,numRows,rows,12,cols,JLocal,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Integrate the L_2 error of our solution over each face
*/
PetscErrorCode L_2Error(DA da, Vec fVec, double *error, AppCtx *user)
{
  DALocalInfo info;
  Vec fLocalVec;
  Field **f;
  Field u, uExact, uLocal[4];
  PetscScalar hx, hy, hxhy, x, y, phi[4];
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
  for (j=info.ys; j<info.ys+info.ym-1; j++) {
    for (i=info.xs; i<info.xs+info.xm-1; i++) {
      uLocal[0] = f[j][i];
      uLocal[1] = f[j][i+1];
      uLocal[2] = f[j+1][i+1];
      uLocal[3] = f[j+1][i];
      for(q = 0; q < 4; q++) {
        phi[0] = (1.0 - quadPoints[q*2])*(1.0 - quadPoints[q*2+1]);
        phi[1] =  quadPoints[q*2]       *(1.0 - quadPoints[q*2+1]);
        phi[2] =  quadPoints[q*2]       * quadPoints[q*2+1];
        phi[3] = (1.0 - quadPoints[q*2])* quadPoints[q*2+1];
        u.u = uLocal[0].u*phi[0]+ uLocal[1].u*phi[1] + uLocal[2].u*phi[2]+ uLocal[3].u*phi[3];
        u.v = uLocal[0].v*phi[0]+ uLocal[1].v*phi[1] + uLocal[2].v*phi[2]+ uLocal[3].v*phi[3];
        u.p = uLocal[0].p*phi[0]+ uLocal[1].p*phi[1] + uLocal[2].p*phi[2]+ uLocal[3].p*phi[3];
        x = (quadPoints[q*2] + i)*hx;
        y = (quadPoints[q*2+1] + j)*hy;
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
