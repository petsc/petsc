
static char help[] = "Solves the Bratu equation in a 2D rectangular\n\
domain, using distributed arrays (DMDAs) to partition the parallel grid.\n\n";

/*T
   Concepts: SNES^parallel Bratu example
   Concepts: DMDA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    The Bratu equation is given by the partial differential equation

            -alpha*Laplacian u + lambda*e^u = f,  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A linear finite element approximation is used to discretize the boundary
    value problem on the two triangles which make up each rectangle in the DMDA
    to obtain a nonlinear system of equations.

  ------------------------------------------------------------------------- */

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscsys.h>
#include <petscbag.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
   PetscReal alpha;          /* parameter controlling linearity */
   PetscReal lambda;         /* parameter controlling nonlinearity */
} AppCtx;

static PetscScalar Kref[36] = { 0.5,  0.5, -0.5,  0,  0, -0.5,
                                0.5,  0.5, -0.5,  0,  0, -0.5,
                               -0.5, -0.5,  0.5,  0,  0,  0.5,
                                  0,    0,    0,  0,  0,    0,
                                  0,    0,    0,  0,  0,    0,
                               -0.5, -0.5,  0.5,  0,  0,  0.5};

/* These are */
static PetscScalar quadPoints[8] = {0.17855873, 0.15505103,
                                    0.07503111, 0.64494897,
                                    0.66639025, 0.15505103,
                                    0.28001992, 0.64494897};
static PetscScalar quadWeights[4] = {0.15902069,  0.09097931,  0.15902069,  0.09097931};

/*
   User-defined routines
*/
extern PetscErrorCode FormInitialGuess(DM,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar**,Mat,AppCtx*);
extern PetscErrorCode L_2Error(DM, Vec, double *, AppCtx *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DM                     da;
  SNES                   snes;                 /* nonlinear solver */
  AppCtx                *user;                 /* user-defined work context */
  PetscBag               bag;
  PetscInt               its;                  /* iterations for convergence */
  SNESConvergedReason    reason;
  PetscErrorCode         ierr;
  PetscReal              lambda_max = 6.81, lambda_min = 0.0, error;
  Vec                    x;

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
  if (user->lambda > lambda_max || user->lambda < lambda_min) SETERRQ3(PETSC_COMM_SELF,1,"Lambda %G is out of range [%G, %G]", user->lambda, lambda_min, lambda_max);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da, 0, "ooblek");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, (DM) da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set the discretization functions
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDASetLocalFunction(da,(DMDALocalFunction1)FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMDASetLocalJacobian(da,(DMDALocalFunction1)FormJacobianLocal);CHKERRQ(ierr);
  ierr = DMSetInitialGuess(da,FormInitialGuess);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D, %s\n",its,SNESConvergedReasons[reason]);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&x);CHKERRQ(ierr);
  ierr = L_2Error(da, x, &error, user);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_2 error in the solution: %G\n", error);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
PetscErrorCode ExactSolution(PetscReal x, PetscReal y, PetscScalar *u)
{
  PetscFunctionBegin;
  *u = x*x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(DM da,Vec X)
{
  AppCtx        *user;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      lambda,temp1,temp,hx,hy;
  PetscScalar    **x;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

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
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,My-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {

      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* boundary conditions are all zero Dirichlet */
        x[j][i] = 0.0;
      } else {
        x[j][i] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,temp));
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "constantResidual"
PetscErrorCode constantResidual(PetscReal lambda, PetscBool  isLower, int i, int j, PetscReal hx, PetscReal hy, PetscScalar r[])
{
  PetscScalar rLocal[3] = {0.0, 0.0, 0.0};
  PetscScalar phi[3] = {0.0, 0.0, 0.0};
  PetscReal   xI = i*hx, yI = j*hy, hxhy = hx*hy, x, y;
  PetscScalar res;
  PetscInt    q, k;

  PetscFunctionBegin;
  for (q = 0; q < 4; q++) {
    phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
    phi[1] = quadPoints[q*2];
    phi[2] = quadPoints[q*2+1];
    /* These are currently wrong */
    x      = xI + quadPoints[q*2]*hx;
    y      = yI + quadPoints[q*2+1]*hy;
    res    = quadWeights[q]*(2.0);
    for (k = 0; k < 3; k++) {
      rLocal[k] += phi[k]*res;
    }
  }
  for (k = 0; k < 3; k++) {
    printf("  constLocal[%d] = %g\n", k, lambda*hxhy*rLocal[k]);
    r[k] += lambda*hxhy*rLocal[k];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "nonlinearResidual"
PetscErrorCode nonlinearResidual(PetscReal lambda, PetscScalar u[], PetscScalar r[])
{
  PetscScalar rLocal[3] = {0.0, 0.0, 0.0};
  PetscScalar phi[3] = {0.0, 0.0, 0.0};
  PetscScalar res;
  PetscInt    q;

  PetscFunctionBegin;
  for (q = 0; q < 4; q++) {
    phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
    phi[1] = quadPoints[q*2];
    phi[2] = quadPoints[q*2+1];
    res    = quadWeights[q]*PetscExpScalar(u[0]*phi[0]+ u[1]*phi[1] + u[2]*phi[2]+ u[3]*phi[3]);
    rLocal[0] += phi[0]*res;
    rLocal[1] += phi[1]*res;
    rLocal[2] += phi[2]*res;
  }
  r[0] += lambda*rLocal[0];
  r[1] += lambda*rLocal[1];
  r[2] += lambda*rLocal[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).

       Process adiC(36): FormFunctionLocal

 */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  PetscScalar    uLocal[3];
  PetscScalar    rLocal[3];
  PetscScalar    G[4];
  PetscScalar    uExact;
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
  for (k = 0; k < 4; k++) {
    printf("G[%d] = %g\n", k, G[k]);
  }

  /* Zero the vector */
  ierr = PetscMemzero((void *) &(f[info->ys][info->xs]), info->xm*info->ym*sizeof(PetscScalar));CHKERRQ(ierr);
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
  for (j = info->ys; j < info->ys+info->ym-1; j++) {
    for (i = info->xs; i < info->xs+info->xm-1; i++) {
      /* Lower element */
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i];
      printf("Solution ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  uLocal[%d] = %g\n", k, uLocal[k]);
      }
      for (k = 0; k < 3; k++) {
        rLocal[k] = 0.0;
        for (l = 0; l < 3; l++) {
          rLocal[k] += (G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1])*uLocal[l];
        }
        rLocal[k] *= alpha;
      }
      printf("Laplacian ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  rLocal[%d] = %g\n", k, rLocal[k]);
      }
      ierr = constantResidual(1.0, PETSC_TRUE, i, j, hx, hy, rLocal);CHKERRQ(ierr);
      printf("Laplacian+Constant ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  rLocal[%d] = %g\n", k, rLocal[k]);
      }
      ierr = nonlinearResidual(0.0*sc, uLocal, rLocal);CHKERRQ(ierr);
      printf("Full nonlinear ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  rLocal[%d] = %g\n", k, rLocal[k]);
      }
      f[j][i]   += rLocal[0];
      f[j][i+1] += rLocal[1];
      f[j+1][i] += rLocal[2];
      /* Upper element */
      uLocal[0] = x[j+1][i+1];
      uLocal[1] = x[j+1][i];
      uLocal[2] = x[j][i+1];
      printf("Solution ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  uLocal[%d] = %g\n", k, uLocal[k]);
      }
      for (k = 0; k < 3; k++) {
        rLocal[k] = 0.0;
        for (l = 0; l < 3; l++) {
          rLocal[k] += (G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1])*uLocal[l];
        }
        rLocal[k] *= alpha;
      }
      printf("Laplacian ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  rLocal[%d] = %g\n", k, rLocal[k]);
      }
      ierr = constantResidual(1.0, PETSC_BOOL, i, j, hx, hy, rLocal);CHKERRQ(ierr);
      printf("Laplacian+Constant ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  rLocal[%d] = %g\n", k, rLocal[k]);
      }
      ierr = nonlinearResidual(0.0*sc, uLocal, rLocal);CHKERRQ(ierr);
      printf("Full nonlinear ElementVector for (%d, %d)\n", i, j);
      for (k = 0; k < 3; k++) {
        printf("  rLocal[%d] = %g\n", k, rLocal[k]);
      }
      f[j+1][i+1] += rLocal[0];
      f[j+1][i]   += rLocal[1];
      f[j][i+1]   += rLocal[2];
      /* Boundary conditions */
      if (i == 0 || j == 0) {
        ierr = ExactSolution(i*hx, j*hy, &uExact);CHKERRQ(ierr);
        f[j][i] = x[j][i] - uExact;
      }
      if ((i == info->mx-2) || (j == 0)) {
        ierr = ExactSolution((i+1)*hx, j*hy, &uExact);CHKERRQ(ierr);
        f[j][i+1] = x[j][i+1] - uExact;
      }
      if ((i == info->mx-2) || (j == info->my-2)) {
        ierr = ExactSolution((i+1)*hx, (j+1)*hy, &uExact);CHKERRQ(ierr);
        f[j+1][i+1] = x[j+1][i+1] - uExact;
      }
      if ((i == 0) || (j == info->my-2)) {
        ierr = ExactSolution(i*hx, (j+1)*hy, &uExact);CHKERRQ(ierr);
        f[j+1][i] = x[j+1][i] - uExact;
      }
    }
  }

  for (j = info->ys+info->ym-1; j >= info->ys; j--) {
    for (i = info->xs; i < info->xs+info->xm; i++) {
      printf("f[%d][%d] = %g ", j, i, f[j][i]);
    }
    printf("\n");
  }
  ierr = PetscLogFlops(68.0*(info->ym-1)*(info->xm-1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "nonlinearJacobian"
PetscErrorCode nonlinearJacobian(PetscReal lambda, PetscScalar u[], PetscScalar J[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
   FormJacobianLocal - Evaluates Jacobian matrix.
*/
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscScalar **x,Mat jac,AppCtx *user)
{
  PetscScalar    JLocal[16], uLocal[4];
  MatStencil     rows[4], cols[4], ident;
  PetscInt       lowerRow[3] = {0, 1, 3};
  PetscInt       upperRow[3] = {2, 3, 1};
  PetscInt       hasLower[3], hasUpper[3], localRows[4];
  PetscScalar    alpha,lambda,hx,hy,hxhy,detJInv,G[4],sc,one = 1.0;
  PetscInt       i,j,k,l,numRows;
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
  for (k = 0; k < 4; k++) {
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
  for (j=info->ys; j<info->ys+info->ym-1; j++) {
    for (i=info->xs; i<info->xs+info->xm-1; i++) {
      ierr = PetscMemzero(JLocal, 16 * sizeof(PetscScalar));CHKERRQ(ierr);
      numRows = 0;
      /* Lower element */
      uLocal[0] = x[j][i];
      uLocal[1] = x[j][i+1];
      uLocal[2] = x[j+1][i+1];
      uLocal[3] = x[j+1][i];
      /* i,j */
      if (i == 0 || j == 0) {
        hasLower[0] = 0;
        ident.i = i; ident.j = j;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasLower[0] = 1;
        localRows[0] = numRows;
        rows[numRows].i = i; rows[numRows].j = j;
        numRows++;
      }
      cols[0].i = i; cols[0].j = j;
      /* i+1,j */
      if ((i == info->mx-2) || (j == 0)) {
        hasLower[1] = 0;
        hasUpper[2] = 0;
        ident.i = i+1; ident.j = j;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        localRows[1] = numRows;
        hasLower[1] = 1;
        hasUpper[2] = 1;
        localRows[1] = numRows;
        rows[numRows].i = i+1; rows[numRows].j = j;
        numRows++;
      }
      cols[1].i = i+1; cols[1].j = j;
      /* i+1,j+1 */
      if ((i == info->mx-2) || (j == info->my-2)) {
        hasUpper[0] = 0;
        ident.i = i+1; ident.j = j+1;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasUpper[0] = 1;
        localRows[2] = numRows;
        rows[numRows].i = i+1; rows[numRows].j = j+1;
        numRows++;
      }
      cols[2].i = i+1; cols[2].j = j+1;
      /* i,j+1 */
      if ((i == 0) || (j == info->my-2)) {
        hasLower[2] = 0;
        hasUpper[1] = 0;
        ident.i = i; ident.j = j+1;
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatSetValuesStencil(jac,1,&ident,1,&ident,&one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      } else {
        hasLower[2] = 1;
        hasUpper[1] = 1;
        localRows[3] = numRows;
        rows[numRows].i = i; rows[numRows].j = j+1;
        numRows++;
      }
      cols[3].i = i; cols[3].j = j+1;

      /* Lower Element */
      for (k = 0; k < 3; k++) {
        if (!hasLower[k]) continue;
        for (l = 0; l < 3; l++) {
          JLocal[localRows[lowerRow[k]]*4 + lowerRow[l]] += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1]);
        }
      }
      /* Upper Element */
      for (k = 0; k < 3; k++) {
        if (!hasUpper[k]) continue;
        for (l = 0; l < 3; l++) {
          JLocal[localRows[upperRow[k]]*4 + upperRow[l]] += alpha*(G[0]*Kref[(k*2*3 + l)*2]+G[1]*Kref[(k*2*3 + l)*2+1]+G[2]*Kref[((k*2+1)*3 + l)*2]+G[3]*Kref[((k*2+1)*3 + l)*2+1]);
        }
      }

      ierr = nonlinearJacobian(-1.0*sc, uLocal, JLocal);CHKERRQ(ierr);
      printf("Element matrix for (%d, %d)\n", i, j);
      printf("   col  ");
      for (l = 0; l < 4; l++) {
        printf("(%d, %d) ", cols[l].i, cols[l].j);
      }
      printf("\n");
      for (k = 0; k < numRows; k++) {
        printf("row (%d, %d): ", rows[k].i, rows[k].j);
        for (l = 0; l < 4; l++) {
          printf("%8.6g ", JLocal[k*4 + l]);
        }
        printf("\n");
      }
      ierr = MatSetValuesStencil(jac,numRows,rows,4,cols,JLocal,ADD_VALUES);CHKERRQ(ierr);
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
PetscErrorCode L_2Error(DM da, Vec fVec, double *error, AppCtx *user)
{
  DMDALocalInfo  info;
  Vec            fLocalVec;
  PetscScalar    **f;
  PetscScalar    u, uExact, uLocal[4];
  PetscScalar    hx, hy, hxhy, x, y, phi[3];
  PetscInt       i, j, q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &fLocalVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,fVec, INSERT_VALUES, fLocalVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,fVec, INSERT_VALUES, fLocalVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, fLocalVec, &f);CHKERRQ(ierr);

  *error = 0.0;
  hx     = 1.0/(PetscReal)(info.mx-1);
  hy     = 1.0/(PetscReal)(info.my-1);
  hxhy   = hx*hy;
  for (j = info.ys; j < info.ys+info.ym-1; j++) {
    for (i = info.xs; i < info.xs+info.xm-1; i++) {
      uLocal[0] = f[j][i];
      uLocal[1] = f[j][i+1];
      uLocal[2] = f[j+1][i+1];
      uLocal[3] = f[j+1][i];
      /* Lower element */
      for (q = 0; q < 4; q++) {
        phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
        phi[1] = quadPoints[q*2];
        phi[2] = quadPoints[q*2+1];
        u = uLocal[0]*phi[0] + uLocal[1]*phi[1] + uLocal[3]*phi[2];
        x = (quadPoints[q*2] + i)*hx;
        y = (quadPoints[q*2+1] + j)*hy;
        ierr = ExactSolution(x, y, &uExact);CHKERRQ(ierr);
        *error += hxhy*quadWeights[q]*((u - uExact)*(u - uExact));
      }
      /* Upper element */
      /*
        The affine map from the lower to the upper is

        / x_U \ = / -1  0 \ / x_L \ + / hx \
        \ y_U /   \  0 -1 / \ y_L /   \ hy /
       */
      for (q = 0; q < 4; q++) {
        phi[0] = 1.0 - quadPoints[q*2] - quadPoints[q*2+1];
        phi[1] = quadPoints[q*2];
        phi[2] = quadPoints[q*2+1];
        u = uLocal[2]*phi[0] + uLocal[3]*phi[1] + uLocal[1]*phi[2];
        x = (1.0 - quadPoints[q*2] + i)*hx;
        y = (1.0 - quadPoints[q*2+1] + j)*hy;
        ierr = ExactSolution(x, y, &uExact);CHKERRQ(ierr);
        *error += hxhy*quadWeights[q]*((u - uExact)*(u - uExact));
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, fLocalVec, &f);CHKERRQ(ierr);
  /* ierr = DMLocalToGlobalBegin(da,xLocalVec,ADD_VALUES,xVec);CHKERRQ(ierr); */
  /* ierr = DMLocalToGlobalEnd(da,xLocalVec,ADD_VALUES,xVec);CHKERRQ(ierr); */
  ierr = DMRestoreLocalVector(da, &fLocalVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
