
static char help[] = "Bratu nonlinear PDE in 3d.\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 3D rectangular\n\
domain, using distributed arrays (DMDAs) to partition the parallel grid.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\n";

/*T
   Concepts: SNES^parallel Bratu example
   Concepts: DMDA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation

            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1

    A finite difference approximation with the usual 7-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

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
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/
typedef struct {
  PetscReal param;             /* test problem parameter */
  DM        da;                /* distributed array data structure */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormFunctionLocal(SNES,Vec,Vec,void*);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormInitialGuess(AppCtx*,Vec);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  SNES           snes;                         /* nonlinear solver */
  Vec            x,r;                          /* solution, residual vectors */
  Mat            J = NULL;                            /* Jacobian matrix */
  AppCtx         user;                         /* user-defined work context */
  PetscInt       its;                          /* iterations for convergence */
  MatFDColoring  matfdcoloring = NULL;
  PetscBool      matrix_free = PETSC_FALSE,coloring = PETSC_FALSE, coloring_ds = PETSC_FALSE,local_coloring = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.,fnorm;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.param = 6.0;
  ierr       = PetscOptionsGetReal(NULL,NULL,"-par",&user.param,NULL);CHKERRQ(ierr);
  PetscCheckFalse(user.param >= bratu_lambda_max || user.param <= bratu_lambda_min,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Lambda is out of range");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&user.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.da);CHKERRQ(ierr);
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set function evaluation routine and vector
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method
     -fdcoloring : using finite differences with coloring to compute the Jacobian

     Note one can use -matfd_coloring wp or ds the only reason for the -fdcoloring_ds option
     below is to test the call to MatFDColoringSetType().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsGetBool(NULL,NULL,"-snes_mf",&matrix_free,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-fdcoloring",&coloring,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-fdcoloring_ds",&coloring_ds,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-fdcoloring_local",&local_coloring,NULL);CHKERRQ(ierr);
  if (!matrix_free) {
    ierr = DMSetMatType(user.da,MATAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(user.da,&J);CHKERRQ(ierr);
    if (coloring) {
      ISColoring iscoloring;
      if (!local_coloring) {
        ierr = DMCreateColoring(user.da,IS_COLORING_GLOBAL,&iscoloring);CHKERRQ(ierr);
        ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
        ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunction,&user);CHKERRQ(ierr);
      } else {
        ierr = DMCreateColoring(user.da,IS_COLORING_LOCAL,&iscoloring);CHKERRQ(ierr);
        ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
        ierr = MatFDColoringUseDM(J,matfdcoloring);CHKERRQ(ierr);
        ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunctionLocal,&user);CHKERRQ(ierr);
      }
      if (coloring_ds) {
        ierr = MatFDColoringSetType(matfdcoloring,MATMFFD_DS);CHKERRQ(ierr);
      }
      ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(J,iscoloring,matfdcoloring);CHKERRQ(ierr);
      ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,matfdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    } else {
      ierr = SNESSetJacobian(snes,J,J,FormJacobian,&user);CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Explicitly check norm of the residual of the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormFunction(snes,x,r,(void*)&user);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&fnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D fnorm %g\n",its,(double)fnorm);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscInt       i,j,k,Mx,My,Mz,xs,ys,zs,xm,ym,zm;
  PetscErrorCode ierr;
  PetscReal      lambda,temp1,hx,hy,hz,tempk,tempj;
  PetscScalar    ***x;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(user->da,PETSC_IGNORE,&Mx,&My,&Mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  lambda = user->param;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  hz     = 1.0/(PetscReal)(Mz-1);
  temp1  = lambda/(lambda + 1.0);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArray(user->da,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 3-dimensional DMDA):
       xs, ys, zs   - starting grid indices (no ghost points)
       xm, ym, zm   - widths of local grid (no ghost points)

  */
  ierr = DMDAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (k=zs; k<zs+zm; k++) {
    tempk = (PetscReal)(PetscMin(k,Mz-k-1))*hz;
    for (j=ys; j<ys+ym; j++) {
      tempj = PetscMin((PetscReal)(PetscMin(j,My-j-1))*hy,tempk);
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || k == 0 || i == Mx-1 || j == My-1 || k == Mz-1) {
          /* boundary conditions are all zero Dirichlet */
          x[k][j][i] = 0.0;
        } else {
          x[k][j][i] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,tempj));
        }
      }
    }
  }

  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArray(user->da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x) on a ghosted patch

   Input Parameters:
.  snes - the SNES context
.  localX - input vector, this contains the ghosted region needed
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector, this does not contain a ghosted region
 */
PetscErrorCode FormFunctionLocal(SNES snes,Vec localX,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,My,Mz,xs,ys,zs,xm,ym,zm;
  PetscReal      two = 2.0,lambda,hx,hy,hz,hxhzdhy,hyhzdhx,hxhydhz,sc;
  PetscScalar    u_north,u_south,u_east,u_west,u_up,u_down,u,u_xx,u_yy,u_zz,***x,***f;
  DM             da;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,&Mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  lambda  = user->param;
  hx      = 1.0/(PetscReal)(Mx-1);
  hy      = 1.0/(PetscReal)(My-1);
  hz      = 1.0/(PetscReal)(Mz-1);
  sc      = hx*hy*hz*lambda;
  hxhzdhy = hx*hz/hy;
  hyhzdhx = hy*hz/hx;
  hxhydhz = hx*hy/hz;

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || k == 0 || i == Mx-1 || j == My-1 || k == Mz-1) {
          f[k][j][i] = x[k][j][i];
        } else {
          u          = x[k][j][i];
          u_east     = x[k][j][i+1];
          u_west     = x[k][j][i-1];
          u_north    = x[k][j+1][i];
          u_south    = x[k][j-1][i];
          u_up       = x[k+1][j][i];
          u_down     = x[k-1][j][i];
          u_xx       = (-u_east + two*u - u_west)*hyhzdhx;
          u_yy       = (-u_north + two*u - u_south)*hxhzdhy;
          u_zz       = (-u_up + two*u - u_down)*hxhydhz;
          f[k][j][i] = u_xx + u_yy + u_zz - sc*PetscExpScalar(u);
        }
      }
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x) on the entire domain

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  DM             da;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = FormFunctionLocal(snes,localX,F,ptr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - optionally different preconditioning matrix

*/
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat J,Mat jac,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;  /* user-defined application context */
  Vec            localX;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,My,Mz;
  MatStencil     col[7],row;
  PetscInt       xs,ys,zs,xm,ym,zm;
  PetscScalar    lambda,v[7],hx,hy,hz,hxhzdhy,hyhzdhx,hxhydhz,sc,***x;
  DM             da;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,&Mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  lambda  = user->param;
  hx      = 1.0/(PetscReal)(Mx-1);
  hy      = 1.0/(PetscReal)(My-1);
  hz      = 1.0/(PetscReal)(Mz-1);
  sc      = hx*hy*hz*lambda;
  hxhzdhy = hx*hz/hy;
  hyhzdhx = hy*hz/hx;
  hxhydhz = hx*hy/hz;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointer to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localX,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

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
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.k = k; row.j = j; row.i = i;
        /* boundary points */
        if (i == 0 || j == 0 || k == 0|| i == Mx-1 || j == My-1 || k == Mz-1) {
          v[0] = 1.0;
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          /* interior grid points */
          v[0] = -hxhydhz; col[0].k=k-1;col[0].j=j;  col[0].i = i;
          v[1] = -hxhzdhy; col[1].k=k;  col[1].j=j-1;col[1].i = i;
          v[2] = -hyhzdhx; col[2].k=k;  col[2].j=j;  col[2].i = i-1;
          v[3] = 2.0*(hyhzdhx+hxhzdhy+hxhydhz)-sc*PetscExpScalar(x[k][j][i]);col[3].k=row.k;col[3].j=row.j;col[3].i = row.i;
          v[4] = -hyhzdhx; col[4].k=k;  col[4].j=j;  col[4].i = i+1;
          v[5] = -hxhzdhy; col[5].k=k;  col[5].j=j+1;col[5].i = i;
          v[6] = -hxhydhz; col[6].k=k+1;col[6].j=j;  col[6].i = i;
          ierr = MatSetValuesStencil(jac,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayRead(da,localX,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Normally since the matrix has already been assembled above; this
     would do nothing. But in the matrix free mode -snes_mf_operator
     this tells the "matrix-free" matrix that a new linear system solve
     is about to be done.
  */

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      nsize: 4
      args: -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3
      nsize: 4
      args: -fdcoloring -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3_ds
      nsize: 4
      args: -fdcoloring -fdcoloring_ds -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 4
      nsize: 4
      args: -fdcoloring_local -fdcoloring -ksp_monitor_short -da_refine 1
      requires: !single

   test:
      suffix: 5
      nsize: 4
      args: -fdcoloring_local -fdcoloring -ksp_monitor_short -da_refine 1 -snes_type newtontrdc
      requires: !single

TEST*/
