
static char help[] = "Krylov methods to solve u''  = f in parallel with periodic boundary conditions,\n\
                      with a singular, inconsistent system.\n\n";

/*T
   Concepts: KSP^basic parallel example
   Concepts: periodic boundary conditions
   Processors: n
T*/

/*

   This tests solving singular inconsistent systems with GMRES

   Default: Solves a symmetric system
   -symmetric false: Solves a non-symmetric system where nullspace(A) != nullspace(A')

  -ksp_pc_side left or right

   See the KSPSolve() for a discussion of when right preconditioning with nullspace(A) != nullspace(A') can fail to produce the
   norm minimizing solution.

   Note that though this example does solve the system with right preconditioning and nullspace(A) != nullspace(A') it does not produce the
   norm minimizing solution, that is the computed solution is not orthogonal to the nullspace(A).

   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscksp.h" so that we can use KSP solvers.  Note that this
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
#include <petsc/private/kspimpl.h>

PetscBool symmetric = PETSC_TRUE;

PetscErrorCode FormMatrix(Mat,void*);
PetscErrorCode FormRightHandSide(Vec,void*);

int main(int argc,char **argv)
{
  KSP            ksp;
  Mat            J;
  DM             da;
  Vec            x,r;              /* vectors */
  PetscErrorCode ierr;
  PetscInt       M = 10;
  MatNullSpace   constants,nconstants;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric",&symmetric,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create linear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,M,1,2,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  ierr = FormRightHandSide(r,da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure;
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&constants);CHKERRQ(ierr);
  if (symmetric) {
    ierr = MatSetOption(J,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    Vec         n;
    PetscInt    zero = 0;
    PetscScalar zeros = 0.0;
    ierr = VecDuplicate(x,&n);CHKERRQ(ierr);
    /* the nullspace(A') is the constant vector but with a zero in the very first entry; hence nullspace(A') != nullspace(A) */
    ierr = VecSet(n,1.0);CHKERRQ(ierr);
    ierr = VecSetValues(n,1,&zero,&zeros,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(n);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(n);CHKERRQ(ierr);
    ierr = VecNormalize(n,NULL);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,1,&n,&nconstants);CHKERRQ(ierr);
    ierr = MatSetTransposeNullSpace(J,nconstants);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nconstants);CHKERRQ(ierr);
    ierr = VecDestroy(&n);CHKERRQ(ierr);
  }
  ierr = MatSetNullSpace(J,constants);CHKERRQ(ierr);
  ierr = FormMatrix(J,da);CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp,J,J);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,r,x);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(ksp,r,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&constants);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*

     This intentionally includes something in the right hand side that is not in the range of the matrix A.
     Since MatSetNullSpace() is called and the matrix is symmetric; the Krylov method automatically removes this
     portion of the right hand side before solving the linear system.
*/
PetscErrorCode FormRightHandSide(Vec f,void *ctx)
{
  DM             da    = (DM) ctx;
  PetscScalar    *ff;
  PetscErrorCode ierr;
  PetscInt       i,M,xs,xm;
  PetscReal      h;

  PetscFunctionBeginUser;
  ierr = DMDAVecGetArray(da,f,&ff);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over locally owned part of the grid
     Note the [i-1] and [i+1] will automatically access the ghost points from other processes or the periodic points.
  */
  h = 1.0/M;
  for (i=xs; i<xs+xm; i++) ff[i] = - PetscSinReal(2.0*PETSC_PI*i*h) + 1.0;

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,f,&ff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
PetscErrorCode FormMatrix(Mat jac,void *ctx)
{
  PetscScalar    A[3];
  PetscErrorCode ierr;
  PetscInt       i,M,xs,xm;
  DM             da = (DM) ctx;
  MatStencil     row,cols[3];
  PetscReal      h;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
    Get range of locally owned matrix
  */
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = MatZeroEntries(jac);CHKERRQ(ierr);
  h = 1.0/M;
  /* because of periodic boundary conditions we can simply loop over all local nodes and access to the left and right */
  if (symmetric) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i;
      cols[0].i = i - 1;
      cols[1].i = i;
      cols[2].i = i + 1;
      A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
      ierr = MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES);CHKERRQ(ierr);
    }
  } else {
    MatStencil  *acols;
    PetscScalar *avals;

    /* only works for one process */
    ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    row.i = 0;
    ierr = PetscMalloc1(M,&acols);CHKERRQ(ierr);
    ierr = PetscMalloc1(M,&avals);CHKERRQ(ierr);
    for (i=0; i<M; i++) {
      acols[i].i = i;
      avals[i]  = (i % 2) ? 1 : -1;
    }
    ierr = MatSetValuesStencil(jac,1,&row,M,acols,avals,ADD_VALUES);CHKERRQ(ierr);
    ierr = PetscFree(acols);CHKERRQ(ierr);
    ierr = PetscFree(avals);CHKERRQ(ierr);
    row.i = 1;
    cols[0].i = - 1;
    cols[1].i = 1;
    cols[2].i = 1 + 1;
    A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
    ierr = MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES);CHKERRQ(ierr);
    for (i=2; i<xs+xm-1; i++) {
      row.i = i;
      cols[0].i = i - 1;
      cols[1].i = i;
      cols[2].i = i + 1;
      A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
      ierr = MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES);CHKERRQ(ierr);
    }
    row.i = M - 1 ;
    cols[0].i = M-2;
    cols[1].i = M-1;
    cols[2].i = M+1;
    A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
    ierr = MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: nonsymmetric_left
      args: -symmetric false -ksp_view -ksp_converged_reason -pc_type jacobi -mat_no_inode -ksp_monitor_true_residual -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side left
      filter: sed 's/ATOL/RTOL/g'
      requires: !single

   test:
      suffix: nonsymmetric_right
      args: -symmetric false -ksp_view -ksp_converged_reason -pc_type jacobi -mat_no_inode -ksp_monitor_true_residual -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side right
      filter: sed 's/ATOL/RTOL/g'
      requires: !single

   test:
      suffix: symmetric_left
      args: -ksp_view -ksp_converged_reason -pc_type sor -mat_no_inode -ksp_monitor_true_residual -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side left
      requires: !single

   test:
      suffix: symmetric_right
      args: -ksp_view -ksp_converged_reason -pc_type sor -mat_no_inode -ksp_monitor_true_residual -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side right
      filter: sed 's/ATOL/RTOL/g'
      requires: !single

   test:
      suffix: transpose_asm
      args: -symmetric false -ksp_monitor -ksp_view -pc_type asm -sub_pc_type lu -sub_pc_factor_zeropivot 1.e-33 -ksp_converged_reason
      filter: sed 's/ATOL/RTOL/g'

TEST*/
