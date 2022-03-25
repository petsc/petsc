
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
  PetscInt       M = 10;
  MatNullSpace   constants,nconstants;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symmetric",&symmetric,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create linear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,M,1,2,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(VecDuplicate(x,&r));

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  PetscCall(FormRightHandSide(r,da));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure;
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateMatrix(da,&J));
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&constants));
  if (symmetric) {
    PetscCall(MatSetOption(J,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
    PetscCall(MatSetOption(J,MAT_SYMMETRIC,PETSC_TRUE));
  } else {
    Vec         n;
    PetscInt    zero = 0;
    PetscScalar zeros = 0.0;
    PetscCall(VecDuplicate(x,&n));
    /* the nullspace(A') is the constant vector but with a zero in the very first entry; hence nullspace(A') != nullspace(A) */
    PetscCall(VecSet(n,1.0));
    PetscCall(VecSetValues(n,1,&zero,&zeros,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(n));
    PetscCall(VecAssemblyEnd(n));
    PetscCall(VecNormalize(n,NULL));
    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,1,&n,&nconstants));
    PetscCall(MatSetTransposeNullSpace(J,nconstants));
    PetscCall(MatNullSpaceDestroy(&nconstants));
    PetscCall(VecDestroy(&n));
  }
  PetscCall(MatSetNullSpace(J,constants));
  PetscCall(FormMatrix(J,da));

  PetscCall(KSPSetOperators(ksp,J,J));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,r,x));
  PetscCall(KSPSolveTranspose(ksp,r,x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(MatNullSpaceDestroy(&constants));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
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
  PetscInt       i,M,xs,xm;
  PetscReal      h;

  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArray(da,f,&ff));

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  PetscCall(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  /*
     Compute function over locally owned part of the grid
     Note the [i-1] and [i+1] will automatically access the ghost points from other processes or the periodic points.
  */
  h = 1.0/M;
  for (i=xs; i<xs+xm; i++) ff[i] = - PetscSinReal(2.0*PETSC_PI*i*h) + 1.0;

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArray(da,f,&ff));
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
PetscErrorCode FormMatrix(Mat jac,void *ctx)
{
  PetscScalar    A[3];
  PetscInt       i,M,xs,xm;
  DM             da = (DM) ctx;
  MatStencil     row,cols[3];
  PetscReal      h;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /*
    Get range of locally owned matrix
  */
  PetscCall(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  PetscCall(MatZeroEntries(jac));
  h = 1.0/M;
  /* because of periodic boundary conditions we can simply loop over all local nodes and access to the left and right */
  if (symmetric) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i;
      cols[0].i = i - 1;
      cols[1].i = i;
      cols[2].i = i + 1;
      A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
      PetscCall(MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES));
    }
  } else {
    MatStencil  *acols;
    PetscScalar *avals;

    /* only works for one process */
    PetscCall(MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
    row.i = 0;
    PetscCall(PetscMalloc1(M,&acols));
    PetscCall(PetscMalloc1(M,&avals));
    for (i=0; i<M; i++) {
      acols[i].i = i;
      avals[i]  = (i % 2) ? 1 : -1;
    }
    PetscCall(MatSetValuesStencil(jac,1,&row,M,acols,avals,ADD_VALUES));
    PetscCall(PetscFree(acols));
    PetscCall(PetscFree(avals));
    row.i = 1;
    cols[0].i = - 1;
    cols[1].i = 1;
    cols[2].i = 1 + 1;
    A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
    PetscCall(MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES));
    for (i=2; i<xs+xm-1; i++) {
      row.i = i;
      cols[0].i = i - 1;
      cols[1].i = i;
      cols[2].i = i + 1;
      A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
      PetscCall(MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES));
    }
    row.i = M - 1 ;
    cols[0].i = M-2;
    cols[1].i = M-1;
    cols[2].i = M+1;
    A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
    PetscCall(MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
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
