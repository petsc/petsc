
static char help[] = "Solves a linear system in parallel with KSP and DM.\n\
Compare this to ex2 which solves the same problem without a DM.\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Concepts: KSP^Laplacian, 2d
   Concepts: DM^using distributed arrays;
   Concepts: Laplacian, 2d
   Processors: n
T*/

/*
  Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

int main(int argc,char **argv)
{
  DM             da;            /* distributed array */
  Vec            x,b,u;         /* approx solution, RHS, exact solution */
  Mat            A;             /* linear system matrix */
  KSP            ksp;           /* linear solver context */
  PetscRandom    rctx;          /* random number generator context */
  PetscReal      norm;          /* norm of solution error */
  PetscInt       i,j,its;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif
  DMDALocalInfo  info;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /*
     Create distributed array to handle parallel distribution.
     The problem size will default to 8 by 7, but this can be
     changed using -da_grid_x M -da_grid_y N
  */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,8,7,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix preallocated according to the DMDA, format AIJ by default.
     To use symmetric storage, run with -dm_mat_type sbaij -mat_ignore_lower_triangular
  */
  CHKERRQ(DMCreateMatrix(da,&A));

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Rows and columns are specified by the stencil
      - Entries are normalized for a domain [0,1]x[0,1]
   */
  CHKERRQ(PetscLogStageRegister("Assembly", &stage));
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      PetscReal   hx  = 1./info.mx,hy = 1./info.my;
      MatStencil  row = {0},col[5] = {{0}};
      PetscScalar v[5];
      PetscInt    ncols = 0;
      row.j        = j; row.i = i;
      col[ncols].j = j; col[ncols].i = i; v[ncols++] = 2*(hx/hy + hy/hx);
      /* boundaries */
      if (i>0)         {col[ncols].j = j;   col[ncols].i = i-1; v[ncols++] = -hy/hx;}
      if (i<info.mx-1) {col[ncols].j = j;   col[ncols].i = i+1; v[ncols++] = -hy/hx;}
      if (j>0)         {col[ncols].j = j-1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      if (j<info.my-1) {col[ncols].j = j+1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      CHKERRQ(MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES));
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogStagePop());

  /*
     Create parallel vectors compatible with the DMDA.
  */
  CHKERRQ(DMCreateGlobalVector(da,&u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(u,&x));

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-random_exact_sol",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    CHKERRQ(PetscRandomSetFromOptions(rctx));
    CHKERRQ(VecSetRandom(u,rctx));
    CHKERRQ(PetscRandomDestroy(&rctx));
  } else {
    CHKERRQ(VecSet(u,1.));
  }
  CHKERRQ(MatMult(A,u,b));

  /*
     View the exact solution vector if desired
  */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL));
  if (flg) CHKERRQ(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  CHKERRQ(KSPSetOperators(ksp,A,A));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  CHKERRQ(VecAXPY(x,-1.,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(DMDestroy(&da));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      requires: cuda
      args: -dm_mat_type aijcusparse -dm_vec_type cuda -random_exact_sol
      output_file: output/ex46_aijcusparse.out

      test:
        suffix: aijcusparse
        args: -use_gpu_aware_mpi 0
      test:
        suffix: aijcusparse_mpi_gpu_aware

   testset:
      requires: cuda
      args: -dm_mat_type aijcusparse -dm_vec_type cuda -random_exact_sol -pc_type ilu -pc_factor_mat_solver_type cusparse
      output_file: output/ex46_aijcusparse_2.out
      test:
        suffix: aijcusparse_2
        args: -use_gpu_aware_mpi 0
      test:
        suffix: aijcusparse_2_mpi_gpu_aware

TEST*/
