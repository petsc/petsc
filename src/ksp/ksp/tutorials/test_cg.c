
static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers

  Note:  The corresponding parallel example is ex23.c
*/
#include <petscksp.h>
#include "mmio_highlevel.h"
#include <sys/time.h>

int main(int argc, char **args)
{
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PC          pc;      /* preconditioner context */
  PetscReal   norm;    /* norm of solution error */
  PetscReal   residue;
  PetscReal   norm_b;
  PetscInt    i, n = 10, col[3], its;
  PetscInt    col_new[5];
  PetscMPIInt size;
  PetscScalar value[3];
  PetscScalar value_new[5];
  PetscLogDouble             time_start, time_mid1 = 0.0, time_mid2 = 0.0, time_end, time_avg, floprate;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  char *filename = args[1];
  int m, n1, nnzR, isSymmetric;
  mmio_info(&m, &n1, &nnzR, &isSymmetric, filename);
  int *RowPtr=(int *)malloc(sizeof(int)*(m+1));
  int *ColIdx=(int *)malloc(sizeof(int)*nnzR);
  VALUE_TYPE *Val=(VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*nnzR);
  mmio_data(RowPtr, ColIdx, Val, filename);
  printf("n=%d, nnz=%d\n",n1,nnzR);
  n=n1;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
   PetscCall(MatCreate(PETSC_COMM_SELF, &A));
   PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
   PetscCall(MatSetFromOptions(A));
   //PetscCall(MatSetUp(A));
  //   MatCreateAIJ(PETSC_COMM_SELF,n,n,n,n,0,NULL,0,NULL,A);

  /*
     Assemble matrix
  */
//   value[0] = -1.0;
//   value[1] = 2.0;
//   value[2] = -1.0;
// //   value_new[2] = -1.0;
// //   value_new[3] = 2.0;
// //   value_new[4] = -1.0;
//   for (i = 1; i < n - 1; i++) {
//     col[0] = i - 1;
//     col[1] = i;
//     col[2] = i + 1;
//     PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
//     //PetscCall(MatSetValues(A, 1, &i, 3, &col[0], &value_new[2], INSERT_VALUES));
//   }
//   i      = n - 1;
//   col[0] = n - 2;
//   col[1] = n - 1;
//   PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
//   i        = 0;
//   col[0]   = 0;
//   col[1]   = 1;
//   value[0] = 2.0;
//   value[1] = -1.0;
//   PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  for(int i=0;i<n;i++)
  {
      int len=RowPtr[i+1]-RowPtr[i];
      int head=RowPtr[i];
      PetscCall(MatSetValues(A, 1, &i, len, &ColIdx[head], &Val[head], INSERT_VALUES));
  }
  free(RowPtr);
  free(ColIdx);
  free(Val);
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));
//   PetscCall(VecSet(u, 0.0));
//   PetscCall(VecSet(b, 1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  //PetscCall(KSPGetPC(ksp, &pc));
  //PetscCall(PCSetType(pc, PCJACOBI));
  //1e-12 for JPCG
  //PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  //迭代次数两万次 for JPCG
  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system (warmup 3x, solve 10x)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#define WARMUP_CNT 3
#define SOLVE_CNT 10
  struct timeval t_start, t_stop;
  double solve_times[SOLVE_CNT];
  int k;

  /* Warmup */
  for (k = 0; k < WARMUP_CNT; k++) {
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
  }

  /* Timed solve runs */
  for (k = 0; k < SOLVE_CNT; k++) {
    PetscCall(VecSet(x, 0.0));
    gettimeofday(&t_start, NULL);
    PetscCall(KSPSolve(ksp, b, x));
    gettimeofday(&t_stop, NULL);
    solve_times[k] = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
    printf("  solve[%d]=%.3f ms\n", k, solve_times[k]);
  }

  /* Average time */
  double total_time = 0.0;
  for (k = 0; k < SOLVE_CNT; k++) total_time += solve_times[k];
  total_time /= SOLVE_CNT;

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_SELF));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(VecNorm(b, NORM_2, &norm_b));
  PetscCall(KSPGetResidualNorm(ksp,&residue));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Norm of error %g, Iterations %d \n", (double)norm, its));
  printf("avg solve time=%.3f ms (over %d runs)\n", total_time, SOLVE_CNT);
  char *s = (char *)malloc(sizeof(char) * 200);
  sprintf(s, "n=%d,nnzR=%d,petsc_norm=%e, iterations=%d, total_time=%lf,\n", n, nnzR, norm, its, total_time);
  //sprintf(s, "nnzR=%d,petsc_norm=%e,petsc_residule=%e,\n", nnzR,norm,residue/norm_b);
//   sprintf(s, "%d,%d,%e,\n", nnzR,n,residue);
  FILE *file1 = fopen("petsc_cg_a100.csv", "a");
  if (file1 == NULL)
  {
      printf("open error!\n");
      return 0;
  }
  fwrite(filename, strlen(filename), 1, file1);
  fwrite(",", strlen(","), 1, file1);
  fwrite(s, strlen(s), 1, file1);
  fclose(file1);
  /* check that KSP automatically handles the fact that the the new non-zero values in the matrix are propagated to the KSP solver */
//   PetscCall(MatShift(A, 2.0));
//   PetscCall(KSPSolve(ksp, b, x));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2_aijcusparse
      requires: cuda
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda

   test:
      suffix: 3
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3_aijcusparse
      requires: cuda
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda

   test:
      suffix: aijcusparse
      requires: cuda
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda
      output_file: output/ex1_1_aijcusparse.out

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_1
      nsize: 3
      filter: sed 's?ATOL?RTOL?g'
      args: -mpi_linear_solver_server -mpi_linear_solver_server_view -pc_type mpi -ksp_type preonly -mpi_ksp_monitor -mpi_ksp_converged_reason -mat_view -mpi_pc_type none -mpi_ksp_view -mpi_mat_view -pc_mpi_minimum_count_per_rank 5

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_2
      nsize: 3
      filter: sed 's?ATOL?RTOL?g'
      args: -mpi_linear_solver_server  -mpi_linear_solver_server_view -pc_type mpi -ksp_type preonly -mpi_ksp_monitor -mpi_ksp_converged_reason -mat_view -mpi_pc_type none -mpi_ksp_view

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_3
      nsize: 3
      filter: sed 's?ATOL?RTOL?g'
      args: -mpi_linear_solver_server  -mpi_linear_solver_server_view -pc_type mpi -ksp_type preonly -mpi_ksp_monitor -mpi_ksp_converged_reason -mat_view -mpi_pc_type none -mpi_ksp_view -mpi_mat_view -pc_mpi_always_use_server

   test:
      requires: !__float128
      suffix: minit
      args: -ksp_monitor -pc_type none -ksp_min_it 8

// TEST*/
//   suffix: minit
//       args: -ksp_monitor -pc_type none -ksp_min_it 8

// TEST*/
