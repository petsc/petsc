
static char help[] = "Tests sequential and parallel DMCreateMatrix(), MatMatMult() and MatPtAP()\n\
  -Mx <xg>, where <xg> = number of coarse grid points in the x-direction\n\
  -My <yg>, where <yg> = number of coarse grid points in the y-direction\n\
  -Mz <zg>, where <zg> = number of coarse grid points in the z-direction\n\
  -Npx <npx>, where <npx> = number of processors in the x-direction\n\
  -Npy <npy>, where <npy> = number of processors in the y-direction\n\
  -Npz <npz>, where <npz> = number of processors in the z-direction\n\n";

/*
    This test is modified from ~src/ksp/tests/ex19.c.
    Example of usage: mpiexec -n 3 ./ex96 -Mx 10 -My 10 -Mz 10
*/

#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application contexts */
typedef struct {
  PetscInt mx, my, mz;     /* number grid points in x, y and z direction */
  Vec      localX, localF; /* local vectors with ghost region */
  DM       da;
  Vec      x, b, r; /* global vectors */
  Mat      J;       /* Jacobian on grid */
} GridCtx;
typedef struct {
  GridCtx  fine;
  GridCtx  coarse;
  PetscInt ratio;
  Mat      Ii; /* interpolation from coarse to fine */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
int main(int argc, char **argv)
{
  AppCtx          user;
  PetscInt        Npx = PETSC_DECIDE, Npy = PETSC_DECIDE, Npz = PETSC_DECIDE;
  PetscMPIInt     size, rank;
  PetscInt        m, n, M, N, i, nrows;
  PetscScalar     one  = 1.0;
  PetscReal       fill = 2.0;
  Mat             A, A_tmp, P, C, C1, C2;
  PetscScalar    *array, none = -1.0, alpha;
  Vec             x, v1, v2, v3, v4;
  PetscReal       norm, norm_tmp, norm_tmp1, tol = 100. * PETSC_MACHINE_EPSILON;
  PetscRandom     rdm;
  PetscBool       Test_MatMatMult = PETSC_TRUE, Test_MatPtAP = PETSC_TRUE, Test_3D = PETSC_TRUE, flg;
  const PetscInt *ia, *ja;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  user.ratio     = 2;
  user.coarse.mx = 20;
  user.coarse.my = 20;
  user.coarse.mz = 20;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Mx", &user.coarse.mx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-My", &user.coarse.my, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Mz", &user.coarse.mz, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ratio", &user.ratio, NULL));

  if (user.coarse.mz) Test_3D = PETSC_TRUE;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Npx", &Npx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Npy", &Npy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Npz", &Npz, NULL));

  /* Set up distributed array for fine grid */
  if (!Test_3D) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, user.coarse.mx, user.coarse.my, Npx, Npy, 1, 1, NULL, NULL, &user.coarse.da));
  } else {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, user.coarse.mx, user.coarse.my, user.coarse.mz, Npx, Npy, Npz, 1, 1, NULL, NULL, NULL, &user.coarse.da));
  }
  PetscCall(DMSetFromOptions(user.coarse.da));
  PetscCall(DMSetUp(user.coarse.da));

  /* This makes sure the coarse DMDA has the same partition as the fine DMDA */
  PetscCall(DMRefine(user.coarse.da, PetscObjectComm((PetscObject)user.coarse.da), &user.fine.da));

  /* Test DMCreateMatrix()                                         */
  /*------------------------------------------------------------*/
  PetscCall(DMSetMatType(user.fine.da, MATAIJ));
  PetscCall(DMCreateMatrix(user.fine.da, &A));
  PetscCall(DMSetMatType(user.fine.da, MATBAIJ));
  PetscCall(DMCreateMatrix(user.fine.da, &C));

  PetscCall(MatConvert(C, MATAIJ, MAT_INITIAL_MATRIX, &A_tmp)); /* not work for mpisbaij matrix! */
  PetscCall(MatEqual(A, A_tmp, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "A != C");
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&A_tmp));

  /*------------------------------------------------------------*/

  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  /* if (rank == 0) printf("A %d, %d\n",M,N); */

  /* set val=one to A */
  if (size == 1) {
    PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(A, &array));
      for (i = 0; i < ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(A, &array));
    }
    PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &flg));
  } else {
    Mat AA, AB;
    PetscCall(MatMPIAIJGetSeqAIJ(A, &AA, &AB, NULL));
    PetscCall(MatGetRowIJ(AA, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(AA, &array));
      for (i = 0; i < ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(AA, &array));
    }
    PetscCall(MatRestoreRowIJ(AA, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &flg));
    PetscCall(MatGetRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &flg));
    if (flg) {
      PetscCall(MatSeqAIJGetArray(AB, &array));
      for (i = 0; i < ia[nrows]; i++) array[i] = one;
      PetscCall(MatSeqAIJRestoreArray(AB, &array));
    }
    PetscCall(MatRestoreRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nrows, &ia, &ja, &flg));
  }
  /* PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD)); */

  /* Create interpolation between the fine and coarse grids */
  PetscCall(DMCreateInterpolation(user.coarse.da, user.fine.da, &P, NULL));
  PetscCall(MatGetLocalSize(P, &m, &n));
  PetscCall(MatGetSize(P, &M, &N));
  /* if (rank == 0) printf("P %d, %d\n",M,N); */

  /* Create vectors v1 and v2 that are compatible with A */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &v1));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(VecSetSizes(v1, m, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(v1));
  PetscCall(VecDuplicate(v1, &v2));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  /* Test MatMatMult(): C = A*P */
  /*----------------------------*/
  if (Test_MatMatMult) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_tmp));
    PetscCall(MatMatMult(A_tmp, P, MAT_INITIAL_MATRIX, fill, &C));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha = 1.0;
    for (i = 0; i < 2; i++) {
      alpha -= 0.1;
      PetscCall(MatScale(A_tmp, alpha));
      PetscCall(MatMatMult(A_tmp, P, MAT_REUSE_MATRIX, fill, &C));
    }
    /* Free intermediate data structures created for reuse of C=Pt*A*P */
    PetscCall(MatProductClear(C));

    /* Test MatDuplicate()        */
    /*----------------------------*/
    PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &C1));
    PetscCall(MatDuplicate(C1, MAT_COPY_VALUES, &C2));
    PetscCall(MatDestroy(&C1));
    PetscCall(MatDestroy(&C2));

    /* Create vector x that is compatible with P */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(MatGetLocalSize(P, NULL, &n));
    PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(x));

    norm = 0.0;
    for (i = 0; i < 10; i++) {
      PetscCall(VecSetRandom(x, rdm));
      PetscCall(MatMult(P, x, v1));
      PetscCall(MatMult(A_tmp, v1, v2)); /* v2 = A*P*x */
      PetscCall(MatMult(C, x, v1));      /* v1 = C*x   */
      PetscCall(VecAXPY(v1, none, v2));
      PetscCall(VecNorm(v1, NORM_1, &norm_tmp));
      PetscCall(VecNorm(v2, NORM_1, &norm_tmp1));
      norm_tmp /= norm_tmp1;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol && rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatMatMult(), |v1 - v2|/|v2|: %g\n", (double)norm));

    PetscCall(VecDestroy(&x));
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&A_tmp));
  }

  /* Test P^T * A * P - MatPtAP() */
  /*------------------------------*/
  if (Test_MatPtAP) {
    PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, fill, &C));
    PetscCall(MatGetLocalSize(C, &m, &n));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha = 1.0;
    for (i = 0; i < 1; i++) {
      alpha -= 0.1;
      PetscCall(MatScale(A, alpha));
      PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, fill, &C));
    }

    /* Free intermediate data structures created for reuse of C=Pt*A*P */
    PetscCall(MatProductClear(C));

    /* Test MatDuplicate()        */
    /*----------------------------*/
    PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &C1));
    PetscCall(MatDuplicate(C1, MAT_COPY_VALUES, &C2));
    PetscCall(MatDestroy(&C1));
    PetscCall(MatDestroy(&C2));

    /* Create vector x that is compatible with P */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(MatGetLocalSize(P, &m, &n));
    PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(x));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &v3));
    PetscCall(VecSetSizes(v3, n, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(v3));
    PetscCall(VecDuplicate(v3, &v4));

    norm = 0.0;
    for (i = 0; i < 10; i++) {
      PetscCall(VecSetRandom(x, rdm));
      PetscCall(MatMult(P, x, v1));
      PetscCall(MatMult(A, v1, v2)); /* v2 = A*P*x */

      PetscCall(MatMultTranspose(P, v2, v3)); /* v3 = Pt*A*P*x */
      PetscCall(MatMult(C, x, v4));           /* v3 = C*x   */
      PetscCall(VecAXPY(v4, none, v3));
      PetscCall(VecNorm(v4, NORM_1, &norm_tmp));
      PetscCall(VecNorm(v3, NORM_1, &norm_tmp1));

      norm_tmp /= norm_tmp1;
      if (norm_tmp > norm) norm = norm_tmp;
    }
    if (norm >= tol && rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatPtAP(), |v3 - v4|/|v3|: %g\n", (double)norm));
    PetscCall(MatDestroy(&C));
    PetscCall(VecDestroy(&v3));
    PetscCall(VecDestroy(&v4));
    PetscCall(VecDestroy(&x));
  }

  /* Clean up */
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&v1));
  PetscCall(VecDestroy(&v2));
  PetscCall(DMDestroy(&user.fine.da));
  PetscCall(DMDestroy(&user.coarse.da));
  PetscCall(MatDestroy(&P));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -Mx 10 -My 5 -Mz 10
      output_file: output/ex96_1.out

   test:
      suffix: nonscalable
      nsize: 3
      args: -Mx 10 -My 5 -Mz 10
      output_file: output/ex96_1.out

   test:
      suffix: scalable
      nsize: 3
      args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable
      output_file: output/ex96_1.out

   test:
     suffix: seq_scalable
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm scalable -inner_offdiag_mat_product_algorithm scalable
     output_file: output/ex96_1.out

   test:
     suffix: seq_sorted
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm sorted -inner_offdiag_mat_product_algorithm sorted
     output_file: output/ex96_1.out

   test:
     suffix: seq_scalable_fast
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm scalable_fast -inner_offdiag_mat_product_algorithm scalable_fast
     output_file: output/ex96_1.out

   test:
     suffix: seq_heap
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm heap -inner_offdiag_mat_product_algorithm heap
     output_file: output/ex96_1.out

   test:
     suffix: seq_btheap
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm btheap -inner_offdiag_mat_product_algorithm btheap
     output_file: output/ex96_1.out

   test:
     suffix: seq_llcondensed
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm llcondensed -inner_offdiag_mat_product_algorithm llcondensed
     output_file: output/ex96_1.out

   test:
     suffix: seq_rowmerge
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via scalable -inner_diag_mat_product_algorithm rowmerge -inner_offdiag_mat_product_algorithm rowmerge
     output_file: output/ex96_1.out

   test:
     suffix: allatonce
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via allatonce
     output_file: output/ex96_1.out

   test:
     suffix: allatonce_merged
     nsize: 3
     args: -Mx 10 -My 5 -Mz 10 -matmatmult_via scalable -matptap_via allatonce_merged
     output_file: output/ex96_1.out

TEST*/
