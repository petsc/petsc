
static char help[] = "Bilinear elements on the unit square for Laplacian.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

/* Addendum: piggy-backing on this example to test KSPChebyshev methods */

#include <petscksp.h>

int FormElementStiffness(PetscReal H, PetscScalar *Ke)
{
  PetscFunctionBeginUser;
  Ke[0]  = H / 6.0;
  Ke[1]  = -.125 * H;
  Ke[2]  = H / 12.0;
  Ke[3]  = -.125 * H;
  Ke[4]  = -.125 * H;
  Ke[5]  = H / 6.0;
  Ke[6]  = -.125 * H;
  Ke[7]  = H / 12.0;
  Ke[8]  = H / 12.0;
  Ke[9]  = -.125 * H;
  Ke[10] = H / 6.0;
  Ke[11] = -.125 * H;
  Ke[12] = -.125 * H;
  Ke[13] = H / 12.0;
  Ke[14] = -.125 * H;
  Ke[15] = H / 6.0;
  PetscFunctionReturn(0);
}
int FormElementRhs(PetscReal x, PetscReal y, PetscReal H, PetscScalar *r)
{
  PetscFunctionBeginUser;
  r[0] = 0.;
  r[1] = 0.;
  r[2] = 0.;
  r[3] = 0.0;
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  Mat         C;
  PetscMPIInt rank, size;
  PetscInt    i, m = 5, N, start, end, M, its;
  PetscScalar val, Ke[16], r[4];
  PetscReal   x, y, h, norm;
  PetscInt    idx[4], count, *rows;
  Vec         u, ustar, b;
  KSP         ksp;
  PetscBool   viewkspest = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ksp_est_view", &viewkspest, NULL));
  N = (m + 1) * (m + 1); /* dimension of matrix */
  M = m * m;             /* number of elements */
  h = 1.0 / m;           /* mesh width */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Create stiffness matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  start = rank * (M / size) + ((M % size) < rank ? (M % size) : rank);
  end   = start + M / size + ((M % size) > rank);

  /* Assemble matrix */
  PetscCall(FormElementStiffness(h * h, Ke)); /* element stiffness for Laplacian */
  for (i = start; i < end; i++) {
    /* node numbers for the four corners of element */
    idx[0] = (m + 1) * (i / m) + (i % m);
    idx[1] = idx[0] + 1;
    idx[2] = idx[1] + m + 1;
    idx[3] = idx[2] - 1;
    PetscCall(MatSetValues(C, 4, idx, 4, idx, Ke, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  /* Create right-hand-side and solution vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Approx. Solution"));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "Right hand side"));
  PetscCall(VecDuplicate(b, &ustar));
  PetscCall(VecSet(u, 0.0));
  PetscCall(VecSet(b, 0.0));

  /* Assemble right-hand-side vector */
  for (i = start; i < end; i++) {
    /* location of lower left corner of element */
    x = h * (i % m);
    y = h * (i / m);
    /* node numbers for the four corners of element */
    idx[0] = (m + 1) * (i / m) + (i % m);
    idx[1] = idx[0] + 1;
    idx[2] = idx[1] + m + 1;
    idx[3] = idx[2] - 1;
    PetscCall(FormElementRhs(x, y, h * h, r));
    PetscCall(VecSetValues(b, 4, idx, r, ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* Modify matrix and right-hand-side for Dirichlet boundary conditions */
  PetscCall(PetscMalloc1(4 * m, &rows));
  for (i = 0; i < m + 1; i++) {
    rows[i]             = i;               /* bottom */
    rows[3 * m - 1 + i] = m * (m + 1) + i; /* top */
  }
  count = m + 1; /* left side */
  for (i = m + 1; i < m * (m + 1); i += m + 1) rows[count++] = i;

  count = 2 * m; /* left side */
  for (i = 2 * m + 1; i < m * (m + 1); i += m + 1) rows[count++] = i;
  for (i = 0; i < 4 * m; i++) {
    val = h * (rows[i] / (m + 1));
    PetscCall(VecSetValues(u, 1, &rows[i], &val, INSERT_VALUES));
    PetscCall(VecSetValues(b, 1, &rows[i], &val, INSERT_VALUES));
  }
  PetscCall(MatZeroRows(C, 4 * m, rows, 1.0, 0, 0));

  PetscCall(PetscFree(rows));
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  {
    Mat A;
    PetscCall(MatConvert(C, MATSAME, MAT_INITIAL_MATRIX, &A));
    PetscCall(MatDestroy(&C));
    PetscCall(MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &C));
    PetscCall(MatDestroy(&A));
  }

  /* Solve linear system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, b, u));

  if (viewkspest) {
    KSP kspest;

    PetscCall(KSPChebyshevEstEigGetKSP(ksp, &kspest));
    if (kspest) PetscCall(KSPView(kspest, PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Check error */
  PetscCall(VecGetOwnershipRange(ustar, &start, &end));
  for (i = start; i < end; i++) {
    val = h * (i / (m + 1));
    PetscCall(VecSetValues(ustar, 1, &i, &val, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(ustar));
  PetscCall(VecAssemblyEnd(ustar));
  PetscCall(VecAXPY(u, -1.0, ustar));
  PetscCall(VecNorm(u, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g Iterations %" PetscInt_FMT "\n", (double)(norm * h), its));

  /* Free work space */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ustar));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 2
      nsize: 2
      args: -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 2_kokkos
      nsize: 2
      args: -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always -mat_type aijkokkos -vec_type kokkos
      output_file: output/ex3_2.out
      requires: kokkos_kernels

    test:
      suffix: nocheby
      args: -ksp_est_view

    test:
      suffix: chebynoest
      args: -ksp_est_view -ksp_type chebyshev -ksp_chebyshev_eigenvalues 0.1,1.0

    test:
      suffix: chebyest
      args: -ksp_est_view -ksp_type chebyshev -ksp_chebyshev_esteig
      filter:  sed -e "s/Iterations 19/Iterations 20/g"

TEST*/
