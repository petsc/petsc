/*
Poisson in 3D. Modeled by the PDE:

  - delta u(x,y,z) = f(x,y,z)

With exact solution:

   u(x,y,z) = 1.0

Exampe usage:

  Run on GPU (requires respective backends installed):
    ./bench_kspsolve -vec_type cuda   -mat_type aijcusparse
    ./bench_kspsolve -vec_type hip    -mat_type aijhipsparse
    ./bench_kspsolve -vec_type kokkos -mat_type aijkokkos

  Test only MatMult:
    ./bench_kspsolve -matmult

  Test MatMult over 1000 iterations:
    ./bench_kspsolve -matmult -its 1000

  Change size of problem (e.g., use a 128x128x128 grid):
    ./bench_kspsolve -n 128
*/
static char help[] = "Solves 3D Laplacian with 27-point finite difference stencil.\n";

#include <petscksp.h>

typedef struct {
  PetscMPIInt rank, size;
  PetscInt    n;        /* global size in each dimension */
  PetscInt    dim;      /* global size */
  PetscInt    nnz;      /* local nonzeros */
  PetscBool   matmult;  /* Do MatMult() only */
  PetscBool   debug;    /* Debug flag for PreallocateCOO() */
  PetscBool   splitksp; /* Split KSPSolve and PCSetUp */
  PetscInt    its;      /* No of matmult_iterations */
  PetscInt    Istart, Iend;
} AppCtx;

static PetscErrorCode PreallocateCOO(Mat A, void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt n = user->n, n2 = n * n, n1 = n - 1;
  PetscInt xstart, ystart, zstart, xend, yend, zend, nm2 = n - 2, idx;
  PetscInt nnz[] = {8, 12, 18, 27}; /* nnz for corner, edge, face, and center. */

  PetscFunctionBeginUser;
  xstart = user->Istart % n;
  ystart = (user->Istart / n) % n;
  zstart = user->Istart / n2;
  xend   = (user->Iend - 1) % n;
  yend   = ((user->Iend - 1) / n) % n;
  zend   = (user->Iend - 1) / n2;

  /* bottom xy-plane */
  user->nnz = 0;
  idx       = !zstart ? 0 : 1;
  if (zstart == zend && (!zstart || zstart == n1)) idx = 0;
  if (zstart == zend && (zstart && zstart < n1)) idx = 1;
  if (!xstart && !ystart) // bottom left
    user->nnz += 4 * nnz[idx] + 4 * nm2 * nnz[idx + 1] + nm2 * nm2 * nnz[idx + 2];
  else if (xstart && xstart < n1 && !ystart) // bottom
    user->nnz += 3 * nnz[idx] + (3 * nm2 + n1 - xstart) * nnz[idx + 1] + nm2 * nm2 * nnz[idx + 2];
  else if (xstart == n1 && !ystart) // bottom right
    user->nnz += 3 * nnz[idx] + (3 * nm2) * nnz[idx + 1] + nm2 * nm2 * nnz[idx + 2];
  else if (!xstart && ystart && ystart < n1) // left
    user->nnz += 2 * nnz[idx] + (nm2 + 2 * (n1 - ystart)) * nnz[idx + 1] + nm2 * (n1 - ystart) * nnz[idx + 2];
  else if (xstart && xstart < n1 && ystart && ystart < n1) // center
    user->nnz += 2 * nnz[idx] + (nm2 + (n1 - ystart) + (nm2 - ystart)) * nnz[idx + 1] + (nm2 * (nm2 - ystart) + (n1 - xstart)) * nnz[idx + 2];
  else if (xstart == n1 && ystart && ystart < n1) // right
    user->nnz += 2 * nnz[idx] + (nm2 + n1 - ystart + nm2 - ystart) * nnz[idx + 1] + nm2 * (nm2 - ystart) * nnz[idx + 2];
  else if (ystart == n1 && !xstart) // top left
    user->nnz += 2 * nnz[idx] + nm2 * nnz[idx + 1];
  else if (ystart == n1 && xstart && xstart < n1) // top
    user->nnz += nnz[idx] + (n1 - xstart) * nnz[idx + 1];
  else if (xstart == n1 && ystart == n1) // top right
    user->nnz += nnz[idx];

  /* top and middle plane the same */
  if (zend == zstart) user->nnz = user->nnz - 4 * nnz[idx] - 4 * nm2 * nnz[idx + 1] - nm2 * nm2 * nnz[idx + 2];

  /* middle xy-planes */
  if (zend - zstart > 1) user->nnz = user->nnz + (zend - zstart - 1) * (4 * nnz[1] + 4 * nnz[2] * nm2 + nnz[3] * nm2 * nm2);

  /* top xy-plane */
  idx = zend == n1 ? 0 : 1;
  if (zstart == zend && (!zend || zend == n1)) idx = 0;
  if (zstart == zend && (zend && zend < n1)) idx = 1;
  if (!xend && !yend) // bottom left
    user->nnz += nnz[idx];
  else if (xend && xend < n1 && !yend) // bottom
    user->nnz += nnz[idx] + xend * nnz[idx + 1];
  else if (xend == n1 && !yend) // bottom right
    user->nnz += 2 * nnz[idx] + nm2 * nnz[idx + 1];
  else if (!xend && yend && yend < n1) // left
    user->nnz += 2 * nnz[idx] + (nm2 + yend + yend - 1) * nnz[idx + 1] + nm2 * (yend - 1) * nnz[idx + 2];
  else if (xend && xend < n1 && yend && yend < n1) // center
    user->nnz += 2 * nnz[idx] + (nm2 + yend + yend - 1) * nnz[idx + 1] + (nm2 * (yend - 1) + xend) * nnz[idx + 2];
  else if (xend == n1 && yend && yend < n1) // right
    user->nnz += 2 * nnz[idx] + (nm2 + 2 * yend) * nnz[idx + 1] + (nm2 * yend) * nnz[idx + 2];
  else if (!xend && yend == n1) // top left
    user->nnz += 3 * nnz[idx] + (3 * nm2) * nnz[idx + 1] + (nm2 * nm2) * nnz[idx + 2];
  else if (xend && xend < n1 && yend == n1) // top
    user->nnz += 3 * nnz[idx] + (3 * nm2 + xend) * nnz[idx + 1] + (nm2 * nm2) * nnz[idx + 2];
  else if (xend == n1 && yend == n1) // top right
    user->nnz += 4 * nnz[idx] + (4 * nm2) * nnz[idx + 1] + (nm2 * nm2) * nnz[idx + 2];
  if (user->debug)
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "rank %d: xyzstart = %" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ", xvzend = %" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ", nnz = %" PetscInt_FMT "\n", user->rank, xstart, ystart, zstart, xend, yend, zend,
                          user->nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FillCOO(Mat A, void *ctx)
{
  AppCtx      *user = (AppCtx *)ctx;
  PetscInt     Ii, x, y, z, n = user->n, n2 = n * n, n1 = n - 1;
  PetscInt     count = 0;
  PetscInt    *coo_i, *coo_j;
  PetscScalar *coo_v;
  PetscScalar  h     = 1.0 / (n - 1);
  PetscScalar  vcorn = -1.0 / 13 * h; //-1.0/12.0*h;
  PetscScalar  vedge = -3.0 / 26 * h; //-1.0/6.0*h;
  PetscScalar  vface = -3.0 / 13 * h; //-1.0e-9*h;
  PetscScalar  vcent = 44.0 / 13 * h; //8.0/3.0*h;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc3(user->nnz, &coo_i, user->nnz, &coo_j, user->nnz, &coo_v));
  /* Fill COO arrays */
  for (Ii = user->Istart; Ii < user->Iend; Ii++) {
    x = Ii % n;
    y = (Ii / n) % n;
    z = Ii / n2;
    if (x > 0 && y > 0 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 - n - n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x > 0 && y < n1 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 + n - n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x < n1 && y > 0 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 - n - n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x < n1 && y < n1 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 + n - n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x > 0 && y > 0 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 - n + n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x > 0 && y < n1 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 + n + n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x < n1 && y > 0 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 - n + n2;
      coo_v[count] = vcorn;
      count++;
    }
    if (x < n1 && y < n1 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 + n + n2;
      coo_v[count] = vcorn;
      count++;
    }
    /* 12 edges */
    if (x > 0 && y > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 - n;
      coo_v[count] = vedge;
      count++;
    }
    if (x > 0 && y < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 + n;
      coo_v[count] = vedge;
      count++;
    }
    if (x < n1 && y > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 - n;
      coo_v[count] = vedge;
      count++;
    }
    if (x < n1 && y < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 + n;
      coo_v[count] = vedge;
      count++;
    }
    if (x > 0 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 - n2;
      coo_v[count] = vedge;
      count++;
    }
    if (x > 0 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1 + n2;
      coo_v[count] = vedge;
      count++;
    }
    if (x < n1 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 - n2;
      coo_v[count] = vedge;
      count++;
    }
    if (x < n1 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1 + n2;
      coo_v[count] = vedge;
      count++;
    }
    if (y > 0 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - n - n2;
      coo_v[count] = vedge;
      count++;
    }
    if (y > 0 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - n + n2;
      coo_v[count] = vedge;
      count++;
    }
    if (y < n1 && z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + n - n2;
      coo_v[count] = vedge;
      count++;
    }
    if (y < n1 && z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + n + n2;
      coo_v[count] = vedge;
      count++;
    }
    /* 6 faces */
    if (x > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - 1;
      coo_v[count] = vface;
      count++;
    }
    if (x < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + 1;
      coo_v[count] = vface;
      count++;
    }
    if (y > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - n;
      coo_v[count] = vface;
      count++;
    }
    if (y < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + n;
      coo_v[count] = vface;
      count++;
    }
    if (z > 0) {
      coo_i[count] = Ii;
      coo_j[count] = Ii - n2;
      coo_v[count] = vface;
      count++;
    }
    if (z < n1) {
      coo_i[count] = Ii;
      coo_j[count] = Ii + n2;
      coo_v[count] = vface;
      count++;
    }
    /* Center */
    coo_i[count] = Ii;
    coo_j[count] = Ii;
    coo_v[count] = vcent;
    count++;
  }
  PetscCheck(count == user->nnz, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Expected %" PetscInt_FMT " nonzeros but got %" PetscInt_FMT " nonzeros in COO format\n", user->nnz, count);
  PetscCall(MatSetPreallocationCOO(A, user->nnz, coo_i, coo_j));
  PetscCall(MatSetValuesCOO(A, coo_v, INSERT_VALUES));
  PetscCall(PetscFree3(coo_i, coo_j, coo_v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx                     user;                               /* Application context */
  Vec                        x, b, u;                            /* approx solution, RHS, and exact solution */
  Mat                        A;                                  /* linear system matrix */
  KSP                        ksp;                                /* linear solver context */
  PC                         pc;                                 /* Preconditioner */
  PetscReal                  norm;                               /* Error norm */
  PetscInt                   nlocal     = PETSC_DECIDE, ksp_its; /* Number of KSP iterations */
  PetscInt                   global_nnz = 0;                     /* Total number of nonzeros */
  PetscLogDouble             time_start, time_mid1 = 0.0, time_mid2 = 0.0, time_end, time_avg, floprate;
  PetscBool                  printTiming = PETSC_TRUE; /* If run in CI, do not print timing result */
  PETSC_UNUSED PetscLogStage stage;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &user.size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &user.rank));

  user.n        = 100;         /* Default grid points per dimension */
  user.matmult  = PETSC_FALSE; /* Test MatMult only */
  user.its      = 100;         /* Default no. of iterations for MatMult test */
  user.debug    = PETSC_FALSE; /* Debug PreallocateCOO() */
  user.splitksp = PETSC_FALSE; /* Split KSPSolve and PCSetUp */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-its", &user.its, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-matmult", &user.matmult, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-debug", &user.debug, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_timing", &printTiming, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-split_ksp", &user.splitksp, NULL));

  user.dim   = user.n * user.n * user.n;
  global_nnz = 64 + 27 * (user.n - 2) * (user.n - 2) * (user.n - 2) + 108 * (user.n - 2) * (user.n - 2) + 144 * (user.n - 2);
  PetscCheck(user.n >= 2, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Requires at least 2 grid points (-n 2), you specified -n %" PetscInt_FMT "\n", user.n);
  PetscCheck(user.dim >= user.size, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "MPI size (%d) exceeds the grid size %" PetscInt_FMT " (-n %" PetscInt_FMT ")\n", user.size, user.dim, user.n);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "===========================================\n"));
  if (user.matmult) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test: MatMult performance - Poisson\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test: KSP performance - Poisson\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\tInput matrix: 27-pt finite difference stencil\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\t-n %" PetscInt_FMT "\n", user.n));
  if (user.matmult) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\t-its %" PetscInt_FMT "\n", user.its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\tDoFs = %" PetscInt_FMT "\n", user.dim));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\tNumber of nonzeros = %" PetscInt_FMT "\n", global_nnz));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*  Create the Vecs and Mat                                            */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStep1  - creating Vecs and Mat...\n"));
  PetscCall(PetscLogStageRegister("Step1  - Vecs and Mat", &stage));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, user.dim));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  if (!user.matmult) PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(u, 1.0)); /* Exact solution */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, user.dim, user.dim));
  PetscCall(MatSetFromOptions(A));

  /* cannot use MatGetOwnershipRange() because the matrix has yet to be preallocated; that happens in MatSetPreallocateCOO() */
  PetscCall(PetscSplitOwnership(PetscObjectComm((PetscObject)A), &nlocal, &user.dim));
  PetscCallMPI(MPI_Scan(&nlocal, &user.Istart, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)A)));
  user.Istart -= nlocal;
  user.Iend = user.Istart + nlocal;

  PetscCall(PreallocateCOO(A, &user)); /* Determine local number of nonzeros */
  PetscCall(FillCOO(A, &user));        /* Fill COO Matrix */
  PetscCall(MatMult(A, u, b));         /* Compute RHS based on exact solution */
  PetscCall(PetscLogStagePop());

  if (user.matmult) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  MatMult                                                            */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step2  - running MatMult() %" PetscInt_FMT " times...\n", user.its));
    PetscCall(PetscLogStageRegister("Step2  - MatMult", &stage));
    PetscCall(PetscLogStagePush(stage));
    PetscCall(PetscTime(&time_start));
    for (int i = 0; i < user.its; i++) PetscCall(MatMult(A, u, b));
    PetscCall(PetscTime(&time_end));
    PetscCall(PetscLogStagePop());
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  Calculate Performance metrics                                      */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    time_avg = (time_end - time_start) / ((PetscLogDouble)user.its);
    floprate = 2 * global_nnz / time_avg * 1e-9;
    if (printTiming) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n%-15s%-7.5f seconds\n", "Average time:", time_avg));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-9.3e Gflops/sec\n", "FOM:", floprate));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "===========================================\n"));
  } else {
    if (!user.splitksp) {
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      /*  Solve the linear system of equations                               */
      /*  Measure only time of PCSetUp() and KSPSolve()                      */
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step2  - running KSPSolve()...\n"));
      PetscCall(PetscLogStageRegister("Step2  - KSPSolve", &stage));
      PetscCall(PetscLogStagePush(stage));
      PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(PetscTime(&time_start));
      PetscCall(KSPSolve(ksp, b, x));
      PetscCall(PetscTime(&time_end));
      PetscCall(PetscLogStagePop());
    } else {
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      /*  Solve the linear system of equations                               */
      /*  Measure only time of PCSetUp() and KSPSolve()                      */
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step2a - running PCSetUp()...\n"));
      PetscCall(PetscLogStageRegister("Step2a - PCSetUp", &stage));
      PetscCall(PetscLogStagePush(stage));
      PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PetscTime(&time_start));
      PetscCall(PCSetUp(pc));
      PetscCall(PetscTime(&time_mid1));
      PetscCall(PetscLogStagePop());
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step2b - running KSPSolve()...\n"));
      PetscCall(PetscLogStageRegister("Step2b - KSPSolve", &stage));
      PetscCall(PetscLogStagePush(stage));
      PetscCall(PetscTime(&time_mid2));
      PetscCall(KSPSolve(ksp, b, x));
      PetscCall(PetscTime(&time_end));
      PetscCall(PetscLogStagePop());
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  Calculate error norm                                               */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step3  - calculating error norm...\n"));
    PetscCall(PetscLogStageRegister("Step3  - Error norm", &stage));
    PetscCall(PetscLogStagePush(stage));
    PetscCall(VecAXPY(x, -1.0, u));
    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCall(PetscLogStagePop());

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  Summary                                                            */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(KSPGetIterationNumber(ksp, &ksp_its));
    if (printTiming) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n%-15s%-1.3e\n", "Error norm:", (double)norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-3" PetscInt_FMT "\n", "KSP iters:", ksp_its));
      if (user.splitksp) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-7.5f seconds\n", "PCSetUp:", time_mid1 - time_start));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-7.5f seconds\n", "KSPSolve:", time_end - time_mid2));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-7.5f seconds\n", "Total Solve:", time_end - time_start));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-7.5f seconds\n", "KSPSolve:", time_end - time_start));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-15s%-1.3e DoFs/sec\n", "FOM:", user.dim / (time_end - time_start)));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "===========================================\n"));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*  Free up memory                                                     */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!user.matmult) {
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&x));
  }
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -print_timing false -matmult -its 10 -n 8
    nsize: {{1 3}}
    output_file: output/bench_kspsolve_matmult.out

    test:
      suffix: matmult

    test:
      suffix: hip_matmult
      requires: hip
      args: -vec_type hip -mat_type aijhipsparse

    test:
      suffix: cuda_matmult
      requires: cuda
      args: -vec_type cuda -mat_type aijcusparse

    test:
      suffix: kok_matmult
      requires: kokkos_kernels
      args: -vec_type kokkos -mat_type aijkokkos

  testset:
    args: -print_timing false -its 10 -n 8
    nsize: {{1 3}}
    output_file: output/bench_kspsolve_ksp.out

    test:
      suffix: ksp

    test:
      suffix: hip_ksp
      requires: hip
      args: -vec_type hip -mat_type aijhipsparse

    test:
      suffix: cuda_ksp
      requires: cuda
      args: -vec_type cuda -mat_type aijcusparse

    test:
      suffix: kok_ksp
      requires: kokkos_kernels
      args: -vec_type kokkos -mat_type aijkokkos
TEST*/
