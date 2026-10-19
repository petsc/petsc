static const char help[] = "Tests PCHPDDM with a user-supplied deflation space and scaling against an explicitly assembled reference operator.\n\n";

/*
  Setup, on two processes:
  - A is an 8 x 8 tridiagonal matrix, complex in complex builds, with 4 rows per process.
  - The overlapping subdomain of process p holds the global rows 2p, ..., 2p + 5, see SubdomainRow().
  - The scaling D_p is given by Scaling() and the single deflation vector U_p by Mode().
    With -empty_rank0, process 0 supplies no deflation vector.
  - ModifySubmatrices() shifts the local matrices of the fine-level PCASM, as optimized transmission
    conditions would, so that the local solvers A_p differ from the restrictions of A.

  After each setup, PCApply() and PCMatApply() must match the reference operator
    T = Q + B (I - A Q)    with -pc_hpddm_coarse_correction deflated,
    T = B + Q (I - A B)    with -pc_hpddm_coarse_correction deflated_reversed,
  where Q = Z (Z^H A Z)^{-1} Z^H is the coarse correction with Z = [R_p^T D_p U_p], and B is the
  fine-level correction, see FineLevelType.
  Also checks without deflation (-scaling_only) and then T = B.
*/

#include <petscksp.h>

#define N_GLOBAL    8 // number of unknowns
#define N_LOCAL     4 // rows owned by each process
#define N_SUBDOMAIN 6 // rows of each overlapping subdomain

// fine-level corrections that the reference operator can reproduce
typedef enum {
  FINE_ASM_WEIGHTED, // default with a user-supplied scaling: B = sum_p R_p^T D_p A_p^{-1} R_p
  FINE_ASM_BASIC,    // -pc_hpddm_levels_1_pc_asm_type basic: B = sum_p R_p^T A_p^{-1} R_p
  FINE_JACOBI        // -pc_hpddm_levels_1_pc_type jacobi:    B = diag(A)^{-1}
} FineLevelType;

static PetscScalar TestScalar(PetscReal real, PetscReal imag)
{
#if PetscDefined(USE_COMPLEX)
  return PetscCMPLX(real, imag);
#else
  (void)imag;
  return real;
#endif
}

// global row of the i-th row of the overlapping subdomain of process p: rows 0-5 for process 0, rows 2-7 for process 1
static PetscInt SubdomainRow(PetscMPIInt p, PetscInt i)
{
  return 2 * p + i;
}

/*
  Scaling of process p at a global row. Process 0 scales its rows 0-5 by 1, 1, 1, 0.65, 0.35, 0, and process 1 its
  rows 2-7 by the complements 0, 0.35, 0.65, 1, 1, 1. The two scalings sum to one on every row and vanish outside each
  subdomain. They also vanish on the subdomain row that A couples to the outside (row 5 for process 0, row 2 for
  process 1), as PCHPDDMSetScaling() requires.
*/
static PetscScalar Scaling(PetscMPIInt p, PetscInt row)
{
  const PetscReal process0[N_GLOBAL] = {1.0, 1.0, 1.0, 0.65, 0.35, 0.0, 0.0, 0.0};

  return p == 0 ? process0[row] : 1.0 - process0[row];
}

// entry of the deflation vector of process p at a global row; complex in complex builds, which exercises conjugation
static PetscScalar Mode(PetscMPIInt p, PetscInt row)
{
  return TestScalar(1.0 + 0.15 * row, 0.3 * (p + 1) - 0.05 * row);
}

// shift applied to every fine-level local matrix by ModifySubmatrices(), and by the reference operator
static PetscScalar LocalShift(void)
{
  return TestScalar(0.35, 0.2);
}

static PetscErrorCode ModifySubmatrices(PC pc, PetscInt n, const IS rows[], const IS cols[], Mat sub[], PetscCtx ctx)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < n; ++i) PetscCall(MatShift(sub[i], LocalShift()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperator(Mat *A)
{
  PetscInt start, end;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, N_LOCAL, N_LOCAL, N_GLOBAL, N_GLOBAL, 3, NULL, 2, NULL, A));
  PetscCall(MatGetOwnershipRange(*A, &start, &end));
  for (PetscInt row = start; row < end; ++row) {
    PetscCall(MatSetValue(*A, row, row, TestScalar(3.0 + 0.1 * row, 0.25), INSERT_VALUES));
    if (row > 0) PetscCall(MatSetValue(*A, row, row - 1, TestScalar(-0.9, 0.07), INSERT_VALUES));
    if (row < N_GLOBAL - 1) PetscCall(MatSetValue(*A, row, row + 1, TestScalar(-0.3, -0.12), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Inputs of PCHPDDMSetDeflationMat() and PCHPDDMSetScaling(), as a user would build them: the subdomain IS,
  the local deflation matrix U (one column, or none on process 0 with -empty_rank0), and the local scaling D. All three
  list the subdomain rows in reverse order, unlike the sorted order used internally by PCASM, so that the test also
  checks how PCHPDDM reorders the scaling.
*/
static PetscErrorCode CreateDeflation(PetscBool empty_rank0, IS *is, Mat *U, Vec *D)
{
  PetscMPIInt  rank;
  PetscInt     rows[N_SUBDOMAIN], nmodes;
  PetscScalar *u, *d;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  nmodes = (empty_rank0 == PETSC_TRUE && rank == 0) ? 0 : 1; // number of local deflation vectors
  for (PetscInt i = 0; i < N_SUBDOMAIN; ++i) rows[i] = SubdomainRow(rank, N_SUBDOMAIN - 1 - i);
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, N_SUBDOMAIN, rows, PETSC_COPY_VALUES, is));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, N_SUBDOMAIN, nmodes, NULL, U));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N_SUBDOMAIN, D));
  PetscCall(MatDenseGetArray(*U, &u));
  PetscCall(VecGetArray(*D, &d));
  for (PetscInt i = 0; i < N_SUBDOMAIN; ++i) {
    d[i] = Scaling(rank, rows[i]);
    if (nmodes > 0) u[i] = Mode(rank, rows[i]);
  }
  PetscCall(VecRestoreArray(*D, &d));
  PetscCall(MatDenseRestoreArray(*U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Fine-level correction B, see FineLevelType. For PCASM, each process adds its term R_p^T [D_p] A_p^{-1} R_p column by
  column, with A_p the shifted restriction of A to its subdomain, factored with LU like the PCASM local solver.
*/
static PetscErrorCode CreateFineReference(Mat A, FineLevelType fine, Mat *B)
{
  Mat               *sub;
  KSP                ksp;
  PC                 lu;
  Vec                x, y;
  IS                 is;
  PetscMPIInt        rank;
  PetscInt           rows[N_SUBDOMAIN], start, end;
  const PetscScalar *values;
  PetscScalar        column[N_SUBDOMAIN];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, N_LOCAL, N_LOCAL, N_GLOBAL, N_GLOBAL, N_LOCAL, NULL, N_LOCAL, NULL, B)); // rows may be dense
  if (fine == FINE_JACOBI) {
    PetscCall(MatCreateVecs(A, NULL, &x));
    PetscCall(MatGetDiagonal(A, x));
    PetscCall(MatGetOwnershipRange(A, &start, &end));
    PetscCall(VecGetArrayRead(x, &values));
    for (PetscInt row = start; row < end; ++row) PetscCall(MatSetValue(*B, row, row, (PetscScalar)1.0 / values[row - start], INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(x, &values));
    PetscCall(VecDestroy(&x));
  } else {
    for (PetscInt i = 0; i < N_SUBDOMAIN; ++i) rows[i] = SubdomainRow(rank, i);
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, N_SUBDOMAIN, rows, PETSC_COPY_VALUES, &is));
    PetscCall(MatCreateSubMatrices(A, 1, &is, &is, MAT_INITIAL_MATRIX, &sub));
    PetscCall(MatShift(sub[0], LocalShift()));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPSetOperators(ksp, sub[0], sub[0]));
    PetscCall(KSPGetPC(ksp, &lu));
    PetscCall(PCSetType(lu, PCLU));
    PetscCall(MatCreateVecs(sub[0], &x, &y));
    for (PetscInt j = 0; j < N_SUBDOMAIN; ++j) { // column j of A_p^{-1}, scaled by D_p for PC_ASM_WEIGHTED
      PetscCall(VecSet(x, 0.0));
      PetscCall(VecSetValue(x, j, 1.0, INSERT_VALUES));
      PetscCall(VecAssemblyBegin(x));
      PetscCall(VecAssemblyEnd(x));
      PetscCall(KSPSolve(ksp, x, y));
      PetscCall(VecGetArrayRead(y, &values));
      for (PetscInt i = 0; i < N_SUBDOMAIN; ++i) column[i] = (fine == FINE_ASM_WEIGHTED ? Scaling(rank, rows[i]) : (PetscScalar)1.0) * values[i];
      PetscCall(MatSetValues(*B, N_SUBDOMAIN, rows, 1, rows + j, column, ADD_VALUES));
      PetscCall(VecRestoreArrayRead(y, &values));
    }
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroySubMatrices(1, &sub));
    PetscCall(ISDestroy(&is));
  }
  PetscCall(MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Coarse correction Q = Z (Z^H A Z)^{-1} Z^H. Column p of Z is the scaled deflation vector R_p^T D_p U_p of process p,
  which is zero outside subdomain p because Scaling() is. With -empty_rank0, Z only has the column of process 1.
*/
static PetscErrorCode CreateCoarseReference(Mat A, PetscBool empty_rank0, Mat *Q)
{
  Mat         Z, Zh, E, EinvZh;
  KSP         ksp;
  PC          redundant;
  PetscMPIInt rank, first_process = empty_rank0 == PETSC_TRUE ? 1 : 0; // first process with a deflation vector
  PetscInt    start, end;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, N_LOCAL, rank < first_process ? 0 : 1, N_GLOBAL, 2 - first_process, 1, NULL, 1, NULL, &Z));
  PetscCall(MatGetOwnershipRange(Z, &start, &end));
  for (PetscInt row = start; row < end; ++row)
    for (PetscMPIInt p = first_process; p < 2; ++p) PetscCall(MatSetValue(Z, row, p - first_process, Scaling(p, row) * Mode(p, row), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Z, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Z, MAT_FINAL_ASSEMBLY));
  PetscCall(MatHermitianTranspose(Z, MAT_INITIAL_MATRIX, &Zh));
  PetscCall(MatMatMatMult(Zh, A, Z, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &E));
  PetscCall(MatConvert(Zh, MATDENSE, MAT_INPLACE_MATRIX, &Zh)); // KSPMatSolve() needs dense right-hand sides
  PetscCall(MatDuplicate(Zh, MAT_DO_NOT_COPY_VALUES, &EinvZh));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPSetOperators(ksp, E, E));
  PetscCall(KSPGetPC(ksp, &redundant));
  PetscCall(PCSetType(redundant, PCREDUNDANT)); // gathers E on every process and factors it with the sequential PETSc LU
  PetscCall(KSPMatSolve(ksp, Zh, EinvZh));
  PetscCall(MatMatMult(Z, EinvZh, MAT_INITIAL_MATRIX, PETSC_DETERMINE, Q));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&Zh));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&EinvZh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Reference two-level operator T = first + second (I - A first), see the description at the top of this file: the
  deflated correction applies the coarse level first, the reversed one applies the fine level first.
*/
static PetscErrorCode CreateReference(Mat A, PetscBool empty_rank0, FineLevelType fine, PCHPDDMCoarseCorrectionType correction, Mat *T)
{
  Mat B, Q, first, second, IA;

  PetscFunctionBeginUser;
  PetscCall(CreateFineReference(A, fine, &B));
  PetscCall(CreateCoarseReference(A, empty_rank0, &Q));
  if (correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED_REVERSED) {
    first  = B;
    second = Q;
  } else {
    first  = Q;
    second = B;
  }
  PetscCall(MatMatMult(A, first, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &IA));
  PetscCall(MatScale(IA, -1.0));
  PetscCall(MatShift(IA, 1.0)); // I - A first
  PetscCall(MatMatMult(second, IA, MAT_INITIAL_MATRIX, PETSC_DETERMINE, T));
  PetscCall(MatConvert(B, MATDENSE, MAT_INPLACE_MATRIX, &B)); // T is dense, and MatAXPY() needs B in the same format; in place, so first still refers to B
  PetscCall(MatAXPY(*T, 1.0, first, SAME_NONZERO_PATTERN));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&IA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Fine-level correction that PCHPDDM builds from the same options, read with the prefix of its fine level:
  pc_hpddm_levels_1_, or pc_hpddm_coarse_ with a single level. Only the corrections of FineLevelType have a reference.
*/
static PetscErrorCode GetFineLevelType(const char prefix[], FineLevelType *fine)
{
  char      type[256] = PCASM;
  PCASMType asm_type  = PC_ASM_WEIGHTED; // default selected by PCHPDDM with a user-supplied scaling
  PetscBool asm_pc, jacobi;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetString(NULL, prefix, "-pc_type", type, sizeof(type), NULL));
  PetscCall(PetscOptionsGetEnum(NULL, prefix, "-pc_asm_type", PCASMTypes, (PetscEnum *)&asm_type, NULL));
  PetscCall(PetscStrcmp(type, PCASM, &asm_pc));
  PetscCall(PetscStrcmp(type, PCJACOBI, &jacobi));
  PetscCheck(jacobi == PETSC_TRUE || (asm_pc == PETSC_TRUE && (asm_type == PC_ASM_WEIGHTED || asm_type == PC_ASM_BASIC)), PETSC_COMM_WORLD, PETSC_ERR_SUP, "No reference for this fine-level PC, use PCJACOBI or PCASM of type PC_ASM_WEIGHTED or PC_ASM_BASIC");
  if (jacobi == PETSC_TRUE) *fine = FINE_JACOBI;
  else *fine = asm_type == PC_ASM_WEIGHTED ? FINE_ASM_WEIGHTED : FINE_ASM_BASIC;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckVector(Vec actual, Vec expected)
{
  PetscReal error, norm;

  PetscFunctionBeginUser;
  PetscCall(VecNorm(expected, NORM_INFINITY, &norm));
  PetscCall(VecAXPY(expected, -1.0, actual));
  PetscCall(VecNorm(expected, NORM_INFINITY, &error));
  PetscCheck(error <= 1000.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, norm), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCHPDDM differs from the reference operator by %g", (double)error);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Compares the PC with the reference operator T: PCApply() on every unit vector, and PCMatApply() on a block of three
  vectors. With check_transpose, also compares PCApplyTranspose() and PCMatApplyTranspose() with T^T. Complex builds only
  check them without a deflation space: with a coarse correction, the complex transpose of PCHPDDM applies Q^H instead of
  Q^T, whether or not a scaling is supplied.
*/
static PetscErrorCode CheckApplications(PC pc, Mat T, PetscBool check_transpose)
{
  Vec      x, y, expected;
  PetscInt N, m, start, end;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(T, &N, NULL));
  PetscCall(MatGetLocalSize(T, &m, NULL));
  PetscCall(MatCreateVecs(T, &x, &y));
  PetscCall(VecDuplicate(y, &expected));
  PetscCall(VecGetOwnershipRange(x, &start, &end));
  for (PetscInt j = 0; j < N; ++j) {
    PetscCall(VecSet(x, 0.0));
    if (start <= j && j < end) PetscCall(VecSetValue(x, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    PetscCall(PCApply(pc, x, y));
    PetscCall(MatMult(T, x, expected));
    PetscCall(CheckVector(y, expected));
    if (check_transpose) {
      PetscCall(PCApplyTranspose(pc, x, y));
      PetscCall(MatMultTranspose(T, x, expected));
      PetscCall(CheckVector(y, expected));
    }
  }
  {
    Mat X, Y;

    PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, N, 3, NULL, &X));
    PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
    PetscCall(MatSetRandom(X, NULL));
    for (PetscInt transpose = 0; transpose < (check_transpose ? 2 : 1); ++transpose) {
      if (transpose) PetscCall(PCMatApplyTranspose(pc, X, Y));
      else PetscCall(PCMatApply(pc, X, Y));
      for (PetscInt j = 0; j < 3; ++j) {
        Vec xc, yc;

        PetscCall(MatDenseGetColumnVecRead(X, j, &xc));
        PetscCall(MatDenseGetColumnVecRead(Y, j, &yc));
        if (transpose) PetscCall(MatMultTranspose(T, xc, expected));
        else PetscCall(MatMult(T, xc, expected));
        PetscCall(CheckVector(yc, expected));
        PetscCall(MatDenseRestoreColumnVecRead(Y, j, &yc));
        PetscCall(MatDenseRestoreColumnVecRead(X, j, &xc));
      }
    }
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&Y));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// checks that the PC holds the shared IS of PCHPDDMSetDeflationMat() and PCHPDDMSetScaling() exactly when one of them uses it
static PetscErrorCode CheckReferences(IS is, PetscBool held)
{
  PetscInt refct;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetReference((PetscObject)is, &refct));
  PetscCheck(refct == (held ? 2 : 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "IS has %" PetscInt_FMT " references, expected %d", refct, held ? 2 : 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PC                          pc;
  Mat                         A, U, T;
  IS                          is;
  Vec                         D;
  PetscMPIInt                 size;
  PetscBool                   empty_rank0 = PETSC_FALSE, scaling_only = PETSC_FALSE;
  FineLevelType               fine = FINE_ASM_WEIGHTED;
  PCHPDDMCoarseCorrectionType correction;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example requires 2 processes");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank0", &empty_rank0, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-scaling_only", &scaling_only, NULL)); // no deflation space, the scaling only applies to the fine level
  PetscCall(GetFineLevelType(scaling_only ? "pc_hpddm_coarse_" : "pc_hpddm_levels_1_", &fine));
  PetscCall(CreateOperator(&A));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCHPDDM));
  /*
    Three setups of the same PC, each checked against the reference operator:
    0. initial setup, after installing and removing the deflation space and scaling in both orders, which must leave
       no trace, and with the PC holding their shared IS exactly while one of them uses it;
    1. new operator values with the same deflation space and scaling: both levels are rebuilt, and the fine-level
       PCASMType selected at the initial setup must persist;
    2. PCReset() discards the deflation space and scaling, which are then installed again, the scaling first.
  */
  for (PetscInt pass = 0; pass < 3; ++pass) {
    if (pass == 1) PetscCall(MatScale(A, 1.1));
    else {
      if (pass == 2) PetscCall(PCReset(pc));
      PetscCall(CreateDeflation(empty_rank0, &is, &U, &D));
      if (scaling_only) PetscCall(PCHPDDMSetScaling(pc, is, D));
      else if (pass == 0) {
        PetscCall(PCHPDDMSetScaling(pc, is, D)); // the scaling alone stores the IS
        PetscCall(CheckReferences(is, PETSC_TRUE));
        PetscCall(PCHPDDMSetDeflationMat(pc, NULL, U)); // reuses the stored IS and keeps the scaling
        PetscCall(PCHPDDMSetScaling(pc, NULL, NULL));
        PetscCall(CheckReferences(is, PETSC_TRUE)); // still used by the deflation space
        PetscCall(PCHPDDMSetScaling(pc, NULL, D));
        PetscCall(PCHPDDMSetDeflationMat(pc, NULL, NULL));
        PetscCall(CheckReferences(is, PETSC_TRUE)); // still used by the scaling
        PetscCall(PCHPDDMSetDeflationMat(pc, is, U));
        PetscCall(PCHPDDMSetScaling(pc, NULL, NULL));
        PetscCall(PCHPDDMSetDeflationMat(pc, NULL, NULL));
        PetscCall(CheckReferences(is, PETSC_FALSE));
      }
      if (!scaling_only) {
        if (pass == 2) { // the deflation space reuses the IS stored by the scaling
          PetscCall(PCHPDDMSetScaling(pc, is, D));
          PetscCall(PCHPDDMSetDeflationMat(pc, NULL, U));
        } else {
          IS copy;

          PetscCall(PCHPDDMSetDeflationMat(pc, is, U));
          PetscCall(ISDuplicate(is, &copy)); // replace the stored IS while keeping the deflation matrix
          PetscCall(PCHPDDMSetScaling(pc, copy, D));
          PetscCall(CheckReferences(is, PETSC_FALSE));
          PetscCall(CheckReferences(copy, PETSC_TRUE));
          PetscCall(ISDestroy(&copy));
        }
      }
      PetscCall(VecDestroy(&D)); // the PC keeps its own reference
      PetscCall(ISDestroy(&is));
      PetscCall(MatDestroy(&U));
    }
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetModifySubMatrices(pc, ModifySubmatrices, NULL));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PCSetUp(pc));
    if (scaling_only) PetscCall(CreateFineReference(A, fine, &T));
    else {
      PetscCall(PCHPDDMGetCoarseCorrectionType(pc, &correction));
      PetscCall(CreateReference(A, empty_rank0, fine, correction, &T));
    }
    PetscCall(CheckApplications(pc, T, (PetscBool)(!PetscDefined(USE_COMPLEX) || scaling_only)));
    PetscCall(MatDestroy(&T));
  }
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    # every case runs with and without a deflation vector on process 0
    nsize: 2
    requires: hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -pc_hpddm_define_subdomains true -pc_hpddm_coarse_mat_type aij -empty_rank0 {{false true}}
    output_file: output/empty.out
    test:
      # default fine level: PCASM of type PC_ASM_WEIGHTED, with the same scaling as the coarse space
      suffix: weighted
      args: -pc_hpddm_levels_1_sub_pc_type lu
    test:
      # same fine level with the fine-level correction applied first
      suffix: weighted_reversed
      args: -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_coarse_correction deflated_reversed
    test:
      # PCASM of type PC_ASM_BASIC: the scaling only applies to the coarse space
      # with one subdomain per process, multiplicative composition gives the same operator as additive, but it checks
      # that the second setup keeps the selected PCASMType instead of reverting to PC_ASM_WEIGHTED
      suffix: basic
      args: -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_pc_asm_type basic -pc_hpddm_levels_1_pc_asm_local_type {{additive multiplicative}}
    test:
      # non-PCASM fine level: the scaling only applies to the coarse space
      suffix: jacobi
      args: -pc_hpddm_levels_1_pc_type jacobi

  test:
    # no deflation space: the scaling only applies to the fine-level PCASM of a single-level PC, whose prefix is -pc_hpddm_coarse_
    suffix: scaling_only
    nsize: 2
    requires: hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -pc_hpddm_define_subdomains true -scaling_only -pc_hpddm_coarse_sub_pc_type lu
    output_file: output/empty.out

TEST*/
