const char help[] = "Coverage and edge case test for LMVM";

#include <petscksp.h>
#include <petscmath.h>

static PetscErrorCode CheckProperties(Mat B, PetscBool symmetric, PetscBool spd)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)B);
  PetscBool known, flg;
  MatType   type;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(B, &type));
  PetscCall(MatIsHermitianKnown(B, &known, &flg));
  PetscCheck(symmetric ? known && flg : !known, comm, PETSC_ERR_PLIB, "Incorrect Hermitian metadata for %s", type);
  if (!PetscDefined(USE_COMPLEX)) {
    PetscCall(MatIsSymmetricKnown(B, &known, &flg));
    PetscCheck(symmetric ? known && flg : !known, comm, PETSC_ERR_PLIB, "Incorrect symmetry metadata for %s", type);
  }
  PetscCall(MatIsSPDKnown(B, &known, &flg));
  PetscCheck(spd ? known && flg : !known, comm, PETSC_ERR_PLIB, "Incorrect SPD metadata for %s", type);
  if (symmetric) {
    Mat       A, AH;
    PetscReal norm, err;

    PetscCall(MatComputeOperator(B, MATDENSE, &A));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &norm));
    PetscCall(MatHermitianTranspose(A, MAT_INITIAL_MATRIX, &AH));
    PetscCall(MatAXPY(AH, -1.0, A, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(AH, NORM_FROBENIUS, &err));
    PetscCheck(err <= 100.0 * PETSC_MACHINE_EPSILON * norm, comm, PETSC_ERR_PLIB, "%s is not Hermitian: relative error %g", type, (double)(err / norm));
    PetscCall(MatDestroy(&AH));
    if (spd) {
      KSP         ksp;
      PC          pc;
      PetscInt    n;
      PetscMPIInt count;
      PetscReal  *real, *imag;

      PetscCall(MatGetSize(A, &n, NULL));
      PetscCall(PetscMalloc2(n, &real, n, &imag));
      PetscCall(KSPCreate(comm, &ksp));
      PetscCall(KSPSetType(ksp, KSPMINRES));
      PetscCall(KSPSetPCSide(ksp, PC_LEFT));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCNONE));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(KSPComputeEigenvaluesExplicitly(ksp, n, real, imag));
      PetscCall(PetscMPIIntCast(n, &count));
      PetscCallMPI(MPI_Bcast(real, count, MPIU_REAL, 0, comm));
      PetscCallMPI(MPI_Bcast(imag, count, MPIU_REAL, 0, comm));
      for (PetscInt i = 0; i < n; ++i) PetscCheck(real[i] > 0.0 && PetscAbsReal(imag[i]) <= 100.0 * PETSC_MACHINE_EPSILON * norm, comm, PETSC_ERR_PLIB, "%s has eigenvalue %g + %gi", type, (double)real[i], (double)imag[i]);
      PetscCall(KSPDestroy(&ksp));
      PetscCall(PetscFree2(real, imag));
    }
    PetscCall(MatDestroy(&A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestProperties(Mat B)
{
  Vec          x, f, d;
  PetscRandom  rand;
  PetscInt     m, start, end;
  PetscScalar *a;
  PetscBool    symmetric, spd, sr1;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &spd, MATLMVMBFGS, MATLMVMDFP, MATLMVMSYMBROYDEN, MATLMVMSYMBADBROYDEN, MATLMVMDIAGBROYDEN, MATLMVMDBFGS, MATLMVMDDFP, MATLMVMDQN, ""));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSR1, &sr1));
  symmetric = (PetscBool)(spd || sr1);
  PetscCall(MatCreateVecs(B, &x, &f));
  PetscCall(VecDuplicate(x, &d));
  PetscCall(VecGetOwnershipRange(d, &start, &end));
  PetscCall(VecGetArray(d, &a));
  for (PetscInt i = start; i < end; ++i) a[i - start] = 1.0 + i;
  PetscCall(VecRestoreArray(d, &a));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)B), &rand));
  PetscCall(MatLMVMGetHistorySize(B, &m));
  PetscCall(CheckProperties(B, symmetric, spd));
  for (PetscInt i = 0; i <= 2 * m; ++i) {
    PetscCall(VecSetRandom(x, rand));
    PetscCall(VecPointwiseMult(f, d, x));
    PetscCall(MatLMVMUpdate(B, x, f));
    PetscCall(CheckProperties(B, symmetric, spd));
  }

  /* Repeated assembly must preserve the declared properties. */
  if (!PetscDefined(USE_COMPLEX)) {
    for (PetscInt i = 0; i < 2; ++i) {
      PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
      PetscCall(CheckProperties(B, symmetric, spd));
    }
  }
  PetscCall(MatLMVMReset(B, PETSC_FALSE));
  PetscCall(CheckProperties(B, symmetric, spd));
  PetscCall(MatLMVMReset(B, PETSC_TRUE));
  PetscCall(MatSetUp(B));
  PetscCall(CheckProperties(B, symmetric, spd));
  for (PetscInt i = 0; i <= m; ++i) {
    PetscCall(VecSetRandom(x, rand));
    PetscCall(VecPointwiseMult(f, d, x));
    PetscCall(MatLMVMUpdate(B, x, f));
    PetscCall(CheckProperties(B, symmetric, spd));
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt  type = 0, n = 10;
  Mat       B;
  PetscBool properties = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(KSPInitializePackage());
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, "KSP");
  /* LMVM Types. 0: LMVMDBFGS, 1: LMVMDDFP, 2: LMVMDQN */
  PetscCall(PetscOptionsInt("-type", "LMVM Type", __FILE__, type, &type, NULL));
  PetscCall(PetscOptionsBool("-test_properties", "Test symmetry and positive definiteness", NULL, properties, &properties, NULL));
  PetscOptionsEnd();
  if (properties) {
    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n, n));
  } else if (type == 0) {
    PetscCall(MatCreateLMVMDBFGS(PETSC_COMM_WORLD, PETSC_DECIDE, n, &B));
  } else if (type == 1) {
    PetscCall(MatCreateLMVMDDFP(PETSC_COMM_WORLD, PETSC_DECIDE, n, &B));
  } else if (type == 2) {
    PetscCall(MatCreateLMVMDQN(PETSC_COMM_WORLD, PETSC_DECIDE, n, &B));
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Incompatible LMVM Type.");
  }
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  if (properties) PetscCall(TestProperties(B));
  else PetscCall(MatLMVMDenseSetType(B, MAT_LMVM_DENSE_INPLACE));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out
    nsize: {{1 2}}
    args: -mat_lmvm_scale_type {{none scalar diagonal}} -type {{0 1 2}}

  test:
    suffix: properties
    requires: !single
    output_file: output/empty.out
    nsize: {{1 2}}
    args: -test_properties -mat_type {{lmvmbfgs lmvmdfp lmvmsymbroyden lmvmsymbadbroyden lmvmdiagbroyden lmvmdbfgs lmvmddfp lmvmdqn lmvmsr1 lmvmbroyden lmvmbadbroyden}}

TEST*/
