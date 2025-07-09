const char help[] = "Test validity of different approaches to specifying a MatLMVM";

#include <petscksp.h>

enum {
  TEST_INIT_NONE,
  TEST_INIT_VECS,
  TEST_INIT_DIAG,
  TEST_INIT_MAT,
  TEST_INIT_PC,
  TEST_INIT_KSP,
  TEST_INIT_COUNT
};

typedef PetscInt TestInitType;

enum {
  TEST_SIZE_NONE,
  TEST_SIZE_LOCAL,
  TEST_SIZE_GLOBAL,
  TEST_SIZE_BOTH,
  TEST_SIZE_COUNT
};

typedef PetscInt TestSizeType;

static PetscErrorCode CreateMatWithTestSizes(MPI_Comm comm, MatType mat_type, PetscInt n, PetscInt N, TestSizeType size_type, PetscBool call_setup, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  switch (size_type) {
  case TEST_SIZE_LOCAL:
    PetscCall(MatSetSizes(*B, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
    break;
  case TEST_SIZE_GLOBAL:
    PetscCall(MatSetSizes(*B, PETSC_DECIDE, PETSC_DECIDE, N, N));
    break;
  case TEST_SIZE_BOTH:
    PetscCall(MatSetSizes(*B, n, n, N, N));
    break;
  default:
    break;
  }
  PetscCall(MatSetType(*B, mat_type));
  if (call_setup) PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestUsability(Mat B, Vec pattern, PetscRandom rand, PetscBool test_pre)
{
  Vec dx, df;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(pattern, &dx));
  PetscCall(VecDuplicate(pattern, &df));
  PetscCall(VecSetRandom(dx, rand));

  if (test_pre) {
    // Mult and Solve should work before any updates
    PetscCall(MatMult(B, dx, df));
    PetscCall(MatSolve(B, dx, df));
  }

  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(VecCopy(dx, df));
    PetscCall(MatLMVMUpdate(B, dx, df));
    PetscCall(MatMult(B, dx, df));
    PetscCall(MatSolve(B, dx, df));
  }

  PetscCall(VecDestroy(&df));
  PetscCall(VecDestroy(&dx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscRandom rand;
  PetscInt    N = 9;
  PetscInt    n = PETSC_DETERMINE;
  Vec         pattern;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(KSPInitializePackage());
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(VecCreate(comm, &pattern));
  PetscCall(PetscSplitOwnership(comm, &n, &N));
  PetscCall(VecSetSizes(pattern, n, N));
  PetscCall(VecSetType(pattern, VECSTANDARD));
  for (TestInitType init = TEST_INIT_NONE; init < TEST_INIT_COUNT; init++) {
    for (PetscInt call_setup = 0; call_setup < 2; call_setup++) {
      for (TestSizeType size = (call_setup ? TEST_SIZE_NONE : TEST_SIZE_LOCAL); size < TEST_SIZE_COUNT; size++) {
        Mat B;

        // skip invalid case
        if (init == TEST_INIT_NONE && size == TEST_SIZE_NONE) continue;
        if (size == TEST_SIZE_NONE && call_setup) continue;

        PetscCall(CreateMatWithTestSizes(comm, MATLMVMBFGS, n, N, size, call_setup ? PETSC_TRUE : PETSC_FALSE, &B));

        switch (init) {
        case TEST_INIT_NONE:
          PetscCall(TestUsability(B, pattern, rand, call_setup ? PETSC_TRUE : PETSC_FALSE));
          break;
        case TEST_INIT_VECS: {
          Vec x, f;

          PetscCall(VecDuplicate(pattern, &x));
          PetscCall(VecDuplicate(pattern, &f));
          PetscCall(MatLMVMAllocate(B, x, f));
          PetscCall(VecDestroy(&x));
          PetscCall(VecDestroy(&f));
          PetscCall(TestUsability(B, pattern, rand, PETSC_TRUE));
          break;
        }
        case TEST_INIT_DIAG: {
          Vec d;

          PetscCall(VecDuplicate(pattern, &d));
          PetscCall(VecSet(d, 1.0));
          PetscCall(MatLMVMSetJ0Diag(B, d));
          PetscCall(VecDestroy(&d));
          PetscCall(TestUsability(B, pattern, rand, PETSC_TRUE));
          break;
        }
        case TEST_INIT_MAT: {
          for (PetscInt j0_call_setup = 0; j0_call_setup < 2; j0_call_setup++) {
            for (TestSizeType j0_size = TEST_SIZE_LOCAL; j0_size < TEST_SIZE_COUNT; j0_size++) {
              Mat J0;

              PetscCall(CreateMatWithTestSizes(comm, MATDENSE, n, N, j0_size, j0_call_setup ? PETSC_TRUE : PETSC_FALSE, &J0));
              PetscCall(MatLMVMSetJ0(B, J0));
              PetscCall(MatZeroEntries(J0));
              PetscCall(MatShift(J0, 1.0));
              PetscCall(MatDestroy(&J0));
              PetscCall(TestUsability(B, pattern, rand, PETSC_TRUE));
            }
          }
          break;
        }
        case TEST_INIT_PC: {
          for (PetscInt j0_call_setup = 0; j0_call_setup < 2; j0_call_setup++) {
            for (TestSizeType j0_size = TEST_SIZE_LOCAL; j0_size < TEST_SIZE_COUNT; j0_size++) {
              PC  J0pc;
              Mat J0;

              PetscCall(CreateMatWithTestSizes(comm, MATCONSTANTDIAGONAL, n, N, j0_size, j0_call_setup ? PETSC_TRUE : PETSC_FALSE, &J0));
              PetscCall(PCCreate(comm, &J0pc));
              PetscCall(PCSetType(J0pc, PCMAT));
              PetscCall(PCMatSetApplyOperation(J0pc, MATOP_SOLVE));
              PetscCall(PCSetOperators(J0pc, J0, J0));
              PetscCall(MatLMVMSetJ0PC(B, J0pc));
              PetscCall(MatZeroEntries(J0));
              PetscCall(MatShift(J0, 1.0));
              PetscCall(MatDestroy(&J0));
              PetscCall(PCDestroy(&J0pc));
              PetscCall(TestUsability(B, pattern, rand, PETSC_TRUE));
            }
          }
          break;
        }
        case TEST_INIT_KSP: {
          for (PetscInt j0_call_setup = 0; j0_call_setup < 2; j0_call_setup++) {
            for (TestSizeType j0_size = TEST_SIZE_LOCAL; j0_size < TEST_SIZE_COUNT; j0_size++) {
              KSP J0ksp;
              PC  J0pc;
              Mat J0;

              PetscCall(CreateMatWithTestSizes(comm, MATCONSTANTDIAGONAL, n, N, j0_size, j0_call_setup ? PETSC_TRUE : PETSC_FALSE, &J0));
              PetscCall(KSPCreate(comm, &J0ksp));
              PetscCall(KSPSetOperators(J0ksp, J0, J0));
              PetscCall(KSPGetPC(J0ksp, &J0pc));
              PetscCall(PCSetType(J0pc, PCNONE));
              PetscCall(MatLMVMSetJ0KSP(B, J0ksp));
              PetscCall(MatZeroEntries(J0));
              PetscCall(MatShift(J0, 1.0));
              PetscCall(MatDestroy(&J0));
              PetscCall(KSPDestroy(&J0ksp));
              PetscCall(TestUsability(B, pattern, rand, PETSC_TRUE));
            }
          }
          break;
        }
        default:
          break;
        }
        PetscCall(MatDestroy(&B));
      }
    }
  }

  PetscCall(VecDestroy(&pattern));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 2
    output_file: output/empty.out

TEST*/
