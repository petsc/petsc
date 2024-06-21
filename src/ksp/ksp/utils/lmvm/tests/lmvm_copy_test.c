const char help[] = "Test that MatCopy() does not affect the copied LMVM matrix";

#include <petsc.h>

static PetscErrorCode positiveVectorUpdate(PetscRandom rand, Vec x, Vec f)
{
  Vec         _x, _f;
  PetscScalar dot;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &_x));
  PetscCall(VecDuplicate(f, &_f));
  PetscCall(VecSetRandom(_x, rand));
  PetscCall(VecSetRandom(_f, rand));
  PetscCall(VecDot(_x, _f, &dot));
  PetscCall(VecAXPY(x, PetscAbsScalar(dot) / dot, _x));
  PetscCall(VecAXPY(f, 1.0, _f));
  PetscCall(VecDestroy(&_f));
  PetscCall(VecDestroy(&_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecEqualToTolerance(Vec a, Vec b, NormType norm_type, PetscReal tol, PetscBool *flg)
{
  Vec       diff;
  PetscReal diff_norm;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(a, &diff));
  PetscCall(VecCopy(a, diff));
  PetscCall(VecAXPY(diff, -1.0, b));
  PetscCall(VecNorm(diff, norm_type, &diff_norm));
  PetscCall(VecDestroy(&diff));
  *flg = (diff_norm <= tol) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// unlike MatTestEqual(), this test tests MatMult() and MatSolve()
static PetscErrorCode testMatEqual(PetscRandom rand, Mat A, Mat B, PetscBool *flg)
{
  Vec x, y_A, y_B;

  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  PetscCall(MatCreateVecs(A, &x, &y_A));
  PetscCall(MatCreateVecs(B, NULL, &y_B));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(MatMult(A, x, y_A));
  PetscCall(MatMult(B, x, y_B));
  PetscCall(VecEqualToTolerance(y_A, y_B, NORM_2, PETSC_SMALL, flg));
  if (*flg == PETSC_TRUE) {
    PetscCall(MatSolve(A, x, y_A));
    PetscCall(MatSolve(B, x, y_B));
    PetscCall(VecEqualToTolerance(y_A, y_B, NORM_2, PETSC_SMALL, flg));
    if (*flg == PETSC_FALSE) {
      PetscReal norm;

      PetscCall(VecAXPY(y_A, -1.0, y_B));
      PetscCall(VecNorm(y_A, NORM_INFINITY, &norm));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "MatSolve() norm error %g\n", (double)norm));
    }
  } else {
    PetscReal norm;

    PetscCall(VecAXPY(y_A, -1.0, y_B));
    PetscCall(VecNorm(y_A, NORM_INFINITY, &norm));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "MatMult() norm error %g\n", (double)norm));
  }
  PetscCall(VecDestroy(&y_B));
  PetscCall(VecDestroy(&y_A));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testUnchangedBegin(PetscRandom rand, Mat A, Vec *x, Vec *y, Vec *z)
{
  Vec _x, _y, _z;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(A, &_x, &_y));
  PetscCall(MatCreateVecs(A, NULL, &_z));
  PetscCall(VecSetRandom(_x, rand));
  PetscCall(MatMult(A, _x, _y));
  PetscCall(MatSolve(A, _x, _z));
  *x = _x;
  *y = _y;
  *z = _z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testUnchangedEnd(PetscRandom rand, Mat A, Vec *x, Vec *y, Vec *z, PetscBool *unchanged)
{
  Vec _x, _y, _z, _y2, _z2;

  PetscFunctionBegin;
  *unchanged = PETSC_TRUE;
  _x         = *x;
  _y         = *y;
  _z         = *z;
  *x         = NULL;
  *y         = NULL;
  *z         = NULL;
  PetscCall(MatCreateVecs(A, NULL, &_y2));
  PetscCall(MatCreateVecs(A, NULL, &_z2));
  PetscCall(MatMult(A, _x, _y2));
  PetscCall(MatSolve(A, _x, _z2));
  PetscCall(VecEqual(_y, _y2, unchanged));
  if (*unchanged == PETSC_TRUE) PetscCall(VecEqual(_z, _z2, unchanged));
  PetscCall(VecDestroy(&_z2));
  PetscCall(VecDestroy(&_y2));
  PetscCall(VecDestroy(&_z));
  PetscCall(VecDestroy(&_y));
  PetscCall(VecDestroy(&_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testMatLMVMCopy(PetscRandom rand)
{
  PetscInt    N     = 10;
  MPI_Comm    comm  = PetscObjectComm((PetscObject)rand);
  PetscInt    k_pre = 2; // number of updates before copy
  Mat         A, A_copy;
  Vec         u, x, f, x_copy, f_copy, v1, v2, v3;
  PetscBool   equal;
  PetscLayout layout;

  PetscFunctionBegin;
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, N, &u));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecSetUp(u));
  PetscCall(VecDuplicate(u, &x));
  PetscCall(VecDuplicate(u, &f));
  PetscCall(VecGetLayout(u, &layout));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetLayouts(A, layout, layout));
  PetscCall(MatSetType(A, MATLMVMBFGS));
  PetscCall(MatSetFromOptions(A));
  PetscCall(positiveVectorUpdate(rand, x, f));
  PetscCall(MatLMVMAllocate(A, x, f));
  PetscCall(MatSetUp(A));
  for (PetscInt k = 0; k <= k_pre; k++) {
    PetscCall(positiveVectorUpdate(rand, x, f));
    PetscCall(MatLMVMUpdate(A, x, f));
  }
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_copy));
  PetscCall(testMatEqual(rand, A, A_copy, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "MatCopy() not the same after initial copy");

  PetscCall(VecDuplicate(x, &x_copy));
  PetscCall(VecCopy(x, x_copy));
  PetscCall(VecDuplicate(f, &f_copy));
  PetscCall(VecCopy(f, f_copy));

  PetscCall(testUnchangedBegin(rand, A_copy, &v1, &v2, &v3));
  PetscCall(positiveVectorUpdate(rand, x, f));
  PetscCall(MatLMVMUpdate(A, x, f));
  PetscCall(testUnchangedEnd(rand, A_copy, &v1, &v2, &v3, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "MatLMVMUpdate() to original matrix affects copy");

  PetscCall(testUnchangedBegin(rand, A, &v1, &v2, &v3));
  PetscCall(positiveVectorUpdate(rand, x_copy, f_copy));
  PetscCall(MatLMVMUpdate(A_copy, x_copy, f_copy));
  PetscCall(testUnchangedEnd(rand, A, &v1, &v2, &v3, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "MatLMVMUpdate() to copy matrix affects original");

  PetscCall(VecDestroy(&f_copy));
  PetscCall(VecDestroy(&x_copy));
  PetscCall(MatDestroy(&A_copy));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscRandom rand;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(KSPInitializePackage());
  PetscCall(testMatLMVMCopy(rand));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -mat_type {{lmvmbfgs lmvmdfp lmvmsr1 lmvmbroyden lmvmbadbroyden lmvmsymbroyden lmvmsymbadbroyden lmvmdiagbroyden lmvmdbfgs lmvmddfp lmvmdqn}}

TEST*/
