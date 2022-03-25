
static char help[] = "Test VecConcatenate both in serial and parallel.\n";

#include <petscvec.h>

int main(int argc,char **args)
{
  Vec                *x, x_test, y, y_test;
  IS                 *x_is;
  VecScatter         y_to_x, x_to_y;
  PetscInt           i, j, nx, shift, x_size, y_size, *idx;
  PetscScalar        *vals;
  PetscBool          flg, x_equal, y_equal;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nx",&nx,&flg));
  if (!flg) nx = 3;

  y_size = 0;
  shift = 0;
  PetscCall(PetscMalloc1(nx, &x));
  for (i=0; i<nx; i++) {
    x_size = 2*(i + 1);
    y_size += x_size;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x[i]));
    PetscCall(VecSetSizes(x[i], PETSC_DECIDE, x_size));
    PetscCall(VecSetFromOptions(x[i]));
    PetscCall(VecSetUp(x[i]));
    PetscCall(PetscMalloc2(x_size, &idx, x_size, &vals));
    for (j=0; j<x_size; j++) {
      idx[j] = j;
      vals[j] = (PetscScalar)(shift + j + 1);
    }
    shift += x_size;
    PetscCall(VecSetValues(x[i], x_size, (const PetscInt*)idx, (const PetscScalar*)vals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x[i]));
    PetscCall(VecAssemblyEnd(x[i]));
    PetscCall(PetscFree2(idx, vals));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Original X[%" PetscInt_FMT "] vector\n", i));
    PetscCall(VecView(x[i], PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, y_size));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecSetUp(y));
  PetscCall(PetscMalloc2(y_size, &idx, y_size, &vals));
  for (j=0; j<y_size; j++) {
    idx[j] = j;
    vals[j] = (PetscScalar)(j + 1);
  }
  PetscCall(VecSetValues(y, y_size, (const PetscInt*)idx, (const PetscScalar*)vals, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(PetscFree2(idx, vals));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Expected Y vector\n"));
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- base VecConcatenate() test ----------- */
  PetscCall(VecConcatenate(nx, (const Vec*)x, &y_test, &x_is));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for Y = [X[1], X[2], ...]\n"));
  PetscCall(VecView(y_test, PETSC_VIEWER_STDOUT_WORLD));
  y_equal = PETSC_FALSE;
  PetscCall(VecEqual(y_test, y, &y_equal));
  if (!y_equal) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- test VecConcatenate() without IS (checks for dangling memory from IS) ----------- */
  PetscCall(VecDestroy(&y_test));
  PetscCall(VecConcatenate(nx, (const Vec*)x, &y_test, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for Y = [X[1], X[2], ...] w/o IS\n"));
  PetscCall(VecView(y_test, PETSC_VIEWER_STDOUT_WORLD));
  y_equal = PETSC_FALSE;
  PetscCall(VecEqual(y_test, y, &y_equal));
  if (!y_equal) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- using index sets on expected Y instead of concatenated Y ----------- */
  for (i=0; i<nx; i++) {
    PetscCall(VecGetSubVector(y, x_is[i], &x_test));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing index set for X[%" PetscInt_FMT "] component\n", i));
    PetscCall(VecView(x_test, PETSC_VIEWER_STDOUT_WORLD));
    x_equal = PETSC_FALSE;
    PetscCall(VecEqual(x_test, x[i], &x_equal));
    if (!x_equal) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
    }
    PetscCall(VecRestoreSubVector(y, x_is[i], &x_test));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- using VecScatter to communicate data from Y to X[i] for all i ----------- */
  for (i=0; i<nx; i++) {
    PetscCall(VecDuplicate(x[i], &x_test));
    PetscCall(VecZeroEntries(x_test));
    PetscCall(VecScatterCreate(y, x_is[i], x[i], NULL, &y_to_x));
    PetscCall(VecScatterBegin(y_to_x, y, x_test, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(y_to_x, y, x_test, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&y_to_x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for Y -> X[%" PetscInt_FMT "]\n", i));
    PetscCall(VecView(x_test, PETSC_VIEWER_STDOUT_WORLD));
    x_equal = PETSC_FALSE;
    PetscCall(VecEqual(x_test, x[i], &x_equal));
    if (!x_equal) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
    }
    PetscCall(VecDestroy(&x_test));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- using VecScatter to communicate data from X[i] to Y for all i ----------- */
  PetscCall(VecZeroEntries(y_test));
  for (i=0; i<nx; i++) {
    PetscCall(VecScatterCreate(x[i], NULL, y, x_is[i], &x_to_y));
    PetscCall(VecScatterBegin(x_to_y, x[i], y_test, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(x_to_y, x[i], y_test, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&x_to_y));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for X[:] -> Y\n"));
  PetscCall(VecView(y_test, PETSC_VIEWER_STDOUT_WORLD));
  y_equal = PETSC_FALSE;
  PetscCall(VecEqual(y_test, y, &y_equal));
  if (!y_equal) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
  }
  PetscCall(VecDestroy(&y_test));

  PetscCall(VecDestroy(&y));
  for (i=0; i<nx; i++) {
    PetscCall(VecDestroy(&x[i]));
    PetscCall(ISDestroy(&x_is[i]));
  }
  PetscCall(PetscFree(x));
  PetscCall(PetscFree(x_is));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
        suffix: serial

    test:
        suffix: parallel
        nsize: 2

    test:
        suffix: cuda
        nsize: 2
        args: -vec_type cuda
        requires: cuda

    test:
        suffix: uneven
        nsize: 5

TEST*/
