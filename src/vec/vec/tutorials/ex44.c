
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nx",&nx,&flg));
  if (!flg) nx = 3;

  y_size = 0;
  shift = 0;
  CHKERRQ(PetscMalloc1(nx, &x));
  for (i=0; i<nx; i++) {
    x_size = 2*(i + 1);
    y_size += x_size;
    CHKERRQ(VecCreate(PETSC_COMM_WORLD, &x[i]));
    CHKERRQ(VecSetSizes(x[i], PETSC_DECIDE, x_size));
    CHKERRQ(VecSetFromOptions(x[i]));
    CHKERRQ(VecSetUp(x[i]));
    CHKERRQ(PetscMalloc2(x_size, &idx, x_size, &vals));
    for (j=0; j<x_size; j++) {
      idx[j] = j;
      vals[j] = (PetscScalar)(shift + j + 1);
    }
    shift += x_size;
    CHKERRQ(VecSetValues(x[i], x_size, (const PetscInt*)idx, (const PetscScalar*)vals, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(x[i]));
    CHKERRQ(VecAssemblyEnd(x[i]));
    CHKERRQ(PetscFree2(idx, vals));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Original X[%" PetscInt_FMT "] vector\n", i));
    CHKERRQ(VecView(x[i], PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &y));
  CHKERRQ(VecSetSizes(y, PETSC_DECIDE, y_size));
  CHKERRQ(VecSetFromOptions(y));
  CHKERRQ(VecSetUp(y));
  CHKERRQ(PetscMalloc2(y_size, &idx, y_size, &vals));
  for (j=0; j<y_size; j++) {
    idx[j] = j;
    vals[j] = (PetscScalar)(j + 1);
  }
  CHKERRQ(VecSetValues(y, y_size, (const PetscInt*)idx, (const PetscScalar*)vals, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));
  CHKERRQ(PetscFree2(idx, vals));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Expected Y vector\n"));
  CHKERRQ(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- base VecConcatenate() test ----------- */
  CHKERRQ(VecConcatenate(nx, (const Vec*)x, &y_test, &x_is));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for Y = [X[1], X[2], ...]\n"));
  CHKERRQ(VecView(y_test, PETSC_VIEWER_STDOUT_WORLD));
  y_equal = PETSC_FALSE;
  CHKERRQ(VecEqual(y_test, y, &y_equal));
  if (!y_equal) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- test VecConcatenate() without IS (checks for dangling memory from IS) ----------- */
  CHKERRQ(VecDestroy(&y_test));
  CHKERRQ(VecConcatenate(nx, (const Vec*)x, &y_test, NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for Y = [X[1], X[2], ...] w/o IS\n"));
  CHKERRQ(VecView(y_test, PETSC_VIEWER_STDOUT_WORLD));
  y_equal = PETSC_FALSE;
  CHKERRQ(VecEqual(y_test, y, &y_equal));
  if (!y_equal) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- using index sets on expected Y instead of concatenated Y ----------- */
  for (i=0; i<nx; i++) {
    CHKERRQ(VecGetSubVector(y, x_is[i], &x_test));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Testing index set for X[%" PetscInt_FMT "] component\n", i));
    CHKERRQ(VecView(x_test, PETSC_VIEWER_STDOUT_WORLD));
    x_equal = PETSC_FALSE;
    CHKERRQ(VecEqual(x_test, x[i], &x_equal));
    if (!x_equal) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
    }
    CHKERRQ(VecRestoreSubVector(y, x_is[i], &x_test));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- using VecScatter to communicate data from Y to X[i] for all i ----------- */
  for (i=0; i<nx; i++) {
    CHKERRQ(VecDuplicate(x[i], &x_test));
    CHKERRQ(VecZeroEntries(x_test));
    CHKERRQ(VecScatterCreate(y, x_is[i], x[i], NULL, &y_to_x));
    CHKERRQ(VecScatterBegin(y_to_x, y, x_test, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(y_to_x, y, x_test, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&y_to_x));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for Y -> X[%" PetscInt_FMT "]\n", i));
    CHKERRQ(VecView(x_test, PETSC_VIEWER_STDOUT_WORLD));
    x_equal = PETSC_FALSE;
    CHKERRQ(VecEqual(x_test, x[i], &x_equal));
    if (!x_equal) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
    }
    CHKERRQ(VecDestroy(&x_test));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n"));

  /* ---------- using VecScatter to communicate data from X[i] to Y for all i ----------- */
  CHKERRQ(VecZeroEntries(y_test));
  for (i=0; i<nx; i++) {
    CHKERRQ(VecScatterCreate(x[i], NULL, y, x_is[i], &x_to_y));
    CHKERRQ(VecScatterBegin(x_to_y, x[i], y_test, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(x_to_y, x[i], y_test, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&x_to_y));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for X[:] -> Y\n"));
  CHKERRQ(VecView(y_test, PETSC_VIEWER_STDOUT_WORLD));
  y_equal = PETSC_FALSE;
  CHKERRQ(VecEqual(y_test, y, &y_equal));
  if (!y_equal) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n"));
  }
  CHKERRQ(VecDestroy(&y_test));

  CHKERRQ(VecDestroy(&y));
  for (i=0; i<nx; i++) {
    CHKERRQ(VecDestroy(&x[i]));
    CHKERRQ(ISDestroy(&x_is[i]));
  }
  CHKERRQ(PetscFree(x));
  CHKERRQ(PetscFree(x_is));
  CHKERRQ(PetscFinalize());
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
