
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
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&nx,&flg);CHKERRQ(ierr);
  if (!flg) nx = 3;

  y_size = 0;
  shift = 0;
  ierr = PetscMalloc1(nx, &x);CHKERRQ(ierr);
  for (i=0; i<nx; i++) {
    x_size = 2*(i + 1);
    y_size += x_size;
    ierr = VecCreate(PETSC_COMM_WORLD, &x[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(x[i], PETSC_DECIDE, x_size);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x[i]);CHKERRQ(ierr);
    ierr = VecSetUp(x[i]);CHKERRQ(ierr);
    ierr = PetscMalloc2(x_size, &idx, x_size, &vals);CHKERRQ(ierr);
    for (j=0; j<x_size; j++) {
      idx[j] = j;
      vals[j] = (PetscScalar)(shift + j + 1);
    }
    shift += x_size;
    ierr = VecSetValues(x[i], x_size, (const PetscInt*)idx, (const PetscScalar*)vals, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x[i]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x[i]);CHKERRQ(ierr);
    ierr = PetscFree2(idx, vals);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Original X[%" PetscInt_FMT "] vector\n", i);CHKERRQ(ierr);
    ierr = VecView(x[i], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecCreate(PETSC_COMM_WORLD, &y);CHKERRQ(ierr);
  ierr = VecSetSizes(y, PETSC_DECIDE, y_size);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSetUp(y);CHKERRQ(ierr);
  ierr = PetscMalloc2(y_size, &idx, y_size, &vals);CHKERRQ(ierr);
  for (j=0; j<y_size; j++) {
    idx[j] = j;
    vals[j] = (PetscScalar)(j + 1);
  }
  ierr = VecSetValues(y, y_size, (const PetscInt*)idx, (const PetscScalar*)vals, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = PetscFree2(idx, vals);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Expected Y vector\n");CHKERRQ(ierr);
  ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n");CHKERRQ(ierr);

  /* ---------- base VecConcatenate() test ----------- */
  ierr = VecConcatenate(nx, (const Vec*)x, &y_test, &x_is);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for Y = [X[1], X[2], ...]\n");CHKERRQ(ierr);
  ierr = VecView(y_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  y_equal = PETSC_FALSE;
  ierr = VecEqual(y_test, y, &y_equal);CHKERRQ(ierr);
  if (!y_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n");CHKERRQ(ierr);

  /* ---------- test VecConcatenate() without IS (checks for dangling memory from IS) ----------- */
  ierr = VecDestroy(&y_test);CHKERRQ(ierr);
  ierr = VecConcatenate(nx, (const Vec*)x, &y_test, NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecConcatenate() for Y = [X[1], X[2], ...] w/o IS\n");CHKERRQ(ierr);
  ierr = VecView(y_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  y_equal = PETSC_FALSE;
  ierr = VecEqual(y_test, y, &y_equal);CHKERRQ(ierr);
  if (!y_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n");CHKERRQ(ierr);

  /* ---------- using index sets on expected Y instead of concatenated Y ----------- */
  for (i=0; i<nx; i++) {
    ierr = VecGetSubVector(y, x_is[i], &x_test);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing index set for X[%" PetscInt_FMT "] component\n", i);CHKERRQ(ierr);
    ierr = VecView(x_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    x_equal = PETSC_FALSE;
    ierr = VecEqual(x_test, x[i], &x_equal);CHKERRQ(ierr);
    if (!x_equal) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");CHKERRQ(ierr);
    }
    ierr = VecRestoreSubVector(y, x_is[i], &x_test);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n");CHKERRQ(ierr);

  /* ---------- using VecScatter to communicate data from Y to X[i] for all i ----------- */
  for (i=0; i<nx; i++) {
    ierr = VecDuplicate(x[i], &x_test);CHKERRQ(ierr);
    ierr = VecZeroEntries(x_test);CHKERRQ(ierr);
    ierr = VecScatterCreate(y, x_is[i], x[i], NULL, &y_to_x);CHKERRQ(ierr);
    ierr = VecScatterBegin(y_to_x, y, x_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(y_to_x, y, x_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&y_to_x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for Y -> X[%" PetscInt_FMT "]\n", i);CHKERRQ(ierr);
    ierr = VecView(x_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    x_equal = PETSC_FALSE;
    ierr = VecEqual(x_test, x[i], &x_equal);CHKERRQ(ierr);
    if (!x_equal) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");CHKERRQ(ierr);
    }
    ierr = VecDestroy(&x_test);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------------------------------------------\n");CHKERRQ(ierr);

  /* ---------- using VecScatter to communicate data from X[i] to Y for all i ----------- */
  ierr = VecZeroEntries(y_test);CHKERRQ(ierr);
  for (i=0; i<nx; i++) {
    ierr = VecScatterCreate(x[i], NULL, y, x_is[i], &x_to_y);CHKERRQ(ierr);
    ierr = VecScatterBegin(x_to_y, x[i], y_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(x_to_y, x[i], y_test, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&x_to_y);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Testing VecScatter for X[:] -> Y\n");CHKERRQ(ierr);
  ierr = VecView(y_test, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  y_equal = PETSC_FALSE;
  ierr = VecEqual(y_test, y, &y_equal);CHKERRQ(ierr);
  if (!y_equal) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  FAIL\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "  PASS\n");CHKERRQ(ierr);
  }
  ierr = VecDestroy(&y_test);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  for (i=0; i<nx; i++) {
    ierr = VecDestroy(&x[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&x_is[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(x_is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
