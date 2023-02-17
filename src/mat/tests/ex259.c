static char help[] = "Test of setting values in a matrix without preallocation\n\n";

#include <petscmat.h>

PetscErrorCode ex1_nonsquare_bs1(void)
{
  Mat       A;
  PetscInt  M, N, m, n, bs = 1;
  char      type[16];
  PetscBool flg;

  /*
     Create the matrix
  */
  PetscFunctionBegin;
  M = 10;
  N = 12;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-type", type, sizeof(type), &flg));
  if (flg) PetscCall(MatSetType(A, type));
  else PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSetFromOptions(A));

  /*
     Get the sizes of the matrix
  */
  PetscCall(MatGetLocalSize(A, &m, &n));

  /*
     Insert non-zero pattern (e.g. perform a sweep over the grid).
     You can use MatSetValues(), MatSetValuesBlocked() or MatSetValue().
  */
  {
    PetscInt    ii, jj;
    PetscScalar vv = 22.0;

    ii = 3;
    jj = 3;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));

    ii = 7;
    jj = 4;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));
    PetscCall(MatSetValue(A, jj, ii, vv, INSERT_VALUES));

    ii = 9;
    jj = 7;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));
    PetscCall(MatSetValue(A, jj, ii, vv, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  /*
     Insert same location non-zero values into A.
  */
  {
    PetscInt    ii, jj;
    PetscScalar vv;

    ii = 3;
    jj = 3;
    vv = 0.3;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));

    ii = 7;
    jj = 4;
    vv = 3.3;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));
    PetscCall(MatSetValue(A, jj, ii, vv, INSERT_VALUES));

    ii = 9;
    jj = 7;
    vv = 4.3;
    PetscCall(MatSetValue(A, ii, jj, vv, INSERT_VALUES));
    PetscCall(MatSetValue(A, jj, ii, vv, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(ex1_nonsquare_bs1());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     args: -bs {{1 2}} -type {{aij baij sbaij}}
     filter: grep -v "type:"
     test:
     test:
       suffix: 2
       nsize: 2

TEST*/
