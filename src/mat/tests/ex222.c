static char help[] = "Tests MatComputeOperator() and MatComputeOperatorTranspose()\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat       A, Ae, Aet;
  char      filename[PETSC_MAX_PATH_LEN];
  char      expltype[128], *etype = NULL;
  PetscInt  bs = 1;
  PetscBool flg, check = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-expl_type", expltype, sizeof(expltype), &flg));
  if (flg) PetscCall(PetscStrallocpy(expltype, &etype));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  if (!flg) {
    PetscInt M = 13, N = 6;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A));
    PetscCall(MatSetBlockSize(A, bs));
    PetscCall(MatSetRandom(A, NULL));
  } else {
    PetscViewer viewer;

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetBlockSize(A, bs));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(PetscObjectSetName((PetscObject)A, "Matrix"));
  PetscCall(MatViewFromOptions(A, NULL, "-view_expl"));

  PetscCall(MatComputeOperator(A, etype, &Ae));
  PetscCall(PetscObjectSetName((PetscObject)Ae, "Explicit matrix"));
  PetscCall(MatViewFromOptions(Ae, NULL, "-view_expl"));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check", &check, NULL));
  if (check) {
    Mat       A2;
    PetscReal err, tol = PETSC_SMALL;

    PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
    PetscCall(MatConvert(A, etype, MAT_INITIAL_MATRIX, &A2));
    PetscCall(MatAXPY(A2, -1.0, Ae, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(A2, NORM_FROBENIUS, &err));
    if (err > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error %g > %g (type %s)\n", (double)err, (double)tol, etype));
    PetscCall(MatDestroy(&A2));
  }

  PetscCall(MatComputeOperatorTranspose(A, etype, &Aet));
  PetscCall(PetscObjectSetName((PetscObject)Aet, "Explicit matrix transpose"));
  PetscCall(MatViewFromOptions(Aet, NULL, "-view_expl"));

  PetscCall(PetscFree(etype));
  PetscCall(MatDestroy(&Ae));
  PetscCall(MatDestroy(&Aet));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex222_null.out

   testset:
     suffix: matexpl_rect
     output_file: output/ex222_null.out
     nsize: {{1 3}}
     args: -expl_type {{dense aij baij}}

   testset:
     suffix: matexpl_square
     output_file: output/ex222_null.out
     nsize: {{1 3}}
     args: -bs {{1 2 3}} -M 36 -N 36 -expl_type {{dense aij baij sbaij}}

TEST*/
