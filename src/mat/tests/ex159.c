static const char help[] = "Test MatGetLocalSubMatrix() with multiple levels of nesting.\n\n";

#include <petscmat.h>

int main(int argc, char *argv[])
{
  IS          is0a, is0b, is0, is1, isl0a, isl0b, isl0, isl1;
  Mat         A, Aexplicit;
  PetscBool   usenest;
  PetscMPIInt rank, size;
  PetscInt    i, j;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  {
    PetscInt ix0a[1], ix0b[1], ix0[2], ix1[1];

    ix0a[0] = rank * 2 + 0;
    ix0b[0] = rank * 2 + 1;
    ix0[0]  = rank * 3 + 0;
    ix0[1]  = rank * 3 + 1;
    ix1[0]  = rank * 3 + 2;
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 1, ix0a, PETSC_COPY_VALUES, &is0a));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 1, ix0b, PETSC_COPY_VALUES, &is0b));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 2, ix0, PETSC_COPY_VALUES, &is0));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 1, ix1, PETSC_COPY_VALUES, &is1));
  }
  {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 6, 0, 1, &isl0));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 3, 0, 1, &isl0a));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 3, 3, 1, &isl0b));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 3, 6, 1, &isl1));
  }

  usenest = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nest", &usenest, NULL));
  if (usenest) {
    ISLocalToGlobalMapping l2g;
    PetscInt               l2gind[3];
    Mat                    B[9];

    l2gind[0] = (rank - 1 + size) % size;
    l2gind[1] = rank;
    l2gind[2] = (rank + 1) % size;
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, l2gind, PETSC_COPY_VALUES, &l2g));
    for (i = 0; i < 9; i++) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 1, 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, NULL, PETSC_DECIDE, NULL, &B[i]));
      PetscCall(MatSetUp(B[i]));
      PetscCall(MatSetLocalToGlobalMapping(B[i], l2g, l2g));
    }
    {
      IS  isx[2];
      Mat Bx00[4], Bx01[2], Bx10[2];
      Mat B00, B01, B10;

      isx[0]  = is0a;
      isx[1]  = is0b;
      Bx00[0] = B[0];
      Bx00[1] = B[1];
      Bx00[2] = B[3];
      Bx00[3] = B[4];
      Bx01[0] = B[2];
      Bx01[1] = B[5];
      Bx10[0] = B[6];
      Bx10[1] = B[7];

      PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, isx, 2, isx, Bx00, &B00));
      PetscCall(MatSetUp(B00));
      PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, isx, 1, NULL, Bx01, &B01));
      PetscCall(MatSetUp(B01));
      PetscCall(MatCreateNest(PETSC_COMM_WORLD, 1, NULL, 2, isx, Bx10, &B10));
      PetscCall(MatSetUp(B10));
      {
        Mat By[4];
        IS  isy[2];

        By[0]  = B00;
        By[1]  = B01;
        By[2]  = B10;
        By[3]  = B[8];
        isy[0] = is0;
        isy[1] = is1;

        PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, isy, 2, isy, By, &A));
        PetscCall(MatSetUp(A));
      }
      PetscCall(MatDestroy(&B00));
      PetscCall(MatDestroy(&B01));
      PetscCall(MatDestroy(&B10));
    }
    for (i = 0; i < 9; i++) PetscCall(MatDestroy(&B[i]));
    PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  } else {
    ISLocalToGlobalMapping l2g;
    PetscInt               l2gind[9];
    for (i = 0; i < 3; i++)
      for (j = 0; j < 3; j++) l2gind[3 * i + j] = ((rank - 1 + j + size) % size) * 3 + i;
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 9, l2gind, PETSC_COPY_VALUES, &l2g));
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 3, 3, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, NULL, PETSC_DECIDE, NULL, &A));
    PetscCall(MatSetLocalToGlobalMapping(A, l2g, l2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  }

  {
    Mat A00, A11, A0a0a, A0a0b;
    PetscCall(MatGetLocalSubMatrix(A, isl0, isl0, &A00));
    PetscCall(MatGetLocalSubMatrix(A, isl1, isl1, &A11));
    PetscCall(MatGetLocalSubMatrix(A00, isl0a, isl0a, &A0a0a));
    PetscCall(MatGetLocalSubMatrix(A00, isl0a, isl0b, &A0a0b));

    PetscCall(MatSetValueLocal(A0a0a, 0, 0, 100 * rank + 1, ADD_VALUES));
    PetscCall(MatSetValueLocal(A0a0a, 0, 1, 100 * rank + 2, ADD_VALUES));
    PetscCall(MatSetValueLocal(A0a0a, 2, 2, 100 * rank + 9, ADD_VALUES));

    PetscCall(MatSetValueLocal(A0a0b, 1, 1, 100 * rank + 50 + 5, ADD_VALUES));

    PetscCall(MatSetValueLocal(A11, 0, 0, 1000 * (rank + 1) + 1, ADD_VALUES));
    PetscCall(MatSetValueLocal(A11, 1, 2, 1000 * (rank + 1) + 6, ADD_VALUES));

    PetscCall(MatRestoreLocalSubMatrix(A00, isl0a, isl0a, &A0a0a));
    PetscCall(MatRestoreLocalSubMatrix(A00, isl0a, isl0b, &A0a0b));
    PetscCall(MatRestoreLocalSubMatrix(A, isl0, isl0, &A00));
    PetscCall(MatRestoreLocalSubMatrix(A, isl1, isl1, &A11));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatComputeOperator(A, MATAIJ, &Aexplicit));
  PetscCall(MatView(Aexplicit, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Aexplicit));
  PetscCall(ISDestroy(&is0a));
  PetscCall(ISDestroy(&is0b));
  PetscCall(ISDestroy(&is0));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&isl0a));
  PetscCall(ISDestroy(&isl0b));
  PetscCall(ISDestroy(&isl0));
  PetscCall(ISDestroy(&isl1));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

   test:
      suffix: nest
      nsize: 3
      args: -nest

TEST*/
