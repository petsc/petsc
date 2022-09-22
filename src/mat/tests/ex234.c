static char help[] = "Basic test of various routines with SBAIJ matrices\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  PetscInt    ia[3] = {0, 2, 4};
  PetscInt    ja[4] = {0, 1, 0, 1};
  PetscScalar c[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  PetscMPIInt size;
  Mat         ssbaij, msbaij;
  Vec         x, y;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is an example with two processors only!");
  PetscCall(MatCreate(PETSC_COMM_SELF, &ssbaij));
  PetscCall(MatSetType(ssbaij, MATSEQSBAIJ));
  PetscCall(MatSetBlockSize(ssbaij, 2));
  PetscCall(MatSetSizes(ssbaij, 4, 8, 4, 8));
  PetscCall(MatSeqSBAIJSetPreallocationCSR(ssbaij, 2, ia, ja, c));
  PetscCall(MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD, ssbaij, PETSC_DECIDE, MAT_INITIAL_MATRIX, &msbaij));
  PetscCall(MatView(msbaij, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&msbaij));
  PetscCall(MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD, ssbaij, 4, MAT_INITIAL_MATRIX, &msbaij));
  PetscCall(MatView(msbaij, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatCreateVecs(msbaij, &x, &y));
  PetscCall(VecSet(x, 1));
  PetscCall(MatMult(msbaij, x, y));
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatMultAdd(msbaij, x, x, y));
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatGetDiagonal(msbaij, y));
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&msbaij));
  PetscCall(MatDestroy(&ssbaij));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     nsize: 2
     filter: sed "s?\.??g"

TEST*/
