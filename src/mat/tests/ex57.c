
static char help[] = "Reads in a binary file, extracts a submatrix from it, and writes to another binary file.\n\
Options:\n\
  -fin  <mat>  : input matrix file\n\
  -fout <mat>  : output marrix file\n\
  -start <row> : the row from where the submat should be extracted\n\
  -m  <sx>  : the size of the submatrix\n";

#include <petscmat.h>
#include <petscvec.h>

int main(int argc, char **args)
{
  char        fin[PETSC_MAX_PATH_LEN], fout[PETSC_MAX_PATH_LEN] = "default.mat";
  PetscViewer fdin, fdout;
  Vec         b;
  MatType     mtype = MATSEQBAIJ;
  Mat         A, *B;
  PetscInt    start = 0;
  PetscInt    m;
  IS          isrow, iscol;
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fin", fin, sizeof(fin), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -fin option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fin, FILE_MODE_READ, &fdin));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-fout", fout, sizeof(fout), &flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Writing submatrix to file : %s\n", fout));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fout, FILE_MODE_WRITE, &fdout));

  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetType(A, mtype));
  PetscCall(MatLoad(A, fdin));
  PetscCall(PetscViewerDestroy(&fdin));

  PetscCall(MatGetSize(A, &m, &m));
  m /= 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-start", &start, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));

  PetscCall(ISCreateStride(PETSC_COMM_SELF, m, start, 1, &isrow));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, m, start, 1, &iscol));
  PetscCall(MatCreateSubMatrices(A, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatView(B[0], fdout));

  PetscCall(VecCreate(PETSC_COMM_SELF, &b));
  PetscCall(VecSetSizes(b, PETSC_DECIDE, m));
  PetscCall(VecSetFromOptions(b));
  PetscCall(MatView(B[0], fdout));
  PetscCall(PetscViewerDestroy(&fdout));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B[0]));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFree(B));
  PetscCall(ISDestroy(&iscol));
  PetscCall(ISDestroy(&isrow));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -fin ${DATAFILESPATH}/matrices/small -fout joe -start 2 -m 4
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
