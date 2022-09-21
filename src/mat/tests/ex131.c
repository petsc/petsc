
static char help[] = "Tests MatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A;
  Vec         x, b;
  PetscViewer fd;                       /* viewer */
  char        file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  /* Determine file from which we read the matrix A */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");

  /* Load matrix A */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  flg = PETSC_FALSE;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vec", file, sizeof(file), &flg));
  if (flg) {
    if (file[0] == '0') {
      PetscInt    m;
      PetscScalar one = 1.0;
      PetscCall(PetscInfo(0, "Using vector of ones for RHS\n"));
      PetscCall(MatGetLocalSize(A, &m, NULL));
      PetscCall(VecSetSizes(x, m, PETSC_DECIDE));
      PetscCall(VecSetFromOptions(x));
      PetscCall(VecSet(x, one));
    }
  } else {
    PetscCall(VecLoad(x, fd));
    PetscCall(PetscViewerDestroy(&fd));
  }
  PetscCall(VecDuplicate(x, &b));
  PetscCall(MatMult(A, x, b));

  /* Print (for testing only) */
  PetscCall(MatView(A, 0));
  PetscCall(VecView(b, 0));
  /* Free data structures */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}
