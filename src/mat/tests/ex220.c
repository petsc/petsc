
#include <petscmat.h>

static char help[PETSC_MAX_PATH_LEN] = "Tests MatLoad() with MatCreateDense() for memory leak ";

int main(int argc, char **argv)
{
    PetscViewer         viewer;
    Mat                 A;
    char                filename[PETSC_MAX_PATH_LEN];
    PetscBool           flg;

    PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &flg));
    PetscCheck(flg,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a filename for input with the -f option");

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 36, 36, NULL, &A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

     test:
       requires: double !complex !defined(PETSC_USE_64BIT_INDICES) datafilespath
       args: -f ${DATAFILESPATH}/matrices/small

TEST*/
