static char help[] = "Check MPI error strings. Crashes with known error with MPICH.\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  for (PetscMPIInt err = 1; err <= MPI_ERR_LASTCODE; err++) {
    PetscMPIInt len;
    char        errorstring[MPI_MAX_ERROR_STRING];

    MPI_Error_string(err, (char *)errorstring, &len);
    PetscCheck(len < MPI_MAX_ERROR_STRING, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Error excessive string length from MPI_Error_string()");
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error code %d length %d string %s\n", err, len, errorstring));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: defined(PETSC_HAVE_OPENMPI)
     output_file: output/empty.out

TEST*/
