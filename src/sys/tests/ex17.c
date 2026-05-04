static char help[] = "Demonstrates PetscFOpens() and PetscSynchronizedFGets().\n\n";

#include <petscsys.h>
int main(int argc, char **argv)
{
  const char  line1[]    = "hello 1\n";
  const char  line2[]    = "hello 2\n";
  const char  filename[] = "testfile";
  PetscMPIInt rank;
  FILE       *fp;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  // -- Create the file
  PetscCall(PetscFOpen(PETSC_COMM_WORLD, filename, "w", &fp));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, line1));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, line2));
  PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "rank: %d\n", rank)); // Print rankid in order
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, fp));
  PetscCall(PetscFClose(PETSC_COMM_WORLD, fp));

  { // -- Read the file
    char        line[512] = {0};
    PetscBool   line_check;
    PetscMPIInt size;
    PetscInt    line_rank;

    PetscCall(PetscFOpen(PETSC_COMM_WORLD, filename, "r", &fp));
    PetscCall(PetscSynchronizedFGets(PETSC_COMM_WORLD, fp, sizeof(line), line));
    PetscCall(PetscStrncmp(line, line1, sizeof(line1), &line_check));
    PetscCheck(line_check, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Line 1 not read correctly. Got '%s', expected '%s'", line, line1);
    PetscCall(PetscSynchronizedFGets(PETSC_COMM_WORLD, fp, sizeof(line), line));
    PetscCall(PetscStrncmp(line, line2, sizeof(line2), &line_check));
    PetscCheck(line_check, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Line 2 not read correctly. Got '%s', expected '%s'", line, line2);
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    for (PetscInt i = 0; i < size; i++) {
      PetscCall(PetscSynchronizedFGets(PETSC_COMM_WORLD, fp, sizeof(line), line));
      sscanf(line, "rank: %" PetscInt_FMT, &line_rank);
      PetscCheck(i == line_rank, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Did not find correct rank line in file. Expected %" PetscInt_FMT ", found %" PetscInt_FMT, i, line_rank);
    }

    PetscCall(PetscFClose(PETSC_COMM_WORLD, fp));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
    nsize: 3
    output_file: output/empty.out

TEST*/
