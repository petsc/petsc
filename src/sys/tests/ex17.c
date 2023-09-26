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
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  // -- Create the file
  PetscCall(PetscFOpen(comm, filename, "w", &fp));
  PetscCall(PetscFPrintf(comm, fp, line1));
  PetscCall(PetscFPrintf(comm, fp, line2));
  PetscCall(PetscSynchronizedFPrintf(comm, fp, "rank: %d\n", rank)); // Print rankid in order
  PetscCall(PetscSynchronizedFlush(comm, fp));
  PetscCall(PetscFClose(comm, fp));

  { // -- Read the file
    char        line[512] = {0};
    PetscBool   line_check;
    PetscMPIInt size;
    PetscInt    line_rank;

    PetscCall(PetscFOpen(comm, filename, "r", &fp));
    PetscCall(PetscSynchronizedFGets(comm, fp, sizeof(line), line));
    PetscCall(PetscStrncmp(line, line1, sizeof(line1), &line_check));
    PetscCheck(line_check, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Line 1 not read correctly. Got '%s', expected '%s'", line, line1);
    PetscCall(PetscSynchronizedFGets(comm, fp, sizeof(line), line));
    PetscCall(PetscStrncmp(line, line2, sizeof(line2), &line_check));
    PetscCheck(line_check, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Line 2 not read correctly. Got '%s', expected '%s'", line, line2);
    PetscCallMPI(MPI_Comm_size(comm, &size));
    for (PetscInt i = 0; i < size; i++) {
      PetscCall(PetscSynchronizedFGets(comm, fp, sizeof(line), line));
      sscanf(line, "rank: %" PetscInt_FMT, &line_rank);
      PetscCheck(i == line_rank, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Did not find correct rank line in file. Expected %" PetscInt_FMT ", found %" PetscInt_FMT, i, line_rank);
    }

    PetscCall(PetscFClose(comm, fp));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
    nsize: 3

TEST*/
