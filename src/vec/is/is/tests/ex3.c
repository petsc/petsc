/*
       Tests ISAllGather()
*/

static char help[] = "Tests ISAllGather().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscInt    i, n, *indices;
  PetscMPIInt rank;
  IS          is, newis;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /*
     Create IS
  */
  n = 4 + rank;
  PetscCall(PetscMalloc1(n, &indices));
  for (i = 0; i < n; i++) indices[i] = rank + i;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, n, indices, PETSC_COPY_VALUES, &is));
  PetscCall(PetscFree(indices));

  /*
      Stick them together from all processors
  */
  PetscCall(ISAllGather(is, &newis));

  if (rank == 0) PetscCall(ISView(newis, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(ISDestroy(&newis));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
