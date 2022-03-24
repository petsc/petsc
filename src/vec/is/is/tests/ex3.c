/*
       Tests ISAllGather()
*/

static char help[] = "Tests ISAllGather().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt       i,n,*indices;
  PetscMPIInt    rank;
  IS             is,newis;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /*
     Create IS
  */
  n    = 4 + rank;
  CHKERRQ(PetscMalloc1(n,&indices));
  for (i=0; i<n; i++) indices[i] = rank + i;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,indices,PETSC_COPY_VALUES,&is));
  CHKERRQ(PetscFree(indices));

  /*
      Stick them together from all processors
  */
  CHKERRQ(ISAllGather(is,&newis));

  if (rank == 0) {
    CHKERRQ(ISView(newis,PETSC_VIEWER_STDOUT_SELF));
  }

  CHKERRQ(ISDestroy(&newis));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
