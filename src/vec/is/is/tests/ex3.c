/*
       Tests ISAllGather()
*/

static char help[] = "Tests ISAllGather().\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n,*indices;
  PetscMPIInt    rank;
  IS             is,newis;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /*
     Create IS
  */
  n    = 4 + rank;
  ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);
  for (i=0; i<n; i++) indices[i] = rank + i;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /*
      Stick them together from all processors
  */
  ierr = ISAllGather(is,&newis);CHKERRQ(ierr);

  if (!rank) {
    ierr = ISView(newis,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&newis);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
