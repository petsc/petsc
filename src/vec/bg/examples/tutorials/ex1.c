static const char help[] = "Test bipartite graph communication (PetscBG)\n\n";

/*T
    Description: Creates a bipartite graph based on a set of integers, communicates broadcasts values using the graph,
    views the graph, then destroys it.
T*/

/*
  Include petscbg.h so we can use PetscBG objects. Note that this automatically
  includes petscsys.h.
*/
#include <petscbg.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       nlocal;
  PetscBGNode    *remote;
  PetscMPIInt    rank,size;
  PetscBG        bg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  nlocal = 2;
  ierr = PetscMalloc(nlocal*sizeof(*remote),&remote);CHKERRQ(ierr);
  remote[0].rank = (rank+size-1)%size;
  remote[0].index = 1;
  remote[1].rank = (rank+1)%size;
  remote[1].index = 0;

  ierr = PetscBGCreate(PETSC_COMM_WORLD,&bg);CHKERRQ(ierr);
  ierr = PetscBGSetGraph(bg,nlocal,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscBGView(bg,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscBGDestroy(&bg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
