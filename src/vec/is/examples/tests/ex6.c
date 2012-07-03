/*
       Formatted test for ISGeneral routines.
*/

static char help[] = "Tests ISComplement.\n\n";

#include <petscis.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       i,j,n,cnt=0,rstart,rend,*indices;
  IS             is,isc;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  n = 3*size;                   /* Number of local indices, same on each process. */
  rstart = 3*(size+2)*rank;     /* start of local range */
  rend   = 3*(size+2)*(rank+1); /* end of local range */
  ierr = PetscMalloc(n*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    for (j=0; j<size; j++) indices[cnt++] = rstart+i*(size+2)+j;
  }
  if (cnt != n) SETERRQ(PETSC_COMM_SELF,1,"inconsistent count");
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = ISComplement(is,rstart,rend,&isc);CHKERRQ(ierr);
  ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(isc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&isc);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
