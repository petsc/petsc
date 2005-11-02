
static char help[] = "Tests AOData loading.\n\n";

#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AOData      aodata;
  PetscViewer      binary;
  int         ierr,indices[4],*intv,i,rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*
        Load the database from the file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"dataoutput",FILE_MODE_READ,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&aodata);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);

  /*
        Access part of the data 
  */
  indices[0] = 0; indices[1] = 2; indices[2] = 1; indices[3] = 5;
  ierr = AODataSegmentGet(aodata,"key1","seg1",4,indices,(void **)&intv);CHKERRQ(ierr);
  for (i=0; i<4; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] %D %D\n",rank,i,intv[i]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = AODataSegmentRestore(aodata,"key1","seg1",4,indices,(void **)&intv);CHKERRQ(ierr);
 
  ierr = AODataDestroy(aodata);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 


