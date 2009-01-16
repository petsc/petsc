
/*
       Formatted test for ISStride routines.
*/

static char help[] = "Tests IS stride routines.\n\n";

#include "petscis.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       i,n,start,stride;
  const PetscInt *ii;
  IS             is;
  PetscTruth     flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /*
     Test IS of size 0 
  */
  ierr = ISCreateStride(PETSC_COMM_SELF,0,0,2,&is);CHKERRQ(ierr);
  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  if (n != 0) SETERRQ(1,"ISCreateStride");
  ierr = ISStrideGetInfo(is,&start,&stride);CHKERRQ(ierr);
  if (start != 0) SETERRQ(1,"ISStrideGetInfo");
  if (stride != 2) SETERRQ(1,"ISStrideGetInfo");
  ierr = ISStride(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"ISStride");
  ierr = ISGetIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  /*
     Test ISGetIndices()
  */
  ierr = ISCreateStride(PETSC_COMM_SELF,10000,-8,3,&is);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ii);CHKERRQ(ierr);
  for (i=0; i<10000; i++) {
    if (ii[i] != -8 + 3*i) SETERRQ(1,"ISGetIndices");
  }
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 






