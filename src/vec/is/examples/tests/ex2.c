/*$Id: ex2.c,v 1.16 2000/10/24 20:24:53 bsmith Exp bsmith $*/

/*
       Formatted test for ISStride routines.
*/

static char help[] = "Tests IS stride routines\n\n";

#include "petscis.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        i,n,ierr,*ii,start,stride;
  IS         is;
  PetscTruth flg;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /*
     Test IS of size 0 
  */
  ierr = ISCreateStride(PETSC_COMM_SELF,0,0,2,&is);CHKERRQ(ierr);
  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  if (n != 0) SETERRQ(1,0);
  ierr = ISStrideGetInfo(is,&start,&stride);CHKERRQ(ierr);
  if (start != 0) SETERRQ(1,0);
  if (stride != 2) SETERRQ(1,0);
  ierr = ISStride(is,&flg);CHKERRQ(ierr);
  if (flg != PETSC_TRUE) SETERRQ(1,0);
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
    if (ii[i] != -8 + 3*i) SETERRQ(1,0);
  }
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
 






