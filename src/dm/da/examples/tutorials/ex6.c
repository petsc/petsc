
static char help[] = "Demonstrates using 3 DA's to manage a slightly non-trivial grid";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       p1 = 6, p2 = 2, r1 = 3, r2 = 2,r1g,r2g,sw = 1;
  PetscErrorCode ierr;

  /* Each DA manages the local vector for the portion of region 1, 2, and 3 for that processor
     Each DA can belong on any subset (overlapping between DA's or not) of processors
     For processes that a particular DA does not exist on, the corresponding dak should be set to zero */
  DA             da1,da2,da3;
  MPI_Comm       comm1 = PETSC_COMM_WORLD,comm2 = PETSC_COMM_WORLD,comm3 = PETSC_COMM_WORLD;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /*
      sw is the stencil width  

      p1 is the width of region 1, p2 the width of region 2
      r1 height of region 1 
      r2 height of region 2
 
      r2 is also the height of region 3-4
      (p1 - p2)/2 is the width of both region 3 and region 4
  */
  r1g = r1 + sw;
  r2g = r2 + sw;
  if (p2 > p1 - 2) SETERRQ(1,"Width of region p2 must be at least 2 less then width of region 1");
  if ((p2 - p1) % 2) SETERRQ(1,"width of region 3 must be divisible by 2");

  ierr = DACreate2d(comm1,DA_NONPERIODIC,DA_STENCIL_BOX,p1,r1g,PETSC_DECIDE,PETSC_DECIDE,1,sw,PETSC_NULL,PETSC_NULL,&da1);CHKERRQ(ierr);
  ierr = DACreate2d(comm2,DA_XPERIODIC,DA_STENCIL_BOX,p2,r2g,PETSC_DECIDE,PETSC_DECIDE,1,sw,PETSC_NULL,PETSC_NULL,&da1);CHKERRQ(ierr);
  ierr = DACreate2d(comm3,DA_NONPERIODIC,DA_STENCIL_BOX,p1-p2,r2g,PETSC_DECIDE,PETSC_DECIDE,1,sw,PETSC_NULL,PETSC_NULL,&da1);CHKERRQ(ierr);

  ierr = DADestroy(da1);CHKERRQ(ierr);
  ierr = DADestroy(da2);CHKERRQ(ierr);
  ierr = DADestroy(da3);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
