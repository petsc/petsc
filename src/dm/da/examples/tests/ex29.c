
static char help[] = "Tests DA wirebasket interpolation.\n\n";

#include "petscda.h"
#include "petscsys.h"

extern PetscErrorCode DAGetWireBasket(DA,Mat);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DA             da;
  Mat            Aglobal;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,3,4,5,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATAIJ,&Aglobal);CHKERRQ(ierr);

  ierr = DAGetWireBasket(da,Aglobal);CHKERRQ(ierr);

  ierr = MatDestroy(Aglobal);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 



