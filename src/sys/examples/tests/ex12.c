
static char help[] = "Tests timing PetscSortInt().\n\n";

#include "petsc.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int         ierr,i,n = 1000,*values,event;
  PetscRandom rand;
  PetscScalar value;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRQ(ierr);

  ierr = PetscMalloc(n*sizeof(int),&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rand,&value);CHKERRQ(ierr);
    values[i] = (int)(n*PetscRealPart(value) + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);

  ierr = PetscLogEventRegister(&event,"Sort",0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(int),&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rand,&value);CHKERRQ(ierr);
    values[i] = (int)(n*PetscRealPart(value) + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  for (i=1; i<n; i++) {
    if (values[i] < values[i-1]) SETERRQ(1,"Values not sorted");
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
