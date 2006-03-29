
static char help[] = "Tests PetscRandom functions.\n\n";

#include "petsc.h"
#include "petscsys.h"

/* Usage: 
   ./ex1 -log_summary
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       i,n = 1000,*values;
  int            event;
  PetscRandom    rand;
  PetscScalar    value;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetType(rand,PETSC_RAND48);CHKERRQ(ierr); 
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr); 

  ierr = PetscMalloc(n*sizeof(PetscInt),&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscRandomSeed(rand);CHKERRQ(ierr);
    ierr = PetscRandomGetValue(rand,&value);CHKERRQ(ierr);
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);

  ierr = PetscLogEventRegister(&event,"Sort",0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
 
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rand,&value);CHKERRQ(ierr);
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
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
 
