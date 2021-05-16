
static char help[] = "Tests PetscOptionsGetScalar(), PetscOptionsScalarArray() for complex numbers\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt    ierr,n,i;
  PetscScalar a,array[10];
  PetscReal   rarray[10];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetScalar(NULL,NULL,"-a",&a,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",(double)PetscRealPart(a),(double)PetscImaginaryPart(a));CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"test options",NULL);CHKERRQ(ierr);
  n = 10; /* max num of input values */
  ierr = PetscOptionsRealArray("-rarray", "Input a real array", "ex14.c", rarray, &n, NULL);CHKERRQ(ierr);
  if (n) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Real rarray of length %d\n",n);CHKERRQ(ierr);
    for (i=0; i<n; i++){
      ierr = PetscPrintf(PETSC_COMM_SELF," %g,\n",rarray[i]);CHKERRQ(ierr);
    }
  }

  n = 10; /* max num of input values */
  ierr = PetscOptionsScalarArray("-array", "Input a scalar array", "ex14.c", array, &n, NULL);CHKERRQ(ierr);
  if (n) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Scalar rarray of length %d\n",n);CHKERRQ(ierr);
    for (i=0; i<n; i++){
      if (PetscImaginaryPart(array[i]) < 0.0) {
        ierr = PetscPrintf(PETSC_COMM_SELF," %g - %gi\n",(double)PetscRealPart(array[i]),(double)PetscAbsReal(PetscImaginaryPart(array[i])));CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF," %g + %gi\n",(double)PetscRealPart(array[i]),(double)PetscImaginaryPart(array[i]));CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: complex
      args: -array 1.0,-2-3i,4.5+6.2i,4.5,6.8+4i,i,-i,-1.2i -rarray 1,2,3 -a 1.5+2.1i

TEST*/
