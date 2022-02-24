
static char help[] = "Tests PetscOptionsGetScalar(), PetscOptionsScalarArray() for complex numbers\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt    ierr,n,i;
  PetscScalar a,array[10];
  PetscReal   rarray[10];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-a",&a,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",(double)PetscRealPart(a),(double)PetscImaginaryPart(a)));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"test options",NULL);CHKERRQ(ierr);
  n = 10; /* max num of input values */
  CHKERRQ(PetscOptionsRealArray("-rarray", "Input a real array", "ex14.c", rarray, &n, NULL));
  if (n) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Real rarray of length %" PetscInt_FMT "\n",n));
    for (i=0; i<n; i++) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %g,\n",(double)rarray[i]));
    }
  }

  n = 10; /* max num of input values */
  CHKERRQ(PetscOptionsScalarArray("-array", "Input a scalar array", "ex14.c", array, &n, NULL));
  if (n) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Scalar rarray of length %" PetscInt_FMT "\n",n));
    for (i=0; i<n; i++) {
      if (PetscImaginaryPart(array[i]) < 0.0) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %g - %gi\n",(double)PetscRealPart(array[i]),(double)PetscAbsReal(PetscImaginaryPart(array[i]))));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %g + %gi\n",(double)PetscRealPart(array[i]),(double)PetscImaginaryPart(array[i])));
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
