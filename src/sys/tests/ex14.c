
static char help[] = "Tests PetscOptionsGetScalar(), PetscOptionsScalarArray() for complex numbers\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt    n,i;
  PetscScalar a,array[10];
  PetscReal   rarray[10];

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetScalar(NULL,NULL,"-a",&a,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",(double)PetscRealPart(a),(double)PetscImaginaryPart(a)));

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"test options",NULL);
  n = 10; /* max num of input values */
  PetscCall(PetscOptionsRealArray("-rarray", "Input a real array", "ex14.c", rarray, &n, NULL));
  if (n) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Real rarray of length %" PetscInt_FMT "\n",n));
    for (i=0; i<n; i++) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF," %g,\n",(double)rarray[i]));
    }
  }

  n = 10; /* max num of input values */
  PetscCall(PetscOptionsScalarArray("-array", "Input a scalar array", "ex14.c", array, &n, NULL));
  if (n) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Scalar rarray of length %" PetscInt_FMT "\n",n));
    for (i=0; i<n; i++) {
      if (PetscImaginaryPart(array[i]) < 0.0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF," %g - %gi\n",(double)PetscRealPart(array[i]),(double)PetscAbsReal(PetscImaginaryPart(array[i]))));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF," %g + %gi\n",(double)PetscRealPart(array[i]),(double)PetscImaginaryPart(array[i])));
      }
    }
  }
  PetscOptionsEnd();
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: complex
      args: -array 1.0,-2-3i,4.5+6.2i,4.5,6.8+4i,i,-i,-1.2i -rarray 1,2,3 -a 1.5+2.1i

TEST*/
