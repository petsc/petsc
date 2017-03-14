static char help[] = "Tests wrapping of math.h functions for real, complex, and scalar types \n";
#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  {
    PetscReal a,b,c;
    a = 0.5;
    c = 2.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Real tests:\n");CHKERRQ(ierr);
    b = PetscSqrtReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sqrt(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscExpReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscLogReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscLog10Real(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log10(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscSinReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sin(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscCosReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cos(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscTanReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tan(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscAsinReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"asin(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscAcosReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"acos(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscAtanReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"atan(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscSinhReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sinh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscCoshReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cosh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscTanhReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tanh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscPowReal(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"pow(%f,%f) = %f\n",(double)a,(double)c,(double)b);CHKERRQ(ierr);
    b = PetscCeilReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ceil(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscFloorReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"floor(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscFmodReal(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fmod(%f,%f) = %f\n",(double)a,(double)c,(double)b);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Scalar tests:\n");CHKERRQ(ierr);
  {
    PetscScalar a,b,c;
    a = 0.5;
    c = 2.0;
    b = PetscSqrtScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sqrt(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscExpScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscLogScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscPowScalar(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"pow(%f,%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(c),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscSinScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sin(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscCosScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cos(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscTanScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tan(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscSinhScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sinh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscCoshScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cosh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscTanhScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tanh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}


/*TEST
   
   test:

TEST*/
