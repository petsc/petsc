static char help[] = "Tests the PetscByteSwap()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       oint[2],sint[2];
  PetscBool      obool[2],sbool[2];
  PetscScalar    oscalar[2],sscalar[2];
  double         odouble[2],sdouble[2];
  float          ofloat[2],sfloat[2];
  short          oshort[2],sshort[2];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  sint[0]    = oint[0]    = 5;
  sint[1]    = oint[1]    = 19;
  sbool[0]   = obool[0]   = PETSC_FALSE;
  sbool[1]   = obool[1]   = PETSC_TRUE;
  sscalar[0] = oscalar[0] = 3.14159265;
  sscalar[1] = oscalar[1] = 1.3806504e-23;
  sdouble[0] = odouble[0] = 3.14159265;
  sdouble[1] = odouble[1] = 1.3806504e-23;
  sfloat[0]  = ofloat[0]  = 3.14159265;
  sfloat[1]  = ofloat[1]  = 1.3806504e-23;
  sshort[0]  = oshort[0]  = 5;
  sshort[1]  = oshort[1]  = 19;

  ierr = PetscByteSwap(sint,PETSC_INT,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sbool,PETSC_BOOL,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sscalar,PETSC_SCALAR,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sdouble,PETSC_DOUBLE,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sfloat,PETSC_FLOAT,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sshort,PETSC_SHORT,2);CHKERRQ(ierr);

  ierr = PetscByteSwap(sint,PETSC_INT,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sbool,PETSC_BOOL,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sscalar,PETSC_SCALAR,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sdouble,PETSC_DOUBLE,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sfloat,PETSC_FLOAT,2);CHKERRQ(ierr);
  ierr = PetscByteSwap(sshort,PETSC_SHORT,2);CHKERRQ(ierr);

  if ((sint[0] !=oint[0])|| (sint[1] != oint[1]))             {ierr = PetscPrintf(PETSC_COMM_SELF,"Byteswap mismatch for PETSC_INT\n");CHKERRQ(ierr);}
  if ((sbool[0] !=obool[0])|| (sbool[1] != obool[1]))         {ierr = PetscPrintf(PETSC_COMM_SELF,"Byteswap mismatch for PETSC_BOOL\n");CHKERRQ(ierr);}
  if ((sscalar[0] !=oscalar[0])|| (sscalar[1] != oscalar[1])) {ierr = PetscPrintf(PETSC_COMM_SELF,"Byteswap mismatch for PETSC_SCALAR\n");CHKERRQ(ierr);}
  if ((sdouble[0] !=odouble[0])|| (sdouble[1] != odouble[1])) {ierr = PetscPrintf(PETSC_COMM_SELF,"Byteswap mismatch for PETSC_DOUBLE\n");CHKERRQ(ierr);}
  if ((sfloat[0] !=ofloat[0])|| (sfloat[1] != ofloat[1]))     {ierr = PetscPrintf(PETSC_COMM_SELF,"Byteswap mismatch for PETSC_FLOAT\n");CHKERRQ(ierr);}
  if ((sshort[0] !=oshort[0])|| (sshort[1] != oshort[1]))     {ierr = PetscPrintf(PETSC_COMM_SELF,"Byteswap mismatch for PETSC_SHORT\n");CHKERRQ(ierr);}

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
