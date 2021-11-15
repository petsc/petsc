static char help[] = "Tests PetscSortReal(), PetscSortRealWithArrayInt(), PetscFindReal\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,loc;
  PetscReal      val;
  PetscReal      x[] = {39, 9, 19, -39, 29, 309, 209, 390, 12, 11};
  PetscReal      x2[] = {39, 9, 19, -39, 29, 309, 209, 390, 12, 11};
  PetscReal      x3[] = {39, 9, 19, -39, 29, 309, 209, 390, 12, 11};
  PetscInt       index2[] = {1, -1, 4, 12, 13, 14, 0, 7, 9, 11};
  PetscInt       index3[] = {1, -1, 4, 12, 13, 14, 0, 7, 9, 11};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_SELF,"1st test\n");CHKERRQ(ierr);
  for (i=0; i<5; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]);CHKERRQ(ierr);}
  ierr = PetscPrintf(PETSC_COMM_SELF,"---------------\n");CHKERRQ(ierr);
  ierr = PetscSortReal(5,x);CHKERRQ(ierr);
  for (i=0; i<5; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]);CHKERRQ(ierr);}

  ierr = PetscPrintf(PETSC_COMM_SELF,"\n2nd test\n");CHKERRQ(ierr);
  for (i=0; i<10; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]);CHKERRQ(ierr);}
  ierr = PetscPrintf(PETSC_COMM_SELF,"---------------\n");CHKERRQ(ierr);
  ierr = PetscSortReal(10,x);CHKERRQ(ierr);
  for (i=0; i<10; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]);CHKERRQ(ierr);}

  ierr = PetscPrintf(PETSC_COMM_SELF,"\n3rd test\n");CHKERRQ(ierr);
  for (i=0; i<5; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index2[i], (double)x2[i]);CHKERRQ(ierr);}
  ierr = PetscPrintf(PETSC_COMM_SELF,"---------------\n");CHKERRQ(ierr);
  ierr = PetscSortRealWithArrayInt(5, x2, index2);CHKERRQ(ierr);
  for (i=0; i<5; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index2[i], (double)x2[i]);CHKERRQ(ierr);}

  ierr = PetscPrintf(PETSC_COMM_SELF,"\n4th test\n");CHKERRQ(ierr);
  for (i=0; i<10; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index3[i], (double)x3[i]);CHKERRQ(ierr);}
  ierr = PetscPrintf(PETSC_COMM_SELF,"---------------\n");CHKERRQ(ierr);
  ierr = PetscSortRealWithArrayInt(10, x3, index3);CHKERRQ(ierr);
  for (i=0; i<10; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index3[i], (double)x3[i]);CHKERRQ(ierr);}

  ierr = PetscPrintf(PETSC_COMM_SELF,"\n5th test\n");CHKERRQ(ierr);
  val  = 44;
  ierr = PetscFindReal(val,10,x3,PETSC_SMALL,&loc);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF," %g in array: loc %" PetscInt_FMT "\n",(double)val,loc);CHKERRQ(ierr);
  val  = 309.2;
  ierr = PetscFindReal(val,10,x3,PETSC_SMALL,&loc);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF," %g in array: loc %" PetscInt_FMT "\n",(double)val,loc);CHKERRQ(ierr);
  val  = 309.2;
  ierr = PetscFindReal(val,10,x3,0.21,&loc);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF," %g in array: loc %" PetscInt_FMT "\n",(double)val,loc);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
