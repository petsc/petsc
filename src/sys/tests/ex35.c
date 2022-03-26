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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"1st test\n"));
  for (i=0; i<5; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"---------------\n"));
  PetscCall(PetscSortReal(5,x));
  for (i=0; i<5; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n2nd test\n"));
  for (i=0; i<10; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"---------------\n"));
  PetscCall(PetscSortReal(10,x));
  for (i=0; i<10; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %g\n",(double)x[i]));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n3rd test\n"));
  for (i=0; i<5; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index2[i], (double)x2[i]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"---------------\n"));
  PetscCall(PetscSortRealWithArrayInt(5, x2, index2));
  for (i=0; i<5; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index2[i], (double)x2[i]));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n4th test\n"));
  for (i=0; i<10; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index3[i], (double)x3[i]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"---------------\n"));
  PetscCall(PetscSortRealWithArrayInt(10, x3, index3));
  for (i=0; i<10; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %2" PetscInt_FMT "     %g\n",index3[i], (double)x3[i]));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n5th test\n"));
  val  = 44;
  PetscCall(PetscFindReal(val,10,x3,PETSC_SMALL,&loc));
  PetscCall(PetscPrintf(PETSC_COMM_SELF," %g in array: loc %" PetscInt_FMT "\n",(double)val,loc));
  val  = 309.2;
  PetscCall(PetscFindReal(val,10,x3,PETSC_SMALL,&loc));
  PetscCall(PetscPrintf(PETSC_COMM_SELF," %g in array: loc %" PetscInt_FMT "\n",(double)val,loc));
  val  = 309.2;
  PetscCall(PetscFindReal(val,10,x3,0.21,&loc));
  PetscCall(PetscPrintf(PETSC_COMM_SELF," %g in array: loc %" PetscInt_FMT "\n",(double)val,loc));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
