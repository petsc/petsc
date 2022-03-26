
static char help[] = "Tests PetscSortIntWithPermutation().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i;
  PetscInt       x[]  = {39, 9, 39, 39, 29},index[5];
  PetscInt       x2[] = {39, 9, 19, 39, 29, 39, 29, 39},index2[8];

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"1st test\n"));
  for (i=0; i<5; i++) index[i] = i;
  PetscCall(PetscSortIntWithPermutation(5, x, index));
  for (i=0; i<5; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %" PetscInt_FMT "     %" PetscInt_FMT "     %" PetscInt_FMT "\n",x[i], index[i],x[index[i]]));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n2nd test\n"));
  for (i=0; i<8; i++) index2[i] = i;
  PetscCall(PetscSortIntWithPermutation(8, x2, index2));
  for (i=0; i<8; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %" PetscInt_FMT "     %" PetscInt_FMT "     %" PetscInt_FMT "\n",x2[i], index2[i],x2[index2[i]]));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
