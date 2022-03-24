
static char help[] = "Tests PetscSortIntWithPermutation().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i;
  PetscInt       x[]  = {39, 9, 39, 39, 29},index[5];
  PetscInt       x2[] = {39, 9, 19, 39, 29, 39, 29, 39},index2[8];

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"1st test\n"));
  for (i=0; i<5; i++) index[i] = i;
  CHKERRQ(PetscSortIntWithPermutation(5, x, index));
  for (i=0; i<5; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %" PetscInt_FMT "     %" PetscInt_FMT "     %" PetscInt_FMT "\n",x[i], index[i],x[index[i]]));

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n2nd test\n"));
  for (i=0; i<8; i++) index2[i] = i;
  CHKERRQ(PetscSortIntWithPermutation(8, x2, index2));
  for (i=0; i<8; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %" PetscInt_FMT "     %" PetscInt_FMT "     %" PetscInt_FMT "\n",x2[i], index2[i],x2[index2[i]]));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
