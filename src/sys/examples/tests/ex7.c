
static char help[] = "Tests PetscSortIntWithPermutation().\n\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       i;
  PetscInt       x[] = {39, 9, 39, 39, 29},index[5];
  PetscInt       x2[] = {39, 9, 19, 39, 29, 39, 29, 39},index2[8];
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  printf("1st test\n");
  for (i=0; i<5; i++) index[i] = i;
  ierr = PetscSortIntWithPermutation( 5, x, index);
  for (i=0; i<5; i++) printf(" %d     %d     %d\n",x[i], index[i],x[index[i]]);

  printf("\n2nd test\n");
  for (i=0; i<8; i++) index2[i] = i;
  ierr = PetscSortIntWithPermutation( 8, x2, index2);
  for (i=0; i<8; i++) printf(" %d     %d     %d\n",x2[i], index2[i],x2[index2[i]]);
  ierr = PetscFinalize();
  return 0;
}

