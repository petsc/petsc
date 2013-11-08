static char help[] = "Test scalability of PetscHash.\n\n";

#include <petscsys.h>
#include <../src/sys/utils/hash.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscHashIJ    table;
  PetscInt       newp = 0, N = 0, i, j;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, "-N", &N, NULL);CHKERRQ(ierr);
  ierr = PetscHashIJCreate(&table);CHKERRQ(ierr);
  ierr = PetscHashIJSetMultivalued(table, PETSC_FALSE);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscHashIJKey key;
      PetscInt       p;

      key.i = PetscMin(i, j);
      key.j = PetscMax(i, j);
      ierr  = PetscHashIJGet(table, key, &p);CHKERRQ(ierr);
      if (p < 0) {ierr = PetscHashIJAdd(table, key, newp++);CHKERRQ(ierr);}
    }
  }
  ierr = PetscHashIJDestroy(&table);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
