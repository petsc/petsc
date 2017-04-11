static char help[] = "Test scalability of PetscHash.\n\n";

#include <petscsys.h>
#include <petsc/private/hash.h>

int main(int argc, char **argv)
{
  PetscHashIJ    table;
  PetscInt       newp = 0, N = 0, i, j;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL);CHKERRQ(ierr);
  ierr = PetscHashIJCreate(&table);CHKERRQ(ierr);
  ierr = PetscHashIJSetMultivalued(table, PETSC_FALSE);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscHashIJKey key;
      PetscInt       p;

      key.i = PetscMin(i, j);
      key.j = PetscMax(i, j);
#if 1
      khint_t ret, idx = kh_put(HASHIJ, table->ht, key, &ret);
      if (ret == 1) kh_val(table->ht, idx).n = newp++;
#else
      ierr  = PetscHashIJGet(table, key, &p);CHKERRQ(ierr);
      if (p < 0) {ierr = PetscHashIJAdd(table, key, newp++);CHKERRQ(ierr);}
#endif
    }
  }
  ierr = PetscHashIJDestroy(&table);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
