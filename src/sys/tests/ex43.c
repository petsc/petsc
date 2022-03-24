static char help[] = "Test scalability of PetscHSetIJ hash set.\n\n";

#include <petscsys.h>
#include <petsctime.h>
#include <petsc/private/hashsetij.h>

int main(int argc, char **argv)
{
  PetscHSetIJ    table;
  PetscInt       N = 0, i, j, n;
  PetscHashIJKey key;
  PetscBool      flag;
  PetscLogDouble t_add = 0;
  PetscLogDouble t_has = 0;
  PetscLogDouble t_del = 0;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscHSetIJCreate(&table));

  /* The following line silences warnings from Clang Static Analyzer */
  CHKERRQ(PetscHSetIJResize(table,0));

  CHKERRQ(PetscTimeSubtract(&t_add));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      key.i = PetscMin(i, j);
      key.j = PetscMax(i, j);
      CHKERRQ(PetscHSetIJQueryAdd(table, key, &flag));
    }
  }
  CHKERRQ(PetscTimeAdd(&t_add));

  CHKERRQ(PetscHSetIJGetSize(table,&n));

  CHKERRQ(PetscTimeSubtract(&t_has));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      key.i = i;
      key.j = j;
      CHKERRQ(PetscHSetIJHas(table, key, &flag));
    }
  }
  CHKERRQ(PetscTimeAdd(&t_has));

  CHKERRQ(PetscTimeSubtract(&t_del));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      key.i = i;
      key.j = j;
      CHKERRQ(PetscHSetIJQueryDel(table, key, &flag));
    }
  }
  CHKERRQ(PetscTimeAdd(&t_del));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"N = %" PetscInt_FMT " - table size: %" PetscInt_FMT ", add: %g, has: %g, del: %g\n",N,n,t_add,t_has,t_del));

  CHKERRQ(PetscHSetIJDestroy(&table));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -N 32

TEST*/
