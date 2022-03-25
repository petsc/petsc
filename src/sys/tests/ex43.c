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

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscHSetIJCreate(&table));

  /* The following line silences warnings from Clang Static Analyzer */
  PetscCall(PetscHSetIJResize(table,0));

  PetscCall(PetscTimeSubtract(&t_add));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      key.i = PetscMin(i, j);
      key.j = PetscMax(i, j);
      PetscCall(PetscHSetIJQueryAdd(table, key, &flag));
    }
  }
  PetscCall(PetscTimeAdd(&t_add));

  PetscCall(PetscHSetIJGetSize(table,&n));

  PetscCall(PetscTimeSubtract(&t_has));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      key.i = i;
      key.j = j;
      PetscCall(PetscHSetIJHas(table, key, &flag));
    }
  }
  PetscCall(PetscTimeAdd(&t_has));

  PetscCall(PetscTimeSubtract(&t_del));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      key.i = i;
      key.j = j;
      PetscCall(PetscHSetIJQueryDel(table, key, &flag));
    }
  }
  PetscCall(PetscTimeAdd(&t_del));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"N = %" PetscInt_FMT " - table size: %" PetscInt_FMT ", add: %g, has: %g, del: %g\n",N,n,t_add,t_has,t_del));

  PetscCall(PetscHSetIJDestroy(&table));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -N 32

TEST*/
