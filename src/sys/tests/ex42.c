static char help[] = "Test scalability of PetscHSetI hash set.\n\n";

#include <petscsys.h>
#include <petsctime.h>
#include <petsc/private/hashseti.h>

int main(int argc, char **argv)
{
  PetscHSetI     table;
  PetscInt       N = 0, i, j, n;
  PetscBool      flag;
  PetscLogDouble t_add = 0;
  PetscLogDouble t_has = 0;
  PetscLogDouble t_del = 0;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscHSetICreate(&table));

  /* The following line silences warnings from Clang Static Analyzer */
  CHKERRQ(PetscHSetIResize(table,0));

  CHKERRQ(PetscTimeSubtract(&t_add));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i+j*N;
      CHKERRQ(PetscHSetIQueryAdd(table, key, &flag));
    }
  }
  CHKERRQ(PetscTimeAdd(&t_add));

  CHKERRQ(PetscHSetIGetSize(table,&n));

  CHKERRQ(PetscTimeSubtract(&t_has));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i+j*N;
      CHKERRQ(PetscHSetIHas(table, key, &flag));
    }
  }
  CHKERRQ(PetscTimeAdd(&t_has));

  CHKERRQ(PetscTimeSubtract(&t_del));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i+j*N;
      CHKERRQ(PetscHSetIQueryDel(table, key, &flag));
    }
  }
  CHKERRQ(PetscTimeAdd(&t_del));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"N = %" PetscInt_FMT " - table size: %" PetscInt_FMT ", add: %g, has: %g, del: %g\n",N,n,t_add,t_has,t_del));

  CHKERRQ(PetscHSetIDestroy(&table));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -N 32

TEST*/
