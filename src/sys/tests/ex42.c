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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscHSetICreate(&table));

  /* The following line silences warnings from Clang Static Analyzer */
  PetscCall(PetscHSetIResize(table, 0));

  PetscCall(PetscTimeSubtract(&t_add));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i + j * N;
      PetscCall(PetscHSetIQueryAdd(table, key, &flag));
    }
  }
  PetscCall(PetscTimeAdd(&t_add));

  PetscCall(PetscHSetIGetSize(table, &n));

  PetscCall(PetscTimeSubtract(&t_has));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i + j * N;
      PetscCall(PetscHSetIHas(table, key, &flag));
    }
  }
  PetscCall(PetscTimeAdd(&t_has));

  PetscCall(PetscTimeSubtract(&t_del));
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      PetscInt key = i + j * N;
      PetscCall(PetscHSetIQueryDel(table, key, &flag));
    }
  }
  PetscCall(PetscTimeAdd(&t_del));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "N = %" PetscInt_FMT " - table size: %" PetscInt_FMT ", add: %g, has: %g, del: %g\n", N, n, t_add, t_has, t_del));

  PetscCall(PetscHSetIDestroy(&table));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -N 32

TEST*/
