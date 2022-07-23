
static char help[] = "Tests ISSortGlobal().\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  IS             is;
  PetscInt       n, i, first, last, nmax=100;
  PetscMPIInt    rank;
  PetscRandom    randsizes, randvalues;
  PetscReal      r;
  PetscInt       *keys, *keyscopy, *keyseven, *keyssorted;
  PetscLayout    map, mapeven;
  PetscBool      sorted;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  comm = MPI_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscOptionsBegin(comm, "", "Parallel Sort Test Options", "IS");
  PetscCall(PetscOptionsBoundedInt("-nmax", "Maximum number of keys per process", "ex11.c", nmax, &nmax, NULL,0));
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(comm, &randsizes));
  PetscCall(PetscRandomSetInterval(randsizes, 0., PetscMax(nmax, 1)));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)randsizes,"sizes_"));
  PetscCall(PetscRandomSetFromOptions(randsizes));

  PetscCall(PetscRandomCreate(comm, &randvalues));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)randvalues,"values_"));
  PetscCall(PetscRandomSetFromOptions(randvalues));

  PetscCall(PetscRandomGetValueReal(randsizes, &r));
  n    = (PetscInt) PetscMin(r, nmax);
  PetscCall(PetscRandomSetInterval(randsizes, 0., 1.));
  PetscCall(PetscRandomGetValueReal(randsizes, &r));
  first = PETSC_MIN_INT + 1 + (PetscInt) ((PETSC_MAX_INT - 1) * r);
  PetscCall(PetscRandomGetValueReal(randsizes, &r));
  last = first + (PetscInt) ((PETSC_MAX_INT - 1) * r);

  PetscCall(PetscRandomSetInterval(randvalues, first, last));
  PetscCall(PetscMalloc3(n, &keys, n, &keyscopy, n, &keyssorted));
  for (i = 0; i < n; i++) {
    PetscCall(PetscRandomGetValueReal(randvalues, &r));
    keys[i] = keyscopy[i] = (PetscInt) r;
  }
  PetscCall(ISCreateGeneral(comm, n, keys, PETSC_USE_POINTER, &is));
  PetscCall(ISViewFromOptions(is, NULL, "-keys_view"));

  PetscCall(ISGetLayout(is, &map));
  PetscCall(PetscLayoutCreateFromSizes(map->comm, PETSC_DECIDE, map->N, 1, &mapeven));
  PetscCall(PetscLayoutSetUp(mapeven));
  PetscCall(PetscMalloc1(mapeven->n, &keyseven));

  PetscCall(PetscParallelSortInt(map, mapeven, keys, keyseven));
  PetscCall(PetscParallelSortedInt(mapeven->comm, mapeven->n, keyseven, &sorted));
  PetscCheck(sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  for (i = 0; i < n; i++) PetscCheck(keys[i] == keyscopy[i],PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscParallelSortInt() modified input array");

  PetscCall(PetscParallelSortInt(map, map, keys, keyssorted));
  PetscCall(PetscParallelSortedInt(map->comm, map->n, keyssorted, &sorted));
  PetscCheck(sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  for (i = 0; i < n; i++) PetscCheck(keys[i] == keyscopy[i],PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscParallelSortInt() modified input array");

  PetscCall(PetscParallelSortInt(map, map, keys, keys));
  PetscCall(PetscParallelSortedInt(map->comm, map->n, keys, &sorted));
  PetscCheck(sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  /* TODO */
#if 0
  PetscCall(ISSortGlobal(is));
#endif

  PetscCall(PetscFree(keyseven));
  PetscCall(PetscLayoutDestroy(&mapeven));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscFree3(keys,keyscopy,keyssorted));
  PetscCall(PetscRandomDestroy(&randvalues));
  PetscCall(PetscRandomDestroy(&randsizes));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 2 3 4 5}}
      args: -nmax {{0 1 5 10 100}}

TEST*/
