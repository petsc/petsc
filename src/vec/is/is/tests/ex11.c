
static char help[] = "Tests ISSortGlobal().\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  IS             is;
  PetscInt       n, i, first, last, nmax=100;
  PetscMPIInt    rank;
  PetscRandom    randsizes, randvalues;
  PetscReal      r;
  PetscInt       *keys, *keyscopy, *keyseven, *keyssorted;
  PetscLayout    map, mapeven;
  PetscBool      sorted;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = MPI_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  ierr = PetscOptionsBegin(comm, "", "Parallel Sort Test Options", "IS");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-nmax", "Maximum number of keys per process", "ex11.c", nmax, &nmax, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscRandomCreate(comm, &randsizes));
  CHKERRQ(PetscRandomSetInterval(randsizes, 0., PetscMax(nmax, 1)));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)randsizes,"sizes_"));
  CHKERRQ(PetscRandomSetFromOptions(randsizes));

  CHKERRQ(PetscRandomCreate(comm, &randvalues));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)randvalues,"values_"));
  CHKERRQ(PetscRandomSetFromOptions(randvalues));

  CHKERRQ(PetscRandomGetValueReal(randsizes, &r));
  n    = (PetscInt) PetscMin(r, nmax);
  CHKERRQ(PetscRandomSetInterval(randsizes, 0., 1.));
  CHKERRQ(PetscRandomGetValueReal(randsizes, &r));
  first = PETSC_MIN_INT + 1 + (PetscInt) ((PETSC_MAX_INT - 1) * r);
  CHKERRQ(PetscRandomGetValueReal(randsizes, &r));
  last = first + (PetscInt) ((PETSC_MAX_INT - 1) * r);

  CHKERRQ(PetscRandomSetInterval(randvalues, first, last));
  CHKERRQ(PetscMalloc3(n, &keys, n, &keyscopy, n, &keyssorted));
  for (i = 0; i < n; i++) {
    CHKERRQ(PetscRandomGetValueReal(randvalues, &r));
    keys[i] = keyscopy[i] = (PetscInt) r;
  }
  CHKERRQ(ISCreateGeneral(comm, n, keys, PETSC_USE_POINTER, &is));
  CHKERRQ(ISViewFromOptions(is, NULL, "-keys_view"));

  CHKERRQ(ISGetLayout(is, &map));
  CHKERRQ(PetscLayoutCreateFromSizes(map->comm, PETSC_DECIDE, map->N, 1, &mapeven));
  CHKERRQ(PetscLayoutSetUp(mapeven));
  CHKERRQ(PetscMalloc1(mapeven->n, &keyseven));

  CHKERRQ(PetscParallelSortInt(map, mapeven, keys, keyseven));
  CHKERRQ(PetscParallelSortedInt(mapeven->comm, mapeven->n, keyseven, &sorted));
  PetscCheck(sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  for (i = 0; i < n; i++) PetscCheckFalse(keys[i] != keyscopy[i],PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscParallelSortInt() modified input array");

  CHKERRQ(PetscParallelSortInt(map, map, keys, keyssorted));
  CHKERRQ(PetscParallelSortedInt(map->comm, map->n, keyssorted, &sorted));
  PetscCheck(sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  for (i = 0; i < n; i++) PetscCheckFalse(keys[i] != keyscopy[i],PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscParallelSortInt() modified input array");

  CHKERRQ(PetscParallelSortInt(map, map, keys, keys));
  CHKERRQ(PetscParallelSortedInt(map->comm, map->n, keys, &sorted));
  PetscCheck(sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  /* TODO */
#if 0
  CHKERRQ(ISSortGlobal(is));
#endif

  CHKERRQ(PetscFree(keyseven));
  CHKERRQ(PetscLayoutDestroy(&mapeven));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(PetscFree3(keys,keyscopy,keyssorted));
  CHKERRQ(PetscRandomDestroy(&randvalues));
  CHKERRQ(PetscRandomDestroy(&randsizes));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{1 2 3 4 5}}
      args: -nmax {{0 1 5 10 100}}

TEST*/
