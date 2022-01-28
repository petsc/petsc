
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscOptionsBegin(comm, "", "Parallel Sort Test Options", "IS");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-nmax", "Maximum number of keys per process", "ex11.c", nmax, &nmax, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &randsizes);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randsizes, 0., PetscMax(nmax, 1));CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)randsizes,"sizes_");CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randsizes);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &randvalues);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)randvalues,"values_");CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randvalues);CHKERRQ(ierr);

  ierr = PetscRandomGetValueReal(randsizes, &r);CHKERRQ(ierr);
  n    = (PetscInt) PetscMin(r, nmax);
  ierr = PetscRandomSetInterval(randsizes, 0., 1.);CHKERRQ(ierr);
  ierr = PetscRandomGetValueReal(randsizes, &r);CHKERRQ(ierr);
  first = PETSC_MIN_INT + 1 + (PetscInt) ((PETSC_MAX_INT - 1) * r);
  ierr = PetscRandomGetValueReal(randsizes, &r);CHKERRQ(ierr);
  last = first + (PetscInt) ((PETSC_MAX_INT - 1) * r);

  ierr = PetscRandomSetInterval(randvalues, first, last);CHKERRQ(ierr);
  ierr = PetscMalloc3(n, &keys, n, &keyscopy, n, &keyssorted);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    ierr = PetscRandomGetValueReal(randvalues, &r);CHKERRQ(ierr);
    keys[i] = keyscopy[i] = (PetscInt) r;
  }
  ierr = ISCreateGeneral(comm, n, keys, PETSC_USE_POINTER, &is);CHKERRQ(ierr);
  ierr = ISViewFromOptions(is, NULL, "-keys_view");CHKERRQ(ierr);

  ierr = ISGetLayout(is, &map);CHKERRQ(ierr);
  ierr = PetscLayoutCreateFromSizes(map->comm, PETSC_DECIDE, map->N, 1, &mapeven);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mapeven);CHKERRQ(ierr);
  ierr = PetscMalloc1(mapeven->n, &keyseven);CHKERRQ(ierr);

  ierr = PetscParallelSortInt(map, mapeven, keys, keyseven);CHKERRQ(ierr);
  ierr = PetscParallelSortedInt(mapeven->comm, mapeven->n, keyseven, &sorted);CHKERRQ(ierr);
  PetscAssertFalse(!sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  for (i = 0; i < n; i++) PetscAssertFalse(keys[i] != keyscopy[i],PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscParallelSortInt() modified input array");

  ierr = PetscParallelSortInt(map, map, keys, keyssorted);CHKERRQ(ierr);
  ierr = PetscParallelSortedInt(map->comm, map->n, keyssorted, &sorted);CHKERRQ(ierr);
  PetscAssertFalse(!sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  for (i = 0; i < n; i++) PetscAssertFalse(keys[i] != keyscopy[i],PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscParallelSortInt() modified input array");

  ierr = PetscParallelSortInt(map, map, keys, keys);CHKERRQ(ierr);
  ierr = PetscParallelSortedInt(map->comm, map->n, keys, &sorted);CHKERRQ(ierr);
  PetscAssertFalse(!sorted,mapeven->comm, PETSC_ERR_PLIB, "PetscParallelSortInt() failed to sort");
  /* TODO */
#if 0
  ierr = ISSortGlobal(is);CHKERRQ(ierr);
#endif

  ierr = PetscFree(keyseven);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&mapeven);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFree3(keys,keyscopy,keyssorted);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randvalues);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randsizes);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{1 2 3 4 5}}
      args: -nmax {{0 1 5 10 100}}

TEST*/
