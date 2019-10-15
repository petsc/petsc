
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
  PetscInt       *keys;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = MPI_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Parallel Sort Test Options", "IS");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-nmax", "Maximum number of keys per process", "ex11.c", nmax, &nmax, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &randsizes);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randsizes, 0., nmax);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)randsizes,"sizes_");CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randsizes);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &randvalues);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)randvalues,"values_");CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randvalues);CHKERRQ(ierr);

  ierr = PetscRandomGetValueReal(randsizes, &r);CHKERRQ(ierr);
  n    = (PetscInt) r;
  ierr = PetscRandomSetInterval(randsizes, 0., 1.);CHKERRQ(ierr);
  ierr = PetscRandomGetValueReal(randsizes, &r);CHKERRQ(ierr);
  first = PETSC_MIN_INT + 1 + (PetscInt) ((PETSC_MAX_INT - 1) * r);
  ierr = PetscRandomGetValueReal(randsizes, &r);CHKERRQ(ierr);
  last = first + (PetscInt) ((PETSC_MAX_INT - 1) * r);

  ierr = PetscRandomSetInterval(randvalues, first, last);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &keys);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    ierr = PetscRandomGetValueReal(randvalues, &r);CHKERRQ(ierr);
    keys[i] = (PetscInt) r;
  }
  ierr = ISCreateGeneral(comm, n, keys, PETSC_OWN_POINTER, &is);CHKERRQ(ierr);
  ierr = ISViewFromOptions(is, NULL, "-keys_view");CHKERRQ(ierr);

  /* TODO */
#if 0
  ierr = ISSortGlobal(is);CHKERRQ(ierr);
#endif

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randvalues);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randsizes);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
