#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

PetscErrorCode DMCoarsen_Plex(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm->coarseMesh) {ierr = DMPlexCoarsen_Internal(dm, NULL, &dm->coarseMesh);CHKERRQ(ierr);}
  ierr = PetscObjectReference((PetscObject) dm->coarseMesh);CHKERRQ(ierr);
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCoarsenHierarchy_Plex(DM dm, PetscInt nlevels, DM dmCoarsened[])
{
  DM             rdm = dm;
  PetscInt       c;
  PetscBool      localized;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  for (c = nlevels-1; c >= 0; --c) {
    ierr = DMCoarsen(rdm, PetscObjectComm((PetscObject) dm), &dmCoarsened[c]);CHKERRQ(ierr);
    ierr = DMCopyDisc(rdm, dmCoarsened[c]);CHKERRQ(ierr);
    if (localized) {ierr = DMLocalizeCoordinates(dmCoarsened[c]);CHKERRQ(ierr);}
    ierr = DMSetCoarseDM(rdm, dmCoarsened[c]);CHKERRQ(ierr);
    rdm  = dmCoarsened[c];
  }
  PetscFunctionReturn(0);
}
