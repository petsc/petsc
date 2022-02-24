#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

PetscErrorCode DMCoarsen_Plex(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
  PetscFunctionBegin;
  if (!dm->coarseMesh) CHKERRQ(DMPlexCoarsen_Internal(dm, NULL, NULL, NULL, &dm->coarseMesh));
  CHKERRQ(PetscObjectReference((PetscObject) dm->coarseMesh));
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCoarsenHierarchy_Plex(DM dm, PetscInt nlevels, DM dmCoarsened[])
{
  DM             rdm = dm;
  PetscInt       c;
  PetscBool      localized;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinatesLocalized(dm, &localized));
  for (c = nlevels-1; c >= 0; --c) {
    CHKERRQ(DMCoarsen(rdm, PetscObjectComm((PetscObject) dm), &dmCoarsened[c]));
    CHKERRQ(DMCopyDisc(rdm, dmCoarsened[c]));
    if (localized) CHKERRQ(DMLocalizeCoordinates(dmCoarsened[c]));
    CHKERRQ(DMSetCoarseDM(rdm, dmCoarsened[c]));
    rdm  = dmCoarsened[c];
  }
  PetscFunctionReturn(0);
}
