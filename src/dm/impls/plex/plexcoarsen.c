#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/

PetscErrorCode DMCoarsen_Plex(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
  PetscFunctionBegin;
  if (!dm->coarseMesh) PetscCall(DMPlexCoarsen_Internal(dm, NULL, NULL, NULL, &dm->coarseMesh));
  PetscCall(PetscObjectReference((PetscObject)dm->coarseMesh));
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCoarsenHierarchy_Plex(DM dm, PetscInt nlevels, DM dmCoarsened[])
{
  DM        rdm = dm;
  PetscInt  c;
  PetscBool localized;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocalized(dm, &localized));
  for (c = nlevels - 1; c >= 0; --c) {
    PetscCall(DMCoarsen(rdm, PetscObjectComm((PetscObject)dm), &dmCoarsened[c]));
    PetscCall(DMCopyDisc(rdm, dmCoarsened[c]));
    if (localized) PetscCall(DMLocalizeCoordinates(dmCoarsened[c]));
    PetscCall(DMSetCoarseDM(rdm, dmCoarsened[c]));
    rdm = dmCoarsened[c];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
