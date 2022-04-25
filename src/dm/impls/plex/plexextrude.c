#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscdmplextransform.h>

/*@C
  DMPlexExtrude - Extrude a volumetric mesh from the input surface mesh

  Input Parameters:
+ dm          - The surface mesh
. layers      - The number of extruded layers
. thickness   - The total thickness of the extruded layers, or PETSC_DETERMINE
. tensor      - Flag to create tensor produt cells
. symmetric   - Flag to extrude symmetrically about the surface
. normal      - Surface normal vector, or NULL
- thicknesses - Thickness of each layer, or NULL

  Output Parameter:
. edm - The volumetric mesh

  Notes:
  Extrusion is implemented as a DMPlexTransform, so that new mesh points are produced from old mesh points. In the exmaple below,
we begin with an edge (v0, v3). It is extruded for two layers. The original vertex v0 produces two edges, e1 and e2, and three vertices,
v0, v2, and v2. Similarly, vertex v3 produces e3, e4, v3, v4, and v5. The original edge produces itself, e5 and e6, as well as face1 and
face2. The new mesh points are given the same labels as the original points which produced them. Thus, if v0 had a label value 1, then so
would v1, v2, e1 and e2.

$  v2----- e6    -----v5
$  |                  |
$  e2     face2       e4
$  |                  |
$  v1----- e5    -----v4
$  |                  |
$  e1     face1       e3
$  |                  |
$  v0--- original ----v3

  Options Database:
+ -dm_plex_transform_extrude_thickness <t>           - The total thickness of extruded layers
. -dm_plex_transform_extrude_use_tensor <bool>       - Use tensor cells when extruding
. -dm_plex_transform_extrude_symmetric <bool>        - Extrude layers symmetrically about the surface
. -dm_plex_transform_extrude_normal <n0,...,nd>      - Specify the extrusion direction
- -dm_plex_transform_extrude_thicknesses <t0,...,tl> - Specify thickness of each layer

  Level: intermediate

.seealso: `DMExtrude()`, `DMPlexTransform`, `DMPlexTransformExtrudeSetThickness()`, `DMPlexTransformExtrudeSetTensor()`
@*/
PetscErrorCode DMPlexExtrude(DM dm, PetscInt layers, PetscReal thickness, PetscBool tensor, PetscBool symmetric, const PetscReal normal[], const PetscReal thicknesses[], DM *edm)
{
  DMPlexTransform tr;
  DM              cdm, ecdm;
  const char     *prefix;
  PetscOptions    options;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXEXTRUDE));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) tr,  prefix));
  PetscCall(PetscObjectGetOptions((PetscObject) dm, &options));
  PetscCall(PetscObjectSetOptions((PetscObject) tr, options));
  PetscCall(DMPlexTransformExtrudeSetLayers(tr, layers));
  if (thickness > 0.) PetscCall(DMPlexTransformExtrudeSetThickness(tr, thickness));
  PetscCall(DMPlexTransformExtrudeSetTensor(tr, tensor));
  PetscCall(DMPlexTransformExtrudeSetSymmetric(tr, symmetric));
  if (normal) PetscCall(DMPlexTransformExtrudeSetNormal(tr, normal));
  if (thicknesses) PetscCall(DMPlexTransformExtrudeSetThicknesses(tr, layers, thicknesses));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(PetscObjectSetOptions((PetscObject) tr, NULL));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject) tr, NULL, "-dm_plex_transform_view"));
  PetscCall(DMPlexTransformApply(tr, dm, edm));
  PetscCall(DMCopyDisc(dm, *edm));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDM(*edm, &ecdm));
  PetscCall(DMCopyDisc(cdm, ecdm));
  PetscCall(DMPlexTransformCreateDiscLabels(tr, *edm));
  PetscCall(DMPlexTransformDestroy(&tr));
  if (*edm) {
    ((DM_Plex *) (*edm)->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
    ((DM_Plex *) (*edm)->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMExtrude_Plex(DM dm, PetscInt layers, DM *edm)
{
  PetscFunctionBegin;
  PetscCall(DMPlexExtrude(dm, layers, PETSC_DETERMINE, PETSC_TRUE, PETSC_FALSE, NULL, NULL, edm));
  PetscCall(DMViewFromOptions(*edm, NULL, "-check_extrude"));
  PetscFunctionReturn(0);
}
