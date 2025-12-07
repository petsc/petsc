#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petscdmplextransform.h>

/*@
  DMPlexExtrude - Extrude a volumetric mesh from the input surface mesh

  Input Parameters:
+ dm          - The surface mesh
. layers      - The number of extruded layers
. thickness   - The total thickness of the extruded layers, or `PETSC_DETERMINE`
. tensor      - Flag to create tensor product cells
. symmetric   - Flag to extrude symmetrically about the surface
. periodic    - Flag to extrude periodically
. normal      - Surface normal vector, or `NULL`
. thicknesses - Thickness of each layer, or `NULL`
- activeLabel - `DMLabel` to extrude from, or `NULL` to extrude entire mesh

  Output Parameter:
. edm - The volumetric mesh

  Options Database Keys:
+ -dm_plex_transform_extrude_thickness <t>           - The total thickness of extruded layers
. -dm_plex_transform_extrude_use_tensor <bool>       - Use tensor cells when extruding
. -dm_plex_transform_extrude_symmetric <bool>        - Extrude layers symmetrically about the surface
. -dm_plex_transform_extrude_periodic <bool>         - Extrude layers periodically
. -dm_plex_transform_extrude_normal <n0,...,nd>      - Specify the extrusion direction
- -dm_plex_transform_extrude_thicknesses <t0,...,tl> - Specify thickness of each layer

  Level: intermediate

  Notes:
  Extrusion is implemented as a `DMPlexTransform`, so that new mesh points are produced from old mesh points. In the example below,
  we begin with an edge (v0, v3). It is extruded for two layers. The original vertex v0 produces two edges, e1 and e2, and three vertices,
  v0, v2, and v2. Similarly, vertex v3 produces e3, e4, v3, v4, and v5. The original edge produces itself, e5 and e6, as well as face1 and
  face2. The new mesh points are given the same labels as the original points which produced them. Thus, if v0 had a label value 1, then so
  would v1, v2, e1 and e2.

.vb
  v2----- e6    -----v5
  |                  |
  e2     face2       e4
  |                  |
  v1----- e5    -----v4
  |                  |
  e1     face1       e3
  |                  |
  v0--- original ----v3
.ve

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMExtrude()`, `DMPlexTransform`, `DMPlexTransformExtrudeSetThickness()`, `DMPlexTransformExtrudeSetTensor()`
@*/
PetscErrorCode DMPlexExtrude(DM dm, PetscInt layers, PetscReal thickness, PetscBool tensor, PetscBool symmetric, PetscBool periodic, const PetscReal normal[], const PetscReal thicknesses[], DMLabel activeLabel, DM *edm)
{
  DMPlexTransform tr;
  DM              cdm;
  PetscObject     disc;
  PetscClassId    id;
  const char     *prefix;
  PetscOptions    options;
  PetscBool       cutMarker = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Extrusion Transform"));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetType(tr, DMPLEXEXTRUDETYPE));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, prefix));
  PetscCall(PetscObjectGetOptions((PetscObject)dm, &options));
  PetscCall(PetscObjectSetOptions((PetscObject)tr, options));
  if (activeLabel) PetscCall(DMPlexTransformSetActive(tr, activeLabel));
  PetscCall(DMPlexTransformExtrudeSetLayers(tr, layers));
  if (thickness > 0.) PetscCall(DMPlexTransformExtrudeSetThickness(tr, thickness));
  PetscCall(DMPlexTransformExtrudeSetTensor(tr, tensor));
  PetscCall(DMPlexTransformExtrudeSetSymmetric(tr, symmetric));
  PetscCall(DMPlexTransformExtrudeSetPeriodic(tr, periodic));
  if (normal) PetscCall(DMPlexTransformExtrudeSetNormal(tr, normal));
  if (thicknesses) PetscCall(DMPlexTransformExtrudeSetThicknesses(tr, layers, thicknesses));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(PetscObjectSetOptions((PetscObject)tr, NULL));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
  PetscCall(DMPlexTransformApply(tr, dm, edm));
  PetscCall(DMCopyDisc(dm, *edm));
  // Handle periodic viewing
  PetscCall(PetscOptionsGetBool(options, ((PetscObject)dm)->prefix, "-dm_plex_periodic_cut", &cutMarker, NULL));
  PetscCall(DMPlexTransformExtrudeGetPeriodic(tr, &periodic));
  if (periodic && cutMarker) {
    DMLabel  cutLabel;
    PetscInt dim, pStart, pEnd;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMCreateLabel(*edm, "periodic_cut"));
    PetscCall(DMGetLabel(*edm, "periodic_cut", &cutLabel));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    for (PetscInt p = pStart; p < pEnd; ++p) {
      DMPolytopeType  ct;
      DMPolytopeType *rct;
      PetscInt       *rsize, *rcone, *rornt;
      PetscInt        Nct;

      PetscCall(DMPlexGetCellType(dm, p, &ct));
      PetscCall(DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
      for (PetscInt n = 0; n < Nct; ++n) {
        PetscInt pNew, pdim = DMPolytopeTypeGetDim(rct[n]);

        if (ct == rct[n] || pdim > dim) {
          PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], p, 0, &pNew));
          PetscCall(DMLabelSetValue(cutLabel, pNew, !pdim ? 1 : 2));
        }
      }
    }
  }
  // It is too hard to raise the dimension of a discretization, so just remake it
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetField(cdm, 0, NULL, &disc));
  PetscCall(PetscObjectGetClassId(disc, &id));
  if (id == PETSCFE_CLASSID) {
    PetscSpace sp;
    PetscInt   deg;

    PetscCall(PetscFEGetBasisSpace((PetscFE)disc, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &deg, NULL));
    PetscCall(DMPlexCreateCoordinateSpace(*edm, deg, PETSC_FALSE, PETSC_TRUE));
  }
  PetscCall(DMPlexTransformCreateDiscLabels(tr, *edm));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexCopy_Internal(dm, PETSC_FALSE, PETSC_FALSE, *edm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMExtrude_Plex(DM dm, PetscInt layers, DM *edm)
{
  PetscFunctionBegin;
  PetscCall(DMPlexExtrude(dm, layers, PETSC_DETERMINE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, NULL, NULL, NULL, edm));
  PetscCall(DMSetMatType(*edm, dm->mattype));
  PetscCall(DMViewFromOptions(*edm, NULL, "-check_extrude"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
