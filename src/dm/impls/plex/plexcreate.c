#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petsc/private/hashseti.h>          /*I   "petscdmplex.h"   I*/
#include <petscsf.h>
#include <petsc/private/kernels/blockmatmult.h>
#include <petsc/private/kernels/blockinvert.h>

PetscLogEvent DMPLEX_CreateFromFile, DMPLEX_BuildFromCellList, DMPLEX_BuildCoordinatesFromCellList;

/* External function declarations here */
static PetscErrorCode DMInitialize_Plex(DM dm);

/* This copies internal things in the Plex structure that we generally want when making a new, related Plex */
PetscErrorCode DMPlexCopy_Internal(DM dmin, PetscBool copyPeriodicity, DM dmout)
{
  const DMBoundaryType *bd;
  const PetscReal      *maxCell, *L;
  PetscBool             isper, dist;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (copyPeriodicity) {
    ierr = DMGetPeriodicity(dmin, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(dmout, isper,  maxCell,  L,  bd);CHKERRQ(ierr);
  }
  ierr = DMPlexDistributeGetDefault(dmin, &dist);CHKERRQ(ierr);
  ierr = DMPlexDistributeSetDefault(dmout, dist);CHKERRQ(ierr);
  ((DM_Plex *) dmout->data)->useHashLocation = ((DM_Plex *) dmin->data)->useHashLocation;
  PetscFunctionReturn(0);
}

/* Replace dm with the contents of ndm, and then destroy ndm
   - Share the DM_Plex structure
   - Share the coordinates
   - Share the SF
*/
static PetscErrorCode DMPlexReplace_Static(DM dm, DM *ndm)
{
  PetscSF               sf;
  DM                    dmNew = *ndm, coordDM, coarseDM;
  Vec                   coords;
  PetscBool             isper;
  const PetscReal      *maxCell, *L;
  const DMBoundaryType *bd;
  PetscInt              dim, cdim;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (dm == dmNew) {
    ierr = DMDestroy(ndm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  dm->setupcalled = dmNew->setupcalled;
  ierr = DMGetDimension(dmNew, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dmNew, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm, cdim);CHKERRQ(ierr);
  ierr = DMGetPointSF(dmNew, &sf);CHKERRQ(ierr);
  ierr = DMSetPointSF(dm, sf);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmNew, &coordDM);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmNew, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dm, coordDM);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coords);CHKERRQ(ierr);
  /* Do not want to create the coordinate field if it does not already exist, so do not call DMGetCoordinateField() */
  ierr = DMFieldDestroy(&dm->coordinateField);CHKERRQ(ierr);
  dm->coordinateField = dmNew->coordinateField;
  ((DM_Plex *) dmNew->data)->coordFunc = ((DM_Plex *) dm->data)->coordFunc;
  ierr = DMGetPeriodicity(dmNew, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(dm, isper, maxCell, L, bd);CHKERRQ(ierr);
  ierr = DMDestroy_Plex(dm);CHKERRQ(ierr);
  ierr = DMInitialize_Plex(dm);CHKERRQ(ierr);
  dm->data = dmNew->data;
  ((DM_Plex *) dmNew->data)->refct++;
  ierr = DMDestroyLabelLinkList_Internal(dm);CHKERRQ(ierr);
  ierr = DMCopyLabels(dmNew, dm, PETSC_OWN_POINTER, PETSC_TRUE, DM_COPY_LABELS_FAIL);CHKERRQ(ierr);
  ierr = DMGetCoarseDM(dmNew,&coarseDM);CHKERRQ(ierr);
  ierr = DMSetCoarseDM(dm,coarseDM);CHKERRQ(ierr);
  ierr = DMDestroy(ndm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Swap dm with the contents of dmNew
   - Swap the DM_Plex structure
   - Swap the coordinates
   - Swap the point PetscSF
*/
static PetscErrorCode DMPlexSwap_Static(DM dmA, DM dmB)
{
  DM              coordDMA, coordDMB;
  Vec             coordsA,  coordsB;
  PetscSF         sfA,      sfB;
  DMField         fieldTmp;
  void            *tmp;
  DMLabelLink     listTmp;
  DMLabel         depthTmp;
  PetscInt        tmpI;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (dmA == dmB) PetscFunctionReturn(0);
  ierr = DMGetPointSF(dmA, &sfA);CHKERRQ(ierr);
  ierr = DMGetPointSF(dmB, &sfB);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) sfA);CHKERRQ(ierr);
  ierr = DMSetPointSF(dmA, sfB);CHKERRQ(ierr);
  ierr = DMSetPointSF(dmB, sfA);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject) sfA);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(dmA, &coordDMA);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmB, &coordDMB);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) coordDMA);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmA, coordDMB);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmB, coordDMA);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject) coordDMA);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dmA, &coordsA);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmB, &coordsB);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) coordsA);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmA, coordsB);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmB, coordsA);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject) coordsA);CHKERRQ(ierr);

  fieldTmp             = dmA->coordinateField;
  dmA->coordinateField = dmB->coordinateField;
  dmB->coordinateField = fieldTmp;
  tmp       = dmA->data;
  dmA->data = dmB->data;
  dmB->data = tmp;
  listTmp   = dmA->labels;
  dmA->labels = dmB->labels;
  dmB->labels = listTmp;
  depthTmp  = dmA->depthLabel;
  dmA->depthLabel = dmB->depthLabel;
  dmB->depthLabel = depthTmp;
  depthTmp  = dmA->celltypeLabel;
  dmA->celltypeLabel = dmB->celltypeLabel;
  dmB->celltypeLabel = depthTmp;
  tmpI         = dmA->levelup;
  dmA->levelup = dmB->levelup;
  dmB->levelup = tmpI;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexInterpolateInPlace_Internal(DM dm)
{
  DM             idm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
  ierr = DMPlexReplace_Static(dm, &idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateCoordinateSpace - Creates a finite element space for the coordinates

  Collective

  Input Parameters:
+ DM        - The DM
. degree    - The degree of the finite element or PETSC_DECIDE
- coordFunc - An optional function to map new points from refinement to the surface

  Level: advanced

.seealso: PetscFECreateLagrange(), DMGetCoordinateDM()
@*/
PetscErrorCode DMPlexCreateCoordinateSpace(DM dm, PetscInt degree, PetscPointFunc coordFunc)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DM             cdm;
  PetscDS        cds;
  PetscFE        fe;
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(cds, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId((PetscObject) fe, &id);CHKERRQ(ierr);
  if (id != PETSCFE_CLASSID) {
    PetscBool simplex;
    PetscInt  dim, dE, qorder;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
    ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
    qorder = degree;
    ierr = PetscObjectOptionsBegin((PetscObject) cdm);CHKERRQ(ierr);
    ierr = PetscOptionsBoundedInt("-coord_dm_default_quadrature_order", "Quadrature order is one less than quadrature points per edge", "DMPlexCreateCoordinateSpace", qorder, &qorder, NULL, 0);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (degree == PETSC_DECIDE) fe = NULL;
    else {
      ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, degree, qorder, &fe);CHKERRQ(ierr);
      ierr = DMSetField(cdm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
      ierr = DMCreateDS(cdm);CHKERRQ(ierr);
    }
    ierr = DMProjectCoordinates(dm, fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  mesh->coordFunc = coordFunc;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateDoublet - Creates a mesh of two cells of the specified type, optionally with later refinement.

  Collective

  Input Parameters:
+ comm - The communicator for the DM object
. dim - The spatial dimension
. simplex - Flag for simplicial cells, otherwise they are tensor product cells
. interpolate - Flag to create intermediate mesh pieces (edges, faces)
- refinementLimit - A nonzero number indicates the largest admissible volume for a refined cell

  Output Parameter:
. dm - The DM object

  Level: beginner

.seealso: DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateDoublet(MPI_Comm comm, PetscInt dim, PetscBool simplex, PetscBool interpolate, PetscReal refinementLimit, DM *newdm)
{
  DM             dm;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  switch (dim) {
  case 2:
    if (simplex) {ierr = PetscObjectSetName((PetscObject) dm, "triangular");CHKERRQ(ierr);}
    else         {ierr = PetscObjectSetName((PetscObject) dm, "quadrilateral");CHKERRQ(ierr);}
    break;
  case 3:
    if (simplex) {ierr = PetscObjectSetName((PetscObject) dm, "tetrahedral");CHKERRQ(ierr);}
    else         {ierr = PetscObjectSetName((PetscObject) dm, "hexahedral");CHKERRQ(ierr);}
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", dim);
  }
  if (rank) {
    PetscInt numPoints[2] = {0, 0};
    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  } else {
    switch (dim) {
    case 2:
      if (simplex) {
        PetscInt    numPoints[2]        = {4, 2};
        PetscInt    coneSize[6]         = {3, 3, 0, 0, 0, 0};
        PetscInt    cones[6]            = {2, 3, 4,  5, 4, 3};
        PetscInt    coneOrientations[6] = {0, 0, 0,  0, 0, 0};
        PetscScalar vertexCoords[8]     = {-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5};

        ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      } else {
        PetscInt    numPoints[2]        = {6, 2};
        PetscInt    coneSize[8]         = {4, 4, 0, 0, 0, 0, 0, 0};
        PetscInt    cones[8]            = {2, 3, 4, 5,  3, 6, 7, 4};
        PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
        PetscScalar vertexCoords[12]    = {-1.0, -0.5,  0.0, -0.5,  0.0, 0.5,  -1.0, 0.5,  1.0, -0.5,  1.0, 0.5};

        ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (simplex) {
        PetscInt    numPoints[2]        = {5, 2};
        PetscInt    coneSize[7]         = {4, 4, 0, 0, 0, 0, 0};
        PetscInt    cones[8]            = {4, 3, 5, 2,  5, 3, 4, 6};
        PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
        PetscScalar vertexCoords[15]    = {-1.0, 0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 0.0};

        ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      } else {
        PetscInt    numPoints[2]         = {12, 2};
        PetscInt    coneSize[14]         = {8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        PetscInt    cones[16]            = {2, 3, 4, 5, 6, 7, 8, 9,  5, 4, 10, 11, 7, 12, 13, 8};
        PetscInt    coneOrientations[16] = {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0,  0,  0, 0};
        PetscScalar vertexCoords[36]     = {-1.0, -0.5, -0.5,  -1.0,  0.5, -0.5,  0.0,  0.5, -0.5,   0.0, -0.5, -0.5,
                                            -1.0, -0.5,  0.5,   0.0, -0.5,  0.5,  0.0,  0.5,  0.5,  -1.0,  0.5,  0.5,
                                             1.0,  0.5, -0.5,   1.0, -0.5, -0.5,  1.0, -0.5,  0.5,   1.0,  0.5,  0.5};

        ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", dim);
    }
  }
  *newdm = dm;
  if (refinementLimit > 0.0) {
    DM rdm;
    const char *name;

    ierr = DMPlexSetRefinementUniform(*newdm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(*newdm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*newdm, comm, &rdm);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) *newdm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)    rdm,  name);CHKERRQ(ierr);
    ierr = DMDestroy(newdm);CHKERRQ(ierr);
    *newdm = rdm;
  }
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*newdm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(newdm);CHKERRQ(ierr);
    *newdm = idm;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxSurfaceMesh_Tensor_1D_Internal(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  const PetscInt numVertices    = 2;
  PetscInt       markerRight    = 1;
  PetscInt       markerLeft     = 1;
  PetscBool      markerSeparate = PETSC_FALSE;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       cdim = 1, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, NULL);CHKERRQ(ierr);
  if (markerSeparate) {
    markerRight  = 2;
    markerLeft   = 1;
  }
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = DMPlexSetChart(dm, 0, numVertices);CHKERRQ(ierr);
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    ierr = DMSetLabelValue(dm, "marker", 0, markerLeft);CHKERRQ(ierr);
    ierr = DMSetLabelValue(dm, "marker", 1, markerRight);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMSetCoordinateDim(dm, cdim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, 0, numVertices);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, cdim);CHKERRQ(ierr);
  for (v = 0; v < numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, cdim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, cdim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, cdim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  coords[0] = lower[0];
  coords[1] = upper[0];
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxSurfaceMesh_Tensor_2D_Internal(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  const PetscInt numVertices    = (edges[0]+1)*(edges[1]+1);
  const PetscInt numEdges       = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
  PetscInt       markerTop      = 1;
  PetscInt       markerBottom   = 1;
  PetscInt       markerRight    = 1;
  PetscInt       markerLeft     = 1;
  PetscBool      markerSeparate = PETSC_FALSE;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, NULL);CHKERRQ(ierr);
  if (markerSeparate) {
    markerTop    = 3;
    markerBottom = 1;
    markerRight  = 2;
    markerLeft   = 4;
  }
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
  if (rank == 0) {
    PetscInt e, ex, ey;

    ierr = DMPlexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
    for (e = 0; e < numEdges; ++e) {
      ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for (vx = 0; vx <= edges[0]; vx++) {
      for (ey = 0; ey < edges[1]; ey++) {
        PetscInt edge   = vx*edges[1] + ey + edges[0]*(edges[1]+1);
        PetscInt vertex = ey*(edges[0]+1) + vx + numEdges;
        PetscInt cone[2];

        cone[0] = vertex; cone[1] = vertex+edges[0]+1;
        ierr    = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vx == edges[0]) {
          ierr = DMSetLabelValue(dm, "marker", edge,    markerRight);CHKERRQ(ierr);
          ierr = DMSetLabelValue(dm, "marker", cone[0], markerRight);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMSetLabelValue(dm, "marker", cone[1], markerRight);CHKERRQ(ierr);
            ierr = DMSetLabelValue(dm, "Face Sets", cone[1], markerRight);CHKERRQ(ierr);
          }
        } else if (vx == 0) {
          ierr = DMSetLabelValue(dm, "marker", edge,    markerLeft);CHKERRQ(ierr);
          ierr = DMSetLabelValue(dm, "marker", cone[0], markerLeft);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMSetLabelValue(dm, "marker", cone[1], markerLeft);CHKERRQ(ierr);
            ierr = DMSetLabelValue(dm, "Face Sets", cone[1], markerLeft);CHKERRQ(ierr);
          }
        }
      }
    }
    for (vy = 0; vy <= edges[1]; vy++) {
      for (ex = 0; ex < edges[0]; ex++) {
        PetscInt edge   = vy*edges[0]     + ex;
        PetscInt vertex = vy*(edges[0]+1) + ex + numEdges;
        PetscInt cone[2];

        cone[0] = vertex; cone[1] = vertex+1;
        ierr    = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vy == edges[1]) {
          ierr = DMSetLabelValue(dm, "marker", edge,    markerTop);CHKERRQ(ierr);
          ierr = DMSetLabelValue(dm, "marker", cone[0], markerTop);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMSetLabelValue(dm, "marker", cone[1], markerTop);CHKERRQ(ierr);
            ierr = DMSetLabelValue(dm, "Face Sets", cone[1], markerTop);CHKERRQ(ierr);
          }
        } else if (vy == 0) {
          ierr = DMSetLabelValue(dm, "marker", edge,    markerBottom);CHKERRQ(ierr);
          ierr = DMSetLabelValue(dm, "marker", cone[0], markerBottom);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMSetLabelValue(dm, "marker", cone[1], markerBottom);CHKERRQ(ierr);
            ierr = DMSetLabelValue(dm, "Face Sets", cone[1], markerBottom);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMSetCoordinateDim(dm, 2);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numEdges, numEdges + numVertices);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, 2);CHKERRQ(ierr);
  for (v = numEdges; v < numEdges+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 2);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, 2);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (vy = 0; vy <= edges[1]; ++vy) {
    for (vx = 0; vx <= edges[0]; ++vx) {
      coords[(vy*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/edges[0])*vx;
      coords[(vy*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/edges[1])*vy;
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxSurfaceMesh_Tensor_3D_Internal(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
{
  PetscInt       vertices[3], numVertices;
  PetscInt       numFaces    = 2*faces[0]*faces[1] + 2*faces[1]*faces[2] + 2*faces[0]*faces[2];
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy, vz;
  PetscInt       voffset, iface=0, cone[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse((faces[0] < 1) || (faces[1] < 1) || (faces[2] < 1),PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Must have at least 1 face per side");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
  vertices[0] = faces[0]+1; vertices[1] = faces[1]+1; vertices[2] = faces[2]+1;
  numVertices = vertices[0]*vertices[1]*vertices[2];
  if (rank == 0) {
    PetscInt f;

    ierr = DMPlexSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMPlexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */

    /* Side 0 (Top) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vertices[0]*vertices[1]*(vertices[2]-1) + vy*vertices[0] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]+1; cone[3] = voffset+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", iface, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+1, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]+1, 1);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 1 (Bottom) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vy*(faces[0]+1) + vx;
        cone[0] = voffset+1; cone[1] = voffset; cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]+1;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", iface, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+1, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]+1, 1);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 2 (Front) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]*vertices[1]+1; cone[3] = voffset+vertices[0]*vertices[1];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", iface, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+1, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+1, 1);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 3 (Back) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vertices[0]*(vertices[1]-1) + vx;
        cone[0] = voffset+vertices[0]*vertices[1]; cone[1] = voffset+vertices[0]*vertices[1]+1;
        cone[2] = voffset+1; cone[3] = voffset;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", iface, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+1, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+1, 1);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 4 (Left) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vy = 0; vy < faces[1]; vy++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vy*vertices[0];
        cone[0] = voffset; cone[1] = voffset+vertices[0]*vertices[1];
        cone[2] = voffset+vertices[0]*vertices[1]+vertices[0]; cone[3] = voffset+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", iface, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[1]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+vertices[0], 1);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 5 (Right) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vy = 0; vy < faces[1]; vy++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vy*vertices[0] + faces[0];
        cone[0] = voffset+vertices[0]*vertices[1]; cone[1] = voffset;
        cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]*vertices[1]+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", iface, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+0, 1);CHKERRQ(ierr);
        ierr    = DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+vertices[0], 1);CHKERRQ(ierr);
        iface++;
      }
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMSetCoordinateDim(dm, 3);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, 3);CHKERRQ(ierr);
  for (v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 3);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, 3);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, 3);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (vz = 0; vz <= faces[2]; ++vz) {
    for (vy = 0; vy <= faces[1]; ++vy) {
      for (vx = 0; vx <= faces[0]; ++vx) {
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+0] = lower[0] + ((upper[0] - lower[0])/faces[0])*vx;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+1] = lower[1] + ((upper[1] - lower[1])/faces[1])*vy;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+2] = lower[2] + ((upper[2] - lower[2])/faces[2])*vz;
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxSurfaceMesh_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], PetscBool interpolate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  ierr = DMSetDimension(dm, dim-1);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm, dim);CHKERRQ(ierr);
  switch (dim) {
    case 1: ierr = DMPlexCreateBoxSurfaceMesh_Tensor_1D_Internal(dm, lower, upper, faces);CHKERRQ(ierr);break;
    case 2: ierr = DMPlexCreateBoxSurfaceMesh_Tensor_2D_Internal(dm, lower, upper, faces);CHKERRQ(ierr);break;
    case 3: ierr = DMPlexCreateBoxSurfaceMesh_Tensor_3D_Internal(dm, lower, upper, faces);CHKERRQ(ierr);break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Dimension not supported: %D", dim);
  }
  if (interpolate) {ierr = DMPlexInterpolateInPlace_Internal(dm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateBoxSurfaceMesh - Creates a mesh on the surface of the tensor product of unit intervals (box) using tensor cells (hexahedra).

  Collective

  Input Parameters:
+ comm        - The communicator for the DM object
. dim         - The spatial dimension of the box, so the resulting mesh is has dimension dim-1
. faces       - Number of faces per dimension, or NULL for (1,) in 1D and (2, 2) in 2D and (1, 1, 1) in 3D
. lower       - The lower left corner, or NULL for (0, 0, 0)
. upper       - The upper right corner, or NULL for (1, 1, 1)
- interpolate - Flag to create intermediate mesh pieces (edges, faces)

  Output Parameter:
. dm  - The DM object

  Level: beginner

.seealso: DMSetFromOptions(), DMPlexCreateBoxMesh(), DMPlexCreateFromFile(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateBoxSurfaceMesh(MPI_Comm comm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], PetscBool interpolate, DM *dm)
{
  PetscInt       fac[3] = {1, 1, 1};
  PetscReal      low[3] = {0, 0, 0};
  PetscReal      upp[3] = {1, 1, 1};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxSurfaceMesh_Internal(*dm, dim, faces ? faces : fac, lower ? lower : low, upper ? upper : upp, interpolate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateLineMesh_Internal(DM dm,PetscInt segments,PetscReal lower,PetscReal upper,DMBoundaryType bd)
{
  PetscInt       i,fStart,fEnd,numCells = 0,numVerts = 0;
  PetscInt       numPoints[2],*coneSize,*cones,*coneOrientations;
  PetscScalar    *vertexCoords;
  PetscReal      L,maxCell;
  PetscBool      markerSeparate = PETSC_FALSE;
  PetscInt       markerLeft  = 1, faceMarkerLeft  = 1;
  PetscInt       markerRight = 1, faceMarkerRight = 2;
  PetscBool      wrap = (bd == DM_BOUNDARY_PERIODIC || bd == DM_BOUNDARY_TWIST) ? PETSC_TRUE : PETSC_FALSE;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,1);

  ierr = DMSetDimension(dm,1);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"marker");CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"Face Sets");CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm),&rank);CHKERRMPI(ierr);
  if (rank == 0) numCells = segments;
  if (rank == 0) numVerts = segments + (wrap ? 0 : 1);

  numPoints[0] = numVerts ; numPoints[1] = numCells;
  ierr = PetscMalloc4(numCells+numVerts,&coneSize,numCells*2,&cones,numCells+numVerts,&coneOrientations,numVerts,&vertexCoords);CHKERRQ(ierr);
  ierr = PetscArrayzero(coneOrientations,numCells+numVerts);CHKERRQ(ierr);
  for (i = 0; i < numCells; ++i) { coneSize[i] = 2; }
  for (i = 0; i < numVerts; ++i) { coneSize[numCells+i] = 0; }
  for (i = 0; i < numCells; ++i) { cones[2*i] = numCells + i%numVerts; cones[2*i+1] = numCells + (i+1)%numVerts; }
  for (i = 0; i < numVerts; ++i) { vertexCoords[i] = lower + (upper-lower)*((PetscReal)i/(PetscReal)numCells); }
  ierr = DMPlexCreateFromDAG(dm,1,numPoints,coneSize,cones,coneOrientations,vertexCoords);CHKERRQ(ierr);
  ierr = PetscFree4(coneSize,cones,coneOrientations,vertexCoords);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(((PetscObject)dm)->options,((PetscObject)dm)->prefix,"-dm_plex_separate_marker",&markerSeparate,NULL);CHKERRQ(ierr);
  if (markerSeparate) { markerLeft = faceMarkerLeft; markerRight = faceMarkerRight;}
  if (!wrap && rank == 0) {
    ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
    ierr = DMSetLabelValue(dm,"marker",fStart,markerLeft);CHKERRQ(ierr);
    ierr = DMSetLabelValue(dm,"marker",fEnd-1,markerRight);CHKERRQ(ierr);
    ierr = DMSetLabelValue(dm,"Face Sets",fStart,faceMarkerLeft);CHKERRQ(ierr);
    ierr = DMSetLabelValue(dm,"Face Sets",fEnd-1,faceMarkerRight);CHKERRQ(ierr);
  }
  if (wrap) {
    L       = upper - lower;
    maxCell = (PetscReal)1.1*(L/(PetscReal)PetscMax(1,segments));
    ierr = DMSetPeriodicity(dm,PETSC_TRUE,&maxCell,&L,&bd);CHKERRQ(ierr);
  }
  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxMesh_Simplex_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate)
{
  DM             boundary, vol;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 1);
  for (i = 0; i < dim; ++i) PetscCheckFalse(periodicity[i] != DM_BOUNDARY_NONE,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Periodicity is not supported for simplex meshes");
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &boundary);CHKERRQ(ierr);
  ierr = DMSetType(boundary, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxSurfaceMesh_Internal(boundary, dim, faces, lower, upper, PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMPlexGenerate(boundary, NULL, interpolate, &vol);CHKERRQ(ierr);
  ierr = DMPlexCopy_Internal(dm, PETSC_TRUE, vol);CHKERRQ(ierr);
  ierr = DMPlexReplace_Static(dm, &vol);CHKERRQ(ierr);
  ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCubeMesh_Internal(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[], DMBoundaryType bdX, DMBoundaryType bdY, DMBoundaryType bdZ)
{
  DMLabel        cutLabel = NULL;
  PetscInt       markerTop      = 1, faceMarkerTop      = 1;
  PetscInt       markerBottom   = 1, faceMarkerBottom   = 1;
  PetscInt       markerFront    = 1, faceMarkerFront    = 1;
  PetscInt       markerBack     = 1, faceMarkerBack     = 1;
  PetscInt       markerRight    = 1, faceMarkerRight    = 1;
  PetscInt       markerLeft     = 1, faceMarkerLeft     = 1;
  PetscInt       dim;
  PetscBool      markerSeparate = PETSC_FALSE, cutMarker = PETSC_FALSE;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
  ierr = DMCreateLabel(dm,"marker");CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"Face Sets");CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_periodic_cut", &cutMarker, NULL);CHKERRQ(ierr);
  if (bdX == DM_BOUNDARY_PERIODIC || bdX == DM_BOUNDARY_TWIST ||
      bdY == DM_BOUNDARY_PERIODIC || bdY == DM_BOUNDARY_TWIST ||
      bdZ == DM_BOUNDARY_PERIODIC || bdZ == DM_BOUNDARY_TWIST) {

    if (cutMarker) {ierr = DMCreateLabel(dm, "periodic_cut");CHKERRQ(ierr); ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);}
  }
  switch (dim) {
  case 2:
    faceMarkerTop    = 3;
    faceMarkerBottom = 1;
    faceMarkerRight  = 2;
    faceMarkerLeft   = 4;
    break;
  case 3:
    faceMarkerBottom = 1;
    faceMarkerTop    = 2;
    faceMarkerFront  = 3;
    faceMarkerBack   = 4;
    faceMarkerRight  = 5;
    faceMarkerLeft   = 6;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Dimension %D not supported",dim);
  }
  ierr = PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, NULL);CHKERRQ(ierr);
  if (markerSeparate) {
    markerBottom = faceMarkerBottom;
    markerTop    = faceMarkerTop;
    markerFront  = faceMarkerFront;
    markerBack   = faceMarkerBack;
    markerRight  = faceMarkerRight;
    markerLeft   = faceMarkerLeft;
  }
  {
    const PetscInt numXEdges    = rank == 0 ? edges[0] : 0;
    const PetscInt numYEdges    = rank == 0 ? edges[1] : 0;
    const PetscInt numZEdges    = rank == 0 ? edges[2] : 0;
    const PetscInt numXVertices = rank == 0 ? (bdX == DM_BOUNDARY_PERIODIC || bdX == DM_BOUNDARY_TWIST ? edges[0] : edges[0]+1) : 0;
    const PetscInt numYVertices = rank == 0 ? (bdY == DM_BOUNDARY_PERIODIC || bdY == DM_BOUNDARY_TWIST ? edges[1] : edges[1]+1) : 0;
    const PetscInt numZVertices = rank == 0 ? (bdZ == DM_BOUNDARY_PERIODIC || bdZ == DM_BOUNDARY_TWIST ? edges[2] : edges[2]+1) : 0;
    const PetscInt numCells     = numXEdges*numYEdges*numZEdges;
    const PetscInt numXFaces    = numYEdges*numZEdges;
    const PetscInt numYFaces    = numXEdges*numZEdges;
    const PetscInt numZFaces    = numXEdges*numYEdges;
    const PetscInt numTotXFaces = numXVertices*numXFaces;
    const PetscInt numTotYFaces = numYVertices*numYFaces;
    const PetscInt numTotZFaces = numZVertices*numZFaces;
    const PetscInt numFaces     = numTotXFaces + numTotYFaces + numTotZFaces;
    const PetscInt numTotXEdges = numXEdges*numYVertices*numZVertices;
    const PetscInt numTotYEdges = numYEdges*numXVertices*numZVertices;
    const PetscInt numTotZEdges = numZEdges*numXVertices*numYVertices;
    const PetscInt numVertices  = numXVertices*numYVertices*numZVertices;
    const PetscInt numEdges     = numTotXEdges + numTotYEdges + numTotZEdges;
    const PetscInt firstVertex  = (dim == 2) ? numFaces : numCells;
    const PetscInt firstXFace   = (dim == 2) ? 0 : numCells + numVertices;
    const PetscInt firstYFace   = firstXFace + numTotXFaces;
    const PetscInt firstZFace   = firstYFace + numTotYFaces;
    const PetscInt firstXEdge   = numCells + numFaces + numVertices;
    const PetscInt firstYEdge   = firstXEdge + numTotXEdges;
    const PetscInt firstZEdge   = firstYEdge + numTotYEdges;
    Vec            coordinates;
    PetscSection   coordSection;
    PetscScalar   *coords;
    PetscInt       coordSize;
    PetscInt       v, vx, vy, vz;
    PetscInt       c, f, fx, fy, fz, e, ex, ey, ez;

    ierr = DMPlexSetChart(dm, 0, numCells+numFaces+numEdges+numVertices);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {
      ierr = DMPlexSetConeSize(dm, c, 6);CHKERRQ(ierr);
    }
    for (f = firstXFace; f < firstXFace+numFaces; ++f) {
      ierr = DMPlexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    for (e = firstXEdge; e < firstXEdge+numEdges; ++e) {
      ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    /* Build cells */
    for (fz = 0; fz < numZEdges; ++fz) {
      for (fy = 0; fy < numYEdges; ++fy) {
        for (fx = 0; fx < numXEdges; ++fx) {
          PetscInt cell    = (fz*numYEdges + fy)*numXEdges + fx;
          PetscInt faceB   = firstZFace + (fy*numXEdges+fx)*numZVertices +   fz;
          PetscInt faceT   = firstZFace + (fy*numXEdges+fx)*numZVertices + ((fz+1)%numZVertices);
          PetscInt faceF   = firstYFace + (fz*numXEdges+fx)*numYVertices +   fy;
          PetscInt faceK   = firstYFace + (fz*numXEdges+fx)*numYVertices + ((fy+1)%numYVertices);
          PetscInt faceL   = firstXFace + (fz*numYEdges+fy)*numXVertices +   fx;
          PetscInt faceR   = firstXFace + (fz*numYEdges+fy)*numXVertices + ((fx+1)%numXVertices);
                            /* B,  T,  F,  K,  R,  L */
          PetscInt ornt[6] = {-2,  0,  0, -3,  0, -2}; /* ??? */
          PetscInt cone[6];

          /* no boundary twisting in 3D */
          cone[0] = faceB; cone[1] = faceT; cone[2] = faceF; cone[3] = faceK; cone[4] = faceR; cone[5] = faceL;
          ierr    = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
          ierr    = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
          if (bdX != DM_BOUNDARY_NONE && fx == numXEdges-1 && cutLabel) {ierr = DMLabelSetValue(cutLabel, cell, 2);CHKERRQ(ierr);}
          if (bdY != DM_BOUNDARY_NONE && fy == numYEdges-1 && cutLabel) {ierr = DMLabelSetValue(cutLabel, cell, 2);CHKERRQ(ierr);}
          if (bdZ != DM_BOUNDARY_NONE && fz == numZEdges-1 && cutLabel) {ierr = DMLabelSetValue(cutLabel, cell, 2);CHKERRQ(ierr);}
        }
      }
    }
    /* Build x faces */
    for (fz = 0; fz < numZEdges; ++fz) {
      for (fy = 0; fy < numYEdges; ++fy) {
        for (fx = 0; fx < numXVertices; ++fx) {
          PetscInt face    = firstXFace + (fz*numYEdges+fy)     *numXVertices+fx;
          PetscInt edgeL   = firstZEdge + (fy                   *numXVertices+fx)*numZEdges + fz;
          PetscInt edgeR   = firstZEdge + (((fy+1)%numYVertices)*numXVertices+fx)*numZEdges + fz;
          PetscInt edgeB   = firstYEdge + (fz                   *numXVertices+fx)*numYEdges + fy;
          PetscInt edgeT   = firstYEdge + (((fz+1)%numZVertices)*numXVertices+fx)*numYEdges + fy;
          PetscInt ornt[4] = {0, 0, -1, -1};
          PetscInt cone[4];

          if (dim == 3) {
            /* markers */
            if (bdX != DM_BOUNDARY_PERIODIC) {
              if (fx == numXVertices-1) {
                ierr = DMSetLabelValue(dm, "Face Sets", face, faceMarkerRight);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", face, markerRight);CHKERRQ(ierr);
              }
              else if (fx == 0) {
                ierr = DMSetLabelValue(dm, "Face Sets", face, faceMarkerLeft);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", face, markerLeft);CHKERRQ(ierr);
              }
            }
          }
          cone[0] = edgeB; cone[1] = edgeR; cone[2] = edgeT; cone[3] = edgeL;
          ierr    = DMPlexSetCone(dm, face, cone);CHKERRQ(ierr);
          ierr    = DMPlexSetConeOrientation(dm, face, ornt);CHKERRQ(ierr);
        }
      }
    }
    /* Build y faces */
    for (fz = 0; fz < numZEdges; ++fz) {
      for (fx = 0; fx < numXEdges; ++fx) {
        for (fy = 0; fy < numYVertices; ++fy) {
          PetscInt face    = firstYFace + (fz*numXEdges+fx)*numYVertices + fy;
          PetscInt edgeL   = firstZEdge + (fy*numXVertices+  fx)*numZEdges + fz;
          PetscInt edgeR   = firstZEdge + (fy*numXVertices+((fx+1)%numXVertices))*numZEdges + fz;
          PetscInt edgeB   = firstXEdge + (fz                   *numYVertices+fy)*numXEdges + fx;
          PetscInt edgeT   = firstXEdge + (((fz+1)%numZVertices)*numYVertices+fy)*numXEdges + fx;
          PetscInt ornt[4] = {0, 0, -1, -1};
          PetscInt cone[4];

          if (dim == 3) {
            /* markers */
            if (bdY != DM_BOUNDARY_PERIODIC) {
              if (fy == numYVertices-1) {
                ierr = DMSetLabelValue(dm, "Face Sets", face, faceMarkerBack);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", face, markerBack);CHKERRQ(ierr);
              }
              else if (fy == 0) {
                ierr = DMSetLabelValue(dm, "Face Sets", face, faceMarkerFront);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", face, markerFront);CHKERRQ(ierr);
              }
            }
          }
          cone[0] = edgeB; cone[1] = edgeR; cone[2] = edgeT; cone[3] = edgeL;
          ierr    = DMPlexSetCone(dm, face, cone);CHKERRQ(ierr);
          ierr    = DMPlexSetConeOrientation(dm, face, ornt);CHKERRQ(ierr);
        }
      }
    }
    /* Build z faces */
    for (fy = 0; fy < numYEdges; ++fy) {
      for (fx = 0; fx < numXEdges; ++fx) {
        for (fz = 0; fz < numZVertices; fz++) {
          PetscInt face    = firstZFace + (fy*numXEdges+fx)*numZVertices + fz;
          PetscInt edgeL   = firstYEdge + (fz*numXVertices+  fx)*numYEdges + fy;
          PetscInt edgeR   = firstYEdge + (fz*numXVertices+((fx+1)%numXVertices))*numYEdges + fy;
          PetscInt edgeB   = firstXEdge + (fz*numYVertices+  fy)*numXEdges + fx;
          PetscInt edgeT   = firstXEdge + (fz*numYVertices+((fy+1)%numYVertices))*numXEdges + fx;
          PetscInt ornt[4] = {0, 0, -1, -1};
          PetscInt cone[4];

          if (dim == 2) {
            if (bdX == DM_BOUNDARY_TWIST && fx == numXEdges-1) {edgeR += numYEdges-1-2*fy; ornt[1] = -1;}
            if (bdY == DM_BOUNDARY_TWIST && fy == numYEdges-1) {edgeT += numXEdges-1-2*fx; ornt[2] =  0;}
            if (bdX != DM_BOUNDARY_NONE && fx == numXEdges-1 && cutLabel) {ierr = DMLabelSetValue(cutLabel, face, 2);CHKERRQ(ierr);}
            if (bdY != DM_BOUNDARY_NONE && fy == numYEdges-1 && cutLabel) {ierr = DMLabelSetValue(cutLabel, face, 2);CHKERRQ(ierr);}
          } else {
            /* markers */
            if (bdZ != DM_BOUNDARY_PERIODIC) {
              if (fz == numZVertices-1) {
                ierr = DMSetLabelValue(dm, "Face Sets", face, faceMarkerTop);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", face, markerTop);CHKERRQ(ierr);
              }
              else if (fz == 0) {
                ierr = DMSetLabelValue(dm, "Face Sets", face, faceMarkerBottom);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", face, markerBottom);CHKERRQ(ierr);
              }
            }
          }
          cone[0] = edgeB; cone[1] = edgeR; cone[2] = edgeT; cone[3] = edgeL;
          ierr    = DMPlexSetCone(dm, face, cone);CHKERRQ(ierr);
          ierr    = DMPlexSetConeOrientation(dm, face, ornt);CHKERRQ(ierr);
        }
      }
    }
    /* Build Z edges*/
    for (vy = 0; vy < numYVertices; vy++) {
      for (vx = 0; vx < numXVertices; vx++) {
        for (ez = 0; ez < numZEdges; ez++) {
          const PetscInt edge    = firstZEdge  + (vy*numXVertices+vx)*numZEdges + ez;
          const PetscInt vertexB = firstVertex + (ez                   *numYVertices+vy)*numXVertices + vx;
          const PetscInt vertexT = firstVertex + (((ez+1)%numZVertices)*numYVertices+vy)*numXVertices + vx;
          PetscInt       cone[2];

          if (dim == 3) {
            if (bdX != DM_BOUNDARY_PERIODIC) {
              if (vx == numXVertices-1) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerRight);CHKERRQ(ierr);
              }
              else if (vx == 0) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerLeft);CHKERRQ(ierr);
              }
            }
            if (bdY != DM_BOUNDARY_PERIODIC) {
              if (vy == numYVertices-1) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerBack);CHKERRQ(ierr);
              }
              else if (vy == 0) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerFront);CHKERRQ(ierr);
              }
            }
          }
          cone[0] = vertexB; cone[1] = vertexT;
          ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        }
      }
    }
    /* Build Y edges*/
    for (vz = 0; vz < numZVertices; vz++) {
      for (vx = 0; vx < numXVertices; vx++) {
        for (ey = 0; ey < numYEdges; ey++) {
          const PetscInt nextv   = (dim == 2 && bdY == DM_BOUNDARY_TWIST && ey == numYEdges-1) ? (numXVertices-vx-1) : (vz*numYVertices+((ey+1)%numYVertices))*numXVertices + vx;
          const PetscInt edge    = firstYEdge  + (vz*numXVertices+vx)*numYEdges + ey;
          const PetscInt vertexF = firstVertex + (vz*numYVertices+ey)*numXVertices + vx;
          const PetscInt vertexK = firstVertex + nextv;
          PetscInt       cone[2];

          cone[0] = vertexF; cone[1] = vertexK;
          ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
          if (dim == 2) {
            if ((bdX != DM_BOUNDARY_PERIODIC) && (bdX != DM_BOUNDARY_TWIST)) {
              if (vx == numXVertices-1) {
                ierr = DMSetLabelValue(dm, "Face Sets", edge, faceMarkerRight);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", edge,    markerRight);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", cone[0], markerRight);CHKERRQ(ierr);
                if (ey == numYEdges-1) {
                  ierr = DMSetLabelValue(dm, "marker", cone[1], markerRight);CHKERRQ(ierr);
                }
              } else if (vx == 0) {
                ierr = DMSetLabelValue(dm, "Face Sets", edge, faceMarkerLeft);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", edge,    markerLeft);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", cone[0], markerLeft);CHKERRQ(ierr);
                if (ey == numYEdges-1) {
                  ierr = DMSetLabelValue(dm, "marker", cone[1], markerLeft);CHKERRQ(ierr);
                }
              }
            } else {
              if (vx == 0 && cutLabel) {
                ierr = DMLabelSetValue(cutLabel, edge,    1);CHKERRQ(ierr);
                ierr = DMLabelSetValue(cutLabel, cone[0], 1);CHKERRQ(ierr);
                if (ey == numYEdges-1) {
                  ierr = DMLabelSetValue(cutLabel, cone[1], 1);CHKERRQ(ierr);
                }
              }
            }
          } else {
            if (bdX != DM_BOUNDARY_PERIODIC) {
              if (vx == numXVertices-1) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerRight);CHKERRQ(ierr);
              } else if (vx == 0) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerLeft);CHKERRQ(ierr);
              }
            }
            if (bdZ != DM_BOUNDARY_PERIODIC) {
              if (vz == numZVertices-1) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerTop);CHKERRQ(ierr);
              } else if (vz == 0) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerBottom);CHKERRQ(ierr);
              }
            }
          }
        }
      }
    }
    /* Build X edges*/
    for (vz = 0; vz < numZVertices; vz++) {
      for (vy = 0; vy < numYVertices; vy++) {
        for (ex = 0; ex < numXEdges; ex++) {
          const PetscInt nextv   = (dim == 2 && bdX == DM_BOUNDARY_TWIST && ex == numXEdges-1) ? (numYVertices-vy-1)*numXVertices : (vz*numYVertices+vy)*numXVertices + (ex+1)%numXVertices;
          const PetscInt edge    = firstXEdge  + (vz*numYVertices+vy)*numXEdges + ex;
          const PetscInt vertexL = firstVertex + (vz*numYVertices+vy)*numXVertices + ex;
          const PetscInt vertexR = firstVertex + nextv;
          PetscInt       cone[2];

          cone[0] = vertexL; cone[1] = vertexR;
          ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
          if (dim == 2) {
            if ((bdY != DM_BOUNDARY_PERIODIC) && (bdY != DM_BOUNDARY_TWIST)) {
              if (vy == numYVertices-1) {
                ierr = DMSetLabelValue(dm, "Face Sets", edge, faceMarkerTop);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", edge,    markerTop);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", cone[0], markerTop);CHKERRQ(ierr);
                if (ex == numXEdges-1) {
                  ierr = DMSetLabelValue(dm, "marker", cone[1], markerTop);CHKERRQ(ierr);
                }
              } else if (vy == 0) {
                ierr = DMSetLabelValue(dm, "Face Sets", edge, faceMarkerBottom);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", edge,    markerBottom);CHKERRQ(ierr);
                ierr = DMSetLabelValue(dm, "marker", cone[0], markerBottom);CHKERRQ(ierr);
                if (ex == numXEdges-1) {
                  ierr = DMSetLabelValue(dm, "marker", cone[1], markerBottom);CHKERRQ(ierr);
                }
              }
            } else {
              if (vy == 0 && cutLabel) {
                ierr = DMLabelSetValue(cutLabel, edge,    1);CHKERRQ(ierr);
                ierr = DMLabelSetValue(cutLabel, cone[0], 1);CHKERRQ(ierr);
                if (ex == numXEdges-1) {
                  ierr = DMLabelSetValue(cutLabel, cone[1], 1);CHKERRQ(ierr);
                }
              }
            }
          } else {
            if (bdY != DM_BOUNDARY_PERIODIC) {
              if (vy == numYVertices-1) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerBack);CHKERRQ(ierr);
              }
              else if (vy == 0) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerFront);CHKERRQ(ierr);
              }
            }
            if (bdZ != DM_BOUNDARY_PERIODIC) {
              if (vz == numZVertices-1) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerTop);CHKERRQ(ierr);
              }
              else if (vz == 0) {
                ierr = DMSetLabelValue(dm, "marker", edge, markerBottom);CHKERRQ(ierr);
              }
            }
          }
        }
      }
    }
    ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
    ierr = DMPlexStratify(dm);CHKERRQ(ierr);
    /* Build coordinates */
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVertices);CHKERRQ(ierr);
    for (v = firstVertex; v < firstVertex+numVertices; ++v) {
      ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(coordSection, v, 0, dim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
    ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
    ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (vz = 0; vz < numZVertices; ++vz) {
      for (vy = 0; vy < numYVertices; ++vy) {
        for (vx = 0; vx < numXVertices; ++vx) {
          coords[((vz*numYVertices+vy)*numXVertices+vx)*dim+0] = lower[0] + ((upper[0] - lower[0])/numXEdges)*vx;
          coords[((vz*numYVertices+vy)*numXVertices+vx)*dim+1] = lower[1] + ((upper[1] - lower[1])/numYEdges)*vy;
          if (dim == 3) {
            coords[((vz*numYVertices+vy)*numXVertices+vx)*dim+2] = lower[2] + ((upper[2] - lower[2])/numZEdges)*vz;
          }
        }
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxMesh_Tensor_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[])
{
  DMBoundaryType bdt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscInt       fac[3] = {0, 0, 0}, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {fac[d] = faces[d]; bdt[d] = periodicity[d];}
  ierr = DMPlexCreateCubeMesh_Internal(dm, lower, upper, fac, bdt[0], bdt[1], bdt[2]);CHKERRQ(ierr);
  if (periodicity[0] == DM_BOUNDARY_PERIODIC || periodicity[0] == DM_BOUNDARY_TWIST ||
      periodicity[1] == DM_BOUNDARY_PERIODIC || periodicity[1] == DM_BOUNDARY_TWIST ||
      (dim > 2 && (periodicity[2] == DM_BOUNDARY_PERIODIC || periodicity[2] == DM_BOUNDARY_TWIST))) {
    PetscReal L[3];
    PetscReal maxCell[3];

    for (d = 0; d < dim; ++d) {
      L[d]       = upper[d] - lower[d];
      maxCell[d] = 1.1 * (L[d] / PetscMax(1, faces[d]));
    }
    ierr = DMSetPeriodicity(dm, PETSC_TRUE, maxCell, L, periodicity);CHKERRQ(ierr);
  }
  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxMesh_Internal(DM dm, PetscInt dim, PetscBool simplex, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim == 1)      {ierr = DMPlexCreateLineMesh_Internal(dm, faces[0], lower[0], upper[0], periodicity[0]);CHKERRQ(ierr);}
  else if (simplex)  {ierr = DMPlexCreateBoxMesh_Simplex_Internal(dm, dim, faces, lower, upper, periodicity, interpolate);CHKERRQ(ierr);}
  else               {ierr = DMPlexCreateBoxMesh_Tensor_Internal(dm, dim, faces, lower, upper, periodicity);CHKERRQ(ierr);}
  if (!interpolate && dim > 1 && !simplex) {
    DM udm;

    ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(dm, udm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &udm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateBoxMesh - Creates a mesh on the tensor product of unit intervals (box) using simplices or tensor cells (hexahedra).

  Collective

  Input Parameters:
+ comm        - The communicator for the DM object
. dim         - The spatial dimension
. simplex     - PETSC_TRUE for simplices, PETSC_FALSE for tensor cells
. faces       - Number of faces per dimension, or NULL for (1,) in 1D and (2, 2) in 2D and (1, 1, 1) in 3D
. lower       - The lower left corner, or NULL for (0, 0, 0)
. upper       - The upper right corner, or NULL for (1, 1, 1)
. periodicity - The boundary type for the X,Y,Z direction, or NULL for DM_BOUNDARY_NONE
- interpolate - Flag to create intermediate mesh pieces (edges, faces)

  Output Parameter:
. dm  - The DM object

  Note: If you want to customize this mesh using options, you just need to
$  DMCreate(comm, &dm);
$  DMSetType(dm, DMPLEX);
$  DMSetFromOptions(dm);
and use the options on the DMSetFromOptions() page.

  Here is the numbering returned for 2 faces in each direction for tensor cells:
$ 10---17---11---18----12
$  |         |         |
$  |         |         |
$ 20    2   22    3    24
$  |         |         |
$  |         |         |
$  7---15----8---16----9
$  |         |         |
$  |         |         |
$ 19    0   21    1   23
$  |         |         |
$  |         |         |
$  4---13----5---14----6

and for simplicial cells

$ 14----8---15----9----16
$  |\     5  |\      7 |
$  | \       | \       |
$ 13   2    14    3    15
$  | 4   \   | 6   \   |
$  |       \ |       \ |
$ 11----6---12----7----13
$  |\        |\        |
$  | \    1  | \     3 |
$ 10   0    11    1    12
$  | 0   \   | 2   \   |
$  |       \ |       \ |
$  8----4----9----5----10

  Level: beginner

.seealso: DMSetFromOptions(), DMPlexCreateFromFile(), DMPlexCreateHexCylinderMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool simplex, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate, DM *dm)
{
  PetscInt       fac[3] = {1, 1, 1};
  PetscReal      low[3] = {0, 0, 0};
  PetscReal      upp[3] = {1, 1, 1};
  DMBoundaryType bdt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh_Internal(*dm, dim, simplex, faces ? faces : fac, lower ? lower : low, upper ? upper : upp, periodicity ? periodicity : bdt, interpolate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateWedgeBoxMesh_Internal(DM dm, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[])
{
  DM             bdm, vol;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i = 0; i < 3; ++i) PetscCheckFalse(periodicity[i] != DM_BOUNDARY_NONE,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Periodicity not yet supported");
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &bdm);CHKERRQ(ierr);
  ierr = DMSetType(bdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(bdm, 2);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh_Simplex_Internal(bdm, 2, faces, lower, upper, periodicity, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexExtrude(bdm, faces[2], upper[2] - lower[2], PETSC_TRUE, PETSC_FALSE, NULL, NULL, &vol);CHKERRQ(ierr);
  ierr = DMDestroy(&bdm);CHKERRQ(ierr);
  ierr = DMPlexReplace_Static(dm, &vol);CHKERRQ(ierr);
  if (lower[2] != 0.0) {
    Vec          v;
    PetscScalar *x;
    PetscInt     cDim, n;

    ierr = DMGetCoordinatesLocal(dm, &v);CHKERRQ(ierr);
    ierr = VecGetBlockSize(v, &cDim);CHKERRQ(ierr);
    ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
    ierr = VecGetArray(v, &x);CHKERRQ(ierr);
    x   += cDim;
    for (i = 0; i < n; i += cDim) x[i] += lower[2];
    ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateWedgeBoxMesh - Creates a 3-D mesh tesselating the (x,y) plane and extruding in the third direction using wedge cells.

  Collective

  Input Parameters:
+ comm        - The communicator for the DM object
. faces       - Number of faces per dimension, or NULL for (1, 1, 1)
. lower       - The lower left corner, or NULL for (0, 0, 0)
. upper       - The upper right corner, or NULL for (1, 1, 1)
. periodicity - The boundary type for the X,Y,Z direction, or NULL for DM_BOUNDARY_NONE
. orderHeight - If PETSC_TRUE, orders the extruded cells in the height first. Otherwise, orders the cell on the layers first
- interpolate - Flag to create intermediate mesh pieces (edges, faces)

  Output Parameter:
. dm  - The DM object

  Level: beginner

.seealso: DMPlexCreateHexCylinderMesh(), DMPlexCreateWedgeCylinderMesh(), DMExtrude(), DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateWedgeBoxMesh(MPI_Comm comm, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool orderHeight, PetscBool interpolate, DM *dm)
{
  PetscInt       fac[3] = {1, 1, 1};
  PetscReal      low[3] = {0, 0, 0};
  PetscReal      upp[3] = {1, 1, 1};
  DMBoundaryType bdt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateWedgeBoxMesh_Internal(*dm, faces ? faces : fac, lower ? lower : low, upper ? upper : upp, periodicity ? periodicity : bdt);CHKERRQ(ierr);
  if (!interpolate) {
    DM udm;

    ierr = DMPlexUninterpolate(*dm, &udm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(*dm, &udm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexSetOptionsPrefix - Sets the prefix used for searching for all DM options in the database.

  Logically Collective on dm

  Input Parameters:
+ dm - the DM context
- prefix - the prefix to prepend to all option names

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  Level: advanced

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode DMPlexSetOptionsPrefix(DM dm, const char prefix[])
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dm, prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) mesh->partitioner, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Remap geometry to cylinder
   TODO: This only works for a single refinement, then it is broken

     Interior square: Linear interpolation is correct
     The other cells all have vertices on rays from the origin. We want to uniformly expand the spacing
     such that the last vertex is on the unit circle. So the closest and farthest vertices are at distance

       phi     = arctan(y/x)
       d_close = sqrt(1/8 + 1/4 sin^2(phi))
       d_far   = sqrt(1/2 + sin^2(phi))

     so we remap them using

       x_new = x_close + (x - x_close) (1 - d_close) / (d_far - d_close)
       y_new = y_close + (y - y_close) (1 - d_close) / (d_far - d_close)

     If pi/4 < phi < 3pi/4 or -3pi/4 < phi < -pi/4, then we switch x and y.
*/
static void snapToCylinder(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal dis = 1.0/PetscSqrtReal(2.0);
  const PetscReal ds2 = 0.5*dis;

  if ((PetscAbsScalar(u[0]) <= ds2) && (PetscAbsScalar(u[1]) <= ds2)) {
    f0[0] = u[0];
    f0[1] = u[1];
  } else {
    PetscReal phi, sinp, cosp, dc, df, x, y, xc, yc;

    x    = PetscRealPart(u[0]);
    y    = PetscRealPart(u[1]);
    phi  = PetscAtan2Real(y, x);
    sinp = PetscSinReal(phi);
    cosp = PetscCosReal(phi);
    if ((PetscAbsReal(phi) > PETSC_PI/4.0) && (PetscAbsReal(phi) < 3.0*PETSC_PI/4.0)) {
      dc = PetscAbsReal(ds2/sinp);
      df = PetscAbsReal(dis/sinp);
      xc = ds2*x/PetscAbsReal(y);
      yc = ds2*PetscSignReal(y);
    } else {
      dc = PetscAbsReal(ds2/cosp);
      df = PetscAbsReal(dis/cosp);
      xc = ds2*PetscSignReal(x);
      yc = ds2*y/PetscAbsReal(x);
    }
    f0[0] = xc + (u[0] - xc)*(1.0 - dc)/(df - dc);
    f0[1] = yc + (u[1] - yc)*(1.0 - dc)/(df - dc);
  }
  f0[2] = u[2];
}

static PetscErrorCode DMPlexCreateHexCylinderMesh_Internal(DM dm, DMBoundaryType periodicZ)
{
  const PetscInt dim = 3;
  PetscInt       numCells, numVertices;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  /* Create topology */
  {
    PetscInt cone[8], c;

    numCells    = rank == 0 ?  5 : 0;
    numVertices = rank == 0 ? 16 : 0;
    if (periodicZ == DM_BOUNDARY_PERIODIC) {
      numCells   *= 3;
      numVertices = rank == 0 ? 24 : 0;
    }
    ierr = DMPlexSetChart(dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {ierr = DMPlexSetConeSize(dm, c, 8);CHKERRQ(ierr);}
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    if (rank == 0) {
      if (periodicZ == DM_BOUNDARY_PERIODIC) {
        cone[0] = 15; cone[1] = 18; cone[2] = 17; cone[3] = 16;
        cone[4] = 31; cone[5] = 32; cone[6] = 33; cone[7] = 34;
        ierr = DMPlexSetCone(dm, 0, cone);CHKERRQ(ierr);
        cone[0] = 16; cone[1] = 17; cone[2] = 24; cone[3] = 23;
        cone[4] = 32; cone[5] = 36; cone[6] = 37; cone[7] = 33; /* 22 25 26 21 */
        ierr = DMPlexSetCone(dm, 1, cone);CHKERRQ(ierr);
        cone[0] = 18; cone[1] = 27; cone[2] = 24; cone[3] = 17;
        cone[4] = 34; cone[5] = 33; cone[6] = 37; cone[7] = 38;
        ierr = DMPlexSetCone(dm, 2, cone);CHKERRQ(ierr);
        cone[0] = 29; cone[1] = 27; cone[2] = 18; cone[3] = 15;
        cone[4] = 35; cone[5] = 31; cone[6] = 34; cone[7] = 38;
        ierr = DMPlexSetCone(dm, 3, cone);CHKERRQ(ierr);
        cone[0] = 29; cone[1] = 15; cone[2] = 16; cone[3] = 23;
        cone[4] = 35; cone[5] = 36; cone[6] = 32; cone[7] = 31;
        ierr = DMPlexSetCone(dm, 4, cone);CHKERRQ(ierr);

        cone[0] = 31; cone[1] = 34; cone[2] = 33; cone[3] = 32;
        cone[4] = 19; cone[5] = 22; cone[6] = 21; cone[7] = 20;
        ierr = DMPlexSetCone(dm, 5, cone);CHKERRQ(ierr);
        cone[0] = 32; cone[1] = 33; cone[2] = 37; cone[3] = 36;
        cone[4] = 22; cone[5] = 25; cone[6] = 26; cone[7] = 21;
        ierr = DMPlexSetCone(dm, 6, cone);CHKERRQ(ierr);
        cone[0] = 34; cone[1] = 38; cone[2] = 37; cone[3] = 33;
        cone[4] = 20; cone[5] = 21; cone[6] = 26; cone[7] = 28;
        ierr = DMPlexSetCone(dm, 7, cone);CHKERRQ(ierr);
        cone[0] = 35; cone[1] = 38; cone[2] = 34; cone[3] = 31;
        cone[4] = 30; cone[5] = 19; cone[6] = 20; cone[7] = 28;
        ierr = DMPlexSetCone(dm, 8, cone);CHKERRQ(ierr);
        cone[0] = 35; cone[1] = 31; cone[2] = 32; cone[3] = 36;
        cone[4] = 30; cone[5] = 25; cone[6] = 22; cone[7] = 19;
        ierr = DMPlexSetCone(dm, 9, cone);CHKERRQ(ierr);

        cone[0] = 19; cone[1] = 20; cone[2] = 21; cone[3] = 22;
        cone[4] = 15; cone[5] = 16; cone[6] = 17; cone[7] = 18;
        ierr = DMPlexSetCone(dm, 10, cone);CHKERRQ(ierr);
        cone[0] = 22; cone[1] = 21; cone[2] = 26; cone[3] = 25;
        cone[4] = 16; cone[5] = 23; cone[6] = 24; cone[7] = 17;
        ierr = DMPlexSetCone(dm, 11, cone);CHKERRQ(ierr);
        cone[0] = 20; cone[1] = 28; cone[2] = 26; cone[3] = 21;
        cone[4] = 18; cone[5] = 17; cone[6] = 24; cone[7] = 27;
        ierr = DMPlexSetCone(dm, 12, cone);CHKERRQ(ierr);
        cone[0] = 30; cone[1] = 28; cone[2] = 20; cone[3] = 19;
        cone[4] = 29; cone[5] = 15; cone[6] = 18; cone[7] = 27;
        ierr = DMPlexSetCone(dm, 13, cone);CHKERRQ(ierr);
        cone[0] = 30; cone[1] = 19; cone[2] = 22; cone[3] = 25;
        cone[4] = 29; cone[5] = 23; cone[6] = 16; cone[7] = 15;
        ierr = DMPlexSetCone(dm, 14, cone);CHKERRQ(ierr);
      } else {
        cone[0] =  5; cone[1] =  8; cone[2] =  7; cone[3] =  6;
        cone[4] =  9; cone[5] = 12; cone[6] = 11; cone[7] = 10;
        ierr = DMPlexSetCone(dm, 0, cone);CHKERRQ(ierr);
        cone[0] =  6; cone[1] =  7; cone[2] = 14; cone[3] = 13;
        cone[4] = 12; cone[5] = 15; cone[6] = 16; cone[7] = 11;
        ierr = DMPlexSetCone(dm, 1, cone);CHKERRQ(ierr);
        cone[0] =  8; cone[1] = 17; cone[2] = 14; cone[3] =  7;
        cone[4] = 10; cone[5] = 11; cone[6] = 16; cone[7] = 18;
        ierr = DMPlexSetCone(dm, 2, cone);CHKERRQ(ierr);
        cone[0] = 19; cone[1] = 17; cone[2] =  8; cone[3] =  5;
        cone[4] = 20; cone[5] =  9; cone[6] = 10; cone[7] = 18;
        ierr = DMPlexSetCone(dm, 3, cone);CHKERRQ(ierr);
        cone[0] = 19; cone[1] =  5; cone[2] =  6; cone[3] = 13;
        cone[4] = 20; cone[5] = 15; cone[6] = 12; cone[7] =  9;
        ierr = DMPlexSetCone(dm, 4, cone);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
    ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  }
  /* Create cube geometry */
  {
    Vec             coordinates;
    PetscSection    coordSection;
    PetscScalar    *coords;
    PetscInt        coordSize, v;
    const PetscReal dis = 1.0/PetscSqrtReal(2.0);
    const PetscReal ds2 = dis/2.0;

    /* Build coordinates */
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(coordSection, numCells, numCells+numVertices);CHKERRQ(ierr);
    for (v = numCells; v < numCells+numVertices; ++v) {
      ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(coordSection, v, 0, dim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
    ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
    ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    if (rank == 0) {
      coords[0*dim+0] = -ds2; coords[0*dim+1] = -ds2; coords[0*dim+2] = 0.0;
      coords[1*dim+0] =  ds2; coords[1*dim+1] = -ds2; coords[1*dim+2] = 0.0;
      coords[2*dim+0] =  ds2; coords[2*dim+1] =  ds2; coords[2*dim+2] = 0.0;
      coords[3*dim+0] = -ds2; coords[3*dim+1] =  ds2; coords[3*dim+2] = 0.0;
      coords[4*dim+0] = -ds2; coords[4*dim+1] = -ds2; coords[4*dim+2] = 1.0;
      coords[5*dim+0] = -ds2; coords[5*dim+1] =  ds2; coords[5*dim+2] = 1.0;
      coords[6*dim+0] =  ds2; coords[6*dim+1] =  ds2; coords[6*dim+2] = 1.0;
      coords[7*dim+0] =  ds2; coords[7*dim+1] = -ds2; coords[7*dim+2] = 1.0;
      coords[ 8*dim+0] =  dis; coords[ 8*dim+1] = -dis; coords[ 8*dim+2] = 0.0;
      coords[ 9*dim+0] =  dis; coords[ 9*dim+1] =  dis; coords[ 9*dim+2] = 0.0;
      coords[10*dim+0] =  dis; coords[10*dim+1] = -dis; coords[10*dim+2] = 1.0;
      coords[11*dim+0] =  dis; coords[11*dim+1] =  dis; coords[11*dim+2] = 1.0;
      coords[12*dim+0] = -dis; coords[12*dim+1] =  dis; coords[12*dim+2] = 0.0;
      coords[13*dim+0] = -dis; coords[13*dim+1] =  dis; coords[13*dim+2] = 1.0;
      coords[14*dim+0] = -dis; coords[14*dim+1] = -dis; coords[14*dim+2] = 0.0;
      coords[15*dim+0] = -dis; coords[15*dim+1] = -dis; coords[15*dim+2] = 1.0;
      if (periodicZ == DM_BOUNDARY_PERIODIC) {
        /* 15 31 19 */ coords[16*dim+0] = -ds2; coords[16*dim+1] = -ds2; coords[16*dim+2] = 0.5;
        /* 16 32 22 */ coords[17*dim+0] =  ds2; coords[17*dim+1] = -ds2; coords[17*dim+2] = 0.5;
        /* 17 33 21 */ coords[18*dim+0] =  ds2; coords[18*dim+1] =  ds2; coords[18*dim+2] = 0.5;
        /* 18 34 20 */ coords[19*dim+0] = -ds2; coords[19*dim+1] =  ds2; coords[19*dim+2] = 0.5;
        /* 29 35 30 */ coords[20*dim+0] = -dis; coords[20*dim+1] = -dis; coords[20*dim+2] = 0.5;
        /* 23 36 25 */ coords[21*dim+0] =  dis; coords[21*dim+1] = -dis; coords[21*dim+2] = 0.5;
        /* 24 37 26 */ coords[22*dim+0] =  dis; coords[22*dim+1] =  dis; coords[22*dim+2] = 0.5;
        /* 27 38 28 */ coords[23*dim+0] = -dis; coords[23*dim+1] =  dis; coords[23*dim+2] = 0.5;
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  }
  /* Create periodicity */
  if (periodicZ == DM_BOUNDARY_PERIODIC || periodicZ == DM_BOUNDARY_TWIST) {
    PetscReal      L[3];
    PetscReal      maxCell[3];
    DMBoundaryType bdType[3];
    PetscReal      lower[3] = {0.0, 0.0, 0.0};
    PetscReal      upper[3] = {1.0, 1.0, 1.5};
    PetscInt       i, numZCells = 3;

    bdType[0] = DM_BOUNDARY_NONE;
    bdType[1] = DM_BOUNDARY_NONE;
    bdType[2] = periodicZ;
    for (i = 0; i < dim; i++) {
      L[i]       = upper[i] - lower[i];
      maxCell[i] = 1.1 * (L[i] / numZCells);
    }
    ierr = DMSetPeriodicity(dm, PETSC_TRUE, maxCell, L, bdType);CHKERRQ(ierr);
  }
  {
    DM          cdm;
    PetscDS     cds;
    PetscScalar c[2] = {1.0, 1.0};

    ierr = DMPlexCreateCoordinateSpace(dm, 1, snapToCylinder);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
    ierr = PetscDSSetConstants(cds, 2, c);CHKERRQ(ierr);
  }
  /* Wait for coordinate creation before doing in-place modification */
  ierr = DMPlexInterpolateInPlace_Internal(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateHexCylinderMesh - Creates a mesh on the tensor product of the unit interval with the circle (cylinder) using hexahedra.

  Collective

  Input Parameters:
+ comm      - The communicator for the DM object
- periodicZ - The boundary type for the Z direction

  Output Parameter:
. dm  - The DM object

  Note:
  Here is the output numbering looking from the bottom of the cylinder:
$       17-----14
$        |     |
$        |  2  |
$        |     |
$ 17-----8-----7-----14
$  |     |     |     |
$  |  3  |  0  |  1  |
$  |     |     |     |
$ 19-----5-----6-----13
$        |     |
$        |  4  |
$        |     |
$       19-----13
$
$ and up through the top
$
$       18-----16
$        |     |
$        |  2  |
$        |     |
$ 18----10----11-----16
$  |     |     |     |
$  |  3  |  0  |  1  |
$  |     |     |     |
$ 20-----9----12-----15
$        |     |
$        |  4  |
$        |     |
$       20-----15

  Level: beginner

.seealso: DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateHexCylinderMesh(MPI_Comm comm, DMBoundaryType periodicZ, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 3);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateHexCylinderMesh_Internal(*dm, periodicZ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateWedgeCylinderMesh_Internal(DM dm, PetscInt n, PetscBool interpolate)
{
  const PetscInt dim = 3;
  PetscInt       numCells, numVertices, v;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(n < 0,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of wedges %D cannot be negative", n);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  /* Must create the celltype label here so that we do not automatically try to compute the types */
  ierr = DMCreateLabel(dm, "celltype");CHKERRQ(ierr);
  /* Create topology */
  {
    PetscInt cone[6], c;

    numCells    = rank == 0 ?        n : 0;
    numVertices = rank == 0 ?  2*(n+1) : 0;
    ierr = DMPlexSetChart(dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {ierr = DMPlexSetConeSize(dm, c, 6);CHKERRQ(ierr);}
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {
      cone[0] =  c+n*1; cone[1] = (c+1)%n+n*1; cone[2] = 0+3*n;
      cone[3] =  c+n*2; cone[4] = (c+1)%n+n*2; cone[5] = 1+3*n;
      ierr = DMPlexSetCone(dm, c, cone);CHKERRQ(ierr);
      ierr = DMPlexSetCellType(dm, c, DM_POLYTOPE_TRI_PRISM_TENSOR);CHKERRQ(ierr);
    }
    ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
    ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  }
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = DMPlexSetCellType(dm, v, DM_POLYTOPE_POINT);CHKERRQ(ierr);
  }
  /* Create cylinder geometry */
  {
    Vec          coordinates;
    PetscSection coordSection;
    PetscScalar *coords;
    PetscInt     coordSize, c;

    /* Build coordinates */
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(coordSection, numCells, numCells+numVertices);CHKERRQ(ierr);
    for (v = numCells; v < numCells+numVertices; ++v) {
      ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(coordSection, v, 0, dim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
    ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(coordinates, dim);CHKERRQ(ierr);
    ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {
      coords[(c+0*n)*dim+0] = PetscCosReal(2.0*c*PETSC_PI/n); coords[(c+0*n)*dim+1] = PetscSinReal(2.0*c*PETSC_PI/n); coords[(c+0*n)*dim+2] = 1.0;
      coords[(c+1*n)*dim+0] = PetscCosReal(2.0*c*PETSC_PI/n); coords[(c+1*n)*dim+1] = PetscSinReal(2.0*c*PETSC_PI/n); coords[(c+1*n)*dim+2] = 0.0;
    }
    if (rank == 0) {
      coords[(2*n+0)*dim+0] = 0.0; coords[(2*n+0)*dim+1] = 0.0; coords[(2*n+0)*dim+2] = 1.0;
      coords[(2*n+1)*dim+0] = 0.0; coords[(2*n+1)*dim+1] = 0.0; coords[(2*n+1)*dim+2] = 0.0;
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  }
  /* Interpolate */
  if (interpolate) {ierr = DMPlexInterpolateInPlace_Internal(dm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateWedgeCylinderMesh - Creates a mesh on the tensor product of the unit interval with the circle (cylinder) using wedges.

  Collective

  Input Parameters:
+ comm - The communicator for the DM object
. n    - The number of wedges around the origin
- interpolate - Create edges and faces

  Output Parameter:
. dm  - The DM object

  Level: beginner

.seealso: DMPlexCreateHexCylinderMesh(), DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateWedgeCylinderMesh(MPI_Comm comm, PetscInt n, PetscBool interpolate, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateWedgeCylinderMesh_Internal(*dm, n, interpolate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscReal DiffNormReal(PetscInt dim, const PetscReal x[], const PetscReal y[])
{
  PetscReal prod = 0.0;
  PetscInt  i;
  for (i = 0; i < dim; ++i) prod += PetscSqr(x[i] - y[i]);
  return PetscSqrtReal(prod);
}
static inline PetscReal DotReal(PetscInt dim, const PetscReal x[], const PetscReal y[])
{
  PetscReal prod = 0.0;
  PetscInt  i;
  for (i = 0; i < dim; ++i) prod += x[i]*y[i];
  return prod;
}

/* The first constant is the sphere radius */
static void snapToSphere(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal r = PetscRealPart(constants[0]);
  PetscReal norm2 = 0.0, fac;
  PetscInt  n = uOff[1] - uOff[0], d;

  for (d = 0; d < n; ++d) norm2 += PetscSqr(PetscRealPart(u[d]));
  fac = r/PetscSqrtReal(norm2);
  for (d = 0; d < n; ++d) f0[d] = u[d]*fac;
}

static PetscErrorCode DMPlexCreateSphereMesh_Internal(DM dm, PetscInt dim, PetscBool simplex, PetscReal R)
{
  const PetscInt  embedDim = dim+1;
  PetscSection    coordSection;
  Vec             coordinates;
  PetscScalar    *coords;
  PetscReal      *coordsIn;
  PetscInt        numCells, numEdges, numVerts, firstVertex, v, firstEdge, coordSize, d, c, e;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(dm, simplex, 3);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm, dim+1);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
  switch (dim) {
  case 2:
    if (simplex) {
      const PetscReal radius    = PetscSqrtReal(1 + PETSC_PHI*PETSC_PHI)/(1.0 + PETSC_PHI);
      const PetscReal edgeLen   = 2.0/(1.0 + PETSC_PHI) * (R/radius);
      const PetscInt  degree    = 5;
      PetscReal       vertex[3] = {0.0, 1.0/(1.0 + PETSC_PHI), PETSC_PHI/(1.0 + PETSC_PHI)};
      PetscInt        s[3]      = {1, 1, 1};
      PetscInt        cone[3];
      PetscInt       *graph, p, i, j, k;

      vertex[0] *= R/radius; vertex[1] *= R/radius; vertex[2] *= R/radius;
      numCells    = rank == 0 ? 20 : 0;
      numVerts    = rank == 0 ? 12 : 0;
      firstVertex = numCells;
      /* Use icosahedron, which for a R-sphere has coordinates which are all cyclic permutations of

           (0, \pm 1/\phi+1, \pm \phi/\phi+1)

         where \phi^2 - \phi - 1 = 0, meaning \phi is the golden ratio \frac{1 + \sqrt{5}}{2}. The edge
         length is then given by 2/(1+\phi) = 2 * 0.38197 = 0.76393.
      */
      /* Construct vertices */
      ierr = PetscCalloc1(numVerts * embedDim, &coordsIn);CHKERRQ(ierr);
      if (rank == 0) {
        for (p = 0, i = 0; p < embedDim; ++p) {
          for (s[1] = -1; s[1] < 2; s[1] += 2) {
            for (s[2] = -1; s[2] < 2; s[2] += 2) {
              for (d = 0; d < embedDim; ++d) coordsIn[i*embedDim+d] = s[(d+p)%embedDim]*vertex[(d+p)%embedDim];
              ++i;
            }
          }
        }
      }
      /* Construct graph */
      ierr = PetscCalloc1(numVerts * numVerts, &graph);CHKERRQ(ierr);
      for (i = 0; i < numVerts; ++i) {
        for (j = 0, k = 0; j < numVerts; ++j) {
          if (PetscAbsReal(DiffNormReal(embedDim, &coordsIn[i*embedDim], &coordsIn[j*embedDim]) - edgeLen) < PETSC_SMALL) {graph[i*numVerts+j] = 1; ++k;}
        }
        PetscCheckFalse(k != degree,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid icosahedron, vertex %D degree %D != %D", i, k, degree);
      }
      /* Build Topology */
      ierr = DMPlexSetChart(dm, 0, numCells+numVerts);CHKERRQ(ierr);
      for (c = 0; c < numCells; c++) {
        ierr = DMPlexSetConeSize(dm, c, embedDim);CHKERRQ(ierr);
      }
      ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
      /* Cells */
      for (i = 0, c = 0; i < numVerts; ++i) {
        for (j = 0; j < i; ++j) {
          for (k = 0; k < j; ++k) {
            if (graph[i*numVerts+j] && graph[j*numVerts+k] && graph[k*numVerts+i]) {
              cone[0] = firstVertex+i; cone[1] = firstVertex+j; cone[2] = firstVertex+k;
              /* Check orientation */
              {
                const PetscInt epsilon[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};
                PetscReal normal[3];
                PetscInt  e, f;

                for (d = 0; d < embedDim; ++d) {
                  normal[d] = 0.0;
                  for (e = 0; e < embedDim; ++e) {
                    for (f = 0; f < embedDim; ++f) {
                      normal[d] += epsilon[d][e][f]*(coordsIn[j*embedDim+e] - coordsIn[i*embedDim+e])*(coordsIn[k*embedDim+f] - coordsIn[i*embedDim+f]);
                    }
                  }
                }
                if (DotReal(embedDim, normal, &coordsIn[i*embedDim]) < 0) {PetscInt tmp = cone[1]; cone[1] = cone[2]; cone[2] = tmp;}
              }
              ierr = DMPlexSetCone(dm, c++, cone);CHKERRQ(ierr);
            }
          }
        }
      }
      ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
      ierr = DMPlexStratify(dm);CHKERRQ(ierr);
      ierr = PetscFree(graph);CHKERRQ(ierr);
    } else {
      /*
        12-21--13
         |     |
        25  4  24
         |     |
  12-25--9-16--8-24--13
   |     |     |     |
  23  5 17  0 15  3  22
   |     |     |     |
  10-20--6-14--7-19--11
         |     |
        20  1  19
         |     |
        10-18--11
         |     |
        23  2  22
         |     |
        12-21--13
       */
      PetscInt cone[4], ornt[4];

      numCells    = rank == 0 ?  6 : 0;
      numEdges    = rank == 0 ? 12 : 0;
      numVerts    = rank == 0 ?  8 : 0;
      firstVertex = numCells;
      firstEdge   = numCells + numVerts;
      /* Build Topology */
      ierr = DMPlexSetChart(dm, 0, numCells+numEdges+numVerts);CHKERRQ(ierr);
      for (c = 0; c < numCells; c++) {
        ierr = DMPlexSetConeSize(dm, c, 4);CHKERRQ(ierr);
      }
      for (e = firstEdge; e < firstEdge+numEdges; ++e) {
        ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
      if (rank == 0) {
        /* Cell 0 */
        cone[0] = 14; cone[1] = 15; cone[2] = 16; cone[3] = 17;
        ierr = DMPlexSetCone(dm, 0, cone);CHKERRQ(ierr);
        ornt[0] = 0; ornt[1] = 0; ornt[2] = 0; ornt[3] = 0;
        ierr = DMPlexSetConeOrientation(dm, 0, ornt);CHKERRQ(ierr);
        /* Cell 1 */
        cone[0] = 18; cone[1] = 19; cone[2] = 14; cone[3] = 20;
        ierr = DMPlexSetCone(dm, 1, cone);CHKERRQ(ierr);
        ornt[0] = 0; ornt[1] = 0; ornt[2] = -1; ornt[3] = 0;
        ierr = DMPlexSetConeOrientation(dm, 1, ornt);CHKERRQ(ierr);
        /* Cell 2 */
        cone[0] = 21; cone[1] = 22; cone[2] = 18; cone[3] = 23;
        ierr = DMPlexSetCone(dm, 2, cone);CHKERRQ(ierr);
        ornt[0] = 0; ornt[1] = 0; ornt[2] = -1; ornt[3] = 0;
        ierr = DMPlexSetConeOrientation(dm, 2, ornt);CHKERRQ(ierr);
        /* Cell 3 */
        cone[0] = 19; cone[1] = 22; cone[2] = 24; cone[3] = 15;
        ierr = DMPlexSetCone(dm, 3, cone);CHKERRQ(ierr);
        ornt[0] = -1; ornt[1] = -1; ornt[2] = 0; ornt[3] = -1;
        ierr = DMPlexSetConeOrientation(dm, 3, ornt);CHKERRQ(ierr);
        /* Cell 4 */
        cone[0] = 16; cone[1] = 24; cone[2] = 21; cone[3] = 25;
        ierr = DMPlexSetCone(dm, 4, cone);CHKERRQ(ierr);
        ornt[0] = -1; ornt[1] = -1; ornt[2] = -1; ornt[3] = 0;
        ierr = DMPlexSetConeOrientation(dm, 4, ornt);CHKERRQ(ierr);
        /* Cell 5 */
        cone[0] = 20; cone[1] = 17; cone[2] = 25; cone[3] = 23;
        ierr = DMPlexSetCone(dm, 5, cone);CHKERRQ(ierr);
        ornt[0] = -1; ornt[1] = -1; ornt[2] = -1; ornt[3] = -1;
        ierr = DMPlexSetConeOrientation(dm, 5, ornt);CHKERRQ(ierr);
        /* Edges */
        cone[0] =  6; cone[1] =  7;
        ierr = DMPlexSetCone(dm, 14, cone);CHKERRQ(ierr);
        cone[0] =  7; cone[1] =  8;
        ierr = DMPlexSetCone(dm, 15, cone);CHKERRQ(ierr);
        cone[0] =  8; cone[1] =  9;
        ierr = DMPlexSetCone(dm, 16, cone);CHKERRQ(ierr);
        cone[0] =  9; cone[1] =  6;
        ierr = DMPlexSetCone(dm, 17, cone);CHKERRQ(ierr);
        cone[0] = 10; cone[1] = 11;
        ierr = DMPlexSetCone(dm, 18, cone);CHKERRQ(ierr);
        cone[0] = 11; cone[1] =  7;
        ierr = DMPlexSetCone(dm, 19, cone);CHKERRQ(ierr);
        cone[0] =  6; cone[1] = 10;
        ierr = DMPlexSetCone(dm, 20, cone);CHKERRQ(ierr);
        cone[0] = 12; cone[1] = 13;
        ierr = DMPlexSetCone(dm, 21, cone);CHKERRQ(ierr);
        cone[0] = 13; cone[1] = 11;
        ierr = DMPlexSetCone(dm, 22, cone);CHKERRQ(ierr);
        cone[0] = 10; cone[1] = 12;
        ierr = DMPlexSetCone(dm, 23, cone);CHKERRQ(ierr);
        cone[0] = 13; cone[1] =  8;
        ierr = DMPlexSetCone(dm, 24, cone);CHKERRQ(ierr);
        cone[0] = 12; cone[1] =  9;
        ierr = DMPlexSetCone(dm, 25, cone);CHKERRQ(ierr);
      }
      ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
      ierr = DMPlexStratify(dm);CHKERRQ(ierr);
      /* Build coordinates */
      ierr = PetscCalloc1(numVerts * embedDim, &coordsIn);CHKERRQ(ierr);
      if (rank == 0) {
        coordsIn[0*embedDim+0] = -R; coordsIn[0*embedDim+1] =  R; coordsIn[0*embedDim+2] = -R;
        coordsIn[1*embedDim+0] =  R; coordsIn[1*embedDim+1] =  R; coordsIn[1*embedDim+2] = -R;
        coordsIn[2*embedDim+0] =  R; coordsIn[2*embedDim+1] = -R; coordsIn[2*embedDim+2] = -R;
        coordsIn[3*embedDim+0] = -R; coordsIn[3*embedDim+1] = -R; coordsIn[3*embedDim+2] = -R;
        coordsIn[4*embedDim+0] = -R; coordsIn[4*embedDim+1] =  R; coordsIn[4*embedDim+2] =  R;
        coordsIn[5*embedDim+0] =  R; coordsIn[5*embedDim+1] =  R; coordsIn[5*embedDim+2] =  R;
        coordsIn[6*embedDim+0] = -R; coordsIn[6*embedDim+1] = -R; coordsIn[6*embedDim+2] =  R;
        coordsIn[7*embedDim+0] =  R; coordsIn[7*embedDim+1] = -R; coordsIn[7*embedDim+2] =  R;
      }
    }
    break;
  case 3:
    if (simplex) {
      const PetscReal edgeLen         = 1.0/PETSC_PHI;
      PetscReal       vertexA[4]      = {0.5, 0.5, 0.5, 0.5};
      PetscReal       vertexB[4]      = {1.0, 0.0, 0.0, 0.0};
      PetscReal       vertexC[4]      = {0.5, 0.5*PETSC_PHI, 0.5/PETSC_PHI, 0.0};
      const PetscInt  degree          = 12;
      PetscInt        s[4]            = {1, 1, 1};
      PetscInt        evenPerm[12][4] = {{0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 3, 2, 0},
                                         {2, 0, 1, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}};
      PetscInt        cone[4];
      PetscInt       *graph, p, i, j, k, l;

      vertexA[0] *= R; vertexA[1] *= R; vertexA[2] *= R; vertexA[3] *= R;
      vertexB[0] *= R; vertexB[1] *= R; vertexB[2] *= R; vertexB[3] *= R;
      vertexC[0] *= R; vertexC[1] *= R; vertexC[2] *= R; vertexC[3] *= R;
      numCells    = rank == 0 ? 600 : 0;
      numVerts    = rank == 0 ? 120 : 0;
      firstVertex = numCells;
      /* Use the 600-cell, which for a unit sphere has coordinates which are

           1/2 (\pm 1, \pm 1,    \pm 1, \pm 1)                          16
               (\pm 1,    0,       0,      0)  all cyclic permutations   8
           1/2 (\pm 1, \pm phi, \pm 1/phi, 0)  all even permutations    96

         where \phi^2 - \phi - 1 = 0, meaning \phi is the golden ratio \frac{1 + \sqrt{5}}{2}. The edge
         length is then given by 1/\phi = 0.61803.

         http://buzzard.pugetsound.edu/sage-practice/ch03s03.html
         http://mathworld.wolfram.com/600-Cell.html
      */
      /* Construct vertices */
      ierr = PetscCalloc1(numVerts * embedDim, &coordsIn);CHKERRQ(ierr);
      i    = 0;
      if (rank == 0) {
        for (s[0] = -1; s[0] < 2; s[0] += 2) {
          for (s[1] = -1; s[1] < 2; s[1] += 2) {
            for (s[2] = -1; s[2] < 2; s[2] += 2) {
              for (s[3] = -1; s[3] < 2; s[3] += 2) {
                for (d = 0; d < embedDim; ++d) coordsIn[i*embedDim+d] = s[d]*vertexA[d];
                ++i;
              }
            }
          }
        }
        for (p = 0; p < embedDim; ++p) {
          s[1] = s[2] = s[3] = 1;
          for (s[0] = -1; s[0] < 2; s[0] += 2) {
            for (d = 0; d < embedDim; ++d) coordsIn[i*embedDim+d] = s[(d+p)%embedDim]*vertexB[(d+p)%embedDim];
            ++i;
          }
        }
        for (p = 0; p < 12; ++p) {
          s[3] = 1;
          for (s[0] = -1; s[0] < 2; s[0] += 2) {
            for (s[1] = -1; s[1] < 2; s[1] += 2) {
              for (s[2] = -1; s[2] < 2; s[2] += 2) {
                for (d = 0; d < embedDim; ++d) coordsIn[i*embedDim+d] = s[evenPerm[p][d]]*vertexC[evenPerm[p][d]];
                ++i;
              }
            }
          }
        }
      }
      PetscCheckFalse(i != numVerts,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid 600-cell, vertices %D != %D", i, numVerts);
      /* Construct graph */
      ierr = PetscCalloc1(numVerts * numVerts, &graph);CHKERRQ(ierr);
      for (i = 0; i < numVerts; ++i) {
        for (j = 0, k = 0; j < numVerts; ++j) {
          if (PetscAbsReal(DiffNormReal(embedDim, &coordsIn[i*embedDim], &coordsIn[j*embedDim]) - edgeLen) < PETSC_SMALL) {graph[i*numVerts+j] = 1; ++k;}
        }
        PetscCheckFalse(k != degree,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid 600-cell, vertex %D degree %D != %D", i, k, degree);
      }
      /* Build Topology */
      ierr = DMPlexSetChart(dm, 0, numCells+numVerts);CHKERRQ(ierr);
      for (c = 0; c < numCells; c++) {
        ierr = DMPlexSetConeSize(dm, c, embedDim);CHKERRQ(ierr);
      }
      ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
      /* Cells */
      if (rank == 0) {
        for (i = 0, c = 0; i < numVerts; ++i) {
          for (j = 0; j < i; ++j) {
            for (k = 0; k < j; ++k) {
              for (l = 0; l < k; ++l) {
                if (graph[i*numVerts+j] && graph[j*numVerts+k] && graph[k*numVerts+i] &&
                    graph[l*numVerts+i] && graph[l*numVerts+j] && graph[l*numVerts+k]) {
                  cone[0] = firstVertex+i; cone[1] = firstVertex+j; cone[2] = firstVertex+k; cone[3] = firstVertex+l;
                  /* Check orientation: https://ef.gy/linear-algebra:normal-vectors-in-higher-dimensional-spaces */
                  {
                    const PetscInt epsilon[4][4][4][4] = {{{{0,  0,  0,  0}, { 0, 0,  0,  0}, { 0,  0, 0,  0}, { 0,  0,  0, 0}},
                                                           {{0,  0,  0,  0}, { 0, 0,  0,  0}, { 0,  0, 0,  1}, { 0,  0, -1, 0}},
                                                           {{0,  0,  0,  0}, { 0, 0,  0, -1}, { 0,  0, 0,  0}, { 0,  1,  0, 0}},
                                                           {{0,  0,  0,  0}, { 0, 0,  1,  0}, { 0, -1, 0,  0}, { 0,  0,  0, 0}}},

                                                          {{{0,  0,  0,  0}, { 0, 0,  0,  0}, { 0,  0, 0, -1}, { 0,  0,  1, 0}},
                                                           {{0,  0,  0,  0}, { 0, 0,  0,  0}, { 0,  0, 0,  0}, { 0,  0,  0, 0}},
                                                           {{0,  0,  0,  1}, { 0, 0,  0,  0}, { 0,  0, 0,  0}, {-1,  0,  0, 0}},
                                                           {{0,  0, -1,  0}, { 0, 0,  0,  0}, { 1,  0, 0,  0}, { 0,  0,  0, 0}}},

                                                          {{{0,  0,  0,  0}, { 0, 0,  0,  1}, { 0,  0, 0,  0}, { 0, -1,  0, 0}},
                                                           {{0,  0,  0, -1}, { 0, 0,  0,  0}, { 0,  0, 0,  0}, { 1,  0,  0, 0}},
                                                           {{0,  0,  0,  0}, { 0, 0,  0,  0}, { 0,  0, 0,  0}, { 0,  0,  0, 0}},
                                                           {{0,  1,  0,  0}, {-1, 0,  0,  0}, { 0,  0, 0,  0}, { 0,  0,  0, 0}}},

                                                          {{{0,  0,  0,  0}, { 0, 0, -1,  0}, { 0,  1, 0,  0}, { 0,  0,  0, 0}},
                                                           {{0,  0,  1,  0}, { 0, 0,  0,  0}, {-1,  0, 0,  0}, { 0,  0,  0, 0}},
                                                           {{0, -1,  0,  0}, { 1, 0,  0,  0}, { 0,  0, 0,  0}, { 0,  0,  0, 0}},
                                                           {{0,  0,  0,  0}, { 0, 0,  0,  0}, { 0,  0, 0,  0}, { 0,  0,  0, 0}}}};
                    PetscReal normal[4];
                    PetscInt  e, f, g;

                    for (d = 0; d < embedDim; ++d) {
                      normal[d] = 0.0;
                      for (e = 0; e < embedDim; ++e) {
                        for (f = 0; f < embedDim; ++f) {
                          for (g = 0; g < embedDim; ++g) {
                            normal[d] += epsilon[d][e][f][g]*(coordsIn[j*embedDim+e] - coordsIn[i*embedDim+e])*(coordsIn[k*embedDim+f] - coordsIn[i*embedDim+f])*(coordsIn[l*embedDim+f] - coordsIn[i*embedDim+f]);
                          }
                        }
                      }
                    }
                    if (DotReal(embedDim, normal, &coordsIn[i*embedDim]) < 0) {PetscInt tmp = cone[1]; cone[1] = cone[2]; cone[2] = tmp;}
                  }
                  ierr = DMPlexSetCone(dm, c++, cone);CHKERRQ(ierr);
                }
              }
            }
          }
        }
      }
      ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
      ierr = DMPlexStratify(dm);CHKERRQ(ierr);
      ierr = PetscFree(graph);CHKERRQ(ierr);
      break;
    }
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Unsupported dimension for sphere: %D", dim);
  }
  /* Create coordinates */
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, embedDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVerts);CHKERRQ(ierr);
  for (v = firstVertex; v < firstVertex+numVerts; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, embedDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, embedDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, embedDim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < numVerts; ++v) for (d = 0; d < embedDim; ++d) {coords[v*embedDim+d] = coordsIn[v*embedDim+d];}
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscFree(coordsIn);CHKERRQ(ierr);
  {
    DM          cdm;
    PetscDS     cds;
    PetscScalar c = R;

    ierr = DMPlexCreateCoordinateSpace(dm, 1, snapToSphere);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
    ierr = PetscDSSetConstants(cds, 1, &c);CHKERRQ(ierr);
  }
  /* Wait for coordinate creation before doing in-place modification */
  if (simplex) {ierr = DMPlexInterpolateInPlace_Internal(dm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

typedef void (*TPSEvaluateFunc)(const PetscReal[], PetscReal*, PetscReal[], PetscReal(*)[3]);

/*
 The Schwarz P implicit surface is

     f(x) = cos(x0) + cos(x1) + cos(x2) = 0
*/
static void TPSEvaluate_SchwarzP(const PetscReal y[3], PetscReal *f, PetscReal grad[], PetscReal (*hess)[3])
{
  PetscReal c[3] = {PetscCosReal(y[0] * PETSC_PI), PetscCosReal(y[1] * PETSC_PI), PetscCosReal(y[2] * PETSC_PI)};
  PetscReal g[3] = {-PetscSinReal(y[0] * PETSC_PI), -PetscSinReal(y[1] * PETSC_PI), -PetscSinReal(y[2] * PETSC_PI)};
  f[0] = c[0] + c[1] + c[2];
  for (PetscInt i=0; i<3; i++) {
    grad[i] = PETSC_PI * g[i];
    for (PetscInt j=0; j<3; j++) {
      hess[i][j] = (i == j) ? -PetscSqr(PETSC_PI) * c[i] : 0.;
    }
  }
}

/*
 The Gyroid implicit surface is

 f(x,y,z) = sin(pi * x) * cos (pi * (y + 1/2))  + sin(pi * (y + 1/2)) * cos(pi * (z + 1/4)) + sin(pi * (z + 1/4)) * cos(pi * x)

*/
static void TPSEvaluate_Gyroid(const PetscReal y[3], PetscReal *f, PetscReal grad[], PetscReal (*hess)[3])
{
  PetscReal s[3] = {PetscSinReal(PETSC_PI * y[0]), PetscSinReal(PETSC_PI * (y[1] + .5)), PetscSinReal(PETSC_PI * (y[2] + .25))};
  PetscReal c[3] = {PetscCosReal(PETSC_PI * y[0]), PetscCosReal(PETSC_PI * (y[1] + .5)), PetscCosReal(PETSC_PI * (y[2] + .25))};
  f[0] = s[0] * c[1] + s[1] * c[2] + s[2] * c[0];
  grad[0] = PETSC_PI * (c[0] * c[1] - s[2] * s[0]);
  grad[1] = PETSC_PI * (c[1] * c[2] - s[0] * s[1]);
  grad[2] = PETSC_PI * (c[2] * c[0] - s[1] * s[2]);
  hess[0][0] = -PetscSqr(PETSC_PI) * (s[0] * c[1] + s[2] * c[0]);
  hess[0][1] = -PetscSqr(PETSC_PI) * (c[0] * s[1]);
  hess[0][2] = -PetscSqr(PETSC_PI) * (c[2] * s[0]);
  hess[1][0] = -PetscSqr(PETSC_PI) * (s[1] * c[2] + s[0] * c[1]);
  hess[1][1] = -PetscSqr(PETSC_PI) * (c[1] * s[2]);
  hess[2][2] = -PetscSqr(PETSC_PI) * (c[0] * s[1]);
  hess[2][0] = -PetscSqr(PETSC_PI) * (s[2] * c[0] + s[1] * c[2]);
  hess[2][1] = -PetscSqr(PETSC_PI) * (c[2] * s[0]);
  hess[2][2] = -PetscSqr(PETSC_PI) * (c[1] * s[2]);
}

/*
   We wish to solve

         min_y || y - x ||^2  subject to f(y) = 0

   Let g(y) = grad(f).  The minimization problem is equivalent to asking to satisfy
   f(y) = 0 and (y-x) is parallel to g(y).  We do this by using Householder QR to obtain a basis for the
   tangent space and ask for both components in the tangent space to be zero.

   Take g to be a column vector and compute the "full QR" factorization Q R = g,
   where Q = I - 2 n n^T is a symmetric orthogonal matrix.
   The first column of Q is parallel to g so the remaining two columns span the null space.
   Let Qn = Q[:,1:] be those remaining columns.  Then Qn Qn^T is an orthogonal projector into the tangent space.
   Since Q is symmetric, this is equivalent to multipyling by Q and taking the last two entries.
   In total, we have a system of 3 equations in 3 unknowns:

     f(y) = 0                       1 equation
     Qn^T (y - x) = 0               2 equations

   Here, we compute the residual and Jacobian of this system.
*/
static void TPSNearestPointResJac(TPSEvaluateFunc feval, const PetscScalar x[], const PetscScalar y[], PetscScalar res[], PetscScalar J[])
{
  PetscReal yreal[3] = {PetscRealPart(y[0]), PetscRealPart(y[1]), PetscRealPart(y[2])};
  PetscReal d[3] = {PetscRealPart(y[0] - x[0]), PetscRealPart(y[1] - x[1]), PetscRealPart(y[2] - x[2])};
  PetscReal f, grad[3], n[3], n_y[3][3], norm, norm_y[3], nd, nd_y[3], sign;

  feval(yreal, &f, grad, n_y);

  for (PetscInt i=0; i<3; i++) n[i] = grad[i];
  norm = PetscSqrtReal(PetscSqr(n[0]) + PetscSqr(n[1]) + PetscSqr(n[2]));
  for (PetscInt i=0; i<3; i++) {
    norm_y[i] = 1. / norm * n[i] * n_y[i][i];
  }

  // Define the Householder reflector
  sign = n[0] >= 0 ? 1. : -1.;
  n[0] += norm * sign;
  for (PetscInt i=0; i<3; i++) n_y[0][i] += norm_y[i] * sign;

  norm = PetscSqrtReal(PetscSqr(n[0]) + PetscSqr(n[1]) + PetscSqr(n[2]));
  norm_y[0] = 1. / norm * (n[0] * n_y[0][0]);
  norm_y[1] = 1. / norm * (n[0] * n_y[0][1] + n[1] * n_y[1][1]);
  norm_y[2] = 1. / norm * (n[0] * n_y[0][2] + n[2] * n_y[2][2]);

  for (PetscInt i=0; i<3; i++) {
    n[i] /= norm;
    for (PetscInt j=0; j<3; j++) {
      // note that n[i] is n_old[i]/norm when executing the code below
      n_y[i][j] = n_y[i][j] / norm - n[i] / norm * norm_y[j];
    }
  }

  nd = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
  for (PetscInt i=0; i<3; i++) nd_y[i] = n[i] + n_y[0][i] * d[0] + n_y[1][i] * d[1] + n_y[2][i] * d[2];

  res[0] = f;
  res[1] = d[1] - 2 * n[1] * nd;
  res[2] = d[2] - 2 * n[2] * nd;
  // J[j][i] is J_{ij} (column major)
  for (PetscInt j=0; j<3; j++) {
    J[0 + j*3] = grad[j];
    J[1 + j*3] = (j == 1)*1. - 2 * (n_y[1][j] * nd + n[1] * nd_y[j]);
    J[2 + j*3] = (j == 2)*1. - 2 * (n_y[2][j] * nd + n[2] * nd_y[j]);
  }
}

/*
   Project x to the nearest point on the implicit surface using Newton's method.
*/
static PetscErrorCode TPSNearestPoint(TPSEvaluateFunc feval, PetscScalar x[])
{
  PetscScalar y[3] = {x[0], x[1], x[2]}; // Initial guess
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (PetscInt iter=0; iter<10; iter++) {
    PetscScalar res[3], J[9];
    PetscReal resnorm;
    TPSNearestPointResJac(feval, x, y, res, J);
    resnorm = PetscSqrtReal(PetscSqr(PetscRealPart(res[0])) + PetscSqr(PetscRealPart(res[1])) + PetscSqr(PetscRealPart(res[2])));
    if (0) { // Turn on this monitor if you need to confirm quadratic convergence
      ierr = PetscPrintf(PETSC_COMM_SELF, "[%D] res [%g %g %g]\n", iter, PetscRealPart(res[0]), PetscRealPart(res[1]), PetscRealPart(res[2]));CHKERRQ(ierr);
    }
    if (resnorm < PETSC_SMALL) break;

    // Take the Newton step
    ierr = PetscKernel_A_gets_inverse_A_3(J, 0., PETSC_FALSE, NULL);CHKERRQ(ierr);
    PetscKernel_v_gets_v_minus_A_times_w_3(y, J, res);
  }
  for (PetscInt i=0; i<3; i++) x[i] = y[i];
  PetscFunctionReturn(0);
}

const char *const DMPlexTPSTypes[] = {"SCHWARZ_P", "GYROID", "DMPlexTPSType", "DMPLEX_TPS_", NULL};

static PetscErrorCode DMPlexCreateTPSMesh_Internal(DM dm, DMPlexTPSType tpstype, const PetscInt extent[], const DMBoundaryType periodic[], PetscBool tps_distribute, PetscInt refinements, PetscInt layers, PetscReal thickness)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt topoDim = 2, spaceDim = 3, numFaces = 0, numVertices = 0, numEdges = 0;
  PetscInt (*edges)[2] = NULL, *edgeSets = NULL;
  PetscInt *cells_flat = NULL;
  PetscReal *vtxCoords = NULL;
  TPSEvaluateFunc evalFunc = NULL;
  DMLabel label;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
  PetscCheck((layers != 0) ^ (thickness == 0.), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Layers %D must be nonzero iff thickness %g is nonzero", layers, (double)thickness);
  switch (tpstype) {
  case DMPLEX_TPS_SCHWARZ_P:
    PetscCheck(!periodic || (periodic[0] == DM_BOUNDARY_NONE && periodic[1] == DM_BOUNDARY_NONE && periodic[2] == DM_BOUNDARY_NONE), PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Schwarz P does not support periodic meshes");
    if (!rank) {
      PetscInt (*cells)[6][4][4] = NULL; // [junction, junction-face, cell, conn]
      PetscInt Njunctions = 0, Ncuts = 0, Npipes[3], vcount;
      PetscReal L = 1;

      Npipes[0] = (extent[0] + 1) * extent[1] * extent[2];
      Npipes[1] = extent[0] * (extent[1] + 1) * extent[2];
      Npipes[2] = extent[0] * extent[1] * (extent[2] + 1);
      Njunctions = extent[0] * extent[1] * extent[2];
      Ncuts = 2 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]);
      numVertices = 4 * (Npipes[0] + Npipes[1] + Npipes[2]) + 8 * Njunctions;
      ierr = PetscMalloc1(3*numVertices, &vtxCoords);CHKERRQ(ierr);
      ierr = PetscMalloc1(Njunctions, &cells);CHKERRQ(ierr);
      ierr = PetscMalloc1(Ncuts*4, &edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(Ncuts*4, &edgeSets);CHKERRQ(ierr);
      // x-normal pipes
      vcount = 0;
      for (PetscInt i=0; i<extent[0]+1; i++) {
        for (PetscInt j=0; j<extent[1]; j++) {
          for (PetscInt k=0; k<extent[2]; k++) {
            for (PetscInt l=0; l<4; l++) {
              vtxCoords[vcount++] = (2*i - 1) * L;
              vtxCoords[vcount++] = 2 * j * L + PetscCosReal((2*l + 1) * PETSC_PI / 4) * L / 2;
              vtxCoords[vcount++] = 2 * k * L + PetscSinReal((2*l + 1) * PETSC_PI / 4) * L / 2;
            }
          }
        }
      }
      // y-normal pipes
      for (PetscInt i=0; i<extent[0]; i++) {
        for (PetscInt j=0; j<extent[1]+1; j++) {
          for (PetscInt k=0; k<extent[2]; k++) {
            for (PetscInt l=0; l<4; l++) {
              vtxCoords[vcount++] = 2 * i * L + PetscSinReal((2*l + 1) * PETSC_PI / 4) * L / 2;
              vtxCoords[vcount++] = (2*j - 1) * L;
              vtxCoords[vcount++] = 2 * k * L + PetscCosReal((2*l + 1) * PETSC_PI / 4) * L / 2;
            }
          }
        }
      }
      // z-normal pipes
      for (PetscInt i=0; i<extent[0]; i++) {
        for (PetscInt j=0; j<extent[1]; j++) {
          for (PetscInt k=0; k<extent[2]+1; k++) {
            for (PetscInt l=0; l<4; l++) {
              vtxCoords[vcount++] = 2 * i * L + PetscCosReal((2*l + 1) * PETSC_PI / 4) * L / 2;
              vtxCoords[vcount++] = 2 * j * L + PetscSinReal((2*l + 1) * PETSC_PI / 4) * L / 2;
              vtxCoords[vcount++] = (2*k - 1) * L;
            }
          }
        }
      }
      // junctions
      for (PetscInt i=0; i<extent[0]; i++) {
        for (PetscInt j=0; j<extent[1]; j++) {
          for (PetscInt k=0; k<extent[2]; k++) {
            const PetscInt J = (i*extent[1] + j)*extent[2] + k, Jvoff = (Npipes[0] + Npipes[1] + Npipes[2])*4 + J*8;
            PetscCheck(vcount / 3 == Jvoff, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected vertex count");
            for (PetscInt ii=0; ii<2; ii++) {
              for (PetscInt jj=0; jj<2; jj++) {
                for (PetscInt kk=0; kk<2; kk++) {
                  double Ls = (1 - sqrt(2) / 4) * L;
                  vtxCoords[vcount++] = 2*i*L + (2*ii-1) * Ls;
                  vtxCoords[vcount++] = 2*j*L + (2*jj-1) * Ls;
                  vtxCoords[vcount++] = 2*k*L + (2*kk-1) * Ls;
                }
              }
            }
            const PetscInt jfaces[3][2][4] = {
              {{3,1,0,2}, {7,5,4,6}}, // x-aligned
              {{5,4,0,1}, {7,6,2,3}}, // y-aligned
              {{6,2,0,4}, {7,3,1,5}}  // z-aligned
            };
            const PetscInt pipe_lo[3] = { // vertex numbers of pipes
              ((i * extent[1] + j) * extent[2] + k)*4,
              ((i * (extent[1] + 1) + j) * extent[2] + k + Npipes[0])*4,
              ((i * extent[1] + j) * (extent[2]+1) + k + Npipes[0] + Npipes[1])*4
            };
            const PetscInt pipe_hi[3] = { // vertex numbers of pipes
              (((i + 1) * extent[1] + j) * extent[2] + k)*4,
              ((i * (extent[1] + 1) + j + 1) * extent[2] + k + Npipes[0])*4,
              ((i * extent[1] + j) * (extent[2]+1) + k + 1 + Npipes[0] + Npipes[1])*4
            };
            for (PetscInt dir=0; dir<3; dir++) { // x,y,z
              const PetscInt ijk[3] = {i, j, k};
              for (PetscInt l=0; l<4; l++) { // rotations
                cells[J][dir*2+0][l][0] = pipe_lo[dir] + l;
                cells[J][dir*2+0][l][1] = Jvoff + jfaces[dir][0][l];
                cells[J][dir*2+0][l][2] = Jvoff + jfaces[dir][0][(l-1+4)%4];
                cells[J][dir*2+0][l][3] = pipe_lo[dir] + (l-1+4)%4;
                cells[J][dir*2+1][l][0] = Jvoff + jfaces[dir][1][l];
                cells[J][dir*2+1][l][1] = pipe_hi[dir] + l;
                cells[J][dir*2+1][l][2] = pipe_hi[dir] + (l-1+4)%4;
                cells[J][dir*2+1][l][3] = Jvoff + jfaces[dir][1][(l-1+4)%4];
                if (ijk[dir] == 0) {
                  edges[numEdges][0] = pipe_lo[dir] + l;
                  edges[numEdges][1] = pipe_lo[dir] + (l+1) % 4;
                  edgeSets[numEdges] = dir*2 + 1;
                  numEdges++;
                }
                if (ijk[dir] + 1 == extent[dir]) {
                  edges[numEdges][0] = pipe_hi[dir] + l;
                  edges[numEdges][1] = pipe_hi[dir] + (l+1) % 4;
                  edgeSets[numEdges] = dir*2 + 2;
                  numEdges++;
                }
              }
            }
          }
        }
      }
      PetscCheck(numEdges == Ncuts * 4, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Edge count %D incompatible with number of cuts %D", numEdges, Ncuts);
      numFaces = 24 * Njunctions;
      cells_flat = cells[0][0][0];
    }
    evalFunc = TPSEvaluate_SchwarzP;
    break;
  case DMPLEX_TPS_GYROID:
    if (!rank) {
      // This is a coarse mesh approximation of the gyroid shifted to being the zero of the level set
      //
      //     sin(pi*x)*cos(pi*(y+1/2)) + sin(pi*(y+1/2))*cos(pi*(z+1/4)) + sin(pi*(z+1/4))*cos(x)
      //
      // on the cell [0,2]^3.
      //
      // Think about dividing that cell into four columns, and focus on the column [0,1]x[0,1]x[0,2].
      // If you looked at the gyroid in that column at different slices of z you would see that it kind of spins
      // like a boomerang:
      //
      //     z = 0          z = 1/4        z = 1/2        z = 3/4     //
      //     -----          -------        -------        -------     //
      //                                                              //
      //     +       +      +       +      +       +      +   \   +   //
      //      \                                   /            \      //
      //       \            `-_   _-'            /              }     //
      //        *-_            `-'            _-'              /      //
      //     +     `-+      +       +      +-'     +      +   /   +   //
      //                                                              //
      //                                                              //
      //     z = 1          z = 5/4        z = 3/2        z = 7/4     //
      //     -----          -------        -------        -------     //
      //                                                              //
      //     +-_     +      +       +      +     _-+      +   /   +   //
      //        `-_            _-_            _-`            /        //
      //           \        _-'   `-_        /              {         //
      //            \                       /                \        //
      //     +       +      +       +      +       +      +   \   +   //
      //
      //
      // This course mesh approximates each of these slices by two line segments,
      // and then connects the segments in consecutive layers with quadrilateral faces.
      // All of the end points of the segments are multiples of 1/4 except for the
      // point * in the picture for z = 0 above and the similar points in other layers.
      // That point is at (gamma, gamma, 0), where gamma is calculated below.
      //
      // The column  [1,2]x[1,2]x[0,2] looks the same as this column;
      // The columns [1,2]x[0,1]x[0,2] and [0,1]x[1,2]x[0,2] are mirror images.
      //
      // As for how this method turned into the names given to the vertices:
      // that was not systematic, it was just the way it worked out in my handwritten notes.

      PetscInt facesPerBlock = 64;
      PetscInt vertsPerBlock = 56;
      PetscInt extentPlus[3];
      PetscInt numBlocks, numBlocksPlus;
      const PetscInt A =  0,   B =  1,   C =  2,   D =  3,   E =  4,   F =  5,   G =  6,   H =  7,
        II =  8,   J =  9,   K = 10,   L = 11,   M = 12,   N = 13,   O = 14,   P = 15,
        Q = 16,   R = 17,   S = 18,   T = 19,   U = 20,   V = 21,   W = 22,   X = 23,
        Y = 24,   Z = 25,  Ap = 26,  Bp = 27,  Cp = 28,  Dp = 29,  Ep = 30,  Fp = 31,
        Gp = 32,  Hp = 33,  Ip = 34,  Jp = 35,  Kp = 36,  Lp = 37,  Mp = 38,  Np = 39,
        Op = 40,  Pp = 41,  Qp = 42,  Rp = 43,  Sp = 44,  Tp = 45,  Up = 46,  Vp = 47,
        Wp = 48,  Xp = 49,  Yp = 50,  Zp = 51,  Aq = 52,  Bq = 53,  Cq = 54,  Dq = 55;
      const PetscInt pattern[64][4] =
        { /* face to vertex within the coarse discretization of a single gyroid block */
          /* layer 0 */
          {A,C,K,G},{C,B,II,K},{D,A,H,L},{B+56*1,D,L,J},{E,B+56*1,J,N},{A+56*2,E,N,H+56*2},{F,A+56*2,G+56*2,M},{B,F,M,II},
          /* layer 1 */
          {G,K,Q,O},{K,II,P,Q},{L,H,O+56*1,R},{J,L,R,P},{N,J,P,S},{H+56*2,N,S,O+56*3},{M,G+56*2,O+56*2,T},{II,M,T,P},
          /* layer 2 */
          {O,Q,Y,U},{Q,P,W,Y},{R,O+56*1,U+56*1,Ap},{P,R,Ap,W},{S,P,X,Bp},{O+56*3,S,Bp,V+56*1},{T,O+56*2,V,Z},{P,T,Z,X},
          /* layer 3 */
          {U,Y,Ep,Dp},{Y,W,Cp,Ep},{Ap,U+56*1,Dp+56*1,Gp},{W,Ap,Gp,Cp},{Bp,X,Cp+56*2,Fp},{V+56*1,Bp,Fp,Dp+56*1},{Z,V,Dp,Hp},{X,Z,Hp,Cp+56*2},
          /* layer 4 */
          {Dp,Ep,Mp,Kp},{Ep,Cp,Ip,Mp},{Gp,Dp+56*1,Lp,Np},{Cp,Gp,Np,Jp},{Fp,Cp+56*2,Jp+56*2,Pp},{Dp+56*1,Fp,Pp,Lp},{Hp,Dp,Kp,Op},{Cp+56*2,Hp,Op,Ip+56*2},
          /* layer 5 */
          {Kp,Mp,Sp,Rp},{Mp,Ip,Qp,Sp},{Np,Lp,Rp,Tp},{Jp,Np,Tp,Qp+56*1},{Pp,Jp+56*2,Qp+56*3,Up},{Lp,Pp,Up,Rp},{Op,Kp,Rp,Vp},{Ip+56*2,Op,Vp,Qp+56*2},
          /* layer 6 */
          {Rp,Sp,Aq,Yp},{Sp,Qp,Wp,Aq},{Tp,Rp,Yp,Cq},{Qp+56*1,Tp,Cq,Wp+56*1},{Up,Qp+56*3,Xp+56*1,Dq},{Rp,Up,Dq,Zp},{Vp,Rp,Zp,Bq},{Qp+56*2,Vp,Bq,Xp},
          /* layer 7 (the top is the periodic image of the bottom of layer 0) */
          {Yp,Aq,C+56*4,A+56*4},{Aq,Wp,B+56*4,C+56*4},{Cq,Yp,A+56*4,D+56*4},{Wp+56*1,Cq,D+56*4,B+56*5},{Dq,Xp+56*1,B+56*5,E+56*4},{Zp,Dq,E+56*4,A+56*6},{Bq,Zp,A+56*6,F+56*4},{Xp,Bq,F+56*4,B+56*4}
        };
      const PetscReal gamma = PetscAcosReal((PetscSqrtReal(3.)-1.) / PetscSqrtReal(2.)) / PETSC_PI;
      const PetscReal patternCoords[56][3] =
        {
          /* A  */ {1.,0.,0.},
          /* B  */ {0.,1.,0.},
          /* C  */ {gamma,gamma,0.},
          /* D  */ {1+gamma,1-gamma,0.},
          /* E  */ {2-gamma,2-gamma,0.},
          /* F  */ {1-gamma,1+gamma,0.},

          /* G  */ {.5,0,.25},
          /* H  */ {1.5,0.,.25},
          /* II */ {.5,1.,.25},
          /* J  */ {1.5,1.,.25},
          /* K  */ {.25,.5,.25},
          /* L  */ {1.25,.5,.25},
          /* M  */ {.75,1.5,.25},
          /* N  */ {1.75,1.5,.25},

          /* O  */ {0.,0.,.5},
          /* P  */ {1.,1.,.5},
          /* Q  */ {gamma,1-gamma,.5},
          /* R  */ {1+gamma,gamma,.5},
          /* S  */ {2-gamma,1+gamma,.5},
          /* T  */ {1-gamma,2-gamma,.5},

          /* U  */ {0.,.5,.75},
          /* V  */ {0.,1.5,.75},
          /* W  */ {1.,.5,.75},
          /* X  */ {1.,1.5,.75},
          /* Y  */ {.5,.75,.75},
          /* Z  */ {.5,1.75,.75},
          /* Ap */ {1.5,.25,.75},
          /* Bp */ {1.5,1.25,.75},

          /* Cp */ {1.,0.,1.},
          /* Dp */ {0.,1.,1.},
          /* Ep */ {1-gamma,1-gamma,1.},
          /* Fp */ {1+gamma,1+gamma,1.},
          /* Gp */ {2-gamma,gamma,1.},
          /* Hp */ {gamma,2-gamma,1.},

          /* Ip */ {.5,0.,1.25},
          /* Jp */ {1.5,0.,1.25},
          /* Kp */ {.5,1.,1.25},
          /* Lp */ {1.5,1.,1.25},
          /* Mp */ {.75,.5,1.25},
          /* Np */ {1.75,.5,1.25},
          /* Op */ {.25,1.5,1.25},
          /* Pp */ {1.25,1.5,1.25},

          /* Qp */ {0.,0.,1.5},
          /* Rp */ {1.,1.,1.5},
          /* Sp */ {1-gamma,gamma,1.5},
          /* Tp */ {2-gamma,1-gamma,1.5},
          /* Up */ {1+gamma,2-gamma,1.5},
          /* Vp */ {gamma,1+gamma,1.5},

          /* Wp */ {0.,.5,1.75},
          /* Xp */ {0.,1.5,1.75},
          /* Yp */ {1.,.5,1.75},
          /* Zp */ {1.,1.5,1.75},
          /* Aq */ {.5,.25,1.75},
          /* Bq */ {.5,1.25,1.75},
          /* Cq */ {1.5,.75,1.75},
          /* Dq */ {1.5,1.75,1.75},
        };
      PetscInt  (*cells)[64][4] = NULL;
      PetscBool *seen;
      PetscInt  *vertToTrueVert;
      PetscInt  count;

      for (PetscInt i = 0; i < 3; i++) extentPlus[i]  = extent[i] + 1;
      numBlocks = 1;
      for (PetscInt i = 0; i < 3; i++)     numBlocks *= extent[i];
      numBlocksPlus = 1;
      for (PetscInt i = 0; i < 3; i++) numBlocksPlus *= extentPlus[i];
      numFaces = numBlocks * facesPerBlock;
      ierr = PetscMalloc1(numBlocks, &cells);CHKERRQ(ierr);
      ierr = PetscCalloc1(numBlocksPlus * vertsPerBlock,&seen);CHKERRQ(ierr);
      for (PetscInt k = 0; k < extent[2]; k++) {
        for (PetscInt j = 0; j < extent[1]; j++) {
          for (PetscInt i = 0; i < extent[0]; i++) {
            for (PetscInt f = 0; f < facesPerBlock; f++) {
              for (PetscInt v = 0; v < 4; v++) {
                PetscInt vertRaw = pattern[f][v];
                PetscInt blockidx = vertRaw / 56;
                PetscInt patternvert = vertRaw % 56;
                PetscInt xplus = (blockidx & 1);
                PetscInt yplus = (blockidx & 2) >> 1;
                PetscInt zplus = (blockidx & 4) >> 2;
                PetscInt zcoord = (periodic && periodic[2] == DM_BOUNDARY_PERIODIC) ? ((k + zplus) % extent[2]) : (k + zplus);
                PetscInt ycoord = (periodic && periodic[1] == DM_BOUNDARY_PERIODIC) ? ((j + yplus) % extent[1]) : (j + yplus);
                PetscInt xcoord = (periodic && periodic[0] == DM_BOUNDARY_PERIODIC) ? ((i + xplus) % extent[0]) : (i + xplus);
                PetscInt vert = ((zcoord * extentPlus[1] + ycoord) * extentPlus[0] + xcoord) * 56 + patternvert;

                cells[(k * extent[1] + j) * extent[0] + i][f][v] = vert;
                seen[vert] = PETSC_TRUE;
              }
            }
          }
        }
      }
      for (PetscInt i = 0; i < numBlocksPlus * vertsPerBlock; i++) if (seen[i]) numVertices++;
      count = 0;
      ierr = PetscMalloc1(numBlocksPlus * vertsPerBlock, &vertToTrueVert);CHKERRQ(ierr);
      ierr = PetscMalloc1(numVertices * 3, &vtxCoords);CHKERRQ(ierr);
      for (PetscInt i = 0; i < numBlocksPlus * vertsPerBlock; i++) vertToTrueVert[i] = -1;
      for (PetscInt k = 0; k < extentPlus[2]; k++) {
        for (PetscInt j = 0; j < extentPlus[1]; j++) {
          for (PetscInt i = 0; i < extentPlus[0]; i++) {
            for (PetscInt v = 0; v < vertsPerBlock; v++) {
              PetscInt vIdx = ((k * extentPlus[1] + j) * extentPlus[0] + i) * vertsPerBlock + v;

              if (seen[vIdx]) {
                PetscInt thisVert;

                vertToTrueVert[vIdx] = thisVert = count++;

                for (PetscInt d = 0; d < 3; d++) vtxCoords[3 * thisVert + d] = patternCoords[v][d];
                vtxCoords[3 * thisVert + 0] += i * 2;
                vtxCoords[3 * thisVert + 1] += j * 2;
                vtxCoords[3 * thisVert + 2] += k * 2;
              }
            }
          }
        }
      }
      for (PetscInt i = 0; i < numBlocks; i++) {
        for (PetscInt f = 0; f < facesPerBlock; f++) {
          for (PetscInt v = 0; v < 4; v++) {
            cells[i][f][v] = vertToTrueVert[cells[i][f][v]];
          }
        }
      }
      ierr = PetscFree(vertToTrueVert);CHKERRQ(ierr);
      ierr = PetscFree(seen);CHKERRQ(ierr);
      cells_flat = cells[0][0];
      numEdges = 0;
      for (PetscInt i = 0; i < numFaces; i++) {
        for (PetscInt e = 0; e < 4; e++) {
          PetscInt ev[] = {cells_flat[i*4 + e], cells_flat[i*4 + ((e+1)%4)]};
          const PetscReal *evCoords[] = {&vtxCoords[3*ev[0]], &vtxCoords[3*ev[1]]};

          for (PetscInt d = 0; d < 3; d++) {
            if (!periodic || periodic[0] != DM_BOUNDARY_PERIODIC) {
              if (evCoords[0][d] == 0. && evCoords[1][d] == 0.) numEdges++;
              if (evCoords[0][d] == 2.*extent[d] && evCoords[1][d] == 2.*extent[d]) numEdges++;
            }
          }
        }
      }
      ierr = PetscMalloc1(numEdges, &edges);CHKERRQ(ierr);
      ierr = PetscMalloc1(numEdges, &edgeSets);CHKERRQ(ierr);
      for (PetscInt edge = 0, i = 0; i < numFaces; i++) {
        for (PetscInt e = 0; e < 4; e++) {
          PetscInt ev[] = {cells_flat[i*4 + e], cells_flat[i*4 + ((e+1)%4)]};
          const PetscReal *evCoords[] = {&vtxCoords[3*ev[0]], &vtxCoords[3*ev[1]]};

          for (PetscInt d = 0; d < 3; d++) {
            if (!periodic || periodic[d] != DM_BOUNDARY_PERIODIC) {
              if (evCoords[0][d] == 0. && evCoords[1][d] == 0.) {
                edges[edge][0] = ev[0];
                edges[edge][1] = ev[1];
                edgeSets[edge++] = 2 * d;
              }
              if (evCoords[0][d] == 2.*extent[d] && evCoords[1][d] == 2.*extent[d]) {
                edges[edge][0] = ev[0];
                edges[edge][1] = ev[1];
                edgeSets[edge++] = 2 * d + 1;
              }
            }
          }
        }
      }
    }
    evalFunc = TPSEvaluate_Gyroid;
    break;
  }

  ierr = DMSetDimension(dm, topoDim);CHKERRQ(ierr);
  if (!rank) {ierr = DMPlexBuildFromCellList(dm, numFaces, numVertices, 4, cells_flat);CHKERRQ(ierr);}
  else       {ierr = DMPlexBuildFromCellList(dm, 0, 0, 0, NULL);CHKERRQ(ierr);}
  ierr = PetscFree(cells_flat);CHKERRQ(ierr);
  {
    DM idm;
    ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &idm);CHKERRQ(ierr);
  }
  if (!rank) {ierr = DMPlexBuildCoordinatesFromCellList(dm, spaceDim, vtxCoords);CHKERRQ(ierr);}
  else       {ierr = DMPlexBuildCoordinatesFromCellList(dm, spaceDim, NULL);CHKERRQ(ierr);}
  ierr = PetscFree(vtxCoords);CHKERRQ(ierr);

  ierr = DMCreateLabel(dm, "Face Sets");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Face Sets", &label);CHKERRQ(ierr);
  for (PetscInt e=0; e<numEdges; e++) {
    PetscInt njoin;
    const PetscInt *join, verts[] = {numFaces + edges[e][0], numFaces + edges[e][1]};
    ierr = DMPlexGetJoin(dm, 2, verts, &njoin, &join);CHKERRQ(ierr);
    PetscCheck(njoin == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected unique join of vertices %D and %D", edges[e][0], edges[e][1]);
    ierr = DMLabelSetValue(label, join[0], edgeSets[e]);CHKERRQ(ierr);
    ierr = DMPlexRestoreJoin(dm, 2, verts, &njoin, &join);CHKERRQ(ierr);
  }
  ierr = PetscFree(edges);CHKERRQ(ierr);
  ierr = PetscFree(edgeSets);CHKERRQ(ierr);
  if (tps_distribute) {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMPlexReplace_Static(dm, &pdm);CHKERRQ(ierr);
    }
    // Do not auto-distribute again
    ierr = DMPlexDistributeSetDefault(dm, PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  for (PetscInt refine=0; refine<refinements; refine++) {
    PetscInt m;
    DM dmf;
    Vec X;
    PetscScalar *x;
    ierr = DMRefine(dm, MPI_COMM_NULL, &dmf);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &dmf);CHKERRQ(ierr);

    ierr = DMGetCoordinatesLocal(dm, &X);CHKERRQ(ierr);
    ierr = VecGetLocalSize(X, &m);CHKERRQ(ierr);
    ierr = VecGetArray(X, &x);CHKERRQ(ierr);
    for (PetscInt i=0; i<m; i+=3) {
      ierr = TPSNearestPoint(evalFunc, &x[i]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
  }

  // Face Sets has already been propagated to new vertices during refinement; this propagates to the initial vertices.
  ierr = DMGetLabel(dm, "Face Sets", &label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);

  if (thickness > 0) {
    DM dm3;
    ierr = DMPlexExtrude(dm, layers, thickness, PETSC_FALSE, PETSC_TRUE, NULL, NULL, &dm3);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &dm3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateTPSMesh - Create a distributed, interpolated mesh of a triply-periodic surface

  Collective

  Input Parameters:
+ comm   - The communicator for the DM object
. tpstype - Type of triply-periodic surface
. extent - Array of length 3 containing number of periods in each direction
. periodic - array of length 3 with periodicity, or NULL for non-periodic
. tps_distribute - Distribute 2D manifold mesh prior to refinement and extrusion (more scalable)
. refinements - Number of factor-of-2 refinements of 2D manifold mesh
. layers - Number of cell layers extruded in normal direction
- thickness - Thickness in normal direction

  Output Parameter:
. dm  - The DM object

  Notes:
  This meshes the surface of the Schwarz P or Gyroid surfaces.  Schwarz P is is the simplest member of the triply-periodic minimal surfaces.
  https://en.wikipedia.org/wiki/Schwarz_minimal_surface#Schwarz_P_(%22Primitive%22) and can be cut with "clean" boundaries.
  The Gyroid (https://en.wikipedia.org/wiki/Gyroid) is another triply-periodic minimal surface with applications in additive manufacturing; it is much more difficult to "cut" since there are no planes of symmetry.
  Our implementation creates a very coarse mesh of the surface and refines (by 4-way splitting) as many times as requested.
  On each refinement, all vertices are projected to their nearest point on the surface.
  This projection could readily be extended to related surfaces.

  The face (edge) sets for the Schwarz P surface are numbered 1(-x), 2(+x), 3(-y), 4(+y), 5(-z), 6(+z).
  When the mesh is refined, "Face Sets" contain the new vertices (created during refinement).  Use DMPlexLabelComplete() to propagate to coarse-level vertices.

  References:
. * - Maskery et al, Insights into the mechanical properties of several triply periodic minimal surface lattice structures made by polymer additive manufacturing, 2017. https://doi.org/10.1016/j.polymer.2017.11.049

  Developer Notes:
  The Gyroid mesh does not currently mark boundary sets.

  Level: beginner

.seealso: DMPlexCreateSphereMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateTPSMesh(MPI_Comm comm, DMPlexTPSType tpstype, const PetscInt extent[], const DMBoundaryType periodic[], PetscBool tps_distribute, PetscInt refinements, PetscInt layers, PetscReal thickness, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateTPSMesh_Internal(*dm, tpstype, extent, periodic, tps_distribute, refinements, layers, thickness);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateSphereMesh - Creates a mesh on the d-dimensional sphere, S^d.

  Collective

  Input Parameters:
+ comm    - The communicator for the DM object
. dim     - The dimension
. simplex - Use simplices, or tensor product cells
- R       - The radius

  Output Parameter:
. dm  - The DM object

  Level: beginner

.seealso: DMPlexCreateBallMesh(), DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateSphereMesh(MPI_Comm comm, PetscInt dim, PetscBool simplex, PetscReal R, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 5);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateSphereMesh_Internal(*dm, dim, simplex, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBallMesh_Internal(DM dm, PetscInt dim, PetscReal R)
{
  DM             sdm, vol;
  DMLabel        bdlabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) sdm, "bd_");CHKERRQ(ierr);
  ierr = DMPlexCreateSphereMesh_Internal(sdm, dim-1, PETSC_TRUE, R);CHKERRQ(ierr);
  ierr = DMSetFromOptions(sdm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(sdm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMPlexGenerate(sdm, NULL, PETSC_TRUE, &vol);CHKERRQ(ierr);
  ierr = DMDestroy(&sdm);CHKERRQ(ierr);
  ierr = DMPlexReplace_Static(dm, &vol);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "marker");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "marker", &bdlabel);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, bdlabel);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, bdlabel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateBallMesh - Creates a simplex mesh on the d-dimensional ball, B^d.

  Collective

  Input Parameters:
+ comm  - The communicator for the DM object
. dim   - The dimension
- R     - The radius

  Output Parameter:
. dm  - The DM object

  Options Database Keys:
- bd_dm_refine - This will refine the surface mesh preserving the sphere geometry

  Level: beginner

.seealso: DMPlexCreateSphereMesh(), DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateBallMesh(MPI_Comm comm, PetscInt dim, PetscReal R, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateBallMesh_Internal(*dm, dim, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateReferenceCell_Internal(DM rdm, DMPolytopeType ct)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (ct) {
    case DM_POLYTOPE_POINT:
    {
      PetscInt    numPoints[1]        = {1};
      PetscInt    coneSize[1]         = {0};
      PetscInt    cones[1]            = {0};
      PetscInt    coneOrientations[1] = {0};
      PetscScalar vertexCoords[1]     = {0.0};

      ierr = DMSetDimension(rdm, 0);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 0, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_SEGMENT:
    {
      PetscInt    numPoints[2]        = {2, 1};
      PetscInt    coneSize[3]         = {2, 0, 0};
      PetscInt    cones[2]            = {1, 2};
      PetscInt    coneOrientations[2] = {0, 0};
      PetscScalar vertexCoords[2]     = {-1.0,  1.0};

      ierr = DMSetDimension(rdm, 1);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    {
      PetscInt    numPoints[2]        = {2, 1};
      PetscInt    coneSize[3]         = {2, 0, 0};
      PetscInt    cones[2]            = {1, 2};
      PetscInt    coneOrientations[2] = {0, 0};
      PetscScalar vertexCoords[2]     = {-1.0,  1.0};

      ierr = DMSetDimension(rdm, 1);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_TRIANGLE:
    {
      PetscInt    numPoints[2]        = {3, 1};
      PetscInt    coneSize[4]         = {3, 0, 0, 0};
      PetscInt    cones[3]            = {1, 2, 3};
      PetscInt    coneOrientations[3] = {0, 0, 0};
      PetscScalar vertexCoords[6]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0};

      ierr = DMSetDimension(rdm, 2);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_QUADRILATERAL:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0};

      ierr = DMSetDimension(rdm, 2);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,  1.0, 1.0};

      ierr = DMSetDimension(rdm, 2);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_TETRAHEDRON:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {-1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, -1.0, 1.0};

      ierr = DMSetDimension(rdm, 3);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_HEXAHEDRON:
    {
      PetscInt    numPoints[2]        = {8, 1};
      PetscInt    coneSize[9]         = {8, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {1, 2, 3, 4, 5, 6, 7, 8};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  -1.0,  1.0, -1.0,  1.0, 1.0, -1.0,   1.0, -1.0, -1.0,
                                         -1.0, -1.0,  1.0,   1.0, -1.0,  1.0,  1.0, 1.0,  1.0,  -1.0,  1.0,  1.0};

      ierr = DMSetDimension(rdm, 3);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_TRI_PRISM:
    {
      PetscInt    numPoints[2]        = {6, 1};
      PetscInt    coneSize[7]         = {6, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[6]            = {1, 2, 3, 4, 5, 6};
      PetscInt    coneOrientations[6] = {0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[18]    = {-1.0, -1.0, -1.0, -1.0,  1.0, -1.0,   1.0, -1.0, -1.0,
                                         -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  -1.0,  1.0,  1.0};

      ierr = DMSetDimension(rdm, 3);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    {
      PetscInt    numPoints[2]        = {6, 1};
      PetscInt    coneSize[7]         = {6, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[6]            = {1, 2, 3, 4, 5, 6};
      PetscInt    coneOrientations[6] = {0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[18]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,
                                         -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  -1.0, 1.0,  1.0};

      ierr = DMSetDimension(rdm, 3);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    {
      PetscInt    numPoints[2]        = {8, 1};
      PetscInt    coneSize[9]         = {8, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {1, 2, 3, 4, 5, 6, 7, 8};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, 1.0, -1.0,  -1.0, 1.0, -1.0,
                                         -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0, 1.0,  1.0,  -1.0, 1.0,  1.0};

      ierr = DMSetDimension(rdm, 3);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case DM_POLYTOPE_PYRAMID:
    {
      PetscInt    numPoints[2]        = {5, 1};
      PetscInt    coneSize[6]         = {5, 0, 0, 0, 0, 0};
      PetscInt    cones[5]            = {1, 2, 3, 4, 5};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  1.0, 1.0, -1.0,  1.0, -1.0, -1.0,
                                          0.0,  0.0,  1.0};

      ierr = DMSetDimension(rdm, 3);CHKERRQ(ierr);
      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    default: SETERRQ(PetscObjectComm((PetscObject) rdm), PETSC_ERR_ARG_WRONG, "Cannot create reference cell for cell type %s", DMPolytopeTypes[ct]);
  }
  {
    PetscInt Nv, v;

    /* Must create the celltype label here so that we do not automatically try to compute the types */
    ierr = DMCreateLabel(rdm, "celltype");CHKERRQ(ierr);
    ierr = DMPlexSetCellType(rdm, 0, ct);CHKERRQ(ierr);
    ierr = DMPlexGetChart(rdm, NULL, &Nv);CHKERRQ(ierr);
    for (v = 1; v < Nv; ++v) {ierr = DMPlexSetCellType(rdm, v, DM_POLYTOPE_POINT);CHKERRQ(ierr);}
  }
  ierr = DMPlexInterpolateInPlace_Internal(rdm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) rdm, DMPolytopeTypes[ct]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateReferenceCell - Create a DMPLEX with the appropriate FEM reference cell

  Collective

  Input Parameters:
+ comm - The communicator
- ct   - The cell type of the reference cell

  Output Parameter:
. refdm - The reference cell

  Level: intermediate

.seealso: DMPlexCreateReferenceCell(), DMPlexCreateBoxMesh()
@*/
PetscErrorCode DMPlexCreateReferenceCell(MPI_Comm comm, DMPolytopeType ct, DM *refdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, refdm);CHKERRQ(ierr);
  ierr = DMSetType(*refdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexCreateReferenceCell_Internal(*refdm, ct);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoundaryLabel_Private(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMHasLabel(dm, name, &hasLabel);CHKERRQ(ierr);
  if (hasLabel) PetscFunctionReturn(0);
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(plex, 1, label);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char * const DMPlexShapes[] = {"box", "box_surface", "ball", "sphere", "cylinder", "schwarz_p", "gyroid", "unknown", "DMPlexShape", "DM_SHAPE_", NULL};

static PetscErrorCode DMPlexCreateFromOptions_Internal(PetscOptionItems *PetscOptionsObject, PetscBool *useCoordSpace, DM dm)
{
  DMPlexShape    shape = DM_SHAPE_BOX;
  DMPolytopeType cell  = DM_POLYTOPE_TRIANGLE;
  PetscInt       dim   = 2;
  PetscBool      simplex = PETSC_TRUE, interpolate = PETSC_TRUE, adjCone = PETSC_FALSE, adjClosure = PETSC_TRUE, refDomain = PETSC_FALSE;
  PetscBool      flg, flg2, fflg, bdfflg, nameflg;
  MPI_Comm       comm;
  char           filename[PETSC_MAX_PATH_LEN]   = "<unspecified>";
  char           bdFilename[PETSC_MAX_PATH_LEN] = "<unspecified>";
  char           plexname[PETSC_MAX_PATH_LEN]   = "";
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  /* TODO Turn this into a registration interface */
  ierr = PetscOptionsString("-dm_plex_filename", "File containing a mesh", "DMPlexCreateFromFile", filename, filename, sizeof(filename), &fflg);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_boundary_filename", "File containing a mesh boundary", "DMPlexCreateFromFile", bdFilename, bdFilename, sizeof(bdFilename), &bdfflg);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_name", "Name of the mesh in the file", "DMPlexCreateFromFile", plexname, plexname, sizeof(plexname), &nameflg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-dm_plex_cell", "Cell shape", "", DMPolytopeTypes, (PetscEnum) cell, (PetscEnum *) &cell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_reference_cell_domain", "Use a reference cell domain", "", refDomain, &refDomain, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-dm_plex_shape", "Shape for built-in mesh", "", DMPlexShapes, (PetscEnum) shape, (PetscEnum *) &shape, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_dim", "Topological dimension of the mesh", "DMGetDimension", dim, &dim, &flg, 0);CHKERRQ(ierr);
  PetscCheckFalse((dim < 0) || (dim > 3),comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D should be in [1, 3]", dim);
  ierr = PetscOptionsBool("-dm_plex_simplex", "Mesh cell shape", "", simplex,  &simplex, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_interpolate", "Flag to create edges and faces automatically", "", interpolate, &interpolate, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_adj_cone", "Set adjacency direction", "DMSetBasicAdjacency", adjCone,  &adjCone, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_adj_closure", "Set adjacency size", "DMSetBasicAdjacency", adjClosure,  &adjClosure, &flg2);CHKERRQ(ierr);
  if (flg || flg2) {ierr = DMSetBasicAdjacency(dm, adjCone, adjClosure);CHKERRQ(ierr);}

  switch (cell) {
    case DM_POLYTOPE_POINT:
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    case DM_POLYTOPE_TRIANGLE:
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_TETRAHEDRON:
    case DM_POLYTOPE_HEXAHEDRON:
      *useCoordSpace = PETSC_TRUE;break;
    default: *useCoordSpace = PETSC_FALSE;break;
  }

  if (fflg) {
    DM dmnew;

    ierr = DMPlexCreateFromFile(PetscObjectComm((PetscObject) dm), filename, plexname, interpolate, &dmnew);CHKERRQ(ierr);
    ierr = DMPlexCopy_Internal(dm, PETSC_FALSE, dmnew);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &dmnew);CHKERRQ(ierr);
  } else if (refDomain) {
    ierr = DMPlexCreateReferenceCell_Internal(dm, cell);CHKERRQ(ierr);
  } else if (bdfflg) {
    DM bdm, dmnew;

    ierr = DMPlexCreateFromFile(PetscObjectComm((PetscObject) dm), bdFilename, plexname, interpolate, &bdm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) bdm, "bd_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(bdm);CHKERRQ(ierr);
    ierr = DMPlexGenerate(bdm, NULL, interpolate, &dmnew);CHKERRQ(ierr);
    ierr = DMDestroy(&bdm);CHKERRQ(ierr);
    ierr = DMPlexCopy_Internal(dm, PETSC_FALSE, dmnew);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &dmnew);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectSetName((PetscObject) dm, DMPlexShapes[shape]);CHKERRQ(ierr);
    switch (shape) {
      case DM_SHAPE_BOX:
      {
        PetscInt       faces[3] = {0, 0, 0};
        PetscReal      lower[3] = {0, 0, 0};
        PetscReal      upper[3] = {1, 1, 1};
        DMBoundaryType bdt[3]   = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
        PetscInt       i, n;

        n    = dim;
        for (i = 0; i < dim; ++i) faces[i] = (dim == 1 ? 1 : 4-dim);
        ierr = PetscOptionsIntArray("-dm_plex_box_faces", "Number of faces along each dimension", "", faces, &n, &flg);CHKERRQ(ierr);
        n    = 3;
        ierr = PetscOptionsRealArray("-dm_plex_box_lower", "Lower left corner of box", "", lower, &n, &flg);CHKERRQ(ierr);
        PetscCheckFalse(flg && (n != dim),comm, PETSC_ERR_ARG_SIZ, "Lower box point had %D values, should have been %D", n, dim);
        n    = 3;
        ierr = PetscOptionsRealArray("-dm_plex_box_upper", "Upper right corner of box", "", upper, &n, &flg);CHKERRQ(ierr);
        PetscCheckFalse(flg && (n != dim),comm, PETSC_ERR_ARG_SIZ, "Upper box point had %D values, should have been %D", n, dim);
        n    = 3;
        ierr = PetscOptionsEnumArray("-dm_plex_box_bd", "Boundary type for each dimension", "", DMBoundaryTypes, (PetscEnum *) bdt, &n, &flg);CHKERRQ(ierr);
        PetscCheckFalse(flg && (n != dim),comm, PETSC_ERR_ARG_SIZ, "Box boundary types had %D values, should have been %D", n, dim);
        switch (cell) {
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
            ierr = DMPlexCreateWedgeBoxMesh_Internal(dm, faces, lower, upper, bdt);CHKERRQ(ierr);
            if (!interpolate) {
              DM udm;

              ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
              ierr = DMPlexReplace_Static(dm, &udm);CHKERRQ(ierr);
            }
            break;
          default:
            ierr = DMPlexCreateBoxMesh_Internal(dm, dim, simplex, faces, lower, upper, bdt, interpolate);CHKERRQ(ierr);
            break;
        }
      }
      break;
      case DM_SHAPE_BOX_SURFACE:
      {
        PetscInt  faces[3] = {0, 0, 0};
        PetscReal lower[3] = {0, 0, 0};
        PetscReal upper[3] = {1, 1, 1};
        PetscInt  i, n;

        n    = dim+1;
        for (i = 0; i < dim+1; ++i) faces[i] = (dim+1 == 1 ? 1 : 4-(dim+1));
        ierr = PetscOptionsIntArray("-dm_plex_box_faces", "Number of faces along each dimension", "", faces, &n, &flg);CHKERRQ(ierr);
        n    = 3;
        ierr = PetscOptionsRealArray("-dm_plex_box_lower", "Lower left corner of box", "", lower, &n, &flg);CHKERRQ(ierr);
        PetscCheckFalse(flg && (n != dim+1),comm, PETSC_ERR_ARG_SIZ, "Lower box point had %D values, should have been %D", n, dim+1);
        n    = 3;
        ierr = PetscOptionsRealArray("-dm_plex_box_upper", "Upper right corner of box", "", upper, &n, &flg);CHKERRQ(ierr);
        PetscCheckFalse(flg && (n != dim+1),comm, PETSC_ERR_ARG_SIZ, "Upper box point had %D values, should have been %D", n, dim+1);
        ierr = DMPlexCreateBoxSurfaceMesh_Internal(dm, dim+1, faces, lower, upper, interpolate);CHKERRQ(ierr);
      }
      break;
      case DM_SHAPE_SPHERE:
      {
        PetscReal R = 1.0;

        ierr = PetscOptionsReal("-dm_plex_sphere_radius", "Radius of the sphere", "", R,  &R, &flg);CHKERRQ(ierr);
        ierr = DMPlexCreateSphereMesh_Internal(dm, dim, simplex, R);CHKERRQ(ierr);
      }
      break;
      case DM_SHAPE_BALL:
      {
        PetscReal R = 1.0;

        ierr = PetscOptionsReal("-dm_plex_ball_radius", "Radius of the ball", "", R,  &R, &flg);CHKERRQ(ierr);
        ierr = DMPlexCreateBallMesh_Internal(dm, dim, R);CHKERRQ(ierr);
      }
      break;
      case DM_SHAPE_CYLINDER:
      {
        DMBoundaryType bdt = DM_BOUNDARY_NONE;
        PetscInt       Nw  = 6;

        ierr = PetscOptionsEnum("-dm_plex_cylinder_bd", "Boundary type in the z direction", "", DMBoundaryTypes, (PetscEnum) bdt, (PetscEnum *) &bdt, NULL);CHKERRQ(ierr);
        ierr = PetscOptionsInt("-dm_plex_cylinder_num_wedges", "Number of wedges around the cylinder", "", Nw, &Nw, NULL);CHKERRQ(ierr);
        switch (cell) {
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
            ierr = DMPlexCreateWedgeCylinderMesh_Internal(dm, Nw, interpolate);CHKERRQ(ierr);
            break;
          default:
            ierr = DMPlexCreateHexCylinderMesh_Internal(dm, bdt);CHKERRQ(ierr);
            break;
        }
      }
      break;
      case DM_SHAPE_SCHWARZ_P: // fallthrough
      case DM_SHAPE_GYROID:
      {
        PetscInt       extent[3] = {1,1,1}, refine = 0, layers = 0, three;
        PetscReal      thickness = 0.;
        DMBoundaryType periodic[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
        DMPlexTPSType  tps_type = shape == DM_SHAPE_SCHWARZ_P ? DMPLEX_TPS_SCHWARZ_P : DMPLEX_TPS_GYROID;
        PetscBool      tps_distribute;
        ierr = PetscOptionsIntArray("-dm_plex_tps_extent", "Number of replicas for each of three dimensions", NULL, extent, (three=3, &three), NULL);CHKERRQ(ierr);
        ierr = PetscOptionsInt("-dm_plex_tps_refine", "Number of refinements", NULL, refine, &refine, NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnumArray("-dm_plex_tps_periodic", "Periodicity in each of three dimensions", NULL, DMBoundaryTypes, (PetscEnum*)periodic, (three=3, &three), NULL);CHKERRQ(ierr);
        ierr = PetscOptionsInt("-dm_plex_tps_layers", "Number of layers in volumetric extrusion (or zero to not extrude)", NULL, layers, &layers, NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-dm_plex_tps_thickness", "Thickness of volumetric extrusion", NULL, thickness, &thickness, NULL);CHKERRQ(ierr);
        ierr = DMPlexDistributeGetDefault(dm, &tps_distribute);CHKERRQ(ierr);
        ierr = PetscOptionsBool("-dm_plex_tps_distribute", "Distribute the 2D mesh prior to refinement and extrusion", NULL, tps_distribute, &tps_distribute, NULL);CHKERRQ(ierr);
        ierr = DMPlexCreateTPSMesh_Internal(dm, tps_type, extent, periodic, tps_distribute, refine, layers, thickness);CHKERRQ(ierr);
      }
      break;
      default: SETERRQ(comm, PETSC_ERR_SUP, "Domain shape %s is unsupported", DMPlexShapes[shape]);
    }
  }
  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  if (!((PetscObject)dm)->name && nameflg) {
    ierr = PetscObjectSetName((PetscObject)dm, plexname);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetFromOptions_NonRefinement_Plex(PetscOptionItems *PetscOptionsObject, DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscBool      flg;
  char           bdLabel[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Handle viewing */
  ierr = PetscOptionsBool("-dm_plex_print_set_values", "Output all set values info", "DMPlexMatSetClosure", PETSC_FALSE, &mesh->printSetValues, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_print_fem", "Debug output level all fem computations", "DMPlexSNESComputeResidualFEM", 0, &mesh->printFEM, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_print_tol", "Tolerance for FEM output", "DMPlexSNESComputeResidualFEM", mesh->printTol, &mesh->printTol, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_print_l2", "Debug output level all L2 diff computations", "DMComputeL2Diff", 0, &mesh->printL2, NULL,0);CHKERRQ(ierr);
  ierr = DMMonitorSetFromOptions(dm, "-dm_plex_monitor_throughput", "Monitor the simulation throughput", "DMPlexMonitorThroughput", DMPlexMonitorThroughput, NULL, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscLogDefaultBegin();CHKERRQ(ierr);}
  /* Labeling */
  ierr = PetscOptionsString("-dm_plex_boundary_label", "Label to mark the mesh boundary", "", bdLabel, bdLabel, sizeof(bdLabel), &flg);CHKERRQ(ierr);
  if (flg) {ierr = DMPlexCreateBoundaryLabel_Private(dm, bdLabel);CHKERRQ(ierr);}
  /* Point Location */
  ierr = PetscOptionsBool("-dm_plex_hash_location", "Use grid hashing for point location", "DMInterpolate", PETSC_FALSE, &mesh->useHashLocation, NULL);CHKERRQ(ierr);
  /* Partitioning and distribution */
  ierr = PetscOptionsBool("-dm_plex_partition_balance", "Attempt to evenly divide points on partition boundary between processes", "DMPlexSetPartitionBalance", PETSC_FALSE, &mesh->partitionBalance, NULL);CHKERRQ(ierr);
  /* Generation and remeshing */
  ierr = PetscOptionsBool("-dm_plex_remesh_bd", "Allow changes to the boundary on remeshing", "DMAdapt", PETSC_FALSE, &mesh->remeshBd, NULL);CHKERRQ(ierr);
  /* Projection behavior */
  ierr = PetscOptionsBoundedInt("-dm_plex_max_projection_height", "Maxmimum mesh point height used to project locally", "DMPlexSetMaxProjectionHeight", 0, &mesh->maxProjectionHeight, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_regular_refinement", "Use special nested projection algorithm for regular refinement", "DMPlexSetRegularRefinement", mesh->regularRefinement, &mesh->regularRefinement, NULL);CHKERRQ(ierr);
  /* Checking structure */
  {
    PetscBool   flg = PETSC_FALSE, flg2 = PETSC_FALSE, all = PETSC_FALSE;

    ierr = PetscOptionsBool("-dm_plex_check_all", "Perform all checks", NULL, PETSC_FALSE, &all, &flg2);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-dm_plex_check_symmetry", "Check that the adjacency information in the mesh is symmetric", "DMPlexCheckSymmetry", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (all || (flg && flg2)) {ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-dm_plex_check_skeleton", "Check that each cell has the correct number of vertices (only for homogeneous simplex or tensor meshes)", "DMPlexCheckSkeleton", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (all || (flg && flg2)) {ierr = DMPlexCheckSkeleton(dm, 0);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-dm_plex_check_faces", "Check that the faces of each cell give a vertex order this is consistent with what we expect from the cell type", "DMPlexCheckFaces", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (all || (flg && flg2)) {ierr = DMPlexCheckFaces(dm, 0);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-dm_plex_check_geometry", "Check that cells have positive volume", "DMPlexCheckGeometry", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (all || (flg && flg2)) {ierr = DMPlexCheckGeometry(dm);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-dm_plex_check_pointsf", "Check some necessary conditions for PointSF", "DMPlexCheckPointSF", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (all || (flg && flg2)) {ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-dm_plex_check_interface_cones", "Check points on inter-partition interfaces have conforming order of cone points", "DMPlexCheckInterfaceCones", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (all || (flg && flg2)) {ierr = DMPlexCheckInterfaceCones(dm);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-dm_plex_check_cell_shape", "Check cell shape", "DMPlexCheckCellShape", PETSC_FALSE, &flg, &flg2);CHKERRQ(ierr);
    if (flg && flg2) {ierr = DMPlexCheckCellShape(dm, PETSC_TRUE, PETSC_DETERMINE);CHKERRQ(ierr);}
  }
  {
    PetscReal scale = 1.0;

    ierr = PetscOptionsReal("-dm_plex_scale", "Scale factor for mesh coordinates", "DMPlexScale", scale, &scale, &flg);CHKERRQ(ierr);
    if (flg) {
      Vec coordinates, coordinatesLocal;

      ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(dm, &coordinatesLocal);CHKERRQ(ierr);
      ierr = VecScale(coordinates, scale);CHKERRQ(ierr);
      ierr = VecScale(coordinatesLocal, scale);CHKERRQ(ierr);
    }
  }
  ierr = PetscPartitionerSetFromOptions(mesh->partitioner);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_Plex(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscFunctionList ordlist;
  char              oname[256];
  PetscReal         volume = -1.0;
  PetscInt          prerefine = 0, refine = 0, r, coarsen = 0, overlap = 0, extLayers = 0, dim;
  PetscBool         uniformOrig, created = PETSC_FALSE, uniform = PETSC_TRUE, distribute, interpolate = PETSC_TRUE, coordSpace = PETSC_TRUE, remap = PETSC_TRUE, ghostCells = PETSC_FALSE, isHierarchy, ignoreModel = PETSC_FALSE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Options");CHKERRQ(ierr);
  /* Handle automatic creation */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 0) {ierr = DMPlexCreateFromOptions_Internal(PetscOptionsObject, &coordSpace, dm);CHKERRQ(ierr);created = PETSC_TRUE;}
  /* Handle interpolation before distribution */
  ierr = PetscOptionsBool("-dm_plex_interpolate_pre", "Flag to interpolate mesh before distribution", "", interpolate, &interpolate, &flg);CHKERRQ(ierr);
  if (flg) {
    DMPlexInterpolatedFlag interpolated;

    ierr = DMPlexIsInterpolated(dm, &interpolated);CHKERRQ(ierr);
    if (interpolated == DMPLEX_INTERPOLATED_FULL && !interpolate) {
      DM udm;

      ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
      ierr = DMPlexReplace_Static(dm, &udm);CHKERRQ(ierr);
    } else if (interpolated != DMPLEX_INTERPOLATED_FULL && interpolate) {
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = DMPlexReplace_Static(dm, &idm);CHKERRQ(ierr);
    }
  }
  /* Handle DMPlex refinement before distribution */
  ierr = PetscOptionsBool("-dm_refine_ignore_model", "Flag to ignore the geometry model when refining", "DMCreate", ignoreModel, &ignoreModel, &flg);CHKERRQ(ierr);
  if (flg) {((DM_Plex *) dm->data)->ignoreModel = ignoreModel;}
  ierr = DMPlexGetRefinementUniform(dm, &uniformOrig);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_refine_pre", "The number of refinements before distribution", "DMCreate", prerefine, &prerefine, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_refine_remap_pre", "Flag to control coordinate remapping", "DMCreate", remap, &remap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_refine_uniform_pre", "Flag for uniform refinement before distribution", "DMCreate", uniform, &uniform, &flg);CHKERRQ(ierr);
  if (flg) {ierr = DMPlexSetRefinementUniform(dm, uniform);CHKERRQ(ierr);}
  ierr = PetscOptionsReal("-dm_refine_volume_limit_pre", "The maximum cell volume after refinement before distribution", "DMCreate", volume, &volume, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMPlexSetRefinementUniform(dm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(dm, volume);CHKERRQ(ierr);
    prerefine = PetscMax(prerefine, 1);
  }
  for (r = 0; r < prerefine; ++r) {
    DM             rdm;
    PetscPointFunc coordFunc = ((DM_Plex*) dm->data)->coordFunc;

    ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
    ierr = DMRefine(dm, PetscObjectComm((PetscObject) dm), &rdm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &rdm);CHKERRQ(ierr);
    ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
    if (coordFunc && remap) {
      ierr = DMPlexRemapGeometry(dm, 0.0, coordFunc);CHKERRQ(ierr);
      ((DM_Plex*) dm->data)->coordFunc = coordFunc;
    }
  }
  ierr = DMPlexSetRefinementUniform(dm, uniformOrig);CHKERRQ(ierr);
  /* Handle DMPlex extrusion before distribution */
  ierr = PetscOptionsBoundedInt("-dm_extrude", "The number of layers to extrude", "", extLayers, &extLayers, NULL, 0);CHKERRQ(ierr);
  if (extLayers) {
    DM edm;

    ierr = DMExtrude(dm, extLayers, &edm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &edm);CHKERRQ(ierr);
    ((DM_Plex *) dm->data)->coordFunc = NULL;
    ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
    extLayers = 0;
  }
  /* Handle DMPlex reordering before distribution */
  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_plex_reorder", "Set mesh reordering type", "DMPlexGetOrdering", ordlist, MATORDERINGNATURAL, oname, sizeof(oname), &flg);CHKERRQ(ierr);
  if (flg) {
    DM pdm;
    IS perm;

    ierr = DMPlexGetOrdering(dm, oname, NULL, &perm);CHKERRQ(ierr);
    ierr = DMPlexPermute(dm, perm, &pdm);CHKERRQ(ierr);
    ierr = ISDestroy(&perm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &pdm);CHKERRQ(ierr);
    ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
  }
  /* Handle DMPlex distribution */
  ierr = DMPlexDistributeGetDefault(dm, &distribute);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_distribute", "Flag to redistribute a mesh among processes", "DMCreate", distribute, &distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_distribute_overlap", "The size of the overlap halo", "DMCreate", overlap, &overlap, NULL, 0);CHKERRQ(ierr);
  if (distribute) {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, overlap, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMPlexReplace_Static(dm, &pdm);CHKERRQ(ierr);
    }
  }
  /* Create coordinate space */
  if (created) {
    DM_Plex  *mesh = (DM_Plex *) dm->data;
    PetscInt  degree = 1;
    PetscBool periodic, flg;

    ierr = PetscOptionsBool("-dm_coord_space", "Use an FEM space for coordinates", "", coordSpace, &coordSpace, &flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dm_coord_petscspace_degree", "FEM degree for coordinate space", "", degree, &degree, NULL);CHKERRQ(ierr);
    if (coordSpace) {ierr = DMPlexCreateCoordinateSpace(dm, degree, mesh->coordFunc);CHKERRQ(ierr);}
    if (flg && !coordSpace) {
      DM           cdm;
      PetscDS      cds;
      PetscObject  obj;
      PetscClassId id;

      ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetDS(cdm, &cds);CHKERRQ(ierr);
      ierr = PetscDSGetDiscretization(cds, 0, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscContainer dummy;

        ierr = PetscContainerCreate(PETSC_COMM_SELF, &dummy);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) dummy, "coordinates");CHKERRQ(ierr);
        ierr = DMSetField(cdm, 0, NULL, (PetscObject) dummy);CHKERRQ(ierr);
        ierr = PetscContainerDestroy(&dummy);CHKERRQ(ierr);
        ierr = DMClearDS(cdm);CHKERRQ(ierr);
      }
      mesh->coordFunc = NULL;
    }
    ierr = DMLocalizeCoordinates(dm);CHKERRQ(ierr);
    ierr = DMGetPeriodicity(dm, &periodic, NULL, NULL, NULL);CHKERRQ(ierr);
    if (periodic) {ierr = DMSetPeriodicity(dm, PETSC_TRUE, NULL, NULL, NULL);CHKERRQ(ierr);}
  }
  /* Handle DMPlex refinement */
  remap = PETSC_TRUE;
  ierr = PetscOptionsBoundedInt("-dm_refine", "The number of uniform refinements", "DMCreate", refine, &refine, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_refine_remap", "Flag to control coordinate remapping", "DMCreate", remap, &remap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_refine_hierarchy", "The number of uniform refinements", "DMCreate", refine, &refine, &isHierarchy,0);CHKERRQ(ierr);
  if (refine) {ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);}
  if (refine && isHierarchy) {
    DM *dms, coarseDM;

    ierr = DMGetCoarseDM(dm, &coarseDM);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)coarseDM);CHKERRQ(ierr);
    ierr = PetscMalloc1(refine,&dms);CHKERRQ(ierr);
    ierr = DMRefineHierarchy(dm, refine, dms);CHKERRQ(ierr);
    /* Total hack since we do not pass in a pointer */
    ierr = DMPlexSwap_Static(dm, dms[refine-1]);CHKERRQ(ierr);
    if (refine == 1) {
      ierr = DMSetCoarseDM(dm, dms[0]);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(dm, PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = DMSetCoarseDM(dm, dms[refine-2]);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(dm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dms[0], dms[refine-1]);CHKERRQ(ierr);
      ierr = DMPlexSetRegularRefinement(dms[0], PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = DMSetCoarseDM(dms[refine-1], coarseDM);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)coarseDM);CHKERRQ(ierr);
    /* Free DMs */
    for (r = 0; r < refine; ++r) {
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dms[r]);CHKERRQ(ierr);
      ierr = DMDestroy(&dms[r]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dms);CHKERRQ(ierr);
  } else {
    for (r = 0; r < refine; ++r) {
      DM             rdm;
      PetscPointFunc coordFunc = ((DM_Plex*) dm->data)->coordFunc;

      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      ierr = DMRefine(dm, PetscObjectComm((PetscObject) dm), &rdm);CHKERRQ(ierr);
      /* Total hack since we do not pass in a pointer */
      ierr = DMPlexReplace_Static(dm, &rdm);CHKERRQ(ierr);
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      if (coordFunc && remap) {
        ierr = DMPlexRemapGeometry(dm, 0.0, coordFunc);CHKERRQ(ierr);
        ((DM_Plex*) dm->data)->coordFunc = coordFunc;
      }
    }
  }
  /* Handle DMPlex coarsening */
  ierr = PetscOptionsBoundedInt("-dm_coarsen", "Coarsen the mesh", "DMCreate", coarsen, &coarsen, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_coarsen_hierarchy", "The number of coarsenings", "DMCreate", coarsen, &coarsen, &isHierarchy,0);CHKERRQ(ierr);
  if (coarsen && isHierarchy) {
    DM *dms;

    ierr = PetscMalloc1(coarsen, &dms);CHKERRQ(ierr);
    ierr = DMCoarsenHierarchy(dm, coarsen, dms);CHKERRQ(ierr);
    /* Free DMs */
    for (r = 0; r < coarsen; ++r) {
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dms[r]);CHKERRQ(ierr);
      ierr = DMDestroy(&dms[r]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dms);CHKERRQ(ierr);
  } else {
    for (r = 0; r < coarsen; ++r) {
      DM             cdm;
      PetscPointFunc coordFunc = ((DM_Plex*) dm->data)->coordFunc;

      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      ierr = DMCoarsen(dm, PetscObjectComm((PetscObject) dm), &cdm);CHKERRQ(ierr);
      /* Total hack since we do not pass in a pointer */
      ierr = DMPlexReplace_Static(dm, &cdm);CHKERRQ(ierr);
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      if (coordFunc) {
        ierr = DMPlexRemapGeometry(dm, 0.0, coordFunc);CHKERRQ(ierr);
        ((DM_Plex*) dm->data)->coordFunc = coordFunc;
      }
    }
  }
  /* Handle ghost cells */
  ierr = PetscOptionsBool("-dm_plex_create_fv_ghost_cells", "Flag to create finite volume ghost cells on the boundary", "DMCreate", ghostCells, &ghostCells, NULL);CHKERRQ(ierr);
  if (ghostCells) {
    DM   gdm;
    char lname[PETSC_MAX_PATH_LEN];

    lname[0] = '\0';
    ierr = PetscOptionsString("-dm_plex_fv_ghost_cells_label", "Label name for ghost cells boundary", "DMCreate", lname, lname, sizeof(lname), &flg);CHKERRQ(ierr);
    ierr = DMPlexConstructGhostCells(dm, flg ? lname : NULL, NULL, &gdm);CHKERRQ(ierr);
    ierr = DMPlexReplace_Static(dm, &gdm);CHKERRQ(ierr);
  }
  /* Handle */
  ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Plex(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector_Section_Private(dm,vec);CHKERRQ(ierr);
  /* ierr = VecSetOperation(*vec, VECOP_DUPLICATE, (void(*)(void)) VecDuplicate_MPI_DM);CHKERRQ(ierr); */
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Plex);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void)) VecView_Plex_Native);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_LOAD, (void (*)(void)) VecLoad_Plex);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void)) VecLoad_Plex_Native);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Plex(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector_Section_Private(dm,vec);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Plex_Local);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_LOAD, (void (*)(void)) VecLoad_Plex_Local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetDimPoints_Plex(DM dm, PetscInt dim, PetscInt *pStart, PetscInt *pEnd)
{
  PetscInt       depth, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth == 1) {
    ierr = DMGetDimension(dm, &d);CHKERRQ(ierr);
    if (dim == 0)      {ierr = DMPlexGetDepthStratum(dm, dim, pStart, pEnd);CHKERRQ(ierr);}
    else if (dim == d) {ierr = DMPlexGetDepthStratum(dm, 1, pStart, pEnd);CHKERRQ(ierr);}
    else               {*pStart = 0; *pEnd = 0;}
  } else {
    ierr = DMPlexGetDepthStratum(dm, dim, pStart, pEnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetNeighbors_Plex(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscSF           sf;
  PetscInt          niranks, njranks, n;
  const PetscMPIInt *iranks, *jranks;
  DM_Plex           *data = (DM_Plex*) dm->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  if (!data->neighbors) {
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = PetscSFGetRootRanks(sf, &njranks, &jranks, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscSFGetLeafRanks(sf, &niranks, &iranks, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(njranks + niranks + 1, &data->neighbors);CHKERRQ(ierr);
    ierr = PetscArraycpy(data->neighbors + 1, jranks, njranks);CHKERRQ(ierr);
    ierr = PetscArraycpy(data->neighbors + njranks + 1, iranks, niranks);CHKERRQ(ierr);
    n = njranks + niranks;
    ierr = PetscSortRemoveDupsMPIInt(&n, data->neighbors + 1);CHKERRQ(ierr);
    /* The following cast should never fail: can't have more neighbors than PETSC_MPI_INT_MAX */
    ierr = PetscMPIIntCast(n, data->neighbors);CHKERRQ(ierr);
  }
  if (nranks) *nranks = data->neighbors[0];
  if (ranks) {
    if (data->neighbors[0]) *ranks = data->neighbors + 1;
    else                    *ranks = NULL;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMInterpolateSolution_Plex(DM, DM, Mat, Vec, Vec);

static PetscErrorCode DMInitialize_Plex(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm->ops->view                            = DMView_Plex;
  dm->ops->load                            = DMLoad_Plex;
  dm->ops->setfromoptions                  = DMSetFromOptions_Plex;
  dm->ops->clone                           = DMClone_Plex;
  dm->ops->setup                           = DMSetUp_Plex;
  dm->ops->createlocalsection              = DMCreateLocalSection_Plex;
  dm->ops->createdefaultconstraints        = DMCreateDefaultConstraints_Plex;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Plex;
  dm->ops->createlocalvector               = DMCreateLocalVector_Plex;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = DMCreateCoordinateDM_Plex;
  dm->ops->createcoordinatefield           = DMCreateCoordinateField_Plex;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = DMCreateMatrix_Plex;
  dm->ops->createinterpolation             = DMCreateInterpolation_Plex;
  dm->ops->createmassmatrix                = DMCreateMassMatrix_Plex;
  dm->ops->createmassmatrixlumped          = DMCreateMassMatrixLumped_Plex;
  dm->ops->createinjection                 = DMCreateInjection_Plex;
  dm->ops->refine                          = DMRefine_Plex;
  dm->ops->coarsen                         = DMCoarsen_Plex;
  dm->ops->refinehierarchy                 = DMRefineHierarchy_Plex;
  dm->ops->coarsenhierarchy                = DMCoarsenHierarchy_Plex;
  dm->ops->extrude                         = DMExtrude_Plex;
  dm->ops->globaltolocalbegin              = NULL;
  dm->ops->globaltolocalend                = NULL;
  dm->ops->localtoglobalbegin              = NULL;
  dm->ops->localtoglobalend                = NULL;
  dm->ops->destroy                         = DMDestroy_Plex;
  dm->ops->createsubdm                     = DMCreateSubDM_Plex;
  dm->ops->createsuperdm                   = DMCreateSuperDM_Plex;
  dm->ops->getdimpoints                    = DMGetDimPoints_Plex;
  dm->ops->locatepoints                    = DMLocatePoints_Plex;
  dm->ops->projectfunctionlocal            = DMProjectFunctionLocal_Plex;
  dm->ops->projectfunctionlabellocal       = DMProjectFunctionLabelLocal_Plex;
  dm->ops->projectfieldlocal               = DMProjectFieldLocal_Plex;
  dm->ops->projectfieldlabellocal          = DMProjectFieldLabelLocal_Plex;
  dm->ops->projectbdfieldlabellocal        = DMProjectBdFieldLabelLocal_Plex;
  dm->ops->computel2diff                   = DMComputeL2Diff_Plex;
  dm->ops->computel2gradientdiff           = DMComputeL2GradientDiff_Plex;
  dm->ops->computel2fielddiff              = DMComputeL2FieldDiff_Plex;
  dm->ops->getneighbors                    = DMGetNeighbors_Plex;
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertBoundaryValues_C",DMPlexInsertBoundaryValues_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertTimeDerviativeBoundaryValues_C",DMPlexInsertTimeDerivativeBoundaryValues_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMCreateNeumannOverlap_C",DMCreateNeumannOverlap_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexGetOverlap_C",DMPlexGetOverlap_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexDistributeGetDefault_C",DMPlexDistributeGetDefault_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexDistributeSetDefault_C",DMPlexDistributeSetDefault_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMInterpolateSolution_C",DMInterpolateSolution_Plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMClone_Plex(DM dm, DM *newdm)
{
  DM_Plex        *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mesh->refct++;
  (*newdm)->data = mesh;
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMInitialize_Plex(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  DMPLEX = "plex" - A DM object that encapsulates an unstructured mesh, or CW Complex, which can be expressed using a Hasse Diagram.
                    In the local representation, Vecs contain all unknowns in the interior and shared boundary. This is
                    specified by a PetscSection object. Ownership in the global representation is determined by
                    ownership of the underlying DMPlex points. This is specified by another PetscSection object.

  Options Database Keys:
+ -dm_refine_pre                     - Refine mesh before distribution
+ -dm_refine_uniform_pre             - Choose uniform or generator-based refinement
+ -dm_refine_volume_limit_pre        - Cell volume limit after pre-refinement using generator
. -dm_distribute                     - Distribute mesh across processes
. -dm_distribute_overlap             - Number of cells to overlap for distribution
. -dm_refine                         - Refine mesh after distribution
. -dm_plex_hash_location             - Use grid hashing for point location
. -dm_plex_hash_box_faces <n,m,p>    - The number of divisions in each direction of the grid hash
. -dm_plex_partition_balance         - Attempt to evenly divide points on partition boundary between processes
. -dm_plex_remesh_bd                 - Allow changes to the boundary on remeshing
. -dm_plex_max_projection_height     - Maxmimum mesh point height used to project locally
. -dm_plex_regular_refinement        - Use special nested projection algorithm for regular refinement
. -dm_plex_check_all                 - Perform all shecks below
. -dm_plex_check_symmetry            - Check that the adjacency information in the mesh is symmetric
. -dm_plex_check_skeleton <celltype> - Check that each cell has the correct number of vertices
. -dm_plex_check_faces <celltype>    - Check that the faces of each cell give a vertex order this is consistent with what we expect from the cell type
. -dm_plex_check_geometry            - Check that cells have positive volume
. -dm_view :mesh.tex:ascii_latex     - View the mesh in LaTeX/TikZ
. -dm_plex_view_scale <num>          - Scale the TikZ
- -dm_plex_print_fem <num>           - View FEM assembly information, such as element vectors and matrices

  Level: intermediate

.seealso: DMType, DMPlexCreate(), DMCreate(), DMSetType()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Plex(DM dm)
{
  DM_Plex       *mesh;
  PetscInt       unit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&mesh);CHKERRQ(ierr);
  dm->data = mesh;

  mesh->refct             = 1;
  ierr                    = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &mesh->coneSection);CHKERRQ(ierr);
  mesh->maxConeSize       = 0;
  mesh->cones             = NULL;
  mesh->coneOrientations  = NULL;
  ierr                    = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &mesh->supportSection);CHKERRQ(ierr);
  mesh->maxSupportSize    = 0;
  mesh->supports          = NULL;
  mesh->refinementUniform = PETSC_TRUE;
  mesh->refinementLimit   = -1.0;
  mesh->distDefault       = PETSC_TRUE;
  mesh->interpolated      = DMPLEX_INTERPOLATED_INVALID;
  mesh->interpolatedCollective = DMPLEX_INTERPOLATED_INVALID;

  mesh->facesTmp = NULL;

  mesh->tetgenOpts   = NULL;
  mesh->triangleOpts = NULL;
  ierr = PetscPartitionerCreate(PetscObjectComm((PetscObject)dm), &mesh->partitioner);CHKERRQ(ierr);
  mesh->remeshBd     = PETSC_FALSE;

  mesh->subpointMap = NULL;

  for (unit = 0; unit < NUM_PETSC_UNITS; ++unit) mesh->scale[unit] = 1.0;

  mesh->regularRefinement   = PETSC_FALSE;
  mesh->depthState          = -1;
  mesh->celltypeState       = -1;
  mesh->globalVertexNumbers = NULL;
  mesh->globalCellNumbers   = NULL;
  mesh->anchorSection       = NULL;
  mesh->anchorIS            = NULL;
  mesh->createanchors       = NULL;
  mesh->computeanchormatrix = NULL;
  mesh->parentSection       = NULL;
  mesh->parents             = NULL;
  mesh->childIDs            = NULL;
  mesh->childSection        = NULL;
  mesh->children            = NULL;
  mesh->referenceTree       = NULL;
  mesh->getchildsymmetry    = NULL;
  mesh->vtkCellHeight       = 0;
  mesh->useAnchors          = PETSC_FALSE;

  mesh->maxProjectionHeight = 0;

  mesh->neighbors           = NULL;

  mesh->printSetValues = PETSC_FALSE;
  mesh->printFEM       = 0;
  mesh->printTol       = 1.0e-10;

  ierr = DMInitialize_Plex(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreate - Creates a DMPlex object, which encapsulates an unstructured mesh, or CW complex, which can be expressed using a Hasse Diagram.

  Collective

  Input Parameter:
. comm - The communicator for the DMPlex object

  Output Parameter:
. mesh  - The DMPlex object

  Level: beginner

@*/
PetscErrorCode DMPlexCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMPLEX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexBuildFromCellListParallel - Build distributed DMPLEX topology from a list of vertices for each cell (common mesh generator output)

  Input Parameters:
+ dm - The DM
. numCells - The number of cells owned by this process
. numVertices - The number of vertices to be owned by this process, or PETSC_DECIDE
. NVertices - The global number of vertices, or PETSC_DETERMINE
. numCorners - The number of vertices for each cell
- cells - An array of numCells*numCorners numbers, the global vertex numbers for each cell

  Output Parameters:
+ vertexSF - (Optional) SF describing complete vertex ownership
- verticesAdjSaved - (Optional) vertex adjacency array

  Notes:
  Two triangles sharing a face
$
$        2
$      / | \
$     /  |  \
$    /   |   \
$   0  0 | 1  3
$    \   |   /
$     \  |  /
$      \ | /
$        1
would have input
$  numCells = 2, numVertices = 4
$  cells = [0 1 2  1 3 2]
$
which would result in the DMPlex
$
$        4
$      / | \
$     /  |  \
$    /   |   \
$   2  0 | 1  5
$    \   |   /
$     \  |  /
$      \ | /
$        3

  Vertices are implicitly numbered consecutively 0,...,NVertices.
  Each rank owns a chunk of numVertices consecutive vertices.
  If numVertices is PETSC_DECIDE, PETSc will distribute them as evenly as possible using PetscLayout.
  If NVertices is PETSC_DETERMINE and numVertices is PETSC_DECIDE, NVertices is computed by PETSc as the maximum vertex index in cells + 1.
  If only NVertices is PETSC_DETERMINE, it is computed as the sum of numVertices over all ranks.

  The cell distribution is arbitrary non-overlapping, independent of the vertex distribution.

  Not currently supported in Fortran.

  Level: advanced

.seealso: DMPlexBuildFromCellList(), DMPlexCreateFromCellListParallelPetsc(), DMPlexBuildCoordinatesFromCellListParallel()
@*/
PetscErrorCode DMPlexBuildFromCellListParallel(DM dm, PetscInt numCells, PetscInt numVertices, PetscInt NVertices, PetscInt numCorners, const PetscInt cells[], PetscSF *vertexSF, PetscInt **verticesAdjSaved)
{
  PetscSF         sfPoint;
  PetscLayout     layout;
  PetscInt        numVerticesAdj, *verticesAdj, *cones, c, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(dm,NVertices,4);
  ierr = PetscLogEventBegin(DMPLEX_BuildFromCellList,dm,0,0,0);CHKERRQ(ierr);
  /* Get/check global number of vertices */
  {
    PetscInt NVerticesInCells, i;
    const PetscInt len = numCells * numCorners;

    /* NVerticesInCells = max(cells) + 1 */
    NVerticesInCells = PETSC_MIN_INT;
    for (i=0; i<len; i++) if (cells[i] > NVerticesInCells) NVerticesInCells = cells[i];
    ++NVerticesInCells;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &NVerticesInCells, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);

    if (numVertices == PETSC_DECIDE && NVertices == PETSC_DECIDE) NVertices = NVerticesInCells;
    else PetscCheckFalse(NVertices != PETSC_DECIDE && NVertices < NVerticesInCells,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Specified global number of vertices %D must be greater than or equal to the number of vertices in cells %D",NVertices,NVerticesInCells);
  }
  /* Count locally unique vertices */
  {
    PetscHSetI vhash;
    PetscInt off = 0;

    ierr = PetscHSetICreate(&vhash);CHKERRQ(ierr);
    for (c = 0; c < numCells; ++c) {
      for (p = 0; p < numCorners; ++p) {
        ierr = PetscHSetIAdd(vhash, cells[c*numCorners+p]);CHKERRQ(ierr);
      }
    }
    ierr = PetscHSetIGetSize(vhash, &numVerticesAdj);CHKERRQ(ierr);
    if (!verticesAdjSaved) { ierr = PetscMalloc1(numVerticesAdj, &verticesAdj);CHKERRQ(ierr); }
    else { verticesAdj = *verticesAdjSaved; }
    ierr = PetscHSetIGetElems(vhash, &off, verticesAdj);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&vhash);CHKERRQ(ierr);
    PetscCheckFalse(off != numVerticesAdj,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of local vertices %D should be %D", off, numVerticesAdj);
  }
  ierr = PetscSortInt(numVerticesAdj, verticesAdj);CHKERRQ(ierr);
  /* Create cones */
  ierr = DMPlexSetChart(dm, 0, numCells+numVerticesAdj);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {ierr = DMPlexSetConeSize(dm, c, numCorners);CHKERRQ(ierr);}
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm,&cones);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      const PetscInt gv = cells[c*numCorners+p];
      PetscInt       lv;

      /* Positions within verticesAdj form 0-based local vertex numbering;
         we need to shift it by numCells to get correct DAG points (cells go first) */
      ierr = PetscFindInt(gv, numVerticesAdj, verticesAdj, &lv);CHKERRQ(ierr);
      PetscCheckFalse(lv < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not find global vertex %D in local connectivity", gv);
      cones[c*numCorners+p] = lv+numCells;
    }
  }
  /* Build point sf */
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)dm), &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(layout, NVertices);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, numVertices);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscSFCreateByMatchingIndices(layout, numVerticesAdj, verticesAdj, NULL, numCells, numVerticesAdj, verticesAdj, NULL, numCells, vertexSF, &sfPoint);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  if (!verticesAdjSaved) { ierr = PetscFree(verticesAdj);CHKERRQ(ierr); }
  ierr = PetscObjectSetName((PetscObject) sfPoint, "point SF");CHKERRQ(ierr);
  if (dm->sf) {
    const char *prefix;

    ierr = PetscObjectGetOptionsPrefix((PetscObject)dm->sf, &prefix);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)sfPoint, prefix);CHKERRQ(ierr);
  }
  ierr = DMSetPointSF(dm, sfPoint);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfPoint);CHKERRQ(ierr);
  if (vertexSF) {ierr = PetscObjectSetName((PetscObject)(*vertexSF), "Vertex Ownership SF");CHKERRQ(ierr);}
  /* Fill in the rest of the topology structure */
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_BuildFromCellList,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexBuildCoordinatesFromCellListParallel - Build DM coordinates from a list of coordinates for each owned vertex (common mesh generator output)

  Input Parameters:
+ dm - The DM
. spaceDim - The spatial dimension used for coordinates
. sfVert - SF describing complete vertex ownership
- vertexCoords - An array of numVertices*spaceDim numbers, the coordinates of each vertex

  Level: advanced

  Notes:
  Not currently supported in Fortran.

.seealso: DMPlexBuildCoordinatesFromCellList(), DMPlexCreateFromCellListParallelPetsc(), DMPlexBuildFromCellListParallel()
@*/
PetscErrorCode DMPlexBuildCoordinatesFromCellListParallel(DM dm, PetscInt spaceDim, PetscSF sfVert, const PetscReal vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       numVertices, numVerticesAdj, coordSize, v, vStart, vEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  PetscCheckFalse(vStart < 0 || vEnd < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not set up properly. DMPlexBuildFromCellList() should be called first.");
  ierr = DMSetCoordinateDim(dm, spaceDim);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfVert, &numVertices, &numVerticesAdj, NULL, NULL);CHKERRQ(ierr);
  PetscCheckFalse(vEnd - vStart != numVerticesAdj,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Supplied sfVert has wrong number of leaves = %D != %D = vEnd - vStart",numVerticesAdj,vEnd - vStart);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, spaceDim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  {
    MPI_Datatype coordtype;

    /* Need a temp buffer for coords if we have complex/single */
    ierr = MPI_Type_contiguous(spaceDim, MPIU_SCALAR, &coordtype);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&coordtype);CHKERRMPI(ierr);
#if defined(PETSC_USE_COMPLEX)
    {
    PetscScalar *svertexCoords;
    PetscInt    i;
    ierr = PetscMalloc1(numVertices*spaceDim,&svertexCoords);CHKERRQ(ierr);
    for (i=0; i<numVertices*spaceDim; i++) svertexCoords[i] = vertexCoords[i];
    ierr = PetscSFBcastBegin(sfVert, coordtype, svertexCoords, coords,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfVert, coordtype, svertexCoords, coords,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscFree(svertexCoords);CHKERRQ(ierr);
    }
#else
    ierr = PetscSFBcastBegin(sfVert, coordtype, vertexCoords, coords,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfVert, coordtype, vertexCoords, coords,MPI_REPLACE);CHKERRQ(ierr);
#endif
    ierr = MPI_Type_free(&coordtype);CHKERRMPI(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateFromCellListParallelPetsc - Create distributed DMPLEX from a list of vertices for each cell (common mesh generator output)

  Input Parameters:
+ comm - The communicator
. dim - The topological dimension of the mesh
. numCells - The number of cells owned by this process
. numVertices - The number of vertices owned by this process, or PETSC_DECIDE
. NVertices - The global number of vertices, or PETSC_DECIDE
. numCorners - The number of vertices for each cell
. interpolate - Flag indicating that intermediate mesh entities (faces, edges) should be created automatically
. cells - An array of numCells*numCorners numbers, the global vertex numbers for each cell
. spaceDim - The spatial dimension used for coordinates
- vertexCoords - An array of numVertices*spaceDim numbers, the coordinates of each vertex

  Output Parameters:
+ dm - The DM
. vertexSF - (Optional) SF describing complete vertex ownership
- verticesAdjSaved - (Optional) vertex adjacency array

  Notes:
  This function is just a convenient sequence of DMCreate(), DMSetType(), DMSetDimension(),
  DMPlexBuildFromCellListParallel(), DMPlexInterpolate(), DMPlexBuildCoordinatesFromCellListParallel()

  See DMPlexBuildFromCellListParallel() for an example and details about the topology-related parameters.
  See DMPlexBuildCoordinatesFromCellListParallel() for details about the geometry-related parameters.

  Level: intermediate

.seealso: DMPlexCreateFromCellListPetsc(), DMPlexBuildFromCellListParallel(), DMPlexBuildCoordinatesFromCellListParallel(), DMPlexCreateFromDAG(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFromCellListParallelPetsc(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numVertices, PetscInt NVertices, PetscInt numCorners, PetscBool interpolate, const PetscInt cells[], PetscInt spaceDim, const PetscReal vertexCoords[], PetscSF *vertexSF, PetscInt **verticesAdj, DM *dm)
{
  PetscSF        sfVert;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(*dm, dim, 2);
  PetscValidLogicalCollectiveInt(*dm, spaceDim, 9);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexBuildFromCellListParallel(*dm, numCells, numVertices, NVertices, numCorners, cells, &sfVert, verticesAdj);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  ierr = DMPlexBuildCoordinatesFromCellListParallel(*dm, spaceDim, sfVert, vertexCoords);CHKERRQ(ierr);
  if (vertexSF) *vertexSF = sfVert;
  else {ierr = PetscSFDestroy(&sfVert);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  DMPlexBuildFromCellList - Build DMPLEX topology from a list of vertices for each cell (common mesh generator output)

  Input Parameters:
+ dm - The DM
. numCells - The number of cells owned by this process
. numVertices - The number of vertices owned by this process, or PETSC_DETERMINE
. numCorners - The number of vertices for each cell
- cells - An array of numCells*numCorners numbers, the global vertex numbers for each cell

  Level: advanced

  Notes:
  Two triangles sharing a face
$
$        2
$      / | \
$     /  |  \
$    /   |   \
$   0  0 | 1  3
$    \   |   /
$     \  |  /
$      \ | /
$        1
would have input
$  numCells = 2, numVertices = 4
$  cells = [0 1 2  1 3 2]
$
which would result in the DMPlex
$
$        4
$      / | \
$     /  |  \
$    /   |   \
$   2  0 | 1  5
$    \   |   /
$     \  |  /
$      \ | /
$        3

  If numVertices is PETSC_DETERMINE, it is computed by PETSc as the maximum vertex index in cells + 1.

  Not currently supported in Fortran.

.seealso: DMPlexBuildFromCellListParallel(), DMPlexBuildCoordinatesFromCellList(), DMPlexCreateFromCellListPetsc()
@*/
PetscErrorCode DMPlexBuildFromCellList(DM dm, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, const PetscInt cells[])
{
  PetscInt      *cones, c, p, dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_BuildFromCellList,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Get/check global number of vertices */
  {
    PetscInt NVerticesInCells, i;
    const PetscInt len = numCells * numCorners;

    /* NVerticesInCells = max(cells) + 1 */
    NVerticesInCells = PETSC_MIN_INT;
    for (i=0; i<len; i++) if (cells[i] > NVerticesInCells) NVerticesInCells = cells[i];
    ++NVerticesInCells;

    if (numVertices == PETSC_DECIDE) numVertices = NVerticesInCells;
    else PetscCheckFalse(numVertices < NVerticesInCells,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Specified number of vertices %D must be greater than or equal to the number of vertices in cells %D",numVertices,NVerticesInCells);
  }
  ierr = DMPlexSetChart(dm, 0, numCells+numVertices);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    ierr = DMPlexSetConeSize(dm, c, numCorners);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm,&cones);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      cones[c*numCorners+p] = cells[c*numCorners+p]+numCells;
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_BuildFromCellList,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexBuildCoordinatesFromCellList - Build DM coordinates from a list of coordinates for each owned vertex (common mesh generator output)

  Input Parameters:
+ dm - The DM
. spaceDim - The spatial dimension used for coordinates
- vertexCoords - An array of numVertices*spaceDim numbers, the coordinates of each vertex

  Level: advanced

  Notes:
  Not currently supported in Fortran.

.seealso: DMPlexBuildCoordinatesFromCellListParallel(), DMPlexCreateFromCellListPetsc(), DMPlexBuildFromCellList()
@*/
PetscErrorCode DMPlexBuildCoordinatesFromCellList(DM dm, PetscInt spaceDim, const PetscReal vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  DM             cdm;
  PetscScalar   *coords;
  PetscInt       v, vStart, vEnd, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  PetscCheckFalse(vStart < 0 || vEnd < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not set up properly. DMPlexBuildFromCellList() should be called first.");
  ierr = DMSetCoordinateDim(dm, spaceDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(cdm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, spaceDim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecGetArrayWrite(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < vEnd-vStart; ++v) {
    for (d = 0; d < spaceDim; ++d) {
      coords[v*spaceDim+d] = vertexCoords[v*spaceDim+d];
    }
  }
  ierr = VecRestoreArrayWrite(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateFromCellListPetsc - Create DMPLEX from a list of vertices for each cell (common mesh generator output), but only process 0 takes in the input

  Collective on comm

  Input Parameters:
+ comm - The communicator
. dim - The topological dimension of the mesh
. numCells - The number of cells, only on process 0
. numVertices - The number of vertices owned by this process, or PETSC_DECIDE, only on process 0
. numCorners - The number of vertices for each cell, only on process 0
. interpolate - Flag indicating that intermediate mesh entities (faces, edges) should be created automatically
. cells - An array of numCells*numCorners numbers, the vertices for each cell, only on process 0
. spaceDim - The spatial dimension used for coordinates
- vertexCoords - An array of numVertices*spaceDim numbers, the coordinates of each vertex, only on process 0

  Output Parameter:
. dm - The DM, which only has points on process 0

  Notes:
  This function is just a convenient sequence of DMCreate(), DMSetType(), DMSetDimension(), DMPlexBuildFromCellList(),
  DMPlexInterpolate(), DMPlexBuildCoordinatesFromCellList()

  See DMPlexBuildFromCellList() for an example and details about the topology-related parameters.
  See DMPlexBuildCoordinatesFromCellList() for details about the geometry-related parameters.
  See DMPlexCreateFromCellListParallelPetsc() for parallel input

  Level: intermediate

.seealso: DMPlexCreateFromCellListParallelPetsc(), DMPlexBuildFromCellList(), DMPlexBuildCoordinatesFromCellList(), DMPlexCreateFromDAG(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFromCellListPetsc(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, PetscBool interpolate, const PetscInt cells[], PetscInt spaceDim, const PetscReal vertexCoords[], DM *dm)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!dim,comm, PETSC_ERR_ARG_OUTOFRANGE, "This is not appropriate for 0-dimensional meshes. Consider either creating the DM using DMPlexCreateFromDAG(), by hand, or using DMSwarm.");
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  if (!rank) {ierr = DMPlexBuildFromCellList(*dm, numCells, numVertices, numCorners, cells);CHKERRQ(ierr);}
  else       {ierr = DMPlexBuildFromCellList(*dm, 0, 0, 0, NULL);CHKERRQ(ierr);}
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  if (!rank) {ierr = DMPlexBuildCoordinatesFromCellList(*dm, spaceDim, vertexCoords);CHKERRQ(ierr);}
  else       {ierr = DMPlexBuildCoordinatesFromCellList(*dm, spaceDim, NULL);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateFromDAG - This takes as input the adjacency-list representation of the Directed Acyclic Graph (Hasse Diagram) encoding a mesh, and produces a DM

  Input Parameters:
+ dm - The empty DM object, usually from DMCreate() and DMSetDimension()
. depth - The depth of the DAG
. numPoints - Array of size depth + 1 containing the number of points at each depth
. coneSize - The cone size of each point
. cones - The concatenation of the cone points for each point, the cone list must be oriented correctly for each point
. coneOrientations - The orientation of each cone point
- vertexCoords - An array of numPoints[0]*spacedim numbers representing the coordinates of each vertex, with spacedim the value set via DMSetCoordinateDim()

  Output Parameter:
. dm - The DM

  Note: Two triangles sharing a face would have input
$  depth = 1, numPoints = [4 2], coneSize = [3 3 0 0 0 0]
$  cones = [2 3 4  3 5 4], coneOrientations = [0 0 0  0 0 0]
$ vertexCoords = [-1.0 0.0  0.0 -1.0  0.0 1.0  1.0 0.0]
$
which would result in the DMPlex
$
$        4
$      / | \
$     /  |  \
$    /   |   \
$   2  0 | 1  5
$    \   |   /
$     \  |  /
$      \ | /
$        3
$
$ Notice that all points are numbered consecutively, unlike DMPlexCreateFromCellListPetsc()

  Level: advanced

.seealso: DMPlexCreateFromCellListPetsc(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFromDAG(DM dm, PetscInt depth, const PetscInt numPoints[], const PetscInt coneSize[], const PetscInt cones[], const PetscInt coneOrientations[], const PetscScalar vertexCoords[])
{
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscInt       coordSize, firstVertex = -1, pStart = 0, pEnd = 0, p, v, dim, dimEmbed, d, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  PetscCheckFalse(dimEmbed < dim,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Embedding dimension %D cannot be less than intrinsic dimension %d",dimEmbed,dim);
  for (d = 0; d <= depth; ++d) pEnd += numPoints[d];
  ierr = DMPlexSetChart(dm, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, p, coneSize[p-pStart]);CHKERRQ(ierr);
    if (firstVertex < 0 && !coneSize[p - pStart]) {
      firstVertex = p - pStart;
    }
  }
  PetscCheckFalse(firstVertex < 0 && numPoints[0],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected %D vertices but could not find any", numPoints[0]);
  ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
  for (p = pStart, off = 0; p < pEnd; off += coneSize[p-pStart], ++p) {
    ierr = DMPlexSetCone(dm, p, &cones[off]);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, p, &coneOrientations[off]);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, dimEmbed);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numPoints[0]);CHKERRQ(ierr);
  for (v = firstVertex; v < firstVertex+numPoints[0]; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dimEmbed);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, dimEmbed);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, dimEmbed);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  if (vertexCoords) {
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = 0; v < numPoints[0]; ++v) {
      PetscInt off;

      ierr = PetscSectionGetOffset(coordSection, v+firstVertex, &off);CHKERRQ(ierr);
      for (d = 0; d < dimEmbed; ++d) {
        coords[off+d] = vertexCoords[v*dimEmbed+d];
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateCellVertexFromFile - Create a DMPlex mesh from a simple cell-vertex file.

+ comm        - The MPI communicator
. filename    - Name of the .dat file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: The format is the simplest possible:
$ Ne
$ v0 v1 ... vk
$ Nv
$ x y z marker

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateMedFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateCellVertexFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  DMLabel         marker;
  PetscViewer     viewer;
  Vec             coordinates;
  PetscSection    coordSection;
  PetscScalar    *coords;
  char            line[PETSC_MAX_PATH_LEN];
  PetscInt        dim = 3, cdim = 3, coordSize, v, c, d;
  PetscMPIInt     rank;
  int             snum, Nv, Nc, Ncn, Nl;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = PetscViewerRead(viewer, line, 4, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d %d %d %d", &Nc, &Nv, &Ncn, &Nl);
    PetscCheckFalse(snum != 4,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
  } else {
    Nc = Nv = Ncn = Nl = 0;
  }
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, Nc+Nv);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*dm, cdim);CHKERRQ(ierr);
  /* Read topology */
  if (rank == 0) {
    char     format[PETSC_MAX_PATH_LEN];
    PetscInt cone[8];
    int      vbuf[8], v;

    for (c = 0; c < Ncn; ++c) {format[c*3+0] = '%'; format[c*3+1] = 'd'; format[c*3+2] = ' ';}
    format[Ncn*3-1] = '\0';
    for (c = 0; c < Nc; ++c) {ierr = DMPlexSetConeSize(*dm, c, Ncn);CHKERRQ(ierr);}
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for (c = 0; c < Nc; ++c) {
      ierr = PetscViewerRead(viewer, line, Ncn, NULL, PETSC_STRING);CHKERRQ(ierr);
      switch (Ncn) {
        case 2: snum = sscanf(line, format, &vbuf[0], &vbuf[1]);break;
        case 3: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2]);break;
        case 4: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2], &vbuf[3]);break;
        case 6: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2], &vbuf[3], &vbuf[4], &vbuf[5]);break;
        case 8: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2], &vbuf[3], &vbuf[4], &vbuf[5], &vbuf[6], &vbuf[7]);break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No cell shape with %D vertices", Ncn);
      }
      PetscCheckFalse(snum != Ncn,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
      for (v = 0; v < Ncn; ++v) cone[v] = vbuf[v] + Nc;
      /* Hexahedra are inverted */
      if (Ncn == 8) {
        PetscInt tmp = cone[1];
        cone[1] = cone[3];
        cone[3] = tmp;
      }
      ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  /* Read coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, cdim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, Nc, Nc + Nv);CHKERRQ(ierr);
  for (v = Nc; v < Nc+Nv; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, cdim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, cdim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, cdim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (rank == 0) {
    char   format[PETSC_MAX_PATH_LEN];
    double x[3];
    int    l, val[3];

    if (Nl) {
      for (l = 0; l < Nl; ++l) {format[l*3+0] = '%'; format[l*3+1] = 'd'; format[l*3+2] = ' ';}
      format[Nl*3-1] = '\0';
      ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, "marker", &marker);CHKERRQ(ierr);
    }
    for (v = 0; v < Nv; ++v) {
      ierr = PetscViewerRead(viewer, line, 3+Nl, NULL, PETSC_STRING);CHKERRQ(ierr);
      snum = sscanf(line, "%lg %lg %lg", &x[0], &x[1], &x[2]);
      PetscCheckFalse(snum != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
      switch (Nl) {
        case 0: snum = 0;break;
        case 1: snum = sscanf(line, format, &val[0]);break;
        case 2: snum = sscanf(line, format, &val[0], &val[1]);break;
        case 3: snum = sscanf(line, format, &val[0], &val[1], &val[2]);break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Request support for %D labels", Nl);
      }
      PetscCheckFalse(snum != Nl,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
      for (d = 0; d < cdim; ++d) coords[v*cdim+d] = x[d];
      for (l = 0; l < Nl; ++l) {ierr = DMLabelSetValue(marker, v+Nc, val[l]);CHKERRQ(ierr);}
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  if (interpolate) {
    DM      idm;
    DMLabel bdlabel;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;

    if (!Nl) {
      ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, "marker", &bdlabel);CHKERRQ(ierr);
      ierr = DMPlexMarkBoundaryFaces(*dm, PETSC_DETERMINE, bdlabel);CHKERRQ(ierr);
      ierr = DMPlexLabelComplete(*dm, bdlabel);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateFromFile - This takes a filename and produces a DM

  Input Parameters:
+ comm - The communicator
. filename - A file name
. plexname - The object name of the resulting DM, also used for intra-datafile lookup by some formats
- interpolate - Flag to create intermediate mesh pieces (edges, faces)

  Output Parameter:
. dm - The DM

  Options Database Keys:
. -dm_plex_create_from_hdf5_xdmf - use the PETSC_VIEWER_HDF5_XDMF format for reading HDF5

  Use -dm_plex_create_ prefix to pass options to the internal PetscViewer, e.g.
$ -dm_plex_create_viewer_hdf5_collective

  Notes:
  Using PETSCVIEWERHDF5 type with PETSC_VIEWER_HDF5_PETSC format, one can save multiple DMPlex
  meshes in a single HDF5 file. This in turn requires one to name the DMPlex object with PetscObjectSetName()
  before saving it with DMView() and before loading it with DMLoad() for identification of the mesh object.
  The input parameter name is thus used to name the DMPlex object when DMPlexCreateFromFile() internally
  calls DMLoad(). Currently, name is ignored for other viewer types and/or formats.

  Level: beginner

.seealso: DMPlexCreateFromDAG(), DMPlexCreateFromCellListPetsc(), DMPlexCreate(), PetscObjectSetName(), DMView(), DMLoad()
@*/
PetscErrorCode DMPlexCreateFromFile(MPI_Comm comm, const char filename[], const char plexname[], PetscBool interpolate, DM *dm)
{
  const char    *extGmsh      = ".msh";
  const char    *extGmsh2     = ".msh2";
  const char    *extGmsh4     = ".msh4";
  const char    *extCGNS      = ".cgns";
  const char    *extExodus    = ".exo";
  const char    *extExodus_e  = ".e";
  const char    *extGenesis   = ".gen";
  const char    *extFluent    = ".cas";
  const char    *extHDF5      = ".h5";
  const char    *extMed       = ".med";
  const char    *extPLY       = ".ply";
  const char    *extEGADSLite = ".egadslite";
  const char    *extEGADS     = ".egads";
  const char    *extIGES      = ".igs";
  const char    *extSTEP      = ".stp";
  const char    *extCV        = ".dat";
  size_t         len;
  PetscBool      isGmsh, isGmsh2, isGmsh4, isCGNS, isExodus, isGenesis, isFluent, isHDF5, isMed, isPLY, isEGADSLite, isEGADS, isIGES, isSTEP, isCV;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  if (plexname) PetscValidCharPointer(plexname, 3);
  PetscValidPointer(dm, 5);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_CreateFromFile,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  PetscCheckFalse(!len,comm, PETSC_ERR_ARG_WRONG, "Filename must be a valid path");
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extGmsh,      4, &isGmsh);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-5)],  extGmsh2,     5, &isGmsh2);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-5)],  extGmsh4,     5, &isGmsh4);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-5)],  extCGNS,      5, &isCGNS);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extExodus,    4, &isExodus);CHKERRQ(ierr);
  if (!isExodus) {
    ierr = PetscStrncmp(&filename[PetscMax(0,len-2)],  extExodus_e,    2, &isExodus);CHKERRQ(ierr);
  }
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extGenesis,   4, &isGenesis);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extFluent,    4, &isFluent);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-3)],  extHDF5,      3, &isHDF5);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extMed,       4, &isMed);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extPLY,       4, &isPLY);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-10)], extEGADSLite, 10, &isEGADSLite);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-6)],  extEGADS,     6, &isEGADS);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extIGES,      4, &isIGES);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extSTEP,      4, &isSTEP);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)],  extCV,        4, &isCV);CHKERRQ(ierr);
  if (isGmsh || isGmsh2 || isGmsh4) {
    ierr = DMPlexCreateGmshFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isCGNS) {
    ierr = DMPlexCreateCGNSFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isExodus || isGenesis) {
    ierr = DMPlexCreateExodusFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isFluent) {
    ierr = DMPlexCreateFluentFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isHDF5) {
    PetscBool      load_hdf5_xdmf = PETSC_FALSE;
    PetscViewer viewer;

    /* PETSC_VIEWER_HDF5_XDMF is used if the filename ends with .xdmf.h5, or if -dm_plex_create_from_hdf5_xdmf option is present */
    ierr = PetscStrncmp(&filename[PetscMax(0,len-8)], ".xdmf",  5, &load_hdf5_xdmf);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_create_from_hdf5_xdmf", &load_hdf5_xdmf, NULL);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
    ierr = PetscViewerSetOptionsPrefix(viewer, "dm_plex_create_");CHKERRQ(ierr);
    ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);

    ierr = DMCreate(comm, dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)(*dm), plexname);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
    if (load_hdf5_xdmf) {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF);CHKERRQ(ierr);}
    ierr = DMLoad(*dm, viewer);CHKERRQ(ierr);
    if (load_hdf5_xdmf) {ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);}
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    if (interpolate) {
      DM idm;

      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
    }
  } else if (isMed) {
    ierr = DMPlexCreateMedFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isPLY) {
    ierr = DMPlexCreatePLYFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isEGADSLite || isEGADS || isIGES || isSTEP) {
    if (isEGADSLite) {ierr = DMPlexCreateEGADSLiteFromFile(comm, filename, dm);CHKERRQ(ierr);}
    else             {ierr = DMPlexCreateEGADSFromFile(comm, filename, dm);CHKERRQ(ierr);}
    if (!interpolate) {
      DM udm;

      ierr = DMPlexUninterpolate(*dm, &udm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = udm;
    }
  } else if (isCV) {
    ierr = DMPlexCreateCellVertexFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);
  ierr = PetscStrlen(plexname, &len);CHKERRQ(ierr);
  if (len) {ierr = PetscObjectSetName((PetscObject)(*dm), plexname);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(DMPLEX_CreateFromFile,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
