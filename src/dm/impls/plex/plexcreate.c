#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petsc/private/hashseti.h>          /*I   "petscdmplex.h"   I*/
#include <petscsf.h>
#include <petscdmplextransform.h>
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

  PetscFunctionBegin;
  if (copyPeriodicity) {
    PetscCall(DMGetPeriodicity(dmin, &isper, &maxCell, &L, &bd));
    PetscCall(DMSetPeriodicity(dmout, isper,  maxCell,  L,  bd));
  }
  PetscCall(DMPlexDistributeGetDefault(dmin, &dist));
  PetscCall(DMPlexDistributeSetDefault(dmout, dist));
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

  PetscFunctionBegin;
  if (dm == dmNew) {
    PetscCall(DMDestroy(ndm));
    PetscFunctionReturn(0);
  }
  dm->setupcalled = dmNew->setupcalled;
  PetscCall(DMGetDimension(dmNew, &dim));
  PetscCall(DMSetDimension(dm, dim));
  PetscCall(DMGetCoordinateDim(dmNew, &cdim));
  PetscCall(DMSetCoordinateDim(dm, cdim));
  PetscCall(DMGetPointSF(dmNew, &sf));
  PetscCall(DMSetPointSF(dm, sf));
  PetscCall(DMGetCoordinateDM(dmNew, &coordDM));
  PetscCall(DMGetCoordinatesLocal(dmNew, &coords));
  PetscCall(DMSetCoordinateDM(dm, coordDM));
  PetscCall(DMSetCoordinatesLocal(dm, coords));
  /* Do not want to create the coordinate field if it does not already exist, so do not call DMGetCoordinateField() */
  PetscCall(DMFieldDestroy(&dm->coordinateField));
  dm->coordinateField = dmNew->coordinateField;
  ((DM_Plex *) dmNew->data)->coordFunc = ((DM_Plex *) dm->data)->coordFunc;
  PetscCall(DMGetPeriodicity(dmNew, &isper, &maxCell, &L, &bd));
  PetscCall(DMSetPeriodicity(dm, isper, maxCell, L, bd));
  PetscCall(DMDestroy_Plex(dm));
  PetscCall(DMInitialize_Plex(dm));
  dm->data = dmNew->data;
  ((DM_Plex *) dmNew->data)->refct++;
  PetscCall(DMDestroyLabelLinkList_Internal(dm));
  PetscCall(DMCopyLabels(dmNew, dm, PETSC_OWN_POINTER, PETSC_TRUE, DM_COPY_LABELS_FAIL));
  PetscCall(DMGetCoarseDM(dmNew,&coarseDM));
  PetscCall(DMSetCoarseDM(dm,coarseDM));
  PetscCall(DMDestroy(ndm));
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

  PetscFunctionBegin;
  if (dmA == dmB) PetscFunctionReturn(0);
  PetscCall(DMGetPointSF(dmA, &sfA));
  PetscCall(DMGetPointSF(dmB, &sfB));
  PetscCall(PetscObjectReference((PetscObject) sfA));
  PetscCall(DMSetPointSF(dmA, sfB));
  PetscCall(DMSetPointSF(dmB, sfA));
  PetscCall(PetscObjectDereference((PetscObject) sfA));

  PetscCall(DMGetCoordinateDM(dmA, &coordDMA));
  PetscCall(DMGetCoordinateDM(dmB, &coordDMB));
  PetscCall(PetscObjectReference((PetscObject) coordDMA));
  PetscCall(DMSetCoordinateDM(dmA, coordDMB));
  PetscCall(DMSetCoordinateDM(dmB, coordDMA));
  PetscCall(PetscObjectDereference((PetscObject) coordDMA));

  PetscCall(DMGetCoordinatesLocal(dmA, &coordsA));
  PetscCall(DMGetCoordinatesLocal(dmB, &coordsB));
  PetscCall(PetscObjectReference((PetscObject) coordsA));
  PetscCall(DMSetCoordinatesLocal(dmA, coordsB));
  PetscCall(DMSetCoordinatesLocal(dmB, coordsA));
  PetscCall(PetscObjectDereference((PetscObject) coordsA));

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

  PetscFunctionBegin;
  PetscCall(DMPlexInterpolate(dm, &idm));
  PetscCall(DMPlexCopyCoordinates(dm, idm));
  PetscCall(DMPlexReplace_Static(dm, &idm));
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
  DM_Plex      *mesh = (DM_Plex *) dm->data;
  DM            cdm;
  PetscDS       cds;
  PetscFE       fe;
  PetscClassId  id;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDS(cdm, &cds));
  PetscCall(PetscDSGetDiscretization(cds, 0, (PetscObject *) &fe));
  PetscCall(PetscObjectGetClassId((PetscObject) fe, &id));
  if (id != PETSCFE_CLASSID) {
    PetscBool      simplex;
    PetscInt       dim, dE, qorder;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetCoordinateDim(dm, &dE));
    qorder = degree;
    PetscObjectOptionsBegin((PetscObject) cdm);
    PetscCall(PetscOptionsBoundedInt("-coord_dm_default_quadrature_order", "Quadrature order is one less than quadrature points per edge", "DMPlexCreateCoordinateSpace", qorder, &qorder, NULL, 0));
    PetscOptionsEnd();
    if (degree == PETSC_DECIDE) fe = NULL;
    else {
      PetscCall(DMPlexIsSimplex(dm, &simplex));
      PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, degree, qorder, &fe));
    }
    PetscCall(DMProjectCoordinates(dm, fe));
    PetscCall(PetscFEDestroy(&fe));
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

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetDimension(dm, dim));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  switch (dim) {
  case 2:
    if (simplex) PetscCall(PetscObjectSetName((PetscObject) dm, "triangular"));
    else         PetscCall(PetscObjectSetName((PetscObject) dm, "quadrilateral"));
    break;
  case 3:
    if (simplex) PetscCall(PetscObjectSetName((PetscObject) dm, "tetrahedral"));
    else         PetscCall(PetscObjectSetName((PetscObject) dm, "hexahedral"));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %" PetscInt_FMT, dim);
  }
  if (rank) {
    PetscInt numPoints[2] = {0, 0};
    PetscCall(DMPlexCreateFromDAG(dm, 1, numPoints, NULL, NULL, NULL, NULL));
  } else {
    switch (dim) {
    case 2:
      if (simplex) {
        PetscInt    numPoints[2]        = {4, 2};
        PetscInt    coneSize[6]         = {3, 3, 0, 0, 0, 0};
        PetscInt    cones[6]            = {2, 3, 4,  5, 4, 3};
        PetscInt    coneOrientations[6] = {0, 0, 0,  0, 0, 0};
        PetscScalar vertexCoords[8]     = {-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5};

        PetscCall(DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      } else {
        PetscInt    numPoints[2]        = {6, 2};
        PetscInt    coneSize[8]         = {4, 4, 0, 0, 0, 0, 0, 0};
        PetscInt    cones[8]            = {2, 3, 4, 5,  3, 6, 7, 4};
        PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
        PetscScalar vertexCoords[12]    = {-1.0, -0.5,  0.0, -0.5,  0.0, 0.5,  -1.0, 0.5,  1.0, -0.5,  1.0, 0.5};

        PetscCall(DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      }
      break;
    case 3:
      if (simplex) {
        PetscInt    numPoints[2]        = {5, 2};
        PetscInt    coneSize[7]         = {4, 4, 0, 0, 0, 0, 0};
        PetscInt    cones[8]            = {4, 3, 5, 2,  5, 3, 4, 6};
        PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
        PetscScalar vertexCoords[15]    = {-1.0, 0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 0.0};

        PetscCall(DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      } else {
        PetscInt    numPoints[2]         = {12, 2};
        PetscInt    coneSize[14]         = {8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        PetscInt    cones[16]            = {2, 3, 4, 5, 6, 7, 8, 9,  5, 4, 10, 11, 7, 12, 13, 8};
        PetscInt    coneOrientations[16] = {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0,  0,  0, 0};
        PetscScalar vertexCoords[36]     = {-1.0, -0.5, -0.5,  -1.0,  0.5, -0.5,  0.0,  0.5, -0.5,   0.0, -0.5, -0.5,
                                            -1.0, -0.5,  0.5,   0.0, -0.5,  0.5,  0.0,  0.5,  0.5,  -1.0,  0.5,  0.5,
                                             1.0,  0.5, -0.5,   1.0, -0.5, -0.5,  1.0, -0.5,  0.5,   1.0,  0.5,  0.5};

        PetscCall(DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      }
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %" PetscInt_FMT, dim);
    }
  }
  *newdm = dm;
  if (refinementLimit > 0.0) {
    DM rdm;
    const char *name;

    PetscCall(DMPlexSetRefinementUniform(*newdm, PETSC_FALSE));
    PetscCall(DMPlexSetRefinementLimit(*newdm, refinementLimit));
    PetscCall(DMRefine(*newdm, comm, &rdm));
    PetscCall(PetscObjectGetName((PetscObject) *newdm, &name));
    PetscCall(PetscObjectSetName((PetscObject)    rdm,  name));
    PetscCall(DMDestroy(newdm));
    *newdm = rdm;
  }
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*newdm, &idm));
    PetscCall(DMDestroy(newdm));
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

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, NULL));
  if (markerSeparate) {
    markerRight  = 2;
    markerLeft   = 1;
  }
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  if (!rank) {
    PetscCall(DMPlexSetChart(dm, 0, numVertices));
    PetscCall(DMSetUp(dm)); /* Allocate space for cones */
    PetscCall(DMSetLabelValue(dm, "marker", 0, markerLeft));
    PetscCall(DMSetLabelValue(dm, "marker", 1, markerRight));
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  /* Build coordinates */
  PetscCall(DMSetCoordinateDim(dm, cdim));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetChart(coordSection, 0, numVertices));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, cdim));
  for (v = 0; v < numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, cdim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, cdim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, cdim));
  PetscCall(VecSetType(coordinates,VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  coords[0] = lower[0];
  coords[1] = upper[0];
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
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

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, NULL));
  if (markerSeparate) {
    markerTop    = 3;
    markerBottom = 1;
    markerRight  = 2;
    markerLeft   = 4;
  }
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  if (rank == 0) {
    PetscInt e, ex, ey;

    PetscCall(DMPlexSetChart(dm, 0, numEdges+numVertices));
    for (e = 0; e < numEdges; ++e) {
      PetscCall(DMPlexSetConeSize(dm, e, 2));
    }
    PetscCall(DMSetUp(dm)); /* Allocate space for cones */
    for (vx = 0; vx <= edges[0]; vx++) {
      for (ey = 0; ey < edges[1]; ey++) {
        PetscInt edge   = vx*edges[1] + ey + edges[0]*(edges[1]+1);
        PetscInt vertex = ey*(edges[0]+1) + vx + numEdges;
        PetscInt cone[2];

        cone[0] = vertex; cone[1] = vertex+edges[0]+1;
        PetscCall(DMPlexSetCone(dm, edge, cone));
        if (vx == edges[0]) {
          PetscCall(DMSetLabelValue(dm, "marker", edge,    markerRight));
          PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerRight));
          if (ey == edges[1]-1) {
            PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerRight));
            PetscCall(DMSetLabelValue(dm, "Face Sets", cone[1], markerRight));
          }
        } else if (vx == 0) {
          PetscCall(DMSetLabelValue(dm, "marker", edge,    markerLeft));
          PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerLeft));
          if (ey == edges[1]-1) {
            PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerLeft));
            PetscCall(DMSetLabelValue(dm, "Face Sets", cone[1], markerLeft));
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
        PetscCall(DMPlexSetCone(dm, edge, cone));
        if (vy == edges[1]) {
          PetscCall(DMSetLabelValue(dm, "marker", edge,    markerTop));
          PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerTop));
          if (ex == edges[0]-1) {
            PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerTop));
            PetscCall(DMSetLabelValue(dm, "Face Sets", cone[1], markerTop));
          }
        } else if (vy == 0) {
          PetscCall(DMSetLabelValue(dm, "marker", edge,    markerBottom));
          PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerBottom));
          if (ex == edges[0]-1) {
            PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerBottom));
            PetscCall(DMSetLabelValue(dm, "Face Sets", cone[1], markerBottom));
          }
        }
      }
    }
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  /* Build coordinates */
  PetscCall(DMSetCoordinateDim(dm, 2));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetChart(coordSection, numEdges, numEdges + numVertices));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, 2));
  for (v = numEdges; v < numEdges+numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, 2));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, 2));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, 2));
  PetscCall(VecSetType(coordinates,VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  for (vy = 0; vy <= edges[1]; ++vy) {
    for (vx = 0; vx <= edges[0]; ++vx) {
      coords[(vy*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/edges[0])*vx;
      coords[(vy*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/edges[1])*vy;
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
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

  PetscFunctionBegin;
  PetscCheck(faces[0] >= 1 && faces[1] >= 1 && faces[2] >= 1,PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Must have at least 1 face per side");
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  vertices[0] = faces[0]+1; vertices[1] = faces[1]+1; vertices[2] = faces[2]+1;
  numVertices = vertices[0]*vertices[1]*vertices[2];
  if (rank == 0) {
    PetscInt f;

    PetscCall(DMPlexSetChart(dm, 0, numFaces+numVertices));
    for (f = 0; f < numFaces; ++f) {
      PetscCall(DMPlexSetConeSize(dm, f, 4));
    }
    PetscCall(DMSetUp(dm)); /* Allocate space for cones */

    /* Side 0 (Top) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vertices[0]*vertices[1]*(vertices[2]-1) + vy*vertices[0] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]+1; cone[3] = voffset+vertices[0];
        PetscCall(DMPlexSetCone(dm, iface, cone));
        PetscCall(DMSetLabelValue(dm, "marker", iface, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+1, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]+1, 1));
        iface++;
      }
    }

    /* Side 1 (Bottom) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vy*(faces[0]+1) + vx;
        cone[0] = voffset+1; cone[1] = voffset; cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]+1;
        PetscCall(DMPlexSetCone(dm, iface, cone));
        PetscCall(DMSetLabelValue(dm, "marker", iface, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+1, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]+1, 1));
        iface++;
      }
    }

    /* Side 2 (Front) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]*vertices[1]+1; cone[3] = voffset+vertices[0]*vertices[1];
        PetscCall(DMPlexSetCone(dm, iface, cone));
        PetscCall(DMSetLabelValue(dm, "marker", iface, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+1, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+1, 1));
        iface++;
      }
    }

    /* Side 3 (Back) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vertices[0]*(vertices[1]-1) + vx;
        cone[0] = voffset+vertices[0]*vertices[1]; cone[1] = voffset+vertices[0]*vertices[1]+1;
        cone[2] = voffset+1; cone[3] = voffset;
        PetscCall(DMPlexSetCone(dm, iface, cone));
        PetscCall(DMSetLabelValue(dm, "marker", iface, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+1, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+1, 1));
        iface++;
      }
    }

    /* Side 4 (Left) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vy = 0; vy < faces[1]; vy++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vy*vertices[0];
        cone[0] = voffset; cone[1] = voffset+vertices[0]*vertices[1];
        cone[2] = voffset+vertices[0]*vertices[1]+vertices[0]; cone[3] = voffset+vertices[0];
        PetscCall(DMPlexSetCone(dm, iface, cone));
        PetscCall(DMSetLabelValue(dm, "marker", iface, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[1]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+vertices[0], 1));
        iface++;
      }
    }

    /* Side 5 (Right) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vy = 0; vy < faces[1]; vy++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vy*vertices[0] + faces[0];
        cone[0] = voffset+vertices[0]*vertices[1]; cone[1] = voffset;
        cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]*vertices[1]+vertices[0];
        PetscCall(DMPlexSetCone(dm, iface, cone));
        PetscCall(DMSetLabelValue(dm, "marker", iface, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+0, 1));
        PetscCall(DMSetLabelValue(dm, "marker", voffset+vertices[0]*vertices[1]+vertices[0], 1));
        iface++;
      }
    }
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  /* Build coordinates */
  PetscCall(DMSetCoordinateDim(dm, 3));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetChart(coordSection, numFaces, numFaces + numVertices));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, 3));
  for (v = numFaces; v < numFaces+numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, 3));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, 3));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, 3));
  PetscCall(VecSetType(coordinates,VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  for (vz = 0; vz <= faces[2]; ++vz) {
    for (vy = 0; vy <= faces[1]; ++vy) {
      for (vx = 0; vx <= faces[0]; ++vx) {
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+0] = lower[0] + ((upper[0] - lower[0])/faces[0])*vx;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+1] = lower[1] + ((upper[1] - lower[1])/faces[1])*vy;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+2] = lower[2] + ((upper[2] - lower[2])/faces[2])*vz;
      }
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxSurfaceMesh_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], PetscBool interpolate)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  PetscCall(DMSetDimension(dm, dim-1));
  PetscCall(DMSetCoordinateDim(dm, dim));
  switch (dim) {
    case 1: PetscCall(DMPlexCreateBoxSurfaceMesh_Tensor_1D_Internal(dm, lower, upper, faces));break;
    case 2: PetscCall(DMPlexCreateBoxSurfaceMesh_Tensor_2D_Internal(dm, lower, upper, faces));break;
    case 3: PetscCall(DMPlexCreateBoxSurfaceMesh_Tensor_3D_Internal(dm, lower, upper, faces));break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Dimension not supported: %" PetscInt_FMT, dim);
  }
  if (interpolate) PetscCall(DMPlexInterpolateInPlace_Internal(dm));
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

  PetscFunctionBegin;
  PetscCall(DMCreate(comm,dm));
  PetscCall(DMSetType(*dm,DMPLEX));
  PetscCall(DMPlexCreateBoxSurfaceMesh_Internal(*dm, dim, faces ? faces : fac, lower ? lower : low, upper ? upper : upp, interpolate));
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

  PetscFunctionBegin;
  PetscValidPointer(dm,1);

  PetscCall(DMSetDimension(dm,1));
  PetscCall(DMCreateLabel(dm,"marker"));
  PetscCall(DMCreateLabel(dm,"Face Sets"));

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm),&rank));
  if (rank == 0) numCells = segments;
  if (rank == 0) numVerts = segments + (wrap ? 0 : 1);

  numPoints[0] = numVerts ; numPoints[1] = numCells;
  PetscCall(PetscMalloc4(numCells+numVerts,&coneSize,numCells*2,&cones,numCells+numVerts,&coneOrientations,numVerts,&vertexCoords));
  PetscCall(PetscArrayzero(coneOrientations,numCells+numVerts));
  for (i = 0; i < numCells; ++i) { coneSize[i] = 2; }
  for (i = 0; i < numVerts; ++i) { coneSize[numCells+i] = 0; }
  for (i = 0; i < numCells; ++i) { cones[2*i] = numCells + i%numVerts; cones[2*i+1] = numCells + (i+1)%numVerts; }
  for (i = 0; i < numVerts; ++i) { vertexCoords[i] = lower + (upper-lower)*((PetscReal)i/(PetscReal)numCells); }
  PetscCall(DMPlexCreateFromDAG(dm,1,numPoints,coneSize,cones,coneOrientations,vertexCoords));
  PetscCall(PetscFree4(coneSize,cones,coneOrientations,vertexCoords));

  PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options,((PetscObject)dm)->prefix,"-dm_plex_separate_marker",&markerSeparate,NULL));
  if (markerSeparate) { markerLeft = faceMarkerLeft; markerRight = faceMarkerRight;}
  if (!wrap && rank == 0) {
    PetscCall(DMPlexGetHeightStratum(dm,1,&fStart,&fEnd));
    PetscCall(DMSetLabelValue(dm,"marker",fStart,markerLeft));
    PetscCall(DMSetLabelValue(dm,"marker",fEnd-1,markerRight));
    PetscCall(DMSetLabelValue(dm,"Face Sets",fStart,faceMarkerLeft));
    PetscCall(DMSetLabelValue(dm,"Face Sets",fEnd-1,faceMarkerRight));
  }
  if (wrap) {
    L       = upper - lower;
    maxCell = (PetscReal)1.1*(L/(PetscReal)PetscMax(1,segments));
    PetscCall(DMSetPeriodicity(dm,PETSC_TRUE,&maxCell,&L,&bd));
  }
  PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxMesh_Simplex_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate)
{
  DM             boundary, vol;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(dm, 1);
  for (i = 0; i < dim; ++i) PetscCheck(periodicity[i] == DM_BOUNDARY_NONE,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Periodicity is not supported for simplex meshes");
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), &boundary));
  PetscCall(DMSetType(boundary, DMPLEX));
  PetscCall(DMPlexCreateBoxSurfaceMesh_Internal(boundary, dim, faces, lower, upper, PETSC_FALSE));
  PetscCall(DMPlexGenerate(boundary, NULL, interpolate, &vol));
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, vol));
  PetscCall(DMPlexReplace_Static(dm, &vol));
  PetscCall(DMDestroy(&boundary));
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

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm,&dim));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMCreateLabel(dm,"marker"));
  PetscCall(DMCreateLabel(dm,"Face Sets"));
  PetscCall(PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_periodic_cut", &cutMarker, NULL));
  if (bdX == DM_BOUNDARY_PERIODIC || bdX == DM_BOUNDARY_TWIST ||
      bdY == DM_BOUNDARY_PERIODIC || bdY == DM_BOUNDARY_TWIST ||
      bdZ == DM_BOUNDARY_PERIODIC || bdZ == DM_BOUNDARY_TWIST) {

    if (cutMarker) {PetscCall(DMCreateLabel(dm, "periodic_cut")); PetscCall(DMGetLabel(dm, "periodic_cut", &cutLabel));}
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Dimension %" PetscInt_FMT " not supported",dim);
  }
  PetscCall(PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, NULL));
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

    PetscCall(DMPlexSetChart(dm, 0, numCells+numFaces+numEdges+numVertices));
    for (c = 0; c < numCells; c++) {
      PetscCall(DMPlexSetConeSize(dm, c, 6));
    }
    for (f = firstXFace; f < firstXFace+numFaces; ++f) {
      PetscCall(DMPlexSetConeSize(dm, f, 4));
    }
    for (e = firstXEdge; e < firstXEdge+numEdges; ++e) {
      PetscCall(DMPlexSetConeSize(dm, e, 2));
    }
    PetscCall(DMSetUp(dm)); /* Allocate space for cones */
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
          PetscCall(DMPlexSetCone(dm, cell, cone));
          PetscCall(DMPlexSetConeOrientation(dm, cell, ornt));
          if (bdX != DM_BOUNDARY_NONE && fx == numXEdges-1 && cutLabel) PetscCall(DMLabelSetValue(cutLabel, cell, 2));
          if (bdY != DM_BOUNDARY_NONE && fy == numYEdges-1 && cutLabel) PetscCall(DMLabelSetValue(cutLabel, cell, 2));
          if (bdZ != DM_BOUNDARY_NONE && fz == numZEdges-1 && cutLabel) PetscCall(DMLabelSetValue(cutLabel, cell, 2));
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
                PetscCall(DMSetLabelValue(dm, "Face Sets", face, faceMarkerRight));
                PetscCall(DMSetLabelValue(dm, "marker", face, markerRight));
              }
              else if (fx == 0) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", face, faceMarkerLeft));
                PetscCall(DMSetLabelValue(dm, "marker", face, markerLeft));
              }
            }
          }
          cone[0] = edgeB; cone[1] = edgeR; cone[2] = edgeT; cone[3] = edgeL;
          PetscCall(DMPlexSetCone(dm, face, cone));
          PetscCall(DMPlexSetConeOrientation(dm, face, ornt));
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
                PetscCall(DMSetLabelValue(dm, "Face Sets", face, faceMarkerBack));
                PetscCall(DMSetLabelValue(dm, "marker", face, markerBack));
              }
              else if (fy == 0) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", face, faceMarkerFront));
                PetscCall(DMSetLabelValue(dm, "marker", face, markerFront));
              }
            }
          }
          cone[0] = edgeB; cone[1] = edgeR; cone[2] = edgeT; cone[3] = edgeL;
          PetscCall(DMPlexSetCone(dm, face, cone));
          PetscCall(DMPlexSetConeOrientation(dm, face, ornt));
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
            if (bdX != DM_BOUNDARY_NONE && fx == numXEdges-1 && cutLabel) PetscCall(DMLabelSetValue(cutLabel, face, 2));
            if (bdY != DM_BOUNDARY_NONE && fy == numYEdges-1 && cutLabel) PetscCall(DMLabelSetValue(cutLabel, face, 2));
          } else {
            /* markers */
            if (bdZ != DM_BOUNDARY_PERIODIC) {
              if (fz == numZVertices-1) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", face, faceMarkerTop));
                PetscCall(DMSetLabelValue(dm, "marker", face, markerTop));
              }
              else if (fz == 0) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", face, faceMarkerBottom));
                PetscCall(DMSetLabelValue(dm, "marker", face, markerBottom));
              }
            }
          }
          cone[0] = edgeB; cone[1] = edgeR; cone[2] = edgeT; cone[3] = edgeL;
          PetscCall(DMPlexSetCone(dm, face, cone));
          PetscCall(DMPlexSetConeOrientation(dm, face, ornt));
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
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerRight));
              }
              else if (vx == 0) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerLeft));
              }
            }
            if (bdY != DM_BOUNDARY_PERIODIC) {
              if (vy == numYVertices-1) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerBack));
              }
              else if (vy == 0) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerFront));
              }
            }
          }
          cone[0] = vertexB; cone[1] = vertexT;
          PetscCall(DMPlexSetCone(dm, edge, cone));
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
          PetscCall(DMPlexSetCone(dm, edge, cone));
          if (dim == 2) {
            if ((bdX != DM_BOUNDARY_PERIODIC) && (bdX != DM_BOUNDARY_TWIST)) {
              if (vx == numXVertices-1) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", edge, faceMarkerRight));
                PetscCall(DMSetLabelValue(dm, "marker", edge,    markerRight));
                PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerRight));
                if (ey == numYEdges-1) {
                  PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerRight));
                }
              } else if (vx == 0) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", edge, faceMarkerLeft));
                PetscCall(DMSetLabelValue(dm, "marker", edge,    markerLeft));
                PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerLeft));
                if (ey == numYEdges-1) {
                  PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerLeft));
                }
              }
            } else {
              if (vx == 0 && cutLabel) {
                PetscCall(DMLabelSetValue(cutLabel, edge,    1));
                PetscCall(DMLabelSetValue(cutLabel, cone[0], 1));
                if (ey == numYEdges-1) {
                  PetscCall(DMLabelSetValue(cutLabel, cone[1], 1));
                }
              }
            }
          } else {
            if (bdX != DM_BOUNDARY_PERIODIC) {
              if (vx == numXVertices-1) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerRight));
              } else if (vx == 0) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerLeft));
              }
            }
            if (bdZ != DM_BOUNDARY_PERIODIC) {
              if (vz == numZVertices-1) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerTop));
              } else if (vz == 0) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerBottom));
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
          PetscCall(DMPlexSetCone(dm, edge, cone));
          if (dim == 2) {
            if ((bdY != DM_BOUNDARY_PERIODIC) && (bdY != DM_BOUNDARY_TWIST)) {
              if (vy == numYVertices-1) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", edge, faceMarkerTop));
                PetscCall(DMSetLabelValue(dm, "marker", edge,    markerTop));
                PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerTop));
                if (ex == numXEdges-1) {
                  PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerTop));
                }
              } else if (vy == 0) {
                PetscCall(DMSetLabelValue(dm, "Face Sets", edge, faceMarkerBottom));
                PetscCall(DMSetLabelValue(dm, "marker", edge,    markerBottom));
                PetscCall(DMSetLabelValue(dm, "marker", cone[0], markerBottom));
                if (ex == numXEdges-1) {
                  PetscCall(DMSetLabelValue(dm, "marker", cone[1], markerBottom));
                }
              }
            } else {
              if (vy == 0 && cutLabel) {
                PetscCall(DMLabelSetValue(cutLabel, edge,    1));
                PetscCall(DMLabelSetValue(cutLabel, cone[0], 1));
                if (ex == numXEdges-1) {
                  PetscCall(DMLabelSetValue(cutLabel, cone[1], 1));
                }
              }
            }
          } else {
            if (bdY != DM_BOUNDARY_PERIODIC) {
              if (vy == numYVertices-1) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerBack));
              }
              else if (vy == 0) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerFront));
              }
            }
            if (bdZ != DM_BOUNDARY_PERIODIC) {
              if (vz == numZVertices-1) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerTop));
              }
              else if (vz == 0) {
                PetscCall(DMSetLabelValue(dm, "marker", edge, markerBottom));
              }
            }
          }
        }
      }
    }
    PetscCall(DMPlexSymmetrize(dm));
    PetscCall(DMPlexStratify(dm));
    /* Build coordinates */
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(PetscSectionSetNumFields(coordSection, 1));
    PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dim));
    PetscCall(PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVertices));
    for (v = firstVertex; v < firstVertex+numVertices; ++v) {
      PetscCall(PetscSectionSetDof(coordSection, v, dim));
      PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dim));
    }
    PetscCall(PetscSectionSetUp(coordSection));
    PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
    PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
    PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
    PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(coordinates, dim));
    PetscCall(VecSetType(coordinates,VECSTANDARD));
    PetscCall(VecGetArray(coordinates, &coords));
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
    PetscCall(VecRestoreArray(coordinates, &coords));
    PetscCall(DMSetCoordinatesLocal(dm, coordinates));
    PetscCall(VecDestroy(&coordinates));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxMesh_Tensor_Internal(DM dm, PetscInt dim, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[])
{
  DMBoundaryType bdt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscInt       fac[3] = {0, 0, 0}, d;

  PetscFunctionBegin;
  PetscValidPointer(dm, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  PetscCall(DMSetDimension(dm, dim));
  for (d = 0; d < dim; ++d) {fac[d] = faces[d]; bdt[d] = periodicity[d];}
  PetscCall(DMPlexCreateCubeMesh_Internal(dm, lower, upper, fac, bdt[0], bdt[1], bdt[2]));
  if (periodicity[0] == DM_BOUNDARY_PERIODIC || periodicity[0] == DM_BOUNDARY_TWIST ||
      periodicity[1] == DM_BOUNDARY_PERIODIC || periodicity[1] == DM_BOUNDARY_TWIST ||
      (dim > 2 && (periodicity[2] == DM_BOUNDARY_PERIODIC || periodicity[2] == DM_BOUNDARY_TWIST))) {
    PetscReal L[3];
    PetscReal maxCell[3];

    for (d = 0; d < dim; ++d) {
      L[d]       = upper[d] - lower[d];
      maxCell[d] = 1.1 * (L[d] / PetscMax(1, faces[d]));
    }
    PetscCall(DMSetPeriodicity(dm, PETSC_TRUE, maxCell, L, periodicity));
  }
  PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoxMesh_Internal(DM dm, PetscInt dim, PetscBool simplex, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate)
{
  PetscFunctionBegin;
  if (dim == 1)      PetscCall(DMPlexCreateLineMesh_Internal(dm, faces[0], lower[0], upper[0], periodicity[0]));
  else if (simplex)  PetscCall(DMPlexCreateBoxMesh_Simplex_Internal(dm, dim, faces, lower, upper, periodicity, interpolate));
  else               PetscCall(DMPlexCreateBoxMesh_Tensor_Internal(dm, dim, faces, lower, upper, periodicity));
  if (!interpolate && dim > 1 && !simplex) {
    DM udm;

    PetscCall(DMPlexUninterpolate(dm, &udm));
    PetscCall(DMPlexCopyCoordinates(dm, udm));
    PetscCall(DMPlexReplace_Static(dm, &udm));
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

  PetscFunctionBegin;
  PetscCall(DMCreate(comm,dm));
  PetscCall(DMSetType(*dm,DMPLEX));
  PetscCall(DMPlexCreateBoxMesh_Internal(*dm, dim, simplex, faces ? faces : fac, lower ? lower : low, upper ? upper : upp, periodicity ? periodicity : bdt, interpolate));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateWedgeBoxMesh_Internal(DM dm, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[])
{
  DM             bdm, vol;
  PetscInt       i;

  PetscFunctionBegin;
  for (i = 0; i < 3; ++i) PetscCheck(periodicity[i] == DM_BOUNDARY_NONE,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Periodicity not yet supported");
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), &bdm));
  PetscCall(DMSetType(bdm, DMPLEX));
  PetscCall(DMSetDimension(bdm, 2));
  PetscCall(DMPlexCreateBoxMesh_Simplex_Internal(bdm, 2, faces, lower, upper, periodicity, PETSC_TRUE));
  PetscCall(DMPlexExtrude(bdm, faces[2], upper[2] - lower[2], PETSC_TRUE, PETSC_FALSE, NULL, NULL, &vol));
  PetscCall(DMDestroy(&bdm));
  PetscCall(DMPlexReplace_Static(dm, &vol));
  if (lower[2] != 0.0) {
    Vec          v;
    PetscScalar *x;
    PetscInt     cDim, n;

    PetscCall(DMGetCoordinatesLocal(dm, &v));
    PetscCall(VecGetBlockSize(v, &cDim));
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(VecGetArray(v, &x));
    x   += cDim;
    for (i = 0; i < n; i += cDim) x[i] += lower[2];
    PetscCall(VecRestoreArray(v,&x));
    PetscCall(DMSetCoordinatesLocal(dm, v));
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

  PetscFunctionBegin;
  PetscCall(DMCreate(comm,dm));
  PetscCall(DMSetType(*dm,DMPLEX));
  PetscCall(DMPlexCreateWedgeBoxMesh_Internal(*dm, faces ? faces : fac, lower ? lower : low, upper ? upper : upp, periodicity ? periodicity : bdt));
  if (!interpolate) {
    DM udm;

    PetscCall(DMPlexUninterpolate(*dm, &udm));
    PetscCall(DMPlexReplace_Static(*dm, &udm));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) dm, prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) mesh->partitioner, prefix));
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

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  PetscCall(DMSetDimension(dm, dim));
  /* Create topology */
  {
    PetscInt cone[8], c;

    numCells    = rank == 0 ?  5 : 0;
    numVertices = rank == 0 ? 16 : 0;
    if (periodicZ == DM_BOUNDARY_PERIODIC) {
      numCells   *= 3;
      numVertices = rank == 0 ? 24 : 0;
    }
    PetscCall(DMPlexSetChart(dm, 0, numCells+numVertices));
    for (c = 0; c < numCells; c++) PetscCall(DMPlexSetConeSize(dm, c, 8));
    PetscCall(DMSetUp(dm));
    if (rank == 0) {
      if (periodicZ == DM_BOUNDARY_PERIODIC) {
        cone[0] = 15; cone[1] = 18; cone[2] = 17; cone[3] = 16;
        cone[4] = 31; cone[5] = 32; cone[6] = 33; cone[7] = 34;
        PetscCall(DMPlexSetCone(dm, 0, cone));
        cone[0] = 16; cone[1] = 17; cone[2] = 24; cone[3] = 23;
        cone[4] = 32; cone[5] = 36; cone[6] = 37; cone[7] = 33; /* 22 25 26 21 */
        PetscCall(DMPlexSetCone(dm, 1, cone));
        cone[0] = 18; cone[1] = 27; cone[2] = 24; cone[3] = 17;
        cone[4] = 34; cone[5] = 33; cone[6] = 37; cone[7] = 38;
        PetscCall(DMPlexSetCone(dm, 2, cone));
        cone[0] = 29; cone[1] = 27; cone[2] = 18; cone[3] = 15;
        cone[4] = 35; cone[5] = 31; cone[6] = 34; cone[7] = 38;
        PetscCall(DMPlexSetCone(dm, 3, cone));
        cone[0] = 29; cone[1] = 15; cone[2] = 16; cone[3] = 23;
        cone[4] = 35; cone[5] = 36; cone[6] = 32; cone[7] = 31;
        PetscCall(DMPlexSetCone(dm, 4, cone));

        cone[0] = 31; cone[1] = 34; cone[2] = 33; cone[3] = 32;
        cone[4] = 19; cone[5] = 22; cone[6] = 21; cone[7] = 20;
        PetscCall(DMPlexSetCone(dm, 5, cone));
        cone[0] = 32; cone[1] = 33; cone[2] = 37; cone[3] = 36;
        cone[4] = 22; cone[5] = 25; cone[6] = 26; cone[7] = 21;
        PetscCall(DMPlexSetCone(dm, 6, cone));
        cone[0] = 34; cone[1] = 38; cone[2] = 37; cone[3] = 33;
        cone[4] = 20; cone[5] = 21; cone[6] = 26; cone[7] = 28;
        PetscCall(DMPlexSetCone(dm, 7, cone));
        cone[0] = 35; cone[1] = 38; cone[2] = 34; cone[3] = 31;
        cone[4] = 30; cone[5] = 19; cone[6] = 20; cone[7] = 28;
        PetscCall(DMPlexSetCone(dm, 8, cone));
        cone[0] = 35; cone[1] = 31; cone[2] = 32; cone[3] = 36;
        cone[4] = 30; cone[5] = 25; cone[6] = 22; cone[7] = 19;
        PetscCall(DMPlexSetCone(dm, 9, cone));

        cone[0] = 19; cone[1] = 20; cone[2] = 21; cone[3] = 22;
        cone[4] = 15; cone[5] = 16; cone[6] = 17; cone[7] = 18;
        PetscCall(DMPlexSetCone(dm, 10, cone));
        cone[0] = 22; cone[1] = 21; cone[2] = 26; cone[3] = 25;
        cone[4] = 16; cone[5] = 23; cone[6] = 24; cone[7] = 17;
        PetscCall(DMPlexSetCone(dm, 11, cone));
        cone[0] = 20; cone[1] = 28; cone[2] = 26; cone[3] = 21;
        cone[4] = 18; cone[5] = 17; cone[6] = 24; cone[7] = 27;
        PetscCall(DMPlexSetCone(dm, 12, cone));
        cone[0] = 30; cone[1] = 28; cone[2] = 20; cone[3] = 19;
        cone[4] = 29; cone[5] = 15; cone[6] = 18; cone[7] = 27;
        PetscCall(DMPlexSetCone(dm, 13, cone));
        cone[0] = 30; cone[1] = 19; cone[2] = 22; cone[3] = 25;
        cone[4] = 29; cone[5] = 23; cone[6] = 16; cone[7] = 15;
        PetscCall(DMPlexSetCone(dm, 14, cone));
      } else {
        cone[0] =  5; cone[1] =  8; cone[2] =  7; cone[3] =  6;
        cone[4] =  9; cone[5] = 12; cone[6] = 11; cone[7] = 10;
        PetscCall(DMPlexSetCone(dm, 0, cone));
        cone[0] =  6; cone[1] =  7; cone[2] = 14; cone[3] = 13;
        cone[4] = 12; cone[5] = 15; cone[6] = 16; cone[7] = 11;
        PetscCall(DMPlexSetCone(dm, 1, cone));
        cone[0] =  8; cone[1] = 17; cone[2] = 14; cone[3] =  7;
        cone[4] = 10; cone[5] = 11; cone[6] = 16; cone[7] = 18;
        PetscCall(DMPlexSetCone(dm, 2, cone));
        cone[0] = 19; cone[1] = 17; cone[2] =  8; cone[3] =  5;
        cone[4] = 20; cone[5] =  9; cone[6] = 10; cone[7] = 18;
        PetscCall(DMPlexSetCone(dm, 3, cone));
        cone[0] = 19; cone[1] =  5; cone[2] =  6; cone[3] = 13;
        cone[4] = 20; cone[5] = 15; cone[6] = 12; cone[7] =  9;
        PetscCall(DMPlexSetCone(dm, 4, cone));
      }
    }
    PetscCall(DMPlexSymmetrize(dm));
    PetscCall(DMPlexStratify(dm));
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
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(PetscSectionSetNumFields(coordSection, 1));
    PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dim));
    PetscCall(PetscSectionSetChart(coordSection, numCells, numCells+numVertices));
    for (v = numCells; v < numCells+numVertices; ++v) {
      PetscCall(PetscSectionSetDof(coordSection, v, dim));
      PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dim));
    }
    PetscCall(PetscSectionSetUp(coordSection));
    PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
    PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
    PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
    PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(coordinates, dim));
    PetscCall(VecSetType(coordinates,VECSTANDARD));
    PetscCall(VecGetArray(coordinates, &coords));
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
    PetscCall(VecRestoreArray(coordinates, &coords));
    PetscCall(DMSetCoordinatesLocal(dm, coordinates));
    PetscCall(VecDestroy(&coordinates));
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
    PetscCall(DMSetPeriodicity(dm, PETSC_TRUE, maxCell, L, bdType));
  }
  {
    DM          cdm;
    PetscDS     cds;
    PetscScalar c[2] = {1.0, 1.0};

    PetscCall(DMPlexCreateCoordinateSpace(dm, 1, snapToCylinder));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetDS(cdm, &cds));
    PetscCall(PetscDSSetConstants(cds, 2, c));
  }
  /* Wait for coordinate creation before doing in-place modification */
  PetscCall(DMPlexInterpolateInPlace_Internal(dm));
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
  PetscFunctionBegin;
  PetscValidPointer(dm, 3);
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexCreateHexCylinderMesh_Internal(*dm, periodicZ));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateWedgeCylinderMesh_Internal(DM dm, PetscInt n, PetscBool interpolate)
{
  const PetscInt dim = 3;
  PetscInt       numCells, numVertices, v;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCheck(n >= 0,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of wedges %" PetscInt_FMT " cannot be negative", n);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  PetscCall(DMSetDimension(dm, dim));
  /* Must create the celltype label here so that we do not automatically try to compute the types */
  PetscCall(DMCreateLabel(dm, "celltype"));
  /* Create topology */
  {
    PetscInt cone[6], c;

    numCells    = rank == 0 ?        n : 0;
    numVertices = rank == 0 ?  2*(n+1) : 0;
    PetscCall(DMPlexSetChart(dm, 0, numCells+numVertices));
    for (c = 0; c < numCells; c++) PetscCall(DMPlexSetConeSize(dm, c, 6));
    PetscCall(DMSetUp(dm));
    for (c = 0; c < numCells; c++) {
      cone[0] =  c+n*1; cone[1] = (c+1)%n+n*1; cone[2] = 0+3*n;
      cone[3] =  c+n*2; cone[4] = (c+1)%n+n*2; cone[5] = 1+3*n;
      PetscCall(DMPlexSetCone(dm, c, cone));
      PetscCall(DMPlexSetCellType(dm, c, DM_POLYTOPE_TRI_PRISM_TENSOR));
    }
    PetscCall(DMPlexSymmetrize(dm));
    PetscCall(DMPlexStratify(dm));
  }
  for (v = numCells; v < numCells+numVertices; ++v) {
    PetscCall(DMPlexSetCellType(dm, v, DM_POLYTOPE_POINT));
  }
  /* Create cylinder geometry */
  {
    Vec          coordinates;
    PetscSection coordSection;
    PetscScalar *coords;
    PetscInt     coordSize, c;

    /* Build coordinates */
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(PetscSectionSetNumFields(coordSection, 1));
    PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dim));
    PetscCall(PetscSectionSetChart(coordSection, numCells, numCells+numVertices));
    for (v = numCells; v < numCells+numVertices; ++v) {
      PetscCall(PetscSectionSetDof(coordSection, v, dim));
      PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dim));
    }
    PetscCall(PetscSectionSetUp(coordSection));
    PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
    PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
    PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
    PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(coordinates, dim));
    PetscCall(VecSetType(coordinates,VECSTANDARD));
    PetscCall(VecGetArray(coordinates, &coords));
    for (c = 0; c < numCells; c++) {
      coords[(c+0*n)*dim+0] = PetscCosReal(2.0*c*PETSC_PI/n); coords[(c+0*n)*dim+1] = PetscSinReal(2.0*c*PETSC_PI/n); coords[(c+0*n)*dim+2] = 1.0;
      coords[(c+1*n)*dim+0] = PetscCosReal(2.0*c*PETSC_PI/n); coords[(c+1*n)*dim+1] = PetscSinReal(2.0*c*PETSC_PI/n); coords[(c+1*n)*dim+2] = 0.0;
    }
    if (rank == 0) {
      coords[(2*n+0)*dim+0] = 0.0; coords[(2*n+0)*dim+1] = 0.0; coords[(2*n+0)*dim+2] = 1.0;
      coords[(2*n+1)*dim+0] = 0.0; coords[(2*n+1)*dim+1] = 0.0; coords[(2*n+1)*dim+2] = 0.0;
    }
    PetscCall(VecRestoreArray(coordinates, &coords));
    PetscCall(DMSetCoordinatesLocal(dm, coordinates));
    PetscCall(VecDestroy(&coordinates));
  }
  /* Interpolate */
  if (interpolate) PetscCall(DMPlexInterpolateInPlace_Internal(dm));
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
  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexCreateWedgeCylinderMesh_Internal(*dm, n, interpolate));
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

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(dm, simplex, 3);
  PetscCall(DMSetDimension(dm, dim));
  PetscCall(DMSetCoordinateDim(dm, dim+1));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
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
      PetscCall(PetscCalloc1(numVerts * embedDim, &coordsIn));
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
      PetscCall(PetscCalloc1(numVerts * numVerts, &graph));
      for (i = 0; i < numVerts; ++i) {
        for (j = 0, k = 0; j < numVerts; ++j) {
          if (PetscAbsReal(DiffNormReal(embedDim, &coordsIn[i*embedDim], &coordsIn[j*embedDim]) - edgeLen) < PETSC_SMALL) {graph[i*numVerts+j] = 1; ++k;}
        }
        PetscCheck(k == degree,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid icosahedron, vertex %" PetscInt_FMT " degree %" PetscInt_FMT " != %" PetscInt_FMT, i, k, degree);
      }
      /* Build Topology */
      PetscCall(DMPlexSetChart(dm, 0, numCells+numVerts));
      for (c = 0; c < numCells; c++) {
        PetscCall(DMPlexSetConeSize(dm, c, embedDim));
      }
      PetscCall(DMSetUp(dm)); /* Allocate space for cones */
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
              PetscCall(DMPlexSetCone(dm, c++, cone));
            }
          }
        }
      }
      PetscCall(DMPlexSymmetrize(dm));
      PetscCall(DMPlexStratify(dm));
      PetscCall(PetscFree(graph));
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
      PetscCall(DMPlexSetChart(dm, 0, numCells+numEdges+numVerts));
      for (c = 0; c < numCells; c++) {
        PetscCall(DMPlexSetConeSize(dm, c, 4));
      }
      for (e = firstEdge; e < firstEdge+numEdges; ++e) {
        PetscCall(DMPlexSetConeSize(dm, e, 2));
      }
      PetscCall(DMSetUp(dm)); /* Allocate space for cones */
      if (rank == 0) {
        /* Cell 0 */
        cone[0] = 14; cone[1] = 15; cone[2] = 16; cone[3] = 17;
        PetscCall(DMPlexSetCone(dm, 0, cone));
        ornt[0] = 0; ornt[1] = 0; ornt[2] = 0; ornt[3] = 0;
        PetscCall(DMPlexSetConeOrientation(dm, 0, ornt));
        /* Cell 1 */
        cone[0] = 18; cone[1] = 19; cone[2] = 14; cone[3] = 20;
        PetscCall(DMPlexSetCone(dm, 1, cone));
        ornt[0] = 0; ornt[1] = 0; ornt[2] = -1; ornt[3] = 0;
        PetscCall(DMPlexSetConeOrientation(dm, 1, ornt));
        /* Cell 2 */
        cone[0] = 21; cone[1] = 22; cone[2] = 18; cone[3] = 23;
        PetscCall(DMPlexSetCone(dm, 2, cone));
        ornt[0] = 0; ornt[1] = 0; ornt[2] = -1; ornt[3] = 0;
        PetscCall(DMPlexSetConeOrientation(dm, 2, ornt));
        /* Cell 3 */
        cone[0] = 19; cone[1] = 22; cone[2] = 24; cone[3] = 15;
        PetscCall(DMPlexSetCone(dm, 3, cone));
        ornt[0] = -1; ornt[1] = -1; ornt[2] = 0; ornt[3] = -1;
        PetscCall(DMPlexSetConeOrientation(dm, 3, ornt));
        /* Cell 4 */
        cone[0] = 16; cone[1] = 24; cone[2] = 21; cone[3] = 25;
        PetscCall(DMPlexSetCone(dm, 4, cone));
        ornt[0] = -1; ornt[1] = -1; ornt[2] = -1; ornt[3] = 0;
        PetscCall(DMPlexSetConeOrientation(dm, 4, ornt));
        /* Cell 5 */
        cone[0] = 20; cone[1] = 17; cone[2] = 25; cone[3] = 23;
        PetscCall(DMPlexSetCone(dm, 5, cone));
        ornt[0] = -1; ornt[1] = -1; ornt[2] = -1; ornt[3] = -1;
        PetscCall(DMPlexSetConeOrientation(dm, 5, ornt));
        /* Edges */
        cone[0] =  6; cone[1] =  7;
        PetscCall(DMPlexSetCone(dm, 14, cone));
        cone[0] =  7; cone[1] =  8;
        PetscCall(DMPlexSetCone(dm, 15, cone));
        cone[0] =  8; cone[1] =  9;
        PetscCall(DMPlexSetCone(dm, 16, cone));
        cone[0] =  9; cone[1] =  6;
        PetscCall(DMPlexSetCone(dm, 17, cone));
        cone[0] = 10; cone[1] = 11;
        PetscCall(DMPlexSetCone(dm, 18, cone));
        cone[0] = 11; cone[1] =  7;
        PetscCall(DMPlexSetCone(dm, 19, cone));
        cone[0] =  6; cone[1] = 10;
        PetscCall(DMPlexSetCone(dm, 20, cone));
        cone[0] = 12; cone[1] = 13;
        PetscCall(DMPlexSetCone(dm, 21, cone));
        cone[0] = 13; cone[1] = 11;
        PetscCall(DMPlexSetCone(dm, 22, cone));
        cone[0] = 10; cone[1] = 12;
        PetscCall(DMPlexSetCone(dm, 23, cone));
        cone[0] = 13; cone[1] =  8;
        PetscCall(DMPlexSetCone(dm, 24, cone));
        cone[0] = 12; cone[1] =  9;
        PetscCall(DMPlexSetCone(dm, 25, cone));
      }
      PetscCall(DMPlexSymmetrize(dm));
      PetscCall(DMPlexStratify(dm));
      /* Build coordinates */
      PetscCall(PetscCalloc1(numVerts * embedDim, &coordsIn));
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
      PetscCall(PetscCalloc1(numVerts * embedDim, &coordsIn));
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
      PetscCheck(i == numVerts,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid 600-cell, vertices %" PetscInt_FMT " != %" PetscInt_FMT, i, numVerts);
      /* Construct graph */
      PetscCall(PetscCalloc1(numVerts * numVerts, &graph));
      for (i = 0; i < numVerts; ++i) {
        for (j = 0, k = 0; j < numVerts; ++j) {
          if (PetscAbsReal(DiffNormReal(embedDim, &coordsIn[i*embedDim], &coordsIn[j*embedDim]) - edgeLen) < PETSC_SMALL) {graph[i*numVerts+j] = 1; ++k;}
        }
        PetscCheck(k == degree,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid 600-cell, vertex %" PetscInt_FMT " degree %" PetscInt_FMT " != %" PetscInt_FMT, i, k, degree);
      }
      /* Build Topology */
      PetscCall(DMPlexSetChart(dm, 0, numCells+numVerts));
      for (c = 0; c < numCells; c++) {
        PetscCall(DMPlexSetConeSize(dm, c, embedDim));
      }
      PetscCall(DMSetUp(dm)); /* Allocate space for cones */
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
                  PetscCall(DMPlexSetCone(dm, c++, cone));
                }
              }
            }
          }
        }
      }
      PetscCall(DMPlexSymmetrize(dm));
      PetscCall(DMPlexStratify(dm));
      PetscCall(PetscFree(graph));
      break;
    }
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Unsupported dimension for sphere: %" PetscInt_FMT, dim);
  }
  /* Create coordinates */
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, embedDim));
  PetscCall(PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVerts));
  for (v = firstVertex; v < firstVertex+numVerts; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, embedDim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, embedDim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(VecSetBlockSize(coordinates, embedDim));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetType(coordinates,VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  for (v = 0; v < numVerts; ++v) for (d = 0; d < embedDim; ++d) {coords[v*embedDim+d] = coordsIn[v*embedDim+d];}
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscCall(PetscFree(coordsIn));
  {
    DM          cdm;
    PetscDS     cds;
    PetscScalar c = R;

    PetscCall(DMPlexCreateCoordinateSpace(dm, 1, snapToSphere));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetDS(cdm, &cds));
    PetscCall(PetscDSSetConstants(cds, 1, &c));
  }
  /* Wait for coordinate creation before doing in-place modification */
  if (simplex) PetscCall(DMPlexInterpolateInPlace_Internal(dm));
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

// u[] is a tentative normal on input. Replace with the implicit function gradient in the same direction
static PetscErrorCode TPSExtrudeNormalFunc_SchwarzP(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt r, PetscScalar u[], void *ctx) {
  for (PetscInt i=0; i<3; i++) {
    u[i] = -PETSC_PI * PetscSinReal(x[i] * PETSC_PI);
  }
  return 0;
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

// u[] is a tentative normal on input. Replace with the implicit function gradient in the same direction
static PetscErrorCode TPSExtrudeNormalFunc_Gyroid(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt r, PetscScalar u[], void *ctx) {
  PetscReal s[3] = {PetscSinReal(PETSC_PI * x[0]), PetscSinReal(PETSC_PI * (x[1] + .5)), PetscSinReal(PETSC_PI * (x[2] + .25))};
  PetscReal c[3] = {PetscCosReal(PETSC_PI * x[0]), PetscCosReal(PETSC_PI * (x[1] + .5)), PetscCosReal(PETSC_PI * (x[2] + .25))};
  u[0] = PETSC_PI * (c[0] * c[1] - s[2] * s[0]);
  u[1] = PETSC_PI * (c[1] * c[2] - s[0] * s[1]);
  u[2] = PETSC_PI * (c[2] * c[0] - s[1] * s[2]);
  return 0;
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

  PetscFunctionBegin;
  for (PetscInt iter=0; iter<10; iter++) {
    PetscScalar res[3], J[9];
    PetscReal resnorm;
    TPSNearestPointResJac(feval, x, y, res, J);
    resnorm = PetscSqrtReal(PetscSqr(PetscRealPart(res[0])) + PetscSqr(PetscRealPart(res[1])) + PetscSqr(PetscRealPart(res[2])));
    if (0) { // Turn on this monitor if you need to confirm quadratic convergence
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%" PetscInt_FMT "] res [%g %g %g]\n", iter, (double)PetscRealPart(res[0]), (double)PetscRealPart(res[1]), (double)PetscRealPart(res[2])));
    }
    if (resnorm < PETSC_SMALL) break;

    // Take the Newton step
    PetscCall(PetscKernel_A_gets_inverse_A_3(J, 0., PETSC_FALSE, NULL));
    PetscKernel_v_gets_v_minus_A_times_w_3(y, J, res);
  }
  for (PetscInt i=0; i<3; i++) x[i] = y[i];
  PetscFunctionReturn(0);
}

const char *const DMPlexTPSTypes[] = {"SCHWARZ_P", "GYROID", "DMPlexTPSType", "DMPLEX_TPS_", NULL};

static PetscErrorCode DMPlexCreateTPSMesh_Internal(DM dm, DMPlexTPSType tpstype, const PetscInt extent[], const DMBoundaryType periodic[], PetscBool tps_distribute, PetscInt refinements, PetscInt layers, PetscReal thickness)
{
  PetscMPIInt rank;
  PetscInt topoDim = 2, spaceDim = 3, numFaces = 0, numVertices = 0, numEdges = 0;
  PetscInt (*edges)[2] = NULL, *edgeSets = NULL;
  PetscInt *cells_flat = NULL;
  PetscReal *vtxCoords = NULL;
  TPSEvaluateFunc evalFunc = NULL;
  PetscSimplePointFunc normalFunc = NULL;
  DMLabel label;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCheck((layers != 0) ^ (thickness == 0.), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Layers %" PetscInt_FMT " must be nonzero iff thickness %g is nonzero", layers, (double)thickness);
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
      PetscCall(PetscMalloc1(3*numVertices, &vtxCoords));
      PetscCall(PetscMalloc1(Njunctions, &cells));
      PetscCall(PetscMalloc1(Ncuts*4, &edges));
      PetscCall(PetscMalloc1(Ncuts*4, &edgeSets));
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
      PetscCheck(numEdges == Ncuts * 4, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Edge count %" PetscInt_FMT " incompatible with number of cuts %" PetscInt_FMT, numEdges, Ncuts);
      numFaces = 24 * Njunctions;
      cells_flat = cells[0][0][0];
    }
    evalFunc = TPSEvaluate_SchwarzP;
    normalFunc = TPSExtrudeNormalFunc_SchwarzP;
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
      PetscCall(PetscMalloc1(numBlocks, &cells));
      PetscCall(PetscCalloc1(numBlocksPlus * vertsPerBlock,&seen));
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
      PetscCall(PetscMalloc1(numBlocksPlus * vertsPerBlock, &vertToTrueVert));
      PetscCall(PetscMalloc1(numVertices * 3, &vtxCoords));
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
      PetscCall(PetscFree(vertToTrueVert));
      PetscCall(PetscFree(seen));
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
      PetscCall(PetscMalloc1(numEdges, &edges));
      PetscCall(PetscMalloc1(numEdges, &edgeSets));
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
    normalFunc = TPSExtrudeNormalFunc_Gyroid;
    break;
  }

  PetscCall(DMSetDimension(dm, topoDim));
  if (!rank) PetscCall(DMPlexBuildFromCellList(dm, numFaces, numVertices, 4, cells_flat));
  else       PetscCall(DMPlexBuildFromCellList(dm, 0, 0, 0, NULL));
  PetscCall(PetscFree(cells_flat));
  {
    DM idm;
    PetscCall(DMPlexInterpolate(dm, &idm));
    PetscCall(DMPlexReplace_Static(dm, &idm));
  }
  if (!rank) PetscCall(DMPlexBuildCoordinatesFromCellList(dm, spaceDim, vtxCoords));
  else       PetscCall(DMPlexBuildCoordinatesFromCellList(dm, spaceDim, NULL));
  PetscCall(PetscFree(vtxCoords));

  PetscCall(DMCreateLabel(dm, "Face Sets"));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  for (PetscInt e=0; e<numEdges; e++) {
    PetscInt njoin;
    const PetscInt *join, verts[] = {numFaces + edges[e][0], numFaces + edges[e][1]};
    PetscCall(DMPlexGetJoin(dm, 2, verts, &njoin, &join));
    PetscCheck(njoin == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected unique join of vertices %" PetscInt_FMT " and %" PetscInt_FMT, edges[e][0], edges[e][1]);
    PetscCall(DMLabelSetValue(label, join[0], edgeSets[e]));
    PetscCall(DMPlexRestoreJoin(dm, 2, verts, &njoin, &join));
  }
  PetscCall(PetscFree(edges));
  PetscCall(PetscFree(edgeSets));
  if (tps_distribute) {
    DM               pdm = NULL;
    PetscPartitioner part;

    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm, 0, NULL, &pdm));
    if (pdm) {
      PetscCall(DMPlexReplace_Static(dm, &pdm));
    }
    // Do not auto-distribute again
    PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  }

  PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  for (PetscInt refine=0; refine<refinements; refine++) {
    PetscInt m;
    DM dmf;
    Vec X;
    PetscScalar *x;
    PetscCall(DMRefine(dm, MPI_COMM_NULL, &dmf));
    PetscCall(DMPlexReplace_Static(dm, &dmf));

    PetscCall(DMGetCoordinatesLocal(dm, &X));
    PetscCall(VecGetLocalSize(X, &m));
    PetscCall(VecGetArray(X, &x));
    for (PetscInt i=0; i<m; i+=3) {
      PetscCall(TPSNearestPoint(evalFunc, &x[i]));
    }
    PetscCall(VecRestoreArray(X, &x));
  }

  // Face Sets has already been propagated to new vertices during refinement; this propagates to the initial vertices.
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCall(DMPlexLabelComplete(dm, label));

  if (thickness > 0) {
    DM edm,cdm,ecdm;
    DMPlexTransform tr;
    const char *prefix;
    PetscOptions options;
    // Code from DMPlexExtrude
    PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
    PetscCall(DMPlexTransformSetDM(tr, dm));
    PetscCall(DMPlexTransformSetType(tr, DMPLEXEXTRUDE));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) tr,  prefix));
    PetscCall(PetscObjectGetOptions((PetscObject) dm, &options));
    PetscCall(PetscObjectSetOptions((PetscObject) tr, options));
    PetscCall(DMPlexTransformExtrudeSetLayers(tr, layers));
    PetscCall(DMPlexTransformExtrudeSetThickness(tr, thickness));
    PetscCall(DMPlexTransformExtrudeSetTensor(tr, PETSC_FALSE));
    PetscCall(DMPlexTransformExtrudeSetSymmetric(tr, PETSC_TRUE));
    PetscCall(DMPlexTransformExtrudeSetNormalFunction(tr, normalFunc));
    PetscCall(DMPlexTransformSetFromOptions(tr));
    PetscCall(PetscObjectSetOptions((PetscObject) tr, NULL));
    PetscCall(DMPlexTransformSetUp(tr));
    PetscCall(PetscObjectViewFromOptions((PetscObject) tr, NULL, "-dm_plex_tps_transform_view"));
    PetscCall(DMPlexTransformApply(tr, dm, &edm));
    PetscCall(DMCopyDisc(dm, edm));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateDM(edm, &ecdm));
    PetscCall(DMCopyDisc(cdm, ecdm));
    PetscCall(DMPlexTransformCreateDiscLabels(tr, edm));
    PetscCall(DMPlexTransformDestroy(&tr));
    if (edm) {
      ((DM_Plex *)edm->data)->printFEM = ((DM_Plex *)dm->data)->printFEM;
      ((DM_Plex *)edm->data)->printL2  = ((DM_Plex *)dm->data)->printL2;
    }
    PetscCall(DMPlexReplace_Static(dm, &edm));
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
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexCreateTPSMesh_Internal(*dm, tpstype, extent, periodic, tps_distribute, refinements, layers, thickness));
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
  PetscFunctionBegin;
  PetscValidPointer(dm, 5);
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexCreateSphereMesh_Internal(*dm, dim, simplex, R));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBallMesh_Internal(DM dm, PetscInt dim, PetscReal R)
{
  DM             sdm, vol;
  DMLabel        bdlabel;

  PetscFunctionBegin;
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), &sdm));
  PetscCall(DMSetType(sdm, DMPLEX));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) sdm, "bd_"));
  PetscCall(DMPlexCreateSphereMesh_Internal(sdm, dim-1, PETSC_TRUE, R));
  PetscCall(DMSetFromOptions(sdm));
  PetscCall(DMViewFromOptions(sdm, NULL, "-dm_view"));
  PetscCall(DMPlexGenerate(sdm, NULL, PETSC_TRUE, &vol));
  PetscCall(DMDestroy(&sdm));
  PetscCall(DMPlexReplace_Static(dm, &vol));
  PetscCall(DMCreateLabel(dm, "marker"));
  PetscCall(DMGetLabel(dm, "marker", &bdlabel));
  PetscCall(DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, bdlabel));
  PetscCall(DMPlexLabelComplete(dm, bdlabel));
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
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexCreateBallMesh_Internal(*dm, dim, R));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateReferenceCell_Internal(DM rdm, DMPolytopeType ct)
{
  PetscFunctionBegin;
  switch (ct) {
    case DM_POLYTOPE_POINT:
    {
      PetscInt    numPoints[1]        = {1};
      PetscInt    coneSize[1]         = {0};
      PetscInt    cones[1]            = {0};
      PetscInt    coneOrientations[1] = {0};
      PetscScalar vertexCoords[1]     = {0.0};

      PetscCall(DMSetDimension(rdm, 0));
      PetscCall(DMPlexCreateFromDAG(rdm, 0, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case DM_POLYTOPE_SEGMENT:
    {
      PetscInt    numPoints[2]        = {2, 1};
      PetscInt    coneSize[3]         = {2, 0, 0};
      PetscInt    cones[2]            = {1, 2};
      PetscInt    coneOrientations[2] = {0, 0};
      PetscScalar vertexCoords[2]     = {-1.0,  1.0};

      PetscCall(DMSetDimension(rdm, 1));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    {
      PetscInt    numPoints[2]        = {2, 1};
      PetscInt    coneSize[3]         = {2, 0, 0};
      PetscInt    cones[2]            = {1, 2};
      PetscInt    coneOrientations[2] = {0, 0};
      PetscScalar vertexCoords[2]     = {-1.0,  1.0};

      PetscCall(DMSetDimension(rdm, 1));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case DM_POLYTOPE_TRIANGLE:
    {
      PetscInt    numPoints[2]        = {3, 1};
      PetscInt    coneSize[4]         = {3, 0, 0, 0};
      PetscInt    cones[3]            = {1, 2, 3};
      PetscInt    coneOrientations[3] = {0, 0, 0};
      PetscScalar vertexCoords[6]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0};

      PetscCall(DMSetDimension(rdm, 2));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case DM_POLYTOPE_QUADRILATERAL:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0};

      PetscCall(DMSetDimension(rdm, 2));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,  1.0, 1.0};

      PetscCall(DMSetDimension(rdm, 2));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case DM_POLYTOPE_TETRAHEDRON:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {-1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, -1.0, 1.0};

      PetscCall(DMSetDimension(rdm, 3));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
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

      PetscCall(DMSetDimension(rdm, 3));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
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

      PetscCall(DMSetDimension(rdm, 3));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
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

      PetscCall(DMSetDimension(rdm, 3));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
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

      PetscCall(DMSetDimension(rdm, 3));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
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

      PetscCall(DMSetDimension(rdm, 3));
      PetscCall(DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    default: SETERRQ(PetscObjectComm((PetscObject) rdm), PETSC_ERR_ARG_WRONG, "Cannot create reference cell for cell type %s", DMPolytopeTypes[ct]);
  }
  {
    PetscInt Nv, v;

    /* Must create the celltype label here so that we do not automatically try to compute the types */
    PetscCall(DMCreateLabel(rdm, "celltype"));
    PetscCall(DMPlexSetCellType(rdm, 0, ct));
    PetscCall(DMPlexGetChart(rdm, NULL, &Nv));
    for (v = 1; v < Nv; ++v) PetscCall(DMPlexSetCellType(rdm, v, DM_POLYTOPE_POINT));
  }
  PetscCall(DMPlexInterpolateInPlace_Internal(rdm));
  PetscCall(PetscObjectSetName((PetscObject) rdm, DMPolytopeTypes[ct]));
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
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, refdm));
  PetscCall(DMSetType(*refdm, DMPLEX));
  PetscCall(DMPlexCreateReferenceCell_Internal(*refdm, ct));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateBoundaryLabel_Private(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;
  PetscBool      hasLabel;

  PetscFunctionBeginUser;
  PetscCall(DMHasLabel(dm, name, &hasLabel));
  if (hasLabel) PetscFunctionReturn(0);
  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexMarkBoundaryFaces(plex, 1, label));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

const char * const DMPlexShapes[] = {"box", "box_surface", "ball", "sphere", "cylinder", "schwarz_p", "gyroid", "doublet", "unknown", "DMPlexShape", "DM_SHAPE_", NULL};

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

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  /* TODO Turn this into a registration interface */
  PetscCall(PetscOptionsString("-dm_plex_filename", "File containing a mesh", "DMPlexCreateFromFile", filename, filename, sizeof(filename), &fflg));
  PetscCall(PetscOptionsString("-dm_plex_boundary_filename", "File containing a mesh boundary", "DMPlexCreateFromFile", bdFilename, bdFilename, sizeof(bdFilename), &bdfflg));
  PetscCall(PetscOptionsString("-dm_plex_name", "Name of the mesh in the file", "DMPlexCreateFromFile", plexname, plexname, sizeof(plexname), &nameflg));
  PetscCall(PetscOptionsEnum("-dm_plex_cell", "Cell shape", "", DMPolytopeTypes, (PetscEnum) cell, (PetscEnum *) &cell, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_reference_cell_domain", "Use a reference cell domain", "", refDomain, &refDomain, NULL));
  PetscCall(PetscOptionsEnum("-dm_plex_shape", "Shape for built-in mesh", "", DMPlexShapes, (PetscEnum) shape, (PetscEnum *) &shape, &flg));
  PetscCall(PetscOptionsBoundedInt("-dm_plex_dim", "Topological dimension of the mesh", "DMGetDimension", dim, &dim, &flg, 0));
  PetscCheck(!(dim < 0) && !(dim > 3),comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " should be in [1, 3]", dim);
  PetscCall(PetscOptionsBool("-dm_plex_simplex", "Mesh cell shape", "", simplex,  &simplex, &flg));
  PetscCall(PetscOptionsBool("-dm_plex_interpolate", "Flag to create edges and faces automatically", "", interpolate, &interpolate, &flg));
  PetscCall(PetscOptionsBool("-dm_plex_adj_cone", "Set adjacency direction", "DMSetBasicAdjacency", adjCone,  &adjCone, &flg));
  PetscCall(PetscOptionsBool("-dm_plex_adj_closure", "Set adjacency size", "DMSetBasicAdjacency", adjClosure,  &adjClosure, &flg2));
  if (flg || flg2) PetscCall(DMSetBasicAdjacency(dm, adjCone, adjClosure));

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

    PetscCall(DMPlexCreateFromFile(PetscObjectComm((PetscObject) dm), filename, plexname, interpolate, &dmnew));
    PetscCall(DMPlexCopy_Internal(dm, PETSC_FALSE, dmnew));
    PetscCall(DMPlexReplace_Static(dm, &dmnew));
  } else if (refDomain) {
    PetscCall(DMPlexCreateReferenceCell_Internal(dm, cell));
  } else if (bdfflg) {
    DM bdm, dmnew;

    PetscCall(DMPlexCreateFromFile(PetscObjectComm((PetscObject) dm), bdFilename, plexname, interpolate, &bdm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) bdm, "bd_"));
    PetscCall(DMSetFromOptions(bdm));
    PetscCall(DMPlexGenerate(bdm, NULL, interpolate, &dmnew));
    PetscCall(DMDestroy(&bdm));
    PetscCall(DMPlexCopy_Internal(dm, PETSC_FALSE, dmnew));
    PetscCall(DMPlexReplace_Static(dm, &dmnew));
  } else {
    PetscCall(PetscObjectSetName((PetscObject) dm, DMPlexShapes[shape]));
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
        PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of faces along each dimension", "", faces, &n, &flg));
        n    = 3;
        PetscCall(PetscOptionsRealArray("-dm_plex_box_lower", "Lower left corner of box", "", lower, &n, &flg));
        PetscCheck(!flg || !(n != dim),comm, PETSC_ERR_ARG_SIZ, "Lower box point had %" PetscInt_FMT " values, should have been %" PetscInt_FMT, n, dim);
        n    = 3;
        PetscCall(PetscOptionsRealArray("-dm_plex_box_upper", "Upper right corner of box", "", upper, &n, &flg));
        PetscCheck(!flg || !(n != dim),comm, PETSC_ERR_ARG_SIZ, "Upper box point had %" PetscInt_FMT " values, should have been %" PetscInt_FMT, n, dim);
        n    = 3;
        PetscCall(PetscOptionsEnumArray("-dm_plex_box_bd", "Boundary type for each dimension", "", DMBoundaryTypes, (PetscEnum *) bdt, &n, &flg));
        PetscCheck(!flg || !(n != dim),comm, PETSC_ERR_ARG_SIZ, "Box boundary types had %" PetscInt_FMT " values, should have been %" PetscInt_FMT, n, dim);
        switch (cell) {
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
            PetscCall(DMPlexCreateWedgeBoxMesh_Internal(dm, faces, lower, upper, bdt));
            if (!interpolate) {
              DM udm;

              PetscCall(DMPlexUninterpolate(dm, &udm));
              PetscCall(DMPlexReplace_Static(dm, &udm));
            }
            break;
          default:
            PetscCall(DMPlexCreateBoxMesh_Internal(dm, dim, simplex, faces, lower, upper, bdt, interpolate));
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
        PetscCall(PetscOptionsIntArray("-dm_plex_box_faces", "Number of faces along each dimension", "", faces, &n, &flg));
        n    = 3;
        PetscCall(PetscOptionsRealArray("-dm_plex_box_lower", "Lower left corner of box", "", lower, &n, &flg));
        PetscCheck(!flg || !(n != dim+1),comm, PETSC_ERR_ARG_SIZ, "Lower box point had %" PetscInt_FMT " values, should have been %" PetscInt_FMT, n, dim+1);
        n    = 3;
        PetscCall(PetscOptionsRealArray("-dm_plex_box_upper", "Upper right corner of box", "", upper, &n, &flg));
        PetscCheck(!flg || !(n != dim+1),comm, PETSC_ERR_ARG_SIZ, "Upper box point had %" PetscInt_FMT " values, should have been %" PetscInt_FMT, n, dim+1);
        PetscCall(DMPlexCreateBoxSurfaceMesh_Internal(dm, dim+1, faces, lower, upper, interpolate));
      }
      break;
      case DM_SHAPE_SPHERE:
      {
        PetscReal R = 1.0;

        PetscCall(PetscOptionsReal("-dm_plex_sphere_radius", "Radius of the sphere", "", R,  &R, &flg));
        PetscCall(DMPlexCreateSphereMesh_Internal(dm, dim, simplex, R));
      }
      break;
      case DM_SHAPE_BALL:
      {
        PetscReal R = 1.0;

        PetscCall(PetscOptionsReal("-dm_plex_ball_radius", "Radius of the ball", "", R,  &R, &flg));
        PetscCall(DMPlexCreateBallMesh_Internal(dm, dim, R));
      }
      break;
      case DM_SHAPE_CYLINDER:
      {
        DMBoundaryType bdt = DM_BOUNDARY_NONE;
        PetscInt       Nw  = 6;

        PetscCall(PetscOptionsEnum("-dm_plex_cylinder_bd", "Boundary type in the z direction", "", DMBoundaryTypes, (PetscEnum) bdt, (PetscEnum *) &bdt, NULL));
        PetscCall(PetscOptionsInt("-dm_plex_cylinder_num_wedges", "Number of wedges around the cylinder", "", Nw, &Nw, NULL));
        switch (cell) {
          case DM_POLYTOPE_TRI_PRISM_TENSOR:
            PetscCall(DMPlexCreateWedgeCylinderMesh_Internal(dm, Nw, interpolate));
            break;
          default:
            PetscCall(DMPlexCreateHexCylinderMesh_Internal(dm, bdt));
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
        PetscCall(PetscOptionsIntArray("-dm_plex_tps_extent", "Number of replicas for each of three dimensions", NULL, extent, (three=3, &three), NULL));
        PetscCall(PetscOptionsInt("-dm_plex_tps_refine", "Number of refinements", NULL, refine, &refine, NULL));
        PetscCall(PetscOptionsEnumArray("-dm_plex_tps_periodic", "Periodicity in each of three dimensions", NULL, DMBoundaryTypes, (PetscEnum*)periodic, (three=3, &three), NULL));
        PetscCall(PetscOptionsInt("-dm_plex_tps_layers", "Number of layers in volumetric extrusion (or zero to not extrude)", NULL, layers, &layers, NULL));
        PetscCall(PetscOptionsReal("-dm_plex_tps_thickness", "Thickness of volumetric extrusion", NULL, thickness, &thickness, NULL));
        PetscCall(DMPlexDistributeGetDefault(dm, &tps_distribute));
        PetscCall(PetscOptionsBool("-dm_plex_tps_distribute", "Distribute the 2D mesh prior to refinement and extrusion", NULL, tps_distribute, &tps_distribute, NULL));
        PetscCall(DMPlexCreateTPSMesh_Internal(dm, tps_type, extent, periodic, tps_distribute, refine, layers, thickness));
      }
      break;
      case DM_SHAPE_DOUBLET:
      {
        DM        dmnew;
        PetscReal rl = 0.0;

        PetscCall(PetscOptionsReal("-dm_plex_doublet_refinementlimit", "Refinement limit", NULL, rl, &rl, NULL));
        PetscCall(DMPlexCreateDoublet(PetscObjectComm((PetscObject)dm), dim, simplex, interpolate, rl, &dmnew));
        PetscCall(DMPlexCopy_Internal(dm, PETSC_FALSE, dmnew));
        PetscCall(DMPlexReplace_Static(dm, &dmnew));
      }
      break;
      default: SETERRQ(comm, PETSC_ERR_SUP, "Domain shape %s is unsupported", DMPlexShapes[shape]);
    }
  }
  PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  if (!((PetscObject)dm)->name && nameflg) {
    PetscCall(PetscObjectSetName((PetscObject)dm, plexname));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetFromOptions_NonRefinement_Plex(PetscOptionItems *PetscOptionsObject, DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscBool      flg;
  char           bdLabel[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  /* Handle viewing */
  PetscCall(PetscOptionsBool("-dm_plex_print_set_values", "Output all set values info", "DMPlexMatSetClosure", PETSC_FALSE, &mesh->printSetValues, NULL));
  PetscCall(PetscOptionsBoundedInt("-dm_plex_print_fem", "Debug output level all fem computations", "DMPlexSNESComputeResidualFEM", 0, &mesh->printFEM, NULL,0));
  PetscCall(PetscOptionsReal("-dm_plex_print_tol", "Tolerance for FEM output", "DMPlexSNESComputeResidualFEM", mesh->printTol, &mesh->printTol, NULL));
  PetscCall(PetscOptionsBoundedInt("-dm_plex_print_l2", "Debug output level all L2 diff computations", "DMComputeL2Diff", 0, &mesh->printL2, NULL,0));
  PetscCall(DMMonitorSetFromOptions(dm, "-dm_plex_monitor_throughput", "Monitor the simulation throughput", "DMPlexMonitorThroughput", DMPlexMonitorThroughput, NULL, &flg));
  if (flg) PetscCall(PetscLogDefaultBegin());
  /* Labeling */
  PetscCall(PetscOptionsString("-dm_plex_boundary_label", "Label to mark the mesh boundary", "", bdLabel, bdLabel, sizeof(bdLabel), &flg));
  if (flg) PetscCall(DMPlexCreateBoundaryLabel_Private(dm, bdLabel));
  /* Point Location */
  PetscCall(PetscOptionsBool("-dm_plex_hash_location", "Use grid hashing for point location", "DMInterpolate", PETSC_FALSE, &mesh->useHashLocation, NULL));
  /* Partitioning and distribution */
  PetscCall(PetscOptionsBool("-dm_plex_partition_balance", "Attempt to evenly divide points on partition boundary between processes", "DMPlexSetPartitionBalance", PETSC_FALSE, &mesh->partitionBalance, NULL));
  /* Generation and remeshing */
  PetscCall(PetscOptionsBool("-dm_plex_remesh_bd", "Allow changes to the boundary on remeshing", "DMAdapt", PETSC_FALSE, &mesh->remeshBd, NULL));
  /* Projection behavior */
  PetscCall(PetscOptionsBoundedInt("-dm_plex_max_projection_height", "Maxmimum mesh point height used to project locally", "DMPlexSetMaxProjectionHeight", 0, &mesh->maxProjectionHeight, NULL,0));
  PetscCall(PetscOptionsBool("-dm_plex_regular_refinement", "Use special nested projection algorithm for regular refinement", "DMPlexSetRegularRefinement", mesh->regularRefinement, &mesh->regularRefinement, NULL));
  /* Checking structure */
  {
    PetscBool   flg = PETSC_FALSE, flg2 = PETSC_FALSE, all = PETSC_FALSE;

    PetscCall(PetscOptionsBool("-dm_plex_check_all", "Perform all checks", NULL, PETSC_FALSE, &all, &flg2));
    PetscCall(PetscOptionsBool("-dm_plex_check_symmetry", "Check that the adjacency information in the mesh is symmetric", "DMPlexCheckSymmetry", PETSC_FALSE, &flg, &flg2));
    if (all || (flg && flg2)) PetscCall(DMPlexCheckSymmetry(dm));
    PetscCall(PetscOptionsBool("-dm_plex_check_skeleton", "Check that each cell has the correct number of vertices (only for homogeneous simplex or tensor meshes)", "DMPlexCheckSkeleton", PETSC_FALSE, &flg, &flg2));
    if (all || (flg && flg2)) PetscCall(DMPlexCheckSkeleton(dm, 0));
    PetscCall(PetscOptionsBool("-dm_plex_check_faces", "Check that the faces of each cell give a vertex order this is consistent with what we expect from the cell type", "DMPlexCheckFaces", PETSC_FALSE, &flg, &flg2));
    if (all || (flg && flg2)) PetscCall(DMPlexCheckFaces(dm, 0));
    PetscCall(PetscOptionsBool("-dm_plex_check_geometry", "Check that cells have positive volume", "DMPlexCheckGeometry", PETSC_FALSE, &flg, &flg2));
    if (all || (flg && flg2)) PetscCall(DMPlexCheckGeometry(dm));
    PetscCall(PetscOptionsBool("-dm_plex_check_pointsf", "Check some necessary conditions for PointSF", "DMPlexCheckPointSF", PETSC_FALSE, &flg, &flg2));
    if (all || (flg && flg2)) PetscCall(DMPlexCheckPointSF(dm));
    PetscCall(PetscOptionsBool("-dm_plex_check_interface_cones", "Check points on inter-partition interfaces have conforming order of cone points", "DMPlexCheckInterfaceCones", PETSC_FALSE, &flg, &flg2));
    if (all || (flg && flg2)) PetscCall(DMPlexCheckInterfaceCones(dm));
    PetscCall(PetscOptionsBool("-dm_plex_check_cell_shape", "Check cell shape", "DMPlexCheckCellShape", PETSC_FALSE, &flg, &flg2));
    if (flg && flg2) PetscCall(DMPlexCheckCellShape(dm, PETSC_TRUE, PETSC_DETERMINE));
  }
  {
    PetscReal scale = 1.0;

    PetscCall(PetscOptionsReal("-dm_plex_scale", "Scale factor for mesh coordinates", "DMPlexScale", scale, &scale, &flg));
    if (flg) {
      Vec coordinates, coordinatesLocal;

      PetscCall(DMGetCoordinates(dm, &coordinates));
      PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));
      PetscCall(VecScale(coordinates, scale));
      PetscCall(VecScale(coordinatesLocal, scale));
    }
  }
  PetscCall(PetscPartitionerSetFromOptions(mesh->partitioner));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_Plex(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscFunctionList ordlist;
  char              oname[256];
  PetscReal         volume = -1.0;
  PetscInt          prerefine = 0, refine = 0, r, coarsen = 0, overlap = 0, extLayers = 0, dim;
  PetscBool         uniformOrig, created = PETSC_FALSE, uniform = PETSC_TRUE, distribute, interpolate = PETSC_TRUE, coordSpace = PETSC_TRUE, remap = PETSC_TRUE, ghostCells = PETSC_FALSE, isHierarchy, ignoreModel = PETSC_FALSE, flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscOptionsHeadBegin(PetscOptionsObject,"DMPlex Options");
  /* Handle automatic creation */
  PetscCall(DMGetDimension(dm, &dim));
  if (dim < 0) {PetscCall(DMPlexCreateFromOptions_Internal(PetscOptionsObject, &coordSpace, dm));created = PETSC_TRUE;}
  /* Handle interpolation before distribution */
  PetscCall(PetscOptionsBool("-dm_plex_interpolate_pre", "Flag to interpolate mesh before distribution", "", interpolate, &interpolate, &flg));
  if (flg) {
    DMPlexInterpolatedFlag interpolated;

    PetscCall(DMPlexIsInterpolated(dm, &interpolated));
    if (interpolated == DMPLEX_INTERPOLATED_FULL && !interpolate) {
      DM udm;

      PetscCall(DMPlexUninterpolate(dm, &udm));
      PetscCall(DMPlexReplace_Static(dm, &udm));
    } else if (interpolated != DMPLEX_INTERPOLATED_FULL && interpolate) {
      DM idm;

      PetscCall(DMPlexInterpolate(dm, &idm));
      PetscCall(DMPlexReplace_Static(dm, &idm));
    }
  }
  /* Handle DMPlex refinement before distribution */
  PetscCall(PetscOptionsBool("-dm_refine_ignore_model", "Flag to ignore the geometry model when refining", "DMCreate", ignoreModel, &ignoreModel, &flg));
  if (flg) {((DM_Plex *) dm->data)->ignoreModel = ignoreModel;}
  PetscCall(DMPlexGetRefinementUniform(dm, &uniformOrig));
  PetscCall(PetscOptionsBoundedInt("-dm_refine_pre", "The number of refinements before distribution", "DMCreate", prerefine, &prerefine, NULL,0));
  PetscCall(PetscOptionsBool("-dm_refine_remap_pre", "Flag to control coordinate remapping", "DMCreate", remap, &remap, NULL));
  PetscCall(PetscOptionsBool("-dm_refine_uniform_pre", "Flag for uniform refinement before distribution", "DMCreate", uniform, &uniform, &flg));
  if (flg) PetscCall(DMPlexSetRefinementUniform(dm, uniform));
  PetscCall(PetscOptionsReal("-dm_refine_volume_limit_pre", "The maximum cell volume after refinement before distribution", "DMCreate", volume, &volume, &flg));
  if (flg) {
    PetscCall(DMPlexSetRefinementUniform(dm, PETSC_FALSE));
    PetscCall(DMPlexSetRefinementLimit(dm, volume));
    prerefine = PetscMax(prerefine, 1);
  }
  for (r = 0; r < prerefine; ++r) {
    DM             rdm;
    PetscPointFunc coordFunc = ((DM_Plex*) dm->data)->coordFunc;

    PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
    PetscCall(DMRefine(dm, PetscObjectComm((PetscObject) dm), &rdm));
    PetscCall(DMPlexReplace_Static(dm, &rdm));
    PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
    if (coordFunc && remap) {
      PetscCall(DMPlexRemapGeometry(dm, 0.0, coordFunc));
      ((DM_Plex*) dm->data)->coordFunc = coordFunc;
    }
  }
  PetscCall(DMPlexSetRefinementUniform(dm, uniformOrig));
  /* Handle DMPlex extrusion before distribution */
  PetscCall(PetscOptionsBoundedInt("-dm_extrude", "The number of layers to extrude", "", extLayers, &extLayers, NULL, 0));
  if (extLayers) {
    DM edm;

    PetscCall(DMExtrude(dm, extLayers, &edm));
    PetscCall(DMPlexReplace_Static(dm, &edm));
    ((DM_Plex *) dm->data)->coordFunc = NULL;
    PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
    extLayers = 0;
  }
  /* Handle DMPlex reordering before distribution */
  PetscCall(MatGetOrderingList(&ordlist));
  PetscCall(PetscOptionsFList("-dm_plex_reorder", "Set mesh reordering type", "DMPlexGetOrdering", ordlist, MATORDERINGNATURAL, oname, sizeof(oname), &flg));
  if (flg) {
    DM pdm;
    IS perm;

    PetscCall(DMPlexGetOrdering(dm, oname, NULL, &perm));
    PetscCall(DMPlexPermute(dm, perm, &pdm));
    PetscCall(ISDestroy(&perm));
    PetscCall(DMPlexReplace_Static(dm, &pdm));
    PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
  }
  /* Handle DMPlex distribution */
  PetscCall(DMPlexDistributeGetDefault(dm, &distribute));
  PetscCall(PetscOptionsBool("-dm_distribute", "Flag to redistribute a mesh among processes", "DMCreate", distribute, &distribute, NULL));
  PetscCall(PetscOptionsBoundedInt("-dm_distribute_overlap", "The size of the overlap halo", "DMCreate", overlap, &overlap, NULL, 0));
  if (distribute) {
    DM               pdm = NULL;
    PetscPartitioner part;

    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm, overlap, NULL, &pdm));
    if (pdm) {
      PetscCall(DMPlexReplace_Static(dm, &pdm));
    }
  }
  /* Create coordinate space */
  if (created) {
    DM_Plex  *mesh = (DM_Plex *) dm->data;
    PetscInt  degree = 1;
    PetscBool periodic, flg;

    PetscCall(PetscOptionsBool("-dm_coord_space", "Use an FEM space for coordinates", "", coordSpace, &coordSpace, &flg));
    PetscCall(PetscOptionsInt("-dm_coord_petscspace_degree", "FEM degree for coordinate space", "", degree, &degree, NULL));
    if (coordSpace) PetscCall(DMPlexCreateCoordinateSpace(dm, degree, mesh->coordFunc));
    if (flg && !coordSpace) {
      DM           cdm;
      PetscDS      cds;
      PetscObject  obj;
      PetscClassId id;

      PetscCall(DMGetCoordinateDM(dm, &cdm));
      PetscCall(DMGetDS(cdm, &cds));
      PetscCall(PetscDSGetDiscretization(cds, 0, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscContainer dummy;

        PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &dummy));
        PetscCall(PetscObjectSetName((PetscObject) dummy, "coordinates"));
        PetscCall(DMSetField(cdm, 0, NULL, (PetscObject) dummy));
        PetscCall(PetscContainerDestroy(&dummy));
        PetscCall(DMClearDS(cdm));
      }
      mesh->coordFunc = NULL;
    }
    PetscCall(DMLocalizeCoordinates(dm));
    PetscCall(DMGetPeriodicity(dm, &periodic, NULL, NULL, NULL));
    if (periodic) PetscCall(DMSetPeriodicity(dm, PETSC_TRUE, NULL, NULL, NULL));
  }
  /* Handle DMPlex refinement */
  remap = PETSC_TRUE;
  PetscCall(PetscOptionsBoundedInt("-dm_refine", "The number of uniform refinements", "DMCreate", refine, &refine, NULL,0));
  PetscCall(PetscOptionsBool("-dm_refine_remap", "Flag to control coordinate remapping", "DMCreate", remap, &remap, NULL));
  PetscCall(PetscOptionsBoundedInt("-dm_refine_hierarchy", "The number of uniform refinements", "DMCreate", refine, &refine, &isHierarchy,0));
  if (refine) PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
  if (refine && isHierarchy) {
    DM *dms, coarseDM;

    PetscCall(DMGetCoarseDM(dm, &coarseDM));
    PetscCall(PetscObjectReference((PetscObject)coarseDM));
    PetscCall(PetscMalloc1(refine,&dms));
    PetscCall(DMRefineHierarchy(dm, refine, dms));
    /* Total hack since we do not pass in a pointer */
    PetscCall(DMPlexSwap_Static(dm, dms[refine-1]));
    if (refine == 1) {
      PetscCall(DMSetCoarseDM(dm, dms[0]));
      PetscCall(DMPlexSetRegularRefinement(dm, PETSC_TRUE));
    } else {
      PetscCall(DMSetCoarseDM(dm, dms[refine-2]));
      PetscCall(DMPlexSetRegularRefinement(dm, PETSC_TRUE));
      PetscCall(DMSetCoarseDM(dms[0], dms[refine-1]));
      PetscCall(DMPlexSetRegularRefinement(dms[0], PETSC_TRUE));
    }
    PetscCall(DMSetCoarseDM(dms[refine-1], coarseDM));
    PetscCall(PetscObjectDereference((PetscObject)coarseDM));
    /* Free DMs */
    for (r = 0; r < refine; ++r) {
      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dms[r]));
      PetscCall(DMDestroy(&dms[r]));
    }
    PetscCall(PetscFree(dms));
  } else {
    for (r = 0; r < refine; ++r) {
      DM             rdm;
      PetscPointFunc coordFunc = ((DM_Plex*) dm->data)->coordFunc;

      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
      PetscCall(DMRefine(dm, PetscObjectComm((PetscObject) dm), &rdm));
      /* Total hack since we do not pass in a pointer */
      PetscCall(DMPlexReplace_Static(dm, &rdm));
      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
      if (coordFunc && remap) {
        PetscCall(DMPlexRemapGeometry(dm, 0.0, coordFunc));
        ((DM_Plex*) dm->data)->coordFunc = coordFunc;
      }
    }
  }
  /* Handle DMPlex coarsening */
  PetscCall(PetscOptionsBoundedInt("-dm_coarsen", "Coarsen the mesh", "DMCreate", coarsen, &coarsen, NULL,0));
  PetscCall(PetscOptionsBoundedInt("-dm_coarsen_hierarchy", "The number of coarsenings", "DMCreate", coarsen, &coarsen, &isHierarchy,0));
  if (coarsen && isHierarchy) {
    DM *dms;

    PetscCall(PetscMalloc1(coarsen, &dms));
    PetscCall(DMCoarsenHierarchy(dm, coarsen, dms));
    /* Free DMs */
    for (r = 0; r < coarsen; ++r) {
      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dms[r]));
      PetscCall(DMDestroy(&dms[r]));
    }
    PetscCall(PetscFree(dms));
  } else {
    for (r = 0; r < coarsen; ++r) {
      DM             cdm;
      PetscPointFunc coordFunc = ((DM_Plex*) dm->data)->coordFunc;

      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
      PetscCall(DMCoarsen(dm, PetscObjectComm((PetscObject) dm), &cdm));
      /* Total hack since we do not pass in a pointer */
      PetscCall(DMPlexReplace_Static(dm, &cdm));
      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
      if (coordFunc) {
        PetscCall(DMPlexRemapGeometry(dm, 0.0, coordFunc));
        ((DM_Plex*) dm->data)->coordFunc = coordFunc;
      }
    }
  }
  /* Handle ghost cells */
  PetscCall(PetscOptionsBool("-dm_plex_create_fv_ghost_cells", "Flag to create finite volume ghost cells on the boundary", "DMCreate", ghostCells, &ghostCells, NULL));
  if (ghostCells) {
    DM   gdm;
    char lname[PETSC_MAX_PATH_LEN];

    lname[0] = '\0';
    PetscCall(PetscOptionsString("-dm_plex_fv_ghost_cells_label", "Label name for ghost cells boundary", "DMCreate", lname, lname, sizeof(lname), &flg));
    PetscCall(DMPlexConstructGhostCells(dm, flg ? lname : NULL, NULL, &gdm));
    PetscCall(DMPlexReplace_Static(dm, &gdm));
  }
  /* Handle 1D order */
  {
    DM           cdm, rdm;
    PetscDS      cds;
    PetscObject  obj;
    PetscClassId id = PETSC_OBJECT_CLASSID;
    IS           perm;
    PetscInt     dim, Nf;
    PetscBool    distributed;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexIsDistributed(dm, &distributed));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetDS(cdm, &cds));
    PetscCall(PetscDSGetNumFields(cds, &Nf));
    if (Nf) {
      PetscCall(PetscDSGetDiscretization(cds, 0, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
    }
    if (dim == 1 && !distributed && id != PETSCFE_CLASSID) {
      PetscCall(DMPlexGetOrdering1D(dm, &perm));
      PetscCall(DMPlexPermute(dm, perm, &rdm));
      PetscCall(DMPlexReplace_Static(dm, &rdm));
      PetscCall(ISDestroy(&perm));
    }
  }
  /* Handle */
  PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Plex(DM dm,Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector_Section_Private(dm,vec));
  /* PetscCall(VecSetOperation(*vec, VECOP_DUPLICATE, (void(*)(void)) VecDuplicate_MPI_DM)); */
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Plex));
  PetscCall(VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void)) VecView_Plex_Native));
  PetscCall(VecSetOperation(*vec, VECOP_LOAD, (void (*)(void)) VecLoad_Plex));
  PetscCall(VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void)) VecLoad_Plex_Native));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Plex(DM dm,Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector_Section_Private(dm,vec));
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Plex_Local));
  PetscCall(VecSetOperation(*vec, VECOP_LOAD, (void (*)(void)) VecLoad_Plex_Local));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetDimPoints_Plex(DM dm, PetscInt dim, PetscInt *pStart, PetscInt *pEnd)
{
  PetscInt       depth, d;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (depth == 1) {
    PetscCall(DMGetDimension(dm, &d));
    if (dim == 0)      PetscCall(DMPlexGetDepthStratum(dm, dim, pStart, pEnd));
    else if (dim == d) PetscCall(DMPlexGetDepthStratum(dm, 1, pStart, pEnd));
    else               {*pStart = 0; *pEnd = 0;}
  } else {
    PetscCall(DMPlexGetDepthStratum(dm, dim, pStart, pEnd));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetNeighbors_Plex(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscSF           sf;
  PetscInt          niranks, njranks, n;
  const PetscMPIInt *iranks, *jranks;
  DM_Plex           *data = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscCall(DMGetPointSF(dm, &sf));
  if (!data->neighbors) {
    PetscCall(PetscSFSetUp(sf));
    PetscCall(PetscSFGetRootRanks(sf, &njranks, &jranks, NULL, NULL, NULL));
    PetscCall(PetscSFGetLeafRanks(sf, &niranks, &iranks, NULL, NULL));
    PetscCall(PetscMalloc1(njranks + niranks + 1, &data->neighbors));
    PetscCall(PetscArraycpy(data->neighbors + 1, jranks, njranks));
    PetscCall(PetscArraycpy(data->neighbors + njranks + 1, iranks, niranks));
    n = njranks + niranks;
    PetscCall(PetscSortRemoveDupsMPIInt(&n, data->neighbors + 1));
    /* The following cast should never fail: can't have more neighbors than PETSC_MPI_INT_MAX */
    PetscCall(PetscMPIIntCast(n, data->neighbors));
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
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertBoundaryValues_C",DMPlexInsertBoundaryValues_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertTimeDerviativeBoundaryValues_C",DMPlexInsertTimeDerivativeBoundaryValues_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMCreateNeumannOverlap_C",DMCreateNeumannOverlap_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexGetOverlap_C",DMPlexGetOverlap_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexDistributeGetDefault_C",DMPlexDistributeGetDefault_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexDistributeSetDefault_C",DMPlexDistributeSetDefault_Plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMInterpolateSolution_C",DMInterpolateSolution_Plex));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMClone_Plex(DM dm, DM *newdm)
{
  DM_Plex        *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  mesh->refct++;
  (*newdm)->data = mesh;
  PetscCall(PetscObjectChangeTypeName((PetscObject) *newdm, DMPLEX));
  PetscCall(DMInitialize_Plex(*newdm));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscNewLog(dm,&mesh));
  dm->data = mesh;

  mesh->refct             = 1;
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &mesh->coneSection));
  mesh->cones             = NULL;
  mesh->coneOrientations  = NULL;
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &mesh->supportSection));
  mesh->supports          = NULL;
  mesh->refinementUniform = PETSC_TRUE;
  mesh->refinementLimit   = -1.0;
  mesh->distDefault       = PETSC_TRUE;
  mesh->interpolated      = DMPLEX_INTERPOLATED_INVALID;
  mesh->interpolatedCollective = DMPLEX_INTERPOLATED_INVALID;

  mesh->facesTmp = NULL;

  mesh->tetgenOpts   = NULL;
  mesh->triangleOpts = NULL;
  PetscCall(PetscPartitionerCreate(PetscObjectComm((PetscObject)dm), &mesh->partitioner));
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

  PetscCall(DMInitialize_Plex(dm));
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
  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  PetscCall(DMCreate(comm, mesh));
  PetscCall(DMSetType(*mesh, DMPLEX));
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

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(dm,NVertices,4);
  PetscCall(PetscLogEventBegin(DMPLEX_BuildFromCellList,dm,0,0,0));
  /* Get/check global number of vertices */
  {
    PetscInt NVerticesInCells, i;
    const PetscInt len = numCells * numCorners;

    /* NVerticesInCells = max(cells) + 1 */
    NVerticesInCells = PETSC_MIN_INT;
    for (i=0; i<len; i++) if (cells[i] > NVerticesInCells) NVerticesInCells = cells[i];
    ++NVerticesInCells;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &NVerticesInCells, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm)));

    if (numVertices == PETSC_DECIDE && NVertices == PETSC_DECIDE) NVertices = NVerticesInCells;
    else PetscCheck(NVertices == PETSC_DECIDE || NVertices >= NVerticesInCells,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Specified global number of vertices %" PetscInt_FMT " must be greater than or equal to the number of vertices in cells %" PetscInt_FMT,NVertices,NVerticesInCells);
  }
  /* Count locally unique vertices */
  {
    PetscHSetI vhash;
    PetscInt off = 0;

    PetscCall(PetscHSetICreate(&vhash));
    for (c = 0; c < numCells; ++c) {
      for (p = 0; p < numCorners; ++p) {
        PetscCall(PetscHSetIAdd(vhash, cells[c*numCorners+p]));
      }
    }
    PetscCall(PetscHSetIGetSize(vhash, &numVerticesAdj));
    if (!verticesAdjSaved) PetscCall(PetscMalloc1(numVerticesAdj, &verticesAdj));
    else { verticesAdj = *verticesAdjSaved; }
    PetscCall(PetscHSetIGetElems(vhash, &off, verticesAdj));
    PetscCall(PetscHSetIDestroy(&vhash));
    PetscCheck(off == numVerticesAdj,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of local vertices %" PetscInt_FMT " should be %" PetscInt_FMT, off, numVerticesAdj);
  }
  PetscCall(PetscSortInt(numVerticesAdj, verticesAdj));
  /* Create cones */
  PetscCall(DMPlexSetChart(dm, 0, numCells+numVerticesAdj));
  for (c = 0; c < numCells; ++c) PetscCall(DMPlexSetConeSize(dm, c, numCorners));
  PetscCall(DMSetUp(dm));
  PetscCall(DMPlexGetCones(dm,&cones));
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      const PetscInt gv = cells[c*numCorners+p];
      PetscInt       lv;

      /* Positions within verticesAdj form 0-based local vertex numbering;
         we need to shift it by numCells to get correct DAG points (cells go first) */
      PetscCall(PetscFindInt(gv, numVerticesAdj, verticesAdj, &lv));
      PetscCheck(lv >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not find global vertex %" PetscInt_FMT " in local connectivity", gv);
      cones[c*numCorners+p] = lv+numCells;
    }
  }
  /* Build point sf */
  PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)dm), &layout));
  PetscCall(PetscLayoutSetSize(layout, NVertices));
  PetscCall(PetscLayoutSetLocalSize(layout, numVertices));
  PetscCall(PetscLayoutSetBlockSize(layout, 1));
  PetscCall(PetscSFCreateByMatchingIndices(layout, numVerticesAdj, verticesAdj, NULL, numCells, numVerticesAdj, verticesAdj, NULL, numCells, vertexSF, &sfPoint));
  PetscCall(PetscLayoutDestroy(&layout));
  if (!verticesAdjSaved) PetscCall(PetscFree(verticesAdj));
  PetscCall(PetscObjectSetName((PetscObject) sfPoint, "point SF"));
  if (dm->sf) {
    const char *prefix;

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm->sf, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)sfPoint, prefix));
  }
  PetscCall(DMSetPointSF(dm, sfPoint));
  PetscCall(PetscSFDestroy(&sfPoint));
  if (vertexSF) PetscCall(PetscObjectSetName((PetscObject)(*vertexSF), "Vertex Ownership SF"));
  /* Fill in the rest of the topology structure */
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  PetscCall(PetscLogEventEnd(DMPLEX_BuildFromCellList,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCheck(vStart >= 0 && vEnd >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not set up properly. DMPlexBuildFromCellList() should be called first.");
  PetscCall(DMSetCoordinateDim(dm, spaceDim));
  PetscCall(PetscSFGetGraph(sfVert, &numVertices, &numVerticesAdj, NULL, NULL));
  PetscCheck(vEnd - vStart == numVerticesAdj,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Supplied sfVert has wrong number of leaves = %" PetscInt_FMT " != %" PetscInt_FMT " = vEnd - vStart",numVerticesAdj,vEnd - vStart);
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, spaceDim));
  PetscCall(PetscSectionSetChart(coordSection, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, spaceDim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, spaceDim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), &coordinates));
  PetscCall(VecSetBlockSize(coordinates, spaceDim));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetType(coordinates,VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  {
    MPI_Datatype coordtype;

    /* Need a temp buffer for coords if we have complex/single */
    PetscCallMPI(MPI_Type_contiguous(spaceDim, MPIU_SCALAR, &coordtype));
    PetscCallMPI(MPI_Type_commit(&coordtype));
#if defined(PETSC_USE_COMPLEX)
    {
    PetscScalar *svertexCoords;
    PetscInt    i;
    PetscCall(PetscMalloc1(numVertices*spaceDim,&svertexCoords));
    for (i=0; i<numVertices*spaceDim; i++) svertexCoords[i] = vertexCoords[i];
    PetscCall(PetscSFBcastBegin(sfVert, coordtype, svertexCoords, coords,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sfVert, coordtype, svertexCoords, coords,MPI_REPLACE));
    PetscCall(PetscFree(svertexCoords));
    }
#else
    PetscCall(PetscSFBcastBegin(sfVert, coordtype, vertexCoords, coords,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sfVert, coordtype, vertexCoords, coords,MPI_REPLACE));
#endif
    PetscCallMPI(MPI_Type_free(&coordtype));
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscCall(PetscLogEventEnd(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscValidLogicalCollectiveInt(*dm, dim, 2);
  PetscValidLogicalCollectiveInt(*dm, spaceDim, 9);
  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMPlexBuildFromCellListParallel(*dm, numCells, numVertices, NVertices, numCorners, cells, &sfVert, verticesAdj));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;
  }
  PetscCall(DMPlexBuildCoordinatesFromCellListParallel(*dm, spaceDim, sfVert, vertexCoords));
  if (vertexSF) *vertexSF = sfVert;
  else PetscCall(PetscSFDestroy(&sfVert));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_BuildFromCellList,dm,0,0,0));
  PetscCall(DMGetDimension(dm, &dim));
  /* Get/check global number of vertices */
  {
    PetscInt NVerticesInCells, i;
    const PetscInt len = numCells * numCorners;

    /* NVerticesInCells = max(cells) + 1 */
    NVerticesInCells = PETSC_MIN_INT;
    for (i=0; i<len; i++) if (cells[i] > NVerticesInCells) NVerticesInCells = cells[i];
    ++NVerticesInCells;

    if (numVertices == PETSC_DECIDE) numVertices = NVerticesInCells;
    else PetscCheck(numVertices >= NVerticesInCells,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Specified number of vertices %" PetscInt_FMT " must be greater than or equal to the number of vertices in cells %" PetscInt_FMT,numVertices,NVerticesInCells);
  }
  PetscCall(DMPlexSetChart(dm, 0, numCells+numVertices));
  for (c = 0; c < numCells; ++c) {
    PetscCall(DMPlexSetConeSize(dm, c, numCorners));
  }
  PetscCall(DMSetUp(dm));
  PetscCall(DMPlexGetCones(dm,&cones));
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      cones[c*numCorners+p] = cells[c*numCorners+p]+numCells;
    }
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  PetscCall(PetscLogEventEnd(DMPLEX_BuildFromCellList,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCheck(vStart >= 0 && vEnd >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "DM is not set up properly. DMPlexBuildFromCellList() should be called first.");
  PetscCall(DMSetCoordinateDim(dm, spaceDim));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, spaceDim));
  PetscCall(PetscSectionSetChart(coordSection, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, spaceDim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, spaceDim));
  }
  PetscCall(PetscSectionSetUp(coordSection));

  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMCreateLocalVector(cdm, &coordinates));
  PetscCall(VecSetBlockSize(coordinates, spaceDim));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  for (v = 0; v < vEnd-vStart; ++v) {
    for (d = 0; d < spaceDim; ++d) {
      coords[v*spaceDim+d] = vertexCoords[v*spaceDim+d];
    }
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscCall(PetscLogEventEnd(DMPLEX_BuildCoordinatesFromCellList,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCheck(dim,comm, PETSC_ERR_ARG_OUTOFRANGE, "This is not appropriate for 0-dimensional meshes. Consider either creating the DM using DMPlexCreateFromDAG(), by hand, or using DMSwarm.");
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetDimension(*dm, dim));
  if (!rank) PetscCall(DMPlexBuildFromCellList(*dm, numCells, numVertices, numCorners, cells));
  else       PetscCall(DMPlexBuildFromCellList(*dm, 0, 0, 0, NULL));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;
  }
  if (!rank) PetscCall(DMPlexBuildCoordinatesFromCellList(*dm, spaceDim, vertexCoords));
  else       PetscCall(DMPlexBuildCoordinatesFromCellList(*dm, spaceDim, NULL));
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

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  PetscCheck(dimEmbed >= dim,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Embedding dimension %" PetscInt_FMT " cannot be less than intrinsic dimension %" PetscInt_FMT,dimEmbed,dim);
  for (d = 0; d <= depth; ++d) pEnd += numPoints[d];
  PetscCall(DMPlexSetChart(dm, pStart, pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscCall(DMPlexSetConeSize(dm, p, coneSize[p-pStart]));
    if (firstVertex < 0 && !coneSize[p - pStart]) {
      firstVertex = p - pStart;
    }
  }
  PetscCheck(firstVertex >= 0 || !numPoints[0],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected %" PetscInt_FMT " vertices but could not find any", numPoints[0]);
  PetscCall(DMSetUp(dm)); /* Allocate space for cones */
  for (p = pStart, off = 0; p < pEnd; off += coneSize[p-pStart], ++p) {
    PetscCall(DMPlexSetCone(dm, p, &cones[off]));
    PetscCall(DMPlexSetConeOrientation(dm, p, &coneOrientations[off]));
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  /* Build coordinates */
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dimEmbed));
  PetscCall(PetscSectionSetChart(coordSection, firstVertex, firstVertex+numPoints[0]));
  for (v = firstVertex; v < firstVertex+numPoints[0]; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, dimEmbed));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dimEmbed));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, dimEmbed));
  PetscCall(VecSetType(coordinates,VECSTANDARD));
  if (vertexCoords) {
    PetscCall(VecGetArray(coordinates, &coords));
    for (v = 0; v < numPoints[0]; ++v) {
      PetscInt off;

      PetscCall(PetscSectionGetOffset(coordSection, v+firstVertex, &off));
      for (d = 0; d < dimEmbed; ++d) {
        coords[off+d] = vertexCoords[v*dimEmbed+d];
      }
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
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

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  if (rank == 0) {
    PetscCall(PetscViewerRead(viewer, line, 4, NULL, PETSC_STRING));
    snum = sscanf(line, "%d %d %d %d", &Nc, &Nv, &Ncn, &Nl);
    PetscCheck(snum == 4,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
  } else {
    Nc = Nv = Ncn = Nl = 0;
  }
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexSetChart(*dm, 0, Nc+Nv));
  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMSetCoordinateDim(*dm, cdim));
  /* Read topology */
  if (rank == 0) {
    char     format[PETSC_MAX_PATH_LEN];
    PetscInt cone[8];
    int      vbuf[8], v;

    for (c = 0; c < Ncn; ++c) {format[c*3+0] = '%'; format[c*3+1] = 'd'; format[c*3+2] = ' ';}
    format[Ncn*3-1] = '\0';
    for (c = 0; c < Nc; ++c) PetscCall(DMPlexSetConeSize(*dm, c, Ncn));
    PetscCall(DMSetUp(*dm));
    for (c = 0; c < Nc; ++c) {
      PetscCall(PetscViewerRead(viewer, line, Ncn, NULL, PETSC_STRING));
      switch (Ncn) {
        case 2: snum = sscanf(line, format, &vbuf[0], &vbuf[1]);break;
        case 3: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2]);break;
        case 4: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2], &vbuf[3]);break;
        case 6: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2], &vbuf[3], &vbuf[4], &vbuf[5]);break;
        case 8: snum = sscanf(line, format, &vbuf[0], &vbuf[1], &vbuf[2], &vbuf[3], &vbuf[4], &vbuf[5], &vbuf[6], &vbuf[7]);break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No cell shape with %d vertices", Ncn);
      }
      PetscCheck(snum == Ncn,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
      for (v = 0; v < Ncn; ++v) cone[v] = vbuf[v] + Nc;
      /* Hexahedra are inverted */
      if (Ncn == 8) {
        PetscInt tmp = cone[1];
        cone[1] = cone[3];
        cone[3] = tmp;
      }
      PetscCall(DMPlexSetCone(*dm, c, cone));
    }
  }
  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  /* Read coordinates */
  PetscCall(DMGetCoordinateSection(*dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, cdim));
  PetscCall(PetscSectionSetChart(coordSection, Nc, Nc + Nv));
  for (v = Nc; v < Nc+Nv; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, cdim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, cdim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, cdim));
  PetscCall(VecSetType(coordinates, VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  if (rank == 0) {
    char   format[PETSC_MAX_PATH_LEN];
    double x[3];
    int    l, val[3];

    if (Nl) {
      for (l = 0; l < Nl; ++l) {format[l*3+0] = '%'; format[l*3+1] = 'd'; format[l*3+2] = ' ';}
      format[Nl*3-1] = '\0';
      PetscCall(DMCreateLabel(*dm, "marker"));
      PetscCall(DMGetLabel(*dm, "marker", &marker));
    }
    for (v = 0; v < Nv; ++v) {
      PetscCall(PetscViewerRead(viewer, line, 3+Nl, NULL, PETSC_STRING));
      snum = sscanf(line, "%lg %lg %lg", &x[0], &x[1], &x[2]);
      PetscCheck(snum == 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
      switch (Nl) {
        case 0: snum = 0;break;
        case 1: snum = sscanf(line, format, &val[0]);break;
        case 2: snum = sscanf(line, format, &val[0], &val[1]);break;
        case 3: snum = sscanf(line, format, &val[0], &val[1], &val[2]);break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Request support for %d labels", Nl);
      }
      PetscCheck(snum == Nl,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse cell-vertex file: %s", line);
      for (d = 0; d < cdim; ++d) coords[v*cdim+d] = x[d];
      for (l = 0; l < Nl; ++l) PetscCall(DMLabelSetValue(marker, v+Nc, val[l]));
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscCall(PetscViewerDestroy(&viewer));
  if (interpolate) {
    DM      idm;
    DMLabel bdlabel;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;

    if (!Nl) {
      PetscCall(DMCreateLabel(*dm, "marker"));
      PetscCall(DMGetLabel(*dm, "marker", &bdlabel));
      PetscCall(DMPlexMarkBoundaryFaces(*dm, PETSC_DETERMINE, bdlabel));
      PetscCall(DMPlexLabelComplete(*dm, bdlabel));
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

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  if (plexname) PetscValidCharPointer(plexname, 3);
  PetscValidPointer(dm, 5);
  PetscCall(DMInitializePackage());
  PetscCall(PetscLogEventBegin(DMPLEX_CreateFromFile,0,0,0,0));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscStrlen(filename, &len));
  PetscCheck(len,comm, PETSC_ERR_ARG_WRONG, "Filename must be a valid path");
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extGmsh,      4, &isGmsh));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-5)],  extGmsh2,     5, &isGmsh2));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-5)],  extGmsh4,     5, &isGmsh4));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-5)],  extCGNS,      5, &isCGNS));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extExodus,    4, &isExodus));
  if (!isExodus) {
    PetscCall(PetscStrncmp(&filename[PetscMax(0,len-2)],  extExodus_e,    2, &isExodus));
  }
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extGenesis,   4, &isGenesis));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extFluent,    4, &isFluent));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-3)],  extHDF5,      3, &isHDF5));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extMed,       4, &isMed));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extPLY,       4, &isPLY));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-10)], extEGADSLite, 10, &isEGADSLite));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-6)],  extEGADS,     6, &isEGADS));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extIGES,      4, &isIGES));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extSTEP,      4, &isSTEP));
  PetscCall(PetscStrncmp(&filename[PetscMax(0,len-4)],  extCV,        4, &isCV));
  if (isGmsh || isGmsh2 || isGmsh4) {
    PetscCall(DMPlexCreateGmshFromFile(comm, filename, interpolate, dm));
  } else if (isCGNS) {
    PetscCall(DMPlexCreateCGNSFromFile(comm, filename, interpolate, dm));
  } else if (isExodus || isGenesis) {
    PetscCall(DMPlexCreateExodusFromFile(comm, filename, interpolate, dm));
  } else if (isFluent) {
    PetscCall(DMPlexCreateFluentFromFile(comm, filename, interpolate, dm));
  } else if (isHDF5) {
    PetscBool      load_hdf5_xdmf = PETSC_FALSE;
    PetscViewer viewer;

    /* PETSC_VIEWER_HDF5_XDMF is used if the filename ends with .xdmf.h5, or if -dm_plex_create_from_hdf5_xdmf option is present */
    PetscCall(PetscStrncmp(&filename[PetscMax(0,len-8)], ".xdmf",  5, &load_hdf5_xdmf));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_create_from_hdf5_xdmf", &load_hdf5_xdmf, NULL));
    PetscCall(PetscViewerCreate(comm, &viewer));
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERHDF5));
    PetscCall(PetscViewerSetOptionsPrefix(viewer, "dm_plex_create_"));
    PetscCall(PetscViewerSetFromOptions(viewer));
    PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(viewer, filename));

    PetscCall(DMCreate(comm, dm));
    PetscCall(PetscObjectSetName((PetscObject)(*dm), plexname));
    PetscCall(DMSetType(*dm, DMPLEX));
    if (load_hdf5_xdmf) PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
    PetscCall(DMLoad(*dm, viewer));
    if (load_hdf5_xdmf) PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    if (interpolate) {
      DM idm;

      PetscCall(DMPlexInterpolate(*dm, &idm));
      PetscCall(DMDestroy(dm));
      *dm  = idm;
    }
  } else if (isMed) {
    PetscCall(DMPlexCreateMedFromFile(comm, filename, interpolate, dm));
  } else if (isPLY) {
    PetscCall(DMPlexCreatePLYFromFile(comm, filename, interpolate, dm));
  } else if (isEGADSLite || isEGADS || isIGES || isSTEP) {
    if (isEGADSLite) PetscCall(DMPlexCreateEGADSLiteFromFile(comm, filename, dm));
    else             PetscCall(DMPlexCreateEGADSFromFile(comm, filename, dm));
    if (!interpolate) {
      DM udm;

      PetscCall(DMPlexUninterpolate(*dm, &udm));
      PetscCall(DMDestroy(dm));
      *dm  = udm;
    }
  } else if (isCV) {
    PetscCall(DMPlexCreateCellVertexFromFile(comm, filename, interpolate, dm));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);
  PetscCall(PetscStrlen(plexname, &len));
  if (len) PetscCall(PetscObjectSetName((PetscObject)(*dm), plexname));
  PetscCall(PetscLogEventEnd(DMPLEX_CreateFromFile,0,0,0,0));
  PetscFunctionReturn(0);
}
