#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petscdmda.h>
#include <petscsf.h>

/*@
  DMPlexCreateDoublet - Creates a mesh of two cells of the specified type, optionally with later refinement.

  Collective on MPI_Comm

  Input Parameters:
+ comm - The communicator for the DM object
. dim - The spatial dimension
. simplex - Flag for simplicial cells, otherwise they are tensor product cells
. interpolate - Flag to create intermediate mesh pieces (edges, faces)
. refinementUniform - Flag for uniform parallel refinement
- refinementLimit - A nonzero number indicates the largest admissible volume for a refined cell

  Output Parameter:
. dm  - The DM object

  Level: beginner

.keywords: DM, create
.seealso: DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateDoublet(MPI_Comm comm, PetscInt dim, PetscBool simplex, PetscBool interpolate, PetscBool refinementUniform, PetscReal refinementLimit, DM *newdm)
{
  DM             dm;
  PetscInt       p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
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
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
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
        PetscInt    markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};

        ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
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
        PetscInt    markerPoints[10]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
        for (p = 0; p < 5; ++p) {ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
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
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
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
    DM idm = NULL;
    const char *name;

    ierr = DMPlexInterpolate(*newdm, &idm);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) *newdm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)    idm,  name);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*newdm, idm);CHKERRQ(ierr);
    ierr = DMCopyLabels(*newdm, idm);CHKERRQ(ierr);
    ierr = DMDestroy(newdm);CHKERRQ(ierr);
    *newdm = idm;
  }
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*newdm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(newdm);CHKERRQ(ierr);
      *newdm = distributedMesh;
    }
    if (refinementUniform) {
      ierr = DMPlexSetRefinementUniform(*newdm, refinementUniform);CHKERRQ(ierr);
      ierr = DMRefine(*newdm, comm, &refinedMesh);CHKERRQ(ierr);
      if (refinedMesh) {
        ierr = DMDestroy(newdm);CHKERRQ(ierr);
        *newdm = refinedMesh;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateSquareBoundary - Creates a 1D mesh the is the boundary of a square lattice.

  Collective on MPI_Comm

  Input Parameters:
+ comm  - The communicator for the DM object
. lower - The lower left corner coordinates
. upper - The upper right corner coordinates
- edges - The number of cells in each direction

  Output Parameter:
. dm  - The DM object

  Note: Here is the numbering returned for 2 cells in each direction:
$ 18--5-17--4--16
$  |     |     |
$  6    10     3
$  |     |     |
$ 19-11-20--9--15
$  |     |     |
$  7     8     2
$  |     |     |
$ 12--0-13--1--14

  Level: beginner

.keywords: DM, create
.seealso: DMPlexCreateBoxMesh(), DMPlexCreateCubeBoundary(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateSquareBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
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
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  if (!rank) {
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

/*@
  DMPlexCreateCubeBoundary - Creates a 2D mesh the is the boundary of a cubic lattice.

  Collective on MPI_Comm

  Input Parameters:
+ comm  - The communicator for the DM object
. lower - The lower left front corner coordinates
. upper - The upper right back corner coordinates
- edges - The number of cells in each direction

  Output Parameter:
. dm  - The DM object

  Level: beginner

.keywords: DM, create
.seealso: DMPlexCreateBoxMesh(), DMPlexCreateSquareBoundary(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateCubeBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
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
  if ((faces[0] < 1) || (faces[1] < 1) || (faces[2] < 1)) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Must have at least 1 face per side");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  vertices[0] = faces[0]+1; vertices[1] = faces[1]+1; vertices[2] = faces[2]+1;
  numVertices = vertices[0]*vertices[1]*vertices[2];
  if (!rank) {
    PetscInt f;

    ierr = DMPlexSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMPlexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for (v = 0; v < numFaces+numVertices; ++v) {
      ierr = DMSetLabelValue(dm, "marker", v, 1);CHKERRQ(ierr);
    }

    /* Side 0 (Top) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vertices[0]*vertices[1]*(vertices[2]-1) + vy*vertices[0] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]+1; cone[3] = voffset+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 1 (Bottom) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vy*(faces[0]+1) + vx;
        cone[0] = voffset+1; cone[1] = voffset; cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]+1;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        iface++;
      }
    }

    /* Side 2 (Front) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]*vertices[1]+1; cone[3] = voffset+vertices[0]*vertices[1];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
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
        iface++;
      }
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMSetCoordinateDim(dm, 3);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  for (v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 3);CHKERRQ(ierr);
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

/*@
  DMPlexCreateBoxMesh - Creates a mesh on the tensor product of unit intervals (box) using simplices.

  Collective on MPI_Comm

  Input Parameters:
+ comm - The communicator for the DM object
. dim - The spatial dimension
. numFaces - Number of faces per dimension
- interpolate - Flag to create intermediate mesh pieces (edges, faces)

  Output Parameter:
. dm  - The DM object

  Level: beginner

.keywords: DM, create
.seealso: DMPlexCreateHexBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscInt numFaces, PetscBool interpolate, DM *dm)
{
  DM             boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(boundary,dim,2);
  ierr = DMSetType(boundary, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(boundary, dim-1);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(boundary, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
  {
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};
    PetscInt  edges[2];

    edges[0] = numFaces; edges[1] = numFaces;
    ierr = DMPlexCreateSquareBoundary(boundary, lower, upper, edges);CHKERRQ(ierr);
    break;
  }
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3];

    faces[0] = numFaces; faces[1] = numFaces; faces[2] = numFaces;
    ierr = DMPlexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  ierr = DMPlexGenerate(boundary, NULL, interpolate, dm);CHKERRQ(ierr);
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
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"marker");CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"Face Sets");CHKERRQ(ierr);
  if (bdX == DM_BOUNDARY_PERIODIC || bdX == DM_BOUNDARY_TWIST ||
      bdY == DM_BOUNDARY_PERIODIC || bdY == DM_BOUNDARY_TWIST ||
      bdZ == DM_BOUNDARY_PERIODIC || bdZ == DM_BOUNDARY_TWIST) {
    ierr = PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_periodic_cut", &cutMarker, NULL);CHKERRQ(ierr);
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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Dimension %d not supported",dim);
    break;
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
    const PetscInt numXEdges    = !rank ? edges[0] : 0;
    const PetscInt numYEdges    = !rank ? edges[1] : 0;
    const PetscInt numZEdges    = !rank ? edges[2] : 0;
    const PetscInt numXVertices = !rank ? (bdX == DM_BOUNDARY_PERIODIC || bdX == DM_BOUNDARY_TWIST ? edges[0] : edges[0]+1) : 0;
    const PetscInt numYVertices = !rank ? (bdY == DM_BOUNDARY_PERIODIC || bdY == DM_BOUNDARY_TWIST ? edges[1] : edges[1]+1) : 0;
    const PetscInt numZVertices = !rank ? (bdZ == DM_BOUNDARY_PERIODIC || bdZ == DM_BOUNDARY_TWIST ? edges[2] : edges[2]+1) : 0;
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
          PetscInt ornt[6] = {-4,  0,  0, -1,  0, -4}; /* ??? */
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
          PetscInt face    = firstXFace + (fz*numYEdges+fy)*numXVertices + fx;
          PetscInt edgeL   = firstZEdge + (  fy*                 numXVertices+fx)*numZEdges + fz;
          PetscInt edgeR   = firstZEdge + (((fy+1)%numYVertices)*numXVertices+fx)*numZEdges + fz;
          PetscInt edgeB   = firstYEdge + (  fz*                 numXVertices+fx)*numYEdges + fy;
          PetscInt edgeT   = firstYEdge + (((fz+1)%numZVertices)*numXVertices+fx)*numYEdges + fy;
          PetscInt ornt[4] = {0, 0, -2, -2};
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
          PetscInt edgeL   = firstZEdge + (fy*numXVertices+  fx                 )*numZEdges + fz;
          PetscInt edgeR   = firstZEdge + (fy*numXVertices+((fx+1)%numXVertices))*numZEdges + fz;
          PetscInt edgeB   = firstXEdge + (  fz                 *numYVertices+fy)*numXEdges + fx;
          PetscInt edgeT   = firstXEdge + (((fz+1)%numZVertices)*numYVertices+fy)*numXEdges + fx;
          PetscInt ornt[4] = {0, 0, -2, -2};
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
          PetscInt edgeL   = firstYEdge + (fz*numXVertices+  fx                 )*numYEdges + fy;
          PetscInt edgeR   = firstYEdge + (fz*numXVertices+((fx+1)%numXVertices))*numYEdges + fy;
          PetscInt edgeB   = firstXEdge + (fz*numYVertices+  fy                 )*numXEdges + fx;
          PetscInt edgeT   = firstXEdge + (fz*numYVertices+((fy+1)%numYVertices))*numXEdges + fx;
          PetscInt ornt[4] = {0, 0, -2, -2};
          PetscInt cone[4];

          if (dim == 2) {
            if (bdX == DM_BOUNDARY_TWIST && fx == numXEdges-1) {edgeR += numYEdges-1-2*fy; ornt[1] = -2;}
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
          const PetscInt vertexB = firstVertex + (  ez                 *numYVertices+vy)*numXVertices + vx;
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
              if (vx == 0 && cutMarker) {
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
              if (vy == 0 && cutMarker) {
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

/*@
  DMPlexCreateHexBoxMesh - Creates a mesh on the tensor product of unit intervals (box) using hexahedra.

  Collective on MPI_Comm

  Input Parameters:
+ comm  - The communicator for the DM object
. dim   - The spatial dimension
. periodicX - The boundary type for the X direction
. periodicY - The boundary type for the Y direction
. periodicZ - The boundary type for the Z direction
- cells - The number of cells in each direction

  Output Parameter:
. dm  - The DM object

  Note: Here is the numbering returned for 2 cells in each direction:
$ 22--8-23--9--24
$  |     |     |
$ 13  2 14  3  15
$  |     |     |
$ 19--6-20--7--21
$  |     |     |
$ 10  0 11  1 12
$  |     |     |
$ 16--4-17--5--18

  Level: beginner

.keywords: DM, create
.seealso: DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateHexBoxMesh(MPI_Comm comm, PetscInt dim, const PetscInt cells[], DMBoundaryType periodicX, DMBoundaryType periodicY, DMBoundaryType periodicZ, DM *dm)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 7);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(*dm,dim,2);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 0.0};
    PetscInt  edges[3];

    edges[0] = cells[0];
    edges[1] = cells[1];
    edges[2] = 0;

    ierr = DMPlexCreateCubeMesh_Internal(*dm, lower, upper, edges, periodicX, periodicY, DM_BOUNDARY_NONE);CHKERRQ(ierr);
    if (periodicX == DM_BOUNDARY_PERIODIC || periodicX == DM_BOUNDARY_TWIST ||
        periodicY == DM_BOUNDARY_PERIODIC || periodicY == DM_BOUNDARY_TWIST) {
      PetscReal      L[2];
      PetscReal      maxCell[2];
      DMBoundaryType bdType[2];

      bdType[0] = periodicX;
      bdType[1] = periodicY;
      for (i = 0; i < dim; i++) {
        L[i]       = upper[i] - lower[i];
        maxCell[i] = 1.1 * (L[i] / PetscMax(1,cells[i]));
      }

      ierr = DMSetPeriodicity(*dm,PETSC_TRUE,maxCell,L,bdType);CHKERRQ(ierr);
    }
    break;
  }
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};

    ierr = DMPlexCreateCubeMesh_Internal(*dm, lower, upper, cells, periodicX, periodicY, periodicZ);CHKERRQ(ierr);
    if (periodicX == DM_BOUNDARY_PERIODIC || periodicX == DM_BOUNDARY_TWIST ||
        periodicY == DM_BOUNDARY_PERIODIC || periodicY == DM_BOUNDARY_TWIST ||
        periodicZ == DM_BOUNDARY_PERIODIC || periodicZ == DM_BOUNDARY_TWIST) {
      PetscReal      L[3];
      PetscReal      maxCell[3];
      DMBoundaryType bdType[3];

      bdType[0] = periodicX;
      bdType[1] = periodicY;
      bdType[2] = periodicZ;
      for (i = 0; i < dim; i++) {
        L[i]       = upper[i] - lower[i];
        maxCell[i] = 1.1 * (L[i] / cells[i]);
      }

      ierr = DMSetPeriodicity(*dm,PETSC_TRUE,maxCell,L,bdType);CHKERRQ(ierr);
    }
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexSetOptionsPrefix - Sets the prefix used for searching for all DM options in the database.

  Logically Collective on DM

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

/*@
  DMPlexCreateHexCylinderMesh - Creates a mesh on the tensor product of the unit interval with the circle (cylinder) using hexahedra.

  Collective on MPI_Comm

  Input Parameters:
+ comm      - The communicator for the DM object
. numRefine - The number of regular refinements to the basic 5 cell structure
- periodicZ - The boundary type for the Z direction

  Output Parameter:
. dm  - The DM object

  Note: Here is the output numbering looking from the bottom of the cylinder:
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

.keywords: DM, create
.seealso: DMPlexCreateHexBoxMesh(), DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateHexCylinderMesh(MPI_Comm comm, PetscInt numRefine, DMBoundaryType periodicZ, DM *dm)
{
  const PetscInt dim = 3;
  PetscInt       numCells, numVertices, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  if (numRefine < 0) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of refinements %D cannot be negative", numRefine);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  /* Create topology */
  {
    PetscInt cone[8], c;

    numCells    = 5;
    numVertices = 16;
    if (periodicZ == DM_BOUNDARY_PERIODIC) {
      numCells   *= 3;
      numVertices = 24;
    }
    ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {ierr = DMPlexSetConeSize(*dm, c, 8);CHKERRQ(ierr);}
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    if (periodicZ == DM_BOUNDARY_PERIODIC) {
      cone[0] = 15; cone[1] = 18; cone[2] = 17; cone[3] = 16;
      cone[4] = 31; cone[5] = 32; cone[6] = 33; cone[7] = 34;
      ierr = DMPlexSetCone(*dm, 0, cone);CHKERRQ(ierr);
      cone[0] = 16; cone[1] = 17; cone[2] = 24; cone[3] = 23;
      cone[4] = 32; cone[5] = 36; cone[6] = 37; cone[7] = 33; /* 22 25 26 21 */
      ierr = DMPlexSetCone(*dm, 1, cone);CHKERRQ(ierr);
      cone[0] = 18; cone[1] = 27; cone[2] = 24; cone[3] = 17;
      cone[4] = 34; cone[5] = 33; cone[6] = 37; cone[7] = 38;
      ierr = DMPlexSetCone(*dm, 2, cone);CHKERRQ(ierr);
      cone[0] = 29; cone[1] = 27; cone[2] = 18; cone[3] = 15;
      cone[4] = 35; cone[5] = 31; cone[6] = 34; cone[7] = 38;
      ierr = DMPlexSetCone(*dm, 3, cone);CHKERRQ(ierr);
      cone[0] = 29; cone[1] = 15; cone[2] = 16; cone[3] = 23;
      cone[4] = 35; cone[5] = 36; cone[6] = 32; cone[7] = 31;
      ierr = DMPlexSetCone(*dm, 4, cone);CHKERRQ(ierr);

      cone[0] = 31; cone[1] = 34; cone[2] = 33; cone[3] = 32;
      cone[4] = 19; cone[5] = 22; cone[6] = 21; cone[7] = 20;
      ierr = DMPlexSetCone(*dm, 5, cone);CHKERRQ(ierr);
      cone[0] = 32; cone[1] = 33; cone[2] = 37; cone[3] = 36;
      cone[4] = 22; cone[5] = 25; cone[6] = 26; cone[7] = 21;
      ierr = DMPlexSetCone(*dm, 6, cone);CHKERRQ(ierr);
      cone[0] = 34; cone[1] = 38; cone[2] = 37; cone[3] = 33;
      cone[4] = 20; cone[5] = 21; cone[6] = 26; cone[7] = 28;
      ierr = DMPlexSetCone(*dm, 7, cone);CHKERRQ(ierr);
      cone[0] = 35; cone[1] = 38; cone[2] = 34; cone[3] = 31;
      cone[4] = 30; cone[5] = 19; cone[6] = 20; cone[7] = 28;
      ierr = DMPlexSetCone(*dm, 8, cone);CHKERRQ(ierr);
      cone[0] = 35; cone[1] = 31; cone[2] = 32; cone[3] = 36;
      cone[4] = 30; cone[5] = 25; cone[6] = 22; cone[7] = 19;
      ierr = DMPlexSetCone(*dm, 9, cone);CHKERRQ(ierr);

      cone[0] = 19; cone[1] = 20; cone[2] = 21; cone[3] = 22;
      cone[4] = 15; cone[5] = 16; cone[6] = 17; cone[7] = 18;
      ierr = DMPlexSetCone(*dm, 10, cone);CHKERRQ(ierr);
      cone[0] = 22; cone[1] = 21; cone[2] = 26; cone[3] = 25;
      cone[4] = 16; cone[5] = 23; cone[6] = 24; cone[7] = 17;
      ierr = DMPlexSetCone(*dm, 11, cone);CHKERRQ(ierr);
      cone[0] = 20; cone[1] = 28; cone[2] = 26; cone[3] = 21;
      cone[4] = 18; cone[5] = 17; cone[6] = 24; cone[7] = 27;
      ierr = DMPlexSetCone(*dm, 12, cone);CHKERRQ(ierr);
      cone[0] = 30; cone[1] = 28; cone[2] = 20; cone[3] = 19;
      cone[4] = 29; cone[5] = 15; cone[6] = 18; cone[7] = 27;
      ierr = DMPlexSetCone(*dm, 13, cone);CHKERRQ(ierr);
      cone[0] = 30; cone[1] = 19; cone[2] = 22; cone[3] = 25;
      cone[4] = 29; cone[5] = 23; cone[6] = 16; cone[7] = 15;
      ierr = DMPlexSetCone(*dm, 14, cone);CHKERRQ(ierr);
    } else {
      cone[0] =  5; cone[1] =  8; cone[2] =  7; cone[3] =  6;
      cone[4] =  9; cone[5] = 12; cone[6] = 11; cone[7] = 10;
      ierr = DMPlexSetCone(*dm, 0, cone);CHKERRQ(ierr);
      cone[0] =  6; cone[1] =  7; cone[2] = 14; cone[3] = 13;
      cone[4] = 12; cone[5] = 15; cone[6] = 16; cone[7] = 11;
      ierr = DMPlexSetCone(*dm, 1, cone);CHKERRQ(ierr);
      cone[0] =  8; cone[1] = 17; cone[2] = 14; cone[3] =  7;
      cone[4] = 10; cone[5] = 11; cone[6] = 16; cone[7] = 18;
      ierr = DMPlexSetCone(*dm, 2, cone);CHKERRQ(ierr);
      cone[0] = 19; cone[1] = 17; cone[2] =  8; cone[3] =  5;
      cone[4] = 20; cone[5] =  9; cone[6] = 10; cone[7] = 18;
      ierr = DMPlexSetCone(*dm, 3, cone);CHKERRQ(ierr);
      cone[0] = 19; cone[1] =  5; cone[2] =  6; cone[3] = 13;
      cone[4] = 20; cone[5] = 15; cone[6] = 12; cone[7] =  9;
      ierr = DMPlexSetCone(*dm, 4, cone);CHKERRQ(ierr);
    }
    ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  }
  /* Interpolate */
  {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
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
    ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
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
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
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
    ierr = DMSetPeriodicity(*dm, PETSC_TRUE, maxCell, L, bdType);CHKERRQ(ierr);
  }
  /* Refine topology */
  for (r = 0; r < numRefine; ++r) {
    DM rdm = NULL;

    ierr = DMRefine(*dm, comm, &rdm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = rdm;
  }
  /* Remap geometry to cylinder
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
  {
    Vec           coordinates;
    PetscSection  coordSection;
    PetscScalar  *coords;
    PetscInt      vStart, vEnd, v;
    const PetscReal dis = 1.0/PetscSqrtReal(2.0);
    const PetscReal ds2 = 0.5*dis;

    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscReal phi, sinp, cosp, dc, df, x, y, xc, yc;
      PetscInt  off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      if ((PetscAbsScalar(coords[off+0]) <= ds2) && (PetscAbsScalar(coords[off+1]) <= ds2)) continue;
      x    = PetscRealPart(coords[off]);
      y    = PetscRealPart(coords[off+1]);
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
      coords[off+0] = xc + (coords[off+0] - xc)*(1.0 - dc)/(df - dc);
      coords[off+1] = yc + (coords[off+1] - yc)*(1.0 - dc)/(df - dc);
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    if (periodicZ == DM_BOUNDARY_PERIODIC || periodicZ == DM_BOUNDARY_TWIST) {
      ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateWedgeCylinderMesh - Creates a mesh on the tensor product of the unit interval with the circle (cylinder) using wedges.

  Collective on MPI_Comm

  Input Parameters:
+ comm - The communicator for the DM object
. n    - The number of wedges around the origin
- interpolate - Create edges and faces

  Output Parameter:
. dm  - The DM object

  Level: beginner

.keywords: DM, create
.seealso: DMPlexCreateHexCylinderMesh(), DMPlexCreateHexBoxMesh(), DMPlexCreateBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateWedgeCylinderMesh(MPI_Comm comm, PetscInt n, PetscBool interpolate, DM *dm)
{
  const PetscInt dim = 3;
  PetscInt       numCells, numVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 3);
  if (n < 0) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Number of wedges %D cannot be negative", n);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  /* Create topology */
  {
    PetscInt cone[6], c;

    numCells    = n;
    numVertices = 2*(n+1);
    ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {ierr = DMPlexSetConeSize(*dm, c, 6);CHKERRQ(ierr);}
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for (c = 0; c < numCells; c++) {
      cone[0] =  c+n*1; cone[1] = (c+1)%n+n*1; cone[2] = 0+3*n;
      cone[3] =  c+n*2; cone[4] = (c+1)%n+n*2; cone[5] = 1+3*n;
      ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
    ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  }
  /* Interpolate */
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  /* Create cylinder geometry */
  {
    Vec          coordinates;
    PetscSection coordSection;
    PetscScalar *coords;
    PetscInt     coordSize, v, c;

    /* Build coordinates */
    ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
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
    coords[(2*n+0)*dim+0] = 0.0; coords[(2*n+0)*dim+1] = 0.0; coords[(2*n+0)*dim+2] = 1.0;
    coords[(2*n+1)*dim+0] = 0.0; coords[(2*n+1)*dim+1] = 0.0; coords[(2*n+1)*dim+2] = 0.0;
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal DiffNormReal(PetscInt dim, const PetscReal x[], const PetscReal y[])
{
  PetscReal prod = 0.0;
  PetscInt  i;
  for (i = 0; i < dim; ++i) prod += PetscSqr(x[i] - y[i]);
  return PetscSqrtReal(prod);
}
PETSC_STATIC_INLINE PetscReal DotReal(PetscInt dim, const PetscReal x[], const PetscReal y[])
{
  PetscReal prod = 0.0;
  PetscInt  i;
  for (i = 0; i < dim; ++i) prod += x[i]*y[i];
  return prod;
}

/*@
  DMPlexCreateSphereMesh - Creates a mesh on the d-dimensional sphere, S^d.

  Collective on MPI_Comm

  Input Parameters:
. comm  - The communicator for the DM object
. dim   - The dimension
- simplex - Use simplices, or tensor product cells

  Output Parameter:
. dm  - The DM object

  Level: beginner

.keywords: DM, create
.seealso: DMPlexCreateBoxMesh(), DMPlexCreateHexBoxMesh(), DMSetType(), DMCreate()
@*/
PetscErrorCode DMPlexCreateSphereMesh(MPI_Comm comm, PetscInt dim, PetscBool simplex, DM *dm)
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
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(*dm, dim+1);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) *dm), &rank);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (simplex) {
      DM              idm         = NULL;
      const PetscReal edgeLen     = 2.0/(1.0 + PETSC_PHI);
      const PetscReal vertex[3]   = {0.0, 1.0/(1.0 + PETSC_PHI), PETSC_PHI/(1.0 + PETSC_PHI)};
      const PetscInt  degree      = 5;
      PetscInt        s[3]        = {1, 1, 1};
      PetscInt        cone[3];
      PetscInt       *graph, p, i, j, k;

      numCells    = !rank ? 20 : 0;
      numVerts    = !rank ? 12 : 0;
      firstVertex = numCells;
      /* Use icosahedron, which for a unit sphere has coordinates which are all cyclic permutations of

           (0, \pm 1/\phi+1, \pm \phi/\phi+1)

         where \phi^2 - \phi - 1 = 0, meaning \phi is the golden ratio \frac{1 + \sqrt{5}}{2}. The edge
         length is then given by 2/\phi = 2 * 2.73606 = 5.47214.
       */
      /* Construct vertices */
      ierr = PetscCalloc1(numVerts * embedDim, &coordsIn);CHKERRQ(ierr);
      for (p = 0, i = 0; p < embedDim; ++p) {
        for (s[1] = -1; s[1] < 2; s[1] += 2) {
          for (s[2] = -1; s[2] < 2; s[2] += 2) {
            for (d = 0; d < embedDim; ++d) coordsIn[i*embedDim+d] = s[(d+p)%embedDim]*vertex[(d+p)%embedDim];
            ++i;
          }
        }
      }
      /* Construct graph */
      ierr = PetscCalloc1(numVerts * numVerts, &graph);CHKERRQ(ierr);
      for (i = 0; i < numVerts; ++i) {
        for (j = 0, k = 0; j < numVerts; ++j) {
          if (PetscAbsReal(DiffNormReal(embedDim, &coordsIn[i*embedDim], &coordsIn[j*embedDim]) - edgeLen) < PETSC_SMALL) {graph[i*numVerts+j] = 1; ++k;}
        }
        if (k != degree) SETERRQ3(comm, PETSC_ERR_PLIB, "Invalid icosahedron, vertex %D degree %D != %D", i, k, degree);
      }
      /* Build Topology */
      ierr = DMPlexSetChart(*dm, 0, numCells+numVerts);CHKERRQ(ierr);
      for (c = 0; c < numCells; c++) {
        ierr = DMPlexSetConeSize(*dm, c, embedDim);CHKERRQ(ierr);
      }
      ierr = DMSetUp(*dm);CHKERRQ(ierr); /* Allocate space for cones */
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
              ierr = DMPlexSetCone(*dm, c++, cone);CHKERRQ(ierr);
            }
          }
        }
      }
      ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
      ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
      ierr = PetscFree(graph);CHKERRQ(ierr);
      /* Interpolate mesh */
      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
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
      const PetscReal dist = 1.0/PetscSqrtReal(3.0);
      PetscInt        cone[4], ornt[4];

      numCells    = !rank ?  6 : 0;
      numEdges    = !rank ? 12 : 0;
      numVerts    = !rank ?  8 : 0;
      firstVertex = numCells;
      firstEdge   = numCells + numVerts;
      /* Build Topology */
      ierr = DMPlexSetChart(*dm, 0, numCells+numEdges+numVerts);CHKERRQ(ierr);
      for (c = 0; c < numCells; c++) {
        ierr = DMPlexSetConeSize(*dm, c, 4);CHKERRQ(ierr);
      }
      for (e = firstEdge; e < firstEdge+numEdges; ++e) {
        ierr = DMPlexSetConeSize(*dm, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(*dm);CHKERRQ(ierr); /* Allocate space for cones */
      /* Cell 0 */
      cone[0] = 14; cone[1] = 15; cone[2] = 16; cone[3] = 17;
      ierr = DMPlexSetCone(*dm, 0, cone);CHKERRQ(ierr);
      ornt[0] = 0; ornt[1] = 0; ornt[2] = 0; ornt[3] = 0;
      ierr = DMPlexSetConeOrientation(*dm, 0, ornt);CHKERRQ(ierr);
      /* Cell 1 */
      cone[0] = 18; cone[1] = 19; cone[2] = 14; cone[3] = 20;
      ierr = DMPlexSetCone(*dm, 1, cone);CHKERRQ(ierr);
      ornt[0] = 0; ornt[1] = 0; ornt[2] = -2; ornt[3] = 0;
      ierr = DMPlexSetConeOrientation(*dm, 1, ornt);CHKERRQ(ierr);
      /* Cell 2 */
      cone[0] = 21; cone[1] = 22; cone[2] = 18; cone[3] = 23;
      ierr = DMPlexSetCone(*dm, 2, cone);CHKERRQ(ierr);
      ornt[0] = 0; ornt[1] = 0; ornt[2] = -2; ornt[3] = 0;
      ierr = DMPlexSetConeOrientation(*dm, 2, ornt);CHKERRQ(ierr);
      /* Cell 3 */
      cone[0] = 19; cone[1] = 22; cone[2] = 24; cone[3] = 15;
      ierr = DMPlexSetCone(*dm, 3, cone);CHKERRQ(ierr);
      ornt[0] = -2; ornt[1] = -2; ornt[2] = 0; ornt[3] = -2;
      ierr = DMPlexSetConeOrientation(*dm, 3, ornt);CHKERRQ(ierr);
      /* Cell 4 */
      cone[0] = 16; cone[1] = 24; cone[2] = 21; cone[3] = 25;
      ierr = DMPlexSetCone(*dm, 4, cone);CHKERRQ(ierr);
      ornt[0] = -2; ornt[1] = -2; ornt[2] = -2; ornt[3] = 0;
      ierr = DMPlexSetConeOrientation(*dm, 4, ornt);CHKERRQ(ierr);
      /* Cell 5 */
      cone[0] = 20; cone[1] = 17; cone[2] = 25; cone[3] = 23;
      ierr = DMPlexSetCone(*dm, 5, cone);CHKERRQ(ierr);
      ornt[0] = -2; ornt[1] = -2; ornt[2] = -2; ornt[3] = -2;
      ierr = DMPlexSetConeOrientation(*dm, 5, ornt);CHKERRQ(ierr);
      /* Edges */
      cone[0] =  6; cone[1] =  7;
      ierr = DMPlexSetCone(*dm, 14, cone);CHKERRQ(ierr);
      cone[0] =  7; cone[1] =  8;
      ierr = DMPlexSetCone(*dm, 15, cone);CHKERRQ(ierr);
      cone[0] =  8; cone[1] =  9;
      ierr = DMPlexSetCone(*dm, 16, cone);CHKERRQ(ierr);
      cone[0] =  9; cone[1] =  6;
      ierr = DMPlexSetCone(*dm, 17, cone);CHKERRQ(ierr);
      cone[0] = 10; cone[1] = 11;
      ierr = DMPlexSetCone(*dm, 18, cone);CHKERRQ(ierr);
      cone[0] = 11; cone[1] =  7;
      ierr = DMPlexSetCone(*dm, 19, cone);CHKERRQ(ierr);
      cone[0] =  6; cone[1] = 10;
      ierr = DMPlexSetCone(*dm, 20, cone);CHKERRQ(ierr);
      cone[0] = 12; cone[1] = 13;
      ierr = DMPlexSetCone(*dm, 21, cone);CHKERRQ(ierr);
      cone[0] = 13; cone[1] = 11;
      ierr = DMPlexSetCone(*dm, 22, cone);CHKERRQ(ierr);
      cone[0] = 10; cone[1] = 12;
      ierr = DMPlexSetCone(*dm, 23, cone);CHKERRQ(ierr);
      cone[0] = 13; cone[1] =  8;
      ierr = DMPlexSetCone(*dm, 24, cone);CHKERRQ(ierr);
      cone[0] = 12; cone[1] =  9;
      ierr = DMPlexSetCone(*dm, 25, cone);CHKERRQ(ierr);
      ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
      ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
      /* Build coordinates */
      ierr = PetscCalloc1(numVerts * embedDim, &coordsIn);CHKERRQ(ierr);
      coordsIn[0*embedDim+0] = -dist; coordsIn[0*embedDim+1] =  dist; coordsIn[0*embedDim+2] = -dist;
      coordsIn[1*embedDim+0] =  dist; coordsIn[1*embedDim+1] =  dist; coordsIn[1*embedDim+2] = -dist;
      coordsIn[2*embedDim+0] =  dist; coordsIn[2*embedDim+1] = -dist; coordsIn[2*embedDim+2] = -dist;
      coordsIn[3*embedDim+0] = -dist; coordsIn[3*embedDim+1] = -dist; coordsIn[3*embedDim+2] = -dist;
      coordsIn[4*embedDim+0] = -dist; coordsIn[4*embedDim+1] =  dist; coordsIn[4*embedDim+2] =  dist;
      coordsIn[5*embedDim+0] =  dist; coordsIn[5*embedDim+1] =  dist; coordsIn[5*embedDim+2] =  dist;
      coordsIn[6*embedDim+0] = -dist; coordsIn[6*embedDim+1] = -dist; coordsIn[6*embedDim+2] =  dist;
      coordsIn[7*embedDim+0] =  dist; coordsIn[7*embedDim+1] = -dist; coordsIn[7*embedDim+2] =  dist;
    }
    break;
  case 3:
    if (simplex) {
      DM              idm             = NULL;
      const PetscReal edgeLen         = 1.0/PETSC_PHI;
      const PetscReal vertexA[4]      = {0.5, 0.5, 0.5, 0.5};
      const PetscReal vertexB[4]      = {1.0, 0.0, 0.0, 0.0};
      const PetscReal vertexC[4]      = {0.5, 0.5*PETSC_PHI, 0.5/PETSC_PHI, 0.0};
      const PetscInt  degree          = 12;
      PetscInt        s[4]            = {1, 1, 1};
      PetscInt        evenPerm[12][4] = {{0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 3, 2, 0},
                                         {2, 0, 1, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}};
      PetscInt        cone[4];
      PetscInt       *graph, p, i, j, k, l;

      numCells    = !rank ? 600 : 0;
      numVerts    = !rank ? 120 : 0;
      firstVertex = numCells;
      /* Use the 600-cell, which for a unit sphere has coordinates which are

           1/2 (\pm 1, \pm 1,    \pm 1, \pm 1)                          16
               (\pm 1,    0,       0,      0)  all cyclic permutations   8
           1/2 (\pm 1, \pm phi, \pm 1/phi, 0)  all even permutations    96

         where \phi^2 - \phi - 1 = 0, meaning \phi is the golden ratio \frac{1 + \sqrt{5}}{2}. The edge
         length is then given by 1/\phi = 2.73606.

         http://buzzard.pugetsound.edu/sage-practice/ch03s03.html
         http://mathworld.wolfram.com/600-Cell.html
       */
      /* Construct vertices */
      ierr = PetscCalloc1(numVerts * embedDim, &coordsIn);CHKERRQ(ierr);
      i    = 0;
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
      if (i != numVerts) SETERRQ2(comm, PETSC_ERR_PLIB, "Invalid 600-cell, vertices %D != %D", i, numVerts);
      /* Construct graph */
      ierr = PetscCalloc1(numVerts * numVerts, &graph);CHKERRQ(ierr);
      for (i = 0; i < numVerts; ++i) {
        for (j = 0, k = 0; j < numVerts; ++j) {
          if (PetscAbsReal(DiffNormReal(embedDim, &coordsIn[i*embedDim], &coordsIn[j*embedDim]) - edgeLen) < PETSC_SMALL) {graph[i*numVerts+j] = 1; ++k;}
        }
        if (k != degree) SETERRQ3(comm, PETSC_ERR_PLIB, "Invalid 600-cell, vertex %D degree %D != %D", i, k, degree);
      }
      /* Build Topology */
      ierr = DMPlexSetChart(*dm, 0, numCells+numVerts);CHKERRQ(ierr);
      for (c = 0; c < numCells; c++) {
        ierr = DMPlexSetConeSize(*dm, c, embedDim);CHKERRQ(ierr);
      }
      ierr = DMSetUp(*dm);CHKERRQ(ierr); /* Allocate space for cones */
      /* Cells */
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
                ierr = DMPlexSetCone(*dm, c++, cone);CHKERRQ(ierr);
              }
            }
          }
        }
      }
      ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
      ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
      ierr = PetscFree(graph);CHKERRQ(ierr);
      /* Interpolate mesh */
      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
      break;
    }
  default: SETERRQ1(comm, PETSC_ERR_SUP, "Unsupported dimension for sphere: %D", dim);
  }
  /* Create coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
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
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscFree(coordsIn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateInterpolation_Plex(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
extern PetscErrorCode DMCreateInjection_Plex(DM dmCoarse, DM dmFine, Mat *mat);
extern PetscErrorCode DMCreateDefaultSection_Plex(DM dm);
extern PetscErrorCode DMCreateDefaultConstraints_Plex(DM dm);
extern PetscErrorCode DMCreateMatrix_Plex(DM dm,  Mat *J);
extern PetscErrorCode DMCreateCoordinateDM_Plex(DM dm, DM *cdm);
PETSC_INTERN PetscErrorCode DMClone_Plex(DM dm, DM *newdm);
extern PetscErrorCode DMSetUp_Plex(DM dm);
extern PetscErrorCode DMDestroy_Plex(DM dm);
extern PetscErrorCode DMView_Plex(DM dm, PetscViewer viewer);
extern PetscErrorCode DMLoad_Plex(DM dm, PetscViewer viewer);
extern PetscErrorCode DMCreateSubDM_Plex(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm);
static PetscErrorCode DMInitialize_Plex(DM dm);

/* Replace dm with the contents of dmNew
   - Share the DM_Plex structure
   - Share the coordinates
   - Share the SF
*/
static PetscErrorCode DMPlexReplace_Static(DM dm, DM dmNew)
{
  PetscSF               sf;
  DM                    coordDM, coarseDM;
  Vec                   coords;
  PetscBool             isper;
  const PetscReal      *maxCell, *L;
  const DMBoundaryType *bd;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dmNew, &sf);CHKERRQ(ierr);
  ierr = DMSetPointSF(dm, sf);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmNew, &coordDM);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmNew, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dm, coordDM);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coords);CHKERRQ(ierr);
  ierr = DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(dmNew, isper, maxCell, L, bd);CHKERRQ(ierr);
  ierr = DMDestroy_Plex(dm);CHKERRQ(ierr);
  ierr = DMInitialize_Plex(dm);CHKERRQ(ierr);
  dm->data = dmNew->data;
  ((DM_Plex *) dmNew->data)->refct++;
  dmNew->labels->refct++;
  if (!--(dm->labels->refct)) {
    DMLabelLink next = dm->labels->next;

    /* destroy the labels */
    while (next) {
      DMLabelLink tmp = next->next;

      ierr = DMLabelDestroy(&next->label);CHKERRQ(ierr);
      ierr = PetscFree(next);CHKERRQ(ierr);
      next = tmp;
    }
    ierr = PetscFree(dm->labels);CHKERRQ(ierr);
  }
  dm->labels = dmNew->labels;
  dm->depthLabel = dmNew->depthLabel;
  ierr = DMGetCoarseDM(dmNew,&coarseDM);CHKERRQ(ierr);
  ierr = DMSetCoarseDM(dm,coarseDM);CHKERRQ(ierr);
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
  void            *tmp;
  DMLabelLinkList listTmp;
  DMLabel         depthTmp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
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

  tmp       = dmA->data;
  dmA->data = dmB->data;
  dmB->data = tmp;
  listTmp   = dmA->labels;
  dmA->labels = dmB->labels;
  dmB->labels = listTmp;
  depthTmp  = dmA->depthLabel;
  dmA->depthLabel = dmB->depthLabel;
  dmB->depthLabel = depthTmp;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetFromOptions_NonRefinement_Plex(PetscOptionItems *PetscOptionsObject,DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Handle viewing */
  ierr = PetscOptionsBool("-dm_plex_print_set_values", "Output all set values info", "DMView", PETSC_FALSE, &mesh->printSetValues, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_plex_print_fem", "Debug output level all fem computations", "DMView", 0, &mesh->printFEM, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_print_tol", "Tolerance for FEM output", "DMView", mesh->printTol, &mesh->printTol, NULL);CHKERRQ(ierr);
  /* Point Location */
  ierr = PetscOptionsBool("-dm_plex_hash_location", "Use grid hashing for point location", "DMView", PETSC_FALSE, &mesh->useHashLocation, NULL);CHKERRQ(ierr);
  /* Generation and remeshing */
  ierr = PetscOptionsBool("-dm_plex_remesh_bd", "Allow changes to the boundary on remeshing", "DMView", PETSC_FALSE, &mesh->remeshBd, NULL);CHKERRQ(ierr);
  /* Projection behavior */
  ierr = PetscOptionsInt("-dm_plex_max_projection_height", "Maxmimum mesh point height used to project locally", "DMPlexSetMaxProjectionHeight", 0, &mesh->maxProjectionHeight, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_regular_refinement", "Use special nested projection algorithm for regular refinement", "DMPlexSetRegularRefinement", mesh->regularRefinement, &mesh->regularRefinement, NULL);CHKERRQ(ierr);

  ierr = PetscPartitionerSetFromOptions(mesh->partitioner);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_Plex(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscInt       refine = 0, coarsen = 0, r;
  PetscBool      isHierarchy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Options");CHKERRQ(ierr);
  /* Handle DMPlex refinement */
  ierr = PetscOptionsInt("-dm_refine", "The number of uniform refinements", "DMCreate", refine, &refine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_refine_hierarchy", "The number of uniform refinements", "DMCreate", refine, &refine, &isHierarchy);CHKERRQ(ierr);
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
      DM refinedMesh;

      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      ierr = DMRefine(dm, PetscObjectComm((PetscObject) dm), &refinedMesh);CHKERRQ(ierr);
      /* Total hack since we do not pass in a pointer */
      ierr = DMPlexReplace_Static(dm, refinedMesh);CHKERRQ(ierr);
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      ierr = DMDestroy(&refinedMesh);CHKERRQ(ierr);
    }
  }
  /* Handle DMPlex coarsening */
  ierr = PetscOptionsInt("-dm_coarsen", "Coarsen the mesh", "DMCreate", coarsen, &coarsen, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_coarsen_hierarchy", "The number of coarsenings", "DMCreate", coarsen, &coarsen, &isHierarchy);CHKERRQ(ierr);
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
      DM coarseMesh;

      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      ierr = DMCoarsen(dm, PetscObjectComm((PetscObject) dm), &coarseMesh);CHKERRQ(ierr);
      /* Total hack since we do not pass in a pointer */
      ierr = DMPlexReplace_Static(dm, coarseMesh);CHKERRQ(ierr);
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject, dm);CHKERRQ(ierr);
      ierr = DMDestroy(&coarseMesh);CHKERRQ(ierr);
    }
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

static PetscErrorCode DMGetNeighors_Plex(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscSF        sf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetRanks(sf, nranks, ranks, NULL, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMInitialize_Plex(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm->ops->view                            = DMView_Plex;
  dm->ops->load                            = DMLoad_Plex;
  dm->ops->setfromoptions                  = DMSetFromOptions_Plex;
  dm->ops->clone                           = DMClone_Plex;
  dm->ops->setup                           = DMSetUp_Plex;
  dm->ops->createdefaultsection            = DMCreateDefaultSection_Plex;
  dm->ops->createdefaultconstraints        = DMCreateDefaultConstraints_Plex;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Plex;
  dm->ops->createlocalvector               = DMCreateLocalVector_Plex;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = DMCreateCoordinateDM_Plex;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = DMCreateMatrix_Plex;
  dm->ops->createinterpolation             = DMCreateInterpolation_Plex;
  dm->ops->getaggregates                   = NULL;
  dm->ops->getinjection                    = DMCreateInjection_Plex;
  dm->ops->refine                          = DMRefine_Plex;
  dm->ops->coarsen                         = DMCoarsen_Plex;
  dm->ops->refinehierarchy                 = DMRefineHierarchy_Plex;
  dm->ops->coarsenhierarchy                = DMCoarsenHierarchy_Plex;
  dm->ops->adaptlabel                      = DMAdaptLabel_Plex;
  dm->ops->adaptmetric                     = DMAdaptMetric_Plex;
  dm->ops->globaltolocalbegin              = NULL;
  dm->ops->globaltolocalend                = NULL;
  dm->ops->localtoglobalbegin              = NULL;
  dm->ops->localtoglobalend                = NULL;
  dm->ops->destroy                         = DMDestroy_Plex;
  dm->ops->createsubdm                     = DMCreateSubDM_Plex;
  dm->ops->getdimpoints                    = DMGetDimPoints_Plex;
  dm->ops->locatepoints                    = DMLocatePoints_Plex;
  dm->ops->projectfunctionlocal            = DMProjectFunctionLocal_Plex;
  dm->ops->projectfunctionlabellocal       = DMProjectFunctionLabelLocal_Plex;
  dm->ops->projectfieldlocal               = DMProjectFieldLocal_Plex;
  dm->ops->projectfieldlabellocal          = DMProjectFieldLabelLocal_Plex;
  dm->ops->computel2diff                   = DMComputeL2Diff_Plex;
  dm->ops->computel2gradientdiff           = DMComputeL2GradientDiff_Plex;
  dm->ops->computel2fielddiff              = DMComputeL2FieldDiff_Plex;
  dm->ops->getneighbors                    = DMGetNeighors_Plex;
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertBoundaryValues_C",DMPlexInsertBoundaryValues_Plex);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Plex);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: DMType, DMPlexCreate(), DMCreate(), DMSetType()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Plex(DM dm)
{
  DM_Plex        *mesh;
  PetscInt       unit, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&mesh);CHKERRQ(ierr);
  dm->dim  = 0;
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

  mesh->facesTmp = NULL;

  mesh->tetgenOpts   = NULL;
  mesh->triangleOpts = NULL;
  ierr = PetscPartitionerCreate(PetscObjectComm((PetscObject)dm), &mesh->partitioner);CHKERRQ(ierr);
  mesh->remeshBd     = PETSC_FALSE;

  mesh->subpointMap = NULL;

  for (unit = 0; unit < NUM_PETSC_UNITS; ++unit) mesh->scale[unit] = 1.0;

  mesh->regularRefinement   = PETSC_FALSE;
  mesh->depthState          = -1;
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
  for (d = 0; d < 8; ++d) mesh->hybridPointMax[d] = PETSC_DETERMINE;
  mesh->vtkCellHeight       = 0;
  mesh->useCone             = PETSC_FALSE;
  mesh->useClosure          = PETSC_TRUE;
  mesh->useAnchors          = PETSC_FALSE;

  mesh->maxProjectionHeight = 0;

  mesh->printSetValues = PETSC_FALSE;
  mesh->printFEM       = 0;
  mesh->printTol       = 1.0e-10;

  ierr = DMInitialize_Plex(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreate - Creates a DMPlex object, which encapsulates an unstructured mesh, or CW complex, which can be expressed using a Hasse Diagram.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMPlex object

  Output Parameter:
. mesh  - The DMPlex object

  Level: beginner

.keywords: DMPlex, create
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

/*
  This takes as input the common mesh generator output, a list of the vertices for each cell, but vertex numbers are global and an SF is built for them
*/
static PetscErrorCode DMPlexBuildFromCellList_Parallel_Private(DM dm, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, const int cells[], PetscSF *sfVert)
{
  PetscSF         sfPoint;
  PetscLayout     vLayout;
  PetscHashI      vhash;
  PetscSFNode    *remoteVerticesAdj, *vertexLocal, *vertexOwner, *remoteVertex;
  const PetscInt *vrange;
  PetscInt        numVerticesAdj, off, *verticesAdj, numVerticesGhost = 0, *localVertex, *cone, c, p, v, g;
  PETSC_UNUSED PetscHashIIter ret, iter;
  PetscMPIInt     rank, size;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  /* Partition vertices */
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject) dm), &vLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(vLayout, numVertices);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(vLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(vLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(vLayout, &vrange);CHKERRQ(ierr);
  /* Count vertices and map them to procs */
  PetscHashICreate(vhash);
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      PetscHashIPut(vhash, cells[c*numCorners+p], ret, iter);
    }
  }
  PetscHashISize(vhash, numVerticesAdj);
  ierr = PetscMalloc1(numVerticesAdj, &verticesAdj);CHKERRQ(ierr);
  off = 0; ierr = PetscHashIGetKeys(vhash, &off, verticesAdj);CHKERRQ(ierr);
  if (off != numVerticesAdj) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of local vertices %D should be %D", off, numVerticesAdj);
  ierr = PetscSortInt(numVerticesAdj, verticesAdj);CHKERRQ(ierr);
  ierr = PetscMalloc1(numVerticesAdj, &remoteVerticesAdj);CHKERRQ(ierr);
  for (v = 0; v < numVerticesAdj; ++v) {
    const PetscInt gv = verticesAdj[v];
    PetscInt       vrank;

    ierr = PetscFindInt(gv, size+1, vrange, &vrank);CHKERRQ(ierr);
    vrank = vrank < 0 ? -(vrank+2) : vrank;
    remoteVerticesAdj[v].index = gv - vrange[vrank];
    remoteVerticesAdj[v].rank  = vrank;
  }
  /* Create cones */
  ierr = DMPlexSetChart(dm, 0, numCells+numVerticesAdj);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {ierr = DMPlexSetConeSize(dm, c, numCorners);CHKERRQ(ierr);}
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numCorners, PETSC_INT, &cone);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      const PetscInt gv = cells[c*numCorners+p];
      PetscInt       lv;

      ierr = PetscFindInt(gv, numVerticesAdj, verticesAdj, &lv);CHKERRQ(ierr);
      if (lv < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not find global vertex %D in local connectivity", gv);
      cone[p] = lv+numCells;
    }
    ierr = DMPlexSetCone(dm, c, cone);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numCorners, PETSC_INT, &cone);CHKERRQ(ierr);
  /* Create SF for vertices */
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfVert);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sfVert, "Vertex Ownership SF");CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sfVert);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfVert, numVertices, numVerticesAdj, NULL, PETSC_OWN_POINTER, remoteVerticesAdj, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree(verticesAdj);CHKERRQ(ierr);
  /* Build pointSF */
  ierr = PetscMalloc2(numVerticesAdj, &vertexLocal, numVertices, &vertexOwner);CHKERRQ(ierr);
  for (v = 0; v < numVerticesAdj; ++v) {vertexLocal[v].index = v+numCells; vertexLocal[v].rank = rank;}
  for (v = 0; v < numVertices;    ++v) {vertexOwner[v].index = -1;         vertexOwner[v].rank = -1;}
  ierr = PetscSFReduceBegin(*sfVert, MPIU_2INT, vertexLocal, vertexOwner, MPI_MAXLOC);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(*sfVert, MPIU_2INT, vertexLocal, vertexOwner, MPI_MAXLOC);CHKERRQ(ierr);
  for (v = 0; v < numVertices;    ++v) if (vertexOwner[v].rank < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Global vertex %d on rank %d was unclaimed", v + vrange[rank], rank);
  ierr = PetscSFBcastBegin(*sfVert, MPIU_2INT, vertexOwner, vertexLocal);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(*sfVert, MPIU_2INT, vertexOwner, vertexLocal);CHKERRQ(ierr);
  for (v = 0; v < numVerticesAdj; ++v) if (vertexLocal[v].rank != rank) ++numVerticesGhost;
  ierr = PetscMalloc1(numVerticesGhost, &localVertex);CHKERRQ(ierr);
  ierr = PetscMalloc1(numVerticesGhost, &remoteVertex);CHKERRQ(ierr);
  for (v = 0, g = 0; v < numVerticesAdj; ++v) {
    if (vertexLocal[v].rank != rank) {
      localVertex[g]        = v+numCells;
      remoteVertex[g].index = vertexLocal[v].index;
      remoteVertex[g].rank  = vertexLocal[v].rank;
      ++g;
    }
  }
  ierr = PetscFree2(vertexLocal, vertexOwner);CHKERRQ(ierr);
  if (g != numVerticesGhost) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of vertex ghosts %D should be %D", g, numVerticesGhost);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) sfPoint, "point SF");CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfPoint, numCells+numVerticesAdj, numVerticesGhost, localVertex, PETSC_OWN_POINTER, remoteVertex, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&vLayout);CHKERRQ(ierr);
  PetscHashIDestroy(vhash);
  /* Fill in the rest of the topology structure */
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This takes as input the coordinates for each owned vertex
*/
static PetscErrorCode DMPlexBuildCoordinates_Parallel_Private(DM dm, PetscInt spaceDim, PetscInt numCells, PetscInt numV, PetscSF sfVert, const PetscReal vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       numVertices, numVerticesAdj, coordSize, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetCoordinateDim(dm, spaceDim);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfVert, &numVertices, &numVerticesAdj, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVerticesAdj);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVerticesAdj; ++v) {
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
    ierr = MPI_Type_contiguous(spaceDim, MPIU_SCALAR, &coordtype);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&coordtype);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    {
    PetscScalar *svertexCoords;
    PetscInt    i;
    ierr = PetscMalloc1(numV*spaceDim,&svertexCoords);CHKERRQ(ierr);
    for (i=0; i<numV*spaceDim; i++) svertexCoords[i] = vertexCoords[i];
    ierr = PetscSFBcastBegin(sfVert, coordtype, svertexCoords, coords);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfVert, coordtype, svertexCoords, coords);CHKERRQ(ierr);
    ierr = PetscFree(svertexCoords);CHKERRQ(ierr);
    }
#else
    ierr = PetscSFBcastBegin(sfVert, coordtype, vertexCoords, coords);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfVert, coordtype, vertexCoords, coords);CHKERRQ(ierr);
#endif
    ierr = MPI_Type_free(&coordtype);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateFromCellListParallel - This takes as input common mesh generator output, a list of the vertices for each cell, and produces a DM

  Input Parameters:
+ comm - The communicator
. dim - The topological dimension of the mesh
. numCells - The number of cells owned by this process
. numVertices - The number of vertices owned by this process
. numCorners - The number of vertices for each cell
. interpolate - Flag indicating that intermediate mesh entities (faces, edges) should be created automatically
. cells - An array of numCells*numCorners numbers, the global vertex numbers for each cell
. spaceDim - The spatial dimension used for coordinates
- vertexCoords - An array of numVertices*spaceDim numbers, the coordinates of each vertex

  Output Parameter:
+ dm - The DM
- vertexSF - Optional, SF describing complete vertex ownership

  Note: Two triangles sharing a face
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

  Level: beginner

.seealso: DMPlexCreateFromCellList(), DMPlexCreateFromDAG(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFromCellListParallel(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, PetscBool interpolate, const int cells[], PetscInt spaceDim, const PetscReal vertexCoords[], PetscSF *vertexSF, DM *dm)
{
  PetscSF        sfVert;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(*dm, dim, 2);
  PetscValidLogicalCollectiveInt(*dm, spaceDim, 8);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexBuildFromCellList_Parallel_Private(*dm, numCells, numVertices, numCorners, cells, &sfVert);CHKERRQ(ierr);
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  ierr = DMPlexBuildCoordinates_Parallel_Private(*dm, spaceDim, numCells, numVertices,sfVert, vertexCoords);CHKERRQ(ierr);
  if (vertexSF) *vertexSF = sfVert;
  else {ierr = PetscSFDestroy(&sfVert);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  This takes as input the common mesh generator output, a list of the vertices for each cell
*/
static PetscErrorCode DMPlexBuildFromCellList_Private(DM dm, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, const int cells[])
{
  PetscInt      *cone, c, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexSetChart(dm, 0, numCells+numVertices);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    ierr = DMPlexSetConeSize(dm, c, numCorners);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numCorners, PETSC_INT, &cone);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    for (p = 0; p < numCorners; ++p) {
      cone[p] = cells[c*numCorners+p]+numCells;
    }
    ierr = DMPlexSetCone(dm, c, cone);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numCorners, PETSC_INT, &cone);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This takes as input the coordinates for each vertex
*/
static PetscErrorCode DMPlexBuildCoordinates_Private(DM dm, PetscInt spaceDim, PetscInt numCells, PetscInt numVertices, const double vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  DM             cdm;
  PetscScalar   *coords;
  PetscInt       v, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetCoordinateDim(dm, spaceDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);

  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(cdm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, spaceDim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < numVertices; ++v) {
    for (d = 0; d < spaceDim; ++d) {
      coords[v*spaceDim+d] = vertexCoords[v*spaceDim+d];
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateFromCellList - This takes as input common mesh generator output, a list of the vertices for each cell, and produces a DM

  Input Parameters:
+ comm - The communicator
. dim - The topological dimension of the mesh
. numCells - The number of cells
. numVertices - The number of vertices
. numCorners - The number of vertices for each cell
. interpolate - Flag indicating that intermediate mesh entities (faces, edges) should be created automatically
. cells - An array of numCells*numCorners numbers, the vertices for each cell
. spaceDim - The spatial dimension used for coordinates
- vertexCoords - An array of numVertices*spaceDim numbers, the coordinates of each vertex

  Output Parameter:
. dm - The DM

  Note: Two triangles sharing a face
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

  Level: beginner

.seealso: DMPlexCreateFromDAG(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFromCellList(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numVertices, PetscInt numCorners, PetscBool interpolate, const int cells[], PetscInt spaceDim, const double vertexCoords[], DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexBuildFromCellList_Private(*dm, numCells, numVertices, numCorners, cells);CHKERRQ(ierr);
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  ierr = DMPlexBuildCoordinates_Private(*dm, spaceDim, numCells, numVertices, vertexCoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateFromDAG - This takes as input the adjacency-list representation of the Directed Acyclic Graph (Hasse Diagram) encoding a mesh, and produces a DM

  Input Parameters:
+ dm - The empty DM object, usually from DMCreate() and DMSetDimension()
. depth - The depth of the DAG
. numPoints - The number of points at each depth
. coneSize - The cone size of each point
. cones - The concatenation of the cone points for each point, the cone list must be oriented correctly for each point
. coneOrientations - The orientation of each cone point
- vertexCoords - An array of numVertices*dim numbers, the coordinates of each vertex

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
$ Notice that all points are numbered consecutively, unlikely DMPlexCreateFromCellList()

  Level: advanced

.seealso: DMPlexCreateFromCellList(), DMPlexCreate()
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
  if (dimEmbed < dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Embedding dimension %d cannot be less than intrinsic dimension %d",dimEmbed,dim);
  for (d = 0; d <= depth; ++d) pEnd += numPoints[d];
  ierr = DMPlexSetChart(dm, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, p, coneSize[p-pStart]);CHKERRQ(ierr);
    if (firstVertex < 0 && !coneSize[p - pStart]) {
      firstVertex = p - pStart;
    }
  }
  if (firstVertex < 0 && numPoints[0]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Expected %d vertices but could not find any", numPoints[0]);
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
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = 0; v < numPoints[0]; ++v) {
    PetscInt off;

    ierr = PetscSectionGetOffset(coordSection, v+firstVertex, &off);CHKERRQ(ierr);
    for (d = 0; d < dimEmbed; ++d) {
      coords[off+d] = vertexCoords[v*dimEmbed+d];
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateFromFile - This takes a filename and produces a DM

  Input Parameters:
+ comm - The communicator
. filename - A file name
- interpolate - Flag to create intermediate mesh pieces (edges, faces)

  Output Parameter:
. dm - The DM

  Level: beginner

.seealso: DMPlexCreateFromDAG(), DMPlexCreateFromCellList(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  const char    *extGmsh    = ".msh";
  const char    *extCGNS    = ".cgns";
  const char    *extExodus  = ".exo";
  const char    *extGenesis = ".gen";
  const char    *extFluent  = ".cas";
  const char    *extHDF5    = ".h5";
  const char    *extMed     = ".med";
  const char    *extPLY     = ".ply";
  size_t         len;
  PetscBool      isGmsh, isCGNS, isExodus, isGenesis, isFluent, isHDF5, isMed, isPLY;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(filename, 2);
  PetscValidPointer(dm, 4);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Filename must be a valid path");
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extGmsh,    4, &isGmsh);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-5)], extCGNS,    5, &isCGNS);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extExodus,  4, &isExodus);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extGenesis, 4, &isGenesis);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extFluent,  4, &isFluent);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-3)], extHDF5,    3, &isHDF5);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extMed,     4, &isMed);CHKERRQ(ierr);
  ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extPLY,     4, &isPLY);CHKERRQ(ierr);
  if (isGmsh) {
    ierr = DMPlexCreateGmshFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isCGNS) {
    ierr = DMPlexCreateCGNSFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isExodus || isGenesis) {
    ierr = DMPlexCreateExodusFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isFluent) {
    ierr = DMPlexCreateFluentFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isHDF5) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = DMCreate(comm, dm);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMLoad(*dm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else if (isMed) {
    ierr = DMPlexCreateMedFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else if (isPLY) {
    ierr = DMPlexCreatePLYFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateReferenceCell - Create a DMPLEX with the appropriate FEM reference cell

  Collective on comm

  Input Parameters:
+ comm    - The communicator
. dim     - The spatial dimension
- simplex - Flag for simplex, otherwise use a tensor-product cell

  Output Parameter:
. refdm - The reference cell

  Level: intermediate

.keywords: reference cell
.seealso:
@*/
PetscErrorCode DMPlexCreateReferenceCell(MPI_Comm comm, PetscInt dim, PetscBool simplex, DM *refdm)
{
  DM             rdm;
  Vec            coords;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, &rdm);CHKERRQ(ierr);
  ierr = DMSetType(rdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(rdm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 0:
  {
    PetscInt    numPoints[1]        = {1};
    PetscInt    coneSize[1]         = {0};
    PetscInt    cones[1]            = {0};
    PetscInt    coneOrientations[1] = {0};
    PetscScalar vertexCoords[1]     = {0.0};

    ierr = DMPlexCreateFromDAG(rdm, 0, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  }
  break;
  case 1:
  {
    PetscInt    numPoints[2]        = {2, 1};
    PetscInt    coneSize[3]         = {2, 0, 0};
    PetscInt    cones[2]            = {1, 2};
    PetscInt    coneOrientations[2] = {0, 0};
    PetscScalar vertexCoords[2]     = {-1.0,  1.0};

    ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  }
  break;
  case 2:
    if (simplex) {
      PetscInt    numPoints[2]        = {3, 1};
      PetscInt    coneSize[4]         = {3, 0, 0, 0};
      PetscInt    cones[3]            = {1, 2, 3};
      PetscInt    coneOrientations[3] = {0, 0, 0};
      PetscScalar vertexCoords[6]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    } else {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 2, 3, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
  break;
  case 3:
    if (simplex) {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {1, 3, 2, 4};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  -1.0, -1.0, 1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    } else {
      PetscInt    numPoints[2]        = {8, 1};
      PetscInt    coneSize[9]         = {8, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {1, 4, 3, 2, 5, 6, 7, 8};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, 1.0, -1.0,  -1.0, 1.0, -1.0,
                                         -1.0, -1.0,  1.0,  1.0, -1.0,  1.0,  1.0, 1.0,  1.0,  -1.0, 1.0,  1.0};

      ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
  break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Cannot create reference cell for dimension %d", dim);
  }
  *refdm = NULL;
  ierr = DMPlexInterpolate(rdm, refdm);CHKERRQ(ierr);
  if (rdm->coordinateDM) {
    DM           ncdm;
    PetscSection cs;
    PetscInt     pEnd = -1;

    ierr = DMGetDefaultSection(rdm->coordinateDM, &cs);CHKERRQ(ierr);
    if (cs) {ierr = PetscSectionGetChart(cs, NULL, &pEnd);CHKERRQ(ierr);}
    if (pEnd >= 0) {
      ierr = DMClone(rdm->coordinateDM, &ncdm);CHKERRQ(ierr);
      ierr = DMSetDefaultSection(ncdm, cs);CHKERRQ(ierr);
      ierr = DMSetCoordinateDM(*refdm, ncdm);CHKERRQ(ierr);
      ierr = DMDestroy(&ncdm);CHKERRQ(ierr);
    }
  }
  ierr = DMGetCoordinatesLocal(rdm, &coords);CHKERRQ(ierr);
  if (coords) {
    ierr = DMSetCoordinatesLocal(*refdm, coords);CHKERRQ(ierr);
  } else {
    ierr = DMGetCoordinates(rdm, &coords);CHKERRQ(ierr);
    if (coords) {ierr = DMSetCoordinates(*refdm, coords);CHKERRQ(ierr);}
  }
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
