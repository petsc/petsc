#define PETSCDM_DLL
#include <petsc-private/pleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Plex"
PetscErrorCode  DMSetFromOptions_Plex(DM dm)
{
  DM_Plex    *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMPlex Options");CHKERRQ(ierr);
    /* Handle DMPlex refinement */
    /* Handle associated vectors */
    /* Handle viewing */
    ierr = PetscOptionsBool("-dm_plex_print_set_values", "Output all set values info", "DMView", PETSC_FALSE, &mesh->printSetValues, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dm_plex_print_fem", "Debug output level all fem computations", "DMView", 0, &mesh->printFEM, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSquareBoundary"
/*
 Simple square boundary:

 18--5-17--4--16
  |     |     |
  6    10     3
  |     |     |
 19-11-20--9--15
  |     |     |
  7     8     2
  |     |     |
 12--0-13--1--14
*/
PetscErrorCode DMPlexCreateSquareBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  PetscInt       numVertices = (edges[0]+1)*(edges[1]+1);
  PetscInt       numEdges    = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
  PetscInt       markerTop      = 1;
  PetscInt       markerBottom   = 1;
  PetscInt       markerRight    = 1;
  PetscInt       markerLeft     = 1;
  PetscBool      markerSeparate = PETSC_FALSE;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, PETSC_NULL);CHKERRQ(ierr);
  if (markerSeparate) {
    markerTop    = 1;
    markerBottom = 0;
    markerRight  = 0;
    markerLeft   = 0;
  }
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt e, ex, ey;

    ierr = DMPlexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
    for (e = 0; e < numEdges; ++e) {
      ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for (vx = 0; vx <= edges[0]; vx++) {
      for (ey = 0; ey < edges[1]; ey++) {
        PetscInt edge    = vx*edges[1] + ey + edges[0]*(edges[1]+1);
        PetscInt vertex  = ey*(edges[0]+1) + vx + numEdges;
        PetscInt cone[2] = {vertex, vertex+edges[0]+1};

        ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vx == edges[0]) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerRight);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerRight);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerRight);CHKERRQ(ierr);
          }
        } else if (vx == 0) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerLeft);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerLeft);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerLeft);CHKERRQ(ierr);
          }
        }
      }
    }
    for (vy = 0; vy <= edges[1]; vy++) {
      for (ex = 0; ex < edges[0]; ex++) {
        PetscInt edge    = vy*edges[0]     + ex;
        PetscInt vertex  = vy*(edges[0]+1) + ex + numEdges;
        PetscInt cone[2] = {vertex, vertex+1};

        ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vy == edges[1]) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerTop);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerTop);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerTop);CHKERRQ(ierr);
          }
        } else if (vy == 0) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerBottom);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerBottom);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerBottom);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numEdges, numEdges + numVertices);CHKERRQ(ierr);
  for (v = numEdges; v < numEdges+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateCubeBoundary"
/*
 Simple cubic boundary:

     2-------3
    /|      /|
   6-------7 |
   | |     | |
   | 0-----|-1
   |/      |/
   4-------5
*/
PetscErrorCode DMPlexCreateCubeBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
{
  PetscInt       numVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
  PetscInt       numFaces    = 6;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy, vz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((faces[0] < 1) || (faces[1] < 1) || (faces[2] < 1)) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Must have at least 1 face per side");
  if ((faces[0] > 1) || (faces[1] > 1) || (faces[2] > 1)) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Currently can't handle more than 1 face per side");
  ierr = PetscMalloc(numVertices*2 * sizeof(PetscReal), &coords);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt f;

    ierr = DMPlexSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMPlexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for (v = 0; v < numFaces+numVertices; ++v) {
      ierr = DMPlexSetLabelValue(dm, "marker", v, 1);CHKERRQ(ierr);
    }
    { /* Side 0 (Front) */
      PetscInt cone[4] = {numFaces+4, numFaces+5, numFaces+7, numFaces+6};
      ierr = DMPlexSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { /* Side 1 (Back) */
      PetscInt cone[4] = {numFaces+1, numFaces+0, numFaces+2, numFaces+3};
      ierr = DMPlexSetCone(dm, 1, cone);CHKERRQ(ierr);
    }
    { /* Side 2 (Bottom) */
      PetscInt cone[4] = {numFaces+0, numFaces+1, numFaces+5, numFaces+4};
      ierr = DMPlexSetCone(dm, 2, cone);CHKERRQ(ierr);
    }
    { /* Side 3 (Top) */
      PetscInt cone[4] = {numFaces+6, numFaces+7, numFaces+3, numFaces+2};
      ierr = DMPlexSetCone(dm, 3, cone);CHKERRQ(ierr);
    }
    { /* Side 4 (Left) */
      PetscInt cone[4] = {numFaces+0, numFaces+4, numFaces+6, numFaces+2};
      ierr = DMPlexSetCone(dm, 4, cone);CHKERRQ(ierr);
    }
    { /* Side 5 (Right) */
      PetscInt cone[4] = {numFaces+5, numFaces+1, numFaces+3, numFaces+7};
      ierr = DMPlexSetCone(dm, 5, cone);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  for (v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 3);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSquareMesh"
/*
 Simple square mesh:

 22--8-23--9--24
  |     |     |
 13  2 14  3  15
  |     |     |
 19--6-20--7--21
  |     |     |
 10  0 11  1 12
  |     |     |
 16--4-17--5--18
*/
PetscErrorCode DMPlexCreateSquareMesh(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  PetscInt       markerTop      = 1;
  PetscInt       markerBottom   = 1;
  PetscInt       markerRight    = 1;
  PetscInt       markerLeft     = 1;
  PetscBool      markerSeparate = PETSC_FALSE;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject) dm)->prefix, "-dm_plex_separate_marker", &markerSeparate, PETSC_NULL);CHKERRQ(ierr);
  if (markerSeparate) {
    markerTop    = 3;
    markerBottom = 1;
    markerRight  = 2;
    markerLeft   = 4;
  }
  {
    const PetscInt numXEdges    = !rank ? edges[0]   : 0;
    const PetscInt numYEdges    = !rank ? edges[1]   : 0;
    const PetscInt numXVertices = !rank ? edges[0]+1 : 0;
    const PetscInt numYVertices = !rank ? edges[1]+1 : 0;
    const PetscInt numTotXEdges = numXEdges*numYVertices;
    const PetscInt numTotYEdges = numYEdges*numXVertices;
    const PetscInt numVertices  = numXVertices*numYVertices;
    const PetscInt numEdges     = numTotXEdges + numTotYEdges;
    const PetscInt numFaces     = numXEdges*numYEdges;
    const PetscInt firstVertex  = numFaces;
    const PetscInt firstXEdge   = numFaces + numVertices;
    const PetscInt firstYEdge   = numFaces + numVertices + numTotXEdges;
    Vec            coordinates;
    PetscSection   coordSection;
    PetscScalar   *coords;
    PetscInt       coordSize;
    PetscInt       v, vx, vy;
    PetscInt       f, fx, fy, e, ex, ey;

    ierr = DMPlexSetChart(dm, 0, numFaces+numEdges+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMPlexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    for (e = firstXEdge; e < firstXEdge+numEdges; ++e) {
      ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    /* Build faces */
    for (fy = 0; fy < numYEdges; fy++) {
      for (fx = 0; fx < numXEdges; fx++) {
        const PetscInt face    = fy*numXEdges + fx;
        const PetscInt edgeL   = firstYEdge + fx*numYEdges + fy;
        const PetscInt edgeB   = firstXEdge + fy*numXEdges + fx;
        const PetscInt cone[4] = {edgeB, edgeL+numYEdges, edgeB+numXEdges, edgeL};
        const PetscInt ornt[4] = {0,     0,               -2,              -2};

        ierr = DMPlexSetCone(dm, face, cone);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(dm, face, ornt);CHKERRQ(ierr);
      }
    }
    /* Build Y edges*/
    for (vx = 0; vx < numXVertices; vx++) {
      for (ey = 0; ey < numYEdges; ey++) {
        const PetscInt edge    = firstYEdge  + vx*numYEdges + ey;
        const PetscInt vertex  = firstVertex + ey*numXVertices + vx;
        const PetscInt cone[2] = {vertex, vertex+numXVertices};

        ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vx == numXVertices-1) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerRight);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerRight);CHKERRQ(ierr);
          if (ey == numYEdges-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerRight);CHKERRQ(ierr);
          }
        } else if (vx == 0) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerLeft);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerLeft);CHKERRQ(ierr);
          if (ey == numYEdges-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerLeft);CHKERRQ(ierr);
          }
        }
      }
    }
    /* Build X edges*/
    for (vy = 0; vy < numYVertices; vy++) {
      for (ex = 0; ex < numXEdges; ex++) {
        const PetscInt edge    = firstXEdge  + vy*numXEdges + ex;
        const PetscInt vertex  = firstVertex + vy*numXVertices + ex;
        const PetscInt cone[2] = {vertex, vertex+1};

        ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vy == numYVertices-1) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerTop);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerTop);CHKERRQ(ierr);
          if (ex == numXEdges-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerTop);CHKERRQ(ierr);
          }
        } else if (vy == 0) {
          ierr = DMPlexSetLabelValue(dm, "marker", edge,    markerBottom);CHKERRQ(ierr);
          ierr = DMPlexSetLabelValue(dm, "marker", cone[0], markerBottom);CHKERRQ(ierr);
          if (ex == numXEdges-1) {
            ierr = DMPlexSetLabelValue(dm, "marker", cone[1], markerBottom);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
    ierr = DMPlexStratify(dm);CHKERRQ(ierr);
    /* Build coordinates */
    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVertices);CHKERRQ(ierr);
    for (v = firstVertex; v < firstVertex+numVertices; ++v) {
      ierr = PetscSectionSetDof(coordSection, v, 2);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(((PetscObject) dm)->comm, &coordinates);CHKERRQ(ierr);
    ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (vy = 0; vy < numYVertices; ++vy) {
      for (vx = 0; vx < numXVertices; ++vx) {
        coords[(vy*numXVertices+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/numXEdges)*vx;
        coords[(vy*numXVertices+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/numYEdges)*vy;
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateBoxMesh"
PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool interpolate, DM *dm)
{
  DM             boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(boundary,dim,2);
  ierr = DMSetType(boundary, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(boundary, dim-1);CHKERRQ(ierr);
  switch(dim) {
  case 2:
  {
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};
    PetscInt  edges[2] = {2, 2};

    ierr = DMPlexCreateSquareBoundary(boundary, lower, upper, edges);CHKERRQ(ierr);
    break;
  }
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3] = {1, 1, 1};

    ierr = DMPlexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  ierr = DMPlexGenerate(boundary, PETSC_NULL, interpolate, dm);CHKERRQ(ierr);
  ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateHexBoxMesh"
PetscErrorCode DMPlexCreateHexBoxMesh(MPI_Comm comm, PetscInt dim, const PetscInt cells[], DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(*dm,dim,2);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*dm, dim);CHKERRQ(ierr);
  switch(dim) {
  case 2:
  {
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};

    ierr = DMPlexCreateSquareMesh(*dm, lower, upper, cells);CHKERRQ(ierr);
    break;
  }
#if 0
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};

    ierr = DMPlexCreateCubeMesh(boundary, lower, upper, cells);CHKERRQ(ierr);
    break;
  }
#endif
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateInterpolation_Plex(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
extern PetscErrorCode DMCreateMatrix_Plex(DM dm, MatType mtype, Mat *J);
extern PetscErrorCode DMCreateCoordinateDM_Plex(DM dm, DM *cdm);
extern PetscErrorCode DMRefine_Plex(DM dm, MPI_Comm comm, DM *dmRefined);
extern PetscErrorCode DMSetUp_Plex(DM dm);
extern PetscErrorCode DMDestroy_Plex(DM dm);
extern PetscErrorCode DMView_Plex(DM dm, PetscViewer viewer);
extern PetscErrorCode DMCreateSubDM_Plex(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm);
extern PetscErrorCode DMLocatePoints_Plex(DM dm, Vec v, IS *cellIS);

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Plex"
static PetscErrorCode DMCreateGlobalVector_Plex(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector_Section_Private(dm,vec);CHKERRQ(ierr);
  /* ierr = VecSetOperation(*vec, VECOP_DUPLICATE, (void(*)(void)) VecDuplicate_MPI_DM);CHKERRQ(ierr); */
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void(*)(void)) VecView_Plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Plex"
static PetscErrorCode DMCreateLocalVector_Plex(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector_Section_Private(dm,vec);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void(*)(void)) VecView_Plex_Local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Plex"
PetscErrorCode DMInitialize_Plex(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(VECSTANDARD, (char**)&dm->vectype);CHKERRQ(ierr);
  dm->ops->view               = DMView_Plex;
  dm->ops->setfromoptions     = DMSetFromOptions_Plex;
  dm->ops->setup              = DMSetUp_Plex;
  dm->ops->createglobalvector = DMCreateGlobalVector_Plex;
  dm->ops->createlocalvector  = DMCreateLocalVector_Plex;
  dm->ops->createlocaltoglobalmapping      = PETSC_NULL;
  dm->ops->createlocaltoglobalmappingblock = PETSC_NULL;
  dm->ops->createfieldis      = PETSC_NULL;
  dm->ops->createcoordinatedm = DMCreateCoordinateDM_Plex;
  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix       = DMCreateMatrix_Plex;
  dm->ops->createinterpolation= 0;
  dm->ops->getaggregates      = 0;
  dm->ops->getinjection       = 0;
  dm->ops->refine             = DMRefine_Plex;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = 0;
  dm->ops->globaltolocalbegin = PETSC_NULL;
  dm->ops->globaltolocalend   = PETSC_NULL;
  dm->ops->localtoglobalbegin = PETSC_NULL;
  dm->ops->localtoglobalend   = PETSC_NULL;
  dm->ops->destroy            = DMDestroy_Plex;
  dm->ops->createsubdm        = DMCreateSubDM_Plex;
  dm->ops->locatepoints       = DMLocatePoints_Plex;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Plex"
PetscErrorCode DMCreate_Plex(DM dm)
{
  DM_Plex       *mesh;
  PetscInt       unit, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Plex, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  mesh->refct             = 1;
  mesh->dim               = 0;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->coneSection);CHKERRQ(ierr);
  mesh->maxConeSize       = 0;
  mesh->cones             = PETSC_NULL;
  mesh->coneOrientations  = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->supportSection);CHKERRQ(ierr);
  mesh->maxSupportSize    = 0;
  mesh->supports          = PETSC_NULL;
  mesh->refinementUniform = PETSC_TRUE;
  mesh->refinementLimit   = -1.0;

  mesh->facesTmp         = PETSC_NULL;

  mesh->subpointMap      = PETSC_NULL;

  for(unit = 0; unit < NUM_PETSC_UNITS; ++unit) {
    mesh->scale[unit]    = 1.0;
  }

  mesh->labels               = PETSC_NULL;
  mesh->globalVertexNumbers  = PETSC_NULL;
  mesh->globalCellNumbers    = PETSC_NULL;
  for(d = 0; d < 8; ++d) {
    mesh->hybridPointMax[d]  = PETSC_DETERMINE;
  }
  mesh->vtkCellHeight        = 0;
  mesh->preallocCenterDim    = -1;

  mesh->integrateResidualFEM       = PETSC_NULL;
  mesh->integrateJacobianActionFEM = PETSC_NULL;
  mesh->integrateJacobianFEM       = PETSC_NULL;

  mesh->printSetValues       = PETSC_FALSE;
  mesh->printFEM             = 0;

  ierr = DMInitialize_Plex(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreate"
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexClone"
/*@
  DMPlexClone - Creates a DMPlex object with the same mesh as the original.

  Collective on MPI_Comm

  Input Parameter:
. dm - The original DMPlex object

  Output Parameter:
. newdm  - The new DMPlex object

  Level: beginner

.keywords: DMPlex, create
@*/
PetscErrorCode DMPlexClone(DM dm, DM *newdm)
{
  DM_Plex    *mesh;
  void          *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(newdm,2);
  ierr = DMCreate(((PetscObject) dm)->comm, newdm);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*newdm)->sf);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) dm->sf);CHKERRQ(ierr);
  (*newdm)->sf = dm->sf;
  mesh = (DM_Plex *) dm->data;
  mesh->refct++;
  (*newdm)->data = mesh;
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMInitialize_Plex(*newdm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*newdm, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
