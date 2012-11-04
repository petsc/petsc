#define PETSCDM_DLL
#include <petsc-private/compleximpl.h>    /*I   "petscdmcomplex.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Complex"
PetscErrorCode  DMSetFromOptions_Complex(DM dm)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMComplex Options");CHKERRQ(ierr);
    /* Handle DMComplex refinement */
    /* Handle associated vectors */
    /* Handle viewing */
    ierr = PetscOptionsBool("-dm_complex_print_set_values", "Output all set values info", "DMView", PETSC_FALSE, &mesh->printSetValues, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dm_complex_print_fem", "Debug output level all fem computations", "DMView", 0, &mesh->printFEM, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSquareBoundary"
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
PetscErrorCode DMComplexCreateSquareBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
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
  ierr = PetscOptionsGetBool(((PetscObject) dm)->prefix, "-dm_complex_separate_marker", &markerSeparate, PETSC_NULL);CHKERRQ(ierr);
  if (markerSeparate) {
    markerTop    = 1;
    markerBottom = 0;
    markerRight  = 0;
    markerLeft   = 0;
  }
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt e, ex, ey;

    ierr = DMComplexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
    for (e = 0; e < numEdges; ++e) {
      ierr = DMComplexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for (vx = 0; vx <= edges[0]; vx++) {
      for (ey = 0; ey < edges[1]; ey++) {
        PetscInt edge    = vx*edges[1] + ey + edges[0]*(edges[1]+1);
        PetscInt vertex  = ey*(edges[0]+1) + vx + numEdges;
        PetscInt cone[2] = {vertex, vertex+edges[0]+1};

        ierr = DMComplexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vx == edges[0]) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerRight);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerRight);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerRight);CHKERRQ(ierr);
          }
        } else if (vx == 0) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerLeft);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerLeft);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerLeft);CHKERRQ(ierr);
          }
        }
      }
    }
    for (vy = 0; vy <= edges[1]; vy++) {
      for (ex = 0; ex < edges[0]; ex++) {
        PetscInt edge    = vy*edges[0]     + ex;
        PetscInt vertex  = vy*(edges[0]+1) + ex + numEdges;
        PetscInt cone[2] = {vertex, vertex+1};

        ierr = DMComplexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vy == edges[1]) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerTop);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerTop);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerTop);CHKERRQ(ierr);
          }
        } else if (vy == 0) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerBottom);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerBottom);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerBottom);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMComplexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMComplexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateCubeBoundary"
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
PetscErrorCode DMComplexCreateCubeBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
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

    ierr = DMComplexSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMComplexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for (v = 0; v < numFaces+numVertices; ++v) {
      ierr = DMComplexSetLabelValue(dm, "marker", v, 1);CHKERRQ(ierr);
    }
    { /* Side 0 (Front) */
      PetscInt cone[4] = {numFaces+4, numFaces+5, numFaces+7, numFaces+6};
      ierr = DMComplexSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { /* Side 1 (Back) */
      PetscInt cone[4] = {numFaces+1, numFaces+0, numFaces+2, numFaces+3};
      ierr = DMComplexSetCone(dm, 1, cone);CHKERRQ(ierr);
    }
    { /* Side 2 (Bottom) */
      PetscInt cone[4] = {numFaces+0, numFaces+1, numFaces+5, numFaces+4};
      ierr = DMComplexSetCone(dm, 2, cone);CHKERRQ(ierr);
    }
    { /* Side 3 (Top) */
      PetscInt cone[4] = {numFaces+6, numFaces+7, numFaces+3, numFaces+2};
      ierr = DMComplexSetCone(dm, 3, cone);CHKERRQ(ierr);
    }
    { /* Side 4 (Left) */
      PetscInt cone[4] = {numFaces+0, numFaces+4, numFaces+6, numFaces+2};
      ierr = DMComplexSetCone(dm, 4, cone);CHKERRQ(ierr);
    }
    { /* Side 5 (Right) */
      PetscInt cone[4] = {numFaces+5, numFaces+1, numFaces+3, numFaces+7};
      ierr = DMComplexSetCone(dm, 5, cone);CHKERRQ(ierr);
    }
  }
  ierr = DMComplexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMComplexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  for (v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 3);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &coordinates);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSquareMesh"
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
PetscErrorCode DMComplexCreateSquareMesh(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
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
  ierr = PetscOptionsGetBool(((PetscObject) dm)->prefix, "-dm_complex_separate_marker", &markerSeparate, PETSC_NULL);CHKERRQ(ierr);
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

    ierr = DMComplexSetChart(dm, 0, numFaces+numEdges+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMComplexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    for (e = firstXEdge; e < firstXEdge+numEdges; ++e) {
      ierr = DMComplexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    /* Build faces */
    for (fy = 0; fy < numYEdges; fy++) {
      for (fx = 0; fx < numXEdges; fx++) {
        const PetscInt face    = fy*numXEdges + fx;
        const PetscInt edgeL   = firstYEdge + fx*numYEdges + fy;
        const PetscInt edgeB   = firstXEdge + fy*numXEdges + fx;
        const PetscInt cone[4] = {edgeB, edgeL+numYEdges, edgeB+numXEdges, edgeL};

        ierr = DMComplexSetCone(dm, face, cone);CHKERRQ(ierr);
      }
    }
    /* Build Y edges*/
    for (vx = 0; vx < numXVertices; vx++) {
      for (ey = 0; ey < numYEdges; ey++) {
        const PetscInt edge    = firstYEdge  + vx*numYEdges + ey;
        const PetscInt vertex  = firstVertex + ey*numXVertices + vx;
        const PetscInt cone[2] = {vertex, vertex+numXVertices};

        ierr = DMComplexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vx == numXVertices-1) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerRight);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerRight);CHKERRQ(ierr);
          if (ey == numYEdges-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerRight);CHKERRQ(ierr);
          }
        } else if (vx == 0) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerLeft);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerLeft);CHKERRQ(ierr);
          if (ey == numYEdges-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerLeft);CHKERRQ(ierr);
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

        ierr = DMComplexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if (vy == numYVertices-1) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerTop);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerTop);CHKERRQ(ierr);
          if (ex == numXEdges-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerTop);CHKERRQ(ierr);
          }
        } else if (vy == 0) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    markerBottom);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], markerBottom);CHKERRQ(ierr);
          if (ex == numXEdges-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], markerBottom);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = DMComplexSymmetrize(dm);CHKERRQ(ierr);
    ierr = DMComplexStratify(dm);CHKERRQ(ierr);
    /* Build coordinates */
    ierr = DMComplexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
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
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateBoxMesh"
PetscErrorCode DMComplexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool interpolate, DM *dm) {
  DM             boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(boundary,dim,2);
  ierr = DMSetType(boundary, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(boundary, dim-1);CHKERRQ(ierr);
  switch(dim) {
  case 2:
  {
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};
    PetscInt  edges[2] = {2, 2};

    ierr = DMComplexCreateSquareBoundary(boundary, lower, upper, edges);CHKERRQ(ierr);
    break;
  }
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3] = {1, 1, 1};

    ierr = DMComplexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  ierr = DMComplexGenerate(boundary, PETSC_NULL, interpolate, dm);CHKERRQ(ierr);
  ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateHexBoxMesh"
PetscErrorCode DMComplexCreateHexBoxMesh(MPI_Comm comm, PetscInt dim, const PetscInt cells[], DM *dm) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(*dm,dim,2);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  switch(dim) {
  case 2:
  {
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};

    ierr = DMComplexCreateSquareMesh(*dm, lower, upper, cells);CHKERRQ(ierr);
    break;
  }
#if 0
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};

    ierr = DMComplexCreateCubeMesh(boundary, lower, upper, cells);CHKERRQ(ierr);
    break;
  }
#endif
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateInterpolation_Complex(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
extern PetscErrorCode DMCreateMatrix_Complex(DM dm, MatType mtype, Mat *J);
extern PetscErrorCode DMCreateCoordinateDM_Complex(DM dm, DM *cdm);
extern PetscErrorCode DMRefine_Complex(DM dm, MPI_Comm comm, DM *dmRefined);
extern PetscErrorCode DMSetUp_Complex(DM dm);
extern PetscErrorCode DMDestroy_Complex(DM dm);
extern PetscErrorCode DMView_Complex(DM dm, PetscViewer viewer);
extern PetscErrorCode DMCreateSubDM_Complex(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm);

#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Complex"
PetscErrorCode DMInitialize_Complex(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(VECSTANDARD, (char**)&dm->vectype);CHKERRQ(ierr);
  dm->ops->view               = DMView_Complex;
  dm->ops->setfromoptions     = DMSetFromOptions_Complex;
  dm->ops->setup              = DMSetUp_Complex;
  dm->ops->createglobalvector = PETSC_NULL;
  dm->ops->createlocalvector  = PETSC_NULL;
  dm->ops->createlocaltoglobalmapping      = PETSC_NULL;
  dm->ops->createlocaltoglobalmappingblock = PETSC_NULL;
  dm->ops->createfieldis      = PETSC_NULL;
  dm->ops->createcoordinatedm = DMCreateCoordinateDM_Complex;
  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix       = DMCreateMatrix_Complex;
  dm->ops->createinterpolation= 0;
  dm->ops->getaggregates      = 0;
  dm->ops->getinjection       = 0;
  dm->ops->refine             = DMRefine_Complex;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = 0;
  dm->ops->globaltolocalbegin = PETSC_NULL;
  dm->ops->globaltolocalend   = PETSC_NULL;
  dm->ops->localtoglobalbegin = PETSC_NULL;
  dm->ops->localtoglobalend   = PETSC_NULL;
  dm->ops->destroy            = DMDestroy_Complex;
  dm->ops->createsubdm        = DMCreateSubDM_Complex;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Complex"
PetscErrorCode DMCreate_Complex(DM dm)
{
  DM_Complex    *mesh;
  PetscInt       unit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Complex, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  mesh->refct            = 1;
  mesh->dim              = 0;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->coneSection);CHKERRQ(ierr);
  mesh->maxConeSize      = 0;
  mesh->cones            = PETSC_NULL;
  mesh->coneOrientations = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->supportSection);CHKERRQ(ierr);
  mesh->maxSupportSize   = 0;
  mesh->supports         = PETSC_NULL;
  mesh->refinementLimit  = -1.0;

  mesh->facesTmp         = PETSC_NULL;

  mesh->subpointMap      = PETSC_NULL;

  for(unit = 0; unit < NUM_PETSC_UNITS; ++unit) {
    mesh->scale[unit]    = 1.0;
  }

  mesh->labels               = PETSC_NULL;
  mesh->globalVertexNumbers  = PETSC_NULL;
  mesh->globalCellNumbers    = PETSC_NULL;
  mesh->vtkCellMax           = PETSC_DETERMINE;
  mesh->vtkVertexMax         = PETSC_DETERMINE;
  mesh->vtkCellHeight        = 0;

  mesh->integrateResidualFEM       = PETSC_NULL;
  mesh->integrateJacobianActionFEM = PETSC_NULL;
  mesh->integrateJacobianFEM       = PETSC_NULL;

  mesh->printSetValues       = PETSC_FALSE;
  mesh->printFEM             = 0;

  ierr = DMInitialize_Complex(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreate"
/*@
  DMComplexCreate - Creates a DMComplex object, which encapsulates an unstructured mesh, or CW complex, which can be expressed using a Hasse Diagram.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMComplex object

  Output Parameter:
. mesh  - The DMComplex object

  Level: beginner

.keywords: DMComplex, create
@*/
PetscErrorCode DMComplexCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMCOMPLEX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexClone"
/*@
  DMComplexClone - Creates a DMComplex object with the same mesh as the original.

  Collective on MPI_Comm

  Input Parameter:
. dm - The original DMComplex object

  Output Parameter:
. newdm  - The new DMComplex object

  Level: beginner

.keywords: DMComplex, create
@*/
PetscErrorCode DMComplexClone(DM dm, DM *newdm)
{
  DM_Complex    *mesh;
  void          *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(newdm,2);
  ierr = DMCreate(((PetscObject) dm)->comm, newdm);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*newdm)->sf);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) dm->sf);CHKERRQ(ierr);
  (*newdm)->sf = dm->sf;
  mesh = (DM_Complex *) dm->data;
  mesh->refct++;
  (*newdm)->data = mesh;
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMInitialize_Complex(*newdm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*newdm, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
