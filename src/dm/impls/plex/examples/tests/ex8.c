static char help[] = "Tests for cell geometry\n\n";

#include <petscdmplex.h>

typedef enum {RUN_REFERENCE, RUN_FILE} RunType;

typedef struct {
  DM        dm;
  RunType   runType;                      /* Type of mesh to use */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool interpolate;                  /* Interpolate the mesh */
  PetscBool transform;                    /* Use random coordinate transformations */
  /* Data for input meshes */
  PetscReal *v0, *J, *invJ, *detJ;        /* FEM data */
  PetscReal *centroid, *normal, *vol;     /* FVM data */
} AppCtx;

PetscErrorCode ReadMesh(MPI_Comm comm, const char *filename, AppCtx *user, DM *dm)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_FALSE, dm);CHKERRQ(ierr);
  if (user->interpolate) {
    DM interpolatedMesh = NULL;

    ierr = DMPlexInterpolate(*dm, &interpolatedMesh);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, interpolatedMesh);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = interpolatedMesh;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Input Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *runTypes[2] = {"reference", "file"};
  PetscInt       run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->runType     = RUN_REFERENCE;
  options->filename[0] = '\0';
  options->interpolate = PETSC_FALSE;
  options->transform   = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Geometry Test Options", "DMPLEX");CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex8.c", runTypes, 2, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsString("-filename", "The mesh file", "ex8.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "ex8.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-transform", "Use random transforms", "ex8.c", options->transform, &options->transform, NULL);CHKERRQ(ierr);

  if (options->runType == RUN_FILE) {
    PetscInt  dim, cStart, cEnd, numCells, n;
    PetscBool flag;

    ierr = ReadMesh(PETSC_COMM_WORLD, options->filename, options, &options->dm);CHKERRQ(ierr);
    ierr = DMGetDimension(options->dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(options->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    numCells = cEnd-cStart;
    ierr = PetscMalloc7(numCells*dim,&options->v0,numCells*dim*dim,&options->J,numCells*dim*dim,&options->invJ,numCells,&options->detJ,
                        numCells*dim,&options->centroid,numCells*dim,&options->normal,numCells,&options->vol);CHKERRQ(ierr);
    n    = numCells*dim;
    ierr = PetscOptionsRealArray("-v0", "Input v0 for each cell", "ex8.c", options->v0, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells*dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of v0 %D should be %D", n, numCells*dim);
    n    = numCells*dim*dim;
    ierr = PetscOptionsRealArray("-J", "Input Jacobian for each cell", "ex8.c", options->J, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells*dim*dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of J %D should be %D", n, numCells*dim*dim);
    n    = numCells*dim*dim;
    ierr = PetscOptionsRealArray("-invJ", "Input inverse Jacobian for each cell", "ex8.c", options->invJ, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells*dim*dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of invJ %D should be %D", n, numCells*dim*dim);
    n    = numCells;
    ierr = PetscOptionsRealArray("-detJ", "Input Jacobian determinant for each cell", "ex8.c", options->detJ, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of detJ %D should be %D", n, numCells);
    n    = numCells*dim;
    ierr = PetscOptionsRealArray("-centroid", "Input centroid for each cell", "ex8.c", options->centroid, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells*dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of centroid %D should be %D", n, numCells*dim);
    n    = numCells*dim;
    ierr = PetscOptionsRealArray("-normal", "Input normal for each cell", "ex8.c", options->normal, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells*dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of normal %D should be %D", n, numCells*dim);
    n    = numCells;
    ierr = PetscOptionsRealArray("-vol", "Input volume for each cell", "ex8.c", options->vol, &n, &flag);CHKERRQ(ierr);
    if (flag && n != numCells) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid size of vol %D should be %D", n, numCells);
  }
  ierr = PetscOptionsEnd();

  if (options->transform) {ierr = PetscPrintf(comm, "Using random transforms");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode ChangeCoordinates(DM dm, PetscInt spaceDim, PetscScalar vertexCoords[])
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       vStart, vEnd, v, d, coordSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
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
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt off;

    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    for (d = 0; d < spaceDim; ++d) {
      coords[off+d] = vertexCoords[(v-vStart)*spaceDim+d];
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm, spaceDim);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define RelativeError(a,b) PetscAbs(a-b)/(1.0+PetscMax(PetscAbs(a),PetscAbs(b)))

PetscErrorCode CheckFEMGeometry(DM dm, PetscInt cell, PetscInt spaceDim, PetscReal v0Ex[], PetscReal JEx[], PetscReal invJEx[], PetscReal detJEx)
{
  PetscReal      v0[3], J[9], invJ[9], detJ;
  PetscInt       d, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
  for (d = 0; d < spaceDim; ++d) {
    if (v0[d] != v0Ex[d]) {
      switch (spaceDim) {
      case 2: SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid v0 (%g, %g) != (%g, %g)", (double)v0[0], (double)v0[1], (double)v0Ex[0], (double)v0Ex[1]);break;
      case 3: SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid v0 (%g, %g, %g) != (%g, %g, %g)", (double)v0[0], (double)v0[1], (double)v0[2], (double)v0Ex[0], (double)v0Ex[1], (double)v0Ex[2]);break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid space dimension %D", spaceDim);
      }
    }
  }
  for (i = 0; i < spaceDim; ++i) {
    for (j = 0; j < spaceDim; ++j) {
      if (RelativeError(J[i*spaceDim+j],JEx[i*spaceDim+j])    > 10*PETSC_SMALL) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid J[%D,%D]: %g != %g", i, j, (double)J[i*spaceDim+j], (double)JEx[i*spaceDim+j]);
      if (RelativeError(invJ[i*spaceDim+j],invJEx[i*spaceDim+j]) > 10*PETSC_SMALL) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid invJ[%D,%D]: %g != %g", i, j, (double)invJ[i*spaceDim+j], (double)invJEx[i*spaceDim+j]);
    }
  }
  if (RelativeError(detJ,detJEx) > 10*PETSC_SMALL) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid |J| = %g != %g diff %g", (double)detJ, (double)detJEx,(double)(detJ - detJEx));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckFVMGeometry(DM dm, PetscInt cell, PetscInt spaceDim, PetscReal centroidEx[], PetscReal normalEx[], PetscReal volEx)
{
  PetscReal      centroid[3], normal[3], vol;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFVM(dm, cell, &vol, centroid, normal);CHKERRQ(ierr);
  for (d = 0; d < spaceDim; ++d) {
    if (RelativeError(centroid[d],centroidEx[d]) > 10*PETSC_SMALL) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid centroid[%D]: %g != %g diff %g", d, (double)centroid[d], (double)centroidEx[d],(double)(centroid[d]-centroidEx[d]));
    if (RelativeError(normal[d],normalEx[d]) > 10*PETSC_SMALL) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid normal[%D]: %g != %g", d, (double)normal[d], (double)normalEx[d]);
  }
  if (RelativeError(volEx,vol) > 10*PETSC_SMALL) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid volume = %g != %g diff %g", (double)vol, (double)volEx,(double)(vol - volEx));
  PetscFunctionReturn(0);
}

PetscErrorCode TestTriangle(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscRandom    r, ang, ang2;
  PetscInt       dim, t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create reference triangle */
  dim  = 2;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "triangle");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    PetscInt    numPoints[2]        = {3, 1};
    PetscInt    coneSize[4]         = {3, 0, 0, 0};
    PetscInt    cones[3]            = {1, 2, 3};
    PetscInt    coneOrientations[3] = {0, 0, 0};
    PetscScalar vertexCoords[6]     = {-1.0, -1.0, 1.0, -1.0, -1.0, 1.0};

    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    if (interpolate) {
      DM idm = NULL;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) idm, "triangle");CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
    }
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume (2.0) */
  {
    PetscReal v0Ex[2]       = {-1.0, -1.0};
    PetscReal JEx[4]        = {1.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[4]     = {1.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[2] = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.)};
    PetscReal normalEx[2]   = {0.0, 0.0};
    PetscReal volEx         = 2.0;

    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random triangles: rotate, scale, then translate */
  if (transform) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 100; ++t) {
      PetscScalar vertexCoords[6] = {-1.0, -1.0, 1.0, -1.0, -1.0, 1.0}, trans[2];
      PetscReal   v0Ex[2]         = {-1.0, -1.0};
      PetscReal   JEx[4]          = {1.0, 0.0, 0.0, 1.0}, R[4], rot[2], rotM[4];
      PetscReal   invJEx[4]       = {1.0, 0.0, 0.0, 1.0};
      PetscReal   detJEx          = 1.0, scale, phi;
      PetscReal   centroidEx[2]   = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.)};
      PetscReal   normalEx[2]     = {0.0, 0.0};
      PetscReal   volEx           = 2.0;
      PetscInt    d, e, f, p;

      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      R[0] = PetscCosReal(phi); R[1] = -PetscSinReal(phi);
      R[2] = PetscSinReal(phi); R[3] =  PetscCosReal(phi);
      for (p = 0; p < 3; ++p) {
        for (d = 0; d < dim; ++d) {
          for (e = 0, rot[d] = 0.0; e < dim; ++e) {
            rot[d] += R[d*dim+e] * PetscRealPart(vertexCoords[p*dim+e]);
          }
        }
        for (d = 0; d < dim; ++d) vertexCoords[p*dim+d] = rot[d];
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * centroidEx[e];
        }
      }
      for (d = 0; d < dim; ++d) centroidEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += R[d*dim+f] * JEx[f*dim+e];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += invJEx[d*dim+f] * R[e*dim+f];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          invJEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {
          vertexCoords[p*dim+d] *= scale;
          vertexCoords[p*dim+d] += trans[d];
        }
        v0Ex[d] = PetscRealPart(vertexCoords[d]);
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e]    *= scale;
          invJEx[d*dim+e] /= scale;
        }
        detJEx *= scale;
        centroidEx[d] *= scale;
        centroidEx[d] += PetscRealPart(trans[d]);
        volEx *= scale;
      }
      ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
      ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
      if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
  }
  /* Move to 3D: Check reference geometry: determinant is scaled by reference volume (2.0) */
  dim = 3;
  {
    PetscScalar vertexCoords[9] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0};
    PetscReal v0Ex[3]       = {-1.0, -1.0, 0.0};
    PetscReal JEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[9]     = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[3] = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.), 0.0};
    PetscReal normalEx[3]   = {0.0, 0.0, 1.0};
    PetscReal volEx         = 2.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Rotated reference element */
  {
    PetscScalar vertexCoords[9] = {0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0};
    PetscReal   v0Ex[3]         = {0.0, -1.0, -1.0};
    PetscReal   JEx[9]          = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    PetscReal   invJEx[9]       = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    PetscReal   detJEx          = 1.0;
    PetscReal   centroidEx[3]   = {0.0, -((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.)};
    PetscReal   normalEx[3]     = {1.0, 0.0, 0.0};
    PetscReal   volEx           = 2.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random triangles: scale, translate, then rotate */
  if (transform) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang2, 0.0, PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 100; ++t) {
      PetscScalar vertexCoords[9] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0}, trans[3];
      PetscReal   v0Ex[3]         = {-1.0, -1.0, 0.0};
      PetscReal   JEx[9]          = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, R[9], rot[3], rotM[9];
      PetscReal   invJEx[9]       = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      PetscReal   detJEx          = 1.0, scale, phi, theta, psi = 0.0;
      PetscReal   centroidEx[3]   = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.), 0.0};
      PetscReal   normalEx[3]     = {0.0, 0.0, 1.0};
      PetscReal   volEx           = 2.0;
      PetscInt    d, e, f, p;

      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang2, &theta);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {
          vertexCoords[p*dim+d] *= scale;
          vertexCoords[p*dim+d] += trans[d];
        }
        centroidEx[d] *= scale;
        centroidEx[d] += PetscRealPart(trans[d]);
        for (e = 0; e < dim-1; ++e) {
          JEx[d*dim+e]    *= scale;
          invJEx[d*dim+e] /= scale;
        }
        if (d < dim-1) {
          detJEx *= scale;
          volEx  *= scale;
        }
      }
      R[0] = PetscCosReal(theta)*PetscCosReal(psi); R[1] = PetscSinReal(phi)*PetscSinReal(theta)*PetscCosReal(psi) - PetscCosReal(phi)*PetscSinReal(psi); R[2] = PetscSinReal(phi)*PetscSinReal(psi) + PetscCosReal(phi)*PetscSinReal(theta)*PetscCosReal(psi);
      R[3] = PetscCosReal(theta)*PetscSinReal(psi); R[4] = PetscCosReal(phi)*PetscCosReal(psi) + PetscSinReal(phi)*PetscSinReal(theta)*PetscSinReal(psi); R[5] = PetscCosReal(phi)*PetscSinReal(theta)*PetscSinReal(psi) - PetscSinReal(phi)*PetscCosReal(psi);
      R[6] = -PetscSinReal(theta);         R[7] = PetscSinReal(phi)*PetscCosReal(theta);                              R[8] = PetscCosReal(phi)*PetscCosReal(theta);
      for (p = 0; p < 3; ++p) {
        for (d = 0; d < dim; ++d) {
          for (e = 0, rot[d] = 0.0; e < dim; ++e) {
            rot[d] += R[d*dim+e] * PetscRealPart(vertexCoords[p*dim+e]);
          }
        }
        for (d = 0; d < dim; ++d) vertexCoords[p*dim+d] = rot[d];
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * centroidEx[e];
        }
      }
      for (d = 0; d < dim; ++d) centroidEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * normalEx[e];
        }
      }
      for (d = 0; d < dim; ++d) normalEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        v0Ex[d] = PetscRealPart(vertexCoords[d]);
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += R[d*dim+f] * JEx[f*dim+e];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += invJEx[d*dim+f] * R[e*dim+f];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          invJEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
      ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
      if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang2);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestQuadrilateral(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscRandom    r, ang, ang2;
  PetscInt       dim, t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create reference quadrilateral */
  dim  = 2;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "quadrilateral");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    PetscInt    numPoints[2]        = {4, 1};
    PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
    PetscInt    cones[4]            = {1, 2, 3, 4};
    PetscInt    coneOrientations[4] = {0, 0, 0, 0};
    PetscScalar vertexCoords[8]     = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0};

    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    if (interpolate) {
      DM idm = NULL;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) idm, "quadrilateral");CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
    }
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume (2.0) */
  {
    PetscReal v0Ex[2]       = {-1.0, -1.0};
    PetscReal JEx[4]        = {1.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[4]     = {1.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[2] = {0.0, 0.0};
    PetscReal normalEx[2]   = {0.0, 0.0};
    PetscReal volEx         = 4.0;

    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random quadrilaterals: rotate, scale, then translate */
  if (transform) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 100; ++t) {
      PetscScalar vertexCoords[8] = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0}, trans[2];
      PetscReal   v0Ex[2]         = {-1.0, -1.0};
      PetscReal   JEx[4]          = {1.0, 0.0, 0.0, 1.0}, R[4], rot[2], rotM[4];
      PetscReal   invJEx[4]       = {1.0, 0.0, 0.0, 1.0};
      PetscReal   detJEx          = 1.0, scale, phi;
      PetscReal   centroidEx[2]   = {0.0, 0.0};
      PetscReal   normalEx[2]     = {0.0, 0.0};
      PetscReal   volEx           = 4.0;
      PetscInt    d, e, f, p;

      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      R[0] = PetscCosReal(phi); R[1] = -PetscSinReal(phi);
      R[2] = PetscSinReal(phi); R[3] =  PetscCosReal(phi);
      for (p = 0; p < 4; ++p) {
        for (d = 0; d < dim; ++d) {
          for (e = 0, rot[d] = 0.0; e < dim; ++e) {
            rot[d] += R[d*dim+e] * PetscRealPart(vertexCoords[p*dim+e]);
          }
        }
        for (d = 0; d < dim; ++d) vertexCoords[p*dim+d] = rot[d];
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * centroidEx[e];
        }
      }
      for (d = 0; d < dim; ++d) centroidEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += R[d*dim+f] * JEx[f*dim+e];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += invJEx[d*dim+f] * R[e*dim+f];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          invJEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {
          vertexCoords[p*dim+d] *= scale;
          vertexCoords[p*dim+d] += trans[d];
        }
        v0Ex[d] = PetscRealPart(vertexCoords[d]);
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e]    *= scale;
          invJEx[d*dim+e] /= scale;
        }
        detJEx *= scale;
        centroidEx[d] *= scale;
        centroidEx[d] += PetscRealPart(trans[d]);
        volEx *= scale;
      }
      ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
      ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
      if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
  }
  /* Move to 3D: Check reference geometry: determinant is scaled by reference volume (4.0) */
  dim = 3;
  {
    PetscScalar vertexCoords[12] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};
    PetscReal v0Ex[3]            = {-1.0, -1.0, 0.0};
    PetscReal JEx[9]             = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[9]          = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal detJEx             = 1.0;
    PetscReal centroidEx[3]      = {0.0, 0.0, 0.0};
    PetscReal normalEx[3]        = {0.0, 0.0, 1.0};
    PetscReal volEx              = 4.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random quadrilaterals: scale, translate, then rotate */
  if (transform) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang2, 0.0, PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 100; ++t) {
      PetscScalar vertexCoords[12] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0}, trans[3];
      PetscReal   v0Ex[3]          = {-1.0, -1.0, 0.0};
      PetscReal   JEx[9]           = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, R[9], rot[3], rotM[9];
      PetscReal   invJEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      PetscReal   detJEx           = 1.0, scale, phi, theta, psi = 0.0;
      PetscReal   centroidEx[3]    = {0.0, 0.0, 0.0};
      PetscReal   normalEx[3]      = {0.0, 0.0, 1.0};
      PetscReal   volEx            = 4.0;
      PetscInt    d, e, f, p;

      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang2, &theta);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {
          vertexCoords[p*dim+d] *= scale;
          vertexCoords[p*dim+d] += trans[d];
        }
        centroidEx[d] *= scale;
        centroidEx[d] += PetscRealPart(trans[d]);
        for (e = 0; e < dim-1; ++e) {
          JEx[d*dim+e]    *= scale;
          invJEx[d*dim+e] /= scale;
        }
        if (d < dim-1) {
          detJEx *= scale;
          volEx  *= scale;
        }
      }
      R[0] = PetscCosReal(theta)*PetscCosReal(psi); R[1] = PetscSinReal(phi)*PetscSinReal(theta)*PetscCosReal(psi) - PetscCosReal(phi)*PetscSinReal(psi); R[2] = PetscSinReal(phi)*PetscSinReal(psi) + PetscCosReal(phi)*PetscSinReal(theta)*PetscCosReal(psi);
      R[3] = PetscCosReal(theta)*PetscSinReal(psi); R[4] = PetscCosReal(phi)*PetscCosReal(psi) + PetscSinReal(phi)*PetscSinReal(theta)*PetscSinReal(psi); R[5] = PetscCosReal(phi)*PetscSinReal(theta)*PetscSinReal(psi) - PetscSinReal(phi)*PetscCosReal(psi);
      R[6] = -PetscSinReal(theta);         R[7] = PetscSinReal(phi)*PetscCosReal(theta);                              R[8] = PetscCosReal(phi)*PetscCosReal(theta);
      for (p = 0; p < 4; ++p) {
        for (d = 0; d < dim; ++d) {
          for (e = 0, rot[d] = 0.0; e < dim; ++e) {
            rot[d] += R[d*dim+e] * PetscRealPart(vertexCoords[p*dim+e]);
          }
        }
        for (d = 0; d < dim; ++d) vertexCoords[p*dim+d] = rot[d];
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * centroidEx[e];
        }
      }
      for (d = 0; d < dim; ++d) centroidEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * normalEx[e];
        }
      }
      for (d = 0; d < dim; ++d) normalEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        v0Ex[d] = PetscRealPart(vertexCoords[d]);
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += R[d*dim+f] * JEx[f*dim+e];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += invJEx[d*dim+f] * R[e*dim+f];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          invJEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
      ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
      if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang2);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestTetrahedron(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscRandom    r, ang, ang2;
  PetscInt       dim, t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create reference tetrahedron */
  dim  = 3;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "tetrahedron");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    PetscInt    numPoints[2]        = {4, 1};
    PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
    PetscInt    cones[4]            = {1, 2, 3, 4};
    PetscInt    coneOrientations[4] = {0, 0, 0, 0};
    PetscScalar vertexCoords[12]    = {-1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, -1.0, 1.0};

    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    if (interpolate) {
      DM idm = NULL;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) idm, "tetrahedron");CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
    }
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume (4/3) */
  {
    PetscReal v0Ex[3]       = {-1.0, -1.0, -1.0};
    PetscReal JEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[9]     = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[3] = {-0.5, -0.5, -0.5};
    PetscReal normalEx[3]   = {0.0, 0.0, 0.0};
    PetscReal volEx         = (PetscReal)4.0/(PetscReal)3.0;

    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random tetrahedra: rotate, scale, then translate */
  if (transform) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang2, 0.0, PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 100; ++t) {
      PetscScalar vertexCoords[12] = {-1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, -1.0, 1.0}, trans[3];
      PetscReal   v0Ex[3]          = {-1.0, -1.0, -1.0};
      PetscReal   JEx[9]           = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, R[9], rot[3], rotM[9];
      PetscReal   invJEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      PetscReal   detJEx           = 1.0, scale, phi, theta, psi = 0.0;
      PetscReal   centroidEx[3]    = {-0.5, -0.5, -0.5};
      PetscReal   normalEx[3]      = {0.0, 0.0, 0.0};
      PetscReal   volEx            = (PetscReal)4.0/(PetscReal)3.0;
      PetscInt    d, e, f, p;

      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang2, &theta);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {
          vertexCoords[p*dim+d] *= scale;
          vertexCoords[p*dim+d] += trans[d];
        }
        centroidEx[d] *= scale;
        centroidEx[d] += PetscRealPart(trans[d]);
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e]    *= scale;
          invJEx[d*dim+e] /= scale;
        }
        detJEx *= scale;
        volEx  *= scale;
      }
      R[0] = PetscCosReal(theta)*PetscCosReal(psi); R[1] = PetscSinReal(phi)*PetscSinReal(theta)*PetscCosReal(psi) - PetscCosReal(phi)*PetscSinReal(psi); R[2] = PetscSinReal(phi)*PetscSinReal(psi) + PetscCosReal(phi)*PetscSinReal(theta)*PetscCosReal(psi);
      R[3] = PetscCosReal(theta)*PetscSinReal(psi); R[4] = PetscCosReal(phi)*PetscCosReal(psi) + PetscSinReal(phi)*PetscSinReal(theta)*PetscSinReal(psi); R[5] = PetscCosReal(phi)*PetscSinReal(theta)*PetscSinReal(psi) - PetscSinReal(phi)*PetscCosReal(psi);
      R[6] = -PetscSinReal(theta);         R[7] = PetscSinReal(phi)*PetscCosReal(theta);                              R[8] = PetscCosReal(phi)*PetscCosReal(theta);
      for (p = 0; p < 4; ++p) {
        for (d = 0; d < dim; ++d) {
          for (e = 0, rot[d] = 0.0; e < dim; ++e) {
            rot[d] += R[d*dim+e] * PetscRealPart(vertexCoords[p*dim+e]);
          }
        }
        for (d = 0; d < dim; ++d) vertexCoords[p*dim+d] = rot[d];
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * centroidEx[e];
        }
      }
      for (d = 0; d < dim; ++d) centroidEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        v0Ex[d] = PetscRealPart(vertexCoords[d]);
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += R[d*dim+f] * JEx[f*dim+e];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += invJEx[d*dim+f] * R[e*dim+f];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          invJEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
      ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
      if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang2);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestHexahedron(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscRandom    r, ang, ang2;
  PetscInt       dim, t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create reference hexahedron */
  dim  = 3;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "hexahedron");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    PetscInt    numPoints[2]        = {8, 1};
    PetscInt    coneSize[9]         = {8, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[8]            = {1, 2, 3, 4, 5, 6, 7, 8};
    PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  -1.0,  1.0, -1.0,  1.0, 1.0, -1.0,   1.0, -1.0, -1.0,
                                       -1.0, -1.0,  1.0,   1.0, -1.0,  1.0,  1.0, 1.0,  1.0,  -1.0,  1.0,  1.0};

    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    if (interpolate) {
      DM idm = NULL;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) idm, "hexahedron");CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
    }
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume 8.0 */
  {
    PetscReal v0Ex[3]       = {-1.0, -1.0, -1.0};
    PetscReal JEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[9]     = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal detJEx        = 1.0;
    PetscReal centroidEx[3] = {0.0, 0.0, 0.0};
    PetscReal normalEx[3]   = {0.0, 0.0, 0.0};
    PetscReal volEx         = 8.0;

    ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
    if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
  }
  /* Check random hexahedra: rotate, scale, then translate */
  if (transform) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang2, 0.0, PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 100; ++t) {
      PetscScalar vertexCoords[24] = {-1.0, -1.0, -1.0,  -1.0,  1.0, -1.0,  1.0, 1.0, -1.0,   1.0, -1.0, -1.0,
                                      -1.0, -1.0,  1.0,   1.0, -1.0,  1.0,  1.0, 1.0,  1.0,  -1.0,  1.0,  1.0}, trans[3];
      PetscReal   v0Ex[3]          = {-1.0, -1.0, -1.0};
      PetscReal   JEx[9]           = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, R[9], rot[3], rotM[9];
      PetscReal   invJEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      PetscReal   detJEx           = 1.0, scale, phi, theta, psi = 0.0;
      PetscReal   centroidEx[3]    = {0.0, 0.0, 0.0};
      PetscReal   normalEx[3]      = {0.0, 0.0, 0.0};
      PetscReal   volEx            = 8.0;
      PetscInt    d, e, f, p;

      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang2, &theta);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
        for (p = 0; p < 8; ++p) {
          vertexCoords[p*dim+d] *= scale;
          vertexCoords[p*dim+d] += trans[d];
        }
        centroidEx[d] *= scale;
        centroidEx[d] += PetscRealPart(trans[d]);
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e]    *= scale;
          invJEx[d*dim+e] /= scale;
        }
        detJEx *= scale;
        volEx  *= scale;
      }
      R[0] = PetscCosReal(theta)*PetscCosReal(psi); R[1] = PetscSinReal(phi)*PetscSinReal(theta)*PetscCosReal(psi) - PetscCosReal(phi)*PetscSinReal(psi); R[2] = PetscSinReal(phi)*PetscSinReal(psi) + PetscCosReal(phi)*PetscSinReal(theta)*PetscCosReal(psi);
      R[3] = PetscCosReal(theta)*PetscSinReal(psi); R[4] = PetscCosReal(phi)*PetscCosReal(psi) + PetscSinReal(phi)*PetscSinReal(theta)*PetscSinReal(psi); R[5] = PetscCosReal(phi)*PetscSinReal(theta)*PetscSinReal(psi) - PetscSinReal(phi)*PetscCosReal(psi);
      R[6] = -PetscSinReal(theta);         R[7] = PetscSinReal(phi)*PetscCosReal(theta);                              R[8] = PetscCosReal(phi)*PetscCosReal(theta);
      for (p = 0; p < 8; ++p) {
        for (d = 0; d < dim; ++d) {
          for (e = 0, rot[d] = 0.0; e < dim; ++e) {
            rot[d] += R[d*dim+e] * PetscRealPart(vertexCoords[p*dim+e]);
          }
        }
        for (d = 0; d < dim; ++d) vertexCoords[p*dim+d] = rot[d];
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0, rot[d] = 0.0; e < dim; ++e) {
          rot[d] += R[d*dim+e] * centroidEx[e];
        }
      }
      for (d = 0; d < dim; ++d) centroidEx[d] = rot[d];
      for (d = 0; d < dim; ++d) {
        v0Ex[d] = PetscRealPart(vertexCoords[d]);
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += R[d*dim+f] * JEx[f*dim+e];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          JEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          for (f = 0, rotM[d*dim+e] = 0.0; f < dim; ++f) {
            rotM[d*dim+e] += invJEx[d*dim+f] * R[e*dim+f];
          }
        }
      }
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; ++e) {
          invJEx[d*dim+e] = rotM[d*dim+e];
        }
      }
      ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
      ierr = CheckFEMGeometry(dm, 0, dim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
      if (interpolate) {ierr = CheckFVMGeometry(dm, 0, dim, centroidEx, normalEx, volEx);CHKERRQ(ierr);}
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang2);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  if (user.runType == RUN_REFERENCE) {
    ierr = TestTriangle(PETSC_COMM_SELF, user.interpolate, user.transform);CHKERRQ(ierr);
    ierr = TestQuadrilateral(PETSC_COMM_SELF, user.interpolate, user.transform);CHKERRQ(ierr);
    ierr = TestTetrahedron(PETSC_COMM_SELF, user.interpolate, user.transform);CHKERRQ(ierr);
    ierr = TestHexahedron(PETSC_COMM_SELF, user.interpolate, user.transform);CHKERRQ(ierr);
  } else if (user.runType == RUN_FILE) {
    PetscInt dim, cStart, cEnd, c;

    ierr = DMGetDimension(user.dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(user.dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = 0; c < cEnd-cStart; ++c) {
      ierr = CheckFEMGeometry(user.dm, c+cStart, dim, &user.v0[c*dim], &user.J[c*dim*dim], &user.invJ[c*dim*dim], user.detJ[c]);CHKERRQ(ierr);
      if (user.interpolate) {ierr = CheckFVMGeometry(user.dm, c+cStart, dim, &user.centroid[c*dim], &user.normal[c*dim], user.vol[c]);CHKERRQ(ierr);}
    }
    ierr = PetscFree7(user.v0,user.J,user.invJ,user.detJ,user.centroid,user.normal,user.vol);CHKERRQ(ierr);
    ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -dm_view ascii::ascii_info_detail
  test:
    suffix: 1
    args: -interpolate -dm_view ascii::ascii_info_detail
  test:
    suffix: 2
    args: -transform
  test:
    suffix: 3
    args: -interpolate -transform
  test:
    suffix: 4
    requires: exodusii
    args: -run_type file -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/simpleblock-100.exo -dm_view ascii::ascii_info_detail -v0 -1.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,0.5 -J 0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0 -invJ 0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0 -detJ 0.125,0.125,0.125
  test:
    suffix: 5
    requires: exodusii
    args: -interpolate -run_type file -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/simpleblock-100.exo -dm_view ascii::ascii_info_detail -v0 -1.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,0.5 -J 0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0 -invJ 0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0 -detJ 0.125,0.125,0.125 -centroid -1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0 -normal 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 -vol 1.0,1.0,1.0

TEST*/
