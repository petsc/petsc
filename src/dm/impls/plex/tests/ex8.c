static char help[] = "Tests for cell geometry\n\n";

#include <petscdmplex.h>

typedef enum {RUN_REFERENCE, RUN_FILE, RUN_DISPLAY} RunType;

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

static PetscErrorCode ReadMesh(MPI_Comm comm, const char *filename, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateFromFile(comm, filename, user->interpolate, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Input Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *runTypes[3] = {"reference", "file", "display"};
  PetscInt       run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->runType     = RUN_REFERENCE;
  options->filename[0] = '\0';
  options->interpolate = PETSC_FALSE;
  options->transform   = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Geometry Test Options", "DMPLEX");CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex8.c", runTypes, 3, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsString("-filename", "The mesh file", "ex8.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
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
  } else if (options->runType == RUN_DISPLAY) {
    ierr = ReadMesh(PETSC_COMM_WORLD, options->filename, options, &options->dm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();

  if (options->transform) {ierr = PetscPrintf(comm, "Using random transforms\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode ChangeCoordinates(DM dm, PetscInt spaceDim, PetscScalar vertexCoords[])
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

static PetscErrorCode CheckFEMGeometry(DM dm, PetscInt cell, PetscInt spaceDim, PetscReal v0Ex[], PetscReal JEx[], PetscReal invJEx[], PetscReal detJEx)
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

static PetscErrorCode CheckFVMGeometry(DM dm, PetscInt cell, PetscInt spaceDim, PetscReal centroidEx[], PetscReal normalEx[], PetscReal volEx)
{
  PetscReal      centroid[3], normal[3], vol;
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeCellGeometryFVM(dm, cell, &vol, centroid, normal);CHKERRQ(ierr);
  for (d = 0; d < spaceDim; ++d) {
    if (RelativeError(centroid[d],centroidEx[d]) > 10*PETSC_SMALL) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %D, Invalid centroid[%D]: %g != %g diff %g", cell, d, (double)centroid[d], (double)centroidEx[d],(double)(centroid[d]-centroidEx[d]));
    if (RelativeError(normal[d],normalEx[d]) > 10*PETSC_SMALL) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %D, Invalid normal[%D]: %g != %g", cell, d, (double)normal[d], (double)normalEx[d]);
  }
  if (RelativeError(volEx,vol) > 10*PETSC_SMALL) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %D, Invalid volume = %g != %g diff %g", cell, (double)vol, (double)volEx,(double)(vol - volEx));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckCell(DM dm, PetscInt cell, PetscBool transform, PetscReal v0Ex[], PetscReal JEx[], PetscReal invJEx[], PetscReal detJEx, PetscReal centroidEx[], PetscReal normalEx[], PetscReal volEx, PetscReal faceCentroidEx[], PetscReal faceNormalEx[], PetscReal faceVolEx[])
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscInt        dim, depth, cdim;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  if (v0Ex) {
    ierr = CheckFEMGeometry(dm, cell, cdim, v0Ex, JEx, invJEx, detJEx);CHKERRQ(ierr);
  }
  if (dim == depth && centroidEx) {
    ierr = CheckFVMGeometry(dm, cell, cdim, centroidEx, normalEx, volEx);CHKERRQ(ierr);
    if (faceCentroidEx) {
      ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        ierr = CheckFVMGeometry(dm, cone[c], dim, &faceCentroidEx[c*dim], &faceNormalEx[c*dim], faceVolEx[c]);CHKERRQ(ierr);
      }
    }
  }
  if (transform) {
    Vec          coordinates;
    PetscSection coordSection;
    PetscScalar *coords = NULL, *origCoords, *newCoords;
    PetscReal   *v0ExT, *JExT, *invJExT, detJExT=0, *centroidExT, *normalExT, volExT=0;
    PetscReal   *faceCentroidExT, *faceNormalExT, faceVolExT;
    PetscRandom  r, ang, ang2;
    PetscInt     coordSize, numCorners, t;

    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
    ierr = PetscMalloc2(coordSize, &origCoords, coordSize, &newCoords);CHKERRQ(ierr);
    ierr = PetscMalloc5(cdim, &v0ExT, cdim*cdim, &JExT, cdim*cdim, &invJExT, cdim, &centroidExT, cdim, &normalExT);CHKERRQ(ierr);
    ierr = PetscMalloc2(cdim, &faceCentroidExT, cdim, &faceNormalExT);CHKERRQ(ierr);
    for (c = 0; c < coordSize; ++c) origCoords[c] = coords[c];
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, cell, &coordSize, &coords);CHKERRQ(ierr);
    numCorners = coordSize/cdim;

    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, 10.0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang, 0.0, 2*PETSC_PI);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(ang2);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(ang2, 0.0, PETSC_PI);CHKERRQ(ierr);
    for (t = 0; t < 1; ++t) {
      PetscScalar trans[3];
      PetscReal   R[9], rot[3], rotM[9];
      PetscReal   scale, phi, theta, psi = 0.0, norm;
      PetscInt    d, e, f, p;

      for (c = 0; c < coordSize; ++c) newCoords[c] = origCoords[c];
      ierr = PetscRandomGetValueReal(r, &scale);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang, &phi);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(ang2, &theta);CHKERRQ(ierr);
      for (d = 0; d < cdim; ++d) {
        ierr = PetscRandomGetValue(r, &trans[d]);CHKERRQ(ierr);
      }
      switch (cdim) {
      case 2:
        R[0] = PetscCosReal(phi); R[1] = -PetscSinReal(phi);
        R[2] = PetscSinReal(phi); R[3] =  PetscCosReal(phi);
        break;
      case 3:
      {
        const PetscReal ct = PetscCosReal(theta), st = PetscSinReal(theta);
        const PetscReal cp = PetscCosReal(phi),   sp = PetscSinReal(phi);
        const PetscReal cs = PetscCosReal(psi),   ss = PetscSinReal(psi);
        R[0] = ct*cs; R[1] = sp*st*cs - cp*ss;    R[2] = sp*ss    + cp*st*cs;
        R[3] = ct*ss; R[4] = cp*cs    + sp*st*ss; R[5] = cp*st*ss - sp*cs;
        R[6] = -st;   R[7] = sp*ct;               R[8] = cp*ct;
        break;
      }
      default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Invalid coordinate dimension %D", cdim);
      }
      if (v0Ex) {
        detJExT = detJEx;
        for (d = 0; d < cdim; ++d) {
          v0ExT[d] = v0Ex[d];
          for (e = 0; e < cdim; ++e) {
            JExT[d*cdim+e]    = JEx[d*cdim+e];
            invJExT[d*cdim+e] = invJEx[d*cdim+e];
          }
        }
        for (d = 0; d < cdim; ++d) {
          v0ExT[d] *= scale;
          v0ExT[d] += PetscRealPart(trans[d]);
          /* Only scale dimensions in the manifold */
          for (e = 0; e < dim; ++e) {
            JExT[d*cdim+e]    *= scale;
            invJExT[d*cdim+e] /= scale;
          }
          if (d < dim) detJExT *= scale;
        }
        /* Do scaling and translation before rotation, so that we can leave out the normal dimension for lower dimensional manifolds */
        for (d = 0; d < cdim; ++d) {
          for (e = 0, rot[d] = 0.0; e < cdim; ++e) {
            rot[d] += R[d*cdim+e] * v0ExT[e];
          }
        }
        for (d = 0; d < cdim; ++d) v0ExT[d] = rot[d];
        for (d = 0; d < cdim; ++d) {
          for (e = 0; e < cdim; ++e) {
            for (f = 0, rotM[d*cdim+e] = 0.0; f < cdim; ++f) {
              rotM[d*cdim+e] += R[d*cdim+f] * JExT[f*cdim+e];
            }
          }
        }
        for (d = 0; d < cdim; ++d) {
          for (e = 0; e < cdim; ++e) {
            JExT[d*cdim+e] = rotM[d*cdim+e];
          }
        }
        for (d = 0; d < cdim; ++d) {
          for (e = 0; e < cdim; ++e) {
            for (f = 0, rotM[d*cdim+e] = 0.0; f < cdim; ++f) {
              rotM[d*cdim+e] += invJExT[d*cdim+f] * R[e*cdim+f];
            }
          }
        }
        for (d = 0; d < cdim; ++d) {
          for (e = 0; e < cdim; ++e) {
            invJExT[d*cdim+e] = rotM[d*cdim+e];
          }
        }
      }
      if (centroidEx) {
        volExT = volEx;
        for (d = 0; d < cdim; ++d) {
          centroidExT[d]  = centroidEx[d];
          normalExT[d]    = normalEx[d];
        }
        for (d = 0; d < cdim; ++d) {
          centroidExT[d] *= scale;
          centroidExT[d] += PetscRealPart(trans[d]);
          normalExT[d]   /= scale;
          /* Only scale dimensions in the manifold */
          if (d < dim) volExT  *= scale;
        }
        /* Do scaling and translation before rotation, so that we can leave out the normal dimension for lower dimensional manifolds */
        for (d = 0; d < cdim; ++d) {
          for (e = 0, rot[d] = 0.0; e < cdim; ++e) {
            rot[d] += R[d*cdim+e] * centroidExT[e];
          }
        }
        for (d = 0; d < cdim; ++d) centroidExT[d] = rot[d];
        for (d = 0; d < cdim; ++d) {
          for (e = 0, rot[d] = 0.0; e < cdim; ++e) {
            rot[d] += R[d*cdim+e] * normalExT[e];
          }
        }
        for (d = 0; d < cdim; ++d) normalExT[d] = rot[d];
        for (d = 0, norm = 0.0; d < cdim; ++d) norm += PetscSqr(normalExT[d]);
        norm = PetscSqrtReal(norm);
        for (d = 0; d < cdim; ++d) normalExT[d] /= norm;
      }
      for (d = 0; d < cdim; ++d) {
        for (p = 0; p < numCorners; ++p) {
          newCoords[p*cdim+d] *= scale;
          newCoords[p*cdim+d] += trans[d];
        }
      }
      for (p = 0; p < numCorners; ++p) {
        for (d = 0; d < cdim; ++d) {
          for (e = 0, rot[d] = 0.0; e < cdim; ++e) {
            rot[d] += R[d*cdim+e] * PetscRealPart(newCoords[p*cdim+e]);
          }
        }
        for (d = 0; d < cdim; ++d) newCoords[p*cdim+d] = rot[d];
      }

      ierr = ChangeCoordinates(dm, cdim, newCoords);CHKERRQ(ierr);
      if (v0Ex) {
        ierr = CheckFEMGeometry(dm, 0, cdim, v0ExT, JExT, invJExT, detJExT);CHKERRQ(ierr);
      }
      if (dim == depth && centroidEx) {
        ierr = CheckFVMGeometry(dm, cell, cdim, centroidExT, normalExT, volExT);CHKERRQ(ierr);
        if (faceCentroidEx) {
          ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            PetscInt off = c*cdim;

            faceVolExT = faceVolEx[c];
            for (d = 0; d < cdim; ++d) {
              faceCentroidExT[d]  = faceCentroidEx[off+d];
              faceNormalExT[d]    = faceNormalEx[off+d];
            }
            for (d = 0; d < cdim; ++d) {
              faceCentroidExT[d] *= scale;
              faceCentroidExT[d] += PetscRealPart(trans[d]);
              faceNormalExT[d]   /= scale;
              /* Only scale dimensions in the manifold */
              if (d < dim-1) {
                faceVolExT *= scale;
              }
            }
            for (d = 0; d < cdim; ++d) {
              for (e = 0, rot[d] = 0.0; e < cdim; ++e) {
                rot[d] += R[d*cdim+e] * faceCentroidExT[e];
              }
            }
            for (d = 0; d < cdim; ++d) faceCentroidExT[d] = rot[d];
            for (d = 0; d < cdim; ++d) {
              for (e = 0, rot[d] = 0.0; e < cdim; ++e) {
                rot[d] += R[d*cdim+e] * faceNormalExT[e];
              }
            }
            for (d = 0; d < cdim; ++d) faceNormalExT[d] = rot[d];
            for (d = 0, norm = 0.0; d < cdim; ++d) norm += PetscSqr(faceNormalExT[d]);
            norm = PetscSqrtReal(norm);
            for (d = 0; d < cdim; ++d) faceNormalExT[d] /= norm;

            ierr = CheckFVMGeometry(dm, cone[c], cdim, faceCentroidExT, faceNormalExT, faceVolExT);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&ang2);CHKERRQ(ierr);
    ierr = PetscFree2(origCoords, newCoords);CHKERRQ(ierr);
    ierr = PetscFree5(v0ExT, JExT, invJExT, centroidExT, normalExT);CHKERRQ(ierr);
    ierr = PetscFree2(faceCentroidExT, faceNormalExT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTriangle(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscInt       dim;
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
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
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

    ierr = CheckCell(dm, 0, transform, v0Ex, JEx, invJEx, detJEx, centroidEx, normalEx, volEx, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  /* Move to 3D: Check reference geometry: determinant is scaled by reference volume (2.0) */
  dim = 3;
  {
    PetscScalar vertexCoords[9] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0};
    PetscReal   v0Ex[3]         = {-1.0, -1.0, 0.0};
    PetscReal   JEx[9]          = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal   invJEx[9]       = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal   detJEx          = 1.0;
    PetscReal   centroidEx[3]   = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.), 0.0};
    PetscReal   normalEx[3]     = {0.0, 0.0, 1.0};
    PetscReal   volEx           = 2.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckCell(dm, 0, transform, v0Ex, JEx, invJEx, detJEx, centroidEx, normalEx, volEx, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestQuadrilateral(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscInt       dim;
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
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
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

    ierr = CheckCell(dm, 0, transform, v0Ex, JEx, invJEx, detJEx, centroidEx, normalEx, volEx, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  /* Move to 3D: Check reference geometry: determinant is scaled by reference volume (4.0) */
  dim = 3;
  {
    PetscScalar vertexCoords[12] = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};
    PetscReal   v0Ex[3]          = {-1.0, -1.0, 0.0};
    PetscReal   JEx[9]           = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal   invJEx[9]        = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal   detJEx           = 1.0;
    PetscReal   centroidEx[3]    = {0.0, 0.0, 0.0};
    PetscReal   normalEx[3]      = {0.0, 0.0, 1.0};
    PetscReal   volEx            = 4.0;

    ierr = ChangeCoordinates(dm, dim, vertexCoords);CHKERRQ(ierr);
    ierr = CheckCell(dm, 0, transform, v0Ex, JEx, invJEx, detJEx, centroidEx, normalEx, volEx, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTetrahedron(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscInt       dim;
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
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
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

    ierr = CheckCell(dm, 0, transform, v0Ex, JEx, invJEx, detJEx, centroidEx, normalEx, volEx, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestHexahedron(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscInt       dim;
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
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
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

    ierr = CheckCell(dm, 0, transform, v0Ex, JEx, invJEx, detJEx, centroidEx, normalEx, volEx, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This wedge is a tensor product cell, rather than a normal wedge */
static PetscErrorCode TestWedge(MPI_Comm comm, PetscBool interpolate, PetscBool transform)
{
  DM             dm;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create reference wedge */
  dim  = 3;
  ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Triangular Prism");CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    PetscInt    numPoints[2]        = {6, 1};
    PetscInt    coneSize[7]         = {6, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[6]            = {1, 2, 3, 4, 5, 6};
    PetscInt    coneOrientations[6] = {0, 0, 0, 0, 0, 0};
    PetscScalar vertexCoords[18]    = {-1.0, -1.0, -1.0,   1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,
                                       -1.0, -1.0,  1.0,   1.0, -1.0,  1.0,  -1.0, 1.0,  1.0};

    ierr = DMPlexCreateFromDAG(dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    if (interpolate) {
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
    }
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  }
  /* Check reference geometry: determinant is scaled by reference volume 4.0 */
  {
#if 0
    /* FEM geometry is not functional for wedges */
    PetscReal v0Ex[3]   = {-1.0, -1.0, -1.0};
    PetscReal JEx[9]    = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal invJEx[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PetscReal detJEx    = 1.0;
#endif

    if (interpolate) {
      PetscReal       centroidEx[3]      = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.), 0.0};
      PetscReal       normalEx[3]        = {0.0, 0.0, 0.0};
      PetscReal       volEx              = 4.0;
      PetscReal       faceVolEx[5]       = {2.0, 2.0, 4.0, PETSC_SQRT2*4.0, 4.0};
      PetscReal       faceNormalEx[15]   = {0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, -1.0, 0.0,  PETSC_SQRT2/2.0, PETSC_SQRT2/2.0, 0.0,  -1.0, 0.0, 0.0};
      PetscReal       faceCentroidEx[15] = {-((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.), -1.0,
                                            -((PetscReal)1.)/((PetscReal)3.), -((PetscReal)1.)/((PetscReal)3.),  1.0,
                                            0.0, -1.0, 0.0,  0.0, 0.0, 0.0,  -1.0, 0.0, 0.0};

      ierr = CheckCell(dm, 0, transform, NULL, NULL, NULL, 0.0, centroidEx, normalEx, volEx, faceCentroidEx, faceNormalEx, faceVolEx);CHKERRQ(ierr);
    }
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
    ierr = TestWedge(PETSC_COMM_SELF, user.interpolate, user.transform);CHKERRQ(ierr);
  } else if (user.runType == RUN_FILE) {
    PetscInt dim, cStart, cEnd, c;

    ierr = DMGetDimension(user.dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(user.dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = 0; c < cEnd-cStart; ++c) {
      ierr = CheckCell(user.dm, c+cStart, PETSC_FALSE, &user.v0[c*dim], &user.J[c*dim*dim], &user.invJ[c*dim*dim], user.detJ[c], &user.centroid[c*dim], &user.normal[c*dim], user.vol[c], NULL, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree7(user.v0,user.J,user.invJ,user.detJ,user.centroid,user.normal,user.vol);CHKERRQ(ierr);
    ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  } else if (user.runType == RUN_DISPLAY) {
    DM                 gdm, dmCell;
    Vec                cellgeom, facegeom;
    const PetscScalar *cgeom;
    PetscInt           dim, d, cStart, cEnd, cEndInterior, c;

    ierr = DMGetCoordinateDim(user.dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexConstructGhostCells(user.dm, NULL, NULL, &gdm);CHKERRQ(ierr);
    if (gdm) {
      ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
      user.dm = gdm;
    }
    ierr = DMPlexComputeGeometryFVM(user.dm, &cellgeom, &facegeom);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(user.dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(user.dm, &cEndInterior, NULL);CHKERRQ(ierr);
    if (cEndInterior >= 0) cEnd = cEndInterior;
    ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    for (c = 0; c < cEnd-cStart; ++c) {
      PetscFVCellGeom *cg;

      ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %4D: Centroid (", c);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        if (d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
        ierr = PetscPrintf(PETSC_COMM_SELF, "%12.2g", cg->centroid[d]);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_SELF, ") Vol %12.2g\n", cg->volume);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    ierr = VecDestroy(&cellgeom);CHKERRQ(ierr);
    ierr = VecDestroy(&facegeom);CHKERRQ(ierr);
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
    args: -run_type file -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/simpleblock-100.exo -dm_view ascii::ascii_info_detail -v0 -1.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,0.5 -J 0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0 -invJ 0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0 -detJ 0.125,0.125,0.125
  test:
    suffix: 5
    requires: exodusii
    args: -interpolate -run_type file -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/simpleblock-100.exo -dm_view ascii::ascii_info_detail -v0 -1.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,0.5 -J 0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0 -invJ 0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0 -detJ 0.125,0.125,0.125 -centroid -1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0 -normal 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 -vol 1.0,1.0,1.0

TEST*/
