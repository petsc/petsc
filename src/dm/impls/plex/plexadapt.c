#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#if defined(PETSC_HAVE_PRAGMATIC)
#include <pragmatic/cpragmatic.h>
#endif

static PetscErrorCode DMPlexLabelToVolumeConstraint(DM dm, DMLabel adaptLabel, PetscInt cStart, PetscInt cEnd, PetscReal refRatio, PetscReal maxVolumes[])
{
  PetscInt       dim, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  refRatio = refRatio == PETSC_DEFAULT ? (PetscReal) ((PetscInt) 1 << dim) : refRatio;
  for (c = cStart; c < cEnd; c++) {
    PetscReal vol;
    PetscInt  closureSize = 0, cl;
    PetscInt *closure     = NULL;
    PetscBool anyRefine   = PETSC_FALSE;
    PetscBool anyCoarsen  = PETSC_FALSE;
    PetscBool anyKeep     = PETSC_FALSE;

    ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
    maxVolumes[c - cStart] = vol;
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];
      PetscInt       refFlag;

      ierr = DMLabelGetValue(adaptLabel, point, &refFlag);CHKERRQ(ierr);
      switch (refFlag) {
      case DM_ADAPT_REFINE:
        anyRefine  = PETSC_TRUE;break;
      case DM_ADAPT_COARSEN:
        anyCoarsen = PETSC_TRUE;break;
      case DM_ADAPT_KEEP:
        anyKeep    = PETSC_TRUE;break;
      case DM_ADAPT_DETERMINE:
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "DMPlex does not support refinement flag %D\n", refFlag);
      }
      if (anyRefine) break;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (anyRefine) {
      maxVolumes[c - cStart] = vol / refRatio;
    } else if (anyKeep) {
      maxVolumes[c - cStart] = vol;
    } else if (anyCoarsen) {
      maxVolumes[c - cStart] = vol * refRatio;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLabelToMetricConstraint(DM dm, DMLabel adaptLabel, PetscInt cStart, PetscInt cEnd, PetscInt vStart, PetscInt vEnd, PetscReal refRatio, Vec *metricVec)
{
  DM              udm, coordDM;
  PetscSection    coordSection;
  Vec             coordinates, mb, mx;
  Mat             A;
  PetscScalar    *metric, *eqns;
  const PetscReal coarseRatio = refRatio == PETSC_DEFAULT ? PetscSqr(0.5) : 1/refRatio;
  PetscInt        dim, Nv, Neq, c, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  ierr = DMGetLocalSection(coordDM, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  Nv   = vEnd - vStart;
  ierr = VecCreateSeq(PETSC_COMM_SELF, Nv*PetscSqr(dim), metricVec);CHKERRQ(ierr);
  ierr = VecGetArray(*metricVec, &metric);CHKERRQ(ierr);
  Neq  = (dim*(dim+1))/2;
  ierr = PetscMalloc1(PetscSqr(Neq), &eqns);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, Neq, Neq, eqns, &A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A, &mx, &mb);CHKERRQ(ierr);
  ierr = VecSet(mb, 1.0);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *sol;
    PetscScalar       *cellCoords = NULL;
    PetscReal          e[3], vol;
    const PetscInt    *cone;
    PetscInt           coneSize, cl, i, j, d, r;

    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &cellCoords);CHKERRQ(ierr);
    /* Only works for simplices */
    for (i = 0, r = 0; i < dim+1; ++i) {
      for (j = 0; j < i; ++j, ++r) {
        for (d = 0; d < dim; ++d) e[d] = PetscRealPart(cellCoords[i*dim+d] - cellCoords[j*dim+d]);
        /* FORTRAN ORDERING */
        switch (dim) {
        case 2:
          eqns[0*Neq+r] = PetscSqr(e[0]);
          eqns[1*Neq+r] = 2.0*e[0]*e[1];
          eqns[2*Neq+r] = PetscSqr(e[1]);
          break;
        case 3:
          eqns[0*Neq+r] = PetscSqr(e[0]);
          eqns[1*Neq+r] = 2.0*e[0]*e[1];
          eqns[2*Neq+r] = 2.0*e[0]*e[2];
          eqns[3*Neq+r] = PetscSqr(e[1]);
          eqns[4*Neq+r] = 2.0*e[1]*e[2];
          eqns[5*Neq+r] = PetscSqr(e[2]);
          break;
        }
      }
    }
    ierr = MatSetUnfactored(A);CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &cellCoords);CHKERRQ(ierr);
    ierr = MatLUFactor(A, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatSolve(A, mb, mx);CHKERRQ(ierr);
    ierr = VecGetArrayRead(mx, &sol);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) {
      const PetscInt v = cone[cl] - vStart;

      if (dim == 2) {
        metric[v*4+0] += vol*coarseRatio*sol[0];
        metric[v*4+1] += vol*coarseRatio*sol[1];
        metric[v*4+2] += vol*coarseRatio*sol[1];
        metric[v*4+3] += vol*coarseRatio*sol[2];
      } else {
        metric[v*9+0] += vol*coarseRatio*sol[0];
        metric[v*9+1] += vol*coarseRatio*sol[1];
        metric[v*9+3] += vol*coarseRatio*sol[1];
        metric[v*9+2] += vol*coarseRatio*sol[2];
        metric[v*9+6] += vol*coarseRatio*sol[2];
        metric[v*9+4] += vol*coarseRatio*sol[3];
        metric[v*9+5] += vol*coarseRatio*sol[4];
        metric[v*9+7] += vol*coarseRatio*sol[4];
        metric[v*9+8] += vol*coarseRatio*sol[5];
      }
    }
    ierr = VecRestoreArrayRead(mx, &sol);CHKERRQ(ierr);
  }
  for (v = 0; v < Nv; ++v) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscReal       vol, totVol = 0.0;

    ierr = DMPlexGetSupport(udm, v+vStart, &support);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(udm, v+vStart, &supportSize);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {ierr = DMPlexComputeCellGeometryFVM(dm, support[s], &vol, NULL, NULL);CHKERRQ(ierr); totVol += vol;}
    for (s = 0; s < PetscSqr(dim); ++s) metric[v*PetscSqr(dim)+s] /= totVol;
  }
  ierr = PetscFree(eqns);CHKERRQ(ierr);
  ierr = VecRestoreArray(*metricVec, &metric);CHKERRQ(ierr);
  ierr = VecDestroy(&mx);CHKERRQ(ierr);
  ierr = VecDestroy(&mb);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Contains the list of registered DMPlexGenerators routines
*/
extern PetscFunctionList DMPlexGenerateList;

struct _n_PetscFunctionList {
  PetscErrorCode    (*generate)(DM, PetscBool, DM*);
  PetscErrorCode    (*refine)(DM,PetscReal*, DM*);
  char              *name;               /* string to identify routine */
  PetscInt          dim;
  PetscFunctionList next;                /* next pointer */
};

PetscErrorCode DMPlexRefine_Internal(DM dm, DMLabel adaptLabel, DM *dmRefined)
{
  PetscErrorCode    (*refinementFunc)(const PetscReal [], PetscReal *);
  PetscReal         refinementLimit;
  PetscInt          dim, cStart, cEnd;
  char              genname[1024], *name = NULL;
  PetscBool         flg, localized;
  PetscErrorCode    ierr;
  PetscErrorCode    (*refine)(DM,PetscReal*,DM*);
  PetscFunctionList fl;
  PetscReal         *maxVolumes;
  PetscInt          c;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  ierr = DMPlexGetRefinementLimit(dm, &refinementLimit);CHKERRQ(ierr);
  ierr = DMPlexGetRefinementFunction(dm, &refinementFunc);CHKERRQ(ierr);
  if (refinementLimit == 0.0 && !refinementFunc && !adaptLabel) PetscFunctionReturn(0);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_generator", genname, sizeof(genname), &flg);CHKERRQ(ierr);
  if (flg) name = genname;

  fl = DMPlexGenerateList;
  if (name) {
    while (fl) {
      ierr = PetscStrcmp(fl->name,name,&flg);CHKERRQ(ierr);
      if (flg) {
        refine = fl->refine;
        goto gotit;
      }
      fl = fl->next;
    }
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Grid refiner %s not registered",name);
  } else {
    while (fl) {
      if (dim-1 == fl->dim) {
        refine = fl->refine;
        goto gotit;
      }
      fl = fl->next;
    }
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No grid refiner of dimension %D registered",dim);
  }

  gotit: switch (dim) {
  case 2:
      ierr = PetscMalloc1(cEnd - cStart, &maxVolumes);CHKERRQ(ierr);
      if (adaptLabel) {
        ierr = DMPlexLabelToVolumeConstraint(dm, adaptLabel, cStart, cEnd, PETSC_DEFAULT, maxVolumes);CHKERRQ(ierr);
      } else if (refinementFunc) {
        for (c = cStart; c < cEnd; ++c) {
          PetscReal vol, centroid[3];
          PetscReal maxVol;

          ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, centroid, NULL);CHKERRQ(ierr);
          ierr = (*refinementFunc)(centroid, &maxVol);CHKERRQ(ierr);
          maxVolumes[c - cStart] = (double) maxVol;
        }
      } else {
        for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = refinementLimit;
      }
      ierr = (*refine)(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
      ierr = PetscFree(maxVolumes);CHKERRQ(ierr);
    break;
  case 3:
      ierr = PetscMalloc1(cEnd - cStart, &maxVolumes);CHKERRQ(ierr);
      if (adaptLabel) {
        ierr = DMPlexLabelToVolumeConstraint(dm, adaptLabel, cStart, cEnd, PETSC_DEFAULT, maxVolumes);CHKERRQ(ierr);
      } else if (refinementFunc) {
        for (c = cStart; c < cEnd; ++c) {
          PetscReal vol, centroid[3];

          ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, centroid, NULL);CHKERRQ(ierr);
          ierr = (*refinementFunc)(centroid, &maxVolumes[c-cStart]);CHKERRQ(ierr);
        }
      } else {
        for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = refinementLimit;
      }
      ierr = (*refine)(dm, maxVolumes, dmRefined);CHKERRQ(ierr);
      ierr = PetscFree(maxVolumes);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mesh refinement in dimension %D is not supported.", dim);
  }
  ierr = DMCopyBoundary(dm, *dmRefined);CHKERRQ(ierr);
  if (localized) {ierr = DMLocalizeCoordinates(*dmRefined);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCoarsen_Internal(DM dm, DMLabel adaptLabel, DM *dmCoarsened)
{
  Vec            metricVec;
  PetscInt       cStart, cEnd, vStart, vEnd;
  DMLabel        bdLabel = NULL;
  char           bdLabelName[PETSC_MAX_PATH_LEN];
  PetscBool      localized, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexLabelToMetricConstraint(dm, adaptLabel, cStart, cEnd, vStart, vEnd, PETSC_DEFAULT, &metricVec);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, dm->hdr.prefix, "-dm_plex_coarsen_bd_label", bdLabelName, sizeof(bdLabelName), &flg);CHKERRQ(ierr);
  if (flg) {ierr = DMGetLabel(dm, bdLabelName, &bdLabel);CHKERRQ(ierr);}
  ierr = DMAdaptMetric_Plex(dm, metricVec, bdLabel, dmCoarsened);CHKERRQ(ierr);
  ierr = VecDestroy(&metricVec);CHKERRQ(ierr);
  if (localized) {ierr = DMLocalizeCoordinates(*dmCoarsened);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptLabel_Plex(DM dm, DMLabel adaptLabel, DM *dmAdapted)
{
  IS              flagIS;
  const PetscInt *flags;
  PetscInt        defFlag, minFlag, maxFlag, numFlags, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetDefaultValue(adaptLabel, &defFlag);CHKERRQ(ierr);
  minFlag = defFlag;
  maxFlag = defFlag;
  ierr = DMLabelGetValueIS(adaptLabel, &flagIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(flagIS, &numFlags);CHKERRQ(ierr);
  ierr = ISGetIndices(flagIS, &flags);CHKERRQ(ierr);
  for (f = 0; f < numFlags; ++f) {
    const PetscInt flag = flags[f];

    minFlag = PetscMin(minFlag, flag);
    maxFlag = PetscMax(maxFlag, flag);
  }
  ierr = ISRestoreIndices(flagIS, &flags);CHKERRQ(ierr);
  ierr = ISDestroy(&flagIS);CHKERRQ(ierr);
  {
    PetscInt minMaxFlag[2], minMaxFlagGlobal[2];

    minMaxFlag[0] =  minFlag;
    minMaxFlag[1] = -maxFlag;
    ierr = MPI_Allreduce(minMaxFlag, minMaxFlagGlobal, 2, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    minFlag =  minMaxFlagGlobal[0];
    maxFlag = -minMaxFlagGlobal[1];
  }
  if (minFlag == maxFlag) {
    switch (minFlag) {
    case DM_ADAPT_DETERMINE:
      *dmAdapted = NULL;break;
    case DM_ADAPT_REFINE:
      ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMRefine(dm, MPI_COMM_NULL, dmAdapted);CHKERRQ(ierr);break;
    case DM_ADAPT_COARSEN:
      ierr = DMCoarsen(dm, MPI_COMM_NULL, dmAdapted);CHKERRQ(ierr);break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP,"DMPlex does not support refinement flag %D\n", minFlag);
    }
  } else {
    ierr = DMPlexSetRefinementUniform(dm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexRefine_Internal(dm, adaptLabel, dmAdapted);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  DMAdaptMetric_Plex - Generates a new mesh conforming to a metric field.

  Input Parameters:
+ dm - The DM object
. vertexMetric - The metric to which the mesh is adapted, defined vertex-wise in a LOCAL vector
- bdLabel - Label for boundary tags which are preserved in dmNew, or NULL. Should not be named "_boundary_".

  Output Parameter:
. dmNew  - the new DM

  Level: advanced

.seealso: DMCoarsen(), DMRefine()
*/
PetscErrorCode DMAdaptMetric_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
{
#if defined(PETSC_HAVE_PRAGMATIC)
  MPI_Comm           comm;
  const char        *bdName = "_boundary_";
#if 0
  DM                 odm = dm;
#endif
  DM                 udm, cdm;
  DMLabel            bdLabelFull;
  const char        *bdLabelName;
  IS                 bdIS, globalVertexNum;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull, *gV;
  PetscInt          *bdFaces, *bdFaceIds, *l2gv;
  PetscReal         *x, *y, *z, *metric;
  PetscInt          *cells;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, numLocVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize;
  PetscBool          flg;
  DMLabel            bdLabelNew;
  PetscReal         *coordsNew;
  PetscInt          *bdTags;
  PetscReal         *xNew[3] = {NULL, NULL, NULL};
  PetscInt          *cellsNew;
  PetscInt           d, numCellsNew, numVerticesNew;
  PetscInt           numCornersNew, fStart, fEnd;
  PetscMPIInt        numProcs;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Check for FEM adjacency flags */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  /* Add overlap for Pragmatic */
#if 0
  /* Check for overlap by looking for cell in the SF */
  if (!overlapped) {
    ierr = DMPlexDistributeOverlap(odm, 1, NULL, &dm);CHKERRQ(ierr);
    if (!dm) {dm = odm; ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);}
  }
#endif
  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices*PetscSqr(dim), &metric, numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }
  ierr = PetscCalloc1(numVertices, &l2gv);CHKERRQ(ierr);
  ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
  ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  for (v = 0, numLocVertices = 0; v < numVertices; ++v) {
    if (gV[v] >= 0) ++numLocVertices;
    l2gv[v] = gV[v] < 0 ? -(gV[v]+1) : gV[v];
  }
  ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    x[v-vStart] = PetscRealPart(coords[off+0]);
    if (dim > 1) y[v-vStart] = PetscRealPart(coords[off+1]);
    if (dim > 2) z[v-vStart] = PetscRealPart(coords[off+2]);
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  /* Get boundary mesh */
  ierr = DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabelFull);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, bdLabelFull);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(bdLabelFull, 1, &bdIS);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(bdLabelFull, 1, &numBdFaces);CHKERRQ(ierr);
  ierr = ISGetIndices(bdIS, &bdFacesFull);CHKERRQ(ierr);
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    ierr = DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds);CHKERRQ(ierr);
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    ierr = DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (bdLabel) {ierr = DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]);CHKERRQ(ierr);}
    else         {bdFaceIds[f] = 1;}
  }
  ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&bdLabelFull);CHKERRQ(ierr);
  /* Get metric */
  ierr = VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view");CHKERRQ(ierr);
  ierr = VecGetArrayRead(vertexMetric, &met);CHKERRQ(ierr);
  for (v = 0; v < (vEnd-vStart)*PetscSqr(dim); ++v) metric[v] = PetscRealPart(met[v]);
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);
#if 0
  /* Destroy overlap mesh */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
#endif
  /* Create new mesh */
  switch (dim) {
  case 2:
    pragmatic_2d_mpi_init(&numVertices, &numCells, cells, x, y, l2gv, numLocVertices, comm);break;
  case 3:
    pragmatic_3d_mpi_init(&numVertices, &numCells, cells, x, y, z, l2gv, numLocVertices, comm);break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %D", dim);
  }
  pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
  pragmatic_set_metric(metric);
  pragmatic_adapt(((DM_Plex *) dm->data)->remeshBd ? 1 : 0);
  ierr = PetscFree(l2gv);CHKERRQ(ierr);
  /* Read out mesh */
  pragmatic_get_info_mpi(&numVerticesNew, &numCellsNew);
  ierr = PetscMalloc1(numVerticesNew*dim, &coordsNew);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    numCornersNew = 3;
    ierr = PetscMalloc2(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1]);CHKERRQ(ierr);
    pragmatic_get_coords_2d_mpi(xNew[0], xNew[1]);
    break;
  case 3:
    numCornersNew = 4;
    ierr = PetscMalloc3(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1], numVerticesNew, &xNew[2]);CHKERRQ(ierr);
    pragmatic_get_coords_3d_mpi(xNew[0], xNew[1], xNew[2]);
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %D", dim);
  }
  for (v = 0; v < numVerticesNew; ++v) {for (d = 0; d < dim; ++d) coordsNew[v*dim+d] = xNew[d][v];}
  ierr = PetscMalloc1(numCellsNew*(dim+1), &cellsNew);CHKERRQ(ierr);
  pragmatic_get_elements(cellsNew);
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, coordsNew, NULL, dmNew);CHKERRQ(ierr);
  /* Read out boundary label */
  pragmatic_get_boundaryTags(&bdTags);
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    /* Only for simplicial meshes */
    coff = (c-cStart)*(dim+1);
    /* d is the local cell number of the vertex opposite to the face we are marking */
    for (d = 0; d < dim+1; ++d) {
      if (bdTags[coff+d]) {
        const PetscInt  perm[4][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {1, 2, 0, -1}, {3, 2, 1, 0}}; /* perm[d] = face opposite */
        const PetscInt *cone;

        /* Mark face opposite to this vertex: This pattern is specified in DMPlexGetRawFaces_Internal() */
        ierr = DMPlexGetCone(*dmNew, c, &cone);CHKERRQ(ierr);
        ierr = DMLabelSetValue(bdLabelNew, cone[perm[dim][d]], bdTags[coff+d]);CHKERRQ(ierr);
      }
    }
  }
  /* Cleanup */
  switch (dim) {
  case 2: ierr = PetscFree2(xNew[0], xNew[1]);CHKERRQ(ierr);break;
  case 3: ierr = PetscFree3(xNew[0], xNew[1], xNew[2]);CHKERRQ(ierr);break;
  }
  ierr = PetscFree(cellsNew);CHKERRQ(ierr);
  ierr = PetscFree5(x, y, z, metric, cells);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree(coordsNew);CHKERRQ(ierr);
  pragmatic_finalize();
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Remeshing needs external package support.\nPlease reconfigure with --download-pragmatic.");
#endif
}
