#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#if defined(PETSC_HAVE_PRAGMATIC)
#include <pragmatic/cpragmatic.h>
#endif
#if defined(PETSC_HAVE_MMG)
#include <mmg/libmmg.h>
#endif
#if defined(PETSC_HAVE_PARMMG)
#include <parmmg/libparmmg.h>
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
extern PlexGeneratorFunctionList DMPlexGenerateList;

PetscErrorCode DMPlexRefine_Internal(DM dm, DMLabel adaptLabel, DM *dmRefined)
{
  PlexGeneratorFunctionList fl;
  PetscErrorCode          (*refine)(DM,PetscReal*,DM*);
  PetscErrorCode          (*adapt)(DM,DMLabel,DM*);
  PetscErrorCode          (*refinementFunc)(const PetscReal [], PetscReal *);
  char                      genname[PETSC_MAX_PATH_LEN], *name = NULL;
  PetscReal                 refinementLimit;
  PetscReal                *maxVolumes;
  PetscInt                  dim, cStart, cEnd, c;
  PetscBool                 flg, flg2, localized;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  ierr = DMPlexGetRefinementLimit(dm, &refinementLimit);CHKERRQ(ierr);
  ierr = DMPlexGetRefinementFunction(dm, &refinementFunc);CHKERRQ(ierr);
  if (refinementLimit == 0.0 && !refinementFunc && !adaptLabel) PetscFunctionReturn(0);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_adaptor", genname, sizeof(genname), &flg);CHKERRQ(ierr);
  if (flg) name = genname;
  else {
    ierr = PetscOptionsGetString(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_generator", genname, sizeof(genname), &flg2);CHKERRQ(ierr);
    if (flg2) name = genname;
  }

  fl = DMPlexGenerateList;
  if (name) {
    while (fl) {
      ierr = PetscStrcmp(fl->name,name,&flg);CHKERRQ(ierr);
      if (flg) {
        refine = fl->refine;
        adapt  = fl->adaptlabel;
        goto gotit;
      }
      fl = fl->next;
    }
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Grid refiner %s not registered",name);
  } else {
    while (fl) {
      if (fl->dim < 0 || dim-1 == fl->dim) {
        refine = fl->refine;
        adapt  = fl->adaptlabel;
        goto gotit;
      }
      fl = fl->next;
    }
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No grid refiner of dimension %D registered",dim);
  }

  gotit:
  switch (dim) {
    case 1:
    case 2:
    case 3:
      if (adapt) {
        ierr = (*adapt)(dm, adaptLabel, dmRefined);CHKERRQ(ierr);
      } else {
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
      }
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mesh refinement in dimension %D is not supported.", dim);
  }
  ((DM_Plex *) (*dmRefined)->data)->useHashLocation = ((DM_Plex *) dm->data)->useHashLocation;
  ierr = DMCopyDisc(dm, *dmRefined);CHKERRQ(ierr);
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
  ((DM_Plex *) (*dmCoarsened)->data)->useHashLocation = ((DM_Plex *) dm->data)->useHashLocation;
  ierr = DMCopyDisc(dm, *dmCoarsened);CHKERRQ(ierr);
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
    ierr = MPI_Allreduce(minMaxFlag, minMaxFlagGlobal, 2, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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


PetscErrorCode DMAdaptMetricPragmatic_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
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
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRMPI(ierr);
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
  numCells = cEnd - cStart;
  if (numCells == 0) {
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot perform mesh adaptation because process %d does not own any cells.", rank);
  }
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

int my_increasing_comparison_function(const void *left, const void *right, void *ctx) {
    PetscInt l = *(PetscInt*) left, r = *(PetscInt *) right;
    return (l < r) ? -1 : (l > r);
}

PetscErrorCode DMAdaptMetricMMG_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
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
  IS                 bdIS; // IS : index set: ensemble d'indices ordonnés BDIS pour les conditions aux limites , global vertex num pour le //
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met; // double
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *vertices, *metric, *verticesNew;
  PetscInt          *cells;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize, fStart, fEnd,j,k;
  PetscBool          flg;
  DMLabel            bdLabelNew;
  PetscInt          *cellsNew;
  PetscInt           numCellsNew, numVerticesNew;
  PetscInt           numCornersNew;
  PetscMPIInt        numProcs,me;
  PetscErrorCode     ierr;
  // nos variables
  PetscInt * tab_cl_verticies, * tab_cl_triangles, *tab_cl_verticies_new, *tab_cl_cells_new;
  PetscInt * tab_areCorners, * tab_areRequiredCells, *tab_areRequiredVerticies;
  PetscInt * faces, * tab_cl_faces, * tab_areRidges, * tab_areRequiredFaces;
  PetscInt numFacesNew;
  MMG5_pMesh mmg_mesh = NULL;
  MMG5_pSol mmg_metric = NULL;
  PetscInt i;
  // Pour le parallèle
  PMMG_pParMesh parmesh = NULL;
  PetscSF starforest;
  const PetscInt *gV;
  PetscInt numleaves=0, numroots=0, num_communicators=0,ct=0,ctt=0,p;
  PetscInt *flags_proc,*communicators_local,*communicators_global,*communicators_local_new;
  PetscInt *num_per_proc, *offset;
  IS globalVertexNum;
  PetscInt *gv_new, *ranks_own,numVerticesNewNew=0;
  PetscReal *VerticesNewNew;

  PetscFunctionBegin;

  // 0. Début du programme 
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &me);CHKERRQ(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  
  // 0. Dans le cas parallèlle il faut récupérer les sommets aux interfaces
  // et le starforest
  if(numProcs>1){
    ierr = DMPlexDistribute(dm, 0, NULL, &udm);dm=udm;
  }

  // 1. Chercher les informations dans le maillage
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr); /*Récupération des cellulles*/
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr); /*Récupération des sommets*/
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr); /*On regarde uniquement les cellulles et les sommets*/
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr); /*regarde la taille maximum du cône ie combien de sommets au max a une cellule donc si ce sont des triangles, des quadrilatères*/
  numCells    = cEnd - cStart; /*indice du début moins indice de fin des tranches*/
  numVertices = vEnd - vStart;

  printf("DEBUG %d : nombre de tétra %d \n",me, numCells);

  // 2. Récupération des cellules
  ierr = PetscCalloc1(numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) { /*boucle sur les cellules*/
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr); /*récupération de la taille du cone*/
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr); /*récupération du cône*/
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart+1; /*translation du tableau*/
  }

  // 3. Récupération de tous les sommets
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  if (dim==2) {ierr=PetscCalloc2(numVertices*3, &metric,2*numVertices, &vertices);CHKERRQ(ierr);}
  if (dim==3) {ierr=PetscCalloc2(numVertices*6, &metric,3*numVertices, &vertices);CHKERRQ(ierr);}

  for (v = 0; v < vEnd-vStart; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v+vStart, &off);CHKERRQ(ierr);
    if (dim==2) {
      vertices[2*v]=PetscRealPart(coords[off+0]);
      vertices[2*v+1]=PetscRealPart(coords[off+1]);
    }
    else if (dim==3) {
      vertices[3*v]=PetscRealPart(coords[off+0]);
      vertices[3*v+1]=PetscRealPart(coords[off+1]);
      vertices[3*v+2]=PetscRealPart(coords[off+2]);
    }
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);

  // 3.5 récupération des interfaces
  if (numProcs>1) {
    
    PetscInt niranks,nranks;
    const PetscMPIInt *iranks,*ranks;
    const PetscInt *ioffset, *irootloc,*roffset,*rmine,*rremote;
    PetscSection       coordSection;
    DM                 cdm;
    PetscInt           off;
    Vec                coordinates;
    const PetscScalar  *coords;
    PetscScalar x, y, z;

    ierr = DMGetPointSF(dm,&starforest);CHKERRQ(ierr);
    ierr = PetscSFSetUp(starforest);CHKERRQ(ierr);
    ierr = PetscSFGetLeafRanks(starforest, &niranks, &iranks, &ioffset, &irootloc); CHKERRQ(ierr);
    ierr = PetscSFGetRootRanks(starforest, &nranks, &ranks, &roffset, &rmine, &rremote); CHKERRQ(ierr);

    ierr = PetscCalloc1(numVertices*numProcs,&flags_proc);CHKERRQ(ierr);
    ierr = PetscCalloc1(numProcs,&num_per_proc);CHKERRQ(ierr);
    ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
    ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    /// FIN TESTS

    // Recherche des feuilles
    for(p=0;p<nranks;p++) 
      for(i=roffset[p];i<roffset[p+1];i++)
        if (rmine[i] >= vStart && rmine[i] < vEnd && flags_proc[(rmine[i]-vStart)*numProcs+ranks[p]]==0) {
          numleaves++;
          flags_proc[(rmine[i]-vStart)*numProcs+ranks[p]]=1;
          num_per_proc[ranks[p]]++;
        }

    // recherche des racines
    for(p=0;p<niranks;p++) 
      for(i=ioffset[p];i<ioffset[p+1];i++)
        if (irootloc[i] >= vStart && irootloc[i] < vEnd && flags_proc[(irootloc[i]-vStart)*numProcs+iranks[p]]==0) {
          numroots++;
          flags_proc[(irootloc[i]-vStart)*numProcs+iranks[p]]=1;
          num_per_proc[iranks[p]]++;
        }

    printf("numleaves + numroots %d sur %d total : %d\n",numleaves+numroots,me,numVertices);

    // nombre de comm
    for (p=0;p<numProcs;p++) if (p!=me && num_per_proc[p] !=0) num_communicators++;
    
    // construction des tableaux non triés
    ierr = PetscCalloc1(num_communicators+1,&offset);CHKERRQ(ierr); offset[0]=0;
    ierr = PetscCalloc3(numroots+numleaves,&communicators_local,numroots+numleaves,&communicators_global,numroots+numleaves,&communicators_local_new); CHKERRQ(ierr);
    for(p=0;p<numProcs;p++) if (p!=me && num_per_proc[p] !=0) {
      for(i=0;i<numVertices;i++) if (flags_proc[i*numProcs+p]==1) {
        communicators_local[ct]=i+1;
        communicators_global[ct]=gV[i] < 0 ? -(gV[i]+1) : gV[i];
        communicators_global[ct]++;
        ct++;
      }
      offset[++ctt]=ct;
    }
    // tri du tableau global
    for(p=0;p<num_communicators;p++) {
      ierr = PetscTimSort(offset[p+1]-offset[p],&communicators_global[offset[p]],sizeof(PetscInt),my_increasing_comparison_function,NULL); CHKERRQ(ierr);
    }
    // reconstruction du tableau local
    for(p=0;p<numroots+numleaves;p++) {
      for(i=0;i<numroots+numleaves;i++) {
        PetscInt tempa=communicators_local[i]-1,tempb;
        tempb=gV[tempa] < 0 ? -(gV[tempa]+1) : gV[tempa];
        tempb++;
        if (tempb==communicators_global[p]) communicators_local_new[p]=communicators_local[i];
      }
    }

    for (p=0;p<numProcs;p++) {
      if (p==me) for(v=0;v<num_communicators+1;v++) {printf("%d",offset[v]);printf(" fin offst\n");}
      if (p==me) for(i=0;i<numroots+numleaves;i++) {
        ierr = PetscSectionGetOffset(coordSection, vStart+communicators_local_new[i]-1, &off);CHKERRQ(ierr);
        x = PetscRealPart(coords[off+0]);
        y = PetscRealPart(coords[off+1]);
        z = PetscRealPart(coords[off+2]);
        //printf("%d %d %d %d diff %d coo x %1.2f coo y %1.2f coo z %1.2f \n",me,i,communicators_local_new[i],communicators_global[i],gV[communicators_local_new[i]-1],x,y,z);
      }

      MPI_Barrier(PETSC_COMM_WORLD);
    }
    ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&udm);CHKERRQ(ierr);


  // 4. Récupératin des conditions aux bords
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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart+1;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (bdLabel) {ierr = DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]);CHKERRQ(ierr);}
    else         {bdFaceIds[f] = 1;}
  }
  ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&bdLabelFull);CHKERRQ(ierr);
  
  // 5. Récupération de la metric
  ierr = VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view");CHKERRQ(ierr);
  ierr = VecGetArrayRead(vertexMetric, &met);CHKERRQ(ierr);
  
  if (dim==2) {
    for (v = 0; v < (vEnd-vStart); ++v) { // *PetscSqr(dim)
      metric[3*v] = PetscRealPart(met[4*v]);
      metric[3*v+1] = PetscRealPart(met[4*v+1]);
      metric[3*v+2] = PetscRealPart(met[4*v+3]);
    }
  }
  else if (dim==3) {
    for (v = 0; v < (vEnd-vStart); ++v) { // *PetscSqr(dim)
      metric[6*v] = PetscRealPart(met[9*v]);
      metric[6*v+1] = PetscRealPart(met[9*v+1]);
      metric[6*v+2] = PetscRealPart(met[9*v+2]);
      metric[6*v+3] = PetscRealPart(met[9*v+4]);
      metric[6*v+4] = PetscRealPart(met[9*v+5]);
      metric[6*v+5] = PetscRealPart(met[9*v+8]);
    } 
  }
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);


  /* PARTIE 2: Transformation du maillage avec mmg*/
  ierr = PetscCalloc2(numVertices,&tab_cl_verticies,numCells,&tab_cl_triangles);CHKERRQ(ierr);

  if (numProcs==1) {
  switch(dim)
  {
  case 2:
    ierr = MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose,10); // quantité d'information à l'écran 10=toutes les informations
    ierr = MMG2D_Set_meshSize(mmg_mesh,numVertices,numCells,0,numBdFaces);
    // Passage des informations sur le maillage
    
    // géométrie
    ierr = MMG2D_Set_vertices(mmg_mesh,vertices,tab_cl_verticies);
    ierr = MMG2D_Set_triangles(mmg_mesh,cells,tab_cl_triangles);
    ierr = MMG2D_Set_edges(mmg_mesh,bdFaces,bdFaceIds);

    // métrique
    ierr = MMG2D_Set_solSize(mmg_mesh,mmg_metric,MMG5_Vertex,numVertices,MMG5_Tensor);
    for (i=0;i<numVertices;i++) MMG2D_Set_tensorSol(mmg_metric,metric[3*i],metric[3*i+1],metric[3*i+2],i+1);
    
    // Remaillage
    ierr =  MMG2D_saveMshMesh(mmg_mesh,NULL,"maillage_avant.msh");
    ierr = MMG2D_mmg2dlib(mmg_mesh,mmg_metric); printf("DEBUG remaillage 2D: %d \n", ierr);
    ierr = MMG2D_saveMshMesh(mmg_mesh,NULL,"maillage_apres.msh");
    break;

  case 3:
    ierr = MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose,10);
    ierr = MMG3D_Set_meshSize(mmg_mesh,numVertices,numCells,0,numBdFaces,0,0);

    ierr = MMG3D_Set_vertices(mmg_mesh,vertices,tab_cl_verticies);
    for (i=0;i<numCells;i++) ierr = MMG3D_Set_tetrahedron(mmg_mesh,cells[4*i+0],cells[4*i+1],cells[4*i+2],cells[4*i+3],0,i+1);
    ierr = MMG3D_Set_triangles(mmg_mesh,bdFaces,bdFaceIds);

    // Métrique
    ierr = MMG3D_Set_solSize(mmg_mesh,mmg_metric,MMG5_Vertex,numVertices,MMG5_Tensor);
    ierr = MMG3D_Set_tensorSols(mmg_metric,metric);

    // Remaillage
    ierr = MMG3D_saveMshMesh(mmg_mesh,NULL,"maillage_avant_3D.msh");
    ierr = MMG3D_mmg3dlib(mmg_mesh,mmg_metric); printf("DEBUG remaillage 3D: %d \n", ierr);  
    ierr = MMG3D_saveMshMesh(mmg_mesh,NULL,"maillage_apres_3D.msh");
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }
  }
  else
  {
    // Initialisation
    ierr = PMMG_Init_parMesh(PMMG_ARG_start,PMMG_ARG_ppParMesh,&parmesh,PMMG_ARG_pMesh,PMMG_ARG_pMet,PMMG_ARG_dim,3,PMMG_ARG_MPIComm,comm,PMMG_ARG_end);
    ierr = PMMG_Set_meshSize(parmesh,numVertices,numCells,0,numBdFaces,0,0);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_APImode,PMMG_APIDISTRIB_nodes);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_verbose,10);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_globalNum,1);

    // Maillage et métrique
    ierr = PMMG_Set_vertices(parmesh,vertices,tab_cl_verticies);
    for (i=0;i<numCells;i++) ierr = PMMG_Set_tetrahedron(parmesh,cells[4*i+0],cells[4*i+1],cells[4*i+2],cells[4*i+3],0,i+1);
    ierr = PMMG_Set_triangles(parmesh,bdFaces,bdFaceIds);
    ierr = PMMG_Set_metSize(parmesh,MMG5_Vertex,numVertices,MMG5_Tensor);
    for (i=0;i<numVertices;i++) PMMG_Set_tensorMet(parmesh,metric[6*i],metric[6*i+1],metric[6*i+2],metric[6*i+3],metric[6*i+4],metric[6*i+5],i+1);

    // Interfaces de communication
    ierr = PMMG_Set_numberOfNodeCommunicators(parmesh,num_communicators);
    printf("nombre de communicateurs sur le proc %d : %d \n",me,num_communicators);
    for(ct=0,p=0;p<numProcs;p++)
      if (num_per_proc[p] !=0 && p!=me) {
        printf("DEBUG %d, taille: %d ct: %d p:%d \n",me,offset[ct+1]-offset[ct],ct,p);
        ierr = PMMG_Set_ithNodeCommunicatorSize(parmesh,ct,p,offset[ct+1]-offset[ct]); printf("DEBUG communicator size: %d \n", ierr);
        ierr = PMMG_Set_ithNodeCommunicator_nodes(parmesh,ct,&communicators_local_new[offset[ct]],&communicators_global[offset[ct]],1); printf("DEBUG communicator: %d \n", ierr);
        
        // printf("DEBUG (%d)  ct: %d\n", me, ct);
        // printf("DEBUG(%d)   communicators_local_new: ", me);
        // for (int iv=offset[ct]; iv<offset[ct+1]; ++iv) {
        //   printf(" %d,", communicators_local_new[iv]);
        // }
        // printf("\n");
        // printf("DEBUG(%d)   communicators_global: ", me);
        // for (int iv=offset[ct]; iv<offset[ct+1]; ++iv) {
        //   printf(" %d,", communicators_global[iv]);
        // }
        // printf("\n");

        ct++;
      }
    
    // Remaillage
    MPI_Barrier(comm);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_globalNum,1);
    //ierr = PMMG_saveMesh_distributed(parmesh,"mesh_depart");
    ierr = PMMG_parmmglib_distributed(parmesh);printf("DEBUG remaillage //: %d \n", ierr);
    //ierr = PMMG_saveMesh_distributed(parmesh,"mesh_apres");
  }

  /*3. Passer du nouveau maillage mmg à un maillage lisible par petsc*/
  if (numProcs==1) {
  switch(dim)
  {
    case(2):
      numCornersNew = 3;
      ierr = MMG2D_Get_meshSize(mmg_mesh,&numVerticesNew,&numCellsNew,0,&numFacesNew);
      
      ierr = PetscCalloc4(2*numVerticesNew,&verticesNew,numVerticesNew,&tab_cl_verticies_new,numVerticesNew,&tab_areCorners,numVerticesNew,&tab_areRequiredVerticies);CHKERRQ(ierr);
      ierr = PetscCalloc3(3*numCellsNew,&cellsNew,numCellsNew,&tab_cl_cells_new,numCellsNew,&tab_areRequiredCells);CHKERRQ(ierr);
      ierr = PetscCalloc4(2*numFacesNew,&faces,numFacesNew,&tab_cl_faces,numFacesNew,&tab_areRidges,numFacesNew,&tab_areRequiredFaces);CHKERRQ(ierr);
      ierr = MMG2D_Get_vertices(mmg_mesh,verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);
      ierr = MMG2D_Get_triangles(mmg_mesh,cellsNew,tab_cl_cells_new,tab_areRequiredCells);
      ierr = MMG2D_Get_edges(mmg_mesh,faces,tab_cl_faces,tab_areRidges,tab_areRequiredFaces);

      for (i=0;i<3*numCellsNew;i++) cellsNew[i]-=1;
      for (i=0;i<2*numFacesNew;i++) faces[i]-=1;

      ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, dmNew);CHKERRQ(ierr);
      ierr = MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end); printf("DEBUG libération mémoire : %d \n", ierr);
    break;
    case(3):
      numCornersNew = 4;
      ierr = MMG3D_Get_meshSize(mmg_mesh,&numVerticesNew,&numCellsNew,0,&numFacesNew,0,0);
      ierr = PetscCalloc4(3*numVerticesNew,&verticesNew,numVerticesNew,&tab_cl_verticies_new,numVerticesNew,&tab_areCorners,numVerticesNew,&tab_areRequiredVerticies);CHKERRQ(ierr);
      ierr = PetscCalloc3(4*numCellsNew,&cellsNew,numCellsNew,&tab_cl_cells_new,numCellsNew,&tab_areRequiredCells);CHKERRQ(ierr);
      ierr = PetscCalloc4(3*numFacesNew,&faces,numFacesNew,&tab_cl_faces,numFacesNew,&tab_areRidges,numFacesNew,&tab_areRequiredFaces);CHKERRQ(ierr);
      ierr = MMG3D_Get_vertices(mmg_mesh,verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);
      ierr = MMG3D_Get_tetrahedra(mmg_mesh,cellsNew,tab_cl_cells_new,tab_areRequiredCells);
      ierr = MMG3D_Get_triangles(mmg_mesh,faces,tab_cl_faces,tab_areRequiredFaces);
      
      for (i=0;i<4*numCellsNew;i++) cellsNew[i]-=1;
      for (i=0;i<3*numFacesNew;i++) faces[i]-=1;

      ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, dmNew);CHKERRQ(ierr);
      ierr = MMG3D_Free_all(MMG5_ARG_start,MMG5_ARG_ppMesh,&mmg_mesh,MMG5_ARG_ppMet,&mmg_metric,MMG5_ARG_end); printf("DEBUG libération mémoire : %d \n", ierr);
    break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }
  }
  else {
    numCornersNew = 4;
    ierr = PMMG_Get_meshSize(parmesh,&numVerticesNew,&numCellsNew,0,&numFacesNew,0,0);
    ierr = PetscCalloc4(3*numVerticesNew,&verticesNew,numVerticesNew,&tab_cl_verticies_new,numVerticesNew,&tab_areCorners,numVerticesNew,&tab_areRequiredVerticies);CHKERRQ(ierr);
    ierr = PetscCalloc3(4*numCellsNew,&cellsNew,numCellsNew,&tab_cl_cells_new,numCellsNew,&tab_areRequiredCells);CHKERRQ(ierr);
    ierr = PetscCalloc4(3*numFacesNew,&faces,numFacesNew,&tab_cl_faces,numFacesNew,&tab_areRidges,numFacesNew,&tab_areRequiredFaces);CHKERRQ(ierr);
    ierr = PMMG_Get_vertices(parmesh,verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);
    ierr = PMMG_Get_tetrahedra(parmesh,cellsNew,tab_cl_cells_new,tab_areRequiredCells);
    ierr = PMMG_Get_triangles(parmesh,faces,tab_cl_faces,tab_areRequiredFaces);

    ierr = PetscCalloc2(numVerticesNew,&gv_new,numVerticesNew,&ranks_own);
    ierr = PMMG_Get_verticesGloNum(parmesh,gv_new,ranks_own);

    // Décallage et changement de numérotation
    for (i=0;i<3*numFacesNew;i++) faces[i]-=1;
    for (i=0;i<4*numCellsNew;i++) cellsNew[i]=gv_new[cellsNew[i]-1]-1;

    // Calcul de la liste des sommets sur chacun des procs
    for (i=0;i<numVerticesNew;i++) if (ranks_own[i]==me) numVerticesNewNew++;
    ierr = PetscCalloc1(numVerticesNewNew,&VerticesNewNew); CHKERRQ(ierr);
    for (ct=0,i=0;i<numVerticesNew;i++) if (ranks_own[i]==me) {
      VerticesNewNew[ct++]=verticesNew[i];
    }

    // tests
    for (p=0;p<numProcs;p++) {
      if (p==me) printf("me : %d , numVerticesNew : %d \n",me,numVerticesNew);
      if (p==me) { for (i=0;i<numVerticesNew;i++) printf("%d ",gv_new[i]); printf("\n");}
      if (p==me) { for (i=0;i<numVerticesNew;i++) printf("%d ",ranks_own[i]); printf("\n");}
      MPI_Barrier(comm);
    }
    for(i=0;i<4*numCellsNew;i++) if (cellsNew[i]<0) printf("DEBUG -1 %d %d\n",i,cellsNew[i]);
    
    for (p=0;p<numProcs;p++) {
      if (p==me) {
        for (i=0;i<numVerticesNew;i++) {
          printf("DBEUG(%d)  VerticesNewNew[%d]: %1.2f %1.2f %1.2f\n", me, i, 
            VerticesNewNew[3*i], VerticesNewNew[3*i+1], VerticesNewNew[3*i+2]);
        }
      }
      MPI_Barrier(comm);
    }

    // printf("DEBUG %d nombre de sommets %d, nombre de tétra %d num total: %d \n",me,numVerticesNew,numCellsNew,num_total);
    // if (me==1) for (i=0;i<numVerticesNew;i++) printf("%d sommets %d %lf %lf %lf\n",me,i,verticesNew[3*i],verticesNew[3*i+1],verticesNew[3*i+2]);
    ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNewNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, VerticesNewNew, NULL, dmNew);CHKERRQ(ierr);
    ierr = PMMG_Free_all(PMMG_ARG_start,PMMG_ARG_ppParMesh,&parmesh,PMMG_ARG_end);

  }

  /*Reconstruction des conditions aux limites*/
  //ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  //ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  //for(i=0;i<numFacesNew;i++) ierr = DMLabelSetValue(bdLabelNew,faces[i],tab_cl_faces[i]);CHKERRQ(ierr);

  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);

  switch(dim)
  {
    case(2):
      for (i=0;i<numFacesNew;i++){
        const PetscInt *supportA, *supportB;
        PetscInt SA,SB,inter=-1;
        ierr = DMPlexGetSupportSize(*dmNew,faces[2*i]+vStart,&SA);
        ierr = DMPlexGetSupportSize(*dmNew,faces[2*i+1]+vStart,&SB);
        ierr = DMPlexGetSupport(*dmNew,faces[2*i]+vStart,&supportA);
        ierr = DMPlexGetSupport(*dmNew,faces[2*i+1]+vStart,&supportB);
        /*printf("Numero des points : %d %d \n",faces[2*i]+1,faces[2*i+1]+1);
        printf("Taille des supports : %d %d \n",SA,SB);
        printf("Support de A ");for(d=0;d<SA;d++) printf("%d ",supportA[d]); printf("\n");
        printf("Support de B ");for(d=0;d<SB;d++) printf("%d ",supportB[d]); printf("\n");*/

        // Calcul de l'intersection:
        for(j=0;j<SA;j++){
          for(k=0;k<SB;k++){
            if(supportA[j]==supportB[k]) inter=supportA[j];
          }
        }
        ierr = DMLabelSetValue(bdLabelNew,inter,tab_cl_faces[i]);CHKERRQ(ierr);
      }
    break;
    case(3):
      for (i=0;i<numFacesNew;i++){
        const PetscInt *supportA, *supportB, *supportC;
        const PetscInt *supportiA, *supportiB, *supportiC;

        PetscInt SA,SB,SC,inter=-1;
        PetscInt SiA,SiB,SiC;
        PetscInt ia, ib, ic;
        PetscInt ja, jb, jc;
        ierr = DMPlexGetSupportSize(*dmNew,faces[3*i]+vStart,&SA);
        ierr = DMPlexGetSupportSize(*dmNew,faces[3*i+1]+vStart,&SB);
        ierr = DMPlexGetSupportSize(*dmNew,faces[3*i+2]+vStart,&SC);
        ierr = DMPlexGetSupport(*dmNew,faces[3*i]+vStart,&supportA);
        ierr = DMPlexGetSupport(*dmNew,faces[3*i+1]+vStart,&supportB);
        ierr = DMPlexGetSupport(*dmNew,faces[3*i+2]+vStart,&supportC);

        //printf("%d %d %d\n",SA,SB,SC);
        inter=-1;
        for (ia=0;ia<SA;ia++) {
          ierr = DMPlexGetSupportSize(*dmNew,supportA[ia],&SiA);
          ierr = DMPlexGetSupport(*dmNew,supportA[ia],&supportiA);
          for (ib=0;ib<SB;ib++) {
            ierr = DMPlexGetSupportSize(*dmNew,supportB[ib],&SiB);
            ierr = DMPlexGetSupport(*dmNew,supportB[ib],&supportiB);
            for (ic=0;ic<SC;ic++) {
              ierr = DMPlexGetSupportSize(*dmNew,supportC[ic],&SiC);
              ierr = DMPlexGetSupport(*dmNew,supportC[ic],&supportiC);
              for(ja=0;ja<SiA;ja++) {
                for(jb=0;jb<SiB;jb++) {
                  for(jc=0;jc<SiC;jc++) {
                    if(supportiA[ja]==supportiB[jb] && supportiA[ja]==supportiC[jc]) inter=supportiA[ja];
                  }
                }
              }
            }
          }
        }
        ierr = DMLabelSetValue(bdLabelNew,inter,tab_cl_faces[i]);CHKERRQ(ierr);
      }
      break;
    default:SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }

  /*libération des tableaux en mémoire TO DO*/
  ierr = PetscFree3(cells,metric,vertices);CHKERRQ(ierr); printf("debug 1 \n");
  ierr = PetscFree4(bdFaces,bdFaceIds,tab_cl_verticies,tab_cl_triangles);CHKERRQ(ierr);printf("debug 2 \n");
  ierr = PetscFree4(verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);CHKERRQ(ierr);printf("debug 3 \n");
  ierr = PetscFree4(faces,tab_cl_faces,tab_areRidges,tab_areRequiredFaces);CHKERRQ(ierr);printf("debug 4 \n");
  ierr = PetscFree4(faces,tab_cl_faces,tab_areRidges,tab_areRequiredFaces);CHKERRQ(ierr);printf("debug 5 \n");
  
  if (numProcs>1) ierr = PetscFree4(communicators_local,communicators_global,num_per_proc,flags_proc); CHKERRQ(ierr);
  if (numProcs>1) ierr = PetscFree3(VerticesNewNew,ranks_own,gv_new); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Remeshing needs external package support.\nPlease reconfigure with --download-pragmatic.");
  PetscFunctionReturn(0);
#endif
}

PetscErrorCode DMAdaptMetricParMMG_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
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
  IS                 bdIS; // IS : index set: ensemble d'indices ordonnés BDIS pour les conditions aux limites , global vertex num pour le //
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met; // double
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *vertices, *metric, *verticesNew;
  PetscInt          *cells;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize, fStart, fEnd,j,k;
  PetscBool          flg;
  DMLabel            bdLabelNew;
  PetscInt          *cellsNew;
  PetscInt           numCellsNew, numVerticesNew;
  PetscInt           numCornersNew;
  PetscMPIInt        numProcs,me;
  PetscErrorCode     ierr;
  // nos variables
  PetscInt * tab_cl_verticies, * tab_cl_triangles, *tab_cl_verticies_new, *tab_cl_cells_new;
  PetscInt * tab_areCorners, * tab_areRequiredCells, *tab_areRequiredVerticies;
  PetscInt * faces, * tab_cl_faces, * tab_areRidges, * tab_areRequiredFaces;
  PetscInt numFacesNew;
  MMG5_pMesh mmg_mesh = NULL;
  MMG5_pSol mmg_metric = NULL;
  PetscInt i;
  // Pour le parallèle
  PMMG_pParMesh parmesh = NULL;
  PetscSF starforest;
  const PetscInt *gV;
  PetscInt numleaves=0, numroots=0, num_communicators=0,ct=0,ctt=0,p;
  PetscInt *flags_proc,*communicators_local,*communicators_global,*communicators_local_new;
  PetscInt *num_per_proc, *offset;
  IS globalVertexNum;
  PetscInt *gv_new, *ranks_own,numVerticesNewNew=0;
  PetscReal *VerticesNewNew;

  PetscFunctionBegin;

  // 0. Début du programme 
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &me);CHKERRQ(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  
  // 0. Dans le cas parallèlle il faut récupérer les sommets aux interfaces
  // et le starforest
  if(numProcs>1){
    ierr = DMPlexDistribute(dm, 0, NULL, &udm);dm=udm;
  }

  // 1. Chercher les informations dans le maillage
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr); /*Récupération des cellulles*/
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr); /*Récupération des sommets*/
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr); /*On regarde uniquement les cellulles et les sommets*/
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr); /*regarde la taille maximum du cône ie combien de sommets au max a une cellule donc si ce sont des triangles, des quadrilatères*/
  numCells    = cEnd - cStart; /*indice du début moins indice de fin des tranches*/
  numVertices = vEnd - vStart;

  printf("DEBUG %d : nombre de tétra %d \n",me, numCells);

  // 2. Récupération des cellules
  ierr = PetscCalloc1(numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) { /*boucle sur les cellules*/
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr); /*récupération de la taille du cone*/
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr); /*récupération du cône*/
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart+1; /*translation du tableau*/
  }

  // 3. Récupération de tous les sommets
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  if (dim==2) {ierr=PetscCalloc2(numVertices*3, &metric,2*numVertices, &vertices);CHKERRQ(ierr);}
  if (dim==3) {ierr=PetscCalloc2(numVertices*6, &metric,3*numVertices, &vertices);CHKERRQ(ierr);}

  for (v = 0; v < vEnd-vStart; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v+vStart, &off);CHKERRQ(ierr);
    if (dim==2) {
      vertices[2*v]=PetscRealPart(coords[off+0]);
      vertices[2*v+1]=PetscRealPart(coords[off+1]);
    }
    else if (dim==3) {
      vertices[3*v]=PetscRealPart(coords[off+0]);
      vertices[3*v+1]=PetscRealPart(coords[off+1]);
      vertices[3*v+2]=PetscRealPart(coords[off+2]);
    }
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);

  // 3.5 récupération des interfaces
  if (numProcs>1) {
    
    PetscInt niranks,nranks;
    const PetscMPIInt *iranks,*ranks;
    const PetscInt *ioffset, *irootloc,*roffset,*rmine,*rremote;
    PetscSection       coordSection;
    DM                 cdm;
    PetscInt           off;
    Vec                coordinates;
    const PetscScalar  *coords;
    PetscScalar x, y, z;

    ierr = DMGetPointSF(dm,&starforest);CHKERRQ(ierr);
    ierr = PetscSFSetUp(starforest);CHKERRQ(ierr);
    ierr = PetscSFGetLeafRanks(starforest, &niranks, &iranks, &ioffset, &irootloc); CHKERRQ(ierr);
    ierr = PetscSFGetRootRanks(starforest, &nranks, &ranks, &roffset, &rmine, &rremote); CHKERRQ(ierr);

    ierr = PetscCalloc1(numVertices*numProcs,&flags_proc);CHKERRQ(ierr);
    ierr = PetscCalloc1(numProcs,&num_per_proc);CHKERRQ(ierr);
    ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
    ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    /// FIN TESTS

    // Recherche des feuilles
    for(p=0;p<nranks;p++) 
      for(i=roffset[p];i<roffset[p+1];i++)
        if (rmine[i] >= vStart && rmine[i] < vEnd && flags_proc[(rmine[i]-vStart)*numProcs+ranks[p]]==0) {
          numleaves++;
          flags_proc[(rmine[i]-vStart)*numProcs+ranks[p]]=1;
          num_per_proc[ranks[p]]++;
        }

    // recherche des racines
    for(p=0;p<niranks;p++) 
      for(i=ioffset[p];i<ioffset[p+1];i++)
        if (irootloc[i] >= vStart && irootloc[i] < vEnd && flags_proc[(irootloc[i]-vStart)*numProcs+iranks[p]]==0) {
          numroots++;
          flags_proc[(irootloc[i]-vStart)*numProcs+iranks[p]]=1;
          num_per_proc[iranks[p]]++;
        }

    printf("numleaves + numroots %d sur %d total : %d\n",numleaves+numroots,me,numVertices);

    // nombre de comm
    for (p=0;p<numProcs;p++) if (p!=me && num_per_proc[p] !=0) num_communicators++;
    
    // construction des tableaux non triés
    ierr = PetscCalloc1(num_communicators+1,&offset);CHKERRQ(ierr); offset[0]=0;
    ierr = PetscCalloc3(numroots+numleaves,&communicators_local,numroots+numleaves,&communicators_global,numroots+numleaves,&communicators_local_new); CHKERRQ(ierr);
    for(p=0;p<numProcs;p++) if (p!=me && num_per_proc[p] !=0) {
      for(i=0;i<numVertices;i++) if (flags_proc[i*numProcs+p]==1) {
        communicators_local[ct]=i+1;
        communicators_global[ct]=gV[i] < 0 ? -(gV[i]+1) : gV[i];
        communicators_global[ct]++;
        ct++;
      }
      offset[++ctt]=ct;
    }
    // tri du tableau global
    for(p=0;p<num_communicators;p++) {
      ierr = PetscTimSort(offset[p+1]-offset[p],&communicators_global[offset[p]],sizeof(PetscInt),my_increasing_comparison_function,NULL); CHKERRQ(ierr);
    }
    // reconstruction du tableau local
    for(p=0;p<numroots+numleaves;p++) {
      for(i=0;i<numroots+numleaves;i++) {
        PetscInt tempa=communicators_local[i]-1,tempb;
        tempb=gV[tempa] < 0 ? -(gV[tempa]+1) : gV[tempa];
        tempb++;
        if (tempb==communicators_global[p]) communicators_local_new[p]=communicators_local[i];
      }
    }

    for (p=0;p<numProcs;p++) {
      if (p==me) for(v=0;v<num_communicators+1;v++) {printf("%d",offset[v]);printf(" fin offst\n");}
      if (p==me) for(i=0;i<numroots+numleaves;i++) {
        ierr = PetscSectionGetOffset(coordSection, vStart+communicators_local_new[i]-1, &off);CHKERRQ(ierr);
        x = PetscRealPart(coords[off+0]);
        y = PetscRealPart(coords[off+1]);
        z = PetscRealPart(coords[off+2]);
        //printf("%d %d %d %d diff %d coo x %1.2f coo y %1.2f coo z %1.2f \n",me,i,communicators_local_new[i],communicators_global[i],gV[communicators_local_new[i]-1],x,y,z);
      }

      MPI_Barrier(PETSC_COMM_WORLD);
    }
    ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&udm);CHKERRQ(ierr);


  // 4. Récupératin des conditions aux bords
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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart+1;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (bdLabel) {ierr = DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]);CHKERRQ(ierr);}
    else         {bdFaceIds[f] = 1;}
  }
  ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&bdLabelFull);CHKERRQ(ierr);
  
  // 5. Récupération de la metric
  ierr = VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view");CHKERRQ(ierr);
  ierr = VecGetArrayRead(vertexMetric, &met);CHKERRQ(ierr);
  
  if (dim==2) {
    for (v = 0; v < (vEnd-vStart); ++v) { // *PetscSqr(dim)
      metric[3*v] = PetscRealPart(met[4*v]);
      metric[3*v+1] = PetscRealPart(met[4*v+1]);
      metric[3*v+2] = PetscRealPart(met[4*v+3]);
    }
  }
  else if (dim==3) {
    for (v = 0; v < (vEnd-vStart); ++v) { // *PetscSqr(dim)
      metric[6*v] = PetscRealPart(met[9*v]);
      metric[6*v+1] = PetscRealPart(met[9*v+1]);
      metric[6*v+2] = PetscRealPart(met[9*v+2]);
      metric[6*v+3] = PetscRealPart(met[9*v+4]);
      metric[6*v+4] = PetscRealPart(met[9*v+5]);
      metric[6*v+5] = PetscRealPart(met[9*v+8]);
    } 
  }
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);


  /* PARTIE 2: Transformation du maillage avec mmg*/
  ierr = PetscCalloc2(numVertices,&tab_cl_verticies,numCells,&tab_cl_triangles);CHKERRQ(ierr);

  if (numProcs==1) {
  switch(dim)
  {
  case 2:
    ierr = MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose,10); // quantité d'information à l'écran 10=toutes les informations
    ierr = MMG2D_Set_meshSize(mmg_mesh,numVertices,numCells,0,numBdFaces);
    // Passage des informations sur le maillage
    
    // géométrie
    ierr = MMG2D_Set_vertices(mmg_mesh,vertices,tab_cl_verticies);
    ierr = MMG2D_Set_triangles(mmg_mesh,cells,tab_cl_triangles);
    ierr = MMG2D_Set_edges(mmg_mesh,bdFaces,bdFaceIds);

    // métrique
    ierr = MMG2D_Set_solSize(mmg_mesh,mmg_metric,MMG5_Vertex,numVertices,MMG5_Tensor);
    for (i=0;i<numVertices;i++) MMG2D_Set_tensorSol(mmg_metric,metric[3*i],metric[3*i+1],metric[3*i+2],i+1);
    
    // Remaillage
    ierr =  MMG2D_saveMshMesh(mmg_mesh,NULL,"maillage_avant.msh");
    ierr = MMG2D_mmg2dlib(mmg_mesh,mmg_metric); printf("DEBUG remaillage 2D: %d \n", ierr);
    ierr = MMG2D_saveMshMesh(mmg_mesh,NULL,"maillage_apres.msh");
    break;

  case 3:
    ierr = MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose,10);
    ierr = MMG3D_Set_meshSize(mmg_mesh,numVertices,numCells,0,numBdFaces,0,0);

    ierr = MMG3D_Set_vertices(mmg_mesh,vertices,tab_cl_verticies);
    for (i=0;i<numCells;i++) ierr = MMG3D_Set_tetrahedron(mmg_mesh,cells[4*i+0],cells[4*i+1],cells[4*i+2],cells[4*i+3],0,i+1);
    ierr = MMG3D_Set_triangles(mmg_mesh,bdFaces,bdFaceIds);

    // Métrique
    ierr = MMG3D_Set_solSize(mmg_mesh,mmg_metric,MMG5_Vertex,numVertices,MMG5_Tensor);
    ierr = MMG3D_Set_tensorSols(mmg_metric,metric);

    // Remaillage
    ierr = MMG3D_saveMshMesh(mmg_mesh,NULL,"maillage_avant_3D.msh");
    ierr = MMG3D_mmg3dlib(mmg_mesh,mmg_metric); printf("DEBUG remaillage 3D: %d \n", ierr);  
    ierr = MMG3D_saveMshMesh(mmg_mesh,NULL,"maillage_apres_3D.msh");
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }
  }
  else
  {
    // Initialisation
    ierr = PMMG_Init_parMesh(PMMG_ARG_start,PMMG_ARG_ppParMesh,&parmesh,PMMG_ARG_pMesh,PMMG_ARG_pMet,PMMG_ARG_dim,3,PMMG_ARG_MPIComm,comm,PMMG_ARG_end);
    ierr = PMMG_Set_meshSize(parmesh,numVertices,numCells,0,numBdFaces,0,0);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_APImode,PMMG_APIDISTRIB_nodes);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_verbose,10);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_globalNum,1);

    // Maillage et métrique
    ierr = PMMG_Set_vertices(parmesh,vertices,tab_cl_verticies);
    for (i=0;i<numCells;i++) ierr = PMMG_Set_tetrahedron(parmesh,cells[4*i+0],cells[4*i+1],cells[4*i+2],cells[4*i+3],0,i+1);
    ierr = PMMG_Set_triangles(parmesh,bdFaces,bdFaceIds);
    ierr = PMMG_Set_metSize(parmesh,MMG5_Vertex,numVertices,MMG5_Tensor);
    for (i=0;i<numVertices;i++) PMMG_Set_tensorMet(parmesh,metric[6*i],metric[6*i+1],metric[6*i+2],metric[6*i+3],metric[6*i+4],metric[6*i+5],i+1);

    // Interfaces de communication
    ierr = PMMG_Set_numberOfNodeCommunicators(parmesh,num_communicators);
    printf("nombre de communicateurs sur le proc %d : %d \n",me,num_communicators);
    for(ct=0,p=0;p<numProcs;p++)
      if (num_per_proc[p] !=0 && p!=me) {
        printf("DEBUG %d, taille: %d ct: %d p:%d \n",me,offset[ct+1]-offset[ct],ct,p);
        ierr = PMMG_Set_ithNodeCommunicatorSize(parmesh,ct,p,offset[ct+1]-offset[ct]); printf("DEBUG communicator size: %d \n", ierr);
        ierr = PMMG_Set_ithNodeCommunicator_nodes(parmesh,ct,&communicators_local_new[offset[ct]],&communicators_global[offset[ct]],1); printf("DEBUG communicator: %d \n", ierr);
        
        // printf("DEBUG (%d)  ct: %d\n", me, ct);
        // printf("DEBUG(%d)   communicators_local_new: ", me);
        // for (int iv=offset[ct]; iv<offset[ct+1]; ++iv) {
        //   printf(" %d,", communicators_local_new[iv]);
        // }
        // printf("\n");
        // printf("DEBUG(%d)   communicators_global: ", me);
        // for (int iv=offset[ct]; iv<offset[ct+1]; ++iv) {
        //   printf(" %d,", communicators_global[iv]);
        // }
        // printf("\n");

        ct++;
      }
    
    // Remaillage
    MPI_Barrier(comm);
    ierr = PMMG_Set_iparameter(parmesh,PMMG_IPARAM_globalNum,1);
    //ierr = PMMG_saveMesh_distributed(parmesh,"mesh_depart");
    ierr = PMMG_parmmglib_distributed(parmesh);printf("DEBUG remaillage //: %d \n", ierr);
    //ierr = PMMG_saveMesh_distributed(parmesh,"mesh_apres");
  }

  /*3. Passer du nouveau maillage mmg à un maillage lisible par petsc*/
  if (numProcs==1) {
  switch(dim)
  {
    case(2):
      numCornersNew = 3;
      ierr = MMG2D_Get_meshSize(mmg_mesh,&numVerticesNew,&numCellsNew,0,&numFacesNew);
      
      ierr = PetscCalloc4(2*numVerticesNew,&verticesNew,numVerticesNew,&tab_cl_verticies_new,numVerticesNew,&tab_areCorners,numVerticesNew,&tab_areRequiredVerticies);CHKERRQ(ierr);
      ierr = PetscCalloc3(3*numCellsNew,&cellsNew,numCellsNew,&tab_cl_cells_new,numCellsNew,&tab_areRequiredCells);CHKERRQ(ierr);
      ierr = PetscCalloc4(2*numFacesNew,&faces,numFacesNew,&tab_cl_faces,numFacesNew,&tab_areRidges,numFacesNew,&tab_areRequiredFaces);CHKERRQ(ierr);
      ierr = MMG2D_Get_vertices(mmg_mesh,verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);
      ierr = MMG2D_Get_triangles(mmg_mesh,cellsNew,tab_cl_cells_new,tab_areRequiredCells);
      ierr = MMG2D_Get_edges(mmg_mesh,faces,tab_cl_faces,tab_areRidges,tab_areRequiredFaces);

      for (i=0;i<3*numCellsNew;i++) cellsNew[i]-=1;
      for (i=0;i<2*numFacesNew;i++) faces[i]-=1;

      ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, dmNew);CHKERRQ(ierr);
      ierr = MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end); printf("DEBUG libération mémoire : %d \n", ierr);
    break;
    case(3):
      numCornersNew = 4;
      ierr = MMG3D_Get_meshSize(mmg_mesh,&numVerticesNew,&numCellsNew,0,&numFacesNew,0,0);
      ierr = PetscCalloc4(3*numVerticesNew,&verticesNew,numVerticesNew,&tab_cl_verticies_new,numVerticesNew,&tab_areCorners,numVerticesNew,&tab_areRequiredVerticies);CHKERRQ(ierr);
      ierr = PetscCalloc3(4*numCellsNew,&cellsNew,numCellsNew,&tab_cl_cells_new,numCellsNew,&tab_areRequiredCells);CHKERRQ(ierr);
      ierr = PetscCalloc4(3*numFacesNew,&faces,numFacesNew,&tab_cl_faces,numFacesNew,&tab_areRidges,numFacesNew,&tab_areRequiredFaces);CHKERRQ(ierr);
      ierr = MMG3D_Get_vertices(mmg_mesh,verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);
      ierr = MMG3D_Get_tetrahedra(mmg_mesh,cellsNew,tab_cl_cells_new,tab_areRequiredCells);
      ierr = MMG3D_Get_triangles(mmg_mesh,faces,tab_cl_faces,tab_areRequiredFaces);
      
      for (i=0;i<4*numCellsNew;i++) cellsNew[i]-=1;
      for (i=0;i<3*numFacesNew;i++) faces[i]-=1;

      ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, dmNew);CHKERRQ(ierr);
      ierr = MMG3D_Free_all(MMG5_ARG_start,MMG5_ARG_ppMesh,&mmg_mesh,MMG5_ARG_ppMet,&mmg_metric,MMG5_ARG_end); printf("DEBUG libération mémoire : %d \n", ierr);
    break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }
  }
  else {
    numCornersNew = 4;
    ierr = PMMG_Get_meshSize(parmesh,&numVerticesNew,&numCellsNew,0,&numFacesNew,0,0);
    ierr = PetscCalloc4(3*numVerticesNew,&verticesNew,numVerticesNew,&tab_cl_verticies_new,numVerticesNew,&tab_areCorners,numVerticesNew,&tab_areRequiredVerticies);CHKERRQ(ierr);
    ierr = PetscCalloc3(4*numCellsNew,&cellsNew,numCellsNew,&tab_cl_cells_new,numCellsNew,&tab_areRequiredCells);CHKERRQ(ierr);
    ierr = PetscCalloc4(3*numFacesNew,&faces,numFacesNew,&tab_cl_faces,numFacesNew,&tab_areRidges,numFacesNew,&tab_areRequiredFaces);CHKERRQ(ierr);
    ierr = PMMG_Get_vertices(parmesh,verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);
    ierr = PMMG_Get_tetrahedra(parmesh,cellsNew,tab_cl_cells_new,tab_areRequiredCells);
    ierr = PMMG_Get_triangles(parmesh,faces,tab_cl_faces,tab_areRequiredFaces);

    ierr = PetscCalloc2(numVerticesNew,&gv_new,numVerticesNew,&ranks_own);
    ierr = PMMG_Get_verticesGloNum(parmesh,gv_new,ranks_own);

    // Décallage et changement de numérotation
    for (i=0;i<3*numFacesNew;i++) faces[i]-=1;
    for (i=0;i<4*numCellsNew;i++) cellsNew[i]=gv_new[cellsNew[i]-1]-1;

    // Calcul de la liste des sommets sur chacun des procs
    for (i=0;i<numVerticesNew;i++) if (ranks_own[i]==me) numVerticesNewNew++;
    ierr = PetscCalloc1(numVerticesNewNew,&VerticesNewNew); CHKERRQ(ierr);
    for (ct=0,i=0;i<numVerticesNew;i++) if (ranks_own[i]==me) {
      VerticesNewNew[ct++]=verticesNew[i];
    }

    // tests
    for (p=0;p<numProcs;p++) {
      if (p==me) printf("me : %d , numVerticesNew : %d \n",me,numVerticesNew);
      if (p==me) { for (i=0;i<numVerticesNew;i++) printf("%d ",gv_new[i]); printf("\n");}
      if (p==me) { for (i=0;i<numVerticesNew;i++) printf("%d ",ranks_own[i]); printf("\n");}
      MPI_Barrier(comm);
    }
    for(i=0;i<4*numCellsNew;i++) if (cellsNew[i]<0) printf("DEBUG -1 %d %d\n",i,cellsNew[i]);
    
    for (p=0;p<numProcs;p++) {
      if (p==me) {
        for (i=0;i<numVerticesNew;i++) {
          printf("DBEUG(%d)  VerticesNewNew[%d]: %1.2f %1.2f %1.2f\n", me, i, 
            VerticesNewNew[3*i], VerticesNewNew[3*i+1], VerticesNewNew[3*i+2]);
        }
      }
      MPI_Barrier(comm);
    }

    // printf("DEBUG %d nombre de sommets %d, nombre de tétra %d num total: %d \n",me,numVerticesNew,numCellsNew,num_total);
    // if (me==1) for (i=0;i<numVerticesNew;i++) printf("%d sommets %d %lf %lf %lf\n",me,i,verticesNew[3*i],verticesNew[3*i+1],verticesNew[3*i+2]);
    ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNewNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, VerticesNewNew, NULL, dmNew);CHKERRQ(ierr);
    ierr = PMMG_Free_all(PMMG_ARG_start,PMMG_ARG_ppParMesh,&parmesh,PMMG_ARG_end);

  }

  /*Reconstruction des conditions aux limites*/
  //ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  //ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  //for(i=0;i<numFacesNew;i++) ierr = DMLabelSetValue(bdLabelNew,faces[i],tab_cl_faces[i]);CHKERRQ(ierr);

  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);

  switch(dim)
  {
    case(2):
      for (i=0;i<numFacesNew;i++){
        const PetscInt *supportA, *supportB;
        PetscInt SA,SB,inter=-1;
        ierr = DMPlexGetSupportSize(*dmNew,faces[2*i]+vStart,&SA);
        ierr = DMPlexGetSupportSize(*dmNew,faces[2*i+1]+vStart,&SB);
        ierr = DMPlexGetSupport(*dmNew,faces[2*i]+vStart,&supportA);
        ierr = DMPlexGetSupport(*dmNew,faces[2*i+1]+vStart,&supportB);
        /*printf("Numero des points : %d %d \n",faces[2*i]+1,faces[2*i+1]+1);
        printf("Taille des supports : %d %d \n",SA,SB);
        printf("Support de A ");for(d=0;d<SA;d++) printf("%d ",supportA[d]); printf("\n");
        printf("Support de B ");for(d=0;d<SB;d++) printf("%d ",supportB[d]); printf("\n");*/

        // Calcul de l'intersection:
        for(j=0;j<SA;j++){
          for(k=0;k<SB;k++){
            if(supportA[j]==supportB[k]) inter=supportA[j];
          }
        }
        ierr = DMLabelSetValue(bdLabelNew,inter,tab_cl_faces[i]);CHKERRQ(ierr);
      }
    break;
    case(3):
      for (i=0;i<numFacesNew;i++){
        const PetscInt *supportA, *supportB, *supportC;
        const PetscInt *supportiA, *supportiB, *supportiC;

        PetscInt SA,SB,SC,inter=-1;
        PetscInt SiA,SiB,SiC;
        PetscInt ia, ib, ic;
        PetscInt ja, jb, jc;
        ierr = DMPlexGetSupportSize(*dmNew,faces[3*i]+vStart,&SA);
        ierr = DMPlexGetSupportSize(*dmNew,faces[3*i+1]+vStart,&SB);
        ierr = DMPlexGetSupportSize(*dmNew,faces[3*i+2]+vStart,&SC);
        ierr = DMPlexGetSupport(*dmNew,faces[3*i]+vStart,&supportA);
        ierr = DMPlexGetSupport(*dmNew,faces[3*i+1]+vStart,&supportB);
        ierr = DMPlexGetSupport(*dmNew,faces[3*i+2]+vStart,&supportC);

        //printf("%d %d %d\n",SA,SB,SC);
        inter=-1;
        for (ia=0;ia<SA;ia++) {
          ierr = DMPlexGetSupportSize(*dmNew,supportA[ia],&SiA);
          ierr = DMPlexGetSupport(*dmNew,supportA[ia],&supportiA);
          for (ib=0;ib<SB;ib++) {
            ierr = DMPlexGetSupportSize(*dmNew,supportB[ib],&SiB);
            ierr = DMPlexGetSupport(*dmNew,supportB[ib],&supportiB);
            for (ic=0;ic<SC;ic++) {
              ierr = DMPlexGetSupportSize(*dmNew,supportC[ic],&SiC);
              ierr = DMPlexGetSupport(*dmNew,supportC[ic],&supportiC);
              for(ja=0;ja<SiA;ja++) {
                for(jb=0;jb<SiB;jb++) {
                  for(jc=0;jc<SiC;jc++) {
                    if(supportiA[ja]==supportiB[jb] && supportiA[ja]==supportiC[jc]) inter=supportiA[ja];
                  }
                }
              }
            }
          }
        }
        ierr = DMLabelSetValue(bdLabelNew,inter,tab_cl_faces[i]);CHKERRQ(ierr);
      }
      break;
    default:SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }

  /*libération des tableaux en mémoire TO DO*/
  ierr = PetscFree3(cells,metric,vertices);CHKERRQ(ierr); printf("debug 1 \n");
  ierr = PetscFree4(bdFaces,bdFaceIds,tab_cl_verticies,tab_cl_triangles);CHKERRQ(ierr);printf("debug 2 \n");
  ierr = PetscFree4(verticesNew,tab_cl_verticies_new,tab_areCorners,tab_areRequiredVerticies);CHKERRQ(ierr);printf("debug 3 \n");
  ierr = PetscFree4(faces,tab_cl_faces,tab_areRidges,tab_areRequiredFaces);CHKERRQ(ierr);printf("debug 4 \n");
  ierr = PetscFree4(faces,tab_cl_faces,tab_areRidges,tab_areRequiredFaces);CHKERRQ(ierr);printf("debug 5 \n");
  
  if (numProcs>1) ierr = PetscFree4(communicators_local,communicators_global,num_per_proc,flags_proc); CHKERRQ(ierr);
  if (numProcs>1) ierr = PetscFree3(VerticesNewNew,ranks_own,gv_new); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Remeshing needs external package support.\nPlease reconfigure with --download-pragmatic.");
  PetscFunctionReturn(0);
#endif
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
  PetscInt remesher = 0;
  switch (remesher) {
  case 0:
    DMAdaptMetricPragmatic_Plex(dm, vertexMetric, bdLabel, dmNew);
    break;
  case 1:
    DMAdaptMetricMMG_Plex(dm, vertexMetric, bdLabel, dmNew);
    break;
  case 2:
    DMAdaptMetricParMMG_Plex(dm, vertexMetric, bdLabel, dmNew);
    break;
  }
}