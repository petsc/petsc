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

PETSC_EXTERN PetscErrorCode DMAdaptMetric_Pragmatic_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
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

  /* Get cell offsets */
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }

  /* Get local-to-global vertex map */
  ierr = PetscCalloc1(numVertices, &l2gv);CHKERRQ(ierr);
  ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
  ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  for (v = 0, numLocVertices = 0; v < numVertices; ++v) {
    if (gV[v] >= 0) ++numLocVertices;
    l2gv[v] = gV[v] < 0 ? -(gV[v]+1) : gV[v];
  }
  ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);

  /* Get vertex coordinate arrays */
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
  /* Send to Pragmatic and remesh */
  switch (dim) {
  case 2:
    pragmatic_2d_mpi_init(&numVertices, &numCells, cells, x, y, l2gv, numLocVertices, comm);
    break;
  case 3:
    pragmatic_3d_mpi_init(&numVertices, &numCells, cells, x, y, z, l2gv, numLocVertices, comm);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %D", dim);
  }
  pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
  pragmatic_set_metric(metric);
  pragmatic_adapt(((DM_Plex *) dm->data)->remeshBd ? 1 : 0);
  ierr = PetscFree(l2gv);CHKERRQ(ierr);

  /* Retrieve mesh from Pragmatic and create new plex */
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
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, coordsNew, NULL, NULL, dmNew);CHKERRQ(ierr);

  /* Rebuild boundary label */
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

  /* Clean up */
  switch (dim) {
  case 2: ierr = PetscFree2(xNew[0], xNew[1]);CHKERRQ(ierr);
  break;
  case 3: ierr = PetscFree3(xNew[0], xNew[1], xNew[2]);CHKERRQ(ierr);
  break;
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

PETSC_EXTERN PetscErrorCode DMAdaptMetric_Mmg_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
{
#if defined(PETSC_HAVE_MMG)
  MPI_Comm           comm;
  const char        *bdName = "_boundary_";
  DM                 udm, cdm;
  DMLabel            bdLabelFull;
  const char        *bdLabelName;
  IS                 bdIS;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *vertices, *metric, *verticesNew;
  PetscInt          *cells, *cellsNew;
  PetscInt          *facesNew, *faceTagsNew;
  PetscInt          *verTags, *cellTags, *verTagsNew, *cellTagsNew;
  PetscInt          *corners, *requiredCells, *requiredVer, *ridges, *requiredFaces;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize, fStart, fEnd, i, j, k, Neq;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew;
  PetscBool          flg;
  DMLabel            bdLabelNew;
  MMG5_pMesh         mmg_mesh = NULL;
  MMG5_pSol          mmg_metric = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }

  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  Neq  = (dim*(dim+1))/2;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;

  /* Get cell offsets */
  ierr = PetscMalloc1(numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) { cells[coff++] = cone[cl] - vStart + 1; }
  }

  /* Get vertex coordinate array */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscMalloc2(numVertices*Neq, &metric, dim*numVertices, &vertices);CHKERRQ(ierr);
  for (v = 0; v < vEnd-vStart; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v+vStart, &off);CHKERRQ(ierr);
    for (i = 0; i < dim; ++i) { vertices[dim*v+i] = PetscRealPart(coords[off+i]); }
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);

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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) { ++bdSize; }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds);CHKERRQ(ierr);
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    ierr = DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) { bdFaces[bdSize++] = closure[cl] - vStart + 1; }
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
  for (v = 0; v < (vEnd-vStart); ++v) {
    for (i = 0, k = 0; i < dim; ++i) {
      for (j = i; j < dim; ++j) {
        metric[Neq*v+k] = PetscRealPart(met[dim*dim*v+dim*i+j]);
        k++;
      }
    }
  }
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);

  /* Send mesh to Mmg and remesh */
  ierr = DMPlexMetricGetVerbosity(dm, &verbosity);CHKERRQ(ierr);
  ierr = PetscCalloc2(numVertices, &verTags, numCells, &cellTags);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose, 10);
    ierr = MMG2D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numBdFaces);
    ierr = MMG2D_Set_vertices(mmg_mesh, vertices, verTags);
    ierr = MMG2D_Set_triangles(mmg_mesh, cells, cellTags);
    ierr = MMG2D_Set_edges(mmg_mesh, bdFaces, bdFaceIds);
    ierr = MMG2D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor);
    ierr = MMG2D_Set_tensorSols(mmg_metric, metric);
    ierr = MMG2D_mmg2dlib(mmg_mesh, mmg_metric);
    break;
  case 3:
    ierr = MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose, ctx->verbosity);
    ierr = MMG3D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numBdFaces, 0, 0);
    ierr = MMG3D_Set_vertices(mmg_mesh, vertices, verTags);
    ierr = MMG3D_Set_tetrahedra(mmg_mesh, cells, cellTags);
    ierr = MMG3D_Set_triangles(mmg_mesh, bdFaces, bdFaceIds);
    ierr = MMG3D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor);
    ierr = MMG3D_Set_tensorSols(mmg_metric, metric);
    ierr = MMG3D_mmg3dlib(mmg_mesh, mmg_metric);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }

  /* Retrieve mesh from Mmg and create new Plex*/
  switch (dim) {
  case 2:
    numCornersNew = 3;
    ierr = MMG2D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew);
    ierr = PetscMalloc4(2*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer);CHKERRQ(ierr);
    ierr = PetscMalloc3(3*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells);CHKERRQ(ierr);
    ierr = PetscMalloc4(2*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces);CHKERRQ(ierr);
    ierr = MMG2D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer);
    ierr = MMG2D_Get_triangles(mmg_mesh, cellsNew, cellTagsNew, requiredCells);
    ierr = MMG2D_Get_edges(mmg_mesh, facesNew, faceTagsNew, ridges, requiredFaces);
    break;
  case 3:
    numCornersNew = 4;
    ierr = MMG3D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0);
    ierr = PetscMalloc4(3*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer);CHKERRQ(ierr);
    ierr = PetscMalloc3(4*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells);CHKERRQ(ierr);
    ierr = PetscMalloc4(3*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces);CHKERRQ(ierr);
    ierr = MMG3D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer);
    ierr = MMG3D_Get_tetrahedra(mmg_mesh, cellsNew, cellTagsNew, requiredCells);
    ierr = MMG3D_Get_triangles(mmg_mesh, facesNew, faceTagsNew, requiredFaces);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }
  for (i = 0; i < (dim+1)*numCellsNew; i++) { cellsNew[i] -= 1; }
  for (i = 0; i < dim*numFacesNew; i++) { facesNew[i] -= 1; }
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, NULL, dmNew);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    break;
  case 3:
    ierr = MMG3D_Free_all(MMG5_ARG_start,MMG5_ARG_ppMesh,&mmg_mesh,MMG5_ARG_ppMet,&mmg_metric,MMG5_ARG_end);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No MMG adaptation defined for dimension %D", dim);
  }

  /* Rebuild boundary labels */
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (i = 0; i < numFacesNew; i++) {
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) { facePoints[j] = facesNew[i*dim+j]+vStart; }
    ierr = DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    if (numFaces != 1) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "%d vertices cannot define more than 1 facet (%d)", dim, numFaces);
    ierr = DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]);CHKERRQ(ierr);
    ierr = DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = PetscFree(cells);CHKERRQ(ierr);
  ierr = PetscFree2(metric, vertices);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree2(verTags, cellTags);CHKERRQ(ierr);
  ierr = PetscFree4(verticesNew, verTagsNew, corners, requiredVer);CHKERRQ(ierr);
  ierr = PetscFree3(cellsNew, cellTagsNew, requiredCells);CHKERRQ(ierr);
  ierr = PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces);CHKERRQ(ierr);

  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Remeshing needs external package support.\nPlease reconfigure with --download-mmg.");
  PetscFunctionReturn(0);
#endif
}

PETSC_EXTERN PetscErrorCode DMAdaptMetric_ParMmg_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
{
#if defined(PETSC_HAVE_PARMMG)
  MPI_Comm           comm, tmpComm;
  const char        *bdName = "_boundary_";
  DM                 udm, cdm;
  DMLabel            bdLabelFull;
  const char        *bdLabelName;
  IS                 bdIS, globalVertexNum;
  PetscSection       coordSection;
  Vec                coordinates;
  PetscSF            sf;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *vertices, *metric, *verticesNew;
  PetscInt          *cells;
  const PetscInt    *gV, *ioffset, *irootloc, *roffset, *rmine, *rremote;
  PetscInt          *numVerInterfaces, *ngbRanks, *verNgbRank, *interfaces_lv, *interfaces_gv, *intOffset;
  PetscInt           niranks, nrranks, numNgbRanks, numVerNgbRanksTotal, count, sliceSize;
  PetscInt          *verTags, *cellTags;
  PetscInt           dim, Neq, cStart, cEnd, numCells, coff, vStart, vEnd, numVertices;
  PetscInt           maxConeSize, numBdFaces, bdSize, fStart, fEnd;
  PetscBool          flg;
  DMLabel            bdLabelNew;
  PetscReal         *verticesNewLoc;
  PetscInt          *verTagsNew, *cellsNew, *cellTagsNew, *corners;
  PetscInt          *requiredCells, *requiredVer, *facesNew, *faceTagsNew, *ridges, *requiredFaces;
  PetscInt          *gv_new, *owners, *verticesNewSorted;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew, numVerticesNewLoc;
  PetscInt           i, j, k, p, r, n, v, c, f, off, lv, gv;
  const PetscMPIInt *iranks, *rranks;
  PetscMPIInt        numProcs, rank;
  PMMG_pParMesh      parmesh = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  MPI_Comm_dup(comm, &tmpComm);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }

  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim != 3) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "ParMmg only works in 3D.\n");
  Neq  = (dim*(dim+1))/2;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;

  /* Get cell offsets */
  ierr = PetscMalloc1(numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart+1;
  }

  /* Get vertex coordinate array */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscMalloc2(numVertices*Neq, &metric, dim*numVertices, &vertices);CHKERRQ(ierr);
  for (v = 0; v < vEnd-vStart; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v+vStart, &off);CHKERRQ(ierr);
    for (i = 0; i < dim; ++i) { vertices[dim*v+i]=PetscRealPart(coords[off+i]); }
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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart+1;
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
  for (v = 0; v < (vEnd-vStart); ++v) {
    for (i = 0, k = 0; i < dim; ++i) {
      for (j = i; j < dim; ++j) {
        metric[Neq*v+k] = PetscRealPart(met[dim*dim*v+dim*i+j]);
        k++;
      }
    }
  }
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);

  /* Build ParMMG communicators: the list of vertices between two partitions  */
  niranks = nrranks = 0;
  numNgbRanks = 0;
  if (numProcs > 1) {
    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc);CHKERRQ(ierr);
    ierr = PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);
    ierr = PetscCalloc1(numProcs, &numVerInterfaces);CHKERRQ(ierr);

    /* Counting */
    for (r = 0; r < niranks; ++r) {
      for (i=ioffset[r], count=0; i<ioffset[r+1]; ++i) {
        if (irootloc[i] >= vStart && irootloc[i] < vEnd) {count++; }
      }
      numVerInterfaces[iranks[r]] += count;
    }
    for (r = 0; r < nrranks; ++r) {
      for (i=roffset[r], count=0; i<roffset[r+1]; ++i) {
        if (rmine[i] >= vStart && rmine[i] < vEnd) {count++; }
      }
      numVerInterfaces[rranks[r]] += count;
    }
    for (p = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) { numNgbRanks++; }
    }
    ierr = PetscMalloc2(numNgbRanks, &ngbRanks, numNgbRanks, &verNgbRank);CHKERRQ(ierr);
    for (p = 0, n = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) {
        ngbRanks[n] = p;
        verNgbRank[n] = numVerInterfaces[p];
        n++;
      }
    }
    numVerNgbRanksTotal = 0;
    for (p = 0; p < numNgbRanks; ++p) { numVerNgbRanksTotal += verNgbRank[p]; }

    /* For each neighbor, fill in interface arrays */
    ierr = PetscMalloc3(numVerNgbRanksTotal, &interfaces_lv, numVerNgbRanksTotal, &interfaces_gv,  numNgbRanks+1, &intOffset);CHKERRQ(ierr);
    intOffset[0] = 0;
    for (p = 0, r = 0, i = 0; p < numNgbRanks; ++p) {
      intOffset[p+1] = intOffset[p];
      if (iranks && iranks[i] == ngbRanks[p]) {

        /* Add the right slice of irootloc at the right place */
        sliceSize = ioffset[i+1]-ioffset[i];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          if (ioffset[i]+j >= ioffset[niranks]) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
          v = irootloc[ioffset[i]+j];
          if (v >= vStart && v < vEnd) {
            if (intOffset[p+1]+count >= numVerNgbRanksTotal) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
            interfaces_lv[intOffset[p+1]+count] = v-vStart;
            count++;
          }
        }
        intOffset[p+1] += count;
        i++;
      }
      if (rranks && rranks[r] == ngbRanks[p]) {

        /* Add the right slice of rmine at the right place */
        sliceSize = roffset[r+1]-roffset[r];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          if (roffset[r]+j >= roffset[nrranks]) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
          v = rmine[roffset[r]+j];
          if (v >= vStart && v < vEnd) {
            if (intOffset[p+1]+count >= numVerNgbRanksTotal) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
            interfaces_lv[intOffset[p+1]+count] = v-vStart;
            count++;
          }
        }
        intOffset[p+1] += count;
        r++;
      }
      if (intOffset[p+1] != intOffset[p] + verNgbRank[p]) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Unequal offsets");
    }
    ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
    ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);
    for (i = 0; i < numVerNgbRanksTotal; ++i) {
      v = gV[interfaces_lv[i]];
      interfaces_gv[i] = v < 0 ? -v-1 : v;
      interfaces_lv[i] += 1;
      interfaces_gv[i] += 1;
    }
    ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
    ierr = PetscFree(numVerInterfaces);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&udm);CHKERRQ(ierr);

  /* Send the data to ParMmg and remesh */
  ierr = PetscCalloc2(numVertices, &verTags, numCells, &cellTags);CHKERRQ(ierr);
  ierr = PMMG_Init_parMesh(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_pMesh, PMMG_ARG_pMet, PMMG_ARG_dim, 3, PMMG_ARG_MPIComm, tmpComm, PMMG_ARG_end);
  ierr = PMMG_Set_meshSize(parmesh, numVertices, numCells, 0, numBdFaces, 0, 0);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_APImode, PMMG_APIDISTRIB_nodes);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_verbose, 10);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1);
  ierr = PMMG_Set_vertices(parmesh, vertices, verTags);
  for (i=0; i<numCells; i++) ierr = PMMG_Set_tetrahedron(parmesh, cells[4*i+0], cells[4*i+1], cells[4*i+2], cells[4*i+3], 0, i+1);
  ierr = PMMG_Set_triangles(parmesh, bdFaces, bdFaceIds);
  ierr = PMMG_Set_metSize(parmesh, MMG5_Vertex, numVertices, MMG5_Tensor);
  for (i=0; i<numVertices; i++) {
    PMMG_Set_tensorMet(parmesh, metric[6*i], metric[6*i+1], metric[6*i+2], metric[6*i+3], metric[6*i+4], metric[6*i+5], i+1);
  }
  ierr = PMMG_Set_numberOfNodeCommunicators(parmesh, numNgbRanks);
  for (c = 0; c < numNgbRanks; ++c) {
    ierr = PMMG_Set_ithNodeCommunicatorSize(parmesh, c, ngbRanks[c], intOffset[c+1]-intOffset[c]);
    ierr = PMMG_Set_ithNodeCommunicator_nodes(parmesh, c, &interfaces_lv[intOffset[c]], &interfaces_gv[intOffset[c]], 1);
  }
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_niter, 3);
  ierr = PMMG_parmmglib_distributed(parmesh);
  ierr = PetscFree(cells);CHKERRQ(ierr);
  ierr = PetscFree2(metric, vertices);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree2(verTags, cellTags);CHKERRQ(ierr);
  if (numProcs > 1) {
    ierr = PetscFree2(ngbRanks, verNgbRank);CHKERRQ(ierr);
    ierr = PetscFree3(interfaces_lv, interfaces_gv, intOffset);CHKERRQ(ierr);
  }

  /* Retrieve mesh from Mmg and create new Plex */
  numCornersNew = 4;
  ierr = PMMG_Get_meshSize(parmesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0);
  ierr = PetscMalloc4(dim*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer);CHKERRQ(ierr);
  ierr = PetscMalloc3((dim+1)*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces);CHKERRQ(ierr);
  ierr = PMMG_Get_vertices(parmesh, verticesNew, verTagsNew, corners, requiredVer);
  ierr = PMMG_Get_tetrahedra(parmesh, cellsNew, cellTagsNew, requiredCells);
  ierr = PMMG_Get_triangles(parmesh, facesNew, faceTagsNew, requiredFaces);
  ierr = PetscMalloc2(numVerticesNew, &owners, numVerticesNew, &gv_new);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1);
  ierr = PMMG_Get_verticesGloNum(parmesh, gv_new, owners);
  for (i = 0; i < dim*numFacesNew; ++i) { facesNew[i] -= 1; }
  for (i = 0; i < (dim+1)*numCellsNew; ++i) {
    cellsNew[i] = gv_new[cellsNew[i]-1]-1;
  }
  numVerticesNewLoc = 0;
  for (i = 0; i < numVerticesNew; ++i) {
    if (owners[i] == rank) { numVerticesNewLoc++; }
  }
  ierr = PetscMalloc2(numVerticesNewLoc*dim, &verticesNewLoc, numVerticesNew, &verticesNewSorted);CHKERRQ(ierr);

  for (i = 0, c = 0; i < numVerticesNew; i++) {
    if (owners[i] == rank) {
      for (j=0; j<dim; ++j) { verticesNewLoc[dim*c+j] = verticesNew[dim*i+j]; }
        c++;
    }
  }
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNewLoc, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNewLoc, NULL, &verticesNewSorted, dmNew);CHKERRQ(ierr);
  ierr = PMMG_Free_all(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_end);

  /* Rebuild boundary label*/
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  for (i = 0; i < numFacesNew; i++) {
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) {
      lv = facesNew[i*dim+j];
      gv = gv_new[lv]-1;
      ierr = PetscFindInt(gv, numVerticesNew, verticesNewSorted, &lv);CHKERRQ(ierr);
      facePoints[j] = lv+vStart;
    }
    ierr = DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    if (numFaces != 1) { SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "%d vertices cannot define more than 1 facet (%d)", dim, numFaces); }
    ierr = DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]);CHKERRQ(ierr);
    ierr = DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = PetscFree4(verticesNew, verTagsNew, corners, requiredVer);CHKERRQ(ierr);
  ierr = PetscFree3(cellsNew, cellTagsNew, requiredCells);CHKERRQ(ierr);
  ierr = PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces);CHKERRQ(ierr);
  ierr = PetscFree2(owners, gv_new);CHKERRQ(ierr);
  ierr = PetscFree2(verticesNewLoc, verticesNewSorted);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Remeshing needs external package support.\nPlease reconfigure with --download-parmmg.");
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
  PetscInt remesher = 2;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}
