#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

static PetscErrorCode DMPlexLabelToVolumeConstraint(DM dm, DMLabel adaptLabel, PetscInt cStart, PetscInt cEnd, PetscReal refRatio, PetscReal maxVolumes[])
{
  PetscInt       dim, c;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  refRatio = refRatio == PETSC_DEFAULT ? (PetscReal) ((PetscInt) 1 << dim) : refRatio;
  for (c = cStart; c < cEnd; c++) {
    PetscReal vol;
    PetscInt  closureSize = 0, cl;
    PetscInt *closure     = NULL;
    PetscBool anyRefine   = PETSC_FALSE;
    PetscBool anyCoarsen  = PETSC_FALSE;
    PetscBool anyKeep     = PETSC_FALSE;

    CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
    maxVolumes[c - cStart] = vol;
    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];
      PetscInt       refFlag;

      CHKERRQ(DMLabelGetValue(adaptLabel, point, &refFlag));
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
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "DMPlex does not support refinement flag %D", refFlag);
      }
      if (anyRefine) break;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexUninterpolate(dm, &udm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDM(dm, &coordDM));
  CHKERRQ(DMGetLocalSection(coordDM, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  Nv   = vEnd - vStart;
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, Nv*PetscSqr(dim), metricVec));
  CHKERRQ(VecGetArray(*metricVec, &metric));
  Neq  = (dim*(dim+1))/2;
  CHKERRQ(PetscMalloc1(PetscSqr(Neq), &eqns));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF, Neq, Neq, eqns, &A));
  CHKERRQ(MatCreateVecs(A, &mx, &mb));
  CHKERRQ(VecSet(mb, 1.0));
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *sol;
    PetscScalar       *cellCoords = NULL;
    PetscReal          e[3], vol;
    const PetscInt    *cone;
    PetscInt           coneSize, cl, i, j, d, r;

    CHKERRQ(DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &cellCoords));
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
    CHKERRQ(MatSetUnfactored(A));
    CHKERRQ(DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &cellCoords));
    CHKERRQ(MatLUFactor(A, NULL, NULL, NULL));
    CHKERRQ(MatSolve(A, mb, mx));
    CHKERRQ(VecGetArrayRead(mx, &sol));
    CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
    CHKERRQ(DMPlexGetCone(udm, c, &cone));
    CHKERRQ(DMPlexGetConeSize(udm, c, &coneSize));
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
    CHKERRQ(VecRestoreArrayRead(mx, &sol));
  }
  for (v = 0; v < Nv; ++v) {
    const PetscInt *support;
    PetscInt        supportSize, s;
    PetscReal       vol, totVol = 0.0;

    CHKERRQ(DMPlexGetSupport(udm, v+vStart, &support));
    CHKERRQ(DMPlexGetSupportSize(udm, v+vStart, &supportSize));
    for (s = 0; s < supportSize; ++s) {CHKERRQ(DMPlexComputeCellGeometryFVM(dm, support[s], &vol, NULL, NULL)); totVol += vol;}
    for (s = 0; s < PetscSqr(dim); ++s) metric[v*PetscSqr(dim)+s] /= totVol;
  }
  CHKERRQ(PetscFree(eqns));
  CHKERRQ(VecRestoreArray(*metricVec, &metric));
  CHKERRQ(VecDestroy(&mx));
  CHKERRQ(VecDestroy(&mb));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(DMDestroy(&udm));
  PetscFunctionReturn(0);
}

/*
   Contains the list of registered DMPlexGenerators routines
*/
PetscErrorCode DMPlexRefine_Internal(DM dm, PETSC_UNUSED Vec metric, DMLabel adaptLabel, PETSC_UNUSED DMLabel rgLabel, DM *dmRefined)
{
  DMGeneratorFunctionList fl;
  PetscErrorCode        (*refine)(DM,PetscReal*,DM*);
  PetscErrorCode        (*adapt)(DM,Vec,DMLabel,DMLabel,DM*);
  PetscErrorCode        (*refinementFunc)(const PetscReal [], PetscReal *);
  char                    genname[PETSC_MAX_PATH_LEN], *name = NULL;
  PetscReal               refinementLimit;
  PetscReal              *maxVolumes;
  PetscInt                dim, cStart, cEnd, c;
  PetscBool               flg, flg2, localized;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinatesLocalized(dm, &localized));
  CHKERRQ(DMPlexGetRefinementLimit(dm, &refinementLimit));
  CHKERRQ(DMPlexGetRefinementFunction(dm, &refinementFunc));
  if (refinementLimit == 0.0 && !refinementFunc && !adaptLabel) PetscFunctionReturn(0);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscOptionsGetString(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_adaptor", genname, sizeof(genname), &flg));
  if (flg) name = genname;
  else {
    CHKERRQ(PetscOptionsGetString(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_generator", genname, sizeof(genname), &flg2));
    if (flg2) name = genname;
  }

  fl = DMGenerateList;
  if (name) {
    while (fl) {
      CHKERRQ(PetscStrcmp(fl->name,name,&flg));
      if (flg) {
        refine = fl->refine;
        adapt  = fl->adapt;
        goto gotit;
      }
      fl = fl->next;
    }
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Grid refiner %s not registered",name);
  } else {
    while (fl) {
      if (fl->dim < 0 || dim-1 == fl->dim) {
        refine = fl->refine;
        adapt  = fl->adapt;
        goto gotit;
      }
      fl = fl->next;
    }
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No grid refiner of dimension %D registered",dim);
  }

  gotit:
  switch (dim) {
    case 1:
    case 2:
    case 3:
      if (adapt) {
        CHKERRQ((*adapt)(dm, NULL, adaptLabel, NULL, dmRefined));
      } else {
        CHKERRQ(PetscMalloc1(cEnd - cStart, &maxVolumes));
        if (adaptLabel) {
          CHKERRQ(DMPlexLabelToVolumeConstraint(dm, adaptLabel, cStart, cEnd, PETSC_DEFAULT, maxVolumes));
        } else if (refinementFunc) {
          for (c = cStart; c < cEnd; ++c) {
            PetscReal vol, centroid[3];

            CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, &vol, centroid, NULL));
            CHKERRQ((*refinementFunc)(centroid, &maxVolumes[c-cStart]));
          }
        } else {
          for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = refinementLimit;
        }
        CHKERRQ((*refine)(dm, maxVolumes, dmRefined));
        CHKERRQ(PetscFree(maxVolumes));
      }
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mesh refinement in dimension %D is not supported.", dim);
  }
  CHKERRQ(DMCopyDisc(dm, *dmRefined));
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *dmRefined));
  if (localized) CHKERRQ(DMLocalizeCoordinates(*dmRefined));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCoarsen_Internal(DM dm, PETSC_UNUSED Vec metric, DMLabel adaptLabel, PETSC_UNUSED DMLabel rgLabel, DM *dmCoarsened)
{
  Vec            metricVec;
  PetscInt       cStart, cEnd, vStart, vEnd;
  DMLabel        bdLabel = NULL;
  char           bdLabelName[PETSC_MAX_PATH_LEN], rgLabelName[PETSC_MAX_PATH_LEN];
  PetscBool      localized, flg;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinatesLocalized(dm, &localized));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexLabelToMetricConstraint(dm, adaptLabel, cStart, cEnd, vStart, vEnd, PETSC_DEFAULT, &metricVec));
  CHKERRQ(PetscOptionsGetString(NULL, dm->hdr.prefix, "-dm_plex_coarsen_bd_label", bdLabelName, sizeof(bdLabelName), &flg));
  if (flg) CHKERRQ(DMGetLabel(dm, bdLabelName, &bdLabel));
  CHKERRQ(PetscOptionsGetString(NULL, dm->hdr.prefix, "-dm_plex_coarsen_rg_label", rgLabelName, sizeof(rgLabelName), &flg));
  if (flg) CHKERRQ(DMGetLabel(dm, rgLabelName, &rgLabel));
  CHKERRQ(DMAdaptMetric(dm, metricVec, bdLabel, rgLabel, dmCoarsened));
  CHKERRQ(VecDestroy(&metricVec));
  CHKERRQ(DMCopyDisc(dm, *dmCoarsened));
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, *dmCoarsened));
  if (localized) CHKERRQ(DMLocalizeCoordinates(*dmCoarsened));
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptLabel_Plex(DM dm, PETSC_UNUSED Vec metric, DMLabel adaptLabel, PETSC_UNUSED DMLabel rgLabel, DM *dmAdapted)
{
  IS              flagIS;
  const PetscInt *flags;
  PetscInt        defFlag, minFlag, maxFlag, numFlags, f;

  PetscFunctionBegin;
  CHKERRQ(DMLabelGetDefaultValue(adaptLabel, &defFlag));
  minFlag = defFlag;
  maxFlag = defFlag;
  CHKERRQ(DMLabelGetValueIS(adaptLabel, &flagIS));
  CHKERRQ(ISGetLocalSize(flagIS, &numFlags));
  CHKERRQ(ISGetIndices(flagIS, &flags));
  for (f = 0; f < numFlags; ++f) {
    const PetscInt flag = flags[f];

    minFlag = PetscMin(minFlag, flag);
    maxFlag = PetscMax(maxFlag, flag);
  }
  CHKERRQ(ISRestoreIndices(flagIS, &flags));
  CHKERRQ(ISDestroy(&flagIS));
  {
    PetscInt minMaxFlag[2], minMaxFlagGlobal[2];

    minMaxFlag[0] =  minFlag;
    minMaxFlag[1] = -maxFlag;
    CHKERRMPI(MPI_Allreduce(minMaxFlag, minMaxFlagGlobal, 2, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm)));
    minFlag =  minMaxFlagGlobal[0];
    maxFlag = -minMaxFlagGlobal[1];
  }
  if (minFlag == maxFlag) {
    switch (minFlag) {
    case DM_ADAPT_DETERMINE:
      *dmAdapted = NULL;break;
    case DM_ADAPT_REFINE:
      CHKERRQ(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
      CHKERRQ(DMRefine(dm, MPI_COMM_NULL, dmAdapted));break;
    case DM_ADAPT_COARSEN:
      CHKERRQ(DMCoarsen(dm, MPI_COMM_NULL, dmAdapted));break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"DMPlex does not support refinement flag %D", minFlag);
    }
  } else {
    CHKERRQ(DMPlexSetRefinementUniform(dm, PETSC_FALSE));
    CHKERRQ(DMPlexRefine_Internal(dm, NULL, adaptLabel, NULL, dmAdapted));
  }
  PetscFunctionReturn(0);
}
