#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#include <petsc/private/petscfeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "DMProjectPoint_Func_Private"
static PetscErrorCode DMProjectPoint_Func_Private(DM dm, PetscReal time, PetscFECellGeom *fegeom, PetscFVCellGeom *fvgeom, PetscBool isFE[], PetscDualSpace sp[],
                                                  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs,
                                                  PetscScalar values[])
{
  PetscDS        prob;
  PetscInt       Nf, *Nc, f, spDim, d, c, v;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  /* Get values for closure */
  for (f = 0, v = 0; f < Nf; ++f) {
    void * const ctx = ctxs ? ctxs[f] : NULL;

    if (!sp[f]) continue;
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    for (d = 0; d < spDim; ++d) {
      if (funcs[f]) {
        if (isFE[f]) {ierr = PetscDualSpaceApply(sp[f], d, time, fegeom, Nc[f], funcs[f], ctx, &values[v]);CHKERRQ(ierr);}
        else         {ierr = PetscDualSpaceApplyFVM(sp[f], d, time, fvgeom, Nc[f], funcs[f], ctx, &values[v]);CHKERRQ(ierr);}
      } else {
        for (c = 0; c < Nc[f]; ++c) values[v+c] = 0.0;
      }
      v += Nc[f];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectPoint_Field_Private"
static PetscErrorCode DMProjectPoint_Field_Private(DM dm, DM dmAux, PetscReal time, Vec localU, Vec localA, PetscFECellGeom *fegeom, PetscDualSpace sp[], PetscInt p,
                                                   PetscReal **basisTab, PetscReal **basisDerTab, PetscReal **basisTabAux, PetscReal **basisDerTabAux,
                                                   void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                  PetscReal, const PetscReal[], PetscScalar[]), void **ctxs,
                                                   PetscScalar values[])
{
  PetscDS        prob, probAux = NULL;
  PetscSection   section, sectionAux = NULL;
  PetscScalar   *u, *u_t = NULL, *u_x, *a = NULL, *a_t = NULL, *a_x = NULL, *refSpaceDer, *refSpaceDerAux = NULL;
  PetscScalar   *coefficients   = NULL, *coefficientsAux   = NULL;
  PetscScalar   *coefficients_t = NULL, *coefficientsAux_t = NULL;
  PetscReal     *x;
  PetscInt      *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt       dim, Nf, NfAux = 0, f, spDim, d, c, v, tp = 0;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetSpatialDimension(prob, &dim);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL /*&u_t*/, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localU, p, NULL, &coefficients);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL /*&a_t*/, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dmAux, sectionAux, localA, p, NULL, &coefficientsAux);CHKERRQ(ierr);
  }
  /* Get values for closure */
  for (f = 0, v = 0; f < Nf; ++f) {
    if (!sp[f]) continue;
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    for (d = 0; d < spDim; ++d) {
      if (funcs[f]) {
        PetscQuadrature  quad;
        const PetscReal *points;
        PetscInt         numPoints, q;

        ierr = PetscDualSpaceGetFunctional(sp[f], d, &quad);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(quad, NULL, &numPoints, &points, NULL);CHKERRQ(ierr);
        for (q = 0; q < numPoints; ++q, ++tp) {
          CoordinatesRefToReal(dim, dim, fegeom->v0, fegeom->J, &points[q*dim], x);
          EvaluateFieldJets(dim, Nf, Nb, Nc, tp, basisTab, basisDerTab, refSpaceDer, fegeom->invJ, coefficients, coefficients_t, u, u_x, u_t);
          if (probAux) {EvaluateFieldJets(dim, NfAux, NbAux, NcAux, tp, basisTabAux, basisDerTabAux, refSpaceDerAux, fegeom->invJ, coefficientsAux, coefficientsAux_t, a, a_x, a_t);}
          (*funcs[f])(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, time, x, &values[v]);
        }
      } else {
        for (c = 0; c < Nc[f]; ++c) values[v+c] = 0.0;
      }
      v += Nc[f];
    }
  }
  ierr = DMPlexVecRestoreClosure(dm, section, localU, p, NULL, &coefficients);CHKERRQ(ierr);
  if (dmAux) {ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, localA, p, NULL, &coefficientsAux);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectPoint_Private"
static PetscErrorCode DMProjectPoint_Private(DM dm, DM dmAux, PetscInt h, PetscReal time, Vec localU, Vec localA, PetscBool hasFE, PetscBool hasFV, PetscBool isFE[],
                                             PetscDualSpace sp[], PetscInt p, PetscReal **basisTab, PetscReal **basisDerTab, PetscReal **basisTabAux, PetscReal **basisDerTabAux,
                                             DMBoundaryConditionType type, void (**funcs)(void), void **ctxs, PetscBool fieldActive[], PetscScalar values[])
{
  PetscFECellGeom fegeom;
  PetscFVCellGeom fvgeom;
  PetscInt        dim, dimEmbed;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  if (hasFE) {
    ierr = DMPlexComputeCellGeometryFEM(dm, p, NULL, fegeom.v0, fegeom.J, fegeom.invJ, &fegeom.detJ);CHKERRQ(ierr);
    fegeom.dim      = dim - h;
    fegeom.dimEmbed = dimEmbed;
  }
  if (hasFV) {ierr = DMPlexComputeCellGeometryFVM(dm, p, &fvgeom.volume, fvgeom.centroid, NULL);CHKERRQ(ierr);}
  switch (type) {
  case DM_BC_ESSENTIAL:
  case DM_BC_NATURAL:
    ierr = DMProjectPoint_Func_Private(dm, time, &fegeom, &fvgeom, isFE, sp, (PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *)) funcs, ctxs, values);CHKERRQ(ierr);break;
  case DM_BC_ESSENTIAL_FIELD:
  case DM_BC_NATURAL_FIELD:
    ierr = DMProjectPoint_Field_Private(dm, dmAux, time, localU, localA, &fegeom, sp, p,
                                        basisTab, basisDerTab, basisTabAux, basisDerTabAux,
                                        (void (**)(PetscInt, PetscInt, PetscInt,
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                   PetscReal, const PetscReal[], PetscScalar[])) funcs, ctxs, values);CHKERRQ(ierr);break;
  default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown boundary condition type: %d", (int) type);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectLocal_Generic_Plex"
PetscErrorCode DMProjectLocal_Generic_Plex(DM dm, PetscReal time, Vec localU,
                                           DMLabel label, PetscInt numIds, const PetscInt ids[],
                                           DMBoundaryConditionType type, void (**funcs)(void), void **ctxs,
                                           InsertMode mode, Vec localX)
{
  DM              dmAux = NULL;
  PetscDS         prob, probAux = NULL;
  Vec             localA = NULL;
  PetscSection    section;
  PetscDualSpace *sp, *cellsp;
  PetscReal     **basisTab = NULL, **basisDerTab = NULL, **basisTabAux = NULL, **basisDerTabAux = NULL;
  PetscInt       *Nc;
  PetscInt        dim, dimEmbed, maxHeight, h, Nf, NfAux = 0, f;
  PetscBool      *isFE, hasFE = PETSC_FALSE, hasFV = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetMaxProjectionHeight(dm, &maxHeight);CHKERRQ(ierr);
  if (maxHeight < 0 || maxHeight > dim) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Maximum projection height %d not in [0, %d)\n", maxHeight, dim);}
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &localA);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
  }
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf, &isFE, Nf, &sp);CHKERRQ(ierr);
  if (maxHeight > 0) {ierr = PetscMalloc1(Nf, &cellsp);CHKERRQ(ierr);}
  else               {cellsp = sp;}
  if (localU && localU != localX) {ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localU, time, NULL, NULL, NULL);CHKERRQ(ierr);}
  /* Get cell dual spaces */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      hasFE   = PETSC_TRUE;
      isFE[f] = PETSC_TRUE;
      ierr  = PetscFEGetDualSpace(fe, &cellsp[f]);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      hasFV   = PETSC_TRUE;
      isFE[f] = PETSC_FALSE;
      ierr = PetscFVGetDualSpace(fv, &cellsp[f]);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
  }
  if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_NATURAL_FIELD) {
    PetscFE    fem;
    PetscReal *points;
    PetscInt   numPoints, spDim, d;

    if (maxHeight) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Field proejction not supported for face interpolation");
    numPoints = 0;
    for (f = 0; f < Nf; ++f) {
      ierr = PetscDualSpaceGetDimension(cellsp[f], &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        if (funcs[f]) {
          PetscQuadrature quad;
          PetscInt        Nq;

          ierr = PetscDualSpaceGetFunctional(cellsp[f], d, &quad);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(quad, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
          numPoints += Nq;
        }
      }
    }
    ierr = PetscMalloc1(numPoints*dim, &points);CHKERRQ(ierr);
    numPoints = 0;
    for (f = 0; f < Nf; ++f) {
      ierr = PetscDualSpaceGetDimension(cellsp[f], &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        if (funcs[f]) {
          PetscQuadrature  quad;
          const PetscReal *qpoints;
          PetscInt         Nq, q;

          ierr = PetscDualSpaceGetFunctional(cellsp[f], d, &quad);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(quad, NULL, &Nq, &qpoints, NULL);CHKERRQ(ierr);
          for (q = 0; q < Nq*dim; ++q) points[numPoints*dim+q] = qpoints[q];
          numPoints += Nq;
        }
      }
    }
    ierr = PetscMalloc4(Nf, &basisTab, Nf, &basisDerTab, NfAux, &basisTabAux, NfAux, &basisDerTabAux);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      if (!isFE[f]) continue;
      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fem);CHKERRQ(ierr);
      ierr = PetscFEGetTabulation(fem, numPoints, points, &basisTab[f], &basisDerTab[f], NULL);CHKERRQ(ierr);
    }
    for (f = 0; f < NfAux; ++f) {
      ierr = PetscDSGetDiscretization(probAux, f, (PetscObject *) &fem);CHKERRQ(ierr);
      ierr = PetscFEGetTabulation(fem, numPoints, points, &basisTabAux[f], &basisDerTabAux[f], NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree(points);CHKERRQ(ierr);
  }
  /* Note: We make no attempt to optimize for height. Higher height things just overwrite the lower height results. */
  for (h = 0; h <= maxHeight; h++) {
    PetscScalar *values;
    PetscBool   *fieldActive;
    PetscInt     pStart, pEnd, p, spDim, totDim, numValues;

    ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
    if (!h) {
      PetscInt cEndInterior;

      ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
      pEnd = cEndInterior < 0 ? pEnd : cEndInterior;
    }
    if (pEnd <= pStart) continue;
    /* Compute totDim, the number of dofs in the closure of a point at this height */
    totDim = 0;
    for (f = 0; f < Nf; ++f) {
      if (!h) {
        sp[f] = cellsp[f];
      } else {
        ierr = PetscDualSpaceGetHeightSubspace(cellsp[f], h, &sp[f]);CHKERRQ(ierr);
        if (!sp[f]) continue;
      }
      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      totDim += spDim*Nc[f];
    }
    ierr = DMPlexVecGetClosure(dm, section, localX, pStart, &numValues, NULL);CHKERRQ(ierr);
    if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The section point closure size %d != dual space dimension %d", numValues, totDim);
    if (!totDim) continue;
    /* Loop over points at this height */
    ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, Nf, PETSC_BOOL, &fieldActive);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) fieldActive[f] = (funcs[f] && sp[f]) ? PETSC_TRUE : PETSC_FALSE;
    if (label) {
      PetscInt i;

      for (i = 0; i < numIds; ++i) {
        IS              pointIS;
        const PetscInt *points;
        PetscInt        n;

        ierr = DMLabelGetStratumIS(label, ids[i], &pointIS);CHKERRQ(ierr);
        if (!pointIS) continue; /* No points with that id on this process */
        ierr = ISGetLocalSize(pointIS, &n);CHKERRQ(ierr);
        ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
        for (p = 0; p < n; ++p) {
          const PetscInt  point = points[p];

          if ((point < pStart) || (point >= pEnd)) continue;
          ierr = DMProjectPoint_Private(dm, dmAux, h, time, localU, localA, hasFE, hasFV, isFE, sp, point, basisTab, basisDerTab, basisTabAux, basisDerTabAux, type, funcs, ctxs, fieldActive, values);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr2);
            ierr2 = DMRestoreWorkArray(dm, Nf, PETSC_BOOL, &fieldActive);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          ierr = DMPlexVecSetFieldClosure_Internal(dm, section, localX, fieldActive, point, values, mode);CHKERRQ(ierr);
        }
        ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      }
    } else {
      for (p = pStart; p < pEnd; ++p) {
        ierr = DMProjectPoint_Private(dm, dmAux, h, time, localU, localA, hasFE, hasFV, isFE, sp, p, basisTab, basisDerTab, basisTabAux, basisDerTabAux, type, funcs, ctxs, fieldActive, values);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr2);
          ierr2 = DMRestoreWorkArray(dm, Nf, PETSC_BOOL, &fieldActive);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        ierr = DMPlexVecSetFieldClosure_Internal(dm, section, localX, fieldActive, p, values, mode);CHKERRQ(ierr);
      }
    }
    ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, Nf, PETSC_BOOL, &fieldActive);CHKERRQ(ierr);
  }
  /* Cleanup */
  if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_NATURAL_FIELD) {
    PetscFE fem;

    for (f = 0; f < Nf; ++f) {
      if (!isFE[f]) continue;
      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fem);CHKERRQ(ierr);
      ierr = PetscFERestoreTabulation(fem, 0, NULL, &basisTab[f], &basisDerTab[f], NULL);CHKERRQ(ierr);
    }
    for (f = 0; f < NfAux; ++f) {
      ierr = PetscDSGetDiscretization(probAux, f, (PetscObject *) &fem);CHKERRQ(ierr);
      ierr = PetscFERestoreTabulation(fem, 0, NULL, &basisTabAux[f], &basisDerTabAux[f], NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree4(basisTab, basisDerTab, basisTabAux, basisDerTabAux);CHKERRQ(ierr);
  }
  ierr = PetscFree2(isFE, sp);CHKERRQ(ierr);
  if (maxHeight > 0) {ierr = PetscFree(cellsp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectFunctionLocal_Plex"
PetscErrorCode DMProjectFunctionLocal_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localX, NULL, 0, NULL, DM_BC_ESSENTIAL, (void (**)(void)) funcs, ctxs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectFunctionLabelLocal_Plex"
PetscErrorCode DMProjectFunctionLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localX, label, numIds, ids, DM_BC_ESSENTIAL, (void (**)(void)) funcs, ctxs, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectFieldLocal_Plex"
PetscErrorCode DMProjectFieldLocal_Plex(DM dm, PetscReal time, Vec localU,
                                        void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscScalar[]),
                                        InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localU, NULL, 0, NULL, DM_BC_ESSENTIAL_FIELD, (void (**)(void)) funcs, NULL, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectFieldLabelLocal_Plex"
PetscErrorCode DMProjectFieldLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], Vec localU,
                                             void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                            PetscReal, const PetscReal[], PetscScalar[]),
                                             InsertMode mode, Vec localX)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMProjectLocal_Generic_Plex(dm, time, localU, label, numIds, ids, DM_BC_ESSENTIAL_FIELD, (void (**)(void)) funcs, NULL, mode, localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
