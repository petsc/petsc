#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMProjectFunctionLocal_Plex"
PetscErrorCode DMProjectFunctionLocal_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscDualSpace *sp, *cellsp;
  PetscInt       *numComp;
  PetscSection    section;
  PetscScalar    *values;
  PetscInt        Nf, dim, dimEmbed, spDim, totDim = 0, numValues, pStart, pEnd, p, cStart, cEnd, cEndInterior, f, d, v, comp, h, maxHeight;
  PetscBool      *isFE, hasFE = PETSC_FALSE, hasFV = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc3(Nf, &isFE, Nf, &sp, Nf, &numComp);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMPlexGetMaxProjectionHeight(dm, &maxHeight);CHKERRQ(ierr);
  if (maxHeight < 0 || maxHeight > dim) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER, "maximum projection height %d not in [0, %d)\n", maxHeight, dim);}
  if (maxHeight > 0) {
    ierr = PetscMalloc1(Nf, &cellsp);CHKERRQ(ierr);
  } else {
    cellsp = sp;
  }
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      hasFE   = PETSC_TRUE;
      isFE[f] = PETSC_TRUE;
      ierr = PetscFEGetNumComponents(fe, &numComp[f]);CHKERRQ(ierr);
      ierr  = PetscFEGetDualSpace(fe, &cellsp[f]);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      hasFV   = PETSC_TRUE;
      isFE[f] = PETSC_FALSE;
      ierr = PetscFVGetNumComponents(fv, &numComp[f]);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fv, &cellsp[f]);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
  }
  /* Note: We make no attempt to optimize for height. Higher height things just overwrite the lower height results. */
  for (h = 0; h <= maxHeight; h++) {
    ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
    if (!h) {pStart = cStart; pEnd = cEnd;} /* Respect hybrid bounds */
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
      totDim += spDim*numComp[f];
    }
    ierr = DMPlexVecGetClosure(dm, section, localX, pStart, &numValues, NULL);CHKERRQ(ierr);
    if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section point closure size %d != dual space dimension %d", numValues, totDim);
    if (!totDim) continue;
    /* Loop over points at this height */
    ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscFECellGeom fegeom;
      PetscFVCellGeom fvgeom;

      if (hasFE) {
        ierr = DMPlexComputeCellGeometryFEM(dm, p, NULL, fegeom.v0, fegeom.J, NULL, &fegeom.detJ);CHKERRQ(ierr);
        fegeom.dim      = dim - h;
        fegeom.dimEmbed = dimEmbed;
      }
      if (hasFV) {ierr = DMPlexComputeCellGeometryFVM(dm, p, &fvgeom.volume, fvgeom.centroid, NULL);CHKERRQ(ierr);}
      /* Get values for closure */
      for (f = 0, v = 0; f < Nf; ++f) {
        void * const ctx = ctxs ? ctxs[f] : NULL;

        if (!sp[f]) continue;o
        ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
        for (d = 0; d < spDim; ++d) {
          if (funcs[f]) {
            if (isFE[f]) {ierr = PetscDualSpaceApply(sp[f], d, time, &fegeom, numComp[f], funcs[f], ctx, &values[v]);}
            else         {ierr = PetscDualSpaceApplyFVM(sp[f], d, time, &fvgeom, numComp[f], funcs[f], ctx, &values[v]);}
            if (ierr) {
              PetscErrorCode ierr2;
              ierr2 = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr2);
              CHKERRQ(ierr);
            }
          } else {
            for (comp = 0; comp < numComp[f]; ++comp) values[v+comp] = 0.0;
          }
          v += numComp[f];
        }
      }
      ierr = DMPlexVecSetClosure(dm, section, localX, p, values, mode);CHKERRQ(ierr);
    }
    ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  }
  ierr = PetscFree3(isFE, sp, numComp);CHKERRQ(ierr);
  if (maxHeight > 0) {ierr = PetscFree(cellsp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectFunctionLabelLocal_Plex"
PetscErrorCode DMProjectFunctionLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscDualSpace *sp, *cellsp;
  PetscInt       *numComp;
  PetscSection    section;
  PetscScalar    *values;
  PetscBool      *fieldActive;
  PetscInt        numFields, dim, dimEmbed, spDim, totDim = 0, numValues, pStart, pEnd, cStart, cEnd, cEndInterior, f, d, v, i, comp, maxHeight, h;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  if (cEnd <= cStart) PetscFunctionReturn(0);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc2(numFields,&sp,numFields,&numComp);CHKERRQ(ierr);
  ierr = DMPlexGetMaxProjectionHeight(dm,&maxHeight);CHKERRQ(ierr);
  if (maxHeight < 0 || maxHeight > dim) {SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"maximum projection height %d not in [0, %d)\n", maxHeight,dim);}
  if (maxHeight > 0) {ierr = PetscMalloc1(numFields,&cellsp);CHKERRQ(ierr);}
  else               {cellsp = sp;}
  for (h = 0; h <= maxHeight; h++) {
    ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
    if (!h) {pStart = cStart; pEnd = cEnd;}
    if (pEnd <= pStart) continue;
    totDim = 0;
    for (f = 0; f < numFields; ++f) {
      PetscObject  obj;
      PetscClassId id;

      ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE fe = (PetscFE) obj;

        ierr = PetscFEGetNumComponents(fe, &numComp[f]);CHKERRQ(ierr);
        if (!h) {
          ierr = PetscFEGetDualSpace(fe, &cellsp[f]);CHKERRQ(ierr);
          sp[f] = cellsp[f];
        } else {
          ierr = PetscDualSpaceGetHeightSubspace(cellsp[f], h, &sp[f]);CHKERRQ(ierr);
          if (!sp[f]) continue;
        }
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        ierr = PetscFVGetNumComponents(fv, &numComp[f]);CHKERRQ(ierr);
        ierr = PetscFVGetDualSpace(fv, &cellsp[f]);CHKERRQ(ierr);
        sp[f] = cellsp[f];
      } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      totDim += spDim*numComp[f];
    }
    ierr = DMPlexVecGetClosure(dm, section, localX, pStart, &numValues, NULL);CHKERRQ(ierr);
    if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section point closure size %d != dual space dimension %d", numValues, totDim);
    if (!totDim) continue;
    ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFields, PETSC_BOOL, &fieldActive);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) fieldActive[f] = (funcs[f] && sp[f]) ? PETSC_TRUE : PETSC_FALSE;
    for (i = 0; i < numIds; ++i) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        n, p;

      ierr = DMLabelGetStratumIS(label, ids[i], &pointIS);CHKERRQ(ierr);
      if (!pointIS) continue; /* No points with that id on this process */
      ierr = ISGetLocalSize(pointIS, &n);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < n; ++p) {
        const PetscInt    point = points[p];
        PetscFECellGeom   geom;

        if ((point < pStart) || (point >= pEnd)) continue;
        ierr          = DMPlexComputeCellGeometryFEM(dm, point, NULL, geom.v0, geom.J, NULL, &geom.detJ);CHKERRQ(ierr);
        geom.dim      = dim - h;
        geom.dimEmbed = dimEmbed;
        for (f = 0, v = 0; f < numFields; ++f) {
          void * const ctx = ctxs ? ctxs[f] : NULL;

          if (!sp[f]) continue;
          ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
          for (d = 0; d < spDim; ++d) {
            if (funcs[f]) {
              ierr = PetscDualSpaceApply(sp[f], d, time, &geom, numComp[f], funcs[f], ctx, &values[v]);
              if (ierr) {
                PetscErrorCode ierr2;
                ierr2 = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr2);
                ierr2 = DMRestoreWorkArray(dm, numFields, PETSC_BOOL, &fieldActive);CHKERRQ(ierr2);
                CHKERRQ(ierr);
              }
            } else {
              for (comp = 0; comp < numComp[f]; ++comp) values[v+comp] = 0.0;
            }
            v += numComp[f];
          }
        }
        ierr = DMPlexVecSetFieldClosure_Internal(dm, section, localX, fieldActive, point, values, mode);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, numFields, PETSC_BOOL, &fieldActive);CHKERRQ(ierr);
  }
  ierr = PetscFree2(sp, numComp);CHKERRQ(ierr);
  if (maxHeight > 0) {
    ierr = PetscFree(cellsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMProjectFieldLocal_Plex"
PetscErrorCode DMProjectFieldLocal_Plex(DM dm, Vec localU,
                                        void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscScalar[]),
                                        InsertMode mode, Vec localX)
{
  DM              dmAux;
  PetscDS         prob, probAux = NULL;
  Vec             A;
  PetscSection    section, sectionAux = NULL;
  PetscDualSpace *sp;
  PetscInt       *Ncf;
  PetscScalar    *values, *u, *u_x, *a, *a_x;
  PetscReal      *x, *v0, *J, *invJ, detJ;
  PetscInt       *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt        Nf, NfAux = 0, dim, spDim, totDim, numValues, cStart, cEnd, cEndInterior, c, f, d, v, comp, maxHeight;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetMaxProjectionHeight(dm,&maxHeight);CHKERRQ(ierr);
  if (maxHeight > 0) {SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Field projection for height > 0 not supported yet");}
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf, &sp, Nf, &Ncf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, NULL);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localU, 0.0, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *coefficients = NULL, *coefficientsAux = NULL;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, section, localU, c, NULL, &coefficients);CHKERRQ(ierr);
    if (dmAux) {ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &coefficientsAux);CHKERRQ(ierr);}
    for (f = 0, v = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;

      ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE fe = (PetscFE) obj;

        ierr = PetscFEGetDualSpace(fe, &sp[f]);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(fe, &Ncf[f]);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        ierr = PetscFVGetNumComponents(fv, &Ncf[f]);CHKERRQ(ierr);
        ierr = PetscFVGetDualSpace(fv, &sp[f]);CHKERRQ(ierr);
      }
      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        PetscQuadrature  quad;
        const PetscReal *points, *weights;
        PetscInt         numPoints, q;

        if (funcs[f]) {
          ierr = PetscDualSpaceGetFunctional(sp[f], d, &quad);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(quad, NULL, &numPoints, &points, &weights);CHKERRQ(ierr);
          for (q = 0; q < numPoints; ++q) {
            CoordinatesRefToReal(dim, dim, v0, J, &points[q*dim], x);
            ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, coefficients,    NULL, u, u_x, NULL);CHKERRQ(ierr);
            ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, coefficientsAux, NULL, a, a_x, NULL);CHKERRQ(ierr);
            (*funcs[f])(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, x, &values[v]);
          }
        } else {
          for (comp = 0; comp < Ncf[f]; ++comp) values[v+comp] = 0.0;
        }
        v += Ncf[f];
      }
    }
    ierr = DMPlexVecRestoreClosure(dm, section, localU, c, NULL, &coefficients);CHKERRQ(ierr);
    if (dmAux) {ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &coefficientsAux);CHKERRQ(ierr);}
    ierr = DMPlexVecSetClosure(dm, section, localX, c, values, mode);CHKERRQ(ierr);
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscFree2(sp, Ncf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
