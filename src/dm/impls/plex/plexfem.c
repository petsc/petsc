#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetScale"
PetscErrorCode DMPlexGetScale(DM dm, PetscUnit unit, PetscReal *scale)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(scale, 3);
  *scale = mesh->scale[unit];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetScale"
PetscErrorCode DMPlexSetScale(DM dm, PetscUnit unit, PetscReal scale)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->scale[unit] = scale;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscInt epsilon(PetscInt i, PetscInt j, PetscInt k)
{
  switch (i) {
  case 0:
    switch (j) {
    case 0: return 0;
    case 1:
      switch (k) {
      case 0: return 0;
      case 1: return 0;
      case 2: return 1;
      }
    case 2:
      switch (k) {
      case 0: return 0;
      case 1: return -1;
      case 2: return 0;
      }
    }
  case 1:
    switch (j) {
    case 0:
      switch (k) {
      case 0: return 0;
      case 1: return 0;
      case 2: return -1;
      }
    case 1: return 0;
    case 2:
      switch (k) {
      case 0: return 1;
      case 1: return 0;
      case 2: return 0;
      }
    }
  case 2:
    switch (j) {
    case 0:
      switch (k) {
      case 0: return 0;
      case 1: return 1;
      case 2: return 0;
      }
    case 1:
      switch (k) {
      case 0: return -1;
      case 1: return 0;
      case 2: return 0;
      }
    case 2: return 0;
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectRigidBody"
static PetscErrorCode DMPlexProjectRigidBody(PetscInt dim, PetscReal t, const PetscReal X[], PetscInt Nf, PetscScalar *mode, void *ctx)
{
  PetscInt *ctxInt  = (PetscInt *) ctx;
  PetscInt  dim2    = ctxInt[0];
  PetscInt  d       = ctxInt[1];
  PetscInt  i, j, k = dim > 2 ? d - dim : d;

  PetscFunctionBegin;
  if (dim != dim2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input dimension %d does not match context dimension %d", dim, dim2);
  for (i = 0; i < dim; i++) mode[i] = 0.;
  if (d < dim) {
    mode[d] = 1.;
  } else {
    for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
        mode[j] += epsilon(i, j, k)*X[i];
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateRigidBody"
/*@C
  DMPlexCreateRigidBody - for the default global section, create rigid body modes from coordinates

  Collective on DM

  Input Arguments:
. dm - the DM

  Output Argument:
. sp - the null space

  Note: This is necessary to take account of Dirichlet conditions on the displacements

  Level: advanced

.seealso: MatNullSpaceCreate()
@*/
PetscErrorCode DMPlexCreateRigidBody(DM dm, MatNullSpace *sp)
{
  MPI_Comm       comm;
  Vec            mode[6];
  PetscSection   section, globalSection;
  PetscInt       dim, dimEmbed, n, m, d, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &n);CHKERRQ(ierr);
  m    = (dim*(dim+1))/2;
  ierr = VecCreate(comm, &mode[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(mode[0], n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(mode[0]);CHKERRQ(ierr);
  for (i = 1; i < m; ++i) {ierr = VecDuplicate(mode[0], &mode[i]);CHKERRQ(ierr);}
  for (d = 0; d < m; d++) {
    PetscInt ctx[2];
    void    *voidctx = (void *) (&ctx[0]);
    PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *) = DMPlexProjectRigidBody;

    ctx[0] = dimEmbed;
    ctx[1] = d;
    ierr = DMProjectFunction(dm, 0.0, &func, &voidctx, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
  }
  for (i = 0; i < dim; ++i) {ierr = VecNormalize(mode[i], NULL);CHKERRQ(ierr);}
  /* Orthonormalize system */
  for (i = dim; i < m; ++i) {
    PetscScalar dots[6];

    ierr = VecMDot(mode[i], i, mode, dots);CHKERRQ(ierr);
    for (j = 0; j < i; ++j) dots[j] *= -1.0;
    ierr = VecMAXPY(mode[i], i, dots, mode);CHKERRQ(ierr);
    ierr = VecNormalize(mode[i], NULL);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, m, mode, sp);CHKERRQ(ierr);
  for (i = 0; i< m; ++i) {ierr = VecDestroy(&mode[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetMaxProjectionHeight"
/*@
  DMPlexSetMaxProjectionHeight - In DMPlexProjectXXXLocal() functions, the projected values of a basis function's dofs
  are computed by associating the basis function with one of the mesh points in its transitively-closed support, and
  evaluating the dual space basis of that point.  A basis function is associated with the point in its
  transitively-closed support whose mesh height is highest (w.r.t. DAG height), but not greater than the maximum
  projection height, which is set with this function.  By default, the maximum projection height is zero, which means
  that only mesh cells are used to project basis functions.  A height of one, for example, evaluates a cell-interior
  basis functions using its cells dual space basis, but all other basis functions with the dual space basis of a face.

  Input Parameters:
+ dm - the DMPlex object
- height - the maximum projection height >= 0

  Level: advanced

.seealso: DMPlexGetMaxProjectionHeight(), DMProjectFunctionLocal(), DMProjectFunctionLabelLocal()
@*/
PetscErrorCode DMPlexSetMaxProjectionHeight(DM dm, PetscInt height)
{
  DM_Plex *plex = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  plex->maxProjectionHeight = height;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetMaxProjectionHeight"
/*@
  DMPlexGetMaxProjectionHeight - Get the maximum height (w.r.t. DAG) of mesh points used to evaluate dual bases in
  DMPlexProjectXXXLocal() functions.

  Input Parameters:
. dm - the DMPlex object

  Output Parameters:
. height - the maximum projection height

  Level: intermediate

.seealso: DMPlexSetMaxProjectionHeight(), DMProjectFunctionLocal(), DMProjectFunctionLabelLocal()
@*/
PetscErrorCode DMPlexGetMaxProjectionHeight(DM dm, PetscInt *height)
{
  DM_Plex *plex = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *height = plex->maxProjectionHeight;
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
        ierr = PetscFVGetDualSpace(fv, &sp[f]);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) sp[f]);CHKERRQ(ierr);
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
  ierr = DMPlexGetMaxProjectionHeight(dm,&maxHeight);CHKERRQ(ierr);
  if (maxHeight < 0 || maxHeight > dim) {SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"maximum projection height %d not in [0, %d)\n", maxHeight,dim);}
  if (maxHeight > 0) {
    ierr = PetscMalloc1(Nf,&cellsp);CHKERRQ(ierr);
  }
  else {
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
  for (h = 0; h <= maxHeight; h++) {
    ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
    if (!h) {pStart = cStart; pEnd = cEnd;}
    if (pEnd <= pStart) continue;
    totDim = 0;
    for (f = 0; f < Nf; ++f) {
      if (!h) {
        sp[f] = cellsp[f];
      }
      else {
        ierr = PetscDualSpaceGetHeightSubspace(cellsp[f], h, &sp[f]);CHKERRQ(ierr);
        if (!sp[f]) {
          continue;
        }
      }
      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      totDim += spDim*numComp[f];
    }
    ierr = DMPlexVecGetClosure(dm, section, localX, pStart, &numValues, NULL);CHKERRQ(ierr);
    if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section point closure size %d != dual space dimension %d", numValues, totDim);
    if (!totDim) continue;
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
      for (f = 0, v = 0; f < Nf; ++f) {
        void * const ctx = ctxs ? ctxs[f] : NULL;

        if (!sp[f]) continue;
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
        ierr = PetscObjectReference((PetscObject) sp[f]);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertBoundaryValues_FEM_Internal"
static PetscErrorCode DMPlexInsertBoundaryValues_FEM_Internal(DM dm, PetscReal time, PetscInt field, DMLabel label, PetscInt numids, const PetscInt ids[], PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void *ctx, Vec locX)
{
  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ctxs;
  PetscInt          numFields;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscCalloc2(numFields,&funcs,numFields,&ctxs);CHKERRQ(ierr);
  funcs[field] = func;
  ctxs[field]  = ctx;
  ierr = DMProjectFunctionLabelLocal(dm, time, label, numids, ids, funcs, ctxs, INSERT_BC_VALUES, locX);CHKERRQ(ierr);
  ierr = PetscFree2(funcs,ctxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertBoundaryValues_FVM_Internal"
/* This ignores numcomps/comps */
static PetscErrorCode DMPlexInsertBoundaryValues_FVM_Internal(DM dm, PetscReal time, Vec faceGeometry, Vec cellGeometry, Vec Grad,
                                                              PetscInt field, DMLabel label, PetscInt numids, const PetscInt ids[], PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*), void *ctx, Vec locX)
{
  PetscDS            prob;
  PetscSF            sf;
  DM                 dmFace, dmCell, dmGrad;
  const PetscScalar *facegeom, *cellgeom = NULL, *grad;
  const PetscInt    *leaves;
  PetscScalar       *x, *fx;
  PetscInt           dim, nleaves, loc, fStart, fEnd, pdim, i;
  PetscErrorCode     ierr, ierru = 0;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL);CHKERRQ(ierr);
  nleaves = PetscMax(0, nleaves);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  if (cellGeometry) {
    ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  }
  if (Grad) {
    PetscFV fv;

    ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fv);CHKERRQ(ierr);
    ierr = VecGetDM(Grad, &dmGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Grad, &grad);CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, pdim, PETSC_SCALAR, &fx);CHKERRQ(ierr);
  }
  ierr = VecGetArray(locX, &x);CHKERRQ(ierr);
  for (i = 0; i < numids; ++i) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    ierr = DMLabelGetStratumIS(label, ids[i], &faceIS);CHKERRQ(ierr);
    if (!faceIS) continue; /* No points with that id on this process */
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      const PetscInt         face = faces[f], *cells;
      PetscFVFaceGeom        *fg;

      if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
      ierr = PetscFindInt(face, nleaves, (PetscInt *) leaves, &loc);CHKERRQ(ierr);
      if (loc >= 0) continue;
      ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
      if (Grad) {
        PetscFVCellGeom       *cg;
        PetscScalar           *cx, *cgrad;
        PetscScalar           *xG;
        PetscReal              dx[3];
        PetscInt               d;

        ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dm, cells[0], x, &cx);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmGrad, cells[0], grad, &cgrad);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG);CHKERRQ(ierr);
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, fg->centroid, dx);
        for (d = 0; d < pdim; ++d) fx[d] = cx[d] + DMPlex_DotD_Internal(dim, &cgrad[d*dim], dx);
        ierru = (*func)(time, fg->centroid, fg->normal, fx, xG, ctx);
        if (ierru) {
          ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
          ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
          goto cleanup;
        }
      } else {
        PetscScalar       *xI;
        PetscScalar       *xG;

        ierr = DMPlexPointLocalRead(dm, cells[0], x, &xI);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG);CHKERRQ(ierr);
        ierru = (*func)(time, fg->centroid, fg->normal, xI, xG, ctx);
        if (ierru) {
          ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
          ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
          goto cleanup;
        }
      }
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  cleanup:
  ierr = VecRestoreArray(locX, &x);CHKERRQ(ierr);
  if (Grad) {
    ierr = DMRestoreWorkArray(dm, pdim, PETSC_SCALAR, &fx);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Grad, &grad);CHKERRQ(ierr);
  }
  if (cellGeometry) {ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);}
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  CHKERRQ(ierru);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertBoundaryValues"
PetscErrorCode DMPlexInsertBoundaryValues(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscInt       numBd, b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 2);
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 4);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 5);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 6);}
  ierr = DMGetNumBoundary(dm, &numBd);CHKERRQ(ierr);
  for (b = 0; b < numBd; ++b) {
    PetscBool       isEssential;
    const char     *labelname;
    DMLabel         label;
    PetscInt        field;
    PetscObject     obj;
    PetscClassId    id;
    void          (*func)();
    PetscInt        numids;
    const PetscInt *ids;
    void           *ctx;

    ierr = DMGetBoundary(dm, b, &isEssential, NULL, &labelname, &field, NULL, NULL, &func, &numids, &ids, &ctx);CHKERRQ(ierr);
    if (insertEssential != isEssential) continue;
    ierr = DMGetLabel(dm, labelname, &label);CHKERRQ(ierr);
    ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      if (!isEssential) continue; /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      ierr = DMPlexLabelAddCells(dm,label);CHKERRQ(ierr);
      ierr = DMPlexInsertBoundaryValues_FEM_Internal(dm, time, field, label, numids, ids, (PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *)) func, ctx, locX);CHKERRQ(ierr);
      ierr = DMPlexLabelClearCells(dm,label);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      if (!faceGeomFVM) continue;
      ierr = DMPlexInsertBoundaryValues_FVM_Internal(dm, time, faceGeomFVM, cellGeomFVM, gradFVM,
                                                     field, label, numids, ids, (PetscErrorCode (*)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*)) func, ctx, locX);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComputeL2Diff_Plex"
PetscErrorCode DMComputeL2Diff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  const PetscInt   debug = 0;
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscReal        localDiff = 0.0;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, numFields, numComponents = 0, numQuadPoints, cStart, cEnd, cEndInterior, c, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, &quadWeights);CHKERRQ(ierr);
  ierr = PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
      else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
      else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb*Nc, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < numQuadPoints; ++q) {
        CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], coords);
        ierr = (*funcs[field])(dim, time, coords, Nc, funcVal, ctx);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
          ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
          ierr2 = PetscFree6(funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
        else if (id == PETSCFV_CLASSID) {ierr = PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
        else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
        for (fc = 0; fc < Nc; ++fc) {
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*quadWeights[q]*detJ;
        }
      }
      fieldOffset += Nb*Nc;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree6(funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComputeL2GradientDiff_Plex"
PetscErrorCode DMComputeL2GradientDiff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, const PetscReal n[], PetscReal *diff)
{
  const PetscInt  debug = 0;
  PetscSection    section;
  PetscQuadrature quad;
  Vec             localX;
  PetscScalar    *funcVal, *interpolantVec;
  PetscReal      *coords, *realSpaceDer, *v0, *J, *invJ, detJ;
  PetscReal       localDiff = 0.0;
  PetscInt        dim, numFields, numComponents = 0, cStart, cEnd, cEndInterior, c, field, fieldOffset, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscFE  fe;
    PetscInt Nc;

    ierr = DMGetField(dm, field, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    numComponents += Nc;
  }
  /* ierr = DMProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr); */
  ierr = PetscMalloc7(numComponents,&funcVal,dim,&coords,dim,&realSpaceDer,dim,&v0,dim*dim,&J,dim*dim,&invJ,dim,&interpolantVec);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscFE          fe;
      void * const     ctx = ctxs ? ctxs[field] : NULL;
      const PetscReal *quadPoints, *quadWeights;
      PetscReal       *basisDer;
      PetscInt         numQuadPoints, Nb, Ncomp, q, d, e, fc, f, g;

      ierr = DMGetField(dm, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, &quadWeights);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Ncomp);CHKERRQ(ierr);
      ierr = PetscFEGetDefaultTabulation(fe, NULL, &basisDer, NULL);CHKERRQ(ierr);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb*Ncomp, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < numQuadPoints; ++q) {
        for (d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for (e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        ierr = (*funcs[field])(dim, time, coords, n, numFields, funcVal, ctx);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
          ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
          ierr2 = PetscFree7(funcVal,coords,realSpaceDer,v0,J,invJ,interpolantVec);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        for (fc = 0; fc < Ncomp; ++fc) {
          PetscScalar interpolant = 0.0;

          for (d = 0; d < dim; ++d) interpolantVec[d] = 0.0;
          for (f = 0; f < Nb; ++f) {
            const PetscInt fidx = f*Ncomp+fc;

            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+fidx)*dim+g];
              }
              interpolantVec[d] += x[fieldOffset+fidx]*realSpaceDer[d];
            }
          }
          for (d = 0; d < dim; ++d) interpolant += interpolantVec[d]*n[d];
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d fieldDer %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q]*detJ;
        }
      }
      comp        += Ncomp;
      fieldOffset += Nb*Ncomp;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree7(funcVal,coords,realSpaceDer,v0,J,invJ,interpolantVec);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComputeL2FieldDiff_Plex"
PetscErrorCode DMComputeL2FieldDiff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  const PetscInt   debug = 0;
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscReal       *localDiff;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, numFields, numComponents = 0, numQuadPoints, cStart, cEnd, cEndInterior, c, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, &quadWeights);CHKERRQ(ierr);
  ierr = PetscCalloc7(numFields,&localDiff,numComponents,&funcVal,numComponents,&interpolant,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      PetscReal       elemDiff = 0.0;

      ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
      else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
      else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb*Nc, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < numQuadPoints; ++q) {
        CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], coords);
        ierr = (*funcs[field])(dim, time, coords, numFields, funcVal, ctx);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
          ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
          ierr2 = PetscFree7(localDiff,funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
        else if (id == PETSCFV_CLASSID) {ierr = PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
        else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
        for (fc = 0; fc < Nc; ++fc) {
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*quadWeights[q]*detJ;
        }
      }
      fieldOffset += Nb*Nc;
      localDiff[field] += elemDiff;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(localDiff, diff, numFields, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) diff[field] = PetscSqrtReal(diff[field]);
  ierr = PetscFree7(localDiff,funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeL2DiffVec"
/*@C
  DMPlexComputeL2DiffVec - This function computes the cellwise L_2 difference between a function u and an FEM interpolant solution u_h, and stores it in a Vec.

  Input Parameters:
+ dm    - The DM
. time  - The time
. funcs - The functions to evaluate for each field component: NULL means that component does not contribute to error calculation
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h

  Output Parameter:
. D - A Vec which holds the difference ||u - u_h||_2 for each cell

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2Diff(), DMPlexComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeL2DiffVec(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, Vec D)
{
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, numFields, numComponents = 0, numQuadPoints, cStart, cEnd, cEndInterior, c, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecSet(D, 0.0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, &quadWeights);CHKERRQ(ierr);
  ierr = PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscScalar  elemDiff = 0.0;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      ierr = DMGetField(dm, field, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
      else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
      else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
      if (funcs[field]) {
        for (q = 0; q < numQuadPoints; ++q) {
          CoordinatesRefToReal(dim, dim, v0, J, &quadPoints[q*dim], coords);
          ierr = (*funcs[field])(dim, time, coords, Nc, funcVal, ctx);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
            ierr2 = PetscFree6(funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr2);
            ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
          else if (id == PETSCFV_CLASSID) {ierr = PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
          else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
          for (fc = 0; fc < Nc; ++fc) {
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*quadWeights[q]*detJ;
          }
        }
      }
      fieldOffset += Nb*Nc;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    ierr = VecSetValue(D, c - cStart, elemDiff, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = VecSqrtAbs(D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeIntegralFEM"
/*@
  DMPlexComputeIntegralFEM - Form the local integral F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. integral - Local integral for each field

  Level: developer

.seealso: DMPlexComputeResidualFEM()
@*/
PetscErrorCode DMPlexComputeIntegralFEM(DM dm, Vec X, PetscReal *integral, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  DM                dmAux;
  Vec               localX, A;
  PetscDS           prob, probAux = NULL;
  PetscSection      section, sectionAux;
  PetscFECellGeom  *cgeom;
  PetscScalar      *u, *a = NULL;
  PetscReal        *lintegral, *vol;
  PetscInt          dim, Nf, f, numCells, cStart, cEnd, cEndInterior, c;
  PetscInt          totDim, totDimAux;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localX, 0.0, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = PetscMalloc4(Nf,&lintegral,numCells*totDim,&u,numCells,&cgeom,numCells,&vol);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  for (f = 0; f < Nf; ++f) {lintegral[f] = 0.0;}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, cgeom[c].v0, cgeom[c].J, cgeom[c].invJ, &cgeom[c].detJ);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol[c], NULL, NULL);CHKERRQ(ierr);
    if (cgeom[c].detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", cgeom[c].detJ, c);
    ierr = DMPlexVecGetClosure(dm, section, localX, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, localX, c, NULL, &x);CHKERRQ(ierr);
    if (dmAux) {
      ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE         fe = (PetscFE) obj;
      PetscQuadrature q;
      PetscInt        Nq, Nb;

      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(q, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      blockSize = Nb*Nq;
      batchSize = numBlocks * blockSize;
      ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      ierr = PetscFEIntegrate(fe, prob, f, Ne, cgeom, u, probAux, a, lintegral);CHKERRQ(ierr);
      ierr = PetscFEIntegrate(fe, prob, f, Nr, &cgeom[offset], &u[offset*totDim], probAux, &a[offset*totDimAux], lintegral);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      /* PetscFV  fv = (PetscFV) obj; */
      PetscInt       foff;
      PetscPointFunc obj_func;
      PetscScalar    lint;

      ierr = PetscDSGetObjective(prob, f, &obj_func);CHKERRQ(ierr);
      ierr = PetscDSGetFieldOffset(prob, f, &foff);CHKERRQ(ierr);
      if (obj_func) {
        for (c = 0; c < numCells; ++c) {
          /* TODO: Need full pointwise interpolation and get centroid for x argument */
          obj_func(dim, Nf, 0, NULL, NULL, &u[totDim*c+foff], NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.0, NULL, &lint);
          lintegral[f] = PetscRealPart(lint)*vol[c];
        }
      }
    } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
  }
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  if (mesh->printFEM) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Local integral:");CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), " %g", lintegral[f]);CHKERRQ(ierr);}
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "\n");CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(lintegral, integral, Nf, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  ierr = PetscFree4(lintegral,u,cgeom,vol);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeInterpolatorNested"
/*@
  DMPlexComputeInterpolatorNested - Form the local portion of the interpolation matrix I from the coarse DM to the uniformly refined DM.

  Input Parameters:
+ dmf  - The fine mesh
. dmc  - The coarse mesh
- user - The user context

  Output Parameter:
. In  - The interpolation matrix

  Level: developer

.seealso: DMPlexComputeInterpolatorGeneral(), DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeInterpolatorNested(DM dmc, DM dmf, Mat In, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dmc->data;
  const char       *name  = "Interpolator";
  PetscDS           prob;
  PetscFE          *feRef;
  PetscFV          *fvRef;
  PetscSection      fsection, fglobalSection;
  PetscSection      csection, cglobalSection;
  PetscScalar      *elemMat;
  PetscInt          dim, Nf, f, fieldI, fieldJ, offsetI, offsetJ, cStart, cEnd, cEndInterior, c;
  PetscInt          cTotDim, rTotDim = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dmf, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmf, &fglobalSection);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmc, &cglobalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(fsection, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmc, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dmc, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMGetDS(dmf, &prob);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf,&feRef,Nf,&fvRef);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     rNb = 0, Nc = 0;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFERefine(fe, &feRef[f]);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(feRef[f], &rNb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      ierr = PetscFVRefine(fv, &fvRef[f]);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvRef[f], &Q);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(Q, &rNb);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    }
    rTotDim += rNb*Nc;
  }
  ierr = PetscDSGetTotalDimension(prob, &cTotDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(rTotDim*cTotDim,&elemMat);CHKERRQ(ierr);
  ierr = PetscMemzero(elemMat, rTotDim*cTotDim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (fieldI = 0, offsetI = 0; fieldI < Nf; ++fieldI) {
    PetscDualSpace   Qref;
    PetscQuadrature  f;
    const PetscReal *qpoints, *qweights;
    PetscReal       *points;
    PetscInt         npoints = 0, Nc, Np, fpdim, i, k, p, d;

    /* Compose points from all dual basis functionals */
    if (feRef[fieldI]) {
      ierr = PetscFEGetDualSpace(feRef[fieldI], &Qref);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feRef[fieldI], &Nc);CHKERRQ(ierr);
    } else {
      ierr = PetscFVGetDualSpace(fvRef[fieldI], &Qref);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fvRef[fieldI], &Nc);CHKERRQ(ierr);
    }
    ierr = PetscDualSpaceGetDimension(Qref, &fpdim);CHKERRQ(ierr);
    for (i = 0; i < fpdim; ++i) {
      ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(f, NULL, &Np, NULL, NULL);CHKERRQ(ierr);
      npoints += Np;
    }
    ierr = PetscMalloc1(npoints*dim,&points);CHKERRQ(ierr);
    for (i = 0, k = 0; i < fpdim; ++i) {
      ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(f, NULL, &Np, &qpoints, NULL);CHKERRQ(ierr);
      for (p = 0; p < Np; ++p, ++k) for (d = 0; d < dim; ++d) points[k*dim+d] = qpoints[p*dim+d];
    }

    for (fieldJ = 0, offsetJ = 0; fieldJ < Nf; ++fieldJ) {
      PetscObject  obj;
      PetscClassId id;
      PetscReal   *B;
      PetscInt     NcJ = 0, cpdim = 0, j;

      ierr = PetscDSGetDiscretization(prob, fieldJ, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE fe = (PetscFE) obj;

        /* Evaluate basis at points */
        ierr = PetscFEGetNumComponents(fe, &NcJ);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &cpdim);CHKERRQ(ierr);
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          if (Nc != NcJ) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %d does not match coarse field %d", Nc, NcJ);
          ierr = PetscFEGetTabulation(fe, npoints, points, &B, NULL, NULL);CHKERRQ(ierr);
          for (i = 0, k = 0; i < fpdim; ++i) {
            ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
            ierr = PetscQuadratureGetData(f, NULL, &Np, NULL, &qweights);CHKERRQ(ierr);
            for (p = 0; p < Np; ++p, ++k) {
              for (j = 0; j < cpdim; ++j) {
                for (c = 0; c < Nc; ++c) elemMat[(offsetI + i*Nc + c)*cTotDim + offsetJ + j*NcJ + c] += B[k*cpdim*NcJ+j*Nc+c]*qweights[p];
              }
            }
          }
          ierr = PetscFERestoreTabulation(fe, npoints, points, &B, NULL, NULL);CHKERRQ(ierr);CHKERRQ(ierr);
        }
      } else if (id == PETSCFV_CLASSID) {
        PetscFV        fv = (PetscFV) obj;

        /* Evaluate constant function at points */
        ierr = PetscFVGetNumComponents(fv, &NcJ);CHKERRQ(ierr);
        cpdim = 1;
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          if (Nc != NcJ) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %d does not match coarse field %d", Nc, NcJ);
          for (i = 0, k = 0; i < fpdim; ++i) {
            ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
            ierr = PetscQuadratureGetData(f, NULL, &Np, NULL, &qweights);CHKERRQ(ierr);
            for (p = 0; p < Np; ++p, ++k) {
              for (j = 0; j < cpdim; ++j) {
                for (c = 0; c < Nc; ++c) elemMat[(offsetI + i*Nc + c)*cTotDim + offsetJ + j*NcJ + c] += 1.0*qweights[p];
              }
            }
          }
        }
      }
      offsetJ += cpdim*NcJ;
    }
    offsetI += fpdim*Nc;
    ierr = PetscFree(points);CHKERRQ(ierr);
  }
  if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(0, name, rTotDim, cTotDim, elemMat);CHKERRQ(ierr);}
  /* Preallocate matrix */
  {
    Mat          preallocator;
    PetscScalar *vals;
    PetscInt    *cellCIndices, *cellFIndices;
    PetscInt     locRows, locCols, cell;

    ierr = MatGetLocalSize(In, &locRows, &locCols);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject) In), &preallocator);CHKERRQ(ierr);
    ierr = MatSetType(preallocator, MATPREALLOCATOR);CHKERRQ(ierr);
    ierr = MatSetSizes(preallocator, locRows, locCols, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetUp(preallocator);CHKERRQ(ierr);
    ierr = PetscCalloc3(rTotDim*cTotDim, &vals,cTotDim,&cellCIndices,rTotDim,&cellFIndices);CHKERRQ(ierr);
    for (cell = cStart; cell < cEnd; ++cell) {
      ierr = DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, cell, cellCIndices, cellFIndices);CHKERRQ(ierr);
      ierr = MatSetValues(preallocator, rTotDim, cellFIndices, cTotDim, cellCIndices, vals, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree3(vals,cellCIndices,cellFIndices);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatPreallocatorPreallocate(preallocator, PETSC_TRUE, In);CHKERRQ(ierr);
    ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  }
  /* Fill matrix */
  ierr = MatZeroEntries(In);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexMatSetClosureRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES);CHKERRQ(ierr);
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&feRef[f]);CHKERRQ(ierr);}
  ierr = PetscFree2(feRef,fvRef);CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s:\n", name);CHKERRQ(ierr);
    ierr = MatChop(In, 1.0e-10);CHKERRQ(ierr);
    ierr = MatView(In, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeInterpolatorGeneral"
/*@
  DMPlexComputeInterpolatorGeneral - Form the local portion of the interpolation matrix I from the coarse DM to a non-nested fine DM.

  Input Parameters:
+ dmf  - The fine mesh
. dmc  - The coarse mesh
- user - The user context

  Output Parameter:
. In  - The interpolation matrix

  Level: developer

.seealso: DMPlexComputeInterpolatorNested(), DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeInterpolatorGeneral(DM dmc, DM dmf, Mat In, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dmf->data;
  const char    *name = "Interpolator";
  PetscDS        prob;
  PetscSection   fsection, csection, globalFSection, globalCSection;
  PetscHashJK    ht;
  PetscLayout    rLayout;
  PetscInt      *dnz, *onz;
  PetscInt       locRows, rStart, rEnd;
  PetscReal     *x, *v0, *J, *invJ, detJ;
  PetscReal     *v0c, *Jc, *invJc, detJc;
  PetscScalar   *elemMat;
  PetscInt       dim, Nf, field, totDim, cStart, cEnd, cell, ccell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dmc, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dmc, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(totDim*totDim, &elemMat);CHKERRQ(ierr);

  ierr = MatGetLocalSize(In, &locRows, NULL);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject) In), &rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rLayout, locRows);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
  ierr = PetscCalloc2(locRows,&dnz,locRows,&onz);CHKERRQ(ierr);
  ierr = PetscHashJKCreate(&ht);CHKERRQ(ierr);
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscDualSpace   Q = NULL;
    PetscQuadrature  f;
    const PetscReal *qpoints;
    PetscInt         Nc, Np, fpdim, i, d;

    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetDualSpace(fv, &Q);CHKERRQ(ierr);
      Nc   = 1;
    }
    ierr = PetscDualSpaceGetDimension(Q, &fpdim);CHKERRQ(ierr);
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices,   *cindices;
      PetscInt  numFIndices, numCIndices;

      ierr = DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      if (numFIndices != fpdim*Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %d != %d dual basis vecs", numFIndices, fpdim*Nc);
      for (i = 0; i < fpdim; ++i) {
        Vec             pointVec;
        PetscScalar    *pV;
        PetscSF         coarseCellSF = NULL;
        const PetscSFNode *coarseCells;
        PetscInt        numCoarseCells, q, r, c;

        /* Get points from the dual basis functional quadrature */
        ierr = PetscDualSpaceGetFunctional(Q, i, &f);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(f, NULL, &Np, &qpoints, NULL);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(pointVec, dim);CHKERRQ(ierr);
        ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
        for (q = 0; q < Np; ++q) {
          /* Transform point to real space */
          CoordinatesRefToReal(dim, dim, v0, J, &qpoints[q*dim], x);
          for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
        }
        ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
        /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
        ierr = DMLocatePoints(dmc, pointVec, &coarseCellSF);CHKERRQ(ierr);
        ierr = PetscSFViewFromOptions(coarseCellSF, NULL, "-interp_sf_view");CHKERRQ(ierr);
        /* Update preallocation info */
        ierr = PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells);CHKERRQ(ierr);
        if (numCoarseCells != Np) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
        for (r = 0; r < Nc; ++r) {
          PetscHashJKKey  key;
          PetscHashJKIter missing, iter;

          key.j = findices[i*Nc+r];
          if (key.j < 0) continue;
          /* Get indices for coarse elements */
          for (ccell = 0; ccell < numCoarseCells; ++ccell) {
            ierr = DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, &numCIndices, &cindices, NULL);CHKERRQ(ierr);
            for (c = 0; c < numCIndices; ++c) {
              key.k = cindices[c];
              if (key.k < 0) continue;
              ierr = PetscHashJKPut(ht, key, &missing, &iter);CHKERRQ(ierr);
              if (missing) {
                ierr = PetscHashJKSet(ht, iter, 1);CHKERRQ(ierr);
                if ((key.k >= rStart) && (key.k < rEnd)) ++dnz[key.j-rStart];
                else                                     ++onz[key.j-rStart];
              }
            }
            ierr = DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, &numCIndices, &cindices, NULL);CHKERRQ(ierr);
          }
        }
        ierr = PetscSFDestroy(&coarseCellSF);CHKERRQ(ierr);
        ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscHashJKDestroy(&ht);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(In, 1, dnz, onz, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetOption(In, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscDualSpace   Q = NULL;
    PetscQuadrature  f;
    const PetscReal *qpoints, *qweights;
    PetscInt         Nc, Np, fpdim, i, d;

    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetDualSpace(fv, &Q);CHKERRQ(ierr);
      Nc   = 1;
    }
    ierr = PetscDualSpaceGetDimension(Q, &fpdim);CHKERRQ(ierr);
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices,   *cindices;
      PetscInt  numFIndices, numCIndices;

      ierr = DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      if (numFIndices != fpdim*Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %d != %d dual basis vecs", numFIndices, fpdim*Nc);
      for (i = 0; i < fpdim; ++i) {
        Vec             pointVec;
        PetscScalar    *pV;
        PetscSF         coarseCellSF = NULL;
        const PetscSFNode *coarseCells;
        PetscInt        numCoarseCells, cpdim, q, c, j;

        /* Get points from the dual basis functional quadrature */
        ierr = PetscDualSpaceGetFunctional(Q, i, &f);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(f, NULL, &Np, &qpoints, &qweights);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(pointVec, dim);CHKERRQ(ierr);
        ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
        for (q = 0; q < Np; ++q) {
          /* Transform point to real space */
          CoordinatesRefToReal(dim, dim, v0, J, &qpoints[q*dim], x);
          for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
        }
        ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
        /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
        ierr = DMLocatePoints(dmc, pointVec, &coarseCellSF);CHKERRQ(ierr);
        /* Update preallocation info */
        ierr = PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells);CHKERRQ(ierr);
        if (numCoarseCells != Np) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
        ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
        for (ccell = 0; ccell < numCoarseCells; ++ccell) {
          PetscReal pVReal[3];

          ierr = DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, &numCIndices, &cindices, NULL);CHKERRQ(ierr);
          /* Transform points from real space to coarse reference space */
          ierr = DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
          CoordinatesRealToRef(dim, dim, v0c, invJc, pVReal, x);

          if (id == PETSCFE_CLASSID) {
            PetscFE    fe = (PetscFE) obj;
            PetscReal *B;

            /* Evaluate coarse basis on contained point */
            ierr = PetscFEGetDimension(fe, &cpdim);CHKERRQ(ierr);
            ierr = PetscFEGetTabulation(fe, 1, x, &B, NULL, NULL);CHKERRQ(ierr);
            /* Get elemMat entries by multiplying by weight */
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[(c*cpdim + j)*Nc + c] = B[j*Nc + c]*qweights[ccell];
            }
            ierr = PetscFERestoreTabulation(fe, 1, x, &B, NULL, NULL);CHKERRQ(ierr);CHKERRQ(ierr);
          } else {
            cpdim = 1;
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[(c*cpdim + j)*Nc + c] = 1.0*qweights[ccell];
            }
          }
          /* Update interpolator */
          if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, Nc, numCIndices, elemMat);CHKERRQ(ierr);}
          ierr = MatSetValues(In, Nc, &findices[i*Nc], numCIndices, cindices, elemMat, INSERT_VALUES);CHKERRQ(ierr);
          ierr = DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, &numCIndices, &cindices, NULL);CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&coarseCellSF);CHKERRQ(ierr);
        ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  ierr = PetscFree3(v0c,Jc,invJc);CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeInjectorFEM"
PetscErrorCode DMPlexComputeInjectorFEM(DM dmc, DM dmf, VecScatter *sc, void *user)
{
  PetscDS        prob;
  PetscFE       *feRef;
  PetscFV       *fvRef;
  Vec            fv, cv;
  IS             fis, cis;
  PetscSection   fsection, fglobalSection, csection, cglobalSection;
  PetscInt      *cmap, *cellCIndices, *cellFIndices, *cindices, *findices;
  PetscInt       cTotDim, fTotDim = 0, Nf, f, field, cStart, cEnd, cEndInterior, c, dim, d, startC, endC, offsetC, offsetF, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_InjectorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dmf, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmf, &fglobalSection);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmc, &cglobalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(fsection, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmc, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dmc, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMGetDS(dmc, &prob);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf,&feRef,Nf,&fvRef);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     fNb = 0, Nc = 0;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFERefine(fe, &feRef[f]);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(feRef[f], &fNb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      ierr = PetscFVRefine(fv, &fvRef[f]);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvRef[f], &Q);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(Q, &fNb);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    }
    fTotDim += fNb*Nc;
  }
  ierr = PetscDSGetTotalDimension(prob, &cTotDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(cTotDim,&cmap);CHKERRQ(ierr);
  for (field = 0, offsetC = 0, offsetF = 0; field < Nf; ++field) {
    PetscFE        feC;
    PetscFV        fvC;
    PetscDualSpace QF, QC;
    PetscInt       NcF, NcC, fpdim, cpdim;

    if (feRef[field]) {
      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &feC);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feC, &NcC);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feRef[field], &NcF);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(feRef[field], &QF);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QF, &fpdim);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(feC, &QC);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QC, &cpdim);CHKERRQ(ierr);
    } else {
      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fvC);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fvC, &NcC);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fvRef[field], &NcF);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvRef[field], &QF);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QF, &fpdim);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvC, &QC);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QC, &cpdim);CHKERRQ(ierr);
    }
    if (NcF != NcC) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %d does not match coarse field %d", NcF, NcC);
    for (c = 0; c < cpdim; ++c) {
      PetscQuadrature  cfunc;
      const PetscReal *cqpoints;
      PetscInt         NpC;
      PetscBool        found = PETSC_FALSE;

      ierr = PetscDualSpaceGetFunctional(QC, c, &cfunc);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(cfunc, NULL, &NpC, &cqpoints, NULL);CHKERRQ(ierr);
      if (NpC != 1 && feRef[field]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not know how to do injection for moments");
      for (f = 0; f < fpdim; ++f) {
        PetscQuadrature  ffunc;
        const PetscReal *fqpoints;
        PetscReal        sum = 0.0;
        PetscInt         NpF, comp;

        ierr = PetscDualSpaceGetFunctional(QF, f, &ffunc);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(ffunc, NULL, &NpF, &fqpoints, NULL);CHKERRQ(ierr);
        if (NpC != NpF) continue;
        for (d = 0; d < dim; ++d) sum += PetscAbsReal(cqpoints[d] - fqpoints[d]);
        if (sum > 1.0e-9) continue;
        for (comp = 0; comp < NcC; ++comp) {
          cmap[(offsetC+c)*NcC+comp] = (offsetF+f)*NcF+comp;
        }
        found = PETSC_TRUE;
        break;
      }
      if (!found) {
        /* TODO We really want the average here, but some asshole put VecScatter in the interface */
        if (fvRef[field]) {
          PetscInt comp;
          for (comp = 0; comp < NcC; ++comp) {
            cmap[(offsetC+c)*NcC+comp] = (offsetF+0)*NcF+comp;
          }
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate matching functional for injection");
      }
    }
    offsetC += cpdim*NcC;
    offsetF += fpdim*NcF;
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&feRef[f]);CHKERRQ(ierr);ierr = PetscFVDestroy(&fvRef[f]);CHKERRQ(ierr);}
  ierr = PetscFree2(feRef,fvRef);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmf, &fv);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmc, &cv);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(cv, &startC, &endC);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(cglobalSection, &m);CHKERRQ(ierr);
  ierr = PetscMalloc2(cTotDim,&cellCIndices,fTotDim,&cellFIndices);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&cindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&findices);CHKERRQ(ierr);
  for (d = 0; d < m; ++d) cindices[d] = findices[d] = -1;
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, c, cellCIndices, cellFIndices);CHKERRQ(ierr);
    for (d = 0; d < cTotDim; ++d) {
      if ((cellCIndices[d] < startC) || (cellCIndices[d] >= endC)) continue;
      if ((findices[cellCIndices[d]-startC] >= 0) && (findices[cellCIndices[d]-startC] != cellFIndices[cmap[d]])) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse dof %d maps to both %d and %d", cindices[cellCIndices[d]-startC], findices[cellCIndices[d]-startC], cellFIndices[cmap[d]]);
      cindices[cellCIndices[d]-startC] = cellCIndices[d];
      findices[cellCIndices[d]-startC] = cellFIndices[cmap[d]];
    }
  }
  ierr = PetscFree(cmap);CHKERRQ(ierr);
  ierr = PetscFree2(cellCIndices,cellFIndices);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, m, cindices, PETSC_OWN_POINTER, &cis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, m, findices, PETSC_OWN_POINTER, &fis);CHKERRQ(ierr);
  ierr = VecScatterCreate(cv, cis, fv, fis, sc);CHKERRQ(ierr);
  ierr = ISDestroy(&cis);CHKERRQ(ierr);
  ierr = ISDestroy(&fis);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf, &fv);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmc, &cv);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InjectorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
