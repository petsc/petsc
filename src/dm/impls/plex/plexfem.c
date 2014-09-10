#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#include <petsc-private/petscfeimpl.h>
#include <petsc-private/petscfvimpl.h>

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
#define __FUNCT__ "DMPlexCreateRigidBody"
/*@C
  DMPlexCreateRigidBody - create rigid body modes from coordinates

  Collective on DM

  Input Arguments:
+ dm - the DM
. section - the local section associated with the rigid field, or NULL for the default section
- globalSection - the global section associated with the rigid field, or NULL for the default section

  Output Argument:
. sp - the null space

  Note: This is necessary to take account of Dirichlet conditions on the displacements

  Level: advanced

.seealso: MatNullSpaceCreate()
@*/
PetscErrorCode DMPlexCreateRigidBody(DM dm, PetscSection section, PetscSection globalSection, MatNullSpace *sp)
{
  MPI_Comm       comm;
  Vec            coordinates, localMode, mode[6];
  PetscSection   coordSection;
  PetscScalar   *coords;
  PetscInt       dim, vStart, vEnd, v, n, m, d, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (!section)       {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  if (!globalSection) {ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);}
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &n);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  m    = (dim*(dim+1))/2;
  ierr = VecCreate(comm, &mode[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(mode[0], n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(mode[0]);CHKERRQ(ierr);
  for (i = 1; i < m; ++i) {ierr = VecDuplicate(mode[0], &mode[i]);CHKERRQ(ierr);}
  /* Assume P1 */
  ierr = DMGetLocalVector(dm, &localMode);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    PetscScalar values[3] = {0.0, 0.0, 0.0};

    values[d] = 1.0;
    ierr      = VecSet(localMode, 0.0);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      ierr = DMPlexVecSetClosure(dm, section, localMode, v, values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = DMLocalToGlobalBegin(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
  }
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (d = dim; d < dim*(dim+1)/2; ++d) {
    PetscInt i, j, k = dim > 2 ? d - dim : d;

    ierr = VecSet(localMode, 0.0);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscScalar values[3] = {0.0, 0.0, 0.0};
      PetscInt    off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
          values[j] += epsilon(i, j, k)*PetscRealPart(coords[off+i]);
        }
      }
      ierr = DMPlexVecSetClosure(dm, section, localMode, v, values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = DMLocalToGlobalBegin(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, localMode, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localMode);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexProjectFunctionLabelLocal"
PetscErrorCode DMPlexProjectFunctionLabelLocal(DM dm, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscFE fe[], void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscDualSpace *sp;
  PetscSection    section;
  PetscScalar    *values;
  PetscReal      *v0, *J, detJ;
  PetscBool      *fieldActive;
  PetscInt        numFields, numComp, dim, spDim, totDim = 0, numValues, cStart, cEnd, f, d, v, i, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc3(numFields,&sp,dim,&v0,dim*dim,&J);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    ierr = PetscFEGetDualSpace(fe[f], &sp[f]);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe[f], &numComp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    totDim += spDim*numComp;
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numFields, PETSC_BOOL, &fieldActive);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) fieldActive[f] = funcs[f] ? PETSC_TRUE : PETSC_FALSE;
  for (i = 0; i < numIds; ++i) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        n, p;

    ierr = DMLabelGetStratumIS(label, ids[i], &pointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(pointIS, &n);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < n; ++p) {
      const PetscInt    point = points[p];
      PetscCellGeometry geom;

      if ((point < cStart) || (point >= cEnd)) continue;
      ierr = DMPlexComputeCellGeometryFEM(dm, point, NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
      geom.v0   = v0;
      geom.J    = J;
      geom.detJ = &detJ;
      for (f = 0, v = 0; f < numFields; ++f) {
        void * const ctx = ctxs ? ctxs[f] : NULL;
        ierr = PetscFEGetNumComponents(fe[f], &numComp);CHKERRQ(ierr);
        ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
        for (d = 0; d < spDim; ++d) {
          if (funcs[f]) {
            ierr = PetscDualSpaceApply(sp[f], d, geom, numComp, funcs[f], ctx, &values[v]);CHKERRQ(ierr);
          } else {
            for (comp = 0; comp < numComp; ++comp) values[v+comp] = 0.0;
          }
          v += numComp;
        }
      }
      ierr = DMPlexVecSetFieldClosure_Internal(dm, section, localX, fieldActive, point, values, mode);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numFields, PETSC_BOOL, &fieldActive);CHKERRQ(ierr);
  ierr = PetscFree3(sp,v0,J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectFunctionLocal"
PetscErrorCode DMPlexProjectFunctionLocal(DM dm, void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscDualSpace *sp;
  PetscInt       *numComp;
  PetscSection    section;
  PetscScalar    *values;
  PetscReal      *v0, *J, detJ;
  PetscInt        numFields, dim, spDim, totDim = 0, numValues, cStart, cEnd, c, f, d, v, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc2(numFields, &sp, numFields, &numComp);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetNumComponents(fe, &numComp[f]);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(fe, &sp[f]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject) sp[f]);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV         fv = (PetscFV) obj;
      PetscQuadrature q;

      ierr = PetscFVGetNumComponents(fv, &numComp[f]);CHKERRQ(ierr);
      ierr = PetscFVGetQuadrature(fv, &q);CHKERRQ(ierr);
      ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) fv), &sp[f]);CHKERRQ(ierr);
      ierr = PetscDualSpaceSetDM(sp[f], dm);CHKERRQ(ierr);
      ierr = PetscDualSpaceSetType(sp[f], PETSCDUALSPACESIMPLE);CHKERRQ(ierr);
      ierr = PetscDualSpaceSimpleSetDimension(sp[f], 1);CHKERRQ(ierr);
      ierr = PetscDualSpaceSimpleSetFunctional(sp[f], 0, q);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    totDim += spDim*numComp[f];
  }
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,&v0,dim*dim,&J);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscCellGeometry geom;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
    geom.v0   = v0;
    geom.J    = J;
    geom.detJ = &detJ;
    for (f = 0, v = 0; f < numFields; ++f) {
      void *const ctx = ctxs ? ctxs[f] : NULL;

      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        if (funcs[f]) {
          ierr = PetscDualSpaceApply(sp[f], d, geom, numComp[f], funcs[f], ctx, &values[v]);CHKERRQ(ierr);
        } else {
          for (comp = 0; comp < numComp[f]; ++comp) values[v+comp] = 0.0;
        }
        v += numComp[f];
      }
    }
    ierr = DMPlexVecSetClosure(dm, section, localX, c, values, mode);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscFree2(v0,J);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {ierr = PetscDualSpaceDestroy(&sp[f]);CHKERRQ(ierr);}
  ierr = PetscFree2(sp, numComp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectFunction"
/*@C
  DMPlexProjectFunction - This projects the given function into the function space provided.

  Input Parameters:
+ dm      - The DM
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

  Level: developer

.seealso: DMPlexComputeL2Diff()
@*/
PetscErrorCode DMPlexProjectFunction(DM dm, void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, InsertMode mode, Vec X)
{
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMPlexProjectFunctionLocal(dm, funcs, ctxs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectFieldLocal"
PetscErrorCode DMPlexProjectFieldLocal(DM dm, Vec localU, void (**funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal [], PetscScalar []), InsertMode mode, Vec localX)
{
  DM                dmAux;
  PetscDS           prob, probAux;
  Vec               A;
  PetscSection      section, sectionAux;
  PetscScalar      *values, *u, *u_x, *a, *a_x;
  PetscReal        *x, *v0, *J, *invJ, detJ, **basisField, **basisFieldDer, **basisFieldAux, **basisFieldDerAux;
  PetscInt          Nf, dim, spDim, totDim, numValues, cStart, cEnd, c, f, d, v, comp;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &basisField, &basisFieldDer);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, NULL);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(prob, &basisFieldAux, &basisFieldDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
  }
  ierr = DMPlexInsertBoundaryValuesFEM(dm, localU);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *coefficients = NULL, *coefficientsAux = NULL;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, section, localU, c, NULL, &coefficients);CHKERRQ(ierr);
    if (dmAux) {ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &coefficientsAux);CHKERRQ(ierr);}
    for (f = 0, v = 0; f < Nf; ++f) {
      PetscFE        fe;
      PetscDualSpace sp;
      PetscInt       Ncf;

      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(fe, &sp);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Ncf);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(sp, &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        PetscQuadrature  quad;
        const PetscReal *points, *weights;
        PetscInt         numPoints, q;

        if (funcs[f]) {
          ierr = PetscDualSpaceGetFunctional(sp, d, &quad);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(quad, NULL, &numPoints, &points, &weights);CHKERRQ(ierr);
          ierr = PetscFEGetTabulation(fe, numPoints, points, &basisField[f], &basisFieldDer[f], NULL);CHKERRQ(ierr);
          for (q = 0; q < numPoints; ++q) {
            CoordinatesRefToReal(dim, dim, v0, J, &points[q*dim], x);
            ierr = EvaluateFieldJets(prob,    PETSC_FALSE, q, invJ, coefficients,    NULL, u, u_x, NULL);CHKERRQ(ierr);
            ierr = EvaluateFieldJets(probAux, PETSC_FALSE, q, invJ, coefficientsAux, NULL, a, a_x, NULL);CHKERRQ(ierr);
            (*funcs[f])(u, NULL, u_x, a, NULL, a_x, x, &values[v]);
          }
          ierr = PetscFERestoreTabulation(fe, numPoints, points, &basisField[f], &basisFieldDer[f], NULL);CHKERRQ(ierr);
        } else {
          for (comp = 0; comp < Ncf; ++comp) values[v+comp] = 0.0;
        }
        v += Ncf;
      }
    }
    ierr = DMPlexVecRestoreClosure(dm, section, localU, c, NULL, &coefficients);CHKERRQ(ierr);
    if (dmAux) {ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &coefficientsAux);CHKERRQ(ierr);}
    ierr = DMPlexVecSetClosure(dm, section, localX, c, values, mode);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectField"
/*@C
  DMPlexProjectField - This projects the given function of the fields into the function space provided.

  Input Parameters:
+ dm      - The DM
. U       - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. X       - The output vector

  Level: developer

.seealso: DMPlexProjectFunction(), DMPlexComputeL2Diff()
@*/
PetscErrorCode DMPlexProjectField(DM dm, Vec U, void (**funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal [], PetscScalar []), InsertMode mode, Vec X)
{
  Vec            localX, localU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, U, INSERT_VALUES, localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, U, INSERT_VALUES, localU);CHKERRQ(ierr);
  ierr = DMPlexProjectFieldLocal(dm, localU, funcs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertBoundaryValuesFEM"
PetscErrorCode DMPlexInsertBoundaryValuesFEM(DM dm, Vec localX)
{
  void        (**funcs)(const PetscReal x[], PetscScalar *u, void *ctx);
  void         **ctxs;
  PetscFE       *fe;
  PetscInt       numFields, f, numBd, b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(localX, VEC_CLASSID, 2);
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc3(numFields,&fe,numFields,&funcs,numFields,&ctxs);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {ierr = DMGetField(dm, f, (PetscObject *) &fe[f]);CHKERRQ(ierr);}
  /* OPT: Could attempt to do multiple BCs at once */
  ierr = DMPlexGetNumBoundary(dm, &numBd);CHKERRQ(ierr);
  for (b = 0; b < numBd; ++b) {
    DMLabel         label;
    const PetscInt *ids;
    const char     *labelname;
    PetscInt        numids, field;
    PetscBool       isEssential;
    void          (*func)();
    void           *ctx;

    ierr = DMPlexGetBoundary(dm, b, &isEssential, NULL, &labelname, &field, &func, &numids, &ids, &ctx);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, labelname, &label);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      funcs[f] = field == f ? (void (*)(const PetscReal[], PetscScalar *, void *)) func : NULL;
      ctxs[f]  = field == f ? ctx : NULL;
    }
    ierr = DMPlexProjectFunctionLabelLocal(dm, label, numids, ids, fe, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  }
  ierr = PetscFree3(fe,funcs,ctxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeL2Diff"
/*@C
  DMPlexComputeL2Diff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h.

  Input Parameters:
+ dm    - The DM
. funcs - The functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

.seealso: DMPlexProjectFunction(), DMPlexComputeL2FieldDiff(), DMPlexComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeL2Diff(DM dm, void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  const PetscInt   debug = 0;
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscReal        localDiff = 0.0;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, numFields, numComponents = 0, numQuadPoints, cStart, cEnd, c, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
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
  ierr = DMPlexProjectFunctionLocal(dm, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
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
        (*funcs[field])(coords, funcVal, ctx);
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
  ierr  = MPI_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeL2GradientDiff"
/*@C
  DMPlexComputeL2GradientDiff - This function computes the L_2 difference between the gradient of a function u and an FEM interpolant solution grad u_h.

  Input Parameters:
+ dm    - The DM
. funcs - The gradient functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
. X     - The coefficient vector u_h
- n     - The vector to project along

  Output Parameter:
. diff - The diff ||(grad u - grad u_h) . n||_2

  Level: developer

.seealso: DMPlexProjectFunction(), DMPlexComputeL2Diff()
@*/
PetscErrorCode DMPlexComputeL2GradientDiff(DM dm, void (**funcs)(const PetscReal [], const PetscReal [], PetscScalar *, void *), void **ctxs, Vec X, const PetscReal n[], PetscReal *diff)
{
  const PetscInt  debug = 0;
  PetscSection    section;
  PetscQuadrature quad;
  Vec             localX;
  PetscScalar    *funcVal, *interpolantVec;
  PetscReal      *coords, *realSpaceDer, *v0, *J, *invJ, detJ;
  PetscReal       localDiff = 0.0;
  PetscInt        dim, numFields, numComponents = 0, cStart, cEnd, c, field, fieldOffset, comp;
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
  /* ierr = DMPlexProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr); */
  ierr = PetscMalloc7(numComponents,&funcVal,dim,&coords,dim,&realSpaceDer,dim,&v0,dim*dim,&J,dim*dim,&invJ,dim,&interpolantVec);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
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
        (*funcs[field])(coords, n, funcVal, ctx);
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
  ierr  = MPI_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeL2FieldDiff"
/*@C
  DMPlexComputeL2FieldDiff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h, separated into field components.

  Input Parameters:
+ dm    - The DM
. funcs - The functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h

  Output Parameter:
. diff - The array of differences, ||u^f - u^f_h||_2

  Level: developer

.seealso: DMPlexProjectFunction(), DMPlexComputeL2Diff(), DMPlexComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeL2FieldDiff(DM dm, void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, Vec X, PetscReal diff[])
{
  const PetscInt   debug = 0;
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscReal       *localDiff;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, numFields, numComponents = 0, numQuadPoints, cStart, cEnd, c, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
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
  ierr = DMPlexProjectFunctionLocal(dm, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscCalloc7(numFields,&localDiff,numComponents,&funcVal,numComponents,&interpolant,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
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
        (*funcs[field])(coords, funcVal, ctx);
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
  ierr = MPI_Allreduce(localDiff, diff, numFields, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) diff[field] = PetscSqrtReal(diff[field]);
  ierr = PetscFree7(localDiff,funcVal,interpolant,coords,v0,J,invJ);CHKERRQ(ierr);
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
  PetscQuadrature   q;
  PetscCellGeometry geom;
  PetscSection      section, sectionAux;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *u, *a = NULL;
  PetscInt          dim, Nf, f, numCells, cStart, cEnd, c;
  PetscInt          totDim, totDimAux;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /*ierr = PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);*/
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (f = 0; f < Nf; ++f) {integral[f]    = 0.0;}
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMPlexInsertBoundaryValuesFEM(dm, localX);CHKERRQ(ierr);
  ierr = PetscMalloc5(numCells*totDim,&u,numCells*dim,&v0,numCells*dim*dim,&J,numCells*dim*dim,&invJ,numCells,&detJ);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
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
    PetscFE  fe;
    PetscInt numQuadPoints, Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks * blockSize;
    ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    geom.v0   = v0;
    geom.J    = J;
    geom.invJ = invJ;
    geom.detJ = detJ;
    ierr = PetscFEIntegrate(fe, prob, f, Ne, geom, u, probAux, a, integral);CHKERRQ(ierr);
    geom.v0   = &v0[offset*dim];
    geom.J    = &J[offset*dim*dim];
    geom.invJ = &invJ[offset*dim*dim];
    geom.detJ = &detJ[offset];
    ierr = PetscFEIntegrate(fe, prob, f, Nr, geom, &u[offset*totDim], probAux, &a[offset*totDimAux], integral);CHKERRQ(ierr);
  }
  ierr = PetscFree5(u,v0,J,invJ,detJ);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  if (mesh->printFEM) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Local integral:");CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), " %g", integral[f]);CHKERRQ(ierr);}
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "\n");CHKERRQ(ierr);
  }
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  /* TODO: Allreduce for integral */
  /*ierr = PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeInterpolatorFEM"
/*@
  DMPlexComputeInterpolatorFEM - Form the local portion of the interpolation matrix I from the coarse DM to the uniformly refined DM.

  Input Parameters:
+ dmf  - The fine mesh
. dmc  - The coarse mesh
- user - The user context

  Output Parameter:
. In  - The interpolation matrix

  Note:
  The first member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeInterpolatorFEM(DM dmc, DM dmf, Mat In, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dmc->data;
  const char       *name  = "Interpolator";
  PetscDS           prob;
  PetscFE          *feRef;
  PetscSection      fsection, fglobalSection;
  PetscSection      csection, cglobalSection;
  PetscScalar      *elemMat;
  PetscInt          dim, Nf, f, fieldI, fieldJ, offsetI, offsetJ, cStart, cEnd, c;
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
  ierr = DMGetDS(dmf, &prob);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf,&feRef);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscFE  fe;
    PetscInt rNb, Nc;

    ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFERefine(fe, &feRef[f]);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(feRef[f], &rNb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
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
    ierr = PetscFEGetDualSpace(feRef[fieldI], &Qref);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(feRef[fieldI], &Nc);CHKERRQ(ierr);
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
      PetscFE    fe;
      PetscReal *B;
      PetscInt   NcJ, cpdim, j;

      /* Evaluate basis at points */
      ierr = PetscDSGetDiscretization(prob, fieldJ, (PetscObject *) &fe);CHKERRQ(ierr);
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
      offsetJ += cpdim*NcJ;
    }
    offsetI += fpdim*Nc;
    ierr = PetscFree(points);CHKERRQ(ierr);
  }
  if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(0, name, rTotDim, cTotDim, elemMat);CHKERRQ(ierr);}
  /* Preallocate matrix */
  {
    PetscHashJK ht;
    PetscLayout rLayout;
    PetscInt   *dnz, *onz, *cellCIndices, *cellFIndices;
    PetscInt    locRows, rStart, rEnd, cell, r;

    ierr = MatGetLocalSize(In, &locRows, NULL);CHKERRQ(ierr);
    ierr = PetscLayoutCreate(PetscObjectComm((PetscObject) In), &rLayout);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(rLayout, locRows);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
    ierr = PetscCalloc4(locRows,&dnz,locRows,&onz,cTotDim,&cellCIndices,rTotDim,&cellFIndices);CHKERRQ(ierr);
    ierr = PetscHashJKCreate(&ht);CHKERRQ(ierr);
    for (cell = cStart; cell < cEnd; ++cell) {
      ierr = DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, cell, cellCIndices, cellFIndices);CHKERRQ(ierr);
      for (r = 0; r < rTotDim; ++r) {
        PetscHashJKKey  key;
        PetscHashJKIter missing, iter;

        key.j = cellFIndices[r];
        if (key.j < 0) continue;
        for (c = 0; c < cTotDim; ++c) {
          key.k = cellCIndices[c];
          if (key.k < 0) continue;
          ierr = PetscHashJKPut(ht, key, &missing, &iter);CHKERRQ(ierr);
          if (missing) {
            ierr = PetscHashJKSet(ht, iter, 1);CHKERRQ(ierr);
            if ((key.k >= rStart) && (key.k < rEnd)) ++dnz[key.j-rStart];
            else                                     ++onz[key.j-rStart];
          }
        }
      }
    }
    ierr = PetscHashJKDestroy(&ht);CHKERRQ(ierr);
    ierr = MatXAIJSetPreallocation(In, 1, dnz, onz, NULL, NULL);CHKERRQ(ierr);
    ierr = MatSetOption(In, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree4(dnz,onz,cellCIndices,cellFIndices);CHKERRQ(ierr);
  }
  /* Fill matrix */
  ierr = MatZeroEntries(In);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexMatSetClosureRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES);CHKERRQ(ierr);
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&feRef[f]);CHKERRQ(ierr);}
  ierr = PetscFree(feRef);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexComputeInjectorFEM"
PetscErrorCode DMPlexComputeInjectorFEM(DM dmc, DM dmf, VecScatter *sc, void *user)
{
  PetscDS        prob;
  PetscFE       *feRef;
  Vec            fv, cv;
  IS             fis, cis;
  PetscSection   fsection, fglobalSection, csection, cglobalSection;
  PetscInt      *cmap, *cellCIndices, *cellFIndices, *cindices, *findices;
  PetscInt       cTotDim, fTotDim = 0, Nf, f, field, cStart, cEnd, c, dim, d, startC, offsetC, offsetF, m;
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
  ierr = DMGetDS(dmc, &prob);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf,&feRef);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscFE  fe;
    PetscInt fNb, Nc;

    ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFERefine(fe, &feRef[f]);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(feRef[f], &fNb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    fTotDim += fNb*Nc;
  }
  ierr = PetscDSGetTotalDimension(prob, &cTotDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(cTotDim,&cmap);CHKERRQ(ierr);
  for (field = 0, offsetC = 0, offsetF = 0; field < Nf; ++field) {
    PetscFE        feC;
    PetscDualSpace QF, QC;
    PetscInt       NcF, NcC, fpdim, cpdim;

    ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &feC);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(feC, &NcC);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(feRef[field], &NcF);CHKERRQ(ierr);
    if (NcF != NcC) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %d does not match coarse field %d", NcF, NcC);
    ierr = PetscFEGetDualSpace(feRef[field], &QF);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(QF, &fpdim);CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(feC, &QC);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(QC, &cpdim);CHKERRQ(ierr);
    for (c = 0; c < cpdim; ++c) {
      PetscQuadrature  cfunc;
      const PetscReal *cqpoints;
      PetscInt         NpC;

      ierr = PetscDualSpaceGetFunctional(QC, c, &cfunc);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(cfunc, NULL, &NpC, &cqpoints, NULL);CHKERRQ(ierr);
      if (NpC != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not know how to do injection for moments");
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
        break;
      }
    }
    offsetC += cpdim*NcC;
    offsetF += fpdim*NcF;
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&feRef[f]);CHKERRQ(ierr);}
  ierr = PetscFree(feRef);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmf, &fv);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmc, &cv);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(cv, &startC, NULL);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(cglobalSection, &m);CHKERRQ(ierr);
  ierr = PetscMalloc2(cTotDim,&cellCIndices,fTotDim,&cellFIndices);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&cindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&findices);CHKERRQ(ierr);
  for (d = 0; d < m; ++d) cindices[d] = findices[d] = -1;
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, c, cellCIndices, cellFIndices);CHKERRQ(ierr);
    for (d = 0; d < cTotDim; ++d) {
      if (cellCIndices[d] < 0) continue;
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
