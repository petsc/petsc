#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#include <petsc-private/petscfeimpl.h>
#include <petscfv.h>

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
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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
      ierr = DMPlexComputeCellGeometry(dm, point, v0, J, NULL, &detJ);CHKERRQ(ierr);
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
  PetscSection    section;
  PetscScalar    *values;
  PetscReal      *v0, *J, detJ;
  PetscInt        numFields, numComp, dim, spDim, totDim = 0, numValues, cStart, cEnd, c, f, d, v, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFields, &sp);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscFE fe;

    ierr = DMGetField(dm, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(fe, &sp[f]);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &numComp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    totDim += spDim*numComp;
  }
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,&v0,dim*dim,&J);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscCellGeometry geom;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, NULL, &detJ);CHKERRQ(ierr);
    geom.v0   = v0;
    geom.J    = J;
    geom.detJ = &detJ;
    for (f = 0, v = 0; f < numFields; ++f) {
      PetscFE      fe;
      void * const ctx = ctxs ? ctxs[f] : NULL;

      ierr = DMGetField(dm, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &numComp);CHKERRQ(ierr);
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
    ierr = DMPlexVecSetClosure(dm, section, localX, c, values, mode);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscFree2(v0,J);CHKERRQ(ierr);
  ierr = PetscFree(sp);CHKERRQ(ierr);
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
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
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

.seealso: DMPlexProjectFunction(), DMPlexComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeL2Diff(DM dm, void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  const PetscInt  debug = 0;
  PetscSection    section;
  PetscQuadrature quad;
  Vec             localX;
  PetscScalar    *funcVal;
  PetscReal      *coords, *v0, *J, *invJ, detJ;
  PetscReal       localDiff = 0.0;
  PetscInt        dim, numFields, numComponents = 0, cStart, cEnd, c, field, fieldOffset, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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
  ierr = DMPlexProjectFunctionLocal(dm, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscMalloc5(numComponents,&funcVal,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscFE          fe;
      void * const     ctx = ctxs ? ctxs[field] : NULL;
      const PetscReal *quadPoints, *quadWeights;
      PetscReal       *basis;
      PetscInt         numQuadPoints, numBasisFuncs, numBasisComps, q, d, e, fc, f;

      ierr = DMGetField(dm, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, &quadWeights);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &numBasisFuncs);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &numBasisComps);CHKERRQ(ierr);
      ierr = PetscFEGetDefaultTabulation(fe, &basis, NULL, NULL);CHKERRQ(ierr);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, numBasisFuncs*numBasisComps, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < numQuadPoints; ++q) {
        for (d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for (e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        (*funcs[field])(coords, funcVal, ctx);
        for (fc = 0; fc < numBasisComps; ++fc) {
          PetscScalar interpolant = 0.0;

          for (f = 0; f < numBasisFuncs; ++f) {
            const PetscInt fidx = f*numBasisComps+fc;
            interpolant += x[fieldOffset+fidx]*basis[q*numBasisFuncs*numBasisComps+fidx];
          }
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q]*detJ;
        }
      }
      comp        += numBasisComps;
      fieldOffset += numBasisFuncs*numBasisComps;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree5(funcVal,coords,v0,J,invJ);CHKERRQ(ierr);
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
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
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
PetscErrorCode DMPlexComputeL2FieldDiff(DM dm, void (**funcs)(const PetscReal [], PetscScalar *, void *), void **ctxs, Vec X, PetscReal diff[])
{
  const PetscInt  debug = 0;
  PetscSection    section;
  PetscQuadrature quad;
  Vec             localX;
  PetscScalar    *funcVal;
  PetscReal      *coords, *v0, *J, *invJ, detJ;
  PetscReal      *localDiff;
  PetscInt        dim, numFields, numComponents = 0, cStart, cEnd, c, field, fieldOffset, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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
  ierr = DMPlexProjectFunctionLocal(dm, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscCalloc6(numFields,&localDiff,numComponents,&funcVal,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscFE          fe;
      void * const     ctx = ctxs ? ctxs[field] : NULL;
      const PetscReal *quadPoints, *quadWeights;
      PetscReal       *basis, elemDiff = 0.0;
      PetscInt         numQuadPoints, numBasisFuncs, numBasisComps, q, d, e, fc, f;

      ierr = DMGetField(dm, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, &quadPoints, &quadWeights);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &numBasisFuncs);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &numBasisComps);CHKERRQ(ierr);
      ierr = PetscFEGetDefaultTabulation(fe, &basis, NULL, NULL);CHKERRQ(ierr);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, numBasisFuncs*numBasisComps, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < numQuadPoints; ++q) {
        for (d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for (e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        (*funcs[field])(coords, funcVal, ctx);
        for (fc = 0; fc < numBasisComps; ++fc) {
          PetscScalar interpolant = 0.0;

          for (f = 0; f < numBasisFuncs; ++f) {
            const PetscInt fidx = f*numBasisComps+fc;
            interpolant += x[fieldOffset+fidx]*basis[q*numBasisFuncs*numBasisComps+fidx];
          }
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q]*detJ;
        }
      }
      comp        += numBasisComps;
      fieldOffset += numBasisFuncs*numBasisComps;
      localDiff[field] += elemDiff;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
  }
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPI_Allreduce(localDiff, diff, numFields, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) diff[field] = PetscSqrtReal(diff[field]);
  ierr  = PetscFree6(localDiff,funcVal,coords,v0,J,invJ);CHKERRQ(ierr);
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
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexComputeResidualFEM_Internal"
PetscErrorCode DMPlexComputeResidualFEM_Internal(DM dm, Vec X, Vec X_t, Vec F, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  const char       *name  = "Residual";
  DM                dmAux;
  DMLabel           depth;
  Vec               A;
  PetscDS           prob, probAux = NULL;
  PetscQuadrature   q;
  PetscCellGeometry geom;
  PetscSection      section, sectionAux;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *elemVec, *u, *u_t, *a = NULL;
  PetscInt          dim, Nf, f, numCells, cStart, cEnd, c, numBd, bd;
  PetscInt          totDim, totDimBd, totDimAux;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTotalBdDimension(prob, &totDimBd);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMPlexInsertBoundaryValuesFEM(dm, X);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc7(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*dim,&v0,numCells*dim*dim,&J,numCells*dim*dim,&invJ,numCells,&detJ,numCells*totDim,&elemVec);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL, *x_t = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
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
    ierr = PetscFEIntegrateResidual(fe, prob, f, Ne, geom, u, u_t, probAux, a, elemVec);CHKERRQ(ierr);
    geom.v0   = &v0[offset*dim];
    geom.J    = &J[offset*dim*dim];
    geom.invJ = &invJ[offset*dim*dim];
    geom.detJ = &detJ[offset];
    ierr = PetscFEIntegrateResidual(fe, prob, f, Nr, geom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], &elemVec[offset*totDim]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(c, name, totDim, &elemVec[c*totDim]);CHKERRQ(ierr);}
    ierr = DMPlexVecSetClosure(dm, section, F, c, &elemVec[c*totDim], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree7(u,u_t,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetNumBoundary(dm, &numBd);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    const char     *bdLabel;
    DMLabel         label;
    IS              pointIS;
    const PetscInt *points;
    const PetscInt *values;
    PetscReal      *n;
    PetscInt        field, numValues, numPoints, p, dep, numFaces;
    PetscBool       isEssential;

    ierr = DMPlexGetBoundary(dm, bd, &isEssential, NULL, &bdLabel, &field, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    if (isEssential) continue;
    if (numValues != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Bug me and I will fix this");
    ierr = DMPlexGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(label, 1, &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, 1, &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0, numFaces = 0; p < numPoints; ++p) {
      ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
      if (dep == dim-1) ++numFaces;
    }
    ierr = PetscMalloc7(numFaces*totDimBd,&u,numFaces*dim,&v0,numFaces*dim,&n,numFaces*dim*dim,&J,numFaces*dim*dim,&invJ,numFaces,&detJ,numFaces*totDimBd,&elemVec);CHKERRQ(ierr);
    if (X_t) {ierr = PetscMalloc1(numFaces*totDimBd,&u_t);CHKERRQ(ierr);}
    for (p = 0, f = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscScalar   *x     = NULL;
      PetscInt       i;

      ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
      if (dep != dim-1) continue;
      ierr = DMPlexComputeCellGeometry(dm, point, &v0[f*dim], &J[f*dim*dim], &invJ[f*dim*dim], &detJ[f]);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFVM(dm, point, NULL, NULL, &n[f*dim]);
      if (detJ[f] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", detJ[f], point);
      ierr = DMPlexVecGetClosure(dm, section, X, point, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimBd; ++i) u[f*totDimBd+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X, point, NULL, &x);CHKERRQ(ierr);
      if (X_t) {
        ierr = DMPlexVecGetClosure(dm, section, X_t, point, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimBd; ++i) u_t[f*totDimBd+i] = x[i];
        ierr = DMPlexVecRestoreClosure(dm, section, X_t, point, NULL, &x);CHKERRQ(ierr);
      }
      ++f;
    }
    for (f = 0; f < Nf; ++f) {
      PetscFE  fe;
      PetscInt numQuadPoints, Nb;
      /* Conforming batches */
      PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt Nr, offset;

      ierr = PetscDSGetBdDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(q, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
      blockSize = Nb*numQuadPoints;
      batchSize = numBlocks * blockSize;
      ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      geom.v0   = v0;
      geom.n    = n;
      geom.J    = J;
      geom.invJ = invJ;
      geom.detJ = detJ;
      ierr = PetscFEIntegrateBdResidual(fe, prob, f, Ne, geom, u, u_t, NULL, NULL, elemVec);CHKERRQ(ierr);
      geom.v0   = &v0[offset*dim];
      geom.n    = &n[offset*dim];
      geom.J    = &J[offset*dim*dim];
      geom.invJ = &invJ[offset*dim*dim];
      geom.detJ = &detJ[offset];
      ierr = PetscFEIntegrateBdResidual(fe, prob, f, Nr, geom, &u[offset*totDimBd], u_t ? &u_t[offset*totDimBd] : NULL, NULL, NULL, &elemVec[offset*totDimBd]);CHKERRQ(ierr);
    }
    for (p = 0, f = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];

      ierr = DMLabelGetValue(depth, point, &dep);CHKERRQ(ierr);
      if (dep != dim-1) continue;
      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(point, "BdResidual", totDimBd, &elemVec[f*totDimBd]);CHKERRQ(ierr);}
      ierr = DMPlexVecSetClosure(dm, NULL, F, point, &elemVec[f*totDimBd], ADD_VALUES);CHKERRQ(ierr);
      ++f;
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree7(u,v0,n,J,invJ,detJ,elemVec);CHKERRQ(ierr);
    if (X_t) {ierr = PetscFree(u_t);CHKERRQ(ierr);}
  }
  if (mesh->printFEM) {ierr = DMPrintLocalVec(dm, name, mesh->printTol, F);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeResidualFEM_Check"
static PetscErrorCode DMPlexComputeResidualFEM_Check(DM dm, Vec X, Vec X_t, Vec F, void *user)
{
  DM                dmCh, dmAux;
  Vec               A;
  PetscDS           prob, probCh, probAux = NULL;
  PetscQuadrature   q;
  PetscCellGeometry geom;
  PetscSection      section, sectionAux;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *elemVec, *elemVecCh, *u, *u_t, *a = NULL;
  PetscInt          dim, Nf, f, numCells, cStart, cEnd, c;
  PetscInt          totDim, totDimAux, diffCell = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmCh", (PetscObject *) &dmCh);CHKERRQ(ierr);
  ierr = DMGetDS(dmCh, &probCh);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMPlexInsertBoundaryValuesFEM(dm, X);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc7(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*dim,&v0,numCells*dim*dim,&J,numCells*dim*dim,&invJ,numCells,&detJ,numCells*totDim,&elemVec);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCells*totDim,&elemVecCh);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL, *x_t = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  for (f = 0; f < Nf; ++f) {
    PetscFE  fe, feCh;
    PetscInt numQuadPoints, Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(probCh, f, (PetscObject *) &feCh);CHKERRQ(ierr);
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
    ierr = PetscFEIntegrateResidual(fe, prob, f, Ne, geom, u, u_t, probAux, a, elemVec);CHKERRQ(ierr);
    ierr = PetscFEIntegrateResidual(feCh, prob, f, Ne, geom, u, u_t, probAux, a, elemVecCh);CHKERRQ(ierr);
    geom.v0   = &v0[offset*dim];
    geom.J    = &J[offset*dim*dim];
    geom.invJ = &invJ[offset*dim*dim];
    geom.detJ = &detJ[offset];
    ierr = PetscFEIntegrateResidual(fe, prob, f, Nr, geom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], &elemVec[offset*totDim]);CHKERRQ(ierr);
    ierr = PetscFEIntegrateResidual(feCh, prob, f, Nr, geom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], &elemVecCh[offset*totDim]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscBool diff = PETSC_FALSE;
    PetscInt  d;

    for (d = 0; d < totDim; ++d) if (PetscAbsScalar(elemVec[c*totDim+d] - elemVecCh[c*totDim+d]) > 1.0e-7) {diff = PETSC_TRUE;break;}
    if (diff) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Different cell %d\n", c);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Residual", totDim, &elemVec[c*totDim]);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Check Residual", totDim, &elemVecCh[c*totDim]);CHKERRQ(ierr);
      ++diffCell;
    }
    if (diffCell > 9) break;
    ierr = DMPlexVecSetClosure(dm, section, F, c, &elemVec[c*totDim], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree7(u,u_t,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  ierr = PetscFree(elemVecCh);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSNESComputeResidualFEM"
/*@
  DMPlexSNESComputeResidualFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Local solution
- user - The user context

  Output Parameter:
. F  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexSNESComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  PetscObject    check;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) dm, "dmCh", &check);CHKERRQ(ierr);
  if (check) {ierr = DMPlexComputeResidualFEM_Check(dm, X, NULL, F, user);CHKERRQ(ierr);}
  else       {ierr = DMPlexComputeResidualFEM_Internal(dm, X, NULL, F, user);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSComputeIFunctionFEM"
/*@
  DMPlexTSComputeIFunctionFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. X  - Local solution
. X_t - Local solution time derivative, or NULL
- user - The user context

  Output Parameter:
. F  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeIFunctionFEM(DM dm, PetscReal time, Vec X, Vec X_t, Vec F, void *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeResidualFEM_Internal(dm, X, X_t, F, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeJacobianFEM_Internal"
PetscErrorCode DMPlexComputeJacobianFEM_Internal(DM dm, Vec X, Vec X_t, Mat Jac, Mat JacP,void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  const char       *name  = "Jacobian";
  DM                dmAux;
  DMLabel           depth;
  Vec               A;
  PetscDS           prob, probAux = NULL;
  PetscQuadrature   quad;
  PetscCellGeometry geom;
  PetscSection      section, globalSection, sectionAux;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *elemMat, *u, *u_t, *a = NULL;
  PetscInt          dim, Nf, f, fieldI, fieldJ, numCells, cStart, cEnd, c;
  PetscInt          totDim, totDimBd, totDimAux, numBd, bd;
  PetscBool         isShell;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTotalBdDimension(prob, &totDimBd);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMPlexInsertBoundaryValuesFEM(dm, X);CHKERRQ(ierr);
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = PetscMalloc7(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*dim,&v0,numCells*dim*dim,&J,numCells*dim*dim,&invJ,numCells,&detJ,numCells*totDim*totDim,&elemMat);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL,  *x_t = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  ierr = PetscMemzero(elemMat, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscFE  fe;
    PetscInt numQuadPoints, Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      geom.v0   = v0;
      geom.J    = J;
      geom.invJ = invJ;
      geom.detJ = detJ;
      ierr = PetscFEIntegrateJacobian(fe, prob, fieldI, fieldJ, Ne, geom, u, u_t, probAux, a, elemMat);CHKERRQ(ierr);
      geom.v0   = &v0[offset*dim];
      geom.J    = &J[offset*dim*dim];
      geom.invJ = &invJ[offset*dim*dim];
      geom.detJ = &detJ[offset];
      ierr = PetscFEIntegrateJacobian(fe, prob, fieldI, fieldJ, Nr, geom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[c*totDim*totDim]);CHKERRQ(ierr);}
    ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, c, &elemMat[c*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree7(u,u_t,v0,J,invJ,detJ,elemMat);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetNumBoundary(dm, &numBd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetNumBoundary(dm, &numBd);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    const char     *bdLabel;
    DMLabel         label;
    IS              pointIS;
    const PetscInt *points;
    const PetscInt *values;
    PetscReal      *n;
    PetscInt        field, numValues, numPoints, p, dep, numFaces;
    PetscBool       isEssential;

    ierr = DMPlexGetBoundary(dm, bd, &isEssential, NULL, &bdLabel, &field, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    if (isEssential) continue;
    if (numValues != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Bug me and I will fix this");
    ierr = DMPlexGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(label, 1, &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, 1, &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0, numFaces = 0; p < numPoints; ++p) {
      ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
      if (dep == dim-1) ++numFaces;
    }
    ierr = PetscMalloc7(numFaces*totDimBd,&u,numFaces*dim,&v0,numFaces*dim,&n,numFaces*dim*dim,&J,numFaces*dim*dim,&invJ,numFaces,&detJ,numFaces*totDimBd*totDimBd,&elemMat);CHKERRQ(ierr);
    if (X_t) {ierr = PetscMalloc1(numFaces*totDimBd,&u_t);CHKERRQ(ierr);}
    for (p = 0, f = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscScalar   *x     = NULL;
      PetscInt       i;

      ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
      if (dep != dim-1) continue;
      ierr = DMPlexComputeCellGeometry(dm, point, &v0[f*dim], &J[f*dim*dim], &invJ[f*dim*dim], &detJ[f]);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFVM(dm, point, NULL, NULL, &n[f*dim]);
      if (detJ[f] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", detJ[f], point);
      ierr = DMPlexVecGetClosure(dm, section, X, point, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimBd; ++i) u[f*totDimBd+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X, point, NULL, &x);CHKERRQ(ierr);
      if (X_t) {
        ierr = DMPlexVecGetClosure(dm, section, X_t, point, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimBd; ++i) u_t[f*totDimBd+i] = x[i];
        ierr = DMPlexVecRestoreClosure(dm, section, X_t, point, NULL, &x);CHKERRQ(ierr);
      }
      ++f;
    }
    ierr = PetscMemzero(elemMat, numFaces*totDimBd*totDimBd * sizeof(PetscScalar));CHKERRQ(ierr);
    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE  fe;
      PetscInt numQuadPoints, Nb;
      /* Conforming batches */
      PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt Nr, offset;

      ierr = PetscDSGetBdDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
      blockSize = Nb*numQuadPoints;
      batchSize = numBlocks * blockSize;
      ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        geom.v0   = v0;
        geom.n    = n;
        geom.J    = J;
        geom.invJ = invJ;
        geom.detJ = detJ;
        ierr = PetscFEIntegrateBdJacobian(fe, prob, fieldI, fieldJ, Ne, geom, u, u_t, NULL, NULL, elemMat);CHKERRQ(ierr);
        geom.v0   = &v0[offset*dim];
        geom.n    = &n[offset*dim];
        geom.J    = &J[offset*dim*dim];
        geom.invJ = &invJ[offset*dim*dim];
        geom.detJ = &detJ[offset];
        ierr = PetscFEIntegrateBdJacobian(fe, prob, fieldI, fieldJ, Nr, geom, &u[offset*totDimBd], u_t ? &u_t[offset*totDimBd] : NULL, NULL, NULL, &elemMat[offset*totDimBd*totDimBd]);CHKERRQ(ierr);
      }
    }
    for (p = 0, f = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];

      ierr = DMLabelGetValue(depth, point, &dep);CHKERRQ(ierr);
      if (dep != dim-1) continue;
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(point, "BdJacobian", totDimBd, totDimBd, &elemMat[f*totDimBd*totDimBd]);CHKERRQ(ierr);}
      ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, point, &elemMat[f*totDimBd*totDimBd], ADD_VALUES);CHKERRQ(ierr);
      ++f;
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree7(u,v0,n,J,invJ,detJ,elemMat);CHKERRQ(ierr);
    if (X_t) {ierr = PetscFree(u_t);CHKERRQ(ierr);}
  }
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s:\n", name);CHKERRQ(ierr);
    ierr = MatChop(JacP, 1.0e-10);CHKERRQ(ierr);
    ierr = MatView(JacP, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) Jac, MATSHELL, &isShell);CHKERRQ(ierr);
  if (isShell) {
    JacActionCtx *jctx;

    ierr = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
    ierr = VecCopy(X, jctx->u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSNESComputeJacobianFEM"
/*@
  DMPlexSNESComputeJacobianFEM - Form the local portion of the Jacobian matrix J at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  The first member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: FormFunctionLocal()
@*/
PetscErrorCode DMPlexSNESComputeJacobianFEM(DM dm, Vec X, Mat Jac, Mat JacP,void *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexComputeJacobianFEM_Internal(dm, X, NULL, Jac, JacP, user);CHKERRQ(ierr);
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
  ierr = DMPlexGetDimension(dmf, &dim);CHKERRQ(ierr);
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
  ierr = DMPlexGetDimension(dmf, &dim);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "BoundaryDuplicate"
static PetscErrorCode BoundaryDuplicate(DMBoundary bd, DMBoundary *boundary)
{
  DMBoundary     b = bd, b2, bold = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *boundary = NULL;
  for (; b; b = b->next, bold = b2) {
    ierr = PetscNew(&b2);CHKERRQ(ierr);
    ierr = PetscStrallocpy(b->name, (char **) &b2->name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(b->labelname, (char **) &b2->labelname);CHKERRQ(ierr);
    ierr = PetscMalloc1(b->numids, &b2->ids);CHKERRQ(ierr);
    ierr = PetscMemcpy(b2->ids, b->ids, b->numids*sizeof(PetscInt));CHKERRQ(ierr);
    b2->label     = NULL;
    b2->essential = b->essential;
    b2->field     = b->field;
    b2->func      = b->func;
    b2->numids    = b->numids;
    b2->ctx       = b->ctx;
    b2->next      = NULL;
    if (!*boundary) *boundary   = b2;
    if (bold)        bold->next = b2;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCopyBoundary"
PetscErrorCode DMPlexCopyBoundary(DM dm, DM dmNew)
{
  DM_Plex       *mesh    = (DM_Plex *) dm->data;
  DM_Plex       *meshNew = (DM_Plex *) dmNew->data;
  DMBoundary     b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BoundaryDuplicate(mesh->boundary, &meshNew->boundary);CHKERRQ(ierr);
  for (b = meshNew->boundary; b; b = b->next) {
    if (b->labelname) {
      ierr = DMPlexGetLabel(dmNew, b->labelname, &b->label);CHKERRQ(ierr);
      if (!b->label) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Label %s does not exist in this DM", b->labelname);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexAddBoundary"
/* The ids can be overridden by the command line option -bc_<boundary name> */
PetscErrorCode DMPlexAddBoundary(DM dm, PetscBool isEssential, const char name[], const char labelname[], PetscInt field, void (*bcFunc)(), PetscInt numids, const PetscInt *ids, void *ctx)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DMBoundary     b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNew(&b);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char **) &b->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(labelname, (char **) &b->labelname);CHKERRQ(ierr);
  ierr = PetscMalloc1(numids, &b->ids);CHKERRQ(ierr);
  ierr = PetscMemcpy(b->ids, ids, numids*sizeof(PetscInt));CHKERRQ(ierr);
  if (b->labelname) {
    ierr = DMPlexGetLabel(dm, b->labelname, &b->label);CHKERRQ(ierr);
    if (!b->label) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Label %s does not exist in this DM", b->labelname);
  }
  b->essential   = isEssential;
  b->field       = field;
  b->func        = bcFunc;
  b->numids      = numids;
  b->ctx         = ctx;
  b->next        = mesh->boundary;
  mesh->boundary = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetNumBoundary"
PetscErrorCode DMPlexGetNumBoundary(DM dm, PetscInt *numBd)
{
  DM_Plex   *mesh = (DM_Plex *) dm->data;
  DMBoundary b    = mesh->boundary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(numBd, 2);
  *numBd = 0;
  while (b) {++(*numBd); b = b->next;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetBoundary"
PetscErrorCode DMPlexGetBoundary(DM dm, PetscInt bd, PetscBool *isEssential, const char **name, const char **labelname, PetscInt *field, void (**func)(), PetscInt *numids, const PetscInt **ids, void **ctx)
{
  DM_Plex   *mesh = (DM_Plex *) dm->data;
  DMBoundary b    = mesh->boundary;
  PetscInt   n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  while (b) {
    if (n == bd) break;
    b = b->next;
    ++n;
  }
  if (n != bd) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Boundary %d is not in [0, %d)", bd, n);
  if (isEssential) {
    PetscValidPointer(isEssential, 3);
    *isEssential = b->essential;
  }
  if (name) {
    PetscValidPointer(name, 4);
    *name = b->name;
  }
  if (labelname) {
    PetscValidPointer(labelname, 5);
    *labelname = b->labelname;
  }
  if (field) {
    PetscValidPointer(field, 6);
    *field = b->field;
  }
  if (func) {
    PetscValidPointer(func, 7);
    *func = b->func;
  }
  if (numids) {
    PetscValidPointer(numids, 8);
    *numids = b->numids;
  }
  if (ids) {
    PetscValidPointer(ids, 9);
    *ids = b->ids;
  }
  if (ctx) {
    PetscValidPointer(ctx, 10);
    *ctx = b->ctx;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexIsBoundaryPoint"
PetscErrorCode DMPlexIsBoundaryPoint(DM dm, PetscInt point, PetscBool *isBd)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DMBoundary     b    = mesh->boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(isBd, 3);
  *isBd = PETSC_FALSE;
  while (b && !(*isBd)) {
    if (b->label) {
      PetscInt i;

      for (i = 0; i < b->numids && !(*isBd); ++i) {
        ierr = DMLabelStratumHasPoint(b->label, b->ids[i], point, isBd);CHKERRQ(ierr);
      }
    }
    b = b->next;
  }
  PetscFunctionReturn(0);
}
