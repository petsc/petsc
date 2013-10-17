#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#include <petscfe.h>

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
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
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
#define __FUNCT__ "DMPlexProjectFunctionLocal"
PetscErrorCode DMPlexProjectFunctionLocal(DM dm, PetscFE fe[], void (**funcs)(const PetscReal [], PetscScalar *), InsertMode mode, Vec localX)
{
  PetscDualSpace *sp;
  PetscSection    section;
  PetscScalar    *values;
  PetscReal      *v0, *J, detJ;
  PetscInt        numFields, numComp, dim, spDim, totDim = 0, numValues, cStart, cEnd, c, f, d, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc(numFields * sizeof(PetscDualSpace), &sp);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    ierr = PetscFEGetDualSpace(fe[f], &sp[f]);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe[f], &numComp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
    totDim += spDim*numComp;
  }
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, PETSC_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,PetscReal,&v0,dim*dim,PetscReal,&J);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscCellGeometry geom;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, NULL, &detJ);CHKERRQ(ierr);
    geom.v0   = v0;
    geom.J    = J;
    geom.detJ = &detJ;
    for (f = 0, v = 0; f < numFields; ++f) {
      ierr = PetscFEGetNumComponents(fe[f], &numComp);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(sp[f], &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        ierr = PetscDualSpaceApply(sp[f], d, geom, numComp, funcs[f], &values[v]);CHKERRQ(ierr);
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
. fe      - The PetscFE associated with the field
. funcs   - The coordinate functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

  Level: developer

.seealso: DMPlexComputeL2Diff()
@*/
PetscErrorCode DMPlexProjectFunction(DM dm, PetscFE fe[], void (**funcs)(const PetscReal [], PetscScalar *), InsertMode mode, Vec X)
{
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMPlexProjectFunctionLocal(dm, fe, funcs, mode, localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeL2Diff"
/*@C
  DMPlexComputeL2Diff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h.

  Input Parameters:
+ dm    - The DM
. fe    - The PetscFE object for each field
. funcs - The functions to evaluate for each field component
- X     - The coefficient vector u_h

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

.seealso: DMPlexProjectFunction()
@*/
PetscErrorCode DMPlexComputeL2Diff(DM dm, PetscFE fe[], void (**funcs)(const PetscReal [], PetscScalar *), Vec X, PetscReal *diff)
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
    PetscInt Nc;

    ierr = PetscFEGetNumComponents(fe[field], &Nc);CHKERRQ(ierr);
    numComponents += Nc;
  }
  ierr = DMPlexProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscMalloc5(numComponents,PetscScalar,&funcVal,dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &quad);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      const PetscInt   numQuadPoints = quad.numPoints;
      const PetscReal *quadPoints    = quad.points;
      const PetscReal *quadWeights   = quad.weights;
      PetscReal       *basis;
      PetscInt         numBasisFuncs, numBasisComps, q, d, e, fc, f;

      ierr = PetscFEGetDimension(fe[field], &numBasisFuncs);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe[field], &numBasisComps);CHKERRQ(ierr);
      ierr = PetscFEGetDefaultTabulation(fe[field], &basis, NULL, NULL);CHKERRQ(ierr);
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
        (*funcs[field])(coords, funcVal);
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

#if 0

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeResidualFEM"
PetscErrorCode DMPlexComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM_Plex         *mesh   = (DM_Plex *) dm->data;
  PetscFEM        *fem    = (PetscFEM *) user;
  PetscQuadrature *quad   = fem->quad;
  PetscQuadrature *quadBd = fem->quadBd;
  PetscSection     section;
  PetscReal       *v0, *n, *J, *invJ, *detJ;
  PetscScalar     *elemVec, *u;
  PetscInt         dim, numFields, field, numBatchesTmp = 1, numCells, cStart, cEnd, c;
  PetscInt         cellDof, numComponents;
  PetscBool        has;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (has && quadBd) {
    DMLabel         label;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;

    ierr = DMPlexGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(label, 1, &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, 1, &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (field = 0, cellDof = 0, numComponents = 0; field < numFields; ++field) {
      cellDof       += quadBd[field].numBasisFuncs*quadBd[field].numComponents;
      numComponents += quadBd[field].numComponents;
    }
    ierr = PetscMalloc7(numPoints*cellDof,PetscScalar,&u,numPoints*dim,PetscReal,&v0,numPoints*dim,PetscReal,&n,numPoints*dim*dim,PetscReal,&J,numPoints*dim*dim,PetscReal,&invJ,numPoints,PetscReal,&detJ,numPoints*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscScalar   *x     = NULL;
      PetscInt       i;

      /* TODO: Add normal determination here */
      ierr = DMPlexComputeCellGeometry(dm, point, &v0[p*dim], &J[p*dim*dim], &invJ[p*dim*dim], &detJ[p]);CHKERRQ(ierr);
      if (detJ[p] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", detJ[p], point);
      ierr = DMPlexVecGetClosure(dm, NULL, X, point, NULL, &x);CHKERRQ(ierr);

      for (i = 0; i < cellDof; ++i) u[p*cellDof+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dm, NULL, X, point, NULL, &x);CHKERRQ(ierr);
    }
    for (field = 0; field < numFields; ++field) {
      const PetscInt numQuadPoints = quadBd[field].numQuadPoints;
      const PetscInt numBasisFuncs = quadBd[field].numBasisFuncs;
      void           (*f0)(const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]) = fem->f0BdFuncs[field];
      void           (*f1)(const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]) = fem->f1BdFuncs[field];
      /* Conforming batches */
      PetscInt blockSize  = numBasisFuncs*numQuadPoints;
      PetscInt numBlocks  = 1;
      PetscInt batchSize  = numBlocks * blockSize;
      PetscInt numBatches = numBatchesTmp;
      PetscInt numChunks  = numPoints / (numBatches*batchSize);
      /* Remainder */
      PetscInt numRemainder = numPoints % (numBatches * batchSize);
      PetscInt offset       = numPoints - numRemainder;

      ierr = (*mesh->integrateBdResidualFEM)(numChunks*numBatches*batchSize, numFields, field, quadBd, u, v0, n, J, invJ, detJ, f0, f1, elemVec);CHKERRQ(ierr);
      ierr = (*mesh->integrateBdResidualFEM)(numRemainder, numFields, field, quadBd, &u[offset*cellDof], &v0[offset*dim], &n[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
                                             f0, f1, &elemVec[offset*cellDof]);CHKERRQ(ierr);
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];

      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(point, "Residual", cellDof, &elemVec[p*cellDof]);CHKERRQ(ierr);}
      ierr = DMPlexVecSetClosure(dm, NULL, F, point, &elemVec[p*cellDof], ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree7(u,v0,n,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeResidualFEM"
/*@
  DMPlexComputeResidualFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  The second member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  PetscFEM         *fem   = (PetscFEM *) user;
  PetscFE          *fe    = fem->fe;
  PetscFE          *feAux = fem->feAux;
  PetscFE          *feBd  = fem->feBd;
  const char       *name  = "Residual";
  DM                dmAux;
  Vec               A;
  PetscQuadrature   q;
  PetscCellGeometry geom;
  PetscSection      section, sectionAux;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *elemVec, *u, *a;
  PetscInt          dim, Nf, NfAux = 0, f, numCells, cStart, cEnd, c;
  PetscInt          cellDof = 0, numComponents = 0;
  PetscInt          cellDofAux = 0, numComponentsAux = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (f = 0; f < Nf; ++f) {
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe[f], &Nc);CHKERRQ(ierr);
    cellDof       += Nb*Nc;
    numComponents += Nc;
  }
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(sectionAux, &NfAux);CHKERRQ(ierr);
  }
  for (f = 0; f < NfAux; ++f) {
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(feAux[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(feAux[f], &Nc);CHKERRQ(ierr);
    cellDofAux       += Nb*Nc;
    numComponentsAux += Nc;
  }
  ierr = DMPlexProjectFunctionLocal(dm, fe, fem->bcFuncs, INSERT_BC_VALUES, X);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc6(numCells*cellDof,PetscScalar,&u,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc(numCells*cellDofAux * sizeof(PetscScalar), &a);CHKERRQ(ierr);}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < cellDof; ++i) u[c*cellDof+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (dmAux) {
      ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < cellDofAux; ++i) a[c*cellDofAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  for (f = 0; f < Nf; ++f) {
    void   (*f0)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]) = fem->f0Funcs[f];
    void   (*f1)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]) = fem->f1Funcs[f];
    PetscInt Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscFEGetQuadrature(fe[f], &q);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe[f], NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    blockSize = Nb*q.numPoints;
    batchSize = numBlocks * blockSize;
    ierr =  PetscFESetTileSizes(fe[f], blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    geom.v0   = v0;
    geom.J    = J;
    geom.invJ = invJ;
    geom.detJ = detJ;
    ierr = PetscFEIntegrateResidual(fe[f], Ne, Nf, fe, f, geom, u, NfAux, feAux, a, f0, f1, elemVec);CHKERRQ(ierr);
    geom.v0   = &v0[offset*dim];
    geom.J    = &J[offset*dim*dim];
    geom.invJ = &invJ[offset*dim*dim];
    geom.detJ = &detJ[offset];
    ierr = PetscFEIntegrateResidual(fe[f], Nr, Nf, fe, f, geom, &u[offset*cellDof], NfAux, feAux, &a[offset*cellDofAux], f0, f1, &elemVec[offset*cellDof]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(c, name, cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMPlexVecSetClosure(dm, section, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  if (feBd) {
    DMLabel         label;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numPoints, p;
    PetscReal      *n;

    ierr = DMPlexGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(label, 1, &numPoints);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(label, 1, &pointIS);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (f = 0, cellDof = 0, numComponents = 0; f < Nf; ++f) {
      PetscInt Nb, Nc;

      ierr = PetscFEGetDimension(feBd[f], &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feBd[f], &Nc);CHKERRQ(ierr);
      cellDof       += Nb*Nc;
      numComponents += Nc;
    }
    ierr = PetscMalloc7(numPoints*cellDof,PetscScalar,&u,numPoints*dim,PetscReal,&v0,numPoints*dim,PetscReal,&n,numPoints*dim*dim,PetscReal,&J,numPoints*dim*dim,PetscReal,&invJ,numPoints,PetscReal,&detJ,numPoints*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];
      PetscScalar   *x     = NULL;
      PetscInt       i;

      /* TODO: Add normal determination here */
      ierr = DMPlexComputeCellGeometry(dm, point, &v0[p*dim], &J[p*dim*dim], &invJ[p*dim*dim], &detJ[p]);CHKERRQ(ierr);
      if (detJ[p] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", detJ[p], point);
      ierr = DMPlexVecGetClosure(dm, section, X, point, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < cellDof; ++i) u[p*cellDof+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X, point, NULL, &x);CHKERRQ(ierr);
    }
    for (f = 0; f < Nf; ++f) {
      void   (*f0)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]) = fem->f0BdFuncs[f];
      void   (*f1)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]) = fem->f1BdFuncs[f];
      PetscInt Nb;
      /* Conforming batches */
      PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt Nr, offset;

      ierr = PetscFEGetQuadrature(feBd[f], &q);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(feBd[f], &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(feBd[f], NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      blockSize = Nb*q.numPoints;
      batchSize = numBlocks * blockSize;
      ierr =  PetscFESetTileSizes(feBd[f], blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numPoints / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numPoints % (numBatches*batchSize);
      offset    = numPoints - Nr;
      geom.v0   = v0;
      geom.n    = n;
      geom.J    = J;
      geom.invJ = invJ;
      geom.detJ = detJ;
      ierr = PetscFEIntegrateBdResidual(feBd[f], Ne, Nf, feBd, f, geom, u, 0, NULL, NULL, f0, f1, elemVec);CHKERRQ(ierr);
      geom.v0   = &v0[offset*dim];
      geom.n    = &n[offset*dim];
      geom.J    = &J[offset*dim*dim];
      geom.invJ = &invJ[offset*dim*dim];
      geom.detJ = &detJ[offset];
      ierr = PetscFEIntegrateBdResidual(feBd[f], Nr, Nf, feBd, f, geom, &u[offset*cellDof], 0, NULL, NULL, f0, f1, &elemVec[offset*cellDof]);CHKERRQ(ierr);
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt point = points[p];

      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(point, "BdResidual", cellDof, &elemVec[p*cellDof]);CHKERRQ(ierr);}
      ierr = DMPlexVecSetClosure(dm, NULL, F, point, &elemVec[p*cellDof], ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree7(u,v0,n,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  }
  if (mesh->printFEM) {ierr = DMPrintLocalVec(dm, name, mesh->printTol, F);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeJacobianActionFEM"
/*@C
  DMPlexComputeJacobianActionFEM - Form the local action of Jacobian J(u) on the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. J  - The Jacobian shell matrix
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  The second member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: DMPlexComputeResidualFEM()
@*/
PetscErrorCode DMPlexComputeJacobianActionFEM(DM dm, Mat Jac, Vec X, Vec F, void *user)
{
  DM_Plex          *mesh = (DM_Plex *) dm->data;
  PetscFEM         *fem  = (PetscFEM *) user;
  PetscFE          *fe   = fem->fe;
  PetscQuadrature   quad;
  PetscCellGeometry geom;
  PetscSection      section;
  JacActionCtx     *jctx;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *elemVec, *u, *a;
  PetscInt          dim, numFields, field, numCells, cStart, cEnd, c;
  PetscInt          cellDof = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(DMPLEX_JacobianActionFEM,dm,0,0,0);CHKERRQ(ierr); */
  ierr = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(fe[field], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe[field], &Nc);CHKERRQ(ierr);
    cellDof += Nb*Nc;
  }
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc7(numCells*cellDof,PetscScalar,&u,numCells*cellDof,PetscScalar,&a,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, NULL, jctx->u, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < cellDof; ++i) u[c*cellDof+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, NULL, jctx->u, c, NULL, &x);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, NULL, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < cellDof; ++i) a[c*cellDof+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, NULL, X, c, NULL, &x);CHKERRQ(ierr);
  }
  for (field = 0; field < numFields; ++field) {
    PetscInt Nb;
    /* Conforming batches */
    PetscInt numBlocks  = 1;
    PetscInt numBatches = 1;
    PetscInt numChunks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscFEGetQuadrature(fe[field], &quad);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe[field], &Nb);CHKERRQ(ierr);
    blockSize = Nb*quad.numPoints;
    batchSize = numBlocks * blockSize;
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    geom.v0   = v0;
    geom.J    = J;
    geom.invJ = invJ;
    geom.detJ = detJ;
    ierr = PetscFEIntegrateJacobianAction(fe[field], Ne, numFields, fe, field, geom, u, a, fem->g0Funcs, fem->g1Funcs, fem->g2Funcs, fem->g3Funcs, elemVec);CHKERRQ(ierr);
    geom.v0   = &v0[offset*dim];
    geom.J    = &J[offset*dim*dim];
    geom.invJ = &invJ[offset*dim*dim];
    geom.detJ = &detJ[offset];
    ierr = PetscFEIntegrateJacobianAction(fe[field], Nr, numFields, fe, field, geom, &u[offset*cellDof], &a[offset*cellDof],
                                          fem->g0Funcs, fem->g1Funcs, fem->g2Funcs, fem->g3Funcs, &elemVec[offset*cellDof]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(c, "Jacobian Action", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMPlexVecSetClosure(dm, NULL, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree7(u,a,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  if (mesh->printFEM) {
    PetscMPIInt rank, numProcs;
    PetscInt    p;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &numProcs);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "Jacobian Action:\n");CHKERRQ(ierr);
    for (p = 0; p < numProcs; ++p) {
      if (p == rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  /* ierr = PetscLogEventEnd(DMPLEX_JacobianActionFEM,dm,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeJacobianFEM"
/*@
  DMPlexComputeJacobianFEM - Form the local portion of the Jacobian matrix J at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  The second member of the user context must be an FEMContext.

  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: FormFunctionLocal()
@*/
PetscErrorCode DMPlexComputeJacobianFEM(DM dm, Vec X, Mat Jac, Mat JacP, MatStructure *str,void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  PetscFEM         *fem   = (PetscFEM *) user;
  PetscFE          *fe    = fem->fe;
  PetscFE          *feAux = fem->feAux;
  const char       *name  = "Jacobian";
  DM                dmAux;
  Vec               A;
  PetscQuadrature   quad;
  PetscCellGeometry geom;
  PetscSection      section, globalSection, sectionAux;
  PetscReal        *v0, *J, *invJ, *detJ;
  PetscScalar      *elemMat, *u, *a;
  PetscInt          dim, Nf, NfAux = 0, f, fieldI, fieldJ, numCells, cStart, cEnd, c;
  PetscInt          cellDof = 0, numComponents = 0;
  PetscInt          cellDofAux = 0, numComponentsAux = 0;
  PetscBool         isShell;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (f = 0; f < Nf; ++f) {
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe[f], &Nc);CHKERRQ(ierr);
    cellDof       += Nb*Nc;
    numComponents += Nc;
  }
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(sectionAux, &NfAux);CHKERRQ(ierr);
  }
  for (f = 0; f < NfAux; ++f) {
    PetscInt Nb, Nc;

    ierr = PetscFEGetDimension(feAux[f], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(feAux[f], &Nc);CHKERRQ(ierr);
    cellDofAux       += Nb*Nc;
    numComponentsAux += Nc;
  }
  ierr = DMPlexProjectFunctionLocal(dm, fe, fem->bcFuncs, INSERT_BC_VALUES, X);CHKERRQ(ierr);
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = PetscMalloc6(numCells*cellDof,PetscScalar,&u,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof*cellDof,PetscScalar,&elemMat);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc(numCells*cellDofAux * sizeof(PetscScalar), &a);CHKERRQ(ierr);}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < cellDof; ++i) u[c*cellDof+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (dmAux) {
      ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < cellDofAux; ++i) a[c*cellDofAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  ierr = PetscMemzero(elemMat, numCells*cellDof*cellDof * sizeof(PetscScalar));CHKERRQ(ierr);
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscInt Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscFEGetQuadrature(fe[fieldI], &quad);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe[fieldI], &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe[fieldI], NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    blockSize = Nb*quad.numPoints;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe[fieldI], blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      void   (*g0)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]) = fem->g0Funcs[fieldI*Nf+fieldJ];
      void   (*g1)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]) = fem->g1Funcs[fieldI*Nf+fieldJ];
      void   (*g2)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]) = fem->g2Funcs[fieldI*Nf+fieldJ];
      void   (*g3)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]) = fem->g3Funcs[fieldI*Nf+fieldJ];

      geom.v0   = v0;
      geom.J    = J;
      geom.invJ = invJ;
      geom.detJ = detJ;
      ierr = PetscFEIntegrateJacobian(fe[fieldI], Ne, Nf, fe, fieldI, fieldJ, geom, u, NfAux, feAux, a, g0, g1, g2, g3, elemMat);CHKERRQ(ierr);
      geom.v0   = &v0[offset*dim];
      geom.J    = &J[offset*dim*dim];
      geom.invJ = &invJ[offset*dim*dim];
      geom.detJ = &detJ[offset];
      ierr = PetscFEIntegrateJacobian(fe[fieldI], Nr, Nf, fe, fieldI, fieldJ, geom, &u[offset*cellDof], NfAux, feAux, &a[offset*cellDofAux], g0, g1, g2, g3, &elemMat[offset*cellDof*cellDof]);CHKERRQ(ierr);
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, name, cellDof, cellDof, &elemMat[c*cellDof*cellDof]);CHKERRQ(ierr);}
    ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, c, &elemMat[c*cellDof*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,v0,J,invJ,detJ,elemMat);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
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
  *str = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
