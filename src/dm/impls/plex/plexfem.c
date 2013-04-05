#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

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
/*******************************************************************************
This should be in a separate Discretization object, but I am not sure how to lay
it out yet, so I am stuffing things here while I experiment.
*******************************************************************************/
#undef __FUNCT__
#define __FUNCT__ "DMPlexSetFEMIntegration"
PetscErrorCode DMPlexSetFEMIntegration(DM dm,
                                          PetscErrorCode (*integrateResidualFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                                                                 const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]),
                                          PetscErrorCode (*integrateBdResidualFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                                                                   const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                   void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                                                   void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]), PetscScalar[]),
                                          PetscErrorCode (*integrateJacobianActionFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[], const PetscScalar[],
                                                                                       const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                       void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]),
                                          PetscErrorCode (*integrateJacobianFEM)(PetscInt, PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                                                                 const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                                                 void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]))
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->integrateResidualFEM       = integrateResidualFEM;
  mesh->integrateBdResidualFEM     = integrateBdResidualFEM;
  mesh->integrateJacobianActionFEM = integrateJacobianActionFEM;
  mesh->integrateJacobianFEM       = integrateJacobianFEM;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectFunctionLocal"
PetscErrorCode DMPlexProjectFunctionLocal(DM dm, PetscInt numComp, PetscScalar (**funcs)(const PetscReal []), InsertMode mode, Vec localX)
{
  Vec            coordinates;
  PetscSection   section, cSection;
  PetscInt       dim, vStart, vEnd, v, c, d;
  PetscScalar   *values, *cArray;
  PetscReal     *coords;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &cSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscMalloc(numComp * sizeof(PetscScalar), &values);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &cArray);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(cSection, vStart, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal),&coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(cSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cSection, v, &off);CHKERRQ(ierr);
    if (dof > dim) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Cannot have more coordinates %d then dimensions %d", dof, dim);
    for (d = 0; d < dof; ++d) coords[d] = PetscRealPart(cArray[off+d]);
    for (c = 0; c < numComp; ++c) values[c] = (*funcs[c])(coords);
    ierr = VecSetValuesSection(localX, section, v, values, mode);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &cArray);CHKERRQ(ierr);
  /* Temporary, must be replaced by a projection on the finite element basis */
  {
    PetscInt eStart = 0, eEnd = 0, e, depth;

    ierr = DMPlexGetLabelSize(dm, "depth", &depth);CHKERRQ(ierr);
    --depth;
    if (depth > 1) {ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);}
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt *cone = NULL;
      PetscInt        coneSize, d;
      PetscScalar    *coordsA, *coordsB;

      ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      if (coneSize != 2) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Cone size %d for point %d should be 2", coneSize, e);
      ierr = VecGetValuesSection(coordinates, cSection, cone[0], &coordsA);CHKERRQ(ierr);
      ierr = VecGetValuesSection(coordinates, cSection, cone[1], &coordsB);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coords[d] = 0.5*(PetscRealPart(coordsA[d]) + PetscRealPart(coordsB[d]));
      }
      for (c = 0; c < numComp; ++c) values[c] = (*funcs[c])(coords);
      ierr = VecSetValuesSection(localX, section, e, values, mode);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
#if 0
  const PetscInt localDof = this->_mesh->sizeWithBC(s, *cells->begin());
  PetscReal      detJ;

  ierr = PetscMalloc(localDof * sizeof(PetscScalar), &values);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,PetscReal,&v0,dim*dim,PetscReal,&J);CHKERRQ(ierr);
  ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV(PetscPowInt(this->_mesh->getSieve()->getMaxConeSize(),dim+1), true);

  for (PetscInt c = cStart; c < cEnd; ++c) {
    ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), c, pV);
    const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
    const int                          oSize   = pV.getSize();
    int                                v       = 0;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, NULL, &detJ);CHKERRQ(ierr);
    for (PetscInt cl = 0; cl < oSize; ++cl) {
      const PetscInt fDim;

      ierr = PetscSectionGetDof(oPoints[cl], &fDim);CHKERRQ(ierr);
      if (pointDim) {
        for (PetscInt d = 0; d < fDim; ++d, ++v) {
          values[v] = (*this->_options.integrate)(v0, J, v, initFunc);
        }
      }
    }
    ierr = DMPlexVecSetClosure(dm, NULL, localX, c, values);CHKERRQ(ierr);
    pV.clear();
  }
  ierr = PetscFree2(v0,J);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexProjectFunction"
/*@C
  DMPlexProjectFunction - This projects the given function into the function space provided.

  Input Parameters:
+ dm      - The DM
. numComp - The number of components (functions)
. funcs   - The coordinate functions to evaluate
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

  Level: developer

  Note:
  This currently just calls the function with the coordinates of each vertex and edge midpoint, and stores the result in a vector.
  We will eventually fix it.

.seealso: DMPlexComputeL2Diff()
@*/
PetscErrorCode DMPlexProjectFunction(DM dm, PetscInt numComp, PetscScalar (**funcs)(const PetscReal []), InsertMode mode, Vec X)
{
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMPlexProjectFunctionLocal(dm, numComp, funcs, mode, localX);CHKERRQ(ierr);
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
. quad  - The PetscQuadrature object for each field
. funcs - The functions to evaluate for each field component
- X     - The coefficient vector u_h

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

.seealso: DMPlexProjectFunction()
@*/
PetscErrorCode DMPlexComputeL2Diff(DM dm, PetscQuadrature quad[], PetscScalar (**funcs)(const PetscReal []), Vec X, PetscReal *diff)
{
  const PetscInt debug = 0;
  PetscSection   section;
  Vec            localX;
  PetscReal     *coords, *v0, *J, *invJ, detJ;
  PetscReal      localDiff = 0.0;
  PetscInt       dim, numFields, numComponents = 0, cStart, cEnd, c, field, fieldOffset, comp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    numComponents += quad[field].numComponents;
  }
  ierr = DMPlexProjectFunctionLocal(dm, numComponents, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x;
    PetscReal    elemDiff = 0.0;

    ierr = DMPlexComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      const PetscInt   numQuadPoints = quad[field].numQuadPoints;
      const PetscReal *quadPoints    = quad[field].quadPoints;
      const PetscReal *quadWeights   = quad[field].quadWeights;
      const PetscInt   numBasisFuncs = quad[field].numBasisFuncs;
      const PetscInt   numBasisComps = quad[field].numComponents;
      const PetscReal *basis         = quad[field].basis;
      PetscInt         q, d, e, fc, f;

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
        for (fc = 0; fc < numBasisComps; ++fc) {
          const PetscReal funcVal     = PetscRealPart((*funcs[comp+fc])(coords));
          PetscReal       interpolant = 0.0;
          for (f = 0; f < numBasisFuncs; ++f) {
            const PetscInt fidx = f*numBasisComps+fc;
            interpolant += PetscRealPart(x[fieldOffset+fidx])*basis[q*numBasisFuncs*numBasisComps+fidx];
          }
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ;
        }
      }
      comp        += numBasisComps;
      fieldOffset += numBasisFuncs*numBasisComps;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPI_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

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
  DM_Plex         *mesh   = (DM_Plex*) dm->data;
  PetscFEM        *fem    = (PetscFEM*) &((DM*) user)[1];
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
  /* ierr = PetscLogEventBegin(ResidualFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr     = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr     = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr     = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr     = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0, cellDof = 0, numComponents = 0; field < numFields; ++field) {
    cellDof       += quad[field].numBasisFuncs*quad[field].numComponents;
    numComponents += quad[field].numComponents;
  }
  ierr = DMPlexProjectFunctionLocal(dm, numComponents, fem->bcFuncs, INSERT_BC_VALUES, X);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc6(numCells*cellDof,PetscScalar,&u,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, NULL, X, c, NULL, &x);CHKERRQ(ierr);

    for (i = 0; i < cellDof; ++i) u[c*cellDof+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, NULL, X, c, NULL, &x);CHKERRQ(ierr);
  }
  for (field = 0; field < numFields; ++field) {
    const PetscInt numQuadPoints = quad[field].numQuadPoints;
    const PetscInt numBasisFuncs = quad[field].numBasisFuncs;
    void           (*f0)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]) = fem->f0Funcs[field];
    void           (*f1)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]) = fem->f1Funcs[field];
    /* Conforming batches */
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = numBatchesTmp;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    /* Remainder */
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;

    ierr = (*mesh->integrateResidualFEM)(numChunks*numBatches*batchSize, numFields, field, quad, u, v0, J, invJ, detJ, f0, f1, elemVec);CHKERRQ(ierr);
    ierr = (*mesh->integrateResidualFEM)(numRemainder, numFields, field, quad, &u[offset*cellDof], &v0[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
                                         f0, f1, &elemVec[offset*cellDof]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(c, "Residual", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMPlexVecSetClosure(dm, NULL, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,v0,J,invJ,detJ,elemVec);CHKERRQ(ierr);
  /* Integration over the boundary:
     - This can probably be generalized to integration over a set of labels, however
       the idea here is to do integration where we need the cell normal
     - We can replace hardcoding with a registration process, and this is how we hook
       up the system to something like FEniCS
  */
  ierr = DMPlexHasLabel(dm, "boundary", &has);CHKERRQ(ierr);
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
      PetscScalar   *x;
      PetscInt       i;

      /* TODO: Add normal determination here */
      ierr = DMPlexComputeCellGeometry(dm, point, &v0[p*dim], &J[p*dim*dim], &invJ[p*dim*dim], &detJ[p]);CHKERRQ(ierr);
      if (detJ[p] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[p], p);
      ierr = DMPlexVecGetClosure(dm, NULL, X, point, NULL, &x);CHKERRQ(ierr);

      for (i = 0; i < cellDof; ++i) u[p*cellDof+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dm, NULL, X, point, NULL, &x);CHKERRQ(ierr);
    }
    for (field = 0; field < numFields; ++field) {
      const PetscInt numQuadPoints = quadBd[field].numQuadPoints;
      const PetscInt numBasisFuncs = quadBd[field].numBasisFuncs;
      void           (*f0)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]) = fem->f0BdFuncs[field];
      void           (*f1)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]) = fem->f1BdFuncs[field];
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
  if (mesh->printFEM) {
    PetscMPIInt rank, numProcs;
    PetscInt    p;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &numProcs);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
    for (p = 0; p < numProcs; ++p) {
      if (p == rank) {
        Vec f;

        ierr = VecDuplicate(F, &f);CHKERRQ(ierr);
        ierr = VecCopy(F, f);CHKERRQ(ierr);
        ierr = VecChop(f, 1.0e-10);CHKERRQ(ierr);
        ierr = VecView(f, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = VecDestroy(&f);CHKERRQ(ierr);
        ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  /* ierr = PetscLogEventEnd(ResidualFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

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
  DM_Plex         *mesh = (DM_Plex*) dm->data;
  PetscFEM        *fem  = (PetscFEM*) &((DM*) user)[1];
  PetscQuadrature *quad = fem->quad;
  PetscSection     section;
  JacActionCtx    *jctx;
  PetscReal       *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec, *u, *a;
  PetscInt         dim, numFields, field, numBatchesTmp = 1, numCells, cStart, cEnd, c;
  PetscInt         cellDof = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(JacobianActionFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr     = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
  ierr     = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr     = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr     = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr     = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc7(numCells*cellDof,PetscScalar,&u,numCells*cellDof,PetscScalar,&a,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x;
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
    const PetscInt numQuadPoints = quad[field].numQuadPoints;
    const PetscInt numBasisFuncs = quad[field].numBasisFuncs;
    /* Conforming batches */
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = numBatchesTmp;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    /* Remainder */
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;

    ierr = (*mesh->integrateJacobianActionFEM)(numChunks*numBatches*batchSize, numFields, field, quad, u, a, v0, J, invJ, detJ, fem->g0Funcs, fem->g1Funcs, fem->g2Funcs, fem->g3Funcs, elemVec);CHKERRQ(ierr);
    ierr = (*mesh->integrateJacobianActionFEM)(numRemainder, numFields, field, quad, &u[offset*cellDof], &a[offset*cellDof], &v0[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
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
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian Action:\n");CHKERRQ(ierr);
    for (p = 0; p < numProcs; ++p) {
      if (p == rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  /* ierr = PetscLogEventEnd(JacobianActionFEMEvent,0,0,0,0);CHKERRQ(ierr); */
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
  DM_Plex         *mesh = (DM_Plex*) dm->data;
  PetscFEM        *fem  = (PetscFEM*) &((DM*) user)[1];
  PetscQuadrature *quad = fem->quad;
  PetscSection     section;
  PetscReal       *v0, *J, *invJ, *detJ;
  PetscScalar     *elemMat, *u;
  PetscInt         dim, numFields, field, fieldI, numBatchesTmp = 1, numCells, cStart, cEnd, c;
  PetscInt         cellDof = 0, numComponents = 0;
  PetscBool        isShell;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* ierr = PetscLogEventBegin(JacobianFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr     = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr     = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr     = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr     = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    cellDof       += quad[field].numBasisFuncs*quad[field].numComponents;
    numComponents += quad[field].numComponents;
  }
  ierr = DMPlexProjectFunctionLocal(dm, numComponents, fem->bcFuncs, INSERT_BC_VALUES, X);CHKERRQ(ierr);
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = PetscMalloc6(numCells*cellDof,PetscScalar,&u,numCells*dim,PetscReal,&v0,numCells*dim*dim,PetscReal,&J,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof*cellDof,PetscScalar,&elemMat);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x;
    PetscInt     i;

    ierr = DMPlexComputeCellGeometry(dm, c, &v0[c*dim], &J[c*dim*dim], &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMPlexVecGetClosure(dm, NULL, X, c, NULL, &x);CHKERRQ(ierr);

    for (i = 0; i < cellDof; ++i) u[c*cellDof+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, NULL, X, c, NULL, &x);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(elemMat, numCells*cellDof*cellDof * sizeof(PetscScalar));CHKERRQ(ierr);
  for (fieldI = 0; fieldI < numFields; ++fieldI) {
    const PetscInt numQuadPoints = quad[fieldI].numQuadPoints;
    const PetscInt numBasisFuncs = quad[fieldI].numBasisFuncs;
    PetscInt       fieldJ;

    for (fieldJ = 0; fieldJ < numFields; ++fieldJ) {
      void (*g0)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g0[]) = fem->g0Funcs[fieldI*numFields+fieldJ];
      void (*g1)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[]) = fem->g1Funcs[fieldI*numFields+fieldJ];
      void (*g2)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[]) = fem->g2Funcs[fieldI*numFields+fieldJ];
      void (*g3)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[]) = fem->g3Funcs[fieldI*numFields+fieldJ];
      /* Conforming batches */
      PetscInt blockSize  = numBasisFuncs*numQuadPoints;
      PetscInt numBlocks  = 1;
      PetscInt batchSize  = numBlocks * blockSize;
      PetscInt numBatches = numBatchesTmp;
      PetscInt numChunks  = numCells / (numBatches*batchSize);
      /* Remainder */
      PetscInt numRemainder = numCells % (numBatches * batchSize);
      PetscInt offset       = numCells - numRemainder;

      ierr = (*mesh->integrateJacobianFEM)(numChunks*numBatches*batchSize, numFields, fieldI, fieldJ, quad, u, v0, J, invJ, detJ, g0, g1, g2, g3, elemMat);CHKERRQ(ierr);
      ierr = (*mesh->integrateJacobianFEM)(numRemainder, numFields, fieldI, fieldJ, quad, &u[offset*cellDof], &v0[offset*dim], &J[offset*dim*dim], &invJ[offset*dim*dim], &detJ[offset],
                                           g0, g1, g2, g3, &elemMat[offset*cellDof*cellDof]);CHKERRQ(ierr);
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, "Jacobian", cellDof, cellDof, &elemMat[c*cellDof*cellDof]);CHKERRQ(ierr);}
    ierr = DMPlexMatSetClosure(dm, NULL, NULL, JacP, c, &elemMat[c*cellDof*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,v0,J,invJ,detJ,elemMat);CHKERRQ(ierr);

  /* Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd(). */
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian:\n");CHKERRQ(ierr);
    ierr = MatChop(JacP, 1.0e-10);CHKERRQ(ierr);
    ierr = MatView(JacP, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* ierr = PetscLogEventEnd(JacobianFEMEvent,0,0,0,0);CHKERRQ(ierr); */
  ierr = PetscObjectTypeCompare((PetscObject)Jac, MATSHELL, &isShell);CHKERRQ(ierr);
  if (isShell) {
    JacActionCtx *jctx;

    ierr = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
    ierr = VecCopy(X, jctx->u);CHKERRQ(ierr);
  }
  *str = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
