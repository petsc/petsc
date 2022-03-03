#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>

static PetscErrorCode DMPlexApplyLimiter_Internal(DM dm, DM dmCell, PetscLimiter lim, PetscInt dim, PetscInt dof, PetscInt cell, PetscInt field, PetscInt face, PetscInt fStart, PetscInt fEnd,
                                                  PetscReal *cellPhi, const PetscScalar *x, const PetscScalar *cellgeom, const PetscFVCellGeom *cg, const PetscScalar *cx, const PetscScalar *cgrad)
{
  const PetscInt *children;
  PetscInt        numChildren;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetTreeChildren(dm,face,&numChildren,&children));
  if (numChildren) {
    PetscInt c;

    for (c = 0; c < numChildren; c++) {
      PetscInt childFace = children[c];

      if (childFace >= fStart && childFace < fEnd) {
        CHKERRQ(DMPlexApplyLimiter_Internal(dm,dmCell,lim,dim,dof,cell,field,childFace,fStart,fEnd,cellPhi,x,cellgeom,cg,cx,cgrad));
      }
    }
  } else {
    PetscScalar     *ncx;
    PetscFVCellGeom *ncg;
    const PetscInt  *fcells;
    PetscInt         ncell, d;
    PetscReal        v[3];

    CHKERRQ(DMPlexGetSupport(dm, face, &fcells));
    ncell = cell == fcells[0] ? fcells[1] : fcells[0];
    if (field >= 0) {
      CHKERRQ(DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx));
    } else {
      CHKERRQ(DMPlexPointLocalRead(dm, ncell, x, &ncx));
    }
    CHKERRQ(DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg));
    DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
    for (d = 0; d < dof; ++d) {
      /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
      PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);
      PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;

      CHKERRQ(PetscLimiterLimit(lim, flim, &phi));
      cellPhi[d] = PetscMin(cellPhi[d], phi);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexReconstructGradients_Internal(DM dm, PetscFV fvm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, Vec locX, Vec grad)
{
  DM                 dmFace, dmCell, dmGrad;
  DMLabel            ghostLabel;
  PetscDS            prob;
  PetscLimiter       lim;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscScalar       *gr;
  PetscReal         *cellPhi;
  PetscInt           dim, face, cell, field, dof, cStart, cEnd, nFields;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &nFields));
  CHKERRQ(PetscDSGetFieldIndex(prob, (PetscObject) fvm, &field));
  CHKERRQ(PetscDSGetFieldSize(prob, field, &dof));
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(PetscFVGetLimiter(fvm, &lim));
  CHKERRQ(VecGetDM(faceGeometry, &dmFace));
  CHKERRQ(VecGetArrayRead(faceGeometry, &facegeom));
  CHKERRQ(VecGetDM(cellGeometry, &dmCell));
  CHKERRQ(VecGetArrayRead(cellGeometry, &cellgeom));
  CHKERRQ(VecGetArrayRead(locX, &x));
  CHKERRQ(VecGetDM(grad, &dmGrad));
  CHKERRQ(VecZeroEntries(grad));
  CHKERRQ(VecGetArray(grad, &gr));
  /* Reconstruct gradients */
  for (face = fStart; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscScalar           *cx[2];
    PetscScalar           *cgrad[2];
    PetscBool              boundary;
    PetscInt               ghost, c, pd, d, numChildren, numCells;

    CHKERRQ(DMLabelGetValue(ghostLabel, face, &ghost));
    CHKERRQ(DMIsBoundaryPoint(dm, face, &boundary));
    CHKERRQ(DMPlexGetTreeChildren(dm, face, &numChildren, NULL));
    if (ghost >= 0 || boundary || numChildren) continue;
    CHKERRQ(DMPlexGetSupportSize(dm, face, &numCells));
    PetscCheckFalse(numCells != 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "facet %d has %d support points: expected 2",face,numCells);
    CHKERRQ(DMPlexGetSupport(dm, face, &cells));
    CHKERRQ(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
    for (c = 0; c < 2; ++c) {
      if (nFields > 1) {
        CHKERRQ(DMPlexPointLocalFieldRead(dm, cells[c], field, x, &cx[c]));
      } else {
        CHKERRQ(DMPlexPointLocalRead(dm, cells[c], x, &cx[c]));
      }
      CHKERRQ(DMPlexPointGlobalRef(dmGrad, cells[c], gr, &cgrad[c]));
    }
    for (pd = 0; pd < dof; ++pd) {
      PetscScalar delta = cx[1][pd] - cx[0][pd];

      for (d = 0; d < dim; ++d) {
        if (cgrad[0]) cgrad[0][pd*dim+d] += fg->grad[0][d] * delta;
        if (cgrad[1]) cgrad[1][pd*dim+d] -= fg->grad[1][d] * delta;
      }
    }
  }
  /* Limit interior gradients (using cell-based loop because it generalizes better to vector limiters) */
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMGetWorkArray(dm, dof, MPIU_REAL, &cellPhi));
  for (cell = (dmGrad && lim) ? cStart : cEnd; cell < cEnd; ++cell) {
    const PetscInt        *faces;
    PetscScalar           *cx;
    PetscFVCellGeom       *cg;
    PetscScalar           *cgrad;
    PetscInt               coneSize, f, pd, d;

    CHKERRQ(DMPlexGetConeSize(dm, cell, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, cell, &faces));
    if (nFields > 1) {
      CHKERRQ(DMPlexPointLocalFieldRead(dm, cell, field, x, &cx));
    }
    else {
      CHKERRQ(DMPlexPointLocalRead(dm, cell, x, &cx));
    }
    CHKERRQ(DMPlexPointLocalRead(dmCell, cell, cellgeom, &cg));
    CHKERRQ(DMPlexPointGlobalRef(dmGrad, cell, gr, &cgrad));
    if (!cgrad) continue; /* Unowned overlap cell, we do not compute */
    /* Limiter will be minimum value over all neighbors */
    for (d = 0; d < dof; ++d) cellPhi[d] = PETSC_MAX_REAL;
    for (f = 0; f < coneSize; ++f) {
      CHKERRQ(DMPlexApplyLimiter_Internal(dm,dmCell,lim,dim,dof,cell,nFields > 1 ? field : -1,faces[f],fStart,fEnd,cellPhi,x,cellgeom,cg,cx,cgrad));
    }
    /* Apply limiter to gradient */
    for (pd = 0; pd < dof; ++pd)
      /* Scalar limiter applied to each component separately */
      for (d = 0; d < dim; ++d) cgrad[pd*dim+d] *= cellPhi[pd];
  }
  CHKERRQ(DMRestoreWorkArray(dm, dof, MPIU_REAL, &cellPhi));
  CHKERRQ(VecRestoreArrayRead(faceGeometry, &facegeom));
  CHKERRQ(VecRestoreArrayRead(cellGeometry, &cellgeom));
  CHKERRQ(VecRestoreArrayRead(locX, &x));
  CHKERRQ(VecRestoreArray(grad, &gr));
  PetscFunctionReturn(0);
}

/*@
  DMPlexReconstructGradientsFVM - reconstruct the gradient of a vector using a finite volume method.

  Input Parameters:
+ dm - the mesh
- locX - the local representation of the vector

  Output Parameter:
. grad - the global representation of the gradient

  Level: developer

.seealso: DMPlexGetGradientDM()
@*/
PetscErrorCode DMPlexReconstructGradientsFVM(DM dm, Vec locX, Vec grad)
{
  PetscDS          prob;
  PetscInt         Nf, f, fStart, fEnd;
  PetscBool        useFVM = PETSC_FALSE;
  PetscFV          fvm = NULL;
  Vec              faceGeometryFVM, cellGeometryFVM;
  PetscFVCellGeom  *cgeomFVM   = NULL;
  PetscFVFaceGeom  *fgeomFVM   = NULL;
  DM               dmGrad = NULL;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  PetscCheck(useFVM,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm does not have a finite volume discretization");
  CHKERRQ(DMPlexGetDataFVM(dm, fvm, &cellGeometryFVM, &faceGeometryFVM, &dmGrad));
  PetscCheck(dmGrad,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm's finite volume discretization does not reconstruct gradients");
  CHKERRQ(VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM));
  CHKERRQ(VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  CHKERRQ(DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad));
  PetscFunctionReturn(0);
}
