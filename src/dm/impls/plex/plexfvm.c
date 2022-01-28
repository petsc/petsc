#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>

static PetscErrorCode DMPlexApplyLimiter_Internal(DM dm, DM dmCell, PetscLimiter lim, PetscInt dim, PetscInt dof, PetscInt cell, PetscInt field, PetscInt face, PetscInt fStart, PetscInt fEnd,
                                                  PetscReal *cellPhi, const PetscScalar *x, const PetscScalar *cellgeom, const PetscFVCellGeom *cg, const PetscScalar *cx, const PetscScalar *cgrad)
{
  const PetscInt *children;
  PetscInt        numChildren;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetTreeChildren(dm,face,&numChildren,&children);CHKERRQ(ierr);
  if (numChildren) {
    PetscInt c;

    for (c = 0; c < numChildren; c++) {
      PetscInt childFace = children[c];

      if (childFace >= fStart && childFace < fEnd) {
        ierr = DMPlexApplyLimiter_Internal(dm,dmCell,lim,dim,dof,cell,field,childFace,fStart,fEnd,cellPhi,x,cellgeom,cg,cx,cgrad);CHKERRQ(ierr);
      }
    }
  } else {
    PetscScalar     *ncx;
    PetscFVCellGeom *ncg;
    const PetscInt  *fcells;
    PetscInt         ncell, d;
    PetscReal        v[3];

    ierr  = DMPlexGetSupport(dm, face, &fcells);CHKERRQ(ierr);
    ncell = cell == fcells[0] ? fcells[1] : fcells[0];
    if (field >= 0) {
      ierr  = DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx);CHKERRQ(ierr);
    } else {
      ierr  = DMPlexPointLocalRead(dm, ncell, x, &ncx);CHKERRQ(ierr);
    }
    ierr  = DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg);CHKERRQ(ierr);
    DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
    for (d = 0; d < dof; ++d) {
      /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
      PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);
      PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;

      ierr = PetscLimiterLimit(lim, flim, &phi);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &nFields);CHKERRQ(ierr);
  ierr = PetscDSGetFieldIndex(prob, (PetscObject) fvm, &field);CHKERRQ(ierr);
  ierr = PetscDSGetFieldSize(prob, field, &dof);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = PetscFVGetLimiter(fvm, &lim);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecGetDM(grad, &dmGrad);CHKERRQ(ierr);
  ierr = VecZeroEntries(grad);CHKERRQ(ierr);
  ierr = VecGetArray(grad, &gr);CHKERRQ(ierr);
  /* Reconstruct gradients */
  for (face = fStart; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscScalar           *cx[2];
    PetscScalar           *cgrad[2];
    PetscBool              boundary;
    PetscInt               ghost, c, pd, d, numChildren, numCells;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    ierr = DMIsBoundaryPoint(dm, face, &boundary);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, face, &numChildren, NULL);CHKERRQ(ierr);
    if (ghost >= 0 || boundary || numChildren) continue;
    ierr = DMPlexGetSupportSize(dm, face, &numCells);CHKERRQ(ierr);
    if (numCells != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "facet %d has %d support points: expected 2",face,numCells);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    for (c = 0; c < 2; ++c) {
      if (nFields > 1) {
        ierr = DMPlexPointLocalFieldRead(dm, cells[c], field, x, &cx[c]);CHKERRQ(ierr);
      } else {
        ierr = DMPlexPointLocalRead(dm, cells[c], x, &cx[c]);CHKERRQ(ierr);
      }
      ierr = DMPlexPointGlobalRef(dmGrad, cells[c], gr, &cgrad[c]);CHKERRQ(ierr);
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
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, dof, MPIU_REAL, &cellPhi);CHKERRQ(ierr);
  for (cell = (dmGrad && lim) ? cStart : cEnd; cell < cEnd; ++cell) {
    const PetscInt        *faces;
    PetscScalar           *cx;
    PetscFVCellGeom       *cg;
    PetscScalar           *cgrad;
    PetscInt               coneSize, f, pd, d;

    ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
    if (nFields > 1) {
      ierr = DMPlexPointLocalFieldRead(dm, cell, field, x, &cx);CHKERRQ(ierr);
    }
    else {
      ierr = DMPlexPointLocalRead(dm, cell, x, &cx);CHKERRQ(ierr);
    }
    ierr = DMPlexPointLocalRead(dmCell, cell, cellgeom, &cg);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmGrad, cell, gr, &cgrad);CHKERRQ(ierr);
    if (!cgrad) continue; /* Unowned overlap cell, we do not compute */
    /* Limiter will be minimum value over all neighbors */
    for (d = 0; d < dof; ++d) cellPhi[d] = PETSC_MAX_REAL;
    for (f = 0; f < coneSize; ++f) {
      ierr = DMPlexApplyLimiter_Internal(dm,dmCell,lim,dim,dof,cell,nFields > 1 ? field : -1,faces[f],fStart,fEnd,cellPhi,x,cellgeom,cg,cx,cgrad);CHKERRQ(ierr);
    }
    /* Apply limiter to gradient */
    for (pd = 0; pd < dof; ++pd)
      /* Scalar limiter applied to each component separately */
      for (d = 0; d < dim; ++d) cgrad[pd*dim+d] *= cellPhi[pd];
  }
  ierr = DMRestoreWorkArray(dm, dof, MPIU_REAL, &cellPhi);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(grad, &gr);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  if (!useFVM) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm does not have a finite volume discretization");
  ierr = DMPlexGetDataFVM(dm, fvm, &cellGeometryFVM, &faceGeometryFVM, &dmGrad);CHKERRQ(ierr);
  if (!dmGrad) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm's finite volume discretization does not reconstruct gradients");
  ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
