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
  PetscCall(DMPlexGetTreeChildren(dm,face,&numChildren,&children));
  if (numChildren) {
    PetscInt c;

    for (c = 0; c < numChildren; c++) {
      PetscInt childFace = children[c];

      if (childFace >= fStart && childFace < fEnd) {
        PetscCall(DMPlexApplyLimiter_Internal(dm,dmCell,lim,dim,dof,cell,field,childFace,fStart,fEnd,cellPhi,x,cellgeom,cg,cx,cgrad));
      }
    }
  } else {
    PetscScalar     *ncx;
    PetscFVCellGeom *ncg;
    const PetscInt  *fcells;
    PetscInt         ncell, d;
    PetscReal        v[3];

    PetscCall(DMPlexGetSupport(dm, face, &fcells));
    ncell = cell == fcells[0] ? fcells[1] : fcells[0];
    if (field >= 0) {
      PetscCall(DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx));
    } else {
      PetscCall(DMPlexPointLocalRead(dm, ncell, x, &ncx));
    }
    PetscCall(DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg));
    DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
    for (d = 0; d < dof; ++d) {
      /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
      PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);
      PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;

      PetscCall(PetscLimiterLimit(lim, flim, &phi));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetNumFields(prob, &nFields));
  PetscCall(PetscDSGetFieldIndex(prob, (PetscObject) fvm, &field));
  PetscCall(PetscDSGetFieldSize(prob, field, &dof));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(PetscFVGetLimiter(fvm, &lim));
  PetscCall(VecGetDM(faceGeometry, &dmFace));
  PetscCall(VecGetArrayRead(faceGeometry, &facegeom));
  PetscCall(VecGetDM(cellGeometry, &dmCell));
  PetscCall(VecGetArrayRead(cellGeometry, &cellgeom));
  PetscCall(VecGetArrayRead(locX, &x));
  PetscCall(VecGetDM(grad, &dmGrad));
  PetscCall(VecZeroEntries(grad));
  PetscCall(VecGetArray(grad, &gr));
  /* Reconstruct gradients */
  for (face = fStart; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscScalar           *cx[2];
    PetscScalar           *cgrad[2];
    PetscBool              boundary;
    PetscInt               ghost, c, pd, d, numChildren, numCells;

    PetscCall(DMLabelGetValue(ghostLabel, face, &ghost));
    PetscCall(DMIsBoundaryPoint(dm, face, &boundary));
    PetscCall(DMPlexGetTreeChildren(dm, face, &numChildren, NULL));
    if (ghost >= 0 || boundary || numChildren) continue;
    PetscCall(DMPlexGetSupportSize(dm, face, &numCells));
    PetscCheckFalse(numCells != 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "facet %d has %d support points: expected 2",face,numCells);
    PetscCall(DMPlexGetSupport(dm, face, &cells));
    PetscCall(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
    for (c = 0; c < 2; ++c) {
      if (nFields > 1) {
        PetscCall(DMPlexPointLocalFieldRead(dm, cells[c], field, x, &cx[c]));
      } else {
        PetscCall(DMPlexPointLocalRead(dm, cells[c], x, &cx[c]));
      }
      PetscCall(DMPlexPointGlobalRef(dmGrad, cells[c], gr, &cgrad[c]));
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
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetWorkArray(dm, dof, MPIU_REAL, &cellPhi));
  for (cell = (dmGrad && lim) ? cStart : cEnd; cell < cEnd; ++cell) {
    const PetscInt        *faces;
    PetscScalar           *cx;
    PetscFVCellGeom       *cg;
    PetscScalar           *cgrad;
    PetscInt               coneSize, f, pd, d;

    PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
    PetscCall(DMPlexGetCone(dm, cell, &faces));
    if (nFields > 1) {
      PetscCall(DMPlexPointLocalFieldRead(dm, cell, field, x, &cx));
    }
    else {
      PetscCall(DMPlexPointLocalRead(dm, cell, x, &cx));
    }
    PetscCall(DMPlexPointLocalRead(dmCell, cell, cellgeom, &cg));
    PetscCall(DMPlexPointGlobalRef(dmGrad, cell, gr, &cgrad));
    if (!cgrad) continue; /* Unowned overlap cell, we do not compute */
    /* Limiter will be minimum value over all neighbors */
    for (d = 0; d < dof; ++d) cellPhi[d] = PETSC_MAX_REAL;
    for (f = 0; f < coneSize; ++f) {
      PetscCall(DMPlexApplyLimiter_Internal(dm,dmCell,lim,dim,dof,cell,nFields > 1 ? field : -1,faces[f],fStart,fEnd,cellPhi,x,cellgeom,cg,cx,cgrad));
    }
    /* Apply limiter to gradient */
    for (pd = 0; pd < dof; ++pd)
      /* Scalar limiter applied to each component separately */
      for (d = 0; d < dim; ++d) cgrad[pd*dim+d] *= cellPhi[pd];
  }
  PetscCall(DMRestoreWorkArray(dm, dof, MPIU_REAL, &cellPhi));
  PetscCall(VecRestoreArrayRead(faceGeometry, &facegeom));
  PetscCall(VecRestoreArrayRead(cellGeometry, &cellgeom));
  PetscCall(VecRestoreArrayRead(locX, &x));
  PetscCall(VecRestoreArray(grad, &gr));
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  PetscCheck(useFVM,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm does not have a finite volume discretization");
  PetscCall(DMPlexGetDataFVM(dm, fvm, &cellGeometryFVM, &faceGeometryFVM, &dmGrad));
  PetscCheck(dmGrad,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm's finite volume discretization does not reconstruct gradients");
  PetscCall(VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM));
  PetscCall(VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad));
  PetscFunctionReturn(0);
}
