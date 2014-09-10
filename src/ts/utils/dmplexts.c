#include <petsc-private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc-private/tsimpl.h>     /*I "petscts.h" I*/
#include <petscds.h>
#include <petscfv.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSGetGeometryFVM"
/*@
  DMPlexTSGetGeometryFVM - Return precomputed geometric data

  Input Parameter:
. dm - The DM

  Output Parameters:
+ facegeom - The values precomputed from face geometry
. cellgeom - The values precomputed from cell geometry
- minRadius - The minimum radius over the mesh of an inscribed sphere in a cell

  Level: developer

.seealso: DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexTSGetGeometryFVM(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DMTS           dmts;
  PetscObject    obj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm, &dmts);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_facegeom_fvm", &obj);CHKERRQ(ierr);
  if (!obj) {
    Vec cellgeom, facegeom;

    ierr = DMPlexComputeGeometryFVM(dm, &cellgeom, &facegeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmts, "DMPlexTS_facegeom_fvm", (PetscObject) facegeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmts, "DMPlexTS_cellgeom_fvm", (PetscObject) cellgeom);CHKERRQ(ierr);
    ierr = VecDestroy(&facegeom);CHKERRQ(ierr);
    ierr = VecDestroy(&cellgeom);CHKERRQ(ierr);
  }
  if (facegeom) {PetscValidPointer(facegeom, 2); ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_facegeom_fvm", (PetscObject *) facegeom);CHKERRQ(ierr);}
  if (cellgeom) {PetscValidPointer(cellgeom, 3); ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_cellgeom_fvm", (PetscObject *) cellgeom);CHKERRQ(ierr);}
  if (minRadius) {ierr = DMPlexGetMinRadius(dm, minRadius);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BuildGradientReconstruction"
static PetscErrorCode BuildGradientReconstruction(DM dm, PetscFV fvm, DM dmFace, PetscScalar *fgeom, DM dmCell, PetscScalar *cgeom)
{
  DMLabel        ghostLabel;
  PetscScalar   *dx, *grad, **gref;
  PetscInt       dim, cStart, cEnd, c, cEndInterior, maxNumFaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxNumFaces, NULL);CHKERRQ(ierr);
  ierr = PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = PetscMalloc3(maxNumFaces*dim, &dx, maxNumFaces*dim, &grad, maxNumFaces, &gref);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; c++) {
    const PetscInt        *faces;
    PetscInt               numFaces, usedFaces, f, d;
    const PetscFVCellGeom *cg;
    PetscBool              boundary;
    PetscInt               ghost;

    ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, c, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);CHKERRQ(ierr);
    if (numFaces < dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D has only %D faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      const PetscFVCellGeom *cg1;
      PetscFVFaceGeom       *fg;
      const PetscInt        *fcells;
      PetscInt               ncell, side;

      ierr = DMLabelGetValue(ghostLabel, faces[f], &ghost);CHKERRQ(ierr);
      ierr = DMPlexIsBoundaryPoint(dm, faces[f], &boundary);CHKERRQ(ierr);
      if ((ghost >= 0) || boundary) continue;
      ierr  = DMPlexGetSupport(dm, faces[f], &fcells);CHKERRQ(ierr);
      side  = (c != fcells[0]); /* c is on left=0 or right=1 of face */
      ncell = fcells[!side];    /* the neighbor */
      ierr  = DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg);CHKERRQ(ierr);
      ierr  = DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) dx[usedFaces*dim+d] = cg1->centroid[d] - cg->centroid[d];
      gref[usedFaces++] = fg->grad[side];  /* Gradient reconstruction term will go here */
    }
    if (!usedFaces) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
    ierr = PetscFVComputeGradient(fvm, usedFaces, dx, grad);CHKERRQ(ierr);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      ierr = DMLabelGetValue(ghostLabel, faces[f], &ghost);CHKERRQ(ierr);
      ierr = DMPlexIsBoundaryPoint(dm, faces[f], &boundary);CHKERRQ(ierr);
      if ((ghost >= 0) || boundary) continue;
      for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces*dim+d];
      ++usedFaces;
    }
  }
  ierr = PetscFree3(dx, grad, gref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSSetupGradientFVM_Internal"
static PetscErrorCode DMPlexTSSetupGradientFVM_Internal(DM dm, PetscFV fvm, DM *dmGrad)
{
  DM             dmFace, dmCell;
  Vec            facegeom, cellgeom;
  PetscScalar   *fgeom, *cgeom;
  PetscSection   sectionGrad;
  PetscInt       dim, pdim, cStart, cEnd, cEndInterior, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
  ierr = DMPlexTSGetGeometryFVM(dm, &facegeom, &cellgeom, NULL);CHKERRQ(ierr);
  ierr = VecGetDM(facegeom, &dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArray(facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecGetArray(cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = BuildGradientReconstruction(dm, fvm, dmFace, fgeom, dmCell, cgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(cellgeom, &cgeom);CHKERRQ(ierr);
  /* Create storage for gradients */
  ierr = DMClone(dm, dmGrad);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionGrad);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionGrad, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {ierr = PetscSectionSetDof(sectionGrad, c, pdim*dim);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionGrad);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(*dmGrad, sectionGrad);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSGetGradientDM"
/*@C
  DMPlexTSGetGradientDM - Return gradient data layout

  Input Parameters:
+ dm - The DM
- fv - The PetscFV

  Output Parameter:
. dmGrad - The layout for gradient values

  Level: developer

.seealso: DMPlexTSGetGeometryFVM(), DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexTSGetGradientDM(DM dm, PetscFV fv, DM *dmGrad)
{
  DMTS           dmts;
  PetscObject    obj;
  PetscBool      computeGradients;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(fv,PETSCFV_CLASSID,2);
  PetscValidPointer(dmGrad,3);
  ierr = PetscFVGetComputeGradients(fv, &computeGradients);CHKERRQ(ierr);
  if (!computeGradients) {*dmGrad = NULL; PetscFunctionReturn(0);}
  ierr = DMGetDMTS(dm, &dmts);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_dmgrad_fvm", &obj);CHKERRQ(ierr);
  if (!obj) {
    DM dmGrad;

    ierr = DMPlexTSSetupGradientFVM_Internal(dm, fv, &dmGrad);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmts, "DMPlexTS_dmgrad_fvm", (PetscObject) dmGrad);CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);CHKERRQ(ierr);
  }
  ierr = PetscObjectQuery((PetscObject) dmts, "DMPlexTS_dmgrad_fvm", (PetscObject *) dmGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertBoundaryValuesFVM_Static"
static PetscErrorCode DMPlexInsertBoundaryValuesFVM_Static(DM dm, PetscFV fvm, PetscReal time, Vec locX, Vec Grad)
{
  Vec                faceGeometry, cellGeometry;
  DM                 dmFace, dmCell, dmGrad;
  const PetscScalar *facegeom, *cellgeom, *grad;
  PetscScalar       *x, *fx;
  PetscInt           numBd, b, dim, pdim, fStart, fEnd;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexTSGetGeometryFVM(dm, &faceGeometry, &cellGeometry, NULL);CHKERRQ(ierr);
  ierr = DMPlexTSGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetNumBoundary(dm, &numBd);CHKERRQ(ierr);
  if (Grad) {
    ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, pdim, PETSC_SCALAR, &fx);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Grad, &grad);CHKERRQ(ierr);
  }
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(locX, &x);CHKERRQ(ierr);
  for (b = 0; b < numBd; ++b) {
    PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*);
    DMLabel          label;
    const char      *labelname;
    const PetscInt  *ids;
    PetscInt         numids, i;
    void            *ctx;

    ierr = DMPlexGetBoundary(dm, b, NULL, NULL, &labelname, NULL, (void (**)()) &func, &numids, &ids, &ctx);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, labelname, &label);CHKERRQ(ierr);
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
        const PetscFVFaceGeom *fg;

        if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
        ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
        if (Grad) {
          const PetscFVCellGeom *cg;
          const PetscScalar     *cx, *cgrad;
          PetscScalar           *xG;
          PetscReal              dx[3];
          PetscInt               d;

          ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRead(dm, cells[0], x, &cx);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRead(dmGrad, cells[0], grad, &cgrad);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRef(dm, cells[1], x, &xG);CHKERRQ(ierr);
          DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, fg->centroid, dx);
          for (d = 0; d < pdim; ++d) fx[d] = cx[d] + DMPlex_DotD_Internal(dim, &cgrad[d*dim], dx);
          ierr = (*func)(time, fg->centroid, fg->normal, fx, xG, ctx);CHKERRQ(ierr);
        } else {
          const PetscScalar *xI;
          PetscScalar       *xG;

          ierr = DMPlexPointLocalRead(dm, cells[0], x, &xI);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRef(dm, cells[1], x, &xG);CHKERRQ(ierr);
          ierr = (*func)(time, fg->centroid, fg->normal, xI, xG, ctx);CHKERRQ(ierr);
        }
      }
      ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
      ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(locX, &x);CHKERRQ(ierr);
  if (Grad) {
    ierr = DMRestoreWorkArray(dm, pdim, PETSC_SCALAR, &fx);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Grad, &grad);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSComputeRHSFunctionFVM"
PetscErrorCode DMPlexTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *ctx)
{
  PetscDS            prob;
  void             (*riemann)(const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscScalar[], void *);
  void              *rctx;
  PetscFV            fvm;
  PetscLimiter       lim;
  Vec                faceGeometry, cellGeometry;
  Vec                Grad = NULL, locGrad;
  DM                 dmFace, dmCell, dmGrad;
  DMLabel            ghostLabel;
  PetscFVFaceGeom   *fgeom;
  const PetscScalar *facegeom, *cellgeom, *x, *lgrad;
  PetscScalar       *grad, *f, *uL, *uR, *fluxL, *fluxR;
  PetscReal         *vol, *cellPhi;
  PetscBool          computeGradients;
  PetscInt           Nf, dim, pdim, fStart, fEnd, numFaces = 0, face, iface, cell, cStart, cEnd, cEndInterior;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(locX,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  ierr = DMPlexTSGetGeometryFVM(dm, &faceGeometry, &cellGeometry, NULL);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetRiemannSolver(prob, 0, &riemann);CHKERRQ(ierr);
  ierr = PetscDSGetContext(prob, 0, &rctx);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, (PetscObject *) &fvm);CHKERRQ(ierr);
  ierr = DMPlexTSGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
  ierr = PetscFVGetLimiter(fvm, &lim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = PetscFVGetComputeGradients(fvm, &computeGradients);CHKERRQ(ierr);
  if (computeGradients) {
    ierr = DMGetGlobalVector(dmGrad, &Grad);CHKERRQ(ierr);
    ierr = VecZeroEntries(Grad);CHKERRQ(ierr);
    ierr = VecGetArray(Grad, &grad);CHKERRQ(ierr);
  }
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
  /* Count faces and reconstruct gradients */
  for (face = fStart; face < fEnd; ++face) {
    const PetscInt        *cells;
    const PetscFVFaceGeom *fg;
    const PetscScalar     *cx[2];
    PetscScalar           *cgrad[2];
    PetscBool              boundary;
    PetscInt               ghost, c, pd, d;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ++numFaces;
    if (!computeGradients) continue;
    ierr = DMPlexIsBoundaryPoint(dm, face, &boundary);CHKERRQ(ierr);
    if (boundary) continue;
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    for (c = 0; c < 2; ++c) {
      ierr = DMPlexPointLocalRead(dm, cells[c], x, &cx[c]);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRef(dmGrad, cells[c], grad, &cgrad[c]);CHKERRQ(ierr);
    }
    for (pd = 0; pd < pdim; ++pd) {
      PetscScalar delta = cx[1][pd] - cx[0][pd];

      for (d = 0; d < dim; ++d) {
        if (cgrad[0]) cgrad[0][pd*dim+d] += fg->grad[0][d] * delta;
        if (cgrad[1]) cgrad[1][pd*dim+d] -= fg->grad[1][d] * delta;
      }
    }
  }
  /* Limit interior gradients (using cell-based loop because it generalizes better to vector limiters) */
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, pdim, PETSC_REAL, &cellPhi);CHKERRQ(ierr);
  for (cell = computeGradients && lim ? cStart : cEnd; cell < cEndInterior; ++cell) {
    const PetscInt        *faces;
    const PetscScalar     *cx;
    const PetscFVCellGeom *cg;
    PetscScalar           *cgrad;
    PetscInt               coneSize, f, pd, d;

    ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, cell, x, &cx);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cell, cellgeom, &cg);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmGrad, cell, grad, &cgrad);CHKERRQ(ierr);
    if (!cgrad) continue; /* Unowned overlap cell, we do not compute */
    /* Limiter will be minimum value over all neighbors */
    for (d = 0; d < pdim; ++d) cellPhi[d] = PETSC_MAX_REAL;
    for (f = 0; f < coneSize; ++f) {
      const PetscScalar     *ncx;
      const PetscFVCellGeom *ncg;
      const PetscInt        *fcells;
      PetscInt               face = faces[f], ncell, ghost;
      PetscReal              v[3];
      PetscBool              boundary;

      ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
      ierr = DMPlexIsBoundaryPoint(dm, face, &boundary);CHKERRQ(ierr);
      if ((ghost >= 0) || boundary) continue;
      ierr  = DMPlexGetSupport(dm, face, &fcells);CHKERRQ(ierr);
      ncell = cell == fcells[0] ? fcells[1] : fcells[0];
      ierr  = DMPlexPointLocalRead(dm, ncell, x, &ncx);CHKERRQ(ierr);
      ierr  = DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg);CHKERRQ(ierr);
      DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
      for (d = 0; d < pdim; ++d) {
        /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
        PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / DMPlex_DotD_Internal(dim, &cgrad[d*dim], v);

        ierr = PetscLimiterLimit(lim, flim, &phi);CHKERRQ(ierr);
        cellPhi[d] = PetscMin(cellPhi[d], phi);
      }
    }
    /* Apply limiter to gradient */
    for (pd = 0; pd < pdim; ++pd)
      /* Scalar limiter applied to each component separately */
      for (d = 0; d < dim; ++d) cgrad[pd*dim+d] *= cellPhi[pd];
  }
  ierr = DMRestoreWorkArray(dm, pdim, PETSC_REAL, &cellPhi);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValuesFVM_Static(dm, fvm, time, locX, Grad);CHKERRQ(ierr);
  if (computeGradients) {
    ierr = VecRestoreArray(Grad, &grad);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmGrad, Grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmGrad, Grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmGrad, &Grad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  ierr = PetscMalloc6(numFaces,&fgeom,numFaces*2,&vol,numFaces*pdim,&uL,numFaces*pdim,&uR,numFaces*pdim,&fluxL,numFaces*pdim,&fluxR);CHKERRQ(ierr);
  /* Read out values */
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    const PetscFVFaceGeom *fg;
    const PetscFVCellGeom *cgL, *cgR;
    const PetscScalar     *xL, *xR, *gL, *gR;
    PetscInt               ghost, d;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, cells[0], x, &xL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, cells[1], x, &xR);CHKERRQ(ierr);
    if (computeGradients) {
      PetscReal dxL[3], dxR[3];

      ierr = DMPlexPointLocalRead(dmGrad, cells[0], lgrad, &gL);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmGrad, cells[1], lgrad, &gR);CHKERRQ(ierr);
      DMPlex_WaxpyD_Internal(dim, -1, cgL->centroid, fg->centroid, dxL);
      DMPlex_WaxpyD_Internal(dim, -1, cgR->centroid, fg->centroid, dxR);
      for (d = 0; d < pdim; ++d) {
        uL[iface*pdim+d] = xL[d] + DMPlex_DotD_Internal(dim, &gL[d*dim], dxL);
        uR[iface*pdim+d] = xR[d] + DMPlex_DotD_Internal(dim, &gR[d*dim], dxR);
      }
    } else {
      for (d = 0; d < pdim; ++d) {
        uL[iface*pdim+d] = xL[d];
        uR[iface*pdim+d] = xR[d];
      }
    }
    for (d = 0; d < dim; ++d) {
      fgeom[iface].centroid[d] = fg->centroid[d];
      fgeom[iface].normal[d]   = fg->normal[d];
    }
    vol[iface*2+0] = cgL->volume;
    vol[iface*2+1] = cgR->volume;
    ++iface;
  }
  if (computeGradients) {
    ierr = VecRestoreArrayRead(locGrad,&lgrad);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  /* Riemann solve */
  ierr = PetscFVIntegrateRHSFunction(fvm, numFaces, Nf, &fvm, 0, fgeom, vol, uL, uR, riemann, fluxL, fluxR, rctx);CHKERRQ(ierr);
  /* Insert fluxes */
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt *cells;
    PetscScalar    *fL, *fR;
    PetscInt        ghost, d;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dm, cells[0], f, &fL);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dm, cells[1], f, &fR);CHKERRQ(ierr);
    for (d = 0; d < pdim; ++d) {
      if (fL) fL[d] -= fluxL[iface*pdim+d];
      if (fR) fR[d] += fluxR[iface*pdim+d];
    }
    ++iface;
  }
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  ierr = PetscFree6(fgeom,vol,uL,uR,fluxL,fluxR);CHKERRQ(ierr);
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
