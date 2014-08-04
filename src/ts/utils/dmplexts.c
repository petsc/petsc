#include <petscdmplex.h>          /*I "petscdmplex.h" I*/
#include <petsc-private/tsimpl.h>   /*I "petscts.h" I*/
#include <petscfv.h>

PETSC_STATIC_INLINE void WaxpyD(PetscInt dim, PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w) {PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];}
PETSC_STATIC_INLINE PetscReal DotD(PetscInt dim, const PetscScalar *x, const PetscReal *y) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += PetscRealPart(x[d])*y[d]; return sum;}
PETSC_STATIC_INLINE PetscReal DotRealD(PetscInt dim, const PetscReal *x, const PetscReal *y) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;}
PETSC_STATIC_INLINE PetscReal NormD(PetscInt dim, const PetscReal *x) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*x[d]; return PetscSqrtReal(sum);}

typedef struct {
  PetscBool setupGeom; /* Flag for geometry setup */
  PetscBool setupGrad; /* Flag for gradient calculation setup */
  Vec       facegeom;  /* FaceGeom struct for each face */
  Vec       cellgeom;  /* CellGeom struct for each cell */
  DM        dmGrad;    /* Layout for the gradient data */
  PetscReal minradius; /* Minimum distance from centroid to face */
  void    (*riemann)(const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscScalar[], void *);
  void     *rhsfunctionlocalctx;
} DMTS_Plex;

#undef __FUNCT__
#define __FUNCT__ "DMTSDestroy_Plex"
static PetscErrorCode DMTSDestroy_Plex(DMTS dmts)
{
  DMTS_Plex     *dmplexts = (DMTS_Plex *) dmts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDestroy(&dmplexts->dmGrad);CHKERRQ(ierr);
  ierr = VecDestroy(&dmplexts->facegeom);CHKERRQ(ierr);
  ierr = VecDestroy(&dmplexts->cellgeom);CHKERRQ(ierr);
  ierr = PetscFree(dmts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSDuplicate_Plex"
static PetscErrorCode DMTSDuplicate_Plex(DMTS olddmts, DMTS dmts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(dmts, (DMTS_Plex **) &dmts->data);CHKERRQ(ierr);
  if (olddmts->data) {ierr = PetscMemcpy(dmts->data, olddmts->data, sizeof(DMTS_Plex));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSGetContext"
static PetscErrorCode DMPlexTSGetContext(DM dm, DMTS dmts, DMTS_Plex **dmplexts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmplexts = NULL;
  if (!dmts->data) {
    ierr = PetscNewLog(dm, (DMTS_Plex **) &dmts->data);CHKERRQ(ierr);
    dmts->ops->destroy   = DMTSDestroy_Plex;
    dmts->ops->duplicate = DMTSDuplicate_Plex;
  }
  *dmplexts = (DMTS_Plex *) dmts->data;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSGetGeometry"
/*@
  DMPlexTSGetGeometry - Return precomputed geometric data

  Input Parameter:
. dm - The DM

  Output Parameters:
+ facegeom - The values precomputed from face geometry
. cellgeom - The values precomputed from cell geometry
- minRadius - The minimum radius over the mesh of an inscribed sphere in a cell

  Level: developer

.seealso: DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexTSGetGeometry(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DMTS           dmts;
  DMTS_Plex     *dmplexts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDMTS(dm, &dmts);CHKERRQ(ierr);
  ierr = DMPlexTSGetContext(dm, dmts, &dmplexts);CHKERRQ(ierr);
  if (facegeom)  *facegeom  = dmplexts->facegeom;
  if (cellgeom)  *cellgeom  = dmplexts->cellgeom;
  if (minRadius) *minRadius = dmplexts->minradius;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSSetupGeometry"
static PetscErrorCode DMPlexTSSetupGeometry(DM dm, PetscFV fvm, DMTS_Plex *dmplexts)
{
  DM             dmFace, dmCell;
  DMLabel        ghostLabel;
  PetscSection   sectionFace, sectionCell;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscReal      minradius;
  PetscScalar   *fgeom, *cgeom;
  PetscInt       dim, cStart, cEnd, cEndInterior, c, fStart, fEnd, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmplexts->setupGeom) PetscFunctionReturn(0);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  /* Make cell centroids and volumes */
  ierr = DMClone(dm, &dmCell);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dmCell, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmCell, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {ierr = PetscSectionSetDof(sectionCell, c, sizeof(CellGeom)/sizeof(PetscScalar));CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmCell, &dmplexts->cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(dmplexts->cellgeom, &cgeom);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; ++c) {
    CellGeom *cg;

    ierr = DMPlexPointLocalRef(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = PetscMemzero(cg, sizeof(*cg));CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dmCell, c, &cg->volume, cg->centroid, NULL);CHKERRQ(ierr);
  }
  /* Compute face normals and minimum cell radius */
  ierr = DMClone(dm, &dmFace);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionFace);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionFace, fStart, fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {ierr = PetscSectionSetDof(sectionFace, f, sizeof(FaceGeom)/sizeof(PetscScalar));CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionFace);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmFace, sectionFace);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionFace);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmFace, &dmplexts->facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(dmplexts->facegeom, &fgeom);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  minradius = PETSC_MAX_REAL;
  for (f = fStart; f < fEnd; ++f) {
    FaceGeom *fg;
    PetscReal area;
    PetscInt  ghost, d;

    ierr = DMLabelGetValue(ghostLabel, f, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ierr = DMPlexPointLocalRef(dmFace, f, fgeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dm, f, &area, fg->centroid, fg->normal);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) fg->normal[d] *= area;
    /* Flip face orientation if necessary to match ordering in support, and Update minimum radius */
    {
      CellGeom       *cL, *cR;
      const PetscInt *cells;
      PetscReal      *lcentroid, *rcentroid;
      PetscReal       v[3];

      ierr = DMPlexGetSupport(dm, f, &cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[0], cgeom, &cL);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[1], cgeom, &cR);CHKERRQ(ierr);
      lcentroid = cells[0] >= cEndInterior ? fg->centroid : cL->centroid;
      rcentroid = cells[1] >= cEndInterior ? fg->centroid : cR->centroid;
      WaxpyD(dim, -1, lcentroid, rcentroid, v);
      if (DotRealD(dim, fg->normal, v) < 0) {
        for (d = 0; d < dim; ++d) fg->normal[d] = -fg->normal[d];
      }
      if (DotRealD(dim, fg->normal, v) <= 0) {
        if (dim == 2) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g) v (%g,%g)", f, (double) fg->normal[0], (double) fg->normal[1], (double) v[0], (double) v[1]);
        if (dim == 3) SETERRQ7(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", f, (double) fg->normal[0], (double) fg->normal[1], (double) fg->normal[2], (double) v[0], (double) v[1], (double) v[2]);
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed", f);
      }
      if (cells[0] < cEndInterior) {
        WaxpyD(dim, -1, fg->centroid, cL->centroid, v);
        minradius = PetscMin(minradius, NormD(dim, v));
      }
      if (cells[1] < cEndInterior) {
        WaxpyD(dim, -1, fg->centroid, cR->centroid, v);
        minradius = PetscMin(minradius, NormD(dim, v));
      }
    }
  }
  ierr = MPI_Allreduce(&minradius, &dmplexts->minradius, 1, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  /* Compute centroids of ghost cells */
  for (c = cEndInterior; c < cEnd; ++c) {
    FaceGeom       *fg;
    const PetscInt *cone,    *support;
    PetscInt        coneSize, supportSize, s;

    ierr = DMPlexGetConeSize(dmCell, c, &coneSize);CHKERRQ(ierr);
    if (coneSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %d has cone size %d != 1", c, coneSize);
    ierr = DMPlexGetCone(dmCell, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmCell, cone[0], &supportSize);CHKERRQ(ierr);
    if (supportSize != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d has support size %d != 1", cone[0], supportSize);
    ierr = DMPlexGetSupport(dmCell, cone[0], &support);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dmFace, cone[0], fgeom, &fg);CHKERRQ(ierr);
    for (s = 0; s < 2; ++s) {
      /* Reflect ghost centroid across plane of face */
      if (support[s] == c) {
        const CellGeom *ci;
        CellGeom       *cg;
        PetscReal       c2f[3], a;

        ierr = DMPlexPointLocalRead(dmCell, support[(s+1)%2], cgeom, &ci);CHKERRQ(ierr);
        WaxpyD(dim, -1, ci->centroid, fg->centroid, c2f); /* cell to face centroid */
        a    = DotRealD(dim, c2f, fg->normal)/DotRealD(dim, fg->normal, fg->normal);
        ierr = DMPlexPointLocalRef(dmCell, support[s], cgeom, &cg);CHKERRQ(ierr);
        WaxpyD(dim, 2*a, fg->normal, ci->centroid, cg->centroid);
        cg->volume = ci->volume;
      }
    }
  }
  ierr = VecRestoreArray(dmplexts->facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(dmplexts->cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  ierr = DMDestroy(&dmFace);CHKERRQ(ierr);
  dmplexts->setupGeom = PETSC_TRUE;
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
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxNumFaces, NULL);CHKERRQ(ierr);
  ierr = PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = PetscMalloc3(maxNumFaces*dim, &dx, maxNumFaces*dim, &grad, maxNumFaces, &gref);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; c++) {
    const PetscInt *faces;
    PetscInt        numFaces, usedFaces, f, d;
    const CellGeom *cg;
    PetscBool       boundary;
    PetscInt        ghost;

    ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, c, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);CHKERRQ(ierr);
    if (numFaces < dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D has only %D faces, not enough for gradient reconstruction", c, numFaces);
    for (f = 0, usedFaces = 0; f < numFaces; ++f) {
      const CellGeom *cg1;
      FaceGeom       *fg;
      const PetscInt *fcells;
      PetscInt        ncell, side;

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
#define __FUNCT__ "DMPlexTSSetupGradient"
static PetscErrorCode DMPlexTSSetupGradient(DM dm, PetscFV fvm, DMTS_Plex *dmplexts)
{
  DM             dmFace, dmCell;
  PetscScalar   *fgeom, *cgeom;
  PetscSection   sectionGrad;
  PetscInt       dim, pdim, cStart, cEnd, cEndInterior, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmplexts->setupGrad) PetscFunctionReturn(0);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  /* Construct the interpolant corresponding to each face from the leat-square solution over the cell neighborhood */
  ierr = VecGetDM(dmplexts->facegeom, &dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(dmplexts->cellgeom, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArray(dmplexts->facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecGetArray(dmplexts->cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = BuildGradientReconstruction(dm, fvm, dmFace, fgeom, dmCell, cgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(dmplexts->facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(dmplexts->cellgeom, &cgeom);CHKERRQ(ierr);
  /* Create storage for gradients */
  ierr = DMClone(dm, &dmplexts->dmGrad);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionGrad);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionGrad, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {ierr = PetscSectionSetDof(sectionGrad, c, pdim*dim);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sectionGrad);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmplexts->dmGrad, sectionGrad);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionGrad);CHKERRQ(ierr);
  dmplexts->setupGrad = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertBoundaryValuesFVM_Static"
static PetscErrorCode DMPlexInsertBoundaryValuesFVM_Static(DM dm, PetscFV fvm, PetscReal time, Vec locX, Vec Grad, DMTS_Plex *dmplexts)
{
  Vec                faceGeometry = dmplexts->facegeom;
  Vec                cellGeometry = dmplexts->cellgeom;
  DM                 dmFace, dmCell;
  const PetscScalar *facegeom, *cellgeom, *grad;
  PetscScalar       *x, *fx;
  PetscInt           numBd, b, dim, pdim, fStart, fEnd;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
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
        const PetscInt     face = faces[f], *cells;
        const FaceGeom    *fg;

        if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
        ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
        if (Grad) {
          const CellGeom    *cg;
          const PetscScalar *cx, *cgrad;
          PetscScalar       *xG;
          PetscReal          dx[3];
          PetscInt           d;

          ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRead(dm, cells[0], x, &cx);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRead(dmplexts->dmGrad, cells[0], grad, &cgrad);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRef(dm, cells[1], x, &xG);CHKERRQ(ierr);
          WaxpyD(dim, -1, cg->centroid, fg->centroid, dx);
          for (d = 0; d < pdim; ++d) fx[d] = cx[d] + DotD(dim, &cgrad[d*dim], dx);
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
#define __FUNCT__ "TSComputeRHSFunction_DMPlex"
static PetscErrorCode TSComputeRHSFunction_DMPlex(TS ts, PetscReal time, Vec X, Vec F, void *ctx)
{
  DM                 dm;
  DMTS_Plex         *dmplexts = (DMTS_Plex *) ctx;
  void             (*riemann)(const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscScalar[], void *) = dmplexts->riemann;
  PetscFV            fvm;
  PetscLimiter       lim;
  Vec                faceGeometry = dmplexts->facegeom;
  Vec                cellGeometry = dmplexts->cellgeom;
  Vec                Grad = NULL, locGrad, locX;
  DM                 dmFace, dmCell;
  DMLabel            ghostLabel;
  PetscCellGeometry  fgeom, cgeom;
  const PetscScalar *facegeom, *cellgeom, *x, *lgrad;
  PetscScalar       *grad, *f, *uL, *uR, *fluxL, *fluxR;
  PetscReal         *centroid, *normal, *vol, *cellPhi;
  PetscBool          computeGradients;
  PetscInt           Nf, dim, pdim, fStart, fEnd, numFaces = 0, face, iface, cell, cStart, cEnd, cEndInterior;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, (PetscObject *) &fvm);CHKERRQ(ierr);
  ierr = PetscFVGetLimiter(fvm, &lim);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = PetscFVGetComputeGradients(fvm, &computeGradients);CHKERRQ(ierr);
  if (computeGradients) {
    ierr = DMGetGlobalVector(dmplexts->dmGrad, &Grad);CHKERRQ(ierr);
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
    const PetscInt    *cells;
    const FaceGeom    *fg;
    const PetscScalar *cx[2];
    PetscScalar       *cgrad[2];
    PetscBool          boundary;
    PetscInt           ghost, c, pd, d;

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
      ierr = DMPlexPointGlobalRef(dmplexts->dmGrad, cells[c], grad, &cgrad[c]);CHKERRQ(ierr);
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
    const PetscInt    *faces;
    const PetscScalar *cx;
    const CellGeom    *cg;
    PetscScalar       *cgrad;
    PetscInt           coneSize, f, pd, d;

    ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, cell, x, &cx);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cell, cellgeom, &cg);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmplexts->dmGrad, cell, grad, &cgrad);CHKERRQ(ierr);
    if (!cgrad) continue; /* Unowned overlap cell, we do not compute */
    /* Limiter will be minimum value over all neighbors */
    for (d = 0; d < pdim; ++d) cellPhi[d] = PETSC_MAX_REAL;
    for (f = 0; f < coneSize; ++f) {
      const PetscScalar *ncx;
      const CellGeom    *ncg;
      const PetscInt    *fcells;
      PetscInt           face = faces[f], ncell, ghost;
      PetscReal          v[3];
      PetscBool          boundary;

      ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
      ierr = DMPlexIsBoundaryPoint(dm, face, &boundary);CHKERRQ(ierr);
      if ((ghost >= 0) || boundary) continue;
      ierr  = DMPlexGetSupport(dm, face, &fcells);CHKERRQ(ierr);
      ncell = cell == fcells[0] ? fcells[1] : fcells[0];
      ierr  = DMPlexPointLocalRead(dm, ncell, x, &ncx);CHKERRQ(ierr);
      ierr  = DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg);CHKERRQ(ierr);
      WaxpyD(dim, -1, cg->centroid, ncg->centroid, v);
      for (d = 0; d < pdim; ++d) {
        /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
        PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / DotD(dim, &cgrad[d*dim], v);

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
  ierr = DMPlexInsertBoundaryValuesFVM_Static(dm, fvm, time, locX, Grad, dmplexts);CHKERRQ(ierr);
  if (computeGradients) {
    ierr = VecRestoreArray(Grad, &grad);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmplexts->dmGrad, &locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmplexts->dmGrad, Grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmplexts->dmGrad, Grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmplexts->dmGrad, &Grad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  ierr = PetscMalloc7(numFaces*dim,&centroid,numFaces*dim,&normal,numFaces*2,&vol,numFaces*pdim,&uL,numFaces*pdim,&uR,numFaces*pdim,&fluxL,numFaces*pdim,&fluxR);CHKERRQ(ierr);
  /* Read out values */
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt    *cells;
    const FaceGeom    *fg;
    const CellGeom    *cgL, *cgR;
    const PetscScalar *xL, *xR, *gL, *gR;
    PetscInt           ghost, d;

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

      ierr = DMPlexPointLocalRead(dmplexts->dmGrad, cells[0], lgrad, &gL);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmplexts->dmGrad, cells[1], lgrad, &gR);CHKERRQ(ierr);
      WaxpyD(dim, -1, cgL->centroid, fg->centroid, dxL);
      WaxpyD(dim, -1, cgR->centroid, fg->centroid, dxR);
      for (d = 0; d < pdim; ++d) {
        uL[iface*pdim+d] = xL[d] + DotD(dim, &gL[d*dim], dxL);
        uR[iface*pdim+d] = xR[d] + DotD(dim, &gR[d*dim], dxR);
      }
    } else {
      for (d = 0; d < pdim; ++d) {
        uL[iface*pdim+d] = xL[d];
        uR[iface*pdim+d] = xR[d];
      }
    }
    for (d = 0; d < dim; ++d) {
      centroid[iface*dim+d] = fg->centroid[d];
      normal[iface*dim+d]   = fg->normal[d];
    }
    vol[iface*2+0] = cgL->volume;
    vol[iface*2+1] = cgR->volume;
    ++iface;
  }
  if (computeGradients) {
    ierr = VecRestoreArrayRead(locGrad,&lgrad);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmplexts->dmGrad, &locGrad);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  fgeom.v0  = centroid;
  fgeom.n   = normal;
  cgeom.vol = vol;
  /* Riemann solve */
  ierr = PetscFVIntegrateRHSFunction(fvm, numFaces, Nf, &fvm, 0, fgeom, cgeom, uL, uR, riemann, fluxL, fluxR, dmplexts->rhsfunctionlocalctx);CHKERRQ(ierr);
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
  ierr = PetscFree7(centroid,normal,vol,uL,uR,fluxL,fluxR);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTSSetRHSFunctionLocal"
/*@C
  DMPlexTSSetRHSFunctionLocal - set a local residual evaluation function

  Logically Collective

  Input Arguments:
+ dm      - DM to associate callback with
. riemann - Riemann solver
- ctx     - optional context for Riemann solve

  Calling sequence for riemann:

$ riemann(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx)

+ x    - The coordinates at a point on the interface
. n    - The normal vector to the interface
. uL   - The state vector to the left of the interface
. uR   - The state vector to the right of the interface
. flux - output array of flux through the interface
- ctx  - optional user context

  Level: beginner

.seealso: DMTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexTSSetRHSFunctionLocal(DM dm, void (*riemann)(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx), void *ctx)
{
  DMTS           dmts;
  DMTS_Plex     *dmplexts;
  PetscFV        fvm;
  PetscInt       Nf;
  PetscBool      computeGradients;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDMTSWrite(dm, &dmts);CHKERRQ(ierr);
  ierr = DMPlexTSGetContext(dm, dmts, &dmplexts);CHKERRQ(ierr);
  dmplexts->riemann             = riemann;
  dmplexts->rhsfunctionlocalctx = ctx;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, (PetscObject *) &fvm);CHKERRQ(ierr);
  ierr = DMPlexTSSetupGeometry(dm, fvm, dmplexts);CHKERRQ(ierr);
  ierr = PetscFVGetComputeGradients(fvm, &computeGradients);CHKERRQ(ierr);
  if (computeGradients) {ierr = DMPlexTSSetupGradient(dm, fvm, dmplexts);CHKERRQ(ierr);}
  ierr = DMTSSetRHSFunction(dm, TSComputeRHSFunction_DMPlex, dmplexts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
