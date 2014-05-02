#include <petscdmplex.h>          /*I "petscdmplex.h" I*/
#include <petscfv.h>

struct _FVPhysics {
  void (*riemann)(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx);
};

#undef __FUNCT__
#define __FUNCT__ "TSComputeRHSFunctionLocalDMPlex"
PetscErrorCode TSComputeRHSFunctionLocalDMPlex(DM dm, PetscReal time, Vec locX, Vec F, void *ctx)
{
  void             (*riemann)(const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscScalar flux[], void *ctx) = ((struct _FVPhysics *) ctx)->riemann;
  PetscFV            fvm;
  Vec                faceGeometry, cellGeometry;
  DM                 dmFace, dmCell;
  DMLabel            ghostLabel;
  PetscCellGeometry  fgeom, cgeom;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscScalar       *f, *uL, *uR, *fluxL, *fluxR;
  PetscReal         *centroid, *normal, *vol;
  PetscInt           Nf, dim, pdim, fStart, fEnd, numFaces = 0, face, iface;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, (PetscObject *) &fvm);CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValuesFVM(dm, time, locX);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  /* TODO Pull this geometry calculation into the library */
  ierr = PetscObjectQuery((PetscObject) dm, "FaceGeometry", (PetscObject *) &faceGeometry);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "CellGeometry", (PetscObject *) &cellGeometry);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
  for (face = fStart; face < fEnd; ++face) {
    PetscInt ghost;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ++numFaces;
  }
  ierr = PetscMalloc7(numFaces*dim,&centroid,numFaces*dim,&normal,numFaces*2,&vol,numFaces*pdim,&uL,numFaces*pdim,&uR,numFaces*pdim,&fluxL,numFaces*pdim,&fluxR);CHKERRQ(ierr);
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt    *cells;
    const FaceGeom    *fg;
    const CellGeom    *cgL, *cgR;
    const PetscScalar *xL, *xR;
    PetscInt           ghost, d;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, cells[0], x, &xL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, cells[1], x, &xR);CHKERRQ(ierr);
    for (d = 0; d < pdim; ++d) {
      uL[iface*pdim+d] = xL[d];
      uR[iface*pdim+d] = xR[d];
    }
    for (d = 0; d < dim; ++d) {
      centroid[iface*dim+d] = fg->centroid[d];
      normal[iface*dim+d]   = fg->normal[d];
    }
    vol[iface*2+0] = cgL->volume;
    vol[iface*2+1] = cgR->volume;
    ++iface;
  }
  ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  fgeom.v0  = centroid;
  fgeom.n   = normal;
  cgeom.vol = vol;
  ierr = PetscFVIntegrateRHSFunction(fvm, numFaces, Nf, &fvm, 0, fgeom, cgeom, uL, uR, riemann, fluxL, fluxR, ctx);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}
