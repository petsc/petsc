#include <petscsf.h>
#include <petsc/private/dmdaimpl.h>     /*I  "petscdmda.h"   I*/


/*@
  DMDAConvertToCell - Convert (i,j,k) to local cell number

  Not Collective

  Input Parameter:
+ da - the distributed array
- s - A MatStencil giving (i,j,k)

  Output Parameter:
. cell - the local cell number

  Level: developer

.seealso: DMDAVecGetClosure()
@*/
PetscErrorCode DMDAConvertToCell(DM dm, MatStencil s, PetscInt *cell)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys /*, mz = da->Ze - da->Zs*/;
  const PetscInt il  = s.i - da->Xs/da->w, jl = dim > 1 ? s.j - da->Ys : 0, kl = dim > 2 ? s.k - da->Zs : 0;

  PetscFunctionBegin;
  *cell = -1;
  if ((s.i < da->Xs/da->w) || (s.i >= da->Xe/da->w))    SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil i %D should be in [%D, %D)", s.i, da->Xs/da->w, da->Xe/da->w);
  if ((dim > 1) && ((s.j < da->Ys) || (s.j >= da->Ye))) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil j %D should be in [%D, %D)", s.j, da->Ys, da->Ye);
  if ((dim > 2) && ((s.k < da->Zs) || (s.k >= da->Ze))) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil k %D should be in [%D, %D)", s.k, da->Zs, da->Ze);
  *cell = (kl*my + jl)*mx + il;
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMDALocatePointsIS_2D_Regular(DM dmregular,Vec pos,IS *iscell)
{
  PetscInt          n,bs,p,npoints;
  PetscInt          xs,xe,Xs,Xe,mxlocal;
  PetscInt          ys,ye,Ys,Ye,mylocal;
  PetscInt          d,c0,c1;
  PetscReal         gmin_l[2],gmax_l[2],dx[2];
  PetscReal         gmin[2],gmax[2];
  PetscInt          *cellidx;
  Vec               coor;
  const PetscScalar *_coor;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(dmregular,&xs,&ys,NULL,&xe,&ye,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dmregular,&Xs,&Ys,NULL,&Xe,&Ye,NULL);CHKERRQ(ierr);
  xe += xs; Xe += Xs; if (xs != Xs) xs -= 1;
  ye += ys; Ye += Ys; if (ys != Ys) ys -= 1;

  mxlocal = xe - xs - 1;
  mylocal = ye - ys - 1;

  ierr = DMGetCoordinatesLocal(dmregular,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&_coor);CHKERRQ(ierr);
  c0 = (xs-Xs) + (ys-Ys)*(Xe-Xs);
  c1 = (xe-2-Xs+1) + (ye-2-Ys+1)*(Xe-Xs);

  gmin_l[0] = PetscRealPart(_coor[2*c0+0]);
  gmin_l[1] = PetscRealPart(_coor[2*c0+1]);

  gmax_l[0] = PetscRealPart(_coor[2*c1+0]);
  gmax_l[1] = PetscRealPart(_coor[2*c1+1]);

  dx[0] = (gmax_l[0]-gmin_l[0])/((PetscReal)mxlocal);
  dx[1] = (gmax_l[1]-gmin_l[1])/((PetscReal)mylocal);

  ierr = VecRestoreArrayRead(coor,&_coor);CHKERRQ(ierr);

  ierr = DMGetBoundingBox(dmregular,gmin,gmax);CHKERRQ(ierr);

  ierr = VecGetLocalSize(pos,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = n/bs;

  ierr = PetscMalloc1(npoints,&cellidx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(pos,&_coor);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal coor_p[2];
    PetscInt  mi[2];

    coor_p[0] = PetscRealPart(_coor[2*p]);
    coor_p[1] = PetscRealPart(_coor[2*p+1]);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    if (coor_p[0] < gmin_l[0]) { continue; }
    if (coor_p[0] > gmax_l[0]) { continue; }
    if (coor_p[1] < gmin_l[1]) { continue; }
    if (coor_p[1] > gmax_l[1]) { continue; }

    for (d=0; d<2; d++) {
      mi[d] = (PetscInt)((coor_p[d] - gmin[d])/dx[d]);
    }

    if (mi[0] < xs)     { continue; }
    if (mi[0] > (xe-1)) { continue; }
    if (mi[1] < ys)     { continue; }
    if (mi[1] > (ye-1)) { continue; }

    if (mi[0] == (xe-1)) { mi[0]--; }
    if (mi[1] == (ye-1)) { mi[1]--; }

    cellidx[p] = (mi[0]-xs) + (mi[1]-ys) * mxlocal;
  }
  ierr = VecRestoreArrayRead(pos,&_coor);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode private_DMDALocatePointsIS_3D_Regular(DM dmregular,Vec pos,IS *iscell)
{
  PetscInt          n,bs,p,npoints;
  PetscInt          xs,xe,Xs,Xe,mxlocal;
  PetscInt          ys,ye,Ys,Ye,mylocal;
  PetscInt          zs,ze,Zs,Ze,mzlocal;
  PetscInt          d,c0,c1;
  PetscReal         gmin_l[3],gmax_l[3],dx[3];
  PetscReal         gmin[3],gmax[3];
  PetscInt          *cellidx;
  Vec               coor;
  const PetscScalar *_coor;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(dmregular,&xs,&ys,&zs,&xe,&ye,&ze);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dmregular,&Xs,&Ys,&Zs,&Xe,&Ye,&Ze);CHKERRQ(ierr);
  xe += xs; Xe += Xs; if (xs != Xs) xs -= 1;
  ye += ys; Ye += Ys; if (ys != Ys) ys -= 1;
  ze += zs; Ze += Zs; if (zs != Zs) zs -= 1;

  mxlocal = xe - xs - 1;
  mylocal = ye - ys - 1;
  mzlocal = ze - zs - 1;

  ierr = DMGetCoordinatesLocal(dmregular,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&_coor);CHKERRQ(ierr);
  c0 = (xs-Xs)     + (ys-Ys)    *(Xe-Xs) + (zs-Zs)    *(Xe-Xs)*(Ye-Ys);
  c1 = (xe-2-Xs+1) + (ye-2-Ys+1)*(Xe-Xs) + (ze-2-Zs+1)*(Xe-Xs)*(Ye-Ys);

  gmin_l[0] = PetscRealPart(_coor[3*c0+0]);
  gmin_l[1] = PetscRealPart(_coor[3*c0+1]);
  gmin_l[2] = PetscRealPart(_coor[3*c0+2]);

  gmax_l[0] = PetscRealPart(_coor[3*c1+0]);
  gmax_l[1] = PetscRealPart(_coor[3*c1+1]);
  gmax_l[2] = PetscRealPart(_coor[3*c1+2]);

  dx[0] = (gmax_l[0]-gmin_l[0])/((PetscReal)mxlocal);
  dx[1] = (gmax_l[1]-gmin_l[1])/((PetscReal)mylocal);
  dx[2] = (gmax_l[2]-gmin_l[2])/((PetscReal)mzlocal);

  ierr = VecRestoreArrayRead(coor,&_coor);CHKERRQ(ierr);

  ierr = DMGetBoundingBox(dmregular,gmin,gmax);CHKERRQ(ierr);

  ierr = VecGetLocalSize(pos,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = n/bs;

  ierr = PetscMalloc1(npoints,&cellidx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(pos,&_coor);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal coor_p[3];
    PetscInt  mi[3];

    coor_p[0] = PetscRealPart(_coor[3*p]);
    coor_p[1] = PetscRealPart(_coor[3*p+1]);
    coor_p[2] = PetscRealPart(_coor[3*p+2]);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    if (coor_p[0] < gmin_l[0]) { continue; }
    if (coor_p[0] > gmax_l[0]) { continue; }
    if (coor_p[1] < gmin_l[1]) { continue; }
    if (coor_p[1] > gmax_l[1]) { continue; }
    if (coor_p[2] < gmin_l[2]) { continue; }
    if (coor_p[2] > gmax_l[2]) { continue; }

    for (d=0; d<3; d++) {
      mi[d] = (PetscInt)((coor_p[d] - gmin[d])/dx[d]);
    }

    if (mi[0] < xs)     { continue; }
    if (mi[0] > (xe-1)) { continue; }
    if (mi[1] < ys)     { continue; }
    if (mi[1] > (ye-1)) { continue; }
    if (mi[2] < zs)     { continue; }
    if (mi[2] > (ze-1)) { continue; }

    if (mi[0] == (xe-1)) { mi[0]--; }
    if (mi[1] == (ye-1)) { mi[1]--; }
    if (mi[2] == (ze-1)) { mi[2]--; }

    cellidx[p] = (mi[0]-xs) + (mi[1]-ys) * mxlocal + (mi[2]-zs) * mxlocal * mylocal;
  }
  ierr = VecRestoreArrayRead(pos,&_coor);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocatePoints_DA_Regular(DM dm,Vec pos,DMPointLocationType ltype,PetscSF cellSF)
{
  IS             iscell;
  PetscSFNode    *cells;
  PetscInt       p,bs,dim,npoints,nfound;
  const PetscInt *boxCells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetBlockSize(pos,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Support not provided for 1D");
    case 2:
      ierr = private_DMDALocatePointsIS_2D_Regular(dm,pos,&iscell);CHKERRQ(ierr);
      break;
    case 3:
      ierr = private_DMDALocatePointsIS_3D_Regular(dm,pos,&iscell);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupport spatial dimension");
  }

  ierr = VecGetLocalSize(pos,&npoints);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = npoints / bs;

  ierr = PetscMalloc1(npoints, &cells);CHKERRQ(ierr);
  ierr = ISGetIndices(iscell, &boxCells);CHKERRQ(ierr);

  for (p=0; p<npoints; p++) {
    cells[p].rank  = 0;
    cells[p].index = boxCells[p];
  }
  ierr = ISRestoreIndices(iscell, &boxCells);CHKERRQ(ierr);

  nfound = npoints;
  ierr = PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISDestroy(&iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
