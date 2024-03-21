#include <petscsf.h>
#include <petsc/private/dmdaimpl.h> /*I  "petscdmda.h"   I*/

/*@
  DMDAConvertToCell - Convert a (i,j,k) location in a `DMDA` to its local cell or vertex number

  Not Collective

  Input Parameters:
+ dm - the `DMDA`
- s  - a `MatStencil` that provides (i,j,k)

  Output Parameter:
. cell - the local cell or vertext number

  Level: developer

  Note:
  The (i,j,k) are in the local numbering of the `DMDA`. That is they are non-negative offsets to the ghost corners returned by `DMDAGetGhostCorners()`

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetGhostCorners()`
@*/
PetscErrorCode DMDAConvertToCell(DM dm, MatStencil s, PetscInt *cell)
{
  DM_DA         *da  = (DM_DA *)dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx = (da->Xe - da->Xs) / da->w, my = da->Ye - da->Ys /*, mz = da->Ze - da->Zs*/;
  const PetscInt il = s.i - da->Xs / da->w, jl = dim > 1 ? s.j - da->Ys : 0, kl = dim > 2 ? s.k - da->Zs : 0;

  PetscFunctionBegin;
  *cell = -1;
  PetscCheck(!(s.i < da->Xs / da->w) && !(s.i >= da->Xe / da->w), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil i %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", s.i, da->Xs / da->w, da->Xe / da->w);
  PetscCheck(dim <= 1 || (s.j >= da->Ys && s.j < da->Ye), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil j %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", s.j, da->Ys, da->Ye);
  PetscCheck(dim <= 2 || (s.k >= da->Zs && s.k < da->Ze), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Stencil k %" PetscInt_FMT " should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")", s.k, da->Zs, da->Ze);
  *cell = (kl * my + jl) * mx + il;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetLocalBoundingBox_DA(DM da, PetscReal lmin[], PetscReal lmax[], PetscInt cs[], PetscInt ce[])
{
  PetscInt           xs, xe, Xs, Xe;
  PetscInt           ys, ye, Ys, Ye;
  PetscInt           zs, ze, Zs, Ze;
  PetscInt           dim, M, N, P, c0, c1;
  PetscReal          gmax[3] = {0., 0., 0.};
  const PetscReal   *L, *Lstart;
  Vec                coordinates;
  const PetscScalar *coor;
  DMBoundaryType     bx, by, bz;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xe, &ye, &ze));
  PetscCall(DMDAGetGhostCorners(da, &Xs, &Ys, &Zs, &Xe, &Ye, &Ze));
  PetscCall(DMDAGetInfo(da, &dim, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, &bx, &by, &bz, NULL));
  // Convert from widths to endpoints
  xe += xs;
  Xe += Xs;
  ye += ys;
  Ye += Ys;
  ze += zs;
  Ze += Zs;
  // What is this doing?
  if (xs != Xs && Xs >= 0) xs -= 1;
  if (ys != Ys && Ys >= 0) ys -= 1;
  if (zs != Zs && Zs >= 0) zs -= 1;

  PetscCall(DMGetCoordinatesLocal(da, &coordinates));
  if (!coordinates) {
    PetscCall(DMGetLocalBoundingIndices_DMDA(da, lmin, lmax));
    for (PetscInt d = 0; d < dim; ++d) {
      if (cs) cs[d] = lmin[d];
      if (ce) ce[d] = lmax[d];
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetArrayRead(coordinates, &coor));
  switch (dim) {
  case 1:
    c0 = (xs - Xs);
    c1 = (xe - 2 - Xs + 1);
    break;
  case 2:
    c0 = (xs - Xs) + (ys - Ys) * (Xe - Xs);
    c1 = (xe - 2 - Xs + 1) + (ye - 2 - Ys + 1) * (Xe - Xs);
    break;
  case 3:
    c0 = (xs - Xs) + (ys - Ys) * (Xe - Xs) + (zs - Zs) * (Xe - Xs) * (Ye - Ys);
    c1 = (xe - 2 - Xs + 1) + (ye - 2 - Ys + 1) * (Xe - Xs) + (ze - 2 - Zs + 1) * (Xe - Xs) * (Ye - Ys);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Invalid dimension %" PetscInt_FMT " for DMDA", dim);
  }
  for (PetscInt d = 0; d < dim; ++d) {
    lmin[d] = PetscRealPart(coor[c0 * dim + d]);
    lmax[d] = PetscRealPart(coor[c1 * dim + d]);
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coor));

  PetscCall(DMGetPeriodicity(da, NULL, &Lstart, &L));
  if (L) {
    for (PetscInt d = 0; d < dim; ++d)
      if (L[d] > 0.0) gmax[d] = Lstart[d] + L[d];
  }
  // Must check for periodic boundary
  if (bx == DM_BOUNDARY_PERIODIC && xe == M) {
    lmax[0] = gmax[0];
    ++xe;
  }
  if (by == DM_BOUNDARY_PERIODIC && ye == N) {
    lmax[1] = gmax[1];
    ++ye;
  }
  if (bz == DM_BOUNDARY_PERIODIC && ze == P) {
    lmax[2] = gmax[2];
    ++ze;
  }
  if (cs) {
    cs[0] = xs;
    if (dim > 1) cs[1] = ys;
    if (dim > 2) cs[2] = zs;
  }
  if (ce) {
    ce[0] = xe;
    if (dim > 1) ce[1] = ye;
    if (dim > 2) ce[2] = ze;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode private_DMDALocatePointsIS_2D_Regular(DM dmregular, Vec pos, IS *iscell)
{
  PetscInt           n, bs, npoints;
  PetscInt           cs[2], ce[2];
  PetscInt           xs, xe, mxlocal;
  PetscInt           ys, ye, mylocal;
  PetscReal          gmin_l[2], gmax_l[2], dx[2];
  PetscReal          gmin[2], gmax[2];
  PetscInt          *cellidx;
  const PetscScalar *coor;

  PetscFunctionBegin;
  PetscCall(DMGetLocalBoundingBox_DA(dmregular, gmin_l, gmax_l, cs, ce));
  xs = cs[0];
  ys = cs[1];
  xe = ce[0];
  ye = ce[1];
  PetscCall(DMGetBoundingBox(dmregular, gmin, gmax));

  mxlocal = xe - xs - 1;
  mylocal = ye - ys - 1;

  dx[0] = (gmax_l[0] - gmin_l[0]) / ((PetscReal)mxlocal);
  dx[1] = (gmax_l[1] - gmin_l[1]) / ((PetscReal)mylocal);

  PetscCall(VecGetLocalSize(pos, &n));
  PetscCall(VecGetBlockSize(pos, &bs));
  npoints = n / bs;

  PetscCall(PetscMalloc1(npoints, &cellidx));
  PetscCall(VecGetArrayRead(pos, &coor));
  for (PetscInt p = 0; p < npoints; p++) {
    PetscReal coor_p[2];
    PetscInt  mi[2];

    coor_p[0] = PetscRealPart(coor[2 * p]);
    coor_p[1] = PetscRealPart(coor[2 * p + 1]);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    if (coor_p[0] < gmin_l[0]) continue;
    if (coor_p[0] > gmax_l[0]) continue;
    if (coor_p[1] < gmin_l[1]) continue;
    if (coor_p[1] > gmax_l[1]) continue;

    for (PetscInt d = 0; d < 2; d++) mi[d] = (PetscInt)((coor_p[d] - gmin[d]) / dx[d]);

    if (mi[0] < xs) continue;
    if (mi[0] > (xe - 1)) continue;
    if (mi[1] < ys) continue;
    if (mi[1] > (ye - 1)) continue;

    if (mi[0] == (xe - 1)) mi[0]--;
    if (mi[1] == (ye - 1)) mi[1]--;

    cellidx[p] = (mi[0] - xs) + (mi[1] - ys) * mxlocal;
  }
  PetscCall(VecRestoreArrayRead(pos, &coor));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, npoints, cellidx, PETSC_OWN_POINTER, iscell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode private_DMDALocatePointsIS_3D_Regular(DM dmregular, Vec pos, IS *iscell)
{
  PetscInt           n, bs, npoints;
  PetscInt           cs[3], ce[3];
  PetscInt           xs, xe, mxlocal;
  PetscInt           ys, ye, mylocal;
  PetscInt           zs, ze, mzlocal;
  PetscReal          gmin_l[3], gmax_l[3], dx[3];
  PetscReal          gmin[3], gmax[3];
  PetscInt          *cellidx;
  const PetscScalar *coor;

  PetscFunctionBegin;
  PetscCall(DMGetLocalBoundingBox_DA(dmregular, gmin_l, gmax_l, cs, ce));
  xs = cs[0];
  ys = cs[1];
  zs = cs[2];
  xe = ce[0];
  ye = ce[1];
  ze = ce[2];
  PetscCall(DMGetBoundingBox(dmregular, gmin, gmax));

  mxlocal = xe - xs - 1;
  mylocal = ye - ys - 1;
  mzlocal = ze - zs - 1;

  dx[0] = (gmax_l[0] - gmin_l[0]) / ((PetscReal)mxlocal);
  dx[1] = (gmax_l[1] - gmin_l[1]) / ((PetscReal)mylocal);
  dx[2] = (gmax_l[2] - gmin_l[2]) / ((PetscReal)mzlocal);

  PetscCall(VecGetLocalSize(pos, &n));
  PetscCall(VecGetBlockSize(pos, &bs));
  npoints = n / bs;

  PetscCall(PetscMalloc1(npoints, &cellidx));
  PetscCall(VecGetArrayRead(pos, &coor));
  for (PetscInt p = 0; p < npoints; p++) {
    PetscReal coor_p[3];
    PetscInt  mi[3];

    coor_p[0] = PetscRealPart(coor[3 * p]);
    coor_p[1] = PetscRealPart(coor[3 * p + 1]);
    coor_p[2] = PetscRealPart(coor[3 * p + 2]);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    if (coor_p[0] < gmin_l[0]) continue;
    if (coor_p[0] > gmax_l[0]) continue;
    if (coor_p[1] < gmin_l[1]) continue;
    if (coor_p[1] > gmax_l[1]) continue;
    if (coor_p[2] < gmin_l[2]) continue;
    if (coor_p[2] > gmax_l[2]) continue;

    for (PetscInt d = 0; d < 3; d++) mi[d] = (PetscInt)((coor_p[d] - gmin[d]) / dx[d]);

    // TODO: Check for periodicity here
    if (mi[0] < xs) continue;
    if (mi[0] > (xe - 1)) continue;
    if (mi[1] < ys) continue;
    if (mi[1] > (ye - 1)) continue;
    if (mi[2] < zs) continue;
    if (mi[2] > (ze - 1)) continue;

    if (mi[0] == (xe - 1)) mi[0]--;
    if (mi[1] == (ye - 1)) mi[1]--;
    if (mi[2] == (ze - 1)) mi[2]--;

    cellidx[p] = (mi[0] - xs) + (mi[1] - ys) * mxlocal + (mi[2] - zs) * mxlocal * mylocal;
  }
  PetscCall(VecRestoreArrayRead(pos, &coor));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, npoints, cellidx, PETSC_OWN_POINTER, iscell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMLocatePoints_DA_Regular(DM dm, Vec pos, DMPointLocationType ltype, PetscSF cellSF)
{
  IS              iscell;
  PetscSFNode    *cells;
  PetscInt        p, bs, dim, npoints, nfound;
  const PetscInt *boxCells;

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(pos, &dim));
  switch (dim) {
  case 1:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Support not provided for 1D");
  case 2:
    PetscCall(private_DMDALocatePointsIS_2D_Regular(dm, pos, &iscell));
    break;
  case 3:
    PetscCall(private_DMDALocatePointsIS_3D_Regular(dm, pos, &iscell));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported spatial dimension");
  }

  PetscCall(VecGetLocalSize(pos, &npoints));
  PetscCall(VecGetBlockSize(pos, &bs));
  npoints = npoints / bs;

  PetscCall(PetscMalloc1(npoints, &cells));
  PetscCall(ISGetIndices(iscell, &boxCells));

  for (p = 0; p < npoints; p++) {
    cells[p].rank  = 0;
    cells[p].index = boxCells[p];
  }
  PetscCall(ISRestoreIndices(iscell, &boxCells));

  nfound = npoints;
  PetscCall(PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER));
  PetscCall(ISDestroy(&iscell));
  PetscFunctionReturn(PETSC_SUCCESS);
}
