/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

/*@
   DMDAGetLogicalCoordinate - Returns a the i,j,k logical coordinate for the closest mesh point to a x,y,z point in the coordinates of the `DMDA`

   Collective on da

   Input Parameters:
+  da - the distributed array
.  x  - the first physical coordinate
.  y  - the second physical coordinate
-  z  - the third physical coordinate

   Output Parameters:
+  II - the first logical coordinate (-1 on processes that do not contain that point)
.  JJ - the second logical coordinate (-1 on processes that do not contain that point)
.  KK - the third logical coordinate (-1 on processes that do not contain that point)
.  X  - (optional) the first coordinate of the located grid point
.  Y  - (optional) the second coordinate of the located grid point
-  Z  - (optional) the third coordinate of the located grid point

   Level: advanced

   Note:
   All processors that share the `DMDA` must call this with the same coordinate value

.seealso: `DM`, `DMDA`
@*/
PetscErrorCode DMDAGetLogicalCoordinate(DM da, PetscScalar x, PetscScalar y, PetscScalar z, PetscInt *II, PetscInt *JJ, PetscInt *KK, PetscScalar *X, PetscScalar *Y, PetscScalar *Z)
{
  Vec          coors;
  DM           dacoors;
  DMDACoor2d **c;
  PetscInt     i, j, xs, xm, ys, ym;
  PetscReal    d, D = PETSC_MAX_REAL, Dv;
  PetscMPIInt  rank, root;

  PetscFunctionBegin;
  PetscCheck(da->dim != 1, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot get point from 1d DMDA");
  PetscCheck(da->dim != 3, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot get point from 3d DMDA");

  *II = -1;
  *JJ = -1;

  PetscCall(DMGetCoordinateDM(da, &dacoors));
  PetscCall(DMDAGetCorners(dacoors, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMGetCoordinates(da, &coors));
  PetscCall(DMDAVecGetArrayRead(dacoors, coors, &c));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      d = PetscSqrtReal(PetscRealPart((c[j][i].x - x) * (c[j][i].x - x) + (c[j][i].y - y) * (c[j][i].y - y)));
      if (d < D) {
        D   = d;
        *II = i;
        *JJ = j;
      }
    }
  }
  PetscCall(MPIU_Allreduce(&D, &Dv, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)da)));
  if (D != Dv) {
    *II  = -1;
    *JJ  = -1;
    rank = 0;
  } else {
    *X = c[*JJ][*II].x;
    *Y = c[*JJ][*II].y;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)da), &rank));
    rank++;
  }
  PetscCall(MPIU_Allreduce(&rank, &root, 1, MPI_INT, MPI_SUM, PetscObjectComm((PetscObject)da)));
  root--;
  PetscCallMPI(MPI_Bcast(X, 1, MPIU_SCALAR, root, PetscObjectComm((PetscObject)da)));
  PetscCallMPI(MPI_Bcast(Y, 1, MPIU_SCALAR, root, PetscObjectComm((PetscObject)da)));
  PetscCall(DMDAVecRestoreArrayRead(dacoors, coors, &c));
  PetscFunctionReturn(0);
}

/*@
   DMDAGetRay - Returns a vector on process zero that contains a row or column of the values in a `DMDA` vector

   Collective on da

   Input Parameters:
+  da - the distributed array
.  dir - Cartesian direction, either `DM_X`, `DM_Y`, or `DM_Z`
-  gp - global grid point number in this direction

   Output Parameters:
+  newvec - the new vector that can hold the values (size zero on all processes except process 0)
-  scatter - the `VecScatter` that will map from the original vector to the slice

   Level: advanced

   Note:
   All processors that share the `DMDA` must call this with the same gp value

.seealso: `DM`, `DMDA`, `DMDirection`, `Vec`, `VecScatter`
@*/
PetscErrorCode DMDAGetRay(DM da, DMDirection dir, PetscInt gp, Vec *newvec, VecScatter *scatter)
{
  PetscMPIInt rank;
  DM_DA      *dd = (DM_DA *)da->data;
  IS          is;
  AO          ao;
  Vec         vec;
  PetscInt   *indices, i, j;

  PetscFunctionBegin;
  PetscCheck(da->dim != 3, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot get slice from 3d DMDA");
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)da), &rank));
  PetscCall(DMDAGetAO(da, &ao));
  if (rank == 0) {
    if (da->dim == 1) {
      if (dir == DM_X) {
        PetscCall(PetscMalloc1(dd->w, &indices));
        indices[0] = dd->w * gp;
        for (i = 1; i < dd->w; ++i) indices[i] = indices[i - 1] + 1;
        PetscCall(AOApplicationToPetsc(ao, dd->w, indices));
        PetscCall(VecCreate(PETSC_COMM_SELF, newvec));
        PetscCall(VecSetBlockSize(*newvec, dd->w));
        PetscCall(VecSetSizes(*newvec, dd->w, PETSC_DETERMINE));
        PetscCall(VecSetType(*newvec, VECSEQ));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, dd->w, indices, PETSC_OWN_POINTER, &is));
      } else {
        PetscCheck(dir != DM_Y, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot get Y slice from 1d DMDA");
        SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unknown DMDirection");
      }
    } else {
      if (dir == DM_Y) {
        PetscCall(PetscMalloc1(dd->w * dd->M, &indices));
        indices[0] = gp * dd->M * dd->w;
        for (i = 1; i < dd->M * dd->w; i++) indices[i] = indices[i - 1] + 1;

        PetscCall(AOApplicationToPetsc(ao, dd->M * dd->w, indices));
        PetscCall(VecCreate(PETSC_COMM_SELF, newvec));
        PetscCall(VecSetBlockSize(*newvec, dd->w));
        PetscCall(VecSetSizes(*newvec, dd->M * dd->w, PETSC_DETERMINE));
        PetscCall(VecSetType(*newvec, VECSEQ));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, dd->w * dd->M, indices, PETSC_OWN_POINTER, &is));
      } else if (dir == DM_X) {
        PetscCall(PetscMalloc1(dd->w * dd->N, &indices));
        indices[0] = dd->w * gp;
        for (j = 1; j < dd->w; j++) indices[j] = indices[j - 1] + 1;
        for (i = 1; i < dd->N; i++) {
          indices[i * dd->w] = indices[i * dd->w - 1] + dd->w * dd->M - dd->w + 1;
          for (j = 1; j < dd->w; j++) indices[i * dd->w + j] = indices[i * dd->w + j - 1] + 1;
        }
        PetscCall(AOApplicationToPetsc(ao, dd->w * dd->N, indices));
        PetscCall(VecCreate(PETSC_COMM_SELF, newvec));
        PetscCall(VecSetBlockSize(*newvec, dd->w));
        PetscCall(VecSetSizes(*newvec, dd->N * dd->w, PETSC_DETERMINE));
        PetscCall(VecSetType(*newvec, VECSEQ));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, dd->w * dd->N, indices, PETSC_OWN_POINTER, &is));
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown DMDirection");
    }
  } else {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 0, newvec));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_COPY_VALUES, &is));
  }
  PetscCall(DMGetGlobalVector(da, &vec));
  PetscCall(VecScatterCreate(vec, is, *newvec, NULL, scatter));
  PetscCall(DMRestoreGlobalVector(da, &vec));
  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetProcessorSubset - Returns a communicator consisting only of the
   processors in a `DMDA` that own a particular global x, y, or z grid point
   (corresponding to a logical plane in a 3D grid or a line in a 2D grid).

   Collective on da

   Input Parameters:
+  da - the distributed array
.  dir - Cartesian direction, either `DM_X`, `DM_Y`, or `DM_Z`
-  gp - global grid point number in this direction

   Output Parameter:
.  comm - new communicator

   Level: advanced

   Notes:
   All processors that share the `DMDA` must call this with the same gp value

   After use, comm should be freed with `MPI_Comm_free()`

   This routine is particularly useful to compute boundary conditions
   or other application-specific calculations that require manipulating
   sets of data throughout a logical plane of grid points.

   Fortran Note:
   Not supported from Fortran

.seealso: `DM`, `DMDA`, `DMDirection`
@*/
PetscErrorCode DMDAGetProcessorSubset(DM da, DMDirection dir, PetscInt gp, MPI_Comm *comm)
{
  MPI_Group   group, subgroup;
  PetscInt    i, ict, flag, *owners, xs, xm, ys, ym, zs, zm;
  PetscMPIInt size, *ranks = NULL;
  DM_DA      *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  flag = 0;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
  if (dir == DM_Z) {
    PetscCheck(da->dim >= 3, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "DM_Z invalid for DMDA dim < 3");
    PetscCheck(gp >= 0 && gp <= dd->P, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid grid point");
    if (gp >= zs && gp < zs + zm) flag = 1;
  } else if (dir == DM_Y) {
    PetscCheck(da->dim != 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "DM_Y invalid for DMDA dim = 1");
    PetscCheck(gp >= 0 && gp <= dd->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid grid point");
    if (gp >= ys && gp < ys + ym) flag = 1;
  } else if (dir == DM_X) {
    PetscCheck(gp >= 0 && gp <= dd->M, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "invalid grid point");
    if (gp >= xs && gp < xs + xm) flag = 1;
  } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");

  PetscCall(PetscMalloc2(size, &owners, size, &ranks));
  PetscCallMPI(MPI_Allgather(&flag, 1, MPIU_INT, owners, 1, MPIU_INT, PetscObjectComm((PetscObject)da)));
  ict = 0;
  PetscCall(PetscInfo(da, "DMDAGetProcessorSubset: dim=%" PetscInt_FMT ", direction=%d, procs: ", da->dim, (int)dir));
  for (i = 0; i < size; i++) {
    if (owners[i]) {
      ranks[ict] = i;
      ict++;
      PetscCall(PetscInfo(da, "%" PetscInt_FMT " ", i));
    }
  }
  PetscCall(PetscInfo(da, "\n"));
  PetscCallMPI(MPI_Comm_group(PetscObjectComm((PetscObject)da), &group));
  PetscCallMPI(MPI_Group_incl(group, ict, ranks, &subgroup));
  PetscCallMPI(MPI_Comm_create(PetscObjectComm((PetscObject)da), subgroup, comm));
  PetscCallMPI(MPI_Group_free(&subgroup));
  PetscCallMPI(MPI_Group_free(&group));
  PetscCall(PetscFree2(owners, ranks));
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetProcessorSubsets - Returns communicators consisting only of the
   processors in a `DMDA` adjacent in a particular dimension,
   corresponding to a logical plane in a 3D grid or a line in a 2D grid.

   Collective on da

   Input Parameters:
+  da - the distributed array
-  dir - Cartesian direction, either `DM_X`, `DM_Y`, or `DM_Z`

   Output Parameter:
.  subcomm - new communicator

   Level: advanced

   Notes:
   This routine is useful for distributing one-dimensional data in a tensor product grid.

   After use, comm should be freed with` MPI_Comm_free()`

   Fortran Note:
   Not supported from Fortran

.seealso: `DM`, `DMDA`, `DMDirection`
@*/
PetscErrorCode DMDAGetProcessorSubsets(DM da, DMDirection dir, MPI_Comm *subcomm)
{
  MPI_Comm    comm;
  MPI_Group   group, subgroup;
  PetscInt    subgroupSize = 0;
  PetscInt   *firstPoints;
  PetscMPIInt size, *subgroupRanks = NULL;
  PetscInt    xs, xm, ys, ym, zs, zm, firstPoint, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (dir == DM_Z) {
    PetscCheck(da->dim >= 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "DM_Z invalid for DMDA dim < 3");
    firstPoint = zs;
  } else if (dir == DM_Y) {
    PetscCheck(da->dim != 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "DM_Y invalid for DMDA dim = 1");
    firstPoint = ys;
  } else if (dir == DM_X) {
    firstPoint = xs;
  } else SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");

  PetscCall(PetscMalloc2(size, &firstPoints, size, &subgroupRanks));
  PetscCallMPI(MPI_Allgather(&firstPoint, 1, MPIU_INT, firstPoints, 1, MPIU_INT, comm));
  PetscCall(PetscInfo(da, "DMDAGetProcessorSubset: dim=%" PetscInt_FMT ", direction=%d, procs: ", da->dim, (int)dir));
  for (p = 0; p < size; ++p) {
    if (firstPoints[p] == firstPoint) {
      subgroupRanks[subgroupSize++] = p;
      PetscCall(PetscInfo(da, "%" PetscInt_FMT " ", p));
    }
  }
  PetscCall(PetscInfo(da, "\n"));
  PetscCallMPI(MPI_Comm_group(comm, &group));
  PetscCallMPI(MPI_Group_incl(group, subgroupSize, subgroupRanks, &subgroup));
  PetscCallMPI(MPI_Comm_create(comm, subgroup, subcomm));
  PetscCallMPI(MPI_Group_free(&subgroup));
  PetscCallMPI(MPI_Group_free(&group));
  PetscCall(PetscFree2(firstPoints, subgroupRanks));
  PetscFunctionReturn(0);
}
