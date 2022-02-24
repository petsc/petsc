
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

/*@
   DMDAGetLogicalCoordinate - Returns a the i,j,k logical coordinate for the closest mesh point to a x,y,z point in the coordinates of the DMDA

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

   Notes:
   All processors that share the DMDA must call this with the same coordinate value

@*/
PetscErrorCode  DMDAGetLogicalCoordinate(DM da,PetscScalar x,PetscScalar y,PetscScalar z,PetscInt *II,PetscInt *JJ,PetscInt *KK,PetscScalar *X,PetscScalar *Y,PetscScalar *Z)
{
  Vec            coors;
  DM             dacoors;
  DMDACoor2d     **c;
  PetscInt       i,j,xs,xm,ys,ym;
  PetscReal      d,D = PETSC_MAX_REAL,Dv;
  PetscMPIInt    rank,root;

  PetscFunctionBegin;
  PetscCheckFalse(da->dim == 1,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Cannot get point from 1d DMDA");
  PetscCheckFalse(da->dim == 3,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Cannot get point from 3d DMDA");

  *II = -1;
  *JJ = -1;

  CHKERRQ(DMGetCoordinateDM(da,&dacoors));
  CHKERRQ(DMDAGetCorners(dacoors,&xs,&ys,NULL,&xm,&ym,NULL));
  CHKERRQ(DMGetCoordinates(da,&coors));
  CHKERRQ(DMDAVecGetArrayRead(dacoors,coors,&c));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      d = PetscSqrtReal(PetscRealPart((c[j][i].x - x)*(c[j][i].x - x) + (c[j][i].y - y)*(c[j][i].y - y)));
      if (d < D) {
        D   = d;
        *II = i;
        *JJ = j;
      }
    }
  }
  CHKERRMPI(MPIU_Allreduce(&D,&Dv,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)da)));
  if (D != Dv) {
    *II  = -1;
    *JJ  = -1;
    rank = 0;
  } else {
    *X = c[*JJ][*II].x;
    *Y = c[*JJ][*II].y;
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)da),&rank));
    rank++;
  }
  CHKERRMPI(MPIU_Allreduce(&rank,&root,1,MPI_INT,MPI_SUM,PetscObjectComm((PetscObject)da)));
  root--;
  CHKERRMPI(MPI_Bcast(X,1,MPIU_SCALAR,root,PetscObjectComm((PetscObject)da)));
  CHKERRMPI(MPI_Bcast(Y,1,MPIU_SCALAR,root,PetscObjectComm((PetscObject)da)));
  CHKERRQ(DMDAVecRestoreArrayRead(dacoors,coors,&c));
  PetscFunctionReturn(0);
}

/*@
   DMDAGetRay - Returns a vector on process zero that contains a row or column of the values in a DMDA vector

   Collective on DMDA

   Input Parameters:
+  da - the distributed array
.  dir - Cartesian direction, either DM_X, DM_Y, or DM_Z
-  gp - global grid point number in this direction

   Output Parameters:
+  newvec - the new vector that can hold the values (size zero on all processes except process 0)
-  scatter - the VecScatter that will map from the original vector to the slice

   Level: advanced

   Notes:
   All processors that share the DMDA must call this with the same gp value

@*/
PetscErrorCode  DMDAGetRay(DM da,DMDirection dir,PetscInt gp,Vec *newvec,VecScatter *scatter)
{
  PetscMPIInt    rank;
  DM_DA          *dd = (DM_DA*)da->data;
  IS             is;
  AO             ao;
  Vec            vec;
  PetscInt       *indices,i,j;

  PetscFunctionBegin;
  PetscCheckFalse(da->dim == 3,PetscObjectComm((PetscObject) da), PETSC_ERR_SUP, "Cannot get slice from 3d DMDA");
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) da), &rank));
  CHKERRQ(DMDAGetAO(da, &ao));
  if (rank == 0) {
    if (da->dim == 1) {
      if (dir == DM_X) {
        CHKERRQ(PetscMalloc1(dd->w, &indices));
        indices[0] = dd->w*gp;
        for (i = 1; i < dd->w; ++i) indices[i] = indices[i-1] + 1;
        CHKERRQ(AOApplicationToPetsc(ao, dd->w, indices));
        CHKERRQ(VecCreate(PETSC_COMM_SELF, newvec));
        CHKERRQ(VecSetBlockSize(*newvec, dd->w));
        CHKERRQ(VecSetSizes(*newvec, dd->w, PETSC_DETERMINE));
        CHKERRQ(VecSetType(*newvec, VECSEQ));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, dd->w, indices, PETSC_OWN_POINTER, &is));
      } else PetscCheckFalse(dir == DM_Y,PetscObjectComm((PetscObject) da), PETSC_ERR_SUP, "Cannot get Y slice from 1d DMDA");
      else SETERRQ(PetscObjectComm((PetscObject) da), PETSC_ERR_ARG_OUTOFRANGE, "Unknown DMDirection");
    } else {
      if (dir == DM_Y) {
        CHKERRQ(PetscMalloc1(dd->w*dd->M,&indices));
        indices[0] = gp*dd->M*dd->w;
        for (i=1; i<dd->M*dd->w; i++) indices[i] = indices[i-1] + 1;

        CHKERRQ(AOApplicationToPetsc(ao,dd->M*dd->w,indices));
        CHKERRQ(VecCreate(PETSC_COMM_SELF,newvec));
        CHKERRQ(VecSetBlockSize(*newvec,dd->w));
        CHKERRQ(VecSetSizes(*newvec,dd->M*dd->w,PETSC_DETERMINE));
        CHKERRQ(VecSetType(*newvec,VECSEQ));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,dd->w*dd->M,indices,PETSC_OWN_POINTER,&is));
      } else if (dir == DM_X) {
        CHKERRQ(PetscMalloc1(dd->w*dd->N,&indices));
        indices[0] = dd->w*gp;
        for (j=1; j<dd->w; j++) indices[j] = indices[j-1] + 1;
        for (i=1; i<dd->N; i++) {
          indices[i*dd->w] = indices[i*dd->w-1] + dd->w*dd->M - dd->w + 1;
          for (j=1; j<dd->w; j++) indices[i*dd->w + j] = indices[i*dd->w + j - 1] + 1;
        }
        CHKERRQ(AOApplicationToPetsc(ao,dd->w*dd->N,indices));
        CHKERRQ(VecCreate(PETSC_COMM_SELF,newvec));
        CHKERRQ(VecSetBlockSize(*newvec,dd->w));
        CHKERRQ(VecSetSizes(*newvec,dd->N*dd->w,PETSC_DETERMINE));
        CHKERRQ(VecSetType(*newvec,VECSEQ));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,dd->w*dd->N,indices,PETSC_OWN_POINTER,&is));
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown DMDirection");
    }
  } else {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, 0, newvec));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_COPY_VALUES, &is));
  }
  CHKERRQ(DMGetGlobalVector(da, &vec));
  CHKERRQ(VecScatterCreate(vec, is, *newvec, NULL, scatter));
  CHKERRQ(DMRestoreGlobalVector(da, &vec));
  CHKERRQ(ISDestroy(&is));
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetProcessorSubset - Returns a communicator consisting only of the
   processors in a DMDA that own a particular global x, y, or z grid point
   (corresponding to a logical plane in a 3D grid or a line in a 2D grid).

   Collective on da

   Input Parameters:
+  da - the distributed array
.  dir - Cartesian direction, either DM_X, DM_Y, or DM_Z
-  gp - global grid point number in this direction

   Output Parameter:
.  comm - new communicator

   Level: advanced

   Notes:
   All processors that share the DMDA must call this with the same gp value

   After use, comm should be freed with MPI_Comm_free()

   This routine is particularly useful to compute boundary conditions
   or other application-specific calculations that require manipulating
   sets of data throughout a logical plane of grid points.

   Not supported from Fortran

@*/
PetscErrorCode  DMDAGetProcessorSubset(DM da,DMDirection dir,PetscInt gp,MPI_Comm *comm)
{
  MPI_Group      group,subgroup;
  PetscInt       i,ict,flag,*owners,xs,xm,ys,ym,zs,zm;
  PetscMPIInt    size,*ranks = NULL;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  flag = 0;
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da),&size));
  if (dir == DM_Z) {
    PetscCheckFalse(da->dim < 3,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"DM_Z invalid for DMDA dim < 3");
    PetscCheckFalse(gp < 0 || gp > dd->P,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= zs && gp < zs+zm) flag = 1;
  } else if (dir == DM_Y) {
    PetscCheckFalse(da->dim == 1,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"DM_Y invalid for DMDA dim = 1");
    PetscCheckFalse(gp < 0 || gp > dd->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= ys && gp < ys+ym) flag = 1;
  } else if (dir == DM_X) {
    PetscCheckFalse(gp < 0 || gp > dd->M,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= xs && gp < xs+xm) flag = 1;
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  CHKERRQ(PetscMalloc2(size,&owners,size,&ranks));
  CHKERRMPI(MPI_Allgather(&flag,1,MPIU_INT,owners,1,MPIU_INT,PetscObjectComm((PetscObject)da)));
  ict  = 0;
  CHKERRQ(PetscInfo(da,"DMDAGetProcessorSubset: dim=%D, direction=%d, procs: ",da->dim,(int)dir));
  for (i=0; i<size; i++) {
    if (owners[i]) {
      ranks[ict] = i; ict++;
      CHKERRQ(PetscInfo(da,"%D ",i));
    }
  }
  CHKERRQ(PetscInfo(da,"\n"));
  CHKERRMPI(MPI_Comm_group(PetscObjectComm((PetscObject)da),&group));
  CHKERRMPI(MPI_Group_incl(group,ict,ranks,&subgroup));
  CHKERRMPI(MPI_Comm_create(PetscObjectComm((PetscObject)da),subgroup,comm));
  CHKERRMPI(MPI_Group_free(&subgroup));
  CHKERRMPI(MPI_Group_free(&group));
  CHKERRQ(PetscFree2(owners,ranks));
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetProcessorSubsets - Returns communicators consisting only of the
   processors in a DMDA adjacent in a particular dimension,
   corresponding to a logical plane in a 3D grid or a line in a 2D grid.

   Collective on da

   Input Parameters:
+  da - the distributed array
-  dir - Cartesian direction, either DM_X, DM_Y, or DM_Z

   Output Parameter:
.  subcomm - new communicator

   Level: advanced

   Notes:
   This routine is useful for distributing one-dimensional data in a tensor product grid.

   After use, comm should be freed with MPI_Comm_free()

   Not supported from Fortran

@*/
PetscErrorCode  DMDAGetProcessorSubsets(DM da, DMDirection dir, MPI_Comm *subcomm)
{
  MPI_Comm       comm;
  MPI_Group      group, subgroup;
  PetscInt       subgroupSize = 0;
  PetscInt       *firstPoints;
  PetscMPIInt    size, *subgroupRanks = NULL;
  PetscInt       xs, xm, ys, ym, zs, zm, firstPoint, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1,DMDA);
  CHKERRQ(PetscObjectGetComm((PetscObject)da,&comm));
  CHKERRQ(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  if (dir == DM_Z) {
    PetscCheckFalse(da->dim < 3,comm,PETSC_ERR_ARG_OUTOFRANGE,"DM_Z invalid for DMDA dim < 3");
    firstPoint = zs;
  } else if (dir == DM_Y) {
    PetscCheckFalse(da->dim == 1,comm,PETSC_ERR_ARG_OUTOFRANGE,"DM_Y invalid for DMDA dim = 1");
    firstPoint = ys;
  } else if (dir == DM_X) {
    firstPoint = xs;
  } else SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  CHKERRQ(PetscMalloc2(size, &firstPoints, size, &subgroupRanks));
  CHKERRMPI(MPI_Allgather(&firstPoint, 1, MPIU_INT, firstPoints, 1, MPIU_INT, comm));
  CHKERRQ(PetscInfo(da,"DMDAGetProcessorSubset: dim=%D, direction=%d, procs: ",da->dim,(int)dir));
  for (p = 0; p < size; ++p) {
    if (firstPoints[p] == firstPoint) {
      subgroupRanks[subgroupSize++] = p;
      CHKERRQ(PetscInfo(da, "%D ", p));
    }
  }
  CHKERRQ(PetscInfo(da, "\n"));
  CHKERRMPI(MPI_Comm_group(comm, &group));
  CHKERRMPI(MPI_Group_incl(group, subgroupSize, subgroupRanks, &subgroup));
  CHKERRMPI(MPI_Comm_create(comm, subgroup, subcomm));
  CHKERRMPI(MPI_Group_free(&subgroup));
  CHKERRMPI(MPI_Group_free(&group));
  CHKERRQ(PetscFree2(firstPoints, subgroupRanks));
  PetscFunctionReturn(0);
}
