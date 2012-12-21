
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMDAGetRay"
/*@C
   DMDAGetRay - Returns a vector on process zero that contains a row or column of the values in a DMDA vector

   Collective on DMDA

   Input Parameters:
+  da - the distributed array
.  vec - the vector
.  dir - Cartesian direction, either DMDA_X, DMDA_Y, or DMDA_Z
-  gp - global grid point number in this direction

   Output Parameters:
+  newvec - the new vector that can hold the values (size zero on all processes except process 0)
-  scatter - the VecScatter that will map from the original vector to the slice

   Level: advanced

   Notes:
   All processors that share the DMDA must call this with the same gp value

.keywords: distributed array, get, processor subset
@*/
PetscErrorCode  DMDAGetRay(DM da,DMDADirection dir,PetscInt gp,Vec *newvec,VecScatter *scatter)
{
  PetscMPIInt    rank;
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;
  IS             is;
  AO             ao;
  Vec            vec;
  PetscInt       *indices,i,j;

  PetscFunctionBegin;
  if (dd->dim == 1) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Cannot get slice from 1d DMDA");
  if (dd->dim == 3) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Cannot get slice from 3d DMDA");
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)da)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (dir == DMDA_Y) {
      ierr = PetscMalloc(dd->w*dd->M*sizeof(PetscInt),&indices);CHKERRQ(ierr);
      indices[0] = gp*dd->M*dd->w;
      for (i=1; i<dd->M*dd->w; i++) {indices[i] = indices[i-1] + 1;}
      ierr = AOApplicationToPetsc(ao,dd->M*dd->w,indices);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_SELF,newvec);CHKERRQ(ierr);
      ierr = VecSetBlockSize(*newvec,dd->w);CHKERRQ(ierr);
      ierr = VecSetSizes(*newvec,dd->M*dd->w,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = VecSetType(*newvec,VECSEQ);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,dd->w*dd->M,indices,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    } else if (dir == DMDA_X) {
      ierr = PetscMalloc(dd->w*dd->N*sizeof(PetscInt),&indices);CHKERRQ(ierr);
      indices[0] = dd->w*gp;
      for (j=1; j<dd->w; j++) indices[j] = indices[j-1] + 1;
      for (i=1; i<dd->N; i++) {
         indices[i*dd->w] = indices[i*dd->w-1] + dd->w*dd->M - dd->w + 1;
         for (j=1; j<dd->w; j++) indices[i*dd->w + j] = indices[i*dd->w + j - 1] + 1;
      }
      ierr = AOApplicationToPetsc(ao,dd->w*dd->N,indices);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_SELF,newvec);CHKERRQ(ierr);
      ierr = VecSetBlockSize(*newvec,dd->w);CHKERRQ(ierr);
      ierr = VecSetSizes(*newvec,dd->N*dd->w,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = VecSetType(*newvec,VECSEQ);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,dd->w*dd->N,indices,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown DMDADirection");
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,newvec);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,0,0,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  }
  ierr = DMGetGlobalVector(da,&vec);CHKERRQ(ierr);
  ierr = VecScatterCreate(vec,is,*newvec,PETSC_NULL,scatter);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&vec);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetProcessorSubset"
/*@C
   DMDAGetProcessorSubset - Returns a communicator consisting only of the
   processors in a DMDA that own a particular global x, y, or z grid point
   (corresponding to a logical plane in a 3D grid or a line in a 2D grid).

   Collective on DMDA

   Input Parameters:
+  da - the distributed array
.  dir - Cartesian direction, either DMDA_X, DMDA_Y, or DMDA_Z
-  gp - global grid point number in this direction

   Output Parameters:
.  comm - new communicator

   Level: advanced

   Notes:
   All processors that share the DMDA must call this with the same gp value

   This routine is particularly useful to compute boundary conditions
   or other application-specific calculations that require manipulating
   sets of data throughout a logical plane of grid points.

.keywords: distributed array, get, processor subset
@*/
PetscErrorCode  DMDAGetProcessorSubset(DM da,DMDADirection dir,PetscInt gp,MPI_Comm *comm)
{
  MPI_Group      group,subgroup;
  PetscErrorCode ierr;
  PetscInt       i,ict,flag,*owners,xs,xm,ys,ym,zs,zm;
  PetscMPIInt    size,*ranks = PETSC_NULL;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  flag = 0;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)da)->comm,&size);CHKERRQ(ierr);
  if (dir == DMDA_Z) {
    if (dd->dim < 3) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"DMDA_Z invalid for DMDA dim < 3");
    if (gp < 0 || gp > dd->P) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= zs && gp < zs+zm) flag = 1;
  } else if (dir == DMDA_Y) {
    if (dd->dim == 1) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"DMDA_Y invalid for DMDA dim = 1");
    if (gp < 0 || gp > dd->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= ys && gp < ys+ym) flag = 1;
  } else if (dir == DMDA_X) {
    if (gp < 0 || gp > dd->M) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= xs && gp < xs+xm) flag = 1;
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  ierr = PetscMalloc2(size,PetscInt,&owners,size,PetscMPIInt,&ranks);CHKERRQ(ierr);
  ierr = MPI_Allgather(&flag,1,MPIU_INT,owners,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
  ict  = 0;
  ierr = PetscInfo2(da,"DMDAGetProcessorSubset: dim=%D, direction=%d, procs: ",dd->dim,(int)dir);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    if (owners[i]) {
      ranks[ict] = i; ict++;
      ierr = PetscInfo1(da,"%D ",i);CHKERRQ(ierr);
    }
  }
  ierr = PetscInfo(da,"\n");CHKERRQ(ierr);
  ierr = MPI_Comm_group(((PetscObject)da)->comm,&group);CHKERRQ(ierr);
  ierr = MPI_Group_incl(group,ict,ranks,&subgroup);CHKERRQ(ierr);
  ierr = MPI_Comm_create(((PetscObject)da)->comm,subgroup,comm);CHKERRQ(ierr);
  ierr = MPI_Group_free(&subgroup);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  ierr = PetscFree2(owners,ranks);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetProcessorSubsets"
/*@C
   DMDAGetProcessorSubsets - Returns communicators consisting only of the
   processors in a DMDA adjacent in a particular dimension,
   corresponding to a logical plane in a 3D grid or a line in a 2D grid.

   Collective on DMDA

   Input Parameters:
+  da - the distributed array
-  dir - Cartesian direction, either DMDA_X, DMDA_Y, or DMDA_Z

   Output Parameters:
.  subcomm - new communicator

   Level: advanced

   Notes:
   This routine is useful for distributing one-dimensional data in a tensor product grid.

.keywords: distributed array, get, processor subset
@*/
PetscErrorCode  DMDAGetProcessorSubsets(DM da, DMDADirection dir, MPI_Comm *subcomm)
{
  MPI_Comm       comm;
  MPI_Group      group, subgroup;
  PetscInt       subgroupSize = 0;
  PetscInt      *firstPoints;
  PetscMPIInt    size, *subgroupRanks = PETSC_NULL;
  PetscInt       xs, xm, ys, ym, zs, zm, firstPoint, p;
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  comm = ((PetscObject) da)->comm;
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (dir == DMDA_Z) {
    if (dd->dim < 3) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"DMDA_Z invalid for DMDA dim < 3");
    firstPoint = zs;
  } else if (dir == DMDA_Y) {
    if (dd->dim == 1) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"DMDA_Y invalid for DMDA dim = 1");
    firstPoint = ys;
  } else if (dir == DMDA_X) {
    firstPoint = xs;
  } else SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  ierr = PetscMalloc2(size, PetscInt, &firstPoints, size, PetscMPIInt, &subgroupRanks);CHKERRQ(ierr);
  ierr = MPI_Allgather(&firstPoint, 1, MPIU_INT, firstPoints, 1, MPIU_INT, comm);CHKERRQ(ierr);
  ierr = PetscInfo2(da,"DMDAGetProcessorSubset: dim=%D, direction=%d, procs: ",dd->dim,(int)dir);CHKERRQ(ierr);
  for (p = 0; p < size; ++p) {
    if (firstPoints[p] == firstPoint) {
      subgroupRanks[subgroupSize++] = p;
      ierr = PetscInfo1(da, "%D ", p);CHKERRQ(ierr);
    }
  }
  ierr = PetscInfo(da, "\n");CHKERRQ(ierr);
  ierr = MPI_Comm_group(comm, &group);CHKERRQ(ierr);
  ierr = MPI_Group_incl(group, subgroupSize, subgroupRanks, &subgroup);CHKERRQ(ierr);
  ierr = MPI_Comm_create(comm, subgroup, subcomm);CHKERRQ(ierr);
  ierr = MPI_Group_free(&subgroup);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  ierr = PetscFree2(firstPoints, subgroupRanks);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
