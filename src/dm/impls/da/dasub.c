#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetProcessorSubset"
/*@C
   DAGetProcessorSubset - Returns a communicator consisting only of the
   processors in a DA that own a particular global x, y, or z grid point
   (corresponding to a logical plane in a 3D grid or a line in a 2D grid).

   Collective on DA

   Input Parameters:
+  da - the distributed array
.  dir - Cartesian direction, either DA_X, DA_Y, or DA_Z
-  gp - global grid point number in this direction

   Output Parameters:
.  comm - new communicator

   Level: advanced

   Notes:
   All processors that share the DA must call this with the same gp value

   This routine is particularly useful to compute boundary conditions
   or other application-specific calculations that require manipulating
   sets of data throughout a logical plane of grid points.

.keywords: distributed array, get, processor subset
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetProcessorSubset(DM da,DADirection dir,PetscInt gp,MPI_Comm *comm)
{
  MPI_Group      group,subgroup;
  PetscErrorCode ierr;
  PetscInt       i,ict,flag,*owners,xs,xm,ys,ym,zs,zm;
  PetscMPIInt    size,*ranks = PETSC_NULL;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  flag = 0; 
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)da)->comm,&size);CHKERRQ(ierr);
  if (dir == DA_Z) {
    if (dd->dim < 3) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"DA_Z invalid for DA dim < 3");
    if (gp < 0 || gp > dd->P) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= zs && gp < zs+zm) flag = 1;
  } else if (dir == DA_Y) {
    if (dd->dim == 1) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"DA_Y invalid for DA dim = 1");
    if (gp < 0 || gp > dd->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= ys && gp < ys+ym) flag = 1;
  } else if (dir == DA_X) {
    if (gp < 0 || gp > dd->M) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= xs && gp < xs+xm) flag = 1;
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  ierr = PetscMalloc2(size,PetscInt,&owners,size,PetscMPIInt,&ranks);CHKERRQ(ierr);
  ierr = MPI_Allgather(&flag,1,MPIU_INT,owners,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
  ict  = 0;
  ierr = PetscInfo2(da,"DAGetProcessorSubset: dim=%D, direction=%d, procs: ",dd->dim,(int)dir);CHKERRQ(ierr);
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
#define __FUNCT__ "DAGetProcessorSubsets"
/*@C
   DAGetProcessorSubsets - Returns communicators consisting only of the
   processors in a DA adjacent in a particular dimension,
   corresponding to a logical plane in a 3D grid or a line in a 2D grid.

   Collective on DA

   Input Parameters:
+  da - the distributed array
-  dir - Cartesian direction, either DA_X, DA_Y, or DA_Z

   Output Parameters:
.  subcomm - new communicator

   Level: advanced

   Notes:
   This routine is useful for distributing one-dimensional data in a tensor product grid.

.keywords: distributed array, get, processor subset
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetProcessorSubsets(DM da, DADirection dir, MPI_Comm *subcomm)
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
  ierr = DAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (dir == DA_Z) {
    if (dd->dim < 3) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"DA_Z invalid for DA dim < 3");
    firstPoint = zs;
  } else if (dir == DA_Y) {
    if (dd->dim == 1) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"DA_Y invalid for DA dim = 1");
    firstPoint = ys;
  } else if (dir == DA_X) {
    firstPoint = xs;
  } else SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  ierr = PetscMalloc2(size, PetscInt, &firstPoints, size, PetscMPIInt, &subgroupRanks);CHKERRQ(ierr);
  ierr = MPI_Allgather(&firstPoint, 1, MPIU_INT, firstPoints, 1, MPIU_INT, comm);CHKERRQ(ierr);
  ierr = PetscInfo2(da,"DAGetProcessorSubset: dim=%D, direction=%d, procs: ",dd->dim,(int)dir);CHKERRQ(ierr);
  for(p = 0; p < size; ++p) {
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
