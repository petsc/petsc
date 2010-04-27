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
PetscErrorCode PETSCDM_DLLEXPORT DAGetProcessorSubset(DA da,DADirection dir,PetscInt gp,MPI_Comm *comm)
{
  MPI_Group      group,subgroup;
  PetscErrorCode ierr;
  PetscInt       i,ict,flag,*owners,xs,xm,ys,ym,zs,zm;
  PetscMPIInt    size,*ranks = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  flag = 0; 
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)da)->comm,&size);CHKERRQ(ierr);
  if (dir == DA_Z) {
    if (da->dim < 3) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"DA_Z invalid for DA dim < 3");
    if (gp < 0 || gp > da->P) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= zs && gp < zs+zm) flag = 1;
  } else if (dir == DA_Y) {
    if (da->dim == 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"DA_Y invalid for DA dim = 1");
    if (gp < 0 || gp > da->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= ys && gp < ys+ym) flag = 1;
  } else if (dir == DA_X) {
    if (gp < 0 || gp > da->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"invalid grid point");
    if (gp >= xs && gp < xs+xm) flag = 1;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid direction");

  ierr = PetscMalloc2(size,PetscInt,&owners,size,PetscMPIInt,&ranks);CHKERRQ(ierr);
  ierr = MPI_Allgather(&flag,1,MPIU_INT,owners,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
  ict  = 0;
  ierr = PetscInfo2(da,"DAGetProcessorSubset: dim=%D, direction=%d, procs: ",da->dim,(int)dir);CHKERRQ(ierr);
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
