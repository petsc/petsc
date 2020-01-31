/* DMStag dimension-independent internal functions. If added to the public API,
   these would move to stagutils.c */

#include <petsc/private/dmstagimpl.h>

/* Populate data created after DMCreate_Stag() is called, which is used by DMSetUp_Stag(),
   such as the grid dimensions and dof information. Arguments are ignored for dimensions
   less than three. */
PetscErrorCode DMStagInitialize(DMBoundaryType bndx,DMBoundaryType bndy,DMBoundaryType bndz,PetscInt M,PetscInt N,PetscInt P,PetscInt m,PetscInt n,PetscInt p,PetscInt dof0,PetscInt dof1,PetscInt dof2,PetscInt dof3,DMStagStencilType stencilType,PetscInt stencilWidth,const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetType(dm,DMSTAG);CHKERRQ(ierr);
  ierr = DMStagSetBoundaryTypes(dm,bndx,bndy,bndz);CHKERRQ(ierr);
  ierr = DMStagSetGlobalSizes(dm,M,N,P);CHKERRQ(ierr);
  ierr = DMStagSetNumRanks(dm,m,n,p);CHKERRQ(ierr);
  ierr = DMStagSetStencilType(dm,stencilType);CHKERRQ(ierr);
  ierr = DMStagSetStencilWidth(dm,stencilWidth);CHKERRQ(ierr);
  ierr = DMStagSetDOF(dm,dof0,dof1,dof2,dof3);CHKERRQ(ierr);
  ierr = DMStagSetOwnershipRanges(dm,lx,ly,lz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
