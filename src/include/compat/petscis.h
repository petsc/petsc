#ifndef _COMPAT_PETSC_IS_H
#define _COMPAT_PETSC_IS_H

#if PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "ISCopy"
static PetscErrorCode ISCopy_Compat(IS isx, IS isy)
{
  PetscInt n,nx,ny;
  const PetscInt *ix,*iy;
  PetscTruth equal;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(isx,IS_COOKIE,1);
  PetscValidHeaderSpecific(isy,IS_COOKIE,1);
  ierr = ISGetLocalSize(isx,&nx);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isy,&ny);CHKERRQ(ierr);
  ierr = ISGetIndices(isx,&ix);CHKERRQ(ierr);
  ierr = ISGetIndices(isy,&iy);CHKERRQ(ierr);
  n = PetscMin(nx,ny);
  ierr = PetscMemcmp(ix,iy,n*sizeof(PetscInt),&equal);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(isx,&ix);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isy,&iy);CHKERRQ(ierr);
  if (nx == ny && equal) PetscFunctionReturn(0);
  SETERRQ(PETSC_ERR_SUP, __FUNCT__"() not supported");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#define ISCopy ISCopy_Compat
#endif

#endif /* _COMPAT_PETSC_IS_H */
