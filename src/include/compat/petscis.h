#ifndef _COMPAT_PETSC_IS_H
#define _COMPAT_PETSC_IS_H

#if PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "ISComplement"
static PetscErrorCode ISComplement_Compat(IS is,PetscInt nmin,PetscInt nmax,IS *isout)
{
  PetscErrorCode ierr;
  const PetscInt *indices;
  PetscInt       n,i,j,unique,cnt,*nindices;
  PetscTruth     sorted;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(isout,3);
  if (nmin < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nmin %D cannot be negative",nmin);
  if (nmin > nmax) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nmin %D cannot be greater than nmax %D",nmin,nmax);
  ierr = ISSorted(is,&sorted);CHKERRQ(ierr);
  if (!sorted) SETERRQ(PETSC_ERR_ARG_WRONG,"Index set must be sorted");

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  for (i=0; i<n; i++) {
    if (indices[i] <  nmin) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Index %D's value %D is smaller than minimum given %D",i,indices[i],nmin);
    if (indices[i] >= nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"Index %D's value %D is larger than maximum given %D",i,indices[i],nmax);
  }
#endif
  /* Count number of unique entries */
  unique = (n>0);
  for (i=0; i<n-1; i++) {
    if (indices[i+1] != indices[i]) unique++;
  }
  ierr = PetscMalloc((nmax-nmin-unique)*sizeof(PetscInt),&nindices);CHKERRQ(ierr);
  cnt = 0;
  for (i=nmin,j=0; i<nmax; i++) {
    if (j<n && i==indices[j]) do { j++; } while (j<n && i==indices[j]);
    else nindices[cnt++] = i;
  }
  if (cnt != nmax-nmin-unique) SETERRQ2(PETSC_ERR_PLIB,"Number of entries found in complement %D does not match expected %D",cnt,nmax-nmin-unique);
  ierr = ISCreateGeneral(((PetscObject)is)->comm,cnt,nindices,isout);CHKERRQ(ierr);
  ierr = PetscFree(nindices);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define ISComplement ISComplement_Compat
#endif

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
