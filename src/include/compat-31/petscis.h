#ifndef _COMPAT_PETSC_IS_H
#define _COMPAT_PETSC_IS_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))

#define ISGENERAL "general"
#define ISSTRIDE  "stride"
#define ISBLOCK   "block"

typedef enum {
  PETSC_COPY_VALUES,
  PETSC_OWN_POINTER,
  PETSC_USE_POINTER
} PetscCopyMode;

#undef __FUNCT__
#define __FUNCT__ "ISCreate"
static PetscErrorCode
ISCreate_Compat(MPI_Comm comm, IS *is)
{
  PetscFunctionBegin;
  PetscValidCharPointer(is,2);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define ISCreate ISCreate_Compat

#undef __FUNCT__
#define __FUNCT__ "ISSetType"
static PetscErrorCode
ISSetType_Compat(IS is, const char *istype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidCharPointer(istype,3);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define ISSetType ISSetType_Compat

#undef __FUNCT__
#define __FUNCT__ "ISGetType"
static PetscErrorCode
ISGetType_Compat(IS is, const char **istype)
{
  static const char* ISTypes[] = {ISGENERAL,ISSTRIDE,ISBLOCK};
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(istype,3);
  *istype = ISTypes[((PetscObject)is)->type];
  PetscFunctionReturn(0);
}
#define ISGetType ISGetType_Compat

#undef __FUNCT__
#define __FUNCT__ "ISGetBlockSize"
static PetscErrorCode
ISGetBlockSize_Compat(IS is, PetscInt *bs)
{
  PetscTruth     match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(bs,2);
  ierr = ISBlock(is,&match);CHKERRQ(ierr);
  if (match) { ierr = ISBlockGetBlockSize(is,bs);CHKERRQ(ierr); }
  else *bs = 1;
  PetscFunctionReturn(0);
}
#define ISGetBlockSize ISGetBlockSize_Compat

#undef __FUNCT__
#define __FUNCT__ "ISGeneralSetIndices"
static PetscErrorCode
ISGeneralSetIndices_Compat(IS is,PetscInt n,const PetscInt idx[],
                           PetscCopyMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  if (n) PetscValidIntPointer(idx,3);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define ISGeneralSetIndices ISGeneralSetIndices_Compat

#undef __FUNCT__
#define __FUNCT__ "ISBlockSetIndices"
static PetscErrorCode
ISBlockSetIndices_Compat(IS is,PetscInt bs, PetscInt n,const PetscInt idx[],
                         PetscCopyMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  if (n) PetscValidIntPointer(idx,3);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define ISBlockSetIndices ISBlockSetIndices_Compat

#undef __FUNCT__
#define __FUNCT__ "ISStrideSetStride"
static PetscErrorCode
ISStrideSetStride_Compat(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define ISStrideSetStride ISStrideSetStride_Compat

#undef __FUNCT__
#define __FUNCT__ "ISCreateGeneral"
static PetscErrorCode
ISCreateGeneral_Compat(MPI_Comm comm,PetscInt n,const PetscInt idx[],PetscCopyMode mode, IS *is)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch(mode) {
  case PETSC_OWN_POINTER:
    ierr = ISCreateGeneralNC(comm,n,idx,is);CHKERRQ(ierr);break;
  case PETSC_USE_POINTER:
    ierr = ISCreateGeneralWithArray(comm,n,(PetscInt*)idx,is);CHKERRQ(ierr);break;
  default:
    ierr = ISCreateGeneral(comm,n,idx,is);CHKERRQ(ierr);break;
  }
  PetscFunctionReturn(0);
}
#define ISCreateGeneral ISCreateGeneral_Compat

#undef __FUNCT__
#define __FUNCT__ "ISCreateBlock"
static PetscErrorCode
ISCreateBlock_Compat(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode, IS *is)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch(mode) {

  case PETSC_COPY_VALUES:
    ierr = ISCreateBlock(comm,bs,n,idx,is);CHKERRQ(ierr);break;
  default:
    SETERRQ(PETSC_ERR_SUP, __FUNCT__"() not supported in this PETSc version");
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  PetscFunctionReturn(0);
}
#define ISCreateBlock ISCreateBlock_Compat

#undef __FUNCT__
#define __FUNCT__ "ISToGeneral"
static PetscErrorCode
ISToGeneral_Compat(IS is)
{
  PetscTruth     match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  ierr = ISBlock(is,&match);CHKERRQ(ierr);
  if (match) {
    SETERRQ(PETSC_ERR_SUP, __FUNCT__"() not supported in this PETSc version");
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
  ierr = ISStride(is,&match);CHKERRQ(ierr);
  if (match) {
    ierr = ISStrideToGeneral(is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define ISToGeneral ISToGeneral_Compat

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingCreate"
static PetscErrorCode
ISLocalToGlobalMappingCreate_Compat(MPI_Comm comm,PetscInt n,const PetscInt idx[],PetscCopyMode mode, ISLocalToGlobalMapping *isltog)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch(mode) {
  case PETSC_OWN_POINTER:
    ierr = ISLocalToGlobalMappingCreateNC(comm,n,idx,isltog);CHKERRQ(ierr);break;
  default:
    ierr = ISLocalToGlobalMappingCreate(comm,n,idx,isltog);CHKERRQ(ierr);break;
  }
  PetscFunctionReturn(0);
}
#define ISLocalToGlobalMappingCreate ISLocalToGlobalMappingCreate_Compat

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingCreate"
static PetscErrorCode
ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping ltog, 
                                 const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_COOKIE,1);
  PetscValidPointer(array,2);
  *array = ltog->indices;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISLocalToGlobalMappingRestoreIndices"
static PetscErrorCode
ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping ltog,
                                     const PetscInt **array)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_COOKIE,1);
  PetscValidPointer(array,2);
  if (*array != ltog->indices) 
    SETERRQ(PETSC_ERR_ARG_BADPTR,
            "Trying to return mismatched pointer");
  *array = PETSC_NULL;
  PetscFunctionReturn(0);
}

#endif

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
  ierr = ISCreateGeneralNC(((PetscObject)is)->comm,cnt,nindices,isout);CHKERRQ(ierr);
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
