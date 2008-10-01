#ifndef _PETSC_COMPAT_VEC_H
#define _PETSC_COMPAT_VEC_H

#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISGetIndices_233(IS is, const PetscInt *ptr[])
{
  PetscInt *idx = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(ptr,2);
  ierr = ISGetIndices(is,&idx);CHKERRQ(ierr);
  *ptr = idx;
  PetscFunctionReturn(0);
  
}
#define ISGetIndices ISGetIndices_233

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreIndices_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISRestoreIndices_233(IS is, const PetscInt *ptr[])
{
  PetscInt *idx = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(ptr,2);
  idx = (PetscInt *) (*ptr);
  ierr = ISRestoreIndices(is,&idx);CHKERRQ(ierr);
  *ptr = idx;
  PetscFunctionReturn(0);
}
#define ISRestoreIndices ISRestoreIndices_233

#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetIndices_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISBlockGetIndices_233(IS is, const PetscInt *ptr[])
{
  PetscInt *idx = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(ptr,2);
  ierr = ISBlockGetIndices(is,&idx);CHKERRQ(ierr);
  *ptr = idx;
  PetscFunctionReturn(0);
  
}
#define ISBlockGetIndices ISBlockGetIndices_233

#undef __FUNCT__  
#define __FUNCT__ "ISBlockRestoreIndices_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISBlockRestoreIndices_233(IS is, const PetscInt *ptr[])
{
  PetscInt *idx = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(ptr,2);
  idx = (PetscInt *) (*ptr);
  ierr = ISBlockRestoreIndices(is,&idx);CHKERRQ(ierr);
  *ptr = idx;
  PetscFunctionReturn(0);
}
#define ISBlockRestoreIndices ISBlockRestoreIndices_233


#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetSize_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISBlockGetSize_233(IS is, PetscInt *size)
{
  PetscInt N, bs=1;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  ierr = ISBlockGetBlockSize(is,&bs);CHKERRQ(ierr);
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  *size = N/bs;
  PetscFunctionReturn(0);
}
#define ISBlockGetSize ISBlockGetSize_233

#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetLocalSize_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISBlockGetLocalSize_233(IS is, PetscInt *size)
{
  PetscInt n, bs=1;
  PetscErrorCode ierr;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  PetscFunctionBegin;
  ierr = ISBlockGetBlockSize(is,&bs);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  *size = n/bs;
  PetscFunctionReturn(0);
}
#define ISBlockGetLocalSize ISBlockGetLocalSize_233

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingApply_233"
PETSC_STATIC_INLINE PetscErrorCode 
ISLocalToGlobalMappingApply_233(ISLocalToGlobalMapping mapping,
				PetscInt N,const PetscInt in[],PetscInt out[])
{
  PetscInt i=0, *idx=0, Nmax=0;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,1);
  if (N > 0) { PetscValidPointer(in,3);PetscValidPointer(out,3); }
  idx = mapping->indices, Nmax = mapping->n;
  for (i=0; i<N; i++) {
    if (in[i] < 0) {out[i] = in[i]; continue;}
    if (in[i] >= Nmax) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,
				"Local index %D too large %D (max) at %D",
				in[i],Nmax,i);
    out[i] = idx[in[i]];
  }
  PetscFunctionReturn(0);
}
#undef  ISLocalToGlobalMappingApply
#define ISLocalToGlobalMappingApply ISLocalToGlobalMappingApply_233


#undef __FUNCT__
#define __FUNCT__ "VecSetOption_233"
static PETSC_UNUSED
PetscErrorCode VecSetOption_233(Vec x,VecOption op,PetscTruth flag) {
  if (op==VEC_IGNORE_OFF_PROC_ENTRIES && flag==PETSC_FALSE) 
    op = VEC_TREAT_OFF_PROC_ENTRIES;
  else if (op==VEC_IGNORE_NEGATIVE_INDICES && flag==PETSC_FALSE)
    op = VEC_TREAT_NEGATIVE_INDICES;
  return VecSetOption(x,op);
}
#define VecSetOption VecSetOption_233

#undef __FUNCT__
#define __FUNCT__ "VecGetOwnershipRanges_233"
static PETSC_UNUSED
PetscErrorCode VecGetOwnershipRanges_233(Vec vec,const PetscInt *ranges[]) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  PetscValidPointer(ranges,2);
  ierr = PetscMapGetGlobalRange(&vec->map,ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define VecGetOwnershipRanges VecGetOwnershipRanges_233

#endif /* _PETSC_COMPAT_VEC_H */
