#ifndef _PETSC_COMPAT_VEC_H
#define _PETSC_COMPAT_VEC_H

#undef __FUNCT__  
#define __FUNCT__ "ISSum_232"
static PETSC_UNUSED
PetscErrorCode ISSum_232(IS is1,IS is2,IS *is3) {
  PetscTruth     f;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_COOKIE,1);
  PetscValidHeaderSpecific(is2,IS_COOKIE,2);
  PetscValidPointer(is3, 3);
  ierr = ISSorted(is1,&f); CHKERRQ(ierr);
  if (!f) SETERRQ(PETSC_ERR_ARG_INCOMP,"Arg 1 is not sorted");
  ierr = ISSorted(is2,&f); CHKERRQ(ierr);
  if (!f) SETERRQ(PETSC_ERR_ARG_INCOMP,"Arg 2 is not sorted");
  ierr = ISDuplicate(is1,is3); CHKERRQ(ierr);
  ierr = ISSum(is3,is2); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define ISSum ISSum_232

#undef __FUNCT__  
#define __FUNCT__ "ISLocalToGlobalMappingApply_232"
PETSC_STATIC_INLINE PetscErrorCode 
ISLocalToGlobalMappingApply_232(ISLocalToGlobalMapping mapping,
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
#define ISLocalToGlobalMappingApply ISLocalToGlobalMappingApply_232

#define VEC_IGNORE_NEGATIVE_INDICES ((VecOption)1)

#undef __FUNCT__  
#define __FUNCT__ "VecSetOption_232"
static PETSC_UNUSED
PetscErrorCode VecSetOption_232(Vec x,VecOption op,PetscTruth flag) {
  if (op==VEC_IGNORE_OFF_PROC_ENTRIES && flag==PETSC_FALSE) 
    op = VEC_TREAT_OFF_PROC_ENTRIES;
  else 
    return 0;
  return VecSetOption(x,op);
}
#define VecSetOption VecSetOption_232

#undef __FUNCT__
#define __FUNCT__ "VecGetOwnershipRanges_232"
static PETSC_UNUSED
PetscErrorCode VecGetOwnershipRanges_232(Vec vec,const PetscInt *ranges[]) 
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


#define VecStrideScale(v,start,scale) VecStrideScale((v),(start),(&scale))

#define VecScatterBegin(ctx,x,y,im,sm) VecScatterBegin((x),(y),(im),(sm),(ctx))
#define VecScatterEnd(ctx,x,y,im,sm) VecScatterEnd((x),(y),(im),(sm),(ctx))

#endif /* _PETSC_COMPAT_VEC_H */
