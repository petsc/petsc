#ifndef _COMPAT_PETSC_VEC_H
#define _COMPAT_PETSC_VEC_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define VecSqrtAbs VecSqrt
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#if  PETSC_VERSION_(2,3,2)
#define VEC_IGNORE_NEGATIVE_INDICES ((VecOption)2)
#endif
#undef __FUNCT__
#define __FUNCT__ "VecSetOption"
static PETSC_UNUSED
PetscErrorCode VecSetOption_Compat(Vec x,VecOption op,PetscTruth flag) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (op==VEC_IGNORE_OFF_PROC_ENTRIES && flag==PETSC_FALSE)
    op = VEC_TREAT_OFF_PROC_ENTRIES;
#if   PETSC_VERSION_(2,3,3)
  else if (op==VEC_IGNORE_NEGATIVE_INDICES && flag==PETSC_FALSE)
    op = VEC_TREAT_NEGATIVE_INDICES;
#elif PETSC_VERSION_(2,3,2)
  else if (op==VEC_IGNORE_NEGATIVE_INDICES) {
    SETERRQ(PETSC_ERR_SUP,"VEC_IGNORE_NEGATIVE_INDICES not supported");
    PetscFunctionReturn(PETSC_ERR_SUP);
  }
#endif
  ierr = VecSetOption(x,op); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define VecSetOption VecSetOption_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "VecGetOwnershipRanges"
static PETSC_UNUSED
PetscErrorCode VecGetOwnershipRanges(Vec vec,const PetscInt *ranges[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  PetscValidPointer(ranges,2);
#if   PETSC_VERSION_(2,3,3)
  ierr = PetscMapGetGlobalRange(&vec->map,ranges);CHKERRQ(ierr);
#elif PETSC_VERSION_(2,3,2)
  ierr = PetscMapGetGlobalRange(&vec->map,(PetscInt**)ranges);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#if defined(PETSC_USE_COMPLEX)
#  if defined(PETSC_CLANGUAGE_CXX)
#    define PetscLogScalar_Compat(a) std::log(a)
#  else
#    if   defined(PETSC_USE_SINGLE)
#      define PetscLogScalar_Compat(a) clogf(a)
#    elif defined(PETSC_USE_LONG_DOUBLE)
#      define PetscLogScalar_Compat(a) clogl(a)
#    else
#      define PetscLogScalar_Compat(a) clog(a)
#    endif
#  endif
#else
#  define PetscLogScalar_Compat(a) log(a)
#endif
#define PetscLogScalar PetscLogScalar_Compat
#undef __FUNCT__
#define __FUNCT__ "VecLog"
static PETSC_UNUSED
PetscErrorCode VecLog(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_COOKIE,1);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &x);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    x[i] = PetscLogScalar(x[i]);
  }
  ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "VecExp"
static PETSC_UNUSED
PetscErrorCode VecExp(Vec v)
{
  PetscScalar    *x;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_COOKIE,1);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &x);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    x[i] = PetscExpScalar(x[i]);
  }
  ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(2,3,2))
#define VecStrideScale(v,start,scale) VecStrideScale((v),(start),(&scale))
#define VecScatterBegin(ctx,x,y,im,sm) VecScatterBegin((x),(y),(im),(sm),(ctx))
#define VecScatterEnd(ctx,x,y,im,sm) VecScatterEnd((x),(y),(im),(sm),(ctx))
#endif

#endif /* _COMPAT_PETSC_VEC_H */
