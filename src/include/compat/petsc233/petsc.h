#ifndef _PETSC_COMPAT_H
#define _PETSC_COMPAT_H

/* attribute recognised by some compilers to avoid 'unused' warnings */
#if !defined(PETSC_UNUSED)
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define PETSC_UNUSED __attribute__ ((__unused__)) 
#   else
#     define PETSC_UNUSED
#   endif
# elif defined(__ICC)
#   define PETSC_UNUSED __attribute__ ((__unused__)) 
# else
#   define PETSC_UNUSED 
# endif
#endif

typedef struct _p_PetscObject _p_PetscObject;

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsInsertFile_233"
static PETSC_UNUSED
PetscErrorCode PetscOptionsInsertFile_233(MPI_Comm comm,const char file[],PetscTruth require)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsInsertFile(file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscOptionsInsertFile PetscOptionsInsertFile_233


#define PetscLogEvent                PetscEvent
#define PetscLogEventRegister(n,c,e) PetscLogEventRegister(e,n,c)
#define PetscLogStageRegister(n,s)   PetscLogStageRegister(s,n)
#define PetscCookieRegister(n,c)     PetscCookieRegister(c)

#endif /* _PETSC_COMPAT_H */
