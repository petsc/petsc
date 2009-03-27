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

#if defined(PETSC_HAVE_MPIUNI)
#if !defined(MPI_Finalized)
static PETSC_UNUSED
int Petsc_MPI_Finalized_232(int *flag)
{
  if (flag) *flag = 0;
  return 0;
}
#define MPI_Finalized Petsc_MPI_Finalized_232
#endif
#endif

#define PetscOptionsMonitorSet PetscOptionsSetMonitor
#define PetscOptionsMonitorCancel PetscOptionsClearMonitor

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsInsertFile_232"
static PETSC_UNUSED
PetscErrorCode PetscOptionsInsertFile_232(MPI_Comm comm,const char file[],PetscTruth require)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsInsertFile(file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscOptionsInsertFile PetscOptionsInsertFile_232


static PETSC_UNUSED FILE *PETSC_STDERR = 0;

PETSC_EXTERN_CXX_BEGIN
EXTERN PETSC_DLLEXPORT PetscCookie CONTAINER_COOKIE;
PETSC_EXTERN_CXX_END
#define PETSC_CONTAINER_COOKIE        CONTAINER_COOKIE
#define PetscContainer                PetscObjectContainer
#define PetscContainerGetPointer      PetscObjectContainerGetPointer
#define PetscContainerSetPointer      PetscObjectContainerSetPointer
#define PetscContainerDestroy         PetscObjectContainerDestroy
#define PetscContainerCreate          PetscObjectContainerCreate
#define PetscContainerSetUserDestroy  PetscObjectContainerSetUserDestroy

#define PetscLogStage                int
#define PetscLogEvent                PetscEvent
#define PetscLogStageRegister(n,s)   PetscLogStageRegister(s,n)
#define PetscLogEventRegister(n,c,e) PetscLogEventRegister(e,n,c)
#define PetscCookieRegister(n,c)     PetscCookieRegister(c)

#endif /* _PETSC_COMPAT_H */
