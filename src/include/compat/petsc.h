#ifndef _COMPAT_PETSC_H
#define _COMPAT_PETSC_H

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
typedef struct _p_PetscObject _p_PetscObject;
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHasName"
static PETSC_UNUSED
PetscErrorCode PetscOptionsHasName_Compat(const char pre[],const char name[],PetscTruth *flg)
{
  char dummy[2] = { 0, 0 };
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsGetString(pre,name,dummy,1,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscOptionsHasName PetscOptionsHasName_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsInsertFile"
static PETSC_UNUSED
PetscErrorCode PetscOptionsInsertFile_Compat(MPI_Comm comm,const char file[],PetscTruth require)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsInsertFile(file);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscOptionsInsertFile PetscOptionsInsertFile_Compat
#endif

#if (PETSC_VERSION_(2,3,2))
#define PetscOptionsMonitorSet PetscOptionsSetMonitor
#define PetscOptionsMonitorCancel PetscOptionsClearMonitor
#endif

#if (PETSC_VERSION_(2,3,2))
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
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PetscLogStage                int
#define PetscLogEvent                PetscEvent
#define PetscLogStageRegister(n,s)   PetscLogStageRegister(s,n)
#define PetscLogEventRegister(n,c,e) PetscLogEventRegister(e,n,c)
#define PetscCookieRegister(n,c)     PetscLogClassRegister(c,n)
#endif

#if (PETSC_VERSION_(2,3,2))
#if defined(PETSC_HAVE_MPIUNI)
#if !defined(MPI_Finalized)
static PETSC_UNUSED
int Petsc_MPI_Finalized(int *flag)
{
  if (flag) *flag = 0;
  return 0;
}
#define MPI_Finalized Petsc_MPI_Finalized
#endif
#endif
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
typedef PetscToken* PetscToken_Compat;
#define PetscToken PetscToken_Compat
#endif

#endif /* _COMPAT_PETSC_H */
