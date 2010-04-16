#ifndef _COMPAT_PETSC_H
#define _COMPAT_PETSC_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))

typedef PetscCookie PetscClassId;

#define  PETSC_OBJECT_CLASSID  	 PETSC_OBJECT_COOKIE
#define  PETSC_VIEWER_CLASSID  	 PETSC_VIEWER_COOKIE
#define  PETSC_RANDOM_CLASSID  	 PETSC_RANDOM_COOKIE
#define  IS_CLASSID            	 IS_COOKIE
#define  IS_LTOGM_CLASSID      	 IS_LTOGM_COOKIE
#define  VEC_CLASSID           	 VEC_COOKIE
#define  VEC_SCATTER_CLASSID   	 VEC_SCATTER_COOKIE
#define  MAT_CLASSID           	 MAT_COOKIE
#define  MAT_NULLSPACE_CLASSID 	 MAT_NULLSPACE_COOKIE
#define  MAT_FDCOLORING_CLASSID  MAT_FDCOLORING_COOKIE
#define  PC_CLASSID            	 PC_COOKIE
#define  KSP_CLASSID           	 KSP_COOKIE
#define  SNES_CLASSID          	 SNES_COOKIE
#define  TS_CLASSID            	 TS_COOKIE
#define  AO_CLASSID            	 AO_COOKIE
#define  DM_CLASSID            	 DM_COOKIE

#define PetscObjectGetClassId PetscObjectGetCookie
#define PetscClassIdRegister  PetscCookieRegister

static StageLog _v_stageLog = 0;
#define _stageLog (PetscLogGetStageLog(&_v_stageLog),_v_stageLog)

#endif

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
