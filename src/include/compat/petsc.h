#ifndef _COMPAT_PETSC_H
#define _COMPAT_PETSC_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
typedef PetscTruth PetscBool;
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
typedef enum {
  PETSC_ERROR_INITIAL=1,
  PETSC_ERROR_REPEAT=0,
  PETSC_ERROR_IN_CXX=2
} PetscErrorType;
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))

typedef PetscCookie PetscClassId;

#define  PETSC_OBJECT_CLASSID    PETSC_OBJECT_COOKIE
#define  PETSC_FWK_CLASSID       PETSC_FWK_COOKIE
#define  PETSC_VIEWER_CLASSID    PETSC_VIEWER_COOKIE
#define  PETSC_RANDOM_CLASSID    PETSC_RANDOM_COOKIE
#define  IS_CLASSID              IS_COOKIE
#define  IS_LTOGM_CLASSID        IS_LTOGM_COOKIE
#define  VEC_CLASSID             VEC_COOKIE
#define  VEC_SCATTER_CLASSID     VEC_SCATTER_COOKIE
#define  MAT_CLASSID             MAT_COOKIE
#define  MAT_NULLSPACE_CLASSID   MAT_NULLSPACE_COOKIE
#define  MAT_FDCOLORING_CLASSID  MAT_FDCOLORING_COOKIE
#define  PC_CLASSID              PC_COOKIE
#define  KSP_CLASSID             KSP_COOKIE
#define  SNES_CLASSID            SNES_COOKIE
#define  TS_CLASSID              TS_COOKIE
#define  AO_CLASSID              AO_COOKIE
#define  DM_CLASSID              DM_COOKIE

#define PetscObjectGetClassId PetscObjectGetCookie
#define PetscClassIdRegister  PetscCookieRegister

static StageLog _v_stageLog = 0;
#define _stageLog (PetscLogGetStageLog(&_v_stageLog),_v_stageLog)

#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHasName"
static PetscErrorCode PetscOptionsHasName_Compat(const char pre[],
                                                 const char name[],
                                                 PetscTruth *flg)
{
  char dummy[2] = { 0, 0 };
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsGetString(pre,name,dummy,1,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscOptionsHasName PetscOptionsHasName_Compat
#endif

#endif /* _COMPAT_PETSC_H */
