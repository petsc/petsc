#ifndef _COMPAT_PETSC_SYS_H
#define _COMPAT_PETSC_SYS_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define PetscSysInitializePackage PetscInitializePackage
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
static StageLog _v_stageLog = 0;
#define _stageLog (PetscLogGetStageLog(&_v_stageLog),_v_stageLog)
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "PetscObjectGetClassName"
static PetscErrorCode
PetscObjectGetClassName(PetscObject obj, const char *class_name[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidPointer(class_name,2);
  *class_name = obj->class_name;
  PetscFunctionReturn(0);
}
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

#if PETSC_VERSION_(3,0,0)
#define MPIU_SUM PetscSum_Op
#endif

#endif /* _COMPAT_PETSC_SYS_H */
