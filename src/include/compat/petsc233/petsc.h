#ifndef _PETSC_COMPAT_H
#define _PETSC_COMPAT_H

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


#define PetscLogStage                int
#define PetscLogEvent                PetscEvent
#define PetscLogStageRegister(n,s)   PetscLogStageRegister(s,n)
#define PetscLogEventRegister(n,c,e) PetscLogEventRegister(e,n,c)
#define PetscCookieRegister(n,c)     PetscLogClassRegister(c,n)

#endif /* _PETSC_COMPAT_H */
