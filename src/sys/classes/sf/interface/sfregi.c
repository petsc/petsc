#include <petsc-private/sfimpl.h>     /*I  "petscsf.h"  I*/

#if defined(PETSC_HAVE_MPI_WIN_CREATE)
PETSC_EXTERN_C PetscErrorCode PetscSFCreate_Window(PetscSF);
#endif
PETSC_EXTERN_C PetscErrorCode PetscSFCreate_Basic(PetscSF);

PetscFList PetscSFList;

#undef __FUNCT__
#define __FUNCT__ "PetscSFRegisterAll"
/*@C
   PetscSFRegisterAll - Registers all the PetscSF communication implementations

   Not Collective

   Level: advanced

.keywords: PetscSF, register, all

.seealso:  PetscSFRegisterDestroy()
@*/
PetscErrorCode  PetscSFRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscSFRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  ierr = PetscSFRegisterDynamic(PETSCSFWINDOW,       path,"PetscSFCreate_Window",       PetscSFCreate_Window);CHKERRQ(ierr);
#endif
  ierr = PetscSFRegisterDynamic(PETSCSFBASIC,        path,"PetscSFCreate_Basic",        PetscSFCreate_Basic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFRegister"
/*@C
  PetscSFRegister - See PetscSFRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  PetscSFRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(PetscSF))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(PETSC_COMM_WORLD,&PetscSFList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFRegisterDestroy"
/*@
   PetscSFRegisterDestroy - Frees the list of communication implementations registered by PetscSFRegisterDynamic()

   Not Collective

   Level: advanced

.keywords: PetscSF, register, destroy

.seealso: PetscSFRegisterAll()
@*/
PetscErrorCode  PetscSFRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PetscSFList);CHKERRQ(ierr);
  PetscSFRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
