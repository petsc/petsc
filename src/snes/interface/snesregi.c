#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesregi.c,v 1.21 1997/10/19 03:29:25 bsmith Exp bsmith $";
#endif

#include "src/snes/snesimpl.h"     /*I  "snes.h"  I*/
extern int SNESCreate_EQ_LS(SNES);
extern int SNESCreate_EQ_TR(SNES);
extern int SNESCreate_UM_TR(SNES);
extern int SNESCreate_UM_LS(SNES);
extern int SNESCreate_Test(SNES);

#if defined(USE_DYNAMIC_LIBRARIES)
#define SNESRegister(a,b,c,d) SNESRegister_Private(a,b,c,0)
#else
#define SNESRegister(a,b,c,d) SNESRegister_Private(a,b,c,d)
#endif

#undef __FUNC__  
#define __FUNC__ "SNESRegister_Private"
static int SNESRegister_Private(char *sname,char *path,char *name,int (*function)(SNES))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = DLRegister(&SNESList,sname,fullname, (int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
/*
      This is used by SNESSetType() to make sure that at least one 
    SNESRegisterAll() is called. In general, if there is more than one
    DLL then SNESRegisterAll() may be called several times.
*/
extern int SNESRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "SNESRegisterAll"
/*@C
  SNESRegisterAll - Registers all of the nonlinear solver methods in the SNES package.

.keywords: SNES, register, all

.seealso:  SNESRegisterDestroy()
@*/
int SNESRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  SNESRegisterAllCalled = 1;

  ierr = SNESRegister("ls",   path,"SNESCreate_EQ_LS",SNESCreate_EQ_LS);CHKERRQ(ierr);
  ierr = SNESRegister("tr",   path,"SNESCreate_EQ_TR",SNESCreate_EQ_TR);CHKERRQ(ierr);
  ierr = SNESRegister("test", path,"SNESCreate_Test", SNESCreate_Test);CHKERRQ(ierr);
  ierr = SNESRegister("umtr", path,"SNESCreate_UM_TR",SNESCreate_UM_TR);CHKERRQ(ierr);
  ierr = SNESRegister("umls", path,"SNESCreate_UM_LS",SNESCreate_UM_LS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

