#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mmregis.c,v 1.4 1998/03/24 20:59:56 balay Exp curfman $";
#endif

#include "petsc.h"
#include "mmimpl.h"          /*I   "mm.h"   I*/

extern int MMCreate_Euler(MM);
extern int MMCreate_FullPotential(MM);
extern int MMCreate_Hybrid_EF1(MM);

/* temporarily undefine this so that we register the routines for
   creation of multimodel contexts (avoiding the use of dynamic libs
   just for simplicity here) */
#undef USE_DYNAMIC_LIBRARIES

/*
    This is used by MMSetType() to make sure that at least one 
    MMRegisterAll() is called. In general, if there is more than one
    DLL, then MMRegisterAll() may be called several times.
*/
extern int MMRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "MMRegisterAll"
/*
  MMRegisterAll - Registers all of the multi-models in the MM package.

  Not collective

.seealso:  MMRegisterDestroy()
*/
int MMRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  MMRegisterAllCalled = 1;

  ierr = MMRegister_Private(MMEULER,path,"MMCreate_Euler",MMCreate_Euler); CHKERRQ(ierr);
  ierr = MMRegister(MMFP,path,"MMCreate_FullPotential",MMCreate_FullPotential); CHKERRQ(ierr);
  ierr = MMRegister(MMHYBRID_EF1,path,"MMCreate_Hybrid_EF1",MMCreate_Euler); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

