#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.63 1997/09/22 15:21:33 balay Exp $";
#endif

#include "petsc.h"
#include "mmimpl.h"          /*I   "mm.h"   I*/

extern int MMCreate_Euler(MM);
extern int MMCreate_FullPotential(MM);
extern int MMCreate_Hybrid_EF1(MM);

#undef __FUNC__  
#define __FUNC__ "MMRegisterAll"
/*
  MMRegisterAll - Registers all of the multi-models in the MM package.

  Notes: 
  Currently the default method MMEULER must be registered.
*/
int MMRegisterAll()
{
  MMRegisterAllCalled = 1;

  MMRegister(MMEULER      ,0, "euler",   MMCreate_Euler);
  MMRegister(MMFP         ,0, "fp",      MMCreate_FullPotential);
  MMRegister(MMHYBRID_EF1 ,0, "hybrid",  MMCreate_Hybrid_EF1);
  return 0;
}

