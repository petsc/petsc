#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesregi.c,v 1.18 1997/02/22 02:28:38 bsmith Exp balay $";
#endif

#include "src/snes/snesimpl.h"     /*I  "snes.h"  I*/
extern int SNESCreate_EQ_LS(SNES);
extern int SNESCreate_EQ_TR(SNES);
extern int SNESCreate_UM_TR(SNES);
extern int SNESCreate_UM_LS(SNES);
extern int SNESCreate_Test(SNES);

#undef __FUNC__  
#define __FUNC__ "SNESRegisterAll" /* ADIC Ignore */
/*@C
  SNESRegisterAll - Registers all of the nonlinear solvers in the SNES 
  package. 

  Adding new methods:
  To add a new method to the registry, copy this routine and modify
  it to incorporate a call to SNESRegister() for the new method.  

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsnes.a.

.keywords: SNES, nonlinear, register, all

.seealso: SNESRegister(), SNESRegisterDestroy()
@*/
int SNESRegisterAll()
{
  SNESRegisterAllCalled = 1;
  SNESRegister(SNES_EQ_LS,         0,"ls",      SNESCreate_EQ_LS);
  SNESRegister(SNES_EQ_TR,         0,"tr",      SNESCreate_EQ_TR);
  SNESRegister(SNES_EQ_TEST,       0,"test",    SNESCreate_Test);
  SNESRegister(SNES_UM_TR,         0,"umtr",    SNESCreate_UM_TR);
  SNESRegister(SNES_UM_LS,         0,"umls",    SNESCreate_UM_LS);
  return 0;
}
