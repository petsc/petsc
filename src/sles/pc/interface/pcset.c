#ifndef lint
static char vcid[] = "$Id: pcset.c,v 1.23 1995/08/22 16:30:13 bsmith Exp curfman $";
#endif

#include "petsc.h"
#include "pcimpl.h"      /*I "pc.h" I*/
#include <stdio.h>
#include "sys/nreg.h"
#include "sys.h"

static NRList *__PCList = 0;

/*@
  PCSetMethod - Builds PC for a particular preconditioner. It is 
                best to use the SLESSetFromOptions() command and 
                set the PC method from the command line rather then
                by using this routine.

  Input Parameter:
.  pc - the preconditioner context.
.  method - a known method

   Options Database Command:
$  -pc_method  <method>
$      Use -help for a list of available methods
$      (for instance, jacobi or bjacobi)

  Notes:
  See "petsc/include/pc.h" for available methods (for instance,
  PCJACOBI, PCILU, or PCBJACOBI).

.keywords: PC, set, method
@*/
int PCSetMethod(PC ctx,PCMethod method)
{
  int ierr,(*r)(PC);
  PETSCVALIDHEADERSPECIFIC(ctx,PC_COOKIE);
  if (ctx->setupcalled) {
    if (ctx->destroy) ierr =  (*ctx->destroy)((PetscObject)ctx);
    else {if (ctx->data) PETSCFREE(ctx->data);}
    ctx->data = 0;
  }
  /* Get the function pointers for the method requested */
  if (!__PCList) {PCRegisterAll();}
  if (!__PCList) {
    SETERRQ(1,"PCSetMethod: Could not acquire list of PC methods"); 
  }
  r =  (int (*)(PC))NRFindRoutine( __PCList, (int)method, (char *)0 );
  if (!r) {SETERRQ(1,"PCSetMethod: Unknown preconditioner method");}
  if (ctx->data) PETSCFREE(ctx->data);
  ctx->setfrom     = ( int (*)(PC) ) 0;
  ctx->printhelp   = ( int (*)(PC) ) 0;
  ctx->setup       = ( int (*)(PC) ) 0;
  ctx->destroy     = ( int (*)(PetscObject) ) 0;
  return (*r)(ctx);
}

/*@C
   PCRegister - Adds the iterative method to the preconditioner
   package,  given an iterative name (PCMethod) and a function pointer.

   Input Parameters:
.  name - for instance PCJACOBI, ...
.  sname -  corresponding string for name
.  create - routine to create method context

.keywords: PC, register

.seealso: PCRegisterAll(), PCRegisterDestroy()
@*/
int  PCRegister(PCMethod name,char *sname,int (*create)(PC))
{
  int ierr;
  if (!__PCList) {ierr = NRCreate(&__PCList); CHKERRQ(ierr);}
  return NRRegister( __PCList, (int) name, sname, (int (*)(void*)) create );
}

/*@C
   PCRegisterDestroy - Frees the list of preconditioners that were
   registered by PCRegister().

.keywords: PC, register, destroy

.seealso: PCRegisterAll(), PCRegisterAll()
@*/
int PCRegisterDestroy()
{
  if (__PCList) {
    NRDestroy( __PCList );
    __PCList = 0;
  }
  return 0;
}

/* 
  PCGetMethodFromOptions_Private - Sets the selected PC method from the 
  options database.

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. method - PC method

  Returns:
  1 if method is found; otherwise 0.

  Options Database Key:
$ -pc_method  method
*/
int PCGetMethodFromOptions_Private(PC pc,PCMethod *method )
{
  char sbuf[50];
  if (OptionsGetString( pc->prefix,"-pc_method", sbuf, 50 )) {
    if (!__PCList) PCRegisterAll();
    *method = (PCMethod)NRFindID( __PCList, sbuf );
    return 1;
  }
  return 0;
}

/*@C
   PCGetMethodName - Gets the PC method name (as a string) from the 
   method type.

   Input Parameter:
.  meth - preconditioner method

   Output Parameter:
.  name - name of preconditioner

.keywords: PC, get, method, name
@*/
int PCGetMethodName(PCMethod meth,char **name)
{
  if (!__PCList) PCRegisterAll();
  *name = NRFindName( __PCList, (int)meth );
  return 0;
}

#include <stdio.h>
/*
   PCPrintMethods_Private - Prints the PC methods available from the options 
   database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "pc_method") 
*/
int PCPrintMethods_Private(char *prefix,char *name)
{
  FuncList *entry;
  if (!__PCList) {PCRegisterAll();}
  entry = __PCList->head;
  printf(" %s%s (one of)",prefix,name);
  while (entry) {
    printf(" %s",entry->name);
    entry = entry->next;
  }
  printf("\n");
  return 0;
}

/*@
   PCSetFromOptions - Sets PC options from the options database.
   This routine must be called before PCSetUp() if the user is to be
   allowed to set the preconditioner method. 

   Input Parameters:
.  pc - the preconditioner context

.keywords: PC, set, from, options, database

.seealso: PCPrintHelp()
@*/
int PCSetFromOptions(PC pc)
{
  PCMethod method;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);

  if (PCGetMethodFromOptions_Private(pc,&method)) {
    PCSetMethod(pc,method);
  }
  if (OptionsHasName(0,"-help")){
    PCPrintHelp(pc);
  }
  if (pc->setfrom) return (*pc->setfrom)(pc);
  return 0;
}
