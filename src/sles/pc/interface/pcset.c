#ifndef lint
static char vcid[] = "$Id: pcset.c,v 1.6 1995/03/06 03:49:52 bsmith Exp curfman $";
#endif

#include "petsc.h"
#include "pcimpl.h"      /*I "pc.h" I*/
#include <stdio.h>
#include "sys/nreg.h"
#include "sys.h"
#include "options.h"

static NRList *__PCList = 0;

/*@
  PCSetMethod - Builds PC for a particular preconditioner.

  Input Parameter:
.  pc - the preconditioner context.
.  method   - One of the known methods.  See "pc.h" for
    available methods (for instance PCJACOBI).
 @*/
int PCSetMethod(PC ctx,PCMETHOD method)
{
  int (*r)(PC);
  VALIDHEADER(ctx,PC_COOKIE);
  if (ctx->setupcalled) {
    SETERR(1,"Method cannot be called after PCSetUp");
  }
  /* Get the function pointers for the method requested */
  if (!__PCList) {PCRegisterAll();}
  if (!__PCList) {
    SETERR(1,"Could not acquire list of preconditioner methods"); 
  }
  r =  (int (*)(PC))NRFindRoutine( __PCList, (int)method, (char *)0 );
  if (!r) {SETERR(1,"Unknown preconditioner method");}
  if (ctx->data) FREE(ctx->data);
  ctx->setfrom     = ( int (*)(PC) ) 0;
  ctx->printhelp   = ( int (*)(PC) ) 0;
  ctx->setup       = ( int (*)(PC) ) 0;
  ctx->destroy     = ( int (*)(PetscObject) ) 0;
  return (*r)(ctx);
}

/*@C
   PCRegister - Adds the iterative method to the preconditioner
   package,  given an iterative name (PCMETHOD) and a function pointer.

   Input Parameters:
.      name - for instance PCJACOBI, ...
.      sname -  corresponding string for name
.      create - routine to create method context
@*/
int  PCRegister(PCMETHOD name,char *sname,int (*create)(PC))
{
  int ierr;
  if (!__PCList) {ierr = NRCreate(&__PCList); CHKERR(ierr);}
  return NRRegister( __PCList, (int) name, sname, (int (*)(void*)) create );
}

/*@
   PCRegisterDestroy - Frees the list of preconditioners
   registered by PCRegister().
@*/
int PCRegisterDestroy()
{
  if (__PCList) {
    NRDestroy( __PCList );
    __PCList = 0;
  }
  return 0;
}

/*@C
  PCGetMethodFromOptions - Sets the selected PC method from the options 
                           database.

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. method - PC method

  Returns:
  1 if method is found; otherwise 0.

  Options Database Key:
$ -pcmethod  method
@*/
int PCGetMethodFromOptions(PC pc,PCMETHOD *method )
{
  char sbuf[50];
  if (OptionsGetString(  0, pc->prefix,"-pcmethod", sbuf, 50 )) {
    if (!__PCList) PCRegisterAll();
    *method = (PCMETHOD)NRFindID( __PCList, sbuf );
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
@*/
int PCGetMethodName(PCMETHOD meth,char **name)
{
  if (!__PCList) PCRegisterAll();
  *name = NRFindName( __PCList, (int)meth );
  return 0;
}

#include <stdio.h>
/*@C
    PCPrintMethods - Prints the preconditioner methods available 
                     from the options database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "pcmethod") 

   Notes:
   This routine is called from PCPrintHelp().
@*/
int PCPrintMethods(char *prefix,char *name)
{
  FuncList *entry;
  if (!__PCList) {PCRegisterAll();}
  entry = __PCList->head;
  fprintf(stderr," %s%s (one of)",prefix,name);
  while (entry) {
    fprintf(stderr," %s",entry->name);
    entry = entry->next;
  }
  fprintf(stderr,"\n");
  return 0;
}
/*@
   PCSetFromOptions - Sets PC options from the command line.
                      This must be called before PCSetUp()
                      if the user is to be allowed to set the 
                      preconditioner method. 

   Input Parameters:
.  pc - the preconditioner context

   Note:
   See PCPrintHelp() for a list of available PC options.   
@*/
int PCSetFromOptions(PC pc)
{
  PCMETHOD method;
  VALIDHEADER(pc,PC_COOKIE);

  if (PCGetMethodFromOptions(pc,&method)) {
    PCSetMethod(pc,method);
  }
  if (OptionsHasName(0,0,"-help")){
    PCPrintHelp(pc);
  }
  if (pc->setfrom) return (*pc->setfrom)(pc);
  return 0;
}
