#ifndef lint
static char vcid[] = "$Id: pcset.c,v 1.1 1994/11/21 06:47:24 bsmith Exp bsmith $";
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
  if (!__PCList) __PCList = NRCreate();
  return NRRegister( __PCList, (int) name, sname, (int (*)()) create );
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
  PCGetMethodFromOptions - Sets the selected method from the options database.

  Input parameters:
. flag - 1 if argument should be removed from list if found 
. sname - name used to indicate solver.  If null, -itmethod is used

  Output parameter:
. pcmethod -  Iterative method type
. returns 1 if method found else 0.
@*/
int PCGetMethodFromOptions(int flag,char *sname,PCMETHOD *method )
{
  char sbuf[50];
  if (!sname) sname = "-pcmethod";
  if (OptionsGetString(  flag, sname, sbuf, 50 )) {
    if (!__PCList) PCRegisterAll();
    *method = (PCMETHOD)NRFindID( __PCList, sbuf );
    return 1;
  }
  return 0;
}

/*@C
   PCGetMethodName - Get the name (as a string) from the method type

   Input Parameter:
.  itctx - Preconditioner context
@*/
int PCGetMethodName(PCMETHOD meth,char **name )
{
  if (!__PCList) PCRegisterAll();
  *name = NRFindName( __PCList, meth );
  return 0;
}

#include <stdio.h>
/*@C
    PCPrintMethods - prints the preconditioner methods available 
              from the command line.

  Input Parameters:
.   name - the command line option (usually -pcmethod) 
@*/
int PCPrintMethods(char *name)
{
  FuncList *entry;
  if (!__PCList) {PCRegisterAll();}
  entry = __PCList->head;
  fprintf(stderr," %s (one of)",name);
  while (entry) {
    fprintf(stderr," %s",entry->name);
    entry = entry->next;
  }
  fprintf(stderr,"\n");
  return 0;
}
/*@
    PCSetFromOptions - sets PC options from the command line.
                            This must be called before PCPSetUp()
                            if the user is to be allowed to set the 
                            preconditioner method. 

  Input Parameters:
.  pc - the preconditioner context
   
@*/
int PCSetFromOptions(PC pc)
{
  char     string[50];
  PCMETHOD method;
  VALIDHEADER(pc,PC_COOKIE);

  if (PCGetMethodFromOptions(0,pc->namemethod,&method)) {
    PCSetMethod(pc,method);
  }
  if (OptionsHasName(0,"-help")){
    PCPrintHelp(pc);
  }
  if (pc->setfrom) return (*pc->setfrom)(pc);
  return 0;
}
