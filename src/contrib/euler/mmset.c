#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.63 1997/09/22 15:21:33 balay Exp $";
#endif

/* 
   This file contains routines to set multi-model methods and options.
 */

#include "petsc.h"
#include "mmimpl.h"
#include <stdio.h>
#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MMList = 0;
int MMRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "MMSetType"
/*
   MMSetType - Builds MM for a particular multi-model.

   Input Parameter:
.  mm - the preconditioner context.
.  type - a known method

   Options Database Command:
$  -mm_type  <type>
$      Use -help for a list of available methods
$      (for instance, jacobi or bjacobi)

  Notes:
  See "mm.h" for available methods (for instance,  MMEULER, MMFP, or MMHYBRID).
*/
int MMSetType(MM ctx,MMType type)
{
  int ierr,(*r)(MM);

  PetscValidHeaderSpecific(ctx,ctx->MM_COOKIE);
  if (ctx->type == (int) type) return 0;

  if (ctx->setupcalled) {
    if (ctx->destroy) ierr =  (*ctx->destroy)((PetscObject)ctx);
    else {if (ctx->data) PetscFree(ctx->data);}
    ctx->data        = 0;
    ctx->setupcalled = 0;
  }
  /* Get the function pointers for the method requested */
  if (!MMRegisterAllCalled) {ierr = MMRegisterAll(); CHKERRQ(ierr);}
  if (!__MMList) {SETERRQ(1,0,"Could not get list of methods");}
  r =  (int (*)(MM))NRFindRoutine( __MMList, (int)type, (char *)0 );
  if (!r) {SETERRQ(1,0,"Unknown type");}
  if (ctx->data) PetscFree(ctx->data);
  ctx->destroy      = ( int (*)(PetscObject) ) 0;
  ctx->view         = ( int (*)(PetscObject,Viewer) ) 0;
  return (*r)(ctx);
}

#undef __FUNC__  
#define __FUNC__ "MMRegister"
/*
   MMRegister - Adds the model to the multi-model package, given a model
   name (MMType) and a function pointer.

   Input Parameters:
.  name - either a predefined name such as MMEULER, or MMNEW
          to indicate a new user-defined model
.  sname -  corresponding string for name
.  create - routine to create method context

   Output Parameter:
.  oname - type associated with this new preconditioner

   Notes:
   Multiple user-defined preconditioners can be added by calling
   MMRegister() with the input parameter "name" set to be MMNEW; 
   each call will return a unique preconditioner type in the output
   parameter "oname".
*/
int MMRegister(MMType name,MMType *oname,char *sname,int (*create)(MM))
{
  int ierr;
  static int numberregistered = 0;

  if (name == MMNEW) name = (MMType) ((int) MMNEW + numberregistered++);

  if (oname) *oname = name;
  if (!__MMList) {ierr = NRCreate(&__MMList); CHKERRQ(ierr);}
  return NRRegister( __MMList, (int) name, sname, (int (*)(void*)) create );
}

#undef __FUNC__  
#define __FUNC__ "MMRegisterDestroy"
/*
   MMRegisterDestroy - Frees the list of multi-models that were
   registered by MMRegister().

.keywords: MM, register, destroy

.seealso: MMRegisterAll(), MMRegisterAll()
*/
int MMRegisterDestroy()
{
  if (__MMList) {
    NRDestroy( __MMList );
    __MMList = 0;
  }
  MMRegisterAllCalled = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MMGetTypeFromOptions_Private"
/* 
  MMGetTypeFromOptions_Private - Sets the selected MM type from the 
  options database.

  Input Parameter:
. MM - the preconditioner context

  Output Parameter:
. method - MM method

  Returns:
  1 if method is found; otherwise 0.

  Options Database Key:
$ -mm_type method
*/
int MMGetTypeFromOptions_Private(MM mm,MMType *method)
{
  int  ierr,flg;
  char sbuf[50];

  ierr = OptionsGetString( mm->prefix,"-mm_type", sbuf, 50,&flg );CHKERRQ(ierr);
  if (flg) {
    if (!__MMList) {ierr = MMRegisterAll(); CHKERRQ(ierr);}
    *method = (MMType)NRFindID( __MMList, sbuf );
    return 1;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MMGetType"
/*@C
   MMGetType - Gets the MM method type and name (as a string) from the MM
   context.

   Input Parameter:
.  mm - the preconditioner context

   Output Parameter:
.  name - name of multi-model (or use PETSC_NULL)
.  meth - multi-model (or use PETSC_NULL)

@*/
int MMGetType(MM mm,MMType *meth,char **name)
{
  int ierr;
  if (!__MMList) {ierr = MMRegisterAll(); CHKERRQ(ierr);}
  if (meth) *meth = (MMType) mm->type;
  if (name)  *name = NRFindName( __MMList, (int)mm->type );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MMPrintTypes_Private"
/*
   MMPrintTypes_Private - Prints the MM methods available from the options 
   database.

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  prefix - prefix (usually "-")
.  name   - the options database name (by default "mm_type") 
*/
int MMPrintTypes_Private(MPI_Comm comm,char *prefix,char *name)
{
  FuncList *entry;
  int      count = 0,ierr;

  if (!__MMList) {ierr = MMRegisterAll(); CHKERRQ(ierr);}
  entry = __MMList->head;
  PetscPrintf(comm," %s%s (one of)",prefix,name);
  while (entry) {
    PetscPrintf(comm," %s",entry->name);
    entry = entry->next;
    count++;
    if (count == 8) PetscPrintf(comm,"\n     ");
  }
  PetscPrintf(comm,"\n");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MMSetFromOptions"
/*@
   MMSetFromOptions - Sets MM options from the options database.
   This routine must be called before MMSetUp() if the user is to be
   allowed to set the preconditioner method. 

   Input Parameters:
.  mm - the preconditioner context

.keywords: MM, set, from, options, database

.seealso: MMPrintHelp()
@*/
int MMSetFromOptions(MM mm)
{
  MMType method;
  int    ierr,flg;

  PetscValidHeaderSpecific(mm,mm->MM_COOKIE);

  if (MMGetTypeFromOptions_Private(mm,&method)) {
    MMSetType(mm,method);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); 
  if (flg){
    MMPrintHelp(mm);
  }
  if (mm->setfrom) return (*mm->setfrom)(mm);
  return 0;
}
