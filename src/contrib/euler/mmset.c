#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mmset.c,v 1.4 1998/03/24 20:59:51 balay Exp curfman $";
#endif

/* 
   This file contains routines to set multi-model methods and options.
 */

#include "petsc.h"
#include "mmimpl.h"
#include <stdio.h>
#include "src/sys/nreg.h"
#include "sys.h"

int MMRegisterAllCalled = 0;
/*
   Contains the list of registered KSP routines
*/
DLList MMList = 0;

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,ctx->MM_COOKIE);
  if (!PetscStrcmp(ctx->type_name,type)) PetscFunctionReturn(0);

  if (ctx->setupcalled) {
    if (ctx->destroy) ierr =  (*ctx->destroy)(ctx);
    else {if (ctx->data) PetscFree(ctx->data);}
    ctx->data        = 0;
    ctx->setupcalled = 0;
  }
  /* Get the function pointers for the method requested */
  if (!MMRegisterAllCalled) {ierr = MMRegisterAll(PETSC_NULL); CHKERRQ(ierr);}
  ierr = DLRegisterFind(ctx->comm, MMList, type,(int (**)(void *))&r); CHKERRQ(ierr);
  if (!r) SETERRQ(1,1,"Unknown MM type given");

  if (ctx->data) PetscFree(ctx->data);
  ctx->data    = 0;
  ierr = (*r)(ctx); CHKERRQ(ierr);

  if (ctx->type_name) PetscFree(ctx->type_name);
  ctx->type_name = (char *) PetscMalloc((PetscStrlen(type)+1)*sizeof(char)); CHKPTRQ(ctx->type_name);
  PetscStrcpy(ctx->type_name,type);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MMRegisterDestroy"
/*
   MMRegisterDestroy - Frees the list of multi-models that were
   registered by MMRegister().

   Not Collective

.keywords: MM, register, destroy

.seealso: MMRegisterAll(), MMRegisterAll()
*/
int MMRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MMList) {
    ierr = DLRegisterDestroy(MMList); CHKERRQ(ierr);
    MMList = 0;
  }
  MMRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MMGetType"
/*@C
   MMGetType - Gets the MM type as a string from the MM context.

   Input Parameter:
.  mm - the preconditioner context

   Output Parameter:
.  type - name of multi-model

@*/
int MMGetType(MM mm,MMType *type)
{
  PetscFunctionBegin;
  *type = mm->type_name;
  PetscFunctionReturn(0);
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
