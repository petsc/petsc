#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mmset.c,v 1.5 1998/05/13 18:58:46 curfman Exp curfman $";
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
+  mm - the preconditioner context.
-  type - a known model

   Options Database Key:
.  -mm_type <type> - Specifies multi-model type.  Use -help
    for a list of available models (for instance, euler or fp)

  Notes:
  See "mm.h" for available methods.
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
  ierr = DLRegisterFind(ctx->comm,MMList,type,(int (**)(void *))&r); CHKERRQ(ierr);
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
  int       flg, ierr;
  char      method[256];

  PetscFunctionBegin;

  PetscValidHeaderSpecific(mm,mm->MM_COOKIE);
  if (!MMRegisterAllCalled) {
    ierr = MMRegisterAll(PETSC_NULL); CHKERRQ(ierr);
  }
  ierr = OptionsGetString(mm->prefix,"-mm_type",method,256,&flg);
  if (flg) {
    ierr = MMSetType(mm,method); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MMPrintHelp(mm); CHKERRQ(ierr);
  }

  /*
    Since the private setfromoptions requires the type to have 
    been set already, we make sure a type is set by this time.
    */
  if (!mm->type_name) {
    ierr = MMSetType(mm,MMEULER); CHKERRQ(ierr);
  }

  if (mm->setfromoptions) {
    ierr = (*mm->setfromoptions)(mm); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MMRegister - Adds a method to the multi-model (MM) package.

   Synopsis:
   MMRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(MM))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined multi-model.
.  path - path (either absolute or relative) the library containing this model
.  name_create - name of routine to create model context
-  routine_create - routine to create model context

   Notes:
   MMRegister() may be called multiple times to add several user-defined models.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MMRegister("my_model",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyModelCreate",MyModelCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MMSetType(mm,"my_model")
   or at runtime via the option
$     -mm_type my_model

.keywords: MM, register

.seealso: MMRegisterAll(), MMRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "MMRegister_Private"
int MMRegister_Private(char *sname,char *path,char *name,int (*function)(MM))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = DLRegister_Private(&MMList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
