#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pcset.c,v 1.57 1997/08/22 15:12:25 bsmith Exp bsmith $";
#endif
/*
    Routines to set PC methods and options.
*/

#include "petsc.h"
#include "src/pc/pcimpl.h"      /*I "pc.h" I*/
#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__PCList = 0;

int  PCRegisterAllCalled = 0;
#undef __FUNC__  
#define __FUNC__ "PCSetType"
/*@
   PCSetType - Builds PC for a particular preconditioner.

   Input Parameter:
.  pc - the preconditioner context.
.  type - a known method

   Options Database Command:
$  -pc_type  <type>
$      Use -help for a list of available methods
$      (for instance, jacobi or bjacobi)

  Notes:
  See "petsc/include/pc.h" for available methods (for instance,
  PCJACOBI, PCILU, or PCBJACOBI).

  Normally, it is best to use the SLESSetFromOptions() command and
  then set the PC type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different preconditioners. 
  The PCSetType() routine is provided for those situations where it
  is necessary to set the preconditioner independently of the command
  line or options database.  This might be the case, for example, when
  the choice of preconditioner changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate preconditioner.  In other words, this
  routine is for the advanced user.

.keywords: PC, set, method, type
@*/
int PCSetType(PC ctx,PCType type)
{
  int ierr,(*r)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,PC_COOKIE);
  if (ctx->type == (int) type) PetscFunctionReturn(0);

  if (ctx->setupcalled) {
    if (ctx->destroy) ierr =  (*ctx->destroy)((PetscObject)ctx);
    else {if (ctx->data) PetscFree(ctx->data);}
    ctx->data        = 0;
    ctx->setupcalled = 0;
  }
  /* Get the function pointers for the method requested */
  if (!PCRegisterAllCalled) {ierr = PCRegisterAll(); CHKERRQ(ierr);}
  if (!__PCList) {SETERRQ(1,0,"Could not get list of methods");}
  r =  (int (*)(PC))NRFindRoutine( __PCList, (int)type, (char *)0 );
  if (!r) {SETERRQ(1,0,"Unknown type");}
  if (ctx->data) PetscFree(ctx->data);

  ctx->destroy      = ( int (*)(PetscObject) ) 0;
  ctx->view         = ( int (*)(PetscObject,Viewer) ) 0;
  ctx->apply        = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->setup        = ( int (*)(PC) ) 0;
  ctx->applyrich    = ( int (*)(PC,Vec,Vec,Vec,int) ) 0;
  ctx->applyBA      = ( int (*)(PC,int,Vec,Vec,Vec) ) 0;
  ctx->setfrom      = ( int (*)(PC) ) 0;
  ctx->printhelp    = ( int (*)(PC,char*) ) 0;
  ctx->applytrans   = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->applyBAtrans = ( int (*)(PC,int,Vec,Vec,Vec) ) 0;
  ctx->presolve     = ( int (*)(PC,KSP) ) 0;
  ctx->postsolve    = ( int (*)(PC,KSP) ) 0;
  ctx->getfactoredmatrix   = ( int (*)(PC,Mat*) ) 0;
  ctx->applysymmetricleft  = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->applysymmetricright = ( int (*)(PC,Vec,Vec) ) 0;
  ctx->setuponblocks       = ( int (*)(PC) ) 0;
  ctx->modifysubmatrices   = ( int (*)(PC,int,IS*,IS*,Mat*,void*) ) 0;
  ierr = (*r)(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCRegister"
/*@C
   PCRegister - Adds the preconditioner to the preconditioner
   package,  given a preconditioner name (PCType) and a function pointer.

   Input Parameters:
.  name - either a predefined name such as PCJACOBI, or PCNEW
          to indicate a new user-defined preconditioner
.  sname -  corresponding string for name
.  create - routine to create method context

   Output Parameter:
.  oname - type associated with this new preconditioner

   Notes:
   Multiple user-defined preconditioners can be added by calling
   PCRegister() with the input parameter "name" set to be PCNEW; 
   each call will return a unique preconditioner type in the output
   parameter "oname".

.keywords: PC, register

.seealso: PCRegisterAll(), PCRegisterDestroy()
@*/
int  PCRegister(PCType name,PCType *oname,char *sname,int (*create)(PC))
{
  int ierr;
  static int numberregistered = 0;

  PetscFunctionBegin;
  if (name == PCNEW) name = (PCType) ((int) PCNEW + numberregistered++);

  if (oname) *oname = name;
  if (!__PCList) {ierr = NRCreate(&__PCList); CHKERRQ(ierr);}
  ierr = NRRegister( __PCList, (int) name, sname, (int (*)(void*)) create );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCRegisterDestroy"
/*@C
   PCRegisterDestroy - Frees the list of preconditioners that were
   registered by PCRegister().

.keywords: PC, register, destroy

.seealso: PCRegisterAll(), PCRegisterAll()
@*/
int PCRegisterDestroy()
{
  PetscFunctionBegin;
  if (__PCList) {
    NRDestroy( __PCList );
    __PCList = 0;
  }
  PCRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetTypeFromOptions_Private"
/* 
  PCGetTypeFromOptions_Private - Sets the selected PC type from the 
  options database.

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. method - PC method

  Returns:
  1 if method is found; otherwise 0.

  Options Database Key:
$ -pc_type  method
*/
int PCGetTypeFromOptions_Private(PC pc,PCType *method )
{
  int  ierr,flg;
  char sbuf[50];

  PetscFunctionBegin;
  ierr = OptionsGetString( pc->prefix,"-pc_type", sbuf, 50,&flg );CHKERRQ(ierr);
  if (flg) {
    if (!__PCList) {ierr = PCRegisterAll(); CHKERRQ(ierr);}
    *method = (PCType)NRFindID( __PCList, sbuf );
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetType"
/*@C
   PCGetType - Gets the PC method type and name (as a string) from the PC
   context.

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - name of preconditioner (or use PETSC_NULL)
.  meth - preconditioner method (or use PETSC_NULL)

.keywords: PC, get, method, name, type
@*/
int PCGetType(PC pc,PCType *meth,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!__PCList) {ierr = PCRegisterAll(); CHKERRQ(ierr);}
  if (meth) *meth = (PCType) pc->type;
  if (name)  *name = NRFindName( __PCList, (int)pc->type );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintTypes_Private"
/*
   PCPrintTypes_Private - Prints the PC methods available from the options 
   database.

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  prefix - prefix (usually "-")
.  name   - the options database name (by default "pc_type") 
*/
int PCPrintTypes_Private(MPI_Comm comm,char *prefix,char *name)
{
  FuncList *entry;
  int      count = 0,ierr;

  PetscFunctionBegin;
  if (!__PCList) {ierr = PCRegisterAll(); CHKERRQ(ierr);}
  entry = __PCList->head;
  PetscPrintf(comm," %s%s (one of)",prefix,name);
  while (entry) {
    PetscPrintf(comm," %s",entry->name);
    entry = entry->next;
    count++;
    if (count == 8) PetscPrintf(comm,"\n     ");
  }
  PetscPrintf(comm,"\n");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions"
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
  PCType method;
  int    ierr,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);

  if (PCGetTypeFromOptions_Private(pc,&method)) {
    PCSetType(pc,method);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); 
  if (flg){
    PCPrintHelp(pc);
  }
  if (pc->setfrom) {
    ierr = (*pc->setfrom)(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
