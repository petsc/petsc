#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsreg.c,v 1.24 1997/10/19 03:28:16 bsmith Exp bsmith $";
#endif

#include "src/ts/tsimpl.h"      /*I "ts.h"  I*/
#include "src/sys/nreg.h"      
#include "pinclude/pviewer.h"
#include <math.h>

static NRList *__TSList = 0;
int TSRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "TSSetType"
/*@
   TSSetType - Sets the method for the timestepping solver.  

   Input Parameters:
.  ts - the TS context
.  method - a known method

  Options Database Command:
$ -ts_type  <method>
$    Use -help for a list of available methods
$    (for instance, euler)

   Notes:
   See "petsc/include/ts.h" for available methods (for instance)
$   TS_EULER
$   TS_PVODE
$   TS_BEULER
$   TS_PSEUDO

  Normally, it is best to use the TSSetFromOptions() command and
  then set the TS type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different solvers.
  The TSSetType() routine is provided for those situations where it
  is necessary to set the timestepping solver independently of the
  command line or options database.  This might be the case, for example,
  when the choice of solver changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate method.  In other words, this routine is
  for the advanced user.

.keywords: TS, set, type
@*/
int TSSetType(TS ts,TSType method)
{
  int ierr,(*r)(TS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  /* Get the function pointers for the method requested */
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
  if (!__TSList) {SETERRQ(1,0,"Could not get methods");}
  r =  (int (*)(TS))NRFindRoutine( __TSList, (int)method, (char *)0 );
  if (!r) {SETERRQ(1,0,"Unknown method");}
  if (ts->data) PetscFree(ts->data);
  ierr = (*r)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "TSRegister"
/*@C
   TSRegister - Adds the method to the timestepping package, given 
   a function pointer and a solver name of the type TSType.

   Input Parameters:
.  name - either a predefined name such as TS_BEULER, or TS_NEW
          to indicate a new user-defined solver
.  sname - corresponding string for name
.  create - routine to create method context

   Output Parameter:
.  oname - type associated with this new method

   Notes:
   Multiple user-defined timestepping solvers can be added by calling
   TSRegister() with the input parameter "name" set to be TS_NEW; 
   each call will return a unique solver type in the output
   parameter "oname".

.keywords: TS, timestepper, register

.seealso: TSRegisterAll(), TSRegisterDestroy()
@*/
int TSRegister(TSType name,TSType *oname, char *sname, int (*create)(TS))
{
  int ierr;
  static int numberregistered = 0;

  PetscFunctionBegin;
  if (name == TS_NEW) name = (TSType) ((int) TS_NEW + numberregistered++);

  if (oname) *oname = name;
  if (!__TSList) {ierr = NRCreate(&__TSList); CHKERRQ(ierr);}
  NRRegister( __TSList, (int) name, sname, (int (*)(void*))create );
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "TSRegisterDestroy"
/*@C
   TSRegisterDestroy - Frees the list of timesteppers that were
   registered by TSRegister().

.keywords: TS, timestepper, register, destroy

.seealso: TSRegisterAll(), TSRegisterAll()
@*/
int TSRegisterDestroy()
{
  PetscFunctionBegin;
  if (__TSList) {
    NRDestroy( __TSList );
    __TSList = 0;
  }
  TSRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetType"
/*@C
   TSGetType - Gets the TS method type and name (as a string).

   Input Parameter:
.  ts - timestepper solver context

   Output Parameter:
.  method - TS method (or use PETSC_NULL)
.  name - name of TS method (or use PETSC_NULL)

.keywords: TS, timestepper, get, method, name
@*/
int TSGetType(TS ts, TSType *method,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
  if (method) *method = (TSType) ts->type;
  if (name)  *name = NRFindName( __TSList, (int) ts->type );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPrintTypes_Private"
/*
   TSPrintTypes_Private - Prints the TS methods available from the 
   options database.

   Input Parameters:
.  comm   - The communicator (usually PETSC_COMM_WORLD)
.  prefix - prefix (usually "-")
.  name   - the options database name (by default "ts_type") 
*/
int TSPrintTypes_Private(MPI_Comm comm,char* prefix,char *name)
{
  FuncList *entry;

  PetscFunctionBegin;
  if (!__TSList) {TSRegisterAll();}
  entry = __TSList->head;
  PetscPrintf(comm," %s%s (one of)",prefix,name);
  while (entry) {
    PetscPrintf(comm," %s",entry->name);
    entry = entry->next;
  }
  PetscPrintf(comm,"\n");
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "TSGetTypeFromOptions_Private"
/*
   TSGetTypeFromOptions_Private - Sets the selected method from the 
   options database.

   Input Parameter:
.  ctx - the TS context

   Output Parameter:
.  method -  solver method
.  flg  - indicates if method found

   Options Database Key:
$  -ts_type  method
*/
int TSGetTypeFromOptions_Private(TS ctx,TSType *method,int *flg)
{
  int  ierr;
  char sbuf[50];

  PetscFunctionBegin;
  ierr = OptionsGetString(ctx->prefix,"-ts_type", sbuf, 50, flg); CHKERRQ(ierr);
  if (*flg) {
    if (!__TSList) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
    *method = (TSType)NRFindID( __TSList, sbuf );
    if (*method == (TSType) -1) SETERRQ(1,1,"Invalid TS Type");
  }
  PetscFunctionReturn(0);
}
