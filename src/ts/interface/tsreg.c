

#ifndef lint
static char vcid[] = "$Id: tsreg.c,v 1.11 1996/11/07 19:00:08 curfman Exp balay $";
#endif

#include "src/ts/tsimpl.h"      /*I "ts.h"  I*/
#include "src/sys/nreg.h"      
#include "pinclude/pviewer.h"
#include <math.h>

static NRList *__TSList = 0;

#undef __FUNCTION__  
#define __FUNCTION__ "TSSetType"
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
  int (*r)(TS);

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  /* Get the function pointers for the method requested */
  if (!__TSList) {TSRegisterAll();}
  if (!__TSList) {SETERRQ(1,"TSSetType:Could not get methods");}
  r =  (int (*)(TS))NRFindRoutine( __TSList, (int)method, (char *)0 );
  if (!r) {SETERRQ(1,"TSSetType:Unknown method");}
  if (ts->data) PetscFree(ts->data);
  return (*r)(ts);
}

/* --------------------------------------------------------------------- */
#undef __FUNCTION__  
#define __FUNCTION__ "TSRegister"
/*@C
   TSRegister - Adds the method to the nonlinear solver package, given 
   a function pointer and a nonlinear solver name of the type TSType.

   Input Parameters:
.  name - for instance TS_EQ_NLS, TS_EQ_NTR, ...
.  sname - corfunPonding string for name
.  create - routine to create method context

.keywords: TS, nonlinear, register

.seealso: TSRegisterAll(), TSRegisterDestroy()
@*/
int TSRegister(int name, char *sname, int (*create)(TS))
{
  int ierr;
  if (!__TSList) {ierr = NRCreate(&__TSList); CHKERRQ(ierr);}
  NRRegister( __TSList, name, sname, (int (*)(void*))create );
  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNCTION__  
#define __FUNCTION__ "TSRegisterDestroy"
/*@C
   TSRegisterDestroy - Frees the list of nonlinear solvers that were
   registered by TSRegister().

.keywords: TS, nonlinear, register, destroy

.seealso: TSRegisterAll(), TSRegisterAll()
@*/
int TSRegisterDestroy()
{
  if (__TSList) {
    NRDestroy( __TSList );
    __TSList = 0;
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "TSGetType"
/*@C
   TSGetType - Gets the TS method type and name (as a string).

   Input Parameter:
.  ts - nonlinear solver context

   Output Parameter:
.  method - TS method (or use PETSC_NULL)
.  name - name of TS method (or use PETSC_NULL)

.keywords: TS, nonlinear, get, method, name
@*/
int TSGetType(TS ts, TSType *method,char **name)
{
  int ierr;
  if (!__TSList) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
  if (method) *method = (TSType) ts->type;
  if (name)  *name = NRFindName( __TSList, (int) ts->type );
  return 0;
}

#include <stdio.h>
#undef __FUNCTION__  
#define __FUNCTION__ "TSPrintTypes_Private"
/*
   TSPrintTypes_Private - Prints the TS methods available from the 
   options database.

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  prefix - prefix (usually "-")
.  name   - the options database name (by default "ts_type") 
*/
int TSPrintTypes_Private(MPI_Comm comm,char* prefix,char *name)
{
  FuncList *entry;
  if (!__TSList) {TSRegisterAll();}
  entry = __TSList->head;
  PetscPrintf(comm," %s%s (one of)",prefix,name);
  while (entry) {
    PetscPrintf(comm," %s",entry->name);
    entry = entry->next;
  }
  PetscPrintf(comm,"\n");
  return 0;
}


#undef __FUNCTION__  
#define __FUNCTION__ "TSGetTypeFromOptions_Private"
/*
   TSGetTypeFromOptions_Private - Sets the selected method from the 
   options database.

   Input Parameter:
.  ctx - the TS context

   Output Parameter:
.  method -  solver method

   Returns:
   Returns 1 if the method is found; 0 otherwise.

   Options Database Key:
$  -ts_type  method
*/
int TSGetTypeFromOptions_Private(TS ctx,TSType *method,int *flg)
{
  int ierr;
  char sbuf[50];
  ierr = OptionsGetString(ctx->prefix,"-ts_type", sbuf, 50, flg); CHKERRQ(ierr);
  if (*flg) {
    if (!__TSList) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
    *method = (TSType)NRFindID( __TSList, sbuf );
  }
  return 0;
}
