#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsreg.c,v 1.25 1997/10/28 14:24:05 bsmith Exp bsmith $";
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
#define __FUNC__ "TSPrintHelp"
/*@
   TSPrintHelp - Prints all options for the TS (timestepping) component.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
$  -help, -h

.keywords: TS, timestep, print, help

.seealso: TSSetFromOptions()
@*/
int TSPrintHelp(TS ts)
{
  char    *prefix = "-";
  int     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->prefix) prefix = ts->prefix;
  PetscPrintf(ts->comm,"TS options --------------------------------------------------\n");
  ierr = NRPrintTypes(ts->comm,stdout,ts->prefix,"ts_type",__TSList);CHKERRQ(ierr);
  PetscPrintf(ts->comm," %sts_monitor: use default TS monitor\n",prefix);
  PetscPrintf(ts->comm," %sts_view: view TS info after each solve\n",prefix);

  PetscPrintf(ts->comm," %sts_max_steps <steps>: maximum steps, defaults to %d\n",prefix,ts->max_steps);
  PetscPrintf(ts->comm," %sts_max_time <steps>: maximum time, defaults to %g\n",prefix,ts->max_time);
  if (ts->printhelp) {ierr = (*ts->printhelp)(ts,prefix);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetFromOptions"
/*@
   TSSetFromOptions - Sets various TS parameters from user options.

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

.keywords: TS, timestep, set, options, database

.seealso: TSPrintHelp()
@*/
int TSSetFromOptions(TS ts)
{
  int    ierr,flg,loc[4],nmax;
  TSType method;

  PetscFunctionBegin;
  loc[0] = PETSC_DECIDE; loc[1] = PETSC_DECIDE; loc[2] = 300; loc[3] = 300;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->setup_called) SETERRQ(1,0,"Call prior to TSSetUp!");
  if (!__TSList) {ierr = TSRegisterAll();CHKERRQ(ierr);}
  ierr = NRGetTypeFromOptions(ts->prefix,"-ts_type",__TSList,&method,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TSSetType(ts,method); CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg)  {ierr = TSPrintHelp(ts);CHKERRQ(ierr);}
  ierr = OptionsGetInt(ts->prefix,"-ts_max_steps",&ts->max_steps,&flg);CHKERRQ(ierr);
  ierr = OptionsGetDouble(ts->prefix,"-ts_max_time",&ts->max_time,&flg);CHKERRQ(ierr);
  ierr = OptionsHasName(ts->prefix,"-ts_monitor",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = TSSetMonitor(ts,TSDefaultMonitor,0);CHKERRQ(ierr);
  }
  nmax = 4;
  ierr = OptionsGetIntArray(ts->prefix,"-ts_xmonitor",loc,&nmax,&flg); CHKERRQ(ierr);
  if (flg) {
    int    rank = 0;
    DrawLG lg;
    MPI_Comm_rank(ts->comm,&rank);
    if (!rank) {
      ierr = TSLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); CHKERRQ(ierr);
      PLogObjectParent(ts,(PetscObject) lg);
      ierr = TSSetMonitor(ts,TSLGMonitor,(void *)lg);CHKERRQ(ierr);
    }
  }
  if (!ts->setfromoptions) PetscFunctionReturn(0);
  ierr = (*ts->setfromoptions)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

