#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsreg.c,v 1.29 1998/01/17 17:38:11 bsmith Exp bsmith $";
#endif

#include "src/ts/tsimpl.h"      /*I "ts.h"  I*/
#include "src/sys/nreg.h"      
#include "pinclude/pviewer.h"
#include <math.h>

static DLList __TSList = 0;
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
  if (ts->type == method) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested */
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
  ierr =  DLFindRoutine( __TSList, (int)method, (char *)0,(int (**)(void *)) &r );CHKERRQ(ierr);
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown method");}
  if (ts->type != TS_UNKNOWN) {
    if (ts->sles) {ierr = SLESDestroy(ts->sles); CHKERRQ(ierr);}
    if (ts->snes) {ierr = SNESDestroy(ts->snes); CHKERRQ(ierr);}
    ierr = (*(ts)->destroy)((PetscObject)ts); CHKERRQ(ierr);
    ts->sles = 0;
    ts->snes = 0;
  }
  ierr = (*r)(ts);CHKERRQ(ierr);

  /* override the type that the create routine put in */
  ts->type = method;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "TSRegister_Private"
/*
   TSRegister_Private - Adds the method to the timestepping package, given 
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
*/
int TSRegister_Private(TSType name, char *sname, char *fname,int (*create)(TS),TSType *oname)
{
  int ierr;

  PetscFunctionBegin;

  if (!__TSList) {ierr = DLCreate((int)TS_NEW,&__TSList); CHKERRQ(ierr);}
  ierr = DLRegister( __TSList, (int) name, sname, fname,(int (*)(void*))create,(int*)oname );
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
  int ierr;

  PetscFunctionBegin;
  if (__TSList) {
    ierr = DLDestroy( __TSList );CHKERRQ(ierr);
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

.keywords: TS, timestepper, get, type, name
@*/
int TSGetType(TS ts, TSType *type,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(); CHKERRQ(ierr);}
  if (type) *type = (TSType) ts->type;
  if (name) {ierr = DLFindName( __TSList, (int) ts->type,name ); CHKERRQ(ierr);}
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
  (*PetscHelpPrintf)(ts->comm,"TS options --------------------------------------------------\n");
  ierr = DLPrintTypes(ts->comm,stdout,ts->prefix,"ts_type",__TSList);CHKERRQ(ierr);
  (*PetscHelpPrintf)(ts->comm," %sts_monitor: use default TS monitor\n",prefix);
  (*PetscHelpPrintf)(ts->comm," %sts_view: view TS info after each solve\n",prefix);

  (*PetscHelpPrintf)(ts->comm," %sts_max_steps <steps>: maximum steps, defaults to %d\n",prefix,ts->max_steps);
  (*PetscHelpPrintf)(ts->comm," %sts_max_time <steps>: maximum time, defaults to %g\n",prefix,ts->max_time);
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
  TSType type;
  char   fname[256];

  PetscFunctionBegin;
  loc[0] = PETSC_DECIDE; loc[1] = PETSC_DECIDE; loc[2] = 300; loc[3] = 300;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->setup_called) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call prior to TSSetUp!");
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll();CHKERRQ(ierr);}
  ierr = DLGetTypeFromOptions(ts->prefix,"-ts_type",__TSList,(int *)&type,fname,256,&flg);CHKERRQ(ierr);
  if (flg) {
#if defined(USE_DYNAMIC_LIBRARIES)
    if (type == (TSType) -1) { /* indicates method not yet registered */
      ierr = TSRegister(TS_NEW,fname,fname,0,&type); CHKERRQ(ierr);
    }
#endif
    ierr = TSSetType(ts,type); CHKERRQ(ierr);
  }

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
  if (ts->type == TS_UNKNOWN) {
    ierr = TSSetType(ts,TS_EULER);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg)  {ierr = TSPrintHelp(ts);CHKERRQ(ierr);}
  if (!ts->setfromoptions) PetscFunctionReturn(0);
  ierr = (*ts->setfromoptions)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

