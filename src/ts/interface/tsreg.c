#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsreg.c,v 1.34 1998/04/03 23:16:43 bsmith Exp bsmith $";
#endif

#include "src/ts/tsimpl.h"      /*I "ts.h"  I*/
#include "src/sys/nreg.h"      
#include "pinclude/pviewer.h"
#include <math.h>

DLList TSList = 0;
int TSRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "TSSetType"
/*@C
   TSSetType - Sets the method for the timestepping solver.  

   Input Parameters:
.  ts - the TS context
.  method - a known method

   Collective on TS

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
  if (!PetscStrcmp(ts->type_name,method)) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested */
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL); CHKERRQ(ierr);}
  ierr =  DLRegisterFind(ts->comm, TSList, method, (int (**)(void *)) &r );CHKERRQ(ierr);
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown method");}

  if (ts->sles) {ierr = SLESDestroy(ts->sles); CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESDestroy(ts->snes); CHKERRQ(ierr);}
  if (ts->destroy) {ierr = (*(ts)->destroy)(ts); CHKERRQ(ierr);}
  ts->sles = 0;
  ts->snes = 0;

  ierr = (*r)(ts);CHKERRQ(ierr);

  if (ts->type_name) PetscFree(ts->type_name);
  ts->type_name = (char *) PetscMalloc((PetscStrlen(method)+1)*sizeof(char));CHKPTRQ(ts->type_name);
  PetscStrcpy(ts->type_name,method);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "TSRegisterDestroy"
/*@C
   TSRegisterDestroy - Frees the list of timesteppers that were
   registered by DLRegister().

   Not Collective

.keywords: TS, timestepper, register, destroy

.seealso: TSRegisterAll()
@*/
int TSRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (TSList) {
    ierr = DLRegisterDestroy( TSList );CHKERRQ(ierr);
    TSList = 0;
  }
  TSRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetType"
/*@C
   TSGetType - Gets the TS method type (as a string).

   Input Parameter:
.  ts - timestepper solver context

   Output Parameter:
.  type - name of TS method

   Not Collective

.keywords: TS, timestepper, get, type, name
@*/
int TSGetType(TS ts, TSType *type)
{
  int ierr;

  PetscFunctionBegin;
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL); CHKERRQ(ierr);}
  *type = ts->type_name;
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

   Collective on TS

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
  ierr = DLRegisterPrintTypes(ts->comm,stdout,ts->prefix,"ts_type",TSList);CHKERRQ(ierr);
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

   Collective on TS

.keywords: TS, timestep, set, options, database

.seealso: TSPrintHelp()
@*/
int TSSetFromOptions(TS ts)
{
  int    ierr,flg,loc[4],nmax;
  char   type[256];

  PetscFunctionBegin;
  loc[0] = PETSC_DECIDE; loc[1] = PETSC_DECIDE; loc[2] = 300; loc[3] = 300;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call prior to TSSetUp!");
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = OptionsGetString(ts->prefix,"-ts_type",(char *) type,256,&flg);
  if (flg) {
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
  if (!ts->type_name) {
    ierr = TSSetType(ts,TS_EULER);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg)  {ierr = TSPrintHelp(ts);CHKERRQ(ierr);}
  if (!ts->setfromoptions) PetscFunctionReturn(0);
  ierr = (*ts->setfromoptions)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


