#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsreg.c,v 1.44 1999/04/16 16:10:38 bsmith Exp bsmith $";
#endif

#include "src/ts/tsimpl.h"      /*I "ts.h"  I*/

FList TSList              = 0;
int   TSRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "TSSetType"
/*@C
   TSSetType - Sets the method for the timestepping solver.  

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  method - a known method

   Options Database Command:
.  -ts_type <method> - Sets the method; use -help for a list
   of available methods (for instance, euler)

   Notes:
   See "petsc/include/ts.h" for available methods (for instance)
+  TS_EULER - Euler
.  TS_PVODE - PVODE interface
.  TS_BEULER - Backward Euler
-  TS_PSEUDO - Pseudo-timestepping

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
   not for beginners.

   Level: intermediate

.keywords: TS, set, type
@*/
int TSSetType(TS ts,TSType method)
{
  int ierr,(*r)(TS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (PetscTypeCompare(ts->type_name,method)) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested */
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL); CHKERRQ(ierr);}
  ierr =  FListFind(ts->comm, TSList, method, (int (**)(void *)) &r );CHKERRQ(ierr);
  if (!r) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown method: %s",method);}

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
   registered by FListAdd().

   Not Collective

   Level: advanced

.keywords: TS, timestepper, register, destroy

.seealso: TSRegisterAll()
@*/
int TSRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (TSList) {
    ierr = FListDestroy( TSList );CHKERRQ(ierr);
    TSList = 0;
  }
  TSRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSGetType"
/*@C
   TSGetType - Gets the TS method type (as a string).

   Not Collective

   Input Parameter:
.  ts - timestepper solver context

   Output Parameter:
.  type - name of TS method

   Level: intermediate

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

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
+  -help - Prints KSP options
-  -h - Prints KSP options

   Level: beginner

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
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL); CHKERRQ(ierr);}
  ierr = (*PetscHelpPrintf)(ts->comm,"TS options --------------------------------------------------\n");CHKERRQ(ierr);
  ierr = FListPrintTypes(ts->comm,stdout,ts->prefix,"ts_type",TSList);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," %sts_monitor: use default TS monitor\n",prefix);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," %sts_view: view TS info after each solve\n",prefix);CHKERRQ(ierr);

  ierr = (*PetscHelpPrintf)(ts->comm," %sts_max_steps <steps>: maximum steps, defaults to %d\n",prefix,ts->max_steps);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ts->comm," %sts_max_time <steps>: maximum time, defaults to %g\n",prefix,ts->max_time);CHKERRQ(ierr);
  if (ts->printhelp) {ierr = (*ts->printhelp)(ts,prefix);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetTypeFromOptions"
/*@
   TSSetTypeFromOptions - Sets the TS type from the options database; sets 
     a default if none is given.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
.  -ts_type <type> - TS_EULER, TS_BEULER, TS_PVODE, TS_PSEUDO, TS_CRANK_NICHOLSON

   Level: beginner

.keywords: TS, timestep, set, options, database, TS type

.seealso: TSPrintHelp(), TSSetFromOptions()
@*/
int TSSetTypeFromOptions(TS ts)
{
  int  ierr,flg;
  char type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  if (ts->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call prior to TSSetUp()");
  ierr = OptionsGetString(ts->prefix,"-ts_type",(char *) type,256,&flg);
  if (flg) {
    ierr = TSSetType(ts,type); CHKERRQ(ierr);
  }
  if (!ts->type_name) {
    ierr = TSSetType(ts,TS_EULER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetFromOptions"
/*@
   TSSetFromOptions - Sets various TS parameters from user options.

   Collective on TS

   Input Parameter:
.  ts - the TS context obtained from TSCreate()

   Options Database Keys:
+  -ts_type <type> - TS_EULER, TS_BEULER, TS_PVODE, TS_PSEUDO, TS_CRANK_NICHOLSON
.  -ts_max_steps maxsteps - maximum number of time-steps to take
.  -ts_max_time time - maximum time to compute to
.  -ts_monitor - print information at each timestep
-  -ts_xmonitor - plot information at each timestep

   Level: beginner

.keywords: TS, timestep, set, options, database

.seealso: TSPrintHelp(), TSSetTypeFromOptions()
@*/
int TSSetFromOptions(TS ts)
{
  int    ierr,flg,loc[4],nmax;

  PetscFunctionBegin;
  loc[0] = PETSC_DECIDE; loc[1] = PETSC_DECIDE; loc[2] = 300; loc[3] = 300;

  PetscValidHeaderSpecific(ts,TS_COOKIE);
  ierr = TSSetTypeFromOptions(ts);CHKERRQ(ierr);

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
    ierr = MPI_Comm_rank(ts->comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = TSLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); CHKERRQ(ierr);
      PLogObjectParent(ts,(PetscObject) lg);
      ierr = TSSetMonitor(ts,TSLGMonitor,(void *)lg);CHKERRQ(ierr);
    }
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg)  {ierr = TSPrintHelp(ts);CHKERRQ(ierr);}
  if (!ts->setfromoptions) PetscFunctionReturn(0);
  ierr = (*ts->setfromoptions)(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


