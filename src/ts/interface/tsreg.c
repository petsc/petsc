/*$Id: tsreg.c,v 1.63 2000/09/02 02:49:53 bsmith Exp bsmith $*/

#include "src/ts/tsimpl.h"      /*I "petscts.h"  I*/

FList      TSList              = 0;
PetscTruth TSRegisterAllCalled = PETSC_FALSE;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSSetType"
/*@C
   TSSetType - Sets the method for the timestepping solver.  

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  type - a known method

   Options Database Command:
.  -ts_type <type> - Sets the method; use -help for a list
   of available methods (for instance, euler)

   Notes:
   See "petsc/include/petscts.h" for available methods (for instance)
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
int TSSetType(TS ts,TSType type)
{
  int        ierr,(*r)(TS);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)ts,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested */
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr =  FListFind(ts->comm,TSList,type,(int (**)(void *)) &r);CHKERRQ(ierr);
  if (!r) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown type: %s",type);}

  if (ts->sles) {ierr = SLESDestroy(ts->sles);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESDestroy(ts->snes);CHKERRQ(ierr);}
  if (ts->destroy) {ierr = (*(ts)->destroy)(ts);CHKERRQ(ierr);}
  ts->sles = 0;
  ts->snes = 0;

  ierr = (*r)(ts);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ts,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSRegisterDestroy"
/*@C
   TSRegisterDestroy - Frees the list of timesteppers that were
   registered by FListAddDynamic().

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
    ierr = FListDestroy(&TSList);CHKERRQ(ierr);
    TSList = 0;
  }
  TSRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSGetType"
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
int TSGetType(TS ts,TSType *type)
{
  int ierr;

  PetscFunctionBegin;
  if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  *type = ts->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSSetFromOptions"
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

.seealso: TSSetTypeFromOptions()
@*/
int TSSetFromOptions(TS ts)
{
  int        ierr;
  PetscTruth flg;
  char       *deft,type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);

  ierr = OptionsBegin(ts->comm,ts->prefix,"Time step options","TS");CHKERRQ(ierr);
    if (ts->type_name) {
      deft = ts->type_name;
    } else {  
      deft = TS_EULER;
    }
    if (!TSRegisterAllCalled) {ierr = TSRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr = OptionsList("-ts_type","Timestep method","TSSetType",TSList,deft,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = TSSetType(ts,type);CHKERRQ(ierr);
    } else if (!ts->type_name) {
      ierr = TSSetType(ts,deft);CHKERRQ(ierr);
    }

    ierr = OptionsInt("-ts_max_steps","Maximum number of time steps","TSSetDuration",ts->max_steps,&ts->max_steps,PETSC_NULL);CHKERRQ(ierr);
    ierr = OptionsDouble("-ts_max_time","Time to run to","TSSetDuration",ts->max_time,&ts->max_time,PETSC_NULL);CHKERRQ(ierr);
    ierr = OptionsName("-ts_monitor","Monitor timestep size","TSDefaultMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = TSSetMonitor(ts,TSDefaultMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = OptionsName("-ts_xmonitor","Monitor timestep size graphically","TSLGMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = TSSetMonitor(ts,TSLGMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    if (ts->setfromoptions) {
      ierr = (*ts->setfromoptions)(ts);CHKERRQ(ierr);
    }
  ierr = OptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



