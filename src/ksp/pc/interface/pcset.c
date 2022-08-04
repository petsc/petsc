
/*
    Routines to set PC methods and options.
*/

#include <petsc/private/pcimpl.h>      /*I "petscpc.h" I*/
#include <petscdm.h>

PetscBool PCRegisterAllCalled = PETSC_FALSE;
/*
   Contains the list of registered PC routines
*/
PetscFunctionList PCList = NULL;

/*@C
   PCSetType - Builds PC for a particular preconditioner type

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context.
-  type - a known method

   Options Database Key:
.  -pc_type <type> - Sets PC type

   Use -help for a list of available methods (for instance,
   jacobi or bjacobi)

  Notes:
  See "petsc/include/petscpc.h" for available methods (for instance,
  PCJACOBI, PCILU, or PCBJACOBI).

  Normally, it is best to use the KSPSetFromOptions() command and
  then set the PC type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many different preconditioners.
  The PCSetType() routine is provided for those situations where it
  is necessary to set the preconditioner independently of the command
  line or options database.  This might be the case, for example, when
  the choice of preconditioner changes during the execution of the
  program, and the user's application is taking responsibility for
  choosing the appropriate preconditioner.  In other words, this
  routine is not for beginners.

  Level: intermediate

  Developer Note: PCRegister() is used to add preconditioner types to PCList from which they
  are accessed by PCSetType().

.seealso: `KSPSetType()`, `PCType`, `PCRegister()`, `PCCreate()`, `KSPGetPC()`

@*/
PetscErrorCode  PCSetType(PC pc,PCType type)
{
  PetscBool      match;
  PetscErrorCode (*r)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(type,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)pc,type,&match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(PCList,type,&r));
  PetscCheck(r,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PC type %s",type);
  /* Destroy the previous private PC context */
  if (pc->ops->destroy) {
    PetscCall((*pc->ops->destroy)(pc));
    pc->ops->destroy = NULL;
    pc->data         = NULL;
  }
  PetscCall(PetscFunctionListDestroy(&((PetscObject)pc)->qlist));
  /* Reinitialize function pointers in PCOps structure */
  PetscCall(PetscMemzero(pc->ops,sizeof(struct _PCOps)));
  /* XXX Is this OK?? */
  pc->modifysubmatrices  = NULL;
  pc->modifysubmatricesP = NULL;
  /* Call the PCCreate_XXX routine for this particular preconditioner */
  pc->setupcalled = 0;

  PetscCall(PetscObjectChangeTypeName((PetscObject)pc,type));
  PetscCall((*r)(pc));
  PetscFunctionReturn(0);
}

/*@C
   PCGetType - Gets the PC method type and name (as a string) from the PC
   context.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  type - name of preconditioner method

   Level: intermediate

.seealso: `PCSetType()`

@*/
PetscErrorCode  PCGetType(PC pc,PCType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)pc)->type_name;
  PetscFunctionReturn(0);
}

extern PetscErrorCode PCGetDefaultType_Private(PC,const char*[]);

/*@
   PCSetFromOptions - Sets PC options from the options database.
   This routine must be called before PCSetUp() if the user is to be
   allowed to set the preconditioner method.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Options Database:
.   -pc_use_amat true,false - see PCSetUseAmat()

   Level: developer

.seealso: `PCSetUseAmat()`

@*/
PetscErrorCode  PCSetFromOptions(PC pc)
{
  char           type[256];
  const char     *def;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);

  PetscCall(PCRegisterAll());
  PetscObjectOptionsBegin((PetscObject)pc);
  if (!((PetscObject)pc)->type_name) {
    PetscCall(PCGetDefaultType_Private(pc,&def));
  } else {
    def = ((PetscObject)pc)->type_name;
  }

  PetscCall(PetscOptionsFList("-pc_type","Preconditioner","PCSetType",PCList,def,type,256,&flg));
  if (flg) {
    PetscCall(PCSetType(pc,type));
  } else if (!((PetscObject)pc)->type_name) {
    PetscCall(PCSetType(pc,def));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCNONE,&flg));
  if (flg) goto skipoptions;

  PetscCall(PetscOptionsBool("-pc_use_amat","use Amat (instead of Pmat) to define preconditioner in nested inner solves","PCSetUseAmat",pc->useAmat,&pc->useAmat,NULL));

  if (pc->ops->setfromoptions) PetscCall((*pc->ops->setfromoptions)(PetscOptionsObject,pc));

  skipoptions:
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)pc));
  PetscOptionsEnd();
  pc->setfromoptionscalled++;
  PetscFunctionReturn(0);
}

/*@
   PCSetDM - Sets the DM that may be used by some preconditioners

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  dm - the dm, can be NULL

   Level: intermediate

   Developer Notes:
    The routines KSP/SNES/TSSetDM() require the dm to be non-NULL, but this one can be NULL since all it does is
    replace the current DM

.seealso: `PCGetDM()`, `KSPSetDM()`, `KSPGetDM()`
@*/
PetscErrorCode  PCSetDM(PC pc,DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (dm) PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(DMDestroy(&pc->dm));
  pc->dm = dm;
  PetscFunctionReturn(0);
}

/*@
   PCGetDM - Gets the DM that may be used by some preconditioners

   Not Collective

   Input Parameter:
. pc - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate

.seealso: `PCSetDM()`, `KSPSetDM()`, `KSPGetDM()`
@*/
PetscErrorCode  PCGetDM(PC pc,DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *dm = pc->dm;
  PetscFunctionReturn(0);
}

/*@
   PCSetApplicationContext - Sets the optional user-defined context for the linear solver.

   Logically Collective on PC

   Input Parameters:
+  pc - the PC context
-  usrP - optional user context

   Level: intermediate

.seealso: `PCGetApplicationContext()`
@*/
PetscErrorCode  PCSetApplicationContext(PC pc,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  pc->user = usrP;
  PetscFunctionReturn(0);
}

/*@
   PCGetApplicationContext - Gets the user-defined context for the linear solver.

   Not Collective

   Input Parameter:
.  pc - PC context

   Output Parameter:
.  usrP - user context

   Level: intermediate

.seealso: `PCSetApplicationContext()`
@*/
PetscErrorCode  PCGetApplicationContext(PC pc,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *(void**)usrP = pc->user;
  PetscFunctionReturn(0);
}
