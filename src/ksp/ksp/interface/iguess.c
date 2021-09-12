#include <petsc/private/kspimpl.h> /*I "petscksp.h"  I*/

PetscFunctionList KSPGuessList = NULL;
static PetscBool KSPGuessRegisterAllCalled;

/*
  KSPGuessRegister -  Adds a method for initial guess computation in Krylov subspace solver package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Notes:
   KSPGuessRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   KSPGuessRegister("my_initial_guess",MyInitialGuessCreate);
.ve

   Then, it can be chosen with the procedural interface via
$     KSPSetGuessType(ksp,"my_initial_guess")
   or at runtime via the option
$     -ksp_guess_type my_initial_guess

   Level: advanced

.seealso: KSPGuess, KSPGuessRegisterAll()

@*/
PetscErrorCode  KSPGuessRegister(const char sname[],PetscErrorCode (*function)(KSPGuess))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&KSPGuessList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  KSPGuessRegisterAll - Registers all KSPGuess implementations in the KSP package.

  Not Collective

  Level: advanced

.seealso: KSPRegisterAll(),  KSPInitializePackage()
*/
PetscErrorCode KSPGuessRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (KSPGuessRegisterAllCalled) PetscFunctionReturn(0);
  KSPGuessRegisterAllCalled = PETSC_TRUE;
  ierr = KSPGuessRegister(KSPGUESSFISCHER,KSPGuessCreate_Fischer);CHKERRQ(ierr);
  ierr = KSPGuessRegister(KSPGUESSPOD,KSPGuessCreate_POD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    KSPGuessSetFromOptions - Sets the options for a KSPGuess from the options database

    Collective on guess

    Input Parameter:
.    guess - KSPGuess object

   Level: intermediate

.seealso: KSPGuess, KSPGetGuess(), KSPSetGuessType(), KSPGuessType
@*/
PetscErrorCode KSPGuessSetFromOptions(KSPGuess guess)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (guess->ops->setfromoptions) { ierr = (*guess->ops->setfromoptions)(guess);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*@
   KSPGuessDestroy - Destroys KSPGuess context.

   Collective on kspGuess

   Input Parameter:
.  guess - initial guess object

   Level: beginner

.seealso: KSPGuessCreate(), KSPGuess, KSPGuessType
@*/
PetscErrorCode  KSPGuessDestroy(KSPGuess *guess)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*guess) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*guess),KSPGUESS_CLASSID,1);
  if (--((PetscObject)(*guess))->refct > 0) {*guess = NULL; PetscFunctionReturn(0);}
  if ((*guess)->ops->destroy) { ierr = (*(*guess)->ops->destroy)(*guess);CHKERRQ(ierr); }
  ierr = MatDestroy(&(*guess)->A);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   KSPGuessView - View the KSPGuess object

   Logically Collective on guess

   Input Parameters:
+  guess  - the initial guess object for the Krylov method
-  viewer - the viewer object

   Notes:

  Level: intermediate

.seealso: KSP, KSPGuess, KSPGuessType, KSPGuessRegister(), KSPGuessCreate(), PetscViewer
@*/
PetscErrorCode  KSPGuessView(KSPGuess guess, PetscViewer view)
{
  PetscErrorCode ierr;
  PetscBool      ascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (!view) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)guess),&view);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(view,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(guess,1,view,2);
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&ascii);CHKERRQ(ierr);
  if (ascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)guess,view);CHKERRQ(ierr);
    if (guess->ops->view) {
      ierr = PetscViewerASCIIPushTab(view);CHKERRQ(ierr);
      ierr = (*guess->ops->view)(guess,view);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(view);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   KSPGuessCreate - Creates the default KSPGuess context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  guess - location to put the KSPGuess context

   Notes:
   The default KSPGuess type is XXX

   Level: beginner

.seealso: KSPSolve(), KSPGuessDestroy(), KSPGuess, KSPGuessType, KSP
@*/
PetscErrorCode  KSPGuessCreate(MPI_Comm comm,KSPGuess *guess)
{
  KSPGuess       tguess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(guess,2);
  *guess = NULL;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(tguess,KSPGUESS_CLASSID,"KSPGuess","Initial guess for Krylov Method","KSPGuess",comm,KSPGuessDestroy,KSPGuessView);CHKERRQ(ierr);
  tguess->omatstate = -1;
  *guess = tguess;
  PetscFunctionReturn(0);
}

/*@C
   KSPGuessSetType - Sets the type of a KSPGuess

   Logically Collective on guess

   Input Parameters:
+  guess - the initial guess object for the Krylov method
-  type  - a known KSPGuess method

   Options Database Key:
.  -ksp_guess_type  <method> - Sets the method; use -help for a list
    of available methods

   Notes:

  Level: intermediate

.seealso: KSP, KSPGuess, KSPGuessType, KSPGuessRegister(), KSPGuessCreate()

@*/
PetscErrorCode  KSPGuessSetType(KSPGuess guess, KSPGuessType type)
{
  PetscErrorCode ierr,(*r)(KSPGuess);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)guess,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(KSPGuessList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)guess),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested KSPGuess type %s",type);
  if (guess->ops->destroy) {
    ierr                = (*guess->ops->destroy)(guess);CHKERRQ(ierr);
    guess->ops->destroy = NULL;
  }
  ierr = PetscMemzero(guess->ops,sizeof(struct _KSPGuessOps));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)guess,type);CHKERRQ(ierr);
  ierr = (*r)(guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   KSPGuessGetType - Gets the KSPGuess type as a string from the KSPGuess object.

   Not Collective

   Input Parameter:
.  guess - the initial guess context

   Output Parameter:
.  name - name of KSPGuess method

   Level: intermediate

.seealso: KSPGuessSetType()
@*/
PetscErrorCode  KSPGuessGetType(KSPGuess guess,KSPGuessType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)guess)->type_name;
  PetscFunctionReturn(0);
}

/*@
    KSPGuessUpdate - Updates the guess object with the current solution and rhs vector

   Collective on guess

   Input Parameters:
+  guess - the initial guess context
.  rhs   - the corresponding rhs
-  sol   - the computed solution

   Level: intermediate

.seealso: KSPGuessCreate(), KSPGuess
@*/
PetscErrorCode  KSPGuessUpdate(KSPGuess guess, Vec rhs, Vec sol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(sol,VEC_CLASSID,3);
  if (guess->ops->update) { ierr = (*guess->ops->update)(guess,rhs,sol);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*@
    KSPGuessFormGuess - Form the initial guess

   Collective on guess

   Input Parameter:
+  guess - the initial guess context
.  rhs   - the current rhs vector
-  sol   - the initial guess vector

   Level: intermediate

.seealso: KSPGuessCreate(), KSPGuess
@*/
PetscErrorCode  KSPGuessFormGuess(KSPGuess guess, Vec rhs, Vec sol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(sol,VEC_CLASSID,3);
  if (guess->ops->formguess) { ierr = (*guess->ops->formguess)(guess,rhs,sol);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*@
    KSPGuessSetUp - Setup the initial guess object

   Collective on guess

   Input Parameter:
-  guess - the initial guess context

   Level: intermediate

.seealso: KSPGuessCreate(), KSPGuess
@*/
PetscErrorCode  KSPGuessSetUp(KSPGuess guess)
{
  PetscErrorCode   ierr;
  PetscObjectState matstate;
  PetscInt         oM = 0, oN = 0, M, N;
  Mat              omat = NULL;
  PC               pc;
  PetscBool        reuse;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  if (guess->A) {
    omat = guess->A;
    ierr = MatGetSize(guess->A,&oM,&oN);CHKERRQ(ierr);
  }
  ierr = KSPGetOperators(guess->ksp,&guess->A,NULL);CHKERRQ(ierr);
  ierr = KSPGetPC(guess->ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetReusePreconditioner(pc,&reuse);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)guess->A);CHKERRQ(ierr);
  ierr = MatGetSize(guess->A,&M,&N);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)guess->A,&matstate);CHKERRQ(ierr);
  if (M != oM || N != oN) {
    ierr = PetscInfo4(guess,"Resetting KSPGuess since matrix sizes have changed (%D != %D, %D != %D)\n",oM,M,oN,N);CHKERRQ(ierr);
  } else if (!reuse && (omat != guess->A || guess->omatstate != matstate)) {
    ierr = PetscInfo1(guess,"Resetting KSPGuess since %s has changed\n",omat != guess->A ? "matrix" : "matrix state");CHKERRQ(ierr);
    if (guess->ops->reset) { ierr = (*guess->ops->reset)(guess);CHKERRQ(ierr); }
  } else if (reuse) {
    ierr = PetscInfo(guess,"Not resettting KSPGuess since reuse preconditioner has been specified\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(guess,"KSPGuess status unchanged\n");CHKERRQ(ierr);
  }
  if (guess->ops->setup) { ierr = (*guess->ops->setup)(guess);CHKERRQ(ierr); }
  guess->omatstate = matstate;
  ierr = MatDestroy(&omat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
