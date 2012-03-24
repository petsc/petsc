
#include <petscdmda.h>      /*I      "petscdmda.h"    I*/
#include <petscdmmesh.h>    /*I      "petscdmmesh.h"  I*/
#include <petscdmcomplex.h> /*I      "petscdmcomplex.h" I*/
#include <petsc-private/daimpl.h> 
/* It appears that preprocessor directives are not respected by bfort */
#include <petscpcmg.h>      /*I      "petscpcmg.h"    I*/
#include <petscdmmg.h>    /*I      "petscdmmg.h"  I*/

#if defined(PETSC_HAVE_ADIC)
extern PetscErrorCode DMMGComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode DMMGSolveFAS(DMMG*,PetscInt);
extern PetscErrorCode DMMGSolveFAS4(DMMG*,PetscInt);
extern PetscErrorCode DMMGSolveFASb(DMMG*,PetscInt);
extern PetscErrorCode DMMGSolveFAS_NGMRES(DMMG*,PetscInt);
extern PetscErrorCode DMMGComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
#endif

EXTERN_C_BEGIN
extern PetscErrorCode  NLFCreate_DAAD(NLF*);
extern PetscErrorCode  NLFDAADSetDA_DAAD(NLF,DM);
extern PetscErrorCode  NLFDAADSetCtx_DAAD(NLF,void*);
extern PetscErrorCode  NLFDAADSetResidual_DAAD(NLF,Vec);
extern PetscErrorCode  NLFDAADSetNewtonIterations_DAAD(NLF,PetscInt);
EXTERN_C_END

/*
      period of -1 indicates update only on zeroth iteration of SNES
*/
#define ShouldUpdate(l,it) (((dmmg[l-1]->updatejacobianperiod == -1) && (it == 0)) || \
                            ((dmmg[l-1]->updatejacobianperiod >   0) && !(it % dmmg[l-1]->updatejacobianperiod)))
/*
   Evaluates the Jacobian on all of the grids. It is used by DMMG to provide the 
   ComputeJacobian() function that SNESSetJacobian() requires.
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_Multigrid"
PetscErrorCode DMMGComputeJacobian_Multigrid(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           *dmmg = (DMMG*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels,it;
  KSP            ksp,lksp;
  PC             pc;
  PetscBool      ismg,galerkin = PETSC_FALSE;
  Vec            W;
  MatStructure   flg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as user context which should contain DMMG");
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);

  /* compute Jacobian on finest grid */
  if (dmmg[nlevels-1]->updatejacobian && ShouldUpdate(nlevels,it)) {
    ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg));CHKERRQ(ierr);
  } else {
    ierr = PetscInfo3(0,"Skipping Jacobian, SNES iteration %D frequence %D level %D\n",it,dmmg[nlevels-1]->updatejacobianperiod,nlevels-1);CHKERRQ(ierr);
    *flag = SAME_PRECONDITIONER;
  }
  ierr = MatMFFDSetBase(DMMGGetFine(dmmg)->J,X,PETSC_NULL);CHKERRQ(ierr);

  /* create coarser grid Jacobians for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {
    ierr = PCMGGetGalerkin(pc,&galerkin);CHKERRQ(ierr);
  }

  if (!galerkin) {
    for (i=nlevels-1; i>0; i--) {
      if (!dmmg[i-1]->w) {
	ierr = VecDuplicate(dmmg[i-1]->x,&dmmg[i-1]->w);CHKERRQ(ierr);
      }
      W    = dmmg[i-1]->w;
      /* restrict X to coarser grid */
      ierr = MatRestrict(dmmg[i]->R,X,W);CHKERRQ(ierr);
      X    = W;      
      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(X,X,dmmg[i]->Rscale);CHKERRQ(ierr);
      /* tell the base vector for matrix free multiplies */
      ierr = MatMFFDSetBase(dmmg[i-1]->J,X,PETSC_NULL);CHKERRQ(ierr);
      /* compute Jacobian on coarse grid */
      if (dmmg[i-1]->updatejacobian && ShouldUpdate(i,it)) {
	ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,&flg,dmmg[i-1]);CHKERRQ(ierr);
	flg = SAME_NONZERO_PATTERN;
      } else {
	ierr = PetscInfo3(0,"Skipping Jacobian, SNES iteration %D frequence %D level %D\n",it,dmmg[i-1]->updatejacobianperiod,i-1);CHKERRQ(ierr);
	flg = SAME_PRECONDITIONER;
      }
      if (ismg) {
	ierr = PCMGGetSmoother(pc,i-1,&lksp);CHKERRQ(ierr);
	ierr = KSPSetOperators(lksp,dmmg[i-1]->J,dmmg[i-1]->B,flg);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------*/


#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunction"
/* 
   DMMGFormFunction - This is a universal global FormFunction used by the DMMG code
   when the user provides a local function.

   Input Parameters:
+  snes - the SNES context
.  X - input vector
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

 */
PetscErrorCode DMMGFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DM             da = dmmg->dm;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAComputeFunction1(da,localX,F,dmmg->user);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunctionGhost"
PetscErrorCode DMMGFormFunctionGhost(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  Vec            localX, localF;
  DM             da = dmmg->dm;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localF);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);CHKERRQ(ierr);
  ierr = DMDAComputeFunction1(da,localX,localF,dmmg->user);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunctionFD"
/* 
   DMMGFormFunctionFD - This is a universal global FormFunction used by the DMMG code
   when the user provides a local function used to compute the Jacobian via FD.

   Input Parameters:
+  snes - the SNES context
.  X - input vector
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

 */
PetscErrorCode DMMGFormFunctionFD(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DM             da = dmmg->dm;
  PetscInt       N,n;
 
  PetscFunctionBegin;
  /* determine whether X=localX */
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);

  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
       DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMDAComputeFunction(da,dmmg->lfj,localX,F,dmmg->user);CHKERRQ(ierr);
  if (n != N){
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
} 

#include <petsc-private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESDMDAComputeFunction"
/*@C 
   SNESDMDAComputeFunction - This is a universal function evaluation routine that
   may be used with SNESSetFunction(). Must be used with SNESSetDM().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  F - function vector
-  ptr - pointer to a user context containing problem specific information used by application function

   Level: intermediate

.seealso: DMDASetLocalFunction(), DMDASetLocalJacobian(), DMDASetLocalAdicFunction(), DMDASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian(), SNESSetDM()

@*/
PetscErrorCode  SNESDMDAComputeFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  DM             da = snes->dm;
  PetscInt       N,n;
  
  PetscFunctionBegin;
  if (!da) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Looks like you called SNESSetFuntion(snes,SNESDMDAComputeFunction,) without also calling SNESSetDM() to set the DMDA context");

  /* determine whether X=localX */
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);
 
  
  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMDAComputeFunction1(da,localX,F,ptr);CHKERRQ(ierr);
  if (n != N){
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
} 

#if defined(PETSC_HAVE_SIEVE)
#undef __FUNCT__
#define __FUNCT__ "SNESDMMeshComputeFunction"
/*@C
  SNESDMMeshComputeFunction - This is a universal function evaluation routine that
  may be used with SNESSetFunction() as long as the user context has a DMMesh
  as its first record and the user has called DMMeshSetLocalFunction().

  Collective on SNES

  Input Parameters:
+ snes - the SNES context
. X - input vector
. F - function vector
- ptr - pointer to a structure that must have a DMMesh as its first entry.
        This ptr must have been passed into SNESDMMeshComputeFunction() as the context.

  Level: intermediate

.seealso: DMMeshSetLocalFunction(), DMMeshSetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()
@*/
PetscErrorCode SNESDMMeshComputeFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  DM               dm = *(DM*) ptr;
  PetscErrorCode (*lf)(DM, Vec, Vec, void *);
  Vec              localX, localF;
  PetscInt         N, n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  if (!dm) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "Looks like you called SNESSetFromFuntion(snes,SNESDMMeshComputeFunction,) without the DMMesh context");
  PetscValidHeaderSpecific(dm, DM_CLASSID, 4);

  /* determine whether X = localX */
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localF);CHKERRQ(ierr);
  ierr = VecGetSize(X, &N);CHKERRQ(ierr);
  ierr = VecGetSize(localX, &n);CHKERRQ(ierr);

  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMMeshGetLocalFunction(dm, &lf);CHKERRQ(ierr);
  ierr = (*lf)(dm, localX, localF, ptr);CHKERRQ(ierr);
  if (n != N){
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(dm, localF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_SIEVE)
#undef __FUNCT__
#define __FUNCT__ "SNESDMMeshComputeJacobian"
/*
  SNESDMMeshComputeJacobian - This is a universal Jacobian evaluation routine that
  may be used with SNESSetJacobian() as long as the user context has a DMMesh
  as its first record and the user has called DMMeshSetLocalJacobian().

  Collective on SNES

  Input Parameters:
+ snes - the SNES context
. X - input vector
. J - Jacobian
. B - Jacobian used in preconditioner (usally same as J)
. flag - indicates if the matrix changed its structure
- ptr - pointer to a structure that must have a DMMesh as its first entry.
        This ptr must have been passed into SNESDMMeshComputeFunction() as the context.

  Level: intermediate

.seealso: DMMeshSetLocalFunction(), DMMeshSetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()
*/
PetscErrorCode SNESDMMeshComputeJacobian(SNES snes, Vec X, Mat *J, Mat *B, MatStructure *flag, void *ptr)
{
  DM               dm = *(DM*) ptr;
  PetscErrorCode (*lj)(DM, Vec, Mat, void *);
  Vec              localX;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMMeshGetLocalJacobian(dm, &lj);CHKERRQ(ierr);
  ierr = (*lj)(dm, localX, *B, ptr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "SNESDMComplexComputeFunction"
/*@C
  SNESDMComplexComputeFunction - This is a universal function evaluation routine that
  may be used with SNESSetFunction() as long as the user context has a DMComplex
  as its first record and the user has called DMComplexSetLocalFunction().

  Collective on SNES

  Input Parameters:
+ snes - the SNES context
. X - input vector
. F - function vector
- ptr - pointer to a structure that must have a DMComplex as its first entry.
        This ptr must have been passed into SNESDMComplexComputeFunction() as the context.

  Level: intermediate

.seealso: DMComplexSetLocalFunction(), DMComplexSetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()
@*/
PetscErrorCode SNESDMComplexComputeFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  DM               dm = *(DM*) ptr;
  PetscErrorCode (*lf)(DM, Vec, Vec, void *);
  Vec              localX, localF;
  PetscInt         N, n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  if (!dm) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "Looks like you called SNESSetFromFuntion(snes,SNESDMComplexComputeFunction,) without the DMComplex context");
  PetscValidHeaderSpecific(dm, DM_CLASSID, 4);

  /* determine whether X = localX */
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localF);CHKERRQ(ierr);
  ierr = VecGetSize(X, &N);CHKERRQ(ierr);
  ierr = VecGetSize(localX, &n);CHKERRQ(ierr);

  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMComplexGetLocalFunction(dm, &lf);CHKERRQ(ierr);
  ierr = (*lf)(dm, localX, localF, ptr);CHKERRQ(ierr);
  if (n != N){
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(dm, localF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDMComplexComputeJacobian"
/*
  SNESDMComplexComputeJacobian - This is a universal Jacobian evaluation routine that
  may be used with SNESSetJacobian() as long as the user context has a DMComplex
  as its first record and the user has called DMComplexSetLocalJacobian().

  Collective on SNES

  Input Parameters:
+ snes - the SNES context
. X - input vector
. J - Jacobian
. B - Jacobian used in preconditioner (usally same as J)
. flag - indicates if the matrix changed its structure
- ptr - pointer to a structure that must have a DMComplex as its first entry.
        This ptr must have been passed into SNESDMComplexComputeFunction() as the context.

  Level: intermediate

.seealso: DMComplexSetLocalFunction(), DMComplexSetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()
*/
PetscErrorCode SNESDMComplexComputeJacobian(SNES snes, Vec X, Mat *J, Mat *B, MatStructure *flag, void *ptr)
{
  DM               dm = *(DM*) ptr;
  PetscErrorCode (*lj)(DM, Vec, Mat, Mat, void *);
  Vec              localX;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMComplexGetLocalJacobian(dm, &lj);CHKERRQ(ierr);
  ierr = (*lj)(dm, localX, *J, *B, ptr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#include <petsc-private/matimpl.h>        /*I "petscmat.h" I*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithFD"
PetscErrorCode DMMGComputeJacobianWithFD(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  MatFDColoring  color = (MatFDColoring)dmmg->fdcoloring;
  
  PetscFunctionBegin;
  if (color->ctype == IS_COLORING_GHOSTED){
    DM            da=dmmg->dm;
    Vec           x1_loc;
    ierr = DMGetLocalVector(da,&x1_loc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da,x1,INSERT_VALUES,x1_loc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,x1,INSERT_VALUES,x1_loc);CHKERRQ(ierr);
    ierr = SNESDefaultComputeJacobianColor(snes,x1_loc,J,B,flag,dmmg->fdcoloring);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da,&x1_loc);CHKERRQ(ierr);
  } else {
    ierr = SNESDefaultComputeJacobianColor(snes,x1,J,B,flag,dmmg->fdcoloring);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithMF"
PetscErrorCode DMMGComputeJacobianWithMF(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian"
/*
    DMMGComputeJacobian - Evaluates the Jacobian when the user has provided
    a local function evaluation routine.
*/
PetscErrorCode DMMGComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DM             da =  dmmg->dm;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAComputeJacobian1(da,localX,*B,dmmg->user);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDMDAComputeJacobianWithAdifor"
/*
    SNESDMDAComputeJacobianWithAdifor - This is a universal Jacobian evaluation routine
    that may be used with SNESSetJacobian() from Fortran as long as the user context has 
    a DMDA as its first record and DMDASetLocalAdiforFunction() has been called.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  J - Jacobian
.  B - Jacobian used in preconditioner (usally same as J)
.  flag - indicates if the matrix changed its structure
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Level: intermediate

.seealso: DMDASetLocalFunction(), DMDASetLocalAdicFunction(), SNESSetFunction(), SNESSetJacobian()

*/
PetscErrorCode  SNESDMDAComputeJacobianWithAdifor(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DM             da = *(DM*) ptr;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAComputeJacobian1WithAdifor(da,localX,*B,ptr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDMDAComputeJacobian"
/*
   SNESDMDAComputeJacobian - This is a universal Jacobian evaluation routine for a
   locally provided Jacobian. Must be used with SNESSetDM().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  J - Jacobian
.  B - Jacobian used in preconditioner (usally same as J)
.  flag - indicates if the matrix changed its structure
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Level: intermediate

.seealso: DMDASetLocalFunction(), DMDASetLocalJacobian(), SNESSetFunction(), SNESSetJacobian(), SNESSetDM()

*/
PetscErrorCode  SNESDMDAComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DM             da = snes->dm;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAComputeJacobian1(da,localX,*B,ptr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveSNES"
PetscErrorCode DMMGSolveSNES(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  dmmg[0]->nlevels = level+1;
  ierr             = SNESSolve(dmmg[level]->snes,PETSC_NULL,dmmg[level]->x);CHKERRQ(ierr);
  dmmg[0]->nlevels = nlevels;
  PetscFunctionReturn(0);
}

/* ===========================================================================================================*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNES"
/*@C
    DMMGSetSNES - Sets the nonlinear function that defines the nonlinear set of equations
    to be solved using the grid hierarchy.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - optional function to compute Jacobian

    Options Database Keys:
+    -snes_monitor
.    -dmmg_coloring_from_mat - use graph coloring on the actual matrix nonzero structure instead of getting the coloring from the DM
.    -dmmg_jacobian_fd
.    -dmmg_jacobian_ad
.    -dmmg_jacobian_mf_fd_operator
.    -dmmg_jacobian_mf_fd
.    -dmmg_jacobian_mf_ad_operator
.    -dmmg_jacobian_mf_ad
.    -dmmg_iscoloring_type
-
                                 The period at which the Jacobian is recomputed can be set differently for different levels
                                 of the Jacobian (for example lag all Jacobians except on the finest level).
                                 There is no user interface currently for setting a different period on the different levels, one must set the
                                 fields dmmg[i]->updatejacobian and dmmg[i]->updatejacobianperiod directly in the DMMG data structure.
                                 

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNESLocal(), DMMGSetFromOptions()

@*/
PetscErrorCode  DMMGSetSNES(DMMG *dmmg,PetscErrorCode (*function)(SNES,Vec,Vec,void*),PetscErrorCode (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  PetscErrorCode          ierr;
  PetscInt                i,nlevels = dmmg[0]->nlevels;
  PetscBool               mffdoperator,mffd,fdjacobian;
  PetscBool               useFAS = PETSC_FALSE, fasBlock=PETSC_FALSE, fasGMRES=PETSC_FALSE;
  PetscBool               monitor, monitorAll;
  PetscInt                fasPresmooth = 1, fasPostsmooth = 1, fasCoarsesmooth = 1, fasMaxIter = 2;
  PetscReal               fasRtol = 1.0e-8, fasAbstol = 1.0e-50;
#if defined(PETSC_HAVE_ADIC)
  PetscBool               mfadoperator,mfad,adjacobian;
#endif
  PetscClassId            classid;

  PetscFunctionBegin;
  if (!dmmg)     SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");
  if (!jacobian) jacobian = DMMGComputeJacobianWithFD;
  ierr = PetscObjectGetClassId((PetscObject) dmmg[0]->dm, &classid);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(dmmg[0]->comm,dmmg[0]->prefix,"DMMG Options","SNES");CHKERRQ(ierr);
    ierr = PetscOptionsName("-dmmg_monitor","Monitor DMMG iterations","DMMG",&monitor);CHKERRQ(ierr);
    ierr = PetscOptionsName("-dmmg_monitor_all","Monitor all DMMG iterations","DMMG",&monitorAll);CHKERRQ(ierr);
    /*
    ierr = PetscOptionsBool("-dmmg_fas","Use the Full Approximation Scheme","DMMGSetSNES",useFAS,&useFAS,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-dmmg_fas_block","Use point-block smoothing","DMMG",&fasBlock);CHKERRQ(ierr);
    ierr = PetscOptionsName("-dmmg_fas_ngmres","Use Nonlinear GMRES","DMMG",&fasGMRES);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dmmg_fas_presmooth","Number of downward smoother iterates","DMMG",fasPresmooth,&fasPresmooth,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dmmg_fas_postsmooth","Number of upward smoother iterates","DMMG",fasPostsmooth,&fasPostsmooth,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dmmg_fas_coarsesmooth","Number of coarse smoother iterates","DMMG",fasCoarsesmooth,&fasCoarsesmooth,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dmmg_fas_rtol","Relative tolerance for FAS","DMMG",fasRtol,&fasRtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dmmg_fas_atol","Absolute tolerance for FAS","DMMG",fasAbstol,&fasAbstol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dmmg_fas_max_its","Maximum number of iterates per smoother","DMMG",fasMaxIter,&fasMaxIter,PETSC_NULL);CHKERRQ(ierr);
    */

    ierr = PetscOptionsBool("-dmmg_coloring_from_mat","Compute the coloring directly from the matrix nonzero structure","DMMGSetSNES",dmmg[0]->getcoloringfrommat,&dmmg[0]->getcoloringfrommat,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsName("-dmmg_jacobian_fd","Compute sparse Jacobian explicitly with finite differencing","DMMGSetSNES",&fdjacobian);CHKERRQ(ierr);
    if (fdjacobian) jacobian = DMMGComputeJacobianWithFD;
#if defined(PETSC_HAVE_ADIC)
    ierr = PetscOptionsName("-dmmg_jacobian_ad","Compute sparse Jacobian explicitly with ADIC (automatic differentiation)","DMMGSetSNES",&adjacobian);CHKERRQ(ierr);
    if (adjacobian) jacobian = DMMGComputeJacobianWithAdic;
#endif

    ierr = PetscOptionsBoolGroupBegin("-dmmg_jacobian_mf_fd_operator","Apply Jacobian via matrix free finite differencing","DMMGSetSNES",&mffdoperator);CHKERRQ(ierr);
    ierr = PetscOptionsBoolGroupEnd("-dmmg_jacobian_mf_fd","Apply Jacobian via matrix free finite differencing even in computing preconditioner","DMMGSetSNES",&mffd);CHKERRQ(ierr);
    if (mffd) mffdoperator = PETSC_TRUE;
#if defined(PETSC_HAVE_ADIC)
    ierr = PetscOptionsBoolGroupBegin("-dmmg_jacobian_mf_ad_operator","Apply Jacobian via matrix free ADIC (automatic differentiation)","DMMGSetSNES",&mfadoperator);CHKERRQ(ierr);
    ierr = PetscOptionsBoolGroupEnd("-dmmg_jacobian_mf_ad","Apply Jacobian via matrix free ADIC (automatic differentiation) even in computing preconditioner","DMMGSetSNES",&mfad);CHKERRQ(ierr);
    if (mfad) mfadoperator = PETSC_TRUE;
#endif
    ierr = PetscOptionsEnum("-dmmg_iscoloring_type","Type of ISColoring","None",ISColoringTypes,(PetscEnum)dmmg[0]->isctype,(PetscEnum*)&dmmg[0]->isctype,PETSC_NULL);CHKERRQ(ierr);
        
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SNESCreate(dmmg[i]->comm,&dmmg[i]->snes);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)dmmg[i]->snes,PETSC_NULL,nlevels - i - 1);CHKERRQ(ierr);
    ierr = SNESSetFunction(dmmg[i]->snes,dmmg[i]->b,function,dmmg[i]);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(dmmg[i]->snes,dmmg[i]->prefix);CHKERRQ(ierr);
    ierr = SNESGetKSP(dmmg[i]->snes,&dmmg[i]->ksp);CHKERRQ(ierr);

    if (mffdoperator) {
      ierr = MatCreateSNESMF(dmmg[i]->snes,&dmmg[i]->J);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work1);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work2);CHKERRQ(ierr);
      ierr = MatMFFDSetFunction(dmmg[i]->J,(PetscErrorCode (*)(void*, Vec,Vec))SNESComputeFunction,dmmg[i]->snes);CHKERRQ(ierr);
      if (mffd) {
        dmmg[i]->B = dmmg[i]->J;
        jacobian   = DMMGComputeJacobianWithMF;
        ierr = PetscObjectReference((PetscObject) dmmg[i]->B);CHKERRQ(ierr);
      }
#if defined(PETSC_HAVE_ADIC)
    } else if (mfadoperator) {
      ierr = MatRegisterDAAD();CHKERRQ(ierr);
      ierr = MatCreateDAAD(dmmg[i]->dm,&dmmg[i]->J);CHKERRQ(ierr);
      ierr = MatDAADSetCtx(dmmg[i]->J,dmmg[i]->user);CHKERRQ(ierr);
      if (mfad) {
        dmmg[i]->B = dmmg[i]->J;
        jacobian   = DMMGComputeJacobianWithMF;
        ierr = PetscObjectReference((PetscObject) dmmg[i]->B);CHKERRQ(ierr);
      }
#endif
    }
    
    if (!useFAS) {
      if (!dmmg[i]->B) {
        ierr = DMCreateMatrix(dmmg[i]->dm,dmmg[i]->mtype,&dmmg[i]->B);CHKERRQ(ierr);
      } 
      if (!dmmg[i]->J) {
        dmmg[i]->J = dmmg[i]->B;
        ierr = PetscObjectReference((PetscObject) dmmg[i]->B);CHKERRQ(ierr);
      }
    }

    ierr = DMMGSetUpLevel(dmmg,dmmg[i]->ksp,i+1);CHKERRQ(ierr);
    
    /*
       if the number of levels is > 1 then we want the coarse solve in the grid sequencing to use LU
       when possible 
    */
    if (nlevels > 1 && i == 0) {
      PC          pc;
      PetscMPIInt size;
      MPI_Comm    comm;
      PetscBool   flg1,flg2,flg3;
      KSP         cksp;

      ierr = KSPGetPC(dmmg[i]->ksp,&pc);CHKERRQ(ierr);
      ierr = PCMGGetCoarseSolve(pc,&cksp);CHKERRQ(ierr);  
      ierr = KSPGetPC(cksp,&pc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCILU,&flg1);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCSOR,&flg2);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PETSC_NULL,&flg3);CHKERRQ(ierr);
      if (flg1 || flg2 || flg3) {
        ierr = KSPSetType(dmmg[i]->ksp,KSPPREONLY);CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject) pc,&comm);CHKERRQ(ierr);
        ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
        if (size > 1) {
          ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
        } else {
          ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
        }
      }
    }

    dmmg[i]->computejacobian = jacobian;
    if (useFAS) {
#if defined(PETSC_HAVE_ADIC)
      if (fasBlock) {
        dmmg[i]->solve     = DMMGSolveFASb;
      } else if(fasGMRES) {
        dmmg[i]->solve     = DMMGSolveFAS_NGMRES;
      } else {
        dmmg[i]->solve     = DMMGSolveFAS4;
      }
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Must use ADIC for structured FAS.");
#endif
    } else {
      dmmg[i]->solve         = DMMGSolveSNES;
    }
  }

  if (jacobian == DMMGComputeJacobianWithFD) {
    ISColoring iscoloring;

    for (i=0; i<nlevels; i++) {
      if (dmmg[0]->getcoloringfrommat) {
        ierr = MatGetColoring(dmmg[i]->B,(MatColoringType)MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
      } else {
        ierr = DMCreateColoring(dmmg[i]->dm,dmmg[0]->isctype,dmmg[i]->mtype,&iscoloring);CHKERRQ(ierr);
      }
      ierr = MatFDColoringCreate(dmmg[i]->B,iscoloring,&dmmg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      if (function == DMMGFormFunction) function = DMMGFormFunctionFD;
      ierr = MatFDColoringSetFunction(dmmg[i]->fdcoloring,(PetscErrorCode(*)(void))function,dmmg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(dmmg[i]->fdcoloring);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_ADIC)
  } else if (jacobian == DMMGComputeJacobianWithAdic) {
    for (i=0; i<nlevels; i++) {
      ISColoring iscoloring;
      ierr = DMCreateColoring(dmmg[i]->dm,IS_COLORING_GHOSTED,dmmg[i]->mtype,&iscoloring);CHKERRQ(ierr);
      ierr = MatSetColoring(dmmg[i]->B,iscoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    }
#endif
  }
  if (!useFAS) {
    for (i=0; i<nlevels; i++) {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_Multigrid,dmmg);CHKERRQ(ierr);
    }
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DMCreateInterpolationScale(dmmg[i-1]->dm,dmmg[i]->dm,dmmg[i]->R,&dmmg[i]->Rscale);CHKERRQ(ierr);
  }

  if (useFAS) {
    PetscBool  flg;

    for(i = 0; i < nlevels; i++) {
      ierr = VecDuplicate(dmmg[i]->b,&dmmg[i]->w);CHKERRQ(ierr);
      if (classid == DM_CLASSID) {
#if defined(PETSC_HAVE_ADIC)
        ierr = NLFCreate_DAAD(&dmmg[i]->nlf);CHKERRQ(ierr);
        ierr = NLFDAADSetDA_DAAD(dmmg[i]->nlf,dmmg[i]->dm);CHKERRQ(ierr);
        ierr = NLFDAADSetCtx_DAAD(dmmg[i]->nlf,dmmg[i]->user);CHKERRQ(ierr);
        ierr = NLFDAADSetResidual_DAAD(dmmg[i]->nlf,dmmg[i]->r);CHKERRQ(ierr);
        ierr = NLFDAADSetNewtonIterations_DAAD(dmmg[i]->nlf,fasMaxIter);CHKERRQ(ierr);
#endif
      } else {
      }

      dmmg[i]->monitor      = monitor;
      dmmg[i]->monitorall   = monitorAll;
      dmmg[i]->presmooth    = fasPresmooth;
      dmmg[i]->postsmooth   = fasPostsmooth;
      dmmg[i]->coarsesmooth = fasCoarsesmooth;
      dmmg[i]->rtol         = fasRtol;
      dmmg[i]->abstol       = fasAbstol;
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(0, "-dmmg_fas_view", &flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      for(i = 0; i < nlevels; i++) {
        if (i == 0) {
          ierr = PetscPrintf(dmmg[i]->comm,"FAS Solver Parameters\n");CHKERRQ(ierr);
          ierr = PetscPrintf(dmmg[i]->comm,"  rtol %G atol %G\n",dmmg[i]->rtol,dmmg[i]->abstol);CHKERRQ(ierr);
          ierr = PetscPrintf(dmmg[i]->comm,"             coarsesmooths %D\n",dmmg[i]->coarsesmooth);CHKERRQ(ierr);
          ierr = PetscPrintf(dmmg[i]->comm,"             SNES iterations %D\n",fasMaxIter);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(dmmg[i]->comm,"  level %D   presmooths    %D\n",i,dmmg[i]->presmooth);CHKERRQ(ierr);
          ierr = PetscPrintf(dmmg[i]->comm,"             postsmooths   %D\n",dmmg[i]->postsmooth);CHKERRQ(ierr);
          ierr = PetscPrintf(dmmg[i]->comm,"             SNES iterations %D\n",fasMaxIter);CHKERRQ(ierr);
        }
        if (fasBlock) {
          ierr = PetscPrintf(dmmg[i]->comm,"  using point-block smoothing\n");CHKERRQ(ierr);
        } else if(fasGMRES) {
          ierr = PetscPrintf(dmmg[i]->comm,"  using non-linear gmres\n");CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetFromOptions"
/*@C
    DMMGSetFromOptions - Sets various options associated with the DMMG object

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on DMMG

    Input Parameter:
.   dmmg - the context


    Notes: this is currently only supported for use with DMMGSetSNES() NOT DMMGSetKSP()

           Most options are checked in DMMGSetSNES(); this does call SNESSetFromOptions()

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNESLocal(), DMMGSetSNES()

@*/
PetscErrorCode  DMMGSetFromOptions(DMMG *dmmg)
{
  PetscErrorCode          ierr;
  PetscInt                i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as DMMG");

  for (i=0; i<nlevels; i++) {
    ierr = SNESSetFromOptions(dmmg[i]->snes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocalFD"
/*@C
    DMMGSetSNESLocalFD - Sets the local user function that is used to approximately compute the Jacobian
        via finite differences.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   function - the function that defines the nonlinear system

    Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetSNESLocal()

@*/
PetscErrorCode DMMGSetSNESLocalFD(DMMG *dmmg,DMDALocalFunction1 function)
{
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    dmmg[i]->lfj = (PetscErrorCode (*)(void))function; 
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocal_Private"
/*@C
  DMMGGetSNESLocal - Returns the local functions for residual and Jacobian evaluation.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

  Not Collective

  Input Parameter:
. dmmg - the context

  Output Parameters:
+ function - the function that defines the nonlinear system
- jacobian - function defines the local part of the Jacobian

  Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetSNESLocal()
@*/
PetscErrorCode DMMGGetSNESLocal(DMMG *dmmg,DMDALocalFunction1 *function, DMDALocalFunction1 *jacobian)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetLocalFunction(dmmg[0]->dm, function);CHKERRQ(ierr);
  ierr = DMDAGetLocalJacobian(dmmg[0]->dm, jacobian);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
    DMMGSetSNESLocal - Sets the local user function that defines the nonlinear set of equations
    that will use the grid hierarchy and (optionally) its derivative.

   Synopsis:
   PetscErrorCode DMMGSetSNESLocal(DMMG *dmmg,DMDALocalFunction1 function, DMDALocalFunction1 jacobian,
                        DMDALocalFunction1 ad_function, DMDALocalFunction1 admf_function);

    Logically Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
.   jacobian - function defines the local part of the Jacobian
.   ad_function - the name of the function with an ad_ prefix. This is ignored currently
-   admf_function - the name of the function with an ad_ prefix. This is ignored currently

    Options Database Keys:
+    -dmmg_jacobian_fd
.    -dmmg_jacobian_ad
.    -dmmg_jacobian_mf_fd_operator
.    -dmmg_jacobian_mf_fd
.    -dmmg_jacobian_mf_ad_operator
-    -dmmg_jacobian_mf_ad


    Level: intermediate

    Notes: 
    If the Jacobian is not provided, this routine
    uses finite differencing to approximate the Jacobian.

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES()

M*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocal_Private"
PetscErrorCode DMMGSetSNESLocal_Private(DMMG *dmmg,DMDALocalFunction1 function,DMDALocalFunction1 jacobian,DMDALocalFunction1 ad_function,DMDALocalFunction1 admf_function)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  const char    *typeName;
  PetscBool      ismesh;
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = 0;


  PetscFunctionBegin;
  if (jacobian)         computejacobian = DMMGComputeJacobian;
#if defined(PETSC_HAVE_ADIC)
  else if (ad_function) computejacobian = DMMGComputeJacobianWithAdic;
#endif
  CHKMEMQ;
  ierr = PetscObjectGetType((PetscObject) dmmg[0]->dm, &typeName);CHKERRQ(ierr);
  ierr = PetscStrcmp(typeName, DMMESH, &ismesh);CHKERRQ(ierr);
  if (ismesh)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Unstructured grids no longer supported since DMMG will be phased out");
  else {
    PetscBool  flag;
    /* it makes no sense to use an option to decide on ghost, it depends on whether the 
       formfunctionlocal computes ghost values in F or not. */
    ierr = PetscOptionsHasName(PETSC_NULL, "-dmmg_form_function_ghost", &flag);CHKERRQ(ierr);
    if (flag) {
      ierr = DMMGSetSNES(dmmg,DMMGFormFunctionGhost,computejacobian);CHKERRQ(ierr);
    } else {
      ierr = DMMGSetSNES(dmmg,DMMGFormFunction,computejacobian);CHKERRQ(ierr);
    }
    for (i=0; i<nlevels; i++) {
      dmmg[i]->isctype  = IS_COLORING_GHOSTED;   /* switch to faster version since have local function evaluation */
      ierr = DMDASetLocalFunction(dmmg[i]->dm,function);CHKERRQ(ierr);
      dmmg[i]->lfj = (PetscErrorCode (*)(void))function;
      ierr = DMDASetLocalJacobian(dmmg[i]->dm,jacobian);CHKERRQ(ierr);
      ierr = DMDASetLocalAdicFunction(dmmg[i]->dm,ad_function);CHKERRQ(ierr);
      ierr = DMDASetLocalAdicMFFunction(dmmg[i]->dm,admf_function);CHKERRQ(ierr);
    }
  }
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctioni"
PetscErrorCode DMMGFunctioni(void* ctx,PetscInt i,Vec u,PetscScalar* r)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;
  VecScatter     gtol;

  PetscFunctionBegin;
  /* copy u into interior part of U */
  ierr = DMDAGetScatter(dmmg->dm,0,&gtol,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(gtol,u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL);CHKERRQ(ierr);
  ierr = VecScatterEnd(gtol,u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL);CHKERRQ(ierr);
  ierr = DMDAComputeFunctioni1(dmmg->dm,i,U,r,dmmg->user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctionib"
PetscErrorCode DMMGFunctionib(PetscInt i,Vec u,PetscScalar* r,void* ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;
  VecScatter     gtol;

  PetscFunctionBegin;
  /* copy u into interior part of U */
  ierr = DMDAGetScatter(dmmg->dm,0,&gtol,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(gtol,u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL);CHKERRQ(ierr);
  ierr = VecScatterEnd(gtol,u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL);CHKERRQ(ierr);
  ierr = DMDAComputeFunctionib1(dmmg->dm,i,U,r,dmmg->user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctioniBase"
PetscErrorCode DMMGFunctioniBase(void* ctx,Vec u)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalBegin(dmmg->dm,u,INSERT_VALUES,U);CHKERRQ(ierr);  
  ierr = DMGlobalToLocalEnd(dmmg->dm,u,INSERT_VALUES,U);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocali_Private"
PetscErrorCode DMMGSetSNESLocali_Private(DMMG *dmmg,PetscErrorCode (*functioni)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode (*adi)(DMDALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode (*adimf)(DMDALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    ierr = DMDASetLocalFunctioni(dmmg[i]->dm,functioni);CHKERRQ(ierr);
    ierr = DMDASetLocalAdicFunctioni(dmmg[i]->dm,adi);CHKERRQ(ierr);
    ierr = DMDASetLocalAdicMFFunctioni(dmmg[i]->dm,adimf);CHKERRQ(ierr);
    ierr = MatMFFDSetFunctioni(dmmg[i]->J,DMMGFunctioni);CHKERRQ(ierr);
    ierr = MatMFFDSetFunctioniBase(dmmg[i]->J,DMMGFunctioniBase);CHKERRQ(ierr);    
    if (!dmmg[i]->lwork1) {
      ierr = DMCreateLocalVector(dmmg[i]->dm,&dmmg[i]->lwork1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocalib_Private"
PetscErrorCode DMMGSetSNESLocalib_Private(DMMG *dmmg,PetscErrorCode (*functioni)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode (*adi)(DMDALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode (*adimf)(DMDALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    ierr = DMDASetLocalFunctionib(dmmg[i]->dm,functioni);CHKERRQ(ierr);
    ierr = DMDASetLocalAdicFunctionib(dmmg[i]->dm,adi);CHKERRQ(ierr);
    ierr = DMDASetLocalAdicMFFunctionib(dmmg[i]->dm,adimf);CHKERRQ(ierr);
    if (!dmmg[i]->lwork1) {
      ierr = DMCreateLocalVector(dmmg[i]->dm,&dmmg[i]->lwork1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetISColoringType"
/*@C
    DMMGSetISColoringType - type of coloring used to compute Jacobian via finite differencing

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Logically Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   isctype - IS_COLORING_GHOSTED (default) or IS_COLORING_GLOBAL

    Options Database:
.   -dmmg_iscoloring_type <ghosted or global>

    Notes: ghosted coloring requires using DMMGSetSNESLocal()

    Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetInitialGuess(), DMMGSetSNESLocal()

@*/
PetscErrorCode DMMGSetISColoringType(DMMG *dmmg,ISColoringType isctype)
{
  PetscFunctionBegin;
  dmmg[0]->isctype = isctype;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMMGSetUp"
/*@C
    DMMGSetUp - Called after DMMGSetSNES() and (optionally) DMMGSetFromOptions()

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

    Collective on DMMG

    Input Parameter:
.   dmmg - the context

    Notes: Currently this must be called by the user (if they want to).

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy(), DMMG, DMMGSetSNES(), DMMGSetKSP(), DMMGSolve(), DMMGSetMatType()

@*/
PetscErrorCode  DMMGSetUp(DMMG *dmmg)
{
  PetscErrorCode ierr;
  KSP            ksp;
  SNES           snes;
  PC             pc;

  PetscFunctionBegin;
  snes = DMMGGetSNES(dmmg);
  if (snes) {
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  } else {
    ksp = DMMGGetKSP(dmmg);
  }
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetDM(pc,DMMGGetDM(dmmg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
