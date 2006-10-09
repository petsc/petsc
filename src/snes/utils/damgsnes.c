#define PETSCSNES_DLL
 
#include "petscda.h"      /*I      "petscda.h"    I*/
#ifdef PETSC_HAVE_SIEVE
#include "petscmesh.h"    /*I      "petscmesh.h"    I*/
#endif
#include "petscmg.h"      /*I      "petscmg.h"    I*/
#include "petscdmmg.h"    /*I      "petscdmmg.h"  I*/

#if defined(PETSC_HAVE_ADIC)
extern PetscErrorCode DMMGComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode DMMGSolveFAS(DMMG*,PetscInt);
extern PetscErrorCode DMMGSolveFAS4(DMMG*,PetscInt);
extern PetscErrorCode DMMGSolveFASb(DMMG*,PetscInt);
extern PetscErrorCode DMMGSolveFAS_NGMRES(DMMG*,PetscInt);
extern PetscErrorCode DMMGComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
#endif

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFCreate_DAAD(NLF*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetDA_DAAD(NLF,DA);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetCtx_DAAD(NLF,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetResidual_DAAD(NLF,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetNewtonIterations_DAAD(NLF,PetscInt);
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
  PetscTruth     ismg;
  Vec            W;
  MatStructure   flg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as user context which should contain DMMG");
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);

  /* compute Jacobian on finest grid */
  if (dmmg[nlevels-1]->updatejacobian && ShouldUpdate(nlevels,it)) {
    ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg));CHKERRQ(ierr);
  } else {
    ierr = PetscInfo3(0,"Skipping Jacobian, SNES iteration %D frequence %D level %D\n",it,dmmg[nlevels-1]->updatejacobianperiod,nlevels-1);CHKERRQ(ierr);
    *flag = SAME_PRECONDITIONER;
  }
  ierr = MatSNESMFSetBase(DMMGGetFine(dmmg)->J,X);CHKERRQ(ierr);

  /* create coarser grid Jacobians for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {
    PetscTruth galerkin;

    ierr = PCMGGetGalerkin(pc,&galerkin);CHKERRQ(ierr);
    ierr = PCMGGetSmoother(pc,nlevels-1,&lksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(lksp,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->B,*flag);CHKERRQ(ierr);

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
	ierr = MatSNESMFSetBase(dmmg[i-1]->J,X);CHKERRQ(ierr);
	/* compute Jacobian on coarse grid */
	if (dmmg[i-1]->updatejacobian && ShouldUpdate(i,it)) {
	  ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,&flg,dmmg[i-1]);CHKERRQ(ierr);
	  flg = SAME_NONZERO_PATTERN;
	} else {
	  ierr = PetscInfo3(0,"Skipping Jacobian, SNES iteration %D frequence %D level %D\n",it,dmmg[i-1]->updatejacobianperiod,i-1);CHKERRQ(ierr);
	  flg = SAME_PRECONDITIONER;
	}
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
  DA             da = (DA)dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAFormFunction1(da,localX,F,dmmg->user);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunctionGhost"
PetscErrorCode DMMGFormFunctionGhost(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  Vec            localX, localF;
  DA             da = (DA)dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da,&localF);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = VecSet(localF, 0.0);CHKERRQ(ierr);
  ierr = DAFormFunction1(da,localX,localF,dmmg->user);CHKERRQ(ierr);
  ierr = DALocalToGlobalBegin(da,localF,F);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(da,localF,F);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#ifdef PETSC_HAVE_SIEVE
#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunctionMesh"
/* 
   DMMGFormFunctionMesh - This is a universal global FormFunction used by the DMMG code
   when the user provides a local function.

   Input Parameters:
+  snes - the SNES context
.  X - input vector
-  ptr - This is the DMMG object

   Output Parameter:
.  F - function vector

 */
PetscErrorCode DMMGFormFunctionMesh(SNES snes, Vec X, Vec F, void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  Mesh           mesh = (Mesh) dmmg->dm;
  SectionReal    sectionX, section;
  Vec            localVec;
  VecScatter     scatter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetSectionReal(mesh, "default", &section);CHKERRQ(ierr);
  ierr = SectionRealDuplicate(section, &sectionX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = MeshGetGlobalScatter(mesh, &scatter);CHKERRQ(ierr);
  ierr = MeshCreateLocalVector(mesh, sectionX, &localVec);CHKERRQ(ierr);
  ierr = VecScatterBegin(X, localVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(X, localVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERRQ(ierr);
  ierr = VecDestroy(localVec);CHKERRQ(ierr);
  ierr = MeshFormFunction(mesh, sectionX, section, dmmg->user);CHKERRQ(ierr);
  ierr = MeshCreateLocalVector(mesh, section, &localVec);CHKERRQ(ierr);
  ierr = VecScatterBegin(localVec, F, INSERT_VALUES, SCATTER_FORWARD, scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(localVec, F, INSERT_VALUES, SCATTER_FORWARD, scatter);CHKERRQ(ierr);
  ierr = VecDestroy(localVec);CHKERRQ(ierr);
  ierr = SectionRealDestroy(sectionX);CHKERRQ(ierr);
  ierr = SectionRealDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianMesh"
/* 
   DMMGComputeJacobianMesh - This is a universal global FormJacobian used by the DMMG code
   when the user provides a local function.

   Input Parameters:
+  snes - the SNES context
.  X - input vector
-  ptr - This is the DMMG object

   Output Parameter:
.  F - function vector

 */
PetscErrorCode DMMGComputeJacobianMesh(SNES snes, Vec X, Mat *J, Mat *B, MatStructure *flag, void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  Mesh           mesh = (Mesh) dmmg->dm;
  SectionReal    sectionX;
  Vec            localVec;
  VecScatter     scatter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetSectionReal(mesh, "default", &sectionX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = MeshGetGlobalScatter(mesh, &scatter);CHKERRQ(ierr);
  ierr = MeshCreateLocalVector(mesh, sectionX, &localVec);CHKERRQ(ierr);
  ierr = VecScatterBegin(X, localVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(X, localVec, INSERT_VALUES, SCATTER_REVERSE, scatter);CHKERRQ(ierr);
  ierr = VecDestroy(localVec);CHKERRQ(ierr);
  ierr = MeshFormJacobian(mesh, sectionX, *B, dmmg->user);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B, MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  ierr = SectionRealDestroy(sectionX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 
#endif

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
  DA             da = (DA)dmmg->dm;
  PetscInt       N,n;
 
  PetscFunctionBegin;
  /* determine whether X=localX */
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);

  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
       DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
    */
    ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DAFormFunction(da,dmmg->lfj,localX,F,dmmg->user);CHKERRQ(ierr);
  if (n != N){
    ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "SNESDAFormFunction"
/*@C 
   SNESDAFormFunction - This is a universal function evaluation routine that
   may be used with SNESSetFunction() as long as the user context has a DA
   as its first record and the user has called DASetLocalFunction().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  F - function vector
-  ptr - pointer to a structure that must have a DA as its first entry. For example this 
         could be a DMMG, this ptr must have been passed into SNESDAFormFunction() as the context

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), DASetLocalAdicFunction(), DASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  DA             da = *(DA*)ptr;
  PetscInt       N,n;
  
  PetscFunctionBegin;
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Looks like you called SNESSetFromFuntion(snes,SNESDAFormFunction,) without the DA context");

  /* determine whether X=localX */
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);
 
  
  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
    */
    ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DAFormFunction1(da,localX,F,ptr);
  if (n != N){
    if (PetscExceptionValue(ierr)) {
      PetscErrorCode pierr = DARestoreLocalVector(da,&localX);CHKERRQ(pierr);
    }
    CHKERRQ(ierr);
    ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
} 

/* ------------------------------------------------------------------------------*/
#include "src/mat/matimpl.h"        /*I "petscmat.h" I*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithFD"
PetscErrorCode DMMGComputeJacobianWithFD(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  MatFDColoring  color = (MatFDColoring)dmmg->fdcoloring;
  
  PetscFunctionBegin;
  if (color->ctype == IS_COLORING_GHOSTED){
    DA            da=(DA)dmmg->dm;
    Vec           x1_loc;
    ierr = DAGetLocalVector(da,&x1_loc);CHKERRQ(ierr);
    ierr = DAGlobalToLocalBegin(da,x1,INSERT_VALUES,x1_loc);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(da,x1,INSERT_VALUES,x1_loc);CHKERRQ(ierr);
    ierr = SNESDefaultComputeJacobianColor(snes,x1_loc,J,B,flag,dmmg->fdcoloring);CHKERRQ(ierr);
    ierr = DARestoreLocalVector(da,&x1_loc);CHKERRQ(ierr);
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
  DA             da = (DA) dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1(da,localX,*B,dmmg->user);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDAComputeJacobianWithAdifor"
/*
    SNESDAComputeJacobianWithAdifor - This is a universal Jacobian evaluation routine
    that may be used with SNESSetJacobian() from Fortran as long as the user context has 
    a DA as its first record and DASetLocalAdiforFunction() has been called.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  J - Jacobian
.  B - Jacobian used in preconditioner (usally same as J)
.  flag - indicates if the matrix changed its structure
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalAdicFunction(), SNESSetFunction(), SNESSetJacobian()

*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobianWithAdifor(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA             da = *(DA*) ptr;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1WithAdifor(da,localX,*B,ptr);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDAComputeJacobian"
/*
   SNESDAComputeJacobian - This is a universal Jacobian evaluation routine for a
   locally provided Jacobian.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  J - Jacobian
.  B - Jacobian used in preconditioner (usally same as J)
.  flag - indicates if the matrix changed its structure
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()

*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA             da = *(DA*) ptr;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1(da,localX,*B,ptr);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
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

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - optional function to compute Jacobian

    Options Database Keys:
+    -dmmg_snes_monitor
.    -dmmg_jacobian_fd
.    -dmmg_jacobian_ad
.    -dmmg_jacobian_mf_fd_operator
.    -dmmg_jacobian_mf_fd
.    -dmmg_jacobian_mf_ad_operator
.    -dmmg_jacobian_mf_ad
.    -dmmg_iscoloring_type
-    -dmmg_jacobian_period <p> - Indicates how often in the SNES solve the Jacobian is recomputed (on all levels)
                                 as suggested by Florin Dobrian if p is -1 then Jacobian is computed only on first
                                 SNES iteration (i.e. -1 is equivalent to infinity) 

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNESLocal()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetSNES(DMMG *dmmg,PetscErrorCode (*function)(SNES,Vec,Vec,void*),PetscErrorCode (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels,period = 1,isctype=0;
  PetscTruth     snesmonitor,mffdoperator,mffd,fdjacobian;
#if defined(PETSC_HAVE_ADIC)
  PetscTruth     mfadoperator,mfad,adjacobian;
#endif
  PetscViewer    ascii;
  MPI_Comm       comm;
  PetscMPIInt    size;
  const char     *isctypes[] = {"IS_COLORING_LOCAL","IS_COLORING_GHOSTED"};
  ISColoringType ctype;

  PetscFunctionBegin;
  if (!dmmg)     SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");
  if (!jacobian) jacobian = DMMGComputeJacobianWithFD;

  ierr = PetscOptionsBegin(dmmg[0]->comm,PETSC_NULL,"DMMG Options","SNES");CHKERRQ(ierr);
    ierr = PetscOptionsName("-dmmg_snes_monitor","Monitor nonlinear convergence","SNESMonitorSet",&snesmonitor);CHKERRQ(ierr);


    ierr = PetscOptionsName("-dmmg_jacobian_fd","Compute sparse Jacobian explicitly with finite differencing","DMMGSetSNES",&fdjacobian);CHKERRQ(ierr);
    if (fdjacobian) jacobian = DMMGComputeJacobianWithFD;
#if defined(PETSC_HAVE_ADIC)
    ierr = PetscOptionsName("-dmmg_jacobian_ad","Compute sparse Jacobian explicitly with ADIC (automatic differentiation)","DMMGSetSNES",&adjacobian);CHKERRQ(ierr);
    if (adjacobian) jacobian = DMMGComputeJacobianWithAdic;
#endif

    ierr = PetscOptionsTruthGroupBegin("-dmmg_jacobian_mf_fd_operator","Apply Jacobian via matrix free finite differencing","DMMGSetSNES",&mffdoperator);CHKERRQ(ierr);
    ierr = PetscOptionsTruthGroupEnd("-dmmg_jacobian_mf_fd","Apply Jacobian via matrix free finite differencing even in computing preconditioner","DMMGSetSNES",&mffd);CHKERRQ(ierr);
    if (mffd) mffdoperator = PETSC_TRUE;
#if defined(PETSC_HAVE_ADIC)
    ierr = PetscOptionsTruthGroupBegin("-dmmg_jacobian_mf_ad_operator","Apply Jacobian via matrix free ADIC (automatic differentiation)","DMMGSetSNES",&mfadoperator);CHKERRQ(ierr);
    ierr = PetscOptionsTruthGroupEnd("-dmmg_jacobian_mf_ad","Apply Jacobian via matrix free ADIC (automatic differentiation) even in computing preconditioner","DMMGSetSNES",&mfad);CHKERRQ(ierr);
    if (mfad) mfadoperator = PETSC_TRUE;
#endif
    ierr = MPI_Comm_size(dmmg[0]->comm,&size);CHKERRQ(ierr);
    if (size == 1){
      isctype = 0;
    } else {
      isctype = 1; 
    }
    ierr = PetscOptionsEList("-dmmg_iscoloring_type","Type of ISColoring","None",isctypes,2,isctypes[isctype],&isctype,PETSC_NULL);CHKERRQ(ierr);
        
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SNESCreate(dmmg[i]->comm,&dmmg[i]->snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(dmmg[i]->snes,&dmmg[i]->ksp);CHKERRQ(ierr);
    if (snesmonitor) {
      ierr = PetscObjectGetComm((PetscObject)dmmg[i]->snes,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(ascii,nlevels-i);CHKERRQ(ierr);
      ierr = SNESMonitorSet(dmmg[i]->snes,SNESMonitorDefault,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }

    if (mffdoperator) {
      ierr = MatCreateSNESMF(dmmg[i]->snes,dmmg[i]->x,&dmmg[i]->J);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work1);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work2);CHKERRQ(ierr);
      ierr = MatSNESMFSetFunction(dmmg[i]->J,dmmg[i]->work1,function,dmmg[i]);CHKERRQ(ierr);
      if (mffd) {
        dmmg[i]->B = dmmg[i]->J;
        jacobian   = DMMGComputeJacobianWithMF;
      }
#if defined(PETSC_HAVE_ADIC)
    } else if (mfadoperator) {
      ierr = MatRegisterDAAD();CHKERRQ(ierr);
      ierr = MatCreateDAAD((DA)dmmg[i]->dm,&dmmg[i]->J);CHKERRQ(ierr);
      ierr = MatDAADSetCtx(dmmg[i]->J,dmmg[i]->user);CHKERRQ(ierr);
      if (mfad) {
        dmmg[i]->B = dmmg[i]->J;
        jacobian   = DMMGComputeJacobianWithMF;
      }
#endif
    }
    
    if (!dmmg[i]->B) {
      ierr = DMGetMatrix(dmmg[i]->dm,dmmg[i]->mtype,&dmmg[i]->B);CHKERRQ(ierr);
    } 
    if (!dmmg[i]->J) {
      dmmg[i]->J = dmmg[i]->B;
    }

    ierr = DMMGSetUpLevel(dmmg,dmmg[i]->ksp,i+1);CHKERRQ(ierr);
    
    /*
       if the number of levels is > 1 then we want the coarse solve in the grid sequencing to use LU
       when possible 
    */
    if (nlevels > 1 && i == 0) {
      PC         pc;
      KSP        cksp;
      PetscTruth flg1,flg2,flg3;

      ierr = KSPGetPC(dmmg[i]->ksp,&pc);CHKERRQ(ierr);
      ierr = PCMGGetCoarseSolve(pc,&cksp);CHKERRQ(ierr);
      ierr = KSPGetPC(cksp,&pc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCILU,&flg1);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCSOR,&flg2);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PETSC_NULL,&flg3);CHKERRQ(ierr);
      if (flg1 || flg2 || flg3) {
        ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      }
    }

    dmmg[i]->solve           = DMMGSolveSNES;
    dmmg[i]->computejacobian = jacobian;
    dmmg[i]->computefunction = function;
  }

  if (jacobian == DMMGComputeJacobianWithFD) {
    ISColoring iscoloring; 
    ctype = (ISColoringType)isctype;

    for (i=0; i<nlevels; i++) {
      ierr = DMGetColoring(dmmg[i]->dm,ctype,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(dmmg[i]->B,iscoloring,&dmmg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      if (function == DMMGFormFunction) function = DMMGFormFunctionFD;
      ierr = MatFDColoringSetFunction(dmmg[i]->fdcoloring,(PetscErrorCode(*)(void))function,dmmg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(dmmg[i]->fdcoloring);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_ADIC)
  } else if (jacobian == DMMGComputeJacobianWithAdic) {
    for (i=0; i<nlevels; i++) {
      ISColoring iscoloring;
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_GHOSTED,&iscoloring);CHKERRQ(ierr);
      ierr = MatSetColoring(dmmg[i]->B,iscoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    }
#endif
  }
  for (i=0; i<nlevels; i++) {
    ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_Multigrid,dmmg);CHKERRQ(ierr);
    ierr = SNESSetFunction(dmmg[i]->snes,dmmg[i]->b,function,dmmg[i]);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(dmmg[i]->snes);CHKERRQ(ierr);
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolationScale(dmmg[i-1]->dm,dmmg[i]->dm,dmmg[i]->R,&dmmg[i]->Rscale);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetInt(PETSC_NULL,"-dmmg_jacobian_period",&period,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    dmmg[i]->updatejacobian       = PETSC_TRUE;
    dmmg[i]->updatejacobianperiod = period;
  }

#if defined(PETSC_HAVE_ADIC)
  { 
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_fas",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscTruth block = PETSC_FALSE;
      PetscTruth ngmres = PETSC_FALSE;
      PetscInt   newton_its;
      ierr = PetscOptionsHasName(0,"-dmmg_fas_view",&flg);CHKERRQ(ierr);
      for (i=0; i<nlevels; i++) {
	ierr = NLFCreate_DAAD(&dmmg[i]->nlf);CHKERRQ(ierr);
	ierr = NLFDAADSetDA_DAAD(dmmg[i]->nlf,(DA)dmmg[i]->dm);CHKERRQ(ierr);
	ierr = NLFDAADSetCtx_DAAD(dmmg[i]->nlf,dmmg[i]->user);CHKERRQ(ierr);
	ierr = NLFDAADSetResidual_DAAD(dmmg[i]->nlf,dmmg[i]->r);CHKERRQ(ierr);
        ierr = VecDuplicate(dmmg[i]->b,&dmmg[i]->w);CHKERRQ(ierr);

        dmmg[i]->monitor    = PETSC_FALSE;
        ierr = PetscOptionsHasName(0,"-dmmg_fas_monitor",&dmmg[i]->monitor);CHKERRQ(ierr);
        dmmg[i]->monitorall = PETSC_FALSE;
        ierr = PetscOptionsHasName(0,"-dmmg_fas_monitor_all",&dmmg[i]->monitorall);CHKERRQ(ierr);
        dmmg[i]->presmooth  = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_presmooth",&dmmg[i]->presmooth,0);CHKERRQ(ierr);
        dmmg[i]->postsmooth = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_postsmooth",&dmmg[i]->postsmooth,0);CHKERRQ(ierr);
        dmmg[i]->coarsesmooth = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_coarsesmooth",&dmmg[i]->coarsesmooth,0);CHKERRQ(ierr);

        dmmg[i]->rtol = 1.e-8;
        ierr = PetscOptionsGetReal(0,"-dmmg_fas_rtol",&dmmg[i]->rtol,0);CHKERRQ(ierr);
        dmmg[i]->abstol = 1.e-50;
        ierr = PetscOptionsGetReal(0,"-dmmg_fas_atol",&dmmg[i]->abstol,0);CHKERRQ(ierr);

        newton_its = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_newton_its",&newton_its,0);CHKERRQ(ierr);
        ierr = NLFDAADSetNewtonIterations_DAAD(dmmg[i]->nlf,newton_its);CHKERRQ(ierr);

        if (flg) {
          if (i == 0) {
            ierr = PetscPrintf(dmmg[i]->comm,"FAS Solver Parameters\n");CHKERRQ(ierr);
            ierr = PetscPrintf(dmmg[i]->comm,"  rtol %G atol %G\n",dmmg[i]->rtol,dmmg[i]->abstol);CHKERRQ(ierr);
	    ierr = PetscPrintf(dmmg[i]->comm,"             coarsesmooths %D\n",dmmg[i]->coarsesmooth);CHKERRQ(ierr);
            ierr = PetscPrintf(dmmg[i]->comm,"             Newton iterations %D\n",newton_its);CHKERRQ(ierr);
          } else {
	    ierr = PetscPrintf(dmmg[i]->comm,"  level %D   presmooths    %D\n",i,dmmg[i]->presmooth);CHKERRQ(ierr);
	    ierr = PetscPrintf(dmmg[i]->comm,"             postsmooths   %D\n",dmmg[i]->postsmooth);CHKERRQ(ierr);
            ierr = PetscPrintf(dmmg[i]->comm,"             Newton iterations %D\n",newton_its);CHKERRQ(ierr);
          }
        }
        ierr = PetscOptionsHasName(0,"-dmmg_fas_block",&block);CHKERRQ(ierr);
        ierr = PetscOptionsHasName(0,"-dmmg_fas_ngmres",&ngmres);CHKERRQ(ierr);
	if (block) {
          dmmg[i]->solve = DMMGSolveFASb;
          if (flg) {
            ierr = PetscPrintf(dmmg[i]->comm,"  using point-block smoothing\n");CHKERRQ(ierr);
          }
	} else if(ngmres) {
          dmmg[i]->solve = DMMGSolveFAS_NGMRES;
          if (flg) {
            ierr = PetscPrintf(dmmg[i]->comm,"  using non-linear gmres\n");CHKERRQ(ierr);
          }
        } else {
          dmmg[i]->solve = DMMGSolveFAS4;
        }
      }
    }
  }
#endif
   
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocalFD"
/*@
    DMMGSetSNESLocalFD - Sets the local user function that is used to approximately compute the Jacobian
        via finite differences.

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   function - the function that defines the nonlinear system

    Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetSNESLocal()

@*/
PetscErrorCode DMMGSetSNESLocalFD(DMMG *dmmg,DALocalFunction1 function)
{
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    dmmg[i]->lfj = (PetscErrorCode (*)(void))function; 
  }
  PetscFunctionReturn(0);
}


/*M
    DMMGSetSNESLocal - Sets the local user function that defines the nonlinear set of equations
    that will use the grid hierarchy and (optionally) its derivative.

    Collective on DMMG

   Synopsis:
   PetscErrorCode DMMGSetSNESLocal(DMMG *dmmg,DALocalFunction1 function, DALocalFunction1 jacobian,
                        DALocalFunction1 ad_function, DALocalFunction1 admf_function);

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
.   jacobian - function defines the local part of the Jacobian
.   ad_function - the name of the function with an ad_ prefix. This is ignored if ADIC is
                  not installed
-   admf_function - the name of the function with an ad_ prefix. This is ignored if ADIC is
                  not installed

    Options Database Keys:
+    -dmmg_snes_monitor
.    -dmmg_jacobian_fd
.    -dmmg_jacobian_ad
.    -dmmg_jacobian_mf_fd_operator
.    -dmmg_jacobian_mf_fd
.    -dmmg_jacobian_mf_ad_operator
.    -dmmg_jacobian_mf_ad
-    -dmmg_jacobian_period <p> - Indicates how often in the SNES solve the Jacobian is recomputed (on all levels)
                                 as suggested by Florin Dobrian if p is -1 then Jacobian is computed only on first
                                 SNES iteration (i.e. -1 is equivalent to infinity) 


    Level: intermediate

    Notes: 
    If ADIC or ADIFOR have been installed, this routine can use ADIC or ADIFOR to compute
    the derivative; however, that function cannot call other functions except those in
    standard C math libraries.

    If ADIC/ADIFOR have not been installed and the Jacobian is not provided, this routine
    uses finite differencing to approximate the Jacobian.

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES()

M*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocal_Private"
PetscErrorCode DMMGSetSNESLocal_Private(DMMG *dmmg,DALocalFunction1 function,DALocalFunction1 jacobian,DALocalFunction1 ad_function,DALocalFunction1 admf_function)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels,cookie;
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = 0;


  PetscFunctionBegin;
  if (jacobian)         computejacobian = DMMGComputeJacobian;
#if defined(PETSC_HAVE_ADIC)
  else if (ad_function) computejacobian = DMMGComputeJacobianWithAdic;
#endif

  ierr = PetscObjectGetCookie((PetscObject) dmmg[0]->dm,&cookie);CHKERRQ(ierr);
  if (cookie == DA_COOKIE) {
    PetscTruth flag;
    ierr = PetscOptionsHasName(PETSC_NULL, "-dmmg_form_function_ghost", &flag);CHKERRQ(ierr);
    if (flag) {
      ierr = DMMGSetSNES(dmmg,DMMGFormFunctionGhost,computejacobian);CHKERRQ(ierr);
    } else {
      ierr = DMMGSetSNES(dmmg,DMMGFormFunction,computejacobian);CHKERRQ(ierr);
    }
    for (i=0; i<nlevels; i++) {
      ierr = DASetLocalFunction((DA)dmmg[i]->dm,function);CHKERRQ(ierr);
      dmmg[i]->lfj = (PetscErrorCode (*)(void))function; 
      ierr = DASetLocalJacobian((DA)dmmg[i]->dm,jacobian);CHKERRQ(ierr);
      ierr = DASetLocalAdicFunction((DA)dmmg[i]->dm,ad_function);CHKERRQ(ierr);
      ierr = DASetLocalAdicMFFunction((DA)dmmg[i]->dm,admf_function);CHKERRQ(ierr);
    }
  } else {
#ifdef PETSC_HAVE_SIEVE
    ierr = DMMGSetSNES(dmmg, DMMGFormFunctionMesh, DMMGComputeJacobianMesh);CHKERRQ(ierr);
    for (i=0; i<nlevels; i++) {
      ierr = MeshSetLocalFunction((Mesh) dmmg[i]->dm, (PetscErrorCode (*)(Mesh,SectionReal,SectionReal,void*)) function);CHKERRQ(ierr);
      dmmg[i]->lfj = (PetscErrorCode (*)(void)) function; 
      ierr = MeshSetLocalJacobian((Mesh) dmmg[i]->dm, (PetscErrorCode (*)(Mesh,SectionReal,Mat,void*)) jacobian);CHKERRQ(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctioni"
PetscErrorCode DMMGFunctioni(PetscInt i,Vec u,PetscScalar* r,void* ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;
  VecScatter     gtol;

  PetscFunctionBegin;
  /* copy u into interior part of U */
  ierr = DAGetScatter((DA)dmmg->dm,0,&gtol,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL,gtol);CHKERRQ(ierr);
  ierr = DAFormFunctioni1((DA)dmmg->dm,i,U,r,dmmg->user);CHKERRQ(ierr);
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
  ierr = DAGetScatter((DA)dmmg->dm,0,&gtol,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL,gtol);CHKERRQ(ierr);
  ierr = DAFormFunctionib1((DA)dmmg->dm,i,U,r,dmmg->user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctioniBase"
PetscErrorCode DMMGFunctioniBase(Vec u,void* ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,u,INSERT_VALUES,U);CHKERRQ(ierr);  
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,u,INSERT_VALUES,U);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocali_Private"
PetscErrorCode DMMGSetSNESLocali_Private(DMMG *dmmg,PetscErrorCode (*functioni)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode (*adi)(DALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode (*adimf)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    ierr = DASetLocalFunctioni((DA)dmmg[i]->dm,functioni);CHKERRQ(ierr);
    ierr = DASetLocalAdicFunctioni((DA)dmmg[i]->dm,adi);CHKERRQ(ierr);
    ierr = DASetLocalAdicMFFunctioni((DA)dmmg[i]->dm,adimf);CHKERRQ(ierr);
    ierr = MatSNESMFSetFunctioni(dmmg[i]->J,DMMGFunctioni);CHKERRQ(ierr);
    ierr = MatSNESMFSetFunctioniBase(dmmg[i]->J,DMMGFunctioniBase);CHKERRQ(ierr);    
    if (!dmmg[i]->lwork1) {
      ierr = DACreateLocalVector((DA)dmmg[i]->dm,&dmmg[i]->lwork1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocalib_Private"
PetscErrorCode DMMGSetSNESLocalib_Private(DMMG *dmmg,PetscErrorCode (*functioni)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode (*adi)(DALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode (*adimf)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    ierr = DASetLocalFunctionib((DA)dmmg[i]->dm,functioni);CHKERRQ(ierr);
    ierr = DASetLocalAdicFunctionib((DA)dmmg[i]->dm,adi);CHKERRQ(ierr);
    ierr = DASetLocalAdicMFFunctionib((DA)dmmg[i]->dm,adimf);CHKERRQ(ierr);
    if (!dmmg[i]->lwork1) {
      ierr = DACreateLocalVector((DA)dmmg[i]->dm,&dmmg[i]->lwork1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode (*localfunc)(DALocalInfo*,void*) = 0;

#undef __FUNCT__  
#define __FUNCT__ "DMMGInitialGuess_Local"
PetscErrorCode PETSCSNES_DLLEXPORT DMMGInitialGuess_Local(DMMG dmmg,Vec x)
{
  PetscErrorCode ierr;
  DALocalInfo    info;
  void           *f;
  DA             da = (DA)dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,x,&f);CHKERRQ(ierr);

  CHKMEMQ;
  ierr = (*localfunc)(&info,f);CHKERRQ(ierr);
  CHKMEMQ;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetInitialGuessLocal"
/*@C
    DMMGSetInitialGuessLocal - sets code to compute the initial guess for each level

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
-   localguess - the function that computes the initial guess

    Level: intermediate

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES(), DMMGSetInitialGuess(), DMMGSetSNESLocal()

@*/
PetscErrorCode DMMGSetInitialGuessLocal(DMMG *dmmg,PetscErrorCode (*localguess)(DALocalInfo*,void*))
{
  PetscErrorCode ierr;


  PetscFunctionBegin;
  localfunc = localguess;  /* stash into ugly static for now */

  ierr = DMMGSetInitialGuess(dmmg,DMMGInitialGuess_Local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


