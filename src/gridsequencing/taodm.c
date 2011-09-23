#include "petscdm.h"            /*I "petscdm.h"   I*/
#include "petscksp.h"           /*I "petscksp.h"  I*/
#include "private/pcimpl.h"     /*I "petscpc.h"   I*/
#include "private/taosolver_impl.h" /*I "taosolver.h" I*/
#include "private/taodm_impl.h" /*I "taodm.h" I*/

PetscClassId TAODM_CLASSID;
static PetscBool taodmclass_registered = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "TaoDMCreate"
/*@C
    TaoDMCreate - Creates a DM based multigrid solver object. This allows one to 
      easily implement MG methods on regular grids.

    Collective on MPI_Comm

    Input Parameter:
+   comm - the processors that will share the grids and solution process
.   nlevels - number of multigrid levels (if this is negative it CANNOT be reset with -taodm_nlevels
-   user - an optional user context

    Output Parameters:
.    - the TaoDM context

    Options Database:
+     -taodm_nlevels <levels> - number of levels to use
-     -taodm_mat_type <type> - matrix type that TaoDM should create, defaults to MATAIJ

    Notes:
      To provide a different user context for each level call TaoDMSetUser() after calling
      this routine

    Level: advanced

.seealso TaoDMDestroy(), TaoDMSetUser(), TaoDMGetUser(), TaoDMSetMatType(),  TaoDMSetNullSpace(), TaoDMSetInitialGuessRoutine(),
         TaoDMSetISColoringType()

@*/
PetscErrorCode  TaoDMCreate(MPI_Comm comm,PetscInt nlevels,void *user,TaoDM **taodm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  TaoDM           *p;
  PetscBool     ftype;
  char           mtype[256];

  PetscFunctionBegin;
  if (nlevels < 0) {
    nlevels = -nlevels;
  } else {
    ierr = PetscOptionsGetInt(0,"-taodm_nlevels",&nlevels,PETSC_IGNORE);CHKERRQ(ierr);
  }
  if (nlevels < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot set levels less than 1");
  if (!taodmclass_registered) {
    ierr = PetscClassIdRegister("TaoDM",&TAODM_CLASSID); CHKERRQ(ierr);
  }
  ierr = PetscMalloc(nlevels*sizeof(TaoDM),&p); CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr = PetscHeaderCreate(p[i],_p_TaoDM,struct _TaoDMOps,TAODM_CLASSID,0,"TaoDM",0,0,comm,TaoDMDestroyLevel,TaoDMView); CHKERRQ(ierr);
    p[i]->nlevels  = nlevels - i;
    p[i]->coarselevel = p[0];
    p[i]->user     = user;
    p[i]->isctype  = IS_COLORING_GLOBAL; 
    ierr           = PetscStrallocpy(MATAIJ,&p[i]->mtype);CHKERRQ(ierr);
    p[i]->ttype = PETSC_NULL;
    p[i]->ops->computeobjectiveandgradientlocal=0;
    p[i]->ops->computeobjectivelocal=0;
    p[i]->ops->computegradientlocal=0;
    p[i]->ops->computehessianlocal=0;
    p[i]->ops->computeobjectiveandgradient=0;
    p[i]->ops->computeobjective=0;
    p[i]->ops->computegradient=0;
    p[i]->ops->computehessian=0;
    p[i]->ops->computebounds=0;
    p[i]->ops->computeinitialguess=0;
    p[i]->npremonitors=0;
    p[i]->npostmonitors=0;
  }
  *taodm = p;

  ierr = PetscOptionsGetString(PETSC_NULL,"-taodm_mat_type",mtype,256,&ftype);CHKERRQ(ierr);
  if (ftype) {
    ierr = TaoDMSetMatType(*taodm,mtype);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoDMSetMatType"
/*@C
    TaoDMSetMatType - Sets the type of matrices that TaoDM will create for its solvers.

    Collective on MPI_Comm 

    Input Parameters:
+    taodm - the TaoDM object created with TaoDMCreate()
-    mtype - the matrix type, defaults to MATAIJ

    Level: intermediate

.seealso TaoDMDestroy(), TaoDMSetUser(), TaoDMGetUser(), TaoDMCreate(), TaoDMSetNullSpace()

@*/
PetscErrorCode  TaoDMSetMatType(TaoDM *taodm,const MatType mtype)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<taodm[0]->nlevels; i++) {
    ierr = PetscFree(taodm[i]->mtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(mtype,&taodm[i]->mtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDMSetSolverType"
/*@C
    TaoDMSetSolverType - Sets the type of solver that TaoDM will use.

    Collective on MPI_Comm 

    Input Parameters:
+    taodm - the TaoDM object created with TaoDMCreate()
-    type - the solver type

    Level: intermediate

    Options Database Keys:
+   -tao_method - select which method TAO should use
-   -tao_type - identical to -tao_method
    
.seealso TaoDMDestroy(), TaoDMSetUser(), TaoDMGetUser(), TaoDMCreate(), TaoDMSetNullSpace()

@*/
PetscErrorCode TaoDMSetSolverType(TaoDM *taodm, const TaoSolverType type)
{
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for (i=0; i<taodm[0]->nlevels; i++) {
    ierr = PetscFree(taodm[i]->ttype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&taodm[i]->ttype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
  
}
#undef __FUNCT__  
#define __FUNCT__ "TaoDMSetOptionsPrefix"
/*@C
    TaoDMSetOptionsPrefix - Sets the prefix used for the solvers inside a TaoDM

    Collective on MPI_Comm 

    Input Parameters:
+    taodm - the TaoDM object created with TaoDMCreate()
-    prefix - the prefix string

    Level: intermediate

.seealso TaoDMDestroy(), TaoDMSetUser(), TaoDMGetUser(), TaoDMCreate(), TaoDMSetNullSpace()

@*/
PetscErrorCode  TaoDMSetOptionsPrefix(TaoDM *taodm,const char prefix[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  for (i=0; i<taodm[0]->nlevels; i++) {
    ierr = PetscStrallocpy(prefix,&(((PetscObject)(taodm[i]))->prefix));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoDMDestroy"
/*@C
    TaoDMDestroy - Destroys a DA based multigrid solver object. 

    Collective on TaoDM

    Input Parameter:
.    - the context

    Level: advanced

.seealso TaoDMCreate()

@*/
PetscErrorCode  TaoDMDestroy(TaoDM *taodm)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels;
  
  if (!taodm) return(0);
    
  nlevels = taodm[0]->nlevels;

  PetscFunctionBegin;

  for (i=0; i<nlevels; i++) {
    ierr = TaoDMDestroyLevel(taodm[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(taodm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMDestroyLevel"
PetscErrorCode  TaoDMDestroyLevel(TaoDM taodmlevel)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(((PetscObject)(taodmlevel))->prefix);CHKERRQ(ierr);
  ierr = PetscFree(taodmlevel->mtype);CHKERRQ(ierr);
  ierr = PetscFree(taodmlevel->ttype);CHKERRQ(ierr);
  if (taodmlevel->dm)      {ierr = DMDestroy(&taodmlevel->dm);CHKERRQ(ierr);}
  if (taodmlevel->x)       {ierr = VecDestroy(&taodmlevel->x);CHKERRQ(ierr);}
  //if (taodmlevel->b)       {ierr = VecDestroy(&taodmlevel->b);CHKERRQ(ierr);}
  //if (taodmlevel->r)       {ierr = VecDestroy(&taodmlevel->r);CHKERRQ(ierr);}
  //if (taodmlevel->work1)   {ierr = VecDestroy(&taodmlevel->work1);CHKERRQ(ierr);}
  //if (taodmlevel->w)       {ierr = VecDestroy(&taodmlevel->w);CHKERRQ(ierr);}
  //if (taodmlevel->work2)   {ierr = VecDestroy(&taodmlevel->work2);CHKERRQ(ierr);}
  //if (taodmlevel->lwork1)  {ierr = VecDestroy(&taodmlevel->lwork1);CHKERRQ(ierr);}
  if (taodmlevel->hessian_pre)         {ierr = MatDestroy(&taodmlevel->hessian_pre);CHKERRQ(ierr);}
  if (taodmlevel->hessian)         {ierr = MatDestroy(&taodmlevel->hessian);CHKERRQ(ierr);}
  //if (taodmlevel->R)    {ierr = MatDestroy(&taodmlevel->R);CHKERRQ(ierr);}
  //if (taodmlevel->fdcoloring){ierr = MatFDColoringDestroy(taodmlevel->fdcoloring);CHKERRQ(ierr);}
  //if (taodmlevel->tao)      {ierr = PetscObjectDestroy((PetscObject)taodmlevel->tao);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(&taodmlevel); CHKERRQ(ierr);
  PetscFunctionReturn(0);

  
}

#undef __FUNCT__  
#define __FUNCT__ "TaoDMSetDM"
/*@C
    TaoDMSetDM - Sets the coarse grid information for the grids

    Collective on TaoDM

    Input Parameter:
+   taodm - the context
-   dm - the DA or DMComposite object

    Options Database Keys:
.   -taodm_refine: Use the input problem as the coarse level and refine. Otherwise, use it as the fine level and coarsen.

    Level: advanced

.seealso TaoDMCreate(), TaoDMDestroy(), TaoDMSetMatType()

@*/
PetscErrorCode  TaoDMSetDM(TaoDM *taodm, DM dm)
{
  PetscInt       nlevels     = taodm[0]->nlevels;
  PetscBool     doRefine    = PETSC_TRUE;
  PetscInt       i;
  DM             *hierarchy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!taodm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as TaoDM");

  /* Create DM data structure for all the levels */
  ierr = PetscOptionsGetBool(PETSC_NULL, "-taodm_refine", &doRefine, PETSC_IGNORE);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  ierr = PetscMalloc(nlevels*sizeof(DM),&hierarchy);CHKERRQ(ierr);
  if (doRefine) {
    ierr = DMRefineHierarchy(dm,nlevels-1,hierarchy);CHKERRQ(ierr);
    taodm[0]->dm = dm;
    for(i=1; i<nlevels; ++i) {
      taodm[i]->dm = hierarchy[i-1];
    }
  } else {
    taodm[nlevels-1]->dm = dm;
    ierr = DMCoarsenHierarchy(dm,nlevels-1,hierarchy);CHKERRQ(ierr);
    for(i=0; i<nlevels-1; ++i) {
      taodm[nlevels-2-i]->dm = hierarchy[i];
    }
  }
  ierr = PetscFree(hierarchy);CHKERRQ(ierr);
  /* Cleanup old structures (should use some private Destroy() instead) */
  for(i = 0; i < nlevels; ++i) {
    if (taodm[i]->hessian) {ierr = MatDestroy(&taodm[i]->hessian);CHKERRQ(ierr); taodm[i]->hessian = PETSC_NULL;}
    if (taodm[i]->hessian_pre) {ierr = MatDestroy(&taodm[i]->hessian_pre);CHKERRQ(ierr); taodm[i]->hessian_pre = PETSC_NULL;}
  }

  /* Clean up work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DMCreateGlobalVector(taodm[i]->dm,&taodm[i]->x);CHKERRQ(ierr);
    //ierr = VecDuplicate(taodm[i]->x,&taodm[i]->b);CHKERRQ(ierr);
    //ierr = VecDuplicate(taodm[i]->x,&taodm[i]->r);CHKERRQ(ierr);
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolation(taodm[i-1]->dm,taodm[i]->dm,&taodm[i]->R,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TaoDMSolve"
/*@C
    TaoDMSolve - Actually solves the (non)linear system defined with the TaoDM

    Collective on TaoDM

    Input Parameter:
.   taodm - the context

    Level: advanced

    Options Database:
-   -taodm_monitor_solution - display the solution at each iteration
 
.seealso TaoDMCreate(), TaoDMDestroy(), TaoDM, TaoDMSetSNES(), TaoDMSetUp(), TaoDMSetMatType()

@*/
PetscErrorCode  TaoDMSolve(TaoDM *taodm)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nlevels = taodm[0]->nlevels;
  PetscBool     gridseq = PETSC_FALSE,vecmonitor = PETSC_FALSE,flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(0,"-taodm_grid_sequence",&gridseq,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(0,"-taodm_monitor_solution",&vecmonitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = TaoDMSetUp(taodm);
  if (taodm[0]->ops->computeinitialguess) {
    ierr = (*taodm[0]->ops->computeinitialguess)(taodm[0],taodm[0]->x);CHKERRQ(ierr);
    ierr = TaoSolverSetInitialVector(taodm[0]->tao,taodm[0]->x); CHKERRQ(ierr);
  }
  for (i=0; i<nlevels; i++) {
    /* generate hessian for this level */
    ierr = DMGetMatrix(taodm[i]->dm,taodm[nlevels-1]->mtype,&taodm[i]->hessian);CHKERRQ(ierr);
    taodm[i]->hessian_pre = 0;
    if (taodm[i]->ops->computehessianlocal) {
      ierr = TaoSolverSetHessianRoutine(taodm[i]->tao,taodm[i]->hessian,taodm[i]->hessian,TaoDMFormHessianLocal,taodm[i]); CHKERRQ(ierr);
    } else if (taodm[i]->ops->computehessian) {
      ierr = TaoSolverSetHessianRoutine(taodm[i]->tao,taodm[i]->hessian,taodm[i]->hessian,taodm[i]->ops->computehessian,taodm[i]); CHKERRQ(ierr);
    }


    for (j=0;j<taodm[i]->npremonitors;j++) {
      ierr = (*taodm[i]->prelevelmonitor[j])(taodm[i],i,taodm[i]->userpremonitor[j]); CHKERRQ(ierr);
    }

    ierr = TaoSolverSolve(taodm[i]->tao);CHKERRQ(ierr);
    if (vecmonitor) {
      ierr = VecView(taodm[i]->x,PETSC_VIEWER_DRAW_(((PetscObject)(taodm[i]))->comm));CHKERRQ(ierr);
    }
    for (j=0;j<taodm[i]->npostmonitors;j++) {
      ierr = (*taodm[i]->postlevelmonitor[j])(taodm[i],i,taodm[i]->userpostmonitor[j]); CHKERRQ(ierr);
    }

    /* get ready for next level (if another exists) */
    if (i < nlevels-1) {
      ierr = MatInterpolate(taodm[i+1]->R,taodm[i]->x,taodm[i+1]->x);CHKERRQ(ierr);
      ierr = TaoSolverSetInitialVector(taodm[i+1]->tao,taodm[i+1]->x);
    }
  }
  
  /*ierr = VecView(taodm[nlevels-1]->x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);*/
  
  //ierr = (*TaoDMGetFine(taodm)->solve)(taodm,nlevels-1);CHKERRQ(ierr);
  if (vecmonitor) {
    ierr = VecView(taodm[nlevels-1]->x,PETSC_VIEWER_DRAW_(((PetscObject)(taodm[nlevels-1]))->comm));CHKERRQ(ierr);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-taodm_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
   PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(((PetscObject)(taodm[0]))->comm,&viewer);CHKERRQ(ierr);
    ierr = TaoDMView(taodm,viewer);CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-taodm_view_binary",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = TaoDMView(taodm,PETSC_VIEWER_BINARY_(((PetscObject)(taodm[0]))->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetTolerances"
/*
  TaoSolverSetTolerances - Sets parameters used in TAO convergence tests

  Collective on TaoSolver

  Input Parameters
+ tao - the TaoSolver context
. fatol - absolute convergence tolerance
. frtol - relative convergence tolerance
. gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by a this factor

  Options Database Keys:
+ -tao_fatol <fatol> - Sets fatol
. -tao_frtol <frtol> - Sets frtol
. -tao_gatol <catol> - Sets gatol
. -tao_grtol <catol> - Sets gatol
- .tao_gttol <crtol> - Sets gttol

  Absolute Stopping Criteria
$ f_{k+1} <= f_k + fatol

  Relative Stopping Criteria
$ f_{k+1} <= f_k + frtol*|f_k|

  Notes: Use PETSC_DEFAULT to leave one or more tolerances unchanged.

  Level: beginner

@*/
PetscErrorCode TaoDMSetTolerances(TaoDM *taodm, PetscReal fatol, PetscReal frtol, PetscReal gatol, PetscReal grtol, PetscReal gttol)
{
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<taodm[0]->nlevels; i++) {
    taodm[i]->fatol = fatol;
    taodm[i]->frtol = frtol;
    taodm[i]->gatol = gatol;
    taodm[i]->grtol = grtol;
    taodm[i]->gttol = gttol;
  }
  PetscFunctionReturn(0);
}



#ifdef SKIP
/*
    For each level (of grid sequencing) this sets the interpolation/restriction and 
    work vectors needed by the multigrid preconditioner within the KSP 
    of that level.

    Also sets the KSP monitoring on all the levels if requested by user.

*/
#undef __FUNCT__  
#define __FUNCT__ "TaoDMSetUpLevel"
PetscErrorCode  TaoDMSetUpLevel(TaoDM *taodm,KSP ksp,PetscInt nlevels)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PC                      pc;
  PetscBool              ismg,ismf,isshell,ismffd;
  KSP                     lksp; /* solver internal to the multigrid preconditioner */
  MPI_Comm                *comms;

  PetscFunctionBegin;
  if (!taodm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as TaoDM");

  ierr  = PetscMalloc(nlevels*sizeof(MPI_Comm),&comms);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    comms[i] = (PetscObject)(taodm[i])->comm;
  }
  /* use fgmres on outer iteration by default */
  ierr  = KSPSetType(ksp,KSPFGMRES);CHKERRQ(ierr);
  ierr  = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr  = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr  = PCMGSetLevels(pc,nlevels,comms);CHKERRQ(ierr);
  ierr  = PetscFree(comms);CHKERRQ(ierr); 
  ierr =  PCMGSetType(pc,PC_MG_FULL);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {
    /* set solvers for each level */
    for (i=0; i<nlevels; i++) {
      if (i < nlevels-1) { /* don't set for finest level, they are set in PCApply_MG()*/
	ierr = PCMGSetX(pc,i,taodm[i]->x);CHKERRQ(ierr); 
	ierr = PCMGSetRhs(pc,i,taodm[i]->b);CHKERRQ(ierr); 
      }
      if (i > 0) {
        ierr = PCMGSetR(pc,i,taodm[i]->r);CHKERRQ(ierr); 
      }
      /* If using a matrix free multiply and did not provide an explicit matrix to build
         the preconditioner then must use no preconditioner 
      */
      ierr = PetscTypeCompare((PetscObject)taodm[i]->B,MATSHELL,&isshell);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)taodm[i]->B,MATDAAD,&ismf);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)taodm[i]->B,MATMFFD,&ismffd);CHKERRQ(ierr);
      if (isshell || ismf || ismffd) {
        PC  lpc;
        ierr = PCMGGetSmoother(pc,i,&lksp);CHKERRQ(ierr); 
        ierr = KSPGetPC(lksp,&lpc);CHKERRQ(ierr);
        ierr = PCSetType(lpc,PCNONE);CHKERRQ(ierr);
      }
    }

    /* Set interpolation/restriction between levels */
    for (i=1; i<nlevels; i++) {
      ierr = PCMGSetInterpolation(pc,i,taodm[i]->R);CHKERRQ(ierr); 
      ierr = PCMGSetRestriction(pc,i,taodm[i]->R);CHKERRQ(ierr); 
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TaoDMSetKSP"
/*@C
    TaoDMSetKSP - Sets the linear solver object that will use the grid hierarchy

    Collective on TaoDM

    Input Parameter:
+   taodm - the context
.   func - function to compute linear system matrix on each grid level
-   rhs - function to compute right hand side on each level (need only work on the finest grid
          if you do not use grid sequencing)

    Level: advanced

    Notes: For linear problems my be called more than once, reevaluates the matrices if it is called more
       than once. Call TaoDMSolve() directly several times to solve with the same matrix but different 
       right hand sides.
   
.seealso TaoDMCreate(), TaoDMDestroy, TaoDMSetDM(), TaoDMSolve(), TaoDMSetMatType()

@*/
PetscErrorCode  TaoDMSetKSP(TaoDM *taodm,PetscErrorCode (*rhs)(TaoDM,Vec),PetscErrorCode (*func)(TaoDM,Mat,Mat))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = taodm[0]->nlevels,level;
  PetscBool     ismg,galerkin=PETSC_FALSE;
  PC             pc;
  KSP            lksp;
  
  PetscFunctionBegin;
  if (!taodm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as TaoDM");


  if (!taodm[0]->ksp) {
    /* create solvers for each level if they don't already exist*/
    for (i=0; i<nlevels; i++) {
      ierr = KSPCreate(taodm[i]->comm,&taodm[i]->ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)taodm[i]->ksp,PETSC_NULL,nlevels-i);CHKERRQ(ierr);
      //ierr = KSPSetOptionsPrefix(taodm[i]->ksp,taodm[i]->prefix);CHKERRQ(ierr);
      //ierr = TaoDMSetUpLevel(taodm,taodm[i]->ksp,i+1);CHKERRQ(ierr);
      //ierr = KSPSetFromOptions(taodm[i]->ksp);CHKERRQ(ierr);

      } 
    }
  }

  /* evaluate matrix on each level */
  ierr = KSPGetPC(taodm[nlevels-1]->ksp,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {
    ierr = PCMGGetGalerkin(pc,&galerkin);CHKERRQ(ierr);
  }
  if (func) {
    if (galerkin) {
      ierr = (*func)(taodm[nlevels-1],taodm[nlevels-1]->J,taodm[nlevels-1]->B);CHKERRQ(ierr);
    } else {
      for (i=0; i<nlevels; i++) {
        ierr = (*func)(taodm[i],taodm[i]->J,taodm[i]->B);CHKERRQ(ierr);
      }
    }
  }

  for (i=0; i<nlevels-1; i++) {
    ierr = KSPSetOptionsPrefix(taodm[i]->ksp,"taodm_");CHKERRQ(ierr);
  }

  for (level=0; level<nlevels; level++) {
    ierr = KSPSetOperators(taodm[level]->ksp,taodm[level]->J,taodm[level]->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPGetPC(taodm[level]->ksp,&pc);CHKERRQ(ierr);
    if (ismg) {
      for (i=0; i<=level; i++) {
        ierr = PCMGGetSmoother(pc,i,&lksp);CHKERRQ(ierr); 
        ierr = KSPSetOperators(lksp,taodm[i]->J,taodm[i]->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);
}


#endif /* ifdef SKIP */

#undef __FUNCT__  
#define __FUNCT__ "TaoDMView"
/*@C
    TaoDMView - prints information on a DA based multi-level preconditioner

    Collective on TaoDM and PetscViewer

    Input Parameter:
+   taodm - the TaoDM context
-   viewer - the viewer

    Level: advanced

.seealso TaoDMCreate(), TaoDMDestroy(), TaoDMSetMatType()

@*/
PetscErrorCode  TaoDMView(TaoDM *taodm,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = taodm[0]->nlevels;
  PetscMPIInt    flag;
  MPI_Comm       comm1,comm2;
  PetscBool      isascii,isbinary;

  PetscFunctionBegin;
  PetscValidPointer(taodm,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm1);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)taodm[0],&comm2); CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm1,comm2,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the TaoDM and the PetscViewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {
    for (i=0; i<nlevels; i++) {
      ierr = MatView(taodm[i]->hessian,viewer);CHKERRQ(ierr);
    }
  } else {
    if (isascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"TaoDM Object with %D levels\n",nlevels);CHKERRQ(ierr);
      /*
      if (taodm[0]->isctype == IS_COLORING_GLOBAL) {
        ierr = PetscViewerASCIIPrintf(viewer,"Using global (nonghosted) hessian coloring computation\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Using ghosted hessian coloring computation\n");CHKERRQ(ierr);
	}*/
    }
    for (i=0; i<nlevels; i++) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DMView(taodm[i]->dm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (isascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"Using matrix type %s\n",taodm[nlevels-1]->mtype);CHKERRQ(ierr);
    }
    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetVariableBoundsRoutine"
/*@C
  TaoDMSolverSetVariableBoundsRoutine - Sets a function to be used to compute variable bounds

  Collective on TaoSolver

  Input Parameters:
+ taodm - the TaoDM context
- func - the bounds computation routine
 
  Calling sequence of func:
$      func (TaoSolver tao, Vec xl, Vec xu);

+ taodm - the TaoDM context
. xl  - vector of lower bounds 
- xu  - vector of upper bounds

  Level: intermediate

.seealso: TaoDMSetInitialGuessRoutine()

@*/
PetscErrorCode TaoDMSetVariableBoundsRoutine(TaoDM *taodm, PetscErrorCode (*func)(TaoDM,Vec,Vec))
{
  PetscInt i, nlevels=taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computebounds = func;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoDMSetInitialGuessRoutine"
/*@C
    TaoDMSetInitialGuessRoutine - Sets the function that computes an initial guess (coarsest grid only)

    Collective on TaoDM

    Input Parameter:
+   taodm - the context
-   func - the function

    Level: intermediate


.seealso TaoDMCreate(), TaoDMDestroy(), TaoDMSetVariableBoundsRoutine()
@*/
PetscErrorCode  TaoDMSetInitialGuessRoutine(TaoDM *taodm,PetscErrorCode (*func)(TaoDM,Vec))
{
  PetscInt       i,nlevels = taodm[0]->nlevels;


  PetscFunctionBegin;
  taodm[0]->ops->computeinitialguess=func;
  for (i=1; i<nlevels; i++) {
    taodm[i]->ops->computeinitialguess = 0;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDMSetObjectiveAndGradientRoutine"
/*@C
  TaoDMSetObjectiveAndRoutine - Sets the function/gradient evaluation routine for minimization

  Collective on TaoDM

  Input Parameter:
+ taodm - the TaoDM context
. func - the objective/gradient evalution routine
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be PETSC_NULL)

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, PetscReal *f, Vec g, void *ctx);

+ tao - a TaoSolver context
. x - input vector
. f - function value
. g - the gradient vector
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetGradientRoutine(), TaoDMSetHessianRoutine() TaoDMSetObjectiveRoutine(), TaoDMSetLocalObjectiveAndGradientRoutine()
@*/
PetscErrorCode TaoDMSetObjectiveAndGradientRoutine(TaoDM* taodm, PetscErrorCode (*func)(TaoSolver,Vec,PetscReal*,Vec,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computeobjectiveandgradientlocal = 0;
    taodm[i]->ops->computeobjective=0;
    taodm[i]->ops->computeobjectiveandgradient=func;
  }
  PetscFunctionReturn(0);
  
}


#undef __FUNCT__
#define __FUNCT__ "TaoDMSetObjectiveRoutine"
/*@C
  TaoDMSetObjectiveRoutine - Sets the function evaluation routine for minimization

  Collective on TaoDM

  Input Parameter:
+ taodm - the TaoDM context
. func - the objective function
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be PETSC_NULL)

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, PetscReal *f, void *ctx);

+ tao - a TaoSolver context
. x - input vector
. f - function value
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetGradientRoutine(), TaoDMSetHessianRoutine() TaoDMSetObjectiveAndGradientRoutine(), TaoDMSetLocalObjectiveRoutine()
@*/
PetscErrorCode TaoDMSetObjectiveRoutine(TaoDM* taodm, PetscErrorCode (*func)(TaoSolver,Vec,PetscReal*,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computeobjectiveandgradientlocal = 0;
    taodm[i]->ops->computeobjectivelocal=0;
    taodm[i]->ops->computeobjective=func;
  }
  PetscFunctionReturn(0);
  
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetObjectiveRoutine"
/*@C
  TaoDMSetObjectiveRoutine - Sets the function evaluation routine for minimization

  Collective on TaoDM

  Input Parameter:
+ taodm- the TaoDM context
. func - the objective function
- ctx - [optional] user-defined context for private data for the function evaluation
        routine (may be PETSC_NULL)

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, Vec g, void *ctx);

+ tao - a TaoSolver context
. x - input vector
. g - the gradient vector
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetGradientRoutine(), TaoDMSetHessianRoutine() TaoDMSetObjectiveAndGradientRoutine(), TaoDMSetLocalGradientRoutine()
@*/
PetscErrorCode TaoDMSetGradientRoutine(TaoDM* taodm, PetscErrorCode (*func)(TaoSolver,Vec,Vec,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computeobjectiveandgradientlocal = 0;
    taodm[i]->ops->computegradientlocal=0;
    taodm[i]->ops->computegradient=func;
  }
  PetscFunctionReturn(0);
  
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetLocalObjectiveRoutine"
/*@C
  TaoDMSetLocalObjectiveRoutine - Sets the local function evaluation routine for minimization.

  Collective on TaoDM

  Input Parameter:
+ taodm- the TaoDM context
- func - the objective function

  Calling sequence of func:
$      func (DMDALocalInfo *info, PetscReal **x, PetscReal *f, void *ctx)

+ info - information about the DMDA grid
. x - input vector
. f - the objective function value at x
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetGradientRoutine(), TaoDMSetHessianRoutine() TaoDMSetObjectiveAndGradientRoutine(), TaoDMSetLocalGradientRoutine()
@*/
PetscErrorCode TaoDMSetLocalObjectiveRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscReal**,PetscReal*,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computeobjectivelocal = func;
    taodm[i]->ops->computeobjective=0;
    taodm[i]->ops->computeobjectiveandgradient=0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetLocalGradientRoutine"
/*@C
  TaoDMSetLocalGradientRoutine - Sets the local gradient evaluation routine for minimization.

  Collective on TaoDM

  Input Parameter:
+ taodm- the TaoDM context
- func - the gradient function

  Calling sequence of func:
$      func (DMDALocalInfo *info, PetscReal **x, PetscReal **g, void *ctx)

+ info - information about the DMDA grid
. x - input array of local x values
. g - output array of local g values
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetLocalObjectiveRoutine(), TaoDMSetHessianRoutine() TaoDMSetObjectiveAndGradientRoutine(), TaoDMSetGradientRoutine()
@*/
PetscErrorCode TaoDMSetLocalGradientRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscReal**,PetscReal**,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computegradientlocal = func;
    taodm[i]->ops->computegradient=0;
    taodm[i]->ops->computeobjectiveandgradient=0;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDMSetLocalObjectiveAndGradientRoutine"
PetscErrorCode TaoDMSetLocalObjectiveAndGradientRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscReal**,PetscReal *,PetscReal**,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computeobjectiveandgradientlocal = func;
    taodm[i]->ops->computeobjective=0;
    taodm[i]->ops->computegradient=0;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDMSetLocalHessianRoutine"
/*@C
  TaoDMSetLocalHessianRoutine - Sets the local hessian evaluation routine for minimization.

  Collective on TaoDM

  Input Parameter:
+ taodm- the TaoDM context
- func - the hessian evaluation routine

  Calling sequence of func:
$      func (DMDALocalInfo *info, PetscReal **x, Mat H, void *ctx)

+ info - information about the DMDA grid
. x - input vector
. H - the Hessian matrix at H
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetLocalGradientRoutine(), TaoDMSetHessianRoutine() TaoDMSetLocalObjectiveAndGradientRoutine(), TaoDMSetLocalObjectiveRoutine()
@*/
PetscErrorCode TaoDMSetLocalHessianRoutine(TaoDM *taodm, PetscErrorCode (*func)(DMDALocalInfo*, PetscReal**,Mat,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computehessianlocal=func;
    taodm[i]->ops->computehessian=0;
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TaoDMSetHessianRoutine"
/*@C
  TaoDMSetHessianRoutine - Sets the hessian evaluation routine for minimization.

  Collective on TaoDM

  Input Parameter:
+ taodm- the TaoDM context
- func - the hessian evaluation routine

  Calling sequence of func:
$      func (TaoSolver tao, Vec x, Mat *H, Mat *Hpre, MatStructure *flg, void *ctx)

+ tao - TaoSolver context
. x - input vector
. H - the Hessian matrix at H
. Hpre - The matrix used in constructing preconditioner (usually same as H)
. flg - flag indicating information about the preconditioner matrix structure
- ctx - [optional] user-defined function context

  Level: intermediate

.seealso: TaoDMSetLocalGradientRoutine(), TaoDMSetHessianRoutine() TaoDMSetLocalObjectiveAndGradientRoutine(), TaoDMSetLocalObjectiveRoutine()
@*/
PetscErrorCode TaoDMSetHessianRoutine(TaoDM* taodm, PetscErrorCode (*func)(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computehessian = func;
    taodm[i]->ops->computehessianlocal = 0;
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "TaoDMSetFromOptions"

PetscErrorCode TaoDMSetFromOptions(TaoDM* taodm)
{
  PetscFunctionBegin;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoDMSetUp"
PetscErrorCode TaoDMSetUp(TaoDM* taodm)
{
  PetscErrorCode ierr;
  PetscInt i,nlevels = taodm[0]->nlevels;
  //  PetscBool monitor,monitorAll;

  PetscFunctionBegin;
  /* Create Solver for each level */
  for (i=0; i<nlevels; i++) {
    ierr = TaoSolverCreate(((PetscObject)(taodm[i]))->comm,&taodm[i]->tao);CHKERRQ(ierr);
    ierr = TaoSolverSetTolerances(taodm[i]->tao,taodm[i]->fatol,taodm[i]->frtol, taodm[i]->gatol,taodm[i]->grtol,taodm[i]->gttol); CHKERRQ(ierr);
    if (taodm[i]->ttype) {
      ierr = TaoSolverSetType(taodm[i]->tao,taodm[i]->ttype); CHKERRQ(ierr);
    }
    ierr = PetscObjectIncrementTabLevel((PetscObject)taodm[i]->tao,PETSC_NULL,nlevels - i - 1);CHKERRQ(ierr);

    if (taodm[i]->ops->computeobjectivelocal) {
      ierr = TaoSolverSetObjectiveRoutine(taodm[i]->tao,TaoDMFormFunctionLocal,taodm[i]); CHKERRQ(ierr);
    } else if (taodm[i]->ops->computeobjective) {
      ierr = TaoSolverSetObjectiveRoutine(taodm[i]->tao,taodm[i]->ops->computeobjective,taodm[i]); CHKERRQ(ierr);
    }

    if (taodm[i]->ops->computegradientlocal) {
      ierr = TaoSolverSetGradientRoutine(taodm[i]->tao,TaoDMFormGradientLocal,taodm[i]); CHKERRQ(ierr);
    } else if (taodm[i]->ops->computegradient) {
      ierr = TaoSolverSetGradientRoutine(taodm[i]->tao,taodm[i]->ops->computegradient,taodm[i]); CHKERRQ(ierr);
    }

    if (taodm[i]->ops->computeobjectiveandgradientlocal) {
      ierr = TaoSolverSetObjectiveAndGradientRoutine(taodm[i]->tao,TaoDMFormFunctionGradientLocal,taodm[i]); CHKERRQ(ierr);
    } else if (taodm[i]->ops->computeobjectiveandgradient) {
      ierr = TaoSolverSetObjectiveAndGradientRoutine(taodm[i]->tao,taodm[i]->ops->computeobjectiveandgradient,taodm[i]); CHKERRQ(ierr);
    }

    /* Hessian hasn't been built yet */
    ierr = TaoSolverSetVariableBoundsRoutine(taodm[i]->tao,TaoDMFormBounds,taodm[i]); CHKERRQ(ierr);


    ierr = TaoSolverSetOptionsPrefix(taodm[i]->tao,((PetscObject)(taodm[i]))->prefix); CHKERRQ(ierr);
    ierr = TaoSolverSetFromOptions(taodm[i]->tao); CHKERRQ(ierr);
    //ierr = TaoDMSetUpLevel(taodm,taodm[i]->ksp,i+1);CHKERRQ(ierr);
    
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolationScale(taodm[i-1]->dm,taodm[i]->dm,taodm[i]->R,&taodm[i]->Rscale);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
  
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoDMGetContext"
/*@
  TaoDMGetContext - Gets a user context from a TaoDM Object

  Not Collective

  Input Parameter:
. taodm - the TaoDM object

  Output Parameter:
. ctx - the user context

  Level: intermediate

.seealse TaoDMGetDM(), TaoDMSetContext(), TaoDMCreate()
@*/
PetscErrorCode TaoDMGetContext(TaoDM taodm, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(taodm,TAODM_CLASSID,1); 
  if (ctx) *ctx = taodm->user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "TaoDMSetContext"
/*@
  TaoDMSetContext - Sets a user context for a TaoDM Object

  Not Collective

  Input Parameter:
+ taodm - the TaoDM object
- ctx - the user context

  Level: intermediate

.seealse TaoDMGetDM(), TaoDMGetContext(), TaoDMCreate()
@*/
PetscErrorCode TaoDMSetContext(TaoDM taodm, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(taodm,TAODM_CLASSID,1); 
  taodm->user = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMGetDM"
/*@
  TaoDMGetDM - Gets the PETSc DM object

  Not Collective

  Input Parameter:
. taodm - the TaoDM object

  Output Parameter:
. dm - the DM context

  Level: intermediate

.seealse TaoDMDSetDM(), TaoDMGetContext(), TaoDMCreate()
@*/
PetscErrorCode TaoDMGetDM(TaoDM taodm, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(taodm,TAODM_CLASSID,1);
  if (dm) *dm = taodm->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormHessianLocal"
/*@C
  TaoDMFormHessianLocal - hessian evaluation routine for a local DM function. This function is an intermediary between the TaoSolver and the TaoDM hessian evaluation routine.

  Collective on TaoSolver
  
  Input Parameters:
+ tao - TaoSolver context
. X - input vector
. H - hessian matrix
. Hpre - matrix for preconditioner
. flg - flag for preconditioner structure
- ptr - user context

  Level: developer
@*/
PetscErrorCode TaoDMFormHessianLocal(TaoSolver tao, Vec X, Mat *H, Mat *Hpre, MatStructure *flg, void* ptr)
{
  TaoDM          taodm = (TaoDM) ptr;
  PetscErrorCode ierr;
  Vec            localX;
  void           *x;
  DM             dm;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,X,&x); CHKERRQ(ierr);
  ierr = (*taodm->ops->computehessianlocal)(&info, (PetscReal**)x, *H, taodm->user); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,X,&x); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormGradientLocal"
/*@C
  TaoDMFormGradientLocal - gradient evaluation routine for a local DM function. This function is an intermediary between the TaoSolver and the TaoDM gradient evaluation routine.

  Collective on TaoSolver
  
  Input Parameters:
+ tao - TaoSolver context
. X - input vector
. G - gradient vector
- ptr - user context

  Level: developer
@*/
PetscErrorCode TaoDMFormGradientLocal(TaoSolver tao, Vec X, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  void           *x, *g;
  TaoDM          taodm = (TaoDM)ptr;
  PetscInt       N,n;
  DMDALocalInfo  info;
  DM             dm;
  
  PetscFunctionBegin;
  
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  /* determine whether X=localX */
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);
 
  
  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
    localX = X;
  }

  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,G,&g);CHKERRQ(ierr);

  CHKMEMQ;
  ierr = (*taodm->ops->computegradientlocal)(&info,(PetscReal**)x,(PetscReal**)g,taodm->user); CHKERRQ(ierr);
  CHKMEMQ;

  ierr = DMDAVecRestoreArray(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,G,&g);CHKERRQ(ierr);

  if (n != N){
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 

}



#undef __FUNCT__
#define __FUNCT__ "TaoDMFormFunctionGradientLocal"
/*@C
  TaoDMFormFunctionGradientLocal - gradient evaluation routine for a local DM function. This function is an intermediary between the TaoSolver and the TaoDM function/gradient evaluation routine.

  Collective on TaoSolver
  
  Input Parameters:
+ tao - TaoSolver context
. X - input vector
. f - objective function value at X
. G - gradient vector
- ptr - user context

  Level: developer
@*/
PetscErrorCode TaoDMFormFunctionGradientLocal(TaoSolver tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  void           *x, *g;
  TaoDM          taodm = (TaoDM)ptr;
  PetscInt       N,n;
  PetscReal    floc;
  MPI_Comm       comm;
  DMDALocalInfo  info;
  DM             dm;
  PetscFunctionBegin;
  
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  /* determine whether X=localX */
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);
  
  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
    localX = X;
  }

  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,G,&g);CHKERRQ(ierr);

  CHKMEMQ;
  ierr = (*taodm->ops->computeobjectiveandgradientlocal)(&info,(PetscReal**)x,&floc,(PetscReal**)g,taodm->user); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = PetscObjectGetComm((PetscObject)X,&comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&floc,f,1,MPIU_SCALAR, MPIU_SUM, comm); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,G,&g);CHKERRQ(ierr);

  if (n != N){
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 

}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormFunctionLocal"
/*@C
  TaoDMFormFunctionLocal - objective function evaluation routine for a local DM function. This function is an intermediary between the TaoSolver and the TaoDM function evaluation routine.

  Collective on TaoSolver
  
  Input Parameters:
+ tao - TaoSolver context
. X - input vector
. f - objective function value at X
- ptr - user context

  Level: developer
@*/
PetscErrorCode TaoDMFormFunctionLocal(TaoSolver tao, Vec X, PetscReal *f, void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  PetscReal    *x;
  TaoDM          taodm = (TaoDM)ptr;
  PetscInt       N,n;
  PetscReal    floc;
  MPI_Comm       comm;
  DMDALocalInfo  info;
  DM             dm;
  
  PetscFunctionBegin;
  
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  /* determine whether X=localX */
  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetSize(localX,&n);CHKERRQ(ierr);
  
  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
    localX = X;
  }

  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,X,&x);CHKERRQ(ierr);

  CHKMEMQ;
  ierr = (*taodm->ops->computeobjectivelocal)(&info,(PetscReal**)x,&floc,taodm->user); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = PetscObjectGetComm((PetscObject)X,&comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&floc,f,1,MPIU_SCALAR, MPIU_SUM, comm); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,X,&x);CHKERRQ(ierr);

  if (n != N){
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 

}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormBounds"
/*@
  TaoDMFormBounds - bounds evaluation routine for a TaoDM application. This function is an intermediary between the TaoSolver and the TaoDM bounds evaluation routine.

  Collective on TaoSolver
  
  Input Parameters:
+ tao - TaoSolver context
. XL - lower bounds vector
. XU - upper bounds vector
- ptr - user context

  Level: developer
@*/


PetscErrorCode TaoDMFormBounds(TaoSolver tao, Vec XL, Vec XU, void *ptr)
{

  TaoDM          taodm = (TaoDM)ptr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = (*taodm->ops->computebounds)(taodm,XL,XU); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetPreLevelMonitor"
/*@C
  TaoDMSetPreLevelMonitor - Sets a user monitor that will be called after a 
  DM level is set up but before a solve is performed.
  
  Collective on TaoDM

  InputParameters:
+ taodm - the TaoDM context
. func - monitoring routine
- ctx - (optional) user-defined context to pass to the monitor routine.
  

  Calling sequence of func:
$    PetscErrorCode func(TaoDM taodm, PetscInt level, void *ctx)

+  taodm - the TaoDM context
.  level - the current depth of mesh refinement
-  ctx - the user-defined monitor context

  Level: intermediate

.seealso TaoDMSetPostLevelMonitor(), TaoDMSolve()
@*/
PetscErrorCode TaoDMSetPreLevelMonitor(TaoDM* taodm, PetscErrorCode (*func)(TaoDM,PetscInt, void*),void *ctx)
{
  PetscInt i,nlevels = taodm[0]->nlevels;

  PetscFunctionBegin;
  if (taodm[0]->npremonitors >= MAXTAODMMONITORS) {
    SETERRQ1(PETSC_COMM_SELF,1,"Cannot attach another monitor -- max=",MAXTAODMMONITORS);
  }
  for (i=0;i<nlevels;i++) {
    taodm[i]->prelevelmonitor[taodm[i]->npremonitors] = func;
    taodm[i]->userpremonitor[taodm[i]->npremonitors] = ctx;
    ++taodm[i]->npremonitors;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoDMSetPostLevelMonitor"
/*@C
  TaoDMSetPostLevelMonitor - Sets a user monitor that will be called after a 
  DM level is set up but before a solve is performed.
  
  Collective on TaoDM

  InputParameters:
+ taodm - the TaoDM context
. func - monitoring routine
- ctx - (optional) user-defined context to pass to the monitor routine.
  

  Calling sequence of func:
$    PetscErrorCode func(TaoDM taodm, PetscInt level, void *ctx)

+  taodm - the TaoDM context
.  level - the current depth of mesh refinement
-  ctx - the user-defined monitor context

  Level: intermediate

.seealso TaoDMSetPreLevelMonitor(), TaoDMSolve()
@*/
PetscErrorCode TaoDMSetPostLevelMonitor(TaoDM* taodm, PetscErrorCode (*func)(TaoDM,PetscInt, void*),void *ctx)
{
  PetscInt i,nlevels = taodm[0]->nlevels;

  PetscFunctionBegin;
  if (taodm[0]->npostmonitors >= MAXTAODMMONITORS) {
    SETERRQ1(PETSC_COMM_SELF,1,"Cannot attach another monitor -- max=",MAXTAODMMONITORS);
  }
  for (i=0;i<nlevels;i++) {
    taodm[i]->postlevelmonitor[taodm[i]->npostmonitors] = func;
    taodm[i]->userpostmonitor[taodm[i]->npostmonitors] = ctx;
    ++taodm[i]->npremonitors;
  }
  PetscFunctionReturn(0);
}
