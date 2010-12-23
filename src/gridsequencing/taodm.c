#include "petscdm.h"            /*I "petscdm.h"   I*/
#include "petscksp.h"           /*I "petscksp.h"  I*/
#include "private/pcimpl.h"     /*I "petscpc.h"   I*/
#include "private/taosolver_impl.h" /*I "taosolver.h" I*/
#include "private/taodm_impl.h" /*I "taodm.h" I*/

PetscClassId TAODM_CLASSID;

#undef __FUNCT__  
#define __FUNCT__ "TaoDMCreate"
/*@C
    TaoDMCreate - Creates a D based multigrid solver object. This allows one to 
      easily implement MG methods on regular grids.

    Collective on MPI_Comm

    Input Parameter:
+   comm - the processors that will share the grids and solution process
.   nlevels - number of multigrid levels (if this is negative it CANNOT be reset with -taodm_nlevels
-   user - an optional user context

    Output Parameters:
.    - the context

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
  
  for (i=0; i<nlevels; i++) {
    ierr = PetscHeaderCreate(p[i],_p_TaoDM,struct _TaoDMOps,TAODM_CLASSID,0,"TaoDM",comm,TaoDMDestroy,TaoDMView); CHKERRQ(ierr);
    p[i]->nlevels  = nlevels - i;
    p[i]->user     = user;
    p[i]->isctype  = IS_COLORING_GLOBAL; 
    ierr           = PetscStrallocpy(MATAIJ,&p[i]->mtype);CHKERRQ(ierr);
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
  PetscInt       i,nlevels = taodm[0]->nlevels;

  PetscFunctionBegin;
  if (!taodm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Passing null as TaoDM");

  for (i=1; i<nlevels; i++) {
    if (taodm[i]->R) {ierr = MatDestroy(taodm[i]->R);CHKERRQ(ierr);}
  }
  for (i=0; i<nlevels; i++) {
    ierr = PetscFree(((PetscObject)(taodm[i]))->prefix);CHKERRQ(ierr);
    ierr = PetscFree(taodm[i]->mtype);CHKERRQ(ierr);
    if (taodm[i]->dm)      {ierr = DMDestroy(taodm[i]->dm);CHKERRQ(ierr);}
    if (taodm[i]->x)       {ierr = VecDestroy(taodm[i]->x);CHKERRQ(ierr);}
    //if (taodm[i]->b)       {ierr = VecDestroy(taodm[i]->b);CHKERRQ(ierr);}
    //if (taodm[i]->r)       {ierr = VecDestroy(taodm[i]->r);CHKERRQ(ierr);}
    //if (taodm[i]->work1)   {ierr = VecDestroy(taodm[i]->work1);CHKERRQ(ierr);}
    //if (taodm[i]->w)       {ierr = VecDestroy(taodm[i]->w);CHKERRQ(ierr);}
    //if (taodm[i]->work2)   {ierr = VecDestroy(taodm[i]->work2);CHKERRQ(ierr);}
    //if (taodm[i]->lwork1)  {ierr = VecDestroy(taodm[i]->lwork1);CHKERRQ(ierr);}
    if (taodm[i]->hessian_pre)         {ierr = MatDestroy(taodm[i]->hessian_pre);CHKERRQ(ierr);}
    if (taodm[i]->hessian)         {ierr = MatDestroy(taodm[i]->hessian);CHKERRQ(ierr);}
    if (taodm[i]->R)    {ierr = MatDestroy(taodm[i]->R);CHKERRQ(ierr);}
    //if (taodm[i]->fdcoloring){ierr = MatFDColoringDestroy(taodm[i]->fdcoloring);CHKERRQ(ierr);}
    if (taodm[i]->tao)      {ierr = PetscObjectDestroy((PetscObject)taodm[i]->tao);CHKERRQ(ierr);} 
    ierr = PetscFree(taodm[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(taodm);CHKERRQ(ierr);
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
    if (taodm[i]->hessian) {ierr = MatDestroy(taodm[i]->hessian);CHKERRQ(ierr); taodm[i]->hessian = PETSC_NULL;}
    if (taodm[i]->hessian_pre) {ierr = MatDestroy(taodm[i]->hessian_pre);CHKERRQ(ierr); taodm[i]->hessian_pre = PETSC_NULL;}
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
  PetscInt       i,nlevels = taodm[0]->nlevels;
  PetscBool     gridseq = PETSC_FALSE,vecmonitor = PETSC_FALSE,flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(0,"-taodm_grid_sequence",&gridseq,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(0,"-taodm_monitor_solution",&vecmonitor,PETSC_NULL);CHKERRQ(ierr);
  if (taodm[0]->ops->computeinitialguess) {
    ierr = (*taodm[0]->ops->computeinitialguess)(taodm[0],taodm[0]->x);CHKERRQ(ierr);
    ierr = TaoSolverSetInitialVector(taodm[0]->tao,taodm[0]->x); CHKERRQ(ierr);
  }
  for (i=0; i<nlevels-1; i++) {
    ierr = TaoSolverSolve(taodm[i]->tao);CHKERRQ(ierr);
    if (vecmonitor) {
      ierr = VecView(taodm[i]->x,PETSC_VIEWER_DRAW_(((PetscObject)(taodm[i]))->comm));CHKERRQ(ierr);
    }
    ierr = MatInterpolate(taodm[i+1]->R,taodm[i]->x,taodm[i+1]->x);CHKERRQ(ierr);
    ierr = TaoSolverSetInitialVector(taodm[i+1]->tao,taodm[i+1]->x);
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

      if (!taodm[i]->hessian) {
	ierr = DMGetMatrix(taodm[i]->dm,taodm[nlevels-1]->mtype,&taodm[i]->hessian);CHKERRQ(ierr);
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
+   taodm - the context
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
  PetscBool      iascii,isbinary;

  PetscFunctionBegin;
  PetscValidPointer(taodm,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm1);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)taodm[0],&comm2); CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm1,comm2,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"Different communicators in the TaoDM and the PetscViewer");
  }

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {
    for (i=0; i<nlevels; i++) {
      ierr = MatView(taodm[i]->hessian,viewer);CHKERRQ(ierr);
    }
  } else {
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"TaoDM Object with %D levels\n",nlevels);CHKERRQ(ierr);
      if (taodm[0]->isctype == IS_COLORING_GLOBAL) {
        ierr = PetscViewerASCIIPrintf(viewer,"Using global (nonghosted) hessian coloring computation\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Using ghosted hessian coloring computation\n");CHKERRQ(ierr);
      }
    }
    for (i=0; i<nlevels; i++) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DMView(taodm[i]->dm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"Using matrix type %s\n",taodm[nlevels-1]->mtype);CHKERRQ(ierr);
    }
    if (taodm[i]->tao != 0) {
      ierr = TaoSolverView(taodm[i]->tao,viewer);CHKERRQ(ierr);
    } else if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"TaoDM does not have a TaoSolver set\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetVariableBoundsRoutine"
PetscErrorCode TaoDMSetVariableBoundsRoutine(TaoDM *taodm, PetscErrorCode (*bounds)(TaoDM,Vec,Vec))
{
  PetscInt i, nlevels=taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computebounds = bounds;
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
-   guess - the function

    Level: intermediate


.seealso TaoDMCreate(), TaoDMDestroy, TaoDMSetKSP(), TaoDMSetSNES(), TaoDMInitialGuessCurrent(), TaoDMSetMatType(), TaoDMSetNullSpace()

@*/
PetscErrorCode  TaoDMSetInitialGuessRoutine(TaoDM *taodm,PetscErrorCode (*guess)(TaoDM,Vec))
{
  PetscInt       i,nlevels = taodm[0]->nlevels;


  PetscFunctionBegin;
  taodm[0]->ops->computeinitialguess=guess;
  for (i=1; i<nlevels; i++) {
    taodm[i]->ops->computeinitialguess = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMSetLocalObjectiveRoutine"
PetscErrorCode TaoDMSetLocalObjectiveRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscScalar**,PetscScalar*,void*))
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
PetscErrorCode TaoDMSetLocalGradientRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscScalar**,PetscScalar**,void*))
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
PetscErrorCode TaoDMSetLocalObjectiveAndGradientRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscScalar**,PetscScalar *,PetscScalar**,void*))
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
PetscErrorCode TaoDMSetLocalHessianRoutine(TaoDM* taodm, PetscErrorCode (*func)(DMDALocalInfo*,PetscScalar**,Mat,void*))
{
  PetscInt i;
  PetscInt nlevels = taodm[0]->nlevels;
  PetscFunctionBegin;
  for (i=0;i<nlevels;i++) {
    taodm[i]->ops->computehessianlocal = func;
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

    if (taodm[i]->ops->computehessianlocal) {
      ierr = TaoSolverSetHessianRoutine(taodm[i]->tao,taodm[i]->hessian,taodm[i]->hessian_pre,TaoDMFormHessianLocal,taodm[i]); CHKERRQ(ierr);
    } else if (taodm[i]->ops->computehessian) {
      ierr = TaoSolverSetHessianRoutine(taodm[i]->tao,taodm[i]->hessian,taodm[i]->hessian_pre,taodm[i]->ops->computehessian,taodm[i]); CHKERRQ(ierr);
    }
    ierr = TaoSolverSetVariableBoundsRoutine(taodm[i]->tao,TaoDMFormBounds,taodm[i]); CHKERRQ(ierr);


    ierr = TaoSolverSetOptionsPrefix(taodm[i]->tao,((PetscObject)(taodm[i]))->prefix); CHKERRQ(ierr);
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
PetscErrorCode TaoDMGetContext(TaoDM taodm, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(taodm,TAODM_CLASSID,1); 
  if (ctx) *ctx = taodm->user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMGetDM"
PetscErrorCode TaoDMGetDM(TaoDM taodm, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(taodm,TAODM_CLASSID,1);
  if (dm) *dm = taodm->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormHessianLocal"
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
  ierr = (*taodm->ops->computehessianlocal)(&info, (PetscScalar**)x, *H, taodm->user); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,X,&x); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormGradientLocal"
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
  ierr = (*taodm->ops->computegradientlocal)(&info,(PetscScalar**)x,(PetscScalar**)g,taodm->user); CHKERRQ(ierr);
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
PetscErrorCode TaoDMFormFunctionGradientLocal(TaoSolver tao, Vec X, PetscScalar *f, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  void           *x, *g;
  TaoDM          taodm = (TaoDM)ptr;
  PetscInt       N,n;
  PetscScalar    floc;
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
  ierr = (*taodm->ops->computeobjectiveandgradientlocal)(&info,(PetscScalar**)x,&floc,(PetscScalar**)g,taodm->user); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = PetscObjectGetComm((PetscObject)X,&comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&floc,f,1,MPIU_SCALAR, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,G,&g);CHKERRQ(ierr);

  if (n != N){
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 

}

#undef __FUNCT__
#define __FUNCT__ "TaoDMFormFunctionLocal"
PetscErrorCode TaoDMFormFunctionLocal(TaoSolver tao, Vec X, PetscScalar *f, void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  PetscScalar    *x;
  TaoDM          taodm = (TaoDM)ptr;
  PetscInt       N,n;
  PetscScalar    floc;
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
  ierr = (*taodm->ops->computeobjectivelocal)(&info,(PetscScalar**)x,&floc,taodm->user); CHKERRQ(ierr);
  CHKMEMQ;
  ierr = PetscObjectGetComm((PetscObject)X,&comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&floc,f,1,MPIU_SCALAR, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,X,&x);CHKERRQ(ierr);

  if (n != N){
    ierr = DMRestoreLocalVector(dm,&localX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 

}

PetscErrorCode TaoDMFormBounds(TaoSolver tao, Vec XL, Vec XU, void *ptr)
{

  TaoDM          taodm = (TaoDM)ptr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = (*taodm->ops->computebounds)(taodm,XL,XU); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
