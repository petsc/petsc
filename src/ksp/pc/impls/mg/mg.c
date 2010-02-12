#define PETSCKSP_DLL

/*
    Defines the multigrid preconditioner interface.
*/
#include "../src/ksp/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/


#undef __FUNCT__  
#define __FUNCT__ "PCMGMCycle_Private"
PetscErrorCode PCMGMCycle_Private(PC pc,PC_MG_Levels **mglevelsin,PCRichardsonConvergedReason *reason)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   *mgc,*mglevels = *mglevelsin;
  PetscErrorCode ierr;
  PetscInt       cycles = (mglevels->level == 1) ? 1 : (PetscInt) mglevels->cycles;

  PetscFunctionBegin;

  if (mg->eventsmoothsolve) {ierr = PetscLogEventBegin(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mglevels->smoothd,mglevels->b,mglevels->x);CHKERRQ(ierr);  /* pre-smooth */
  if (mg->eventsmoothsolve) {ierr = PetscLogEventEnd(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  if (mglevels->level) {  /* not the coarsest grid */
    if (mg->eventresidual) {ierr = PetscLogEventBegin(mg->eventresidual,0,0,0,0);CHKERRQ(ierr);}
    ierr = (*mglevels->residual)(mglevels->A,mglevels->b,mglevels->x,mglevels->r);CHKERRQ(ierr);
    if (mg->eventresidual) {ierr = PetscLogEventEnd(mg->eventresidual,0,0,0,0);CHKERRQ(ierr);}

    /* if on finest level and have convergence criteria set */
    if (mglevels->level == mglevels->levels-1 && mg->ttol && reason) {
      PetscReal rnorm;
      ierr = VecNorm(mglevels->r,NORM_2,&rnorm);CHKERRQ(ierr);
      if (rnorm <= mg->ttol) {
        if (rnorm < mg->abstol) {
          *reason = PCRICHARDSON_CONVERGED_ATOL;
          ierr = PetscInfo2(pc,"Linear solver has converged. Residual norm %G is less than absolute tolerance %G\n",rnorm,mg->abstol);CHKERRQ(ierr);
        } else {
          *reason = PCRICHARDSON_CONVERGED_RTOL;
          ierr = PetscInfo2(pc,"Linear solver has converged. Residual norm %G is less than relative tolerance times initial residual norm %G\n",rnorm,mg->ttol);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      }
    }

    mgc = *(mglevelsin - 1);
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mglevels->restrct,mglevels->r,mgc->b);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = VecSet(mgc->x,0.0);CHKERRQ(ierr);
    while (cycles--) {
      ierr = PCMGMCycle_Private(pc,mglevelsin-1,reason);CHKERRQ(ierr); 
    }
    if (mg->eventinterprestrict) {ierr = PetscLogEventBegin(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolateAdd(mglevels->interpolate,mgc->x,mglevels->x,mglevels->x);CHKERRQ(ierr);
    if (mg->eventinterprestrict) {ierr = PetscLogEventEnd(mg->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (mg->eventsmoothsolve) {ierr = PetscLogEventBegin(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mglevels->smoothu,mglevels->b,mglevels->x);CHKERRQ(ierr);    /* post smooth */
    if (mg->eventsmoothsolve) {ierr = PetscLogEventEnd(mg->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_MG"
static PetscErrorCode PCApplyRichardson_MG(PC pc,Vec b,Vec x,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscTruth zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       levels = mglevels[0]->levels,i;

  PetscFunctionBegin;
  mglevels[levels-1]->b    = b; 
  mglevels[levels-1]->x    = x;

  mg->rtol = rtol;
  mg->abstol = abstol;
  mg->dtol = dtol;
  if (rtol) {
    /* compute initial residual norm for relative convergence test */
    PetscReal rnorm;
    if (zeroguess) {
      ierr               = VecNorm(b,NORM_2,&rnorm);CHKERRQ(ierr);
    } else {
      ierr               = (*mglevels[levels-1]->residual)(mglevels[levels-1]->A,b,x,w);CHKERRQ(ierr);
      ierr               = VecNorm(w,NORM_2,&rnorm);CHKERRQ(ierr);
    }
    mg->ttol = PetscMax(rtol*rnorm,abstol);
  } else if (abstol) {
    mg->ttol = abstol;
  } else {
    mg->ttol = 0.0;
  }

  /* since smoother is applied to full system, not just residual we need to make sure that smoothers don't 
     stop prematurely do to small residual */
  for (i=1; i<levels; i++) {  
    ierr = KSPSetTolerances(mglevels[i]->smoothu,0,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    if (mglevels[i]->smoothu != mglevels[i]->smoothd) {
      ierr = KSPSetTolerances(mglevels[i]->smoothd,0,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    }
  }

  *reason = (PCRichardsonConvergedReason)0;
  for (i=0; i<its; i++) {
    ierr = PCMGMCycle_Private(pc,mglevels+levels-1,reason);CHKERRQ(ierr);
    if (*reason) break;
  }
  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = i;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetLevels"
/*@C
   PCMGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  levels - the number of levels
-  comms - optional communicators for each level; this is to allow solving the coarser problems
           on smaller sets of processors. Use PETSC_NULL_OBJECT for default in Fortran

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -mg_levels prefix
  for setting the level options rather than the -mg_coarse prefix.

.keywords: MG, set, levels, multigrid

.seealso: PCMGSetType(), PCMGGetLevels()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetLevels(PC pc,PetscInt levels,MPI_Comm *comms)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  MPI_Comm       comm = ((PetscObject)pc)->comm;
  PC_MG_Levels   **mglevels;
  PetscInt       i;
  PetscMPIInt    size;
  const char     *prefix;
  PC             ipc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (mg->nlevels > -1) {
    SETERRQ(PETSC_ERR_ORDER,"Number levels already set for MG\n  make sure that you call PCMGSetLevels() before KSPSetFromOptions()");
  }
  if (mg->levels) SETERRQ(PETSC_ERR_PLIB,"Internal error in PETSc, this array should not yet exist");

  mg->nlevels = levels;

  ierr = PetscMalloc(levels*sizeof(PC_MG*),&mglevels);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,levels*(sizeof(PC_MG*)));CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  for (i=0; i<levels; i++) {
    ierr = PetscNewLog(pc,PC_MG_Levels,&mglevels[i]);CHKERRQ(ierr);
    mglevels[i]->level           = i;
    mglevels[i]->levels          = levels;
    mglevels[i]->cycles          = PC_MG_CYCLE_V;
    mglevels[i]->galerkin        = PETSC_FALSE;
    mglevels[i]->galerkinused    = PETSC_FALSE;
    mg->default_smoothu = 1;
    mg->default_smoothd = 1;

    if (comms) comm = comms[i];
    ierr = KSPCreate(comm,&mglevels[i]->smoothd);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)mglevels[i]->smoothd,(PetscObject)pc,levels-i);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mglevels[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT, mg->default_smoothd);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(mglevels[i]->smoothd,prefix);CHKERRQ(ierr);

    /* do special stuff for coarse grid */
    if (!i && levels > 1) {
      ierr = KSPAppendOptionsPrefix(mglevels[0]->smoothd,"mg_coarse_");CHKERRQ(ierr);

      /* coarse solve is (redundant) LU by default */
      ierr = KSPSetType(mglevels[0]->smoothd,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(mglevels[0]->smoothd,&ipc);CHKERRQ(ierr);
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      if (size > 1) {
        ierr = PCSetType(ipc,PCREDUNDANT);CHKERRQ(ierr);
      } else {
        ierr = PCSetType(ipc,PCLU);CHKERRQ(ierr);
      }

    } else {
      char tprefix[128];
      sprintf(tprefix,"mg_levels_%d_",(int)i);
      ierr = KSPAppendOptionsPrefix(mglevels[i]->smoothd,tprefix);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectParent(pc,mglevels[i]->smoothd);CHKERRQ(ierr);
    mglevels[i]->smoothu    = mglevels[i]->smoothd;
    mg->rtol                = 0.0;
    mg->abstol              = 0.0;
    mg->dtol                = 0.0;
    mg->ttol                = 0.0;
    mg->eventsmoothsetup    = 0;
    mg->eventsmoothsolve    = 0;
    mg->eventresidual       = 0;
    mg->eventinterprestrict = 0;
    mg->cyclesperpcapply    = 1; 
  }
  mg->am          = PC_MG_MULTIPLICATIVE;
  mg->levels      = mglevels;
  pc->ops->applyrichardson = PCApplyRichardson_MG;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_MG_Private"
PetscErrorCode PCDestroy_MG_Private(PC pc)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  if (mglevels) {
    n = mglevels[0]->levels;
    for (i=0; i<n-1; i++) {
      if (mglevels[i+1]->r) {ierr = VecDestroy(mglevels[i+1]->r);CHKERRQ(ierr);}
      if (mglevels[i]->b) {ierr = VecDestroy(mglevels[i]->b);CHKERRQ(ierr);}
      if (mglevels[i]->x) {ierr = VecDestroy(mglevels[i]->x);CHKERRQ(ierr);}
      if (mglevels[i+1]->restrct) {ierr = MatDestroy(mglevels[i+1]->restrct);CHKERRQ(ierr);}
      if (mglevels[i+1]->interpolate) {ierr = MatDestroy(mglevels[i+1]->interpolate);CHKERRQ(ierr);}
    }

    for (i=0; i<n; i++) {
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
	ierr = KSPDestroy(mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPDestroy(mglevels[i]->smoothu);CHKERRQ(ierr);
      ierr = PetscFree(mglevels[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mglevels);CHKERRQ(ierr);
  }
  mg->nlevels = -1;
  mg->levels  = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_MG"
PetscErrorCode PCDestroy_MG(PC pc)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCDestroy_MG_Private(pc);CHKERRQ(ierr);
  ierr = PetscFree(mg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



EXTERN PetscErrorCode PCMGACycle_Private(PC,PC_MG_Levels**);
EXTERN PetscErrorCode PCMGFCycle_Private(PC,PC_MG_Levels**);
EXTERN PetscErrorCode PCMGKCycle_Private(PC,PC_MG_Levels**);

/*
   PCApply_MG - Runs either an additive, multiplicative, Kaskadic
             or full cycle of multigrid. 

  Note: 
  A simple wrapper which calls PCMGMCycle(),PCMGACycle(), or PCMGFCycle(). 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "PCApply_MG"
static PetscErrorCode PCApply_MG(PC pc,Vec b,Vec x)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       levels = mglevels[0]->levels,i;

  PetscFunctionBegin;
  mglevels[levels-1]->b = b; 
  mglevels[levels-1]->x = x;
  if (mg->am == PC_MG_MULTIPLICATIVE) {
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
    for (i=0; i<mg->cyclesperpcapply; i++) {
      ierr = PCMGMCycle_Private(pc,mglevels+levels-1,PETSC_NULL);CHKERRQ(ierr);
    }
  } 
  else if (mg->am == PC_MG_ADDITIVE) {
    ierr = PCMGACycle_Private(pc,mglevels);CHKERRQ(ierr);
  }
  else if (mg->am == PC_MG_KASKADE) {
    ierr = PCMGKCycle_Private(pc,mglevels);CHKERRQ(ierr);
  }
  else {
    ierr = PCMGFCycle_Private(pc,mglevels);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_MG"
PetscErrorCode PCSetFromOptions_MG(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       m,levels = 1,cycles;
  PetscTruth     flg;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PCMGType       mgtype;
  PCMGCycleType  mgctype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Multigrid options");CHKERRQ(ierr);
    if (!pc->data) {
      ierr = PetscOptionsInt("-pc_mg_levels","Number of Levels","PCMGSetLevels",levels,&levels,&flg);CHKERRQ(ierr);
      ierr = PCMGSetLevels(pc,levels,PETSC_NULL);CHKERRQ(ierr);
      mglevels = mg->levels;
    }
    mgctype = (PCMGCycleType) mglevels[0]->cycles;
    ierr = PetscOptionsEnum("-pc_mg_cycle_type","V cycle or for W-cycle","PCMGSetCycleType",PCMGCycleTypes,(PetscEnum)mgctype,(PetscEnum*)&mgctype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetCycleType(pc,mgctype);CHKERRQ(ierr);
    };
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_mg_galerkin","Use Galerkin process to compute coarser operators","PCMGSetGalerkin",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetGalerkin(pc);CHKERRQ(ierr);
    } 
    ierr = PetscOptionsInt("-pc_mg_smoothup","Number of post-smoothing steps","PCMGSetNumberSmoothUp",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetNumberSmoothUp(pc,m);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-pc_mg_smoothdown","Number of pre-smoothing steps","PCMGSetNumberSmoothDown",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetNumberSmoothDown(pc,m);CHKERRQ(ierr);
    }
    mgtype = mg->am;
    ierr = PetscOptionsEnum("-pc_mg_type","Multigrid type","PCMGSetType",PCMGTypes,(PetscEnum)mgtype,(PetscEnum*)&mgtype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetType(pc,mgtype);CHKERRQ(ierr);
    }
    if (mg->am == PC_MG_MULTIPLICATIVE) {
      ierr = PetscOptionsInt("-pc_mg_multiplicative_cycles","Number of cycles for each preconditioner step","PCMGSetLevels",mg->cyclesperpcapply,&cycles,&flg);CHKERRQ(ierr);
      if (flg) {
	ierr = PCMGMultiplicativeSetCycles(pc,cycles);CHKERRQ(ierr);
      }
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_mg_log","Log times for each multigrid level","None",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscInt i;
      char     eventname[128];
      if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
      levels = mglevels[0]->levels;
      for (i=0; i<levels; i++) {  
        sprintf(eventname,"MGSetup Level %d",(int)i);
        ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->cookie,&mg->eventsmoothsetup);CHKERRQ(ierr);
        sprintf(eventname,"MGSmooth Level %d",(int)i);
        ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->cookie,&mg->eventsmoothsolve);CHKERRQ(ierr);
        if (i) {
          sprintf(eventname,"MGResid Level %d",(int)i);
          ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->cookie,&mg->eventresidual);CHKERRQ(ierr);
          sprintf(eventname,"MGInterp Level %d",(int)i);
          ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->cookie,&mg->eventinterprestrict);CHKERRQ(ierr);
        }
      }
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char *PCMGTypes[] = {"MULTIPLICATIVE","ADDITIVE","FULL","KASKADE","PCMGType","PC_MG",0};
const char *PCMGCycleTypes[] = {"invalid","v","w","PCMGCycleType","PC_MG_CYCLE",0};

#undef __FUNCT__  
#define __FUNCT__ "PCView_MG"
PetscErrorCode PCView_MG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       levels = mglevels[0]->levels,i;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  MG: type is %s, levels=%D cycles=%s\n", PCMGTypes[mg->am],levels,(mglevels[0]->cycles == PC_MG_CYCLE_V) ? "v" : "w");CHKERRQ(ierr);
    if (mg->am == PC_MG_MULTIPLICATIVE) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Cycles per PCApply=%d\n",mg->cyclesperpcapply);CHKERRQ(ierr);
    }
    if (mglevels[0]->galerkin) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid matrices\n");CHKERRQ(ierr);
    }
    for (i=0; i<levels; i++) {
      if (!i) {
        ierr = PetscViewerASCIIPrintf(viewer,"Coarse grid solver -- level %D presmooths=%D postsmooths=%D -----\n",i,mg->default_smoothd,mg->default_smoothu);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level %D smooths=%D --------------------\n",i,mg->default_smoothd);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(mglevels[i]->smoothd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (i && mglevels[i]->smoothd == mglevels[i]->smoothu) {
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
      } else if (i){
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level %D smooths=%D --------------------\n",i,mg->default_smoothu);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->smoothu,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCMG",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
    Calls setup for the KSP on each level
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_MG"
PetscErrorCode PCSetUp_MG(PC pc)
{
  PC_MG                   *mg = (PC_MG*)pc->data;
  PC_MG_Levels            **mglevels = mg->levels;
  PetscErrorCode          ierr;
  PetscInt                i,n = mglevels[0]->levels;
  PC                      cpc,mpc;
  PetscTruth              preonly,lu,redundant,cholesky,monitor = PETSC_FALSE,dump = PETSC_FALSE,opsset;
  PetscViewerASCIIMonitor ascii;
  PetscViewer             viewer = PETSC_NULL;
  MPI_Comm                comm;
  Mat                     dA,dB;
  MatStructure            uflag;
  Vec                     tvec;

  PetscFunctionBegin;

  /* If user did not provide fine grid operators OR operator was not updated since last global KSPSetOperators() */
  /* so use those from global PC */
  /* Is this what we always want? What if user wants to keep old one? */
  ierr = KSPGetOperatorsSet(mglevels[n-1]->smoothd,PETSC_NULL,&opsset);CHKERRQ(ierr);
  ierr = KSPGetPC(mglevels[0]->smoothd,&cpc);CHKERRQ(ierr);
  ierr = KSPGetPC(mglevels[n-1]->smoothd,&mpc);CHKERRQ(ierr);
  if (!opsset || ((cpc->setupcalled == 1) && (mpc->setupcalled == 2)) || ((mpc == cpc) && (mpc->setupcalled == 2))) {
    ierr = PetscInfo(pc,"Using outer operators to define finest grid operator \n  because PCMGGetSmoother(pc,nlevels-1,&ksp);KSPSetOperators(ksp,...); was not called.\n");CHKERRQ(ierr);
    ierr = KSPSetOperators(mglevels[n-1]->smoothd,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  }

  if (mglevels[0]->galerkin) {
    Mat B;
    mglevels[0]->galerkinused = PETSC_TRUE;
    /* currently only handle case where mat and pmat are the same on coarser levels */
    ierr = KSPGetOperators(mglevels[n-1]->smoothd,&dA,&dB,&uflag);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      for (i=n-2; i>-1; i--) {
        ierr = MatPtAP(dB,mglevels[i+1]->interpolate,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothd,B,B,uflag);CHKERRQ(ierr);
	if (i != n-2) {ierr = PetscObjectDereference((PetscObject)dB);CHKERRQ(ierr);} 
        dB   = B;
      }
      ierr = PetscObjectDereference((PetscObject)dB);CHKERRQ(ierr);
    } else {
      for (i=n-2; i>-1; i--) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,PETSC_NULL,&B,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatPtAP(dB,mglevels[i+1]->interpolate,MAT_REUSE_MATRIX,1.0,&B);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothd,B,B,uflag);CHKERRQ(ierr);
        dB   = B;
      }
    }
  }

  if (!pc->setupcalled) {
    ierr = PetscOptionsGetTruth(0,"-pc_mg_monitor",&monitor,PETSC_NULL);CHKERRQ(ierr);
     
    for (i=0; i<n; i++) {
      if (monitor) {
        ierr = PetscObjectGetComm((PetscObject)mglevels[i]->smoothd,&comm);CHKERRQ(ierr);
        ierr = PetscViewerASCIIMonitorCreate(comm,"stdout",n-i,&ascii);CHKERRQ(ierr);
        ierr = KSPMonitorSet(mglevels[i]->smoothd,KSPMonitorDefault,ascii,(PetscErrorCode(*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
      }
      ierr = KSPSetFromOptions(mglevels[i]->smoothd);CHKERRQ(ierr);
    }
    for (i=1; i<n; i++) {
      if (mglevels[i]->smoothu && (mglevels[i]->smoothu != mglevels[i]->smoothd)) {
        if (monitor) {
          ierr = PetscObjectGetComm((PetscObject)mglevels[i]->smoothu,&comm);CHKERRQ(ierr);
          ierr = PetscViewerASCIIMonitorCreate(comm,"stdout",n-i,&ascii);CHKERRQ(ierr);
          ierr = KSPMonitorSet(mglevels[i]->smoothu,KSPMonitorDefault,ascii,(PetscErrorCode(*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
        }
        ierr = KSPSetFromOptions(mglevels[i]->smoothu);CHKERRQ(ierr);
      }
    }
    for (i=1; i<n; i++) {
      if (!mglevels[i]->residual) {
        Mat mat;
        ierr = KSPGetOperators(mglevels[i]->smoothd,PETSC_NULL,&mat,PETSC_NULL);CHKERRQ(ierr);
        ierr = PCMGSetResidual(pc,i,PCMGDefaultResidual,mat);CHKERRQ(ierr);
      }
      if (mglevels[i]->restrct && !mglevels[i]->interpolate) {
        ierr = PCMGSetInterpolation(pc,i,mglevels[i]->restrct);CHKERRQ(ierr);
      }
      if (!mglevels[i]->restrct && mglevels[i]->interpolate) {
        ierr = PCMGSetRestriction(pc,i,mglevels[i]->interpolate);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_DEBUG)
      if (!mglevels[i]->restrct || !mglevels[i]->interpolate) {
        SETERRQ1(PETSC_ERR_ARG_WRONGSTATE,"Need to set restriction or interpolation on level %d",(int)i);
      }
#endif
    }
    for (i=0; i<n-1; i++) {
      if (!mglevels[i]->b) {
        Vec *vec;
        ierr = KSPGetVecs(mglevels[i]->smoothd,1,&vec,0,PETSC_NULL);CHKERRQ(ierr);
        ierr = PCMGSetRhs(pc,i,*vec);CHKERRQ(ierr);
        ierr = VecDestroy(*vec);CHKERRQ(ierr);
        ierr = PetscFree(vec);CHKERRQ(ierr);
      }
      if (!mglevels[i]->r && i) {
        ierr = VecDuplicate(mglevels[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetR(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(tvec);CHKERRQ(ierr);
      }
      if (!mglevels[i]->x) {
        ierr = VecDuplicate(mglevels[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetX(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(tvec);CHKERRQ(ierr);
      }
    }
    if (n != 1 && !mglevels[n-1]->r) {
      /* PCMGSetR() on the finest level if user did not supply it */
      Vec *vec;
      ierr = KSPGetVecs(mglevels[n-1]->smoothd,1,&vec,0,PETSC_NULL);CHKERRQ(ierr);
      ierr = PCMGSetR(pc,n-1,*vec);CHKERRQ(ierr);
      ierr = VecDestroy(*vec);CHKERRQ(ierr);
      ierr = PetscFree(vec);CHKERRQ(ierr);
    }
  }


  for (i=1; i<n; i++) {
    if (mglevels[i]->smoothu == mglevels[i]->smoothd) {
      /* if doing only down then initial guess is zero */
      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothd,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (mg->eventsmoothsetup) {ierr = PetscLogEventBegin(mg->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSetUp(mglevels[i]->smoothd);CHKERRQ(ierr);
    if (mg->eventsmoothsetup) {ierr = PetscLogEventEnd(mg->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
  }
  for (i=1; i<n; i++) {
    if (mglevels[i]->smoothu && mglevels[i]->smoothu != mglevels[i]->smoothd) {
      Mat          downmat,downpmat;
      MatStructure matflag;
      PetscTruth   opsset;

      /* check if operators have been set for up, if not use down operators to set them */
      ierr = KSPGetOperatorsSet(mglevels[i]->smoothu,&opsset,PETSC_NULL);CHKERRQ(ierr);
      if (!opsset) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,&downmat,&downpmat,&matflag);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothu,downmat,downpmat,matflag);CHKERRQ(ierr);
      }

      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothu,PETSC_TRUE);CHKERRQ(ierr);
      if (mg->eventsmoothsetup) {ierr = PetscLogEventBegin(mg->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
      ierr = KSPSetUp(mglevels[i]->smoothu);CHKERRQ(ierr);
      if (mg->eventsmoothsetup) {ierr = PetscLogEventEnd(mg->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    }
  }

  /*
      If coarse solver is not direct method then DO NOT USE preonly 
  */
  ierr = PetscTypeCompare((PetscObject)mglevels[0]->smoothd,KSPPREONLY,&preonly);CHKERRQ(ierr);
  if (preonly) {
    ierr = PetscTypeCompare((PetscObject)cpc,PCLU,&lu);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)cpc,PCREDUNDANT,&redundant);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)cpc,PCCHOLESKY,&cholesky);CHKERRQ(ierr);
    if (!lu && !redundant && !cholesky) {
      ierr = KSPSetType(mglevels[0]->smoothd,KSPGMRES);CHKERRQ(ierr);
    }
  }

  if (!pc->setupcalled) {
    if (monitor) {
      ierr = PetscObjectGetComm((PetscObject)mglevels[0]->smoothd,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIMonitorCreate(comm,"stdout",n,&ascii);CHKERRQ(ierr);
      ierr = KSPMonitorSet(mglevels[0]->smoothd,KSPMonitorDefault,ascii,(PetscErrorCode(*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(mglevels[0]->smoothd);CHKERRQ(ierr);
  }

  if (mg->eventsmoothsetup) {ierr = PetscLogEventBegin(mg->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSetUp(mglevels[0]->smoothd);CHKERRQ(ierr);
  if (mg->eventsmoothsetup) {ierr = PetscLogEventEnd(mg->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}

  /*
     Dump the interpolation/restriction matrices plus the 
   Jacobian/stiffness on each level. This allows Matlab users to 
   easily check if the Galerkin condition A_c = R A_f R^T is satisfied.

   Only support one or the other at the same time.
  */
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = PetscOptionsGetTruth(((PetscObject)pc)->prefix,"-pc_mg_dump_matlab",&dump,PETSC_NULL);CHKERRQ(ierr);
  if (dump) {
    viewer = PETSC_VIEWER_SOCKET_(((PetscObject)pc)->comm);
  }
  dump = PETSC_FALSE;
#endif
  ierr = PetscOptionsGetTruth(((PetscObject)pc)->prefix,"-pc_mg_dump_binary",&dump,PETSC_NULL);CHKERRQ(ierr);
  if (dump) {
    viewer = PETSC_VIEWER_BINARY_(((PetscObject)pc)->comm);
  }

  if (viewer) {
    for (i=1; i<n; i++) {
      ierr = MatView(mglevels[i]->restrct,viewer);CHKERRQ(ierr);
    }
    for (i=0; i<n; i++) {
      ierr = KSPGetPC(mglevels[i]->smoothd,&pc);CHKERRQ(ierr);
      ierr = MatView(pc->mat,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetLevels"
/*@
   PCMGGetLevels - Gets the number of levels to use with MG.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.keywords: MG, get, levels, multigrid

.seealso: PCMGSetLevels()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetLevels(PC pc,PetscInt *levels)
{
  PC_MG *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(levels,2);
  *levels = mg->nlevels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetType"
/*@
   PCMGSetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  form - multigrid form, one of PC_MG_MULTIPLICATIVE, PC_MG_ADDITIVE,
   PC_MG_FULL, PC_MG_KASKADE

   Options Database Key:
.  -pc_mg_type <form> - Sets <form>, one of multiplicative,
   additive, full, kaskade   

   Level: advanced

.keywords: MG, set, method, multiplicative, additive, full, Kaskade, multigrid

.seealso: PCMGSetLevels()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetType(PC pc,PCMGType form)
{
  PC_MG                   *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  mg->am = form;
  if (form == PC_MG_MULTIPLICATIVE) pc->ops->applyrichardson = PCApplyRichardson_MG;
  else pc->ops->applyrichardson = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetCycleType"
/*@
   PCMGSetCycleType - Sets the type cycles to use.  Use PCMGSetCycleTypeOnLevel() for more 
   complicated cycling.

   Collective on PC

   Input Parameters:
+  pc - the multigrid context 
-  PC_MG_CYCLE_V or PC_MG_CYCLE_W

   Options Database Key:
$  -pc_mg_cycle_type v or w

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: PCMGSetCycleTypeOnLevel()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetCycleType(PC pc,PCMGCycleType n)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;
  PetscInt     i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!mglevels) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mglevels[0]->levels;

  for (i=0; i<levels; i++) {  
    mglevels[i]->cycles  = n; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGMultiplicativeSetCycles"
/*@
   PCMGMultiplicativeSetCycles - Sets the number of cycles to use for each preconditioner step 
         of multigrid when PCMGType of PC_MG_MULTIPLICATIVE is used

   Collective on PC

   Input Parameters:
+  pc - the multigrid context 
-  n - number of cycles (default is 1)

   Options Database Key:
$  -pc_mg_multiplicative_cycles n

   Level: advanced

   Notes: This is not associated with setting a v or w cycle, that is set with PCMGSetCycleType()

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: PCMGSetCycleTypeOnLevel(), PCMGSetCycleType()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGMultiplicativeSetCycles(PC pc,PetscInt n)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;
  PetscInt     i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!mglevels) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mglevels[0]->levels;

  for (i=0; i<levels; i++) {  
    mg->cyclesperpcapply  = n; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetGalerkin"
/*@
   PCMGSetGalerkin - Causes the coarser grid matrices to be computed from the
      finest grid via the Galerkin process: A_i-1 = r_i * A_i * r_i^t

   Collective on PC

   Input Parameters:
.  pc - the multigrid context 

   Options Database Key:
$  -pc_mg_galerkin

   Level: intermediate

.keywords: MG, set, Galerkin

.seealso: PCMGGetGalerkin()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetGalerkin(PC pc)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;
  PetscInt     i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!mglevels) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mglevels[0]->levels;

  for (i=0; i<levels; i++) {  
    mglevels[i]->galerkin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGGetGalerkin"
/*@
   PCMGGetGalerkin - Checks if Galerkin multigrid is being used, i.e.
      A_i-1 = r_i * A_i * r_i^t

   Not Collective

   Input Parameter:
.  pc - the multigrid context 

   Output Parameter:
.  gelerkin - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
$  -pc_mg_galerkin

   Level: intermediate

.keywords: MG, set, Galerkin

.seealso: PCMGSetGalerkin()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGGetGalerkin(PC pc,PetscTruth *galerkin)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!mglevels) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  *galerkin = mglevels[0]->galerkin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetNumberSmoothDown"
/*@
   PCMGSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels. Use PCMGGetSmootherDown() to set different 
   pre-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: PCMGSetNumberSmoothUp()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothDown(PC pc,PetscInt n)
{ 
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!mglevels) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mglevels[0]->levels;

  for (i=1; i<levels; i++) {  
    /* make sure smoother up and down are different */
    ierr = PCMGGetSmootherUp(pc,i,PETSC_NULL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mglevels[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg->default_smoothd = n;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetNumberSmoothUp"
/*@
   PCMGSetNumberSmoothUp - Sets the number of post-smoothing steps to use 
   on all levels. Use PCMGGetSmootherUp() to set different numbers of 
   post-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothup <n> - Sets number of post-smoothing steps

   Level: advanced

   Note: this does not set a value on the coarsest grid, since we assume that
    there is no separate smooth up on the coarsest grid.

.keywords: MG, smooth, up, post-smoothing, steps, multigrid

.seealso: PCMGSetNumberSmoothDown()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCMGSetNumberSmoothUp(PC pc,PetscInt n)
{ 
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!mglevels) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  levels = mglevels[0]->levels;

  for (i=1; i<levels; i++) {  
    /* make sure smoother up and down are different */
    ierr = PCMGGetSmootherUp(pc,i,PETSC_NULL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mglevels[i]->smoothu,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg->default_smoothu = n;
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

/*MC
   PCMG - Use multigrid preconditioning. This preconditioner requires you provide additional
    information about the coarser grid matrices and restriction/interpolation operators.

   Options Database Keys:
+  -pc_mg_levels <nlevels> - number of levels including finest
.  -pc_mg_cycles v or w
.  -pc_mg_smoothup <n> - number of smoothing steps after interpolation
.  -pc_mg_smoothdown <n> - number of smoothing steps before applying restriction operator
.  -pc_mg_type <additive,multiplicative,full,cascade> - multiplicative is the default
.  -pc_mg_log - log information about time spent on each level of the solver
.  -pc_mg_monitor - print information on the multigrid convergence
.  -pc_mg_galerkin - use Galerkin process to compute coarser operators
-  -pc_mg_dump_matlab - dumps the matrices for each level and the restriction/interpolation matrices
                        to the Socket viewer for reading from Matlab.

   Notes:

   Level: intermediate

   Concepts: multigrid/multilevel

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType, 
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), PCMGSetCycleType(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCycleTypeOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()           
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_MG"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_MG(PC pc)
{
  PC_MG          *mg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr        = PetscNewLog(pc,PC_MG,&mg);CHKERRQ(ierr);
  pc->data    = (void*)mg;
  mg->nlevels = -1;

  pc->ops->apply          = PCApply_MG;
  pc->ops->setup          = PCSetUp_MG;
  pc->ops->destroy        = PCDestroy_MG;
  pc->ops->setfromoptions = PCSetFromOptions_MG;
  pc->ops->view           = PCView_MG;
  PetscFunctionReturn(0);
}
EXTERN_C_END
