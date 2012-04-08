
/*
    Defines the multigrid preconditioner interface.
*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/


#undef __FUNCT__  
#define __FUNCT__ "PCMGMCycle_Private"
PetscErrorCode PCMGMCycle_Private(PC pc,PC_MG_Levels **mglevelsin,PCRichardsonConvergedReason *reason)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   *mgc,*mglevels = *mglevelsin;
  PetscErrorCode ierr;
  PetscInt       cycles = (mglevels->level == 1) ? 1 : (PetscInt) mglevels->cycles;

  PetscFunctionBegin;

  if (mglevels->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSolve(mglevels->smoothd,mglevels->b,mglevels->x);CHKERRQ(ierr);  /* pre-smooth */
  if (mglevels->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  if (mglevels->level) {  /* not the coarsest grid */
    if (mglevels->eventresidual) {ierr = PetscLogEventBegin(mglevels->eventresidual,0,0,0,0);CHKERRQ(ierr);}
    ierr = (*mglevels->residual)(mglevels->A,mglevels->b,mglevels->x,mglevels->r);CHKERRQ(ierr);
    if (mglevels->eventresidual) {ierr = PetscLogEventEnd(mglevels->eventresidual,0,0,0,0);CHKERRQ(ierr);}

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
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatRestrict(mglevels->restrct,mglevels->r,mgc->b);CHKERRQ(ierr);
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = VecSet(mgc->x,0.0);CHKERRQ(ierr);
    while (cycles--) {
      ierr = PCMGMCycle_Private(pc,mglevelsin-1,reason);CHKERRQ(ierr); 
    }
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    ierr = MatInterpolateAdd(mglevels->interpolate,mgc->x,mglevels->x,mglevels->x);CHKERRQ(ierr);
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (mglevels->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSolve(mglevels->smoothu,mglevels->b,mglevels->x);CHKERRQ(ierr);    /* post smooth */
    if (mglevels->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_MG"
static PetscErrorCode PCApplyRichardson_MG(PC pc,Vec b,Vec x,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool  zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
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
#define __FUNCT__ "PCReset_MG"
PetscErrorCode PCReset_MG(PC pc)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  if (mglevels) {
    n = mglevels[0]->levels;
    for (i=0; i<n-1; i++) {
      ierr = VecDestroy(&mglevels[i+1]->r);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i]->b);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i]->x);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i+1]->restrct);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i+1]->interpolate);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i+1]->rscale);CHKERRQ(ierr);
    }

    for (i=0; i<n; i++) {
      ierr = MatDestroy(&mglevels[i]->A);CHKERRQ(ierr);
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
	ierr = KSPReset(mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPReset(mglevels[i]->smoothu);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetLevels"
/*@C
   PCMGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Logically Collective on PC

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
PetscErrorCode  PCMGSetLevels(PC pc,PetscInt levels,MPI_Comm *comms)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  MPI_Comm       comm = ((PetscObject)pc)->comm;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscInt       i;
  PetscMPIInt    size;
  const char     *prefix;
  PC             ipc;
  PetscInt       n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,levels,2);
  if (mg->nlevels == levels) PetscFunctionReturn(0);
  if (mglevels) {
    /* changing the number of levels so free up the previous stuff */
    ierr = PCReset_MG(pc);CHKERRQ(ierr);
    n = mglevels[0]->levels;
    for (i=0; i<n; i++) {
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
	ierr = KSPDestroy(&mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPDestroy(&mglevels[i]->smoothu);CHKERRQ(ierr);
      ierr = PetscFree(mglevels[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mg->levels);CHKERRQ(ierr);
  }

  mg->nlevels      = levels;

  ierr = PetscMalloc(levels*sizeof(PC_MG*),&mglevels);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,levels*(sizeof(PC_MG*)));CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  for (i=0; i<levels; i++) {
    ierr = PetscNewLog(pc,PC_MG_Levels,&mglevels[i]);CHKERRQ(ierr);
    mglevels[i]->level           = i;
    mglevels[i]->levels          = levels;
    mglevels[i]->cycles          = PC_MG_CYCLE_V;
    mg->default_smoothu = 1;
    mg->default_smoothd = 1;
    mglevels[i]->eventsmoothsetup    = 0;
    mglevels[i]->eventsmoothsolve    = 0;
    mglevels[i]->eventresidual       = 0;
    mglevels[i]->eventinterprestrict = 0;

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
    mg->cyclesperpcapply    = 1; 
  }
  mg->am          = PC_MG_MULTIPLICATIVE;
  mg->levels      = mglevels;
  pc->ops->applyrichardson = PCApplyRichardson_MG;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_MG"
PetscErrorCode PCDestroy_MG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscInt       i,n;

  PetscFunctionBegin;
  ierr = PCReset_MG(pc);CHKERRQ(ierr);
  if (mglevels) {
    n = mglevels[0]->levels;
    for (i=0; i<n; i++) {
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
	ierr = KSPDestroy(&mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPDestroy(&mglevels[i]->smoothu);CHKERRQ(ierr);
      ierr = PetscFree(mglevels[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mg->levels);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



extern PetscErrorCode PCMGACycle_Private(PC,PC_MG_Levels**);
extern PetscErrorCode PCMGFCycle_Private(PC,PC_MG_Levels**);
extern PetscErrorCode PCMGKCycle_Private(PC,PC_MG_Levels**);

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

  /* When the DM is supplying the matrix then it will not exist until here */
  for (i=0; i<levels; i++) {
    if (!mglevels[i]->A) {
      ierr = KSPGetOperators(mglevels[i]->smoothu,&mglevels[i]->A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)mglevels[i]->A);CHKERRQ(ierr);
    }
  }

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
  PetscBool      flg,set;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PCMGType       mgtype;
  PCMGCycleType  mgctype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Multigrid options");CHKERRQ(ierr);
    if (!mg->levels) {
      ierr = PetscOptionsInt("-pc_mg_levels","Number of Levels","PCMGSetLevels",levels,&levels,&flg);CHKERRQ(ierr);
      if (!flg && pc->dm) {
        ierr = DMGetRefineLevel(pc->dm,&levels);CHKERRQ(ierr);
        levels++;
        mg->usedmfornumberoflevels = PETSC_TRUE;
      }
      ierr = PCMGSetLevels(pc,levels,PETSC_NULL);CHKERRQ(ierr);
    }
    mglevels = mg->levels;

    mgctype = (PCMGCycleType) mglevels[0]->cycles;
    ierr = PetscOptionsEnum("-pc_mg_cycle_type","V cycle or for W-cycle","PCMGSetCycleType",PCMGCycleTypes,(PetscEnum)mgctype,(PetscEnum*)&mgctype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGSetCycleType(pc,mgctype);CHKERRQ(ierr);
    };
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-pc_mg_galerkin","Use Galerkin process to compute coarser operators","PCMGSetGalerkin",flg,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PCMGSetGalerkin(pc,flg);CHKERRQ(ierr);
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
    ierr = PetscOptionsBool("-pc_mg_log","Log times for each multigrid level","None",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscInt i;
      char     eventname[128];
      if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
      levels = mglevels[0]->levels;
      for (i=0; i<levels; i++) {  
        sprintf(eventname,"MGSetup Level %d",(int)i);
        ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventsmoothsetup);CHKERRQ(ierr);
        sprintf(eventname,"MGSmooth Level %d",(int)i);
        ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventsmoothsolve);CHKERRQ(ierr);
        if (i) {
          sprintf(eventname,"MGResid Level %d",(int)i);
          ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventresidual);CHKERRQ(ierr);
          sprintf(eventname,"MGInterp Level %d",(int)i);
          ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventinterprestrict);CHKERRQ(ierr);
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
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  MG: type is %s, levels=%D cycles=%s\n", PCMGTypes[mg->am],levels,(mglevels[0]->cycles == PC_MG_CYCLE_V) ? "v" : "w");CHKERRQ(ierr);
    if (mg->am == PC_MG_MULTIPLICATIVE) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Cycles per PCApply=%d\n",mg->cyclesperpcapply);CHKERRQ(ierr);
    }
    if (mg->galerkin) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid matrices\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"    Not using Galerkin computed coarse grid matrices\n");CHKERRQ(ierr);
    }
    for (i=0; i<levels; i++) {
      if (!i) {
        ierr = PetscViewerASCIIPrintf(viewer,"Coarse grid solver -- level -------------------------------\n",i);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(mglevels[i]->smoothd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (i && mglevels[i]->smoothd == mglevels[i]->smoothu) {
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
      } else if (i){
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->smoothu,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCMG",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#include <petsc-private/dmimpl.h>
#include <petsc-private/kspimpl.h>

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
  PC                      cpc;
  PetscBool               preonly,lu,redundant,cholesky,svd,dump = PETSC_FALSE,opsset;
  Mat                     dA,dB;
  MatStructure            uflag;
  Vec                     tvec;
  DM                      *dms;
  PetscViewer             viewer = 0;

  PetscFunctionBegin;
  if (mg->usedmfornumberoflevels) {
    PetscInt levels;
    ierr = DMGetRefineLevel(pc->dm,&levels);CHKERRQ(ierr);
    levels++;
    if (levels > n) { /* the problem is now being solved on a finer grid */
      ierr = PCMGSetLevels(pc,levels,PETSC_NULL);CHKERRQ(ierr);
      n    = levels;
      ierr = PCSetFromOptions(pc);CHKERRQ(ierr);  /* it is bad to call this here, but otherwise will never be called for the new hierarchy */
      mglevels =  mg->levels;
    }
  }
  ierr = KSPGetPC(mglevels[0]->smoothd,&cpc);CHKERRQ(ierr);


  /* If user did not provide fine grid operators OR operator was not updated since last global KSPSetOperators() */
  /* so use those from global PC */
  /* Is this what we always want? What if user wants to keep old one? */
  ierr = KSPGetOperatorsSet(mglevels[n-1]->smoothd,PETSC_NULL,&opsset);CHKERRQ(ierr);
  if (opsset) {
    Mat mmat;
    ierr = KSPGetOperators(mglevels[n-1]->smoothd,PETSC_NULL,&mmat,PETSC_NULL);CHKERRQ(ierr);
    if (mmat == pc->pmat) opsset = PETSC_FALSE;
  }
  if (!opsset) {
    ierr = PetscInfo(pc,"Using outer operators to define finest grid operator \n  because PCMGGetSmoother(pc,nlevels-1,&ksp);KSPSetOperators(ksp,...); was not called.\n");CHKERRQ(ierr);
    ierr = KSPSetOperators(mglevels[n-1]->smoothd,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  }

  /* Skipping this for galerkin==2 (externally managed hierarchy such as ML and GAMG). Cleaner logic here would be great. */
  if (pc->dm && mg->galerkin != 2 && !pc->setupcalled) {
    /* construct the interpolation from the DMs */
    Mat p;
    Vec rscale;
    ierr = PetscMalloc(n*sizeof(DM),&dms);CHKERRQ(ierr);
    dms[n-1] = pc->dm;
    for (i=n-2; i>-1; i--) {
      KSPDM kdm;
      ierr = DMCoarsen(dms[i+1],MPI_COMM_NULL,&dms[i]);CHKERRQ(ierr);
      ierr = KSPSetDM(mglevels[i]->smoothd,dms[i]);CHKERRQ(ierr);
      if (mg->galerkin) {ierr = KSPSetDMActive(mglevels[i]->smoothd,PETSC_FALSE);CHKERRQ(ierr);}
      ierr = DMKSPGetContextWrite(dms[i],&kdm);CHKERRQ(ierr);
      /* Ugly hack so that the next KSPSetUp() will use the RHS that we set */
      kdm->computerhs = PETSC_NULL;
      kdm->rhsctx = PETSC_NULL;
      ierr = DMSetFunction(dms[i],0);
      ierr = DMSetInitialGuess(dms[i],0);
      if (!mglevels[i+1]->interpolate) {
	ierr = DMCreateInterpolation(dms[i],dms[i+1],&p,&rscale);CHKERRQ(ierr);
	ierr = PCMGSetInterpolation(pc,i+1,p);CHKERRQ(ierr);
	if (rscale) {ierr = PCMGSetRScale(pc,i+1,rscale);CHKERRQ(ierr);}
        ierr = VecDestroy(&rscale);CHKERRQ(ierr);
        ierr = MatDestroy(&p);CHKERRQ(ierr);
      }
    }

    for (i=n-2; i>-1; i--) {
      ierr = DMDestroy(&dms[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dms);CHKERRQ(ierr);
  }

  if (pc->dm && !pc->setupcalled) {
    /* finest smoother also gets DM but it is not active, independent of whether galerkin==2 */
    ierr = KSPSetDM(mglevels[n-1]->smoothd,pc->dm);CHKERRQ(ierr);
    ierr = KSPSetDMActive(mglevels[n-1]->smoothd,PETSC_FALSE);CHKERRQ(ierr);
  }

  if (mg->galerkin == 1) {
    Mat B;
    /* currently only handle case where mat and pmat are the same on coarser levels */
    ierr = KSPGetOperators(mglevels[n-1]->smoothd,&dA,&dB,&uflag);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      for (i=n-2; i>-1; i--) {
        ierr = MatPtAP(dB,mglevels[i+1]->interpolate,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothd,B,B,uflag);CHKERRQ(ierr);
	if (i != n-2) {ierr = PetscObjectDereference((PetscObject)dB);CHKERRQ(ierr);} 
        dB   = B;
      }
      if (n > 1) {ierr = PetscObjectDereference((PetscObject)dB);CHKERRQ(ierr);}
    } else {
      for (i=n-2; i>-1; i--) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,PETSC_NULL,&B,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatPtAP(dB,mglevels[i+1]->interpolate,MAT_REUSE_MATRIX,1.0,&B);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothd,B,B,uflag);CHKERRQ(ierr);
        dB   = B;
      }
    }
  } else if (!mg->galerkin && pc->dm && pc->dm->x) {
    /* need to restrict Jacobian location to coarser meshes for evaluation */
    for (i=n-2;i>-1; i--) {
      Mat R;
      Vec rscale;
      if (!mglevels[i]->smoothd->dm->x) {
        Vec *vecs;
        ierr = KSPGetVecs(mglevels[i]->smoothd,1,&vecs,0,PETSC_NULL);CHKERRQ(ierr);
        mglevels[i]->smoothd->dm->x = vecs[0];
        ierr = PetscFree(vecs);CHKERRQ(ierr);
      }
      ierr = PCMGGetRestriction(pc,i+1,&R);CHKERRQ(ierr);
      ierr = PCMGGetRScale(pc,i+1,&rscale);CHKERRQ(ierr);
      ierr = MatRestrict(R,mglevels[i+1]->smoothd->dm->x,mglevels[i]->smoothd->dm->x);CHKERRQ(ierr);
      ierr = VecPointwiseMult(mglevels[i]->smoothd->dm->x,mglevels[i]->smoothd->dm->x,rscale);CHKERRQ(ierr);
    }
  }
  if (!mg->galerkin && pc->dm) {
    for (i=n-2;i>=0; i--) {
      DM dmfine,dmcoarse;
      Mat Restrict,Inject;
      Vec rscale;
      ierr = KSPGetDM(mglevels[i+1]->smoothd,&dmfine);CHKERRQ(ierr);
      ierr = KSPGetDM(mglevels[i]->smoothd,&dmcoarse);CHKERRQ(ierr);
      ierr = PCMGGetRestriction(pc,i+1,&Restrict);CHKERRQ(ierr);
      ierr = PCMGGetRScale(pc,i+1,&rscale);CHKERRQ(ierr);
      Inject = PETSC_NULL;      /* Callback should create it if it needs Injection */
      ierr = DMRestrict(dmfine,Restrict,rscale,Inject,dmcoarse);CHKERRQ(ierr);
    }
  }

  if (!pc->setupcalled) {
    for (i=0; i<n; i++) {
      ierr = KSPSetFromOptions(mglevels[i]->smoothd);CHKERRQ(ierr);
    }
    for (i=1; i<n; i++) {
      if (mglevels[i]->smoothu && (mglevels[i]->smoothu != mglevels[i]->smoothd)) {
        ierr = KSPSetFromOptions(mglevels[i]->smoothu);CHKERRQ(ierr);
      }
    }
    for (i=1; i<n; i++) {
      ierr = PCMGGetInterpolation(pc,i,&mglevels[i]->interpolate);CHKERRQ(ierr);
      ierr = PCMGGetRestriction(pc,i,&mglevels[i]->restrct);CHKERRQ(ierr);
    }
    for (i=0; i<n-1; i++) {
      if (!mglevels[i]->b) {
        Vec *vec;
        ierr = KSPGetVecs(mglevels[i]->smoothd,1,&vec,0,PETSC_NULL);CHKERRQ(ierr);
        ierr = PCMGSetRhs(pc,i,*vec);CHKERRQ(ierr);
        ierr = VecDestroy(vec);CHKERRQ(ierr);
        ierr = PetscFree(vec);CHKERRQ(ierr);
      }
      if (!mglevels[i]->r && i) {
        ierr = VecDuplicate(mglevels[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetR(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(&tvec);CHKERRQ(ierr);
      }
      if (!mglevels[i]->x) {
        ierr = VecDuplicate(mglevels[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetX(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(&tvec);CHKERRQ(ierr);
      }
    }
    if (n != 1 && !mglevels[n-1]->r) {
      /* PCMGSetR() on the finest level if user did not supply it */
      Vec *vec;
      ierr = KSPGetVecs(mglevels[n-1]->smoothd,1,&vec,0,PETSC_NULL);CHKERRQ(ierr);
      ierr = PCMGSetR(pc,n-1,*vec);CHKERRQ(ierr);
      ierr = VecDestroy(vec);CHKERRQ(ierr);
      ierr = PetscFree(vec);CHKERRQ(ierr);
    }
  }

  if (pc->dm) {
    /* need to tell all the coarser levels to rebuild the matrix using the DM for that level */
    for (i=0; i<n-1; i++){
      if (mglevels[i]->smoothd->setupstage != KSP_SETUP_NEW) mglevels[i]->smoothd->setupstage = KSP_SETUP_NEWMATRIX;
    }
  }

  for (i=1; i<n; i++) {
    if (mglevels[i]->smoothu == mglevels[i]->smoothd) {
      /* if doing only down then initial guess is zero */
      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothd,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSetUp(mglevels[i]->smoothd);CHKERRQ(ierr);
    if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    if (!mglevels[i]->residual) {
      Mat mat;
      ierr = KSPGetOperators(mglevels[i]->smoothd,PETSC_NULL,&mat,PETSC_NULL);CHKERRQ(ierr);
      ierr = PCMGSetResidual(pc,i,PCMGDefaultResidual,mat);CHKERRQ(ierr);
    }
  }
  for (i=1; i<n; i++) {
    if (mglevels[i]->smoothu && mglevels[i]->smoothu != mglevels[i]->smoothd) {
      Mat          downmat,downpmat;
      MatStructure matflag;

      /* check if operators have been set for up, if not use down operators to set them */
      ierr = KSPGetOperatorsSet(mglevels[i]->smoothu,&opsset,PETSC_NULL);CHKERRQ(ierr);
      if (!opsset) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,&downmat,&downpmat,&matflag);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothu,downmat,downpmat,matflag);CHKERRQ(ierr);
      }

      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothu,PETSC_TRUE);CHKERRQ(ierr);
      if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
      ierr = KSPSetUp(mglevels[i]->smoothu);CHKERRQ(ierr);
      if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
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
    ierr = PetscTypeCompare((PetscObject)cpc,PCSVD,&svd);CHKERRQ(ierr);
    if (!lu && !redundant && !cholesky && !svd) {
      ierr = KSPSetType(mglevels[0]->smoothd,KSPGMRES);CHKERRQ(ierr);
    }
  }

  if (!pc->setupcalled) {
    ierr = KSPSetFromOptions(mglevels[0]->smoothd);CHKERRQ(ierr);
  }

  if (mglevels[0]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[0]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSetUp(mglevels[0]->smoothd);CHKERRQ(ierr);
  if (mglevels[0]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[0]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}

  /*
     Dump the interpolation/restriction matrices plus the 
   Jacobian/stiffness on each level. This allows MATLAB users to 
   easily check if the Galerkin condition A_c = R A_f R^T is satisfied.

   Only support one or the other at the same time.
  */
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_mg_dump_matlab",&dump,PETSC_NULL);CHKERRQ(ierr);
  if (dump) {
    viewer = PETSC_VIEWER_SOCKET_(((PetscObject)pc)->comm);
  }
  dump = PETSC_FALSE;
#endif
  ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_mg_dump_binary",&dump,PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode  PCMGGetLevels(PC pc,PetscInt *levels)
{
  PC_MG *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(levels,2);
  *levels = mg->nlevels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetType"
/*@
   PCMGSetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Logically Collective on PC

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
PetscErrorCode  PCMGSetType(PC pc,PCMGType form)
{
  PC_MG                   *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,form,2);
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

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context 
-  PC_MG_CYCLE_V or PC_MG_CYCLE_W

   Options Database Key:
$  -pc_mg_cycle_type v or w

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: PCMGSetCycleTypeOnLevel()
@*/
PetscErrorCode  PCMGSetCycleType(PC pc,PCMGCycleType n)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;
  PetscInt     i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscValidLogicalCollectiveInt(pc,n,2);
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

   Logically Collective on PC

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
PetscErrorCode  PCMGMultiplicativeSetCycles(PC pc,PetscInt n)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;
  PetscInt     i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscValidLogicalCollectiveInt(pc,n,2);
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

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  use - PETSC_TRUE to use the Galerkin process to compute coarse-level operators

   Options Database Key:
$  -pc_mg_galerkin

   Level: intermediate

.keywords: MG, set, Galerkin

.seealso: PCMGGetGalerkin()

@*/
PetscErrorCode PCMGSetGalerkin(PC pc,PetscBool use)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  mg->galerkin = (PetscInt)use;
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
PetscErrorCode  PCMGGetGalerkin(PC pc,PetscBool  *galerkin)
{ 
  PC_MG        *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *galerkin = (PetscBool)mg->galerkin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCMGSetNumberSmoothDown"
/*@
   PCMGSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels. Use PCMGGetSmootherDown() to set different 
   pre-smoothing steps on different levels.

   Logically Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: PCMGSetNumberSmoothUp()
@*/
PetscErrorCode  PCMGSetNumberSmoothDown(PC pc,PetscInt n)
{ 
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscValidLogicalCollectiveInt(pc,n,2);
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

   Logically Collective on PC

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
PetscErrorCode  PCMGSetNumberSmoothUp(PC pc,PetscInt n)
{ 
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!mglevels) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  PetscValidLogicalCollectiveInt(pc,n,2);
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
.  -pc_mg_galerkin - use Galerkin process to compute coarser operators, i.e. Acoarse = R A R'
.  -pc_mg_multiplicative_cycles - number of cycles to use as the preconditioner (defaults to 1)
.  -pc_mg_dump_matlab - dumps the matrices for each level and the restriction/interpolation matrices
                        to the Socket viewer for reading from MATLAB.
-  -pc_mg_dump_binary - dumps the matrices for each level and the restriction/interpolation matrices
                        to the binary output file called binaryoutput

   Notes: By default this uses GMRES on the fine grid smoother so this should be used with KSPFGMRES or the smoother changed to not use GMRES

       When run with a single level the smoother options are used on that level NOT the coarse grid solver options

   Level: intermediate

   Concepts: multigrid/multilevel

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType, PCEXOTIC, PCGAMG, PCML, PCHYPRE
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), PCMGSetCycleType(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCycleTypeOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()           
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_MG"
PetscErrorCode  PCCreate_MG(PC pc)
{
  PC_MG          *mg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr        = PetscNewLog(pc,PC_MG,&mg);CHKERRQ(ierr);
  pc->data    = (void*)mg;
  mg->nlevels = -1;

  pc->ops->apply          = PCApply_MG;
  pc->ops->setup          = PCSetUp_MG;
  pc->ops->reset          = PCReset_MG;
  pc->ops->destroy        = PCDestroy_MG;
  pc->ops->setfromoptions = PCSetFromOptions_MG;
  pc->ops->view           = PCView_MG;
  PetscFunctionReturn(0);
}
EXTERN_C_END
