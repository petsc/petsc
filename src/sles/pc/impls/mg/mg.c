/*$Id: mg.c,v 1.113 2000/09/02 02:48:50 bsmith Exp bsmith $*/
/*
    Defines the multigrid preconditioner interface.
*/
#include "src/sles/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/


/*
       MGMCycle_Private - Given an MG structure created with MGCreate() runs 
                  one multiplicative cycle down through the levels and
                  back up.

    Input Parameter:
.   mg - structure created with  MGCreate().
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGMCycle_Private"
int MGMCycle_Private(MG *mglevels)
{
  MG     mg = *mglevels,mgc = *(mglevels - 1);
  int    cycles = mg->cycles,ierr,its;
  Scalar zero = 0.0;

  PetscFunctionBegin;
  if (!mg->level) {
    ierr = SLESSolve(mg->smoothd,mg->b,mg->x,&its);CHKERRQ(ierr);
  } else {
    while (cycles--) {
      ierr = SLESSolve(mg->smoothd,mg->b,mg->x,&its);CHKERRQ(ierr);
      ierr = (*mg->residual)(mg->A,mg->b,mg->x,mg->r);CHKERRQ(ierr);
      ierr = MatRestrict(mg->restrct,mg->r,mgc->b);CHKERRQ(ierr);
      ierr = VecSet(&zero,mgc->x);CHKERRQ(ierr);
      ierr = MGMCycle_Private(mglevels-1);CHKERRQ(ierr); 
      ierr = MatInterpolateAdd(mg->interpolate,mgc->x,mg->x,mg->x);CHKERRQ(ierr);
      ierr = SLESSolve(mg->smoothu,mg->b,mg->x,&its);CHKERRQ(ierr); 
    }
  }
  PetscFunctionReturn(0);
}

/*
       MGCreate_Private - Creates a MG structure for use with the
               multigrid code. Level 0 is the coarsest. (But the 
               finest level is stored first in the array).

*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGCreate_Private"
static int MGCreate_Private(MPI_Comm comm,int levels,PC pc,MPI_Comm *comms,MG **result)
{
  MG   *mg;
  int  i,ierr,size;
  char *prefix;
  KSP  ksp;
  PC   ipc;

  PetscFunctionBegin;
  mg = (MG*)PetscMalloc(levels*sizeof(MG));CHKPTRQ(mg);
  PLogObjectMemory(pc,levels*(sizeof(MG)+sizeof(struct _MG)));

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  for (i=0; i<levels; i++) {
    mg[i]         = (MG)PetscMalloc(sizeof(struct _MG));CHKPTRQ(mg[i]);
    ierr          = PetscMemzero(mg[i],sizeof(struct _MG));CHKERRQ(ierr);
    mg[i]->level  = i;
    mg[i]->levels = levels;
    mg[i]->cycles = 1;

    if (comms) comm = comms[i];
    ierr = SLESCreate(comm,&mg[i]->smoothd);CHKERRQ(ierr);
    ierr = SLESGetKSP(mg[i]->smoothd,&ksp);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix(mg[i]->smoothd,prefix);CHKERRQ(ierr);

    /* do special stuff for coarse grid */
    if (!i && levels > 1) {
      ierr = SLESAppendOptionsPrefix(mg[0]->smoothd,"mg_coarse_");CHKERRQ(ierr);

      /* coarse solve is (redundant) LU by default */
      ierr = SLESGetKSP(mg[0]->smoothd,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = SLESGetPC(mg[0]->smoothd,&ipc);CHKERRQ(ierr);
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      if (size > 1) {
        ierr = PCSetType(ipc,PCREDUNDANT);CHKERRQ(ierr);
        ierr = PCRedundantGetPC(ipc,&ipc);CHKERRQ(ierr);
      }
      ierr = PCSetType(ipc,PCLU);CHKERRQ(ierr);

    } else {
      ierr = SLESAppendOptionsPrefix(mg[i]->smoothd,"mg_levels_");CHKERRQ(ierr);
    }
    PLogObjectParent(pc,mg[i]->smoothd);
    mg[i]->smoothu         = mg[i]->smoothd;
    mg[i]->default_smoothu = 10000;
    mg[i]->default_smoothd = 10000;
  }
  *result = mg;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCDestroy_MG"
static int PCDestroy_MG(PC pc)
{
  MG  *mg = (MG*)pc->data;
  int i,n = mg[0]->levels,ierr;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (mg[i]->smoothd != mg[i]->smoothu) {
      ierr = SLESDestroy(mg[i]->smoothd);CHKERRQ(ierr);
    }
    ierr = SLESDestroy(mg[i]->smoothu);CHKERRQ(ierr);
    ierr = PetscFree(mg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(mg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



EXTERN int MGACycle_Private(MG*);
EXTERN int MGFCycle_Private(MG*);
EXTERN int MGKCycle_Private(MG*);

/*
   MGCycle - Runs either an additive, multiplicative, Kaskadic
             or full cycle of multigrid. 

  Note: 
  A simple wrapper which calls MGMCycle(),MGACycle(), or MGFCycle(). 
*/ 
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGCycle"
static int MGCycle(PC pc,Vec b,Vec x)
{
  MG     *mg = (MG*)pc->data;
  Scalar zero = 0.0;
  int    levels = mg[0]->levels,ierr;

  PetscFunctionBegin;
  mg[levels-1]->b = b; 
  mg[levels-1]->x = x;
  if (mg[0]->am == MGMULTIPLICATIVE) {
    ierr = VecSet(&zero,x);CHKERRQ(ierr);
    ierr = MGMCycle_Private(mg+levels-1);CHKERRQ(ierr);
  } 
  else if (mg[0]->am == MGADDITIVE) {
    ierr = MGACycle_Private(mg);CHKERRQ(ierr);
  }
  else if (mg[0]->am == MGKASKADE) {
    ierr = MGKCycle_Private(mg);CHKERRQ(ierr);
  }
  else {
    ierr = MGFCycle_Private(mg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGCycleRichardson"
static int MGCycleRichardson(PC pc,Vec b,Vec x,Vec w,int its)
{
  MG  *mg = (MG*)pc->data;
  int ierr,levels = mg[0]->levels;

  PetscFunctionBegin;
  mg[levels-1]->b = b; 
  mg[levels-1]->x = x;
  while (its--) {
    ierr = MGMCycle_Private(mg+levels-1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCSetFromOptions_MG"
static int PCSetFromOptions_MG(PC pc)
{
  int        ierr,m,levels = 1;
  PetscTruth flg;
  char       buff[16],*type[] = {"additive","multiplicative","full","cascade"};

  PetscFunctionBegin;

  ierr = OptionsHead("Multigrid options");CHKERRQ(ierr);
    if (!pc->data) {
      ierr = OptionsInt("-pc_mg_levels","Number of Levels","MGSetLevels",levels,&levels,&flg);CHKERRQ(ierr);
      ierr = MGSetLevels(pc,levels,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = OptionsInt("-pc_mg_cycles","1 for V cycle, 2 for W-cycle","MGSetCycles",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MGSetCycles(pc,m);CHKERRQ(ierr);
    } 
    ierr = OptionsInt("-pc_mg_smoothup","Number of post-smoothing steps","MGSetNumberSmoothUp",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MGSetNumberSmoothUp(pc,m);CHKERRQ(ierr);
    }
    ierr = OptionsInt("-pc_mg_smoothdown","Number of pre-smoothing steps","MGSetNumberSmoothDown",1,&m,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MGSetNumberSmoothDown(pc,m);CHKERRQ(ierr);
    }
    ierr = OptionsEList("-pc_mg_type","Multigrid type","MGSetType",type,4,"multiplicative",buff,15,&flg);CHKERRQ(ierr);
    if (flg) {
      MGType     mg;
      PetscTruth isadd,ismult,isfull,iskask,iscasc;

      ierr = PetscStrcmp(buff,type[0],&isadd);CHKERRQ(ierr);
      ierr = PetscStrcmp(buff,type[1],&ismult);CHKERRQ(ierr);
      ierr = PetscStrcmp(buff,type[2],&isfull);CHKERRQ(ierr);
      ierr = PetscStrcmp(buff,type[3],&iscasc);CHKERRQ(ierr);
      ierr = PetscStrcmp(buff,"kaskade",&iskask);CHKERRQ(ierr);

      if      (isadd)  mg = MGADDITIVE;
      else if (ismult) mg = MGMULTIPLICATIVE;
      else if (isfull) mg = MGFULL;
      else if (iskask) mg = MGKASKADE;
      else if (iscasc) mg = MGKASKADE;
      else SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Unknown type: %s",buff);
      ierr = MGSetType(pc,mg);CHKERRQ(ierr);
    }
  ierr = OptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCView_MG"
static int PCView_MG(PC pc,Viewer viewer)
{
  MG         *mg = (MG*)pc->data;
  KSP        kspu,kspd;
  int        itu,itd,ierr,levels = mg[0]->levels,i;
  PetscReal  dtol,atol,rtol;
  char       *cstring;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SLESGetKSP(mg[0]->smoothu,&kspu);CHKERRQ(ierr);
    ierr = SLESGetKSP(mg[0]->smoothd,&kspd);CHKERRQ(ierr);
    ierr = KSPGetTolerances(kspu,&dtol,&atol,&rtol,&itu);CHKERRQ(ierr);
    ierr = KSPGetTolerances(kspd,&dtol,&atol,&rtol,&itd);CHKERRQ(ierr);
    if (mg[0]->am == MGMULTIPLICATIVE) cstring = "multiplicative";
    else if (mg[0]->am == MGADDITIVE)  cstring = "additive";
    else if (mg[0]->am == MGFULL)      cstring = "full";
    else if (mg[0]->am == MGKASKADE)   cstring = "Kaskade";
    else cstring = "unknown";
    ierr = ViewerASCIIPrintf(viewer,"  MG: type is %s, cycles=%d, pre-smooths=%d, post-smooths=%d\n",
                      cstring,mg[0]->cycles,mg[0]->default_smoothu,mg[0]->default_smoothd);CHKERRQ(ierr);
    for (i=0; i<levels; i++) {
      ierr = ViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level %d -------------------------------\n",i);CHKERRQ(ierr);
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = SLESView(mg[i]->smoothd,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (mg[i]->smoothd == mg[i]->smoothu) {
        ierr = ViewerASCIIPrintf(viewer,"Up solver same as down solver\n");CHKERRQ(ierr);
      } else {
        ierr = ViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level %d -------------------------------\n",i);CHKERRQ(ierr);
        ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = SLESView(mg[i]->smoothu,viewer);CHKERRQ(ierr);
        ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
  } else {
    SETERRQ1(1,"Viewer type %s not supported for PCMG",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
    Calls setup for the SLES on each level
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCSetUp_MG"
static int PCSetUp_MG(PC pc)
{
  MG         *mg = (MG*)pc->data;
  int        ierr,i,n = mg[0]->levels;
  KSP        ksp;

  PetscFunctionBegin;
  /*
     temporarily stick pc->vec into mg[0]->b and x so that 
   SLESSetUp is happy. Since currently those slots are empty.
  */
  mg[n-1]->x = pc->vec;
  mg[n-1]->b = pc->vec;

  for (i=1; i<n; i++) {
    if (mg[i]->smoothd) {
      ierr = SLESSetFromOptions(mg[i]->smoothd);CHKERRQ(ierr);
      ierr = SLESGetKSP(mg[i]->smoothd,&ksp);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
      ierr = SLESSetUp(mg[i]->smoothd,mg[i]->b,mg[i]->x);CHKERRQ(ierr);
    }
  }
  for (i=0; i<n; i++) {
    if (mg[i]->smoothu && mg[i]->smoothu != mg[i]->smoothd) {
      ierr = SLESSetFromOptions(mg[i]->smoothu);CHKERRQ(ierr);
      ierr = SLESGetKSP(mg[i]->smoothu,&ksp);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
      ierr = SLESSetUp(mg[i]->smoothu,mg[i]->b,mg[i]->x);CHKERRQ(ierr);
    }
  }
  ierr = SLESSetFromOptions(mg[0]->smoothd);CHKERRQ(ierr);
  ierr = SLESSetUp(mg[0]->smoothd,mg[0]->b,mg[0]->x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGSetLevels"
/*@
   MGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  levels - the number of levels
-  comms - optional communicators for each level; this is to allow solving the coarser problems
           on smaller sets of processors

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -mg_fine prefix
  for setting the level options rather than the -mg_coarse prefix.

.keywords: MG, set, levels, multigrid

.seealso: MGSetType(), MGGetLevels()
@*/
int MGSetLevels(PC pc,int levels,MPI_Comm *comms)
{
  int ierr;
  MG  *mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);

  if (pc->data) {
    SETERRQ(1,"Number levels already set for MG\n\
    make sure that you call MGSetLevels() before SLESSetFromOptions()");
  }
  ierr                     = MGCreate_Private(pc->comm,levels,pc,comms,&mg);CHKERRQ(ierr);
  mg[0]->am                = MGMULTIPLICATIVE;
  pc->data                 = (void*)mg;
  pc->ops->applyrichardson = MGCycleRichardson;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGGetLevels"
/*@
   MGGetLevels - Gets the number of levels to use with MG.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.keywords: MG, get, levels, multigrid

.seealso: MGSetLevels()
@*/
int MGGetLevels(PC pc,int *levels)
{
  MG  *mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);

  mg      = (MG*)pc->data;
  *levels = mg[0]->levels;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGSetType"
/*@
   MGSetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  form - multigrid form, one of MGMULTIPLICATIVE, MGADDITIVE,
   MGFULL, MGKASKADE

   Options Database Key:
.  -pc_mg_type <form> - Sets <form>, one of multiplicative,
   additive, full, kaskade   

   Level: advanced

.keywords: MG, set, method, multiplicative, additive, full, Kaskade, multigrid

.seealso: MGSetLevels()
@*/
int MGSetType(PC pc,MGType form)
{
  MG *mg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  mg = (MG*)pc->data;

  mg[0]->am = form;
  if (form == MGMULTIPLICATIVE) pc->ops->applyrichardson = MGCycleRichardson;
  else pc->ops->applyrichardson = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGSetCycles"
/*@
   MGSetCycles - Sets the number of cycles to use. 1 denotes a
   V-cycle; 2 denotes a W-cycle. Use MGSetCyclesOnLevel() for more 
   complicated cycling.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of cycles

   Options Database Key:
$  -pc_mg_cycles n - Sets number of multigrid cycles

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: MGSetCyclesOnLevel()
@*/
int MGSetCycles(PC pc,int n)
{ 
  MG  *mg;
  int i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  mg     = (MG*)pc->data;
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    mg[i]->cycles  = n; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGCheck"
/*@
   MGCheck - Checks that all components of the MG structure have 
   been set.

   Collective on PC

   Input Parameters:
.  mg - the MG structure

   Level: advanced

.keywords: MG, check, set, multigrid
@*/
int MGCheck(PC pc)
{
  MG  *mg;
  int i,n,count = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  mg = (MG*)pc->data;

  if (!mg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");

  n = mg[0]->levels;

  for (i=1; i<n; i++) {
    if (!mg[i]->restrct) {
      (*PetscErrorPrintf)("No restrict set level %d \n",n-i); count++;
    }    
    if (!mg[i]->interpolate) {
      (*PetscErrorPrintf)("No interpolate set level %d \n",n-i); count++;
    }
    if (!mg[i]->residual) {
      (*PetscErrorPrintf)("No residual set level %d \n",n-i); count++;
    }
    if (!mg[i]->smoothu) {
      (*PetscErrorPrintf)("No smoothup set level %d \n",n-i); count++;
    }  
    if (!mg[i]->smoothd) {
      (*PetscErrorPrintf)("No smoothdown set level %d \n",n-i); count++;
    }
    if (!mg[i]->r) {
      (*PetscErrorPrintf)("No r set level %d \n",n-i); count++;
    } 
    if (!mg[i-1]->x) {
      (*PetscErrorPrintf)("No x set level %d \n",n-i); count++;
    }
    if (!mg[i-1]->b) {
      (*PetscErrorPrintf)("No b set level %d \n",n-i); count++;
    }
  }
  PetscFunctionReturn(count);
}


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGSetNumberSmoothDown"
/*@
   MGSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels. Use MGGetSmootherDown() to set different 
   pre-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.keywords: MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: MGSetNumberSmoothUp()
@*/
int MGSetNumberSmoothDown(PC pc,int n)
{ 
  MG  *mg;
  int i,levels,ierr;
  KSP ksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  mg     = (MG*)pc->data;
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    ierr = SLESGetKSP(mg[i]->smoothd,&ksp);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg[i]->default_smoothd = n;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MGSetNumberSmoothUp"
/*@
   MGSetNumberSmoothUp - Sets the number of post-smoothing steps to use 
   on all levels. Use MGGetSmootherUp() to set different numbers of 
   post-smoothing steps on different levels.

   Collective on PC

   Input Parameters:
+  mg - the multigrid context 
-  n - the number of smoothing steps

   Options Database Key:
.  -pc_mg_smoothup <n> - Sets number of post-smoothing steps

   Level: advanced

.keywords: MG, smooth, up, post-smoothing, steps, multigrid

.seealso: MGSetNumberSmoothDown()
@*/
int  MGSetNumberSmoothUp(PC pc,int n)
{ 
  MG  *mg;
  int i,levels,ierr;
  KSP ksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  mg     = (MG*)pc->data;
  levels = mg[0]->levels;

  for (i=0; i<levels; i++) {  
    ierr = SLESGetKSP(mg[i]->smoothu,&ksp);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg[i]->default_smoothu = n;
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCCreate_MG"
int PCCreate_MG(PC pc)
{
  PetscFunctionBegin;
  pc->ops->apply          = MGCycle;
  pc->ops->setup          = PCSetUp_MG;
  pc->ops->destroy        = PCDestroy_MG;
  pc->ops->setfromoptions = PCSetFromOptions_MG;
  pc->ops->view           = PCView_MG;

  pc->data                = (void*)0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
