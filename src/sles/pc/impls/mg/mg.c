#ifndef lint
static char vcid[] = "$Id: mg.c,v 1.42 1996/01/12 22:06:33 bsmith Exp bsmith $";
#endif
/*
    Defines the multigrid preconditioner interface.
*/
#include "mgimpl.h"                    /*I "pc.h" I*/
#include "pinclude/pviewer.h"          /*I "mg.h" I*/

/*
       MGMCycle_Private - Given an MG structure created with MGCreate() runs 
                  one multiplicative cycle down through the levels and
                  back up.

    Input Parameter:
.   mg - structure created with  MGCreate().
*/
int MGMCycle_Private(MG *mglevels)
{
  MG     mg = *mglevels, mgc = *(mglevels + 1);
  int    cycles = mg->cycles, ierr,its;
  Scalar zero = 0.0;

  if (mg->level == 0) {
    ierr = SLESSolve(mg->csles,mg->b,mg->x,&its); CHKERRQ(ierr);
  }
  else {
    while (cycles--) {
      ierr = SLESSolve(mg->smoothd,mg->b,mg->x,&its); CHKERRQ(ierr);
      ierr = (*mg->residual)(mg->A, mg->b, mg->x, mg->r ); CHKERRQ(ierr);
      ierr = MatMult(mg->restrct,  mg->r, mgc->b ); CHKERRQ(ierr);
      ierr = VecSet(&zero,mgc->x); CHKERRQ(ierr);
      ierr = MGMCycle_Private(mglevels + 1); CHKERRQ(ierr); 
      ierr = MatMultTransAdd(mg->interpolate,mgc->x,mg->x,mg->x); CHKERRQ(ierr);
      ierr = SLESSolve(mg->smoothu,mg->b,mg->x,&its);CHKERRQ(ierr); 
    }
  }
  return 0;
}

/*
       MGCreate_Private - Creates a MG structure for use with the
               multigrid code.
               Level 0 is the coarsest. (But the finest level is stored

*/
static int MGCreate_Private(MPI_Comm comm,int levels,PC pc,MG **result)
{
  MG  *mg;
  int i,ierr;

  mg = (MG *) PetscMalloc( levels*sizeof(MG) ); CHKPTRQ(mg);
  PLogObjectMemory(pc,levels*(sizeof(MG)+sizeof(struct _MG)));

  for ( i=0; i<levels; i++ ) {
    mg[i]         = (MG) PetscMalloc( sizeof(struct _MG) ); CHKPTRQ(mg[i]);
    PetscMemzero(mg[i],sizeof(MG));
    mg[i]->level  = levels - i - 1;
    mg[i]->cycles = 1;
    if ( i==levels-1) {
      ierr = SLESCreate(comm,&mg[i]->csles); CHKERRQ(ierr);
      PLogObjectParent(pc,mg[i]->csles);
    }
    else {
      ierr = SLESCreate(comm,&mg[i]->smoothd); CHKERRQ(ierr);
      PLogObjectParent(pc,mg[i]->smoothd);
      mg[i]->smoothu = mg[i]->smoothd;
    }
  }
  *result = mg;
  return 0;
}

static int PCDestroy_MG(PetscObject obj)
{
  PC pc = (PC) obj;
  MG *mg = (MG *) pc->data;
  int i, n = mg[0]->level + 1;
  for ( i=0; i<n; i++ ) {
    if ( i==n-1 ) {
      SLESDestroy(mg[i]->csles);
    }
    else {
      if (mg[i]->smoothd != mg[i]->smoothu) SLESDestroy(mg[i]->smoothd);
      SLESDestroy(mg[i]->smoothu);
    }
     PetscFree(mg[i]);
  }
  PetscFree(mg);
  return 0;
}

#include <stdio.h>
/*@
   MGCheck - Checks that all components of the MG structure have 
   been set; use before MGCycle().

   Iput Parameters:
.  mg - the MG structure

.keywords: MG, check, set, multigrid
@*/
int MGCheck(PC pc)
{
  MG *mg = (MG *) pc->data;
  int i, n, count = 0;
  if (pc->type != PCMG) return 0;
  n = mg[0]->level;

  if (!mg[n]->csles) {
    fprintf(stderr,"No coarse solver set \n"); count++;
  }
  for (i=1; i<n; i++) {
    if (!mg[i]->restrct) {
      fprintf(stderr,"No restrict set level %d \n",n-i); count++;
    }    
    if (!mg[i]->interpolate) {
      fprintf(stderr,"No interpolate set level %d \n",n-i); count++;
    }
    if (!mg[i]->residual) {
      fprintf(stderr,"No residual set level %d \n",n-i); count++;
    }
    if (!mg[i]->smoothu) {
      fprintf(stderr,"No smoothup set level %d \n",n-i); count++;
    }  
    if (!mg[i]->smoothd) {
      fprintf(stderr,"No smoothdown set level %d \n",n-i); count++;
    }
    if (!mg[i]->r) {
      fprintf(stderr,"No r set level %d \n",n-i); count++;
    } 
    if (i > 0 && !mg[i]->x) {
      fprintf(stderr,"No x set level %d \n",n-i); count++;
    }
    if (i > 0 && !mg[i]->b) {
      fprintf(stderr,"No b set level %d \n",n-i); count++;
    }
  }
  if (!mg[0]->r) {
    fprintf(stderr,"No r set level %d \n",0); count++;
  } 
  if (!mg[0]->x) {
    fprintf(stderr,"No x set level %d \n",0); count++;
  }
  if (!mg[0]->b) {
    fprintf(stderr,"No b set level %d \n",0); count++;
  }
  return count;
}

/*@
   MGSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels. Use MGSetSmootherDown() to set different 
   pre-smoothing steps on different levels.

   Input Parameters:
.  mg - the multigrid context 
.  n - the number of smoothing steps

   Options Database Key:
$  -pc_mg_smoothdown  n

.keywords: MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: MGSetNumberSmoothUp()
@*/
int MGSetNumberSmoothDown(PC pc,int n)
{ 
  MG *mg = (MG *) pc->data;
  int i;
  KSP ksp;
  for ( i=0; i<mg[0]->level; i++ ) {  
     SLESGetKSP(mg[i]->smoothd,&ksp);
     KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);
  }
  return 0;
}

/*@
   MGSetNumberSmoothUp - Sets the number of post-smoothing steps to use 
   on all levels. Use MGSetSmootherUp() to set different numbers of 
   post-smoothing steps on different levels.

   Input Parameters:
.  mg - the multigrid context 
.  n - the number of smoothing steps

   Options Database Key:
$  -pc_mg_smoothup  n

.keywords: MG, smooth, up, post-smoothing, steps, multigrid

.seealso: MGSetNumberSmoothDown()
@*/
int  MGSetNumberSmoothUp(PC pc,int n)
{ 
  MG *mg = (MG *) pc->data;
  int i;
  KSP ksp;
  for ( i=0; i<mg[0]->level; i++ ) {  
     SLESGetKSP(mg[i]->smoothu,&ksp);
     KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);
  }
  return 0;
}

/*@
   MGSetCycles - Sets the number of cycles to use. 1 denotes a
   V-cycle; 2 denotes a W-cycle. Use MGSetCyclesOnLevel() for more 
   complicated cycling.

   Input Parameters:
.  mg - the multigrid context 
.  n - the number of cycles

   Options Database Key:
$  -pc_mg_cycles n

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: MGSetCyclesOnLevel()
@*/
int MGSetCycles(PC pc,int n)
{ 
  MG *mg = (MG *) pc->data;
  int i;
  for ( i=0; i<mg[0]->level; i++ ) {  
     mg[i]->cycles  = n; 
  }
  return 0;
}

extern int MGACycle_Private(MG*);
extern int MGFCycle_Private(MG*);

/*
   MGCycle - Runs either an additive, multiplicative or full cycle of 
   multigrid. 

  Note: 
  A simple wrapper which calls MGMCycle(),MGACycle(), or MGFCycle(). 
*/ 
static int MGCycle(PC pc,Vec b,Vec x)
{
   MG *mg = (MG*) pc->data;
   Scalar zero = 0.0;
   mg[0]->b = b; mg[0]->x = x;
   if (mg[0]->am == MGMULTIPLICATIVE) {
     VecSet(&zero,x);
     return MGMCycle_Private(mg);
   } 
   else if (mg[0]->am == MGADDITIVE) {
     return MGACycle_Private(mg);
   }
   else {
     return MGFCycle_Private(mg);
   }
}

static int MGCycleRichardson(PC pc,Vec b,Vec x,Vec w,int its)
{
  int ierr;
  MG  *mg = (MG*) pc->data;
  mg[0]->b = b; mg[0]->x = x;
  while (its--) {
    ierr = MGMCycle_Private(mg); CHKERRQ(ierr);
  }
  return 0;
}

static int PCSetFromOptions_MG(PC pc)
{
  int    ierr, m,levels = 1,flg;
  char   buff[16];

  if (pc->type != PCMG) return 0;
  if (!pc->data) {
    ierr = OptionsGetInt(pc->prefix,"-pc_mg_levels",&levels,&flg);CHKERRQ(ierr);
    ierr = MGSetLevels(pc,levels); CHKERRQ(ierr);
  }
  ierr = OptionsGetInt(pc->prefix,"-pc_mg_cycles",&m,&flg);CHKERRQ(ierr);
  if (flg) {
    MGSetCycles(pc,m);
  } 
  ierr = OptionsGetInt(pc->prefix,"-pc_mg_smoothup",&m,&flg);CHKERRQ(ierr);
  if (flg) {
    MGSetNumberSmoothUp(pc,m);
  }
  ierr = OptionsGetInt(pc->prefix,"-pc_mg_smoothdown",&m,&flg);CHKERRQ(ierr);
  if (flg) {
    MGSetNumberSmoothDown(pc,m);
  }
  ierr = OptionsGetString(pc->prefix,"-pc_mg_method",buff,15,&flg);CHKERRQ(ierr);
  if (flg) {
    MGType mg;
    if (!PetscStrcmp(buff,"additive")) mg = MGADDITIVE;
    else if (!PetscStrcmp(buff,"multiplicative")) mg = MGMULTIPLICATIVE;
    else if (!PetscStrcmp(buff,"full")) mg = MGFULL;
    else if (!PetscStrcmp(buff,"kaskade")) mg = MGKASKADE;
    else SETERRQ(1,"PCSetFromOptions_MG:Unknown method");
    MGSetType(pc,mg);
  }
  return 0;
}

static int PCPrintHelp_MG(PC pc,char *p)
{
  MPIU_printf(pc->comm," Options for PCMG preconditioner:\n");
  MPIU_fprintf(pc->comm,stdout," %spc_mg_method [additive,multiplicative,fullmultigrid,kaskade\
                  : type of multigrid method\n",p);
  MPIU_fprintf(pc->comm,stdout," %spc_mg_smoothdown m: number of pre-smooths\n",p);
  MPIU_fprintf(pc->comm,stdout," %spc_mg_smoothup m: number of post-smooths\n",p);
  MPIU_fprintf(pc->comm,stdout," %spc_mg_cycles m: 1 for V-cycle, 2 for W-cycle\n",p);
  return 0;
}

static int PCView_MG(PetscObject obj,Viewer viewer)
{
  PC     pc = (PC)obj;
  FILE   *fd;
  MG     *mg = (MG *) pc->data;
  KSP    kspu, kspd;
  int    itu, itd,ierr;
  double dtol, atol, rtol;
  char   *cstring;
  ierr = ViewerFileGetPointer(viewer,&fd); CHKERRQ(ierr);
  SLESGetKSP(mg[0]->smoothu,&kspu);
  SLESGetKSP(mg[0]->smoothd,&kspd);
  KSPGetTolerances(kspu,&dtol,&atol,&rtol,&itu);
  KSPGetTolerances(kspd,&dtol,&atol,&rtol,&itd);
  if (mg[0]->am == MGMULTIPLICATIVE) cstring = "multiplicative";
  else if (mg[0]->am == MGADDITIVE)  cstring = "additive";
  else if (mg[0]->am == MGFULL)      cstring = "full";
  else if (mg[0]->am == MGKASKADE)   cstring = "Kaskade";
  else cstring = "unknown";
  MPIU_fprintf(pc->comm,fd,
    "   MG: method is %s, cycles=%d, pre-smooths=%d, post-smooths=%d\n",
    cstring,mg[0]->cycles,itu,itd); 
  return 0;
}

int PCCreate_MG(PC pc)
{
  pc->apply     = MGCycle;
  pc->setup     = 0;
  pc->destroy   = PCDestroy_MG;
  pc->type      = PCMG;
  pc->data      = (void *) 0;
  pc->setfrom   = PCSetFromOptions_MG;
  pc->printhelp = PCPrintHelp_MG;
  pc->view      = PCView_MG;
  return 0;
}

/*@
   MGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Input Parameters:
.  pc - the preconditioner context
.  levels - the number of levels

.keywords: MG, set, levels, multigrid

.seealso: MGSetType()
@*/
int MGSetLevels(PC pc,int levels)
{
  int ierr;
  MG  *mg;
  if (pc->type != PCMG) return 0;
  ierr          = MGCreate_Private(pc->comm,levels,pc,&mg); CHKERRQ(ierr);
  mg[0]->am     = MGMULTIPLICATIVE;
  pc->data      = (void *) mg;
  pc->applyrich = MGCycleRichardson;
  return 0;
}

/*@
   MGSetType - Determines the form of multigrid to use, either 
   multiplicative, additive, full, or the Kaskade algorithm.

   Input Parameters:
.  pc - the preconditioner context
.  form - multigrid form, one of the following:
$      MGMULTIPLICATIVE, MGADDITIVE, MGFULL, MGKASKADE

   Options Database Key:
$  -pc_mg_method <form>, where <form> is one of the following:
$      multiplicative, additive, full, kaskade   

.keywords: MG, set, method, multiplicative, additive, full, Kaskade, multigrid

.seealso: MGSetLevels()
@*/
int MGSetType(PC pc,MGType form)
{
  MG *mg = (MG *) pc->data;
  if (pc->type != PCMG) return 0;
  mg[0]->am = form;
  if (form == MGMULTIPLICATIVE) pc->applyrich = MGCycleRichardson;
  else pc->applyrich = 0;
  return 0;
}
