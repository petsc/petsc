#ifndef lint
static char vcid[] = "$Id: mg.c,v 1.14 1995/04/17 03:21:45 bsmith Exp bsmith $";
#endif
/*
     Classical Multigrid V or W Cycle routine    

*/
#include "mgimpl.h"
#include "options.h"

/*
       MGMCycle - Given an MG structure created with MGCreate() runs 
                  one multiplicative cycle down through the levels and
                  back up.

    Input Parameter:
.   mg - structure created with  MGCreate().
*/
int MGMCycle(MG *mglevels)
{
  MG     mg = *mglevels, mgc = *(mglevels + 1);
  int    cycles = mg->cycles, ierr,its;
  Scalar zero = 0.0;

  if (mg->level == 0) {
    ierr = SLESSolve(mg->csles,mg->b,mg->x,&its); CHKERR(ierr);
  }
  else {
    while (cycles--) {
      ierr = SLESSolve(mg->smoothd,mg->b,mg->x,&its); CHKERR(ierr);
      ierr = (*mg->residual)(mg->A, mg->b, mg->x, mg->r ); CHKERR(ierr);
      ierr = MatMult(mg->restrict,  mg->r, mgc->b ); CHKERR(ierr);
      ierr = VecSet(&zero,mgc->x); CHKERR(ierr);
      ierr = MGMCycle(mglevels + 1); CHKERR(ierr); 
      ierr = MatMultTransAdd(mg->interpolate,mgc->x,mg->x,mg->x); CHKERR(ierr);
      ierr = SLESSolve(mg->smoothu,mg->b,mg->x,&its);CHKERR(ierr); 
    }
  }
  return 0;
}

/*
       MGCreate - Creates a MG structure for use with the multigrid code.
               Level 0 is the coarsest. (But the finest level is stored
               first in the MG array.)                  
   Useage:
.                    mg = MGCreate(levels) 
.                    MGSet* - set various options
.                    MGCheck() - make sure all options are set.
.                    MGCycle(mg); - run a single cycle.
.                    MGDestroy(mg); - free up space.

    Iput Parameters:
.   levels - the number of levels to use.

*/
static int MGCreate(MPI_Comm comm,int levels,MG **result)
{
  MG  *mg;
  int i,ierr;

  mg = (MG *) MALLOC( levels*sizeof(MG) ); CHKPTR(mg);
  
  for ( i=0; i<levels; i++ ) {
    mg[i]         = (MG) MALLOC( sizeof(struct _MG) ); CHKPTR(mg[i]);
    MEMSET(mg[i],0,sizeof(MG));
    mg[i]->level  = levels - i - 1;
    mg[i]->cycles = 1;
    if ( i==levels-1) {
      ierr = SLESCreate(comm,&mg[i]->csles); CHKERR(ierr);
    }
    else {
      ierr = SLESCreate(comm,&mg[i]->smoothd); CHKERR(ierr);
      mg[i]->smoothu = mg[i]->smoothd;
    }
  }
  *result = mg;
  return 0;
}

/*
       MGDestroy - Frees space used by an MG structure created with 
                    MGCreate().

    Iput Parameters:
.   mg - the MG structure

*/
static int MGDestroy(PetscObject obj)
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
     FREE(mg[i]);
  }
  FREE(mg);
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

#include <stdio.h>
/*@C
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
    if (!mg[i]->restrict) {
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

/*@C
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
     KSPSetMaxIterations(ksp,n);
  }
  return 0;
}

/*@C
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
     KSPSetMaxIterations(ksp,n);
  }
  return 0;
}

/*@C
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

extern int MGACycle(MG*);
extern int MGFCycle(MG*);

/*
   MGCycle - Runs either an additive, multiplicative or full cycle of 
   multigrid. 

  Input Parameters:
.   mg - the multigrid context 
.   am - either Multiplicative, Additive or FullMultigrid 

  Note: 
  A simple wrapper which calls MGMCycle(),MGACycle(), or MGFCycle(). 
*/ 
static int MGCycle(PC pc,Vec b,Vec x)
{
   MG *mg = (MG*) pc->data;
   Scalar zero = 0.0;
   mg[0]->b = b; mg[0]->x = x;
   if (mg[0]->am == Multiplicative) {
     VecSet(&zero,x);
     return MGMCycle(mg);
   } 
   else if (mg[0]->am == Additive) {
     return MGACycle(mg);
   }
   else {
     return MGFCycle(mg);
   }
}

static int MGCycleRichardson(PC pc,Vec b,Vec x,Vec w,int its)
{
  int ierr;
  MG  *mg = (MG*) pc->data;
  mg[0]->b = b; mg[0]->x = x;
  while (its--) {
    ierr = MGMCycle(mg); CHKERR(ierr);
  }
  return 0;
}

static int PCSetFromOptions_MG(PC pc)
{
  int    m;
  char   buff[16];

  if (pc->type != PCMG) return 0;
  if (!pc->data) {
    SETERR(1,"For multigrid PCSetFromOptions() must be after MGSetLevels");
  }
  if (OptionsGetInt(0,pc->prefix,"-pc_mg_cycles",&m)) {
    MGSetCycles(pc,m);
  } 
  if (OptionsGetInt(0,pc->prefix,"-pc_mg_smoothup",&m)) {
    MGSetNumberSmoothUp(pc,m);
  }
  if (OptionsGetInt(0,pc->prefix,"-pc_mg_smoothdown",&m)) {
    MGSetNumberSmoothDown(pc,m);
  }
  if (OptionsGetString(0,pc->prefix,"-pc_mg_method",buff,15)) {
    if (!strcmp(buff,"additive")) m = Additive;
    else if (!strcmp(buff,"multiplicative")) m = Multiplicative;
    else if (!strcmp(buff,"fullmultigrid")) m = FullMultigrid;
    else if (!strcmp(buff,"kaskade")) m = Kaskade;
    else SETERR(1,"Unknown MG method");
    MGSetMethod(pc,m);
  }
  return 0;
}

static int PCPrintHelp_MG(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr," %spc_mg_method [additive,multiplicative,fullmultigrid,kaskade\
                  : type of multigrid method\n",p);
  fprintf(stderr," %spc_mg_smoothdown m: number of pre-smooths\n",p);
  fprintf(stderr," %spc_mg_smoothup m: number of post-smooths\n",p);
  fprintf(stderr," %spc_mg_cycles m: 1 for V-cycle, 2 for W-cycle\n",p);
  return 0;
}

int PCCreate_MG(PC pc)
{
  pc->apply     = MGCycle;
  pc->setup     = 0;
  pc->destroy   = MGDestroy;
  pc->type      = PCMG;
  pc->data      = (void *) 0;
  pc->setfrom   = PCSetFromOptions_MG;
  pc->printhelp = PCPrintHelp_MG;
  return 0;
}

/*@
   MGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Input Parameters:
.  pc - the preconditioner context
.  levels - the number of levels

.keywords: MG, set, levels, multigrid

.seealso: MGSetMethod()
@*/
int MGSetLevels(PC pc,int levels)
{
  int ierr;
  MG  *mg;
  if (pc->type != PCMG) return 0;
  ierr          = MGCreate(pc->comm,levels,&mg); CHKERR(ierr);
  mg[0]->am     = Multiplicative;
  pc->data      = (void *) mg;
  pc->applyrich = MGCycleRichardson;
  return 0;
}

/*@
   MGSetMethod - Determines the form of multigrid to use, either 
   multiplicative, additive, full, or the Kaskade algorithm.

   Input Parameters:
.  pc - the preconditioner context
.  flag - multigrid flag, one of the following:
$      Multiplicative, Additive, FullMultigrid, Kaskade

   Options Database Key:
$  -pc_mg_method <flag>, where <flag> is one of the following:
$      multiplicative, additive, fullmultigrid, kaskade   

.keywords: MG, set, method, multiplicative, additive, full, Kaskade, multigrid

.seealso: MGSetLevels()
@*/
int MGSetMethod(PC pc,int flag)
{
  MG *mg = (MG *) pc->data;
  if (pc->type != PCMG) return 0;
  mg[0]->am = flag;
  if (flag == Multiplicative) pc->applyrich = MGCycleRichardson;
  else pc->applyrich = 0;
  return 0;
}
