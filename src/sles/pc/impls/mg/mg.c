/*
     Classical Multigrid V or W Cycle routine    

*/
#include "mgimpl.h"

/*@C
       MGMCycle - Given an MG structure created with MGCreate() runs 
                  one multiplicative cycle down through the levels and
                  back up.

    Iput Parameters:
.   mg - structure created with  MGCreate().

@*/
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

/*@
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

@*/
int MGCreate(int levels,MG **result)
{
  MG  *mg;
  int i;

  mg = (MG *) MALLOC( levels*sizeof(MG) ); CHKPTR(mg);
  
  for ( i=0; i<levels; i++ ) {
    mg[i]        = (MG) MALLOC( sizeof(struct _MG) ); CHKPTR(mg[i]);
    MEMSET(mg[i],0,sizeof(MG));
    mg[i]->level  = levels - i - 1;
    mg[i]->cycles = 1;
  }
  *result = mg;
  return 0;
}

/*@C
       MGDestroy - Frees space used by an MG structure created with 
                    MGCreate().

    Iput Parameters:
.   mg - the MG structure

@*/
int MGDestroy(MG *mg)
{
  int i, n = mg[0]->level + 1;
  for ( i=0; i<n; i++ ) FREE(mg[i]);
  FREE(mg);
  return 0;
}

#include <stdio.h>
/*@C
       MGCheck - Checks that all components of MG structure have 
                 been set, use before MGCycle().

    Iput Parameters:
.   mg - the MG structure

@*/
int MGCheck(MG *mg)
{
  int i, n = mg[0]->level, count = 0;

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
    if (!mg[i]->x) {
      fprintf(stderr,"No x set level %d \n",n-i); count++;
    }
    if (!mg[i]->b) {
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
      MGSetNumberSmoothDown - Sets the number of pre smoothing steps to
                    use on all levels. Use MGSetSmootherDown() to set 
                    it different on different levels.

  Input Parameters:
.   mg - the multigrid context 
.   n - the number of smoothing steps

@*/
int MGSetNumberSmoothDown(MG *mg,int n)
{ 
  int i;
  KSP ksp;
  for ( i=0; i<mg[0]->level; i++ ) {  
     SLESGetKSP(mg[i]->smoothd,&ksp);
     KSPSetIterations(ksp,n);
  }
  return 0;
}

/*@C
      MGSetNumberSmoothUp - Sets the number of post smoothing steps to use 
                    on all levels. Use MGSetSmootherUp() to set 
                    it different on different levels.

  Input Parameters:
.   mg - the multigrid context 
.   n - the number of smoothing steps

@*/
int  MGSetNumberSmoothUp(MG *mg,int n)
{ 
  int i;
  KSP ksp;
  for ( i=0; i<mg[0]->level; i++ ) {  
     SLESGetKSP(mg[i]->smoothu,&ksp);
     KSPSetIterations(ksp,n);
  }
  return 0;
}

/*@C
      MGSetCycles - Sets the number of cycles to use. 1 is V cycle, 2
                    is W cycle. Use MGSetCyclesOnLevel() for more 
                    complicated cycling.

  Input Parameters:
.   mg - the multigrid context 
.   n - the number of cycles

@*/
int MGSetCycles(MG *mg,int n)
{ 
  int i;
  for ( i=0; i<mg[0]->level; i++ ) {  
     mg[i]->cycles  = n; 
  }
  return 0;
}

