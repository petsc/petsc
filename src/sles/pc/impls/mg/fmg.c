#ifndef lint
static char vcid[] = "$Id: fmg.c,v 1.4 1995/07/20 23:43:16 bsmith Exp bsmith $";
#endif
/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include "mgimpl.h"

extern int MGMCycle_Private(MG *);

/*
       MGFCycle_Private - Given an MG structure created with MGCreate() runs 
               full multigrid. 

    Iput Parameters:
.   mg - structure created with MGCreate().

    Note: This may not be what others call full multigrid. What we
          do is restrict the rhs to all levels, then starting 
          on the coarsest level work our way up generating 
          initial guess for the next level. This provides an
          improved preconditioner but not a great improvement.
*/
int MGFCycle_Private(MG *mg)
{
  int    i, l = mg[0]->level;
  Scalar zero = 0.0;

  /* restrict the RHS through all levels to coarsest. */
  for ( i=0; i<l; i++ ){
    MatMult(mg[i]->restrct,  mg[i]->b, mg[i+1]->b ); 
  }
  
  /* work our way up through the levels */
  for ( i=l; i>0; i-- ) {
    MGMCycle_Private(&mg[i]); 
    VecSet(&zero, mg[i-1]->x ); 
    MatMultTransAdd(mg[i-1]->interpolate,mg[i]->x,mg[i-1]->x,mg[i-1]->x); 
  }
  MGMCycle_Private(mg); 
  return 0;
}


/*
       MGKCycle_Private - Given an MG structure created with MGCreate() runs 
               full Kascade MG solve.

    Iput Parameters:
.   mg - structure created with MGCreate().

    Note: This may not be what others call Kascadic MG.
*/
int MGKCycle_Private(MG *mg)
{
  int    i, l = mg[0]->level,its,ierr;
  Scalar zero = 0.0;

  /* restrict the RHS through all levels to coarsest. */
  for ( i=0; i<l; i++ ){
    MatMult(mg[i]->restrct,  mg[i]->b, mg[i+1]->b ); 
  }
  
  /* work our way up through the levels */
  for ( i=l; i>0; i-- ) {
    ierr = SLESSolve(mg[i]->smoothd,mg[i]->b,mg[i]->x,&its); CHKERRQ(ierr);
    VecSet(&zero, mg[i-1]->x ); 
    MatMultTransAdd(mg[i-1]->interpolate,mg[i]->x,mg[i-1]->x,mg[i-1]->x); 
  }
  ierr = SLESSolve(mg[0]->smoothd,mg[0]->b,mg[0]->x,&its); CHKERRQ(ierr);
  return 0;
}
