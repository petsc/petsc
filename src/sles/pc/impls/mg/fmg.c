#ifndef lint
static char vcid[] = "$Id: $";
#endif
/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include "mgimpl.h"

extern int MGMCycle(MG *);

/*
       MGFCycle - Given an MG structure created with MGCreate() runs 
               full multigrid. 

    Iput Parameters:
.   mg - structure created with MGCreate().

    Note: This may not be what others call full multigrid. What we
          do is restrict the rhs to all levels, then starting 
          on the coarsest level work our way up generating 
          initial guess for the next level. This provides an
          improved preconditioner but not a great improvement.
*/
int MGFCycle(MG *mg)
{
  int    i, l = mg[0]->level;
  Scalar zero = 0.0;

  /* restrict the RHS through all levels to coarsest. */
  for ( i=0; i<l; i++ ){
    MatMult(mg[i]->restrict,  mg[i]->b, mg[i+1]->b ); 
  }
  
  /* work our way up through the levels */
  for ( i=l; i>0; i-- ) {
    MGMCycle(&mg[i]); 
    VecSet(&zero, mg[i-1]->x ); 
    MatMultTransAdd(mg[i-1]->interpolate,mg[i]->x,mg[i-1]->x,mg[i-1]->x); 
  }
  MGMCycle(mg); 
  return 0;
}

