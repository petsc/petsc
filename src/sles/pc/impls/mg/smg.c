#ifndef lint
static char vcid[] = "$Id: $";
#endif
/*
     Additive Multigrid V Cycle routine    
*/
#include "mgimpl.h"

/*
       MGACycle - Given an MG structure created with MGCreate() runs 
                  one cycle down through the levels and back up. Applys
                  the smoothers in an additive manner.

    Iput Parameters:
.   mg - structure created with  MGCreate().

*/
int MGACycle(MG *mg)
{
  int    i, l = mg[0]->level,its,ierr;
  Scalar zero = 0.0;

  for ( i=0; i<l; i++ ) {
    MatMult(mg[i]->restrict,  mg[i]->b, mg[i+1]->b); 
  }
  for ( i=0; i<l; i++ ) {
    VecSet(&zero,mg[i]->x); 
    SLESSolve(mg[i]->smoothd, mg[i]->b, mg[i]->x,&its); 
  }
  VecSet(&zero,mg[l]->x); 
  ierr = SLESSolve(mg[l]->csles, mg[l]->b, mg[l]->x,&its); CHKERR(ierr);
  for ( i=l-1; i>-1; i-- ) {  
    MatMultTransAdd(mg[i]->interpolate,mg[i+1]->x,mg[i]->x,mg[i]->x); 
  }
  return 0;
}
