#ifndef lint
static char vcid[] = "$Id: smg.c,v 1.6 1995/07/20 23:43:16 bsmith Exp bsmith $";
#endif
/*
     Additive Multigrid V Cycle routine    
*/
#include "mgimpl.h"

/*
       MGACycle_Private - Given an MG structure created with MGCreate() runs 
                  one cycle down through the levels and back up. Applys
                  the smoothers in an additive manner.

    Iput Parameters:
.   mg - structure created with  MGCreate().

*/
int MGACycle_Private(MG *mg)
{
  int    i, l = mg[0]->level,its,ierr;
  Scalar zero = 0.0;

  for ( i=0; i<l; i++ ) {
    ierr = MatMult(mg[i]->restrct,  mg[i]->b, mg[i+1]->b);CHKERRQ(ierr);
  }
  for ( i=0; i<l; i++ ) {
    ierr = VecSet(&zero,mg[i]->x);CHKERRQ(ierr); 
    ierr = SLESSolve(mg[i]->smoothd, mg[i]->b, mg[i]->x,&its);CHKERRQ(ierr); 
  }
  ierr = VecSet(&zero,mg[l]->x); 
  ierr = SLESSolve(mg[l]->csles, mg[l]->b, mg[l]->x,&its);CHKERRQ(ierr);
  for ( i=l-1; i>-1; i-- ) {  
    ierr = MatMultTransAdd(mg[i]->interpolate,mg[i+1]->x,mg[i]->x,mg[i]->x);
    CHKERRQ(ierr);
  }
  return 0;
}
