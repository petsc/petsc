

#ifndef lint
static char vcid[] = "$Id: snestest.c,v 1.1 1995/05/09 02:41:22 bsmith Exp bsmith $";
#endif

#include "draw.h"
#include "snesimpl.h"
#include "options.h"


/*@
     SNESTestJacobian - Tests whether a hand computed Jacobian 
        matches one compute via finite differences

  Input Parameters:

  Output Parameters:

@*/
int SNESTestJacobian(SNES snes)
{
  Mat    A = snes->jacobian,B;
  Vec    x = snes->vec_sol;
  int    ierr,flg;
  Scalar mone = -1.0;
  double norm,gnorm;

  MatConvert(A,MATSAME,&B);
  /* compute both versions of Jacobian */
  ierr = (*snes->ComputeJacobian)(snes,x,&A,&B,&flg,snes->jacP);CHKERR(ierr);
  ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,snes->funP);CHKERR(ierr);
 
  /* compare */
  ierr = MatAXPY(&mone,A,B);
  MatNorm(B,NORM_FROBENIUS,&norm);
  MatNorm(A,NORM_FROBENIUS,&gnorm);
  
  fprintf(stderr,"Norm of difference %g ratio %g\n",norm,norm/gnorm);
  return 0;
}
