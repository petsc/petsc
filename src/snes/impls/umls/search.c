/*$Id: search.c,v 1.27 2001/03/23 23:24:18 balay Exp bsmith $*/

/*
     The subroutine mcstep is taken from the work of Jorge Nocedal.
     this is a variant of More' and Thuente's routine.

c     subroutine mcstep
c
c     the purpose of mcstep is to compute a safeguarded step for
c     a linesearch and to update an interval of uncertainty for
c     a minimizer of the function.
c
c     the parameter stx contains the step with the least function
c     value. the parameter stp contains the current step. it is
c     assumed that the derivative at stx is negative in the
c     direction of the step. if bracket is set true then a
c     minimizer has been bracketed in an interval of uncertainty
c     with endpoints stx and sty.
c
c     the subroutine statement is
c
c       subroutine mcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,bracket,
c                        stpmin,stpmax,info)
c
c     where
c
c       stx, fx, and dx are variables which specify the step,
c         the function, and the derivative at the best step obtained
c         so far. the derivative must be negative in the direction
c         of the step, that is, dx and stp-stx must have opposite
c         signs. on output these parameters are updated appropriately.
c
c       sty, fy, and dy are variables which specify the step,
c         the function, and the derivative at the other endpoint of
c         the interval of uncertainty. on output these parameters are
c         updated appropriately.
c
c       stp, fp, and dp are variables which specify the step,
c         the function, and the derivative at the current step.
c         if bracket is set true then on input stp must be
c         between stx and sty. on output stp is set to the new step.
c
c       bracket is a logical variable which specifies if a minimizer
c         has been bracketed. if the minimizer has not been bracketed
c         then on input bracket must be set false. if the minimizer
c         is bracketed then on output bracket is set true.
c
c       stpmin and stpmax are input variables which specify lower
c         and upper bounds for the step.
c
c       info is an integer output variable set as follows:
c         if info = 1,2,3,4,5, then the step has been computed
c         according to one of the five cases below. otherwise
c         info = 0, and this indicates improper input parameters.
c
c     subprograms called
c
c       fortran-supplied ... abs,max,min,sqrt
c
c     argonne national laboratory. minpack project. june 1983
c     jorge j. more', david j. thuente
nc
 */
#include "src/snes/impls/umls/umls.h"

#undef __FUNCT__  
#define __FUNCT__ "SNESStep"
int SNESStep(SNES snes,PetscReal *stx,PetscReal *fx,PetscReal *dx,
    PetscReal *sty,PetscReal *fy,PetscReal *dy,PetscReal *stp,PetscReal *fp,PetscReal *dp)
{
  SNES_UM_LS *neP = (SNES_UM_LS*)snes->data;
  PetscReal     gamma1,p,q,r,s,sgnd,stpc,stpf,stpq,theta;
  PetscReal     two = 2.0,zero = 0.0;
  int        bound;

  PetscFunctionBegin;
  /* Check the input parameters for errors */
  neP->infoc = 0;
  if (neP->bracket && (*stp <= PetscMin(*stx,*sty) || (*stp >= PetscMax(*stx,*sty))))
    SETERRQ(PETSC_ERR_PLIB,"bad stp in bracket");
  if (*dx * (*stp-*stx) >= zero) SETERRQ(PETSC_ERR_PLIB,"dx * (stp-stx) >= 0");
  if (neP->stepmax < neP->stepmin) SETERRQ(PETSC_ERR_PLIB,"stepmax > stepmin");

  /* Determine if the derivatives have opposite sign */
  sgnd = *dp * (*dx/PetscAbsDouble(*dx));

/*   Case 1: a higher function value.
     the minimum is bracketed. if the cubic step is closer
     to stx than the quadratic step, the cubic step is taken,
     else the average of the cubic and quadratic steps is taken.
 */
  if (*fp > *fx) {
    neP->infoc = 1;
    bound = 1;
    theta = 3 * (*fx - *fp) / (*stp - *stx) + *dx + *dp;
    s = PetscMax(PetscAbsDouble(theta),PetscAbsDouble(*dx));
    s = PetscMax(s,PetscAbsDouble(*dp));
    gamma1 = s*sqrt(pow(theta/s,two) - (*dx/s)*(*dp/s));
    if (*stp < *stx) gamma1 = -gamma1;
    p = (gamma1 - *dx) + theta;
    q = ((gamma1 - *dx) + gamma1) + *dp;
    r = p/q;
    stpc = *stx + r*(*stp - *stx);
    stpq = *stx + ((*dx/((*fx-*fp)/(*stp-*stx)+*dx))*0.5) * (*stp - *stx);
    if (PetscAbsDouble(stpc-*stx) < PetscAbsDouble(stpq-*stx)) {
      stpf = stpc;
    } else {
      stpf = stpc + 0.5*(stpq - stpc);
    }
    neP->bracket = 1;
  }
  /* 
     Case 2: A lower function value and derivatives of
     opposite sign. the minimum is bracketed. if the cubic
     step is closer to stx than the quadratic (secant) step,
     the cubic step is taken, else the quadratic step is taken.
  */
  else if (sgnd < zero) {
    neP->infoc = 2;
    bound = 0;
    theta = 3*(*fx - *fp)/(*stp - *stx) + *dx + *dp;
    s = PetscMax(PetscAbsDouble(theta),PetscAbsDouble(*dx));
    s = PetscMax(s,PetscAbsDouble(*dp));
    gamma1 = s*sqrt(pow(theta/s,two) - (*dx/s)*(*dp/s));
    if (*stp > *stx) gamma1 = -gamma1;
    p = (gamma1 - *dp) + theta;
    q = ((gamma1 - *dp) + gamma1) + *dx;
    r = p/q;
    stpc = *stp + r*(*stx - *stp);
    stpq = *stp + (*dp/(*dp-*dx))*(*stx - *stp);
    if (PetscAbsDouble(stpc-*stp) > PetscAbsDouble(stpq-*stp)) stpf = stpc;
    else                                                       stpf = stpq;
    neP->bracket = 1;
  }

/*   Case 3: A lower function value, derivatives of the
     same sign, and the magnitude of the derivative decreases.
     the cubic step is only used if the cubic tends to infinity
     in the direction of the step or if the minimum of the cubic
     is beyond stp. otherwise the cubic step is defined to be
     either stepmin or stepmax. the quadratic (secant) step is also
     computed and if the minimum is bracketed then the the step
     closest to stx is taken, else the step farthest away is taken.
 */

  else if (PetscAbsDouble(*dp) < PetscAbsDouble(*dx)) {
    neP->infoc = 3;
    bound = 1;
    theta = 3*(*fx - *fp)/(*stp - *stx) + *dx + *dp;
    s = PetscMax(PetscAbsDouble(theta),PetscAbsDouble(*dx));
    s = PetscMax(s,PetscAbsDouble(*dp));

    /* The case gamma1 = 0 only arises if the cubic does not tend
       to infinity in the direction of the step. */
    gamma1 = s*sqrt(PetscMax(zero,pow(theta/s,two) - (*dx/s)*(*dp/s)));
    if (*stp > *stx) gamma1 = -gamma1;
    p = (gamma1 - *dp) + theta;
    q = (gamma1 + (*dx - *dp)) + gamma1;
    r = p/q;
    if (r < zero && gamma1 != zero) stpc = *stp + r*(*stx - *stp);
    else if (*stp > *stx)        stpc = neP->stepmax;
    else                         stpc = neP->stepmin;
    stpq = *stp + (*dp/(*dp-*dx)) * (*stx - *stp);
    if (neP->bracket) {
      if (PetscAbsDouble(*stp-stpc) < PetscAbsDouble(*stp-stpq)) stpf = stpc;
      else                                                       stpf = stpq;
    }
    else {
      if (PetscAbsDouble(*stp-stpc) > PetscAbsDouble(*stp-stpq)) stpf = stpc;
      else                                                       stpf = stpq;
    }
  }

/*   Case 4: A lower function value, derivatives of the
     same sign, and the magnitude of the derivative does
     not decrease. if the minimum is not bracketed, the step
     is either stpmin or stpmax, else the cubic step is taken.
 */
  else {
    neP->infoc = 4;
    bound = 0;
    if (neP->bracket) {
      theta = 3*(*fp - *fy)/(*sty - *stp) + *dy + *dp;
      s = PetscMax(PetscAbsDouble(theta),PetscAbsDouble(*dy));
      s = PetscMax(s,PetscAbsDouble(*dp));
      gamma1 = s*sqrt(pow(theta/s,two) - (*dy/s)*(*dp/s));
      if (*stp > *sty) gamma1 = -gamma1;
      p = (gamma1 - *dp) + theta;
      q = ((gamma1 - *dp) + gamma1) + *dy;
      r = p/q;
      stpc = *stp + r*(*sty - *stp);
      stpf = stpc;
    } else if (*stp > *stx) {
      stpf = neP->stepmax;
    } else {
      stpf = neP->stepmin;
    }
  }

  /* Update the interval of uncertainty.  This update does not
     depend on the new step or the case analysis above. */

  if (*fp > *fx) {
    *sty = *stp;
    *fy = *fp;
    *dy = *dp;
  } else {
    if (sgnd < zero) {
      *sty = *stx;
      *fy = *fx;
      *dy = *dx;
    }
    *stx = *stp;
    *fx = *fp;
    *dx = *dp;
  }

  /* Compute the new step and safeguard it */
  stpf = PetscMin(neP->stepmax,stpf);
  stpf = PetscMax(neP->stepmin,stpf);
  *stp = stpf;
  if (neP->bracket && bound) {
    if (*sty > *stx) *stp = PetscMin(*stx+0.66*(*sty-*stx),*stp);
    else             *stp = PetscMax(*stx+0.66*(*sty-*stx),*stp);
  }
  PetscFunctionReturn(0);
}
