#ifndef lint
static char vcid[] = "$Id: qcg.c,v 1.3 1995/07/24 22:23:49 curfman Exp curfman $";
#endif

#include <stdio.h>
#include <math.h>
#include "kspimpl.h"
#include "qcg.h"

static int QuadraticRoots_Private(Vec,Vec,double*,double*,double*);

/* 
  KSPSolve_QCG - Use preconditioned conjugate gradient to compute 
  an approximate minimizer of the quadratic function 

            q(s) = g^T * s + .5 * s^T * H * s

   subject to the Euclidean norm trust region constraint

            || D * s || <= delta,

   where 

     delta is the trust region radius, 
     g is the gradient vector, and
     H is Hessian matrix,
     D is a scaling matrix.

   On termination pcgP->info is set as follows:
$  1 if convergence is reached along a negative curvature direction,
$  2 if convergence is reached along a constrained step,
$  3 if convergence is reached along a truncated step.

  This method is intended primarily for use in conjunction with the
  SNES trust region method for unconstrained minimization, SNES_UM_TR.

  Note:  This currently is coded with forward/backward solves for
  incomplete Cholesky preconditioning.  Eventually this will be generalized.
 */

int KSPSolve_QCG(KSP itP,int *its)
{
/* 
   Correpondence with documentation above:  
      B = g = gradient,
      X = s = step
   Note:  This is not coded correctly for complex arithmetic!
 */

  KSP_QCG      *pcgP = (KSP_QCG *) itP->MethodPrivate;
  MatStructure pflag;
  Mat          Amat, Pmat;
  Vec          W, WA, R, P, ASP, BS, X, B;
  Scalar       zero = 0.0, negone = -1.0, scal, nstep, btx, xtax;
  Scalar       beta, rntrn, step;
  double       dzero = 0.0, bsnrm, ptasp, q1, q2, wtasp, bstp, rtr;
  double       xnorm, step1, step2, rnrm, p5 = 0.5, *history;
  int          i, cerr, hist_len, maxit, ierr;
  PC           pc = itP->B;
/*  PCMethod     pcmethod; */
#if defined(PETSC_COMPLEX)
  Scalar       cstep1, cstep2, ctasp, cbstp, crtr, cwtasp, cptasp;
#endif

  history  = itP->residual_history;
  hist_len = itP->res_hist_size;
  maxit    = itP->max_it;
  WA       = itP->work[0];
  R        = itP->work[1];
  P        = itP->work[2]; 
  ASP      = itP->work[3];
  BS       = itP->work[4];
  W        = itP->work[5]; 
  X        = itP->vec_sol;
  B        = itP->vec_rhs;

/*  PCGetMethodFromContext(pc,&pcmethod);
  if (pcmethod != PCICC && pcmethod != PCJACOBI) 
    SETERRQ(1,"Use only PCICC and PCJACOBI methods"); */
  *its = 0;
  pcgP->info = 0;
  if (pcgP->delta <= dzero) SETERRQ(1,"KSPSolve_QCG: Input error: delta <= 0");

  /* Initialize variables */
  VecSet(&zero,W);			/* W = 0 */
  VecSet(&zero,X);			/* X = 0 */
  ierr = PCGetOperators(pc,&Amat,&Pmat,&pflag); CHKERRQ(ierr);

  /* Compute:  BS = D^{-1} B */
  PCApply(pc,B,BS);
  /* ierr = MatForwardSolve(Factmat,B,BS); CHKERRQ(ierr); */
  VecNorm(BS,&bsnrm);
  MONITOR(itP,bsnrm,0);
  if (history) history[0] = bsnrm;
  cerr = (*itP->converged)(itP,0,bsnrm,itP->cnvP);
  if (cerr) {*its =  0; return 0;}

  /* Compute the initial scaled direction and scaled residual */
  VecCopy(BS,R);
  VecScale(&negone,R);
  VecCopy(R,P);
#if defined(PETSC_COMPLEX)
  VecDot(R,R,&crtr); rtr = real(crtr);
#else
  VecDot(R,R,&rtr);
#endif

  for (i=0; i<=maxit; i++) {

    /* Compute:  asp = L^{-1}*A*L^{-T}*p  */
    /* VecCopy(P,WA);
       MatBackwardSolve(Factmat,WA,WA);
       MatMult(Amat,WA,ASP);
       MatForwardSolve(Factmat,ASP,ASP); */
    MatMult(Amat,P,WA);
    PCApply(pc,WA,ASP);

    /* Check for negative curvature */
#if defined(PETSC_COMPLEX)
    VecDot(P,ASP,&cptasp);
    ptasp = real(cptasp);
#else
    VecDot(P,ASP,&ptasp);		/* ptasp = p^T asp */
#endif
    if (ptasp <= dzero) {

      /* Scaled negative curvature direction:  Compute a step so that
         ||w + step*p|| = delta and QS(w + step*p) is least */

       if (i == 0) {
         VecCopy(P,X);
         VecNorm(X,&xnorm);
         scal = pcgP->delta / xnorm;
         VecScale(&scal,X);
       } else {
         /* Compute roots of quadratic */
         ierr = QuadraticRoots_Private(W,P,&pcgP->delta,&step1,&step2);
         CHKERRQ(ierr);
#if defined(PETSC_COMPLEX)
         VecDot(W,ASP,&cwtasp); wtasp = real(cwtasp);
         VecDot(BS,P,&cbstp);   bstp  = real(cbstp);
#else
         VecDot(W,ASP,&wtasp);
         VecDot(BS,P,&bstp);
#endif
         VecCopy(W,X);
         q1 = step1*(bstp + wtasp + p5*step1*ptasp);
         q2 = step2*(bstp + wtasp + p5*step2*ptasp);
#if defined(PETSC_COMPLEX)
         if (q1 <= q2) {cstep1 = step1; VecAXPY(&cstep1,P,X);}
         else          {cstep2 = step2; VecAXPY(&cstep2,P,X);}
#else
         if (q1 <= q2) VecAXPY(&step1,P,X);
         else          VecAXPY(&step2,P,X);
#endif
       }
       pcgP->ltsnrm = pcgP->delta;    /* convergence in direction of */
       pcgP->info = 1;	        /* negative curvature */
       if (i == 0) {
         PLogInfo((PetscObject)itP,
           "negative curvature:  delta=%g\n", pcgP->delta );
       } else {
         PLogInfo((PetscObject)itP,
           "negative curvature:  step1=%g, step2=%g, delta=%g\n",
                    step1, step2, pcgP->delta );
       }
         
    } else {
 
       /* Compute step along p */

       step = rtr/ptasp;
       VecCopy(W,X);		/*  x = w  */
       VecAXPY(&step,P,X);	/*  x <- step*p + x  */
       VecNorm(X,&pcgP->ltsnrm);

       if (pcgP->ltsnrm > pcgP->delta ) {

         /* Since the trial iterate is outside the trust region, 
             evaluate a constrained step along p so that 
                      ||w + step*p|| = delta 
            The positive step is always better in this case. */

         if (i == 0) {
           scal = pcgP->delta / pcgP->ltsnrm;
           VecScale(&scal,X); 
         } else {
           /* Compute roots of quadratic */
           ierr = QuadraticRoots_Private(W,P,&pcgP->delta,&step1,&step2);
           CHKERRQ(ierr);
           VecCopy(W,X);
#if defined(PETSC_COMPLEX)
           cstep1 = step1; VecAXPY(&cstep1,P,X);
#else
           VecAXPY(&step1,P,X);	/*  x <- step1*p + x  */
#endif
         }
         pcgP->ltsnrm = pcgP->delta;
         pcgP->info = 2;	/* convergence along constrained step */
         if (i == 0) {
           PLogInfo((PetscObject)itP,
             "constrained step:   delta=%g\n", pcgP->delta );
         } else {
           PLogInfo((PetscObject)itP,
             "constrained step:  step1=%g, step2=%g, delta=%g\n",
              step1, step2, pcgP->delta );
         }

       } else {

         /* Evaluate the current step */

         VecCopy(X,W);		/* update interior iterate */
         nstep = -step;
         VecAXPY(&nstep,ASP,R);	/*  r <- -step*asp + r  */
         VecNorm(R,&rnrm);

         if (history && hist_len > i + 1) history[i+1] = rnrm;
         MONITOR(itP,rnrm,i+1);
         cerr = (*itP->converged)(itP,i+1,rnrm,itP->cnvP);
         if (cerr) {                 /* convergence for */
           pcgP->info = 3;          /* truncated step */
#if defined(PETSC_COMPLEX)               
           PLogInfo((PetscObject)itP,
             "truncated step:  step=%g, rnrm=%g, delta=%g\n", 
              real(step), rnrm, pcgP->delta );
#else
           PLogInfo((PetscObject)itP,
             "truncated step:  step=%g, rnrm=%g, delta=%g\n", 
              step, rnrm, pcgP->delta );
#endif
         }
      }
    }
    if (pcgP->info) break;	/* Convergence has been attained */
    else {		/* Compute a new AS-orthogonal direction */
      VecDot(R,R,&rntrn);
      beta = rntrn/rtr;
      VecAYPX(&beta,R,P);		/*  p <- r + beta*p  */
#if defined(PETSC_COMPLEX)
      rtr = real(rntrn);
#else
      rtr = rntrn;
#endif
    }
  }
/*  MatBackwardSolve(Factmat,X,X); */                   /* Unscale x */
  MatMult(Amat,X,WA);
  VecDot(B,X,&btx);
  VecDot(X,WA,&xtax);
#if defined(PETSC_COMPLEX)
  pcgP->quadratic = real(btx) + p5* real(xtax);
#else
  pcgP->quadratic = btx + p5*xtax;              /* Compute q(x) */
#endif
  *its = i+1;
  return 0;
}

static int KSPSetUp_QCG(KSP itP)
{
  int ierr;

  /* Check user parameters and functions */
  if ( itP->right_pre ) {
    SETERRQ(2,"KSPSetUp_QCG: no right-inverse preconditioning for QCG");}
  if ((ierr = KSPCheckDef( itP ))) return ierr;

  /* Get work vectors from user code */
  if ((ierr = KSPiDefaultGetWork( itP, 6 ))) return ierr;
  return 0;
}

static int KSPDestroy_QCG(PetscObject obj)
{
  KSP itP = (KSP) obj;
  KSP_QCG *cgP = (KSP_QCG *) itP->MethodPrivate;

  KSPiDefaultFreeWork( itP );
  
  /* Free the context variable */
  PETSCFREE(cgP); 
  return 0;
}

int KSPCreate_QCG(KSP itP)
{
  KSP_QCG *cgP;
  cgP = (KSP_QCG*) PETSCMALLOC(sizeof(KSP_QCG));  CHKPTRQ(cgP);
  itP->MethodPrivate = (void *) cgP;
  itP->type                 = KSPQCG;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_QCG;
  itP->solver               = KSPSolve_QCG;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPDestroy_QCG;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
/* ---------------------------------------------------------- */
/* 
  QuadraticRoots_Private - Computes the roots of the quadratic,
         ||s + step*p|| - delta = 0 
   such that step1 >= 0 >= step2.
   where
      delta:
        On entry delta must contain scalar delta.
        On exit delta is unchanged.
      step1:
        On entry step1 need not be specified.
        On exit step1 contains the non-negative root.
      step2:
        On entry step2 need not be specified.
        On exit step2 contains the non-positive root.
   C code is translated from the Fortran version of the MINPACK-2 Project,
   Argonne National Laboratory, Brett M. Averick and Richard G. Carter.
*/
#if !defined(NLSQR)
#define NLSQR(a)        ( (a)*(a) )
#endif
static int QuadraticRoots_Private(Vec s,Vec p,double *delta,
                                  double *step1,double *step2)
{ 
#if defined(PETSC_COMPLEX)
  SETERRQ(1,"QuadraticRoots_Private:  not done for complex numbers");
#else
  double zero = 0.0, dsq, ptp, pts, rad, sts;
  VecDot(p,s,&pts);
  VecDot(p,p,&ptp);
  VecDot(s,s,&sts);
  dsq = NLSQR(*delta);
  rad = sqrt(NLSQR(pts) - ptp*(sts - dsq));
  if (pts > zero) {
    *step2 = -(pts + rad)/ptp;
    *step1 = (sts - dsq)/(ptp * *step2);
  } else {
    *step1 = -(pts - rad)/ptp;
    *step2 = (sts - dsq)/(ptp * *step1);
  }
  return 0;
#endif
}
