/*$Id: qcg.c,v 1.67 1999/10/24 14:03:20 bsmith Exp bsmith $*/
/*
         Code to run conjugate gradient method subject to a constraint
   on the solution norm. This is used in Trust Region methods.
*/

#include "src/sles/ksp/kspimpl.h"
#include "src/sles/ksp/impls/qcg/qcg.h"

static int QuadraticRoots_Private(Vec,Vec,double*,double*,double*);

#undef __FUNC__  
#define __FUNC__ "KSPSolve_QCG"
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

  This method is intended for use in conjunction with the SNES trust region method
  for unconstrained minimization, SNESUMTR.

  Notes:
  Currently we allow symmetric preconditioning with the following scaling matrices:
      PCNONE:   D = Identity matrix
      PCJACOBI: D = diag [d_1, d_2, ...., d_n], where d_i = sqrt(H[i,i])
      PCICC:    D = L^T, implemented with forward and backward solves.
                Here L is an incomplete Cholesky factor of H.

 We should perhaps rewrite using PCApplyBAorAB().
 */
int KSPSolve_QCG(KSP ksp,int *its)
{
/* 
   Correpondence with documentation above:  
      B = g = gradient,
      X = s = step
   Note:  This is not coded correctly for complex arithmetic!
 */

  KSP_QCG      *pcgP = (KSP_QCG *) ksp->data;
  MatStructure pflag;
  Mat          Amat, Pmat;
  Vec          W, WA, WA2, R, P, ASP, BS, X, B;
  Scalar       zero = 0.0, negone = -1.0, scal, nstep, btx, xtax,beta, rntrn, step;
  double       ptasp, q1, q2, wtasp, bstp, rtr, xnorm, step1, step2, rnrm, p5 = 0.5;
  double       dzero = 0.0, bsnrm;
  int          i, cerr,  maxit, ierr;
  PC           pc = ksp->B;
  PCSide       side;
#if defined(PETSC_USE_COMPLEX)
  Scalar       cstep1, cstep2, ctasp, cbstp, crtr, cwtasp, cptasp;
#endif

  PetscFunctionBegin;

  if (ksp->transpose_solve) {
    SETERRQ(1,1,"Currently does not support transpose solve");
  }

  ksp->its = 0;
  maxit    = ksp->max_it;
  WA       = ksp->work[0];
  R        = ksp->work[1];
  P        = ksp->work[2]; 
  ASP      = ksp->work[3];
  BS       = ksp->work[4];
  W        = ksp->work[5];
  WA2      = ksp->work[6]; 
  X        = ksp->vec_sol;
  B        = ksp->vec_rhs;

  *its = 0;
  pcgP->info = 0;
  if (pcgP->delta <= dzero) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Input error: delta <= 0");
  ierr = KSPGetPreconditionerSide(ksp,&side);CHKERRQ(ierr);
  if (side != PC_SYMMETRIC) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Requires symmetric preconditioner!");

  /* Initialize variables */
  ierr = VecSet(&zero,W);CHKERRQ(ierr);	/* W = 0 */
  ierr = VecSet(&zero,X);CHKERRQ(ierr);	/* X = 0 */
  ierr = PCGetOperators(pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  /* Compute:  BS = D^{-1} B */
  ierr = PCApplySymmetricLeft(pc,B,BS);CHKERRQ(ierr);

  ierr = VecNorm(BS,NORM_2,&bsnrm);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its    = 0;
  ksp->rnorm  = bsnrm;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,bsnrm);
  KSPMonitor(ksp,0,bsnrm);
  cerr = (*ksp->converged)(ksp,0,bsnrm,ksp->cnvP);
  if (cerr) {*its =  0; PetscFunctionReturn(0);}

  /* Compute the initial scaled direction and scaled residual */
  ierr = VecCopy(BS,R);CHKERRQ(ierr);
  ierr = VecScale(&negone,R);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(R,R,&crtr);CHKERRQ(ierr); rtr = PetscReal(crtr);
#else
  ierr = VecDot(R,R,&rtr);CHKERRQ(ierr);
#endif

  for (i=0; i<=maxit; i++) {
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

    /* Compute:  asp = D^{-T}*A*D^{-1}*p  */
    ierr = PCApplySymmetricRight(pc,P,WA);CHKERRQ(ierr);
    ierr = MatMult(Amat,WA,WA2);CHKERRQ(ierr);
    ierr = PCApplySymmetricLeft(pc,WA2,ASP);CHKERRQ(ierr);

    /* Check for negative curvature */
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot(P,ASP,&cptasp);CHKERRQ(ierr);
    ptasp = PetscReal(cptasp);
#else
    ierr = VecDot(P,ASP,&ptasp);CHKERRQ(ierr);	/* ptasp = p^T asp */
#endif
    if (ptasp <= dzero) {

      /* Scaled negative curvature direction:  Compute a step so that
         ||w + step*p|| = delta and QS(w + step*p) is least */

       if (i == 0) {
         ierr = VecCopy(P,X);CHKERRQ(ierr);
         ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);
         scal = pcgP->delta / xnorm;
         ierr = VecScale(&scal,X);CHKERRQ(ierr);
       } else {
         /* Compute roots of quadratic */
         ierr = QuadraticRoots_Private(W,P,&pcgP->delta,&step1,&step2);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
         ierr = VecDot(W,ASP,&cwtasp);CHKERRQ(ierr); wtasp = PetscReal(cwtasp);
         ierr = VecDot(BS,P,&cbstp);CHKERRQ(ierr);   bstp  = PetscReal(cbstp);
#else
         ierr = VecDot(W,ASP,&wtasp);CHKERRQ(ierr);
         ierr = VecDot(BS,P,&bstp);CHKERRQ(ierr);
#endif
         ierr = VecCopy(W,X);CHKERRQ(ierr);
         q1 = step1*(bstp + wtasp + p5*step1*ptasp);
         q2 = step2*(bstp + wtasp + p5*step2*ptasp);
#if defined(PETSC_USE_COMPLEX)
         if (q1 <= q2) {
           cstep1 = step1; ierr = VecAXPY(&cstep1,P,X);CHKERRQ(ierr);
         } else {
           cstep2 = step2; ierr = VecAXPY(&cstep2,P,X);CHKERRQ(ierr);
         }
#else
         if (q1 <= q2) {ierr = VecAXPY(&step1,P,X);CHKERRQ(ierr);}
         else          {ierr = VecAXPY(&step2,P,X);CHKERRQ(ierr);}
#endif
       }
       pcgP->ltsnrm = pcgP->delta;    /* convergence in direction of */
       pcgP->info = 1;	        /* negative curvature */
       if (i == 0) {
         PLogInfo(ksp,"KSPSolve_QCG: negative curvature: delta=%g\n",pcgP->delta);
       } else {
         PLogInfo(ksp,"KSPSolve_QCG: negative curvature: step1=%g, step2=%g, delta=%g\n",step1,step2,pcgP->delta);
       }
         
    } else {
 
       /* Compute step along p */

       step = rtr/ptasp;
       ierr = VecCopy(W,X);CHKERRQ(ierr);	   /*  x = w  */
       ierr = VecAXPY(&step,P,X);CHKERRQ(ierr);   /*  x <- step*p + x  */
       ierr = VecNorm(X,NORM_2,&pcgP->ltsnrm);CHKERRQ(ierr);

       if (pcgP->ltsnrm > pcgP->delta ) {

         /* Since the trial iterate is outside the trust region, 
             evaluate a constrained step along p so that 
                      ||w + step*p|| = delta 
            The positive step is always better in this case. */

         if (i == 0) {
           scal = pcgP->delta / pcgP->ltsnrm;
           ierr = VecScale(&scal,X);CHKERRQ(ierr);
         } else {
           /* Compute roots of quadratic */
           ierr = QuadraticRoots_Private(W,P,&pcgP->delta,&step1,&step2);CHKERRQ(ierr);
           ierr = VecCopy(W,X);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
           cstep1 = step1; ierr = VecAXPY(&cstep1,P,X);CHKERRQ(ierr);
#else
           ierr = VecAXPY(&step1,P,X);CHKERRQ(ierr);  /*  x <- step1*p + x  */
#endif
         }
         pcgP->ltsnrm = pcgP->delta;
         pcgP->info = 2;	/* convergence along constrained step */
         if (i == 0) {
           PLogInfo(ksp,"KSPSolve_QCG: constrained step: delta=%g\n",pcgP->delta);
         } else {
           PLogInfo(ksp,"KSPSolve_QCG: constrained step: step1=%g, step2=%g, delta=%g\n",step1,step2,pcgP->delta);
         }

       } else {

         /* Evaluate the current step */

         ierr = VecCopy(X,W);CHKERRQ(ierr);	/* update interior iterate */
         nstep = -step;
         ierr = VecAXPY(&nstep,ASP,R);CHKERRQ(ierr); /* r <- -step*asp + r */
         ierr = VecNorm(R,NORM_2,&rnrm);CHKERRQ(ierr);

         ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
         ksp->rnorm                                    = rnrm;
         ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
         KSPLogResidualHistory(ksp,rnrm);
         KSPMonitor(ksp,i+1,rnrm);
         cerr = (*ksp->converged)(ksp,i+1,rnrm,ksp->cnvP);
         if (cerr) {                 /* convergence for */
           pcgP->info = 3;          /* truncated step */
#if defined(PETSC_USE_COMPLEX)               
           PLogInfo(ksp,"KSPSolve_QCG: truncated step: step=%g, rnrm=%g, delta=%g\n",PetscReal(step),rnrm,pcgP->delta);
#else
           PLogInfo(ksp,"KSPSolve_QCG: truncated step: step=%g, rnrm=%g, delta=%g\n",step,rnrm,pcgP->delta);
#endif
         }
      }
    }
    if (pcgP->info) break;	/* Convergence has been attained */
    else {		/* Compute a new AS-orthogonal direction */
      ierr = VecDot(R,R,&rntrn);CHKERRQ(ierr);
      beta = rntrn/rtr;
      ierr = VecAYPX(&beta,R,P);CHKERRQ(ierr);	/*  p <- r + beta*p  */
#if defined(PETSC_USE_COMPLEX)
      rtr = PetscReal(rntrn);
#else
      rtr = rntrn;
#endif
    }
  }
  /* Unscale x */
  ierr = VecCopy(X,WA2);CHKERRQ(ierr);
  ierr = PCApplySymmetricRight(pc,WA2,X);CHKERRQ(ierr);

  ierr = MatMult(Amat,X,WA);CHKERRQ(ierr);
  ierr = VecDot(B,X,&btx);CHKERRQ(ierr);
  ierr = VecDot(X,WA,&xtax);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  pcgP->quadratic = PetscReal(btx) + p5* PetscReal(xtax);
#else
  pcgP->quadratic = btx + p5*xtax;              /* Compute q(x) */
#endif
  *its = i+1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_QCG"
int KSPSetUp_QCG(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  /* Check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(2,0,"no right preconditioning for QCG");
  } else if (ksp->pc_side == PC_LEFT) {
    SETERRQ(2,0,"no left preconditioning for QCG");
  }

  /* Get work vectors from user code */
  ierr = KSPDefaultGetWork(ksp, 7);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDestroy_QCG" 
int KSPDestroy_QCG(KSP ksp)
{
  KSP_QCG *cgP = (KSP_QCG *) ksp->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultFreeWork( ksp );CHKERRQ(ierr);
  
  /* Free the context variable */
  ierr = PetscFree(cgP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_QCG"
int KSPCreate_QCG(KSP ksp)
{
  int     ierr;
  KSP_QCG *cgP;

  PetscFunctionBegin;
  cgP  = (KSP_QCG*) PetscMalloc(sizeof(KSP_QCG));CHKPTRQ(cgP);
  ierr = PetscMemzero(cgP,sizeof(KSP_QCG));CHKERRQ(ierr);
  PLogObjectMemory(ksp,sizeof(KSP_QCG));
  ksp->data                      = (void *) cgP;
  ksp->pc_side                   = PC_SYMMETRIC;
  ksp->calc_res                  = 1;
  ksp->guess_zero                = 1; 
  ksp->ops->setup                = KSPSetUp_QCG;
  ksp->ops->solve                = KSPSolve_QCG;
  ksp->ops->destroy              = KSPDestroy_QCG;
  ksp->converged                 = KSPDefaultConverged;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ---------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "QuadraticRoots_Private"
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
static int QuadraticRoots_Private(Vec s,Vec p,double *delta,double *step1,double *step2)
{ 
  double zero = 0.0, dsq, ptp, pts, rad, sts;
  int    ierr;
#if defined(PETSC_USE_COMPLEX)
  Scalar cptp, cpts, csts;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(p,s,&cpts);CHKERRQ(ierr); pts = PetscReal(cpts);
  ierr = VecDot(p,p,&cptp);CHKERRQ(ierr); ptp = PetscReal(cptp);
  ierr = VecDot(s,s,&csts);CHKERRQ(ierr); sts = PetscReal(csts);
#else
  ierr = VecDot(p,s,&pts);CHKERRQ(ierr);
  ierr = VecDot(p,p,&ptp);CHKERRQ(ierr);
  ierr = VecDot(s,s,&sts);CHKERRQ(ierr);
#endif
  dsq  = (*delta)*(*delta);
  rad  = sqrt((pts*pts) - ptp*(sts - dsq));
  if (pts > zero) {
    *step2 = -(pts + rad)/ptp;
    *step1 = (sts - dsq)/(ptp * *step2);
  } else {
    *step1 = -(pts - rad)/ptp;
    *step2 = (sts - dsq)/(ptp * *step1);
  }
  PetscFunctionReturn(0);
}
