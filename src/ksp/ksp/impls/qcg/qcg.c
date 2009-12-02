#define PETSCKSP_DLL

#include "private/kspimpl.h"             /*I "petscksp.h" I*/
#include "../src/ksp/ksp/impls/qcg/qcgimpl.h"

static PetscErrorCode QuadraticRoots_Private(Vec,Vec,PetscReal*,PetscReal*,PetscReal*);

#undef __FUNCT__  
#define __FUNCT__ "KSPQCGSetTrustRegionRadius" 
/*@
    KSPQCGSetTrustRegionRadius - Sets the radius of the trust region.

    Collective on KSP

    Input Parameters:
+   ksp   - the iterative context
-   delta - the trust region radius (Infinity is the default)

    Options Database Key:
.   -ksp_qcg_trustregionradius <delta>

    Level: advanced

.keywords: KSP, QCG, set, trust region radius
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGSetTrustRegionRadius(KSP ksp,PetscReal delta)
{
  PetscErrorCode ierr,(*f)(KSP,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (delta < 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPQCGSetTrustRegionRadius_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,delta);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPQCGGetTrialStepNorm"
/*@
    KSPQCGGetTrialStepNorm - Gets the norm of a trial step vector.  The WCG step may be
    constrained, so this is not necessarily the length of the ultimate step taken in QCG.

    Collective on KSP

    Input Parameter:
.   ksp - the iterative context

    Output Parameter:
.   tsnorm - the norm

    Level: advanced
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetTrialStepNorm(KSP ksp,PetscReal *tsnorm)
{
  PetscErrorCode ierr,(*f)(KSP,PetscReal*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPQCGGetTrialStepNorm_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,tsnorm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPQCGGetQuadratic"
/*@
    KSPQCGGetQuadratic - Gets the value of the quadratic function, evaluated at the new iterate:

       q(s) = g^T * s + 0.5 * s^T * H * s

    which satisfies the Euclidian Norm trust region constraint

       || D * s || <= delta,

    where

     delta is the trust region radius, 
     g is the gradient vector, and
     H is Hessian matrix,
     D is a scaling matrix.

    Collective on KSP

    Input Parameter:
.   ksp - the iterative context

    Output Parameter:
.   quadratic - the quadratic function evaluated at the new iterate

    Level: advanced
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetQuadratic(KSP ksp,PetscReal *quadratic)
{
  PetscErrorCode ierr,(*f)(KSP,PetscReal*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPQCGGetQuadratic_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp,quadratic);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_QCG"

PetscErrorCode KSPSolve_QCG(KSP ksp)
{
/* 
   Correpondence with documentation above:  
      B = g = gradient,
      X = s = step
   Note:  This is not coded correctly for complex arithmetic!
 */

  KSP_QCG        *pcgP = (KSP_QCG*)ksp->data;
  MatStructure   pflag;
  Mat            Amat,Pmat;
  Vec            W,WA,WA2,R,P,ASP,BS,X,B;
  PetscScalar    scal,btx,xtax,beta,rntrn,step;
  PetscReal      ptasp,q1,q2,wtasp,bstp,rtr,xnorm,step1,step2,rnrm,p5 = 0.5;
  PetscReal      dzero = 0.0,bsnrm;
  PetscErrorCode ierr;
  PetscInt       i,maxit;
  PC             pc = ksp->pc;
  PCSide         side;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cstep1,cstep2,cbstp,crtr,cwtasp,cptasp;
#endif
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  if (ksp->transpose_solve) {
    SETERRQ(PETSC_ERR_SUP,"Currently does not support transpose solve");
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

  if (pcgP->delta <= dzero) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Input error: delta <= 0");
  ierr = KSPGetPreconditionerSide(ksp,&side);CHKERRQ(ierr);
  if (side != PC_SYMMETRIC) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Requires symmetric preconditioner!");

  /* Initialize variables */
  ierr = VecSet(W,0.0);CHKERRQ(ierr);	/* W = 0 */
  ierr = VecSet(X,0.0);CHKERRQ(ierr);	/* X = 0 */
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
  ierr = (*ksp->converged)(ksp,0,bsnrm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Compute the initial scaled direction and scaled residual */
  ierr = VecCopy(BS,R);CHKERRQ(ierr);
  ierr = VecScale(R,-1.0);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(R,R,&crtr);CHKERRQ(ierr); rtr = PetscRealPart(crtr);
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
    ierr  = VecDot(P,ASP,&cptasp);CHKERRQ(ierr);
    ptasp = PetscRealPart(cptasp);
#else
    ierr = VecDot(P,ASP,&ptasp);CHKERRQ(ierr);	/* ptasp = p^T asp */
#endif
    if (ptasp <= dzero) {

      /* Scaled negative curvature direction:  Compute a step so that
         ||w + step*p|| = delta and QS(w + step*p) is least */

       if (!i) {
         ierr = VecCopy(P,X);CHKERRQ(ierr);
         ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);
         scal = pcgP->delta / xnorm;
         ierr = VecScale(X,scal);CHKERRQ(ierr);
       } else {
         /* Compute roots of quadratic */
         ierr = QuadraticRoots_Private(W,P,&pcgP->delta,&step1,&step2);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
         ierr = VecDot(W,ASP,&cwtasp);CHKERRQ(ierr); wtasp = PetscRealPart(cwtasp);
         ierr = VecDot(BS,P,&cbstp);CHKERRQ(ierr);   bstp  = PetscRealPart(cbstp);
#else
         ierr = VecDot(W,ASP,&wtasp);CHKERRQ(ierr);
         ierr = VecDot(BS,P,&bstp);CHKERRQ(ierr);
#endif
         ierr = VecCopy(W,X);CHKERRQ(ierr);
         q1 = step1*(bstp + wtasp + p5*step1*ptasp);
         q2 = step2*(bstp + wtasp + p5*step2*ptasp);
#if defined(PETSC_USE_COMPLEX)
         if (q1 <= q2) {
           cstep1 = step1; ierr = VecAXPY(X,cstep1,P);CHKERRQ(ierr);
         } else {
           cstep2 = step2; ierr = VecAXPY(X,cstep2,P);CHKERRQ(ierr);
         }
#else
         if (q1 <= q2) {ierr = VecAXPY(X,step1,P);CHKERRQ(ierr);}
         else          {ierr = VecAXPY(X,step2,P);CHKERRQ(ierr);}
#endif
       }
       pcgP->ltsnrm = pcgP->delta;                       /* convergence in direction of */
       ksp->reason  = KSP_CONVERGED_CG_NEG_CURVE;  /* negative curvature */
       if (!i) {
         ierr = PetscInfo1(ksp,"negative curvature: delta=%G\n",pcgP->delta);CHKERRQ(ierr);
       } else {
         ierr = PetscInfo3(ksp,"negative curvature: step1=%G, step2=%G, delta=%G\n",step1,step2,pcgP->delta);CHKERRQ(ierr);
       }
         
    } else {
 
       /* Compute step along p */

       step = rtr/ptasp;
       ierr = VecCopy(W,X);CHKERRQ(ierr);	   /*  x = w  */
       ierr = VecAXPY(X,step,P);CHKERRQ(ierr);   /*  x <- step*p + x  */
       ierr = VecNorm(X,NORM_2,&pcgP->ltsnrm);CHKERRQ(ierr);

       if (pcgP->ltsnrm > pcgP->delta) {

         /* Since the trial iterate is outside the trust region, 
             evaluate a constrained step along p so that 
                      ||w + step*p|| = delta 
            The positive step is always better in this case. */

         if (!i) {
           scal = pcgP->delta / pcgP->ltsnrm;
           ierr = VecScale(X,scal);CHKERRQ(ierr);
         } else {
           /* Compute roots of quadratic */
           ierr = QuadraticRoots_Private(W,P,&pcgP->delta,&step1,&step2);CHKERRQ(ierr);
           ierr = VecCopy(W,X);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
           cstep1 = step1; ierr = VecAXPY(X,cstep1,P);CHKERRQ(ierr);
#else
           ierr = VecAXPY(X,step1,P);CHKERRQ(ierr);  /*  x <- step1*p + x  */
#endif
         }
         pcgP->ltsnrm = pcgP->delta;
         ksp->reason  = KSP_CONVERGED_CG_CONSTRAINED;	/* convergence along constrained step */
         if (!i) {
           ierr = PetscInfo1(ksp,"constrained step: delta=%G\n",pcgP->delta);CHKERRQ(ierr);
         } else {
           ierr = PetscInfo3(ksp,"constrained step: step1=%G, step2=%G, delta=%G\n",step1,step2,pcgP->delta);CHKERRQ(ierr);
         }

       } else {

         /* Evaluate the current step */

         ierr = VecCopy(X,W);CHKERRQ(ierr);	/* update interior iterate */
         ierr = VecAXPY(R,-step,ASP);CHKERRQ(ierr); /* r <- -step*asp + r */
         ierr = VecNorm(R,NORM_2,&rnrm);CHKERRQ(ierr);

         ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
         ksp->rnorm                                    = rnrm;
         ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
         KSPLogResidualHistory(ksp,rnrm);
         KSPMonitor(ksp,i+1,rnrm);
         ierr = (*ksp->converged)(ksp,i+1,rnrm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
         if (ksp->reason) {                 /* convergence for */
#if defined(PETSC_USE_COMPLEX)               
           ierr = PetscInfo3(ksp,"truncated step: step=%G, rnrm=%G, delta=%G\n",PetscRealPart(step),rnrm,pcgP->delta);CHKERRQ(ierr);
#else
           ierr = PetscInfo3(ksp,"truncated step: step=%G, rnrm=%G, delta=%G\n",step,rnrm,pcgP->delta);CHKERRQ(ierr);
#endif
         }
      }
    }
    if (ksp->reason) break;	/* Convergence has been attained */
    else {		/* Compute a new AS-orthogonal direction */
      ierr = VecDot(R,R,&rntrn);CHKERRQ(ierr);
      beta = rntrn/rtr;
      ierr = VecAYPX(P,beta,R);CHKERRQ(ierr);	/*  p <- r + beta*p  */
#if defined(PETSC_USE_COMPLEX)
      rtr = PetscRealPart(rntrn);
#else
      rtr = rntrn;
#endif
    }
  }
  if (!ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  /* Unscale x */
  ierr = VecCopy(X,WA2);CHKERRQ(ierr);
  ierr = PCApplySymmetricRight(pc,WA2,X);CHKERRQ(ierr);

  ierr = MatMult(Amat,X,WA);CHKERRQ(ierr);
  ierr = VecDot(B,X,&btx);CHKERRQ(ierr);
  ierr = VecDot(X,WA,&xtax);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  pcgP->quadratic = PetscRealPart(btx) + p5* PetscRealPart(xtax);
#else
  pcgP->quadratic = btx + p5*xtax;              /* Compute q(x) */
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_QCG"
PetscErrorCode KSPSetUp_QCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP,"no right preconditioning for QCG");
  } else if (ksp->pc_side == PC_LEFT) {
    SETERRQ(PETSC_ERR_SUP,"no left preconditioning for QCG");
  }

  /* Get work vectors from user code */
  ierr = KSPDefaultGetWork(ksp,7);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_QCG" 
PetscErrorCode KSPDestroy_QCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPQCGGetQuadratic_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPQCGGetTrialStepNorm_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPQCGSetTrustRegionRadius_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPQCGSetTrustRegionRadius_QCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGSetTrustRegionRadius_QCG(KSP ksp,PetscReal delta)
{
  KSP_QCG *cgP = (KSP_QCG*)ksp->data;

  PetscFunctionBegin;
  cgP->delta = delta;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPQCGGetTrialStepNorm_QCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetTrialStepNorm_QCG(KSP ksp,PetscReal *ltsnrm)
{
  KSP_QCG *cgP = (KSP_QCG*)ksp->data;

  PetscFunctionBegin;
  *ltsnrm = cgP->ltsnrm;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPQCGGetQuadratic_QCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetQuadratic_QCG(KSP ksp,PetscReal *quadratic)
{
  KSP_QCG *cgP = (KSP_QCG*)ksp->data;

  PetscFunctionBegin;
  *quadratic = cgP->quadratic;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_QCG"
PetscErrorCode KSPSetFromOptions_QCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscReal      delta;
  KSP_QCG        *cgP = (KSP_QCG*)ksp->data;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP QCG Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_qcg_trustregionradius","Trust Region Radius","KSPQCGSetTrustRegionRadius",cgP->delta,&delta,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPQCGSetTrustRegionRadius(ksp,delta);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPQCG -   Code to run conjugate gradient method subject to a constraint
         on the solution norm. This is used in Trust Region methods for nonlinear equations, SNESTR

   Options Database Keys:
.      -ksp_qcg_trustregionradius <r> - Trust Region Radius

   Notes: This is rarely used directly

   Level: developer

  Notes:  Use preconditioned conjugate gradient to compute 
      an approximate minimizer of the quadratic function 

            q(s) = g^T * s + .5 * s^T * H * s

   subject to the Euclidean norm trust region constraint

            || D * s || <= delta,

   where 

     delta is the trust region radius, 
     g is the gradient vector, and
     H is Hessian matrix,
     D is a scaling matrix.

   KSPConvergedReason may be 
$  KSP_CONVERGED_CG_NEG_CURVE if convergence is reached along a negative curvature direction,
$  KSP_CONVERGED_CG_CONSTRAINED if convergence is reached along a constrained step,
$  other KSP converged/diverged reasons


  Notes:
  Currently we allow symmetric preconditioning with the following scaling matrices:
      PCNONE:   D = Identity matrix
      PCJACOBI: D = diag [d_1, d_2, ...., d_n], where d_i = sqrt(H[i,i])
      PCICC:    D = L^T, implemented with forward and backward solves.
                Here L is an incomplete Cholesky factor of H.

  References:
   The Conjugate Gradient Method and Trust Regions in Large Scale Optimization, Trond Steihaug
   SIAM Journal on Numerical Analysis, Vol. 20, No. 3 (Jun., 1983), pp. 626-637

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPQCGSetTrustRegionRadius()
           KSPQCGGetTrialStepNorm(), KSPQCGGetQuadratic()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_QCG"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_QCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_QCG        *cgP;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_QCG,&cgP);CHKERRQ(ierr);
  ksp->data                      = (void*)cgP;
  ksp->pc_side                   = PC_SYMMETRIC;
  ksp->ops->setup                = KSPSetUp_QCG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_QCG;
  ksp->ops->solve                = KSPSolve_QCG;
  ksp->ops->destroy              = KSPDestroy_QCG;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->view                 = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPQCGGetQuadratic_C",
                                    "KSPQCGGetQuadratic_QCG",
                                     KSPQCGGetQuadratic_QCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPQCGGetTrialStepNorm_C",
                                    "KSPQCGGetTrialStepNorm_QCG",
                                     KSPQCGGetTrialStepNorm_QCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPQCGSetTrustRegionRadius_C",
                                    "KSPQCGSetTrustRegionRadius_QCG",
                                     KSPQCGSetTrustRegionRadius_QCG);CHKERRQ(ierr);
  cgP->delta = PETSC_MAX; /* default trust region radius is infinite */ 
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "QuadraticRoots_Private"
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
static PetscErrorCode QuadraticRoots_Private(Vec s,Vec p,PetscReal *delta,PetscReal *step1,PetscReal *step2)
{ 
  PetscReal      dsq,ptp,pts,rad,sts;
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cptp,cpts,csts;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(p,s,&cpts);CHKERRQ(ierr); pts = PetscRealPart(cpts);
  ierr = VecDot(p,p,&cptp);CHKERRQ(ierr); ptp = PetscRealPart(cptp);
  ierr = VecDot(s,s,&csts);CHKERRQ(ierr); sts = PetscRealPart(csts);
#else
  ierr = VecDot(p,s,&pts);CHKERRQ(ierr);
  ierr = VecDot(p,p,&ptp);CHKERRQ(ierr);
  ierr = VecDot(s,s,&sts);CHKERRQ(ierr);
#endif
  dsq  = (*delta)*(*delta);
  rad  = sqrt((pts*pts) - ptp*(sts - dsq));
  if (pts > 0.0) {
    *step2 = -(pts + rad)/ptp;
    *step1 = (sts - dsq)/(ptp * *step2);
  } else {
    *step1 = -(pts - rad)/ptp;
    *step2 = (sts - dsq)/(ptp * *step1);
  }
  PetscFunctionReturn(0);
}
