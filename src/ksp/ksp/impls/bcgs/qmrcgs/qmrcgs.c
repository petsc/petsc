
/*
    This file implements QMRCGS (QMRCGStab).
    Only right preconditioning is supported.

    Contributed by: Xiangmin Jiao (xiangmin.jiao@stonybrook.edu)

    References:
.   * - Chan, Gallopoulos, Simoncini, Szeto, and Tong (SISC 1994), Ghai, Lu, and Jiao (NLAA 2019)
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

static PetscErrorCode KSPSetUp_QMRCGS(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,14));
  PetscFunctionReturn(0);
}

/* Only need a few hacks from KSPSolve_BCGS */

static PetscErrorCode  KSPSolve_QMRCGS(KSP ksp)
{
  PetscInt       i;
  PetscScalar    eta,rho1,rho2,alpha,eta2,omega,beta,cf,cf1,uu;
  Vec            X,B,R,P,PH,V,D2,X2,S,SH,T,D,S2,RP,AX,Z;
  PetscReal      dp = 0.0,final,tau,tau2,theta,theta2,c,F,NV,vv;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;
  PC             pc;
  Mat            mat;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  P  = ksp->work[1];
  PH = ksp->work[2];
  V  = ksp->work[3];
  D2 = ksp->work[4];
  X2 = ksp->work[5];
  S  = ksp->work[6];
  SH = ksp->work[7];
  T  = ksp->work[8];
  D  = ksp->work[9];
  S2 = ksp->work[10];
  RP = ksp->work[11];
  AX = ksp->work[12];
  Z  = ksp->work[13];

  /*  Only supports right preconditioning */
  PetscCheckFalse(ksp->pc_side != PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP qmrcgs does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      CHKERRQ(VecDuplicate(X,&bcgs->guess));
    }
    CHKERRQ(VecCopy(X,bcgs->guess));
  } else {
    CHKERRQ(VecSet(X,0.0));
  }

  /* Compute initial residual */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetUp(pc));
  CHKERRQ(PCGetOperators(pc,&mat,NULL));
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,mat,X,S2));
    CHKERRQ(VecCopy(B,R));
    CHKERRQ(VecAXPY(R,-1.0,S2));
  } else {
    CHKERRQ(VecCopy(B,R));
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    CHKERRQ(VecNorm(R,NORM_2,&dp));
  }
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  CHKERRQ(KSPLogResidualHistory(ksp,dp));
  CHKERRQ(KSPMonitor(ksp,0,dp));
  CHKERRQ((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  CHKERRQ(VecCopy(R,RP));

  eta   = 1.0;
  theta = 1.0;
  if (dp == 0.0) {
    CHKERRQ(VecNorm(R,NORM_2,&tau));
  } else {
    tau = dp;
  }

  CHKERRQ(VecDot(RP,RP,&rho1));
  CHKERRQ(VecCopy(R,P));

  i=0;
  do {
    CHKERRQ(KSP_PCApply(ksp,P,PH)); /*  ph <- K p */
    CHKERRQ(KSP_MatMult(ksp,mat,PH,V)); /* v <- A ph */

    CHKERRQ(VecDot(V,RP,&rho2)); /* rho2 <- (v,rp) */
    if (rho2 == 0.0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to division by zero");
      else {
        ksp->reason = KSP_DIVERGED_NANORINF;
        break;
      }
    }

    if (rho1 == 0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has stagnated");
      else {
        ksp->reason = KSP_DIVERGED_BREAKDOWN; /* Stagnation */
        break;
      }
    }

    alpha = rho1 / rho2;
    CHKERRQ(VecWAXPY(S,-alpha,V,R)); /* s <- r - alpha v */

    /* First quasi-minimization step */
    CHKERRQ(VecNorm(S,NORM_2,&F)); /* f <- norm(s) */
    theta2 =  F / tau;

    c = 1.0 / PetscSqrtReal(1.0 + theta2 * theta2);

    tau2 = tau * theta2 * c;
    eta2 = c * c * alpha;
    cf  = theta * theta * eta / alpha;
    CHKERRQ(VecWAXPY(D2,cf,D,PH));        /* d2 <- ph + cf d */
    CHKERRQ(VecWAXPY(X2,eta2,D2,X));      /* x2 <- x + eta2 d2 */

    /* Apply the right preconditioner again */
    CHKERRQ(KSP_PCApply(ksp,S,SH)); /*  sh <- K s */
    CHKERRQ(KSP_MatMult(ksp,mat,SH,T)); /* t <- A sh */

    CHKERRQ(VecDotNorm2(S,T,&uu,&vv));
    if (vv == 0.0) {
      CHKERRQ(VecDot(S,S,&uu));
      if (uu != 0.0) {
        PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to division by zero");
        else {
          ksp->reason = KSP_DIVERGED_NANORINF;
          break;
        }
      }
      CHKERRQ(VecAXPY(X,alpha,SH));   /* x <- x + alpha sh */
      CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      CHKERRQ(KSPLogResidualHistory(ksp,dp));
      CHKERRQ(KSPMonitor(ksp,i+1,0.0));
      break;
    }
    CHKERRQ(VecNorm(V,NORM_2,&NV)); /* nv <- norm(v) */

    if (NV == 0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to singular matrix");
      else {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
    }

    if (uu == 0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has stagnated");
      else {
        ksp->reason = KSP_DIVERGED_BREAKDOWN; /* Stagnation */
        break;
      }
    }
    omega = uu / vv; /* omega <- uu/vv; */

    /* Computing the residual */
    CHKERRQ(VecWAXPY(R,-omega,T,S));  /* r <- s - omega t */

    /* Second quasi-minimization step */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      CHKERRQ(VecNorm(R,NORM_2,&dp));
    }

    if (tau2 == 0) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to division by zero");
      else {
        ksp->reason = KSP_DIVERGED_NANORINF;
        break;
      }
    }
    theta = dp / tau2;
    c = 1.0 / PetscSqrtReal(1.0 + theta * theta);
    if (dp == 0.0) {
      CHKERRQ(VecNorm(R,NORM_2,&tau));
    } else {
      tau = dp;
    }
    tau = tau * c;
    eta = c * c * omega;

    cf1  = theta2 * theta2 * eta2 / omega;
    CHKERRQ(VecWAXPY(D,cf1,D2,SH));     /* d <- sh + cf1 d2 */
    CHKERRQ(VecWAXPY(X,eta,D,X2));      /* x <- x2 + eta d */

    CHKERRQ(VecDot(R,RP,&rho2));
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = dp;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    CHKERRQ(KSPLogResidualHistory(ksp,dp));
    CHKERRQ(KSPMonitor(ksp,i+1,dp));

    beta = (alpha*rho2)/ (omega*rho1);
    CHKERRQ(VecAXPBYPCZ(P,1.0,-omega*beta,beta,R,V)); /* p <- r - omega * beta* v + beta * p */
    rho1 = rho2;
    CHKERRQ(KSP_MatMult(ksp,mat,X,AX)); /* Ax <- A x */
    CHKERRQ(VecWAXPY(Z,-1.0,AX,B));  /* r <- b - Ax */
    CHKERRQ(VecNorm(Z,NORM_2,&final));
    CHKERRQ((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;
    i++;
  } while (i<ksp->max_it);

  /* mark lack of convergence */
  if (ksp->its >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPQMRCGS - Implements the QMRCGStab method.

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   Notes:
    Only right preconditioning is supported.

   References:
. * - Chan, Gallopoulos, Simoncini, Szeto, and Tong (SISC 1994), Ghai, Lu, and Jiao (NLAA 2019)

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBICGS, KSPFBCGSL, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_QMRCGS(KSP ksp)
{
  KSP_BCGS       *bcgs;
  static const char citations[] =
    "@article{chan1994qmrcgs,\n"
    "  title={A quasi-minimal residual variant of the {Bi-CGSTAB} algorithm for nonsymmetric systems},\n"
    "  author={Chan, Tony F and Gallopoulos, Efstratios and Simoncini, Valeria and Szeto, Tedd and Tong, Charles H},\n"
    "  journal={SIAM Journal on Scientific Computing},\n"
    "  volume={15},\n"
    "  number={2},\n"
    "  pages={338--347},\n"
    "  year={1994},\n"
    "  publisher={SIAM}\n"
    "}\n"
    "@article{ghai2019comparison,\n"
    "  title={A comparison of preconditioned {K}rylov subspace methods for large-scale nonsymmetric linear systems},\n"
    "  author={Ghai, Aditi and Lu, Cao and Jiao, Xiangmin},\n"
    "  journal={Numerical Linear Algebra with Applications},\n"
    "  volume={26},\n"
    "  number={1},\n"
    "  pages={e2215},\n"
    "  year={2019},\n"
    "  publisher={Wiley Online Library}\n"
    "}\n";
  PetscBool      cite=PETSC_FALSE;

  PetscFunctionBegin;

  CHKERRQ(PetscCitationsRegister(citations,&cite));
  CHKERRQ(PetscNewLog(ksp,&bcgs));

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_QMRCGS;
  ksp->ops->solve          = KSPSolve_QMRCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;
  ksp->pc_side             = PC_RIGHT;  /* set default PC side */

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));
  PetscFunctionReturn(0);
}
