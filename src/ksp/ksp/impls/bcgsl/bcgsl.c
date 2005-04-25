#define PETSCKSP_DLL
/*
 * Implementation of BiCGstab(L) the paper by D.R. Fokkema,
 * "Enhanced implementation of BiCGStab(L) for solving linear systems
 * of equations". This uses tricky delayed updating ideas to prevent
 * round-off buildup.
 */
#include "petscblaslapack.h"
#include "src/ksp/ksp/kspimpl.h"
#include "bcgsl.h"


#undef __FUNCT__
#define __FUNCT__ "KSPSolve_BCGSL"
static PetscErrorCode  KSPSolve_BCGSL(KSP ksp)
{
  KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *) ksp->data;
  PetscScalar    alpha, beta, nu, omega, sigma;
  PetscScalar    zero = 0;
  PetscScalar    rho0, rho1;
  PetscReal      kappa0, kappaA, kappa1;
  PetscReal      ghat, epsilon, abstol;
  PetscReal      zeta, zeta0, rnmax_computed, rnmax_true, nrm0;
  PetscTruth     bUpdateX;
  PetscTruth     bBombed = PETSC_FALSE;

  PetscInt       maxit;
  PetscInt       h, i, j, k, vi, ell;
  PetscMPIInt    rank;
  PetscBLASInt   ldMZ,bierr;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* set up temporary vectors */
  vi = 0;
  ell = bcgsl->ell;
  bcgsl->vB    = ksp->work[vi]; vi++;
  bcgsl->vRt   = ksp->work[vi]; vi++;
  bcgsl->vTm   = ksp->work[vi]; vi++;
  bcgsl->vvR   = ksp->work+vi; vi += ell+1;
  bcgsl->vvU   = ksp->work+vi; vi += ell+1;
  bcgsl->vXr   = ksp->work[vi]; vi++;
  ldMZ = ell+1;
  {
    ierr = PetscMalloc(ldMZ*sizeof(PetscScalar), &AY0c);CHKERRQ(ierr);
    ierr = PetscMalloc(ldMZ*sizeof(PetscScalar), &AYlc);CHKERRQ(ierr);
    ierr = PetscMalloc(ldMZ*sizeof(PetscScalar), &AYtc);CHKERRQ(ierr);
    ierr = PetscMalloc(ldMZ*ldMZ*sizeof(PetscScalar), &MZa);CHKERRQ(ierr);
    ierr = PetscMalloc(ldMZ*ldMZ*sizeof(PetscScalar), &MZb);CHKERRQ(ierr);
  }

  /* Prime the iterative solver */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  ierr = KSPInitialResidual(ksp, VX, VTM, VB, VVR[0], ksp->vec_rhs);CHKERRQ(ierr);
  ierr = VecNorm(VVR[0], NORM_2, &zeta0);CHKERRQ(ierr);
  rnmax_computed = zeta0;
  rnmax_true = zeta0;

  ierr = (*ksp->converged)(ksp, 0, zeta0, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) {
    ierr = PetscFree(AY0c);CHKERRQ(ierr);
    ierr = PetscFree(AYlc);CHKERRQ(ierr);
    ierr = PetscFree(AYtc);CHKERRQ(ierr);
    ierr = PetscFree(MZa);CHKERRQ(ierr);
    ierr = PetscFree(MZb);CHKERRQ(ierr);

    PetscFunctionReturn(0);
  }

  ierr = VecSet(VVU[0],zero);CHKERRQ(ierr);
  alpha = 0;
  rho0 = omega = 1;

  if (bcgsl->delta>0.0) {
    ierr = VecCopy(VX, VXR);CHKERRQ(ierr);
    ierr = VecSet(VX,zero);CHKERRQ(ierr);
    ierr = VecCopy(VVR[0], VB);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(ksp->vec_rhs, VB);CHKERRQ(ierr);
  }

  /* Life goes on */
  ierr = VecCopy(VVR[0], VRT);CHKERRQ(ierr);
  zeta = zeta0;

  ierr = KSPGetTolerances(ksp, &epsilon, &abstol, PETSC_NULL, &maxit);CHKERRQ(ierr);

  for (k=0; k<maxit; k += bcgsl->ell) {
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its   = k;
    ksp->rnorm = zeta;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

    KSPLogResidualHistory(ksp, zeta);
    KSPMonitor(ksp, ksp->its, zeta);

    ierr = (*ksp->converged)(ksp, k, zeta, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    /* BiCG part */
    rho0 = -omega*rho0;
    nrm0 = zeta;
    for (j=0; j<bcgsl->ell; j++) {
      /* rho1 <- r_j' * r_tilde */
      ierr = VecDot(VVR[j], VRT, &rho1);CHKERRQ(ierr);
      if (rho1 == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        bBombed = PETSC_TRUE;
        break;
      }
      beta = alpha*(rho1/rho0);
      rho0 = rho1;
      nu = -beta;
      for (i=0; i<=j; i++) {
        /* u_i <- r_i - beta*u_i */
        ierr = VecAYPX(VVU[i], nu, VVR[i]);CHKERRQ(ierr);
      }
      /* u_{j+1} <- inv(K)*A*u_j */
      ierr = KSP_PCApplyBAorAB(ksp, VVU[j], VVU[j+1], VTM);CHKERRQ(ierr);

      ierr = VecDot(VVU[j+1], VRT, &sigma);CHKERRQ(ierr);
      if (sigma == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        bBombed = PETSC_TRUE;
        break;
      }
      alpha = rho1/sigma;

      /* x <- x + alpha*u_0 */
      ierr = VecAXPY(VX, alpha, VVU[0]);CHKERRQ(ierr);

      nu = -alpha;
      for (i=0; i<=j; i++) {
        /* r_i <- r_i - alpha*u_{i+1} */
        ierr = VecAXPY(VVR[i], nu, VVU[i+1]);CHKERRQ(ierr);
      }

      /* r_{j+1} <- inv(K)*A*r_j */
      ierr = KSP_PCApplyBAorAB(ksp, VVR[j], VVR[j+1], VTM);CHKERRQ(ierr);

      ierr = VecNorm(VVR[0], NORM_2, &nrm0);CHKERRQ(ierr);
      if (bcgsl->delta>0.0) {
        if (rnmax_computed<nrm0) rnmax_computed = nrm0;
        if (rnmax_true<nrm0) rnmax_true = nrm0;
      }

      /* NEW: check for early exit */
      ierr = (*ksp->converged)(ksp, k+j, nrm0, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) {
        ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
        ksp->its   = k+j;
        ksp->rnorm = nrm0;
        ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
        break;
      }
    }

    if (bBombed==PETSC_TRUE) break;

    /* Polynomial part */

    for (i=0; i<=bcgsl->ell; i++) {
      for (j=0; j<i; j++) {
        ierr = VecDot(VVR[j], VVR[i], &nu);CHKERRQ(ierr);
        MZa[i+ldMZ*j] = nu;
        MZa[j+ldMZ*i] = nu;
        MZb[i+ldMZ*j] = nu;
        MZb[j+ldMZ*i] = nu;
      }

      ierr = VecDot(VVR[i], VVR[i], &nu);CHKERRQ(ierr);
      MZa[i+ldMZ*i] = nu;
      MZb[i+ldMZ*i] = nu;
    }

    if (!bcgsl->bConvex || bcgsl->ell==1) {
      PetscBLASInt ione = 1,bell = bcgsl->ell;

      AY0c[0] = -1;
      LAPACKpotrf_("Lower", &bell, &MZa[1+ldMZ], &ldMZ, &bierr);
      if (ierr!=0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        bBombed = PETSC_TRUE;
        break;
      }
      BLAScopy_(&bell, &MZb[1], &ione, &AY0c[1], &ione);
      LAPACKpotrs_("Lower", &bell, &ione, &MZa[1+ldMZ], &ldMZ, &AY0c[1], &ldMZ, &bierr);
    } else {
      PetscBLASInt neqs = bcgsl->ell-1;
      PetscBLASInt ione = 1;
      PetscScalar aone = 1.0, azero = 0.0;

      LAPACKpotrf_("Lower", &neqs, &MZa[1+ldMZ], &ldMZ, &bierr);
      if (ierr!=0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        bBombed = PETSC_TRUE;
        break;
      }
      BLAScopy_(&neqs, &MZb[1], &ione, &AY0c[1], &ione);
      LAPACKpotrs_("Lower", &neqs, &ione, &MZa[1+ldMZ], &ldMZ, &AY0c[1], &ldMZ, &bierr);
      AY0c[0] = -1;
      AY0c[bcgsl->ell] = 0;

      BLAScopy_(&neqs, &MZb[1+ldMZ*(bcgsl->ell)], &ione, &AYlc[1], &ione);
      LAPACKpotrs_("Lower", &neqs, &ione, &MZa[1+ldMZ], &ldMZ, &AYlc[1], &ldMZ, &bierr);

      AYlc[0] = 0;
      AYlc[bcgsl->ell] = -1;

      BLASgemv_("NoTr", &ldMZ, &ldMZ, &aone, MZb, &ldMZ, AY0c, &ione, &azero, AYtc, &ione);

      kappa0 = BLASdot_(&ldMZ, AY0c, &ione, AYtc, &ione);

      /* round-off can cause negative kappa's */
      if (kappa0<0) kappa0 = -kappa0;
      kappa0 = sqrt(kappa0);

      kappaA = BLASdot_(&ldMZ, AYlc, &ione, AYtc, &ione);

      BLASgemv_("noTr", &ldMZ, &ldMZ, &aone, MZb, &ldMZ, AYlc, &ione, &azero, AYtc, &ione);

      kappa1 = BLASdot_(&ldMZ, AYlc, &ione, AYtc, &ione);

      if (kappa1<0) kappa1 = -kappa1;
      kappa1 = sqrt(kappa1);

      if (kappa0!=0.0 && kappa1!=0.0) {
        if (kappaA<0.7*kappa0*kappa1) {
          ghat = (kappaA<0.0) ?  -0.7*kappa0/kappa1 : 0.7*kappa0/kappa1;
        } else {
          ghat = kappaA/(kappa1*kappa1);
        }
        for (i=0; i<=bcgsl->ell; i++) {
          AY0c[i] = AY0c[i] - ghat* AYlc[i];
        }
      }
    }

    omega = AY0c[bcgsl->ell];
    for (h=bcgsl->ell; h>0 && omega==0.0; h--) {
      omega = AY0c[h];
    }
    if (omega==0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }

    for (i=1; i<=bcgsl->ell; i++) {
      nu = -AY0c[i];
      ierr = VecAXPY(VVU[0], nu, VVU[i]);CHKERRQ(ierr);
      nu = AY0c[i];
      ierr = VecAXPY(VX, nu, VVR[i-1]);CHKERRQ(ierr);
      nu = -AY0c[i];
      ierr = VecAXPY(VVR[0], nu, VVR[i]);CHKERRQ(ierr);
    }

    ierr = VecNorm(VVR[0], NORM_2, &zeta);CHKERRQ(ierr);

    /* Accurate Update */
    if (bcgsl->delta>0.0) {
      if (rnmax_computed<zeta) rnmax_computed = zeta;
      if (rnmax_true<zeta) rnmax_true = zeta;

      bUpdateX = (PetscTruth) (zeta<bcgsl->delta*zeta0 && zeta0<=rnmax_computed);
      if ((zeta<bcgsl->delta*rnmax_true && zeta0<=rnmax_true) || bUpdateX) {
        /* r0 <- b-inv(K)*A*X */
        ierr = KSP_PCApplyBAorAB(ksp, VX, VVR[0], VTM);CHKERRQ(ierr);
        nu = -1;
        ierr = VecAYPX(VVR[0], nu, VB);CHKERRQ(ierr);
        rnmax_true = zeta;

        if (bUpdateX) {
          nu = 1;
          ierr = VecAXPY(VXR,nu,VX);CHKERRQ(ierr);
          ierr = VecSet(VX,zero);CHKERRQ(ierr);
          ierr = VecCopy(VVR[0], VB);CHKERRQ(ierr);
          rnmax_computed = zeta;
        }
      }
    }
  }

  KSPMonitor(ksp, ksp->its, zeta);

  if (bcgsl->delta>0.0) {
    nu   = 1;
    ierr = VecAXPY(VX,nu,VXR);CHKERRQ(ierr);
  }

  ierr = (*ksp->converged)(ksp, k, zeta, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  ierr = PetscFree(AY0c);CHKERRQ(ierr);
  ierr = PetscFree(AYlc);CHKERRQ(ierr);
  ierr = PetscFree(AYtc);CHKERRQ(ierr);
  ierr = PetscFree(MZa);CHKERRQ(ierr);
  ierr = PetscFree(MZb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPBCGSLSetXRes"
/*@C
   KSPBCGSLSetXRes - Sets the parameter governing when
   exact residuals will be used instead of computed residuals.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  delta - computed residuals are used alone when delta is not positive

   Options Database Keys:

.  -ksp_bcgsl_xres delta

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals

.seealso: KSPBCGSLSetEll(), KSPBCGSLSetPol()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetXRes(KSP ksp, PetscReal delta)
{
  KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->setupcalled) {
    if ((delta<=0 && bcgsl->delta>0) || (delta>0 && bcgsl->delta<=0)) {
      ierr = KSPDefaultFreeWork(ksp);CHKERRQ(ierr);
      ksp->setupcalled = 0;
    }
  }
  bcgsl->delta = delta;
  PetscFunctionReturn(0);
}
EXTERN_C_END
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPBCGSLSetPol"
/*@C
   KSPBCGSLSetPol - Sets the type of polynomial part will
   be used in the BiCGSTab(L) solver.

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  uMROR - set to PETSC_TRUE when the polynomial is a convex combination of an MR and an OR step.

   Options Database Keys:

+  -ksp_bcgsl_cxpoly - use enhanced polynomial
.  -ksp_bcgsl_mrpoly - use standard polynomial

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, polynomial

.seealso: @()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetPol(KSP ksp, PetscTruth uMROR)
{
  KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ksp->setupcalled) {
    bcgsl->bConvex = uMROR;
  } else if (bcgsl->bConvex != uMROR) {
    /* free the data structures,
       then create them again
     */
   ierr = KSPDefaultFreeWork(ksp);CHKERRQ(ierr);
    bcgsl->bConvex = uMROR;
    ksp->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPBCGSLSetEll"
/*@C
   KSPBCGSLSetEll - Sets the number of search directions in BiCGStab(L).

   Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  ell - number of search directions

   Options Database Keys:

.  -ksp_bcgsl_ell ell

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals,

.seealso: @()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetEll(KSP ksp, int ell)
{
  KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ell < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "KSPBCGSLSetEll: second argument must be positive");

  if (!ksp->setupcalled) {
    bcgsl->ell = ell;
  } else if (bcgsl->ell != ell) {
    /* free the data structures, then create them again */
   ierr = KSPDefaultFreeWork(ksp);CHKERRQ(ierr);

    bcgsl->ell = ell;
    ksp->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "KSPView_BCGSL"
PetscErrorCode KSPView_BCGSL(KSP ksp, PetscViewer viewer)
{
  KSP_BiCGStabL       *bcgsl = (KSP_BiCGStabL *)ksp->data;
  PetscErrorCode      ierr;
  PetscTruth          isascii, isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer, PETSC_VIEWER_ASCII, &isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer, PETSC_VIEWER_STRING, &isstring);CHKERRQ(ierr);

  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  BCGSL: Ell = %D\n", bcgsl->ell);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  BCGSL: Delta = %lg\n", bcgsl->delta);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Viewer type %s not supported for KSP BCGSL", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_BCGSL"
PetscErrorCode KSPSetFromOptions_BCGSL(KSP ksp)
{
  KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *)ksp->data;
  PetscErrorCode ierr;
  PetscInt       this_ell;
  PetscReal      delta;
  PetscTruth     flga, flg;

  PetscFunctionBegin;
  /* PetscOptionsBegin/End are called in KSPSetFromOptions. They
     don't need to be called here.
  */
  ierr = PetscOptionsHead("KSP BiCGStab(L) Options");CHKERRQ(ierr);

  /* Set number of search directions */
  ierr = PetscOptionsInt("-ksp_bcgsl_ell","Number of Krylov search directions","KSPBCGSLSetEll",bcgsl->ell,&this_ell,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPBCGSLSetEll(ksp, this_ell);CHKERRQ(ierr);
  }

  /* Set polynomial type */
  ierr = PetscOptionsName("-ksp_bcgsl_cxpoly", "Polynomial part of BiCGStabL is MinRes + OR", "KSPBCGSLSetPol", &flga);CHKERRQ(ierr);
  if (flga) {
    ierr = KSPBCGSLSetPol(ksp, PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsName("-ksp_bcgsl_mrpoly", "Polynomial part of BiCGStabL is MinRes", "KSPBCGSLSetPol", &flg);CHKERRQ(ierr);
    ierr = KSPBCGSLSetPol(ksp, PETSC_FALSE);CHKERRQ(ierr);
  }

  /* Will computed residual be refreshed? */
  ierr = PetscOptionsReal("-ksp_bcgsl_xres", "Threshold used to decide when to refresh computed residuals", "KSPBCGSLSetXRes", bcgsl->delta, &delta, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPBCGSLSetXRes(ksp, delta);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_BCGSL"
PetscErrorCode KSPSetUp_BCGSL(KSP ksp)
{
  KSP_BiCGStabL  *bcgsl = (KSP_BiCGStabL *)ksp->data;
  PetscInt        ell = bcgsl->ell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Support left preconditioners only */
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP, "no symmetric preconditioning for KSPBCGSL");
  } else if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP, "no right preconditioning for KSPBCGSL");
  }
  ierr = KSPDefaultGetWork(ksp, 6+2*ell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPBCGSL - Implements a slight variant of the Enhanced
                BiCGStab(L) algorithm in (3) and (2).  The variation
                concerns cases when either kappa0**2 or kappa1**2 is
                negative due to round-off. Kappa0 has also been pulled
                out of the denominator in the formula for ghat.

    References:
      1. G.L.G. Sleijpen, H.A. van der Vorst, "An overview of
         approaches for the stable computation of hybrid BiCG
         methods", Applied Numerical Mathematics: Transactions
         f IMACS, 19(3), pp 235-54, 1996.
      2. G.L.G. Sleijpen, H.A. van der Vorst, D.R. Fokkema,
         "BiCGStab(L) and other hybrid Bi-CG methods",
          Numerical Algorithms, 7, pp 75-109, 1994.
      3. D.R. Fokkema, "Enhanced implementation of BiCGStab(L)
         for solving linear systems of equations", preprint
         from www.citeseer.com.

   Contributed by: Joel M. Malard, email jm.malard@pnl.gov

   Options Database Keys:
+  -ksp_bcgsl_ell <ell> Number of Krylov search directions
-  -ksp_bcgsl_cxpol Use a convex function of the MR and OR polynomials after the BiCG step
-  -ksp_bcgsl_xres <res> Threshold used to decide when to refresh computed residuals

   Level: beginner

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPBCGS

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_BCGSL"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_BCGSL(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BiCGStabL  *bcgsl;

  PetscFunctionBegin;
  /* allocate BiCGStab(L) context */
  ierr = PetscNew(KSP_BiCGStabL, &bcgsl);CHKERRQ(ierr);
  ksp->data = (void*)bcgsl;

  ksp->pc_side              = PC_LEFT;
  ksp->ops->setup           = KSPSetUp_BCGSL;
  ksp->ops->solve           = KSPSolve_BCGSL;
  ksp->ops->destroy         = KSPDefaultDestroy;
  ksp->ops->buildsolution   = KSPDefaultBuildSolution;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = KSPSetFromOptions_BCGSL;
  ksp->ops->view            = KSPView_BCGSL;

  /* Let the user redefine the number of directions vectors */
  bcgsl->ell = 2;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp, "KSPBCGSLSetEll_C", "KSP_BCGS_SetEll", KSPBCGSLSetEll);CHKERRQ(ierr);

  /*Choose between a single MR step or an averaged MR/OR */
  bcgsl->bConvex = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp, "KSPBCGSLSetPol_C", "KSP_BCGS_SetPol", KSPBCGSLSetPol);CHKERRQ(ierr);

  /* Set the threshold for when exact residuals will be used */
  bcgsl->delta = 0.0;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp, "KSPBCGSLSetXRes_C", "KSP_BCGS_SetXRes", KSPBCGSLSetXRes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__
