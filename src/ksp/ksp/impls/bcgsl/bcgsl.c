
/*
 * Implementation of BiCGstab(L) the paper by D.R. Fokkema,
 * "Enhanced implementation of BiCGStab(L) for solving linear systems
 * of equations". This uses tricky delayed updating ideas to prevent
 * round-off buildup.
 *
 * This has not been completely cleaned up into PETSc style.
 *
 * All the BLAS and LAPACK calls below should be removed and replaced with 
 * loops and the macros for block solvers converted from LINPACK; there is no way
 * calls to BLAS/LAPACK make sense for size 2, 3, 4, etc.
 */
#include <petsc-private/kspimpl.h>              /*I   "petscksp.h" I*/
#include <../src/ksp/ksp/impls/bcgsl/bcgslimpl.h>
#include <petscblaslapack.h>


#undef __FUNCT__
#define __FUNCT__ "KSPSolve_BCGSL"
static PetscErrorCode  KSPSolve_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *) ksp->data;
  PetscScalar    alpha, beta, omega, sigma;
  PetscScalar    rho0, rho1;
  PetscReal      kappa0, kappaA, kappa1;
  PetscReal      ghat;
  PetscReal      zeta, zeta0, rnmax_computed, rnmax_true, nrm0;
  PetscBool      bUpdateX;
  PetscInt       maxit;
  PetscInt       h, i, j, k, vi, ell;
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
  ldMZ = PetscBLASIntCast(ell+1);

  /* Prime the iterative solver */
  ierr = KSPInitialResidual(ksp, VX, VTM, VB, VVR[0], ksp->vec_rhs);CHKERRQ(ierr);
  ierr = VecNorm(VVR[0], NORM_2, &zeta0);CHKERRQ(ierr);
  rnmax_computed = zeta0;
  rnmax_true = zeta0;

  ierr = (*ksp->converged)(ksp, 0, zeta0, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) {
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its   = 0;
    ksp->rnorm = zeta0;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecSet(VVU[0],0.0);CHKERRQ(ierr);
  alpha = 0.;
  rho0 = omega = 1;

  if (bcgsl->delta>0.0) {
    ierr = VecCopy(VX, VXR);CHKERRQ(ierr);
    ierr = VecSet(VX,0.0);CHKERRQ(ierr);
    ierr = VecCopy(VVR[0], VB);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(ksp->vec_rhs, VB);CHKERRQ(ierr);
  }

  /* Life goes on */
  ierr = VecCopy(VVR[0], VRT);CHKERRQ(ierr);
  zeta = zeta0;

  ierr = KSPGetTolerances(ksp, PETSC_NULL, PETSC_NULL, PETSC_NULL, &maxit);CHKERRQ(ierr);

  for (k=0; k<maxit; k += bcgsl->ell) {
    ksp->its   = k;
    ksp->rnorm = zeta;

    KSPLogResidualHistory(ksp, zeta);
    ierr = KSPMonitor(ksp, ksp->its, zeta);CHKERRQ(ierr);

    ierr = (*ksp->converged)(ksp, k, zeta, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason < 0) PetscFunctionReturn(0);
    else if (ksp->reason) break;

    /* BiCG part */
    rho0 = -omega*rho0;
    nrm0 = zeta;
    for (j=0; j<bcgsl->ell; j++) {
      /* rho1 <- r_j' * r_tilde */
      ierr = VecDot(VVR[j], VRT, &rho1);CHKERRQ(ierr);
      if (rho1 == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        PetscFunctionReturn(0);
      }
      beta = alpha*(rho1/rho0);
      rho0 = rho1;
      for (i=0; i<=j; i++) {
        /* u_i <- r_i - beta*u_i */
        ierr = VecAYPX(VVU[i], -beta, VVR[i]);CHKERRQ(ierr);
      }
      /* u_{j+1} <- inv(K)*A*u_j */
      ierr = KSP_PCApplyBAorAB(ksp, VVU[j], VVU[j+1], VTM);CHKERRQ(ierr);

      ierr = VecDot(VVU[j+1], VRT, &sigma);CHKERRQ(ierr);
      if (sigma == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        PetscFunctionReturn(0);
      }
      alpha = rho1/sigma;

      /* x <- x + alpha*u_0 */
      ierr = VecAXPY(VX, alpha, VVU[0]);CHKERRQ(ierr);

      for (i=0; i<=j; i++) {
        /* r_i <- r_i - alpha*u_{i+1} */
        ierr = VecAXPY(VVR[i], -alpha, VVU[i+1]);CHKERRQ(ierr);
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
        if (ksp->reason < 0) PetscFunctionReturn(0);
      }
    }

    /* Polynomial part */
    for(i = 0; i <= bcgsl->ell; ++i) {
      ierr = VecMDot(VVR[i], i+1, VVR, &MZa[i*ldMZ]);CHKERRQ(ierr);
    }
    /* Symmetrize MZa */
    for(i = 0; i <= bcgsl->ell; ++i) {
      for(j = i+1; j <= bcgsl->ell; ++j) {
        MZa[i*ldMZ+j] = MZa[j*ldMZ+i] = PetscConj(MZa[j*ldMZ+i]);
      }
    }
    /* Copy MZa to MZb */
    ierr = PetscMemcpy(MZb,MZa,ldMZ*ldMZ*sizeof(PetscScalar));CHKERRQ(ierr);

    if (!bcgsl->bConvex || bcgsl->ell==1) {
      PetscBLASInt ione = 1,bell = PetscBLASIntCast(bcgsl->ell);

      AY0c[0] = -1;
      LAPACKpotrf_("Lower", &bell, &MZa[1+ldMZ], &ldMZ, &bierr);
      if (ierr!=0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscFunctionReturn(0);
      }
      ierr = PetscMemcpy(&AY0c[1],&MZb[1],bcgsl->ell*sizeof(PetscScalar));CHKERRQ(ierr);
      LAPACKpotrs_("Lower", &bell, &ione, &MZa[1+ldMZ], &ldMZ, &AY0c[1], &ldMZ, &bierr);
    } else {
      PetscBLASInt ione = 1;
      PetscScalar aone = 1.0, azero = 0.0;
      PetscBLASInt neqs = PetscBLASIntCast(bcgsl->ell-1);

      LAPACKpotrf_("Lower", &neqs, &MZa[1+ldMZ], &ldMZ, &bierr);
      if (ierr!=0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscFunctionReturn(0);
      }
      ierr = PetscMemcpy(&AY0c[1],&MZb[1],(bcgsl->ell-1)*sizeof(PetscScalar));CHKERRQ(ierr);
      LAPACKpotrs_("Lower", &neqs, &ione, &MZa[1+ldMZ], &ldMZ, &AY0c[1], &ldMZ, &bierr);
      AY0c[0] = -1;
      AY0c[bcgsl->ell] = 0.;

      ierr = PetscMemcpy(&AYlc[1],&MZb[1+ldMZ*(bcgsl->ell)],(bcgsl->ell-1)*sizeof(PetscScalar));CHKERRQ(ierr);
      LAPACKpotrs_("Lower", &neqs, &ione, &MZa[1+ldMZ], &ldMZ, &AYlc[1], &ldMZ, &bierr);

      AYlc[0] = 0.;
      AYlc[bcgsl->ell] = -1;

      BLASgemv_("NoTr", &ldMZ, &ldMZ, &aone, MZb, &ldMZ, AY0c, &ione, &azero, AYtc, &ione);

      kappa0 = PetscRealPart(BLASdot_(&ldMZ, AY0c, &ione, AYtc, &ione));

      /* round-off can cause negative kappa's */
      if (kappa0<0) kappa0 = -kappa0;
      kappa0 = PetscSqrtReal(kappa0);

      kappaA = PetscRealPart(BLASdot_(&ldMZ, AYlc, &ione, AYtc, &ione));

      BLASgemv_("noTr", &ldMZ, &ldMZ, &aone, MZb, &ldMZ, AYlc, &ione, &azero, AYtc, &ione);

      kappa1 = PetscRealPart(BLASdot_(&ldMZ, AYlc, &ione, AYtc, &ione));

      if (kappa1<0) kappa1 = -kappa1;
      kappa1 = PetscSqrtReal(kappa1);

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
      PetscFunctionReturn(0);
    }


    ierr = VecMAXPY(VX, bcgsl->ell,AY0c+1, VVR);CHKERRQ(ierr);
    for (i=1; i<=bcgsl->ell; i++) {
      AY0c[i] *= -1.0;
    }
    ierr = VecMAXPY(VVU[0], bcgsl->ell,AY0c+1, VVU+1);CHKERRQ(ierr);
    ierr = VecMAXPY(VVR[0], bcgsl->ell,AY0c+1, VVR+1);CHKERRQ(ierr);
    for (i=1; i<=bcgsl->ell; i++) {
      AY0c[i] *= -1.0;
    }
    ierr = VecNorm(VVR[0], NORM_2, &zeta);CHKERRQ(ierr);

    /* Accurate Update */
    if (bcgsl->delta>0.0) {
      if (rnmax_computed<zeta) rnmax_computed = zeta;
      if (rnmax_true<zeta) rnmax_true = zeta;

      bUpdateX = (PetscBool) (zeta<bcgsl->delta*zeta0 && zeta0<=rnmax_computed);
      if ((zeta<bcgsl->delta*rnmax_true && zeta0<=rnmax_true) || bUpdateX) {
        /* r0 <- b-inv(K)*A*X */
        ierr = KSP_PCApplyBAorAB(ksp, VX, VVR[0], VTM);CHKERRQ(ierr);
        ierr = VecAYPX(VVR[0], -1.0, VB);CHKERRQ(ierr);
        rnmax_true = zeta;

        if (bUpdateX) {
          ierr = VecAXPY(VXR,1.0,VX);CHKERRQ(ierr);
          ierr = VecSet(VX,0.0);CHKERRQ(ierr);
          ierr = VecCopy(VVR[0], VB);CHKERRQ(ierr);
          rnmax_computed = zeta;
        }
      }
    }
  }
  if (bcgsl->delta>0.0) {
    ierr = VecAXPY(VX,1.0,VXR);CHKERRQ(ierr);
  }

  ierr = (*ksp->converged)(ksp, k, zeta, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPBCGSLSetXRes"
/*@
   KSPBCGSLSetXRes - Sets the parameter governing when
   exact residuals will be used instead of computed residuals.

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  delta - computed residuals are used alone when delta is not positive

   Options Database Keys:

.  -ksp_bcgsl_xres delta

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals

.seealso: KSPBCGSLSetEll(), KSPBCGSLSetPol()
@*/
PetscErrorCode  KSPBCGSLSetXRes(KSP ksp, PetscReal delta)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ksp,delta,2);
  if (ksp->setupstage) {
    if ((delta<=0 && bcgsl->delta>0) || (delta>0 && bcgsl->delta<=0)) {
      ierr = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
      ierr = PetscFree5(AY0c,AYlc,AYtc,MZa,MZb);CHKERRQ(ierr);
      ksp->setupstage = KSP_SETUP_NEW;
    }
  }
  bcgsl->delta = delta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPBCGSLSetPol"
/*@
   KSPBCGSLSetPol - Sets the type of polynomial part will
   be used in the BiCGSTab(L) solver.

   Logically Collective on KSP

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
PetscErrorCode  KSPBCGSLSetPol(KSP ksp, PetscBool  uMROR)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(ksp,uMROR,2);

  if (!ksp->setupstage) {
    bcgsl->bConvex = uMROR;
  } else if (bcgsl->bConvex != uMROR) {
    /* free the data structures,
       then create them again
     */
    ierr = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
    ierr = PetscFree5(AY0c,AYlc,AYtc,MZa,MZb);CHKERRQ(ierr);
    bcgsl->bConvex = uMROR;
    ksp->setupstage = KSP_SETUP_NEW;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPBCGSLSetEll"
/*@
   KSPBCGSLSetEll - Sets the number of search directions in BiCGStab(L).

   Logically Collective on KSP

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  ell - number of search directions

   Options Database Keys:

.  -ksp_bcgsl_ell ell

   Level: intermediate

.keywords: KSP, BiCGStab(L), set, exact residuals,

.seealso: @()
@*/
PetscErrorCode  KSPBCGSLSetEll(KSP ksp, PetscInt ell)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ell < 1) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE, "KSPBCGSLSetEll: second argument must be positive");
  PetscValidLogicalCollectiveInt(ksp,ell,2);

  if (!ksp->setupstage) {
    bcgsl->ell = ell;
  } else if (bcgsl->ell != ell) {
    /* free the data structures, then create them again */
    ierr = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
    ierr = PetscFree5(AY0c,AYlc,AYtc,MZa,MZb);CHKERRQ(ierr);
    bcgsl->ell = ell;
    ksp->setupstage = KSP_SETUP_NEW;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_BCGSL"
PetscErrorCode KSPView_BCGSL(KSP ksp, PetscViewer viewer)
{
  KSP_BCGSL       *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscErrorCode  ierr;
  PetscBool       isascii, isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring);CHKERRQ(ierr);

  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  BCGSL: Ell = %D\n", bcgsl->ell);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  BCGSL: Delta = %lg\n", bcgsl->delta);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Viewer type %s not supported for KSP BCGSL", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_BCGSL"
PetscErrorCode KSPSetFromOptions_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscErrorCode ierr;
  PetscInt       this_ell;
  PetscReal      delta;
  PetscBool      flga = PETSC_FALSE, flg;

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
  ierr = PetscOptionsBool("-ksp_bcgsl_cxpoly", "Polynomial part of BiCGStabL is MinRes + OR", "KSPBCGSLSetPol", flga,&flga,PETSC_NULL);CHKERRQ(ierr);
  if (flga) {
    ierr = KSPBCGSLSetPol(ksp, PETSC_TRUE);CHKERRQ(ierr);
  } else {
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_bcgsl_mrpoly", "Polynomial part of BiCGStabL is MinRes", "KSPBCGSLSetPol", flg,&flg,PETSC_NULL);CHKERRQ(ierr);
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
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscInt       ell = bcgsl->ell,ldMZ = ell+1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP, "no symmetric preconditioning for KSPBCGSL");
  else if (ksp->pc_side == PC_RIGHT) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP, "no right preconditioning for KSPBCGSL");
  ierr = KSPDefaultGetWork(ksp, 6+2*ell);CHKERRQ(ierr);
  ierr = PetscMalloc5(ldMZ,PetscScalar,&AY0c,ldMZ,PetscScalar,&AYlc,ldMZ,PetscScalar,&AYtc,ldMZ*ldMZ,PetscScalar,&MZa,ldMZ*ldMZ,PetscScalar,&MZb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPReset_BCGSL"
PetscErrorCode KSPReset_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL *)ksp->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
  ierr = PetscFree5(AY0c,AYlc,AYtc,MZa,MZb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_BCGSL"
PetscErrorCode KSPDestroy_BCGSL(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_BCGSL(ksp);CHKERRQ(ierr);
  ierr = KSPDefaultDestroy(ksp);CHKERRQ(ierr);
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
+  -ksp_bcgsl_ell <ell> Number of Krylov search directions, defaults to 2 -- KSPBCGSLSetEll()
.  -ksp_bcgsl_cxpol - Use a convex function of the MinRes and OR polynomials after the BiCG step instead of default MinRes -- KSPBCGSLSetPol()
.  -ksp_bcgsl_mrpoly - Use the default MinRes polynomial after the BiCG step  -- KSPBCGSLSetPol()
-  -ksp_bcgsl_xres <res> Threshold used to decide when to refresh computed residuals -- KSPBCGSLSetXRes()

   Notes: Supports left preconditioning only

   Level: beginner

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPBCGS, KSPSetPCSide(), KSPBCGSLSetEll(), KSPBCGSLSetXRes()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_BCGSL"
PetscErrorCode  KSPCreate_BCGSL(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGSL      *bcgsl;

  PetscFunctionBegin;
  /* allocate BiCGStab(L) context */
  ierr = PetscNewLog(ksp, KSP_BCGSL, &bcgsl);CHKERRQ(ierr);
  ksp->data = (void*)bcgsl;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);

  ksp->ops->setup           = KSPSetUp_BCGSL;
  ksp->ops->solve           = KSPSolve_BCGSL;
  ksp->ops->reset           = KSPReset_BCGSL;
  ksp->ops->destroy         = KSPDestroy_BCGSL;
  ksp->ops->buildsolution   = KSPDefaultBuildSolution;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = KSPSetFromOptions_BCGSL;
  ksp->ops->view            = KSPView_BCGSL;

  /* Let the user redefine the number of directions vectors */
  bcgsl->ell = 2;

  /*Choose between a single MR step or an averaged MR/OR */
  bcgsl->bConvex = PETSC_FALSE;

  /* Set the threshold for when exact residuals will be used */
  bcgsl->delta = 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

