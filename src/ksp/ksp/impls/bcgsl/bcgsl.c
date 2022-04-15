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
#include <petsc/private/kspimpl.h>              /*I   "petscksp.h" I*/
#include <../src/ksp/ksp/impls/bcgsl/bcgslimpl.h>
#include <petscblaslapack.h>

static PetscErrorCode  KSPSolve_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*) ksp->data;
  PetscScalar    alpha, beta, omega, sigma;
  PetscScalar    rho0, rho1;
  PetscReal      kappa0, kappaA, kappa1;
  PetscReal      ghat;
  PetscReal      zeta, zeta0, rnmax_computed, rnmax_true, nrm0;
  PetscBool      bUpdateX;
  PetscInt       maxit;
  PetscInt       h, i, j, k, vi, ell;
  PetscBLASInt   ldMZ,bierr;
  PetscScalar    utb;
  PetscReal      max_s, pinv_tol;

  PetscFunctionBegin;
  /* set up temporary vectors */
  vi         = 0;
  ell        = bcgsl->ell;
  bcgsl->vB  = ksp->work[vi]; vi++;
  bcgsl->vRt = ksp->work[vi]; vi++;
  bcgsl->vTm = ksp->work[vi]; vi++;
  bcgsl->vvR = ksp->work+vi; vi += ell+1;
  bcgsl->vvU = ksp->work+vi; vi += ell+1;
  bcgsl->vXr = ksp->work[vi]; vi++;
  PetscCall(PetscBLASIntCast(ell+1,&ldMZ));

  /* Prime the iterative solver */
  PetscCall(KSPInitialResidual(ksp, VX, VTM, VB, VVR[0], ksp->vec_rhs));
  PetscCall(VecNorm(VVR[0], NORM_2, &zeta0));
  KSPCheckNorm(ksp,zeta0);
  rnmax_computed = zeta0;
  rnmax_true     = zeta0;

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = zeta0;
  else ksp->rnorm = 0.0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  PetscCall(VecSet(VVU[0],0.0));
  alpha = 0.;
  rho0  = omega = 1;

  if (bcgsl->delta>0.0) {
    PetscCall(VecCopy(VX, VXR));
    PetscCall(VecSet(VX,0.0));
    PetscCall(VecCopy(VVR[0], VB));
  } else {
    PetscCall(VecCopy(ksp->vec_rhs, VB));
  }

  /* Life goes on */
  PetscCall(VecCopy(VVR[0], VRT));
  zeta = zeta0;

  PetscCall(KSPGetTolerances(ksp, NULL, NULL, NULL, &maxit));

  for (k=0; k<maxit; k += bcgsl->ell) {
    ksp->its = k;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = zeta;
    else ksp->rnorm = 0.0;

    PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
    PetscCall(KSPMonitor(ksp, ksp->its, ksp->rnorm));

    PetscCall((*ksp->converged)(ksp, k, ksp->rnorm, &ksp->reason, ksp->cnvP));
    if (ksp->reason < 0) PetscFunctionReturn(0);
    if (ksp->reason) {
      if (bcgsl->delta>0.0) {
        PetscCall(VecAXPY(VX,1.0,VXR));
      }
      PetscFunctionReturn(0);
    }

    /* BiCG part */
    rho0 = -omega*rho0;
    nrm0 = zeta;
    for (j=0; j<bcgsl->ell; j++) {
      /* rho1 <- r_j' * r_tilde */
      PetscCall(VecDot(VVR[j], VRT, &rho1));
      KSPCheckDot(ksp,rho1);
      if (rho1 == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        PetscFunctionReturn(0);
      }
      beta = alpha*(rho1/rho0);
      rho0 = rho1;
      for (i=0; i<=j; i++) {
        /* u_i <- r_i - beta*u_i */
        PetscCall(VecAYPX(VVU[i], -beta, VVR[i]));
      }
      /* u_{j+1} <- inv(K)*A*u_j */
      PetscCall(KSP_PCApplyBAorAB(ksp, VVU[j], VVU[j+1], VTM));

      PetscCall(VecDot(VVU[j+1], VRT, &sigma));
      KSPCheckDot(ksp,sigma);
      if (sigma == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        PetscFunctionReturn(0);
      }
      alpha = rho1/sigma;

      /* x <- x + alpha*u_0 */
      PetscCall(VecAXPY(VX, alpha, VVU[0]));

      for (i=0; i<=j; i++) {
        /* r_i <- r_i - alpha*u_{i+1} */
        PetscCall(VecAXPY(VVR[i], -alpha, VVU[i+1]));
      }

      /* r_{j+1} <- inv(K)*A*r_j */
      PetscCall(KSP_PCApplyBAorAB(ksp, VVR[j], VVR[j+1], VTM));

      PetscCall(VecNorm(VVR[0], NORM_2, &nrm0));
      KSPCheckNorm(ksp,nrm0);
      if (bcgsl->delta>0.0) {
        if (rnmax_computed<nrm0) rnmax_computed = nrm0;
        if (rnmax_true<nrm0) rnmax_true = nrm0;
      }

      /* NEW: check for early exit */
      PetscCall((*ksp->converged)(ksp, k+j, nrm0, &ksp->reason, ksp->cnvP));
      if (ksp->reason) {
        PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
        ksp->its   = k+j;
        ksp->rnorm = nrm0;

        PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
        if (ksp->reason < 0) PetscFunctionReturn(0);
      }
    }

    /* Polynomial part */
    for (i = 0; i <= bcgsl->ell; ++i) {
      PetscCall(VecMDot(VVR[i], i+1, VVR, &MZa[i*ldMZ]));
    }
    /* Symmetrize MZa */
    for (i = 0; i <= bcgsl->ell; ++i) {
      for (j = i+1; j <= bcgsl->ell; ++j) {
        MZa[i*ldMZ+j] = MZa[j*ldMZ+i] = PetscConj(MZa[j*ldMZ+i]);
      }
    }
    /* Copy MZa to MZb */
    PetscCall(PetscArraycpy(MZb,MZa,ldMZ*ldMZ));

    if (!bcgsl->bConvex || bcgsl->ell==1) {
      PetscBLASInt ione = 1,bell;
      PetscCall(PetscBLASIntCast(bcgsl->ell,&bell));

      AY0c[0] = -1;
      if (bcgsl->pinv) {
#  if defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&bell,&bell,&MZa[1+ldMZ],&ldMZ,bcgsl->s,bcgsl->u,&bell,bcgsl->v,&bell,bcgsl->work,&bcgsl->lwork,bcgsl->realwork,&bierr));
#  else
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&bell,&bell,&MZa[1+ldMZ],&ldMZ,bcgsl->s,bcgsl->u,&bell,bcgsl->v,&bell,bcgsl->work,&bcgsl->lwork,&bierr));
#  endif
        if (bierr!=0) {
          ksp->reason = KSP_DIVERGED_BREAKDOWN;
          PetscFunctionReturn(0);
        }
        /* Apply pseudo-inverse */
        max_s = bcgsl->s[0];
        for (i=1; i<bell; i++) {
          if (bcgsl->s[i] > max_s) {
            max_s = bcgsl->s[i];
          }
        }
        /* tolerance is hardwired to bell*max(s)*PETSC_MACHINE_EPSILON */
        pinv_tol = bell*max_s*PETSC_MACHINE_EPSILON;
        PetscCall(PetscArrayzero(&AY0c[1],bell));
        for (i=0; i<bell; i++) {
          if (bcgsl->s[i] >= pinv_tol) {
            utb=0.;
            for (j=0; j<bell; j++) {
              utb += MZb[1+j]*bcgsl->u[i*bell+j];
            }

            for (j=0; j<bell; j++) {
              AY0c[1+j] += utb/bcgsl->s[i]*bcgsl->v[j*bell+i];
            }
          }
        }
      } else {
        PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("Lower", &bell, &MZa[1+ldMZ], &ldMZ, &bierr));
        if (bierr!=0) {
          ksp->reason = KSP_DIVERGED_BREAKDOWN;
          PetscFunctionReturn(0);
        }
        PetscCall(PetscArraycpy(&AY0c[1],&MZb[1],bcgsl->ell));
        PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("Lower", &bell, &ione, &MZa[1+ldMZ], &ldMZ, &AY0c[1], &ldMZ, &bierr));
      }
    } else {
      PetscBLASInt ione = 1;
      PetscScalar  aone = 1.0, azero = 0.0;
      PetscBLASInt neqs;
      PetscCall(PetscBLASIntCast(bcgsl->ell-1,&neqs));

      PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("Lower", &neqs, &MZa[1+ldMZ], &ldMZ, &bierr));
      if (bierr!=0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscFunctionReturn(0);
      }
      PetscCall(PetscArraycpy(&AY0c[1],&MZb[1],bcgsl->ell-1));
      PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("Lower", &neqs, &ione, &MZa[1+ldMZ], &ldMZ, &AY0c[1], &ldMZ, &bierr));
      AY0c[0]          = -1;
      AY0c[bcgsl->ell] = 0.;

      PetscCall(PetscArraycpy(&AYlc[1],&MZb[1+ldMZ*(bcgsl->ell)],bcgsl->ell-1));
      PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("Lower", &neqs, &ione, &MZa[1+ldMZ], &ldMZ, &AYlc[1], &ldMZ, &bierr));

      AYlc[0]          = 0.;
      AYlc[bcgsl->ell] = -1;

      PetscStackCallBLAS("BLASgemv",BLASgemv_("NoTr", &ldMZ, &ldMZ, &aone, MZb, &ldMZ, AY0c, &ione, &azero, AYtc, &ione));

      kappa0 = PetscRealPart(BLASdot_(&ldMZ, AY0c, &ione, AYtc, &ione));

      /* round-off can cause negative kappa's */
      if (kappa0<0) kappa0 = -kappa0;
      kappa0 = PetscSqrtReal(kappa0);

      kappaA = PetscRealPart(BLASdot_(&ldMZ, AYlc, &ione, AYtc, &ione));

      PetscStackCallBLAS("BLASgemv",BLASgemv_("noTr", &ldMZ, &ldMZ, &aone, MZb, &ldMZ, AYlc, &ione, &azero, AYtc, &ione));

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
    for (h=bcgsl->ell; h>0 && omega==0.0; h--) omega = AY0c[h];
    if (omega==0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscFunctionReturn(0);
    }

    PetscCall(VecMAXPY(VX, bcgsl->ell,AY0c+1, VVR));
    for (i=1; i<=bcgsl->ell; i++) AY0c[i] *= -1.0;
    PetscCall(VecMAXPY(VVU[0], bcgsl->ell,AY0c+1, VVU+1));
    PetscCall(VecMAXPY(VVR[0], bcgsl->ell,AY0c+1, VVR+1));
    for (i=1; i<=bcgsl->ell; i++) AY0c[i] *= -1.0;
    PetscCall(VecNorm(VVR[0], NORM_2, &zeta));
    KSPCheckNorm(ksp,zeta);

    /* Accurate Update */
    if (bcgsl->delta>0.0) {
      if (rnmax_computed<zeta) rnmax_computed = zeta;
      if (rnmax_true<zeta) rnmax_true = zeta;

      bUpdateX = (PetscBool) (zeta<bcgsl->delta*zeta0 && zeta0<=rnmax_computed);
      if ((zeta<bcgsl->delta*rnmax_true && zeta0<=rnmax_true) || bUpdateX) {
        /* r0 <- b-inv(K)*A*X */
        PetscCall(KSP_PCApplyBAorAB(ksp, VX, VVR[0], VTM));
        PetscCall(VecAYPX(VVR[0], -1.0, VB));
        rnmax_true = zeta;

        if (bUpdateX) {
          PetscCall(VecAXPY(VXR,1.0,VX));
          PetscCall(VecSet(VX,0.0));
          PetscCall(VecCopy(VVR[0], VB));
          rnmax_computed = zeta;
        }
      }
    }
  }
  if (bcgsl->delta>0.0) {
    PetscCall(VecAXPY(VX,1.0,VXR));
  }

  ksp->its = k;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = zeta;
  else ksp->rnorm = 0.0;
  PetscCall(KSPMonitor(ksp, ksp->its, ksp->rnorm));
  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, k, ksp->rnorm, &ksp->reason, ksp->cnvP));
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*@
   KSPBCGSLSetXRes - Sets the parameter governing when
   exact residuals will be used instead of computed residuals.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  delta - computed residuals are used alone when delta is not positive

   Options Database Keys:

.  -ksp_bcgsl_xres delta - Threshold used to decide when to refresh computed residuals

   Level: intermediate

.seealso: KSPBCGSLSetEll(), KSPBCGSLSetPol(), KSP
@*/
PetscErrorCode  KSPBCGSLSetXRes(KSP ksp, PetscReal delta)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ksp,delta,2);
  if (ksp->setupstage) {
    if ((delta<=0 && bcgsl->delta>0) || (delta>0 && bcgsl->delta<=0)) {
      PetscCall(VecDestroyVecs(ksp->nwork,&ksp->work));
      PetscCall(PetscFree5(AY0c,AYlc,AYtc,MZa,MZb));
      PetscCall(PetscFree4(bcgsl->work,bcgsl->s,bcgsl->u,bcgsl->v));
      ksp->setupstage = KSP_SETUP_NEW;
    }
  }
  bcgsl->delta = delta;
  PetscFunctionReturn(0);
}

/*@
   KSPBCGSLSetUsePseudoinverse - Use pseudoinverse (via SVD) to solve polynomial part of update

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  use_pinv - set to PETSC_TRUE when using pseudoinverse

   Options Database Keys:

.  -ksp_bcgsl_pinv - use pseudoinverse

   Level: intermediate

.seealso: KSPBCGSLSetEll(), KSP
@*/
PetscErrorCode KSPBCGSLSetUsePseudoinverse(KSP ksp,PetscBool use_pinv)
{
  KSP_BCGSL *bcgsl = (KSP_BCGSL*)ksp->data;

  PetscFunctionBegin;
  bcgsl->pinv = use_pinv;
  PetscFunctionReturn(0);
}

/*@
   KSPBCGSLSetPol - Sets the type of polynomial part will
   be used in the BiCGSTab(L) solver.

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  uMROR - set to PETSC_TRUE when the polynomial is a convex combination of an MR and an OR step.

   Options Database Keys:

+  -ksp_bcgsl_cxpoly - use enhanced polynomial
-  -ksp_bcgsl_mrpoly - use standard polynomial

   Level: intermediate

.seealso: KSP, KSPBCGSL, KSPCreate(), KSPSetType()
@*/
PetscErrorCode  KSPBCGSLSetPol(KSP ksp, PetscBool uMROR)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(ksp,uMROR,2);

  if (!ksp->setupstage) bcgsl->bConvex = uMROR;
  else if (bcgsl->bConvex != uMROR) {
    /* free the data structures,
       then create them again
     */
    PetscCall(VecDestroyVecs(ksp->nwork,&ksp->work));
    PetscCall(PetscFree5(AY0c,AYlc,AYtc,MZa,MZb));
    PetscCall(PetscFree4(bcgsl->work,bcgsl->s,bcgsl->u,bcgsl->v));

    bcgsl->bConvex  = uMROR;
    ksp->setupstage = KSP_SETUP_NEW;
  }
  PetscFunctionReturn(0);
}

/*@
   KSPBCGSLSetEll - Sets the number of search directions in BiCGStab(L).

   Logically Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate
-  ell - number of search directions

   Options Database Keys:

.  -ksp_bcgsl_ell ell - Number of Krylov search directions

   Level: intermediate

   Notes:
   For large ell it is common for the polynomial update problem to become singular (due to happy breakdown for smallish
   test problems, but also for larger problems). Consequently, by default, the system is solved by pseudoinverse, which
   allows the iteration to complete successfully. See KSPBCGSLSetUsePseudoinverse() to switch to a conventional solve.

.seealso: KSPBCGSLSetUsePseudoinverse(), KSP, KSPBCGSL
@*/
PetscErrorCode  KSPBCGSLSetEll(KSP ksp, PetscInt ell)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;

  PetscFunctionBegin;
  PetscCheck(ell >= 1,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE, "KSPBCGSLSetEll: second argument must be positive");
  PetscValidLogicalCollectiveInt(ksp,ell,2);

  if (!ksp->setupstage) bcgsl->ell = ell;
  else if (bcgsl->ell != ell) {
    /* free the data structures, then create them again */
    PetscCall(VecDestroyVecs(ksp->nwork,&ksp->work));
    PetscCall(PetscFree5(AY0c,AYlc,AYtc,MZa,MZb));
    PetscCall(PetscFree4(bcgsl->work,bcgsl->s,bcgsl->u,bcgsl->v));

    bcgsl->ell      = ell;
    ksp->setupstage = KSP_SETUP_NEW;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_BCGSL(KSP ksp, PetscViewer viewer)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ell = %" PetscInt_FMT "\n", bcgsl->ell));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Delta = %g\n", (double)bcgsl->delta));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_BCGSL(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;
  PetscInt       this_ell;
  PetscReal      delta;
  PetscBool      flga = PETSC_FALSE, flg;

  PetscFunctionBegin;
  /* PetscOptionsBegin/End are called in KSPSetFromOptions. They
     don't need to be called here.
  */
  PetscOptionsHeadBegin(PetscOptionsObject,"KSP BiCGStab(L) Options");

  /* Set number of search directions */
  PetscCall(PetscOptionsInt("-ksp_bcgsl_ell","Number of Krylov search directions","KSPBCGSLSetEll",bcgsl->ell,&this_ell,&flg));
  if (flg) {
    PetscCall(KSPBCGSLSetEll(ksp, this_ell));
  }

  /* Set polynomial type */
  PetscCall(PetscOptionsBool("-ksp_bcgsl_cxpoly", "Polynomial part of BiCGStabL is MinRes + OR", "KSPBCGSLSetPol", flga,&flga,NULL));
  if (flga) {
    PetscCall(KSPBCGSLSetPol(ksp, PETSC_TRUE));
  } else {
    flg  = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-ksp_bcgsl_mrpoly", "Polynomial part of BiCGStabL is MinRes", "KSPBCGSLSetPol", flg,&flg,NULL));
    PetscCall(KSPBCGSLSetPol(ksp, PETSC_FALSE));
  }

  /* Will computed residual be refreshed? */
  PetscCall(PetscOptionsReal("-ksp_bcgsl_xres", "Threshold used to decide when to refresh computed residuals", "KSPBCGSLSetXRes", bcgsl->delta, &delta, &flg));
  if (flg) {
    PetscCall(KSPBCGSLSetXRes(ksp, delta));
  }

  /* Use pseudoinverse? */
  flg  = bcgsl->pinv;
  PetscCall(PetscOptionsBool("-ksp_bcgsl_pinv", "Polynomial correction via pseudoinverse", "KSPBCGSLSetUsePseudoinverse",flg,&flg,NULL));
  PetscCall(KSPBCGSLSetUsePseudoinverse(ksp,flg));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUp_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;
  PetscInt       ell    = bcgsl->ell,ldMZ = ell+1;

  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp, 6+2*ell));
  PetscCall(PetscMalloc5(ldMZ,&AY0c,ldMZ,&AYlc,ldMZ,&AYtc,ldMZ*ldMZ,&MZa,ldMZ*ldMZ,&MZb));
  PetscCall(PetscBLASIntCast(5*ell,&bcgsl->lwork));
  PetscCall(PetscMalloc5(bcgsl->lwork,&bcgsl->work,ell,&bcgsl->s,ell*ell,&bcgsl->u,ell*ell,&bcgsl->v,5*ell,&bcgsl->realwork));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl = (KSP_BCGSL*)ksp->data;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(ksp->nwork,&ksp->work));
  PetscCall(PetscFree5(AY0c,AYlc,AYtc,MZa,MZb));
  PetscCall(PetscFree5(bcgsl->work,bcgsl->s,bcgsl->u,bcgsl->v,bcgsl->realwork));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_BCGSL(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_BCGSL(ksp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(0);
}

/*MC
     KSPBCGSL - Implements a slight variant of the Enhanced
                BiCGStab(L) algorithm in (3) and (2).  The variation
                concerns cases when either kappa0**2 or kappa1**2 is
                negative due to round-off. Kappa0 has also been pulled
                out of the denominator in the formula for ghat.

    References:
+   * - G.L.G. Sleijpen, H.A. van der Vorst, "An overview of
         approaches for the stable computation of hybrid BiCG
         methods", Applied Numerical Mathematics: Transactions
         f IMACS, 19(3), 1996.
.   * - G.L.G. Sleijpen, H.A. van der Vorst, D.R. Fokkema,
         "BiCGStab(L) and other hybrid BiCG methods",
          Numerical Algorithms, 7, 1994.
-   * - D.R. Fokkema, "Enhanced implementation of BiCGStab(L)
         for solving linear systems of equations", preprint
         from www.citeseer.com.

   Contributed by: Joel M. Malard, email jm.malard@pnl.gov

   Options Database Keys:
+  -ksp_bcgsl_ell <ell> Number of Krylov search directions, defaults to 2 -- KSPBCGSLSetEll()
.  -ksp_bcgsl_cxpol - Use a convex function of the MinRes and OR polynomials after the BiCG step instead of default MinRes -- KSPBCGSLSetPol()
.  -ksp_bcgsl_mrpoly - Use the default MinRes polynomial after the BiCG step  -- KSPBCGSLSetPol()
.  -ksp_bcgsl_xres <res> Threshold used to decide when to refresh computed residuals -- KSPBCGSLSetXRes()
-  -ksp_bcgsl_pinv <true/false> - (de)activate use of pseudoinverse -- KSPBCGSLSetUsePseudoinverse()

   Level: beginner

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPFGMRES, KSPBCGS, KSPSetPCSide(), KSPBCGSLSetEll(), KSPBCGSLSetXRes()

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_BCGSL(KSP ksp)
{
  KSP_BCGSL      *bcgsl;

  PetscFunctionBegin;
  /* allocate BiCGStab(L) context */
  PetscCall(PetscNewLog(ksp,&bcgsl));
  ksp->data = (void*)bcgsl;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));

  ksp->ops->setup          = KSPSetUp_BCGSL;
  ksp->ops->solve          = KSPSolve_BCGSL;
  ksp->ops->reset          = KSPReset_BCGSL;
  ksp->ops->destroy        = KSPDestroy_BCGSL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGSL;
  ksp->ops->view           = KSPView_BCGSL;

  /* Let the user redefine the number of directions vectors */
  bcgsl->ell = 2;

  /*Choose between a single MR step or an averaged MR/OR */
  bcgsl->bConvex = PETSC_FALSE;

  bcgsl->pinv = PETSC_TRUE;     /* Use the reliable method by default */

  /* Set the threshold for when exact residuals will be used */
  bcgsl->delta = 0.0;
  PetscFunctionReturn(0);
}
