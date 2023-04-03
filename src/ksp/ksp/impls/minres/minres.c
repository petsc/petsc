#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

PetscBool  QLPcite       = PETSC_FALSE;
const char QLPCitation[] = "@article{choi2011minres,\n"
                           "  title={MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems},\n"
                           "  author={Choi, Sou-Cheng T and Paige, Christopher C and Saunders, Michael A},\n"
                           "  journal={SIAM Journal on Scientific Computing},\n"
                           "  volume={33},\n"
                           "  number={4},\n"
                           "  pages={1810--1836},\n"
                           "  year={2011},\n}\n";

typedef struct {
  PetscReal         haptol;
  PetscBool         qlp;
  PetscReal         maxxnorm;
  PetscReal         TranCond;
  PetscBool         monitor;
  PetscViewer       viewer;
  PetscViewerFormat viewer_fmt;
} KSP_MINRES;

static PetscErrorCode KSPSetUp_MINRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp, 9));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Convenience functions */
#define KSPMinresSwap3(V1, V2, V3) \
  do { \
    Vec T = V1; \
    V1    = V2; \
    V2    = V3; \
    V3    = T; \
  } while (0)

static inline PetscReal Norm3(PetscReal a, PetscReal b, PetscReal c)
{
  return PetscSqrtReal(PetscSqr(a) + PetscSqr(b) + PetscSqr(c));
}

static inline void SymOrtho(PetscReal a, PetscReal b, PetscReal *c, PetscReal *s, PetscReal *r)
{
  if (b == 0.0) {
    if (a == 0.0) *c = 1.0;
    else *c = PetscCopysignReal(1.0, a);
    *s = 0.0;
    *r = PetscAbsReal(a);
  } else if (a == 0.0) {
    *c = 0.0;
    *s = PetscCopysignReal(1.0, b);
    *r = PetscAbsReal(b);
  } else if (PetscAbsReal(b) > PetscAbsReal(a)) {
    PetscReal t = a / b;

    *s = PetscCopysignReal(1.0, b) / PetscSqrtReal(1.0 + t * t);
    *c = (*s) * t;
    *r = b / (*s); // computationally better than d = a / c since |c| <= |s|
  } else {
    PetscReal t = b / a;

    *c = PetscCopysignReal(1.0, a) / PetscSqrtReal(1.0 + t * t);
    *s = (*c) * t;
    *r = a / (*c); // computationally better than d = b / s since |s| <= |c|
  }
}

/*
   Code adapted from https://stanford.edu/group/SOL/software/minresqlp/minresqlp-matlab/CPS11.zip
      CSP11/Algorithms/MINRESQLP/minresQLP.m
*/
static PetscErrorCode KSPSolve_MINRES(KSP ksp)
{
  Mat         Amat;
  Vec         X, B, R1, R2, R3, V, W, WL, WL2, XL2, RN;
  PetscReal   alpha, beta, beta1, betan, betal;
  PetscBool   diagonalscale;
  PetscReal   zero = 0.0, dbar, dltan = 0.0, dlta, cs = -1.0, sn = 0.0, epln, eplnn = 0.0, gbar, dlta_QLP;
  PetscReal   gamal3 = 0.0, gamal2 = 0.0, gamal = 0.0, gama = 0.0, gama_tmp;
  PetscReal   taul2 = 0.0, taul = 0.0, tau = 0.0, phi;
  PetscReal   Axnorm, xnorm, xnorm_tmp, xl2norm = 0.0, pnorm, Anorm = 0.0, gmin = 0.0, gminl = 0.0, gminl2 = 0.0;
  PetscReal   Acond = 1.0, Acondl = 0.0, rnorml, rnorm, rootl, relAresl, relres, relresl, Arnorml, Anorml = 0.0, xnorml = 0.0;
  PetscReal   epsx, realmin = PETSC_REAL_MIN, eps = PETSC_MACHINE_EPSILON;
  PetscReal   veplnl2 = 0.0, veplnl = 0.0, vepln = 0.0, etal2 = 0.0, etal = 0.0, eta = 0.0;
  PetscReal   dlta_tmp, sr2 = 0.0, cr2 = -1.0, cr1 = -1.0, sr1 = 0.0;
  PetscReal   ul4 = 0.0, ul3 = 0.0, ul2 = 0.0, ul = 0.0, u = 0.0, ul_QLP = 0.0, u_QLP = 0.0;
  PetscReal   vepln_QLP = 0.0, gamal_QLP = 0.0, gama_QLP = 0.0, gamal_tmp, abs_gama;
  PetscInt    flag = -2, flag0 = -2, QLPiter = 0;
  KSP_MINRES *minres = (KSP_MINRES *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(QLPCitation, &QLPcite));
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;
  R1  = ksp->work[0];
  R2  = ksp->work[1];
  R3  = ksp->work[2];
  V   = ksp->work[3];
  W   = ksp->work[4];
  WL  = ksp->work[5];
  WL2 = ksp->work[6];
  XL2 = ksp->work[7];
  RN  = ksp->work[8];
  PetscCall(PCGetOperators(ksp->pc, &Amat, NULL));

  ksp->its   = 0;
  ksp->rnorm = 0.0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, R2));
    PetscCall(VecNorm(R2, NORM_2, &Axnorm));
    PetscCall(VecNorm(X, NORM_2, &xnorm));
    PetscCall(VecAYPX(R2, -1.0, B));
  } else {
    PetscCall(VecCopy(B, R2));
    Axnorm = 0.0;
    xnorm  = 0.0;
  }
  if (ksp->converged_neg_curve) PetscCall(VecCopy(R2, RN));
  PetscCall(KSP_PCApply(ksp, R2, R3));
  PetscCall(VecDotRealPart(R3, R2, &beta1));
  KSPCheckDot(ksp, beta1);
  if (beta1 < 0.0) {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Detected indefinite operator %g", (double)beta1);
    ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  beta1 = PetscSqrtReal(beta1);

  rnorm = beta1;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rnorm;
  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
  PetscCall(KSPMonitor(ksp, 0, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  relres = rnorm / beta1;
  betan  = beta1;
  phi    = beta1;
  betan  = beta1;
  beta   = 0.0;
  do {
    /* Lanczos */
    ksp->its++;
    betal = beta;
    beta  = betan;
    PetscCall(VecAXPBY(V, 1.0 / beta, 0.0, R3));
    PetscCall(KSP_MatMult(ksp, Amat, V, R3));
    if (ksp->its > 1) PetscCall(VecAXPY(R3, -beta / betal, R1));
    PetscCall(VecDotRealPart(R3, V, &alpha));
    PetscCall(VecAXPY(R3, -alpha / beta, R2));
    KSPMinresSwap3(R1, R2, R3);

    PetscCall(KSP_PCApply(ksp, R2, R3));
    PetscCall(VecDotRealPart(R3, R2, &betan));
    KSPCheckDot(ksp, betan);
    if (betan < 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Detected indefinite preconditioner %g", (double)betan);
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    betan = PetscSqrtReal(betan);

    pnorm = Norm3(betal, alpha, betan);

    // Apply previous left rotation Q_{k-1}
    dbar     = dltan;
    epln     = eplnn;
    dlta     = cs * dbar + sn * alpha;
    gbar     = sn * dbar - cs * alpha;
    eplnn    = sn * betan;
    dltan    = -cs * betan;
    dlta_QLP = dlta;

    // Stop if negative curvature is detected and return residual
    // This is very experimental and maybe changed in the future
    // based on https://arxiv.org/pdf/2208.07095.pdf
    if (ksp->converged_neg_curve) {
      if (cs * gbar >= 0.0) {
        PetscCall(PetscInfo(ksp, "Detected negative curvature c_nm1 %g, gbar %g\n", (double)cs, (double)gbar));
        ksp->reason = KSP_CONVERGED_NEG_CURVE;
        PetscCall(VecCopy(RN, X));
        break;
      } else {
        PetscCall(VecAXPBY(RN, -phi * cs, PetscSqr(sn), V));
      }
    }

    // Compute the current left plane rotation Q_k
    gamal3 = gamal2;
    gamal2 = gamal;
    gamal  = gama;
    SymOrtho(gbar, betan, &cs, &sn, &gama);

    gama_tmp = gama;
    taul2    = taul;
    taul     = tau;
    tau      = cs * phi;
    Axnorm   = Norm3(Axnorm, tau, zero);
    phi      = sn * phi;

    //Apply the previous right plane rotation P{k-2,k}
    if (ksp->its > 2) {
      veplnl2  = veplnl;
      etal2    = etal;
      etal     = eta;
      dlta_tmp = sr2 * vepln - cr2 * dlta;
      veplnl   = cr2 * vepln + sr2 * dlta;
      dlta     = dlta_tmp;
      eta      = sr2 * gama;
      gama     = -cr2 * gama;
    }

    // Compute the current right plane rotation P{k-1,k}, P_12, P_23,...
    if (ksp->its > 1) {
      SymOrtho(gamal, dlta, &cr1, &sr1, &gamal);
      vepln = sr1 * gama;
      gama  = -cr1 * gama;
    }

    // Update xnorm
    xnorml = xnorm;
    ul4    = ul3;
    ul3    = ul2;
    if (ksp->its > 2) ul2 = (taul2 - etal2 * ul4 - veplnl2 * ul3) / gamal2;
    if (ksp->its > 1) ul = (taul - etal * ul3 - veplnl * ul2) / gamal;
    xnorm_tmp = Norm3(xl2norm, ul2, ul);
    if (PetscAbsReal(gama) > realmin && xnorm_tmp < minres->maxxnorm) {
      u = (tau - eta * ul2 - vepln * ul) / gama;
      if (Norm3(xnorm_tmp, u, zero) > minres->maxxnorm) {
        u    = 0;
        flag = 6;
      }
    } else {
      u    = 0;
      flag = 9;
    }
    xl2norm = Norm3(xl2norm, ul2, zero);
    xnorm   = Norm3(xl2norm, ul, u);

    // Update w. Update x except if it will become too big
    //if (Acond < minres->TranCond && flag != flag0 && QLPiter == 0) { // I believe they have a typo in the matlab code
    if ((Acond < minres->TranCond || !minres->qlp) && flag == flag0 && QLPiter == 0) { // MINRES
      KSPMinresSwap3(WL2, WL, W);
      PetscCall(VecAXPBY(W, 1.0 / gama_tmp, 0.0, V));
      if (ksp->its > 1) {
        Vec         T[]      = {WL, WL2};
        PetscScalar alphas[] = {-dlta_QLP / gama_tmp, -epln / gama_tmp};
        PetscInt    nv       = (ksp->its == 2 ? 1 : 2);

        PetscCall(VecMAXPY(W, nv, alphas, T));
      }
      if (xnorm < minres->maxxnorm) {
        PetscCall(VecAXPY(X, tau, W));
      } else {
        flag = 6;
      }
    } else if (minres->qlp) { //MINRES-QLP updates
      QLPiter = QLPiter + 1;
      if (QLPiter == 1) {
        // xl2 = x - wl*ul_QLP - w*u_QLP;
        PetscScalar maxpys[] = {1.0, -ul_QLP, -u_QLP};
        Vec         maxpyv[] = {X, WL, W};

        PetscCall(VecSet(XL2, 0.0));
        // construct w_{k-3}, w_{k-2}, w_{k-1}
        if (ksp->its > 1) {
          if (ksp->its > 3) { // w_{k-3}
            //wl2 = gamal3*wl2 + veplnl2*wl + etal*w;
            PetscCall(VecAXPBYPCZ(WL2, veplnl2, etal, gamal3, WL, W));
          }
          if (ksp->its > 2) { // w_{k-2}
            //wl = gamal_QLP*wl + vepln_QLP*w;
            PetscCall(VecAXPBY(WL, vepln_QLP, gamal_QLP, W));
          }
          // w = gama_QLP*w;
          PetscCall(VecScale(W, gama_QLP));
          // xl2 = x - wl*ul_QLP - w*u_QLP;
          PetscCall(VecMAXPY(XL2, 3, maxpys, maxpyv));
        }
      }
      if (ksp->its == 1) {
        //wl2 = wl;      wl = v*sr1;     w  = -v*cr1;
        PetscCall(VecCopy(WL, WL2));
        PetscCall(VecAXPBY(WL, sr1, 0, V));
        PetscCall(VecAXPBY(W, -cr1, 0, V));
      } else if (ksp->its == 2) {
        //wl2 = wl;
        //wl  = w*cr1 + v*sr1;
        //w   = w*sr1 - v*cr1;
        PetscCall(VecCopy(WL, WL2));
        PetscCall(VecAXPBYPCZ(WL, cr1, sr1, 0.0, W, V));
        PetscCall(VecAXPBY(W, -cr1, sr1, V));
      } else {
        //wl2 = wl;      wl = w;         w  = wl2*sr2 - v*cr2;
        //wl2 = wl2*cr2 + v*sr2;         v  = wl *cr1 + w*sr1;
        //w   = wl *sr1 - w*cr1;         wl = v;
        PetscCall(VecCopy(WL, WL2));
        PetscCall(VecCopy(W, WL));
        PetscCall(VecAXPBYPCZ(W, sr2, -cr2, 0.0, WL2, V));
        PetscCall(VecAXPBY(WL2, sr2, cr2, V));
        PetscCall(VecAXPBYPCZ(V, cr1, sr1, 0.0, WL, W));
        PetscCall(VecAXPBY(W, sr1, -cr1, WL));
        PetscCall(VecCopy(V, WL));
      }

      //xl2 = xl2 + wl2*ul2;
      PetscCall(VecAXPY(XL2, ul2, WL2));
      //x   = xl2 + wl *ul + w*u;
      PetscCall(VecCopy(XL2, X));
      PetscCall(VecAXPBYPCZ(X, ul, u, 1.0, WL, W));
    }
    // Compute the next right plane rotation P{k-1,k+1}
    gamal_tmp = gamal;
    SymOrtho(gamal, eplnn, &cr2, &sr2, &gamal);

    //Store quantities for transferring from MINRES to MINRES-QLP
    gamal_QLP = gamal_tmp;
    vepln_QLP = vepln;
    gama_QLP  = gama;
    ul_QLP    = ul;
    u_QLP     = u;

    // Estimate various norms
    abs_gama = PetscAbsReal(gama);
    Anorml   = Anorm;
    Anorm    = PetscMax(PetscMax(Anorm, pnorm), PetscMax(gamal, abs_gama));
    if (ksp->its == 1) {
      gmin  = gama;
      gminl = gmin;
    } else {
      gminl2 = gminl;
      gminl  = gmin;
      gmin   = PetscMin(gminl2, PetscMin(gamal, abs_gama));
    }
    Acondl  = Acond;
    Acond   = Anorm / gmin;
    rnorml  = rnorm;
    relresl = relres;
    if (flag != 9) rnorm = phi;
    relres   = rnorm / (Anorm * xnorm + beta1);
    rootl    = Norm3(gbar, dltan, zero);
    Arnorml  = rnorml * rootl;
    relAresl = rootl / Anorm;

    // See if any of the stopping criteria are satisfied.
    epsx = Anorm * xnorm * eps;
    if (flag == flag0 || flag == 9) {
      //if (Acond >= Acondlim) flag = 7; // Huge Acond
      if (epsx >= beta1) flag = 5; // x is an eigenvector
      if (minres->qlp) {           /* We use these indicators only if the QLP variant has been selected */
        PetscReal t1 = 1.0 + relres;
        PetscReal t2 = 1.0 + relAresl;
        if (xnorm >= minres->maxxnorm) flag = 6; // xnorm exceeded its limit
        if (t2 <= 1) flag = 4;                   // Accurate LS solution
        if (t1 <= 1) flag = 3;                   // Accurate Ax=b solution
        if (relAresl <= ksp->rtol) flag = 2;     // Good enough LS solution
        if (relres <= ksp->rtol) flag = 1;       // Good enough Ax=b solution
      }
    }

    if (flag == 2 || flag == 4 || flag == 6 || flag == 7) {
      Acond  = Acondl;
      rnorm  = rnorml;
      relres = relresl;
    }

    if (minres->monitor) { /* Mimics matlab code with extra flag */
      PetscCall(PetscViewerPushFormat(minres->viewer, minres->viewer_fmt));
      if (ksp->its == 1) PetscCall(PetscViewerASCIIPrintf(minres->viewer, "        flag      rnorm     Arnorm   Compatible         LS      Anorm      Acond      xnorm\n"));
      PetscCall(PetscViewerASCIIPrintf(minres->viewer, "%s %5d   %2d %10.2e %10.2e   %10.2e %10.2e %10.2e %10.2e %10.2e\n", QLPiter == 1 ? "P" : " ", (int)ksp->its - 1, (int)flag, (double)rnorml, (double)Arnorml, (double)relresl, (double)relAresl, (double)Anorml, (double)Acondl, (double)xnorml));
      PetscCall(PetscViewerPopFormat(minres->viewer));
    }

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rnorm;
    PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
    PetscCall(KSPMonitor(ksp, ksp->its, ksp->rnorm));
    PetscCall((*ksp->converged)(ksp, ksp->its, ksp->rnorm, &ksp->reason, ksp->cnvP));
    if (!ksp->reason) {
      switch (flag) {
      case 1:
      case 2:
      case 5: /* XXX */
        ksp->reason = KSP_CONVERGED_RTOL;
        break;
      case 3:
      case 4:
        ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
        break;
      case 6:
        ksp->reason = KSP_CONVERGED_STEP_LENGTH;
        break;
      default:
        break;
      }
    }
    if (ksp->reason) break;
  } while (ksp->its < ksp->max_it);

  if (minres->monitor && flag != 2 && flag != 4 && flag != 6 && flag != 7) {
    PetscCall(VecNorm(X, NORM_2, &xnorm));
    PetscCall(KSP_MatMult(ksp, Amat, X, R1));
    PetscCall(VecAYPX(R1, -1.0, B));
    PetscCall(VecNorm(R1, NORM_2, &rnorml));
    PetscCall(KSP_MatMult(ksp, Amat, R1, R2));
    PetscCall(VecNorm(R2, NORM_2, &Arnorml));
    relresl  = rnorml / (Anorm * xnorm + beta1);
    relAresl = rnorml > realmin ? Arnorml / (Anorm * rnorml) : 0.0;
    PetscCall(PetscViewerPushFormat(minres->viewer, minres->viewer_fmt));
    PetscCall(PetscViewerASCIIPrintf(minres->viewer, "%s %5d   %2d %10.2e %10.2e   %10.2e %10.2e %10.2e %10.2e %10.2e\n", QLPiter == 1 ? "P" : " ", (int)ksp->its, (int)flag, (double)rnorml, (double)Arnorml, (double)relresl, (double)relAresl, (double)Anorml, (double)Acondl, (double)xnorml));
    PetscCall(PetscViewerPopFormat(minres->viewer));
  }
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This was the original implementation provided by R. Scheichl */
static PetscErrorCode KSPSolve_MINRES_OLD(KSP ksp)
{
  PetscInt          i;
  PetscScalar       alpha, beta, betaold, eta, c = 1.0, ceta, cold = 1.0, coold, s = 0.0, sold = 0.0, soold;
  PetscScalar       rho0, rho1, rho2, rho3, dp = 0.0;
  const PetscScalar none = -1.0;
  PetscReal         np;
  Vec               X, B, R, Z, U, V, W, UOLD, VOLD, WOLD, WOOLD;
  Mat               Amat;
  KSP_MINRES       *minres = (KSP_MINRES *)ksp->data;
  PetscBool         diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  X     = ksp->vec_sol;
  B     = ksp->vec_rhs;
  R     = ksp->work[0];
  Z     = ksp->work[1];
  U     = ksp->work[2];
  V     = ksp->work[3];
  W     = ksp->work[4];
  UOLD  = ksp->work[5];
  VOLD  = ksp->work[6];
  WOLD  = ksp->work[7];
  WOOLD = ksp->work[8];

  PetscCall(PCGetOperators(ksp->pc, &Amat, NULL));

  ksp->its = 0;

  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, X, R)); /*     r <- b - A*x    */
    PetscCall(VecAYPX(R, -1.0, B));
  } else {
    PetscCall(VecCopy(B, R)); /*     r <- b (x is 0) */
  }
  PetscCall(KSP_PCApply(ksp, R, Z));  /*     z  <- B*r       */
  PetscCall(VecNorm(Z, NORM_2, &np)); /*   np <- ||z||        */
  KSPCheckNorm(ksp, np);
  PetscCall(VecDot(R, Z, &dp));
  KSPCheckDot(ksp, dp);

  if (PetscRealPart(dp) < minres->haptol && np > minres->haptol) {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Detected indefinite operator %g tolerance %g", (double)PetscRealPart(dp), (double)minres->haptol);
    PetscCall(PetscInfo(ksp, "Detected indefinite operator %g tolerance %g\n", (double)PetscRealPart(dp), (double)minres->haptol));
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ksp->rnorm = 0.0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
  PetscCall(KSPMonitor(ksp, 0, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  beta = dp; /*  beta <- sqrt(r'*z)  */
  eta  = beta;
  PetscCall(VecAXPBY(V, 1.0 / beta, 0, R)); /* v <- r / beta */
  PetscCall(VecAXPBY(U, 1.0 / beta, 0, Z)); /* u <- z / beta */

  i = 0;
  do {
    ksp->its = i + 1;

    /*   Lanczos  */

    PetscCall(KSP_MatMult(ksp, Amat, U, R)); /*      r <- A*u   */
    PetscCall(VecDot(U, R, &alpha));         /*  alpha <- r'*u  */
    PetscCall(KSP_PCApply(ksp, R, Z));       /*      z <- B*r   */

    if (ksp->its > 1) {
      Vec         T[2];
      PetscScalar alphas[] = {-alpha, -beta};
      /*  r <- r - alpha v - beta v_old    */
      T[0] = V;
      T[1] = VOLD;
      PetscCall(VecMAXPY(R, 2, alphas, T));
      /*  z <- z - alpha u - beta u_old    */
      T[0] = U;
      T[1] = UOLD;
      PetscCall(VecMAXPY(Z, 2, alphas, T));
    } else {
      PetscCall(VecAXPY(R, -alpha, V)); /*  r <- r - alpha v     */
      PetscCall(VecAXPY(Z, -alpha, U)); /*  z <- z - alpha u     */
    }

    betaold = beta;

    PetscCall(VecDot(R, Z, &dp));
    KSPCheckDot(ksp, dp);
    dp   = PetscAbsScalar(dp);
    beta = PetscSqrtScalar(dp); /*  beta <- sqrt(r'*z)   */

    /*    QR factorisation    */

    coold = cold;
    cold  = c;
    soold = sold;
    sold  = s;

    rho0 = cold * alpha - coold * sold * betaold;
    rho1 = PetscSqrtScalar(rho0 * rho0 + beta * beta);
    rho2 = sold * alpha + coold * cold * betaold;
    rho3 = soold * betaold;

    /* Stop if negative curvature is detected */
    if (ksp->converged_neg_curve && PetscRealPart(cold * rho0) <= 0.0) {
      PetscCall(PetscInfo(ksp, "Detected negative curvature c_nm1=%g, gbar %g\n", (double)PetscRealPart(cold), -(double)PetscRealPart(rho0)));
      ksp->reason = KSP_CONVERGED_NEG_CURVE;
      break;
    }

    /*     Givens rotation    */

    c = rho0 / rho1;
    s = beta / rho1;

    /* Update */
    /*  w_oold <- w_old */
    /*  w_old  <- w     */
    KSPMinresSwap3(WOOLD, WOLD, W);

    /* w <- (u - rho2 w_old - rho3 w_oold)/rho1 */
    PetscCall(VecAXPBY(W, 1.0 / rho1, 0.0, U));
    if (ksp->its > 1) {
      Vec         T[]      = {WOLD, WOOLD};
      PetscScalar alphas[] = {-rho2 / rho1, -rho3 / rho1};
      PetscInt    nv       = (ksp->its == 2 ? 1 : 2);

      PetscCall(VecMAXPY(W, nv, alphas, T));
    }

    ceta = c * eta;
    PetscCall(VecAXPY(X, ceta, W)); /*  x <- x + c eta w     */

    /*
        when dp is really small we have either convergence or an indefinite operator so compute true
        residual norm to check for convergence
    */
    if (PetscRealPart(dp) < minres->haptol) {
      PetscCall(PetscInfo(ksp, "Possible indefinite operator %g tolerance %g\n", (double)PetscRealPart(dp), (double)minres->haptol));
      PetscCall(KSP_MatMult(ksp, Amat, X, VOLD));
      PetscCall(VecAXPY(VOLD, none, B));
      PetscCall(VecNorm(VOLD, NORM_2, &np));
      KSPCheckNorm(ksp, np);
    } else {
      /* otherwise compute new residual norm via recurrence relation */
      np *= PetscAbsScalar(s);
    }

    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = np;
    PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
    PetscCall(KSPMonitor(ksp, i + 1, ksp->rnorm));
    PetscCall((*ksp->converged)(ksp, i + 1, ksp->rnorm, &ksp->reason, ksp->cnvP)); /* test for convergence */
    if (ksp->reason) break;

    if (PetscRealPart(dp) < minres->haptol) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Detected indefinite operator %g tolerance %g", (double)PetscRealPart(dp), (double)minres->haptol);
      PetscCall(PetscInfo(ksp, "Detected indefinite operator %g tolerance %g\n", (double)PetscRealPart(dp), (double)minres->haptol));
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    eta = -s * eta;
    KSPMinresSwap3(VOLD, V, R);
    KSPMinresSwap3(UOLD, U, Z);
    PetscCall(VecScale(V, 1.0 / beta)); /* v <- r / beta */
    PetscCall(VecScale(U, 1.0 / beta)); /* u <- z / beta */

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_MINRES(KSP ksp)
{
  KSP_MINRES *minres = (KSP_MINRES *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&minres->viewer));
  PetscCall(PetscFree(ksp->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMINRESSetRadius_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMINRESSetUseQLP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMINRESGetUseQLP_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMINRESSetUseQLP_MINRES(KSP ksp, PetscBool qlp)
{
  KSP_MINRES *minres = (KSP_MINRES *)ksp->data;

  PetscFunctionBegin;
  minres->qlp = qlp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMINRESSetRadius_MINRES(KSP ksp, PetscReal radius)
{
  KSP_MINRES *minres = (KSP_MINRES *)ksp->data;

  PetscFunctionBegin;
  minres->maxxnorm = radius;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMINRESGetUseQLP_MINRES(KSP ksp, PetscBool *qlp)
{
  KSP_MINRES *minres = (KSP_MINRES *)ksp->data;

  PetscFunctionBegin;
  *qlp = minres->qlp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_MINRES(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_MINRES *minres = (KSP_MINRES *)ksp->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP MINRES options");
  { /* Allow comparing with the old code (to be removed in a few releases) */
    PetscBool flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-ksp_minres_old", "Use old implementation (to be removed)", "None", flg, &flg, NULL));
    if (flg) ksp->ops->solve = KSPSolve_MINRES_OLD;
    else ksp->ops->solve = KSPSolve_MINRES;
  }
  PetscCall(PetscOptionsBool("-ksp_minres_qlp", "Solve with QLP variant", "KSPMINRESSetUseQLP", minres->qlp, &minres->qlp, NULL));
  PetscCall(PetscOptionsReal("-ksp_minres_radius", "Maximum allowed norm of solution", "KSPMINRESSetRadius", minres->maxxnorm, &minres->maxxnorm, NULL));
  PetscCall(PetscOptionsReal("-ksp_minres_trancond", "Threshold on condition number to dynamically switch to QLP", "None", minres->TranCond, &minres->TranCond, NULL));
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp), PetscOptionsObject->options, PetscOptionsObject->prefix, "-ksp_minres_monitor", &minres->viewer, &minres->viewer_fmt, &minres->monitor));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPMINRESSetUseQLP - Use the QLP variant of the algorithm.

    Logically Collective

    Input Parameters:
+   ksp - the iterative context
-   qlp - a Boolean indicating if the QLP variant should be used

    Level: beginner

    Note:
    By default, the QLP variant is not used.

.seealso: [](chapter_ksp), `KSP`, `KSPMINRES`, `KSPMINRESGetUseQLP()`
@*/
PetscErrorCode KSPMINRESSetUseQLP(KSP ksp, PetscBool qlp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveBool(ksp, qlp, 2);
  PetscTryMethod(ksp, "KSPMINRESSetUseQLP_C", (KSP, PetscBool), (ksp, qlp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPMINRESSetRadius - Set the maximum solution norm allowed.

    Logically Collective

    Input Parameters:
+   ksp - the iterative context
-   radius - the value

    Level: beginner

    Options Database Key:
.   -ksp_minres_radius <real> - maximum allowed solution norm

.seealso: [](chapter_ksp), `KSP`, `KSPMINRES`, `KSPMINRESSetUseQLP()`
@*/
PetscErrorCode KSPMINRESSetRadius(KSP ksp, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, radius, 2);
  PetscTryMethod(ksp, "KSPMINRESSetRadius_C", (KSP, PetscReal), (ksp, radius));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPMINRESGetUseQLP - Get the flag for the QLP variant.

    Logically Collective

    Input Parameter:
.   ksp - the iterative context

    Output Parameter:
.   qlp - a Boolean indicating if the QLP variant is used

    Level: beginner

.seealso: [](chapter_ksp), `KSP`, `KSPMINRES`, `KSPMINRESSetUseQLP()`
@*/
PetscErrorCode KSPMINRESGetUseQLP(KSP ksp, PetscBool *qlp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidBoolPointer(qlp, 2);
  PetscUseMethod(ksp, "KSPMINRESGetUseQLP_C", (KSP, PetscBool *), (ksp, qlp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPMINRES - This code implements the MINRES (Minimum Residual) method and its QLP variant.

   Options Database Keys:
+   -ksp_minres_qlp <bool> - activates QLP code
.   -ksp_minres_radius <real> - maximum allowed solution norm
.   -ksp_minres_trancond <real> - threshold on condition number to dynamically switch to QLP iterations when QLP has been activated
-   -ksp_minres_monitor - monitors convergence quantities

   Level: beginner

   Notes:
   The operator and the preconditioner must be symmetric and the preconditioner must be positive definite for this method.

   Supports only left preconditioning.

   Reference:
+ * - Paige & Saunders, Solution of sparse indefinite systems of linear equations, SIAM J. Numer. Anal. 12, 1975.
- * - S.-C. T. Choi, C. C. Paige and M. A. Saunders. MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems, SIAM J. Sci. Comput. 33:4, 2011.

   Original MINRES code contributed by: Robert Scheichl: maprs@maths.bath.ac.uk
   QLP variant adapted from: https://stanford.edu/group/SOL/software/minresqlp/minresqlp-matlab/CPS11.zip

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPCG`, `KSPCR`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_MINRES(KSP ksp)
{
  KSP_MINRES *minres;

  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));
  PetscCall(PetscNew(&minres));

  /* this parameter is arbitrary and belongs to the old implementation; but e-50 didn't work for __float128 in one example */
#if defined(PETSC_USE_REAL___FLOAT128)
  minres->haptol = 1.e-100;
#elif defined(PETSC_USE_REAL_SINGLE)
  minres->haptol = 1.e-25;
#else
  minres->haptol = 1.e-50;
#endif
  /* those are set as 1.e7 in the matlab code -> use 1.0/sqrt(eps) to support single precision */
  minres->maxxnorm = 1.0 / PETSC_SQRT_MACHINE_EPSILON;
  minres->TranCond = 1.0 / PETSC_SQRT_MACHINE_EPSILON;

  ksp->data = (void *)minres;

  ksp->ops->setup          = KSPSetUp_MINRES;
  ksp->ops->solve          = KSPSolve_MINRES;
  ksp->ops->destroy        = KSPDestroy_MINRES;
  ksp->ops->setfromoptions = KSPSetFromOptions_MINRES;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMINRESSetRadius_C", KSPMINRESSetRadius_MINRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMINRESSetUseQLP_C", KSPMINRESSetUseQLP_MINRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMINRESGetUseQLP_C", KSPMINRESGetUseQLP_MINRES));
  PetscFunctionReturn(PETSC_SUCCESS);
}
