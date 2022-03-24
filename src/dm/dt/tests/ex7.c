static char help[] = "Test the PetscDTAltV interface for k-forms (alternating k-linear maps).\n\n";

#include <petscviewer.h>
#include <petscdt.h>

static PetscErrorCode CheckPullback(PetscInt N, PetscInt M, const PetscReal *L, PetscInt k,
                                    const PetscReal *w, PetscReal *x, PetscBool verbose, PetscViewer viewer)
{
  PetscInt        Nk, Mk, i, j, l;
  PetscReal       *Lstarw, *Lx, *Lstar, *Lstarwcheck, wLx, Lstarwx;
  PetscReal       diff, diffMat, normMat;
  PetscReal       *walloc = NULL;
  const PetscReal *ww = NULL;
  PetscBool       negative = (PetscBool) (k < 0);

  PetscFunctionBegin;
  k = PetscAbsInt(k);
  CHKERRQ(PetscDTBinomialInt(N, k, &Nk));
  CHKERRQ(PetscDTBinomialInt(M, k, &Mk));
  if (negative) {
    CHKERRQ(PetscMalloc1(Mk, &walloc));
    CHKERRQ(PetscDTAltVStar(M, M - k, 1, w, walloc));
    ww = walloc;
  } else {
    ww = w;
  }
  CHKERRQ(PetscMalloc2(Nk, &Lstarw, (M*k), &Lx));
  CHKERRQ(PetscMalloc2(Nk * Mk, &Lstar, Nk, &Lstarwcheck));
  CHKERRQ(PetscDTAltVPullback(N, M, L, negative ? -k : k, w, Lstarw));
  CHKERRQ(PetscDTAltVPullbackMatrix(N, M, L, negative ? -k : k, Lstar));
  if (negative) {
    PetscReal *sLsw;

    CHKERRQ(PetscMalloc1(Nk, &sLsw));
    CHKERRQ(PetscDTAltVStar(N, N - k, 1, Lstarw, sLsw));
    CHKERRQ(PetscDTAltVApply(N, k, sLsw, x, &Lstarwx));
    CHKERRQ(PetscFree(sLsw));
  } else {
    CHKERRQ(PetscDTAltVApply(N, k, Lstarw, x, &Lstarwx));
  }
  for (l = 0; l < k; l++) {
    for (i = 0; i < M; i++) {
      PetscReal sum = 0.;

      for (j = 0; j < N; j++) sum += L[i * N + j] * x[l * N + j];
      Lx[l * M + i] = sum;
    }
  }
  diffMat = 0.;
  normMat = 0.;
  for (i = 0; i < Nk; i++) {
    PetscReal sum = 0.;
    for (j = 0; j < Mk; j++) {
      sum += Lstar[i * Mk + j] * w[j];
    }
    Lstarwcheck[i] = sum;
    diffMat += PetscSqr(PetscAbsReal(Lstarwcheck[i] - Lstarw[i]));
    normMat += PetscSqr(Lstarwcheck[i]) +  PetscSqr(Lstarw[i]);
  }
  diffMat = PetscSqrtReal(diffMat);
  normMat = PetscSqrtReal(normMat);
  if (verbose) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "L:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    if (M*N > 0) CHKERRQ(PetscRealView(M*N, L, viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));

    CHKERRQ(PetscViewerASCIIPrintf(viewer, "L*:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    if (Nk * Mk > 0) CHKERRQ(PetscRealView(Nk * Mk, Lstar, viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));

    CHKERRQ(PetscViewerASCIIPrintf(viewer, "L*w:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    if (Nk > 0) CHKERRQ(PetscRealView(Nk, Lstarw, viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  CHKERRQ(PetscDTAltVApply(M, k, ww, Lx, &wLx));
  diff = PetscAbsReal(wLx - Lstarwx);
  PetscCheckFalse(diff > 10. * PETSC_SMALL * (PetscAbsReal(wLx) + PetscAbsReal(Lstarwx)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "pullback check: pullback does not commute with application: w(Lx)(%g) != (L* w)(x)(%g)", wLx, Lstarwx);
  PetscCheckFalse(diffMat > PETSC_SMALL * normMat,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "pullback check: pullback matrix does match matrix free result");
  CHKERRQ(PetscFree2(Lstar, Lstarwcheck));
  CHKERRQ(PetscFree2(Lstarw, Lx));
  CHKERRQ(PetscFree(walloc));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       i, numTests = 5, n[5] = {0, 1, 2, 3, 4};
  PetscBool      verbose = PETSC_FALSE;
  PetscRandom    rand;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for exterior algebra tests","none");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsIntArray("-N", "Up to 5 vector space dimensions to test","ex7.c",n,&numTests,NULL));
  CHKERRQ(PetscOptionsBool("-verbose", "Verbose test output","ex7.c",verbose,&verbose,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rand));
  CHKERRQ(PetscRandomSetInterval(rand, -1., 1.));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  if (!numTests) numTests = 5;
  viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  for (i = 0; i < numTests; i++) {
    PetscInt       k, N = n[i];

    if (verbose) CHKERRQ(PetscViewerASCIIPrintf(viewer, "N = %D:\n", N));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));

    if (verbose) {
      PetscInt *perm;
      PetscInt fac = 1;

      CHKERRQ(PetscMalloc1(N, &perm));

      for (k = 1; k <= N; k++) fac *= k;
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "Permutations of %D:\n", N));
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      for (k = 0; k < fac; k++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  j, kCheck;

        CHKERRQ(PetscDTEnumPerm(N, k, perm, &isOdd));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "%D:", k));
        for (j = 0; j < N; j++) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, " %D", perm[j]));
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even"));
        CHKERRQ(PetscDTPermIndex(N, perm, &kCheck, &isOddCheck));
        PetscCheckFalse(kCheck != k || isOddCheck != isOdd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTEnumPerm / PetscDTPermIndex mismatch for (%D, %D)", N, k);
      }
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
      CHKERRQ(PetscFree(perm));
    }
    for (k = 0; k <= N; k++) {
      PetscInt   j, Nk, M;
      PetscReal *w, *v, wv;
      PetscInt  *subset;

      CHKERRQ(PetscDTBinomialInt(N, k, &Nk));
      if (verbose) CHKERRQ(PetscViewerASCIIPrintf(viewer, "k = %D:\n", k));
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      if (verbose) CHKERRQ(PetscViewerASCIIPrintf(viewer, "(%D choose %D): %D\n", N, k, Nk));

      /* Test subset and complement enumeration */
      CHKERRQ(PetscMalloc1(N, &subset));
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      for (j = 0; j < Nk; j++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  jCheck, kCheck;

        CHKERRQ(PetscDTEnumSplit(N, k, j, subset, &isOdd));
        CHKERRQ(PetscDTPermIndex(N, subset, &kCheck, &isOddCheck));
        PetscCheckFalse(isOddCheck != isOdd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTEnumSplit sign does not mmatch PetscDTPermIndex sign");
        if (verbose) {
          PetscInt l;

          CHKERRQ(PetscViewerASCIIPrintf(viewer, "subset %D:", j));
          for (l = 0; l < k; l++) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]));
          }
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, " |"));
          for (l = k; l < N; l++) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]));
          }
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even"));
        }
        CHKERRQ(PetscDTSubsetIndex(N, k, subset, &jCheck));
        PetscCheckFalse(jCheck != j,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "jCheck (%D) != j (%D)", jCheck, j);
      }
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
      CHKERRQ(PetscFree(subset));

      /* Make a random k form */
      CHKERRQ(PetscMalloc1(Nk, &w));
      for (j = 0; j < Nk; j++) CHKERRQ(PetscRandomGetValueReal(rand, &w[j]));
      /* Make a set of random vectors */
      CHKERRQ(PetscMalloc1(N*k, &v));
      for (j = 0; j < N*k; j++) CHKERRQ(PetscRandomGetValueReal(rand, &v[j]));

      CHKERRQ(PetscDTAltVApply(N, k, w, v, &wv));

      if (verbose) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "w:\n"));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        if (Nk) CHKERRQ(PetscRealView(Nk, w, viewer));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "v:\n"));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        if (N*k > 0) CHKERRQ(PetscRealView(N*k, v, viewer));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "w(v): %g\n", (double) wv));
      }

      /* sanity checks */
      if (k == 1) { /* 1-forms are functionals (dot products) */
        PetscInt  l;
        PetscReal wvcheck = 0.;
        PetscReal diff;

        for (l = 0; l < N; l++) wvcheck += w[l] * v[l];
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));
        PetscCheckFalse(diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "1-form / dot product equivalence: wvcheck (%g) != wv (%g)", (double) wvcheck, (double) wv);
      }
      if (k == N && N < 5) { /* n-forms are scaled determinants */
        PetscReal det, wvcheck, diff;

        switch (k) {
        case 0:
          det = 1.;
          break;
        case 1:
          det = v[0];
          break;
        case 2:
          det = v[0] * v[3] - v[1] * v[2];
          break;
        case 3:
          det = v[0] * (v[4] * v[8] - v[5] * v[7]) +
                v[1] * (v[5] * v[6] - v[3] * v[8]) +
                v[2] * (v[3] * v[7] - v[4] * v[6]);
          break;
        case 4:
          det = v[0] * (v[5] * (v[10] * v[15] - v[11] * v[14]) +
                        v[6] * (v[11] * v[13] - v[ 9] * v[15]) +
                        v[7] * (v[ 9] * v[14] - v[10] * v[13])) -
                v[1] * (v[4] * (v[10] * v[15] - v[11] * v[14]) +
                        v[6] * (v[11] * v[12] - v[ 8] * v[15]) +
                        v[7] * (v[ 8] * v[14] - v[10] * v[12])) +
                v[2] * (v[4] * (v[ 9] * v[15] - v[11] * v[13]) +
                        v[5] * (v[11] * v[12] - v[ 8] * v[15]) +
                        v[7] * (v[ 8] * v[13] - v[ 9] * v[12])) -
                v[3] * (v[4] * (v[ 9] * v[14] - v[10] * v[13]) +
                        v[5] * (v[10] * v[12] - v[ 8] * v[14]) +
                        v[6] * (v[ 8] * v[13] - v[ 9] * v[12]));
          break;
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "invalid k");
        }
        wvcheck = det * w[0];
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));
        PetscCheckFalse(diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "n-form / determinant equivalence: wvcheck (%g) != wv (%g) %g", (double) wvcheck, (double) wv, (double) diff);
      }
      if (k > 0) { /* k-forms are linear in each component */
        PetscReal alpha;
        PetscReal *x, *axv, wx, waxv, waxvcheck;
        PetscReal diff;
        PetscReal rj;
        PetscInt  l;

        CHKERRQ(PetscMalloc2(N * k, &x, N * k, &axv));
        CHKERRQ(PetscRandomGetValueReal(rand, &alpha));
        CHKERRQ(PetscRandomSetInterval(rand, 0, k));
        CHKERRQ(PetscRandomGetValueReal(rand, &rj));
        j = (PetscInt) rj;
        CHKERRQ(PetscRandomSetInterval(rand, -1., 1.));
        for (l = 0; l < N*k; l++) x[l] = v[l];
        for (l = 0; l < N*k; l++) axv[l] = v[l];
        for (l = 0; l < N; l++) {
          PetscReal val;

          CHKERRQ(PetscRandomGetValueReal(rand, &val));
          x[j * N + l] = val;
          axv[j * N + l] += alpha * val;
        }
        CHKERRQ(PetscDTAltVApply(N, k, w, x, &wx));
        CHKERRQ(PetscDTAltVApply(N, k, w, axv, &waxv));
        waxvcheck = alpha * wx + wv;
        diff = waxv - waxvcheck;
        PetscCheckFalse(PetscAbsReal(diff) > 10. * PETSC_SMALL * (PetscAbsReal(waxv) + PetscAbsReal(waxvcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "linearity check: component %D, waxvcheck (%g) != waxv (%g)", j, (double) waxvcheck, (double) waxv);
        CHKERRQ(PetscFree2(x,axv));
      }
      if (k > 1) { /* k-forms are antisymmetric */
        PetscReal rj, rl, *swapv, wswapv, diff;
        PetscInt  l, m;

        CHKERRQ(PetscRandomSetInterval(rand, 0, k));
        CHKERRQ(PetscRandomGetValueReal(rand, &rj));
        j = (PetscInt) rj;
        l = j;
        while (l == j) {
          CHKERRQ(PetscRandomGetValueReal(rand, &rl));
          l = (PetscInt) rl;
        }
        CHKERRQ(PetscRandomSetInterval(rand, -1., 1.));
        CHKERRQ(PetscMalloc1(N * k, &swapv));
        for (m = 0; m < N * k; m++) swapv[m] = v[m];
        for (m = 0; m < N; m++) {
          swapv[j * N + m] = v[l * N + m];
          swapv[l * N + m] = v[j * N + m];
        }
        CHKERRQ(PetscDTAltVApply(N, k, w, swapv, &wswapv));
        diff = PetscAbsReal(wswapv + wv);
        PetscCheckFalse(diff > PETSC_SMALL * (PetscAbsReal(wswapv) + PetscAbsReal(wv)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "antisymmetry check: components %D & %D, wswapv (%g) != -wv (%g)", j, l, (double) wswapv, (double) wv);
        CHKERRQ(PetscFree(swapv));
      }
      for (j = 0; j <= k && j + k <= N; j++) { /* wedge product */
        PetscInt   Nj, Njk, l, JKj;
        PetscReal *u, *uWw, *uWwcheck, *uWwmat, *x, *xsplit, uWwx, uWwxcheck, diff, norm;
        PetscInt  *split;

        if (verbose) CHKERRQ(PetscViewerASCIIPrintf(viewer, "wedge j = %D:\n", j));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        CHKERRQ(PetscDTBinomialInt(N, j,   &Nj));
        CHKERRQ(PetscDTBinomialInt(N, j+k, &Njk));
        CHKERRQ(PetscMalloc4(Nj, &u, Njk, &uWw, N*(j+k), &x, N*(j+k), &xsplit));
        CHKERRQ(PetscMalloc1(j+k,&split));
        for (l = 0; l < Nj; l++) CHKERRQ(PetscRandomGetValueReal(rand, &u[l]));
        for (l = 0; l < N*(j+k); l++) CHKERRQ(PetscRandomGetValueReal(rand, &x[l]));
        CHKERRQ(PetscDTAltVWedge(N, j, k, u, w, uWw));
        CHKERRQ(PetscDTAltVApply(N, j+k, uWw, x, &uWwx));
        if (verbose) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "u:\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (Nj) CHKERRQ(PetscRealView(Nj, u, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "u wedge w:\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (Njk) CHKERRQ(PetscRealView(Njk, uWw, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "x:\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (N*(j+k) > 0) CHKERRQ(PetscRealView(N*(j+k), x, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "u wedge w(x): %g\n", (double) uWwx));
        }
        /* verify wedge formula */
        uWwxcheck = 0.;
        CHKERRQ(PetscDTBinomialInt(j+k, j, &JKj));
        for (l = 0; l < JKj; l++) {
          PetscBool isOdd;
          PetscReal ux, wx;
          PetscInt  m, p;

          CHKERRQ(PetscDTEnumSplit(j+k, j, l, split, &isOdd));
          for (m = 0; m < j+k; m++) {for (p = 0; p < N; p++) {xsplit[m * N + p] = x[split[m] * N + p];}}
          CHKERRQ(PetscDTAltVApply(N, j, u, xsplit, &ux));
          CHKERRQ(PetscDTAltVApply(N, k, w, &xsplit[j*N], &wx));
          uWwxcheck += isOdd ? -(ux * wx) : (ux * wx);
        }
        diff = PetscAbsReal(uWwx - uWwxcheck);
        PetscCheckFalse(diff > 10. * PETSC_SMALL * (PetscAbsReal(uWwx) + PetscAbsReal(uWwxcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wedge check: forms %D & %D, uWwxcheck (%g) != uWwx (%g)", j, k, (double) uWwxcheck, (double) uWwx);
        CHKERRQ(PetscFree(split));
        CHKERRQ(PetscMalloc2(Nk * Njk, &uWwmat, Njk, &uWwcheck));
        CHKERRQ(PetscDTAltVWedgeMatrix(N, j, k, u, uWwmat));
        if (verbose) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "(u wedge):\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if ((Nk * Njk) > 0) CHKERRQ(PetscRealView(Nk * Njk, uWwmat, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        diff = 0.;
        norm = 0.;
        for (l = 0; l < Njk; l++) {
          PetscInt  m;
          PetscReal sum = 0.;

          for (m = 0; m < Nk; m++) sum += uWwmat[l * Nk + m] * w[m];
          uWwcheck[l] = sum;
          diff += PetscSqr(uWwcheck[l] - uWw[l]);
          norm += PetscSqr(uWwcheck[l]) + PetscSqr(uWw[l]);
        }
        diff = PetscSqrtReal(diff);
        norm = PetscSqrtReal(norm);
        PetscCheckFalse(diff > PETSC_SMALL * norm,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wedge matrix check: wedge matrix application does not match wedge direct application");
        CHKERRQ(PetscFree2(uWwmat, uWwcheck));
        CHKERRQ(PetscFree4(u, uWw, x, xsplit));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
      }
      for (M = PetscMax(1,k); M <= N; M++) { /* pullback */
        PetscReal   *L, *u, *x;
        PetscInt     Mk, l;

        CHKERRQ(PetscDTBinomialInt(M, k, &Mk));
        CHKERRQ(PetscMalloc3(M*N, &L, Mk, &u, M*k, &x));
        for (l = 0; l < M*N; l++) CHKERRQ(PetscRandomGetValueReal(rand, &L[l]));
        for (l = 0; l < Mk; l++) CHKERRQ(PetscRandomGetValueReal(rand, &u[l]));
        for (l = 0; l < M*k; l++) CHKERRQ(PetscRandomGetValueReal(rand, &x[l]));
        if (verbose) CHKERRQ(PetscViewerASCIIPrintf(viewer, "pullback M = %D:\n", M));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        CHKERRQ(CheckPullback(M, N, L, k, w, x, verbose, viewer));
        if (M != N) CHKERRQ(CheckPullback(N, M, L, k, u, v, PETSC_FALSE, viewer));
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
        if ((k % N) && (N > 1)) {
          if (verbose) CHKERRQ(PetscViewerASCIIPrintf(viewer, "negative pullback M = %D:\n", M));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          CHKERRQ(CheckPullback(M, N, L, -k, w, x, verbose, viewer));
          if (M != N) CHKERRQ(CheckPullback(N, M, L, -k, u, v, PETSC_FALSE, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        CHKERRQ(PetscFree3(L, u, x));
      }
      if (k > 0) { /* Interior */
        PetscInt    Nkm, l, m;
        PetscReal  *wIntv0, *wIntv0check, wvcheck, diff, diffMat, normMat;
        PetscReal  *intv0mat, *matcheck;
        PetscInt  (*indices)[3];

        CHKERRQ(PetscDTBinomialInt(N, k-1, &Nkm));
        CHKERRQ(PetscMalloc5(Nkm, &wIntv0, Nkm, &wIntv0check, Nk * Nkm, &intv0mat, Nk * Nkm, &matcheck, Nk * k, &indices));
        CHKERRQ(PetscDTAltVInterior(N, k, w, v, wIntv0));
        CHKERRQ(PetscDTAltVInteriorMatrix(N, k, v, intv0mat));
        CHKERRQ(PetscDTAltVInteriorPattern(N, k, indices));
        if (verbose) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "interior product matrix pattern:\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          for (l = 0; l < Nk * k; l++) {
            PetscInt row = indices[l][0];
            PetscInt col = indices[l][1];
            PetscInt x   = indices[l][2];

            CHKERRQ(PetscViewerASCIIPrintf(viewer,"intV[%D,%D] = %sV[%D]\n", row, col, x < 0 ? "-" : " ", x < 0 ? -(x + 1) : x));
          }
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        for (l = 0; l < Nkm * Nk; l++) matcheck[l] = 0.;
        for (l = 0; l < Nk * k; l++) {
          PetscInt row = indices[l][0];
          PetscInt col = indices[l][1];
          PetscInt x   = indices[l][2];

          if (x < 0) {
            matcheck[row * Nk + col] = -v[-(x+1)];
          } else {
            matcheck[row * Nk + col] = v[x];
          }
        }
        diffMat = 0.;
        normMat = 0.;
        for (l = 0; l < Nkm * Nk; l++) {
          diffMat += PetscSqr(PetscAbsReal(matcheck[l] - intv0mat[l]));
          normMat += PetscSqr(matcheck[l]) + PetscSqr(intv0mat[l]);
        }
        diffMat = PetscSqrtReal(diffMat);
        normMat = PetscSqrtReal(normMat);
        PetscCheckFalse(diffMat > PETSC_SMALL * normMat,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: matrix pattern does not match matrix");
        diffMat = 0.;
        normMat = 0.;
        for (l = 0; l < Nkm; l++) {
          PetscReal sum = 0.;

          for (m = 0; m < Nk; m++) sum += intv0mat[l * Nk + m] * w[m];
          wIntv0check[l] = sum;

          diffMat += PetscSqr(PetscAbsReal(wIntv0check[l] - wIntv0[l]));
          normMat += PetscSqr(wIntv0check[l]) + PetscSqr(wIntv0[l]);
        }
        diffMat = PetscSqrtReal(diffMat);
        normMat = PetscSqrtReal(normMat);
        PetscCheckFalse(diffMat > PETSC_SMALL * normMat,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: application does not match matrix");
        if (verbose) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "(w int v_0):\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (Nkm) CHKERRQ(PetscRealView(Nkm, wIntv0, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));

          CHKERRQ(PetscViewerASCIIPrintf(viewer, "(int v_0):\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (Nk * Nkm > 0) CHKERRQ(PetscRealView(Nk * Nkm, intv0mat, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        CHKERRQ(PetscDTAltVApply(N, k - 1, wIntv0, &v[N], &wvcheck));
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));
        PetscCheckFalse(diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: (w Int v0)(v_rem) (%g) != w(v) (%g)", (double) wvcheck, (double) wv);
        CHKERRQ(PetscFree5(wIntv0,wIntv0check,intv0mat,matcheck,indices));
      }
      if (k >= N - k) { /* Hodge star */
        PetscReal *u, *starw, *starstarw, wu, starwdotu;
        PetscReal diff, norm;
        PetscBool isOdd;
        PetscInt l;

        isOdd = (PetscBool) ((k * (N - k)) & 1);
        CHKERRQ(PetscMalloc3(Nk, &u, Nk, &starw, Nk, &starstarw));
        CHKERRQ(PetscDTAltVStar(N, k, 1, w, starw));
        CHKERRQ(PetscDTAltVStar(N, N-k, 1, starw, starstarw));
        if (verbose) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "star w:\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (Nk) CHKERRQ(PetscRealView(Nk, starw, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));

          CHKERRQ(PetscViewerASCIIPrintf(viewer, "star star w:\n"));
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          if (Nk) CHKERRQ(PetscRealView(Nk, starstarw, viewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        for (l = 0; l < Nk; l++) CHKERRQ(PetscRandomGetValueReal(rand,&u[l]));
        CHKERRQ(PetscDTAltVWedge(N, k, N - k, w, u, &wu));
        starwdotu = 0.;
        for (l = 0; l < Nk; l++) starwdotu += starw[l] * u[l];
        diff = PetscAbsReal(wu - starwdotu);
        PetscCheckFalse(diff > PETSC_SMALL * (PetscAbsReal(wu) + PetscAbsReal(starwdotu)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Hodge star check: (star w, u) (%g) != (w wedge u) (%g)", (double) starwdotu, (double) wu);

        diff = 0.;
        norm = 0.;
        for (l = 0; l < Nk; l++) {
          diff += PetscSqr(w[l] - (isOdd ? -starstarw[l] : starstarw[l]));
          norm += PetscSqr(w[l]) + PetscSqr(starstarw[l]);
        }
        diff = PetscSqrtReal(diff);
        norm = PetscSqrtReal(norm);
        PetscCheckFalse(diff > PETSC_SMALL * norm,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Hodge star check: star(star(w)) != (-1)^(N*(N-k)) w");
        CHKERRQ(PetscFree3(u, starw, starstarw));
      }
      CHKERRQ(PetscFree(v));
      CHKERRQ(PetscFree(w));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 1234
    args: -verbose
  test:
    suffix: 56
    args: -N 5,6
TEST*/
