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
  PetscCall(PetscDTBinomialInt(N, k, &Nk));
  PetscCall(PetscDTBinomialInt(M, k, &Mk));
  if (negative) {
    PetscCall(PetscMalloc1(Mk, &walloc));
    PetscCall(PetscDTAltVStar(M, M - k, 1, w, walloc));
    ww = walloc;
  } else {
    ww = w;
  }
  PetscCall(PetscMalloc2(Nk, &Lstarw, (M*k), &Lx));
  PetscCall(PetscMalloc2(Nk * Mk, &Lstar, Nk, &Lstarwcheck));
  PetscCall(PetscDTAltVPullback(N, M, L, negative ? -k : k, w, Lstarw));
  PetscCall(PetscDTAltVPullbackMatrix(N, M, L, negative ? -k : k, Lstar));
  if (negative) {
    PetscReal *sLsw;

    PetscCall(PetscMalloc1(Nk, &sLsw));
    PetscCall(PetscDTAltVStar(N, N - k, 1, Lstarw, sLsw));
    PetscCall(PetscDTAltVApply(N, k, sLsw, x, &Lstarwx));
    PetscCall(PetscFree(sLsw));
  } else {
    PetscCall(PetscDTAltVApply(N, k, Lstarw, x, &Lstarwx));
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "L:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (M*N > 0) PetscCall(PetscRealView(M*N, L, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));

    PetscCall(PetscViewerASCIIPrintf(viewer, "L*:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (Nk * Mk > 0) PetscCall(PetscRealView(Nk * Mk, Lstar, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));

    PetscCall(PetscViewerASCIIPrintf(viewer, "L*w:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (Nk > 0) PetscCall(PetscRealView(Nk, Lstarw, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscCall(PetscDTAltVApply(M, k, ww, Lx, &wLx));
  diff = PetscAbsReal(wLx - Lstarwx);
  PetscCheckFalse(diff > 10. * PETSC_SMALL * (PetscAbsReal(wLx) + PetscAbsReal(Lstarwx)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "pullback check: pullback does not commute with application: w(Lx)(%g) != (L* w)(x)(%g)", wLx, Lstarwx);
  PetscCheckFalse(diffMat > PETSC_SMALL * normMat,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "pullback check: pullback matrix does match matrix free result");
  PetscCall(PetscFree2(Lstar, Lstarwcheck));
  PetscCall(PetscFree2(Lstarw, Lx));
  PetscCall(PetscFree(walloc));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       i, numTests = 5, n[5] = {0, 1, 2, 3, 4};
  PetscBool      verbose = PETSC_FALSE;
  PetscRandom    rand;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for exterior algebra tests","none");PetscCall(ierr);
  PetscCall(PetscOptionsIntArray("-N", "Up to 5 vector space dimensions to test","ex7.c",n,&numTests,NULL));
  PetscCall(PetscOptionsBool("-verbose", "Verbose test output","ex7.c",verbose,&verbose,NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1., 1.));
  PetscCall(PetscRandomSetFromOptions(rand));
  if (!numTests) numTests = 5;
  viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  for (i = 0; i < numTests; i++) {
    PetscInt       k, N = n[i];

    if (verbose) PetscCall(PetscViewerASCIIPrintf(viewer, "N = %D:\n", N));
    PetscCall(PetscViewerASCIIPushTab(viewer));

    if (verbose) {
      PetscInt *perm;
      PetscInt fac = 1;

      PetscCall(PetscMalloc1(N, &perm));

      for (k = 1; k <= N; k++) fac *= k;
      PetscCall(PetscViewerASCIIPrintf(viewer, "Permutations of %D:\n", N));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      for (k = 0; k < fac; k++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  j, kCheck;

        PetscCall(PetscDTEnumPerm(N, k, perm, &isOdd));
        PetscCall(PetscViewerASCIIPrintf(viewer, "%D:", k));
        for (j = 0; j < N; j++) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %D", perm[j]));
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even"));
        PetscCall(PetscDTPermIndex(N, perm, &kCheck, &isOddCheck));
        PetscCheckFalse(kCheck != k || isOddCheck != isOdd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTEnumPerm / PetscDTPermIndex mismatch for (%D, %D)", N, k);
      }
      PetscCall(PetscViewerASCIIPopTab(viewer));
      PetscCall(PetscFree(perm));
    }
    for (k = 0; k <= N; k++) {
      PetscInt   j, Nk, M;
      PetscReal *w, *v, wv;
      PetscInt  *subset;

      PetscCall(PetscDTBinomialInt(N, k, &Nk));
      if (verbose) PetscCall(PetscViewerASCIIPrintf(viewer, "k = %D:\n", k));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      if (verbose) PetscCall(PetscViewerASCIIPrintf(viewer, "(%D choose %D): %D\n", N, k, Nk));

      /* Test subset and complement enumeration */
      PetscCall(PetscMalloc1(N, &subset));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      for (j = 0; j < Nk; j++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  jCheck, kCheck;

        PetscCall(PetscDTEnumSplit(N, k, j, subset, &isOdd));
        PetscCall(PetscDTPermIndex(N, subset, &kCheck, &isOddCheck));
        PetscCheck(isOddCheck == isOdd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTEnumSplit sign does not mmatch PetscDTPermIndex sign");
        if (verbose) {
          PetscInt l;

          PetscCall(PetscViewerASCIIPrintf(viewer, "subset %D:", j));
          for (l = 0; l < k; l++) {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]));
          }
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, " |"));
          for (l = k; l < N; l++) {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]));
          }
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even"));
        }
        PetscCall(PetscDTSubsetIndex(N, k, subset, &jCheck));
        PetscCheck(jCheck == j,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "jCheck (%D) != j (%D)", jCheck, j);
      }
      PetscCall(PetscViewerASCIIPopTab(viewer));
      PetscCall(PetscFree(subset));

      /* Make a random k form */
      PetscCall(PetscMalloc1(Nk, &w));
      for (j = 0; j < Nk; j++) PetscCall(PetscRandomGetValueReal(rand, &w[j]));
      /* Make a set of random vectors */
      PetscCall(PetscMalloc1(N*k, &v));
      for (j = 0; j < N*k; j++) PetscCall(PetscRandomGetValueReal(rand, &v[j]));

      PetscCall(PetscDTAltVApply(N, k, w, v, &wv));

      if (verbose) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "w:\n"));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        if (Nk) PetscCall(PetscRealView(Nk, w, viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "v:\n"));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        if (N*k > 0) PetscCall(PetscRealView(N*k, v, viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "w(v): %g\n", (double) wv));
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

        PetscCall(PetscMalloc2(N * k, &x, N * k, &axv));
        PetscCall(PetscRandomGetValueReal(rand, &alpha));
        PetscCall(PetscRandomSetInterval(rand, 0, k));
        PetscCall(PetscRandomGetValueReal(rand, &rj));
        j = (PetscInt) rj;
        PetscCall(PetscRandomSetInterval(rand, -1., 1.));
        for (l = 0; l < N*k; l++) x[l] = v[l];
        for (l = 0; l < N*k; l++) axv[l] = v[l];
        for (l = 0; l < N; l++) {
          PetscReal val;

          PetscCall(PetscRandomGetValueReal(rand, &val));
          x[j * N + l] = val;
          axv[j * N + l] += alpha * val;
        }
        PetscCall(PetscDTAltVApply(N, k, w, x, &wx));
        PetscCall(PetscDTAltVApply(N, k, w, axv, &waxv));
        waxvcheck = alpha * wx + wv;
        diff = waxv - waxvcheck;
        PetscCheckFalse(PetscAbsReal(diff) > 10. * PETSC_SMALL * (PetscAbsReal(waxv) + PetscAbsReal(waxvcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "linearity check: component %D, waxvcheck (%g) != waxv (%g)", j, (double) waxvcheck, (double) waxv);
        PetscCall(PetscFree2(x,axv));
      }
      if (k > 1) { /* k-forms are antisymmetric */
        PetscReal rj, rl, *swapv, wswapv, diff;
        PetscInt  l, m;

        PetscCall(PetscRandomSetInterval(rand, 0, k));
        PetscCall(PetscRandomGetValueReal(rand, &rj));
        j = (PetscInt) rj;
        l = j;
        while (l == j) {
          PetscCall(PetscRandomGetValueReal(rand, &rl));
          l = (PetscInt) rl;
        }
        PetscCall(PetscRandomSetInterval(rand, -1., 1.));
        PetscCall(PetscMalloc1(N * k, &swapv));
        for (m = 0; m < N * k; m++) swapv[m] = v[m];
        for (m = 0; m < N; m++) {
          swapv[j * N + m] = v[l * N + m];
          swapv[l * N + m] = v[j * N + m];
        }
        PetscCall(PetscDTAltVApply(N, k, w, swapv, &wswapv));
        diff = PetscAbsReal(wswapv + wv);
        PetscCheckFalse(diff > PETSC_SMALL * (PetscAbsReal(wswapv) + PetscAbsReal(wv)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "antisymmetry check: components %D & %D, wswapv (%g) != -wv (%g)", j, l, (double) wswapv, (double) wv);
        PetscCall(PetscFree(swapv));
      }
      for (j = 0; j <= k && j + k <= N; j++) { /* wedge product */
        PetscInt   Nj, Njk, l, JKj;
        PetscReal *u, *uWw, *uWwcheck, *uWwmat, *x, *xsplit, uWwx, uWwxcheck, diff, norm;
        PetscInt  *split;

        if (verbose) PetscCall(PetscViewerASCIIPrintf(viewer, "wedge j = %D:\n", j));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscDTBinomialInt(N, j,   &Nj));
        PetscCall(PetscDTBinomialInt(N, j+k, &Njk));
        PetscCall(PetscMalloc4(Nj, &u, Njk, &uWw, N*(j+k), &x, N*(j+k), &xsplit));
        PetscCall(PetscMalloc1(j+k,&split));
        for (l = 0; l < Nj; l++) PetscCall(PetscRandomGetValueReal(rand, &u[l]));
        for (l = 0; l < N*(j+k); l++) PetscCall(PetscRandomGetValueReal(rand, &x[l]));
        PetscCall(PetscDTAltVWedge(N, j, k, u, w, uWw));
        PetscCall(PetscDTAltVApply(N, j+k, uWw, x, &uWwx));
        if (verbose) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "u:\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (Nj) PetscCall(PetscRealView(Nj, u, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
          PetscCall(PetscViewerASCIIPrintf(viewer, "u wedge w:\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (Njk) PetscCall(PetscRealView(Njk, uWw, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
          PetscCall(PetscViewerASCIIPrintf(viewer, "x:\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (N*(j+k) > 0) PetscCall(PetscRealView(N*(j+k), x, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
          PetscCall(PetscViewerASCIIPrintf(viewer, "u wedge w(x): %g\n", (double) uWwx));
        }
        /* verify wedge formula */
        uWwxcheck = 0.;
        PetscCall(PetscDTBinomialInt(j+k, j, &JKj));
        for (l = 0; l < JKj; l++) {
          PetscBool isOdd;
          PetscReal ux, wx;
          PetscInt  m, p;

          PetscCall(PetscDTEnumSplit(j+k, j, l, split, &isOdd));
          for (m = 0; m < j+k; m++) {for (p = 0; p < N; p++) {xsplit[m * N + p] = x[split[m] * N + p];}}
          PetscCall(PetscDTAltVApply(N, j, u, xsplit, &ux));
          PetscCall(PetscDTAltVApply(N, k, w, &xsplit[j*N], &wx));
          uWwxcheck += isOdd ? -(ux * wx) : (ux * wx);
        }
        diff = PetscAbsReal(uWwx - uWwxcheck);
        PetscCheckFalse(diff > 10. * PETSC_SMALL * (PetscAbsReal(uWwx) + PetscAbsReal(uWwxcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wedge check: forms %D & %D, uWwxcheck (%g) != uWwx (%g)", j, k, (double) uWwxcheck, (double) uWwx);
        PetscCall(PetscFree(split));
        PetscCall(PetscMalloc2(Nk * Njk, &uWwmat, Njk, &uWwcheck));
        PetscCall(PetscDTAltVWedgeMatrix(N, j, k, u, uWwmat));
        if (verbose) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "(u wedge):\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if ((Nk * Njk) > 0) PetscCall(PetscRealView(Nk * Njk, uWwmat, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
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
        PetscCall(PetscFree2(uWwmat, uWwcheck));
        PetscCall(PetscFree4(u, uWw, x, xsplit));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
      for (M = PetscMax(1,k); M <= N; M++) { /* pullback */
        PetscReal   *L, *u, *x;
        PetscInt     Mk, l;

        PetscCall(PetscDTBinomialInt(M, k, &Mk));
        PetscCall(PetscMalloc3(M*N, &L, Mk, &u, M*k, &x));
        for (l = 0; l < M*N; l++) PetscCall(PetscRandomGetValueReal(rand, &L[l]));
        for (l = 0; l < Mk; l++) PetscCall(PetscRandomGetValueReal(rand, &u[l]));
        for (l = 0; l < M*k; l++) PetscCall(PetscRandomGetValueReal(rand, &x[l]));
        if (verbose) PetscCall(PetscViewerASCIIPrintf(viewer, "pullback M = %D:\n", M));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(CheckPullback(M, N, L, k, w, x, verbose, viewer));
        if (M != N) PetscCall(CheckPullback(N, M, L, k, u, v, PETSC_FALSE, viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        if ((k % N) && (N > 1)) {
          if (verbose) PetscCall(PetscViewerASCIIPrintf(viewer, "negative pullback M = %D:\n", M));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(CheckPullback(M, N, L, -k, w, x, verbose, viewer));
          if (M != N) PetscCall(CheckPullback(N, M, L, -k, u, v, PETSC_FALSE, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscFree3(L, u, x));
      }
      if (k > 0) { /* Interior */
        PetscInt    Nkm, l, m;
        PetscReal  *wIntv0, *wIntv0check, wvcheck, diff, diffMat, normMat;
        PetscReal  *intv0mat, *matcheck;
        PetscInt  (*indices)[3];

        PetscCall(PetscDTBinomialInt(N, k-1, &Nkm));
        PetscCall(PetscMalloc5(Nkm, &wIntv0, Nkm, &wIntv0check, Nk * Nkm, &intv0mat, Nk * Nkm, &matcheck, Nk * k, &indices));
        PetscCall(PetscDTAltVInterior(N, k, w, v, wIntv0));
        PetscCall(PetscDTAltVInteriorMatrix(N, k, v, intv0mat));
        PetscCall(PetscDTAltVInteriorPattern(N, k, indices));
        if (verbose) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "interior product matrix pattern:\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          for (l = 0; l < Nk * k; l++) {
            PetscInt row = indices[l][0];
            PetscInt col = indices[l][1];
            PetscInt x   = indices[l][2];

            PetscCall(PetscViewerASCIIPrintf(viewer,"intV[%D,%D] = %sV[%D]\n", row, col, x < 0 ? "-" : " ", x < 0 ? -(x + 1) : x));
          }
          PetscCall(PetscViewerASCIIPopTab(viewer));
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
          PetscCall(PetscViewerASCIIPrintf(viewer, "(w int v_0):\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (Nkm) PetscCall(PetscRealView(Nkm, wIntv0, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "(int v_0):\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (Nk * Nkm > 0) PetscCall(PetscRealView(Nk * Nkm, intv0mat, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscDTAltVApply(N, k - 1, wIntv0, &v[N], &wvcheck));
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));
        PetscCheckFalse(diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck)),PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: (w Int v0)(v_rem) (%g) != w(v) (%g)", (double) wvcheck, (double) wv);
        PetscCall(PetscFree5(wIntv0,wIntv0check,intv0mat,matcheck,indices));
      }
      if (k >= N - k) { /* Hodge star */
        PetscReal *u, *starw, *starstarw, wu, starwdotu;
        PetscReal diff, norm;
        PetscBool isOdd;
        PetscInt l;

        isOdd = (PetscBool) ((k * (N - k)) & 1);
        PetscCall(PetscMalloc3(Nk, &u, Nk, &starw, Nk, &starstarw));
        PetscCall(PetscDTAltVStar(N, k, 1, w, starw));
        PetscCall(PetscDTAltVStar(N, N-k, 1, starw, starstarw));
        if (verbose) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "star w:\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (Nk) PetscCall(PetscRealView(Nk, starw, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));

          PetscCall(PetscViewerASCIIPrintf(viewer, "star star w:\n"));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (Nk) PetscCall(PetscRealView(Nk, starstarw, viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        for (l = 0; l < Nk; l++) PetscCall(PetscRandomGetValueReal(rand,&u[l]));
        PetscCall(PetscDTAltVWedge(N, k, N - k, w, u, &wu));
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
        PetscCall(PetscFree3(u, starw, starstarw));
      }
      PetscCall(PetscFree(v));
      PetscCall(PetscFree(w));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
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
