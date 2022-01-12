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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  k = PetscAbsInt(k);
  ierr = PetscDTBinomialInt(N, k, &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(M, k, &Mk);CHKERRQ(ierr);
  if (negative) {
    ierr = PetscMalloc1(Mk, &walloc);CHKERRQ(ierr);
    ierr = PetscDTAltVStar(M, M - k, 1, w, walloc);CHKERRQ(ierr);
    ww = walloc;
  } else {
    ww = w;
  }
  ierr = PetscMalloc2(Nk, &Lstarw, (M*k), &Lx);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nk * Mk, &Lstar, Nk, &Lstarwcheck);CHKERRQ(ierr);
  ierr = PetscDTAltVPullback(N, M, L, negative ? -k : k, w, Lstarw);CHKERRQ(ierr);
  ierr = PetscDTAltVPullbackMatrix(N, M, L, negative ? -k : k, Lstar);CHKERRQ(ierr);
  if (negative) {
    PetscReal *sLsw;

    ierr = PetscMalloc1(Nk, &sLsw);CHKERRQ(ierr);
    ierr = PetscDTAltVStar(N, N - k, 1, Lstarw, sLsw);CHKERRQ(ierr);
    ierr = PetscDTAltVApply(N, k, sLsw, x, &Lstarwx);CHKERRQ(ierr);
    ierr = PetscFree(sLsw);CHKERRQ(ierr);
  } else {
    ierr = PetscDTAltVApply(N, k, Lstarw, x, &Lstarwx);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer, "L:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (M*N > 0) {ierr = PetscRealView(M*N, L, viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer, "L*:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (Nk * Mk > 0) {ierr = PetscRealView(Nk * Mk, Lstar, viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer, "L*w:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (Nk > 0) {ierr = PetscRealView(Nk, Lstarw, viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscDTAltVApply(M, k, ww, Lx, &wLx);CHKERRQ(ierr);
  diff = PetscAbsReal(wLx - Lstarwx);
  if (diff > 10. * PETSC_SMALL * (PetscAbsReal(wLx) + PetscAbsReal(Lstarwx))) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "pullback check: pullback does not commute with application: w(Lx)(%g) != (L* w)(x)(%g)", wLx, Lstarwx);
  if (diffMat > PETSC_SMALL * normMat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "pullback check: pullback matrix does match matrix free result");
  ierr = PetscFree2(Lstar, Lstarwcheck);CHKERRQ(ierr);
  ierr = PetscFree2(Lstarw, Lx);CHKERRQ(ierr);
  ierr = PetscFree(walloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       i, numTests = 5, n[5] = {0, 1, 2, 3, 4};
  PetscBool      verbose = PETSC_FALSE;
  PetscRandom    rand;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for exterior algebra tests","none");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-N", "Up to 5 vector space dimensions to test","ex7.c",n,&numTests,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-verbose", "Verbose test output","ex7.c",verbose,&verbose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand, -1., 1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  if (!numTests) numTests = 5;
  viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  for (i = 0; i < numTests; i++) {
    PetscInt       k, N = n[i];

    if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "N = %D:\n", N);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);

    if (verbose) {
      PetscInt *perm;
      PetscInt fac = 1;

      ierr = PetscMalloc1(N, &perm);CHKERRQ(ierr);

      for (k = 1; k <= N; k++) fac *= k;
      ierr = PetscViewerASCIIPrintf(viewer, "Permutations of %D:\n", N);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      for (k = 0; k < fac; k++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  j, kCheck;

        ierr = PetscDTEnumPerm(N, k, perm, &isOdd);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "%D:", k);CHKERRQ(ierr);
        for (j = 0; j < N; j++) {
          ierr = PetscPrintf(PETSC_COMM_WORLD, " %D", perm[j]);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even");CHKERRQ(ierr);
        ierr = PetscDTPermIndex(N, perm, &kCheck, &isOddCheck);CHKERRQ(ierr);
        if (kCheck != k || isOddCheck != isOdd) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTEnumPerm / PetscDTPermIndex mismatch for (%D, %D)", N, k);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscFree(perm);CHKERRQ(ierr);
    }
    for (k = 0; k <= N; k++) {
      PetscInt   j, Nk, M;
      PetscReal *w, *v, wv;
      PetscInt  *subset;

      ierr = PetscDTBinomialInt(N, k, &Nk);CHKERRQ(ierr);
      if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "k = %D:\n", k);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "(%D choose %D): %D\n", N, k, Nk);CHKERRQ(ierr);}

      /* Test subset and complement enumeration */
      ierr = PetscMalloc1(N, &subset);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      for (j = 0; j < Nk; j++) {
        PetscBool isOdd, isOddCheck;
        PetscInt  jCheck, kCheck;

        ierr = PetscDTEnumSplit(N, k, j, subset, &isOdd);CHKERRQ(ierr);
        ierr = PetscDTPermIndex(N, subset, &kCheck, &isOddCheck);CHKERRQ(ierr);
        if (isOddCheck != isOdd) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDTEnumSplit sign does not mmatch PetscDTPermIndex sign");
        if (verbose) {
          PetscInt l;

          ierr = PetscViewerASCIIPrintf(viewer, "subset %D:", j);CHKERRQ(ierr);
          for (l = 0; l < k; l++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]);CHKERRQ(ierr);
          }
          ierr = PetscPrintf(PETSC_COMM_WORLD, " |");CHKERRQ(ierr);
          for (l = k; l < N; l++) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, " %D", subset[l]);CHKERRQ(ierr);
          }
          ierr = PetscPrintf(PETSC_COMM_WORLD, ", %s\n", isOdd ? "odd" : "even");CHKERRQ(ierr);
        }
        ierr = PetscDTSubsetIndex(N, k, subset, &jCheck);CHKERRQ(ierr);
        if (jCheck != j) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "jCheck (%D) != j (%D)", jCheck, j);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscFree(subset);CHKERRQ(ierr);

      /* Make a random k form */
      ierr = PetscMalloc1(Nk, &w);CHKERRQ(ierr);
      for (j = 0; j < Nk; j++) {ierr = PetscRandomGetValueReal(rand, &w[j]);CHKERRQ(ierr);}
      /* Make a set of random vectors */
      ierr = PetscMalloc1(N*k, &v);CHKERRQ(ierr);
      for (j = 0; j < N*k; j++) {ierr = PetscRandomGetValueReal(rand, &v[j]);CHKERRQ(ierr);}

      ierr = PetscDTAltVApply(N, k, w, v, &wv);CHKERRQ(ierr);

      if (verbose) {
        ierr = PetscViewerASCIIPrintf(viewer, "w:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        if (Nk) {ierr = PetscRealView(Nk, w, viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "v:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        if (N*k > 0) {ierr = PetscRealView(N*k, v, viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "w(v): %g\n", (double) wv);CHKERRQ(ierr);
      }

      /* sanity checks */
      if (k == 1) { /* 1-forms are functionals (dot products) */
        PetscInt  l;
        PetscReal wvcheck = 0.;
        PetscReal diff;

        for (l = 0; l < N; l++) wvcheck += w[l] * v[l];
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));
        if (diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck))) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "1-form / dot product equivalence: wvcheck (%g) != wv (%g)", (double) wvcheck, (double) wv);
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
        if (diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck))) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "n-form / determinant equivalence: wvcheck (%g) != wv (%g) %g", (double) wvcheck, (double) wv, (double) diff);
      }
      if (k > 0) { /* k-forms are linear in each component */
        PetscReal alpha;
        PetscReal *x, *axv, wx, waxv, waxvcheck;
        PetscReal diff;
        PetscReal rj;
        PetscInt  l;

        ierr = PetscMalloc2(N * k, &x, N * k, &axv);CHKERRQ(ierr);
        ierr = PetscRandomGetValueReal(rand, &alpha);CHKERRQ(ierr);
        ierr = PetscRandomSetInterval(rand, 0, k);CHKERRQ(ierr);
        ierr = PetscRandomGetValueReal(rand, &rj);CHKERRQ(ierr);
        j = (PetscInt) rj;
        ierr = PetscRandomSetInterval(rand, -1., 1.);CHKERRQ(ierr);
        for (l = 0; l < N*k; l++) x[l] = v[l];
        for (l = 0; l < N*k; l++) axv[l] = v[l];
        for (l = 0; l < N; l++) {
          PetscReal val;

          ierr = PetscRandomGetValueReal(rand, &val);CHKERRQ(ierr);
          x[j * N + l] = val;
          axv[j * N + l] += alpha * val;
        }
        ierr = PetscDTAltVApply(N, k, w, x, &wx);CHKERRQ(ierr);
        ierr = PetscDTAltVApply(N, k, w, axv, &waxv);CHKERRQ(ierr);
        waxvcheck = alpha * wx + wv;
        diff = waxv - waxvcheck;
        if (PetscAbsReal(diff) > 10. * PETSC_SMALL * (PetscAbsReal(waxv) + PetscAbsReal(waxvcheck))) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "linearity check: component %D, waxvcheck (%g) != waxv (%g)", j, (double) waxvcheck, (double) waxv);
        ierr = PetscFree2(x,axv);CHKERRQ(ierr);
      }
      if (k > 1) { /* k-forms are antisymmetric */
        PetscReal rj, rl, *swapv, wswapv, diff;
        PetscInt  l, m;

        ierr = PetscRandomSetInterval(rand, 0, k);CHKERRQ(ierr);
        ierr = PetscRandomGetValueReal(rand, &rj);CHKERRQ(ierr);
        j = (PetscInt) rj;
        l = j;
        while (l == j) {
          ierr = PetscRandomGetValueReal(rand, &rl);CHKERRQ(ierr);
          l = (PetscInt) rl;
        }
        ierr = PetscRandomSetInterval(rand, -1., 1.);CHKERRQ(ierr);
        ierr = PetscMalloc1(N * k, &swapv);CHKERRQ(ierr);
        for (m = 0; m < N * k; m++) swapv[m] = v[m];
        for (m = 0; m < N; m++) {
          swapv[j * N + m] = v[l * N + m];
          swapv[l * N + m] = v[j * N + m];
        }
        ierr = PetscDTAltVApply(N, k, w, swapv, &wswapv);CHKERRQ(ierr);
        diff = PetscAbsReal(wswapv + wv);
        if (diff > PETSC_SMALL * (PetscAbsReal(wswapv) + PetscAbsReal(wv))) SETERRQ4(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "antisymmetry check: components %D & %D, wswapv (%g) != -wv (%g)", j, l, (double) wswapv, (double) wv);
        ierr = PetscFree(swapv);CHKERRQ(ierr);
      }
      for (j = 0; j <= k && j + k <= N; j++) { /* wedge product */
        PetscInt   Nj, Njk, l, JKj;
        PetscReal *u, *uWw, *uWwcheck, *uWwmat, *x, *xsplit, uWwx, uWwxcheck, diff, norm;
        PetscInt  *split;

        if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "wedge j = %D:\n", j);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscDTBinomialInt(N, j,   &Nj);CHKERRQ(ierr);
        ierr = PetscDTBinomialInt(N, j+k, &Njk);CHKERRQ(ierr);
        ierr = PetscMalloc4(Nj, &u, Njk, &uWw, N*(j+k), &x, N*(j+k), &xsplit);CHKERRQ(ierr);
        ierr = PetscMalloc1(j+k,&split);CHKERRQ(ierr);
        for (l = 0; l < Nj; l++) {ierr = PetscRandomGetValueReal(rand, &u[l]);CHKERRQ(ierr);}
        for (l = 0; l < N*(j+k); l++) {ierr = PetscRandomGetValueReal(rand, &x[l]);CHKERRQ(ierr);}
        ierr = PetscDTAltVWedge(N, j, k, u, w, uWw);CHKERRQ(ierr);
        ierr = PetscDTAltVApply(N, j+k, uWw, x, &uWwx);CHKERRQ(ierr);
        if (verbose) {
          ierr = PetscViewerASCIIPrintf(viewer, "u:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (Nj) {ierr = PetscRealView(Nj, u, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer, "u wedge w:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (Njk) {ierr = PetscRealView(Njk, uWw, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer, "x:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (N*(j+k) > 0) {ierr = PetscRealView(N*(j+k), x, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer, "u wedge w(x): %g\n", (double) uWwx);CHKERRQ(ierr);
        }
        /* verify wedge formula */
        uWwxcheck = 0.;
        ierr = PetscDTBinomialInt(j+k, j, &JKj);CHKERRQ(ierr);
        for (l = 0; l < JKj; l++) {
          PetscBool isOdd;
          PetscReal ux, wx;
          PetscInt  m, p;

          ierr = PetscDTEnumSplit(j+k, j, l, split, &isOdd);CHKERRQ(ierr);
          for (m = 0; m < j+k; m++) {for (p = 0; p < N; p++) {xsplit[m * N + p] = x[split[m] * N + p];}}
          ierr = PetscDTAltVApply(N, j, u, xsplit, &ux);CHKERRQ(ierr);
          ierr = PetscDTAltVApply(N, k, w, &xsplit[j*N], &wx);CHKERRQ(ierr);
          uWwxcheck += isOdd ? -(ux * wx) : (ux * wx);
        }
        diff = PetscAbsReal(uWwx - uWwxcheck);
        if (diff > 10. * PETSC_SMALL * (PetscAbsReal(uWwx) + PetscAbsReal(uWwxcheck))) SETERRQ4(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wedge check: forms %D & %D, uWwxcheck (%g) != uWwx (%g)", j, k, (double) uWwxcheck, (double) uWwx);
        ierr = PetscFree(split);CHKERRQ(ierr);
        ierr = PetscMalloc2(Nk * Njk, &uWwmat, Njk, &uWwcheck);CHKERRQ(ierr);
        ierr = PetscDTAltVWedgeMatrix(N, j, k, u, uWwmat);CHKERRQ(ierr);
        if (verbose) {
          ierr = PetscViewerASCIIPrintf(viewer, "(u wedge):\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if ((Nk * Njk) > 0) {ierr = PetscRealView(Nk * Njk, uWwmat, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
        if (diff > PETSC_SMALL * norm) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "wedge matrix check: wedge matrix application does not match wedge direct application");
        ierr = PetscFree2(uWwmat, uWwcheck);CHKERRQ(ierr);
        ierr = PetscFree4(u, uWw, x, xsplit);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      for (M = PetscMax(1,k); M <= N; M++) { /* pullback */
        PetscReal   *L, *u, *x;
        PetscInt     Mk, l;

        ierr = PetscDTBinomialInt(M, k, &Mk);CHKERRQ(ierr);
        ierr = PetscMalloc3(M*N, &L, Mk, &u, M*k, &x);CHKERRQ(ierr);
        for (l = 0; l < M*N; l++) {ierr = PetscRandomGetValueReal(rand, &L[l]);CHKERRQ(ierr);}
        for (l = 0; l < Mk; l++) {ierr = PetscRandomGetValueReal(rand, &u[l]);CHKERRQ(ierr);}
        for (l = 0; l < M*k; l++) {ierr = PetscRandomGetValueReal(rand, &x[l]);CHKERRQ(ierr);}
        if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "pullback M = %D:\n", M);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = CheckPullback(M, N, L, k, w, x, verbose, viewer);CHKERRQ(ierr);
        if (M != N) {ierr = CheckPullback(N, M, L, k, u, v, PETSC_FALSE, viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        if ((k % N) && (N > 1)) {
          if (verbose) {ierr = PetscViewerASCIIPrintf(viewer, "negative pullback M = %D:\n", M);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = CheckPullback(M, N, L, -k, w, x, verbose, viewer);CHKERRQ(ierr);
          if (M != N) {ierr = CheckPullback(N, M, L, -k, u, v, PETSC_FALSE, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscFree3(L, u, x);CHKERRQ(ierr);
      }
      if (k > 0) { /* Interior */
        PetscInt    Nkm, l, m;
        PetscReal  *wIntv0, *wIntv0check, wvcheck, diff, diffMat, normMat;
        PetscReal  *intv0mat, *matcheck;
        PetscInt  (*indices)[3];

        ierr = PetscDTBinomialInt(N, k-1, &Nkm);CHKERRQ(ierr);
        ierr = PetscMalloc5(Nkm, &wIntv0, Nkm, &wIntv0check, Nk * Nkm, &intv0mat, Nk * Nkm, &matcheck, Nk * k, &indices);CHKERRQ(ierr);
        ierr = PetscDTAltVInterior(N, k, w, v, wIntv0);CHKERRQ(ierr);
        ierr = PetscDTAltVInteriorMatrix(N, k, v, intv0mat);CHKERRQ(ierr);
        ierr = PetscDTAltVInteriorPattern(N, k, indices);CHKERRQ(ierr);
        if (verbose) {
          ierr = PetscViewerASCIIPrintf(viewer, "interior product matrix pattern:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          for (l = 0; l < Nk * k; l++) {
            PetscInt row = indices[l][0];
            PetscInt col = indices[l][1];
            PetscInt x   = indices[l][2];

            ierr = PetscViewerASCIIPrintf(viewer,"intV[%D,%D] = %sV[%D]\n", row, col, x < 0 ? "-" : " ", x < 0 ? -(x + 1) : x);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
        if (diffMat > PETSC_SMALL * normMat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: matrix pattern does not match matrix");
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
        if (diffMat > PETSC_SMALL * normMat) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: application does not match matrix");
        if (verbose) {
          ierr = PetscViewerASCIIPrintf(viewer, "(w int v_0):\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (Nkm) {ierr = PetscRealView(Nkm, wIntv0, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

          ierr = PetscViewerASCIIPrintf(viewer, "(int v_0):\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (Nk * Nkm > 0) {ierr = PetscRealView(Nk * Nkm, intv0mat, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscDTAltVApply(N, k - 1, wIntv0, &v[N], &wvcheck);CHKERRQ(ierr);
        diff = PetscSqrtReal(PetscSqr(wvcheck - wv));
        if (diff >= PETSC_SMALL * (PetscAbsReal(wv) + PetscAbsReal(wvcheck))) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Interior product check: (w Int v0)(v_rem) (%g) != w(v) (%g)", (double) wvcheck, (double) wv);
        ierr = PetscFree5(wIntv0,wIntv0check,intv0mat,matcheck,indices);CHKERRQ(ierr);
      }
      if (k >= N - k) { /* Hodge star */
        PetscReal *u, *starw, *starstarw, wu, starwdotu;
        PetscReal diff, norm;
        PetscBool isOdd;
        PetscInt l;

        isOdd = (PetscBool) ((k * (N - k)) & 1);
        ierr = PetscMalloc3(Nk, &u, Nk, &starw, Nk, &starstarw);CHKERRQ(ierr);
        ierr = PetscDTAltVStar(N, k, 1, w, starw);CHKERRQ(ierr);
        ierr = PetscDTAltVStar(N, N-k, 1, starw, starstarw);CHKERRQ(ierr);
        if (verbose) {
          ierr = PetscViewerASCIIPrintf(viewer, "star w:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (Nk) {ierr = PetscRealView(Nk, starw, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);

          ierr = PetscViewerASCIIPrintf(viewer, "star star w:\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          if (Nk) {ierr = PetscRealView(Nk, starstarw, viewer);CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        for (l = 0; l < Nk; l++) {ierr = PetscRandomGetValueReal(rand,&u[l]);CHKERRQ(ierr);}
        ierr = PetscDTAltVWedge(N, k, N - k, w, u, &wu);CHKERRQ(ierr);
        starwdotu = 0.;
        for (l = 0; l < Nk; l++) starwdotu += starw[l] * u[l];
        diff = PetscAbsReal(wu - starwdotu);
        if (diff > PETSC_SMALL * (PetscAbsReal(wu) + PetscAbsReal(starwdotu))) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Hodge star check: (star w, u) (%g) != (w wedge u) (%g)", (double) starwdotu, (double) wu);

        diff = 0.;
        norm = 0.;
        for (l = 0; l < Nk; l++) {
          diff += PetscSqr(w[l] - (isOdd ? -starstarw[l] : starstarw[l]));
          norm += PetscSqr(w[l]) + PetscSqr(starstarw[l]);
        }
        diff = PetscSqrtReal(diff);
        norm = PetscSqrtReal(norm);
        if (diff > PETSC_SMALL * norm) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Hodge star check: star(star(w)) != (-1)^(N*(N-k)) w");
        ierr = PetscFree3(u, starw, starstarw);CHKERRQ(ierr);
      }
      ierr = PetscFree(v);CHKERRQ(ierr);
      ierr = PetscFree(w);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 1234
    args: -verbose
  test:
    suffix: 56
    args: -N 5,6
TEST*/
