
#include <petsc/private/snesimpl.h>

PETSC_INTERN PetscErrorCode SNESDiffParameterCreate_More(SNES, Vec, void **);
PETSC_INTERN PetscErrorCode SNESDiffParameterCompute_More(SNES, void *, Vec, Vec, PetscReal *, PetscReal *);
PETSC_INTERN PetscErrorCode SNESDiffParameterDestroy_More(void *);

/* Data used by Jorge's diff parameter computation method */
typedef struct {
  Vec     *workv;          /* work vectors */
  FILE    *fp;             /* output file */
  PetscInt function_count; /* count of function evaluations for diff param estimation */
  double   fnoise_min;     /* minimim allowable noise */
  double   hopt_min;       /* minimum allowable hopt */
  double   h_first_try;    /* first try for h used in diff parameter estimate */
  PetscInt fnoise_resets;  /* number of times we've reset the noise estimate */
  PetscInt hopt_resets;    /* number of times we've reset the hopt estimate */
} DIFFPAR_MORE;

PETSC_INTERN PetscErrorCode SNESUnSetMatrixFreeParameter(SNES snes);
PETSC_INTERN PetscErrorCode SNESNoise_dnest_(PetscInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscScalar *, PetscScalar *, PetscInt *, PetscScalar *);

static PetscErrorCode JacMatMultCompare(SNES, Vec, Vec, double);

PetscErrorCode SNESDiffParameterCreate_More(SNES snes, Vec x, void **outneP)
{
  DIFFPAR_MORE *neP;
  Vec           w;
  PetscRandom   rctx; /* random number generator context */
  PetscBool     flg;
  char          noise_file[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(PetscNew(&neP));

  neP->function_count = 0;
  neP->fnoise_min     = 1.0e-20;
  neP->hopt_min       = 1.0e-8;
  neP->h_first_try    = 1.0e-3;
  neP->fnoise_resets  = 0;
  neP->hopt_resets    = 0;

  /* Create work vectors */
  PetscCall(VecDuplicateVecs(x, 3, &neP->workv));
  w = neP->workv[0];

  /* Set components of vector w to random numbers */
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)snes), &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(VecSetRandom(w, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  /* Open output file */
  PetscCall(PetscOptionsGetString(((PetscObject)snes)->options, ((PetscObject)snes)->prefix, "-snes_mf_noise_file", noise_file, sizeof(noise_file), &flg));
  if (flg) neP->fp = fopen(noise_file, "w");
  else neP->fp = fopen("noise.out", "w");
  PetscCheck(neP->fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file");
  PetscCall(PetscInfo(snes, "Creating Jorge's differencing parameter context\n"));

  *outneP = neP;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESDiffParameterDestroy_More(void *nePv)
{
  DIFFPAR_MORE *neP = (DIFFPAR_MORE *)nePv;
  int           err;

  PetscFunctionBegin;
  /* Destroy work vectors and close output file */
  PetscCall(VecDestroyVecs(3, &neP->workv));
  err = fclose(neP->fp);
  PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
  PetscCall(PetscFree(neP));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESDiffParameterCompute_More(SNES snes, void *nePv, Vec x, Vec p, double *fnoise, double *hopt)
{
  DIFFPAR_MORE *neP = (DIFFPAR_MORE *)nePv;
  Vec           w, xp, fvec; /* work vectors to use in computing h */
  double        zero = 0.0, hl, hu, h, fnoise_s, fder2_s;
  PetscScalar   alpha;
  PetscScalar   fval[7], tab[7][7], eps[7], f = -1;
  double        rerrf = -1., fder2;
  PetscInt      iter, k, i, j, info;
  PetscInt      nf = 7; /* number of function evaluations */
  PetscInt      fcount;
  MPI_Comm      comm;
  FILE         *fp;
  PetscBool     noise_test = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  /* Call to SNESSetUp() just to set data structures in SNES context */
  if (!snes->setupcalled) PetscCall(SNESSetUp(snes));

  w    = neP->workv[0];
  xp   = neP->workv[1];
  fvec = neP->workv[2];
  fp   = neP->fp;

  /* Initialize parameters */
  hl       = zero;
  hu       = zero;
  h        = neP->h_first_try;
  fnoise_s = zero;
  fder2_s  = zero;
  fcount   = neP->function_count;

  /* We have 5 tries to attempt to compute a good hopt value */
  PetscCall(SNESGetIterationNumber(snes, &i));
  PetscCall(PetscFPrintf(comm, fp, "\n ------- SNES iteration %" PetscInt_FMT " ---------\n", i));
  for (iter = 0; iter < 5; iter++) {
    neP->h_first_try = h;

    /* Compute the nf function values needed to estimate the noise from
       the difference table */
    for (k = 0; k < nf; k++) {
      alpha = h * (k + 1 - (nf + 1) / 2);
      PetscCall(VecWAXPY(xp, alpha, p, x));
      PetscCall(SNESComputeFunction(snes, xp, fvec));
      neP->function_count++;
      PetscCall(VecDot(fvec, w, &fval[k]));
    }
    f = fval[(nf + 1) / 2 - 1];

    /* Construct the difference table */
    for (i = 0; i < nf; i++) tab[i][0] = fval[i];

    for (j = 0; j < nf - 1; j++) {
      for (i = 0; i < nf - j - 1; i++) tab[i][j + 1] = tab[i + 1][j] - tab[i][j];
    }

    /* Print the difference table */
    PetscCall(PetscFPrintf(comm, fp, "Difference Table: iter = %" PetscInt_FMT "\n", iter));
    for (i = 0; i < nf; i++) {
      for (j = 0; j < nf - i; j++) PetscCall(PetscFPrintf(comm, fp, " %10.2e ", tab[i][j]));
      PetscCall(PetscFPrintf(comm, fp, "\n"));
    }

    /* Call the noise estimator */
    PetscCall(SNESNoise_dnest_(&nf, fval, &h, fnoise, &fder2, hopt, &info, eps));

    /* Output statements */
    rerrf = *fnoise / PetscAbsScalar(f);
    if (info == 1) PetscCall(PetscFPrintf(comm, fp, "%s\n", "Noise detected"));
    if (info == 2) PetscCall(PetscFPrintf(comm, fp, "%s\n", "Noise not detected; h is too small"));
    if (info == 3) PetscCall(PetscFPrintf(comm, fp, "%s\n", "Noise not detected; h is too large"));
    if (info == 4) PetscCall(PetscFPrintf(comm, fp, "%s\n", "Noise detected, but unreliable hopt"));
    PetscCall(PetscFPrintf(comm, fp, "Approximate epsfcn %g  %g  %g  %g  %g  %g\n", (double)eps[0], (double)eps[1], (double)eps[2], (double)eps[3], (double)eps[4], (double)eps[5]));
    PetscCall(PetscFPrintf(comm, fp, "h = %g, fnoise = %g, fder2 = %g, rerrf = %g, hopt = %g\n\n", (double)h, (double)*fnoise, (double)fder2, (double)rerrf, (double)*hopt));

    /* Save fnoise and fder2. */
    if (*fnoise) fnoise_s = *fnoise;
    if (fder2) fder2_s = fder2;

    /* Check for noise detection. */
    if (fnoise_s && fder2_s) {
      *fnoise = fnoise_s;
      fder2   = fder2_s;
      *hopt   = 1.68 * sqrt(*fnoise / PetscAbsScalar(fder2));
      goto theend;
    } else {
      /* Update hl and hu, and determine new h */
      if (info == 2 || info == 4) {
        hl = h;
        if (hu == zero) h = 100 * h;
        else h = PetscMin(100 * h, 0.1 * hu);
      } else if (info == 3) {
        hu = h;
        h  = PetscMax(1.0e-3, sqrt(hl / hu)) * hu;
      }
    }
  }
theend:

  if (*fnoise < neP->fnoise_min) {
    PetscCall(PetscFPrintf(comm, fp, "Resetting fnoise: fnoise1 = %g, fnoise_min = %g\n", (double)*fnoise, (double)neP->fnoise_min));
    *fnoise = neP->fnoise_min;
    neP->fnoise_resets++;
  }
  if (*hopt < neP->hopt_min) {
    PetscCall(PetscFPrintf(comm, fp, "Resetting hopt: hopt1 = %g, hopt_min = %g\n", (double)*hopt, (double)neP->hopt_min));
    *hopt = neP->hopt_min;
    neP->hopt_resets++;
  }

  PetscCall(PetscFPrintf(comm, fp, "Errors in derivative:\n"));
  PetscCall(PetscFPrintf(comm, fp, "f = %g, fnoise = %g, fder2 = %g, hopt = %g\n", (double)f, (double)*fnoise, (double)fder2, (double)*hopt));

  /* For now, compute h **each** MV Mult!! */
  /*
  PetscCall(PetscOptionsHasName(NULL,"-matrix_free_jorge_each_mvp",&flg));
  if (!flg) {
    Mat mat;
    PetscCall(SNESGetJacobian(snes,&mat,NULL,NULL));
    PetscCall(MatSNESMFMoreSetParameters(mat,PETSC_DEFAULT,PETSC_DEFAULT,*hopt));
  }
  */
  fcount = neP->function_count - fcount;
  PetscCall(PetscInfo(snes, "fct_now = %" PetscInt_FMT ", fct_cum = %" PetscInt_FMT ", rerrf=%g, sqrt(noise)=%g, h_more=%g\n", fcount, neP->function_count, (double)rerrf, (double)PetscSqrtReal(*fnoise), (double)*hopt));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-noise_test", &noise_test, NULL));
  if (noise_test) PetscCall(JacMatMultCompare(snes, x, p, *hopt));
  PetscFunctionReturn(0);
}

PetscErrorCode JacMatMultCompare(SNES snes, Vec x, Vec p, double hopt)
{
  Vec         yy1, yy2; /* work vectors */
  PetscViewer view2;    /* viewer */
  Mat         J;        /* analytic Jacobian (set as preconditioner matrix) */
  Mat         Jmf;      /* matrix-free Jacobian (set as true system matrix) */
  double      h;        /* differencing parameter */
  Vec         f;
  PetscScalar alpha;
  PetscReal   yy1n, yy2n, enorm;
  PetscInt    i;
  PetscBool   printv = PETSC_FALSE;
  char        filename[32];
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  /* Compute function and analytic Jacobian at x */
  PetscCall(SNESGetJacobian(snes, &Jmf, &J, NULL, NULL));
  PetscCall(SNESComputeJacobian(snes, x, Jmf, J));
  PetscCall(SNESGetFunction(snes, &f, NULL, NULL));
  PetscCall(SNESComputeFunction(snes, x, f));

  /* Duplicate work vectors */
  PetscCall(VecDuplicate(x, &yy2));
  PetscCall(VecDuplicate(x, &yy1));

  /* Compute true matrix-vector product */
  PetscCall(MatMult(J, p, yy1));
  PetscCall(VecNorm(yy1, NORM_2, &yy1n));

  /* View product vector if desired */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_vecs", &printv, NULL));
  if (printv) {
    PetscCall(PetscViewerASCIIOpen(comm, "y1.out", &view2));
    PetscCall(PetscViewerPushFormat(view2, PETSC_VIEWER_ASCII_COMMON));
    PetscCall(VecView(yy1, view2));
    PetscCall(PetscViewerPopFormat(view2));
    PetscCall(PetscViewerDestroy(&view2));
  }

  /* Test Jacobian-vector product computation */
  alpha = -1.0;
  h     = 0.01 * hopt;
  for (i = 0; i < 5; i++) {
    /* Set differencing parameter for matrix-free multiplication */
    PetscCall(MatSNESMFMoreSetParameters(Jmf, PETSC_DEFAULT, PETSC_DEFAULT, h));

    /* Compute matrix-vector product via differencing approximation */
    PetscCall(MatMult(Jmf, p, yy2));
    PetscCall(VecNorm(yy2, NORM_2, &yy2n));

    /* View product vector if desired */
    if (printv) {
      sprintf(filename, "y2.%d.out", (int)i);
      PetscCall(PetscViewerASCIIOpen(comm, filename, &view2));
      PetscCall(PetscViewerPushFormat(view2, PETSC_VIEWER_ASCII_COMMON));
      PetscCall(VecView(yy2, view2));
      PetscCall(PetscViewerPopFormat(view2));
      PetscCall(PetscViewerDestroy(&view2));
    }

    /* Compute relative error */
    PetscCall(VecAXPY(yy2, alpha, yy1));
    PetscCall(VecNorm(yy2, NORM_2, &enorm));
    enorm = enorm / yy1n;
    PetscCall(PetscFPrintf(comm, stdout, "h = %g: relative error = %g\n", (double)h, (double)enorm));
    h *= 10.0;
  }
  PetscFunctionReturn(0);
}

static PetscInt lin_its_total = 0;

PetscErrorCode SNESNoiseMonitor(SNES snes, PetscInt its, double fnorm, void *dummy)
{
  PetscInt lin_its;

  PetscFunctionBegin;
  PetscCall(SNESGetLinearSolveIterations(snes, &lin_its));
  lin_its_total += lin_its;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)snes), "iter = %" PetscInt_FMT ", SNES Function norm = %g, lin_its = %" PetscInt_FMT ", total_lin_its = %" PetscInt_FMT "\n", its, (double)fnorm, lin_its, lin_its_total));

  PetscCall(SNESUnSetMatrixFreeParameter(snes));
  PetscFunctionReturn(0);
}
