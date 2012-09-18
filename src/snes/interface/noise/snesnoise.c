
#include <petsc-private/snesimpl.h>

/* Data used by Jorge's diff parameter computation method */
typedef struct {
  Vec       *workv;          /* work vectors */
  FILE      *fp;             /* output file */
  int       function_count;  /* count of function evaluations for diff param estimation */
  double    fnoise_min;      /* minimim allowable noise */
  double    hopt_min;        /* minimum allowable hopt */
  double    h_first_try;     /* first try for h used in diff parameter estimate */
  PetscInt  fnoise_resets;   /* number of times we've reset the noise estimate */
  PetscInt  hopt_resets;     /* number of times we've reset the hopt estimate */
} DIFFPAR_MORE;


extern PetscErrorCode SNESNoise_dnest_(PetscInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*);
static PetscErrorCode JacMatMultCompare(SNES,Vec,Vec,double);
extern PetscErrorCode SNESDefaultMatrixFreeSetParameters2(Mat,double,double,double);
extern PetscErrorCode SNESUnSetMatrixFreeParameter(SNES snes);

#undef __FUNCT__
#define __FUNCT__ "SNESDiffParameterCreate_More"
PetscErrorCode SNESDiffParameterCreate_More(SNES snes,Vec x,void **outneP)
{
  DIFFPAR_MORE   *neP;
  Vec            w;
  PetscRandom    rctx;  /* random number generator context */
  PetscErrorCode ierr;
  PetscBool      flg;
  char           noise_file[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;

  ierr = PetscNewLog(snes,DIFFPAR_MORE,&neP);CHKERRQ(ierr);
  neP->function_count = 0;
  neP->fnoise_min     = 1.0e-20;
  neP->hopt_min       = 1.0e-8;
  neP->h_first_try    = 1.0e-3;
  neP->fnoise_resets  = 0;
  neP->hopt_resets    = 0;

  /* Create work vectors */
  ierr = VecDuplicateVecs(x,3,&neP->workv);CHKERRQ(ierr);
  w = neP->workv[0];

  /* Set components of vector w to random numbers */
  ierr = PetscRandomCreate(((PetscObject)snes)->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(w,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);

  /* Open output file */
  ierr = PetscOptionsGetString(((PetscObject)snes)->prefix,"-snes_mf_noise_file",noise_file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) neP->fp = fopen(noise_file,"w");
  else     neP->fp = fopen("noise.out","w");
  if (!neP->fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file");
  ierr = PetscInfo(snes,"Creating Jorge's differencing parameter context\n");CHKERRQ(ierr);

  *outneP = neP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDiffParameterDestroy_More"
PetscErrorCode SNESDiffParameterDestroy_More(void *nePv)
{
  DIFFPAR_MORE   *neP = (DIFFPAR_MORE *)nePv;
  PetscErrorCode ierr;
  int            err;

  PetscFunctionBegin;
  /* Destroy work vectors and close output file */
  ierr = VecDestroyVecs(3,&neP->workv);CHKERRQ(ierr);
  err = fclose(neP->fp);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
  ierr = PetscFree(neP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDiffParameterCompute_More"
PetscErrorCode SNESDiffParameterCompute_More(SNES snes,void *nePv,Vec x,Vec p,double *fnoise,double *hopt)
{
  DIFFPAR_MORE   *neP = (DIFFPAR_MORE *)nePv;
  Vec            w, xp, fvec;    /* work vectors to use in computing h */
  double         zero = 0.0, hl, hu, h, fnoise_s, fder2_s;
  PetscScalar    alpha;
  PetscScalar    fval[7], tab[7][7], eps[7], f;
  double         rerrf, fder2;
  PetscErrorCode ierr;
  PetscInt       iter, k, i, j,  info;
  PetscInt       nf = 7;         /* number of function evaluations */
  PetscInt       fcount;
  MPI_Comm       comm = ((PetscObject)snes)->comm;
  FILE           *fp;
  PetscBool      noise_test = PETSC_FALSE;

  PetscFunctionBegin;
  /* Call to SNESSetUp() just to set data structures in SNES context */
  if (!snes->setupcalled) {ierr = SNESSetUp(snes);CHKERRQ(ierr);}

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
  ierr = SNESGetIterationNumber(snes,&i);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"\n ------- SNES iteration %D ---------\n",i);CHKERRQ(ierr);
  for (iter=0; iter<5; iter++) {
    neP->h_first_try = h;

    /* Compute the nf function values needed to estimate the noise from
       the difference table */
    for (k=0; k<nf; k++) {
      alpha = h * ( k+1 - (nf+1)/2 );
      ierr = VecWAXPY(xp,alpha,p,x);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes,xp,fvec);CHKERRQ(ierr);
      neP->function_count++;
      ierr = VecDot(fvec,w,&fval[k]);CHKERRQ(ierr);
    }
    f = fval[(nf+1)/2 - 1];

    /* Construct the difference table */
    for (i=0; i<nf; i++) {
      tab[i][0] = fval[i];
    }
    for (j=0; j<6; j++) {
      for (i=0; i<nf-j; i++) {
        tab[i][j+1] = tab[i+1][j] - tab[i][j];
      }
    }

    /* Print the difference table */
    ierr = PetscFPrintf(comm,fp,"Difference Table: iter = %D\n",iter);CHKERRQ(ierr);
    for (i=0; i<nf; i++) {
      for (j=0; j<nf-i; j++) {
        ierr = PetscFPrintf(comm,fp," %10.2e ",tab[i][j]);CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(comm,fp,"\n");CHKERRQ(ierr);
    }

    /* Call the noise estimator */
    ierr = SNESNoise_dnest_(&nf,fval,&h,fnoise,&fder2,hopt,&info,eps);CHKERRQ(ierr);

    /* Output statements */
    rerrf = *fnoise/PetscAbsScalar(f);
    if (info == 1) {ierr = PetscFPrintf(comm,fp,"%s\n","Noise detected");CHKERRQ(ierr);}
    if (info == 2) {ierr = PetscFPrintf(comm,fp,"%s\n","Noise not detected; h is too small");CHKERRQ(ierr);}
    if (info == 3) {ierr = PetscFPrintf(comm,fp,"%s\n","Noise not detected; h is too large");CHKERRQ(ierr);}
    if (info == 4) {ierr = PetscFPrintf(comm,fp,"%s\n","Noise detected, but unreliable hopt");CHKERRQ(ierr);}
    ierr = PetscFPrintf(comm,fp,"Approximate epsfcn %G  %G  %G  %G  %G  %G\n",
        eps[0],eps[1],eps[2],eps[3],eps[4],eps[5]);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,fp,"h = %G, fnoise = %G, fder2 = %G, rerrf = %G, hopt = %G\n\n",
            h, *fnoise, fder2, rerrf, *hopt);CHKERRQ(ierr);

    /* Save fnoise and fder2. */
    if (*fnoise) fnoise_s = *fnoise;
    if (fder2)  fder2_s = fder2;

    /* Check for noise detection. */
    if (fnoise_s && fder2_s) {
      *fnoise = fnoise_s;
      fder2 = fder2_s;
      *hopt = 1.68*sqrt(*fnoise/PetscAbsScalar(fder2));
      goto theend;
    } else {

      /* Update hl and hu, and determine new h */
      if (info == 2 || info == 4) {
        hl = h;
        if (hu == zero) h = 100*h;
        else            h = PetscMin(100*h,0.1*hu);
       } else if (info == 3) {
        hu = h;
        h = PetscMax(1.0e-3,sqrt(hl/hu))*hu;
      }
    }
  }
  theend:

  if (*fnoise < neP->fnoise_min) {
    ierr = PetscFPrintf(comm,fp,"Resetting fnoise: fnoise1 = %G, fnoise_min = %G\n",*fnoise,neP->fnoise_min);CHKERRQ(ierr);
    *fnoise = neP->fnoise_min;
    neP->fnoise_resets++;
  }
  if (*hopt < neP->hopt_min) {
    ierr = PetscFPrintf(comm,fp,"Resetting hopt: hopt1 = %G, hopt_min = %G\n",*hopt,neP->hopt_min);CHKERRQ(ierr);
    *hopt = neP->hopt_min;
    neP->hopt_resets++;
  }

  ierr = PetscFPrintf(comm,fp,"Errors in derivative:\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"f = %G, fnoise = %G, fder2 = %G, hopt = %G\n",f,*fnoise,fder2,*hopt);CHKERRQ(ierr);

  /* For now, compute h **each** MV Mult!! */
  /*
  ierr = PetscOptionsHasName(PETSC_NULL,"-matrix_free_jorge_each_mvp",&flg);CHKERRQ(ierr);
  if (!flg) {
    Mat mat;
    ierr = SNESGetJacobian(snes,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = SNESDefaultMatrixFreeSetParameters2(mat,PETSC_DEFAULT,PETSC_DEFAULT,*hopt);CHKERRQ(ierr);
  }
  */
  fcount = neP->function_count - fcount;
  ierr = PetscInfo5(snes,"fct_now = %D, fct_cum = %D, rerrf=%G, sqrt(noise)=%G, h_more=%G\n",fcount,neP->function_count,rerrf,sqrt(*fnoise),*hopt);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-noise_test",&noise_test,PETSC_NULL);CHKERRQ(ierr);
  if (noise_test) {
    ierr = JacMatMultCompare(snes,x,p,*hopt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "JacMatMultCompare"
PetscErrorCode JacMatMultCompare(SNES snes,Vec x,Vec p,double hopt)
{
  Vec            yy1, yy2; /* work vectors */
  PetscViewer    view2;    /* viewer */
  Mat            J;        /* analytic Jacobian (set as preconditioner matrix) */
  Mat            Jmf;      /* matrix-free Jacobian (set as true system matrix) */
  double         h;        /* differencing parameter */
  Vec            f;
  MatStructure   sparsity = DIFFERENT_NONZERO_PATTERN;
  PetscScalar    alpha;
  PetscReal      yy1n,yy2n,enorm;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      printv = PETSC_FALSE;
  char           filename[32];
  MPI_Comm       comm = ((PetscObject)snes)->comm;

  PetscFunctionBegin;

  /* Compute function and analytic Jacobian at x */
  ierr = SNESGetJacobian(snes,&Jmf,&J,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESComputeJacobian(snes,x,&Jmf,&J,&sparsity);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&f,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);

  /* Duplicate work vectors */
  ierr = VecDuplicate(x,&yy2);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&yy1);CHKERRQ(ierr);

  /* Compute true matrix-vector product */
  ierr = MatMult(J,p,yy1);CHKERRQ(ierr);
  ierr = VecNorm(yy1,NORM_2,&yy1n);CHKERRQ(ierr);

  /* View product vector if desired */
  ierr = PetscOptionsGetBool(PETSC_NULL,"-print_vecs",&printv,PETSC_NULL);CHKERRQ(ierr);
  if (printv) {
    ierr = PetscViewerASCIIOpen(comm,"y1.out",&view2);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(view2,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
    ierr = VecView(yy1,view2);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view2);CHKERRQ(ierr);
  }

  /* Test Jacobian-vector product computation */
  alpha = -1.0;
  h = 0.01 * hopt;
  for (i=0; i<5; i++) {
    /* Set differencing parameter for matrix-free multiplication */
    ierr = SNESDefaultMatrixFreeSetParameters2(Jmf,PETSC_DEFAULT,PETSC_DEFAULT,h);CHKERRQ(ierr);

    /* Compute matrix-vector product via differencing approximation */
    ierr = MatMult(Jmf,p,yy2);CHKERRQ(ierr);
    ierr = VecNorm(yy2,NORM_2,&yy2n);CHKERRQ(ierr);

    /* View product vector if desired */
    if (printv) {
      sprintf(filename,"y2.%d.out",(int)i);
      ierr = PetscViewerASCIIOpen(comm,filename,&view2);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(view2,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
      ierr = VecView(yy2,view2);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&view2);CHKERRQ(ierr);
    }

    /* Compute relative error */
    ierr = VecAXPY(yy2,alpha,yy1);CHKERRQ(ierr);
    ierr = VecNorm(yy2,NORM_2,&enorm);CHKERRQ(ierr);
    enorm = enorm/yy1n;
    ierr = PetscFPrintf(comm,stdout,"h = %G: relative error = %G\n",h,enorm);CHKERRQ(ierr);
    h *= 10.0;
  }
  PetscFunctionReturn(0);
}

static PetscInt lin_its_total = 0;

PetscErrorCode SNESNoiseMonitor(SNES snes,PetscInt its,double fnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscInt       lin_its;

  PetscFunctionBegin;
  ierr = SNESGetLinearSolveIterations(snes,&lin_its);CHKERRQ(ierr);
  lin_its_total += lin_its;
  ierr = PetscPrintf(((PetscObject)snes)->comm, "iter = %D, SNES Function norm = %G, lin_its = %D, total_lin_its = %D\n",its,fnorm,lin_its,lin_its_total);CHKERRQ(ierr);

  ierr = SNESUnSetMatrixFreeParameter(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
