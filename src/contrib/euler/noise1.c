#ifndef lint
static char vcid[] = "$Id: noise.c,v 1.15 1997/05/07 19:28:06 curfman Exp $";
#endif

#include "src/snes/snesimpl.h"
#include <math.h>

/* Data used by Jorge's diff parameter computation method */
typedef struct {
  Vec    *workv;          /* work vectors */
  FILE   *fp;             /* output file */
  int    function_count;  /* count of function evaluations for diff param estimation */
  double fnoise_min;      /* minimim allowable noise */
  double hopt_min;        /* minimum allowable hopt */
  int    fnoise_resets;   /* number of times we've reset the noise estimate */
  int    hopt_resets;     /* number of times we've reset the hopt estimate */
} DIFFPAR_MORE;

#if defined(HAVE_FORTRAN_CAPS)
#define dnest_ DNEST
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define dnest_ dnest
#endif

extern void dnest_(int*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,int*,Scalar*);
extern int JacMatMultCompare(SNES,Vec,Vec,double);
extern int SNESSetMatrixFreeParameters2(SNES,double,double,double);
extern int SNESUnSetMatrixFreeParameter(SNES snes);

#undef __FUNC__  
#define __FUNC__ "DiffParameterCreate_More"
int DiffParameterCreate_More(SNES snes,Vec x,void **outneP)
{
  DIFFPAR_MORE *neP;
  Vec          w;
  PetscRandom  rctx;  /* random number generator context */
  int          ierr;

  neP = PetscNew(DIFFPAR_MORE); CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(DIFFPAR_MORE));
  neP->function_count = 0;
  neP->fnoise_min     = 1.0e-16;
  neP->hopt_min       = 1.0e-8;
  neP->fnoise_resets  = 0;
  neP->hopt_resets    = 0;

  /* Create work vectors */
  ierr = VecDuplicateVecs(x,3,&neP->workv); CHKERRQ(ierr);
  w = neP->workv[0];

  /* Set components of vector w to random numbers */
  ierr = PetscRandomCreate(snes->comm,RANDOM_DEFAULT,&rctx); CHKERRQ(ierr);
  ierr = VecSetRandom(rctx,w); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rctx); CHKERRQ(ierr);

  /* Open output file */
  neP->fp = fopen("noise.out","w"); 
  if (!neP->fp) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file");
  PLogInfo(snes,"DiffParameterCreate_More: Creating Jorge's differencing parameter context\n");

  *outneP = neP;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DiffParameterDestroy_More"
int DiffParameterDestroy_More(void *nePv)
{
  DIFFPAR_MORE *neP = (DIFFPAR_MORE *)nePv;
  int          ierr;

  /* Destroy work vectors and close output file */
  ierr = VecDestroyVecs(neP->workv,3); CHKERRQ(ierr);
  fclose(neP->fp);
  PetscFree(neP);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "DiffParameterCompute_More"
int DiffParameterCompute_More(SNES snes,void *nePv,Vec x,Vec p,double *fnoise_rel,double *hopt)
{
  DIFFPAR_MORE *neP = (DIFFPAR_MORE *)nePv;
  Vec         w, xp, fvec;    /* work vectors to use in computing h */
  double      zero = 0.0, hl, hu, h, fnoise_s, fder2_s, alpha;
  double      fval[7], tab[7][7], eps[7];
  double      f, rerrf, fnoise, fder2;
  int         iter, k, i, j, ierr, info;
  int         nf = 7;         /* number of function evaluations */
  int         noise_test, fcount;
  MPI_Comm    comm = snes->comm;
  FILE        *fp;

  /* Call to SNESSetUp() just to set data structures in SNES context */
  if (!snes->setup_called) {ierr = SNESSetUp(snes,x); CHKERRQ(ierr);}

  w    = neP->workv[0];
  xp   = neP->workv[1];
  fvec = neP->workv[2];
  fp   = neP->fp;

  /* Initialize parameters */
  hl       = zero;
  hu       = zero;
  h        = 1.0e-3;
  fnoise_s = zero;
  fder2_s  = zero;
  fcount   = neP->function_count;

  /* We have 5 tries to attempt to compute a good hopt value */
  ierr = SNESGetIterationNumber(snes,&i); CHKERRQ(ierr);
  PetscFPrintf(comm,fp,"\n ------- SNES iteration %d ---------\n",i);
  for (iter=0; iter<5; iter++) {

    /* Compute the nf function values needed to estimate the noise from
       the difference table. */
    for (k=0; k<nf; k++) {
      alpha = h * ( k+1 - (nf+1)/2 );
      ierr = VecWAXPY(&alpha,p,x,xp); CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes,xp,fvec); CHKERRQ(ierr);
      neP->function_count++;
      ierr = VecDot(fvec,w,&fval[k]); CHKERRQ(ierr);
    }
    f = fval[(nf+1)/2 - 1];

    /* Construct the difference table. */
    for (i=0; i<nf; i++) {
      tab[i][0] = fval[i];
    }
    for (j=0; j<6; j++) {
      for (i=0; i<nf-j; i++) {
        tab[i][j+1] = tab[i+1][j] - tab[i][j];
      }
    }

    /* Print the difference table. */
    PetscFPrintf(comm,fp,"Difference Table: iter = %d\n",iter);
    for (i=0; i<nf; i++) {
      for (j=0; j<nf-i; j++) {
        PetscFPrintf(comm,fp," %10.2e ",tab[i][j]);
      }
      PetscFPrintf(comm,fp,"\n");
    }

    /* Call the noise estimator. */
    dnest_(&nf,fval,&h,&fnoise,&fder2,hopt,&info,eps);

    /* Output statements. */
    rerrf = fnoise/PetscAbsScalar(f);
    if (info == 1) PetscFPrintf(comm,fp,"%s\n","Noise detected");
    if (info == 2) PetscFPrintf(comm,fp,"%s\n","Noise not detected; h is too small");
    if (info == 3) PetscFPrintf(comm,fp,"%s\n","Noise not detected; h is too large");
    if (info == 4) PetscFPrintf(comm,fp,"%s\n","Noise detected, but unreliable hopt");
    PetscFPrintf(comm,fp,"Approximate epsfcn %g  %g  %g  %g  %g  %g\n",
        eps[0],eps[1],eps[2],eps[3],eps[4],eps[5]);
    PetscFPrintf(comm,fp,"h = %g, fnoise = %g, fder2 = %g, rerrf = %g, hopt = %g\n\n",
            h, fnoise, fder2, rerrf, *hopt);

    /* Save fnoise and fder2. */
    if (fnoise) fnoise_s = fnoise;
    if (fder2)  fder2_s = fder2;

    /* Check for noise detection. */
    if (fnoise_s && fder2_s) {
      fnoise = fnoise_s;
      fder2 = fder2_s;
      *hopt = 1.68*sqrt(fnoise/PetscAbsScalar(fder2));
      goto theend;
    } else {

      /* Update hl and hu, and determine new h. */
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

  if (fnoise < neP->fnoise_min) {
    PetscFPrintf(comm,fp,"Resetting fnoise: fnoise1 = %g, fnoise_min = %g\n",fnoise,neP->fnoise_min);
    fnoise = neP->fnoise_min;
    neP->fnoise_resets++;
  }
  if (*hopt < neP->hopt_min) {
    PetscFPrintf(comm,fp,"Resetting hopt: hopt1 = %g, hopt_min = %g\n",*hopt,neP->hopt_min);
    *hopt = neP->hopt_min;
    neP->hopt_resets++;
  }

  PetscFPrintf(comm,fp,"Errors in derivative:\n");
  *fnoise_rel = fnoise/PetscAbsScalar(f);
  PetscFPrintf(comm,fp,"f = %g, fnoise = %g, fder2 = %g, hopt = %g, fnoise_rel = %g\n",
          f, fnoise, fder2, *hopt, *fnoise_rel);


  /* For now, compute h **each** MV Mult!! */
  /*
  ierr = OptionsHasName(PETSC_NULL,"-matrix_free_jorge_each_mvp",&flg); CHKERRQ(ierr);
  if (!flg) {
    ierr = SNESSetMatrixFreeParameters2(snes,PETSC_DEFAULT,PETSC_DEFAULT,*hopt); CHKERRQ(ierr);
  }
  */
  fcount = neP->function_count - fcount;
  PLogInfo(snes,"DiffParameterCompute_More: fct_now = %d, fct_cum = %d, sqrt(noise)=%g, h_more=%g\n",
           fcount,neP->function_count, sqrt(*fnoise_rel),*hopt);


  ierr = OptionsHasName(PETSC_NULL,"-noise_test",&noise_test); CHKERRQ(ierr);
  if (noise_test) {
    ierr = JacMatMultCompare(snes,x,p,*hopt); CHKERRQ(ierr); 
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "JacMatMultCompare"
int JacMatMultCompare(SNES snes,Vec x,Vec p,double hopt)
{
  Vec          yy1, yy2; /* work vectors */
  Viewer       view2;    /* viewer */
  Mat          J;        /* analytic Jacobian (set as preconditioner matrix) */
  Mat          Jmf;      /* matrix-free Jacobian (set as true system matrix) */
  double       h;        /* differencing parameter */
  Vec          f;
  MatStructure sparsity = DIFFERENT_NONZERO_PATTERN;
  Scalar       alpha, yy1n, yy2n, enorm;
  int          i, ierr, printv;
  char         filename[32];
  MPI_Comm     comm = snes->comm;

  /* Compute function and analytic Jacobian at x */
  ierr = SNESGetJacobian(snes,&Jmf,&J,PETSC_NULL); CHKERRQ(ierr);
  ierr = SNESComputeJacobian(snes,x,&Jmf,&J,&sparsity); CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&f); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,x,f); CHKERRQ(ierr);

  /* Duplicate work vectors */
  ierr = VecDuplicate(x,&yy2); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&yy1); CHKERRQ(ierr);

  /* Compute true matrix-vector product */
  ierr = MatMult(J,p,yy1); CHKERRQ(ierr);
  ierr = VecNorm(yy1,NORM_2,&yy1n); CHKERRQ(ierr);

  /* View product vector if desired */
  ierr = OptionsHasName(PETSC_NULL,"-print_vecs",&printv); CHKERRQ(ierr);
  if (printv) {
    ierr = ViewerFileOpenASCII(comm,"y1.out",&view2); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = VecView(yy1,view2); CHKERRQ(ierr);
    ierr = ViewerDestroy(view2); CHKERRQ(ierr);
  }

  /* Test Jacobian-vector product computation */
  alpha = -1.0;  
  h = 0.01 * hopt;
  for (i=0; i<5; i++) {
    /* Set differencing parameter for matrix-free multiplication */
    ierr = SNESSetMatrixFreeParameters2(snes,PETSC_DEFAULT,PETSC_DEFAULT,h); CHKERRQ(ierr);

    /* Compute matrix-vector product via differencing approximation */
    ierr = MatMult(Jmf,p,yy2); CHKERRQ(ierr);
    ierr = VecNorm(yy2,NORM_2,&yy2n); CHKERRQ(ierr);

    /* View product vector if desired */
    if (printv) {
      sprintf(filename,"y2.%d.out",i);
      ierr = ViewerFileOpenASCII(comm,filename,&view2); CHKERRQ(ierr);
      ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
      ierr = VecView(yy2,view2); CHKERRQ(ierr);
      ierr = ViewerDestroy(view2); CHKERRQ(ierr);
    }

    /* Compute relative error */
    ierr = VecAXPY(&alpha,yy1,yy2); CHKERRQ(ierr);
    ierr = VecNorm(yy2,NORM_2,&enorm); CHKERRQ(ierr);
    enorm = enorm/yy1n;
    PetscFPrintf(comm,stdout,"h = %g: relative error = %g\n",h,enorm);
    h *= 10.0;
  }

  return 0;
}

static int lin_its_total = 0;

int MyMonitor(SNES snes,int its,double fnorm,void *dummy)
{
  int ierr, lin_its;

  ierr = SNESGetNumberLinearIterations(snes,&lin_its); CHKERRQ(ierr);
  lin_its_total += lin_its;
  PetscPrintf(snes->comm, "iter = %d, SNES Function norm = %g, lin_its = %d, total_lin_its = %d\n",its,fnorm,lin_its,lin_its_total);

  return SNESUnSetMatrixFreeParameter(snes); 
}
