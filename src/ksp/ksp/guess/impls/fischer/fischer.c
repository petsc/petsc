#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>

typedef struct {
  PetscInt         method;        /* 1, 2 or 3 */
  PetscInt         curl;          /* Current number of basis vectors */
  PetscInt         maxl;          /* Maximum number of basis vectors */
  PetscBool        monitor;
  PetscScalar      *alpha;        /* */
  Vec              *xtilde;       /* Saved x vectors */
  Vec              *btilde;       /* Saved b vectors, methods 1 and 3 */
  Vec              Ax;            /* method 2 */
  Vec              guess;
  PetscScalar      *corr;         /* correlation matrix in column-major format, method 3 */
  PetscReal        tol;           /* tolerance for determining rank, method 3 */
  Vec              last_b;        /* last b provided to FormGuess (not owned by this object), method 3 */
  PetscObjectState last_b_state;  /* state of last_b as of the last call to FormGuess, method 3 */
  PetscScalar      *last_b_coefs; /* dot products of last_b and btilde, method 3 */
} KSPGuessFischer;

static PetscErrorCode KSPGuessReset_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscLayout     Alay = NULL,vlay = NULL;
  PetscBool       cong;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  itg->curl = 0;
  /* destroy vectors if the size of the linear system has changed */
  if (guess->A) {
    ierr = MatGetLayouts(guess->A,&Alay,NULL);CHKERRQ(ierr);
  }
  if (itg->xtilde) {
    ierr = VecGetLayout(itg->xtilde[0],&vlay);CHKERRQ(ierr);
  }
  cong = PETSC_FALSE;
  if (vlay && Alay) {
    ierr = PetscLayoutCompare(Alay,vlay,&cong);CHKERRQ(ierr);
  }
  if (!cong) {
    ierr = VecDestroyVecs(itg->maxl,&itg->btilde);CHKERRQ(ierr);
    ierr = VecDestroyVecs(itg->maxl,&itg->xtilde);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->guess);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->Ax);CHKERRQ(ierr);
  }
  if (itg->corr) {
    ierr = PetscMemzero(itg->corr,sizeof(*itg->corr)*itg->maxl*itg->maxl);CHKERRQ(ierr);
  }
  itg->last_b = NULL;
  itg->last_b_state = 0;
  if (itg->last_b_coefs) {
    ierr = PetscMemzero(itg->last_b_coefs,sizeof(*itg->last_b_coefs)*itg->maxl);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetUp_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!itg->alpha) {
    ierr = PetscMalloc1(itg->maxl,&itg->alpha);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)guess,itg->maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (!itg->xtilde) {
    ierr = KSPCreateVecs(guess->ksp,itg->maxl,&itg->xtilde,0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(guess,itg->maxl,itg->xtilde);CHKERRQ(ierr);
  }
  if (!itg->btilde && (itg->method == 1 || itg->method == 3)) {
    ierr = KSPCreateVecs(guess->ksp,itg->maxl,&itg->btilde,0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(guess,itg->maxl,itg->btilde);CHKERRQ(ierr);
  }
  if (!itg->Ax && itg->method == 2) {
    ierr = VecDuplicate(itg->xtilde[0],&itg->Ax);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)guess,(PetscObject)itg->Ax);CHKERRQ(ierr);
  }
  if (!itg->guess && (itg->method == 1 || itg->method == 2)) {
    ierr = VecDuplicate(itg->xtilde[0],&itg->guess);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)guess,(PetscObject)itg->guess);CHKERRQ(ierr);
  }
  if (!itg->corr && itg->method == 3) {
    ierr = PetscCalloc1(itg->maxl*itg->maxl,&itg->corr);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)guess,itg->maxl*itg->maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (!itg->last_b_coefs && itg->method == 3) {
    ierr = PetscCalloc1(itg->maxl,&itg->last_b_coefs);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)guess,itg->maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessDestroy_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->maxl,&itg->btilde);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->maxl,&itg->xtilde);CHKERRQ(ierr);
  ierr = VecDestroy(&itg->guess);CHKERRQ(ierr);
  ierr = VecDestroy(&itg->Ax);CHKERRQ(ierr);
  ierr = PetscFree(itg->corr);CHKERRQ(ierr);
  ierr = PetscFree(itg->last_b_coefs);CHKERRQ(ierr);
  ierr = PetscFree(itg);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)guess,"KSPGuessFischerSetModel_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Note: do not change the b right hand side as is done in the publication */
static PetscErrorCode KSPGuessFormGuess_Fischer_1(KSPGuess guess,Vec b,Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
  if (itg->monitor) {
    ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess alphas =");CHKERRQ(ierr);
    for (i=0; i<itg->curl; i++) {
      ierr = PetscPrintf(((PetscObject)guess)->comm," %g",(double)PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(((PetscObject)guess)->comm,"\n");CHKERRQ(ierr);
  }
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
  ierr = VecCopy(x,itg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_1(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscReal       norm;
  PetscErrorCode  ierr;
  int             curl = itg->curl,i;

  PetscFunctionBegin;
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(guess->ksp,guess->A,x,itg->btilde[0]);CHKERRQ(ierr);
    /* ierr = VecCopy(b,itg->btilde[0]);CHKERRQ(ierr); */
    ierr = VecNormalize(itg->btilde[0],&norm);CHKERRQ(ierr);
    ierr = VecCopy(x,itg->xtilde[0]);CHKERRQ(ierr);
    ierr = VecScale(itg->xtilde[0],1.0/norm);CHKERRQ(ierr);
    itg->curl = 1;
  } else {
    if (!curl) {
      ierr = VecCopy(x,itg->xtilde[curl]);CHKERRQ(ierr);
    } else {
      ierr = VecWAXPY(itg->xtilde[curl],-1.0,itg->guess,x);CHKERRQ(ierr);
    }
    ierr = KSP_MatMult(guess->ksp,guess->A,itg->xtilde[curl],itg->btilde[curl]);CHKERRQ(ierr);
    ierr = VecMDot(itg->btilde[curl],curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->btilde[curl],curl,itg->alpha,itg->btilde);CHKERRQ(ierr);
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
    ierr = VecNormalize(itg->btilde[curl],&norm);CHKERRQ(ierr);
    if (norm) {
      ierr = VecScale(itg->xtilde[curl],1.0/norm);CHKERRQ(ierr);
      itg->curl++;
    } else {
      ierr = PetscInfo(guess->ksp,"Not increasing dimension of Fischer space because new direction is identical to previous\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  Given a basis generated already this computes a new guess x from the new right hand side b
  Figures out the components of b in each btilde direction and adds them to x
  Note: do not change the b right hand side as is done in the publication
*/
static PetscErrorCode KSPGuessFormGuess_Fischer_2(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->xtilde,itg->alpha);CHKERRQ(ierr);
  if (itg->monitor) {
    ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess alphas =");CHKERRQ(ierr);
    for (i=0; i<itg->curl; i++) {
      ierr = PetscPrintf(((PetscObject)guess)->comm," %g",(double)PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(((PetscObject)guess)->comm,"\n");CHKERRQ(ierr);
  }
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
  ierr = VecCopy(x,itg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_2(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscScalar     norm;
  PetscErrorCode  ierr;
  int             curl = itg->curl,i;

  PetscFunctionBegin;
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(guess->ksp,guess->A,x,itg->Ax);CHKERRQ(ierr); /* norm = sqrt(x'Ax) */
    ierr = VecDot(x,itg->Ax,&norm);CHKERRQ(ierr);
    ierr = VecCopy(x,itg->xtilde[0]);CHKERRQ(ierr);
    ierr = VecScale(itg->xtilde[0],1.0/PetscSqrtScalar(norm));CHKERRQ(ierr);
    itg->curl = 1;
  } else {
    if (!curl) {
      ierr = VecCopy(x,itg->xtilde[curl]);CHKERRQ(ierr);
    } else {
      ierr = VecWAXPY(itg->xtilde[curl],-1.0,itg->guess,x);CHKERRQ(ierr);
    }
    ierr = KSP_MatMult(guess->ksp,guess->A,itg->xtilde[curl],itg->Ax);CHKERRQ(ierr);
    ierr = VecMDot(itg->Ax,curl,itg->xtilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);

    ierr = KSP_MatMult(guess->ksp,guess->A,itg->xtilde[curl],itg->Ax);CHKERRQ(ierr); /* norm = sqrt(xtilde[curl]'Axtilde[curl]) */
    ierr = VecDot(itg->xtilde[curl],itg->Ax,&norm);CHKERRQ(ierr);
    if (PetscAbsScalar(norm) != 0.0) {
      ierr = VecScale(itg->xtilde[curl],1.0/PetscSqrtScalar(norm));CHKERRQ(ierr);
      itg->curl++;
    } else {
      ierr = PetscInfo(guess->ksp,"Not increasing dimension of Fischer space because new direction is identical to previous\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  Rather than the standard algorithm implemented in 2, we treat the provided x and b vectors to be spanning sets (not necessarily linearly independent) and use them to compute a windowed correlation matrix. Since the correlation matrix may be singular we solve it with the pseudoinverse, provided by SYEV/HEEV.
*/
static PetscErrorCode KSPGuessFormGuess_Fischer_3(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;
  PetscInt        i,j,m;
  PetscReal       *s_values;
  PetscScalar     *corr,*work,*scratch_vec,zero=0.0,one=1.0;
  PetscBLASInt    blas_m,blas_info,blas_rank=0,blas_lwork,blas_one = 1;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif

  /* project provided b onto space of stored btildes */
  PetscFunctionBegin;
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  m = itg->curl;
  itg->last_b = b;
  ierr = PetscObjectStateGet((PetscObject)b,&itg->last_b_state);CHKERRQ(ierr);
  if (m > 0) {
    ierr = PetscBLASIntCast(m,&blas_m);CHKERRQ(ierr);
    blas_lwork = (/* assume a block size of m */blas_m+2)*blas_m;
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscCalloc5(m*m,&corr,m,&s_values,blas_lwork,&work,3*m-2,&rwork,m,&scratch_vec);CHKERRQ(ierr);
#else
    ierr = PetscCalloc4(m*m,&corr,m,&s_values,blas_lwork,&work,m,&scratch_vec);CHKERRQ(ierr);
#endif
    ierr = VecMDot(b,itg->curl,itg->btilde,itg->last_b_coefs);CHKERRQ(ierr);
    for (j=0;j<m;++j) {
      for (i=0;i<m;++i) {
        corr[m*j+i] = itg->corr[(itg->maxl)*j+i];
      }
    }
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscReal max_s_value = 0.0;
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKheev", LAPACKheev_("V", "L", &blas_m, corr, &blas_m, s_values, work, &blas_lwork, rwork, &blas_info));
#else
    PetscStackCallBLAS(
      "LAPACKsyev", LAPACKsyev_("V", "L", &blas_m, corr, &blas_m, s_values, work, &blas_lwork, &blas_info));
#endif

    if (blas_info == 0) {
      /* make corr store singular vectors and s_values store singular values */
      for (j=0; j<m; ++j) {
        if (s_values[j] < 0.0) {
          s_values[j] = PetscAbsReal(s_values[j]);
          for (i=0; i<m; ++i) {
            corr[m*j + i] *= -1.0;
          }
        }
        max_s_value = PetscMax(max_s_value, s_values[j]);
      }

      /* manually apply the action of the pseudoinverse */
      PetscStackCallBLAS("BLASgemv", BLASgemv_("T", &blas_m, &blas_m, &one, corr, &blas_m, itg->last_b_coefs, &blas_one, &zero, scratch_vec, &blas_one));
      for (j=0; j<m; ++j) {
        if (s_values[j] > itg->tol*max_s_value) {
          scratch_vec[j] /= s_values[j];
          blas_rank += 1;
        } else {
          scratch_vec[j] = 0.0;
        }
      }
      PetscStackCallBLAS("BLASgemv", BLASgemv_("N", &blas_m, &blas_m, &one, corr, &blas_m, scratch_vec, &blas_one, &zero, itg->alpha, &blas_one));

    } else {
      ierr = PetscInfo(guess, "Warning eigenvalue solver failed with error code %d - setting initial guess to zero\n", (int)blas_info);CHKERRQ(ierr);
      ierr = PetscMemzero(itg->alpha,sizeof(*itg->alpha)*itg->maxl);CHKERRQ(ierr);
    }
    ierr = PetscFPTrapPop();CHKERRQ(ierr);

    if (itg->monitor && blas_info == 0) {
      ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess correlation rank = %d\n",(int)blas_rank);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess singular values = ");CHKERRQ(ierr);
      for (i=0; i<itg->curl; i++) {
        ierr = PetscPrintf(((PetscObject)guess)->comm," %g",(double)s_values[i]);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(((PetscObject)guess)->comm,"\n");CHKERRQ(ierr);

      ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess alphas =");CHKERRQ(ierr);
      for (i=0; i<itg->curl; i++) {
        ierr = PetscPrintf(((PetscObject)guess)->comm," %g",(double)PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
      }
      ierr = PetscPrintf(((PetscObject)guess)->comm,"\n");CHKERRQ(ierr);
    }
    /* Form the initial guess by using b's projection coefficients with the xs */
    ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscFree5(corr, s_values, work, rwork, scratch_vec);CHKERRQ(ierr);
#else
    ierr = PetscFree4(corr, s_values, work, scratch_vec);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_3(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer  *itg = (KSPGuessFischer*)guess->data;
  PetscBool        rotate = itg->curl == itg->maxl ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode   ierr;
  PetscInt         i,j;
  PetscObjectState b_state;
  PetscScalar      *last_column;
  Vec              oldest;

  PetscFunctionBegin;
  if (rotate) {
    /* we have the maximum number of vectors so rotate: oldest vector is at index 0 */
    oldest = itg->xtilde[0];
    for (i=1;i<itg->curl;++i) {
      itg->xtilde[i-1] = itg->xtilde[i];
    }
    itg->xtilde[itg->curl-1] = oldest;
    ierr = VecCopy(x,itg->xtilde[itg->curl-1]);CHKERRQ(ierr);

    oldest = itg->btilde[0];
    for (i=1;i<itg->curl;++i) {
      itg->btilde[i-1] = itg->btilde[i];
    }
    itg->btilde[itg->curl-1] = oldest;
    ierr = VecCopy(b,itg->btilde[itg->curl-1]);CHKERRQ(ierr);
    /* shift correlation matrix up and left */
    for (j=1; j<itg->maxl; ++j) {
      for (i=1; i<itg->maxl; ++i) {
        itg->corr[(j-1)*itg->maxl+i-1]=itg->corr[j*itg->maxl+i];
      }
    }
  } else {
    /* append new vectors */
    ierr = VecCopy(x,itg->xtilde[itg->curl]);CHKERRQ(ierr);
    ierr = VecCopy(b,itg->btilde[itg->curl]);CHKERRQ(ierr);
    itg->curl++;
  }

  /*
      Populate new column of the correlation matrix and then copy it into the
      row. itg->maxl is the allocated length per column: itg->curl is the actual
      column length.
      If possible reuse the dot products from FormGuess
  */
  last_column = itg->corr+(itg->curl-1)*itg->maxl;
  ierr = PetscObjectStateGet((PetscObject)b,&b_state);CHKERRQ(ierr);
  if (b_state == itg->last_b_state && b == itg->last_b) {
    if (rotate) {
      for (i=1; i<itg->maxl; ++i) {
        itg->last_b_coefs[i-1] = itg->last_b_coefs[i];
      }
    }
    ierr = VecDot(b,b,&itg->last_b_coefs[itg->curl-1]);CHKERRQ(ierr);
    ierr = PetscArraycpy(last_column,itg->last_b_coefs,itg->curl);CHKERRQ(ierr);
  } else {
    ierr = VecMDot(b,itg->curl,itg->btilde,last_column);CHKERRQ(ierr);
  }
  for (i=0;i<itg->curl;++i) {
    itg->corr[i*itg->maxl+itg->curl-1] = last_column[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetFromOptions_Fischer(KSPGuess guess)
{
  KSPGuessFischer *ITG = (KSPGuessFischer *)guess->data;
  PetscInt        nmax = 2, model[2];
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  model[0] = ITG->method;
  model[1] = ITG->maxl;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)guess),((PetscObject)guess)->prefix,"Fischer guess options","KSPGuess");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-ksp_guess_fischer_model","Model type and dimension of basis","KSPGuessFischerSetModel",model,&nmax,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPGuessFischerSetModel(guess,model[0],model[1]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-ksp_guess_fischer_tol","Tolerance to determine rank via ratio of singular values","KSPGuessSetTolerance",ITG->tol,&ITG->tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_guess_fischer_monitor","Monitor the guess",NULL,ITG->monitor,&ITG->monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetTolerance_Fischer(KSPGuess guess,PetscReal tol)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;

  PetscFunctionBegin;
  itg->tol = tol;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessView_Fischer(KSPGuess guess,PetscViewer viewer)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscBool       isascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Model %D, size %D\n",itg->method,itg->maxl);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   KSPGuessFischerSetModel - Use the Paul Fischer algorithm or its variants

   Logically Collective on guess

   Input Parameters:
+  guess - the initial guess context
.  model - use model 1, model 2, model 3, or any other number to turn it off
-  size  - size of subspace used to generate initial guess

    Options Database:
.   -ksp_guess_fischer_model <model,size> - uses the Fischer initial guess generator for repeated linear solves

   Level: advanced

.seealso: KSPGuess, KSPGuessCreate(), KSPSetUseFischerGuess(), KSPSetGuess(), KSPGetGuess(), KSP
@*/
PetscErrorCode  KSPGuessFischerSetModel(KSPGuess guess,PetscInt model,PetscInt size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidLogicalCollectiveInt(guess,model,2);
  ierr = PetscTryMethod(guess,"KSPGuessFischerSetModel_C",(KSPGuess,PetscInt,PetscInt),(guess,model,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessFischerSetModel_Fischer(KSPGuess guess,PetscInt model,PetscInt size)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (model == 1) {
    guess->ops->update    = KSPGuessUpdate_Fischer_1;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_1;
  } else if (model == 2) {
    guess->ops->update    = KSPGuessUpdate_Fischer_2;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_2;
  } else if (model == 3) {
    guess->ops->update    = KSPGuessUpdate_Fischer_3;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_3;
  } else {
    guess->ops->update    = NULL;
    guess->ops->formguess = NULL;
    itg->method           = 0;
    PetscFunctionReturn(0);
  }
  if (size != itg->maxl) {
    ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
    ierr = VecDestroyVecs(itg->maxl,&itg->btilde);CHKERRQ(ierr);
    ierr = VecDestroyVecs(itg->maxl,&itg->xtilde);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->guess);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->Ax);CHKERRQ(ierr);
  }
  itg->method = model;
  itg->maxl   = size;
  PetscFunctionReturn(0);
}

/*
    KSPGUESSFISCHER - Implements Paul Fischer's two initial guess algorithms and a nonorthogonalizing variant for situations where
    a linear system is solved repeatedly

  References:
.   1. -   https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020363_1994020363.pdf

   Notes:
    the algorithm is different from Fischer's paper because we do not CHANGE the right hand side of the new
    problem and solve the problem with an initial guess of zero, rather we solve the original problem
    with a nonzero initial guess (this is done so that the linear solver convergence tests are based on
    the original RHS). We use the xtilde = x - xguess as the new direction so that it is not
    mostly orthogonal to the previous solutions.

    These are not intended to be used directly, they are called by KSP automatically with the command line options -ksp_guess_type fischer -ksp_guess_fischer_model <int,int> or programmatically as
.vb
    KSPGetGuess(ksp,&guess);
    KSPGuessSetType(guess,KSPGUESSFISCHER);
    KSPGuessFischerSetModel(guess,model,basis);
    KSPGuessSetTolerance(guess,PETSC_MACHINE_EPSILON);

    The default tolerance (which is only used in Method 3) is 32*PETSC_MACHINE_EPSILON. This value was chosen
    empirically by trying a range of tolerances and picking the one that lowered the solver iteration count the most
    with five vectors.

    Method 2 is only for positive definite matrices, since it uses the A norm.

    Method 3 is not in the original paper. It is the same as the first two methods except that it
    does not orthogonalize the input vectors or use A at all. This choice is faster but provides a
    less effective initial guess for large (about 10) numbers of stored vectors.

    Developer note:
      The option -ksp_fischer_guess <int,int> is still available for backward compatibility

    Level: intermediate

@*/
PetscErrorCode KSPGuessCreate_Fischer(KSPGuess guess)
{
  KSPGuessFischer *fischer;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(guess,&fischer);CHKERRQ(ierr);
  fischer->method = 1;  /* defaults to method 1 */
  fischer->maxl   = 10;
  fischer->tol    = 32.0*PETSC_MACHINE_EPSILON;
  guess->data     = fischer;

  guess->ops->setfromoptions = KSPGuessSetFromOptions_Fischer;
  guess->ops->destroy        = KSPGuessDestroy_Fischer;
  guess->ops->settolerance   = KSPGuessSetTolerance_Fischer;
  guess->ops->setup          = KSPGuessSetUp_Fischer;
  guess->ops->view           = KSPGuessView_Fischer;
  guess->ops->reset          = KSPGuessReset_Fischer;
  guess->ops->update         = KSPGuessUpdate_Fischer_1;
  guess->ops->formguess      = KSPGuessFormGuess_Fischer_1;

  ierr = PetscObjectComposeFunction((PetscObject)guess,"KSPGuessFischerSetModel_C",KSPGuessFischerSetModel_Fischer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
