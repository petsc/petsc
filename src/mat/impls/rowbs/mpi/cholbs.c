#ifndef lint
static char vcid[] = "$Id: $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
#include "src/mat/matimpl.h"
#include "mpirowbs.h"

int MatIncompleteCholeskyFactorSymbolic_MPIRowbs( Mat mat,IS perm,
                                                int fill,Mat *newfact )
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data, *fbs;
  Mat          fact;
  int          ierr;

  trvalid(__LINE__,__FILE__);
  VALIDHEADER(mat,MAT_COOKIE);
  /* Form empty factor matrix context; just set pointers from mat */
  ierr = MatCreateShellMPIRowbs(mat->comm,mbs->m,mbs->n,mbs->M,mbs->N,
                         0,0,0,0,&fact); CHKERR(ierr);

  /* Copy permuted matrix */
  mbs->fpA = BScopy_par_mat(mbs->pA); CHKERRBS(0);

  /* Set up the communication for factorization */
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo); CHKERRBS(0);

  fbs = (Mat_MPIRowbs *) fact->data;
  fbs->procinfo = mbs->procinfo;
  fbs->bsmap    = mbs->bsmap;
  fbs->A        = mbs->A;
  fbs->pA       = mbs->pA;
  fbs->fpA      = mbs->fpA;
  fbs->comm_pA  = mbs->comm_pA;
  fbs->comm_fpA = mbs->comm_fpA;
  fbs->diag     = mbs->diag;

  *newfact = fact;
  trvalid(__LINE__,__FILE__);
  return 0;
}
/* ----------------------------------------------------------------- */
/* 
   MatCholeskyFactorNumeric_MPIRowbs - Performs numeric factorization 
   of a symmetric parallel matrix, using BlockSolve.
   
.  failout - number of failures
.  alphaout -  alpha
 */
int MatCholeskyFactorNumeric_MPIRowbs(Mat mat,Mat *factp)
{
  Mat           fact = *factp;
  Mat_MPIRowbs  *mbs = (Mat_MPIRowbs *) mat->data;
  int           i, ierr, ldim, loc;
  Scalar        *da;

  trvalid(__LINE__,__FILE__);
  VALIDHEADER(mat,MAT_COOKIE); VALIDHEADER(fact,MAT_COOKIE);
  /* Do prep work if same nonzero structure as previously factored matrix */
  if (fact->factor == FACTOR_CHOLESKY) {
    /* Repermute the matrix */
    BSmain_reperm(mbs->procinfo,mbs->A,mbs->pA); CHKERRBS(0);
    /* Symmetrically scale the matrix by the diagonal */
    BSscale_diag(mbs->pA,mbs->pA->diag,mbs->procinfo); CHKERRBS(0);
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
    /* Store inverse of square root of permuted diagonal scaling matrix;
       this is done the first time in MatEndAssembly. */
    ierr = VecGetLocalSize( mbs->diag, &ldim ); CHKERR(ierr);
    ierr = VecGetArray( mbs->diag, &da ); CHKERR(ierr);
    for (i=0; i<ldim; i++) {
      da[i] = 1.0/sqrt(mbs->pA->scale_diag[i]);
    }
    ierr = VecBeginAssembly( mbs->diag ); CHKERR(ierr);
    ierr = VecEndAssembly( mbs->diag ); CHKERR(ierr);
  }
  /* Form incomplete Cholesky factor */
  while ( mbs->ierr = BSfactor( mbs->fpA, mbs->comm_fpA, mbs->procinfo ) ) {
    CHKERRBS(0);	mbs->failures++;
    /* Copy only the nonzeros */
    BScopy_nz( mbs->pA, mbs->fpA );			CHKERRBS(0);
    /* Increment the diagonal shift */
    mbs->alpha += 0.1;
    BSset_diag( mbs->fpA, mbs->alpha, mbs->procinfo );		CHKERRBS(0);
#if defined(PETSC_DEBUG)
    MPE_printf(mat->comm,"BlockSolve failed factors=%d, ierr=%d, alpha=%g\n",
               mbs->failures, mbs->ierr, mbs->alpha ); 
#endif
  }
  fact->factor = FACTOR_CHOLESKY;
  trvalid(__LINE__,__FILE__);
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   This routine assumes that the factored matrix has been produced by
   the ICC factorization of BlockSolve.  In particular, this routine
   assumes that the input/output vectors are permuted according to the
   BlockSolve coloring scheme.
 */

int MatSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;
  Vec          diag = mbs->diag;
  int          ierr;
  Scalar       *ywork;

  trvalid(__LINE__,__FILE__);
  /* Apply diagonal scaling to vector, where D^{-1/2} is stored */
  ierr = VecPMult( diag, x, y ); CHKERR(ierr);
  VecGetArray(y,&ywork);  

#if defined(PETSC_DEBUG)
  MLOG_ELM(mbs->procinfo->procset);
#endif
  if (mbs->procinfo->single)
      /* Use BlockSolve routine for no cliques/inodes */
      BSfor_solve1( mbs->fpA, ywork, mbs->comm_pA, mbs->procinfo );
  else
      BSfor_solve( mbs->fpA, ywork, mbs->comm_pA, mbs->procinfo );
  CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_ACC(MS_FORWARD);
  MLOG_ELM(mbs->procinfo->procset);
#endif
  if (mbs->procinfo->single)
      /* Use BlockSolve routine for no cliques/inodes */
      BSback_solve1( mbs->fpA, ywork, mbs->comm_pA, mbs->procinfo );
  else
      BSback_solve( mbs->fpA, ywork, mbs->comm_pA, mbs->procinfo );
  CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_ACC(MS_BACKWARD);
#endif

  /* Apply diagonal scaling to vector, where D^{-1/2} is stored */
  ierr = VecPMult( y, diag, y );  CHKERR(ierr);

  return 0;
}
#else
static int MatNull_MPIRowbs()
{return 0;}
#endif
