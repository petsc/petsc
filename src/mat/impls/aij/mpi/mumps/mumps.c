
/*
    Provides an interface to the MUMPS sparse solver
*/

#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I  "petscmat.h"  I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#include <cmumps_c.h>
#else
#include <zmumps_c.h>
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
#include <smumps_c.h>
#else
#include <dmumps_c.h>
#endif
#endif
EXTERN_C_END
#define JOB_INIT -1
#define JOB_FACTSYMBOLIC 1
#define JOB_FACTNUMERIC 2
#define JOB_SOLVE 3
#define JOB_END -2

/* calls to MUMPS */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscMUMPS_c cmumps_c
#else
#define PetscMUMPS_c zmumps_c
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscMUMPS_c smumps_c
#else
#define PetscMUMPS_c dmumps_c
#endif
#endif

/* declare MumpsScalar */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define MumpsScalar mumps_complex
#else
#define MumpsScalar mumps_double_complex
#endif
#else
#define MumpsScalar PetscScalar
#endif

/* macros s.t. indices match MUMPS documentation */
#define ICNTL(I) icntl[(I)-1]
#define CNTL(I) cntl[(I)-1]
#define INFOG(I) infog[(I)-1]
#define INFO(I) info[(I)-1]
#define RINFOG(I) rinfog[(I)-1]
#define RINFO(I) rinfo[(I)-1]

typedef struct {
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  CMUMPS_STRUC_C id;
#else
  ZMUMPS_STRUC_C id;
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  SMUMPS_STRUC_C id;
#else
  DMUMPS_STRUC_C id;
#endif
#endif

  MatStructure matstruc;
  PetscMPIInt  myid,size;
  PetscInt     *irn,*jcn,nz,sym;
  PetscScalar  *val;
  MPI_Comm     comm_mumps;
  PetscBool    isAIJ;
  PetscInt     ICNTL9_pre;           /* check if ICNTL(9) is changed from previous MatSolve */
  VecScatter   scat_rhs, scat_sol;   /* used by MatSolve() */
  Vec          b_seq,x_seq;
  PetscInt     ninfo,*info;          /* display INFO */
  PetscInt     sizeredrhs;
  PetscScalar  *schur_sol;
  PetscInt     schur_sizesol;

  PetscErrorCode (*ConvertToTriples)(Mat, int, MatReuse, int*, int**, int**, PetscScalar**);
} Mat_MUMPS;

extern PetscErrorCode MatDuplicate_MUMPS(Mat,MatDuplicateOption,Mat*);

static PetscErrorCode MatMumpsResetSchur_Private(Mat_MUMPS* mumps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(mumps->id.listvar_schur,mumps->id.schur);CHKERRQ(ierr);
  ierr = PetscFree(mumps->id.redrhs);CHKERRQ(ierr);
  ierr = PetscFree(mumps->schur_sol);CHKERRQ(ierr);
  mumps->id.size_schur = 0;
  mumps->id.schur_lld  = 0;
  mumps->id.ICNTL(19)  = 0;
  PetscFunctionReturn(0);
}

/* solve with rhs in mumps->id.redrhs and return in the same location */
static PetscErrorCode MatMumpsSolveSchur_Private(Mat F)
{
  Mat_MUMPS            *mumps=(Mat_MUMPS*)F->data;
  Mat                  S,B,X;
  MatFactorSchurStatus schurstatus;
  PetscInt             sizesol;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MatFactorFactorizeSchurComplement(F);CHKERRQ(ierr);
  ierr = MatFactorGetSchurComplement(F,&S,&schurstatus);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,(PetscScalar*)mumps->id.redrhs,&B);CHKERRQ(ierr);
  switch (schurstatus) {
  case MAT_FACTOR_SCHUR_FACTORED:
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,(PetscScalar*)mumps->id.redrhs,&X);CHKERRQ(ierr);
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      ierr = MatMatSolveTranspose(S,B,X);CHKERRQ(ierr);
    } else {
      ierr = MatMatSolve(S,B,X);CHKERRQ(ierr);
    }
    break;
  case MAT_FACTOR_SCHUR_INVERTED:
    sizesol = mumps->id.nrhs*mumps->id.size_schur;
    if (!mumps->schur_sol || sizesol > mumps->schur_sizesol) {
      ierr = PetscFree(mumps->schur_sol);CHKERRQ(ierr);
      ierr = PetscMalloc1(sizesol,&mumps->schur_sol);CHKERRQ(ierr);
      mumps->schur_sizesol = sizesol;
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,mumps->schur_sol,&X);CHKERRQ(ierr);
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      ierr = MatTransposeMatMult(S,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    } else {
      ierr = MatMatMult(S,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    }
    ierr = MatCopy(X,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %D",F->schur_status);
    break;
  }
  ierr = MatFactorRestoreSchurComplement(F,&S,schurstatus);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMumpsHandleSchur_Private(Mat F, PetscBool expansion)
{
  Mat_MUMPS     *mumps=(Mat_MUMPS*)F->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mumps->id.ICNTL(19)) { /* do nothing when Schur complement has not been computed */
    PetscFunctionReturn(0);
  }
  if (!expansion) { /* prepare for the condensation step */
    PetscInt sizeredrhs = mumps->id.nrhs*mumps->id.size_schur;
    /* allocate MUMPS internal array to store reduced right-hand sides */
    if (!mumps->id.redrhs || sizeredrhs > mumps->sizeredrhs) {
      ierr = PetscFree(mumps->id.redrhs);CHKERRQ(ierr);
      mumps->id.lredrhs = mumps->id.size_schur;
      ierr = PetscMalloc1(mumps->id.nrhs*mumps->id.lredrhs,&mumps->id.redrhs);CHKERRQ(ierr);
      mumps->sizeredrhs = mumps->id.nrhs*mumps->id.lredrhs;
    }
    mumps->id.ICNTL(26) = 1; /* condensation phase */
  } else { /* prepare for the expansion step */
    /* solve Schur complement (this has to be done by the MUMPS user, so basically us) */
    ierr = MatMumpsSolveSchur_Private(F);CHKERRQ(ierr);
    mumps->id.ICNTL(26) = 2; /* expansion phase */
    PetscMUMPS_c(&mumps->id);
    if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));
    /* restore defaults */
    mumps->id.ICNTL(26) = -1;
    /* free MUMPS internal array for redrhs if we have solved for multiple rhs in order to save memory space */
    if (mumps->id.nrhs > 1) {
      ierr = PetscFree(mumps->id.redrhs);CHKERRQ(ierr);
      mumps->id.lredrhs = 0;
      mumps->sizeredrhs = 0;
    }
  }
  PetscFunctionReturn(0);
}

/*
  MatConvertToTriples_A_B - convert Petsc matrix to triples: row[nz], col[nz], val[nz] 

  input:
    A       - matrix in aij,baij or sbaij (bs=1) format
    shift   - 0: C style output triple; 1: Fortran style output triple.
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple
              MAT_REUSE_MATRIX:   only the values in v array are updated
  output:
    nnz     - dim of r, c, and v (number of local nonzero entries of A)
    r, c, v - row and col index, matrix values (matrix triples)

  The returned values r, c, and sometimes v are obtained in a single PetscMalloc(). Then in MatDestroy_MUMPS() it is
  freed with PetscFree((mumps->irn);  This is not ideal code, the fact that v is ONLY sometimes part of mumps->irn means
  that the PetscMalloc() cannot easily be replaced with a PetscMalloc3(). 

 */

PetscErrorCode MatConvertToTriples_seqaij_seqaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  const PetscInt *ai,*aj,*ajj,M=A->rmap->n;
  PetscInt       nz,rnz,i,j;
  PetscErrorCode ierr;
  PetscInt       *row,*col;
  Mat_SeqAIJ     *aa=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  *v=aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz;
    ai   = aa->i;
    aj   = aa->j;
    *nnz = nz;
    ierr = PetscMalloc1(2*nz, &row);CHKERRQ(ierr);
    col  = row + nz;

    nz = 0;
    for (i=0; i<M; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for (j=0; j<rnz; j++) {
        row[nz] = i+shift; col[nz++] = ajj[j] + shift;
      }
    }
    *r = row; *c = col;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqbaij_seqaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  Mat_SeqBAIJ    *aa=(Mat_SeqBAIJ*)A->data;
  const PetscInt *ai,*aj,*ajj,bs2 = aa->bs2;
  PetscInt       bs,M,nz,idx=0,rnz,i,j,k,m;
  PetscErrorCode ierr;
  PetscInt       *row,*col;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  M = A->rmap->N/bs;
  *v = aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    ai   = aa->i; aj = aa->j;
    nz   = bs2*aa->nz;
    *nnz = nz;
    ierr = PetscMalloc1(2*nz, &row);CHKERRQ(ierr);
    col  = row + nz;

    for (i=0; i<M; i++) {
      ajj = aj + ai[i];
      rnz = ai[i+1] - ai[i];
      for (k=0; k<rnz; k++) {
        for (j=0; j<bs; j++) {
          for (m=0; m<bs; m++) {
            row[idx]   = i*bs + m + shift;
            col[idx++] = bs*(ajj[k]) + j + shift;
          }
        }
      }
    }
    *r = row; *c = col;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqsbaij_seqsbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  const PetscInt *ai, *aj,*ajj,M=A->rmap->n;
  PetscInt       nz,rnz,i,j;
  PetscErrorCode ierr;
  PetscInt       *row,*col;
  Mat_SeqSBAIJ   *aa=(Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  *v = aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz;
    ai   = aa->i;
    aj   = aa->j;
    *v   = aa->a;
    *nnz = nz;
    ierr = PetscMalloc1(2*nz, &row);CHKERRQ(ierr);
    col  = row + nz;

    nz = 0;
    for (i=0; i<M; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for (j=0; j<rnz; j++) {
        row[nz] = i+shift; col[nz++] = ajj[j] + shift;
      }
    }
    *r = row; *c = col;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqaij_seqsbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  const PetscInt    *ai,*aj,*ajj,*adiag,M=A->rmap->n;
  PetscInt          nz,rnz,i,j;
  const PetscScalar *av,*v1;
  PetscScalar       *val;
  PetscErrorCode    ierr;
  PetscInt          *row,*col;
  Mat_SeqAIJ        *aa=(Mat_SeqAIJ*)A->data;
  PetscBool         missing;

  PetscFunctionBegin;
  ai   =aa->i; aj=aa->j;av=aa->a;
  adiag=aa->diag;
  ierr  = MatMissingDiagonal_SeqAIJ(A,&missing,&i);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    /* count nz in the uppper triangular part of A */
    nz = 0;
    if (missing) {
      for (i=0; i<M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i+1])) {
          for (j=ai[i];j<ai[i+1];j++) {
            if (aj[j] < i) continue;
            nz++;
          }
        } else {
          nz += ai[i+1] - adiag[i];
        }
      }
    } else {
      for (i=0; i<M; i++) nz += ai[i+1] - adiag[i];
    }
    *nnz = nz;

    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    nz = 0;
    if (missing) {
      for (i=0; i<M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i+1])) {
          for (j=ai[i];j<ai[i+1];j++) {
            if (aj[j] < i) continue;
            row[nz] = i+shift;
            col[nz] = aj[j]+shift;
            val[nz] = av[j];
            nz++;
          }
        } else {
          rnz = ai[i+1] - adiag[i];
          ajj = aj + adiag[i];
          v1  = av + adiag[i];
          for (j=0; j<rnz; j++) {
            row[nz] = i+shift; col[nz] = ajj[j] + shift; val[nz++] = v1[j];
          }
        }
      }
    } else {
      for (i=0; i<M; i++) {
        rnz = ai[i+1] - adiag[i];
        ajj = aj + adiag[i];
        v1  = av + adiag[i];
        for (j=0; j<rnz; j++) {
          row[nz] = i+shift; col[nz] = ajj[j] + shift; val[nz++] = v1[j];
        }
      }
    }
    *r = row; *c = col; *v = val;
  } else {
    nz = 0; val = *v;
    if (missing) {
      for (i=0; i <M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i+1])) {
          for (j=ai[i];j<ai[i+1];j++) {
            if (aj[j] < i) continue;
            val[nz++] = av[j];
          }
        } else {
          rnz = ai[i+1] - adiag[i];
          v1  = av + adiag[i];
          for (j=0; j<rnz; j++) {
            val[nz++] = v1[j];
          }
        }
      }
    } else {
      for (i=0; i <M; i++) {
        rnz = ai[i+1] - adiag[i];
        v1  = av + adiag[i];
        for (j=0; j<rnz; j++) {
          val[nz++] = v1[j];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpisbaij_mpisbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  const PetscInt    *ai, *aj, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode    ierr;
  PetscInt          rstart,nz,i,j,jj,irow,countA,countB;
  PetscInt          *row,*col;
  const PetscScalar *av, *bv,*v1,*v2;
  PetscScalar       *val;
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ      *aa  = (Mat_SeqSBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ       *bb  = (Mat_SeqBAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= A->rmap->rstart;
  av=aa->a; bv=bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  jj = 0; irow = rstart;
  for (i=0; i<m; i++) {
    ajj    = aj + ai[i];                 /* ptr to the beginning of this row */
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j=0; j<countA; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift;
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      }
      val[jj++] = v2[j];
    }
    irow++;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpiaij_mpiaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  const PetscInt    *ai, *aj, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode    ierr;
  PetscInt          rstart,nz,i,j,jj,irow,countA,countB;
  PetscInt          *row,*col;
  const PetscScalar *av, *bv,*v1,*v2;
  PetscScalar       *val;
  Mat_MPIAIJ        *mat = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ        *aa  = (Mat_SeqAIJ*)(mat->A)->data;
  Mat_SeqAIJ        *bb  = (Mat_SeqAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= A->rmap->rstart;
  av=aa->a; bv=bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  jj = 0; irow = rstart;
  for (i=0; i<m; i++) {
    ajj    = aj + ai[i];                 /* ptr to the beginning of this row */
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j=0; j<countA; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift;
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      }
      val[jj++] = v2[j];
    }
    irow++;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpibaij_mpiaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  Mat_MPIBAIJ       *mat    = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ       *aa     = (Mat_SeqBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ       *bb     = (Mat_SeqBAIJ*)(mat->B)->data;
  const PetscInt    *ai     = aa->i, *bi = bb->i, *aj = aa->j, *bj = bb->j,*ajj, *bjj;
  const PetscInt    *garray = mat->garray,mbs=mat->mbs,rstart=A->rmap->rstart;
  const PetscInt    bs2=mat->bs2;
  PetscErrorCode    ierr;
  PetscInt          bs,nz,i,j,k,n,jj,irow,countA,countB,idx;
  PetscInt          *row,*col;
  const PetscScalar *av=aa->a, *bv=bb->a,*v1,*v2;
  PetscScalar       *val;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = bs2*(aa->nz + bb->nz);
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  jj = 0; irow = rstart;
  for (i=0; i<mbs; i++) {
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    ajj    = aj + ai[i];
    bjj    = bj + bi[i];
    v1     = av + bs2*ai[i];
    v2     = bv + bs2*bi[i];

    idx = 0;
    /* A-part */
    for (k=0; k<countA; k++) {
      for (j=0; j<bs; j++) {
        for (n=0; n<bs; n++) {
          if (reuse == MAT_INITIAL_MATRIX) {
            row[jj] = irow + n + shift;
            col[jj] = rstart + bs*ajj[k] + j + shift;
          }
          val[jj++] = v1[idx++];
        }
      }
    }

    idx = 0;
    /* B-part */
    for (k=0; k<countB; k++) {
      for (j=0; j<bs; j++) {
        for (n=0; n<bs; n++) {
          if (reuse == MAT_INITIAL_MATRIX) {
            row[jj] = irow + n + shift;
            col[jj] = bs*garray[bjj[k]] + j + shift;
          }
          val[jj++] = v2[idx++];
        }
      }
    }
    irow += bs;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpiaij_mpisbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v)
{
  const PetscInt    *ai, *aj,*adiag, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode    ierr;
  PetscInt          rstart,nz,nza,nzb,i,j,jj,irow,countA,countB;
  PetscInt          *row,*col;
  const PetscScalar *av, *bv,*v1,*v2;
  PetscScalar       *val;
  Mat_MPIAIJ        *mat =  (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ        *aa  =(Mat_SeqAIJ*)(mat->A)->data;
  Mat_SeqAIJ        *bb  =(Mat_SeqAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; adiag=aa->diag;
  bi=bb->i; bj=bb->j; garray = mat->garray;
  av=aa->a; bv=bb->a;

  rstart = A->rmap->rstart;

  if (reuse == MAT_INITIAL_MATRIX) {
    nza = 0;    /* num of upper triangular entries in mat->A, including diagonals */
    nzb = 0;    /* num of upper triangular entries in mat->B */
    for (i=0; i<m; i++) {
      nza   += (ai[i+1] - adiag[i]);
      countB = bi[i+1] - bi[i];
      bjj    = bj + bi[i];
      for (j=0; j<countB; j++) {
        if (garray[bjj[j]] > rstart) nzb++;
      }
    }

    nz   = nza + nzb; /* total nz of upper triangular part of mat */
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v;
  }

  jj = 0; irow = rstart;
  for (i=0; i<m; i++) {
    ajj    = aj + adiag[i];                 /* ptr to the beginning of the diagonal of this row */
    v1     = av + adiag[i];
    countA = ai[i+1] - adiag[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j=0; j<countA; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift;
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (garray[bjj[j]] > rstart) {
        if (reuse == MAT_INITIAL_MATRIX) {
          row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
        }
        val[jj++] = v2[j];
      }
    }
    irow++;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MUMPS(Mat A)
{
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(mumps->id.sol_loc,mumps->id.isol_loc);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mumps->scat_rhs);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mumps->scat_sol);CHKERRQ(ierr);
  ierr = VecDestroy(&mumps->b_seq);CHKERRQ(ierr);
  ierr = VecDestroy(&mumps->x_seq);CHKERRQ(ierr);
  ierr = PetscFree(mumps->id.perm_in);CHKERRQ(ierr);
  ierr = PetscFree(mumps->irn);CHKERRQ(ierr);
  ierr = PetscFree(mumps->info);CHKERRQ(ierr);
  ierr = MatMumpsResetSchur_Private(mumps);CHKERRQ(ierr);
  mumps->id.job = JOB_END;
  PetscMUMPS_c(&mumps->id);
  ierr = MPI_Comm_free(&mumps->comm_mumps);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorSetSchurIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorCreateSchurComplement_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsSetIcntl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetIcntl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsSetCntl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetCntl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInfo_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInfog_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetRinfo_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetRinfog_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_MUMPS(Mat A,Vec b,Vec x)
{
  Mat_MUMPS        *mumps=(Mat_MUMPS*)A->data;
  PetscScalar      *array;
  Vec              b_seq;
  IS               is_iden,is_petsc;
  PetscErrorCode   ierr;
  PetscInt         i;
  PetscBool        second_solve = PETSC_FALSE;
  static PetscBool cite1 = PETSC_FALSE,cite2 = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister("@article{MUMPS01,\n  author = {P.~R. Amestoy and I.~S. Duff and J.-Y. L'Excellent and J. Koster},\n  title = {A fully asynchronous multifrontal solver using distributed dynamic scheduling},\n  journal = {SIAM Journal on Matrix Analysis and Applications},\n  volume = {23},\n  number = {1},\n  pages = {15--41},\n  year = {2001}\n}\n",&cite1);CHKERRQ(ierr);
  ierr = PetscCitationsRegister("@article{MUMPS02,\n  author = {P.~R. Amestoy and A. Guermouche and J.-Y. L'Excellent and S. Pralet},\n  title = {Hybrid scheduling for the parallel solution of linear systems},\n  journal = {Parallel Computing},\n  volume = {32},\n  number = {2},\n  pages = {136--156},\n  year = {2006}\n}\n",&cite2);CHKERRQ(ierr);

  if (A->factorerrortype) {
    ierr = PetscInfo2(A,"MatSolve is called with singular matrix factor, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
    ierr = VecSetInf(x);CHKERRQ(ierr); 
    PetscFunctionReturn(0);
  }

  mumps->id.nrhs = 1;
  b_seq          = mumps->b_seq;
  if (mumps->size > 1) {
    /* MUMPS only supports centralized rhs. Scatter b into a seqential rhs vector */
    ierr = VecScatterBegin(mumps->scat_rhs,b,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mumps->scat_rhs,b,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (!mumps->myid) {ierr = VecGetArray(b_seq,&array);CHKERRQ(ierr);}
  } else {  /* size == 1 */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  }
  if (!mumps->myid) { /* define rhs on the host */
    mumps->id.nrhs = 1;
    mumps->id.rhs = (MumpsScalar*)array;
  }

  /*
     handle condensation step of Schur complement (if any)
     We set by default ICNTL(26) == -1 when Schur indices have been provided by the user.
     According to MUMPS (5.0.0) manual, any value should be harmful during the factorization phase
     Unless the user provides a valid value for ICNTL(26), MatSolve and MatMatSolve routines solve the full system.
     This requires an extra call to PetscMUMPS_c and the computation of the factors for S
  */
  if (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2) {
    if (mumps->size > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Parallel Schur complements not yet supported from PETSc\n");
    second_solve = PETSC_TRUE;
    ierr = MatMumpsHandleSchur_Private(A,PETSC_FALSE);CHKERRQ(ierr);
  }
  /* solve phase */
  /*-------------*/
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(&mumps->id);
  if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));

  /* handle expansion step of Schur complement (if any) */
  if (second_solve) {
    ierr = MatMumpsHandleSchur_Private(A,PETSC_TRUE);CHKERRQ(ierr);
  }

  if (mumps->size > 1) { /* convert mumps distributed solution to petsc mpi x */
    if (mumps->scat_sol && mumps->ICNTL9_pre != mumps->id.ICNTL(9)) {
      /* when id.ICNTL(9) changes, the contents of lsol_loc may change (not its size, lsol_loc), recreates scat_sol */
      ierr = VecScatterDestroy(&mumps->scat_sol);CHKERRQ(ierr);
    }
    if (!mumps->scat_sol) { /* create scatter scat_sol */
      ierr = ISCreateStride(PETSC_COMM_SELF,mumps->id.lsol_loc,0,1,&is_iden);CHKERRQ(ierr); /* from */
      for (i=0; i<mumps->id.lsol_loc; i++) {
        mumps->id.isol_loc[i] -= 1; /* change Fortran style to C style */
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,mumps->id.lsol_loc,mumps->id.isol_loc,PETSC_COPY_VALUES,&is_petsc);CHKERRQ(ierr);  /* to */
      ierr = VecScatterCreate(mumps->x_seq,is_iden,x,is_petsc,&mumps->scat_sol);CHKERRQ(ierr);
      ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
      ierr = ISDestroy(&is_petsc);CHKERRQ(ierr);

      mumps->ICNTL9_pre = mumps->id.ICNTL(9); /* save current value of id.ICNTL(9) */
    }

    ierr = VecScatterBegin(mumps->scat_sol,mumps->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mumps->scat_sol,mumps->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_MUMPS(Mat A,Vec b,Vec x)
{
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mumps->id.ICNTL(9) = 0;
  ierr = MatSolve_MUMPS(A,b,x);CHKERRQ(ierr);
  mumps->id.ICNTL(9) = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatSolve_MUMPS(Mat A,Mat B,Mat X)
{
  PetscErrorCode ierr;
  Mat            Bt = NULL;
  PetscBool      flg, flgT;
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->data;
  PetscInt       i,nrhs,M;
  PetscScalar    *array,*bray;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&flgT);CHKERRQ(ierr);
  if (flgT) {
    if (mumps->size > 1) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
    ierr = MatTransposeGetMat(B,&Bt);CHKERRQ(ierr);
  } else {
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  if (B->rmap->n != X->rmap->n) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONG,"Matrix B and X must have same row distribution");

  ierr = MatGetSize(B,&M,&nrhs);CHKERRQ(ierr);
  mumps->id.nrhs = nrhs;
  mumps->id.lrhs = M;

  if (mumps->size == 1) {
    PetscScalar *aa;
    PetscInt    spnr,*ia,*ja;
    PetscBool   second_solve = PETSC_FALSE;

    /* copy B to X */
    ierr = MatDenseGetArray(X,&array);CHKERRQ(ierr);
    mumps->id.rhs = (MumpsScalar*)array;
    if (!Bt) {
      ierr = MatDenseGetArray(B,&bray);CHKERRQ(ierr);
      ierr = PetscMemcpy(array,bray,M*nrhs*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(B,&bray);CHKERRQ(ierr);
    } else {
      PetscBool done;

      ierr = MatSeqAIJGetArray(Bt,&aa);CHKERRQ(ierr);
      ierr = MatGetRowIJ(Bt,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&done);CHKERRQ(ierr);
      if (!done) SETERRQ(PetscObjectComm((PetscObject)Bt),PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
      mumps->id.irhs_ptr    = ia;
      mumps->id.irhs_sparse = ja;
      mumps->id.nz_rhs      = ia[spnr] - 1;
      mumps->id.rhs_sparse  = (MumpsScalar*)aa;
      mumps->id.ICNTL(20)   = 1;
    }
    /* handle condensation step of Schur complement (if any) */
    if (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2) {
      second_solve = PETSC_TRUE;
      ierr = MatMumpsHandleSchur_Private(A,PETSC_FALSE);CHKERRQ(ierr);
    }
    /* solve phase */
    /*-------------*/
    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(&mumps->id);
    if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));

    /* handle expansion step of Schur complement (if any) */
    if (second_solve) {
      ierr = MatMumpsHandleSchur_Private(A,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (Bt) {
      PetscBool done;

      ierr = MatSeqAIJRestoreArray(Bt,&aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(Bt,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&done);CHKERRQ(ierr);
      if (!done) SETERRQ(PetscObjectComm((PetscObject)Bt),PETSC_ERR_ARG_WRONG,"Cannot restore IJ structure");
      mumps->id.ICNTL(20) = 0;
    }
    ierr = MatDenseRestoreArray(X,&array);CHKERRQ(ierr);
  } else {  /*--------- parallel case --------*/
    PetscInt       lsol_loc,nlsol_loc,*isol_loc,*idx,*iidx,*idxx,*isol_loc_save;
    MumpsScalar    *sol_loc,*sol_loc_save;
    IS             is_to,is_from;
    PetscInt       k,proc,j,m;
    const PetscInt *rstart;
    Vec            v_mpi,b_seq,x_seq;
    VecScatter     scat_rhs,scat_sol;

    if (mumps->size > 1 && mumps->id.ICNTL(19)) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Parallel Schur complements not yet supported from PETSc\n");

    /* create x_seq to hold local solution */
    isol_loc_save = mumps->id.isol_loc; /* save it for MatSovle() */
    sol_loc_save  = mumps->id.sol_loc;

    lsol_loc  = mumps->id.INFO(23); 
    nlsol_loc = nrhs*lsol_loc;     /* length of sol_loc */
    ierr = PetscMalloc2(nlsol_loc,&sol_loc,nlsol_loc,&isol_loc);CHKERRQ(ierr);
    mumps->id.sol_loc = (MumpsScalar*)sol_loc;
    mumps->id.isol_loc = isol_loc;

    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nlsol_loc,(PetscScalar*)sol_loc,&x_seq);CHKERRQ(ierr);

    /* copy rhs matrix B into vector v_mpi */
    ierr = MatGetLocalSize(B,&m,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(B,&bray);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),1,nrhs*m,nrhs*M,(const PetscScalar*)bray,&v_mpi);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(B,&bray);CHKERRQ(ierr);

    /* scatter v_mpi to b_seq because MUMPS only supports centralized rhs */
    /* idx: maps from k-th index of v_mpi to (i,j)-th global entry of B;
      iidx: inverse of idx, will be used by scattering xx_seq -> X       */
    ierr = PetscMalloc2(nrhs*M,&idx,nrhs*M,&iidx);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(B,&rstart);CHKERRQ(ierr);
    k = 0;
    for (proc=0; proc<mumps->size; proc++){
      for (j=0; j<nrhs; j++){
        for (i=rstart[proc]; i<rstart[proc+1]; i++){
          iidx[j*M + i] = k;
          idx[k++]      = j*M + i; 
        }
      }
    }

    if (!mumps->myid) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,nrhs*M,&b_seq);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,nrhs*M,idx,PETSC_COPY_VALUES,&is_to);CHKERRQ(ierr); 
      ierr = ISCreateStride(PETSC_COMM_SELF,nrhs*M,0,1,&is_from);CHKERRQ(ierr); 
    } else {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_to);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_from);CHKERRQ(ierr);
    }
    ierr = VecScatterCreate(v_mpi,is_from,b_seq,is_to,&scat_rhs);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_rhs,v_mpi,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = ISDestroy(&is_to);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_rhs,v_mpi,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    if (!mumps->myid) { /* define rhs on the host */
      ierr = VecGetArray(b_seq,&bray);CHKERRQ(ierr);
      mumps->id.rhs = (MumpsScalar*)bray;
      ierr = VecRestoreArray(b_seq,&bray);CHKERRQ(ierr);
    }

    /* solve phase */
    /*-------------*/
    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(&mumps->id);
    if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));

    /* scatter mumps distributed solution to petsc vector v_mpi, which shares local arrays with solution matrix X */
    ierr = MatDenseGetArray(X,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(v_mpi,array);CHKERRQ(ierr);
    
    /* create scatter scat_sol */
    ierr = PetscMalloc1(nlsol_loc,&idxx);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,nlsol_loc,0,1,&is_from);CHKERRQ(ierr); 
    for (i=0; i<lsol_loc; i++) {
      isol_loc[i] -= 1; /* change Fortran style to C style */
      idxx[i] = iidx[isol_loc[i]]; 
      for (j=1; j<nrhs; j++){
        idxx[j*lsol_loc+i] = iidx[isol_loc[i]+j*M];
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nlsol_loc,idxx,PETSC_COPY_VALUES,&is_to);CHKERRQ(ierr);  
    ierr = VecScatterCreate(x_seq,is_from,v_mpi,is_to,&scat_sol);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_sol,x_seq,v_mpi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from);CHKERRQ(ierr);
    ierr = ISDestroy(&is_to);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_sol,x_seq,v_mpi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(X,&array);CHKERRQ(ierr);

    /* free spaces */
    mumps->id.sol_loc = sol_loc_save;
    mumps->id.isol_loc = isol_loc_save;

    ierr = PetscFree2(sol_loc,isol_loc);CHKERRQ(ierr);
    ierr = PetscFree2(idx,iidx);CHKERRQ(ierr);
    ierr = PetscFree(idxx);CHKERRQ(ierr);
    ierr = VecDestroy(&x_seq);CHKERRQ(ierr);
    ierr = VecDestroy(&v_mpi);CHKERRQ(ierr);
    ierr = VecDestroy(&b_seq);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_rhs);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_sol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)
/*
  input:
   F:        numeric factor
  output:
   nneg:     total number of negative pivots
   nzero:    total number of zero pivots
   npos:     (global dimension of F) - nneg - nzero
*/
PetscErrorCode MatGetInertia_SBAIJMUMPS(Mat F,int *nneg,int *nzero,int *npos)
{
  Mat_MUMPS      *mumps =(Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)F),&size);CHKERRQ(ierr);
  /* MUMPS 4.3.1 calls ScaLAPACK when ICNTL(13)=0 (default), which does not offer the possibility to compute the inertia of a dense matrix. Set ICNTL(13)=1 to skip ScaLAPACK */
  if (size > 1 && mumps->id.ICNTL(13) != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ICNTL(13)=%d. -mat_mumps_icntl_13 must be set as 1 for correct global matrix inertia\n",mumps->id.INFOG(13));

  if (nneg) *nneg = mumps->id.INFOG(12); 
  if (nzero || npos) {
    if (mumps->id.ICNTL(24) != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"-mat_mumps_icntl_24 must be set as 1 for null pivot row detection");
    if (nzero) *nzero = mumps->id.INFOG(28);
    if (npos) *npos   = F->rmap->N - (mumps->id.INFOG(12) + mumps->id.INFOG(28));
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatFactorNumeric_MUMPS(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps =(Mat_MUMPS*)(F)->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0) {
    if (mumps->id.INFOG(1) == -6) {
      ierr = PetscInfo2(A,"MatFactorNumeric is called with singular matrix structure, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
    } 
    ierr = PetscInfo2(A,"MatFactorNumeric is called after analysis phase fails, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_REUSE_MATRIX, &mumps->nz, &mumps->irn, &mumps->jcn, &mumps->val);CHKERRQ(ierr);

  /* numerical factorization phase */
  /*-------------------------------*/
  mumps->id.job = JOB_FACTNUMERIC;
  if (!mumps->id.ICNTL(18)) { /* A is centralized */
    if (!mumps->myid) {
      mumps->id.a = (MumpsScalar*)mumps->val;
    }
  } else {
    mumps->id.a_loc = (MumpsScalar*)mumps->val;
  }
  PetscMUMPS_c(&mumps->id);
  if (mumps->id.INFOG(1) < 0) {
    if (A->erroriffailure) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));
    } else {
      if (mumps->id.INFOG(1) == -10) { /* numerically singular matrix */
        ierr = PetscInfo2(F,"matrix is numerically singular, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      } else if (mumps->id.INFOG(1) == -13) {
        ierr = PetscInfo2(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, cannot allocate required memory %d megabytes\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else if (mumps->id.INFOG(1) == -8 || mumps->id.INFOG(1) == -9 || (-16 < mumps->id.INFOG(1) && mumps->id.INFOG(1) < -10) ) {
        ierr = PetscInfo2(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d, problem with workarray \n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else {
        ierr = PetscInfo2(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OTHER;
      }
    }
  }
  if (!mumps->myid && mumps->id.ICNTL(16) > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"  mumps->id.ICNTL(16):=%d\n",mumps->id.INFOG(16));

  F->assembled    = PETSC_TRUE;
  mumps->matstruc = SAME_NONZERO_PATTERN;
  if (F->schur) { /* reset Schur status to unfactored */
    if (mumps->id.ICNTL(19) == 1) { /* stored by rows */
      mumps->id.ICNTL(19) = 2;
      ierr = MatTranspose(F->schur,MAT_INPLACE_MATRIX,&F->schur);CHKERRQ(ierr);
    }
    ierr = MatFactorRestoreSchurComplement(F,NULL,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
  }

  /* just to be sure that ICNTL(19) value returned by a call from MatMumpsGetIcntl is always consistent */
  if (!mumps->sym && mumps->id.ICNTL(19) && mumps->id.ICNTL(19) != 1) mumps->id.ICNTL(19) = 3;

  if (mumps->size > 1) {
    PetscInt    lsol_loc;
    PetscScalar *sol_loc;

    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);

    /* distributed solution; Create x_seq=sol_loc for repeated use */
    if (mumps->x_seq) {
      ierr = VecScatterDestroy(&mumps->scat_sol);CHKERRQ(ierr);
      ierr = PetscFree2(mumps->id.sol_loc,mumps->id.isol_loc);CHKERRQ(ierr);
      ierr = VecDestroy(&mumps->x_seq);CHKERRQ(ierr);
    }
    lsol_loc = mumps->id.INFO(23); /* length of sol_loc */
    ierr = PetscMalloc2(lsol_loc,&sol_loc,lsol_loc,&mumps->id.isol_loc);CHKERRQ(ierr);
    mumps->id.lsol_loc = lsol_loc;
    mumps->id.sol_loc = (MumpsScalar*)sol_loc;
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,lsol_loc,sol_loc,&mumps->x_seq);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Sets MUMPS options from the options database */
PetscErrorCode PetscSetMUMPSFromOptions(Mat F, Mat A)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  PetscInt       icntl,info[40],i,ninfo=40;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MUMPS Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_1","ICNTL(1): output stream for error messages","None",mumps->id.ICNTL(1),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(1) = icntl;
  ierr = PetscOptionsInt("-mat_mumps_icntl_2","ICNTL(2): output stream for diagnostic printing, statistics, and warning","None",mumps->id.ICNTL(2),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(2) = icntl;
  ierr = PetscOptionsInt("-mat_mumps_icntl_3","ICNTL(3): output stream for global information, collected on the host","None",mumps->id.ICNTL(3),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(3) = icntl;

  ierr = PetscOptionsInt("-mat_mumps_icntl_4","ICNTL(4): level of printing (0 to 4)","None",mumps->id.ICNTL(4),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(4) = icntl;
  if (mumps->id.ICNTL(4) || PetscLogPrintInfo) mumps->id.ICNTL(3) = 6; /* resume MUMPS default id.ICNTL(3) = 6 */

  ierr = PetscOptionsInt("-mat_mumps_icntl_6","ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7)","None",mumps->id.ICNTL(6),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(6) = icntl;

  ierr = PetscOptionsInt("-mat_mumps_icntl_7","ICNTL(7): computes a symmetric permutation in sequential analysis (0 to 7). 3=Scotch, 4=PORD, 5=Metis","None",mumps->id.ICNTL(7),&icntl,&flg);CHKERRQ(ierr);
  if (flg) {
    if (icntl== 1 && mumps->size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"pivot order be set by the user in PERM_IN -- not supported by the PETSc/MUMPS interface\n");
    else mumps->id.ICNTL(7) = icntl;
  }

  ierr = PetscOptionsInt("-mat_mumps_icntl_8","ICNTL(8): scaling strategy (-2 to 8 or 77)","None",mumps->id.ICNTL(8),&mumps->id.ICNTL(8),NULL);CHKERRQ(ierr);
  /* ierr = PetscOptionsInt("-mat_mumps_icntl_9","ICNTL(9): computes the solution using A or A^T","None",mumps->id.ICNTL(9),&mumps->id.ICNTL(9),NULL);CHKERRQ(ierr); handled by MatSolveTranspose_MUMPS() */
  ierr = PetscOptionsInt("-mat_mumps_icntl_10","ICNTL(10): max num of refinements","None",mumps->id.ICNTL(10),&mumps->id.ICNTL(10),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_11","ICNTL(11): statistics related to an error analysis (via -ksp_view)","None",mumps->id.ICNTL(11),&mumps->id.ICNTL(11),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_12","ICNTL(12): an ordering strategy for symmetric matrices (0 to 3)","None",mumps->id.ICNTL(12),&mumps->id.ICNTL(12),NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-mat_mumps_icntl_13","ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting","None",mumps->id.ICNTL(13),&mumps->id.ICNTL(13),NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-mat_mumps_icntl_14","ICNTL(14): percentage increase in the estimated working space","None",mumps->id.ICNTL(14),&mumps->id.ICNTL(14),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_19","ICNTL(19): computes the Schur complement","None",mumps->id.ICNTL(19),&mumps->id.ICNTL(19),NULL);CHKERRQ(ierr);
  if (mumps->id.ICNTL(19) <= 0 || mumps->id.ICNTL(19) > 3) { /* reset any schur data (if any) */
    ierr = MatDestroy(&F->schur);CHKERRQ(ierr);
    ierr = MatMumpsResetSchur_Private(mumps);CHKERRQ(ierr);
  }
  /* ierr = PetscOptionsInt("-mat_mumps_icntl_20","ICNTL(20): the format (dense or sparse) of the right-hand sides","None",mumps->id.ICNTL(20),&mumps->id.ICNTL(20),NULL);CHKERRQ(ierr); -- sparse rhs is not supported in PETSc API */
  /* ierr = PetscOptionsInt("-mat_mumps_icntl_21","ICNTL(21): the distribution (centralized or distributed) of the solution vectors","None",mumps->id.ICNTL(21),&mumps->id.ICNTL(21),NULL);CHKERRQ(ierr); we only use distributed solution vector */

  ierr = PetscOptionsInt("-mat_mumps_icntl_22","ICNTL(22): in-core/out-of-core factorization and solve (0 or 1)","None",mumps->id.ICNTL(22),&mumps->id.ICNTL(22),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_23","ICNTL(23): max size of the working memory (MB) that can allocate per processor","None",mumps->id.ICNTL(23),&mumps->id.ICNTL(23),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_24","ICNTL(24): detection of null pivot rows (0 or 1)","None",mumps->id.ICNTL(24),&mumps->id.ICNTL(24),NULL);CHKERRQ(ierr);
  if (mumps->id.ICNTL(24)) {
    mumps->id.ICNTL(13) = 1; /* turn-off ScaLAPACK to help with the correct detection of null pivots */
  }

  ierr = PetscOptionsInt("-mat_mumps_icntl_25","ICNTL(25): compute a solution of a deficient matrix and a null space basis","None",mumps->id.ICNTL(25),&mumps->id.ICNTL(25),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_26","ICNTL(26): drives the solution phase if a Schur complement matrix","None",mumps->id.ICNTL(26),&mumps->id.ICNTL(26),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_27","ICNTL(27): the blocking size for multiple right-hand sides","None",mumps->id.ICNTL(27),&mumps->id.ICNTL(27),NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-mat_mumps_icntl_28","ICNTL(28): use 1 for sequential analysis and ictnl(7) ordering, or 2 for parallel analysis and ictnl(29) ordering","None",mumps->id.ICNTL(28),&mumps->id.ICNTL(28),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_29","ICNTL(29): parallel ordering 1 = ptscotch, 2 = parmetis","None",mumps->id.ICNTL(29),&mumps->id.ICNTL(29),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_30","ICNTL(30): compute user-specified set of entries in inv(A)","None",mumps->id.ICNTL(30),&mumps->id.ICNTL(30),NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-mat_mumps_icntl_31","ICNTL(31): indicates which factors may be discarded during factorization","None",mumps->id.ICNTL(31),&mumps->id.ICNTL(31),NULL);CHKERRQ(ierr);
  /* ierr = PetscOptionsInt("-mat_mumps_icntl_32","ICNTL(32): performs the forward elemination of the right-hand sides during factorization","None",mumps->id.ICNTL(32),&mumps->id.ICNTL(32),NULL);CHKERRQ(ierr);  -- not supported by PETSc API */
  ierr = PetscOptionsInt("-mat_mumps_icntl_33","ICNTL(33): compute determinant","None",mumps->id.ICNTL(33),&mumps->id.ICNTL(33),NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-mat_mumps_cntl_1","CNTL(1): relative pivoting threshold","None",mumps->id.CNTL(1),&mumps->id.CNTL(1),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_2","CNTL(2): stopping criterion of refinement","None",mumps->id.CNTL(2),&mumps->id.CNTL(2),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_3","CNTL(3): absolute pivoting threshold","None",mumps->id.CNTL(3),&mumps->id.CNTL(3),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_4","CNTL(4): value for static pivoting","None",mumps->id.CNTL(4),&mumps->id.CNTL(4),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_5","CNTL(5): fixation for null pivots","None",mumps->id.CNTL(5),&mumps->id.CNTL(5),NULL);CHKERRQ(ierr);

  ierr = PetscOptionsString("-mat_mumps_ooc_tmpdir", "out of core directory", "None", mumps->id.ooc_tmpdir, mumps->id.ooc_tmpdir, 256, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsIntArray("-mat_mumps_view_info","request INFO local to each processor","",info,&ninfo,NULL);CHKERRQ(ierr);
  if (ninfo) {
    if (ninfo > 40) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"number of INFO %d must <= 40\n",ninfo);
    ierr = PetscMalloc1(ninfo,&mumps->info);CHKERRQ(ierr);
    mumps->ninfo = ninfo;
    for (i=0; i<ninfo; i++) {
      if (info[i] < 0 || info[i]>40) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index of INFO %d must between 1 and 40\n",ninfo);
      else  mumps->info[i] = info[i];
    }
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscInitializeMUMPS(Mat A,Mat_MUMPS *mumps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A), &mumps->myid);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&mumps->size);CHKERRQ(ierr);
  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)A),&(mumps->comm_mumps));CHKERRQ(ierr);

  mumps->id.comm_fortran = MPI_Comm_c2f(mumps->comm_mumps);

  mumps->id.job = JOB_INIT;
  mumps->id.par = 1;  /* host participates factorizaton and solve */
  mumps->id.sym = mumps->sym;
  PetscMUMPS_c(&mumps->id);

  mumps->scat_rhs     = NULL;
  mumps->scat_sol     = NULL;

  /* set PETSc-MUMPS default options - override MUMPS default */
  mumps->id.ICNTL(3) = 0;
  mumps->id.ICNTL(4) = 0;
  if (mumps->size == 1) {
    mumps->id.ICNTL(18) = 0;   /* centralized assembled matrix input */
  } else {
    mumps->id.ICNTL(18) = 3;   /* distributed assembled matrix input */
    mumps->id.ICNTL(20) = 0;   /* rhs is in dense format */
    mumps->id.ICNTL(21) = 1;   /* distributed solution */
  }

  /* schur */
  mumps->id.size_schur      = 0;
  mumps->id.listvar_schur   = NULL;
  mumps->id.schur           = NULL;
  mumps->sizeredrhs         = 0;
  mumps->schur_sol          = NULL;
  mumps->schur_sizesol      = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorSymbolic_MUMPS_ReportIfError(Mat F,Mat A,const MatFactorInfo *info,Mat_MUMPS *mumps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0) {
    if (A->erroriffailure) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",mumps->id.INFOG(1));
    } else {
      if (mumps->id.INFOG(1) == -6) {
        ierr = PetscInfo2(F,"matrix is singular in structure, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_STRUCT_ZEROPIVOT;
      } else if (mumps->id.INFOG(1) == -5 || mumps->id.INFOG(1) == -7) {
        ierr = PetscInfo2(F,"problem of workspace, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else {
        ierr = PetscInfo2(F,"Error reported by MUMPS in analysis phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OTHER;
      }
    } 
  }
  PetscFunctionReturn(0);
}

/* Note Petsc r(=c) permutation is used when mumps->id.ICNTL(7)==1 with centralized assembled matrix input; otherwise r and c are ignored */
PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  Vec            b;
  IS             is_iden;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, &mumps->nz, &mumps->irn, &mumps->jcn, &mumps->val);CHKERRQ(ierr);

  /* analysis phase */
  /*----------------*/
  mumps->id.job = JOB_FACTSYMBOLIC;
  mumps->id.n   = M;
  switch (mumps->id.ICNTL(18)) {
  case 0:  /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nz =mumps->nz; mumps->id.irn=mumps->irn; mumps->id.jcn=mumps->jcn;
      if (mumps->id.ICNTL(6)>1) {
        mumps->id.a = (MumpsScalar*)mumps->val;
      }
      if (mumps->id.ICNTL(7) == 1) { /* use user-provide matrix ordering - assuming r = c ordering */
        /*
        PetscBool      flag;
        ierr = ISEqual(r,c,&flag);CHKERRQ(ierr);
        if (!flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"row_perm != col_perm");
        ierr = ISView(r,PETSC_VIEWER_STDOUT_SELF);
         */
        if (!mumps->myid) {
          const PetscInt *idx;
          PetscInt       i,*perm_in;

          ierr = PetscMalloc1(M,&perm_in);CHKERRQ(ierr);
          ierr = ISGetIndices(r,&idx);CHKERRQ(ierr);

          mumps->id.perm_in = perm_in;
          for (i=0; i<M; i++) perm_in[i] = idx[i]+1; /* perm_in[]: start from 1, not 0! */
          ierr = ISRestoreIndices(r,&idx);CHKERRQ(ierr);
        }
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    mumps->id.nz_loc = mumps->nz;
    mumps->id.irn_loc=mumps->irn; mumps->id.jcn_loc=mumps->jcn;
    if (mumps->id.ICNTL(6)>1) {
      mumps->id.a_loc = (MumpsScalar*)mumps->val;
    }
    /* MUMPS only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
    if (!mumps->myid) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,A->rmap->N,&mumps->b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,A->rmap->N,0,1,&is_iden);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&mumps->b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_iden);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
    ierr = VecScatterCreate(b,is_iden,mumps->b_seq,is_iden,&mumps->scat_rhs);CHKERRQ(ierr);
    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    break;
  }
  PetscMUMPS_c(&mumps->id);
  ierr = MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps);CHKERRQ(ierr);

  F->ops->lufactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve           = MatSolve_MUMPS;
  F->ops->solvetranspose  = MatSolveTranspose_MUMPS;
  F->ops->matsolve        = MatMatSolve_MUMPS; 
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
PetscErrorCode MatLUFactorSymbolic_BAIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  Vec            b;
  IS             is_iden;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, &mumps->nz, &mumps->irn, &mumps->jcn, &mumps->val);CHKERRQ(ierr);

  /* analysis phase */
  /*----------------*/
  mumps->id.job = JOB_FACTSYMBOLIC;
  mumps->id.n   = M;
  switch (mumps->id.ICNTL(18)) {
  case 0:  /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nz =mumps->nz; mumps->id.irn=mumps->irn; mumps->id.jcn=mumps->jcn;
      if (mumps->id.ICNTL(6)>1) {
        mumps->id.a = (MumpsScalar*)mumps->val;
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    mumps->id.nz_loc = mumps->nz;
    mumps->id.irn_loc=mumps->irn; mumps->id.jcn_loc=mumps->jcn;
    if (mumps->id.ICNTL(6)>1) {
      mumps->id.a_loc = (MumpsScalar*)mumps->val;
    }
    /* MUMPS only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
    if (!mumps->myid) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,A->cmap->N,&mumps->b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,A->cmap->N,0,1,&is_iden);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&mumps->b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_iden);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
    ierr = VecScatterCreate(b,is_iden,mumps->b_seq,is_iden,&mumps->scat_rhs);CHKERRQ(ierr);
    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    break;
  }
  PetscMUMPS_c(&mumps->id);
  ierr = MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps);CHKERRQ(ierr);

  F->ops->lufactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve           = MatSolve_MUMPS;
  F->ops->solvetranspose  = MatSolveTranspose_MUMPS;
  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation and factor info are ignored */
PetscErrorCode MatCholeskyFactorSymbolic_MUMPS(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  Vec            b;
  IS             is_iden;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, &mumps->nz, &mumps->irn, &mumps->jcn, &mumps->val);CHKERRQ(ierr);

  /* analysis phase */
  /*----------------*/
  mumps->id.job = JOB_FACTSYMBOLIC;
  mumps->id.n   = M;
  switch (mumps->id.ICNTL(18)) {
  case 0:  /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nz =mumps->nz; mumps->id.irn=mumps->irn; mumps->id.jcn=mumps->jcn;
      if (mumps->id.ICNTL(6)>1) {
        mumps->id.a = (MumpsScalar*)mumps->val;
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    mumps->id.nz_loc = mumps->nz;
    mumps->id.irn_loc=mumps->irn; mumps->id.jcn_loc=mumps->jcn;
    if (mumps->id.ICNTL(6)>1) {
      mumps->id.a_loc = (MumpsScalar*)mumps->val;
    }
    /* MUMPS only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
    if (!mumps->myid) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,A->cmap->N,&mumps->b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,A->cmap->N,0,1,&is_iden);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&mumps->b_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_iden);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
    ierr = VecScatterCreate(b,is_iden,mumps->b_seq,is_iden,&mumps->scat_rhs);CHKERRQ(ierr);
    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    break;
  }
  PetscMUMPS_c(&mumps->id);
  ierr = MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps);CHKERRQ(ierr);

  F->ops->choleskyfactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve                 = MatSolve_MUMPS;
  F->ops->solvetranspose        = MatSolve_MUMPS;
  F->ops->matsolve              = MatMatSolve_MUMPS;
#if defined(PETSC_USE_COMPLEX)
  F->ops->getinertia = NULL;
#else
  F->ops->getinertia = MatGetInertia_SBAIJMUMPS;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MUMPS(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_MUMPS         *mumps=(Mat_MUMPS*)A->data;

  PetscFunctionBegin;
  /* check if matrix is mumps type */
  if (A->ops->solve != MatSolve_MUMPS) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"MUMPS run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  SYM (matrix type):                   %d \n",mumps->id.sym);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  PAR (host participation):            %d \n",mumps->id.par);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(1) (output for error):         %d \n",mumps->id.ICNTL(1));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(2) (output of diagnostic msg): %d \n",mumps->id.ICNTL(2));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(3) (output for global info):   %d \n",mumps->id.ICNTL(3));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(4) (level of printing):        %d \n",mumps->id.ICNTL(4));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(5) (input mat struct):         %d \n",mumps->id.ICNTL(5));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(6) (matrix prescaling):        %d \n",mumps->id.ICNTL(6));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(7) (sequential matrix ordering):%d \n",mumps->id.ICNTL(7));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(8) (scaling strategy):        %d \n",mumps->id.ICNTL(8));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(10) (max num of refinements):  %d \n",mumps->id.ICNTL(10));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(11) (error analysis):          %d \n",mumps->id.ICNTL(11));CHKERRQ(ierr);
      if (mumps->id.ICNTL(11)>0) {
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(4) (inf norm of input mat):        %g\n",mumps->id.RINFOG(4));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(5) (inf norm of solution):         %g\n",mumps->id.RINFOG(5));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(6) (inf norm of residual):         %g\n",mumps->id.RINFOG(6));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(7),RINFOG(8) (backward error est): %g, %g\n",mumps->id.RINFOG(7),mumps->id.RINFOG(8));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(9) (error estimate):               %g \n",mumps->id.RINFOG(9));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n",mumps->id.RINFOG(10),mumps->id.RINFOG(11));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(12) (efficiency control):                         %d \n",mumps->id.ICNTL(12));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(13) (efficiency control):                         %d \n",mumps->id.ICNTL(13));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(14) (percentage of estimated workspace increase): %d \n",mumps->id.ICNTL(14));CHKERRQ(ierr);
      /* ICNTL(15-17) not used */
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(18) (input mat struct):                           %d \n",mumps->id.ICNTL(18));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(19) (Schur complement info):                       %d \n",mumps->id.ICNTL(19));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(20) (rhs sparse pattern):                         %d \n",mumps->id.ICNTL(20));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(21) (solution struct):                            %d \n",mumps->id.ICNTL(21));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(22) (in-core/out-of-core facility):               %d \n",mumps->id.ICNTL(22));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(23) (max size of memory can be allocated locally):%d \n",mumps->id.ICNTL(23));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(24) (detection of null pivot rows):               %d \n",mumps->id.ICNTL(24));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(25) (computation of a null space basis):          %d \n",mumps->id.ICNTL(25));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(26) (Schur options for rhs or solution):          %d \n",mumps->id.ICNTL(26));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(27) (experimental parameter):                     %d \n",mumps->id.ICNTL(27));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(28) (use parallel or sequential ordering):        %d \n",mumps->id.ICNTL(28));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(29) (parallel ordering):                          %d \n",mumps->id.ICNTL(29));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(30) (user-specified set of entries in inv(A)):    %d \n",mumps->id.ICNTL(30));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(31) (factors is discarded in the solve phase):    %d \n",mumps->id.ICNTL(31));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(33) (compute determinant):                        %d \n",mumps->id.ICNTL(33));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(1) (relative pivoting threshold):      %g \n",mumps->id.CNTL(1));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(2) (stopping criterion of refinement): %g \n",mumps->id.CNTL(2));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(3) (absolute pivoting threshold):      %g \n",mumps->id.CNTL(3));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(4) (value of static pivoting):         %g \n",mumps->id.CNTL(4));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(5) (fixation for null pivots):         %g \n",mumps->id.CNTL(5));CHKERRQ(ierr);

      /* infomation local to each processor */
      ierr = PetscViewerASCIIPrintf(viewer, "  RINFO(1) (local estimated flops for the elimination after analysis): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %g \n",mumps->myid,mumps->id.RINFO(1));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  RINFO(2) (local estimated flops for the assembly after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d]  %g \n",mumps->myid,mumps->id.RINFO(2));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  RINFO(3) (local estimated flops for the elimination after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d]  %g \n",mumps->myid,mumps->id.RINFO(3));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(15) (estimated size of (in MB) MUMPS internal data for running numerical factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] %d \n",mumps->myid,mumps->id.INFO(15));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(16) (size of (in MB) MUMPS internal data used during numerical factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d \n",mumps->myid,mumps->id.INFO(16));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(23) (num of pivots eliminated on this processor after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d \n",mumps->myid,mumps->id.INFO(23));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      if (mumps->ninfo && mumps->ninfo <= 40){
        PetscInt i;
        for (i=0; i<mumps->ninfo; i++){
          ierr = PetscViewerASCIIPrintf(viewer, "  INFO(%d): \n",mumps->info[i]);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d \n",mumps->myid,mumps->id.INFO(mumps->info[i]));CHKERRQ(ierr);
          ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        }
      }


      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);

      if (!mumps->myid) { /* information from the host */
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(1) (global estimated flops for the elimination after analysis): %g \n",mumps->id.RINFOG(1));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(2) (global estimated flops for the assembly after factorization): %g \n",mumps->id.RINFOG(2));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(3) (global estimated flops for the elimination after factorization): %g \n",mumps->id.RINFOG(3));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  (RINFOG(12) RINFOG(13))*2^INFOG(34) (determinant): (%g,%g)*(2^%d)\n",mumps->id.RINFOG(12),mumps->id.RINFOG(13),mumps->id.INFOG(34));CHKERRQ(ierr);

        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d \n",mumps->id.INFOG(3));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d \n",mumps->id.INFOG(4));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(5) (estimated maximum front size in the complete tree): %d \n",mumps->id.INFOG(5));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(6) (number of nodes in the complete tree): %d \n",mumps->id.INFOG(6));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(7) (ordering option effectively use after analysis): %d \n",mumps->id.INFOG(7));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(8) (structural symmetry in percent of the permuted matrix after analysis): %d \n",mumps->id.INFOG(8));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(9) (total real/complex workspace to store the matrix factors after factorization): %d \n",mumps->id.INFOG(9));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(10) (total integer space store the matrix factors after factorization): %d \n",mumps->id.INFOG(10));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(11) (order of largest frontal matrix after factorization): %d \n",mumps->id.INFOG(11));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(12) (number of off-diagonal pivots): %d \n",mumps->id.INFOG(12));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(13) (number of delayed pivots after factorization): %d \n",mumps->id.INFOG(13));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(14) (number of memory compress after factorization): %d \n",mumps->id.INFOG(14));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(15) (number of steps of iterative refinement after solution): %d \n",mumps->id.INFOG(15));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(16) (estimated size (in MB) of all MUMPS internal data for factorization after analysis: value on the most memory consuming processor): %d \n",mumps->id.INFOG(16));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(17) (estimated size of all MUMPS internal data for factorization after analysis: sum over all processors): %d \n",mumps->id.INFOG(17));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(18) (size of all MUMPS internal data allocated during factorization: value on the most memory consuming processor): %d \n",mumps->id.INFOG(18));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(19) (size of all MUMPS internal data allocated during factorization: sum over all processors): %d \n",mumps->id.INFOG(19));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(20) (estimated number of entries in the factors): %d \n",mumps->id.INFOG(20));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(21) (size in MB of memory effectively used during factorization - value on the most memory consuming processor): %d \n",mumps->id.INFOG(21));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(22) (size in MB of memory effectively used during factorization - sum over all processors): %d \n",mumps->id.INFOG(22));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(23) (after analysis: value of ICNTL(6) effectively used): %d \n",mumps->id.INFOG(23));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(24) (after analysis: value of ICNTL(12) effectively used): %d \n",mumps->id.INFOG(24));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(25) (after factorization: number of pivots modified by static pivoting): %d \n",mumps->id.INFOG(25));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(28) (after factorization: number of null pivots encountered): %d\n",mumps->id.INFOG(28));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(29) (after factorization: effective number of entries in the factors (sum over all processors)): %d\n",mumps->id.INFOG(29));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(30, 31) (after solution: size in Mbytes of memory used during solution phase): %d, %d\n",mumps->id.INFOG(30),mumps->id.INFOG(31));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(32) (after analysis: type of analysis done): %d\n",mumps->id.INFOG(32));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(33) (value used for ICNTL(8)): %d\n",mumps->id.INFOG(33));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(34) (exponent of the determinant if determinant is requested): %d\n",mumps->id.INFOG(34));CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MUMPS(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)A->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mumps->id.INFOG(20);
  info->nz_used           = mumps->id.INFOG(20);
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
PetscErrorCode MatFactorSetSchurIS_MUMPS(Mat F, IS is)
{
  Mat_MUMPS      *mumps =(Mat_MUMPS*)F->data;
  const PetscInt *idxs;
  PetscInt       size,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&size);CHKERRQ(ierr);
  if (mumps->size > 1) {
    PetscBool ls,gs;

    ls   = mumps->myid ? (size ? PETSC_FALSE : PETSC_TRUE) : PETSC_TRUE;
    ierr = MPI_Allreduce(&ls,&gs,1,MPIU_BOOL,MPI_LAND,mumps->comm_mumps);CHKERRQ(ierr);
    if (!gs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MUMPS distributed parallel Schur complements not yet supported from PETSc\n");
  }
  if (mumps->id.size_schur != size) {
    ierr = PetscFree2(mumps->id.listvar_schur,mumps->id.schur);CHKERRQ(ierr);
    mumps->id.size_schur = size;
    mumps->id.schur_lld  = size;
    ierr = PetscMalloc2(size,&mumps->id.listvar_schur,size*size,&mumps->id.schur);CHKERRQ(ierr);
  }

  /* Schur complement matrix */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.size_schur,(PetscScalar*)mumps->id.schur,&F->schur);CHKERRQ(ierr);
  if (mumps->sym == 1) {
    ierr = MatSetOption(F->schur,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* MUMPS expects Fortran style indices */
  ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
  ierr = PetscMemcpy(mumps->id.listvar_schur,idxs,size*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0;i<size;i++) mumps->id.listvar_schur[i]++;
  ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
  if (mumps->size > 1) {
    mumps->id.ICNTL(19) = 1; /* MUMPS returns Schur centralized on the host */
  } else {
    if (F->factortype == MAT_FACTOR_LU) {
      mumps->id.ICNTL(19) = 3; /* MUMPS returns full matrix */
    } else {
      mumps->id.ICNTL(19) = 2; /* MUMPS returns lower triangular part */
    }
  }
  /* set a special value of ICNTL (not handled my MUMPS) to be used in the solve phase by PETSc */
  mumps->id.ICNTL(26) = -1;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
PetscErrorCode MatFactorCreateSchurComplement_MUMPS(Mat F,Mat* S)
{
  Mat            St;
  Mat_MUMPS      *mumps =(Mat_MUMPS*)F->data;
  PetscScalar    *array;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    im = PetscSqrtScalar((PetscScalar)-1.0);
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mumps->id.ICNTL(19)) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ORDER,"Schur complement mode not selected! You should call MatFactorSetSchurIS to enable it");
  ierr = MatCreate(PETSC_COMM_SELF,&St);CHKERRQ(ierr);
  ierr = MatSetSizes(St,PETSC_DECIDE,PETSC_DECIDE,mumps->id.size_schur,mumps->id.size_schur);CHKERRQ(ierr);
  ierr = MatSetType(St,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(St);CHKERRQ(ierr);
  ierr = MatDenseGetArray(St,&array);CHKERRQ(ierr);
  if (!mumps->sym) { /* MUMPS always return a full matrix */
    if (mumps->id.ICNTL(19) == 1) { /* stored by rows */
      PetscInt i,j,N=mumps->id.size_schur;
      for (i=0;i<N;i++) {
        for (j=0;j<N;j++) {
#if !defined(PETSC_USE_COMPLEX)
          PetscScalar val = mumps->id.schur[i*N+j];
#else
          PetscScalar val = mumps->id.schur[i*N+j].r + im*mumps->id.schur[i*N+j].i;
#endif
          array[j*N+i] = val;
        }
      }
    } else { /* stored by columns */
      ierr = PetscMemcpy(array,mumps->id.schur,mumps->id.size_schur*mumps->id.size_schur*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else { /* either full or lower-triangular (not packed) */
    if (mumps->id.ICNTL(19) == 2) { /* lower triangular stored by columns */
      PetscInt i,j,N=mumps->id.size_schur;
      for (i=0;i<N;i++) {
        for (j=i;j<N;j++) {
#if !defined(PETSC_USE_COMPLEX)
          PetscScalar val = mumps->id.schur[i*N+j];
#else
          PetscScalar val = mumps->id.schur[i*N+j].r + im*mumps->id.schur[i*N+j].i;
#endif
          array[i*N+j] = val;
          array[j*N+i] = val;
        }
      }
    } else if (mumps->id.ICNTL(19) == 3) { /* full matrix */
      ierr = PetscMemcpy(array,mumps->id.schur,mumps->id.size_schur*mumps->id.size_schur*sizeof(PetscScalar));CHKERRQ(ierr);
    } else { /* ICNTL(19) == 1 lower triangular stored by rows */
      PetscInt i,j,N=mumps->id.size_schur;
      for (i=0;i<N;i++) {
        for (j=0;j<i+1;j++) {
#if !defined(PETSC_USE_COMPLEX)
          PetscScalar val = mumps->id.schur[i*N+j];
#else
          PetscScalar val = mumps->id.schur[i*N+j].r + im*mumps->id.schur[i*N+j].i;
#endif
          array[i*N+j] = val;
          array[j*N+i] = val;
        }
      }
    }
  }
  ierr = MatDenseRestoreArray(St,&array);CHKERRQ(ierr);
  *S   = St;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
PetscErrorCode MatMumpsSetIcntl_MUMPS(Mat F,PetscInt icntl,PetscInt ival)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  mumps->id.ICNTL(icntl) = ival;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetIcntl_MUMPS(Mat F,PetscInt icntl,PetscInt *ival)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  *ival = mumps->id.ICNTL(icntl);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsSetIcntl - Set MUMPS parameter ICNTL()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
.  icntl - index of MUMPS parameter array ICNTL()
-  ival - value of MUMPS ICNTL(icntl)

  Options Database:
.   -mat_mumps_icntl_<icntl> <ival>

   Level: beginner

   References:
.     MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
 @*/
PetscErrorCode MatMumpsSetIcntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidLogicalCollectiveInt(F,ival,3);
  ierr = PetscTryMethod(F,"MatMumpsSetIcntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetIcntl - Get MUMPS parameter ICNTL()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  icntl - index of MUMPS parameter array ICNTL()

  Output Parameter:
.  ival - value of MUMPS ICNTL(icntl)

   Level: beginner

   References:
.     MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetIcntl(Mat F,PetscInt icntl,PetscInt *ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidIntPointer(ival,3);
  ierr = PetscUseMethod(F,"MatMumpsGetIcntl_C",(Mat,PetscInt,PetscInt*),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
PetscErrorCode MatMumpsSetCntl_MUMPS(Mat F,PetscInt icntl,PetscReal val)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  mumps->id.CNTL(icntl) = val;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetCntl_MUMPS(Mat F,PetscInt icntl,PetscReal *val)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  *val = mumps->id.CNTL(icntl);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsSetCntl - Set MUMPS parameter CNTL()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
.  icntl - index of MUMPS parameter array CNTL()
-  val - value of MUMPS CNTL(icntl)

  Options Database:
.   -mat_mumps_cntl_<icntl> <val>

   Level: beginner

   References:
.     MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsSetCntl(Mat F,PetscInt icntl,PetscReal val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidLogicalCollectiveReal(F,val,3);
  ierr = PetscTryMethod(F,"MatMumpsSetCntl_C",(Mat,PetscInt,PetscReal),(F,icntl,val));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetCntl - Get MUMPS parameter CNTL()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  icntl - index of MUMPS parameter array CNTL()

  Output Parameter:
.  val - value of MUMPS CNTL(icntl)

   Level: beginner

   References:
.      MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetCntl(Mat F,PetscInt icntl,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidRealPointer(val,3);
  ierr = PetscUseMethod(F,"MatMumpsGetCntl_C",(Mat,PetscInt,PetscReal*),(F,icntl,val));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetInfo_MUMPS(Mat F,PetscInt icntl,PetscInt *info)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  *info = mumps->id.INFO(icntl);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetInfog_MUMPS(Mat F,PetscInt icntl,PetscInt *infog)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  *infog = mumps->id.INFOG(icntl);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetRinfo_MUMPS(Mat F,PetscInt icntl,PetscReal *rinfo)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  *rinfo = mumps->id.RINFO(icntl);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetRinfog_MUMPS(Mat F,PetscInt icntl,PetscReal *rinfog)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  *rinfog = mumps->id.RINFOG(icntl); 
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetInfo - Get MUMPS parameter INFO()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  icntl - index of MUMPS parameter array INFO()

  Output Parameter:
.  ival - value of MUMPS INFO(icntl)

   Level: beginner

   References:
.      MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetInfo(Mat F,PetscInt icntl,PetscInt *ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidIntPointer(ival,3);
  ierr = PetscUseMethod(F,"MatMumpsGetInfo_C",(Mat,PetscInt,PetscInt*),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetInfog - Get MUMPS parameter INFOG()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  icntl - index of MUMPS parameter array INFOG()

  Output Parameter:
.  ival - value of MUMPS INFOG(icntl)

   Level: beginner

   References:
.      MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetInfog(Mat F,PetscInt icntl,PetscInt *ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidIntPointer(ival,3);
  ierr = PetscUseMethod(F,"MatMumpsGetInfog_C",(Mat,PetscInt,PetscInt*),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetRinfo - Get MUMPS parameter RINFO()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  icntl - index of MUMPS parameter array RINFO()

  Output Parameter:
.  val - value of MUMPS RINFO(icntl)

   Level: beginner

   References:
.       MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetRinfo(Mat F,PetscInt icntl,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidRealPointer(val,3);
  ierr = PetscUseMethod(F,"MatMumpsGetRinfo_C",(Mat,PetscInt,PetscReal*),(F,icntl,val));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetRinfog - Get MUMPS parameter RINFOG()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  icntl - index of MUMPS parameter array RINFOG()

  Output Parameter:
.  val - value of MUMPS RINFOG(icntl)

   Level: beginner

   References:
.      MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetRinfog(Mat F,PetscInt icntl,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidRealPointer(val,3);
  ierr = PetscUseMethod(F,"MatMumpsGetRinfog_C",(Mat,PetscInt,PetscReal*),(F,icntl,val));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERMUMPS -  A matrix type providing direct solvers (LU and Cholesky) for
  distributed and sequential matrices via the external package MUMPS.

  Works with MATAIJ and MATSBAIJ matrices

  Use ./configure --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch  to have PETSc installed with MUMPS

  Use -pc_type cholesky or lu -pc_factor_mat_solver_package mumps to us this direct solver

  Options Database Keys:
+  -mat_mumps_icntl_1 - ICNTL(1): output stream for error messages 
.  -mat_mumps_icntl_2 - ICNTL(2): output stream for diagnostic printing, statistics, and warning 
.  -mat_mumps_icntl_3 -  ICNTL(3): output stream for global information, collected on the host 
.  -mat_mumps_icntl_4 -  ICNTL(4): level of printing (0 to 4) 
.  -mat_mumps_icntl_6 - ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7) 
.  -mat_mumps_icntl_7 - ICNTL(7): computes a symmetric permutation in sequential analysis (0 to 7). 3=Scotch, 4=PORD, 5=Metis 
.  -mat_mumps_icntl_8  - ICNTL(8): scaling strategy (-2 to 8 or 77) 
.  -mat_mumps_icntl_10  - ICNTL(10): max num of refinements 
.  -mat_mumps_icntl_11  - ICNTL(11): statistics related to an error analysis (via -ksp_view) 
.  -mat_mumps_icntl_12  - ICNTL(12): an ordering strategy for symmetric matrices (0 to 3) 
.  -mat_mumps_icntl_13  - ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting 
.  -mat_mumps_icntl_14  - ICNTL(14): percentage increase in the estimated working space 
.  -mat_mumps_icntl_19  - ICNTL(19): computes the Schur complement 
.  -mat_mumps_icntl_22  - ICNTL(22): in-core/out-of-core factorization and solve (0 or 1) 
.  -mat_mumps_icntl_23  - ICNTL(23): max size of the working memory (MB) that can allocate per processor 
.  -mat_mumps_icntl_24  - ICNTL(24): detection of null pivot rows (0 or 1) 
.  -mat_mumps_icntl_25  - ICNTL(25): compute a solution of a deficient matrix and a null space basis 
.  -mat_mumps_icntl_26  - ICNTL(26): drives the solution phase if a Schur complement matrix 
.  -mat_mumps_icntl_28  - ICNTL(28): use 1 for sequential analysis and ictnl(7) ordering, or 2 for parallel analysis and ictnl(29) ordering 
.  -mat_mumps_icntl_29 - ICNTL(29): parallel ordering 1 = ptscotch, 2 = parmetis 
.  -mat_mumps_icntl_30 - ICNTL(30): compute user-specified set of entries in inv(A) 
.  -mat_mumps_icntl_31 - ICNTL(31): indicates which factors may be discarded during factorization 
.  -mat_mumps_icntl_33 - ICNTL(33): compute determinant 
.  -mat_mumps_cntl_1  - CNTL(1): relative pivoting threshold 
.  -mat_mumps_cntl_2  -  CNTL(2): stopping criterion of refinement 
.  -mat_mumps_cntl_3 - CNTL(3): absolute pivoting threshold 
.  -mat_mumps_cntl_4 - CNTL(4): value for static pivoting 
-  -mat_mumps_cntl_5 - CNTL(5): fixation for null pivots 

  Level: beginner

    Notes: When a MUMPS factorization fails inside a KSP solve, for example with a KSP_DIVERGED_PCSETUP_FAILED, one can find the MUMPS information about the failure by calling 
$          KSPGetPC(ksp,&pc);
$          PCFactorGetMatrix(pc,&mat);
$          MatMumpsGetInfo(mat,....);
$          MatMumpsGetInfog(mat,....); etc.
           Or you can run with -ksp_error_if_not_converged and the program will be stopped and the information printed in the error message.

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage, MatMumpsSetICntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog(), KSPGetPC(), PCGetFactor(), PCFactorGetMatrix()

M*/

static PetscErrorCode MatFactorGetSolverPackage_mumps(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMUMPS;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq and MPI AIJ matrices */
static PetscErrorCode MatGetFactor_aij_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mumps",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mumps);CHKERRQ(ierr);

  B->ops->view        = MatView_MUMPS;
  B->ops->getinfo     = MatGetInfo_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpiaij;
    mumps->sym = 0;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
    B->factortype                  = MAT_FACTOR_CHOLESKY;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqsbaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpisbaij;
#if defined(PETSC_USE_COMPLEX)
    mumps->sym = 2;
#else
    if (A->spd_set && A->spd) mumps->sym = 1;
    else                      mumps->sym = 2;
#endif
  }

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype);CHKERRQ(ierr);

  mumps->isAIJ    = PETSC_TRUE;
  B->ops->destroy = MatDestroy_MUMPS;
  B->data        = (void*)mumps;

  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq and MPI SBAIJ matrices */
static PetscErrorCode MatGetFactor_sbaij_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqSBAIJ;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with MUMPS LU, use AIJ matrix");
  if (A->rmap->bs > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with block size > 1 with MUMPS Cholesky, use AIJ matrix instead");
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mumps",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mumps);CHKERRQ(ierr);
  if (isSeqSBAIJ) {
    mumps->ConvertToTriples = MatConvertToTriples_seqsbaij_seqsbaij;
  } else {
    mumps->ConvertToTriples = MatConvertToTriples_mpisbaij_mpisbaij;
  }

  B->ops->getinfo                = MatGetInfo_External;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
  B->ops->view                   = MatView_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS);CHKERRQ(ierr);

  B->factortype = MAT_FACTOR_CHOLESKY;
#if defined(PETSC_USE_COMPLEX)
  mumps->sym = 2;
#else
  if (A->spd_set && A->spd) mumps->sym = 1;
  else                      mumps->sym = 2;
#endif

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype);CHKERRQ(ierr);

  mumps->isAIJ    = PETSC_FALSE;
  B->ops->destroy = MatDestroy_MUMPS;
  B->data        = (void*)mumps;

  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_baij_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqBAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQBAIJ,&isSeqBAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mumps",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mumps);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_BAIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqBAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqbaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpibaij_mpiaij;
    mumps->sym = 0;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc BAIJ matrices with MUMPS Cholesky, use SBAIJ or AIJ matrix instead\n");

  B->ops->getinfo     = MatGetInfo_External;
  B->ops->view        = MatView_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS);CHKERRQ(ierr);

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype);CHKERRQ(ierr);

  mumps->isAIJ    = PETSC_TRUE;
  B->ops->destroy = MatDestroy_MUMPS;
  B->data        = (void*)mumps;

  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_MUMPS(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATMPIAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATMPIBAIJ,MAT_FACTOR_LU,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATMPIBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATMPISBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATSEQAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATSEQBAIJ,MAT_FACTOR_LU,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATSEQBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERMUMPS,MATSEQSBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mumps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

