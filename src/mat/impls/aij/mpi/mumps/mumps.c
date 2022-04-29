
/*
    Provides an interface to the MUMPS sparse solver
*/
#include <petscpkg_version.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I  "petscmat.h"  I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <../src/mat/impls/sell/mpi/mpisell.h>

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
#define MUMPS_c cmumps_c
#else
#define MUMPS_c zmumps_c
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define MUMPS_c smumps_c
#else
#define MUMPS_c dmumps_c
#endif
#endif

/* MUMPS uses MUMPS_INT for nonzero indices such as irn/jcn, irn_loc/jcn_loc and uses int64_t for
   number of nonzeros such as nnz, nnz_loc. We typedef MUMPS_INT to PetscMUMPSInt to follow the
   naming convention in PetscMPIInt, PetscBLASInt etc.
*/
typedef MUMPS_INT PetscMUMPSInt;

#if PETSC_PKG_MUMPS_VERSION_GE(5,3,0)
  #if defined(MUMPS_INTSIZE64) /* MUMPS_INTSIZE64 is in MUMPS headers if it is built in full 64-bit mode, therefore the macro is more reliable */
    #error "Petsc has not been tested with full 64-bit MUMPS and we choose to error out"
  #endif
#else
  #if defined(INTSIZE64) /* INTSIZE64 is a command line macro one used to build MUMPS in full 64-bit mode */
    #error "Petsc has not been tested with full 64-bit MUMPS and we choose to error out"
  #endif
#endif

#define MPIU_MUMPSINT             MPI_INT
#define PETSC_MUMPS_INT_MAX       2147483647
#define PETSC_MUMPS_INT_MIN       -2147483648

/* Cast PetscInt to PetscMUMPSInt. Usually there is no overflow since <a> is row/col indices or some small integers*/
static inline PetscErrorCode PetscMUMPSIntCast(PetscInt a,PetscMUMPSInt *b)
{
  PetscFunctionBegin;
#if PetscDefined(USE_64BIT_INDICES)
  PetscAssert(a <= PETSC_MUMPS_INT_MAX && a >= PETSC_MUMPS_INT_MIN,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscInt too long for PetscMUMPSInt");
#endif
  *b = (PetscMUMPSInt)(a);
  PetscFunctionReturn(0);
}

/* Put these utility routines here since they are only used in this file */
static inline PetscErrorCode PetscOptionsMUMPSInt_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscMUMPSInt currentvalue,PetscMUMPSInt *value,PetscBool *set,PetscMUMPSInt lb,PetscMUMPSInt ub)
{
  PetscInt       myval;
  PetscBool      myset;
  PetscFunctionBegin;
  /* PetscInt's size should be always >= PetscMUMPSInt's. It is safe to call PetscOptionsInt_Private to read a PetscMUMPSInt */
  PetscCall(PetscOptionsInt_Private(PetscOptionsObject,opt,text,man,(PetscInt)currentvalue,&myval,&myset,lb,ub));
  if (myset) PetscCall(PetscMUMPSIntCast(myval,value));
  if (set) *set = myset;
  PetscFunctionReturn(0);
}
#define PetscOptionsMUMPSInt(a,b,c,d,e,f) PetscOptionsMUMPSInt_Private(PetscOptionsObject,a,b,c,d,e,f,PETSC_MUMPS_INT_MIN,PETSC_MUMPS_INT_MAX)

/* if using PETSc OpenMP support, we only call MUMPS on master ranks. Before/after the call, we change/restore CPUs the master ranks can run on */
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
#define PetscMUMPS_c(mumps) \
  do { \
    if (mumps->use_petsc_omp_support) { \
      if (mumps->is_omp_master) { \
        PetscCall(PetscOmpCtrlOmpRegionOnMasterBegin(mumps->omp_ctrl)); \
        MUMPS_c(&mumps->id); \
        PetscCall(PetscOmpCtrlOmpRegionOnMasterEnd(mumps->omp_ctrl)); \
      } \
      PetscCall(PetscOmpCtrlBarrier(mumps->omp_ctrl)); \
      /* Global info is same on all processes so we Bcast it within omp_comm. Local info is specific      \
         to processes, so we only Bcast info[1], an error code and leave others (since they do not have   \
         an easy translation between omp_comm and petsc_comm). See MUMPS-5.1.2 manual p82.                   \
         omp_comm is a small shared memory communicator, hence doing multiple Bcast as shown below is OK. \
      */ \
      PetscCallMPI(MPI_Bcast(mumps->id.infog, 40,MPIU_MUMPSINT, 0,mumps->omp_comm));\
      PetscCallMPI(MPI_Bcast(mumps->id.rinfog,20,MPIU_REAL,     0,mumps->omp_comm));\
      PetscCallMPI(MPI_Bcast(mumps->id.info,  1, MPIU_MUMPSINT, 0,mumps->omp_comm));\
    } else { \
      MUMPS_c(&mumps->id); \
    } \
  } while (0)
#else
#define PetscMUMPS_c(mumps) \
  do { MUMPS_c(&mumps->id); } while (0)
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

typedef struct Mat_MUMPS Mat_MUMPS;
struct Mat_MUMPS {
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

  MatStructure   matstruc;
  PetscMPIInt    myid,petsc_size;
  PetscMUMPSInt  *irn,*jcn;             /* the (i,j,v) triplets passed to mumps. */
  PetscScalar    *val,*val_alloc;       /* For some matrices, we can directly access their data array without a buffer. For others, we need a buffer. So comes val_alloc. */
  PetscInt64     nnz;                   /* number of nonzeros. The type is called selective 64-bit in mumps */
  PetscMUMPSInt  sym;
  MPI_Comm       mumps_comm;
  PetscMUMPSInt  ICNTL9_pre;            /* check if ICNTL(9) is changed from previous MatSolve */
  VecScatter     scat_rhs, scat_sol;    /* used by MatSolve() */
  PetscMUMPSInt  ICNTL20;               /* use centralized (0) or distributed (10) dense RHS */
  PetscMUMPSInt  lrhs_loc,nloc_rhs,*irhs_loc;
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  PetscInt       *rhs_nrow,max_nrhs;
  PetscMPIInt    *rhs_recvcounts,*rhs_disps;
  PetscScalar    *rhs_loc,*rhs_recvbuf;
#endif
  Vec            b_seq,x_seq;
  PetscInt       ninfo,*info;           /* which INFO to display */
  PetscInt       sizeredrhs;
  PetscScalar    *schur_sol;
  PetscInt       schur_sizesol;
  PetscMUMPSInt  *ia_alloc,*ja_alloc;   /* work arrays used for the CSR struct for sparse rhs */
  PetscInt64     cur_ilen,cur_jlen;     /* current len of ia_alloc[], ja_alloc[] */
  PetscErrorCode (*ConvertToTriples)(Mat,PetscInt,MatReuse,Mat_MUMPS*);

  /* stuff used by petsc/mumps OpenMP support*/
  PetscBool      use_petsc_omp_support;
  PetscOmpCtrl   omp_ctrl;              /* an OpenMP controler that blocked processes will release their CPU (MPI_Barrier does not have this guarantee) */
  MPI_Comm       petsc_comm,omp_comm;   /* petsc_comm is petsc matrix's comm */
  PetscInt64     *recvcount;            /* a collection of nnz on omp_master */
  PetscMPIInt    tag,omp_comm_size;
  PetscBool      is_omp_master;         /* is this rank the master of omp_comm */
  MPI_Request    *reqs;
};

/* Cast a 1-based CSR represented by (nrow, ia, ja) of type PetscInt to a CSR of type PetscMUMPSInt.
   Here, nrow is number of rows, ia[] is row pointer and ja[] is column indices.
 */
static PetscErrorCode PetscMUMPSIntCSRCast(Mat_MUMPS *mumps,PetscInt nrow,PetscInt *ia,PetscInt *ja,PetscMUMPSInt **ia_mumps,PetscMUMPSInt **ja_mumps,PetscMUMPSInt *nnz_mumps)
{
  PetscInt       nnz=ia[nrow]-1; /* mumps uses 1-based indices. Uses PetscInt instead of PetscInt64 since mumps only uses PetscMUMPSInt for rhs */

  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  {
    PetscInt i;
    if (nrow+1 > mumps->cur_ilen) { /* realloc ia_alloc/ja_alloc to fit ia/ja */
      PetscCall(PetscFree(mumps->ia_alloc));
      PetscCall(PetscMalloc1(nrow+1,&mumps->ia_alloc));
      mumps->cur_ilen = nrow+1;
    }
    if (nnz > mumps->cur_jlen) {
      PetscCall(PetscFree(mumps->ja_alloc));
      PetscCall(PetscMalloc1(nnz,&mumps->ja_alloc));
      mumps->cur_jlen = nnz;
    }
    for (i=0; i<nrow+1; i++) PetscCall(PetscMUMPSIntCast(ia[i],&(mumps->ia_alloc[i])));
    for (i=0; i<nnz; i++)    PetscCall(PetscMUMPSIntCast(ja[i],&(mumps->ja_alloc[i])));
    *ia_mumps = mumps->ia_alloc;
    *ja_mumps = mumps->ja_alloc;
  }
#else
  *ia_mumps = ia;
  *ja_mumps = ja;
#endif
  PetscCall(PetscMUMPSIntCast(nnz,nnz_mumps));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMumpsResetSchur_Private(Mat_MUMPS* mumps)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mumps->id.listvar_schur));
  PetscCall(PetscFree(mumps->id.redrhs));
  PetscCall(PetscFree(mumps->schur_sol));
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

  PetscFunctionBegin;
  PetscCall(MatFactorFactorizeSchurComplement(F));
  PetscCall(MatFactorGetSchurComplement(F,&S,&schurstatus));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,(PetscScalar*)mumps->id.redrhs,&B));
  PetscCall(MatSetType(B,((PetscObject)S)->type_name));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  PetscCall(MatBindToCPU(B,S->boundtocpu));
#endif
  switch (schurstatus) {
  case MAT_FACTOR_SCHUR_FACTORED:
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,(PetscScalar*)mumps->id.redrhs,&X));
    PetscCall(MatSetType(X,((PetscObject)S)->type_name));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscCall(MatBindToCPU(X,S->boundtocpu));
#endif
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      PetscCall(MatMatSolveTranspose(S,B,X));
    } else {
      PetscCall(MatMatSolve(S,B,X));
    }
    break;
  case MAT_FACTOR_SCHUR_INVERTED:
    sizesol = mumps->id.nrhs*mumps->id.size_schur;
    if (!mumps->schur_sol || sizesol > mumps->schur_sizesol) {
      PetscCall(PetscFree(mumps->schur_sol));
      PetscCall(PetscMalloc1(sizesol,&mumps->schur_sol));
      mumps->schur_sizesol = sizesol;
    }
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,mumps->schur_sol,&X));
    PetscCall(MatSetType(X,((PetscObject)S)->type_name));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscCall(MatBindToCPU(X,S->boundtocpu));
#endif
    PetscCall(MatProductCreateWithMat(S,B,NULL,X));
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      PetscCall(MatProductSetType(X,MATPRODUCT_AtB));
    } else {
      PetscCall(MatProductSetType(X,MATPRODUCT_AB));
    }
    PetscCall(MatProductSetFromOptions(X));
    PetscCall(MatProductSymbolic(X));
    PetscCall(MatProductNumeric(X));

    PetscCall(MatCopy(X,B,SAME_NONZERO_PATTERN));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %d",F->schur_status);
  }
  PetscCall(MatFactorRestoreSchurComplement(F,&S,schurstatus));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMumpsHandleSchur_Private(Mat F, PetscBool expansion)
{
  Mat_MUMPS     *mumps=(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  if (!mumps->id.ICNTL(19)) { /* do nothing when Schur complement has not been computed */
    PetscFunctionReturn(0);
  }
  if (!expansion) { /* prepare for the condensation step */
    PetscInt sizeredrhs = mumps->id.nrhs*mumps->id.size_schur;
    /* allocate MUMPS internal array to store reduced right-hand sides */
    if (!mumps->id.redrhs || sizeredrhs > mumps->sizeredrhs) {
      PetscCall(PetscFree(mumps->id.redrhs));
      mumps->id.lredrhs = mumps->id.size_schur;
      PetscCall(PetscMalloc1(mumps->id.nrhs*mumps->id.lredrhs,&mumps->id.redrhs));
      mumps->sizeredrhs = mumps->id.nrhs*mumps->id.lredrhs;
    }
    mumps->id.ICNTL(26) = 1; /* condensation phase */
  } else { /* prepare for the expansion step */
    /* solve Schur complement (this has to be done by the MUMPS user, so basically us) */
    PetscCall(MatMumpsSolveSchur_Private(F));
    mumps->id.ICNTL(26) = 2; /* expansion phase */
    PetscMUMPS_c(mumps);
    PetscCheck(mumps->id.INFOG(1) >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d",mumps->id.INFOG(1));
    /* restore defaults */
    mumps->id.ICNTL(26) = -1;
    /* free MUMPS internal array for redrhs if we have solved for multiple rhs in order to save memory space */
    if (mumps->id.nrhs > 1) {
      PetscCall(PetscFree(mumps->id.redrhs));
      mumps->id.lredrhs = 0;
      mumps->sizeredrhs = 0;
    }
  }
  PetscFunctionReturn(0);
}

/*
  MatConvertToTriples_A_B - convert Petsc matrix to triples: row[nz], col[nz], val[nz]

  input:
    A       - matrix in aij,baij or sbaij format
    shift   - 0: C style output triple; 1: Fortran style output triple.
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple
              MAT_REUSE_MATRIX:   only the values in v array are updated
  output:
    nnz     - dim of r, c, and v (number of local nonzero entries of A)
    r, c, v - row and col index, matrix values (matrix triples)

  The returned values r, c, and sometimes v are obtained in a single PetscMalloc(). Then in MatDestroy_MUMPS() it is
  freed with PetscFree(mumps->irn);  This is not ideal code, the fact that v is ONLY sometimes part of mumps->irn means
  that the PetscMalloc() cannot easily be replaced with a PetscMalloc3().

 */

PetscErrorCode MatConvertToTriples_seqaij_seqaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  const PetscScalar *av;
  const PetscInt    *ai,*aj,*ajj,M=A->rmap->n;
  PetscInt64        nz,rnz,i,j,k;
  PetscMUMPSInt     *row,*col;
  Mat_SeqAIJ        *aa=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A,&av));
  mumps->val = (PetscScalar*)av;
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz;
    ai   = aa->i;
    aj   = aa->j;
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    for (i=k=0; i<M; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for (j=0; j<rnz; j++) {
        PetscCall(PetscMUMPSIntCast(i+shift,&row[k]));
        PetscCall(PetscMUMPSIntCast(ajj[j] + shift,&col[k]));
        k++;
      }
    }
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqsell_seqaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  PetscInt64     nz,i,j,k,r;
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscMUMPSInt  *row,*col;

  PetscFunctionBegin;
  mumps->val = a->val;
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = a->sliidx[a->totalslices];
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    for (i=k=0; i<a->totalslices; i++) {
      for (j=a->sliidx[i],r=0; j<a->sliidx[i+1]; j++,r=((r+1)&0x07)) {
        PetscCall(PetscMUMPSIntCast(8*i+r+shift,&row[k++]));
      }
    }
    for (i=0;i<nz;i++) PetscCall(PetscMUMPSIntCast(a->colidx[i]+shift,&col[i]));
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqbaij_seqaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  Mat_SeqBAIJ    *aa=(Mat_SeqBAIJ*)A->data;
  const PetscInt *ai,*aj,*ajj,bs2 = aa->bs2;
  PetscInt64     M,nz,idx=0,rnz,i,j,k,m;
  PetscInt       bs;
  PetscMUMPSInt  *row,*col;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(A,&bs));
  M          = A->rmap->N/bs;
  mumps->val = aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    ai   = aa->i; aj = aa->j;
    nz   = bs2*aa->nz;
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    for (i=0; i<M; i++) {
      ajj = aj + ai[i];
      rnz = ai[i+1] - ai[i];
      for (k=0; k<rnz; k++) {
        for (j=0; j<bs; j++) {
          for (m=0; m<bs; m++) {
            PetscCall(PetscMUMPSIntCast(i*bs + m + shift,&row[idx]));
            PetscCall(PetscMUMPSIntCast(bs*ajj[k] + j + shift,&col[idx]));
            idx++;
          }
        }
      }
    }
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqsbaij_seqsbaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  const PetscInt *ai, *aj,*ajj;
  PetscInt        bs;
  PetscInt64      nz,rnz,i,j,k,m;
  PetscMUMPSInt   *row,*col;
  PetscScalar     *val;
  Mat_SeqSBAIJ    *aa=(Mat_SeqSBAIJ*)A->data;
  const PetscInt  bs2=aa->bs2,mbs=aa->mbs;
#if defined(PETSC_USE_COMPLEX)
  PetscBool       hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatGetOption(A,MAT_HERMITIAN,&hermitian));
  PetscCheck(!hermitian,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  ai   = aa->i;
  aj   = aa->j;
  PetscCall(MatGetBlockSize(A,&bs));
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz;
    PetscCall(PetscMalloc2(bs2*nz,&row,bs2*nz,&col));
    if (bs>1) {
      PetscCall(PetscMalloc1(bs2*nz,&mumps->val_alloc));
      mumps->val = mumps->val_alloc;
    } else {
      mumps->val = aa->a;
    }
    mumps->irn = row;
    mumps->jcn = col;
  } else {
    if (bs == 1) mumps->val = aa->a;
    row = mumps->irn;
    col = mumps->jcn;
  }
  val = mumps->val;

  nz = 0;
  if (bs>1) {
    for (i=0; i<mbs; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for (j=0; j<rnz; j++) {
        for (k=0; k<bs; k++) {
          for (m=0; m<bs; m++) {
            if (ajj[j]>i || k>=m) {
              if (reuse == MAT_INITIAL_MATRIX) {
                PetscCall(PetscMUMPSIntCast(i*bs + m + shift,&row[nz]));
                PetscCall(PetscMUMPSIntCast(ajj[j]*bs + k + shift,&col[nz]));
              }
              val[nz++] = aa->a[(ai[i]+j)*bs2 + m + k*bs];
            }
          }
        }
      }
    }
  } else if (reuse == MAT_INITIAL_MATRIX) {
    for (i=0; i<mbs; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for (j=0; j<rnz; j++) {
        PetscCall(PetscMUMPSIntCast(i+shift,&row[nz]));
        PetscCall(PetscMUMPSIntCast(ajj[j] + shift,&col[nz]));
        nz++;
      }
    }
    PetscCheck(nz == aa->nz,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Different numbers of nonzeros %" PetscInt64_FMT " != %" PetscInt_FMT,nz,aa->nz);
  }
  if (reuse == MAT_INITIAL_MATRIX) mumps->nnz = nz;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqaij_seqsbaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  const PetscInt    *ai,*aj,*ajj,*adiag,M=A->rmap->n;
  PetscInt64        nz,rnz,i,j;
  const PetscScalar *av,*v1;
  PetscScalar       *val;
  PetscMUMPSInt     *row,*col;
  Mat_SeqAIJ        *aa=(Mat_SeqAIJ*)A->data;
  PetscBool         missing;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatGetOption(A,MAT_HERMITIAN,&hermitian));
  PetscCheck(!hermitian,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  PetscCall(MatSeqAIJGetArrayRead(A,&av));
  ai    = aa->i; aj = aa->j;
  adiag = aa->diag;
  PetscCall(MatMissingDiagonal_SeqAIJ(A,&missing,NULL));
  if (reuse == MAT_INITIAL_MATRIX) {
    /* count nz in the upper triangular part of A */
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
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    PetscCall(PetscMalloc1(nz,&val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;

    nz = 0;
    if (missing) {
      for (i=0; i<M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i+1])) {
          for (j=ai[i];j<ai[i+1];j++) {
            if (aj[j] < i) continue;
            PetscCall(PetscMUMPSIntCast(i+shift,&row[nz]));
            PetscCall(PetscMUMPSIntCast(aj[j]+shift,&col[nz]));
            val[nz] = av[j];
            nz++;
          }
        } else {
          rnz = ai[i+1] - adiag[i];
          ajj = aj + adiag[i];
          v1  = av + adiag[i];
          for (j=0; j<rnz; j++) {
            PetscCall(PetscMUMPSIntCast(i+shift,&row[nz]));
            PetscCall(PetscMUMPSIntCast(ajj[j] + shift,&col[nz]));
            val[nz++] = v1[j];
          }
        }
      }
    } else {
      for (i=0; i<M; i++) {
        rnz = ai[i+1] - adiag[i];
        ajj = aj + adiag[i];
        v1  = av + adiag[i];
        for (j=0; j<rnz; j++) {
          PetscCall(PetscMUMPSIntCast(i+shift,&row[nz]));
          PetscCall(PetscMUMPSIntCast(ajj[j] + shift,&col[nz]));
          val[nz++] = v1[j];
        }
      }
    }
  } else {
    nz = 0;
    val = mumps->val;
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
  PetscCall(MatSeqAIJRestoreArrayRead(A,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpisbaij_mpisbaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  const PetscInt    *ai,*aj,*bi,*bj,*garray,*ajj,*bjj;
  PetscInt          bs;
  PetscInt64        rstart,nz,i,j,k,m,jj,irow,countA,countB;
  PetscMUMPSInt     *row,*col;
  const PetscScalar *av,*bv,*v1,*v2;
  PetscScalar       *val;
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ      *aa  = (Mat_SeqSBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ       *bb  = (Mat_SeqBAIJ*)(mat->B)->data;
  const PetscInt    bs2=aa->bs2,mbs=aa->mbs;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatGetOption(A,MAT_HERMITIAN,&hermitian));
  PetscCheck(!hermitian,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  PetscCall(MatGetBlockSize(A,&bs));
  rstart = A->rmap->rstart;
  ai = aa->i;
  aj = aa->j;
  bi = bb->i;
  bj = bb->j;
  av = aa->a;
  bv = bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = (aa->nz+bb->nz)*bs2; /* just a conservative estimate */
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    PetscCall(PetscMalloc1(nz,&val));
    /* can not decide the exact mumps->nnz now because of the SBAIJ */
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
  }

  jj = 0; irow = rstart;
  for (i=0; i<mbs; i++) {
    ajj    = aj + ai[i];                 /* ptr to the beginning of this row */
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i]*bs2;
    v2     = bv + bi[i]*bs2;

    if (bs>1) {
      /* A-part */
      for (j=0; j<countA; j++) {
        for (k=0; k<bs; k++) {
          for (m=0; m<bs; m++) {
            if (rstart + ajj[j]*bs>irow || k>=m) {
              if (reuse == MAT_INITIAL_MATRIX) {
                PetscCall(PetscMUMPSIntCast(irow + m + shift,&row[jj]));
                PetscCall(PetscMUMPSIntCast(rstart + ajj[j]*bs + k + shift,&col[jj]));
              }
              val[jj++] = v1[j*bs2 + m + k*bs];
            }
          }
        }
      }

      /* B-part */
      for (j=0; j < countB; j++) {
        for (k=0; k<bs; k++) {
          for (m=0; m<bs; m++) {
            if (reuse == MAT_INITIAL_MATRIX) {
              PetscCall(PetscMUMPSIntCast(irow + m + shift,&row[jj]));
              PetscCall(PetscMUMPSIntCast(garray[bjj[j]]*bs + k + shift,&col[jj]));
            }
            val[jj++] = v2[j*bs2 + m + k*bs];
          }
        }
      }
    } else {
      /* A-part */
      for (j=0; j<countA; j++) {
        if (reuse == MAT_INITIAL_MATRIX) {
          PetscCall(PetscMUMPSIntCast(irow + shift,&row[jj]));
          PetscCall(PetscMUMPSIntCast(rstart + ajj[j] + shift,&col[jj]));
        }
        val[jj++] = v1[j];
      }

      /* B-part */
      for (j=0; j < countB; j++) {
        if (reuse == MAT_INITIAL_MATRIX) {
          PetscCall(PetscMUMPSIntCast(irow + shift,&row[jj]));
          PetscCall(PetscMUMPSIntCast(garray[bjj[j]] + shift,&col[jj]));
        }
        val[jj++] = v2[j];
      }
    }
    irow+=bs;
  }
  mumps->nnz = jj;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpiaij_mpiaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  const PetscInt    *ai, *aj, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscInt64        rstart,nz,i,j,jj,irow,countA,countB;
  PetscMUMPSInt     *row,*col;
  const PetscScalar *av, *bv,*v1,*v2;
  PetscScalar       *val;
  Mat               Ad,Ao;
  Mat_SeqAIJ        *aa;
  Mat_SeqAIJ        *bb;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray));
  PetscCall(MatSeqAIJGetArrayRead(Ad,&av));
  PetscCall(MatSeqAIJGetArrayRead(Ao,&bv));

  aa = (Mat_SeqAIJ*)(Ad)->data;
  bb = (Mat_SeqAIJ*)(Ao)->data;
  ai = aa->i;
  aj = aa->j;
  bi = bb->i;
  bj = bb->j;

  rstart = A->rmap->rstart;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = (PetscInt64)aa->nz + bb->nz; /* make sure the sum won't overflow PetscInt */
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    PetscCall(PetscMalloc1(nz,&val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
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
        PetscCall(PetscMUMPSIntCast(irow + shift,&row[jj]));
        PetscCall(PetscMUMPSIntCast(rstart + ajj[j] + shift,&col[jj]));
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(PetscMUMPSIntCast(irow + shift,&row[jj]));
        PetscCall(PetscMUMPSIntCast(garray[bjj[j]] + shift,&col[jj]));
      }
      val[jj++] = v2[j];
    }
    irow++;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(Ad,&av));
  PetscCall(MatSeqAIJRestoreArrayRead(Ao,&bv));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpibaij_mpiaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  Mat_MPIBAIJ       *mat    = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ       *aa     = (Mat_SeqBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ       *bb     = (Mat_SeqBAIJ*)(mat->B)->data;
  const PetscInt    *ai     = aa->i, *bi = bb->i, *aj = aa->j, *bj = bb->j,*ajj, *bjj;
  const PetscInt    *garray = mat->garray,mbs=mat->mbs,rstart=A->rmap->rstart;
  const PetscInt    bs2=mat->bs2;
  PetscInt          bs;
  PetscInt64        nz,i,j,k,n,jj,irow,countA,countB,idx;
  PetscMUMPSInt     *row,*col;
  const PetscScalar *av=aa->a, *bv=bb->a,*v1,*v2;
  PetscScalar       *val;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(A,&bs));
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = bs2*(aa->nz + bb->nz);
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    PetscCall(PetscMalloc1(nz,&val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
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
            PetscCall(PetscMUMPSIntCast(irow + n + shift,&row[jj]));
            PetscCall(PetscMUMPSIntCast(rstart + bs*ajj[k] + j + shift,&col[jj]));
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
            PetscCall(PetscMUMPSIntCast(irow + n + shift,&row[jj]));
            PetscCall(PetscMUMPSIntCast(bs*garray[bjj[k]] + j + shift,&col[jj]));
          }
          val[jj++] = v2[idx++];
        }
      }
    }
    irow += bs;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpiaij_mpisbaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  const PetscInt    *ai, *aj,*adiag, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscInt64        rstart,nz,nza,nzb,i,j,jj,irow,countA,countB;
  PetscMUMPSInt     *row,*col;
  const PetscScalar *av, *bv,*v1,*v2;
  PetscScalar       *val;
  Mat               Ad,Ao;
  Mat_SeqAIJ        *aa;
  Mat_SeqAIJ        *bb;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatGetOption(A,MAT_HERMITIAN,&hermitian));
  PetscCheck(!hermitian,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  PetscCall(MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray));
  PetscCall(MatSeqAIJGetArrayRead(Ad,&av));
  PetscCall(MatSeqAIJGetArrayRead(Ao,&bv));

  aa    = (Mat_SeqAIJ*)(Ad)->data;
  bb    = (Mat_SeqAIJ*)(Ao)->data;
  ai    = aa->i;
  aj    = aa->j;
  adiag = aa->diag;
  bi    = bb->i;
  bj    = bb->j;

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
    PetscCall(PetscMalloc2(nz,&row,nz,&col));
    PetscCall(PetscMalloc1(nz,&val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
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
        PetscCall(PetscMUMPSIntCast(irow + shift,&row[jj]));
        PetscCall(PetscMUMPSIntCast(rstart + ajj[j] + shift,&col[jj]));
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (garray[bjj[j]] > rstart) {
        if (reuse == MAT_INITIAL_MATRIX) {
          PetscCall(PetscMUMPSIntCast(irow + shift,&row[jj]));
          PetscCall(PetscMUMPSIntCast(garray[bjj[j]] + shift,&col[jj]));
        }
        val[jj++] = v2[j];
      }
    }
    irow++;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(Ad,&av));
  PetscCall(MatSeqAIJRestoreArrayRead(Ao,&bv));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MUMPS(Mat A)
{
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->data;

  PetscFunctionBegin;
  PetscCall(PetscFree2(mumps->id.sol_loc,mumps->id.isol_loc));
  PetscCall(VecScatterDestroy(&mumps->scat_rhs));
  PetscCall(VecScatterDestroy(&mumps->scat_sol));
  PetscCall(VecDestroy(&mumps->b_seq));
  PetscCall(VecDestroy(&mumps->x_seq));
  PetscCall(PetscFree(mumps->id.perm_in));
  PetscCall(PetscFree2(mumps->irn,mumps->jcn));
  PetscCall(PetscFree(mumps->val_alloc));
  PetscCall(PetscFree(mumps->info));
  PetscCall(MatMumpsResetSchur_Private(mumps));
  mumps->id.job = JOB_END;
  PetscMUMPS_c(mumps);
  PetscCheck(mumps->id.INFOG(1) >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in MatDestroy_MUMPS: INFOG(1)=%d",mumps->id.INFOG(1));
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  if (mumps->use_petsc_omp_support) {
    PetscCall(PetscOmpCtrlDestroy(&mumps->omp_ctrl));
    PetscCall(PetscFree2(mumps->rhs_loc,mumps->rhs_recvbuf));
    PetscCall(PetscFree3(mumps->rhs_nrow,mumps->rhs_recvcounts,mumps->rhs_disps));
  }
#endif
  PetscCall(PetscFree(mumps->ia_alloc));
  PetscCall(PetscFree(mumps->ja_alloc));
  PetscCall(PetscFree(mumps->recvcount));
  PetscCall(PetscFree(mumps->reqs));
  PetscCall(PetscFree(mumps->irhs_loc));
  if (mumps->mumps_comm != MPI_COMM_NULL) PetscCall(PetscCommRestoreComm(PetscObjectComm((PetscObject)A),&mumps->mumps_comm));
  PetscCall(PetscFree(A->data));

  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorSetSchurIS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorCreateSchurComplement_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsSetIcntl_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetIcntl_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsSetCntl_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetCntl_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInfo_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInfog_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetRinfo_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetRinfog_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInverse_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInverseTranspose_C",NULL));
  PetscFunctionReturn(0);
}

/* Set up the distributed RHS info for MUMPS. <nrhs> is the number of RHS. <array> points to start of RHS on the local processor. */
static PetscErrorCode MatMumpsSetUpDistRHSInfo(Mat A,PetscInt nrhs,const PetscScalar *array)
{
  Mat_MUMPS          *mumps=(Mat_MUMPS*)A->data;
  const PetscMPIInt  ompsize=mumps->omp_comm_size;
  PetscInt           i,m,M,rstart;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&M,NULL));
  PetscCall(MatGetLocalSize(A,&m,NULL));
  PetscCheck(M <= PETSC_MUMPS_INT_MAX,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscInt too long for PetscMUMPSInt");
  if (ompsize == 1) {
    if (!mumps->irhs_loc) {
      mumps->nloc_rhs = m;
      PetscCall(PetscMalloc1(m,&mumps->irhs_loc));
      PetscCall(MatGetOwnershipRange(A,&rstart,NULL));
      for (i=0; i<m; i++) mumps->irhs_loc[i] = rstart+i+1; /* use 1-based indices */
    }
    mumps->id.rhs_loc = (MumpsScalar*)array;
  } else {
  #if defined(PETSC_HAVE_OPENMP_SUPPORT)
    const PetscInt  *ranges;
    PetscMPIInt     j,k,sendcount,*petsc_ranks,*omp_ranks;
    MPI_Group       petsc_group,omp_group;
    PetscScalar     *recvbuf=NULL;

    if (mumps->is_omp_master) {
      /* Lazily initialize the omp stuff for distributed rhs */
      if (!mumps->irhs_loc) {
        PetscCall(PetscMalloc2(ompsize,&omp_ranks,ompsize,&petsc_ranks));
        PetscCall(PetscMalloc3(ompsize,&mumps->rhs_nrow,ompsize,&mumps->rhs_recvcounts,ompsize,&mumps->rhs_disps));
        PetscCallMPI(MPI_Comm_group(mumps->petsc_comm,&petsc_group));
        PetscCallMPI(MPI_Comm_group(mumps->omp_comm,&omp_group));
        for (j=0; j<ompsize; j++) omp_ranks[j] = j;
        PetscCallMPI(MPI_Group_translate_ranks(omp_group,ompsize,omp_ranks,petsc_group,petsc_ranks));

        /* Populate mumps->irhs_loc[], rhs_nrow[] */
        mumps->nloc_rhs = 0;
        PetscCall(MatGetOwnershipRanges(A,&ranges));
        for (j=0; j<ompsize; j++) {
          mumps->rhs_nrow[j] = ranges[petsc_ranks[j]+1] - ranges[petsc_ranks[j]];
          mumps->nloc_rhs   += mumps->rhs_nrow[j];
        }
        PetscCall(PetscMalloc1(mumps->nloc_rhs,&mumps->irhs_loc));
        for (j=k=0; j<ompsize; j++) {
          for (i=ranges[petsc_ranks[j]]; i<ranges[petsc_ranks[j]+1]; i++,k++) mumps->irhs_loc[k] = i+1; /* uses 1-based indices */
        }

        PetscCall(PetscFree2(omp_ranks,petsc_ranks));
        PetscCallMPI(MPI_Group_free(&petsc_group));
        PetscCallMPI(MPI_Group_free(&omp_group));
      }

      /* Realloc buffers when current nrhs is bigger than what we have met */
      if (nrhs > mumps->max_nrhs) {
        PetscCall(PetscFree2(mumps->rhs_loc,mumps->rhs_recvbuf));
        PetscCall(PetscMalloc2(mumps->nloc_rhs*nrhs,&mumps->rhs_loc,mumps->nloc_rhs*nrhs,&mumps->rhs_recvbuf));
        mumps->max_nrhs = nrhs;
      }

      /* Setup recvcounts[], disps[], recvbuf on omp rank 0 for the upcoming MPI_Gatherv */
      for (j=0; j<ompsize; j++) PetscCall(PetscMPIIntCast(mumps->rhs_nrow[j]*nrhs,&mumps->rhs_recvcounts[j]));
      mumps->rhs_disps[0] = 0;
      for (j=1; j<ompsize; j++) {
        mumps->rhs_disps[j] = mumps->rhs_disps[j-1] + mumps->rhs_recvcounts[j-1];
        PetscCheck(mumps->rhs_disps[j] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscMPIInt overflow!");
      }
      recvbuf = (nrhs == 1) ? mumps->rhs_loc : mumps->rhs_recvbuf; /* Directly use rhs_loc[] as recvbuf. Single rhs is common in Ax=b */
    }

    PetscCall(PetscMPIIntCast(m*nrhs,&sendcount));
    PetscCallMPI(MPI_Gatherv(array,sendcount,MPIU_SCALAR,recvbuf,mumps->rhs_recvcounts,mumps->rhs_disps,MPIU_SCALAR,0,mumps->omp_comm));

    if (mumps->is_omp_master) {
      if (nrhs > 1) { /* Copy & re-arrange data from rhs_recvbuf[] to mumps->rhs_loc[] only when there are multiple rhs */
        PetscScalar *dst,*dstbase = mumps->rhs_loc;
        for (j=0; j<ompsize; j++) {
          const PetscScalar *src = mumps->rhs_recvbuf + mumps->rhs_disps[j];
          dst = dstbase;
          for (i=0; i<nrhs; i++) {
            PetscCall(PetscArraycpy(dst,src,mumps->rhs_nrow[j]));
            src += mumps->rhs_nrow[j];
            dst += mumps->nloc_rhs;
          }
          dstbase += mumps->rhs_nrow[j];
        }
      }
      mumps->id.rhs_loc = (MumpsScalar*)mumps->rhs_loc;
    }
  #endif /* PETSC_HAVE_OPENMP_SUPPORT */
  }
  mumps->id.nrhs     = nrhs;
  mumps->id.nloc_rhs = mumps->nloc_rhs;
  mumps->id.lrhs_loc = mumps->nloc_rhs;
  mumps->id.irhs_loc = mumps->irhs_loc;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_MUMPS(Mat A,Vec b,Vec x)
{
  Mat_MUMPS          *mumps=(Mat_MUMPS*)A->data;
  const PetscScalar  *rarray = NULL;
  PetscScalar        *array;
  IS                 is_iden,is_petsc;
  PetscInt           i;
  PetscBool          second_solve = PETSC_FALSE;
  static PetscBool   cite1 = PETSC_FALSE,cite2 = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister("@article{MUMPS01,\n  author = {P.~R. Amestoy and I.~S. Duff and J.-Y. L'Excellent and J. Koster},\n  title = {A fully asynchronous multifrontal solver using distributed dynamic scheduling},\n  journal = {SIAM Journal on Matrix Analysis and Applications},\n  volume = {23},\n  number = {1},\n  pages = {15--41},\n  year = {2001}\n}\n",&cite1));
  PetscCall(PetscCitationsRegister("@article{MUMPS02,\n  author = {P.~R. Amestoy and A. Guermouche and J.-Y. L'Excellent and S. Pralet},\n  title = {Hybrid scheduling for the parallel solution of linear systems},\n  journal = {Parallel Computing},\n  volume = {32},\n  number = {2},\n  pages = {136--156},\n  year = {2006}\n}\n",&cite2));

  if (A->factorerrortype) {
    PetscCall(PetscInfo(A,"MatSolve is called with singular matrix factor, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
    PetscCall(VecSetInf(x));
    PetscFunctionReturn(0);
  }

  mumps->id.nrhs = 1;
  if (mumps->petsc_size > 1) {
    if (mumps->ICNTL20 == 10) {
      mumps->id.ICNTL(20) = 10; /* dense distributed RHS */
      PetscCall(VecGetArrayRead(b,&rarray));
      PetscCall(MatMumpsSetUpDistRHSInfo(A,1,rarray));
    } else {
      mumps->id.ICNTL(20) = 0; /* dense centralized RHS; Scatter b into a sequential rhs vector*/
      PetscCall(VecScatterBegin(mumps->scat_rhs,b,mumps->b_seq,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mumps->scat_rhs,b,mumps->b_seq,INSERT_VALUES,SCATTER_FORWARD));
      if (!mumps->myid) {
        PetscCall(VecGetArray(mumps->b_seq,&array));
        mumps->id.rhs = (MumpsScalar*)array;
      }
    }
  } else {  /* petsc_size == 1 */
    mumps->id.ICNTL(20) = 0; /* dense centralized RHS */
    PetscCall(VecCopy(b,x));
    PetscCall(VecGetArray(x,&array));
    mumps->id.rhs = (MumpsScalar*)array;
  }

  /*
     handle condensation step of Schur complement (if any)
     We set by default ICNTL(26) == -1 when Schur indices have been provided by the user.
     According to MUMPS (5.0.0) manual, any value should be harmful during the factorization phase
     Unless the user provides a valid value for ICNTL(26), MatSolve and MatMatSolve routines solve the full system.
     This requires an extra call to PetscMUMPS_c and the computation of the factors for S
  */
  if (mumps->id.size_schur > 0 && (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2)) {
    PetscCheck(mumps->petsc_size <= 1,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Parallel Schur complements not yet supported from PETSc");
    second_solve = PETSC_TRUE;
    PetscCall(MatMumpsHandleSchur_Private(A,PETSC_FALSE));
  }
  /* solve phase */
  /*-------------*/
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps);
  PetscCheck(mumps->id.INFOG(1) >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d",mumps->id.INFOG(1));

  /* handle expansion step of Schur complement (if any) */
  if (second_solve) {
    PetscCall(MatMumpsHandleSchur_Private(A,PETSC_TRUE));
  }

  if (mumps->petsc_size > 1) { /* convert mumps distributed solution to petsc mpi x */
    if (mumps->scat_sol && mumps->ICNTL9_pre != mumps->id.ICNTL(9)) {
      /* when id.ICNTL(9) changes, the contents of lsol_loc may change (not its size, lsol_loc), recreates scat_sol */
      PetscCall(VecScatterDestroy(&mumps->scat_sol));
    }
    if (!mumps->scat_sol) { /* create scatter scat_sol */
      PetscInt *isol2_loc=NULL;
      PetscCall(ISCreateStride(PETSC_COMM_SELF,mumps->id.lsol_loc,0,1,&is_iden)); /* from */
      PetscCall(PetscMalloc1(mumps->id.lsol_loc,&isol2_loc));
      for (i=0; i<mumps->id.lsol_loc; i++) isol2_loc[i] = mumps->id.isol_loc[i]-1; /* change Fortran style to C style */
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,mumps->id.lsol_loc,isol2_loc,PETSC_OWN_POINTER,&is_petsc));  /* to */
      PetscCall(VecScatterCreate(mumps->x_seq,is_iden,x,is_petsc,&mumps->scat_sol));
      PetscCall(ISDestroy(&is_iden));
      PetscCall(ISDestroy(&is_petsc));
      mumps->ICNTL9_pre = mumps->id.ICNTL(9); /* save current value of id.ICNTL(9) */
    }

    PetscCall(VecScatterBegin(mumps->scat_sol,mumps->x_seq,x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mumps->scat_sol,mumps->x_seq,x,INSERT_VALUES,SCATTER_FORWARD));
  }

  if (mumps->petsc_size > 1) {
    if (mumps->ICNTL20 == 10) {
      PetscCall(VecRestoreArrayRead(b,&rarray));
    } else if (!mumps->myid) {
      PetscCall(VecRestoreArray(mumps->b_seq,&array));
    }
  } else PetscCall(VecRestoreArray(x,&array));

  PetscCall(PetscLogFlops(2.0*mumps->id.RINFO(3)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_MUMPS(Mat A,Vec b,Vec x)
{
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->data;

  PetscFunctionBegin;
  mumps->id.ICNTL(9) = 0;
  PetscCall(MatSolve_MUMPS(A,b,x));
  mumps->id.ICNTL(9) = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatSolve_MUMPS(Mat A,Mat B,Mat X)
{
  Mat               Bt = NULL;
  PetscBool         denseX,denseB,flg,flgT;
  Mat_MUMPS         *mumps=(Mat_MUMPS*)A->data;
  PetscInt          i,nrhs,M;
  PetscScalar       *array;
  const PetscScalar *rbray;
  PetscInt          lsol_loc,nlsol_loc,*idxx,iidx = 0;
  PetscMUMPSInt     *isol_loc,*isol_loc_save;
  PetscScalar       *bray,*sol_loc,*sol_loc_save;
  IS                is_to,is_from;
  PetscInt          k,proc,j,m,myrstart;
  const PetscInt    *rstart;
  Vec               v_mpi,msol_loc;
  VecScatter        scat_sol;
  Vec               b_seq;
  VecScatter        scat_rhs;
  PetscScalar       *aa;
  PetscInt          spnr,*ia,*ja;
  Mat_MPIAIJ        *b = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X,&denseX,MATSEQDENSE,MATMPIDENSE,NULL));
  PetscCheck(denseX,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&denseB,MATSEQDENSE,MATMPIDENSE,NULL));
  if (denseB) {
    PetscCheck(B->rmap->n == X->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix B and X must have same row distribution");
    mumps->id.ICNTL(20)= 0; /* dense RHS */
  } else { /* sparse B */
    PetscCheck(X != B,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_IDN,"X and B must be different matrices");
    PetscCall(PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&flgT));
    if (flgT) { /* input B is transpose of actural RHS matrix,
                 because mumps requires sparse compressed COLUMN storage! See MatMatTransposeSolve_MUMPS() */
      PetscCall(MatTransposeGetMat(B,&Bt));
    } else SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONG,"Matrix B must be MATTRANSPOSEMAT matrix");
    mumps->id.ICNTL(20)= 1; /* sparse RHS */
  }

  PetscCall(MatGetSize(B,&M,&nrhs));
  mumps->id.nrhs = nrhs;
  mumps->id.lrhs = M;
  mumps->id.rhs  = NULL;

  if (mumps->petsc_size == 1) {
    PetscScalar *aa;
    PetscInt    spnr,*ia,*ja;
    PetscBool   second_solve = PETSC_FALSE;

    PetscCall(MatDenseGetArray(X,&array));
    mumps->id.rhs = (MumpsScalar*)array;

    if (denseB) {
      /* copy B to X */
      PetscCall(MatDenseGetArrayRead(B,&rbray));
      PetscCall(PetscArraycpy(array,rbray,M*nrhs));
      PetscCall(MatDenseRestoreArrayRead(B,&rbray));
    } else { /* sparse B */
      PetscCall(MatSeqAIJGetArray(Bt,&aa));
      PetscCall(MatGetRowIJ(Bt,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
      PetscCall(PetscMUMPSIntCSRCast(mumps,spnr,ia,ja,&mumps->id.irhs_ptr,&mumps->id.irhs_sparse,&mumps->id.nz_rhs));
      mumps->id.rhs_sparse  = (MumpsScalar*)aa;
    }
    /* handle condensation step of Schur complement (if any) */
    if (mumps->id.size_schur > 0 && (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2)) {
      second_solve = PETSC_TRUE;
      PetscCall(MatMumpsHandleSchur_Private(A,PETSC_FALSE));
    }
    /* solve phase */
    /*-------------*/
    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(mumps);
    PetscCheck(mumps->id.INFOG(1) >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d",mumps->id.INFOG(1));

    /* handle expansion step of Schur complement (if any) */
    if (second_solve) {
      PetscCall(MatMumpsHandleSchur_Private(A,PETSC_TRUE));
    }
    if (!denseB) { /* sparse B */
      PetscCall(MatSeqAIJRestoreArray(Bt,&aa));
      PetscCall(MatRestoreRowIJ(Bt,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot restore IJ structure");
    }
    PetscCall(MatDenseRestoreArray(X,&array));
    PetscFunctionReturn(0);
  }

  /*--------- parallel case: MUMPS requires rhs B to be centralized on the host! --------*/
  PetscCheck(mumps->petsc_size <= 1 || !mumps->id.ICNTL(19),PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Parallel Schur complements not yet supported from PETSc");

  /* create msol_loc to hold mumps local solution */
  isol_loc_save = mumps->id.isol_loc; /* save it for MatSolve() */
  sol_loc_save  = (PetscScalar*)mumps->id.sol_loc;

  lsol_loc  = mumps->id.lsol_loc;
  nlsol_loc = nrhs*lsol_loc;     /* length of sol_loc */
  PetscCall(PetscMalloc2(nlsol_loc,&sol_loc,lsol_loc,&isol_loc));
  mumps->id.sol_loc  = (MumpsScalar*)sol_loc;
  mumps->id.isol_loc = isol_loc;

  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,nlsol_loc,(PetscScalar*)sol_loc,&msol_loc));

  if (denseB) {
    if (mumps->ICNTL20 == 10) {
      mumps->id.ICNTL(20) = 10; /* dense distributed RHS */
      PetscCall(MatDenseGetArrayRead(B,&rbray));
      PetscCall(MatMumpsSetUpDistRHSInfo(A,nrhs,rbray));
      PetscCall(MatDenseRestoreArrayRead(B,&rbray));
      PetscCall(MatGetLocalSize(B,&m,NULL));
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),1,nrhs*m,nrhs*M,NULL,&v_mpi));
    } else {
      mumps->id.ICNTL(20) = 0; /* dense centralized RHS */
      /* TODO: Because of non-contiguous indices, the created vecscatter scat_rhs is not done in MPI_Gather, resulting in
        very inefficient communication. An optimization is to use VecScatterCreateToZero to gather B to rank 0. Then on rank
        0, re-arrange B into desired order, which is a local operation.
      */

      /* scatter v_mpi to b_seq because MUMPS before 5.3.0 only supports centralized rhs */
      /* wrap dense rhs matrix B into a vector v_mpi */
      PetscCall(MatGetLocalSize(B,&m,NULL));
      PetscCall(MatDenseGetArray(B,&bray));
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),1,nrhs*m,nrhs*M,(const PetscScalar*)bray,&v_mpi));
      PetscCall(MatDenseRestoreArray(B,&bray));

      /* scatter v_mpi to b_seq in proc[0]. MUMPS requires rhs to be centralized on the host! */
      if (!mumps->myid) {
        PetscInt *idx;
        /* idx: maps from k-th index of v_mpi to (i,j)-th global entry of B */
        PetscCall(PetscMalloc1(nrhs*M,&idx));
        PetscCall(MatGetOwnershipRanges(B,&rstart));
        k = 0;
        for (proc=0; proc<mumps->petsc_size; proc++) {
          for (j=0; j<nrhs; j++) {
            for (i=rstart[proc]; i<rstart[proc+1]; i++) idx[k++] = j*M + i;
          }
        }

        PetscCall(VecCreateSeq(PETSC_COMM_SELF,nrhs*M,&b_seq));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nrhs*M,idx,PETSC_OWN_POINTER,&is_to));
        PetscCall(ISCreateStride(PETSC_COMM_SELF,nrhs*M,0,1,&is_from));
      } else {
        PetscCall(VecCreateSeq(PETSC_COMM_SELF,0,&b_seq));
        PetscCall(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_to));
        PetscCall(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_from));
      }
      PetscCall(VecScatterCreate(v_mpi,is_from,b_seq,is_to,&scat_rhs));
      PetscCall(VecScatterBegin(scat_rhs,v_mpi,b_seq,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(ISDestroy(&is_to));
      PetscCall(ISDestroy(&is_from));
      PetscCall(VecScatterEnd(scat_rhs,v_mpi,b_seq,INSERT_VALUES,SCATTER_FORWARD));

      if (!mumps->myid) { /* define rhs on the host */
        PetscCall(VecGetArray(b_seq,&bray));
        mumps->id.rhs = (MumpsScalar*)bray;
        PetscCall(VecRestoreArray(b_seq,&bray));
      }
    }
  } else { /* sparse B */
    b = (Mat_MPIAIJ*)Bt->data;

    /* wrap dense X into a vector v_mpi */
    PetscCall(MatGetLocalSize(X,&m,NULL));
    PetscCall(MatDenseGetArray(X,&bray));
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)X),1,nrhs*m,nrhs*M,(const PetscScalar*)bray,&v_mpi));
    PetscCall(MatDenseRestoreArray(X,&bray));

    if (!mumps->myid) {
      PetscCall(MatSeqAIJGetArray(b->A,&aa));
      PetscCall(MatGetRowIJ(b->A,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
      PetscCall(PetscMUMPSIntCSRCast(mumps,spnr,ia,ja,&mumps->id.irhs_ptr,&mumps->id.irhs_sparse,&mumps->id.nz_rhs));
      mumps->id.rhs_sparse  = (MumpsScalar*)aa;
    } else {
      mumps->id.irhs_ptr    = NULL;
      mumps->id.irhs_sparse = NULL;
      mumps->id.nz_rhs      = 0;
      mumps->id.rhs_sparse  = NULL;
    }
  }

  /* solve phase */
  /*-------------*/
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps);
  PetscCheck(mumps->id.INFOG(1) >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d",mumps->id.INFOG(1));

  /* scatter mumps distributed solution to petsc vector v_mpi, which shares local arrays with solution matrix X */
  PetscCall(MatDenseGetArray(X,&array));
  PetscCall(VecPlaceArray(v_mpi,array));

  /* create scatter scat_sol */
  PetscCall(MatGetOwnershipRanges(X,&rstart));
  /* iidx: index for scatter mumps solution to petsc X */

  PetscCall(ISCreateStride(PETSC_COMM_SELF,nlsol_loc,0,1,&is_from));
  PetscCall(PetscMalloc1(nlsol_loc,&idxx));
  for (i=0; i<lsol_loc; i++) {
    isol_loc[i] -= 1; /* change Fortran style to C style. isol_loc[i+j*lsol_loc] contains x[isol_loc[i]] in j-th vector */

    for (proc=0; proc<mumps->petsc_size; proc++) {
      if (isol_loc[i] >= rstart[proc] && isol_loc[i] < rstart[proc+1]) {
        myrstart = rstart[proc];
        k        = isol_loc[i] - myrstart;        /* local index on 1st column of petsc vector X */
        iidx     = k + myrstart*nrhs;             /* maps mumps isol_loc[i] to petsc index in X */
        m        = rstart[proc+1] - rstart[proc]; /* rows of X for this proc */
        break;
      }
    }

    for (j=0; j<nrhs; j++) idxx[i+j*lsol_loc] = iidx + j*m;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nlsol_loc,idxx,PETSC_COPY_VALUES,&is_to));
  PetscCall(VecScatterCreate(msol_loc,is_from,v_mpi,is_to,&scat_sol));
  PetscCall(VecScatterBegin(scat_sol,msol_loc,v_mpi,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(ISDestroy(&is_from));
  PetscCall(ISDestroy(&is_to));
  PetscCall(VecScatterEnd(scat_sol,msol_loc,v_mpi,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(MatDenseRestoreArray(X,&array));

  /* free spaces */
  mumps->id.sol_loc  = (MumpsScalar*)sol_loc_save;
  mumps->id.isol_loc = isol_loc_save;

  PetscCall(PetscFree2(sol_loc,isol_loc));
  PetscCall(PetscFree(idxx));
  PetscCall(VecDestroy(&msol_loc));
  PetscCall(VecDestroy(&v_mpi));
  if (!denseB) {
    if (!mumps->myid) {
      b = (Mat_MPIAIJ*)Bt->data;
      PetscCall(MatSeqAIJRestoreArray(b->A,&aa));
      PetscCall(MatRestoreRowIJ(b->A,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot restore IJ structure");
    }
  } else {
    if (mumps->ICNTL20 == 0) {
      PetscCall(VecDestroy(&b_seq));
      PetscCall(VecScatterDestroy(&scat_rhs));
    }
  }
  PetscCall(VecScatterDestroy(&scat_sol));
  PetscCall(PetscLogFlops(2.0*nrhs*mumps->id.RINFO(3)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeSolve_MUMPS(Mat A,Mat Bt,Mat X)
{
  PetscBool      flg;
  Mat            B;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)Bt,&flg,MATSEQAIJ,MATMPIAIJ,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)Bt),PETSC_ERR_ARG_WRONG,"Matrix Bt must be MATAIJ matrix");

  /* Create B=Bt^T that uses Bt's data structure */
  PetscCall(MatCreateTranspose(Bt,&B));

  PetscCall(MatMatSolve_MUMPS(A,B,X));
  PetscCall(MatDestroy(&B));
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
PetscErrorCode MatGetInertia_SBAIJMUMPS(Mat F,PetscInt *nneg,PetscInt *nzero,PetscInt *npos)
{
  Mat_MUMPS      *mumps =(Mat_MUMPS*)F->data;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)F),&size));
  /* MUMPS 4.3.1 calls ScaLAPACK when ICNTL(13)=0 (default), which does not offer the possibility to compute the inertia of a dense matrix. Set ICNTL(13)=1 to skip ScaLAPACK */
  PetscCheck(size <= 1 || mumps->id.ICNTL(13) == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ICNTL(13)=%d. -mat_mumps_icntl_13 must be set as 1 for correct global matrix inertia",mumps->id.INFOG(13));

  if (nneg) *nneg = mumps->id.INFOG(12);
  if (nzero || npos) {
    PetscCheck(mumps->id.ICNTL(24) == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"-mat_mumps_icntl_24 must be set as 1 for null pivot row detection");
    if (nzero) *nzero = mumps->id.INFOG(28);
    if (npos) *npos   = F->rmap->N - (mumps->id.INFOG(12) + mumps->id.INFOG(28));
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatMumpsGatherNonzerosOnMaster(MatReuse reuse,Mat_MUMPS *mumps)
{
  PetscInt       i,nreqs;
  PetscMUMPSInt  *irn,*jcn;
  PetscMPIInt    count;
  PetscInt64     totnnz,remain;
  const PetscInt osize=mumps->omp_comm_size;
  PetscScalar    *val;

  PetscFunctionBegin;
  if (osize > 1) {
    if (reuse == MAT_INITIAL_MATRIX) {
      /* master first gathers counts of nonzeros to receive */
      if (mumps->is_omp_master) PetscCall(PetscMalloc1(osize,&mumps->recvcount));
      PetscCallMPI(MPI_Gather(&mumps->nnz,1,MPIU_INT64,mumps->recvcount,1,MPIU_INT64,0/*master*/,mumps->omp_comm));

      /* Then each computes number of send/recvs */
      if (mumps->is_omp_master) {
        /* Start from 1 since self communication is not done in MPI */
        nreqs = 0;
        for (i=1; i<osize; i++) nreqs += (mumps->recvcount[i]+PETSC_MPI_INT_MAX-1)/PETSC_MPI_INT_MAX;
      } else {
        nreqs = (mumps->nnz+PETSC_MPI_INT_MAX-1)/PETSC_MPI_INT_MAX;
      }
      PetscCall(PetscMalloc1(nreqs*3,&mumps->reqs)); /* Triple the requests since we send irn, jcn and val seperately */

      /* The following code is doing a very simple thing: omp_master rank gathers irn/jcn/val from others.
         MPI_Gatherv would be enough if it supports big counts > 2^31-1. Since it does not, and mumps->nnz
         might be a prime number > 2^31-1, we have to slice the message. Note omp_comm_size
         is very small, the current approach should have no extra overhead compared to MPI_Gatherv.
       */
      nreqs = 0; /* counter for actual send/recvs */
      if (mumps->is_omp_master) {
        for (i=0,totnnz=0; i<osize; i++) totnnz += mumps->recvcount[i]; /* totnnz = sum of nnz over omp_comm */
        PetscCall(PetscMalloc2(totnnz,&irn,totnnz,&jcn));
        PetscCall(PetscMalloc1(totnnz,&val));

        /* Self communication */
        PetscCall(PetscArraycpy(irn,mumps->irn,mumps->nnz));
        PetscCall(PetscArraycpy(jcn,mumps->jcn,mumps->nnz));
        PetscCall(PetscArraycpy(val,mumps->val,mumps->nnz));

        /* Replace mumps->irn/jcn etc on master with the newly allocated bigger arrays */
        PetscCall(PetscFree2(mumps->irn,mumps->jcn));
        PetscCall(PetscFree(mumps->val_alloc));
        mumps->nnz = totnnz;
        mumps->irn = irn;
        mumps->jcn = jcn;
        mumps->val = mumps->val_alloc = val;

        irn += mumps->recvcount[0]; /* recvcount[0] is old mumps->nnz on omp rank 0 */
        jcn += mumps->recvcount[0];
        val += mumps->recvcount[0];

        /* Remote communication */
        for (i=1; i<osize; i++) {
          count  = PetscMin(mumps->recvcount[i],PETSC_MPI_INT_MAX);
          remain = mumps->recvcount[i] - count;
          while (count>0) {
            PetscCallMPI(MPI_Irecv(irn,count,MPIU_MUMPSINT,i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
            PetscCallMPI(MPI_Irecv(jcn,count,MPIU_MUMPSINT,i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
            PetscCallMPI(MPI_Irecv(val,count,MPIU_SCALAR,  i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
            irn    += count;
            jcn    += count;
            val    += count;
            count   = PetscMin(remain,PETSC_MPI_INT_MAX);
            remain -= count;
          }
        }
      } else {
        irn    = mumps->irn;
        jcn    = mumps->jcn;
        val    = mumps->val;
        count  = PetscMin(mumps->nnz,PETSC_MPI_INT_MAX);
        remain = mumps->nnz - count;
        while (count>0) {
          PetscCallMPI(MPI_Isend(irn,count,MPIU_MUMPSINT,0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
          PetscCallMPI(MPI_Isend(jcn,count,MPIU_MUMPSINT,0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
          PetscCallMPI(MPI_Isend(val,count,MPIU_SCALAR,  0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
          irn    += count;
          jcn    += count;
          val    += count;
          count   = PetscMin(remain,PETSC_MPI_INT_MAX);
          remain -= count;
        }
      }
    } else {
      nreqs = 0;
      if (mumps->is_omp_master) {
        val = mumps->val + mumps->recvcount[0];
        for (i=1; i<osize; i++) { /* Remote communication only since self data is already in place */
          count  = PetscMin(mumps->recvcount[i],PETSC_MPI_INT_MAX);
          remain = mumps->recvcount[i] - count;
          while (count>0) {
            PetscCallMPI(MPI_Irecv(val,count,MPIU_SCALAR,i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
            val    += count;
            count   = PetscMin(remain,PETSC_MPI_INT_MAX);
            remain -= count;
          }
        }
      } else {
        val    = mumps->val;
        count  = PetscMin(mumps->nnz,PETSC_MPI_INT_MAX);
        remain = mumps->nnz - count;
        while (count>0) {
          PetscCallMPI(MPI_Isend(val,count,MPIU_SCALAR,0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]));
          val    += count;
          count   = PetscMin(remain,PETSC_MPI_INT_MAX);
          remain -= count;
        }
      }
    }
    PetscCallMPI(MPI_Waitall(nreqs,mumps->reqs,MPI_STATUSES_IGNORE));
    mumps->tag++; /* It is totally fine for above send/recvs to share one mpi tag */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorNumeric_MUMPS(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps =(Mat_MUMPS*)(F)->data;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0 && !(mumps->id.INFOG(1) == -16 && mumps->id.INFOG(1) == 0)) {
    if (mumps->id.INFOG(1) == -6) {
      PetscCall(PetscInfo(A,"MatFactorNumeric is called with singular matrix structure, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
    }
    PetscCall(PetscInfo(A,"MatFactorNumeric is called after analysis phase fails, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
    PetscFunctionReturn(0);
  }

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_REUSE_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_REUSE_MATRIX,mumps));

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
  PetscMUMPS_c(mumps);
  if (mumps->id.INFOG(1) < 0) {
    if (A->erroriffailure) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d",mumps->id.INFOG(1),mumps->id.INFO(2));
    } else {
      if (mumps->id.INFOG(1) == -10) { /* numerically singular matrix */
        PetscCall(PetscInfo(F,"matrix is numerically singular, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      } else if (mumps->id.INFOG(1) == -13) {
        PetscCall(PetscInfo(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, cannot allocate required memory %d megabytes\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else if (mumps->id.INFOG(1) == -8 || mumps->id.INFOG(1) == -9 || (-16 < mumps->id.INFOG(1) && mumps->id.INFOG(1) < -10)) {
        PetscCall(PetscInfo(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d, problem with workarray \n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else {
        PetscCall(PetscInfo(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_OTHER;
      }
    }
  }
  PetscCheck(mumps->myid || mumps->id.ICNTL(16) <= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"  mumps->id.ICNTL(16):=%d",mumps->id.INFOG(16));

  F->assembled    = PETSC_TRUE;

  if (F->schur) { /* reset Schur status to unfactored */
#if defined(PETSC_HAVE_CUDA)
    F->schur->offloadmask = PETSC_OFFLOAD_CPU;
#endif
    if (mumps->id.ICNTL(19) == 1) { /* stored by rows */
      mumps->id.ICNTL(19) = 2;
      PetscCall(MatTranspose(F->schur,MAT_INPLACE_MATRIX,&F->schur));
    }
    PetscCall(MatFactorRestoreSchurComplement(F,NULL,MAT_FACTOR_SCHUR_UNFACTORED));
  }

  /* just to be sure that ICNTL(19) value returned by a call from MatMumpsGetIcntl is always consistent */
  if (!mumps->sym && mumps->id.ICNTL(19) && mumps->id.ICNTL(19) != 1) mumps->id.ICNTL(19) = 3;

  if (!mumps->is_omp_master) mumps->id.INFO(23) = 0;
  if (mumps->petsc_size > 1) {
    PetscInt    lsol_loc;
    PetscScalar *sol_loc;

    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ));

    /* distributed solution; Create x_seq=sol_loc for repeated use */
    if (mumps->x_seq) {
      PetscCall(VecScatterDestroy(&mumps->scat_sol));
      PetscCall(PetscFree2(mumps->id.sol_loc,mumps->id.isol_loc));
      PetscCall(VecDestroy(&mumps->x_seq));
    }
    lsol_loc = mumps->id.INFO(23); /* length of sol_loc */
    PetscCall(PetscMalloc2(lsol_loc,&sol_loc,lsol_loc,&mumps->id.isol_loc));
    mumps->id.lsol_loc = lsol_loc;
    mumps->id.sol_loc = (MumpsScalar*)sol_loc;
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,lsol_loc,sol_loc,&mumps->x_seq));
  }
  PetscCall(PetscLogFlops(mumps->id.RINFO(2)));
  PetscFunctionReturn(0);
}

/* Sets MUMPS options from the options database */
PetscErrorCode PetscSetMUMPSFromOptions(Mat F, Mat A)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscMUMPSInt  icntl=0;
  PetscInt       info[80],i,ninfo=80,rbs,cbs;
  PetscBool      flg=PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MUMPS Options","Mat");
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_1","ICNTL(1): output stream for error messages","None",mumps->id.ICNTL(1),&icntl,&flg));
  if (flg) mumps->id.ICNTL(1) = icntl;
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_2","ICNTL(2): output stream for diagnostic printing, statistics, and warning","None",mumps->id.ICNTL(2),&icntl,&flg));
  if (flg) mumps->id.ICNTL(2) = icntl;
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_3","ICNTL(3): output stream for global information, collected on the host","None",mumps->id.ICNTL(3),&icntl,&flg));
  if (flg) mumps->id.ICNTL(3) = icntl;

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_4","ICNTL(4): level of printing (0 to 4)","None",mumps->id.ICNTL(4),&icntl,&flg));
  if (flg) mumps->id.ICNTL(4) = icntl;
  if (mumps->id.ICNTL(4) || PetscLogPrintInfo) mumps->id.ICNTL(3) = 6; /* resume MUMPS default id.ICNTL(3) = 6 */

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_6","ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7)","None",mumps->id.ICNTL(6),&icntl,&flg));
  if (flg) mumps->id.ICNTL(6) = icntl;

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_7","ICNTL(7): computes a symmetric permutation in sequential analysis. 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto(default)","None",mumps->id.ICNTL(7),&icntl,&flg));
  if (flg) {
    PetscCheck(icntl != 1 && icntl >= 0 && icntl <= 7,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Valid values are 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto");
    mumps->id.ICNTL(7) = icntl;
  }

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_8","ICNTL(8): scaling strategy (-2 to 8 or 77)","None",mumps->id.ICNTL(8),&mumps->id.ICNTL(8),NULL));
  /* PetscCall(PetscOptionsInt("-mat_mumps_icntl_9","ICNTL(9): computes the solution using A or A^T","None",mumps->id.ICNTL(9),&mumps->id.ICNTL(9),NULL)); handled by MatSolveTranspose_MUMPS() */
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_10","ICNTL(10): max num of refinements","None",mumps->id.ICNTL(10),&mumps->id.ICNTL(10),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_11","ICNTL(11): statistics related to an error analysis (via -ksp_view)","None",mumps->id.ICNTL(11),&mumps->id.ICNTL(11),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_12","ICNTL(12): an ordering strategy for symmetric matrices (0 to 3)","None",mumps->id.ICNTL(12),&mumps->id.ICNTL(12),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_13","ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting","None",mumps->id.ICNTL(13),&mumps->id.ICNTL(13),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_14","ICNTL(14): percentage increase in the estimated working space","None",mumps->id.ICNTL(14),&mumps->id.ICNTL(14),NULL));
  PetscCall(MatGetBlockSizes(A,&rbs,&cbs));
  if (rbs == cbs && rbs > 1) mumps->id.ICNTL(15) = -rbs;
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_15","ICNTL(15): compression of the input matrix resulting from a block format","None",mumps->id.ICNTL(15),&mumps->id.ICNTL(15),&flg));
  if (flg) {
    PetscCheck(mumps->id.ICNTL(15) <= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Positive -mat_mumps_icntl_15 not handled");
    PetscCheck((-mumps->id.ICNTL(15) % cbs == 0) && (-mumps->id.ICNTL(15) % rbs == 0),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The opposite of -mat_mumps_icntl_15 must be a multiple of the column and row blocksizes");
  }
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_19","ICNTL(19): computes the Schur complement","None",mumps->id.ICNTL(19),&mumps->id.ICNTL(19),NULL));
  if (mumps->id.ICNTL(19) <= 0 || mumps->id.ICNTL(19) > 3) { /* reset any schur data (if any) */
    PetscCall(MatDestroy(&F->schur));
    PetscCall(MatMumpsResetSchur_Private(mumps));
  }

  /* Two MPICH Fortran MPI_IN_PLACE binding bugs prevented the use of 'mpich + mumps'. One happened with "mpi4py + mpich + mumps",
     and was reported by Firedrake. See https://bitbucket.org/mpi4py/mpi4py/issues/162/mpi4py-initialization-breaks-fortran
     and a petsc-maint mailing list thread with subject 'MUMPS segfaults in parallel because of ...'
     This bug was fixed by https://github.com/pmodels/mpich/pull/4149. But the fix brought a new bug,
     see https://github.com/pmodels/mpich/issues/5589. This bug was fixed by https://github.com/pmodels/mpich/pull/5590.
     In short, we could not use distributed RHS with MPICH until v4.0b1.
   */
#if PETSC_PKG_MUMPS_VERSION_LT(5,3,0) || (defined(PETSC_HAVE_MPICH_NUMVERSION) && (PETSC_HAVE_MPICH_NUMVERSION < 40000101))
  mumps->ICNTL20 = 0;  /* Centralized dense RHS*/
#else
  mumps->ICNTL20 = 10; /* Distributed dense RHS*/
#endif
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_20","ICNTL(20): give mumps centralized (0) or distributed (10) dense right-hand sides","None",mumps->ICNTL20,&mumps->ICNTL20,&flg));
  PetscCheck(!flg || mumps->ICNTL20 == 10 || mumps->ICNTL20 == 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"ICNTL(20)=%d is not supported by the PETSc/MUMPS interface. Allowed values are 0, 10",(int)mumps->ICNTL20);
#if PETSC_PKG_MUMPS_VERSION_LT(5,3,0)
  PetscCheck(!flg || mumps->ICNTL20 != 10,PETSC_COMM_SELF,PETSC_ERR_SUP,"ICNTL(20)=10 is not supported before MUMPS-5.3.0");
#endif
  /* PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_21","ICNTL(21): the distribution (centralized or distributed) of the solution vectors","None",mumps->id.ICNTL(21),&mumps->id.ICNTL(21),NULL)); we only use distributed solution vector */

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_22","ICNTL(22): in-core/out-of-core factorization and solve (0 or 1)","None",mumps->id.ICNTL(22),&mumps->id.ICNTL(22),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_23","ICNTL(23): max size of the working memory (MB) that can allocate per processor","None",mumps->id.ICNTL(23),&mumps->id.ICNTL(23),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_24","ICNTL(24): detection of null pivot rows (0 or 1)","None",mumps->id.ICNTL(24),&mumps->id.ICNTL(24),NULL));
  if (mumps->id.ICNTL(24)) {
    mumps->id.ICNTL(13) = 1; /* turn-off ScaLAPACK to help with the correct detection of null pivots */
  }

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_25","ICNTL(25): computes a solution of a deficient matrix and a null space basis","None",mumps->id.ICNTL(25),&mumps->id.ICNTL(25),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_26","ICNTL(26): drives the solution phase if a Schur complement matrix","None",mumps->id.ICNTL(26),&mumps->id.ICNTL(26),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_27","ICNTL(27): controls the blocking size for multiple right-hand sides","None",mumps->id.ICNTL(27),&mumps->id.ICNTL(27),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_28","ICNTL(28): use 1 for sequential analysis and ictnl(7) ordering, or 2 for parallel analysis and ictnl(29) ordering","None",mumps->id.ICNTL(28),&mumps->id.ICNTL(28),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_29","ICNTL(29): parallel ordering 1 = ptscotch, 2 = parmetis","None",mumps->id.ICNTL(29),&mumps->id.ICNTL(29),NULL));
  /* PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_30","ICNTL(30): compute user-specified set of entries in inv(A)","None",mumps->id.ICNTL(30),&mumps->id.ICNTL(30),NULL)); */ /* call MatMumpsGetInverse() directly */
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_31","ICNTL(31): indicates which factors may be discarded during factorization","None",mumps->id.ICNTL(31),&mumps->id.ICNTL(31),NULL));
  /* PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_32","ICNTL(32): performs the forward elemination of the right-hand sides during factorization","None",mumps->id.ICNTL(32),&mumps->id.ICNTL(32),NULL));  -- not supported by PETSc API */
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_33","ICNTL(33): compute determinant","None",mumps->id.ICNTL(33),&mumps->id.ICNTL(33),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_35","ICNTL(35): activates Block Low Rank (BLR) based factorization","None",mumps->id.ICNTL(35),&mumps->id.ICNTL(35),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_36","ICNTL(36): choice of BLR factorization variant","None",mumps->id.ICNTL(36),&mumps->id.ICNTL(36),NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_38","ICNTL(38): estimated compression rate of LU factors with BLR","None",mumps->id.ICNTL(38),&mumps->id.ICNTL(38),NULL));

  PetscCall(PetscOptionsReal("-mat_mumps_cntl_1","CNTL(1): relative pivoting threshold","None",mumps->id.CNTL(1),&mumps->id.CNTL(1),NULL));
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_2","CNTL(2): stopping criterion of refinement","None",mumps->id.CNTL(2),&mumps->id.CNTL(2),NULL));
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_3","CNTL(3): absolute pivoting threshold","None",mumps->id.CNTL(3),&mumps->id.CNTL(3),NULL));
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_4","CNTL(4): value for static pivoting","None",mumps->id.CNTL(4),&mumps->id.CNTL(4),NULL));
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_5","CNTL(5): fixation for null pivots","None",mumps->id.CNTL(5),&mumps->id.CNTL(5),NULL));
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_7","CNTL(7): dropping parameter used during BLR","None",mumps->id.CNTL(7),&mumps->id.CNTL(7),NULL));

  PetscCall(PetscOptionsString("-mat_mumps_ooc_tmpdir", "out of core directory", "None", mumps->id.ooc_tmpdir, mumps->id.ooc_tmpdir, sizeof(mumps->id.ooc_tmpdir), NULL));

  PetscCall(PetscOptionsIntArray("-mat_mumps_view_info","request INFO local to each processor","",info,&ninfo,NULL));
  if (ninfo) {
    PetscCheck(ninfo <= 80,PETSC_COMM_SELF,PETSC_ERR_USER,"number of INFO %" PetscInt_FMT " must <= 80",ninfo);
    PetscCall(PetscMalloc1(ninfo,&mumps->info));
    mumps->ninfo = ninfo;
    for (i=0; i<ninfo; i++) {
      PetscCheck(info[i] >= 0 && info[i] <= 80,PETSC_COMM_SELF,PETSC_ERR_USER,"index of INFO %" PetscInt_FMT " must between 1 and 80",ninfo);
      else  mumps->info[i] = info[i];
    }
  }

  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode PetscInitializeMUMPS(Mat A,Mat_MUMPS *mumps)
{
  PetscInt       nthreads=0;

  PetscFunctionBegin;
  mumps->petsc_comm = PetscObjectComm((PetscObject)A);
  PetscCallMPI(MPI_Comm_size(mumps->petsc_comm,&mumps->petsc_size));
  PetscCallMPI(MPI_Comm_rank(mumps->petsc_comm,&mumps->myid));/* "if (!myid)" still works even if mumps_comm is different */

  PetscCall(PetscOptionsHasName(NULL,((PetscObject)A)->prefix,"-mat_mumps_use_omp_threads",&mumps->use_petsc_omp_support));
  if (mumps->use_petsc_omp_support) nthreads = -1; /* -1 will let PetscOmpCtrlCreate() guess a proper value when user did not supply one */
  PetscCall(PetscOptionsGetInt(NULL,((PetscObject)A)->prefix,"-mat_mumps_use_omp_threads",&nthreads,NULL));
  if (mumps->use_petsc_omp_support) {
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
    PetscCall(PetscOmpCtrlCreate(mumps->petsc_comm,nthreads,&mumps->omp_ctrl));
    PetscCall(PetscOmpCtrlGetOmpComms(mumps->omp_ctrl,&mumps->omp_comm,&mumps->mumps_comm,&mumps->is_omp_master));
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"the system does not have PETSc OpenMP support but you added the -%smat_mumps_use_omp_threads option. Configure PETSc with --with-openmp --download-hwloc (or --with-hwloc) to enable it, see more in MATSOLVERMUMPS manual",((PetscObject)A)->prefix?((PetscObject)A)->prefix:"");
#endif
  } else {
    mumps->omp_comm      = PETSC_COMM_SELF;
    mumps->mumps_comm    = mumps->petsc_comm;
    mumps->is_omp_master = PETSC_TRUE;
  }
  PetscCallMPI(MPI_Comm_size(mumps->omp_comm,&mumps->omp_comm_size));
  mumps->reqs = NULL;
  mumps->tag  = 0;

  /* It looks like MUMPS does not dup the input comm. Dup a new comm for MUMPS to avoid any tag mismatches. */
  if (mumps->mumps_comm != MPI_COMM_NULL) {
    PetscCall(PetscCommGetComm(PetscObjectComm((PetscObject)A),&mumps->mumps_comm));
  }

  mumps->id.comm_fortran = MPI_Comm_c2f(mumps->mumps_comm);
  mumps->id.job = JOB_INIT;
  mumps->id.par = 1;  /* host participates factorizaton and solve */
  mumps->id.sym = mumps->sym;

  PetscMUMPS_c(mumps);
  PetscCheck(mumps->id.INFOG(1) >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in PetscInitializeMUMPS: INFOG(1)=%d",mumps->id.INFOG(1));

  /* copy MUMPS default control values from master to slaves. Although slaves do not call MUMPS, they may access these values in code.
     For example, ICNTL(9) is initialized to 1 by MUMPS and slaves check ICNTL(9) in MatSolve_MUMPS.
   */
  PetscCallMPI(MPI_Bcast(mumps->id.icntl,40,MPI_INT,  0,mumps->omp_comm));
  PetscCallMPI(MPI_Bcast(mumps->id.cntl, 15,MPIU_REAL,0,mumps->omp_comm));

  mumps->scat_rhs = NULL;
  mumps->scat_sol = NULL;

  /* set PETSc-MUMPS default options - override MUMPS default */
  mumps->id.ICNTL(3) = 0;
  mumps->id.ICNTL(4) = 0;
  if (mumps->petsc_size == 1) {
    mumps->id.ICNTL(18) = 0;   /* centralized assembled matrix input */
    mumps->id.ICNTL(7)  = 7;   /* automatic choice of ordering done by the package */
  } else {
    mumps->id.ICNTL(18) = 3;   /* distributed assembled matrix input */
    mumps->id.ICNTL(21) = 1;   /* distributed solution */
  }

  /* schur */
  mumps->id.size_schur    = 0;
  mumps->id.listvar_schur = NULL;
  mumps->id.schur         = NULL;
  mumps->sizeredrhs       = 0;
  mumps->schur_sol        = NULL;
  mumps->schur_sizesol    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorSymbolic_MUMPS_ReportIfError(Mat F,Mat A,const MatFactorInfo *info,Mat_MUMPS *mumps)
{
  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0) {
    if (A->erroriffailure) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d",mumps->id.INFOG(1));
    } else {
      if (mumps->id.INFOG(1) == -6) {
        PetscCall(PetscInfo(F,"matrix is singular in structure, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_STRUCT_ZEROPIVOT;
      } else if (mumps->id.INFOG(1) == -5 || mumps->id.INFOG(1) == -7) {
        PetscCall(PetscInfo(F,"problem of workspace, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else if (mumps->id.INFOG(1) == -16 && mumps->id.INFOG(1) == 0) {
        PetscCall(PetscInfo(F,"Empty matrix\n"));
      } else {
        PetscCall(PetscInfo(F,"Error reported by MUMPS in analysis phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2)));
        F->factorerrortype = MAT_FACTOR_OTHER;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  if (mumps->matstruc == SAME_NONZERO_PATTERN) {
    /* F is assembled by a previous call of MatLUFactorSymbolic_AIJMUMPS() */
    PetscFunctionReturn(0);
  }

  /* Set MUMPS options from the options database */
  PetscCall(PetscSetMUMPSFromOptions(F,A));

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX,mumps));

  /* analysis phase */
  /*----------------*/
  mumps->id.job = JOB_FACTSYMBOLIC;
  mumps->id.n   = M;
  switch (mumps->id.ICNTL(18)) {
  case 0:  /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nnz = mumps->nnz;
      mumps->id.irn = mumps->irn;
      mumps->id.jcn = mumps->jcn;
      if (mumps->id.ICNTL(6)>1) mumps->id.a = (MumpsScalar*)mumps->val;
      if (r) {
        mumps->id.ICNTL(7) = 1;
        if (!mumps->myid) {
          const PetscInt *idx;
          PetscInt       i;

          PetscCall(PetscMalloc1(M,&mumps->id.perm_in));
          PetscCall(ISGetIndices(r,&idx));
          for (i=0; i<M; i++) PetscCall(PetscMUMPSIntCast(idx[i]+1,&(mumps->id.perm_in[i]))); /* perm_in[]: start from 1, not 0! */
          PetscCall(ISRestoreIndices(r,&idx));
        }
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    mumps->id.nnz_loc = mumps->nnz;
    mumps->id.irn_loc = mumps->irn;
    mumps->id.jcn_loc = mumps->jcn;
    if (mumps->id.ICNTL(6)>1) mumps->id.a_loc = (MumpsScalar*)mumps->val;
    if (mumps->ICNTL20 == 0) { /* Centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(MatCreateVecs(A,NULL,&b));
      PetscCall(VecScatterCreateToZero(b,&mumps->scat_rhs,&mumps->b_seq));
      PetscCall(VecDestroy(&b));
    }
    break;
  }
  PetscMUMPS_c(mumps);
  PetscCall(MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps));

  F->ops->lufactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve           = MatSolve_MUMPS;
  F->ops->solvetranspose  = MatSolveTranspose_MUMPS;
  F->ops->matsolve        = MatMatSolve_MUMPS;
  F->ops->mattransposesolve = MatMatTransposeSolve_MUMPS;

  mumps->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
PetscErrorCode MatLUFactorSymbolic_BAIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  if (mumps->matstruc == SAME_NONZERO_PATTERN) {
    /* F is assembled by a previous call of MatLUFactorSymbolic_AIJMUMPS() */
    PetscFunctionReturn(0);
  }

  /* Set MUMPS options from the options database */
  PetscCall(PetscSetMUMPSFromOptions(F,A));

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX,mumps));

  /* analysis phase */
  /*----------------*/
  mumps->id.job = JOB_FACTSYMBOLIC;
  mumps->id.n   = M;
  switch (mumps->id.ICNTL(18)) {
  case 0:  /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nnz = mumps->nnz;
      mumps->id.irn = mumps->irn;
      mumps->id.jcn = mumps->jcn;
      if (mumps->id.ICNTL(6)>1) {
        mumps->id.a = (MumpsScalar*)mumps->val;
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    mumps->id.nnz_loc = mumps->nnz;
    mumps->id.irn_loc = mumps->irn;
    mumps->id.jcn_loc = mumps->jcn;
    if (mumps->id.ICNTL(6)>1) {
      mumps->id.a_loc = (MumpsScalar*)mumps->val;
    }
    if (mumps->ICNTL20 == 0) { /* Centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(MatCreateVecs(A,NULL,&b));
      PetscCall(VecScatterCreateToZero(b,&mumps->scat_rhs,&mumps->b_seq));
      PetscCall(VecDestroy(&b));
    }
    break;
  }
  PetscMUMPS_c(mumps);
  PetscCall(MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps));

  F->ops->lufactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve           = MatSolve_MUMPS;
  F->ops->solvetranspose  = MatSolveTranspose_MUMPS;

  mumps->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation and factor info are ignored */
PetscErrorCode MatCholeskyFactorSymbolic_MUMPS(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  if (mumps->matstruc == SAME_NONZERO_PATTERN) {
    /* F is assembled by a previous call of MatLUFactorSymbolic_AIJMUMPS() */
    PetscFunctionReturn(0);
  }

  /* Set MUMPS options from the options database */
  PetscCall(PetscSetMUMPSFromOptions(F,A));

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX,mumps));

  /* analysis phase */
  /*----------------*/
  mumps->id.job = JOB_FACTSYMBOLIC;
  mumps->id.n   = M;
  switch (mumps->id.ICNTL(18)) {
  case 0:  /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nnz = mumps->nnz;
      mumps->id.irn = mumps->irn;
      mumps->id.jcn = mumps->jcn;
      if (mumps->id.ICNTL(6)>1) {
        mumps->id.a = (MumpsScalar*)mumps->val;
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    mumps->id.nnz_loc = mumps->nnz;
    mumps->id.irn_loc = mumps->irn;
    mumps->id.jcn_loc = mumps->jcn;
    if (mumps->id.ICNTL(6)>1) {
      mumps->id.a_loc = (MumpsScalar*)mumps->val;
    }
    if (mumps->ICNTL20 == 0) { /* Centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(MatCreateVecs(A,NULL,&b));
      PetscCall(VecScatterCreateToZero(b,&mumps->scat_rhs,&mumps->b_seq));
      PetscCall(VecDestroy(&b));
    }
    break;
  }
  PetscMUMPS_c(mumps);
  PetscCall(MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps));

  F->ops->choleskyfactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve                 = MatSolve_MUMPS;
  F->ops->solvetranspose        = MatSolve_MUMPS;
  F->ops->matsolve              = MatMatSolve_MUMPS;
  F->ops->mattransposesolve     = MatMatTransposeSolve_MUMPS;
#if defined(PETSC_USE_COMPLEX)
  F->ops->getinertia = NULL;
#else
  F->ops->getinertia = MatGetInertia_SBAIJMUMPS;
#endif

  mumps->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MUMPS(Mat A,PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_MUMPS         *mumps=(Mat_MUMPS*)A->data;

  PetscFunctionBegin;
  /* check if matrix is mumps type */
  if (A->ops->solve != MatSolve_MUMPS) PetscFunctionReturn(0);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"MUMPS run parameters:\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  SYM (matrix type):                   %d\n",mumps->id.sym));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  PAR (host participation):            %d\n",mumps->id.par));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(1) (output for error):         %d\n",mumps->id.ICNTL(1)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(2) (output of diagnostic msg): %d\n",mumps->id.ICNTL(2)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(3) (output for global info):   %d\n",mumps->id.ICNTL(3)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(4) (level of printing):        %d\n",mumps->id.ICNTL(4)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(5) (input mat struct):         %d\n",mumps->id.ICNTL(5)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(6) (matrix prescaling):        %d\n",mumps->id.ICNTL(6)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(7) (sequential matrix ordering):%d\n",mumps->id.ICNTL(7)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(8) (scaling strategy):        %d\n",mumps->id.ICNTL(8)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(10) (max num of refinements):  %d\n",mumps->id.ICNTL(10)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(11) (error analysis):          %d\n",mumps->id.ICNTL(11)));
      if (mumps->id.ICNTL(11)>0) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"    RINFOG(4) (inf norm of input mat):        %g\n",mumps->id.RINFOG(4)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    RINFOG(5) (inf norm of solution):         %g\n",mumps->id.RINFOG(5)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    RINFOG(6) (inf norm of residual):         %g\n",mumps->id.RINFOG(6)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    RINFOG(7),RINFOG(8) (backward error est): %g, %g\n",mumps->id.RINFOG(7),mumps->id.RINFOG(8)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    RINFOG(9) (error estimate):               %g \n",mumps->id.RINFOG(9)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n",mumps->id.RINFOG(10),mumps->id.RINFOG(11)));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(12) (efficiency control):                         %d\n",mumps->id.ICNTL(12)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(13) (sequential factorization of the root node):  %d\n",mumps->id.ICNTL(13)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(14) (percentage of estimated workspace increase): %d\n",mumps->id.ICNTL(14)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(15) (compression of the input matrix):            %d\n",mumps->id.ICNTL(15)));
      /* ICNTL(15-17) not used */
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(18) (input mat struct):                           %d\n",mumps->id.ICNTL(18)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(19) (Schur complement info):                      %d\n",mumps->id.ICNTL(19)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(20) (RHS sparse pattern):                         %d\n",mumps->id.ICNTL(20)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(21) (solution struct):                            %d\n",mumps->id.ICNTL(21)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(22) (in-core/out-of-core facility):               %d\n",mumps->id.ICNTL(22)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(23) (max size of memory can be allocated locally):%d\n",mumps->id.ICNTL(23)));

      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(24) (detection of null pivot rows):               %d\n",mumps->id.ICNTL(24)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(25) (computation of a null space basis):          %d\n",mumps->id.ICNTL(25)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(26) (Schur options for RHS or solution):          %d\n",mumps->id.ICNTL(26)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(27) (blocking size for multiple RHS):             %d\n",mumps->id.ICNTL(27)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(28) (use parallel or sequential ordering):        %d\n",mumps->id.ICNTL(28)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(29) (parallel ordering):                          %d\n",mumps->id.ICNTL(29)));

      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(30) (user-specified set of entries in inv(A)):    %d\n",mumps->id.ICNTL(30)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(31) (factors is discarded in the solve phase):    %d\n",mumps->id.ICNTL(31)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(33) (compute determinant):                        %d\n",mumps->id.ICNTL(33)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(35) (activate BLR based factorization):           %d\n",mumps->id.ICNTL(35)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(36) (choice of BLR factorization variant):        %d\n",mumps->id.ICNTL(36)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  ICNTL(38) (estimated compression rate of LU factors):   %d\n",mumps->id.ICNTL(38)));

      PetscCall(PetscViewerASCIIPrintf(viewer,"  CNTL(1) (relative pivoting threshold):      %g \n",mumps->id.CNTL(1)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  CNTL(2) (stopping criterion of refinement): %g \n",mumps->id.CNTL(2)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  CNTL(3) (absolute pivoting threshold):      %g \n",mumps->id.CNTL(3)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  CNTL(4) (value of static pivoting):         %g \n",mumps->id.CNTL(4)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  CNTL(5) (fixation for null pivots):         %g \n",mumps->id.CNTL(5)));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  CNTL(7) (dropping parameter for BLR):       %g \n",mumps->id.CNTL(7)));

      /* information local to each processor */
      PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFO(1) (local estimated flops for the elimination after analysis): \n"));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %g \n",mumps->myid,mumps->id.RINFO(1)));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFO(2) (local estimated flops for the assembly after factorization): \n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    [%d]  %g \n",mumps->myid,mumps->id.RINFO(2)));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFO(3) (local estimated flops for the elimination after factorization): \n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    [%d]  %g \n",mumps->myid,mumps->id.RINFO(3)));
      PetscCall(PetscViewerFlush(viewer));

      PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(15) (estimated size of (in MB) MUMPS internal data for running numerical factorization): \n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] %d\n",mumps->myid,mumps->id.INFO(15)));
      PetscCall(PetscViewerFlush(viewer));

      PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(16) (size of (in MB) MUMPS internal data used during numerical factorization): \n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d\n",mumps->myid,mumps->id.INFO(16)));
      PetscCall(PetscViewerFlush(viewer));

      PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(23) (num of pivots eliminated on this processor after factorization): \n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d\n",mumps->myid,mumps->id.INFO(23)));
      PetscCall(PetscViewerFlush(viewer));

      if (mumps->ninfo && mumps->ninfo <= 80) {
        PetscInt i;
        for (i=0; i<mumps->ninfo; i++) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(%" PetscInt_FMT "): \n",mumps->info[i]));
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d\n",mumps->myid,mumps->id.INFO(mumps->info[i])));
          PetscCall(PetscViewerFlush(viewer));
        }
      }
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));

      if (!mumps->myid) { /* information from the host */
        PetscCall(PetscViewerASCIIPrintf(viewer,"  RINFOG(1) (global estimated flops for the elimination after analysis): %g \n",mumps->id.RINFOG(1)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  RINFOG(2) (global estimated flops for the assembly after factorization): %g \n",mumps->id.RINFOG(2)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  RINFOG(3) (global estimated flops for the elimination after factorization): %g \n",mumps->id.RINFOG(3)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  (RINFOG(12) RINFOG(13))*2^INFOG(34) (determinant): (%g,%g)*(2^%d)\n",mumps->id.RINFOG(12),mumps->id.RINFOG(13),mumps->id.INFOG(34)));

        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d\n",mumps->id.INFOG(3)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d\n",mumps->id.INFOG(4)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(5) (estimated maximum front size in the complete tree): %d\n",mumps->id.INFOG(5)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(6) (number of nodes in the complete tree): %d\n",mumps->id.INFOG(6)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(7) (ordering option effectively used after analysis): %d\n",mumps->id.INFOG(7)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(8) (structural symmetry in percent of the permuted matrix after analysis): %d\n",mumps->id.INFOG(8)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(9) (total real/complex workspace to store the matrix factors after factorization): %d\n",mumps->id.INFOG(9)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(10) (total integer space store the matrix factors after factorization): %d\n",mumps->id.INFOG(10)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(11) (order of largest frontal matrix after factorization): %d\n",mumps->id.INFOG(11)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(12) (number of off-diagonal pivots): %d\n",mumps->id.INFOG(12)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(13) (number of delayed pivots after factorization): %d\n",mumps->id.INFOG(13)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(14) (number of memory compress after factorization): %d\n",mumps->id.INFOG(14)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(15) (number of steps of iterative refinement after solution): %d\n",mumps->id.INFOG(15)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(16) (estimated size (in MB) of all MUMPS internal data for factorization after analysis: value on the most memory consuming processor): %d\n",mumps->id.INFOG(16)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(17) (estimated size of all MUMPS internal data for factorization after analysis: sum over all processors): %d\n",mumps->id.INFOG(17)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(18) (size of all MUMPS internal data allocated during factorization: value on the most memory consuming processor): %d\n",mumps->id.INFOG(18)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(19) (size of all MUMPS internal data allocated during factorization: sum over all processors): %d\n",mumps->id.INFOG(19)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(20) (estimated number of entries in the factors): %d\n",mumps->id.INFOG(20)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(21) (size in MB of memory effectively used during factorization - value on the most memory consuming processor): %d\n",mumps->id.INFOG(21)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(22) (size in MB of memory effectively used during factorization - sum over all processors): %d\n",mumps->id.INFOG(22)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(23) (after analysis: value of ICNTL(6) effectively used): %d\n",mumps->id.INFOG(23)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(24) (after analysis: value of ICNTL(12) effectively used): %d\n",mumps->id.INFOG(24)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(25) (after factorization: number of pivots modified by static pivoting): %d\n",mumps->id.INFOG(25)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(28) (after factorization: number of null pivots encountered): %d\n",mumps->id.INFOG(28)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(29) (after factorization: effective number of entries in the factors (sum over all processors)): %d\n",mumps->id.INFOG(29)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(30, 31) (after solution: size in Mbytes of memory used during solution phase): %d, %d\n",mumps->id.INFOG(30),mumps->id.INFOG(31)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(32) (after analysis: type of analysis done): %d\n",mumps->id.INFOG(32)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(33) (value used for ICNTL(8)): %d\n",mumps->id.INFOG(33)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(34) (exponent of the determinant if determinant is requested): %d\n",mumps->id.INFOG(34)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(35) (after factorization: number of entries taking into account BLR factor compression - sum over all processors): %d\n",mumps->id.INFOG(35)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(36) (after analysis: estimated size of all MUMPS internal data for running BLR in-core - value on the most memory consuming processor): %d\n",mumps->id.INFOG(36)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(37) (after analysis: estimated size of all MUMPS internal data for running BLR in-core - sum over all processors): %d\n",mumps->id.INFOG(37)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(38) (after analysis: estimated size of all MUMPS internal data for running BLR out-of-core - value on the most memory consuming processor): %d\n",mumps->id.INFOG(38)));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  INFOG(39) (after analysis: estimated size of all MUMPS internal data for running BLR out-of-core - sum over all processors): %d\n",mumps->id.INFOG(39)));
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
  Mat_MUMPS         *mumps =(Mat_MUMPS*)F->data;
  const PetscScalar *arr;
  const PetscInt    *idxs;
  PetscInt          size,i;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is,&size));
  if (mumps->petsc_size > 1) {
    PetscBool ls,gs; /* gs is false if any rank other than root has non-empty IS */

    ls   = mumps->myid ? (size ? PETSC_FALSE : PETSC_TRUE) : PETSC_TRUE; /* always true on root; false on others if their size != 0 */
    PetscCallMPI(MPI_Allreduce(&ls,&gs,1,MPIU_BOOL,MPI_LAND,mumps->petsc_comm));
    PetscCheck(gs,PETSC_COMM_SELF,PETSC_ERR_SUP,"MUMPS distributed parallel Schur complements not yet supported from PETSc");
  }

  /* Schur complement matrix */
  PetscCall(MatDestroy(&F->schur));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,size,size,NULL,&F->schur));
  PetscCall(MatDenseGetArrayRead(F->schur,&arr));
  mumps->id.schur      = (MumpsScalar*)arr;
  mumps->id.size_schur = size;
  mumps->id.schur_lld  = size;
  PetscCall(MatDenseRestoreArrayRead(F->schur,&arr));
  if (mumps->sym == 1) {
    PetscCall(MatSetOption(F->schur,MAT_SPD,PETSC_TRUE));
  }

  /* MUMPS expects Fortran style indices */
  PetscCall(PetscFree(mumps->id.listvar_schur));
  PetscCall(PetscMalloc1(size,&mumps->id.listvar_schur));
  PetscCall(ISGetIndices(is,&idxs));
  for (i=0; i<size; i++) PetscCall(PetscMUMPSIntCast(idxs[i]+1,&(mumps->id.listvar_schur[i])));
  PetscCall(ISRestoreIndices(is,&idxs));
  if (mumps->petsc_size > 1) {
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

  PetscFunctionBegin;
  PetscCheck(mumps->id.ICNTL(19),PetscObjectComm((PetscObject)F),PETSC_ERR_ORDER,"Schur complement mode not selected! You should call MatFactorSetSchurIS to enable it");
  PetscCall(MatCreate(PETSC_COMM_SELF,&St));
  PetscCall(MatSetSizes(St,PETSC_DECIDE,PETSC_DECIDE,mumps->id.size_schur,mumps->id.size_schur));
  PetscCall(MatSetType(St,MATDENSE));
  PetscCall(MatSetUp(St));
  PetscCall(MatDenseGetArray(St,&array));
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
      PetscCall(PetscArraycpy(array,mumps->id.schur,mumps->id.size_schur*mumps->id.size_schur));
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
      PetscCall(PetscArraycpy(array,mumps->id.schur,mumps->id.size_schur*mumps->id.size_schur));
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
  PetscCall(MatDenseRestoreArray(St,&array));
  *S   = St;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
PetscErrorCode MatMumpsSetIcntl_MUMPS(Mat F,PetscInt icntl,PetscInt ival)
{
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  PetscCall(PetscMUMPSIntCast(ival,&mumps->id.ICNTL(icntl)));
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
.   -mat_mumps_icntl_<icntl> <ival> - change the option numbered icntl to ival

   Level: beginner

   References:
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsSetIcntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidLogicalCollectiveInt(F,ival,3);
  PetscTryMethod(F,"MatMumpsSetIcntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));
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
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetIcntl(Mat F,PetscInt icntl,PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidIntPointer(ival,3);
  PetscUseMethod(F,"MatMumpsGetIcntl_C",(Mat,PetscInt,PetscInt*),(F,icntl,ival));
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
.   -mat_mumps_cntl_<icntl> <val>  - change the option numbered icntl to ival

   Level: beginner

   References:
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsSetCntl(Mat F,PetscInt icntl,PetscReal val)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidLogicalCollectiveReal(F,val,3);
  PetscTryMethod(F,"MatMumpsSetCntl_C",(Mat,PetscInt,PetscReal),(F,icntl,val));
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
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetCntl(Mat F,PetscInt icntl,PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidRealPointer(val,3);
  PetscUseMethod(F,"MatMumpsGetCntl_C",(Mat,PetscInt,PetscReal*),(F,icntl,val));
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

PetscErrorCode MatMumpsGetInverse_MUMPS(Mat F,Mat spRHS)
{
  Mat            Bt = NULL,Btseq = NULL;
  PetscBool      flg;
  Mat_MUMPS      *mumps =(Mat_MUMPS*)F->data;
  PetscScalar    *aa;
  PetscInt       spnr,*ia,*ja,M,nrhs;

  PetscFunctionBegin;
  PetscValidPointer(spRHS,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)spRHS,MATTRANSPOSEMAT,&flg));
  if (flg) {
    PetscCall(MatTransposeGetMat(spRHS,&Bt));
  } else SETERRQ(PetscObjectComm((PetscObject)spRHS),PETSC_ERR_ARG_WRONG,"Matrix spRHS must be type MATTRANSPOSEMAT matrix");

  PetscCall(MatMumpsSetIcntl(F,30,1));

  if (mumps->petsc_size > 1) {
    Mat_MPIAIJ *b = (Mat_MPIAIJ*)Bt->data;
    Btseq = b->A;
  } else {
    Btseq = Bt;
  }

  PetscCall(MatGetSize(spRHS,&M,&nrhs));
  mumps->id.nrhs = nrhs;
  mumps->id.lrhs = M;
  mumps->id.rhs  = NULL;

  if (!mumps->myid) {
    PetscCall(MatSeqAIJGetArray(Btseq,&aa));
    PetscCall(MatGetRowIJ(Btseq,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
    PetscCall(PetscMUMPSIntCSRCast(mumps,spnr,ia,ja,&mumps->id.irhs_ptr,&mumps->id.irhs_sparse,&mumps->id.nz_rhs));
    mumps->id.rhs_sparse  = (MumpsScalar*)aa;
  } else {
    mumps->id.irhs_ptr    = NULL;
    mumps->id.irhs_sparse = NULL;
    mumps->id.nz_rhs      = 0;
    mumps->id.rhs_sparse  = NULL;
  }
  mumps->id.ICNTL(20)   = 1; /* rhs is sparse */
  mumps->id.ICNTL(21)   = 0; /* solution is in assembled centralized format */

  /* solve phase */
  /*-------------*/
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps);
  if (mumps->id.INFOG(1) < 0)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d INFO(2)=%d",mumps->id.INFOG(1),mumps->id.INFO(2));

  if (!mumps->myid) {
    PetscCall(MatSeqAIJRestoreArray(Btseq,&aa));
    PetscCall(MatRestoreRowIJ(Btseq,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
  }
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetInverse - Get user-specified set of entries in inverse of A

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  spRHS - sequential sparse matrix in MATTRANSPOSEMAT format holding specified indices in processor[0]

  Output Parameter:
. spRHS - requested entries of inverse of A

   Level: beginner

   References:
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatCreateTranspose()
@*/
PetscErrorCode MatMumpsGetInverse(Mat F,Mat spRHS)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscUseMethod(F,"MatMumpsGetInverse_C",(Mat,Mat),(F,spRHS));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetInverseTranspose_MUMPS(Mat F,Mat spRHST)
{
  Mat            spRHS;

  PetscFunctionBegin;
  PetscCall(MatCreateTranspose(spRHST,&spRHS));
  PetscCall(MatMumpsGetInverse_MUMPS(F,spRHS));
  PetscCall(MatDestroy(&spRHS));
  PetscFunctionReturn(0);
}

/*@
  MatMumpsGetInverseTranspose - Get user-specified set of entries in inverse of matrix A^T

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix of A obtained by calling MatGetFactor() from PETSc-MUMPS interface
-  spRHST - sequential sparse matrix in MATAIJ format holding specified indices of A^T in processor[0]

  Output Parameter:
. spRHST - requested entries of inverse of A^T

   Level: beginner

   References:
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatCreateTranspose(), MatMumpsGetInverse()
@*/
PetscErrorCode MatMumpsGetInverseTranspose(Mat F,Mat spRHST)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)spRHST,&flg,MATSEQAIJ,MATMPIAIJ,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)spRHST),PETSC_ERR_ARG_WRONG,"Matrix spRHST must be MATAIJ matrix");

  PetscUseMethod(F,"MatMumpsGetInverseTranspose_C",(Mat,Mat),(F,spRHST));
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
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetInfo(Mat F,PetscInt icntl,PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidIntPointer(ival,3);
  PetscUseMethod(F,"MatMumpsGetInfo_C",(Mat,PetscInt,PetscInt*),(F,icntl,ival));
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
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetInfog(Mat F,PetscInt icntl,PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidIntPointer(ival,3);
  PetscUseMethod(F,"MatMumpsGetInfog_C",(Mat,PetscInt,PetscInt*),(F,icntl,ival));
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
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfog()
@*/
PetscErrorCode MatMumpsGetRinfo(Mat F,PetscInt icntl,PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidRealPointer(val,3);
  PetscUseMethod(F,"MatMumpsGetRinfo_C",(Mat,PetscInt,PetscReal*),(F,icntl,val));
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
.  * - MUMPS Users' Guide

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo()
@*/
PetscErrorCode MatMumpsGetRinfog(Mat F,PetscInt icntl,PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscCheck(F->factortype,PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscValidRealPointer(val,3);
  PetscUseMethod(F,"MatMumpsGetRinfog_C",(Mat,PetscInt,PetscReal*),(F,icntl,val));
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERMUMPS -  A matrix type providing direct solvers (LU and Cholesky) for
  distributed and sequential matrices via the external package MUMPS.

  Works with MATAIJ and MATSBAIJ matrices

  Use ./configure --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch to have PETSc installed with MUMPS

  Use ./configure --with-openmp --download-hwloc (or --with-hwloc) to enable running MUMPS in MPI+OpenMP hybrid mode and non-MUMPS in flat-MPI mode. See details below.

  Use -pc_type cholesky or lu -pc_factor_mat_solver_type mumps to use this direct solver

  Options Database Keys:
+  -mat_mumps_icntl_1 - ICNTL(1): output stream for error messages
.  -mat_mumps_icntl_2 - ICNTL(2): output stream for diagnostic printing, statistics, and warning
.  -mat_mumps_icntl_3 -  ICNTL(3): output stream for global information, collected on the host
.  -mat_mumps_icntl_4 -  ICNTL(4): level of printing (0 to 4)
.  -mat_mumps_icntl_6 - ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7)
.  -mat_mumps_icntl_7 - ICNTL(7): computes a symmetric permutation in sequential analysis, 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto
                        Use -pc_factor_mat_ordering_type <type> to have PETSc perform the ordering (sequential only)
.  -mat_mumps_icntl_8  - ICNTL(8): scaling strategy (-2 to 8 or 77)
.  -mat_mumps_icntl_10  - ICNTL(10): max num of refinements
.  -mat_mumps_icntl_11  - ICNTL(11): statistics related to an error analysis (via -ksp_view)
.  -mat_mumps_icntl_12  - ICNTL(12): an ordering strategy for symmetric matrices (0 to 3)
.  -mat_mumps_icntl_13  - ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting
.  -mat_mumps_icntl_14  - ICNTL(14): percentage increase in the estimated working space
.  -mat_mumps_icntl_15  - ICNTL(15): compression of the input matrix resulting from a block format
.  -mat_mumps_icntl_19  - ICNTL(19): computes the Schur complement
.  -mat_mumps_icntl_20  - ICNTL(20): give MUMPS centralized (0) or distributed (10) dense RHS
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
.  -mat_mumps_icntl_35 - ICNTL(35): level of activation of BLR (Block Low-Rank) feature
.  -mat_mumps_icntl_36 - ICNTL(36): controls the choice of BLR factorization variant
.  -mat_mumps_icntl_38 - ICNTL(38): sets the estimated compression rate of LU factors with BLR
.  -mat_mumps_cntl_1  - CNTL(1): relative pivoting threshold
.  -mat_mumps_cntl_2  -  CNTL(2): stopping criterion of refinement
.  -mat_mumps_cntl_3 - CNTL(3): absolute pivoting threshold
.  -mat_mumps_cntl_4 - CNTL(4): value for static pivoting
.  -mat_mumps_cntl_5 - CNTL(5): fixation for null pivots
.  -mat_mumps_cntl_7 - CNTL(7): precision of the dropping parameter used during BLR factorization
-  -mat_mumps_use_omp_threads [m] - run MUMPS in MPI+OpenMP hybrid mode as if omp_set_num_threads(m) is called before calling MUMPS.
                                   Default might be the number of cores per CPU package (socket) as reported by hwloc and suggested by the MUMPS manual.

  Level: beginner

    Notes:
    MUMPS Cholesky does not handle (complex) Hermitian matrices http://mumps.enseeiht.fr/doc/userguide_5.2.1.pdf so using it will error if the matrix is Hermitian.

    When a MUMPS factorization fails inside a KSP solve, for example with a KSP_DIVERGED_PC_FAILED, one can find the MUMPS information about the failure by calling
$          KSPGetPC(ksp,&pc);
$          PCFactorGetMatrix(pc,&mat);
$          MatMumpsGetInfo(mat,....);
$          MatMumpsGetInfog(mat,....); etc.
           Or you can run with -ksp_error_if_not_converged and the program will be stopped and the information printed in the error message.

  Using MUMPS with 64-bit integers
    MUMPS provides 64-bit integer support in two build modes:
      full 64-bit: here MUMPS is built with C preprocessing flag -DINTSIZE64 and Fortran compiler option -i8, -fdefault-integer-8 or equivalent, and
      requires all dependent libraries MPI, ScaLAPACK, LAPACK and BLAS built the same way with 64-bit integers (for example ILP64 Intel MKL and MPI).

      selective 64-bit: with the default MUMPS build, 64-bit integers have been introduced where needed. In compressed sparse row (CSR) storage of matrices,
      MUMPS stores column indices in 32-bit, but row offsets in 64-bit, so you can have a huge number of non-zeros, but must have less than 2^31 rows and
      columns. This can lead to significant memory and performance gains with respect to a full 64-bit integer MUMPS version. This requires a regular (32-bit
      integer) build of all dependent libraries MPI, ScaLAPACK, LAPACK and BLAS.

    With --download-mumps=1, PETSc always build MUMPS in selective 64-bit mode, which can be used by both --with-64-bit-indices=0/1 variants of PETSc.

  Two modes to run MUMPS/PETSc with OpenMP
$     Set OMP_NUM_THREADS and run with fewer MPI ranks than cores. For example, if you want to have 16 OpenMP
$     threads per rank, then you may use "export OMP_NUM_THREADS=16 && mpirun -n 4 ./test".

$     -mat_mumps_use_omp_threads [m] and run your code with as many MPI ranks as the number of cores. For example,
$     if a compute node has 32 cores and you run on two nodes, you may use "mpirun -n 64 ./test -mat_mumps_use_omp_threads 16"

   To run MUMPS in MPI+OpenMP hybrid mode (i.e., enable multithreading in MUMPS), but still run the non-MUMPS part
   (i.e., PETSc part) of your code in the so-called flat-MPI (aka pure-MPI) mode, you need to configure PETSc with --with-openmp --download-hwloc
   (or --with-hwloc), and have an MPI that supports MPI-3.0's process shared memory (which is usually available). Since MUMPS calls BLAS
   libraries, to really get performance, you should have multithreaded BLAS libraries such as Intel MKL, AMD ACML, Cray libSci or OpenBLAS
   (PETSc will automatically try to utilized a threaded BLAS if --with-openmp is provided).

   If you run your code through a job submission system, there are caveats in MPI rank mapping. We use MPI_Comm_split_type() to obtain MPI
   processes on each compute node. Listing the processes in rank ascending order, we split processes on a node into consecutive groups of
   size m and create a communicator called omp_comm for each group. Rank 0 in an omp_comm is called the master rank, and others in the omp_comm
   are called slave ranks (or slaves). Only master ranks are seen to MUMPS and slaves are not. We will free CPUs assigned to slaves (might be set
   by CPU binding policies in job scripts) and make the CPUs available to the master so that OMP threads spawned by MUMPS can run on the CPUs.
   In a multi-socket compute node, MPI rank mapping is an issue. Still use the above example and suppose your compute node has two sockets,
   if you interleave MPI ranks on the two sockets, in other words, even ranks are placed on socket 0, and odd ranks are on socket 1, and bind
   MPI ranks to cores, then with -mat_mumps_use_omp_threads 16, a master rank (and threads it spawns) will use half cores in socket 0, and half
   cores in socket 1, that definitely hurts locality. On the other hand, if you map MPI ranks consecutively on the two sockets, then the
   problem will not happen. Therefore, when you use -mat_mumps_use_omp_threads, you need to keep an eye on your MPI rank mapping and CPU binding.
   For example, with the Slurm job scheduler, one can use srun --cpu-bind=verbose -m block:block to map consecutive MPI ranks to sockets and
   examine the mapping result.

   PETSc does not control thread binding in MUMPS. So to get best performance, one still has to set OMP_PROC_BIND and OMP_PLACES in job scripts,
   for example, export OMP_PLACES=threads and export OMP_PROC_BIND=spread. One does not need to export OMP_NUM_THREADS=m in job scripts as PETSc
   calls omp_set_num_threads(m) internally before calling MUMPS.

   References:
+  * - Heroux, Michael A., R. Brightwell, and Michael M. Wolf. "Bi-modal MPI and MPI+ threads computing on scalable multicore systems." IJHPCA (Submitted) (2011).
-  * - Gutierrez, Samuel K., et al. "Accommodating Thread-Level Heterogeneity in Coupled Parallel Applications." Parallel and Distributed Processing Symposium (IPDPS), 2017 IEEE International. IEEE, 2017.

.seealso: PCFactorSetMatSolverType(), MatSolverType, MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog(), KSPGetPC(), PCFactorGetMatrix()

M*/

static PetscErrorCode MatFactorGetSolverType_mumps(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMUMPS;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq and MPI AIJ matrices */
static PetscErrorCode MatGetFactor_aij_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqAIJ;
  PetscMPIInt    size;

  PetscFunctionBegin;
 #if defined(PETSC_USE_COMPLEX)
  PetscCheck(!A->hermitian || A->symmetric || ftype != MAT_FACTOR_CHOLESKY,PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian CHOLESKY Factor is not supported");
 #endif
  /* Create the factorization matrix */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("mumps",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNewLog(B,&mumps));

  B->ops->view    = MatView_MUMPS;
  B->ops->getinfo = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverse_C",MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverseTranspose_C",MatMumpsGetInverseTranspose_MUMPS));

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpiaij;
    PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]));
    mumps->sym = 0;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
    B->factortype                  = MAT_FACTOR_CHOLESKY;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqsbaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpisbaij;
    PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]));
#if defined(PETSC_USE_COMPLEX)
    mumps->sym = 2;
#else
    if (A->spd_set && A->spd) mumps->sym = 1;
    else                      mumps->sym = 2;
#endif
  }

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

  PetscCall(PetscInitializeMUMPS(A,mumps));

  *F = B;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq and MPI SBAIJ matrices */
static PetscErrorCode MatGetFactor_sbaij_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqSBAIJ;
  PetscMPIInt    size;

  PetscFunctionBegin;
 #if defined(PETSC_USE_COMPLEX)
  PetscCheck(!A->hermitian || A->symmetric,PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian CHOLESKY Factor is not supported");
 #endif
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("mumps",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNewLog(B,&mumps));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ));
  if (isSeqSBAIJ) {
    mumps->ConvertToTriples = MatConvertToTriples_seqsbaij_seqsbaij;
  } else {
    mumps->ConvertToTriples = MatConvertToTriples_mpisbaij_mpisbaij;
  }

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
  B->ops->view                   = MatView_MUMPS;
  B->ops->getinfo                = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverse_C",MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverseTranspose_C",MatMumpsGetInverseTranspose_MUMPS));

  B->factortype = MAT_FACTOR_CHOLESKY;
#if defined(PETSC_USE_COMPLEX)
  mumps->sym = 2;
#else
  if (A->spd_set && A->spd) mumps->sym = 1;
  else                      mumps->sym = 2;
#endif

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]));
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

  PetscCall(PetscInitializeMUMPS(A,mumps));

  *F = B;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_baij_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqBAIJ;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQBAIJ,&isSeqBAIJ));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("mumps",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNewLog(B,&mumps));
  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_BAIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqBAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqbaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpibaij_mpiaij;
    mumps->sym = 0;
    PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc BAIJ matrices with MUMPS Cholesky, use SBAIJ or AIJ matrix instead");

  B->ops->view        = MatView_MUMPS;
  B->ops->getinfo     = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverse_C",MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverseTranspose_C",MatMumpsGetInverseTranspose_MUMPS));

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

  PetscCall(PetscInitializeMUMPS(A,mumps));

  *F = B;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq and MPI SELL matrices */
static PetscErrorCode MatGetFactor_sell_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqSELL;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQSELL,&isSeqSELL));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("mumps",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNewLog(B,&mumps));

  B->ops->view        = MatView_MUMPS;
  B->ops->getinfo     = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorSetSchurIS_C",MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorCreateSchurComplement_C",MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetIcntl_C",MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetIcntl_C",MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsSetCntl_C",MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetCntl_C",MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfo_C",MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInfog_C",MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfo_C",MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetRinfog_C",MatMumpsGetRinfog_MUMPS));

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqSELL) mumps->ConvertToTriples = MatConvertToTriples_seqsell_seqaij;
    else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"To be implemented");
    mumps->sym = 0;
    PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]));
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"To be implemented");

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization  */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

  PetscCall(PetscInitializeMUMPS(A,mumps));

  *F = B;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MUMPS(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIBAIJ,MAT_FACTOR_LU,MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATMPISBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQBAIJ,MAT_FACTOR_LU,MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQSBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQSELL,MAT_FACTOR_LU,MatGetFactor_sell_mumps));
  PetscFunctionReturn(0);
}
