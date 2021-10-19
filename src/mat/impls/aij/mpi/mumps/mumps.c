
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
PETSC_STATIC_INLINE PetscErrorCode PetscMUMPSIntCast(PetscInt a,PetscMUMPSInt *b)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_64BIT_INDICES) && PetscUnlikelyDebug(a > PETSC_MUMPS_INT_MAX || a < PETSC_MUMPS_INT_MIN)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscInt too long for PetscMUMPSInt");
  *b = (PetscMUMPSInt)(a);
  PetscFunctionReturn(0);
}

/* Put these utility routines here since they are only used in this file */
PETSC_STATIC_INLINE PetscErrorCode PetscOptionsMUMPSInt_Private(PetscOptionItems *PetscOptionsObject,const char opt[],const char text[],const char man[],PetscMUMPSInt currentvalue,PetscMUMPSInt *value,PetscBool *set,PetscMUMPSInt lb,PetscMUMPSInt ub)
{
  PetscErrorCode ierr;
  PetscInt       myval;
  PetscBool      myset;
  PetscFunctionBegin;
  /* PetscInt's size should be always >= PetscMUMPSInt's. It is safe to call PetscOptionsInt_Private to read a PetscMUMPSInt */
  ierr = PetscOptionsInt_Private(PetscOptionsObject,opt,text,man,(PetscInt)currentvalue,&myval,&myset,lb,ub);CHKERRQ(ierr);
  if (myset) {ierr = PetscMUMPSIntCast(myval,value);CHKERRQ(ierr);}
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
        ierr = PetscOmpCtrlOmpRegionOnMasterBegin(mumps->omp_ctrl);CHKERRQ(ierr); \
        MUMPS_c(&mumps->id); \
        ierr = PetscOmpCtrlOmpRegionOnMasterEnd(mumps->omp_ctrl);CHKERRQ(ierr); \
      } \
      ierr = PetscOmpCtrlBarrier(mumps->omp_ctrl);CHKERRQ(ierr); \
      /* Global info is same on all processes so we Bcast it within omp_comm. Local info is specific      \
         to processes, so we only Bcast info[1], an error code and leave others (since they do not have   \
         an easy translation between omp_comm and petsc_comm). See MUMPS-5.1.2 manual p82.                   \
         omp_comm is a small shared memory communicator, hence doing multiple Bcast as shown below is OK. \
      */ \
      ierr = MPI_Bcast(mumps->id.infog, 40,MPIU_MUMPSINT, 0,mumps->omp_comm);CHKERRMPI(ierr);\
      ierr = MPI_Bcast(mumps->id.rinfog,20,MPIU_REAL,     0,mumps->omp_comm);CHKERRMPI(ierr);\
      ierr = MPI_Bcast(mumps->id.info,  1, MPIU_MUMPSINT, 0,mumps->omp_comm);CHKERRMPI(ierr);\
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
  PetscErrorCode ierr;
  PetscInt       nnz=ia[nrow]-1; /* mumps uses 1-based indices. Uses PetscInt instead of PetscInt64 since mumps only uses PetscMUMPSInt for rhs */

  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  {
    PetscInt i;
    if (nrow+1 > mumps->cur_ilen) { /* realloc ia_alloc/ja_alloc to fit ia/ja */
      ierr = PetscFree(mumps->ia_alloc);CHKERRQ(ierr);
      ierr = PetscMalloc1(nrow+1,&mumps->ia_alloc);CHKERRQ(ierr);
      mumps->cur_ilen = nrow+1;
    }
    if (nnz > mumps->cur_jlen) {
      ierr = PetscFree(mumps->ja_alloc);CHKERRQ(ierr);
      ierr = PetscMalloc1(nnz,&mumps->ja_alloc);CHKERRQ(ierr);
      mumps->cur_jlen = nnz;
    }
    for (i=0; i<nrow+1; i++) {ierr = PetscMUMPSIntCast(ia[i],&(mumps->ia_alloc[i]));CHKERRQ(ierr);}
    for (i=0; i<nnz; i++)    {ierr = PetscMUMPSIntCast(ja[i],&(mumps->ja_alloc[i]));CHKERRQ(ierr);}
    *ia_mumps = mumps->ia_alloc;
    *ja_mumps = mumps->ja_alloc;
  }
#else
  *ia_mumps = ia;
  *ja_mumps = ja;
#endif
  ierr = PetscMUMPSIntCast(nnz,nnz_mumps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMumpsResetSchur_Private(Mat_MUMPS* mumps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mumps->id.listvar_schur);CHKERRQ(ierr);
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
  ierr = MatSetType(B,((PetscObject)S)->type_name);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  ierr = MatBindToCPU(B,S->boundtocpu);CHKERRQ(ierr);
#endif
  switch (schurstatus) {
  case MAT_FACTOR_SCHUR_FACTORED:
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,mumps->id.size_schur,mumps->id.nrhs,(PetscScalar*)mumps->id.redrhs,&X);CHKERRQ(ierr);
    ierr = MatSetType(X,((PetscObject)S)->type_name);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    ierr = MatBindToCPU(X,S->boundtocpu);CHKERRQ(ierr);
#endif
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
    ierr = MatSetType(X,((PetscObject)S)->type_name);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    ierr = MatBindToCPU(X,S->boundtocpu);CHKERRQ(ierr);
#endif
    ierr = MatProductCreateWithMat(S,B,NULL,X);CHKERRQ(ierr);
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      ierr = MatProductSetType(X,MATPRODUCT_AtB);CHKERRQ(ierr);
    } else {
      ierr = MatProductSetType(X,MATPRODUCT_AB);CHKERRQ(ierr);
    }
    ierr = MatProductSetFromOptions(X);CHKERRQ(ierr);
    ierr = MatProductSymbolic(X);CHKERRQ(ierr);
    ierr = MatProductNumeric(X);CHKERRQ(ierr);

    ierr = MatCopy(X,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %D",F->schur_status);
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
    PetscMUMPS_c(mumps);
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
  PetscErrorCode    ierr;
  PetscMUMPSInt     *row,*col;
  Mat_SeqAIJ        *aa=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  ierr       = MatSeqAIJGetArrayRead(A,&av);CHKERRQ(ierr);
  mumps->val = (PetscScalar*)av;
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz;
    ai   = aa->i;
    aj   = aa->j;
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    for (i=k=0; i<M; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for (j=0; j<rnz; j++) {
        ierr = PetscMUMPSIntCast(i+shift,&row[k]);CHKERRQ(ierr);
        ierr = PetscMUMPSIntCast(ajj[j] + shift,&col[k]);CHKERRQ(ierr);
        k++;
      }
    }
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
  }
  ierr = MatSeqAIJRestoreArrayRead(A,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_seqsell_seqaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  PetscErrorCode ierr;
  PetscInt64     nz,i,j,k,r;
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscMUMPSInt  *row,*col;

  PetscFunctionBegin;
  mumps->val = a->val;
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = a->sliidx[a->totalslices];
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    for (i=k=0; i<a->totalslices; i++) {
      for (j=a->sliidx[i],r=0; j<a->sliidx[i+1]; j++,r=((r+1)&0x07)) {
        ierr = PetscMUMPSIntCast(8*i+r+shift,&row[k++]);CHKERRQ(ierr);
      }
    }
    for (i=0;i<nz;i++) {ierr = PetscMUMPSIntCast(a->colidx[i]+shift,&col[i]);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;
  PetscMUMPSInt  *row,*col;

  PetscFunctionBegin;
  ierr       = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  M          = A->rmap->N/bs;
  mumps->val = aa->a;
  if (reuse == MAT_INITIAL_MATRIX) {
    ai   = aa->i; aj = aa->j;
    nz   = bs2*aa->nz;
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    for (i=0; i<M; i++) {
      ajj = aj + ai[i];
      rnz = ai[i+1] - ai[i];
      for (k=0; k<rnz; k++) {
        for (j=0; j<bs; j++) {
          for (m=0; m<bs; m++) {
            ierr = PetscMUMPSIntCast(i*bs + m + shift,&row[idx]);CHKERRQ(ierr);
            ierr = PetscMUMPSIntCast(bs*ajj[k] + j + shift,&col[idx]);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;
  PetscMUMPSInt   *row,*col;
  PetscScalar     *val;
  Mat_SeqSBAIJ    *aa=(Mat_SeqSBAIJ*)A->data;
  const PetscInt  bs2=aa->bs2,mbs=aa->mbs;
#if defined(PETSC_USE_COMPLEX)
  PetscBool       hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = MatGetOption(A,MAT_HERMITIAN,&hermitian);CHKERRQ(ierr);
  if (hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  ai   = aa->i;
  aj   = aa->j;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = aa->nz;
    ierr = PetscMalloc2(bs2*nz,&row,bs2*nz,&col);CHKERRQ(ierr);
    if (bs>1) {
      ierr       = PetscMalloc1(bs2*nz,&mumps->val_alloc);CHKERRQ(ierr);
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
                ierr = PetscMUMPSIntCast(i*bs + m + shift,&row[nz]);CHKERRQ(ierr);
                ierr = PetscMUMPSIntCast(ajj[j]*bs + k + shift,&col[nz]);CHKERRQ(ierr);
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
        ierr = PetscMUMPSIntCast(i+shift,&row[nz]);CHKERRQ(ierr);
        ierr = PetscMUMPSIntCast(ajj[j] + shift,&col[nz]);CHKERRQ(ierr);
        nz++;
      }
    }
    if (nz != aa->nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Different numbers of nonzeros %D != %D",nz,aa->nz);
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
  PetscErrorCode    ierr;
  PetscMUMPSInt     *row,*col;
  Mat_SeqAIJ        *aa=(Mat_SeqAIJ*)A->data;
  PetscBool         missing;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = MatGetOption(A,MAT_HERMITIAN,&hermitian);CHKERRQ(ierr);
  if (hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  ierr  = MatSeqAIJGetArrayRead(A,&av);CHKERRQ(ierr);
  ai    = aa->i; aj = aa->j;
  adiag = aa->diag;
  ierr  = MatMissingDiagonal_SeqAIJ(A,&missing,NULL);CHKERRQ(ierr);
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
    ierr       = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    ierr       = PetscMalloc1(nz,&val);CHKERRQ(ierr);
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
            ierr = PetscMUMPSIntCast(i+shift,&row[nz]);CHKERRQ(ierr);
            ierr = PetscMUMPSIntCast(aj[j]+shift,&col[nz]);CHKERRQ(ierr);
            val[nz] = av[j];
            nz++;
          }
        } else {
          rnz = ai[i+1] - adiag[i];
          ajj = aj + adiag[i];
          v1  = av + adiag[i];
          for (j=0; j<rnz; j++) {
            ierr = PetscMUMPSIntCast(i+shift,&row[nz]);CHKERRQ(ierr);
            ierr = PetscMUMPSIntCast(ajj[j] + shift,&col[nz]);CHKERRQ(ierr);
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
          ierr = PetscMUMPSIntCast(i+shift,&row[nz]);CHKERRQ(ierr);
          ierr = PetscMUMPSIntCast(ajj[j] + shift,&col[nz]);CHKERRQ(ierr);
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
  ierr = MatSeqAIJRestoreArrayRead(A,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertToTriples_mpisbaij_mpisbaij(Mat A,PetscInt shift,MatReuse reuse,Mat_MUMPS *mumps)
{
  PetscErrorCode    ierr;
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
  ierr = MatGetOption(A,MAT_HERMITIAN,&hermitian);CHKERRQ(ierr);
  if (hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
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
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz,&val);CHKERRQ(ierr);
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
                ierr = PetscMUMPSIntCast(irow + m + shift,&row[jj]);CHKERRQ(ierr);
                ierr = PetscMUMPSIntCast(rstart + ajj[j]*bs + k + shift,&col[jj]);CHKERRQ(ierr);
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
              ierr = PetscMUMPSIntCast(irow + m + shift,&row[jj]);CHKERRQ(ierr);
              ierr = PetscMUMPSIntCast(garray[bjj[j]]*bs + k + shift,&col[jj]);CHKERRQ(ierr);
            }
            val[jj++] = v2[j*bs2 + m + k*bs];
          }
        }
      }
    } else {
      /* A-part */
      for (j=0; j<countA; j++) {
        if (reuse == MAT_INITIAL_MATRIX) {
          ierr = PetscMUMPSIntCast(irow + shift,&row[jj]);CHKERRQ(ierr);
          ierr = PetscMUMPSIntCast(rstart + ajj[j] + shift,&col[jj]);CHKERRQ(ierr);
        }
        val[jj++] = v1[j];
      }

      /* B-part */
      for (j=0; j < countB; j++) {
        if (reuse == MAT_INITIAL_MATRIX) {
          ierr = PetscMUMPSIntCast(irow + shift,&row[jj]);CHKERRQ(ierr);
          ierr = PetscMUMPSIntCast(garray[bjj[j]] + shift,&col[jj]);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt64        rstart,nz,i,j,jj,irow,countA,countB;
  PetscMUMPSInt     *row,*col;
  const PetscScalar *av, *bv,*v1,*v2;
  PetscScalar       *val;
  Mat               Ad,Ao;
  Mat_SeqAIJ        *aa;
  Mat_SeqAIJ        *bb;

  PetscFunctionBegin;
  ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(Ad,&av);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(Ao,&bv);CHKERRQ(ierr);

  aa = (Mat_SeqAIJ*)(Ad)->data;
  bb = (Mat_SeqAIJ*)(Ao)->data;
  ai = aa->i;
  aj = aa->j;
  bi = bb->i;
  bj = bb->j;

  rstart = A->rmap->rstart;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = (PetscInt64)aa->nz + bb->nz; /* make sure the sum won't overflow PetscInt */
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz,&val);CHKERRQ(ierr);
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
        ierr = PetscMUMPSIntCast(irow + shift,&row[jj]);CHKERRQ(ierr);
        ierr = PetscMUMPSIntCast(rstart + ajj[j] + shift,&col[jj]);CHKERRQ(ierr);
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        ierr = PetscMUMPSIntCast(irow + shift,&row[jj]);CHKERRQ(ierr);
        ierr = PetscMUMPSIntCast(garray[bjj[j]] + shift,&col[jj]);CHKERRQ(ierr);
      }
      val[jj++] = v2[j];
    }
    irow++;
  }
  ierr = MatSeqAIJRestoreArrayRead(Ad,&av);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(Ao,&bv);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          bs;
  PetscInt64        nz,i,j,k,n,jj,irow,countA,countB,idx;
  PetscMUMPSInt     *row,*col;
  const PetscScalar *av=aa->a, *bv=bb->a,*v1,*v2;
  PetscScalar       *val;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    nz   = bs2*(aa->nz + bb->nz);
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz,&val);CHKERRQ(ierr);
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
            ierr = PetscMUMPSIntCast(irow + n + shift,&row[jj]);CHKERRQ(ierr);
            ierr = PetscMUMPSIntCast(rstart + bs*ajj[k] + j + shift,&col[jj]);CHKERRQ(ierr);
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
            ierr = PetscMUMPSIntCast(irow + n + shift,&row[jj]);CHKERRQ(ierr);
            ierr = PetscMUMPSIntCast(bs*garray[bjj[k]] + j + shift,&col[jj]);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
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
  ierr = MatGetOption(A,MAT_HERMITIAN,&hermitian);CHKERRQ(ierr);
  if (hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&garray);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(Ad,&av);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArrayRead(Ao,&bv);CHKERRQ(ierr);

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
    ierr = PetscMalloc2(nz,&row,nz,&col);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz,&val);CHKERRQ(ierr);
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
        ierr = PetscMUMPSIntCast(irow + shift,&row[jj]);CHKERRQ(ierr);
        ierr = PetscMUMPSIntCast(rstart + ajj[j] + shift,&col[jj]);CHKERRQ(ierr);
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j=0; j < countB; j++) {
      if (garray[bjj[j]] > rstart) {
        if (reuse == MAT_INITIAL_MATRIX) {
          ierr = PetscMUMPSIntCast(irow + shift,&row[jj]);CHKERRQ(ierr);
          ierr = PetscMUMPSIntCast(garray[bjj[j]] + shift,&col[jj]);CHKERRQ(ierr);
        }
        val[jj++] = v2[j];
      }
    }
    irow++;
  }
  ierr = MatSeqAIJRestoreArrayRead(Ad,&av);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(Ao,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MUMPS(Mat A)
{
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->data;

  PetscFunctionBegin;
  ierr = PetscFree2(mumps->id.sol_loc,mumps->id.isol_loc);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mumps->scat_rhs);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mumps->scat_sol);CHKERRQ(ierr);
  ierr = VecDestroy(&mumps->b_seq);CHKERRQ(ierr);
  ierr = VecDestroy(&mumps->x_seq);CHKERRQ(ierr);
  ierr = PetscFree(mumps->id.perm_in);CHKERRQ(ierr);
  ierr = PetscFree2(mumps->irn,mumps->jcn);CHKERRQ(ierr);
  ierr = PetscFree(mumps->val_alloc);CHKERRQ(ierr);
  ierr = PetscFree(mumps->info);CHKERRQ(ierr);
  ierr = MatMumpsResetSchur_Private(mumps);CHKERRQ(ierr);
  mumps->id.job = JOB_END;
  PetscMUMPS_c(mumps);
  if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in MatDestroy_MUMPS: INFOG(1)=%d\n",mumps->id.INFOG(1));
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  if (mumps->use_petsc_omp_support) {
    ierr = PetscOmpCtrlDestroy(&mumps->omp_ctrl);CHKERRQ(ierr);
    ierr = PetscFree2(mumps->rhs_loc,mumps->rhs_recvbuf);CHKERRQ(ierr);
    ierr = PetscFree3(mumps->rhs_nrow,mumps->rhs_recvcounts,mumps->rhs_disps);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(mumps->ia_alloc);CHKERRQ(ierr);
  ierr = PetscFree(mumps->ja_alloc);CHKERRQ(ierr);
  ierr = PetscFree(mumps->recvcount);CHKERRQ(ierr);
  ierr = PetscFree(mumps->reqs);CHKERRQ(ierr);
  ierr = PetscFree(mumps->irhs_loc);CHKERRQ(ierr);
  if (mumps->mumps_comm != MPI_COMM_NULL) {ierr = MPI_Comm_free(&mumps->mumps_comm);CHKERRMPI(ierr);}
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInverse_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMumpsGetInverseTranspose_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set up the distributed RHS info for MUMPS. <nrhs> is the number of RHS. <array> points to start of RHS on the local processor. */
static PetscErrorCode MatMumpsSetUpDistRHSInfo(Mat A,PetscInt nrhs,const PetscScalar *array)
{
  PetscErrorCode     ierr;
  Mat_MUMPS          *mumps=(Mat_MUMPS*)A->data;
  const PetscMPIInt  ompsize=mumps->omp_comm_size;
  PetscInt           i,m,M,rstart;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  if (M > PETSC_MUMPS_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscInt too long for PetscMUMPSInt");
  if (ompsize == 1) {
    if (!mumps->irhs_loc) {
      mumps->nloc_rhs = m;
      ierr = PetscMalloc1(m,&mumps->irhs_loc);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
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
        ierr = PetscMalloc2(ompsize,&omp_ranks,ompsize,&petsc_ranks);CHKERRQ(ierr);
        ierr = PetscMalloc3(ompsize,&mumps->rhs_nrow,ompsize,&mumps->rhs_recvcounts,ompsize,&mumps->rhs_disps);CHKERRQ(ierr);
        ierr = MPI_Comm_group(mumps->petsc_comm,&petsc_group);CHKERRMPI(ierr);
        ierr = MPI_Comm_group(mumps->omp_comm,&omp_group);CHKERRMPI(ierr);
        for (j=0; j<ompsize; j++) omp_ranks[j] = j;
        ierr = MPI_Group_translate_ranks(omp_group,ompsize,omp_ranks,petsc_group,petsc_ranks);CHKERRMPI(ierr);

        /* Populate mumps->irhs_loc[], rhs_nrow[] */
        mumps->nloc_rhs = 0;
        ierr = MatGetOwnershipRanges(A,&ranges);CHKERRQ(ierr);
        for (j=0; j<ompsize; j++) {
          mumps->rhs_nrow[j] = ranges[petsc_ranks[j]+1] - ranges[petsc_ranks[j]];
          mumps->nloc_rhs   += mumps->rhs_nrow[j];
        }
        ierr = PetscMalloc1(mumps->nloc_rhs,&mumps->irhs_loc);CHKERRQ(ierr);
        for (j=k=0; j<ompsize; j++) {
          for (i=ranges[petsc_ranks[j]]; i<ranges[petsc_ranks[j]+1]; i++,k++) mumps->irhs_loc[k] = i+1; /* uses 1-based indices */
        }

        ierr = PetscFree2(omp_ranks,petsc_ranks);CHKERRQ(ierr);
        ierr = MPI_Group_free(&petsc_group);CHKERRMPI(ierr);
        ierr = MPI_Group_free(&omp_group);CHKERRMPI(ierr);
      }

      /* Realloc buffers when current nrhs is bigger than what we have met */
      if (nrhs > mumps->max_nrhs) {
        ierr = PetscFree2(mumps->rhs_loc,mumps->rhs_recvbuf);CHKERRQ(ierr);
        ierr = PetscMalloc2(mumps->nloc_rhs*nrhs,&mumps->rhs_loc,mumps->nloc_rhs*nrhs,&mumps->rhs_recvbuf);CHKERRQ(ierr);
        mumps->max_nrhs = nrhs;
      }

      /* Setup recvcounts[], disps[], recvbuf on omp rank 0 for the upcoming MPI_Gatherv */
      for (j=0; j<ompsize; j++) {ierr = PetscMPIIntCast(mumps->rhs_nrow[j]*nrhs,&mumps->rhs_recvcounts[j]);CHKERRQ(ierr);}
      mumps->rhs_disps[0] = 0;
      for (j=1; j<ompsize; j++) {
        mumps->rhs_disps[j] = mumps->rhs_disps[j-1] + mumps->rhs_recvcounts[j-1];
        if (mumps->rhs_disps[j] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscMPIInt overflow!");
      }
      recvbuf = (nrhs == 1) ? mumps->rhs_loc : mumps->rhs_recvbuf; /* Directly use rhs_loc[] as recvbuf. Single rhs is common in Ax=b */
    }

    ierr = PetscMPIIntCast(m*nrhs,&sendcount);CHKERRQ(ierr);
    ierr = MPI_Gatherv(array,sendcount,MPIU_SCALAR,recvbuf,mumps->rhs_recvcounts,mumps->rhs_disps,MPIU_SCALAR,0,mumps->omp_comm);CHKERRMPI(ierr);

    if (mumps->is_omp_master) {
      if (nrhs > 1) { /* Copy & re-arrange data from rhs_recvbuf[] to mumps->rhs_loc[] only when there are multiple rhs */
        PetscScalar *dst,*dstbase = mumps->rhs_loc;
        for (j=0; j<ompsize; j++) {
          const PetscScalar *src = mumps->rhs_recvbuf + mumps->rhs_disps[j];
          dst = dstbase;
          for (i=0; i<nrhs; i++) {
            ierr = PetscArraycpy(dst,src,mumps->rhs_nrow[j]);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;
  PetscInt           i;
  PetscBool          second_solve = PETSC_FALSE;
  static PetscBool   cite1 = PETSC_FALSE,cite2 = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister("@article{MUMPS01,\n  author = {P.~R. Amestoy and I.~S. Duff and J.-Y. L'Excellent and J. Koster},\n  title = {A fully asynchronous multifrontal solver using distributed dynamic scheduling},\n  journal = {SIAM Journal on Matrix Analysis and Applications},\n  volume = {23},\n  number = {1},\n  pages = {15--41},\n  year = {2001}\n}\n",&cite1);CHKERRQ(ierr);
  ierr = PetscCitationsRegister("@article{MUMPS02,\n  author = {P.~R. Amestoy and A. Guermouche and J.-Y. L'Excellent and S. Pralet},\n  title = {Hybrid scheduling for the parallel solution of linear systems},\n  journal = {Parallel Computing},\n  volume = {32},\n  number = {2},\n  pages = {136--156},\n  year = {2006}\n}\n",&cite2);CHKERRQ(ierr);

  if (A->factorerrortype) {
    ierr = PetscInfo2(A,"MatSolve is called with singular matrix factor, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
    ierr = VecSetInf(x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  mumps->id.nrhs = 1;
  if (mumps->petsc_size > 1) {
    if (mumps->ICNTL20 == 10) {
      mumps->id.ICNTL(20) = 10; /* dense distributed RHS */
      ierr = VecGetArrayRead(b,&rarray);CHKERRQ(ierr);
      ierr = MatMumpsSetUpDistRHSInfo(A,1,rarray);CHKERRQ(ierr);
    } else {
      mumps->id.ICNTL(20) = 0; /* dense centralized RHS; Scatter b into a sequential rhs vector*/
      ierr = VecScatterBegin(mumps->scat_rhs,b,mumps->b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mumps->scat_rhs,b,mumps->b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      if (!mumps->myid) {
        ierr = VecGetArray(mumps->b_seq,&array);CHKERRQ(ierr);
        mumps->id.rhs = (MumpsScalar*)array;
      }
    }
  } else {  /* petsc_size == 1 */
    mumps->id.ICNTL(20) = 0; /* dense centralized RHS */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
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
    if (mumps->petsc_size > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Parallel Schur complements not yet supported from PETSc\n");
    second_solve = PETSC_TRUE;
    ierr = MatMumpsHandleSchur_Private(A,PETSC_FALSE);CHKERRQ(ierr);
  }
  /* solve phase */
  /*-------------*/
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps);
  if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));

  /* handle expansion step of Schur complement (if any) */
  if (second_solve) {
    ierr = MatMumpsHandleSchur_Private(A,PETSC_TRUE);CHKERRQ(ierr);
  }

  if (mumps->petsc_size > 1) { /* convert mumps distributed solution to petsc mpi x */
    if (mumps->scat_sol && mumps->ICNTL9_pre != mumps->id.ICNTL(9)) {
      /* when id.ICNTL(9) changes, the contents of lsol_loc may change (not its size, lsol_loc), recreates scat_sol */
      ierr = VecScatterDestroy(&mumps->scat_sol);CHKERRQ(ierr);
    }
    if (!mumps->scat_sol) { /* create scatter scat_sol */
      PetscInt *isol2_loc=NULL;
      ierr = ISCreateStride(PETSC_COMM_SELF,mumps->id.lsol_loc,0,1,&is_iden);CHKERRQ(ierr); /* from */
      ierr = PetscMalloc1(mumps->id.lsol_loc,&isol2_loc);CHKERRQ(ierr);
      for (i=0; i<mumps->id.lsol_loc; i++) isol2_loc[i] = mumps->id.isol_loc[i]-1; /* change Fortran style to C style */
      ierr = ISCreateGeneral(PETSC_COMM_SELF,mumps->id.lsol_loc,isol2_loc,PETSC_OWN_POINTER,&is_petsc);CHKERRQ(ierr);  /* to */
      ierr = VecScatterCreate(mumps->x_seq,is_iden,x,is_petsc,&mumps->scat_sol);CHKERRQ(ierr);
      ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
      ierr = ISDestroy(&is_petsc);CHKERRQ(ierr);
      mumps->ICNTL9_pre = mumps->id.ICNTL(9); /* save current value of id.ICNTL(9) */
    }

    ierr = VecScatterBegin(mumps->scat_sol,mumps->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mumps->scat_sol,mumps->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  if (mumps->petsc_size > 1) {
    if (mumps->ICNTL20 == 10) {
      ierr = VecRestoreArrayRead(b,&rarray);CHKERRQ(ierr);
    } else if (!mumps->myid) {
      ierr = VecRestoreArray(mumps->b_seq,&array);CHKERRQ(ierr);
    }
  } else {ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);}

  ierr = PetscLogFlops(2.0*mumps->id.RINFO(3));CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
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
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&denseX,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!denseX) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  ierr = PetscObjectTypeCompareAny((PetscObject)B,&denseB,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (denseB) {
    if (B->rmap->n != X->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix B and X must have same row distribution");
    mumps->id.ICNTL(20)= 0; /* dense RHS */
  } else { /* sparse B */
    if (X == B) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_IDN,"X and B must be different matrices");
    ierr = PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&flgT);CHKERRQ(ierr);
    if (flgT) { /* input B is transpose of actural RHS matrix,
                 because mumps requires sparse compressed COLUMN storage! See MatMatTransposeSolve_MUMPS() */
      ierr = MatTransposeGetMat(B,&Bt);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONG,"Matrix B must be MATTRANSPOSEMAT matrix");
    mumps->id.ICNTL(20)= 1; /* sparse RHS */
  }

  ierr = MatGetSize(B,&M,&nrhs);CHKERRQ(ierr);
  mumps->id.nrhs = nrhs;
  mumps->id.lrhs = M;
  mumps->id.rhs  = NULL;

  if (mumps->petsc_size == 1) {
    PetscScalar *aa;
    PetscInt    spnr,*ia,*ja;
    PetscBool   second_solve = PETSC_FALSE;

    ierr = MatDenseGetArray(X,&array);CHKERRQ(ierr);
    mumps->id.rhs = (MumpsScalar*)array;

    if (denseB) {
      /* copy B to X */
      ierr = MatDenseGetArrayRead(B,&rbray);CHKERRQ(ierr);
      ierr = PetscArraycpy(array,rbray,M*nrhs);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(B,&rbray);CHKERRQ(ierr);
    } else { /* sparse B */
      ierr = MatSeqAIJGetArray(Bt,&aa);CHKERRQ(ierr);
      ierr = MatGetRowIJ(Bt,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
      ierr = PetscMUMPSIntCSRCast(mumps,spnr,ia,ja,&mumps->id.irhs_ptr,&mumps->id.irhs_sparse,&mumps->id.nz_rhs);CHKERRQ(ierr);
      mumps->id.rhs_sparse  = (MumpsScalar*)aa;
    }
    /* handle condensation step of Schur complement (if any) */
    if (mumps->id.size_schur > 0 && (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2)) {
      second_solve = PETSC_TRUE;
      ierr = MatMumpsHandleSchur_Private(A,PETSC_FALSE);CHKERRQ(ierr);
    }
    /* solve phase */
    /*-------------*/
    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(mumps);
    if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));

    /* handle expansion step of Schur complement (if any) */
    if (second_solve) {
      ierr = MatMumpsHandleSchur_Private(A,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (!denseB) { /* sparse B */
      ierr = MatSeqAIJRestoreArray(Bt,&aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(Bt,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot restore IJ structure");
    }
    ierr = MatDenseRestoreArray(X,&array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*--------- parallel case: MUMPS requires rhs B to be centralized on the host! --------*/
  if (mumps->petsc_size > 1 && mumps->id.ICNTL(19)) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Parallel Schur complements not yet supported from PETSc\n");

  /* create msol_loc to hold mumps local solution */
  isol_loc_save = mumps->id.isol_loc; /* save it for MatSolve() */
  sol_loc_save  = (PetscScalar*)mumps->id.sol_loc;

  lsol_loc  = mumps->id.lsol_loc;
  nlsol_loc = nrhs*lsol_loc;     /* length of sol_loc */
  ierr = PetscMalloc2(nlsol_loc,&sol_loc,lsol_loc,&isol_loc);CHKERRQ(ierr);
  mumps->id.sol_loc  = (MumpsScalar*)sol_loc;
  mumps->id.isol_loc = isol_loc;

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nlsol_loc,(PetscScalar*)sol_loc,&msol_loc);CHKERRQ(ierr);

  if (denseB) {
    if (mumps->ICNTL20 == 10) {
      mumps->id.ICNTL(20) = 10; /* dense distributed RHS */
      ierr = MatDenseGetArrayRead(B,&rbray);CHKERRQ(ierr);
      ierr = MatMumpsSetUpDistRHSInfo(A,nrhs,rbray);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(B,&rbray);CHKERRQ(ierr);
      ierr = MatGetLocalSize(B,&m,NULL);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),1,nrhs*m,nrhs*M,NULL,&v_mpi);CHKERRQ(ierr);
    } else {
      mumps->id.ICNTL(20) = 0; /* dense centralized RHS */
      /* TODO: Because of non-contiguous indices, the created vecscatter scat_rhs is not done in MPI_Gather, resulting in
        very inefficient communication. An optimization is to use VecScatterCreateToZero to gather B to rank 0. Then on rank
        0, re-arrange B into desired order, which is a local operation.
      */

      /* scatter v_mpi to b_seq because MUMPS before 5.3.0 only supports centralized rhs */
      /* wrap dense rhs matrix B into a vector v_mpi */
      ierr = MatGetLocalSize(B,&m,NULL);CHKERRQ(ierr);
      ierr = MatDenseGetArray(B,&bray);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),1,nrhs*m,nrhs*M,(const PetscScalar*)bray,&v_mpi);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(B,&bray);CHKERRQ(ierr);

      /* scatter v_mpi to b_seq in proc[0]. MUMPS requires rhs to be centralized on the host! */
      if (!mumps->myid) {
        PetscInt *idx;
        /* idx: maps from k-th index of v_mpi to (i,j)-th global entry of B */
        ierr = PetscMalloc1(nrhs*M,&idx);CHKERRQ(ierr);
        ierr = MatGetOwnershipRanges(B,&rstart);CHKERRQ(ierr);
        k = 0;
        for (proc=0; proc<mumps->petsc_size; proc++) {
          for (j=0; j<nrhs; j++) {
            for (i=rstart[proc]; i<rstart[proc+1]; i++) idx[k++] = j*M + i;
          }
        }

        ierr = VecCreateSeq(PETSC_COMM_SELF,nrhs*M,&b_seq);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,nrhs*M,idx,PETSC_OWN_POINTER,&is_to);CHKERRQ(ierr);
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
    }
  } else { /* sparse B */
    b = (Mat_MPIAIJ*)Bt->data;

    /* wrap dense X into a vector v_mpi */
    ierr = MatGetLocalSize(X,&m,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(X,&bray);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)X),1,nrhs*m,nrhs*M,(const PetscScalar*)bray,&v_mpi);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(X,&bray);CHKERRQ(ierr);

    if (!mumps->myid) {
      ierr = MatSeqAIJGetArray(b->A,&aa);CHKERRQ(ierr);
      ierr = MatGetRowIJ(b->A,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
      ierr = PetscMUMPSIntCSRCast(mumps,spnr,ia,ja,&mumps->id.irhs_ptr,&mumps->id.irhs_sparse,&mumps->id.nz_rhs);CHKERRQ(ierr);
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
  if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",mumps->id.INFOG(1));

  /* scatter mumps distributed solution to petsc vector v_mpi, which shares local arrays with solution matrix X */
  ierr = MatDenseGetArray(X,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(v_mpi,array);CHKERRQ(ierr);

  /* create scatter scat_sol */
  ierr = MatGetOwnershipRanges(X,&rstart);CHKERRQ(ierr);
  /* iidx: index for scatter mumps solution to petsc X */

  ierr = ISCreateStride(PETSC_COMM_SELF,nlsol_loc,0,1,&is_from);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlsol_loc,&idxx);CHKERRQ(ierr);
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
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nlsol_loc,idxx,PETSC_COPY_VALUES,&is_to);CHKERRQ(ierr);
  ierr = VecScatterCreate(msol_loc,is_from,v_mpi,is_to,&scat_sol);CHKERRQ(ierr);
  ierr = VecScatterBegin(scat_sol,msol_loc,v_mpi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = ISDestroy(&is_from);CHKERRQ(ierr);
  ierr = ISDestroy(&is_to);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat_sol,msol_loc,v_mpi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(X,&array);CHKERRQ(ierr);

  /* free spaces */
  mumps->id.sol_loc  = (MumpsScalar*)sol_loc_save;
  mumps->id.isol_loc = isol_loc_save;

  ierr = PetscFree2(sol_loc,isol_loc);CHKERRQ(ierr);
  ierr = PetscFree(idxx);CHKERRQ(ierr);
  ierr = VecDestroy(&msol_loc);CHKERRQ(ierr);
  ierr = VecDestroy(&v_mpi);CHKERRQ(ierr);
  if (!denseB) {
    if (!mumps->myid) {
      b = (Mat_MPIAIJ*)Bt->data;
      ierr = MatSeqAIJRestoreArray(b->A,&aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(b->A,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot restore IJ structure");
    }
  } else {
    if (mumps->ICNTL20 == 0) {
      ierr = VecDestroy(&b_seq);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&scat_rhs);CHKERRQ(ierr);
    }
  }
  ierr = VecScatterDestroy(&scat_sol);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*nrhs*mumps->id.RINFO(3));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeSolve_MUMPS(Mat A,Mat Bt,Mat X)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  Mat            B;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)Bt,&flg,MATSEQAIJ,MATMPIAIJ,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)Bt),PETSC_ERR_ARG_WRONG,"Matrix Bt must be MATAIJ matrix");

  /* Create B=Bt^T that uses Bt's data structure */
  ierr = MatCreateTranspose(Bt,&B);CHKERRQ(ierr);

  ierr = MatMatSolve_MUMPS(A,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)F),&size);CHKERRMPI(ierr);
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

PetscErrorCode MatMumpsGatherNonzerosOnMaster(MatReuse reuse,Mat_MUMPS *mumps)
{
  PetscErrorCode ierr;
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
      if (mumps->is_omp_master) {ierr = PetscMalloc1(osize,&mumps->recvcount);CHKERRQ(ierr);}
      ierr = MPI_Gather(&mumps->nnz,1,MPIU_INT64,mumps->recvcount,1,MPIU_INT64,0/*master*/,mumps->omp_comm);CHKERRMPI(ierr);

      /* Then each computes number of send/recvs */
      if (mumps->is_omp_master) {
        /* Start from 1 since self communication is not done in MPI */
        nreqs = 0;
        for (i=1; i<osize; i++) nreqs += (mumps->recvcount[i]+PETSC_MPI_INT_MAX-1)/PETSC_MPI_INT_MAX;
      } else {
        nreqs = (mumps->nnz+PETSC_MPI_INT_MAX-1)/PETSC_MPI_INT_MAX;
      }
      ierr = PetscMalloc1(nreqs*3,&mumps->reqs);CHKERRQ(ierr); /* Triple the requests since we send irn, jcn and val seperately */

      /* The following code is doing a very simple thing: omp_master rank gathers irn/jcn/val from others.
         MPI_Gatherv would be enough if it supports big counts > 2^31-1. Since it does not, and mumps->nnz
         might be a prime number > 2^31-1, we have to slice the message. Note omp_comm_size
         is very small, the current approach should have no extra overhead compared to MPI_Gatherv.
       */
      nreqs = 0; /* counter for actual send/recvs */
      if (mumps->is_omp_master) {
        for (i=0,totnnz=0; i<osize; i++) totnnz += mumps->recvcount[i]; /* totnnz = sum of nnz over omp_comm */
        ierr = PetscMalloc2(totnnz,&irn,totnnz,&jcn);CHKERRQ(ierr);
        ierr = PetscMalloc1(totnnz,&val);CHKERRQ(ierr);

        /* Self communication */
        ierr = PetscArraycpy(irn,mumps->irn,mumps->nnz);CHKERRQ(ierr);
        ierr = PetscArraycpy(jcn,mumps->jcn,mumps->nnz);CHKERRQ(ierr);
        ierr = PetscArraycpy(val,mumps->val,mumps->nnz);CHKERRQ(ierr);

        /* Replace mumps->irn/jcn etc on master with the newly allocated bigger arrays */
        ierr = PetscFree2(mumps->irn,mumps->jcn);CHKERRQ(ierr);
        ierr = PetscFree(mumps->val_alloc);CHKERRQ(ierr);
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
            ierr    = MPI_Irecv(irn,count,MPIU_MUMPSINT,i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
            ierr    = MPI_Irecv(jcn,count,MPIU_MUMPSINT,i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
            ierr    = MPI_Irecv(val,count,MPIU_SCALAR,  i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
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
          ierr    = MPI_Isend(irn,count,MPIU_MUMPSINT,0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
          ierr    = MPI_Isend(jcn,count,MPIU_MUMPSINT,0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
          ierr    = MPI_Isend(val,count,MPIU_SCALAR,  0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
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
            ierr    = MPI_Irecv(val,count,MPIU_SCALAR,i,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
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
          ierr    = MPI_Isend(val,count,MPIU_SCALAR,0,mumps->tag,mumps->omp_comm,&mumps->reqs[nreqs++]);CHKERRMPI(ierr);
          val    += count;
          count   = PetscMin(remain,PETSC_MPI_INT_MAX);
          remain -= count;
        }
      }
    }
    ierr = MPI_Waitall(nreqs,mumps->reqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
    mumps->tag++; /* It is totally fine for above send/recvs to share one mpi tag */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorNumeric_MUMPS(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps =(Mat_MUMPS*)(F)->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0 && !(mumps->id.INFOG(1) == -16 && mumps->id.INFOG(1) == 0)) {
    if (mumps->id.INFOG(1) == -6) {
      ierr = PetscInfo2(A,"MatFactorNumeric is called with singular matrix structure, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
    }
    ierr = PetscInfo2(A,"MatFactorNumeric is called after analysis phase fails, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_REUSE_MATRIX, mumps);CHKERRQ(ierr);
  ierr = MatMumpsGatherNonzerosOnMaster(MAT_REUSE_MATRIX,mumps);CHKERRQ(ierr);

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
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));
    } else {
      if (mumps->id.INFOG(1) == -10) { /* numerically singular matrix */
        ierr = PetscInfo2(F,"matrix is numerically singular, INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      } else if (mumps->id.INFOG(1) == -13) {
        ierr = PetscInfo2(F,"MUMPS in numerical factorization phase: INFOG(1)=%d, cannot allocate required memory %d megabytes\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
      } else if (mumps->id.INFOG(1) == -8 || mumps->id.INFOG(1) == -9 || (-16 < mumps->id.INFOG(1) && mumps->id.INFOG(1) < -10)) {
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
#if defined(PETSC_HAVE_CUDA)
    F->schur->offloadmask = PETSC_OFFLOAD_CPU;
#endif
    if (mumps->id.ICNTL(19) == 1) { /* stored by rows */
      mumps->id.ICNTL(19) = 2;
      ierr = MatTranspose(F->schur,MAT_INPLACE_MATRIX,&F->schur);CHKERRQ(ierr);
    }
    ierr = MatFactorRestoreSchurComplement(F,NULL,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
  }

  /* just to be sure that ICNTL(19) value returned by a call from MatMumpsGetIcntl is always consistent */
  if (!mumps->sym && mumps->id.ICNTL(19) && mumps->id.ICNTL(19) != 1) mumps->id.ICNTL(19) = 3;

  if (!mumps->is_omp_master) mumps->id.INFO(23) = 0;
  if (mumps->petsc_size > 1) {
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
  ierr = PetscLogFlops(mumps->id.RINFO(2));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Sets MUMPS options from the options database */
PetscErrorCode PetscSetMUMPSFromOptions(Mat F, Mat A)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  PetscMUMPSInt  icntl=0;
  PetscInt       info[80],i,ninfo=80;
  PetscBool      flg=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MUMPS Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_1","ICNTL(1): output stream for error messages","None",mumps->id.ICNTL(1),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(1) = icntl;
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_2","ICNTL(2): output stream for diagnostic printing, statistics, and warning","None",mumps->id.ICNTL(2),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(2) = icntl;
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_3","ICNTL(3): output stream for global information, collected on the host","None",mumps->id.ICNTL(3),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(3) = icntl;

  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_4","ICNTL(4): level of printing (0 to 4)","None",mumps->id.ICNTL(4),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(4) = icntl;
  if (mumps->id.ICNTL(4) || PetscLogPrintInfo) mumps->id.ICNTL(3) = 6; /* resume MUMPS default id.ICNTL(3) = 6 */

  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_6","ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7)","None",mumps->id.ICNTL(6),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(6) = icntl;

  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_7","ICNTL(7): computes a symmetric permutation in sequential analysis. 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto(default)","None",mumps->id.ICNTL(7),&icntl,&flg);CHKERRQ(ierr);
  if (flg) {
    if (icntl == 1 || icntl < 0 || icntl > 7) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Valid values are 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto\n");
    mumps->id.ICNTL(7) = icntl;
  }

  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_8","ICNTL(8): scaling strategy (-2 to 8 or 77)","None",mumps->id.ICNTL(8),&mumps->id.ICNTL(8),NULL);CHKERRQ(ierr);
  /* ierr = PetscOptionsInt("-mat_mumps_icntl_9","ICNTL(9): computes the solution using A or A^T","None",mumps->id.ICNTL(9),&mumps->id.ICNTL(9),NULL);CHKERRQ(ierr); handled by MatSolveTranspose_MUMPS() */
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_10","ICNTL(10): max num of refinements","None",mumps->id.ICNTL(10),&mumps->id.ICNTL(10),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_11","ICNTL(11): statistics related to an error analysis (via -ksp_view)","None",mumps->id.ICNTL(11),&mumps->id.ICNTL(11),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_12","ICNTL(12): an ordering strategy for symmetric matrices (0 to 3)","None",mumps->id.ICNTL(12),&mumps->id.ICNTL(12),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_13","ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting","None",mumps->id.ICNTL(13),&mumps->id.ICNTL(13),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_14","ICNTL(14): percentage increase in the estimated working space","None",mumps->id.ICNTL(14),&mumps->id.ICNTL(14),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_19","ICNTL(19): computes the Schur complement","None",mumps->id.ICNTL(19),&mumps->id.ICNTL(19),NULL);CHKERRQ(ierr);
  if (mumps->id.ICNTL(19) <= 0 || mumps->id.ICNTL(19) > 3) { /* reset any schur data (if any) */
    ierr = MatDestroy(&F->schur);CHKERRQ(ierr);
    ierr = MatMumpsResetSchur_Private(mumps);CHKERRQ(ierr);
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
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_20","ICNTL(20): give mumps centralized (0) or distributed (10) dense right-hand sides","None",mumps->ICNTL20,&mumps->ICNTL20,&flg);CHKERRQ(ierr);
  if (flg && mumps->ICNTL20 != 10 && mumps->ICNTL20 != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"ICNTL(20)=%d is not supported by the PETSc/MUMPS interface. Allowed values are 0, 10\n",(int)mumps->ICNTL20);
#if PETSC_PKG_MUMPS_VERSION_LT(5,3,0)
  if (flg && mumps->ICNTL20 == 10) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"ICNTL(20)=10 is not supported before MUMPS-5.3.0\n");
#endif
  /* ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_21","ICNTL(21): the distribution (centralized or distributed) of the solution vectors","None",mumps->id.ICNTL(21),&mumps->id.ICNTL(21),NULL);CHKERRQ(ierr); we only use distributed solution vector */

  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_22","ICNTL(22): in-core/out-of-core factorization and solve (0 or 1)","None",mumps->id.ICNTL(22),&mumps->id.ICNTL(22),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_23","ICNTL(23): max size of the working memory (MB) that can allocate per processor","None",mumps->id.ICNTL(23),&mumps->id.ICNTL(23),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_24","ICNTL(24): detection of null pivot rows (0 or 1)","None",mumps->id.ICNTL(24),&mumps->id.ICNTL(24),NULL);CHKERRQ(ierr);
  if (mumps->id.ICNTL(24)) {
    mumps->id.ICNTL(13) = 1; /* turn-off ScaLAPACK to help with the correct detection of null pivots */
  }

  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_25","ICNTL(25): computes a solution of a deficient matrix and a null space basis","None",mumps->id.ICNTL(25),&mumps->id.ICNTL(25),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_26","ICNTL(26): drives the solution phase if a Schur complement matrix","None",mumps->id.ICNTL(26),&mumps->id.ICNTL(26),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_27","ICNTL(27): controls the blocking size for multiple right-hand sides","None",mumps->id.ICNTL(27),&mumps->id.ICNTL(27),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_28","ICNTL(28): use 1 for sequential analysis and ictnl(7) ordering, or 2 for parallel analysis and ictnl(29) ordering","None",mumps->id.ICNTL(28),&mumps->id.ICNTL(28),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_29","ICNTL(29): parallel ordering 1 = ptscotch, 2 = parmetis","None",mumps->id.ICNTL(29),&mumps->id.ICNTL(29),NULL);CHKERRQ(ierr);
  /* ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_30","ICNTL(30): compute user-specified set of entries in inv(A)","None",mumps->id.ICNTL(30),&mumps->id.ICNTL(30),NULL);CHKERRQ(ierr); */ /* call MatMumpsGetInverse() directly */
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_31","ICNTL(31): indicates which factors may be discarded during factorization","None",mumps->id.ICNTL(31),&mumps->id.ICNTL(31),NULL);CHKERRQ(ierr);
  /* ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_32","ICNTL(32): performs the forward elemination of the right-hand sides during factorization","None",mumps->id.ICNTL(32),&mumps->id.ICNTL(32),NULL);CHKERRQ(ierr);  -- not supported by PETSc API */
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_33","ICNTL(33): compute determinant","None",mumps->id.ICNTL(33),&mumps->id.ICNTL(33),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_35","ICNTL(35): activates Block Low Rank (BLR) based factorization","None",mumps->id.ICNTL(35),&mumps->id.ICNTL(35),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_36","ICNTL(36): choice of BLR factorization variant","None",mumps->id.ICNTL(36),&mumps->id.ICNTL(36),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsMUMPSInt("-mat_mumps_icntl_38","ICNTL(38): estimated compression rate of LU factors with BLR","None",mumps->id.ICNTL(38),&mumps->id.ICNTL(38),NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-mat_mumps_cntl_1","CNTL(1): relative pivoting threshold","None",mumps->id.CNTL(1),&mumps->id.CNTL(1),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_2","CNTL(2): stopping criterion of refinement","None",mumps->id.CNTL(2),&mumps->id.CNTL(2),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_3","CNTL(3): absolute pivoting threshold","None",mumps->id.CNTL(3),&mumps->id.CNTL(3),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_4","CNTL(4): value for static pivoting","None",mumps->id.CNTL(4),&mumps->id.CNTL(4),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_5","CNTL(5): fixation for null pivots","None",mumps->id.CNTL(5),&mumps->id.CNTL(5),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_7","CNTL(7): dropping parameter used during BLR","None",mumps->id.CNTL(7),&mumps->id.CNTL(7),NULL);CHKERRQ(ierr);

  ierr = PetscOptionsString("-mat_mumps_ooc_tmpdir", "out of core directory", "None", mumps->id.ooc_tmpdir, mumps->id.ooc_tmpdir, sizeof(mumps->id.ooc_tmpdir), NULL);CHKERRQ(ierr);

  ierr = PetscOptionsIntArray("-mat_mumps_view_info","request INFO local to each processor","",info,&ninfo,NULL);CHKERRQ(ierr);
  if (ninfo) {
    if (ninfo > 80) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"number of INFO %d must <= 80\n",ninfo);
    ierr = PetscMalloc1(ninfo,&mumps->info);CHKERRQ(ierr);
    mumps->ninfo = ninfo;
    for (i=0; i<ninfo; i++) {
      if (info[i] < 0 || info[i]>80) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index of INFO %d must between 1 and 80\n",ninfo);
      else  mumps->info[i] = info[i];
    }
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscInitializeMUMPS(Mat A,Mat_MUMPS *mumps)
{
  PetscErrorCode ierr;
  PetscInt       nthreads=0;
  MPI_Comm       newcomm=MPI_COMM_NULL;

  PetscFunctionBegin;
  mumps->petsc_comm = PetscObjectComm((PetscObject)A);
  ierr = MPI_Comm_size(mumps->petsc_comm,&mumps->petsc_size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(mumps->petsc_comm,&mumps->myid);CHKERRMPI(ierr);/* "if (!myid)" still works even if mumps_comm is different */

  ierr = PetscOptionsHasName(NULL,((PetscObject)A)->prefix,"-mat_mumps_use_omp_threads",&mumps->use_petsc_omp_support);CHKERRQ(ierr);
  if (mumps->use_petsc_omp_support) nthreads = -1; /* -1 will let PetscOmpCtrlCreate() guess a proper value when user did not supply one */
  ierr = PetscOptionsGetInt(NULL,((PetscObject)A)->prefix,"-mat_mumps_use_omp_threads",&nthreads,NULL);CHKERRQ(ierr);
  if (mumps->use_petsc_omp_support) {
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
    ierr = PetscOmpCtrlCreate(mumps->petsc_comm,nthreads,&mumps->omp_ctrl);CHKERRQ(ierr);
    ierr = PetscOmpCtrlGetOmpComms(mumps->omp_ctrl,&mumps->omp_comm,&mumps->mumps_comm,&mumps->is_omp_master);CHKERRQ(ierr);
#else
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"the system does not have PETSc OpenMP support but you added the -%smat_mumps_use_omp_threads option. Configure PETSc with --with-openmp --download-hwloc (or --with-hwloc) to enable it, see more in MATSOLVERMUMPS manual\n",((PetscObject)A)->prefix?((PetscObject)A)->prefix:"");
#endif
  } else {
    mumps->omp_comm      = PETSC_COMM_SELF;
    mumps->mumps_comm    = mumps->petsc_comm;
    mumps->is_omp_master = PETSC_TRUE;
  }
  ierr = MPI_Comm_size(mumps->omp_comm,&mumps->omp_comm_size);CHKERRMPI(ierr);
  mumps->reqs = NULL;
  mumps->tag  = 0;

  /* It looks like MUMPS does not dup the input comm. Dup a new comm for MUMPS to avoid any tag mismatches. */
  if (mumps->mumps_comm != MPI_COMM_NULL) {
    ierr = MPI_Comm_dup(mumps->mumps_comm,&newcomm);CHKERRMPI(ierr);
    mumps->mumps_comm = newcomm;
  }

  mumps->id.comm_fortran = MPI_Comm_c2f(mumps->mumps_comm);
  mumps->id.job = JOB_INIT;
  mumps->id.par = 1;  /* host participates factorizaton and solve */
  mumps->id.sym = mumps->sym;

  PetscMUMPS_c(mumps);
  if (mumps->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in PetscInitializeMUMPS: INFOG(1)=%d\n",mumps->id.INFOG(1));

  /* copy MUMPS default control values from master to slaves. Although slaves do not call MUMPS, they may access these values in code.
     For example, ICNTL(9) is initialized to 1 by MUMPS and slaves check ICNTL(9) in MatSolve_MUMPS.
   */
  ierr = MPI_Bcast(mumps->id.icntl,40,MPI_INT,  0,mumps->omp_comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(mumps->id.cntl, 15,MPIU_REAL,0,mumps->omp_comm);CHKERRMPI(ierr);

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
      } else if (mumps->id.INFOG(1) == -16 && mumps->id.INFOG(1) == 0) {
        ierr = PetscInfo(F,"Empty matrix\n");CHKERRQ(ierr);
      } else {
        ierr = PetscInfo2(F,"Error reported by MUMPS in analysis phase: INFOG(1)=%d, INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));CHKERRQ(ierr);
        F->factorerrortype = MAT_FACTOR_OTHER;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps);CHKERRQ(ierr);
  ierr = MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX,mumps);CHKERRQ(ierr);

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

          ierr = PetscMalloc1(M,&mumps->id.perm_in);CHKERRQ(ierr);
          ierr = ISGetIndices(r,&idx);CHKERRQ(ierr);
          for (i=0; i<M; i++) {ierr = PetscMUMPSIntCast(idx[i]+1,&(mumps->id.perm_in[i]));CHKERRQ(ierr);} /* perm_in[]: start from 1, not 0! */
          ierr = ISRestoreIndices(r,&idx);CHKERRQ(ierr);
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
      ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
      ierr = VecScatterCreateToZero(b,&mumps->scat_rhs,&mumps->b_seq);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
    }
    break;
  }
  PetscMUMPS_c(mumps);
  ierr = MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps);CHKERRQ(ierr);

  F->ops->lufactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve           = MatSolve_MUMPS;
  F->ops->solvetranspose  = MatSolveTranspose_MUMPS;
  F->ops->matsolve        = MatMatSolve_MUMPS;
  F->ops->mattransposesolve = MatMatTransposeSolve_MUMPS;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
PetscErrorCode MatLUFactorSymbolic_BAIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *mumps = (Mat_MUMPS*)F->data;
  PetscErrorCode ierr;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps);CHKERRQ(ierr);
  ierr = MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX,mumps);CHKERRQ(ierr);

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
      ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
      ierr = VecScatterCreateToZero(b,&mumps->scat_rhs,&mumps->b_seq);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
    }
    break;
  }
  PetscMUMPS_c(mumps);
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
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  mumps->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps);CHKERRQ(ierr);
  ierr = MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX,mumps);CHKERRQ(ierr);

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
      ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
      ierr = VecScatterCreateToZero(b,&mumps->scat_rhs,&mumps->b_seq);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
    }
    break;
  }
  PetscMUMPS_c(mumps);
  ierr = MatFactorSymbolic_MUMPS_ReportIfError(F,A,info,mumps);CHKERRQ(ierr);

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
      ierr = PetscViewerASCIIPrintf(viewer,"  SYM (matrix type):                   %d\n",mumps->id.sym);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  PAR (host participation):            %d\n",mumps->id.par);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(1) (output for error):         %d\n",mumps->id.ICNTL(1));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(2) (output of diagnostic msg): %d\n",mumps->id.ICNTL(2));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(3) (output for global info):   %d\n",mumps->id.ICNTL(3));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(4) (level of printing):        %d\n",mumps->id.ICNTL(4));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(5) (input mat struct):         %d\n",mumps->id.ICNTL(5));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(6) (matrix prescaling):        %d\n",mumps->id.ICNTL(6));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(7) (sequential matrix ordering):%d\n",mumps->id.ICNTL(7));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(8) (scaling strategy):        %d\n",mumps->id.ICNTL(8));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(10) (max num of refinements):  %d\n",mumps->id.ICNTL(10));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(11) (error analysis):          %d\n",mumps->id.ICNTL(11));CHKERRQ(ierr);
      if (mumps->id.ICNTL(11)>0) {
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(4) (inf norm of input mat):        %g\n",mumps->id.RINFOG(4));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(5) (inf norm of solution):         %g\n",mumps->id.RINFOG(5));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(6) (inf norm of residual):         %g\n",mumps->id.RINFOG(6));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(7),RINFOG(8) (backward error est): %g, %g\n",mumps->id.RINFOG(7),mumps->id.RINFOG(8));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(9) (error estimate):               %g \n",mumps->id.RINFOG(9));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n",mumps->id.RINFOG(10),mumps->id.RINFOG(11));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(12) (efficiency control):                         %d\n",mumps->id.ICNTL(12));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(13) (sequential factorization of the root node):  %d\n",mumps->id.ICNTL(13));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(14) (percentage of estimated workspace increase): %d\n",mumps->id.ICNTL(14));CHKERRQ(ierr);
      /* ICNTL(15-17) not used */
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(18) (input mat struct):                           %d\n",mumps->id.ICNTL(18));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(19) (Schur complement info):                      %d\n",mumps->id.ICNTL(19));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(20) (RHS sparse pattern):                         %d\n",mumps->id.ICNTL(20));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(21) (solution struct):                            %d\n",mumps->id.ICNTL(21));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(22) (in-core/out-of-core facility):               %d\n",mumps->id.ICNTL(22));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(23) (max size of memory can be allocated locally):%d\n",mumps->id.ICNTL(23));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(24) (detection of null pivot rows):               %d\n",mumps->id.ICNTL(24));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(25) (computation of a null space basis):          %d\n",mumps->id.ICNTL(25));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(26) (Schur options for RHS or solution):          %d\n",mumps->id.ICNTL(26));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(27) (blocking size for multiple RHS):             %d\n",mumps->id.ICNTL(27));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(28) (use parallel or sequential ordering):        %d\n",mumps->id.ICNTL(28));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(29) (parallel ordering):                          %d\n",mumps->id.ICNTL(29));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(30) (user-specified set of entries in inv(A)):    %d\n",mumps->id.ICNTL(30));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(31) (factors is discarded in the solve phase):    %d\n",mumps->id.ICNTL(31));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(33) (compute determinant):                        %d\n",mumps->id.ICNTL(33));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(35) (activate BLR based factorization):           %d\n",mumps->id.ICNTL(35));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(36) (choice of BLR factorization variant):        %d\n",mumps->id.ICNTL(36));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(38) (estimated compression rate of LU factors):   %d\n",mumps->id.ICNTL(38));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(1) (relative pivoting threshold):      %g \n",mumps->id.CNTL(1));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(2) (stopping criterion of refinement): %g \n",mumps->id.CNTL(2));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(3) (absolute pivoting threshold):      %g \n",mumps->id.CNTL(3));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(4) (value of static pivoting):         %g \n",mumps->id.CNTL(4));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(5) (fixation for null pivots):         %g \n",mumps->id.CNTL(5));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(7) (dropping parameter for BLR):       %g \n",mumps->id.CNTL(7));CHKERRQ(ierr);

      /* information local to each processor */
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] %d\n",mumps->myid,mumps->id.INFO(15));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(16) (size of (in MB) MUMPS internal data used during numerical factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d\n",mumps->myid,mumps->id.INFO(16));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(23) (num of pivots eliminated on this processor after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d\n",mumps->myid,mumps->id.INFO(23));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      if (mumps->ninfo && mumps->ninfo <= 80) {
        PetscInt i;
        for (i=0; i<mumps->ninfo; i++) {
          ierr = PetscViewerASCIIPrintf(viewer, "  INFO(%d): \n",mumps->info[i]);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d\n",mumps->myid,mumps->id.INFO(mumps->info[i]));CHKERRQ(ierr);
          ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);

      if (!mumps->myid) { /* information from the host */
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(1) (global estimated flops for the elimination after analysis): %g \n",mumps->id.RINFOG(1));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(2) (global estimated flops for the assembly after factorization): %g \n",mumps->id.RINFOG(2));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(3) (global estimated flops for the elimination after factorization): %g \n",mumps->id.RINFOG(3));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  (RINFOG(12) RINFOG(13))*2^INFOG(34) (determinant): (%g,%g)*(2^%d)\n",mumps->id.RINFOG(12),mumps->id.RINFOG(13),mumps->id.INFOG(34));CHKERRQ(ierr);

        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d\n",mumps->id.INFOG(3));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d\n",mumps->id.INFOG(4));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(5) (estimated maximum front size in the complete tree): %d\n",mumps->id.INFOG(5));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(6) (number of nodes in the complete tree): %d\n",mumps->id.INFOG(6));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(7) (ordering option effectively used after analysis): %d\n",mumps->id.INFOG(7));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(8) (structural symmetry in percent of the permuted matrix after analysis): %d\n",mumps->id.INFOG(8));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(9) (total real/complex workspace to store the matrix factors after factorization): %d\n",mumps->id.INFOG(9));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(10) (total integer space store the matrix factors after factorization): %d\n",mumps->id.INFOG(10));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(11) (order of largest frontal matrix after factorization): %d\n",mumps->id.INFOG(11));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(12) (number of off-diagonal pivots): %d\n",mumps->id.INFOG(12));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(13) (number of delayed pivots after factorization): %d\n",mumps->id.INFOG(13));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(14) (number of memory compress after factorization): %d\n",mumps->id.INFOG(14));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(15) (number of steps of iterative refinement after solution): %d\n",mumps->id.INFOG(15));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(16) (estimated size (in MB) of all MUMPS internal data for factorization after analysis: value on the most memory consuming processor): %d\n",mumps->id.INFOG(16));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(17) (estimated size of all MUMPS internal data for factorization after analysis: sum over all processors): %d\n",mumps->id.INFOG(17));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(18) (size of all MUMPS internal data allocated during factorization: value on the most memory consuming processor): %d\n",mumps->id.INFOG(18));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(19) (size of all MUMPS internal data allocated during factorization: sum over all processors): %d\n",mumps->id.INFOG(19));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(20) (estimated number of entries in the factors): %d\n",mumps->id.INFOG(20));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(21) (size in MB of memory effectively used during factorization - value on the most memory consuming processor): %d\n",mumps->id.INFOG(21));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(22) (size in MB of memory effectively used during factorization - sum over all processors): %d\n",mumps->id.INFOG(22));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(23) (after analysis: value of ICNTL(6) effectively used): %d\n",mumps->id.INFOG(23));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(24) (after analysis: value of ICNTL(12) effectively used): %d\n",mumps->id.INFOG(24));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(25) (after factorization: number of pivots modified by static pivoting): %d\n",mumps->id.INFOG(25));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(28) (after factorization: number of null pivots encountered): %d\n",mumps->id.INFOG(28));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(29) (after factorization: effective number of entries in the factors (sum over all processors)): %d\n",mumps->id.INFOG(29));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(30, 31) (after solution: size in Mbytes of memory used during solution phase): %d, %d\n",mumps->id.INFOG(30),mumps->id.INFOG(31));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(32) (after analysis: type of analysis done): %d\n",mumps->id.INFOG(32));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(33) (value used for ICNTL(8)): %d\n",mumps->id.INFOG(33));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(34) (exponent of the determinant if determinant is requested): %d\n",mumps->id.INFOG(34));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(35) (after factorization: number of entries taking into account BLR factor compression - sum over all processors): %d\n",mumps->id.INFOG(35));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(36) (after analysis: estimated size of all MUMPS internal data for running BLR in-core - value on the most memory consuming processor): %d\n",mumps->id.INFOG(36));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(37) (after analysis: estimated size of all MUMPS internal data for running BLR in-core - sum over all processors): %d\n",mumps->id.INFOG(37));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(38) (after analysis: estimated size of all MUMPS internal data for running BLR out-of-core - value on the most memory consuming processor): %d\n",mumps->id.INFOG(38));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(39) (after analysis: estimated size of all MUMPS internal data for running BLR out-of-core - sum over all processors): %d\n",mumps->id.INFOG(39));CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&size);CHKERRQ(ierr);
  if (mumps->petsc_size > 1) {
    PetscBool ls,gs; /* gs is false if any rank other than root has non-empty IS */

    ls   = mumps->myid ? (size ? PETSC_FALSE : PETSC_TRUE) : PETSC_TRUE; /* always true on root; false on others if their size != 0 */
    ierr = MPI_Allreduce(&ls,&gs,1,MPIU_BOOL,MPI_LAND,mumps->petsc_comm);CHKERRMPI(ierr);
    if (!gs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MUMPS distributed parallel Schur complements not yet supported from PETSc\n");
  }

  /* Schur complement matrix */
  ierr = MatDestroy(&F->schur);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,size,size,NULL,&F->schur);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(F->schur,&arr);CHKERRQ(ierr);
  mumps->id.schur      = (MumpsScalar*)arr;
  mumps->id.size_schur = size;
  mumps->id.schur_lld  = size;
  ierr = MatDenseRestoreArrayRead(F->schur,&arr);CHKERRQ(ierr);
  if (mumps->sym == 1) {
    ierr = MatSetOption(F->schur,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* MUMPS expects Fortran style indices */
  ierr = PetscFree(mumps->id.listvar_schur);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&mumps->id.listvar_schur);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
  for (i=0; i<size; i++) {ierr = PetscMUMPSIntCast(idxs[i]+1,&(mumps->id.listvar_schur[i]));CHKERRQ(ierr);}
  ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
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
      ierr = PetscArraycpy(array,mumps->id.schur,mumps->id.size_schur*mumps->id.size_schur);CHKERRQ(ierr);
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
      ierr = PetscArraycpy(array,mumps->id.schur,mumps->id.size_schur*mumps->id.size_schur);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Mat_MUMPS *mumps =(Mat_MUMPS*)F->data;

  PetscFunctionBegin;
  ierr = PetscMUMPSIntCast(ival,&mumps->id.ICNTL(icntl));CHKERRQ(ierr);
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

.seealso: MatGetFactor(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
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

PetscErrorCode MatMumpsGetInverse_MUMPS(Mat F,Mat spRHS)
{
  PetscErrorCode ierr;
  Mat            Bt = NULL,Btseq = NULL;
  PetscBool      flg;
  Mat_MUMPS      *mumps =(Mat_MUMPS*)F->data;
  PetscScalar    *aa;
  PetscInt       spnr,*ia,*ja,M,nrhs;

  PetscFunctionBegin;
  PetscValidPointer(spRHS,2);
  ierr = PetscObjectTypeCompare((PetscObject)spRHS,MATTRANSPOSEMAT,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatTransposeGetMat(spRHS,&Bt);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)spRHS),PETSC_ERR_ARG_WRONG,"Matrix spRHS must be type MATTRANSPOSEMAT matrix");

  ierr = MatMumpsSetIcntl(F,30,1);CHKERRQ(ierr);

  if (mumps->petsc_size > 1) {
    Mat_MPIAIJ *b = (Mat_MPIAIJ*)Bt->data;
    Btseq = b->A;
  } else {
    Btseq = Bt;
  }

  ierr = MatGetSize(spRHS,&M,&nrhs);CHKERRQ(ierr);
  mumps->id.nrhs = nrhs;
  mumps->id.lrhs = M;
  mumps->id.rhs  = NULL;

  if (!mumps->myid) {
    ierr = MatSeqAIJGetArray(Btseq,&aa);CHKERRQ(ierr);
    ierr = MatGetRowIJ(Btseq,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
    ierr = PetscMUMPSIntCSRCast(mumps,spnr,ia,ja,&mumps->id.irhs_ptr,&mumps->id.irhs_sparse,&mumps->id.nz_rhs);CHKERRQ(ierr);
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
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d INFO(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFO(2));

  if (!mumps->myid) {
    ierr = MatSeqAIJRestoreArray(Btseq,&aa);CHKERRQ(ierr);
    ierr = MatRestoreRowIJ(Btseq,1,PETSC_FALSE,PETSC_FALSE,&spnr,(const PetscInt**)&ia,(const PetscInt**)&ja,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot get IJ structure");
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
.      MUMPS Users' Guide

.seealso: MatGetFactor(), MatCreateTranspose()
@*/
PetscErrorCode MatMumpsGetInverse(Mat F,Mat spRHS)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  ierr = PetscUseMethod(F,"MatMumpsGetInverse_C",(Mat,Mat),(F,spRHS));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMumpsGetInverseTranspose_MUMPS(Mat F,Mat spRHST)
{
  PetscErrorCode ierr;
  Mat            spRHS;

  PetscFunctionBegin;
  ierr = MatCreateTranspose(spRHST,&spRHS);CHKERRQ(ierr);
  ierr = MatMumpsGetInverse_MUMPS(F,spRHS);CHKERRQ(ierr);
  ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
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
.      MUMPS Users' Guide

.seealso: MatGetFactor(), MatCreateTranspose(), MatMumpsGetInverse()
@*/
PetscErrorCode MatMumpsGetInverseTranspose(Mat F,Mat spRHST)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidType(F,1);
  if (!F->factortype) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)spRHST,&flg,MATSEQAIJ,MATMPIAIJ,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)spRHST),PETSC_ERR_ARG_WRONG,"Matrix spRHST must be MATAIJ matrix");

  ierr = PetscUseMethod(F,"MatMumpsGetInverseTranspose_C",(Mat,Mat),(F,spRHST));CHKERRQ(ierr);
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfog(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetRinfo(), MatMumpsGetRinfog()
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfog()
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

.seealso: MatGetFactor(), MatMumpsSetIcntl(), MatMumpsGetIcntl(), MatMumpsSetCntl(), MatMumpsGetCntl(), MatMumpsGetInfo(), MatMumpsGetInfog(), MatMumpsGetRinfo()
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
+   1. - Heroux, Michael A., R. Brightwell, and Michael M. Wolf. "Bi-modal MPI and MPI+ threads computing on scalable multicore systems." IJHPCA (Submitted) (2011).
-   2. - Gutierrez, Samuel K., et al. "Accommodating Thread-Level Heterogeneity in Coupled Parallel Applications." Parallel and Distributed Processing Symposium (IPDPS), 2017 IEEE International. IEEE, 2017.

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
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqAIJ;
  PetscMPIInt    size;

  PetscFunctionBegin;
 #if defined(PETSC_USE_COMPLEX)
  if (A->hermitian && !A->symmetric && ftype == MAT_FACTOR_CHOLESKY) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian CHOLESKY Factor is not supported");
 #endif
  /* Create the factorization matrix */
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mumps",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mumps);CHKERRQ(ierr);

  B->ops->view    = MatView_MUMPS;
  B->ops->getinfo = MatGetInfo_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverse_C",MatMumpsGetInverse_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverseTranspose_C",MatMumpsGetInverseTranspose_MUMPS);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpiaij;
    ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
    mumps->sym = 0;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
    B->factortype                  = MAT_FACTOR_CHOLESKY;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqsbaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpisbaij;
    ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

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
  PetscMPIInt    size;

  PetscFunctionBegin;
 #if defined(PETSC_USE_COMPLEX)
  if (A->hermitian && !A->symmetric) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian CHOLESKY Factor is not supported");
 #endif
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mumps",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mumps);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  if (isSeqSBAIJ) {
    mumps->ConvertToTriples = MatConvertToTriples_seqsbaij_seqsbaij;
  } else {
    mumps->ConvertToTriples = MatConvertToTriples_mpisbaij_mpisbaij;
  }

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
  B->ops->view                   = MatView_MUMPS;
  B->ops->getinfo                = MatGetInfo_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverse_C",MatMumpsGetInverse_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverseTranspose_C",MatMumpsGetInverseTranspose_MUMPS);CHKERRQ(ierr);

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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

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
  PetscMPIInt    size;

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
    ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc BAIJ matrices with MUMPS Cholesky, use SBAIJ or AIJ matrix instead\n");

  B->ops->view        = MatView_MUMPS;
  B->ops->getinfo     = MatGetInfo_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverse_C",MatMumpsGetInverse_MUMPS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMumpsGetInverseTranspose_C",MatMumpsGetInverseTranspose_MUMPS);CHKERRQ(ierr);

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

/* MatGetFactor for Seq and MPI SELL matrices */
static PetscErrorCode MatGetFactor_sell_mumps(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqSELL;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSELL,&isSeqSELL);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("mumps",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = PetscNewLog(B,&mumps);CHKERRQ(ierr);

  B->ops->view        = MatView_MUMPS;
  B->ops->getinfo     = MatGetInfo_MUMPS;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_mumps);CHKERRQ(ierr);
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
    if (isSeqSELL) mumps->ConvertToTriples = MatConvertToTriples_seqsell_seqaij;
    else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"To be implemented");
    mumps->sym = 0;
    ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"To be implemented");

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMUMPS,&B->solvertype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization  */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void*)mumps;

  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MUMPS(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIBAIJ,MAT_FACTOR_LU,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATMPIBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATMPISBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQBAIJ,MAT_FACTOR_LU,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_baij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQSBAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_sbaij_mumps);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERMUMPS,MATSEQSELL,MAT_FACTOR_LU,MatGetFactor_sell_mumps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

