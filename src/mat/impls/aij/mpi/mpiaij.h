#ifndef __MPIAIJ_H
#define __MPIAIJ_H

#include <../src/mat/impls/aij/seq/aij.h>

typedef struct { /* used by MatCreateMPIAIJSumSeqAIJ for reusing the merged matrix */
  PetscLayout  rowmap;
  PetscInt   **buf_ri, **buf_rj;
  PetscMPIInt *len_s, *len_r, *id_r; /* array of length of comm->size, store send/recv matrix values */
  PetscMPIInt  nsend, nrecv;
  PetscInt    *bi, *bj;               /* i and j array of the local portion of mpi C (matrix product) - rename to ci, cj! */
  PetscInt    *owners_co, *coi, *coj; /* i and j array of (p->B)^T*A*P - used in the communication */
} Mat_Merge_SeqsToMPI;

typedef struct {                                /* used by MatPtAPXXX_MPIAIJ_MPIAIJ() and MatMatMultXXX_MPIAIJ_MPIAIJ() */
  PetscInt              *startsj_s, *startsj_r; /* used by MatGetBrowsOfAoCols_MPIAIJ */
  PetscScalar           *bufa;                  /* used by MatGetBrowsOfAoCols_MPIAIJ */
  Mat                    P_loc, P_oth;          /* partial B_seq -- intend to replace B_seq */
  PetscInt              *api, *apj;             /* symbolic i and j arrays of the local product A_loc*B_seq */
  PetscScalar           *apv;
  MatReuse               reuse; /* flag to skip MatGetBrowsOfAoCols_MPIAIJ() and MatMPIAIJGetLocalMat() in 1st call of MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  PetscScalar           *apa;   /* tmp array for store a row of A*P used in MatMatMult() */
  Mat                    A_loc; /* used by MatTransposeMatMult(), contains api and apj */
  ISLocalToGlobalMapping ltog;  /* mapping from local column indices to global column indices for A_loc */
  Mat                    Pt;    /* used by MatTransposeMatMult(), Pt = P^T */
  Mat                    Rd, Ro, AP_loc, C_loc, C_oth;
  PetscInt               algType; /* implementation algorithm */
  PetscSF                sf;      /* use it to communicate remote part of C */
  PetscInt              *c_othi, *c_rmti;

  Mat_Merge_SeqsToMPI *merge;
} Mat_APMPI;

typedef struct {
  Mat         A, B; /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt size; /* size of communicator */
  PetscMPIInt rank; /* rank of proc in communicator */

  /* The following variables are used for matrix assembly */
  PetscBool    donotstash;        /* PETSC_TRUE if off processor entries dropped */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  PetscInt     nsends, nrecvs;    /* numbers of sends and receives */
  PetscScalar *svalues, *rvalues; /* sending and receiving data */
  PetscInt     rmax;              /* maximum message length */
#if defined(PETSC_USE_CTABLE)
  PetscHMapI colmap;
#else
  PetscInt *colmap; /* local col number of off-diag col */
#endif
  PetscInt *garray; /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */
  Vec        lvec; /* local vector */
  Vec        diag;
  VecScatter Mvctx;       /* scatter context for vector */
  PetscBool  roworiented; /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */
  PetscInt    *rowindices;   /* column indices for row */
  PetscScalar *rowvalues;    /* nonzero values in row */
  PetscBool    getrowactive; /* indicates MatGetRow(), not restored */

  PetscInt *ld; /* number of entries per row left of diagonal block */

  /* Used by device classes */
  void *spptr;

  /* MatSetValuesCOO() related stuff */
  PetscCount   coo_n;                      /* Number of COOs passed to MatSetPreallocationCOO)() */
  PetscSF      coo_sf;                     /* SF to send/recv remote values in MatSetValuesCOO() */
  PetscCount   Annz, Bnnz;                 /* Number of entries in diagonal A and off-diagonal B */
  PetscCount   Annz2, Bnnz2;               /* Number of unique remote entries belonging to A and B */
  PetscCount   Atot1, Atot2, Btot1, Btot2; /* Total local (tot1) and remote (tot2) entries (which might contain repeats) belonging to A and B */
  PetscCount  *Ajmap1, *Aperm1;            /* Lengths: [Annz+1], [Atot1]. Local entries to diag */
  PetscCount  *Bjmap1, *Bperm1;            /* Lengths: [Bnnz+1], [Btot1]. Local entries to offdiag */
  PetscCount  *Aimap2, *Ajmap2, *Aperm2;   /* Lengths: [Annz2], [Annz2+1], [Atot2]. Remote entries to diag */
  PetscCount  *Bimap2, *Bjmap2, *Bperm2;   /* Lengths: [Bnnz2], [Bnnz2+1], [Btot2]. Remote entries to offdiag */
  PetscCount  *Cperm1;                     /* [sendlen] Permutation to fill MPI send buffer. 'C' for communication */
  PetscScalar *sendbuf, *recvbuf;          /* Buffers for remote values in MatSetValuesCOO() */
  PetscInt     sendlen, recvlen;           /* Lengths (in unit of PetscScalar) of send/recvbuf */

  struct _MatOps cops;
} Mat_MPIAIJ;

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJ(Mat);

PETSC_INTERN PetscErrorCode MatAssemblyEnd_MPIAIJ(Mat, MatAssemblyType);

PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDisAssemble_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDuplicate_MPIAIJ(Mat, MatDuplicateOption, Mat *);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_MPIAIJ(Mat, PetscInt, IS[], PetscInt);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_MPIAIJ_Scalable(Mat, PetscInt, IS[], PetscInt);
PETSC_INTERN PetscErrorCode MatFDColoringCreate_MPIXAIJ(Mat, ISColoring, MatFDColoring);
PETSC_INTERN PetscErrorCode MatFDColoringSetUp_MPIXAIJ(Mat, ISColoring, MatFDColoring);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_MPIAIJ(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_INTERN PetscErrorCode MatCreateSubMatricesMPI_MPIAIJ(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_MPIAIJ_All(Mat, MatCreateSubMatrixOption, MatReuse, Mat *[]);
PETSC_INTERN PetscErrorCode MatView_MPIAIJ(Mat, PetscViewer);

PETSC_INTERN PetscErrorCode MatCreateSubMatrix_MPIAIJ(Mat, IS, IS, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_MPIAIJ_nonscalable(Mat, IS, IS, PetscInt, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_MPIAIJ_SameRowDist(Mat, IS, IS, IS, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_MPIAIJ_SameRowColDist(Mat, IS, IS, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatGetMultiProcBlock_MPIAIJ(Mat, MPI_Comm, MatReuse, Mat *);

PETSC_INTERN PetscErrorCode MatLoad_MPIAIJ(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_MPIAIJ_Binary(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatCreateColmap_MPIAIJ_Private(Mat);

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJBACKEND(Mat);
PETSC_INTERN PetscErrorCode MatProductSymbolic_MPIAIJBACKEND(Mat);
PETSC_INTERN PetscErrorCode MatProductSymbolic_AB_MPIAIJ_MPIAIJ(Mat);

PETSC_INTERN PetscErrorCode MatProductSymbolic_PtAP_MPIAIJ_MPIAIJ(Mat);

PETSC_INTERN PetscErrorCode MatProductSymbolic_RARt_MPIAIJ_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_RARt_MPIAIJ_MPIAIJ(Mat);

PETSC_INTERN PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_seqMPI(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable(Mat, Mat, Mat);

PETSC_INTERN PetscErrorCode MatMatMatMultSymbolic_MPIAIJ_MPIAIJ_MPIAIJ(Mat, Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ(Mat, Mat, Mat, Mat);

PETSC_INTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat, Mat, Mat);

PETSC_INTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce_merged(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged(Mat, Mat, Mat);

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatPtAPSymbolic_AIJ_AIJ_wHYPRE(Mat, Mat, PetscReal, Mat);
#endif
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIDense(Mat, MatType, MatReuse, Mat *);
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_ScaLAPACK(Mat, MatType, MatReuse, Mat *);
#endif

PETSC_INTERN PetscErrorCode MatDestroy_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_MPIAIJ_PtAP(void *);
PETSC_INTERN PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(void *);

PETSC_INTERN PetscErrorCode MatGetBrowsOfAoCols_MPIAIJ(Mat, Mat, MatReuse, PetscInt **, PetscInt **, MatScalar **, Mat *);
PETSC_INTERN PetscErrorCode MatSetValues_MPIAIJ(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode MatSetValues_MPIAIJ_CopyFromCSRFormat(Mat, const PetscInt[], const PetscInt[], const PetscScalar[]);
PETSC_INTERN PetscErrorCode MatSetValues_MPIAIJ_CopyFromCSRFormat_Symbolic(Mat, const PetscInt[], const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetOption_MPIAIJ(Mat, MatOption, PetscBool);

PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_nonscalable(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_nonscalable(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ_matmatmult(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIDense(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatGetSeqNonzeroStructure_MPIAIJ(Mat, Mat *);

PETSC_INTERN PetscErrorCode MatSetFromOptions_MPIAIJ(Mat, PetscOptionItems *);
PETSC_INTERN PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJ(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_REAL_SINGLE) && !defined(PETSC_USE_REAL___FLOAT128) && !defined(PETSC_USE_REAL___FP16)
PETSC_INTERN PetscErrorCode MatLUFactorSymbolic_MPIAIJ_TFS(Mat, IS, IS, const MatFactorInfo *, Mat *);
#endif
PETSC_INTERN PetscErrorCode MatSolve_MPIAIJ(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatILUFactor_MPIAIJ(Mat, IS, IS, const MatFactorInfo *);

PETSC_INTERN PetscErrorCode MatAXPYGetPreallocation_MPIX_private(PetscInt, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, PetscInt *);

extern PetscErrorCode MatGetDiagonalBlock_MPIAIJ(Mat, Mat *);
extern PetscErrorCode MatDiagonalScaleLocal_MPIAIJ(Mat, Vec);

PETSC_INTERN PetscErrorCode MatGetSeqMats_MPIAIJ(Mat, Mat *, Mat *);
PETSC_INTERN PetscErrorCode MatSetSeqMats_MPIAIJ(Mat, IS, IS, IS, MatStructure, Mat, Mat);

PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_MPIAIJ(Mat, PetscCount, PetscInt[], PetscInt[]);
PETSC_INTERN PetscErrorCode MatResetPreallocationCOO_MPIAIJ(Mat);

/* compute apa = A[i,:]*P = Ad[i,:]*P_loc + Ao*[i,:]*P_oth using sparse axpy */
#define AProw_scalable(i, ad, ao, p_loc, p_oth, api, apj, apa) \
  { \
    PetscInt     _anz, _pnz, _j, _k, *_ai, *_aj, _row, *_pi, *_pj, _nextp, *_apJ; \
    PetscScalar *_aa, _valtmp, *_pa; \
    _apJ = apj + api[i]; \
    /* diagonal portion of A */ \
    _ai  = ad->i; \
    _anz = _ai[i + 1] - _ai[i]; \
    _aj  = ad->j + _ai[i]; \
    _aa  = ad->a + _ai[i]; \
    for (_j = 0; _j < _anz; _j++) { \
      _row = _aj[_j]; \
      _pi  = p_loc->i; \
      _pnz = _pi[_row + 1] - _pi[_row]; \
      _pj  = p_loc->j + _pi[_row]; \
      _pa  = p_loc->a + _pi[_row]; \
      /* perform sparse axpy */ \
      _valtmp = _aa[_j]; \
      _nextp  = 0; \
      for (_k = 0; _nextp < _pnz; _k++) { \
        if (_apJ[_k] == _pj[_nextp]) { /* column of AP == column of P */ \
          apa[_k] += _valtmp * _pa[_nextp++]; \
        } \
      } \
      (void)PetscLogFlops(2.0 * _pnz); \
    } \
    /* off-diagonal portion of A */ \
    if (p_oth) { \
      _ai  = ao->i; \
      _anz = _ai[i + 1] - _ai[i]; \
      _aj  = ao->j + _ai[i]; \
      _aa  = ao->a + _ai[i]; \
      for (_j = 0; _j < _anz; _j++) { \
        _row = _aj[_j]; \
        _pi  = p_oth->i; \
        _pnz = _pi[_row + 1] - _pi[_row]; \
        _pj  = p_oth->j + _pi[_row]; \
        _pa  = p_oth->a + _pi[_row]; \
        /* perform sparse axpy */ \
        _valtmp = _aa[_j]; \
        _nextp  = 0; \
        for (_k = 0; _nextp < _pnz; _k++) { \
          if (_apJ[_k] == _pj[_nextp]) { /* column of AP == column of P */ \
            apa[_k] += _valtmp * _pa[_nextp++]; \
          } \
        } \
        (void)PetscLogFlops(2.0 * _pnz); \
      } \
    } \
  }

#define AProw_nonscalable(i, ad, ao, p_loc, p_oth, apa) \
  { \
    PetscInt     _anz, _pnz, _j, _k, *_ai, *_aj, _row, *_pi, *_pj; \
    PetscScalar *_aa, _valtmp, *_pa; \
    /* diagonal portion of A */ \
    _ai  = ad->i; \
    _anz = _ai[i + 1] - _ai[i]; \
    _aj  = ad->j + _ai[i]; \
    _aa  = ad->a + _ai[i]; \
    for (_j = 0; _j < _anz; _j++) { \
      _row = _aj[_j]; \
      _pi  = p_loc->i; \
      _pnz = _pi[_row + 1] - _pi[_row]; \
      _pj  = p_loc->j + _pi[_row]; \
      _pa  = p_loc->a + _pi[_row]; \
      /* perform dense axpy */ \
      _valtmp = _aa[_j]; \
      for (_k = 0; _k < _pnz; _k++) apa[_pj[_k]] += _valtmp * _pa[_k]; \
      (void)PetscLogFlops(2.0 * _pnz); \
    } \
    /* off-diagonal portion of A */ \
    if (p_oth) { \
      _ai  = ao->i; \
      _anz = _ai[i + 1] - _ai[i]; \
      _aj  = ao->j + _ai[i]; \
      _aa  = ao->a + _ai[i]; \
      for (_j = 0; _j < _anz; _j++) { \
        _row = _aj[_j]; \
        _pi  = p_oth->i; \
        _pnz = _pi[_row + 1] - _pi[_row]; \
        _pj  = p_oth->j + _pi[_row]; \
        _pa  = p_oth->a + _pi[_row]; \
        /* perform dense axpy */ \
        _valtmp = _aa[_j]; \
        for (_k = 0; _k < _pnz; _k++) apa[_pj[_k]] += _valtmp * _pa[_k]; \
        (void)PetscLogFlops(2.0 * _pnz); \
      } \
    } \
  }

#endif
