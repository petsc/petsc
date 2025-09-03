/*
    Creates hypre ijmatrix from PETSc matrix
*/

#include <petscpkg_version.h>
#include <petsc/private/petschypre.h>
#include <petscmathypre.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/deviceimpl.h>
#include <../src/mat/impls/hypre/mhypre.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/vec/vec/impls/hypre/vhyp.h>
#include <HYPRE.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_sstruct_ls.h>

#if PETSC_PKG_HYPRE_VERSION_LT(2, 18, 0)
  #define hypre_ParCSRMatrixClone(A, B) hypre_ParCSRMatrixCompleteClone(A)
#endif

#if PETSC_PKG_HYPRE_VERSION_GE(2, 15, 0)
  #define HYPRE_AssumedPartitionCheck() 1
#endif

static PetscErrorCode MatHYPRE_CreateFromMat(Mat, Mat_HYPRE *);
static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat, Mat, HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixCopyIJ_MPIAIJ(Mat, HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixCopyIJ_SeqAIJ(Mat, HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat, HYPRE_Complex, Vec, HYPRE_Complex, Vec, PetscBool);
static PetscErrorCode MatSetValues_HYPRE(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode ins);

static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat A_d, Mat A_o, HYPRE_IJMatrix ij)
{
  PetscInt        i, n_d, n_o;
  const PetscInt *ia_d, *ia_o;
  PetscBool       done_d = PETSC_FALSE, done_o = PETSC_FALSE;
  HYPRE_Int      *nnz_d = NULL, *nnz_o = NULL;

  PetscFunctionBegin;
  if (A_d) { /* determine number of nonzero entries in local diagonal part */
    PetscCall(MatGetRowIJ(A_d, 0, PETSC_FALSE, PETSC_FALSE, &n_d, &ia_d, NULL, &done_d));
    if (done_d) {
      PetscCall(PetscMalloc1(n_d, &nnz_d));
      for (i = 0; i < n_d; i++) nnz_d[i] = ia_d[i + 1] - ia_d[i];
    }
    PetscCall(MatRestoreRowIJ(A_d, 0, PETSC_FALSE, PETSC_FALSE, NULL, &ia_d, NULL, &done_d));
  }
  if (A_o) { /* determine number of nonzero entries in local off-diagonal part */
    PetscCall(MatGetRowIJ(A_o, 0, PETSC_FALSE, PETSC_FALSE, &n_o, &ia_o, NULL, &done_o));
    if (done_o) {
      PetscCall(PetscMalloc1(n_o, &nnz_o));
      for (i = 0; i < n_o; i++) nnz_o[i] = ia_o[i + 1] - ia_o[i];
    }
    PetscCall(MatRestoreRowIJ(A_o, 0, PETSC_FALSE, PETSC_FALSE, &n_o, &ia_o, NULL, &done_o));
  }
  if (done_d) {    /* set number of nonzeros in HYPRE IJ matrix */
    if (!done_o) { /* only diagonal part */
      PetscCall(PetscCalloc1(n_d, &nnz_o));
    }
#if PETSC_PKG_HYPRE_VERSION_GE(2, 16, 0)
    { /* If we don't do this, the columns of the matrix will be all zeros! */
      hypre_AuxParCSRMatrix *aux_matrix;
      aux_matrix = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(ij);
      hypre_AuxParCSRMatrixDestroy(aux_matrix);
      hypre_IJMatrixTranslator(ij) = NULL;
      PetscCallExternal(HYPRE_IJMatrixSetDiagOffdSizes, ij, nnz_d, nnz_o);
      /* it seems they partially fixed it in 2.19.0 */
  #if PETSC_PKG_HYPRE_VERSION_LT(2, 19, 0)
      aux_matrix                               = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(ij);
      hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 1;
  #endif
    }
#else
    PetscCallExternal(HYPRE_IJMatrixSetDiagOffdSizes, ij, nnz_d, nnz_o);
#endif
    PetscCall(PetscFree(nnz_d));
    PetscCall(PetscFree(nnz_o));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRE_CreateFromMat(Mat A, Mat_HYPRE *hA)
{
  PetscInt rstart, rend, cstart, cend;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  rstart = A->rmap->rstart;
  rend   = A->rmap->rend;
  cstart = A->cmap->rstart;
  cend   = A->cmap->rend;
  PetscHYPREInitialize();
  if (hA->ij) {
    if (!hA->inner_free) hypre_IJMatrixObject(hA->ij) = NULL;
    PetscCallExternal(HYPRE_IJMatrixDestroy, hA->ij);
  }
  PetscCallExternal(HYPRE_IJMatrixCreate, hA->comm, rstart, rend - 1, cstart, cend - 1, &hA->ij);
  PetscCallExternal(HYPRE_IJMatrixSetObjectType, hA->ij, HYPRE_PARCSR);
  {
    PetscBool       same;
    Mat             A_d, A_o;
    const PetscInt *colmap;
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &same));
    if (same) {
      PetscCall(MatMPIAIJGetSeqAIJ(A, &A_d, &A_o, &colmap));
      PetscCall(MatHYPRE_IJMatrixPreallocate(A_d, A_o, hA->ij));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIBAIJ, &same));
    if (same) {
      PetscCall(MatMPIBAIJGetSeqBAIJ(A, &A_d, &A_o, &colmap));
      PetscCall(MatHYPRE_IJMatrixPreallocate(A_d, A_o, hA->ij));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &same));
    if (same) {
      PetscCall(MatHYPRE_IJMatrixPreallocate(A, NULL, hA->ij));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQBAIJ, &same));
    if (same) {
      PetscCall(MatHYPRE_IJMatrixPreallocate(A, NULL, hA->ij));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRE_IJMatrixCopyIJ(Mat A, HYPRE_IJMatrix ij)
{
  PetscBool flg;

  PetscFunctionBegin;
#if PETSC_PKG_HYPRE_VERSION_LT(2, 19, 0)
  PetscCallExternal(HYPRE_IJMatrixInitialize, ij);
#else
  PetscCallExternal(HYPRE_IJMatrixInitialize_v2, ij, HYPRE_MEMORY_HOST);
#endif
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &flg));
  if (flg) {
    PetscCall(MatHYPRE_IJMatrixCopyIJ_MPIAIJ(A, ij));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &flg));
  if (flg) {
    PetscCall(MatHYPRE_IJMatrixCopyIJ_SeqAIJ(A, ij));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(PETSC_FALSE, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "No support for matrix type %s", ((PetscObject)A)->type_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRE_IJMatrixCopyIJ_SeqAIJ(Mat A, HYPRE_IJMatrix ij)
{
  Mat_SeqAIJ            *pdiag = (Mat_SeqAIJ *)A->data;
  HYPRE_Int              type;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag;
  PetscBool              sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));

  PetscFunctionBegin;
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObject, ij, (void **)&par_matrix);
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  /*
       this is the Hack part where we monkey directly with the hypre datastructures
  */
  if (sameint) {
    PetscCall(PetscArraycpy(hdiag->i, pdiag->i, A->rmap->n + 1));
    PetscCall(PetscArraycpy(hdiag->j, pdiag->j, pdiag->nz));
  } else {
    PetscInt i;

    for (i = 0; i < A->rmap->n + 1; i++) hdiag->i[i] = (HYPRE_Int)pdiag->i[i];
    for (i = 0; i < pdiag->nz; i++) hdiag->j[i] = (HYPRE_Int)pdiag->j[i];
  }

  aux_matrix                               = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(ij);
  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRE_IJMatrixCopyIJ_MPIAIJ(Mat A, HYPRE_IJMatrix ij)
{
  Mat_MPIAIJ            *pA = (Mat_MPIAIJ *)A->data;
  Mat_SeqAIJ            *pdiag, *poffd;
  PetscInt               i, *garray = pA->garray, *jj, cstart, *pjj;
  HYPRE_Int             *hjj, type;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag, *hoffd;
  PetscBool              sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));

  PetscFunctionBegin;
  pdiag = (Mat_SeqAIJ *)pA->A->data;
  poffd = (Mat_SeqAIJ *)pA->B->data;
  /* cstart is only valid for square MPIAIJ laid out in the usual way */
  PetscCall(MatGetOwnershipRange(A, &cstart, NULL));

  PetscCallExternal(HYPRE_IJMatrixGetObjectType, ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObject, ij, (void **)&par_matrix);
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  hoffd = hypre_ParCSRMatrixOffd(par_matrix);

  if (sameint) {
    PetscCall(PetscArraycpy(hdiag->i, pdiag->i, pA->A->rmap->n + 1));
  } else {
    for (i = 0; i < pA->A->rmap->n + 1; i++) hdiag->i[i] = (HYPRE_Int)pdiag->i[i];
  }

  hjj = hdiag->j;
  pjj = pdiag->j;
#if PETSC_PKG_HYPRE_VERSION_GE(2, 16, 0)
  for (i = 0; i < pdiag->nz; i++) hjj[i] = pjj[i];
#else
  for (i = 0; i < pdiag->nz; i++) hjj[i] = cstart + pjj[i];
#endif
  if (sameint) {
    PetscCall(PetscArraycpy(hoffd->i, poffd->i, pA->A->rmap->n + 1));
  } else {
    for (i = 0; i < pA->A->rmap->n + 1; i++) hoffd->i[i] = (HYPRE_Int)poffd->i[i];
  }

  jj = (PetscInt *)hoffd->j;
#if PETSC_PKG_HYPRE_VERSION_GE(2, 16, 0)
  PetscCallExternal(hypre_CSRMatrixBigInitialize, hoffd);
  jj = (PetscInt *)hoffd->big_j;
#endif
  pjj = poffd->j;
  for (i = 0; i < poffd->nz; i++) jj[i] = garray[pjj[i]];

  aux_matrix                               = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(ij);
  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_HYPRE_IS(Mat A, MatType mtype, MatReuse reuse, Mat *B)
{
  Mat_HYPRE             *mhA = (Mat_HYPRE *)A->data;
  Mat                    lA;
  ISLocalToGlobalMapping rl2g, cl2g;
  IS                     is;
  hypre_ParCSRMatrix    *hA;
  hypre_CSRMatrix       *hdiag, *hoffd;
  MPI_Comm               comm;
  HYPRE_Complex         *hdd, *hod, *aa;
  PetscScalar           *data;
  HYPRE_BigInt          *col_map_offd;
  HYPRE_Int             *hdi, *hdj, *hoi, *hoj;
  PetscInt              *ii, *jj, *iptr, *jptr;
  PetscInt               cum, dr, dc, oc, str, stc, nnz, i, jd, jo, M, N;
  HYPRE_Int              type;
  MatType                lmattype   = NULL;
  PetscBool              freeparcsr = PETSC_FALSE;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A);
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, mhA->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, comm, PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObject, mhA->ij, (void **)&hA);
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_DEVICE == hypre_IJMatrixMemoryLocation(mhA->ij)) {
    /* Support by copying back on the host and copy to GPU
       Kind of inefficient, but this is the best we can do now */
  #if defined(HYPRE_USING_HIP)
    lmattype = MATSEQAIJHIPSPARSE;
  #elif defined(HYPRE_USING_CUDA)
    lmattype = MATSEQAIJCUSPARSE;
  #endif
    hA         = hypre_ParCSRMatrixClone_v2(hA, 1, HYPRE_MEMORY_HOST);
    freeparcsr = PETSC_TRUE;
  }
#endif
  M     = hypre_ParCSRMatrixGlobalNumRows(hA);
  N     = hypre_ParCSRMatrixGlobalNumCols(hA);
  str   = hypre_ParCSRMatrixFirstRowIndex(hA);
  stc   = hypre_ParCSRMatrixFirstColDiag(hA);
  hdiag = hypre_ParCSRMatrixDiag(hA);
  hoffd = hypre_ParCSRMatrixOffd(hA);
  dr    = hypre_CSRMatrixNumRows(hdiag);
  dc    = hypre_CSRMatrixNumCols(hdiag);
  nnz   = hypre_CSRMatrixNumNonzeros(hdiag);
  hdi   = hypre_CSRMatrixI(hdiag);
  hdj   = hypre_CSRMatrixJ(hdiag);
  hdd   = hypre_CSRMatrixData(hdiag);
  oc    = hypre_CSRMatrixNumCols(hoffd);
  nnz += hypre_CSRMatrixNumNonzeros(hoffd);
  hoi = hypre_CSRMatrixI(hoffd);
  hoj = hypre_CSRMatrixJ(hoffd);
  hod = hypre_CSRMatrixData(hoffd);
  if (reuse != MAT_REUSE_MATRIX) {
    PetscInt *aux;

    /* generate l2g maps for rows and cols */
    PetscCall(ISCreateStride(comm, dr, str, 1, &is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is, &rl2g));
    PetscCall(ISDestroy(&is));
    col_map_offd = hypre_ParCSRMatrixColMapOffd(hA);
    PetscCall(PetscMalloc1(dc + oc, &aux));
    for (i = 0; i < dc; i++) aux[i] = i + stc;
    for (i = 0; i < oc; i++) aux[i + dc] = col_map_offd[i];
    PetscCall(ISCreateGeneral(comm, dc + oc, aux, PETSC_OWN_POINTER, &is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is, &cl2g));
    PetscCall(ISDestroy(&is));
    /* create MATIS object */
    PetscCall(MatCreate(comm, B));
    PetscCall(MatSetSizes(*B, dr, dc, M, N));
    PetscCall(MatSetType(*B, MATIS));
    PetscCall(MatSetLocalToGlobalMapping(*B, rl2g, cl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));

    /* allocate CSR for local matrix */
    PetscCall(PetscMalloc1(dr + 1, &iptr));
    PetscCall(PetscMalloc1(nnz, &jptr));
    PetscCall(PetscMalloc1(nnz, &data));
  } else {
    PetscInt  nr;
    PetscBool done;
    PetscCall(MatISGetLocalMat(*B, &lA));
    PetscCall(MatGetRowIJ(lA, 0, PETSC_FALSE, PETSC_FALSE, &nr, (const PetscInt **)&iptr, (const PetscInt **)&jptr, &done));
    PetscCheck(nr == dr, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot reuse mat: invalid number of rows in local mat! %" PetscInt_FMT " != %" PetscInt_FMT, nr, dr);
    PetscCheck(iptr[nr] >= nnz, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot reuse mat: invalid number of nonzeros in local mat! reuse %" PetscInt_FMT " requested %" PetscInt_FMT, iptr[nr], nnz);
    PetscCall(MatSeqAIJGetArrayWrite(lA, &data));
  }
  /* merge local matrices */
  ii  = iptr;
  jj  = jptr;
  aa  = (HYPRE_Complex *)data; /* this cast fixes the clang error when doing the assignments below: implicit conversion from 'HYPRE_Complex' (aka '_Complex double') to 'double' is not permitted in C++ */
  *ii = *(hdi++) + *(hoi++);
  for (jd = 0, jo = 0, cum = 0; *ii < nnz; cum++) {
    PetscScalar *aold = (PetscScalar *)aa;
    PetscInt    *jold = jj, nc = jd + jo;
    for (; jd < *hdi; jd++) {
      *jj++ = *hdj++;
      *aa++ = *hdd++;
    }
    for (; jo < *hoi; jo++) {
      *jj++ = *hoj++ + dc;
      *aa++ = *hod++;
    }
    *(++ii) = *(hdi++) + *(hoi++);
    PetscCall(PetscSortIntWithScalarArray(jd + jo - nc, jold, aold));
  }
  for (; cum < dr; cum++) *(++ii) = nnz;
  if (reuse != MAT_REUSE_MATRIX) {
    Mat_SeqAIJ *a;

    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, dr, dc + oc, iptr, jptr, data, &lA));
    /* hack SeqAIJ */
    a          = (Mat_SeqAIJ *)lA->data;
    a->free_a  = PETSC_TRUE;
    a->free_ij = PETSC_TRUE;
    if (lmattype) PetscCall(MatConvert(lA, lmattype, MAT_INPLACE_MATRIX, &lA));
    PetscCall(MatISSetLocalMat(*B, lA));
    PetscCall(MatDestroy(&lA));
  } else {
    PetscCall(MatSeqAIJRestoreArrayWrite(lA, &data));
  }
  PetscCall(MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, B));
  if (freeparcsr) PetscCallExternal(hypre_ParCSRMatrixDestroy, hA);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRE_DestroyCOOMat(Mat mat)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)mat->data;

  PetscFunctionBegin;
  if (hA->cooMat) { /* If cooMat is present we need to destroy the column indices */
    PetscCall(MatDestroy(&hA->cooMat));
    if (hA->cooMatAttached) {
      hypre_CSRMatrix     *csr;
      hypre_ParCSRMatrix  *parcsr;
      HYPRE_MemoryLocation mem;

      PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)&parcsr);
      csr = hypre_ParCSRMatrixDiag(parcsr);
      if (csr) {
        mem = hypre_CSRMatrixMemoryLocation(csr);
        PetscStackCallExternalVoid("hypre_TFree", hypre_TFree(hypre_CSRMatrixJ(csr), mem));
        PetscStackCallExternalVoid("hypre_TFree", hypre_TFree(hypre_CSRMatrixBigJ(csr), mem));
      }
      csr = hypre_ParCSRMatrixOffd(parcsr);
      if (csr) {
        mem = hypre_CSRMatrixMemoryLocation(csr);
        PetscStackCallExternalVoid("hypre_TFree", hypre_TFree(hypre_CSRMatrixJ(csr), mem));
        PetscStackCallExternalVoid("hypre_TFree", hypre_TFree(hypre_CSRMatrixBigJ(csr), mem));
      }
    }
  }
  hA->cooMatAttached = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRE_CreateCOOMat(Mat mat)
{
  MPI_Comm    comm;
  PetscMPIInt size;
  PetscLayout rmap, cmap;
  Mat_HYPRE  *hmat    = (Mat_HYPRE *)mat->data;
  MatType     matType = MATAIJ; /* default type of cooMat */

  PetscFunctionBegin;
  /* Build an agent matrix cooMat with AIJ format
     It has the same sparsity pattern as mat, and also shares the data array with mat. We use cooMat to do the COO work.
   */
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));
  PetscCall(MatGetLayouts(mat, &rmap, &cmap));

#if defined(PETSC_HAVE_HYPRE_DEVICE)
  if (!mat->boundtocpu) { /* mat will be on device, so will cooMat */
  #if defined(HYPRE_USING_HIP)
    matType = MATAIJHIPSPARSE;
  #elif defined(HYPRE_USING_CUDA)
    matType = MATAIJCUSPARSE;
  #elif defined(HYPRE_USING_SYCL) && defined(PETSC_HAVE_KOKKOS_KERNELS)
    matType = MATAIJKOKKOS;
  #else
    SETERRQ(comm, PETSC_ERR_SUP, "No HYPRE device available. Suggest re-installing with Kokkos Kernels");
  #endif
  }
#endif

  /* Do COO preallocation through cooMat */
  PetscCall(MatHYPRE_DestroyCOOMat(mat));
  PetscCall(MatCreate(comm, &hmat->cooMat));
  PetscCall(MatSetType(hmat->cooMat, matType));
  PetscCall(MatSetLayouts(hmat->cooMat, rmap, cmap));

  /* allocate local matrices if needed */
  PetscCall(MatMPIAIJSetPreallocation(hmat->cooMat, 0, NULL, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Attach cooMat data array to hypre matrix.
   When AIJCUPMSPARSE will support raw device pointers and not THRUSTARRAY
   we should swap the arrays: i.e., attach hypre matrix array to cooMat
   This is because hypre should be in charge of handling the memory,
   cooMat is only a way to reuse PETSc COO code.
   attaching the memory will then be done at MatSetValuesCOO time and it will dynamically
   support hypre matrix migrating to host.
*/
static PetscErrorCode MatHYPRE_AttachCOOMat(Mat mat)
{
  Mat_HYPRE           *hmat = (Mat_HYPRE *)mat->data;
  hypre_CSRMatrix     *diag, *offd;
  hypre_ParCSRMatrix  *parCSR;
  HYPRE_MemoryLocation hmem = HYPRE_MEMORY_HOST;
  PetscMemType         pmem;
  Mat                  A, B;
  PetscScalar         *a;
  PetscMPIInt          size;
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscCheck(hmat->cooMat, PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "HYPRE COO delegate matrix has not been created yet");
  if (hmat->cooMatAttached) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(hmat->cooMat->preallocated, PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "HYPRE COO delegate matrix is not preallocated");
  PetscCall(PetscObjectSetName((PetscObject)hmat->cooMat, "_internal_COO_mat_for_hypre"));
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* Alias cooMat's data array to IJMatrix's */
  PetscCallExternal(HYPRE_IJMatrixGetObject, hmat->ij, (void **)&parCSR);
  diag = hypre_ParCSRMatrixDiag(parCSR);
  offd = hypre_ParCSRMatrixOffd(parCSR);

  A = (size == 1) ? hmat->cooMat : ((Mat_MPIAIJ *)hmat->cooMat->data)->A;
  B = (size == 1) ? NULL : ((Mat_MPIAIJ *)hmat->cooMat->data)->B;

  PetscCall(PetscObjectSetName((PetscObject)A, "_internal_COO_mat_for_hypre"));
  hmem = hypre_CSRMatrixMemoryLocation(diag);
  PetscCall(MatSeqAIJGetCSRAndMemType(A, NULL, NULL, &a, &pmem));
  PetscAssert((PetscMemTypeHost(pmem) && hmem == HYPRE_MEMORY_HOST) || (PetscMemTypeDevice(pmem) && hmem == HYPRE_MEMORY_DEVICE), comm, PETSC_ERR_PLIB, "PETSc and hypre's memory types mismatch");
  PetscStackCallExternalVoid("hypre_TFree", hypre_TFree(hypre_CSRMatrixData(diag), hmem));
  hypre_CSRMatrixData(diag)     = (HYPRE_Complex *)a;
  hypre_CSRMatrixOwnsData(diag) = 0; /* Take ownership of (j,a) away from hypre. As a result, we need to free them on our own */

  if (B) {
    hmem = hypre_CSRMatrixMemoryLocation(offd);
    PetscCall(MatSeqAIJGetCSRAndMemType(B, NULL, NULL, &a, &pmem));
    PetscAssert((PetscMemTypeHost(pmem) && hmem == HYPRE_MEMORY_HOST) || (PetscMemTypeDevice(pmem) && hmem == HYPRE_MEMORY_DEVICE), comm, PETSC_ERR_PLIB, "PETSc and hypre's memory types mismatch");
    PetscStackCallExternalVoid("hypre_TFree", hypre_TFree(hypre_CSRMatrixData(offd), hmem));
    hypre_CSRMatrixData(offd)     = (HYPRE_Complex *)a;
    hypre_CSRMatrixOwnsData(offd) = 0;
  }
  hmat->cooMatAttached = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Build COO's coordinate list i[], j[] based on CSR's i[], j[] arrays and the number of local rows 'n'
static PetscErrorCode CSRtoCOO_Private(PetscInt n, const PetscInt ii[], const PetscInt jj[], PetscCount *ncoo, PetscInt **coo_i, PetscInt **coo_j)
{
  PetscInt *cooi, *cooj;

  PetscFunctionBegin;
  *ncoo = ii[n];
  PetscCall(PetscMalloc2(*ncoo, &cooi, *ncoo, &cooj));
  for (PetscInt i = 0; i < n; i++) {
    for (PetscInt j = ii[i]; j < ii[i + 1]; j++) cooi[j] = i;
  }
  PetscCall(PetscArraycpy(cooj, jj, *ncoo));
  *coo_i = cooi;
  *coo_j = cooj;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Similar to CSRtoCOO_Private, but the CSR's i[], j[] are of type HYPRE_Int
static PetscErrorCode CSRtoCOO_HYPRE_Int_Private(PetscInt n, const HYPRE_Int ii[], const HYPRE_Int jj[], PetscCount *ncoo, PetscInt **coo_i, PetscInt **coo_j)
{
  PetscInt *cooi, *cooj;

  PetscFunctionBegin;
  *ncoo = ii[n];
  PetscCall(PetscMalloc2(*ncoo, &cooi, *ncoo, &cooj));
  for (PetscInt i = 0; i < n; i++) {
    for (HYPRE_Int j = ii[i]; j < ii[i + 1]; j++) cooi[j] = i;
  }
  for (PetscCount i = 0; i < *ncoo; i++) cooj[i] = jj[i];
  *coo_i = cooi;
  *coo_j = cooj;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Build a COO data structure for the seqaij matrix, as if the nonzeros are laid out in the same order as in the CSR
static PetscErrorCode MatSeqAIJGetCOO_Private(Mat A, PetscCount *ncoo, PetscInt **coo_i, PetscInt **coo_j)
{
  PetscInt        n;
  const PetscInt *ii, *jj;
  PetscBool       done;

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &n, &ii, &jj, &done));
  PetscCheck(done, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Failure for MatGetRowIJ");
  PetscCall(CSRtoCOO_Private(n, ii, jj, ncoo, coo_i, coo_j));
  PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &n, &ii, &jj, &done));
  PetscCheck(done, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Failure for MatRestoreRowIJ");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Build a COO data structure for the hypreCSRMatrix, as if the nonzeros are laid out in the same order as in the hypreCSRMatrix
static PetscErrorCode hypreCSRMatrixGetCOO_Private(hypre_CSRMatrix *A, PetscCount *ncoo, PetscInt **coo_i, PetscInt **coo_j)
{
  PetscInt             n = hypre_CSRMatrixNumRows(A);
  HYPRE_Int           *ii, *jj;
  HYPRE_MemoryLocation mem = HYPRE_MEMORY_HOST;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  mem = hypre_CSRMatrixMemoryLocation(A);
  if (mem != HYPRE_MEMORY_HOST) {
    PetscCount nnz = hypre_CSRMatrixNumNonzeros(A);
    PetscCall(PetscMalloc2(n + 1, &ii, nnz, &jj));
    hypre_TMemcpy(ii, hypre_CSRMatrixI(A), HYPRE_Int, n + 1, HYPRE_MEMORY_HOST, mem);
    hypre_TMemcpy(jj, hypre_CSRMatrixJ(A), HYPRE_Int, nnz, HYPRE_MEMORY_HOST, mem);
  } else {
#else
  {
#endif
    ii = hypre_CSRMatrixI(A);
    jj = hypre_CSRMatrixJ(A);
  }
  PetscCall(CSRtoCOO_HYPRE_Int_Private(n, ii, jj, ncoo, coo_i, coo_j));
  if (mem != HYPRE_MEMORY_HOST) PetscCall(PetscFree2(ii, jj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOOFromCSRMatrix_Private(Mat A, hypre_CSRMatrix *H)
{
  PetscBool            iscpu = PETSC_TRUE;
  PetscScalar         *a;
  HYPRE_MemoryLocation mem = HYPRE_MEMORY_HOST;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  mem = hypre_CSRMatrixMemoryLocation(H);
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &iscpu));
#endif
  if (iscpu && mem != HYPRE_MEMORY_HOST) {
    PetscCount nnz = hypre_CSRMatrixNumNonzeros(H);
    PetscCall(PetscMalloc1(nnz, &a));
    hypre_TMemcpy(a, hypre_CSRMatrixData(H), PetscScalar, nnz, HYPRE_MEMORY_HOST, mem);
  } else {
    a = (PetscScalar *)hypre_CSRMatrixData(H);
  }
  PetscCall(MatSetValuesCOO(A, a, INSERT_VALUES));
  if (iscpu && mem != HYPRE_MEMORY_HOST) PetscCall(PetscFree(a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat A, MatType type, MatReuse reuse, Mat *B)
{
  MPI_Comm     comm = PetscObjectComm((PetscObject)A);
  Mat          M = NULL, dH = NULL, oH = NULL, dA = NULL, oA = NULL;
  PetscBool    ismpiaij, issbaij, isbaij, boundtocpu = PETSC_TRUE;
  Mat_HYPRE   *hA;
  PetscMemType memtype = PETSC_MEMTYPE_HOST;

  PetscFunctionBegin;
  if (PetscDefined(HAVE_HYPRE_DEVICE)) {
    PetscCall(MatGetCurrentMemType(A, &memtype));
    PetscHYPREInitialize();
    boundtocpu = PetscMemTypeHost(memtype) ? PETSC_TRUE : PETSC_FALSE;
    PetscCallExternal(HYPRE_SetMemoryLocation, boundtocpu ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE);
  }

  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &issbaij, MATSEQSBAIJ, MATMPIBAIJ, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &isbaij, MATSEQBAIJ, MATMPIBAIJ, ""));
  if (isbaij || issbaij) { /* handle BAIJ and SBAIJ */
    PetscBool ismpi;
    MatType   newtype;

    PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &ismpi, MATMPISBAIJ, MATMPIBAIJ, ""));
    newtype = ismpi ? MATMPIAIJ : MATSEQAIJ;
    if (reuse == MAT_REUSE_MATRIX) {
      PetscCall(MatConvert(*B, newtype, MAT_INPLACE_MATRIX, B));
      PetscCall(MatConvert(A, newtype, MAT_REUSE_MATRIX, B));
      PetscCall(MatConvert(*B, MATHYPRE, MAT_INPLACE_MATRIX, B));
    } else if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(MatConvert(A, newtype, MAT_INITIAL_MATRIX, B));
      PetscCall(MatConvert(*B, MATHYPRE, MAT_INPLACE_MATRIX, B));
    } else {
      PetscCall(MatConvert(A, newtype, MAT_INPLACE_MATRIX, &A));
      PetscCall(MatConvert(A, MATHYPRE, MAT_INPLACE_MATRIX, &A));
    }
#if defined(PETSC_HAVE_DEVICE)
    (*B)->boundtocpu = boundtocpu;
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  dA = A;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &ismpiaij));
  if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(A, &dA, &oA, NULL));

  if (reuse != MAT_REUSE_MATRIX) {
    PetscCount coo_n;
    PetscInt  *coo_i, *coo_j;

    PetscCall(MatCreate(comm, &M));
    PetscCall(MatSetType(M, MATHYPRE));
    PetscCall(MatSetSizes(M, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
    PetscCall(MatSetOption(M, MAT_SORTED_FULL, PETSC_TRUE));
    PetscCall(MatSetOption(M, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

    hA = (Mat_HYPRE *)M->data;
    PetscCall(MatHYPRE_CreateFromMat(A, hA));
    PetscCall(MatHYPRE_IJMatrixCopyIJ(A, hA->ij));

    PetscCall(MatHYPRE_CreateCOOMat(M));

    dH = hA->cooMat;
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)hA->cooMat, MATMPIAIJ, &ismpiaij));
    if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(hA->cooMat, &dH, &oH, NULL));

    PetscCall(PetscObjectSetName((PetscObject)dH, "_internal_COO_mat_for_hypre"));
    PetscCall(MatSeqAIJGetCOO_Private(dA, &coo_n, &coo_i, &coo_j));
    PetscCall(MatSetPreallocationCOO(dH, coo_n, coo_i, coo_j));
    PetscCall(PetscFree2(coo_i, coo_j));
    if (oH) {
      PetscCall(PetscLayoutDestroy(&oH->cmap));
      PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)oH), oA->cmap->n, oA->cmap->n, 1, &oH->cmap));
      PetscCall(MatSeqAIJGetCOO_Private(oA, &coo_n, &coo_i, &coo_j));
      PetscCall(MatSetPreallocationCOO(oH, coo_n, coo_i, coo_j));
      PetscCall(PetscFree2(coo_i, coo_j));
    }
    hA->cooMat->assembled = PETSC_TRUE;

    M->preallocated = PETSC_TRUE;
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

    PetscCall(MatHYPRE_AttachCOOMat(M));
    if (reuse == MAT_INITIAL_MATRIX) *B = M;
  } else M = *B;

  hA = (Mat_HYPRE *)M->data;
  PetscCheck(hA->cooMat, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "HYPRE COO delegate matrix has not been created yet");

  dH = hA->cooMat;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)hA->cooMat, MATMPIAIJ, &ismpiaij));
  if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(hA->cooMat, &dH, &oH, NULL));

  PetscScalar *a;
  PetscCall(MatSeqAIJGetCSRAndMemType(dA, NULL, NULL, &a, NULL));
  PetscCall(MatSetValuesCOO(dH, a, INSERT_VALUES));
  if (oH) {
    PetscCall(MatSeqAIJGetCSRAndMemType(oA, NULL, NULL, &a, NULL));
    PetscCall(MatSetValuesCOO(oH, a, INSERT_VALUES));
  }

  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &M));
#if defined(PETSC_HAVE_DEVICE)
  (*B)->boundtocpu = boundtocpu;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_HYPRE_AIJ(Mat A, MatType mtype, MatReuse reuse, Mat *B)
{
  Mat                 M, dA = NULL, oA = NULL;
  hypre_ParCSRMatrix *parcsr;
  hypre_CSRMatrix    *dH, *oH;
  MPI_Comm            comm;
  PetscBool           ismpiaij, isseqaij;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A);
  if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)*B, MATMPIAIJ, &ismpiaij));
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)*B, MATSEQAIJ, &isseqaij));
    PetscCheck(ismpiaij || isseqaij, comm, PETSC_ERR_SUP, "Only MATMPIAIJ or MATSEQAIJ base types are supported");
  }
  PetscCall(MatHYPREGetParCSR(A, &parcsr));
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_DEVICE == hypre_ParCSRMatrixMemoryLocation(parcsr)) {
    PetscBool isaij;

    PetscCall(PetscStrcmp(mtype, MATAIJ, &isaij));
    if (isaij) {
      PetscMPIInt size;

      PetscCallMPI(MPI_Comm_size(comm, &size));
  #if defined(HYPRE_USING_HIP)
      mtype = size > 1 ? MATMPIAIJHIPSPARSE : MATSEQAIJHIPSPARSE;
  #elif defined(HYPRE_USING_CUDA)
      mtype = size > 1 ? MATMPIAIJCUSPARSE : MATSEQAIJCUSPARSE;
  #else
      mtype = size > 1 ? MATMPIAIJ : MATSEQAIJ;
  #endif
    }
  }
#endif
  dH = hypre_ParCSRMatrixDiag(parcsr);
  oH = hypre_ParCSRMatrixOffd(parcsr);
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCount coo_n;
    PetscInt  *coo_i, *coo_j;

    PetscCall(MatCreate(comm, &M));
    PetscCall(MatSetType(M, mtype));
    PetscCall(MatSetSizes(M, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
    PetscCall(MatMPIAIJSetPreallocation(M, 0, NULL, 0, NULL));

    dA = M;
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)M, MATMPIAIJ, &ismpiaij));
    if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(M, &dA, &oA, NULL));

    PetscCall(hypreCSRMatrixGetCOO_Private(dH, &coo_n, &coo_i, &coo_j));
    PetscCall(MatSetPreallocationCOO(dA, coo_n, coo_i, coo_j));
    PetscCall(PetscFree2(coo_i, coo_j));
    if (ismpiaij) {
      HYPRE_Int nc = hypre_CSRMatrixNumCols(oH);

      PetscCall(PetscLayoutDestroy(&oA->cmap));
      PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)oA), nc, nc, 1, &oA->cmap));
      PetscCall(hypreCSRMatrixGetCOO_Private(oH, &coo_n, &coo_i, &coo_j));
      PetscCall(MatSetPreallocationCOO(oA, coo_n, coo_i, coo_j));
      PetscCall(PetscFree2(coo_i, coo_j));

      /* garray */
      Mat_MPIAIJ   *aij    = (Mat_MPIAIJ *)M->data;
      HYPRE_BigInt *harray = hypre_ParCSRMatrixColMapOffd(parcsr);
      PetscInt     *garray;

      PetscCall(PetscFree(aij->garray));
      PetscCall(PetscMalloc1(nc, &garray));
      for (HYPRE_Int i = 0; i < nc; i++) garray[i] = (PetscInt)harray[i];
      aij->garray = garray;
      PetscCall(MatSetUpMultiply_MPIAIJ(M));
    }
    if (reuse == MAT_INITIAL_MATRIX) *B = M;
  } else M = *B;

  dA = M;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)M, MATMPIAIJ, &ismpiaij));
  if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(M, &dA, &oA, NULL));
  PetscCall(MatSetValuesCOOFromCSRMatrix_Private(dA, dH));
  if (oA) PetscCall(MatSetValuesCOOFromCSRMatrix_Private(oA, oH));
  M->assembled = PETSC_TRUE;
  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAIJGetParCSR_Private(Mat A, hypre_ParCSRMatrix **hA)
{
  hypre_ParCSRMatrix *tA;
  hypre_CSRMatrix    *hdiag, *hoffd;
  Mat_SeqAIJ         *diag, *offd;
  PetscInt           *garray, i, noffd, dnnz, onnz, *row_starts, *col_starts;
  MPI_Comm            comm = PetscObjectComm((PetscObject)A);
  PetscBool           ismpiaij, isseqaij;
  PetscBool           sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  HYPRE_Int          *hdi = NULL, *hdj = NULL, *hoi = NULL, *hoj = NULL;
  PetscInt           *pdi = NULL, *pdj = NULL, *poi = NULL, *poj = NULL;
  PetscBool           iscuda, iship;
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscBool boundtocpu = A->boundtocpu;
#else
  PetscBool boundtocpu = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &ismpiaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &isseqaij));
  PetscCheck(ismpiaij || isseqaij, comm, PETSC_ERR_SUP, "Unsupported type %s", ((PetscObject)A)->type_name);
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iscuda, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iship, MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, ""));
  PetscHYPREInitialize();
  if (ismpiaij) {
    Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

    diag = (Mat_SeqAIJ *)a->A->data;
    offd = (Mat_SeqAIJ *)a->B->data;
    if (!boundtocpu && (iscuda || iship)) {
#if defined(HYPRE_USING_CUDA) && defined(PETSC_HAVE_CUDA)
      if (iscuda) {
        sameint = PETSC_TRUE;
        PetscCall(MatSeqAIJCUSPARSEGetIJ(a->A, PETSC_FALSE, (const HYPRE_Int **)&hdi, (const HYPRE_Int **)&hdj));
        PetscCall(MatSeqAIJCUSPARSEGetIJ(a->B, PETSC_FALSE, (const HYPRE_Int **)&hoi, (const HYPRE_Int **)&hoj));
      }
#endif
#if defined(HYPRE_USING_HIP) && defined(PETSC_HAVE_HIP)
      if (iship) {
        sameint = PETSC_TRUE;
        PetscCall(MatSeqAIJHIPSPARSEGetIJ(a->A, PETSC_FALSE, (const HYPRE_Int **)&hdi, (const HYPRE_Int **)&hdj));
        PetscCall(MatSeqAIJHIPSPARSEGetIJ(a->B, PETSC_FALSE, (const HYPRE_Int **)&hoi, (const HYPRE_Int **)&hoj));
      }
#endif
    } else {
      boundtocpu = PETSC_TRUE;
      pdi        = diag->i;
      pdj        = diag->j;
      poi        = offd->i;
      poj        = offd->j;
      if (sameint) {
        hdi = (HYPRE_Int *)pdi;
        hdj = (HYPRE_Int *)pdj;
        hoi = (HYPRE_Int *)poi;
        hoj = (HYPRE_Int *)poj;
      }
    }
    garray = a->garray;
    noffd  = a->B->cmap->N;
    dnnz   = diag->nz;
    onnz   = offd->nz;
  } else {
    diag = (Mat_SeqAIJ *)A->data;
    offd = NULL;
    if (!boundtocpu && (iscuda || iship)) {
#if defined(HYPRE_USING_CUDA) && defined(PETSC_HAVE_CUDA)
      if (iscuda) {
        sameint = PETSC_TRUE;
        PetscCall(MatSeqAIJCUSPARSEGetIJ(A, PETSC_FALSE, (const HYPRE_Int **)&hdi, (const HYPRE_Int **)&hdj));
      }
#endif
#if defined(HYPRE_USING_HIP) && defined(PETSC_HAVE_HIP)
      if (iship) {
        sameint = PETSC_TRUE;
        PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, (const HYPRE_Int **)&hdi, (const HYPRE_Int **)&hdj));
      }
#endif
    } else {
      boundtocpu = PETSC_TRUE;
      pdi        = diag->i;
      pdj        = diag->j;
      if (sameint) {
        hdi = (HYPRE_Int *)pdi;
        hdj = (HYPRE_Int *)pdj;
      }
    }
    garray = NULL;
    noffd  = 0;
    dnnz   = diag->nz;
    onnz   = 0;
  }

  /* create a temporary ParCSR */
  if (HYPRE_AssumedPartitionCheck()) {
    PetscMPIInt myid;

    PetscCallMPI(MPI_Comm_rank(comm, &myid));
    row_starts = A->rmap->range + myid;
    col_starts = A->cmap->range + myid;
  } else {
    row_starts = A->rmap->range;
    col_starts = A->cmap->range;
  }
  tA = hypre_ParCSRMatrixCreate(comm, A->rmap->N, A->cmap->N, (HYPRE_BigInt *)row_starts, (HYPRE_BigInt *)col_starts, noffd, dnnz, onnz);
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  hypre_ParCSRMatrixSetRowStartsOwner(tA, 0);
  hypre_ParCSRMatrixSetColStartsOwner(tA, 0);
#endif

  /* set diagonal part */
  hdiag = hypre_ParCSRMatrixDiag(tA);
  if (!sameint) { /* malloc CSR pointers */
    PetscCall(PetscMalloc2(A->rmap->n + 1, &hdi, dnnz, &hdj));
    for (i = 0; i < A->rmap->n + 1; i++) hdi[i] = (HYPRE_Int)pdi[i];
    for (i = 0; i < dnnz; i++) hdj[i] = (HYPRE_Int)pdj[i];
  }
  hypre_CSRMatrixI(hdiag)           = hdi;
  hypre_CSRMatrixJ(hdiag)           = hdj;
  hypre_CSRMatrixData(hdiag)        = (HYPRE_Complex *)diag->a;
  hypre_CSRMatrixNumNonzeros(hdiag) = diag->nz;
  hypre_CSRMatrixSetDataOwner(hdiag, 0);

  /* set off-diagonal part */
  hoffd = hypre_ParCSRMatrixOffd(tA);
  if (offd) {
    if (!sameint) { /* malloc CSR pointers */
      PetscCall(PetscMalloc2(A->rmap->n + 1, &hoi, onnz, &hoj));
      for (i = 0; i < A->rmap->n + 1; i++) hoi[i] = (HYPRE_Int)poi[i];
      for (i = 0; i < onnz; i++) hoj[i] = (HYPRE_Int)poj[i];
    }
    hypre_CSRMatrixI(hoffd)           = hoi;
    hypre_CSRMatrixJ(hoffd)           = hoj;
    hypre_CSRMatrixData(hoffd)        = (HYPRE_Complex *)offd->a;
    hypre_CSRMatrixNumNonzeros(hoffd) = offd->nz;
    hypre_CSRMatrixSetDataOwner(hoffd, 0);
  }
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscCallExternal(hypre_ParCSRMatrixInitialize_v2, tA, !boundtocpu ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST);
#else
  #if PETSC_PKG_HYPRE_VERSION_LT(2, 18, 0)
  PetscCallExternal(hypre_ParCSRMatrixInitialize, tA);
  #else
  PetscCallExternal(hypre_ParCSRMatrixInitialize_v2, tA, HYPRE_MEMORY_HOST);
  #endif
#endif

  /* MatrixSetRownnz comes after MatrixInitialize, so the first uses the right memory location */
  hypre_CSRMatrixSetRownnz(hdiag);
  if (offd) hypre_CSRMatrixSetRownnz(hoffd);

  hypre_TFree(hypre_ParCSRMatrixColMapOffd(tA), HYPRE_MEMORY_HOST);
  hypre_ParCSRMatrixSetNumNonzeros(tA);
  hypre_ParCSRMatrixColMapOffd(tA) = (HYPRE_BigInt *)garray;
  if (!hypre_ParCSRMatrixCommPkg(tA)) PetscCallExternal(hypre_MatvecCommPkgCreate, tA);
  *hA = tA;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAIJRestoreParCSR_Private(Mat A, hypre_ParCSRMatrix **hA)
{
  hypre_CSRMatrix *hdiag, *hoffd;
  PetscBool        ismpiaij, sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  PetscBool        iscuda, iship;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &ismpiaij));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iscuda, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iship, MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, ""));
#if defined(HYPRE_USING_CUDA) && defined(PETSC_HAVE_CUDA)
  if (iscuda) sameint = PETSC_TRUE;
#elif defined(HYPRE_USING_HIP) && defined(PETSC_HAVE_HIP)
  if (iship) sameint = PETSC_TRUE;
#endif
  hdiag = hypre_ParCSRMatrixDiag(*hA);
  hoffd = hypre_ParCSRMatrixOffd(*hA);
  /* free temporary memory allocated by PETSc
     set pointers to NULL before destroying tA */
  if (!sameint) {
    HYPRE_Int *hi, *hj;

    hi = hypre_CSRMatrixI(hdiag);
    hj = hypre_CSRMatrixJ(hdiag);
    PetscCall(PetscFree2(hi, hj));
    if (ismpiaij) {
      hi = hypre_CSRMatrixI(hoffd);
      hj = hypre_CSRMatrixJ(hoffd);
      PetscCall(PetscFree2(hi, hj));
    }
  }
  hypre_CSRMatrixI(hdiag)    = NULL;
  hypre_CSRMatrixJ(hdiag)    = NULL;
  hypre_CSRMatrixData(hdiag) = NULL;
  if (ismpiaij) {
    hypre_CSRMatrixI(hoffd)    = NULL;
    hypre_CSRMatrixJ(hoffd)    = NULL;
    hypre_CSRMatrixData(hoffd) = NULL;
  }
  hypre_ParCSRMatrixColMapOffd(*hA) = NULL;
  hypre_ParCSRMatrixDestroy(*hA);
  *hA = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* calls RAP from BoomerAMG:
   the resulting ParCSR will not own the column and row starts
   It looks like we don't need to have the diagonal entries ordered first */
static PetscErrorCode MatHYPRE_ParCSR_RAP(hypre_ParCSRMatrix *hR, hypre_ParCSRMatrix *hA, hypre_ParCSRMatrix *hP, hypre_ParCSRMatrix **hRAP)
{
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  HYPRE_Int P_owns_col_starts, R_owns_row_starts;
#endif

  PetscFunctionBegin;
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  P_owns_col_starts = hypre_ParCSRMatrixOwnsColStarts(hP);
  R_owns_row_starts = hypre_ParCSRMatrixOwnsRowStarts(hR);
#endif
  /* can be replaced by version test later */
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscStackPushExternal("hypre_ParCSRMatrixRAP");
  *hRAP = hypre_ParCSRMatrixRAP(hR, hA, hP);
  PetscStackPop;
#else
  PetscCallExternal(hypre_BoomerAMGBuildCoarseOperator, hR, hA, hP, hRAP);
  PetscCallExternal(hypre_ParCSRMatrixSetNumNonzeros, *hRAP);
#endif
  /* hypre_BoomerAMGBuildCoarseOperator steals the col_starts from P and the row_starts from R */
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  hypre_ParCSRMatrixSetRowStartsOwner(*hRAP, 0);
  hypre_ParCSRMatrixSetColStartsOwner(*hRAP, 0);
  if (P_owns_col_starts) hypre_ParCSRMatrixSetColStartsOwner(hP, 1);
  if (R_owns_row_starts) hypre_ParCSRMatrixSetRowStartsOwner(hR, 1);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPtAPNumeric_AIJ_AIJ_wHYPRE(Mat A, Mat P, Mat C)
{
  Mat                 B;
  hypre_ParCSRMatrix *hA, *hP, *hPtAP = NULL;
  Mat_Product        *product = C->product;

  PetscFunctionBegin;
  PetscCall(MatAIJGetParCSR_Private(A, &hA));
  PetscCall(MatAIJGetParCSR_Private(P, &hP));
  PetscCall(MatHYPRE_ParCSR_RAP(hP, hA, hP, &hPtAP));
  PetscCall(MatCreateFromParCSR(hPtAP, MATAIJ, PETSC_OWN_POINTER, &B));

  PetscCall(MatHeaderMerge(C, &B));
  C->product = product;

  PetscCall(MatAIJRestoreParCSR_Private(A, &hA));
  PetscCall(MatAIJRestoreParCSR_Private(P, &hP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatPtAPSymbolic_AIJ_AIJ_wHYPRE(Mat A, Mat P, PetscReal fill, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(C, MATAIJ));
  C->ops->ptapnumeric    = MatPtAPNumeric_AIJ_AIJ_wHYPRE;
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPtAPNumeric_AIJ_HYPRE(Mat A, Mat P, Mat C)
{
  Mat                 B;
  Mat_HYPRE          *hP;
  hypre_ParCSRMatrix *hA = NULL, *Pparcsr, *ptapparcsr = NULL;
  HYPRE_Int           type;
  MPI_Comm            comm = PetscObjectComm((PetscObject)A);
  PetscBool           ishypre;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATHYPRE, &ishypre));
  PetscCheck(ishypre, comm, PETSC_ERR_USER, "P should be of type %s", MATHYPRE);
  hP = (Mat_HYPRE *)P->data;
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, hP->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, comm, PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObject, hP->ij, (void **)&Pparcsr);

  PetscCall(MatAIJGetParCSR_Private(A, &hA));
  PetscCall(MatHYPRE_ParCSR_RAP(Pparcsr, hA, Pparcsr, &ptapparcsr));
  PetscCall(MatAIJRestoreParCSR_Private(A, &hA));

  /* create temporary matrix and merge to C */
  PetscCall(MatCreateFromParCSR(ptapparcsr, ((PetscObject)C)->type_name, PETSC_OWN_POINTER, &B));
  PetscCall(MatHeaderMerge(C, &B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPtAPNumeric_HYPRE_HYPRE(Mat A, Mat P, Mat C)
{
  Mat                 B;
  hypre_ParCSRMatrix *Aparcsr, *Pparcsr, *ptapparcsr = NULL;
  Mat_HYPRE          *hA, *hP;
  PetscBool           ishypre;
  HYPRE_Int           type;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATHYPRE, &ishypre));
  PetscCheck(ishypre, PetscObjectComm((PetscObject)P), PETSC_ERR_USER, "P should be of type %s", MATHYPRE);
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHYPRE, &ishypre));
  PetscCheck(ishypre, PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "A should be of type %s", MATHYPRE);
  hA = (Mat_HYPRE *)A->data;
  hP = (Mat_HYPRE *)P->data;
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, hA->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, hP->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)P), PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)&Aparcsr);
  PetscCallExternal(HYPRE_IJMatrixGetObject, hP->ij, (void **)&Pparcsr);
  PetscCall(MatHYPRE_ParCSR_RAP(Pparcsr, Aparcsr, Pparcsr, &ptapparcsr));
  PetscCall(MatCreateFromParCSR(ptapparcsr, MATHYPRE, PETSC_OWN_POINTER, &B));
  PetscCall(MatHeaderMerge(C, &B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* calls hypre_ParMatmul
   hypre_ParMatMul uses hypre_ParMatrixCreate with the communicator of hA
   hypre_ParMatrixCreate does not duplicate the communicator
   It looks like we don't need to have the diagonal entries ordered first */
static PetscErrorCode MatHYPRE_ParCSR_MatMatMult(hypre_ParCSRMatrix *hA, hypre_ParCSRMatrix *hB, hypre_ParCSRMatrix **hAB)
{
  PetscFunctionBegin;
  /* can be replaced by version test later */
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscStackPushExternal("hypre_ParCSRMatMat");
  *hAB = hypre_ParCSRMatMat(hA, hB);
#else
  PetscStackPushExternal("hypre_ParMatmul");
  *hAB = hypre_ParMatmul(hA, hB);
#endif
  PetscStackPop;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatMultNumeric_AIJ_AIJ_wHYPRE(Mat A, Mat B, Mat C)
{
  Mat                 D;
  hypre_ParCSRMatrix *hA, *hB, *hAB = NULL;
  Mat_Product        *product = C->product;

  PetscFunctionBegin;
  PetscCall(MatAIJGetParCSR_Private(A, &hA));
  PetscCall(MatAIJGetParCSR_Private(B, &hB));
  PetscCall(MatHYPRE_ParCSR_MatMatMult(hA, hB, &hAB));
  PetscCall(MatCreateFromParCSR(hAB, MATAIJ, PETSC_OWN_POINTER, &D));

  PetscCall(MatHeaderMerge(C, &D));
  C->product = product;

  PetscCall(MatAIJRestoreParCSR_Private(A, &hA));
  PetscCall(MatAIJRestoreParCSR_Private(B, &hB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatMatMultSymbolic_AIJ_AIJ_wHYPRE(Mat A, Mat B, PetscReal fill, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(C, MATAIJ));
  C->ops->matmultnumeric = MatMatMultNumeric_AIJ_AIJ_wHYPRE;
  C->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatMultNumeric_HYPRE_HYPRE(Mat A, Mat B, Mat C)
{
  Mat                 D;
  hypre_ParCSRMatrix *Aparcsr, *Bparcsr, *ABparcsr = NULL;
  Mat_HYPRE          *hA, *hB;
  PetscBool           ishypre;
  HYPRE_Int           type;
  Mat_Product        *product;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATHYPRE, &ishypre));
  PetscCheck(ishypre, PetscObjectComm((PetscObject)B), PETSC_ERR_USER, "B should be of type %s", MATHYPRE);
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHYPRE, &ishypre));
  PetscCheck(ishypre, PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "A should be of type %s", MATHYPRE);
  hA = (Mat_HYPRE *)A->data;
  hB = (Mat_HYPRE *)B->data;
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, hA->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, hB->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "Only HYPRE_PARCSR is supported");
  PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)&Aparcsr);
  PetscCallExternal(HYPRE_IJMatrixGetObject, hB->ij, (void **)&Bparcsr);
  PetscCall(MatHYPRE_ParCSR_MatMatMult(Aparcsr, Bparcsr, &ABparcsr));
  PetscCall(MatCreateFromParCSR(ABparcsr, MATHYPRE, PETSC_OWN_POINTER, &D));

  /* need to use HeaderReplace because HeaderMerge messes up with the communicator */
  product    = C->product; /* save it from MatHeaderReplace() */
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &D));
  C->product             = product;
  C->ops->matmultnumeric = MatMatMultNumeric_HYPRE_HYPRE;
  C->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Mat A, Mat B, Mat C, Mat D)
{
  Mat                 E;
  hypre_ParCSRMatrix *hA, *hB, *hC, *hABC = NULL;

  PetscFunctionBegin;
  PetscCall(MatAIJGetParCSR_Private(A, &hA));
  PetscCall(MatAIJGetParCSR_Private(B, &hB));
  PetscCall(MatAIJGetParCSR_Private(C, &hC));
  PetscCall(MatHYPRE_ParCSR_RAP(hA, hB, hC, &hABC));
  PetscCall(MatCreateFromParCSR(hABC, MATAIJ, PETSC_OWN_POINTER, &E));
  PetscCall(MatHeaderMerge(D, &E));
  PetscCall(MatAIJRestoreParCSR_Private(A, &hA));
  PetscCall(MatAIJRestoreParCSR_Private(B, &hB));
  PetscCall(MatAIJRestoreParCSR_Private(C, &hC));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Mat A, Mat B, Mat C, PetscReal fill, Mat D)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(D, MATAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_AB_HYPRE(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_HYPRE_AB(Mat C)
{
  Mat_Product *product = C->product;
  PetscBool    Ahypre;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)product->A, MATHYPRE, &Ahypre));
  if (Ahypre) { /* A is a Hypre matrix */
    PetscCall(MatSetType(C, MATHYPRE));
    C->ops->productsymbolic = MatProductSymbolic_AB_HYPRE;
    C->ops->matmultnumeric  = MatMatMultNumeric_HYPRE_HYPRE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_PtAP_HYPRE(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_HYPRE_PtAP(Mat C)
{
  Mat_Product *product = C->product;
  PetscBool    flg;
  PetscInt     type        = 0;
  const char  *outTypes[4] = {"aij", "seqaij", "mpiaij", "hypre"};
  PetscInt     ntype       = 4;
  Mat          A           = product->A;
  PetscBool    Ahypre;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHYPRE, &Ahypre));
  if (Ahypre) { /* A is a Hypre matrix */
    PetscCall(MatSetType(C, MATHYPRE));
    C->ops->productsymbolic = MatProductSymbolic_PtAP_HYPRE;
    C->ops->ptapnumeric     = MatPtAPNumeric_HYPRE_HYPRE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* A is AIJ, P is Hypre, C = PtAP can be either AIJ or Hypre format */
  /* Get runtime option */
  if (product->api_user) {
    PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatPtAP_HYPRE", "Mat");
    PetscCall(PetscOptionsEList("-matptap_hypre_outtype", "MatPtAP outtype", "MatPtAP outtype", outTypes, ntype, outTypes[type], &type, &flg));
    PetscOptionsEnd();
  } else {
    PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatProduct_PtAP_HYPRE", "Mat");
    PetscCall(PetscOptionsEList("-mat_product_algorithm_hypre_outtype", "MatProduct_PtAP outtype", "MatProduct_PtAP", outTypes, ntype, outTypes[type], &type, &flg));
    PetscOptionsEnd();
  }

  if (type == 0 || type == 1 || type == 2) {
    PetscCall(MatSetType(C, MATAIJ));
  } else if (type == 3) {
    PetscCall(MatSetType(C, MATHYPRE));
  } else SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "MatPtAP outtype is not supported");
  C->ops->productsymbolic = MatProductSymbolic_PtAP_HYPRE;
  C->ops->ptapnumeric     = MatPtAPNumeric_AIJ_HYPRE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_HYPRE(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatProductSetFromOptions_HYPRE_AB(C));
    break;
  case MATPRODUCT_PtAP:
    PetscCall(MatProductSetFromOptions_HYPRE_PtAP(C));
    break;
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_HYPRE(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatHYPRE_MultKernel_Private(A, 1.0, x, 0.0, y, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_HYPRE(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatHYPRE_MultKernel_Private(A, 1.0, x, 0.0, y, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_HYPRE(Mat A, Vec x, Vec y, Vec z)
{
  PetscFunctionBegin;
  if (y != z) PetscCall(VecCopy(y, z));
  PetscCall(MatHYPRE_MultKernel_Private(A, 1.0, x, 1.0, z, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAdd_HYPRE(Mat A, Vec x, Vec y, Vec z)
{
  PetscFunctionBegin;
  if (y != z) PetscCall(VecCopy(y, z));
  PetscCall(MatHYPRE_MultKernel_Private(A, 1.0, x, 1.0, z, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = a * A * x + b * y or y = a * A^t * x + b * y depending on trans */
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat A, HYPRE_Complex a, Vec x, HYPRE_Complex b, Vec y, PetscBool trans)
{
  Mat_HYPRE          *hA = (Mat_HYPRE *)A->data;
  hypre_ParCSRMatrix *parcsr;
  hypre_ParVector    *hx, *hy;

  PetscFunctionBegin;
  if (trans) {
    PetscCall(VecHYPRE_IJVectorPushVecRead(hA->b, x));
    if (b != 0.0) PetscCall(VecHYPRE_IJVectorPushVec(hA->x, y));
    else PetscCall(VecHYPRE_IJVectorPushVecWrite(hA->x, y));
    PetscCallExternal(HYPRE_IJVectorGetObject, hA->b->ij, (void **)&hx);
    PetscCallExternal(HYPRE_IJVectorGetObject, hA->x->ij, (void **)&hy);
  } else {
    PetscCall(VecHYPRE_IJVectorPushVecRead(hA->x, x));
    if (b != 0.0) PetscCall(VecHYPRE_IJVectorPushVec(hA->b, y));
    else PetscCall(VecHYPRE_IJVectorPushVecWrite(hA->b, y));
    PetscCallExternal(HYPRE_IJVectorGetObject, hA->x->ij, (void **)&hx);
    PetscCallExternal(HYPRE_IJVectorGetObject, hA->b->ij, (void **)&hy);
  }
  PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)&parcsr);
  if (trans) {
    PetscCallExternal(hypre_ParCSRMatrixMatvecT, a, parcsr, hx, b, hy);
  } else {
    PetscCallExternal(hypre_ParCSRMatrixMatvec, a, parcsr, hx, b, hy);
  }
  PetscCall(VecHYPRE_IJVectorPopVec(hA->x));
  PetscCall(VecHYPRE_IJVectorPopVec(hA->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_HYPRE(Mat A)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;

  PetscFunctionBegin;
  PetscCall(VecHYPRE_IJVectorDestroy(&hA->x));
  PetscCall(VecHYPRE_IJVectorDestroy(&hA->b));
  PetscCall(MatHYPRE_DestroyCOOMat(A)); /* must be called before destroying the individual CSR */
  if (hA->ij) {
    if (!hA->inner_free) hypre_IJMatrixObject(hA->ij) = NULL;
    PetscCallExternal(HYPRE_IJMatrixDestroy, hA->ij);
  }
  if (hA->comm) PetscCall(PetscCommRestoreComm(PetscObjectComm((PetscObject)A), &hA->comm));

  PetscCall(MatStashDestroy_Private(&A->stash));
  PetscCall(PetscFree(hA->array));
  if (hA->rows_d) PetscStackCallExternalVoid("hypre_Free", hypre_Free(hA->rows_d, HYPRE_MEMORY_DEVICE));

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_hypre_aij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_hypre_is_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_mpiaij_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijhipsparse_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_mpiaijhipsparse_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijcusparse_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_mpiaijcusparse_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHYPRESetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHYPREGetParCSR_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_HYPRE(Mat A)
{
  PetscFunctionBegin;
  if (!A->preallocated) PetscCall(MatHYPRESetPreallocation(A, PETSC_DEFAULT, NULL, PETSC_DEFAULT, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO FIX hypre_CSRMatrixMatvecOutOfPlace
#if defined(PETSC_HAVE_HYPRE_DEVICE)
static PetscErrorCode MatBindToCPU_HYPRE(Mat A, PetscBool bind)
{
  Mat_HYPRE           *hA   = (Mat_HYPRE *)A->data;
  HYPRE_MemoryLocation hmem = bind ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;

  PetscFunctionBegin;
  A->boundtocpu = bind;
  if (hA->ij && hypre_IJMatrixAssembleFlag(hA->ij) && hmem != hypre_IJMatrixMemoryLocation(hA->ij)) {
    hypre_ParCSRMatrix *parcsr;
    PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)&parcsr);
    PetscCallExternal(hypre_ParCSRMatrixMigrate, parcsr, hmem);
  }
  if (hA->x) PetscCall(VecHYPRE_IJBindToCPU(hA->x, bind));
  if (hA->b) PetscCall(VecHYPRE_IJBindToCPU(hA->b, bind));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatAssemblyEnd_HYPRE(Mat A, MatAssemblyType mode)
{
  Mat_HYPRE   *hA = (Mat_HYPRE *)A->data;
  PetscMPIInt  n;
  PetscInt     i, j, rstart, ncols, flg;
  PetscInt    *row, *col;
  PetscScalar *val;

  PetscFunctionBegin;
  PetscCheck(mode != MAT_FLUSH_ASSEMBLY, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MAT_FLUSH_ASSEMBLY currently not supported with MATHYPRE");

  if (!A->nooffprocentries) {
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&A->stash, &n, &row, &col, &val, &flg));
      if (!flg) break;

      for (i = 0; i < n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j = i, rstart = row[j]; j < n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j - i;
        else ncols = n - i;
        /* Now assemble all these values with a single function call */
        PetscCall(MatSetValues_HYPRE(A, 1, row + i, ncols, col + i, val + i, A->insertmode));

        i = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&A->stash));
  }

  PetscCallExternal(HYPRE_IJMatrixAssemble, hA->ij);
  /* The assembly routine destroys the aux_matrix, we recreate it here by calling HYPRE_IJMatrixInitialize */
  /* If the option MAT_SORTED_FULL is set to true, the indices and values can be passed to hypre directly, so we don't need the aux_matrix */
  if (!A->sortedfull) {
    hypre_AuxParCSRMatrix *aux_matrix;

    /* call destroy just to make sure we do not leak anything */
    aux_matrix = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(hA->ij);
    PetscCallExternal(hypre_AuxParCSRMatrixDestroy, aux_matrix);
    hypre_IJMatrixTranslator(hA->ij) = NULL;

    /* Initialize with assembled flag -> it only recreates the aux_par_matrix */
    PetscCallExternal(HYPRE_IJMatrixInitialize, hA->ij);
    aux_matrix = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(hA->ij);
    if (aux_matrix) {
      hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 1; /* see comment in MatHYPRESetPreallocation_HYPRE */
#if PETSC_PKG_HYPRE_VERSION_LT(2, 19, 0)
      PetscCallExternal(hypre_AuxParCSRMatrixInitialize, aux_matrix);
#else
      PetscCallExternal(hypre_AuxParCSRMatrixInitialize_v2, aux_matrix, HYPRE_MEMORY_HOST);
#endif
    }
  }
  {
    hypre_ParCSRMatrix *parcsr;

    PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)&parcsr);
    if (!hypre_ParCSRMatrixCommPkg(parcsr)) PetscCallExternal(hypre_MatvecCommPkgCreate, parcsr);
  }
  if (!hA->x) PetscCall(VecHYPRE_IJVectorCreate(A->cmap, &hA->x));
  if (!hA->b) PetscCall(VecHYPRE_IJVectorCreate(A->rmap, &hA->b));
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscCall(MatBindToCPU_HYPRE(A, A->boundtocpu));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetArray_HYPRE(Mat A, PetscInt size, void **array)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;

  PetscFunctionBegin;
  PetscCheck(hA->array_available, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Temporary space is in use");

  if (hA->array_size >= size) {
    *array = hA->array;
  } else {
    PetscCall(PetscFree(hA->array));
    hA->array_size = size;
    PetscCall(PetscMalloc(hA->array_size, &hA->array));
    *array = hA->array;
  }

  hA->array_available = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreArray_HYPRE(Mat A, void **array)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;

  PetscFunctionBegin;
  *array              = NULL;
  hA->array_available = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValues_HYPRE(Mat A, PetscInt nr, const PetscInt rows[], PetscInt nc, const PetscInt cols[], const PetscScalar v[], InsertMode ins)
{
  Mat_HYPRE     *hA   = (Mat_HYPRE *)A->data;
  PetscScalar   *vals = (PetscScalar *)v;
  HYPRE_Complex *sscr;
  PetscInt      *cscr[2];
  PetscInt       i, nzc;
  PetscInt       rst = A->rmap->rstart, ren = A->rmap->rend;
  void          *array = NULL;

  PetscFunctionBegin;
  PetscCall(MatGetArray_HYPRE(A, sizeof(PetscInt) * (2 * nc) + sizeof(HYPRE_Complex) * nc * nr, &array));
  cscr[0] = (PetscInt *)array;
  cscr[1] = ((PetscInt *)array) + nc;
  sscr    = (HYPRE_Complex *)(((PetscInt *)array) + nc * 2);
  for (i = 0, nzc = 0; i < nc; i++) {
    if (cols[i] >= 0) {
      cscr[0][nzc]   = cols[i];
      cscr[1][nzc++] = i;
    }
  }
  if (!nzc) {
    PetscCall(MatRestoreArray_HYPRE(A, &array));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

#if 0 //defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_HOST != hypre_IJMatrixMemoryLocation(hA->ij)) {
    hypre_ParCSRMatrix *parcsr;

    PetscCallExternal(HYPRE_IJMatrixGetObject,hA->ij,(void**)&parcsr);
    PetscCallExternal(hypre_ParCSRMatrixMigrate,parcsr, HYPRE_MEMORY_HOST);
  }
#endif

  if (ins == ADD_VALUES) {
    for (i = 0; i < nr; i++) {
      if (rows[i] >= 0) {
        PetscInt  j;
        HYPRE_Int hnc = (HYPRE_Int)nzc;

        if (!nzc) continue;
        /* nonlocal values */
        if (rows[i] < rst || rows[i] >= ren) {
          PetscCheck(!A->nooffprocentries, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set", rows[i]);
          if (hA->donotstash) continue;
        }
        PetscCheck((PetscInt)hnc == nzc, PETSC_COMM_SELF, PETSC_ERR_SUP, "Hypre overflow! number of columns %" PetscInt_FMT " for row %" PetscInt_FMT, nzc, rows[i]);
        for (j = 0; j < nzc; j++) PetscCall(PetscHYPREScalarCast(vals[cscr[1][j]], &sscr[j]));
        PetscCallExternal(HYPRE_IJMatrixAddToValues, hA->ij, 1, &hnc, (HYPRE_BigInt *)(rows + i), (HYPRE_BigInt *)cscr[0], sscr);
      }
      vals += nc;
    }
  } else { /* INSERT_VALUES */
    for (i = 0; i < nr; i++) {
      if (rows[i] >= 0) {
        PetscInt  j;
        HYPRE_Int hnc = (HYPRE_Int)nzc;

        if (!nzc) continue;
        PetscCheck((PetscInt)hnc == nzc, PETSC_COMM_SELF, PETSC_ERR_SUP, "Hypre overflow! number of columns %" PetscInt_FMT " for row %" PetscInt_FMT, nzc, rows[i]);
        for (j = 0; j < nzc; j++) PetscCall(PetscHYPREScalarCast(vals[cscr[1][j]], &sscr[j]));
        /* nonlocal values */
        if (rows[i] < rst || rows[i] >= ren) {
          PetscCheck(!A->nooffprocentries, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set", rows[i]);
          if (!hA->donotstash) PetscCall(MatStashValuesRow_Private(&A->stash, rows[i], nzc, cscr[0], (PetscScalar *)sscr, PETSC_FALSE));
        }
        /* local values */
        else
          PetscCallExternal(HYPRE_IJMatrixSetValues, hA->ij, 1, &hnc, (HYPRE_BigInt *)(rows + i), (HYPRE_BigInt *)cscr[0], sscr);
      }
      vals += nc;
    }
  }

  PetscCall(MatRestoreArray_HYPRE(A, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPRESetPreallocation_HYPRE(Mat A, PetscInt dnz, const PetscInt dnnz[], PetscInt onz, const PetscInt onnz[])
{
  Mat_HYPRE  *hA = (Mat_HYPRE *)A->data;
  HYPRE_Int  *hdnnz, *honnz;
  PetscInt    i, rs, re, cs, ce, bs;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  rs = A->rmap->rstart;
  re = A->rmap->rend;
  cs = A->cmap->rstart;
  ce = A->cmap->rend;
  if (!hA->ij) {
    PetscCallExternal(HYPRE_IJMatrixCreate, hA->comm, rs, re - 1, cs, ce - 1, &hA->ij);
    PetscCallExternal(HYPRE_IJMatrixSetObjectType, hA->ij, HYPRE_PARCSR);
  } else {
    HYPRE_BigInt hrs, hre, hcs, hce;
    PetscCallExternal(HYPRE_IJMatrixGetLocalRange, hA->ij, &hrs, &hre, &hcs, &hce);
    PetscCheck(hre - hrs + 1 == re - rs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent local rows: IJMatrix [%" PetscHYPRE_BigInt_FMT ",%" PetscHYPRE_BigInt_FMT "), PETSc [%" PetscInt_FMT ",%" PetscInt_FMT ")", hrs, hre + 1, rs, re);
    PetscCheck(hce - hcs + 1 == ce - cs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent local cols: IJMatrix [%" PetscHYPRE_BigInt_FMT ",%" PetscHYPRE_BigInt_FMT "), PETSc [%" PetscInt_FMT ",%" PetscInt_FMT ")", hcs, hce + 1, cs, ce);
  }
  PetscCall(MatHYPRE_DestroyCOOMat(A));
  PetscCall(MatGetBlockSize(A, &bs));
  if (dnz == PETSC_DEFAULT || dnz == PETSC_DECIDE) dnz = 10 * bs;
  if (onz == PETSC_DEFAULT || onz == PETSC_DECIDE) onz = 10 * bs;

  if (!dnnz) {
    PetscCall(PetscMalloc1(A->rmap->n, &hdnnz));
    for (i = 0; i < A->rmap->n; i++) hdnnz[i] = dnz;
  } else {
    hdnnz = (HYPRE_Int *)dnnz;
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size > 1) {
    hypre_AuxParCSRMatrix *aux_matrix;
    if (!onnz) {
      PetscCall(PetscMalloc1(A->rmap->n, &honnz));
      for (i = 0; i < A->rmap->n; i++) honnz[i] = onz;
    } else honnz = (HYPRE_Int *)onnz;
    /* SetDiagOffdSizes sets hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0, since it seems
       they assume the user will input the entire row values, properly sorted
       In PETSc, we don't make such an assumption and set this flag to 1,
       unless the option MAT_SORTED_FULL is set to true.
       Also, to avoid possible memory leaks, we destroy and recreate the translator
       This has to be done here, as HYPRE_IJMatrixInitialize will properly initialize
       the IJ matrix for us */
    aux_matrix = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(hA->ij);
    hypre_AuxParCSRMatrixDestroy(aux_matrix);
    hypre_IJMatrixTranslator(hA->ij) = NULL;
    PetscCallExternal(HYPRE_IJMatrixSetDiagOffdSizes, hA->ij, hdnnz, honnz);
    aux_matrix                               = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(hA->ij);
    hypre_AuxParCSRMatrixNeedAux(aux_matrix) = !A->sortedfull;
  } else {
    honnz = NULL;
    PetscCallExternal(HYPRE_IJMatrixSetRowSizes, hA->ij, hdnnz);
  }

  /* reset assembled flag and call the initialize method */
  hypre_IJMatrixAssembleFlag(hA->ij) = 0;
#if PETSC_PKG_HYPRE_VERSION_LT(2, 19, 0)
  PetscCallExternal(HYPRE_IJMatrixInitialize, hA->ij);
#else
  PetscCallExternal(HYPRE_IJMatrixInitialize_v2, hA->ij, HYPRE_MEMORY_HOST);
#endif
  if (!dnnz) PetscCall(PetscFree(hdnnz));
  if (!onnz && honnz) PetscCall(PetscFree(honnz));
  /* Match AIJ logic */
  A->preallocated = PETSC_TRUE;
  A->assembled    = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatHYPRESetPreallocation - Preallocates memory for a sparse parallel matrix in HYPRE IJ format

  Collective

  Input Parameters:
+ A    - the matrix
. dnz  - number of nonzeros per row in DIAGONAL portion of local submatrix
          (same value is used for all local rows)
. dnnz - array containing the number of nonzeros in the various rows of the
          DIAGONAL portion of the local submatrix (possibly different for each row)
          or `NULL` (`PETSC_NULL_INTEGER` in Fortran), if `d_nz` is used to specify the nonzero structure.
          The size of this array is equal to the number of local rows, i.e `m`.
          For matrices that will be factored, you must leave room for (and set)
          the diagonal entry even if it is zero.
. onz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
          submatrix (same value is used for all local rows).
- onnz - array containing the number of nonzeros in the various rows of the
          OFF-DIAGONAL portion of the local submatrix (possibly different for
          each row) or `NULL` (`PETSC_NULL_INTEGER` in Fortran), if `o_nz` is used to specify the nonzero
          structure. The size of this array is equal to the number
          of local rows, i.e `m`.

  Level: intermediate

  Note:
  If the *nnz parameter is given then the *nz parameter is ignored; for sequential matrices, `onz` and `onnz` are ignored.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatMPIAIJSetPreallocation()`, `MATHYPRE`, `MATAIJ`
@*/
PetscErrorCode MatHYPRESetPreallocation(Mat A, PetscInt dnz, const PetscInt dnnz[], PetscInt onz, const PetscInt onnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscTryMethod(A, "MatHYPRESetPreallocation_C", (Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]), (A, dnz, dnnz, onz, onnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateFromParCSR - Creates a `Mat` from a `hypre_ParCSRMatrix`

  Collective

  Input Parameters:
+ parcsr   - the pointer to the `hypre_ParCSRMatrix`
. mtype    - matrix type to be created. Currently `MATAIJ`, `MATIS` and `MATHYPRE` are supported.
- copymode - PETSc copying options, see  `PetscCopyMode`

  Output Parameter:
. A - the matrix

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATHYPRE`, `PetscCopyMode`
@*/
PETSC_EXTERN PetscErrorCode MatCreateFromParCSR(hypre_ParCSRMatrix *parcsr, MatType mtype, PetscCopyMode copymode, Mat *A)
{
  Mat        T;
  Mat_HYPRE *hA;
  MPI_Comm   comm;
  PetscInt   rstart, rend, cstart, cend, M, N;
  PetscBool  isseqaij, isseqaijmkl, ismpiaij, isaij, ishyp, isis;

  PetscFunctionBegin;
  comm = hypre_ParCSRMatrixComm(parcsr);
  PetscCall(PetscStrcmp(mtype, MATSEQAIJ, &isseqaij));
  PetscCall(PetscStrcmp(mtype, MATSEQAIJMKL, &isseqaijmkl));
  PetscCall(PetscStrcmp(mtype, MATMPIAIJ, &ismpiaij));
  PetscCall(PetscStrcmp(mtype, MATAIJ, &isaij));
  PetscCall(PetscStrcmp(mtype, MATHYPRE, &ishyp));
  PetscCall(PetscStrcmp(mtype, MATIS, &isis));
  isaij = (PetscBool)(isseqaij || isseqaijmkl || ismpiaij || isaij);
  /* TODO */
  PetscCheck(isaij || ishyp || isis, comm, PETSC_ERR_SUP, "Unsupported MatType %s! Supported types are %s, %s, %s, %s, %s, and %s", mtype, MATAIJ, MATSEQAIJ, MATSEQAIJMKL, MATMPIAIJ, MATIS, MATHYPRE);
  /* access ParCSRMatrix */
  rstart = hypre_ParCSRMatrixFirstRowIndex(parcsr);
  rend   = hypre_ParCSRMatrixLastRowIndex(parcsr);
  cstart = hypre_ParCSRMatrixFirstColDiag(parcsr);
  cend   = hypre_ParCSRMatrixLastColDiag(parcsr);
  M      = hypre_ParCSRMatrixGlobalNumRows(parcsr);
  N      = hypre_ParCSRMatrixGlobalNumCols(parcsr);

  /* create PETSc matrix with MatHYPRE */
  PetscCall(MatCreate(comm, &T));
  PetscCall(MatSetSizes(T, PetscMax(rend - rstart + 1, 0), PetscMax(cend - cstart + 1, 0), M, N));
  PetscCall(MatSetType(T, MATHYPRE));
  hA = (Mat_HYPRE *)T->data;

  /* create HYPRE_IJMatrix */
  PetscCallExternal(HYPRE_IJMatrixCreate, hA->comm, rstart, rend, cstart, cend, &hA->ij);
  PetscCallExternal(HYPRE_IJMatrixSetObjectType, hA->ij, HYPRE_PARCSR);

  /* create new ParCSR object if needed */
  if (ishyp && copymode == PETSC_COPY_VALUES) {
    hypre_ParCSRMatrix *new_parcsr;
#if PETSC_PKG_HYPRE_VERSION_LT(2, 18, 0)
    hypre_CSRMatrix *hdiag, *hoffd, *ndiag, *noffd;

    new_parcsr = hypre_ParCSRMatrixClone(parcsr, 0);
    hdiag      = hypre_ParCSRMatrixDiag(parcsr);
    hoffd      = hypre_ParCSRMatrixOffd(parcsr);
    ndiag      = hypre_ParCSRMatrixDiag(new_parcsr);
    noffd      = hypre_ParCSRMatrixOffd(new_parcsr);
    PetscCall(PetscArraycpy(hypre_CSRMatrixData(ndiag), hypre_CSRMatrixData(hdiag), hypre_CSRMatrixNumNonzeros(hdiag)));
    PetscCall(PetscArraycpy(hypre_CSRMatrixData(noffd), hypre_CSRMatrixData(hoffd), hypre_CSRMatrixNumNonzeros(hoffd)));
#else
    new_parcsr = hypre_ParCSRMatrixClone(parcsr, 1);
#endif
    parcsr   = new_parcsr;
    copymode = PETSC_OWN_POINTER;
  }

  /* set ParCSR object */
  hypre_IJMatrixObject(hA->ij) = parcsr;
  T->preallocated              = PETSC_TRUE;

  /* set assembled flag */
  hypre_IJMatrixAssembleFlag(hA->ij) = 1;
#if 0
  PetscCallExternal(HYPRE_IJMatrixInitialize,hA->ij);
#endif
  if (ishyp) {
    PetscMPIInt myid = 0;

    /* make sure we always have row_starts and col_starts available */
    if (HYPRE_AssumedPartitionCheck()) PetscCallMPI(MPI_Comm_rank(comm, &myid));
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
    if (!hypre_ParCSRMatrixOwnsColStarts(parcsr)) {
      PetscLayout map;

      PetscCall(MatGetLayouts(T, NULL, &map));
      PetscCall(PetscLayoutSetUp(map));
      hypre_ParCSRMatrixColStarts(parcsr) = (HYPRE_BigInt *)(map->range + myid);
    }
    if (!hypre_ParCSRMatrixOwnsRowStarts(parcsr)) {
      PetscLayout map;

      PetscCall(MatGetLayouts(T, &map, NULL));
      PetscCall(PetscLayoutSetUp(map));
      hypre_ParCSRMatrixRowStarts(parcsr) = (HYPRE_BigInt *)(map->range + myid);
    }
#endif
    /* prevent from freeing the pointer */
    if (copymode == PETSC_USE_POINTER) hA->inner_free = PETSC_FALSE;
    *A = T;
    PetscCall(MatSetOption(*A, MAT_SORTED_FULL, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  } else if (isaij) {
    if (copymode != PETSC_OWN_POINTER) {
      /* prevent from freeing the pointer */
      hA->inner_free = PETSC_FALSE;
      PetscCall(MatConvert_HYPRE_AIJ(T, MATAIJ, MAT_INITIAL_MATRIX, A));
      PetscCall(MatDestroy(&T));
    } else { /* AIJ return type with PETSC_OWN_POINTER */
      PetscCall(MatConvert_HYPRE_AIJ(T, MATAIJ, MAT_INPLACE_MATRIX, &T));
      *A = T;
    }
  } else if (isis) {
    PetscCall(MatConvert_HYPRE_IS(T, MATIS, MAT_INITIAL_MATRIX, A));
    if (copymode != PETSC_OWN_POINTER) hA->inner_free = PETSC_FALSE;
    PetscCall(MatDestroy(&T));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHYPREGetParCSR_HYPRE(Mat A, hypre_ParCSRMatrix **parcsr)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;
  HYPRE_Int  type;

  PetscFunctionBegin;
  PetscCheck(hA->ij, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "HYPRE_IJMatrix not present");
  PetscCallExternal(HYPRE_IJMatrixGetObjectType, hA->ij, &type);
  PetscCheck(type == HYPRE_PARCSR, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "HYPRE_IJMatrix is not of type HYPRE_PARCSR");
  PetscCallExternal(HYPRE_IJMatrixGetObject, hA->ij, (void **)parcsr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatHYPREGetParCSR - Gets the pointer to the ParCSR matrix

  Not Collective, No Fortran Support

  Input Parameter:
. A - the `MATHYPRE` object

  Output Parameter:
. parcsr - the pointer to the `hypre_ParCSRMatrix`

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATHYPRE`, `PetscCopyMode`
@*/
PetscErrorCode MatHYPREGetParCSR(Mat A, hypre_ParCSRMatrix **parcsr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscUseMethod(A, "MatHYPREGetParCSR_C", (Mat, hypre_ParCSRMatrix **), (A, parcsr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMissingDiagonal_HYPRE(Mat A, PetscBool *missing, PetscInt *dd)
{
  hypre_ParCSRMatrix *parcsr;
  hypre_CSRMatrix    *ha;
  PetscInt            rst;

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented with non-square diagonal blocks");
  PetscCall(MatGetOwnershipRange(A, &rst, NULL));
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  if (missing) *missing = PETSC_FALSE;
  if (dd) *dd = -1;
  ha = hypre_ParCSRMatrixDiag(parcsr);
  if (ha) {
    PetscInt   size, i;
    HYPRE_Int *ii, *jj;

    size = hypre_CSRMatrixNumRows(ha);
    ii   = hypre_CSRMatrixI(ha);
    jj   = hypre_CSRMatrixJ(ha);
    for (i = 0; i < size; i++) {
      PetscInt  j;
      PetscBool found = PETSC_FALSE;

      for (j = ii[i]; j < ii[i + 1] && !found; j++) found = (jj[j] == i) ? PETSC_TRUE : PETSC_FALSE;

      if (!found) {
        PetscCall(PetscInfo(A, "Matrix is missing local diagonal entry %" PetscInt_FMT "\n", i));
        if (missing) *missing = PETSC_TRUE;
        if (dd) *dd = i + rst;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    if (!size) {
      PetscCall(PetscInfo(A, "Matrix has no diagonal entries therefore is missing diagonal\n"));
      if (missing) *missing = PETSC_TRUE;
      if (dd) *dd = rst;
    }
  } else {
    PetscCall(PetscInfo(A, "Matrix has no diagonal entries therefore is missing diagonal\n"));
    if (missing) *missing = PETSC_TRUE;
    if (dd) *dd = rst;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_HYPRE(Mat A, PetscScalar s)
{
  hypre_ParCSRMatrix *parcsr;
#if PETSC_PKG_HYPRE_VERSION_LT(2, 19, 0)
  hypre_CSRMatrix *ha;
#endif
  HYPRE_Complex hs;

  PetscFunctionBegin;
  PetscCall(PetscHYPREScalarCast(s, &hs));
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
#if PETSC_PKG_HYPRE_VERSION_GE(2, 19, 0)
  PetscCallExternal(hypre_ParCSRMatrixScale, parcsr, hs);
#else /* diagonal part */
  ha = hypre_ParCSRMatrixDiag(parcsr);
  if (ha) {
    PetscInt       size, i;
    HYPRE_Int     *ii;
    HYPRE_Complex *a;

    size = hypre_CSRMatrixNumRows(ha);
    a    = hypre_CSRMatrixData(ha);
    ii   = hypre_CSRMatrixI(ha);
    for (i = 0; i < ii[size]; i++) a[i] *= hs;
  }
  /* off-diagonal part */
  ha = hypre_ParCSRMatrixOffd(parcsr);
  if (ha) {
    PetscInt       size, i;
    HYPRE_Int     *ii;
    HYPRE_Complex *a;

    size = hypre_CSRMatrixNumRows(ha);
    a    = hypre_CSRMatrixData(ha);
    ii   = hypre_CSRMatrixI(ha);
    for (i = 0; i < ii[size]; i++) a[i] *= hs;
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroRowsColumns_HYPRE(Mat A, PetscInt numRows, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  hypre_ParCSRMatrix *parcsr;
  HYPRE_Int          *lrows;
  PetscInt            rst, ren, i;

  PetscFunctionBegin;
  PetscCheck(!x && !b, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "To be implemented");
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  PetscCall(PetscMalloc1(numRows, &lrows));
  PetscCall(MatGetOwnershipRange(A, &rst, &ren));
  for (i = 0; i < numRows; i++) {
    PetscCheck(rows[i] >= rst && rows[i] < ren, PETSC_COMM_SELF, PETSC_ERR_SUP, "Non-local rows not yet supported");
    lrows[i] = rows[i] - rst;
  }
  PetscCallExternal(hypre_ParCSRMatrixEliminateRowsCols, parcsr, numRows, lrows);
  PetscCall(PetscFree(lrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_HYPRE_CSRMatrix(hypre_CSRMatrix *ha)
{
  PetscFunctionBegin;
  if (ha) {
    HYPRE_Int     *ii, size;
    HYPRE_Complex *a;

    size = hypre_CSRMatrixNumRows(ha);
    a    = hypre_CSRMatrixData(ha);
    ii   = hypre_CSRMatrixI(ha);

    if (a) PetscCall(PetscArrayzero(a, ii[size]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_HYPRE(Mat A)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;

  PetscFunctionBegin;
  if (HYPRE_MEMORY_DEVICE == hypre_IJMatrixMemoryLocation(hA->ij)) {
    PetscCallExternal(HYPRE_IJMatrixSetConstantValues, hA->ij, 0.0);
  } else {
    hypre_ParCSRMatrix *parcsr;

    PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
    PetscCall(MatZeroEntries_HYPRE_CSRMatrix(hypre_ParCSRMatrixDiag(parcsr)));
    PetscCall(MatZeroEntries_HYPRE_CSRMatrix(hypre_ParCSRMatrixOffd(parcsr)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroRows_HYPRE_CSRMatrix(hypre_CSRMatrix *hA, PetscInt N, const PetscInt rows[], HYPRE_Complex diag)
{
  PetscInt       ii;
  HYPRE_Int     *i, *j;
  HYPRE_Complex *a;

  PetscFunctionBegin;
  if (!hA) PetscFunctionReturn(PETSC_SUCCESS);

  i = hypre_CSRMatrixI(hA);
  j = hypre_CSRMatrixJ(hA);
  a = hypre_CSRMatrixData(hA);
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_DEVICE == hypre_CSRMatrixMemoryLocation(hA)) {
  #if defined(HYPRE_USING_CUDA)
    MatZeroRows_CUDA(N, rows, i, j, a, diag);
  #elif defined(HYPRE_USING_HIP)
    MatZeroRows_HIP(N, rows, i, j, a, diag);
  #elif defined(PETSC_HAVE_KOKKOS)
    MatZeroRows_Kokkos(N, rows, i, j, a, diag);
  #else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for MatZeroRows on a hypre matrix in this memory location");
  #endif
  } else
#endif
  {
    for (ii = 0; ii < N; ii++) {
      HYPRE_Int jj, ibeg, iend, irow;

      irow = rows[ii];
      ibeg = i[irow];
      iend = i[irow + 1];
      for (jj = ibeg; jj < iend; jj++)
        if (j[jj] == irow) a[jj] = diag;
        else a[jj] = 0.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroRows_HYPRE(Mat A, PetscInt N, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  hypre_ParCSRMatrix *parcsr;
  PetscInt           *lrows, len, *lrows2;
  HYPRE_Complex       hdiag;

  PetscFunctionBegin;
  PetscCheck(!x && !b, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Does not support to modify the solution and the right hand size");
  PetscCall(PetscHYPREScalarCast(diag, &hdiag));
  /* retrieve the internal matrix */
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  /* get locally owned rows */
  PetscCall(MatZeroRowsMapLocal_Private(A, N, rows, &len, &lrows));

#if defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_DEVICE == hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(parcsr))) {
    Mat_HYPRE *hA = (Mat_HYPRE *)A->data;
    PetscInt   m;
    PetscCall(MatGetLocalSize(A, &m, NULL));
    if (!hA->rows_d) {
      hA->rows_d = hypre_TAlloc(PetscInt, m, HYPRE_MEMORY_DEVICE);
      if (m) PetscCheck(hA->rows_d, PETSC_COMM_SELF, PETSC_ERR_MEM, "HYPRE_TAlloc failed");
    }
    PetscCheck(len <= m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Too many rows in rows[]");
    PetscStackCallExternalVoid("hypre_Memcpy", hypre_Memcpy(hA->rows_d, lrows, sizeof(PetscInt) * len, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST));
    lrows2 = hA->rows_d;
  } else
#endif
  {
    lrows2 = lrows;
  }

  /* zero diagonal part */
  PetscCall(MatZeroRows_HYPRE_CSRMatrix(hypre_ParCSRMatrixDiag(parcsr), len, lrows2, hdiag));
  /* zero off-diagonal part */
  PetscCall(MatZeroRows_HYPRE_CSRMatrix(hypre_ParCSRMatrixOffd(parcsr), len, lrows2, 0.0));

  PetscCall(PetscFree(lrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyBegin_HYPRE(Mat mat, MatAssemblyType mode)
{
  PetscFunctionBegin;
  if (mat->nooffprocentries) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatStashScatterBegin_Private(mat, &mat->stash, mat->rmap->range));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRow_HYPRE(Mat A, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  hypre_ParCSRMatrix *parcsr;
  HYPRE_Int           hnz;

  PetscFunctionBegin;
  /* retrieve the internal matrix */
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  /* call HYPRE API */
  PetscCallExternal(HYPRE_ParCSRMatrixGetRow, parcsr, row, &hnz, (HYPRE_BigInt **)idx, (HYPRE_Complex **)v);
  if (nz) *nz = (PetscInt)hnz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRow_HYPRE(Mat A, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  hypre_ParCSRMatrix *parcsr;
  HYPRE_Int           hnz;

  PetscFunctionBegin;
  /* retrieve the internal matrix */
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  /* call HYPRE API */
  hnz = nz ? (HYPRE_Int)(*nz) : 0;
  PetscCallExternal(HYPRE_ParCSRMatrixRestoreRow, parcsr, row, &hnz, (HYPRE_BigInt **)idx, (HYPRE_Complex **)v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetValues_HYPRE(Mat A, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], PetscScalar v[])
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;
  PetscInt   i;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(PETSC_SUCCESS);
  /* Ignore negative row indices
   * And negative column indices should be automatically ignored in hypre
   * */
  for (i = 0; i < m; i++) {
    if (idxm[i] >= 0) {
      HYPRE_Int hn = (HYPRE_Int)n;
      PetscCallExternal(HYPRE_IJMatrixGetValues, hA->ij, 1, &hn, (HYPRE_BigInt *)&idxm[i], (HYPRE_BigInt *)idxn, (HYPRE_Complex *)(v + i * n));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOption_HYPRE(Mat A, MatOption op, PetscBool flg)
{
  Mat_HYPRE *hA = (Mat_HYPRE *)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NO_OFF_PROC_ENTRIES:
    if (flg) PetscCallExternal(HYPRE_IJMatrixSetMaxOffProcElmts, hA->ij, 0);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    hA->donotstash = flg;
    break;
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_HYPRE(Mat A, PetscViewer view)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(view, &format));
  if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(PETSC_SUCCESS);
  if (format != PETSC_VIEWER_NATIVE) {
    Mat                 B;
    hypre_ParCSRMatrix *parcsr;
    PetscErrorCode (*mview)(Mat, PetscViewer) = NULL;

    PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
    PetscCall(MatCreateFromParCSR(parcsr, MATAIJ, PETSC_USE_POINTER, &B));
    PetscCall(MatGetOperation(B, MATOP_VIEW, (PetscErrorCodeFn **)&mview));
    PetscCheck(mview, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing view operation");
    PetscCall((*mview)(B, view));
    PetscCall(MatDestroy(&B));
  } else {
    Mat_HYPRE  *hA = (Mat_HYPRE *)A->data;
    PetscMPIInt size;
    PetscBool   isascii;
    const char *filename;

    /* HYPRE uses only text files */
    PetscCall(PetscObjectTypeCompare((PetscObject)view, PETSCVIEWERASCII, &isascii));
    PetscCheck(isascii, PetscObjectComm((PetscObject)view), PETSC_ERR_SUP, "PetscViewerType %s: native HYPRE format needs PETSCVIEWERASCII", ((PetscObject)view)->type_name);
    PetscCall(PetscViewerFileGetName(view, &filename));
    PetscCallExternal(HYPRE_IJMatrixPrint, hA->ij, filename);
    PetscCallMPI(MPI_Comm_size(hA->comm, &size));
    if (size > 1) {
      PetscCall(PetscViewerASCIIPrintf(view, "Matrix files: %s.%05d ... %s.%05d\n", filename, 0, filename, size - 1));
    } else {
      PetscCall(PetscViewerASCIIPrintf(view, "Matrix file: %s.%05d\n", filename, 0));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_HYPRE(Mat A, Mat B, MatStructure str)
{
  hypre_ParCSRMatrix *acsr, *bcsr;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN && A->ops->copy == B->ops->copy) {
    PetscCall(MatHYPREGetParCSR_HYPRE(A, &acsr));
    PetscCall(MatHYPREGetParCSR_HYPRE(B, &bcsr));
    PetscCallExternal(hypre_ParCSRMatrixCopy, acsr, bcsr, 1);
    PetscCall(MatSetOption(B, MAT_SORTED_FULL, PETSC_TRUE)); /* "perfect" preallocation, so no need for hypre_AuxParCSRMatrixNeedAux */
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  } else {
    PetscCall(MatCopy_Basic(A, B, str));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_HYPRE(Mat A, Vec d)
{
  hypre_ParCSRMatrix *parcsr;
  hypre_CSRMatrix    *dmat;
  HYPRE_Complex      *a;
  PetscBool           cong;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &cong));
  PetscCheck(cong, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only for square matrices with same local distributions of rows and columns");
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  dmat = hypre_ParCSRMatrixDiag(parcsr);
  if (dmat) {
#if defined(PETSC_HAVE_HYPRE_DEVICE)
    HYPRE_MemoryLocation mem = hypre_CSRMatrixMemoryLocation(dmat);
#else
    HYPRE_MemoryLocation mem = HYPRE_MEMORY_HOST;
#endif

    if (mem != HYPRE_MEMORY_HOST) PetscCall(VecGetArrayWriteAndMemType(d, (PetscScalar **)&a, NULL));
    else PetscCall(VecGetArrayWrite(d, (PetscScalar **)&a));
    hypre_CSRMatrixExtractDiagonal(dmat, a, 0);
    if (mem != HYPRE_MEMORY_HOST) PetscCall(VecRestoreArrayWriteAndMemType(d, (PetscScalar **)&a));
    else PetscCall(VecRestoreArrayWrite(d, (PetscScalar **)&a));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscblaslapack.h>

static PetscErrorCode MatAXPY_HYPRE(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  {
    Mat                 B;
    hypre_ParCSRMatrix *x, *y, *z;

    PetscCall(MatHYPREGetParCSR(Y, &y));
    PetscCall(MatHYPREGetParCSR(X, &x));
    PetscCallExternal(hypre_ParCSRMatrixAdd, 1.0, y, 1.0, x, &z);
    PetscCall(MatCreateFromParCSR(z, MATHYPRE, PETSC_OWN_POINTER, &B));
    PetscCall(MatHeaderMerge(Y, &B));
  }
#else
  if (str == SAME_NONZERO_PATTERN) {
    hypre_ParCSRMatrix *x, *y;
    hypre_CSRMatrix    *xloc, *yloc;
    PetscInt            xnnz, ynnz;
    HYPRE_Complex      *xarr, *yarr;
    PetscBLASInt        one = 1, bnz;

    PetscCall(MatHYPREGetParCSR(Y, &y));
    PetscCall(MatHYPREGetParCSR(X, &x));

    /* diagonal block */
    xloc = hypre_ParCSRMatrixDiag(x);
    yloc = hypre_ParCSRMatrixDiag(y);
    xnnz = 0;
    ynnz = 0;
    xarr = NULL;
    yarr = NULL;
    if (xloc) {
      xarr = hypre_CSRMatrixData(xloc);
      xnnz = hypre_CSRMatrixNumNonzeros(xloc);
    }
    if (yloc) {
      yarr = hypre_CSRMatrixData(yloc);
      ynnz = hypre_CSRMatrixNumNonzeros(yloc);
    }
    PetscCheck(xnnz == ynnz, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Different number of nonzeros in diagonal block %" PetscInt_FMT " != %" PetscInt_FMT, xnnz, ynnz);
    PetscCall(PetscBLASIntCast(xnnz, &bnz));
    PetscCallBLAS("BLASaxpy", BLASaxpy_(&bnz, &a, (PetscScalar *)xarr, &one, (PetscScalar *)yarr, &one));

    /* off-diagonal block */
    xloc = hypre_ParCSRMatrixOffd(x);
    yloc = hypre_ParCSRMatrixOffd(y);
    xnnz = 0;
    ynnz = 0;
    xarr = NULL;
    yarr = NULL;
    if (xloc) {
      xarr = hypre_CSRMatrixData(xloc);
      xnnz = hypre_CSRMatrixNumNonzeros(xloc);
    }
    if (yloc) {
      yarr = hypre_CSRMatrixData(yloc);
      ynnz = hypre_CSRMatrixNumNonzeros(yloc);
    }
    PetscCheck(xnnz == ynnz, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Different number of nonzeros in off-diagonal block %" PetscInt_FMT " != %" PetscInt_FMT, xnnz, ynnz);
    PetscCall(PetscBLASIntCast(xnnz, &bnz));
    PetscCallBLAS("BLASaxpy", BLASaxpy_(&bnz, &a, (PetscScalar *)xarr, &one, (PetscScalar *)yarr, &one));
  } else if (str == SUBSET_NONZERO_PATTERN) {
    PetscCall(MatAXPY_Basic(Y, a, X, str));
  } else {
    Mat B;

    PetscCall(MatAXPY_Basic_Preallocate(Y, X, &B));
    PetscCall(MatAXPY_BasicWithPreallocation(B, Y, a, X, str));
    PetscCall(MatHeaderReplace(Y, &B));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_HYPRE(Mat A, MatDuplicateOption op, Mat *B)
{
  hypre_ParCSRMatrix *parcsr = NULL;
  PetscCopyMode       cpmode;
  Mat_HYPRE          *hA;

  PetscFunctionBegin;
  PetscCall(MatHYPREGetParCSR_HYPRE(A, &parcsr));
  if (op == MAT_DO_NOT_COPY_VALUES || op == MAT_SHARE_NONZERO_PATTERN) {
    parcsr = hypre_ParCSRMatrixClone(parcsr, 0);
    cpmode = PETSC_OWN_POINTER;
  } else {
    cpmode = PETSC_COPY_VALUES;
  }
  PetscCall(MatCreateFromParCSR(parcsr, MATHYPRE, cpmode, B));
  hA = (Mat_HYPRE *)A->data;
  if (hA->cooMat) {
    Mat_HYPRE *hB = (Mat_HYPRE *)((*B)->data);
    op            = (op == MAT_DO_NOT_COPY_VALUES) ? op : MAT_COPY_VALUES;
    /* Cannot simply increase the reference count of hA->cooMat, since B needs to share cooMat's data array */
    PetscCall(MatDuplicate(hA->cooMat, op, &hB->cooMat));
    PetscCall(MatHYPRE_AttachCOOMat(*B));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_HYPRE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_HYPRE *hmat = (Mat_HYPRE *)mat->data;

  PetscFunctionBegin;
  /* Build an agent matrix cooMat with AIJ format
     It has the same sparsity pattern as mat, and also shares the data array with mat. We use cooMat to do the COO work.
   */
  PetscCall(MatHYPRE_CreateCOOMat(mat));
  PetscCall(MatSetOption(hmat->cooMat, MAT_IGNORE_OFF_PROC_ENTRIES, hmat->donotstash));
  PetscCall(MatSetOption(hmat->cooMat, MAT_NO_OFF_PROC_ENTRIES, mat->nooffprocentries));

  /* MatSetPreallocationCOO_SeqAIJ and MatSetPreallocationCOO_MPIAIJ uses this specific
     name to automatically put the diagonal entries first */
  PetscCall(PetscObjectSetName((PetscObject)hmat->cooMat, "_internal_COO_mat_for_hypre"));
  PetscCall(MatSetPreallocationCOO(hmat->cooMat, coo_n, coo_i, coo_j));
  hmat->cooMat->assembled = PETSC_TRUE;

  /* Copy the sparsity pattern from cooMat to hypre IJMatrix hmat->ij */
  PetscCall(MatSetOption(mat, MAT_SORTED_FULL, PETSC_TRUE));
  PetscCall(MatHYPRE_CreateFromMat(hmat->cooMat, hmat));      /* Create hmat->ij and preallocate it */
  PetscCall(MatHYPRE_IJMatrixCopyIJ(hmat->cooMat, hmat->ij)); /* Copy A's (i,j) to hmat->ij */

  mat->preallocated = PETSC_TRUE;
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)); /* Migrate mat to device if it is bound to. Hypre builds its own SpMV context here */

  /* Attach cooMat to mat */
  PetscCall(MatHYPRE_AttachCOOMat(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_HYPRE(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_HYPRE *hmat = (Mat_HYPRE *)mat->data;

  PetscFunctionBegin;
  PetscCheck(hmat->cooMat, PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "HYPRE COO delegate matrix has not been created yet");
  PetscCall(MatSetValuesCOO(hmat->cooMat, v, imode));
  PetscCall(MatViewFromOptions(hmat->cooMat, (PetscObject)mat, "-cooMat_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetCurrentMemType_HYPRE(Mat A, PetscMemType *m)
{
  PetscBool petsconcpu;

  PetscFunctionBegin;
  PetscCall(MatBoundToCPU(A, &petsconcpu));
  *m = petsconcpu ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATHYPRE - "hypre" - A matrix type to be used for sequential and parallel sparse matrices
          based on the hypre IJ interface.

   Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatHYPRESetPreallocation`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_HYPRE(Mat B)
{
  Mat_HYPRE *hB;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  HYPRE_MemoryLocation memory_location;
#endif

  PetscFunctionBegin;
  PetscHYPREInitialize();
  PetscCall(PetscNew(&hB));

  hB->inner_free      = PETSC_TRUE;
  hB->array_available = PETSC_TRUE;

  B->data = (void *)hB;

  PetscCall(PetscMemzero(B->ops, sizeof(struct _MatOps)));
  B->ops->mult                  = MatMult_HYPRE;
  B->ops->multtranspose         = MatMultTranspose_HYPRE;
  B->ops->multadd               = MatMultAdd_HYPRE;
  B->ops->multtransposeadd      = MatMultTransposeAdd_HYPRE;
  B->ops->setup                 = MatSetUp_HYPRE;
  B->ops->destroy               = MatDestroy_HYPRE;
  B->ops->assemblyend           = MatAssemblyEnd_HYPRE;
  B->ops->assemblybegin         = MatAssemblyBegin_HYPRE;
  B->ops->setvalues             = MatSetValues_HYPRE;
  B->ops->missingdiagonal       = MatMissingDiagonal_HYPRE;
  B->ops->scale                 = MatScale_HYPRE;
  B->ops->zerorowscolumns       = MatZeroRowsColumns_HYPRE;
  B->ops->zeroentries           = MatZeroEntries_HYPRE;
  B->ops->zerorows              = MatZeroRows_HYPRE;
  B->ops->getrow                = MatGetRow_HYPRE;
  B->ops->restorerow            = MatRestoreRow_HYPRE;
  B->ops->getvalues             = MatGetValues_HYPRE;
  B->ops->setoption             = MatSetOption_HYPRE;
  B->ops->duplicate             = MatDuplicate_HYPRE;
  B->ops->copy                  = MatCopy_HYPRE;
  B->ops->view                  = MatView_HYPRE;
  B->ops->getdiagonal           = MatGetDiagonal_HYPRE;
  B->ops->axpy                  = MatAXPY_HYPRE;
  B->ops->productsetfromoptions = MatProductSetFromOptions_HYPRE;
  B->ops->getcurrentmemtype     = MatGetCurrentMemType_HYPRE;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  B->ops->bindtocpu = MatBindToCPU_HYPRE;
  /* Get hypre's default memory location. Users can control this using the corresponding HYPRE_SetMemoryLocation API */
  PetscCallExternal(HYPRE_GetMemoryLocation, &memory_location);
  B->boundtocpu = (memory_location == HYPRE_MEMORY_HOST) ? PETSC_TRUE : PETSC_FALSE;
#endif

  /* build cache for off array entries formed */
  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)B), 1, &B->stash));

  PetscCall(PetscCommGetComm(PetscObjectComm((PetscObject)B), &hB->comm));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATHYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_hypre_aij_C", MatConvert_HYPRE_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_hypre_is_C", MatConvert_HYPRE_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_hypre_C", MatProductSetFromOptions_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpiaij_hypre_C", MatProductSetFromOptions_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatHYPRESetPreallocation_C", MatHYPRESetPreallocation_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatHYPREGetParCSR_C", MatHYPREGetParCSR_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetValuesCOO_C", MatSetValuesCOO_HYPRE));
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  #if defined(HYPRE_USING_HIP)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaijhipsparse_hypre_C", MatProductSetFromOptions_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpiaijhipsparse_hypre_C", MatProductSetFromOptions_HYPRE));
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  PetscCall(MatSetVecType(B, VECHIP));
  #endif
  #if defined(HYPRE_USING_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaijcusparse_hypre_C", MatProductSetFromOptions_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpiaijcusparse_hypre_C", MatProductSetFromOptions_HYPRE));
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatSetVecType(B, VECCUDA));
  #endif
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
