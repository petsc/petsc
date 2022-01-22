
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

#if PETSC_PKG_HYPRE_VERSION_LT(2,18,0)
#define  hypre_ParCSRMatrixClone(A,B) hypre_ParCSRMatrixCompleteClone(A)
#endif

static PetscErrorCode MatHYPRE_CreateFromMat(Mat,Mat_HYPRE*);
static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat,Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat,HYPRE_Complex,Vec,HYPRE_Complex,Vec,PetscBool);
static PetscErrorCode hypre_array_destroy(void*);
static PetscErrorCode MatSetValues_HYPRE(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode ins);

static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat A_d, Mat A_o, HYPRE_IJMatrix ij)
{
  PetscErrorCode ierr;
  PetscInt       i,n_d,n_o;
  const PetscInt *ia_d,*ia_o;
  PetscBool      done_d=PETSC_FALSE,done_o=PETSC_FALSE;
  HYPRE_Int      *nnz_d=NULL,*nnz_o=NULL;

  PetscFunctionBegin;
  if (A_d) { /* determine number of nonzero entries in local diagonal part */
    ierr = MatGetRowIJ(A_d,0,PETSC_FALSE,PETSC_FALSE,&n_d,&ia_d,NULL,&done_d);CHKERRQ(ierr);
    if (done_d) {
      ierr = PetscMalloc1(n_d,&nnz_d);CHKERRQ(ierr);
      for (i=0; i<n_d; i++) {
        nnz_d[i] = ia_d[i+1] - ia_d[i];
      }
    }
    ierr = MatRestoreRowIJ(A_d,0,PETSC_FALSE,PETSC_FALSE,NULL,&ia_d,NULL,&done_d);CHKERRQ(ierr);
  }
  if (A_o) { /* determine number of nonzero entries in local off-diagonal part */
    ierr = MatGetRowIJ(A_o,0,PETSC_FALSE,PETSC_FALSE,&n_o,&ia_o,NULL,&done_o);CHKERRQ(ierr);
    if (done_o) {
      ierr = PetscMalloc1(n_o,&nnz_o);CHKERRQ(ierr);
      for (i=0; i<n_o; i++) {
        nnz_o[i] = ia_o[i+1] - ia_o[i];
      }
    }
    ierr = MatRestoreRowIJ(A_o,0,PETSC_FALSE,PETSC_FALSE,&n_o,&ia_o,NULL,&done_o);CHKERRQ(ierr);
  }
  if (done_d) {    /* set number of nonzeros in HYPRE IJ matrix */
    if (!done_o) { /* only diagonal part */
      ierr = PetscCalloc1(n_d,&nnz_o);CHKERRQ(ierr);
    }
#if PETSC_PKG_HYPRE_VERSION_GE(2,16,0)
    { /* If we don't do this, the columns of the matrix will be all zeros! */
      hypre_AuxParCSRMatrix *aux_matrix;
      aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
      hypre_AuxParCSRMatrixDestroy(aux_matrix);
      hypre_IJMatrixTranslator(ij) = NULL;
      PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,(ij,nnz_d,nnz_o));
      /* it seems they partially fixed it in 2.19.0 */
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
      aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
      hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 1;
#endif
    }
#else
    PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,(ij,nnz_d,nnz_o));
#endif
    ierr = PetscFree(nnz_d);CHKERRQ(ierr);
    ierr = PetscFree(nnz_o);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHYPRE_CreateFromMat(Mat A, Mat_HYPRE *hA)
{
  PetscErrorCode ierr;
  PetscInt       rstart,rend,cstart,cend;

  PetscFunctionBegin;
  ierr   = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  rstart = A->rmap->rstart;
  rend   = A->rmap->rend;
  cstart = A->cmap->rstart;
  cend   = A->cmap->rend;
  PetscStackCallStandard(HYPRE_IJMatrixCreate,(hA->comm,rstart,rend-1,cstart,cend-1,&hA->ij));
  PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,(hA->ij,HYPRE_PARCSR));
  {
    PetscBool      same;
    Mat            A_d,A_o;
    const PetscInt *colmap;
    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIAIJGetSeqAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIBAIJGetSeqBAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,NULL,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,NULL,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHYPRE_IJMatrixCopy(Mat A, HYPRE_IJMatrix ij)
{
  PetscErrorCode    ierr;
  PetscInt          i,rstart,rend,ncols,nr,nc;
  const PetscScalar *values;
  const PetscInt    *cols;
  PetscBool         flg;

  PetscFunctionBegin;
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(ij));
#else
  PetscStackCallStandard(HYPRE_IJMatrixInitialize_v2,(ij,HYPRE_MEMORY_HOST));
#endif
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  ierr = MatGetSize(A,&nr,&nc);CHKERRQ(ierr);
  if (flg && nr == nc) {
    ierr = MatHYPRE_IJMatrixFastCopy_MPIAIJ(A,ij);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatHYPRE_IJMatrixFastCopy_SeqAIJ(A,ij);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
    if (ncols) {
      HYPRE_Int nc = (HYPRE_Int)ncols;

      if ((PetscInt)nc != ncols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hypre overflow! number of columns %" PetscInt_FMT " for row %" PetscInt_FMT,ncols,i);
      PetscStackCallStandard(HYPRE_IJMatrixSetValues,(ij,1,&nc,(HYPRE_BigInt *)&i,(HYPRE_BigInt *)cols,(HYPRE_Complex *)values));
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat A, HYPRE_IJMatrix ij)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJ            *pdiag = (Mat_SeqAIJ*)A->data;
  HYPRE_Int             type;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag;
  PetscBool             sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  const PetscScalar     *pa;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(ij,(void**)&par_matrix));
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  /*
       this is the Hack part where we monkey directly with the hypre datastructures
  */
  if (sameint) {
    ierr = PetscArraycpy(hdiag->i,pdiag->i,A->rmap->n + 1);CHKERRQ(ierr);
    ierr = PetscArraycpy(hdiag->j,pdiag->j,pdiag->nz);CHKERRQ(ierr);
  } else {
    PetscInt i;

    for (i=0;i<A->rmap->n + 1;i++) hdiag->i[i] = (HYPRE_Int)pdiag->i[i];
    for (i=0;i<pdiag->nz;i++)      hdiag->j[i] = (HYPRE_Int)pdiag->j[i];
  }

  ierr = MatSeqAIJGetArrayRead(A,&pa);CHKERRQ(ierr);
  ierr = PetscArraycpy(hdiag->data,pa,pdiag->nz);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(A,&pa);CHKERRQ(ierr);

  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat A, HYPRE_IJMatrix ij)
{
  PetscErrorCode        ierr;
  Mat_MPIAIJ            *pA = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ            *pdiag,*poffd;
  PetscInt              i,*garray = pA->garray,*jj,cstart,*pjj;
  HYPRE_Int             *hjj,type;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag,*hoffd;
  PetscBool             sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  const PetscScalar     *pa;

  PetscFunctionBegin;
  pdiag = (Mat_SeqAIJ*) pA->A->data;
  poffd = (Mat_SeqAIJ*) pA->B->data;
  /* cstart is only valid for square MPIAIJ layed out in the usual way */
  ierr = MatGetOwnershipRange(A,&cstart,NULL);CHKERRQ(ierr);

  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(ij,(void**)&par_matrix));
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  hoffd = hypre_ParCSRMatrixOffd(par_matrix);

  /*
       this is the Hack part where we monkey directly with the hypre datastructures
  */
  if (sameint) {
    ierr = PetscArraycpy(hdiag->i,pdiag->i,pA->A->rmap->n + 1);CHKERRQ(ierr);
  } else {
    for (i=0; i<pA->A->rmap->n + 1; i++) hdiag->i[i] = (HYPRE_Int)(pdiag->i[i]);
  }
  /* need to shift the diag column indices (hdiag->j) back to global numbering since hypre is expecting this */
  hjj = hdiag->j;
  pjj = pdiag->j;
#if PETSC_PKG_HYPRE_VERSION_GE(2,16,0)
  for (i=0; i<pdiag->nz; i++) hjj[i] = pjj[i];
#else
  for (i=0; i<pdiag->nz; i++) hjj[i] = cstart + pjj[i];
#endif
  ierr = MatSeqAIJGetArrayRead(pA->A,&pa);CHKERRQ(ierr);
  ierr = PetscArraycpy(hdiag->data,pa,pdiag->nz);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(pA->A,&pa);CHKERRQ(ierr);
  if (sameint) {
    ierr = PetscArraycpy(hoffd->i,poffd->i,pA->A->rmap->n + 1);CHKERRQ(ierr);
  } else {
    for (i=0; i<pA->A->rmap->n + 1; i++) hoffd->i[i] = (HYPRE_Int)(poffd->i[i]);
  }

  /* need to move the offd column indices (hoffd->j) back to global numbering since hypre is expecting this
     If we hacked a hypre a bit more we might be able to avoid this step */
#if PETSC_PKG_HYPRE_VERSION_GE(2,16,0)
  PetscStackCallStandard(hypre_CSRMatrixBigInitialize,(hoffd));
  jj  = (PetscInt*) hoffd->big_j;
#else
  jj  = (PetscInt*) hoffd->j;
#endif
  pjj = poffd->j;
  for (i=0; i<poffd->nz; i++) jj[i] = garray[pjj[i]];

  ierr = MatSeqAIJGetArrayRead(pA->B,&pa);CHKERRQ(ierr);
  ierr = PetscArraycpy(hoffd->data,pa,poffd->nz);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(pA->B,&pa);CHKERRQ(ierr);

  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_HYPRE_IS(Mat A, MatType mtype, MatReuse reuse, Mat* B)
{
  Mat_HYPRE*             mhA = (Mat_HYPRE*)(A->data);
  Mat                    lA;
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  hypre_ParCSRMatrix     *hA;
  hypre_CSRMatrix        *hdiag,*hoffd;
  MPI_Comm               comm;
  HYPRE_Complex          *hdd,*hod,*aa;
  PetscScalar            *data;
  HYPRE_BigInt           *col_map_offd;
  HYPRE_Int              *hdi,*hdj,*hoi,*hoj;
  PetscInt               *ii,*jj,*iptr,*jptr;
  PetscInt               cum,dr,dc,oc,str,stc,nnz,i,jd,jo,M,N;
  HYPRE_Int              type;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A);
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(mhA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(comm,PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(mhA->ij,(void**)&hA));
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
  nnz  += hypre_CSRMatrixNumNonzeros(hoffd);
  hoi   = hypre_CSRMatrixI(hoffd);
  hoj   = hypre_CSRMatrixJ(hoffd);
  hod   = hypre_CSRMatrixData(hoffd);
  if (reuse != MAT_REUSE_MATRIX) {
    PetscInt *aux;

    /* generate l2g maps for rows and cols */
    ierr = ISCreateStride(comm,dr,str,1,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    col_map_offd = hypre_ParCSRMatrixColMapOffd(hA);
    ierr = PetscMalloc1(dc+oc,&aux);CHKERRQ(ierr);
    for (i=0; i<dc; i++) aux[i] = i+stc;
    for (i=0; i<oc; i++) aux[i+dc] = col_map_offd[i];
    ierr = ISCreateGeneral(comm,dc+oc,aux,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    /* create MATIS object */
    ierr = MatCreate(comm,B);CHKERRQ(ierr);
    ierr = MatSetSizes(*B,dr,dc,M,N);CHKERRQ(ierr);
    ierr = MatSetType(*B,MATIS);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(*B,rl2g,cl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rl2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cl2g);CHKERRQ(ierr);

    /* allocate CSR for local matrix */
    ierr = PetscMalloc1(dr+1,&iptr);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnz,&jptr);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnz,&data);CHKERRQ(ierr);
  } else {
    PetscInt  nr;
    PetscBool done;
    ierr = MatISGetLocalMat(*B,&lA);CHKERRQ(ierr);
    ierr = MatGetRowIJ(lA,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&iptr,(const PetscInt**)&jptr,&done);CHKERRQ(ierr);
    if (nr != dr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of rows in local mat! %" PetscInt_FMT " != %" PetscInt_FMT,nr,dr);
    if (iptr[nr] < nnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in local mat! reuse %" PetscInt_FMT " requested %" PetscInt_FMT,iptr[nr],nnz);
    ierr = MatSeqAIJGetArray(lA,&data);CHKERRQ(ierr);
  }
  /* merge local matrices */
  ii  = iptr;
  jj  = jptr;
  aa  = (HYPRE_Complex*)data; /* this cast fixes the clang error when doing the assignments below: implicit conversion from 'HYPRE_Complex' (aka '_Complex double') to 'double' is not permitted in C++ */
  *ii = *(hdi++) + *(hoi++);
  for (jd=0,jo=0,cum=0; *ii<nnz; cum++) {
    PetscScalar *aold = (PetscScalar*)aa;
    PetscInt    *jold = jj,nc = jd+jo;
    for (; jd<*hdi; jd++) { *jj++ = *hdj++;      *aa++ = *hdd++; }
    for (; jo<*hoi; jo++) { *jj++ = *hoj++ + dc; *aa++ = *hod++; }
    *(++ii) = *(hdi++) + *(hoi++);
    ierr = PetscSortIntWithScalarArray(jd+jo-nc,jold,aold);CHKERRQ(ierr);
  }
  for (; cum<dr; cum++) *(++ii) = nnz;
  if (reuse != MAT_REUSE_MATRIX) {
    Mat_SeqAIJ* a;

    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,dc+oc,iptr,jptr,data,&lA);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(*B,lA);CHKERRQ(ierr);
    /* hack SeqAIJ */
    a          = (Mat_SeqAIJ*)(lA->data);
    a->free_a  = PETSC_TRUE;
    a->free_ij = PETSC_TRUE;
    ierr = MatDestroy(&lA);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat A, MatType type, MatReuse reuse, Mat *B)
{
  Mat            M = NULL;
  Mat_HYPRE      *hB;
  MPI_Comm       comm = PetscObjectComm((PetscObject)A);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    /* always destroy the old matrix and create a new memory;
       hope this does not churn the memory too much. The problem
       is I do not know if it is possible to put the matrix back to
       its initial state so that we can directly copy the values
       the second time through. */
    hB = (Mat_HYPRE*)((*B)->data);
    PetscStackCallStandard(HYPRE_IJMatrixDestroy,(hB->ij));
  } else {
    ierr = MatCreate(comm,&M);CHKERRQ(ierr);
    ierr = MatSetType(M,MATHYPRE);CHKERRQ(ierr);
    ierr = MatSetSizes(M,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    hB   = (Mat_HYPRE*)(M->data);
    if (reuse == MAT_INITIAL_MATRIX) *B = M;
  }
  ierr = MatSetOption(*B,MAT_SORTED_FULL,PETSC_TRUE);CHKERRQ(ierr); /* "perfect" preallocation, so no need for hypre_AuxParCSRMatrixNeedAux */
  ierr = MatSetOption(*B,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatHYPRE_CreateFromMat(A,hB);CHKERRQ(ierr);
  ierr = MatHYPRE_IJMatrixCopy(A,hB->ij);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&M);CHKERRQ(ierr);
  }
  (*B)->preallocated = PETSC_TRUE;
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_HYPRE_AIJ(Mat A, MatType mtype, MatReuse reuse, Mat *B)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  hypre_ParCSRMatrix *parcsr;
  hypre_CSRMatrix    *hdiag,*hoffd;
  MPI_Comm           comm;
  PetscScalar        *da,*oa,*aptr;
  PetscInt           *dii,*djj,*oii,*ojj,*iptr;
  PetscInt           i,dnnz,onnz,m,n;
  HYPRE_Int          type;
  PetscMPIInt        size;
  PetscBool          sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A);
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(comm,PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool ismpiaij,isseqaij;
    ierr = PetscObjectBaseTypeCompare((PetscObject)*B,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
    ierr = PetscObjectBaseTypeCompare((PetscObject)*B,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    if (!ismpiaij && !isseqaij) SETERRQ(comm,PETSC_ERR_SUP,"Only MATMPIAIJ or MATSEQAIJ are supported");
  }
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_DEVICE == hypre_IJMatrixMemoryLocation(hA->ij)) SETERRQ(comm,PETSC_ERR_SUP,"Not yet implemented");
#endif
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&parcsr));
  hdiag = hypre_ParCSRMatrixDiag(parcsr);
  hoffd = hypre_ParCSRMatrixOffd(parcsr);
  m     = hypre_CSRMatrixNumRows(hdiag);
  n     = hypre_CSRMatrixNumCols(hdiag);
  dnnz  = hypre_CSRMatrixNumNonzeros(hdiag);
  onnz  = hypre_CSRMatrixNumNonzeros(hoffd);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc1(m+1,&dii);CHKERRQ(ierr);
    ierr = PetscMalloc1(dnnz,&djj);CHKERRQ(ierr);
    ierr = PetscMalloc1(dnnz,&da);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscInt  nr;
    PetscBool done;
    if (size > 1) {
      Mat_MPIAIJ *b = (Mat_MPIAIJ*)((*B)->data);

      ierr = MatGetRowIJ(b->A,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&dii,(const PetscInt**)&djj,&done);CHKERRQ(ierr);
      if (nr != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows in diag part! %" PetscInt_FMT " != %" PetscInt_FMT,nr,m);
      if (dii[nr] < dnnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in diag part! reuse %" PetscInt_FMT " hypre %" PetscInt_FMT,dii[nr],dnnz);
      ierr = MatSeqAIJGetArray(b->A,&da);CHKERRQ(ierr);
    } else {
      ierr = MatGetRowIJ(*B,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&dii,(const PetscInt**)&djj,&done);CHKERRQ(ierr);
      if (nr != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows! %" PetscInt_FMT " != %" PetscInt_FMT,nr,m);
      if (dii[nr] < dnnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros! reuse %" PetscInt_FMT " hypre %" PetscInt_FMT,dii[nr],dnnz);
      ierr = MatSeqAIJGetArray(*B,&da);CHKERRQ(ierr);
    }
  } else { /* MAT_INPLACE_MATRIX */
    if (!sameint) {
      ierr = PetscMalloc1(m+1,&dii);CHKERRQ(ierr);
      ierr = PetscMalloc1(dnnz,&djj);CHKERRQ(ierr);
    } else {
      dii = (PetscInt*)hypre_CSRMatrixI(hdiag);
      djj = (PetscInt*)hypre_CSRMatrixJ(hdiag);
    }
    da = (PetscScalar*)hypre_CSRMatrixData(hdiag);
  }

  if (!sameint) {
    for (i=0;i<m+1;i++)  dii[i] = (PetscInt)(hypre_CSRMatrixI(hdiag)[i]);
    for (i=0;i<dnnz;i++) djj[i] = (PetscInt)(hypre_CSRMatrixJ(hdiag)[i]);
  } else {
    ierr = PetscArraycpy(dii,hypre_CSRMatrixI(hdiag),m+1);CHKERRQ(ierr);
    ierr = PetscArraycpy(djj,hypre_CSRMatrixJ(hdiag),dnnz);CHKERRQ(ierr);
  }
  ierr = PetscArraycpy(da,hypre_CSRMatrixData(hdiag),dnnz);CHKERRQ(ierr);
  iptr = djj;
  aptr = da;
  for (i=0; i<m; i++) {
    PetscInt nc = dii[i+1]-dii[i];
    ierr = PetscSortIntWithScalarArray(nc,iptr,aptr);CHKERRQ(ierr);
    iptr += nc;
    aptr += nc;
  }
  if (size > 1) {
    HYPRE_BigInt *coffd;
    HYPRE_Int    *offdj;

    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = PetscMalloc1(m+1,&oii);CHKERRQ(ierr);
      ierr = PetscMalloc1(onnz,&ojj);CHKERRQ(ierr);
      ierr = PetscMalloc1(onnz,&oa);CHKERRQ(ierr);
    } else if (reuse == MAT_REUSE_MATRIX) {
      Mat_MPIAIJ *b = (Mat_MPIAIJ*)((*B)->data);
      PetscInt   nr,hr = hypre_CSRMatrixNumRows(hoffd);
      PetscBool  done;

      ierr = MatGetRowIJ(b->B,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&oii,(const PetscInt**)&ojj,&done);CHKERRQ(ierr);
      if (nr != hr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows in offdiag part! %" PetscInt_FMT " != %" PetscInt_FMT,nr,hr);
      if (oii[nr] < onnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in offdiag part! reuse %" PetscInt_FMT " hypre %" PetscInt_FMT,oii[nr],onnz);
      ierr = MatSeqAIJGetArray(b->B,&oa);CHKERRQ(ierr);
    } else { /* MAT_INPLACE_MATRIX */
      if (!sameint) {
        ierr = PetscMalloc1(m+1,&oii);CHKERRQ(ierr);
        ierr = PetscMalloc1(onnz,&ojj);CHKERRQ(ierr);
      } else {
        oii = (PetscInt*)hypre_CSRMatrixI(hoffd);
        ojj = (PetscInt*)hypre_CSRMatrixJ(hoffd);
      }
      oa = (PetscScalar*)hypre_CSRMatrixData(hoffd);
    }
    if (!sameint) {
      for (i=0;i<m+1;i++) oii[i] = (PetscInt)(hypre_CSRMatrixI(hoffd)[i]);
    } else {
      ierr = PetscArraycpy(oii,hypre_CSRMatrixI(hoffd),m+1);CHKERRQ(ierr);
    }
    offdj = hypre_CSRMatrixJ(hoffd);
    coffd = hypre_ParCSRMatrixColMapOffd(parcsr);
    for (i=0; i<onnz; i++) ojj[i] = coffd[offdj[i]];
    ierr = PetscArraycpy(oa,hypre_CSRMatrixData(hoffd),onnz);CHKERRQ(ierr);
    iptr = ojj;
    aptr = oa;
    for (i=0; i<m; i++) {
       PetscInt nc = oii[i+1]-oii[i];
       ierr = PetscSortIntWithScalarArray(nc,iptr,aptr);CHKERRQ(ierr);
       iptr += nc;
       aptr += nc;
    }
    if (reuse == MAT_INITIAL_MATRIX) {
      Mat_MPIAIJ *b;
      Mat_SeqAIJ *d,*o;

      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,djj,da,oii,ojj,oa,B);CHKERRQ(ierr);
      /* hack MPIAIJ */
      b          = (Mat_MPIAIJ*)((*B)->data);
      d          = (Mat_SeqAIJ*)b->A->data;
      o          = (Mat_SeqAIJ*)b->B->data;
      d->free_a  = PETSC_TRUE;
      d->free_ij = PETSC_TRUE;
      o->free_a  = PETSC_TRUE;
      o->free_ij = PETSC_TRUE;
    } else if (reuse == MAT_INPLACE_MATRIX) {
      Mat T;

      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,djj,da,oii,ojj,oa,&T);CHKERRQ(ierr);
      if (sameint) { /* ownership of CSR pointers is transferred to PETSc */
        hypre_CSRMatrixI(hdiag) = NULL;
        hypre_CSRMatrixJ(hdiag) = NULL;
        hypre_CSRMatrixI(hoffd) = NULL;
        hypre_CSRMatrixJ(hoffd) = NULL;
      } else { /* Hack MPIAIJ -> free ij but not a */
        Mat_MPIAIJ *b = (Mat_MPIAIJ*)(T->data);
        Mat_SeqAIJ *d = (Mat_SeqAIJ*)(b->A->data);
        Mat_SeqAIJ *o = (Mat_SeqAIJ*)(b->B->data);

        d->free_ij = PETSC_TRUE;
        o->free_ij = PETSC_TRUE;
      }
      hypre_CSRMatrixData(hdiag) = NULL;
      hypre_CSRMatrixData(hoffd) = NULL;
      ierr = MatHeaderReplace(A,&T);CHKERRQ(ierr);
    }
  } else {
    oii  = NULL;
    ojj  = NULL;
    oa   = NULL;
    if (reuse == MAT_INITIAL_MATRIX) {
      Mat_SeqAIJ* b;

      ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,B);CHKERRQ(ierr);
      /* hack SeqAIJ */
      b          = (Mat_SeqAIJ*)((*B)->data);
      b->free_a  = PETSC_TRUE;
      b->free_ij = PETSC_TRUE;
    } else if (reuse == MAT_INPLACE_MATRIX) {
      Mat T;

      ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,&T);CHKERRQ(ierr);
      if (sameint) { /* ownership of CSR pointers is transferred to PETSc */
        hypre_CSRMatrixI(hdiag) = NULL;
        hypre_CSRMatrixJ(hdiag) = NULL;
      } else { /* free ij but not a */
        Mat_SeqAIJ* b = (Mat_SeqAIJ*)(T->data);

        b->free_ij = PETSC_TRUE;
      }
      hypre_CSRMatrixData(hdiag) = NULL;
      ierr = MatHeaderReplace(A,&T);CHKERRQ(ierr);
    }
  }

  /* we have to use hypre_Tfree to free the HYPRE arrays
     that PETSc now onws */
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscInt nh;
    void *ptrs[6] = {da,oa,dii,djj,oii,ojj};
    const char *names[6] = {"_hypre_csr_da",
                            "_hypre_csr_oa",
                            "_hypre_csr_dii",
                            "_hypre_csr_djj",
                            "_hypre_csr_oii",
                            "_hypre_csr_ojj"};
    nh = sameint ? 6 : 2;
    for (i=0; i<nh; i++) {
      PetscContainer c;

      ierr = PetscContainerCreate(comm,&c);CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]);CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,hypre_array_destroy);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*B),names[i],(PetscObject)c);CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAIJGetParCSR_Private(Mat A, hypre_ParCSRMatrix **hA)
{
  hypre_ParCSRMatrix *tA;
  hypre_CSRMatrix    *hdiag,*hoffd;
  Mat_SeqAIJ         *diag,*offd;
  PetscInt           *garray,i,noffd,dnnz,onnz,*row_starts,*col_starts;
  MPI_Comm           comm = PetscObjectComm((PetscObject)A);
  PetscBool          ismpiaij,isseqaij;
  PetscBool          sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  PetscErrorCode     ierr;
  HYPRE_Int          *hdi = NULL,*hdj = NULL,*hoi = NULL,*hoj = NULL;
  PetscInt           *pdi = NULL,*pdj = NULL,*poi = NULL,*poj = NULL;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscBool          iscuda = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  if (!ismpiaij && !isseqaij) SETERRQ1(comm,PETSC_ERR_SUP,"Unsupported type %s",((PetscObject)A)->type_name);
  if (ismpiaij) {
    Mat_MPIAIJ *a = (Mat_MPIAIJ*)(A->data);

    diag = (Mat_SeqAIJ*)a->A->data;
    offd = (Mat_SeqAIJ*)a->B->data;
#if defined(PETSC_HAVE_CUDA) && defined(PETSC_HAVE_HYPRE_DEVICE) && defined(HYPRE_USING_CUDA)
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJCUSPARSE,&iscuda);CHKERRQ(ierr);
    if (iscuda && !A->boundtocpu) {
      sameint = PETSC_TRUE;
      ierr = MatSeqAIJCUSPARSEGetIJ(a->A,PETSC_FALSE,(const HYPRE_Int**)&hdi,(const HYPRE_Int**)&hdj);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSEGetIJ(a->B,PETSC_FALSE,(const HYPRE_Int**)&hoi,(const HYPRE_Int**)&hoj);CHKERRQ(ierr);
    } else {
#else
    {
#endif
      pdi = diag->i;
      pdj = diag->j;
      poi = offd->i;
      poj = offd->j;
      if (sameint) {
        hdi = (HYPRE_Int*)pdi;
        hdj = (HYPRE_Int*)pdj;
        hoi = (HYPRE_Int*)poi;
        hoj = (HYPRE_Int*)poj;
      }
    }
    garray = a->garray;
    noffd  = a->B->cmap->N;
    dnnz   = diag->nz;
    onnz   = offd->nz;
  } else {
    diag = (Mat_SeqAIJ*)A->data;
    offd = NULL;
#if defined(PETSC_HAVE_CUDA) && defined(PETSC_HAVE_HYPRE_DEVICE)
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJCUSPARSE,&iscuda);CHKERRQ(ierr);
    if (iscuda && !A->boundtocpu) {
      sameint = PETSC_TRUE;
      ierr = MatSeqAIJCUSPARSEGetIJ(A,PETSC_FALSE,(const HYPRE_Int**)&hdi,(const HYPRE_Int**)&hdj);CHKERRQ(ierr);
    } else {
#else
    {
#endif
      pdi = diag->i;
      pdj = diag->j;
      if (sameint) {
        hdi = (HYPRE_Int*)pdi;
        hdj = (HYPRE_Int*)pdj;
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

    ierr       = MPI_Comm_rank(comm,&myid);CHKERRMPI(ierr);
    row_starts = A->rmap->range + myid;
    col_starts = A->cmap->range + myid;
  } else {
    row_starts = A->rmap->range;
    col_starts = A->cmap->range;
  }
  tA = hypre_ParCSRMatrixCreate(comm,A->rmap->N,A->cmap->N,(HYPRE_BigInt*)row_starts,(HYPRE_BigInt*)col_starts,noffd,dnnz,onnz);
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  hypre_ParCSRMatrixSetRowStartsOwner(tA,0);
  hypre_ParCSRMatrixSetColStartsOwner(tA,0);
#endif

  /* set diagonal part */
  hdiag = hypre_ParCSRMatrixDiag(tA);
  if (!sameint) { /* malloc CSR pointers */
    ierr = PetscMalloc2(A->rmap->n+1,&hdi,dnnz,&hdj);CHKERRQ(ierr);
    for (i = 0; i < A->rmap->n+1; i++) hdi[i] = (HYPRE_Int)(pdi[i]);
    for (i = 0; i < dnnz; i++)         hdj[i] = (HYPRE_Int)(pdj[i]);
  }
  hypre_CSRMatrixI(hdiag)           = hdi;
  hypre_CSRMatrixJ(hdiag)           = hdj;
  hypre_CSRMatrixData(hdiag)        = (HYPRE_Complex*)diag->a;
  hypre_CSRMatrixNumNonzeros(hdiag) = diag->nz;
  hypre_CSRMatrixSetRownnz(hdiag);
  hypre_CSRMatrixSetDataOwner(hdiag,0);

  /* set offdiagonal part */
  hoffd = hypre_ParCSRMatrixOffd(tA);
  if (offd) {
    if (!sameint) { /* malloc CSR pointers */
      ierr = PetscMalloc2(A->rmap->n+1,&hoi,onnz,&hoj);CHKERRQ(ierr);
      for (i = 0; i < A->rmap->n+1; i++) hoi[i] = (HYPRE_Int)(poi[i]);
      for (i = 0; i < onnz; i++)         hoj[i] = (HYPRE_Int)(poj[i]);
    }
    hypre_CSRMatrixI(hoffd)           = hoi;
    hypre_CSRMatrixJ(hoffd)           = hoj;
    hypre_CSRMatrixData(hoffd)        = (HYPRE_Complex*)offd->a;
    hypre_CSRMatrixNumNonzeros(hoffd) = offd->nz;
    hypre_CSRMatrixSetRownnz(hoffd);
    hypre_CSRMatrixSetDataOwner(hoffd,0);
  }
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscStackCallStandard(hypre_ParCSRMatrixInitialize_v2,(tA,iscuda ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST));
#else
#if PETSC_PKG_HYPRE_VERSION_LT(2,18,0)
  PetscStackCallStandard(hypre_ParCSRMatrixInitialize,(tA));
#else
  PetscStackCallStandard(hypre_ParCSRMatrixInitialize_v2,(tA,HYPRE_MEMORY_HOST));
#endif
#endif
  hypre_TFree(hypre_ParCSRMatrixColMapOffd(tA),HYPRE_MEMORY_HOST);
  hypre_ParCSRMatrixSetNumNonzeros(tA);
  hypre_ParCSRMatrixColMapOffd(tA) = (HYPRE_BigInt*)garray;
  if (!hypre_ParCSRMatrixCommPkg(tA)) PetscStackCallStandard(hypre_MatvecCommPkgCreate,(tA));
  *hA = tA;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAIJRestoreParCSR_Private(Mat A, hypre_ParCSRMatrix **hA)
{
  hypre_CSRMatrix *hdiag,*hoffd;
  PetscBool       ismpiaij,sameint = (PetscBool)(sizeof(PetscInt) == sizeof(HYPRE_Int));
  PetscErrorCode  ierr;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscBool       iscuda = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  ierr = PetscObjectTypeCompareAny((PetscObject)A,&iscuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,"");CHKERRQ(ierr);
  if (iscuda) sameint = PETSC_TRUE;
#endif
  hdiag = hypre_ParCSRMatrixDiag(*hA);
  hoffd = hypre_ParCSRMatrixOffd(*hA);
  /* free temporary memory allocated by PETSc
     set pointers to NULL before destroying tA */
  if (!sameint) {
    HYPRE_Int *hi,*hj;

    hi = hypre_CSRMatrixI(hdiag);
    hj = hypre_CSRMatrixJ(hdiag);
    ierr = PetscFree2(hi,hj);CHKERRQ(ierr);
    if (ismpiaij) {
      hi = hypre_CSRMatrixI(hoffd);
      hj = hypre_CSRMatrixJ(hoffd);
      ierr = PetscFree2(hi,hj);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

/* calls RAP from BoomerAMG:
   the resulting ParCSR will not own the column and row starts
   It looks like we don't need to have the diagonal entries ordered first */
static PetscErrorCode MatHYPRE_ParCSR_RAP(hypre_ParCSRMatrix *hR, hypre_ParCSRMatrix *hA,hypre_ParCSRMatrix *hP, hypre_ParCSRMatrix **hRAP)
{
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  HYPRE_Int P_owns_col_starts,R_owns_row_starts;
#endif

  PetscFunctionBegin;
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  P_owns_col_starts = hypre_ParCSRMatrixOwnsColStarts(hP);
  R_owns_row_starts = hypre_ParCSRMatrixOwnsRowStarts(hR);
#endif
  /* can be replaced by version test later */
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscStackPush("hypre_ParCSRMatrixRAP");
  *hRAP = hypre_ParCSRMatrixRAP(hR,hA,hP);
  PetscStackPop;
#else
  PetscStackCallStandard(hypre_BoomerAMGBuildCoarseOperator,(hR,hA,hP,hRAP));
  PetscStackCallStandard(hypre_ParCSRMatrixSetNumNonzeros,(*hRAP));
#endif
  /* hypre_BoomerAMGBuildCoarseOperator steals the col_starts from P and the row_starts from R */
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
  hypre_ParCSRMatrixSetRowStartsOwner(*hRAP,0);
  hypre_ParCSRMatrixSetColStartsOwner(*hRAP,0);
  if (P_owns_col_starts) hypre_ParCSRMatrixSetColStartsOwner(hP,1);
  if (R_owns_row_starts) hypre_ParCSRMatrixSetRowStartsOwner(hR,1);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPtAPNumeric_AIJ_AIJ_wHYPRE(Mat A,Mat P,Mat C)
{
  Mat                B;
  hypre_ParCSRMatrix *hA,*hP,*hPtAP = NULL;
  PetscErrorCode     ierr;
  Mat_Product        *product=C->product;

  PetscFunctionBegin;
  ierr = MatAIJGetParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatAIJGetParCSR_Private(P,&hP);CHKERRQ(ierr);
  ierr = MatHYPRE_ParCSR_RAP(hP,hA,hP,&hPtAP);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(hPtAP,MATAIJ,PETSC_OWN_POINTER,&B);CHKERRQ(ierr);

  ierr = MatHeaderMerge(C,&B);CHKERRQ(ierr);
  C->product = product;

  ierr = MatAIJRestoreParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatAIJRestoreParCSR_Private(P,&hP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatPtAPSymbolic_AIJ_AIJ_wHYPRE(Mat A,Mat P,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                   = MatSetType(C,MATAIJ);CHKERRQ(ierr);
  C->ops->ptapnumeric    = MatPtAPNumeric_AIJ_AIJ_wHYPRE;
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPtAPNumeric_AIJ_HYPRE(Mat A,Mat P,Mat C)
{
  Mat                B;
  Mat_HYPRE          *hP;
  hypre_ParCSRMatrix *hA = NULL,*Pparcsr,*ptapparcsr = NULL;
  HYPRE_Int          type;
  MPI_Comm           comm = PetscObjectComm((PetscObject)A);
  PetscBool          ishypre;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)P,MATHYPRE,&ishypre);CHKERRQ(ierr);
  if (!ishypre) SETERRQ1(comm,PETSC_ERR_USER,"P should be of type %s",MATHYPRE);
  hP = (Mat_HYPRE*)P->data;
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hP->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(comm,PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hP->ij,(void**)&Pparcsr));

  ierr = MatAIJGetParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatHYPRE_ParCSR_RAP(Pparcsr,hA,Pparcsr,&ptapparcsr);CHKERRQ(ierr);
  ierr = MatAIJRestoreParCSR_Private(A,&hA);CHKERRQ(ierr);

  /* create temporary matrix and merge to C */
  ierr = MatCreateFromParCSR(ptapparcsr,((PetscObject)C)->type_name,PETSC_OWN_POINTER,&B);CHKERRQ(ierr);
  ierr = MatHeaderMerge(C,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPtAPNumeric_HYPRE_HYPRE(Mat A,Mat P,Mat C)
{
  Mat                B;
  hypre_ParCSRMatrix *Aparcsr,*Pparcsr,*ptapparcsr = NULL;
  Mat_HYPRE          *hA,*hP;
  PetscBool          ishypre;
  HYPRE_Int          type;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)P,MATHYPRE,&ishypre);CHKERRQ(ierr);
  if (!ishypre) SETERRQ1(PetscObjectComm((PetscObject)P),PETSC_ERR_USER,"P should be of type %s",MATHYPRE);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATHYPRE,&ishypre);CHKERRQ(ierr);
  if (!ishypre) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"A should be of type %s",MATHYPRE);
  hA = (Mat_HYPRE*)A->data;
  hP = (Mat_HYPRE*)P->data;
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hP->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)P),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&Aparcsr));
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hP->ij,(void**)&Pparcsr));
  ierr = MatHYPRE_ParCSR_RAP(Pparcsr,Aparcsr,Pparcsr,&ptapparcsr);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(ptapparcsr,MATHYPRE,PETSC_OWN_POINTER,&B);CHKERRQ(ierr);
  ierr = MatHeaderMerge(C,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
  PetscStackPush("hypre_ParCSRMatMat");
  *hAB = hypre_ParCSRMatMat(hA,hB);
#else
  PetscStackPush("hypre_ParMatmul");
  *hAB = hypre_ParMatmul(hA,hB);
#endif
  PetscStackPop;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_AIJ_AIJ_wHYPRE(Mat A,Mat B,Mat C)
{
  Mat                D;
  hypre_ParCSRMatrix *hA,*hB,*hAB = NULL;
  PetscErrorCode     ierr;
  Mat_Product        *product=C->product;

  PetscFunctionBegin;
  ierr = MatAIJGetParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatAIJGetParCSR_Private(B,&hB);CHKERRQ(ierr);
  ierr = MatHYPRE_ParCSR_MatMatMult(hA,hB,&hAB);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(hAB,MATAIJ,PETSC_OWN_POINTER,&D);CHKERRQ(ierr);

  ierr = MatHeaderMerge(C,&D);CHKERRQ(ierr);
  C->product = product;

  ierr = MatAIJRestoreParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatAIJRestoreParCSR_Private(B,&hB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMultSymbolic_AIJ_AIJ_wHYPRE(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(C,MATAIJ);CHKERRQ(ierr);
  C->ops->matmultnumeric = MatMatMultNumeric_AIJ_AIJ_wHYPRE;
  C->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_HYPRE_HYPRE(Mat A,Mat B,Mat C)
{
  Mat                D;
  hypre_ParCSRMatrix *Aparcsr,*Bparcsr,*ABparcsr = NULL;
  Mat_HYPRE          *hA,*hB;
  PetscBool          ishypre;
  HYPRE_Int          type;
  PetscErrorCode     ierr;
  Mat_Product        *product;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B,MATHYPRE,&ishypre);CHKERRQ(ierr);
  if (!ishypre) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_USER,"B should be of type %s",MATHYPRE);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATHYPRE,&ishypre);CHKERRQ(ierr);
  if (!ishypre) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"A should be of type %s",MATHYPRE);
  hA = (Mat_HYPRE*)A->data;
  hB = (Mat_HYPRE*)B->data;
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hB->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&Aparcsr));
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hB->ij,(void**)&Bparcsr));
  ierr = MatHYPRE_ParCSR_MatMatMult(Aparcsr,Bparcsr,&ABparcsr);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(ABparcsr,MATHYPRE,PETSC_OWN_POINTER,&D);CHKERRQ(ierr);

  /* need to use HeaderReplace because HeaderMerge messes up with the communicator */
  product    = C->product;  /* save it from MatHeaderReplace() */
  C->product = NULL;
  ierr = MatHeaderReplace(C,&D);CHKERRQ(ierr);
  C->product = product;
  C->ops->matmultnumeric = MatMatMultNumeric_HYPRE_HYPRE;
  C->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatTransposeMatMatMultNumeric_AIJ_AIJ_AIJ_wHYPRE(Mat A,Mat B,Mat C,Mat D)
{
  Mat                E;
  hypre_ParCSRMatrix *hA,*hB,*hC,*hABC = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatAIJGetParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatAIJGetParCSR_Private(B,&hB);CHKERRQ(ierr);
  ierr = MatAIJGetParCSR_Private(C,&hC);CHKERRQ(ierr);
  ierr = MatHYPRE_ParCSR_RAP(hA,hB,hC,&hABC);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(hABC,MATAIJ,PETSC_OWN_POINTER,&E);CHKERRQ(ierr);
  ierr = MatHeaderMerge(D,&E);CHKERRQ(ierr);
  ierr = MatAIJRestoreParCSR_Private(A,&hA);CHKERRQ(ierr);
  ierr = MatAIJRestoreParCSR_Private(B,&hB);CHKERRQ(ierr);
  ierr = MatAIJRestoreParCSR_Private(C,&hC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatTransposeMatMatMultSymbolic_AIJ_AIJ_AIJ_wHYPRE(Mat A,Mat B,Mat C,PetscReal fill,Mat D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(D,MATAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------- */
static PetscErrorCode MatProductSymbolic_AB_HYPRE(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumeric_AB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_HYPRE_AB(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      Ahypre;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)product->A,MATHYPRE,&Ahypre);CHKERRQ(ierr);
  if (Ahypre) { /* A is a Hypre matrix */
    ierr = MatSetType(C,MATHYPRE);CHKERRQ(ierr);
    C->ops->productsymbolic = MatProductSymbolic_AB_HYPRE;
    C->ops->matmultnumeric  = MatMatMultNumeric_HYPRE_HYPRE;
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_PtAP_HYPRE(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_HYPRE_PtAP(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      flg;
  PetscInt       type = 0;
  const char     *outTypes[4] = {"aij","seqaij","mpiaij","hypre"};
  PetscInt       ntype = 4;
  Mat            A = product->A;
  PetscBool      Ahypre;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATHYPRE,&Ahypre);CHKERRQ(ierr);
  if (Ahypre) { /* A is a Hypre matrix */
    ierr = MatSetType(C,MATHYPRE);CHKERRQ(ierr);
    C->ops->productsymbolic = MatProductSymbolic_PtAP_HYPRE;
    C->ops->ptapnumeric     = MatPtAPNumeric_HYPRE_HYPRE;
    PetscFunctionReturn(0);
  }

  /* A is AIJ, P is Hypre, C = PtAP can be either AIJ or Hypre format */
  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatPtAP_HYPRE","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matptap_hypre_outtype","MatPtAP outtype","MatPtAP outtype",outTypes,ntype,outTypes[type],&type,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_PtAP_HYPRE","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_ptap_hypre_outtype","MatProduct_PtAP outtype","MatProduct_PtAP",outTypes,ntype,outTypes[type],&type,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }

  if (type == 0 || type == 1 || type == 2) {
    ierr = MatSetType(C,MATAIJ);CHKERRQ(ierr);
  } else if (type == 3) {
    ierr = MatSetType(C,MATHYPRE);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatPtAP outtype is not supported");
  C->ops->productsymbolic = MatProductSymbolic_PtAP_HYPRE;
  C->ops->ptapnumeric     = MatPtAPNumeric_AIJ_HYPRE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_HYPRE(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_HYPRE_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_PtAP:
    ierr = MatProductSetFromOptions_HYPRE_PtAP(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

static PetscErrorCode MatMultTranspose_HYPRE(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHYPRE_MultKernel_Private(A,1.0,x,0.0,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_HYPRE(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHYPRE_MultKernel_Private(A,1.0,x,0.0,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_HYPRE(Mat A, Vec x, Vec y, Vec z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (y != z) {
    ierr = VecCopy(y,z);CHKERRQ(ierr);
  }
  ierr = MatHYPRE_MultKernel_Private(A,1.0,x,1.0,z,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_HYPRE(Mat A, Vec x, Vec y, Vec z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (y != z) {
    ierr = VecCopy(y,z);CHKERRQ(ierr);
  }
  ierr = MatHYPRE_MultKernel_Private(A,1.0,x,1.0,z,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = a * A * x + b * y or y = a * A^t * x + b * y depending on trans */
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat A, HYPRE_Complex a, Vec x, HYPRE_Complex b, Vec y, PetscBool trans)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  hypre_ParCSRMatrix *parcsr;
  hypre_ParVector    *hx,*hy;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (trans) {
    ierr = VecHYPRE_IJVectorPushVecRead(hA->b,x);CHKERRQ(ierr);
    if (b != 0.0) { ierr = VecHYPRE_IJVectorPushVec(hA->x,y);CHKERRQ(ierr); }
    else { ierr = VecHYPRE_IJVectorPushVecWrite(hA->x,y);CHKERRQ(ierr); }
    PetscStackCallStandard(HYPRE_IJVectorGetObject,(hA->b->ij,(void**)&hx));
    PetscStackCallStandard(HYPRE_IJVectorGetObject,(hA->x->ij,(void**)&hy));
  } else {
    ierr = VecHYPRE_IJVectorPushVecRead(hA->x,x);CHKERRQ(ierr);
    if (b != 0.0) { ierr = VecHYPRE_IJVectorPushVec(hA->b,y);CHKERRQ(ierr); }
    else { ierr = VecHYPRE_IJVectorPushVecWrite(hA->b,y);CHKERRQ(ierr); }
    PetscStackCallStandard(HYPRE_IJVectorGetObject,(hA->x->ij,(void**)&hx));
    PetscStackCallStandard(HYPRE_IJVectorGetObject,(hA->b->ij,(void**)&hy));
  }
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&parcsr));
  if (trans) {
    PetscStackCallStandard(hypre_ParCSRMatrixMatvecT,(a,parcsr,hx,b,hy));
  } else {
    PetscStackCallStandard(hypre_ParCSRMatrixMatvec,(a,parcsr,hx,b,hy));
  }
  ierr = VecHYPRE_IJVectorPopVec(hA->x);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorPopVec(hA->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_HYPRE(Mat A)
{
  Mat_HYPRE      *hA = (Mat_HYPRE*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHYPRE_IJVectorDestroy(&hA->x);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorDestroy(&hA->b);CHKERRQ(ierr);
  if (hA->ij) {
    if (!hA->inner_free) hypre_IJMatrixObject(hA->ij) = NULL;
    PetscStackCallStandard(HYPRE_IJMatrixDestroy,(hA->ij));
  }
  if (hA->comm) {ierr = PetscCommRestoreComm(PetscObjectComm((PetscObject)A),&hA->comm);CHKERRQ(ierr);}

  ierr = MatStashDestroy_Private(&A->stash);CHKERRQ(ierr);

  ierr = PetscFree(hA->array);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_hypre_aij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_hypre_is_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaij_hypre_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_mpiaij_hypre_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHYPRESetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHYPREGetParCSR_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_HYPRE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHYPRESetPreallocation(A,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//TODO FIX hypre_CSRMatrixMatvecOutOfPlace
#if defined(PETSC_HAVE_HYPRE_DEVICE)
static PetscErrorCode MatBindToCPU_HYPRE(Mat A, PetscBool bind)
{
  Mat_HYPRE            *hA = (Mat_HYPRE*)A->data;
  HYPRE_MemoryLocation hmem = bind ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  A->boundtocpu = bind;
  if (hypre_IJMatrixAssembleFlag(hA->ij) && hmem != hypre_IJMatrixMemoryLocation(hA->ij)) {
    hypre_ParCSRMatrix *parcsr;
    PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&parcsr));
    PetscStackCallStandard(hypre_ParCSRMatrixMigrate,(parcsr, hmem));
  }
  if (hA->x) { ierr  = VecHYPRE_IJBindToCPU(hA->x,bind);CHKERRQ(ierr); }
  if (hA->b) { ierr  = VecHYPRE_IJBindToCPU(hA->b,bind);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatAssemblyEnd_HYPRE(Mat A, MatAssemblyType mode)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  PetscMPIInt        n;
  PetscInt           i,j,rstart,ncols,flg;
  PetscInt           *row,*col;
  PetscScalar        *val;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MAT_FLUSH_ASSEMBLY currently not supported with MATHYPRE");

  if (!A->nooffprocentries) {
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&A->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_HYPRE(A,1,row+i,ncols,col+i,val+i,A->insertmode);CHKERRQ(ierr);

        i = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&A->stash);CHKERRQ(ierr);
  }

  PetscStackCallStandard(HYPRE_IJMatrixAssemble,(hA->ij));
  /* The assembly routine destroys the aux_matrix, we recreate it here by calling HYPRE_IJMatrixInitialize */
  /* If the option MAT_SORTED_FULL is set to true, the indices and values can be passed to hypre directly, so we don't need the aux_matrix */
  if (!hA->sorted_full) {
    hypre_AuxParCSRMatrix *aux_matrix;

    /* call destroy just to make sure we do not leak anything */
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(hA->ij);
    PetscStackCallStandard(hypre_AuxParCSRMatrixDestroy,(aux_matrix));
    hypre_IJMatrixTranslator(hA->ij) = NULL;

    /* Initialize with assembled flag -> it only recreates the aux_par_matrix */
    PetscStackCallStandard(HYPRE_IJMatrixInitialize,(hA->ij));
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(hA->ij);
    if (aux_matrix) {
      hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 1; /* see comment in MatHYPRESetPreallocation_HYPRE */
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
      PetscStackCallStandard(hypre_AuxParCSRMatrixInitialize,(aux_matrix));
#else
      PetscStackCallStandard(hypre_AuxParCSRMatrixInitialize_v2,(aux_matrix,HYPRE_MEMORY_HOST));
#endif
    }
  }
  {
    hypre_ParCSRMatrix *parcsr;

    PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&parcsr));
    if (!hypre_ParCSRMatrixCommPkg(parcsr)) PetscStackCallStandard(hypre_MatvecCommPkgCreate,(parcsr));
  }
  if (!hA->x) { ierr = VecHYPRE_IJVectorCreate(A->cmap,&hA->x);CHKERRQ(ierr); }
  if (!hA->b) { ierr = VecHYPRE_IJVectorCreate(A->rmap,&hA->b);CHKERRQ(ierr); }
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  ierr = MatBindToCPU_HYPRE(A,A->boundtocpu);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetArray_HYPRE(Mat A, PetscInt size, void **array)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!hA->available) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Temporary space is in use");

  if (hA->size >= size) {
    *array = hA->array;
  } else {
    ierr = PetscFree(hA->array);CHKERRQ(ierr);
    hA->size = size;
    ierr = PetscMalloc(hA->size,&hA->array);CHKERRQ(ierr);
    *array = hA->array;
  }

  hA->available = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreArray_HYPRE(Mat A, void **array)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;

  PetscFunctionBegin;
  *array = NULL;
  hA->available = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_HYPRE(Mat A, PetscInt nr, const PetscInt rows[], PetscInt nc, const PetscInt cols[], const PetscScalar v[], InsertMode ins)
{
  Mat_HYPRE      *hA = (Mat_HYPRE*)A->data;
  PetscScalar    *vals = (PetscScalar *)v;
  HYPRE_Complex  *sscr;
  PetscInt       *cscr[2];
  PetscInt       i,nzc;
  void           *array = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetArray_HYPRE(A,sizeof(PetscInt)*(2*nc)+sizeof(HYPRE_Complex)*nc*nr,&array);CHKERRQ(ierr);
  cscr[0] = (PetscInt*)array;
  cscr[1] = ((PetscInt*)array)+nc;
  sscr = (HYPRE_Complex*)(((PetscInt*)array)+nc*2);
  for (i=0,nzc=0;i<nc;i++) {
    if (cols[i] >= 0) {
      cscr[0][nzc  ] = cols[i];
      cscr[1][nzc++] = i;
    }
  }
  if (!nzc) {
    ierr = MatRestoreArray_HYPRE(A,&array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

#if 0 //defined(PETSC_HAVE_HYPRE_DEVICE)
  if (HYPRE_MEMORY_HOST != hypre_IJMatrixMemoryLocation(hA->ij)) {
    hypre_ParCSRMatrix *parcsr;

    PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&parcsr));
    PetscStackCallStandard(hypre_ParCSRMatrixMigrate,(parcsr, HYPRE_MEMORY_HOST));
  }
#endif

  if (ins == ADD_VALUES) {
    for (i=0;i<nr;i++) {
      if (rows[i] >= 0) {
        PetscInt  j;
        HYPRE_Int hnc = (HYPRE_Int)nzc;

        if ((PetscInt)hnc != nzc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hypre overflow! number of columns %" PetscInt_FMT " for row %" PetscInt_FMT,nzc,rows[i]);
        for (j=0;j<nzc;j++) { ierr = PetscHYPREScalarCast(vals[cscr[1][j]],&sscr[j]);CHKERRQ(ierr); }
        PetscStackCallStandard(HYPRE_IJMatrixAddToValues,(hA->ij,1,&hnc,(HYPRE_BigInt*)(rows+i),(HYPRE_BigInt*)cscr[0],sscr));
      }
      vals += nc;
    }
  } else { /* INSERT_VALUES */
    PetscInt rst,ren;

    ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
    for (i=0;i<nr;i++) {
      if (rows[i] >= 0) {
        PetscInt  j;
        HYPRE_Int hnc = (HYPRE_Int)nzc;

        if ((PetscInt)hnc != nzc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hypre overflow! number of columns %" PetscInt_FMT " for row %" PetscInt_FMT,nzc,rows[i]);
        for (j=0;j<nzc;j++) { ierr = PetscHYPREScalarCast(vals[cscr[1][j]],&sscr[j]);CHKERRQ(ierr); }
        /* nonlocal values */
        if (rows[i] < rst || rows[i] >= ren) { ierr = MatStashValuesRow_Private(&A->stash,rows[i],nzc,cscr[0],(PetscScalar*)sscr,PETSC_FALSE);CHKERRQ(ierr); }
        /* local values */
        else PetscStackCallStandard(HYPRE_IJMatrixSetValues,(hA->ij,1,&hnc,(HYPRE_BigInt*)(rows+i),(HYPRE_BigInt*)cscr[0],sscr));
      }
      vals += nc;
    }
  }

  ierr = MatRestoreArray_HYPRE(A,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHYPRESetPreallocation_HYPRE(Mat A, PetscInt dnz, const PetscInt dnnz[], PetscInt onz, const PetscInt onnz[])
{
  Mat_HYPRE      *hA = (Mat_HYPRE*)A->data;
  HYPRE_Int      *hdnnz,*honnz;
  PetscInt       i,rs,re,cs,ce,bs;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  rs   = A->rmap->rstart;
  re   = A->rmap->rend;
  cs   = A->cmap->rstart;
  ce   = A->cmap->rend;
  if (!hA->ij) {
    PetscStackCallStandard(HYPRE_IJMatrixCreate,(hA->comm,rs,re-1,cs,ce-1,&hA->ij));
    PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,(hA->ij,HYPRE_PARCSR));
  } else {
    HYPRE_BigInt hrs,hre,hcs,hce;
    PetscStackCallStandard(HYPRE_IJMatrixGetLocalRange,(hA->ij,&hrs,&hre,&hcs,&hce));
    if (hre-hrs+1 != re -rs) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent local rows: IJMatrix [%" PetscInt_FMT ",%" PetscInt_FMT "), PETSc [%" PetscInt_FMT ",%" PetscInt_FMT ")",hrs,hre+1,rs,re);
    if (hce-hcs+1 != ce -cs) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent local cols: IJMatrix [%" PetscInt_FMT ",%" PetscInt_FMT "), PETSc [%" PetscInt_FMT ",%" PetscInt_FMT ")",hcs,hce+1,cs,ce);
  }
  if (dnz == PETSC_DEFAULT || dnz == PETSC_DECIDE) dnz = 10*bs;
  if (onz == PETSC_DEFAULT || onz == PETSC_DECIDE) onz = 10*bs;

  if (!dnnz) {
    ierr = PetscMalloc1(A->rmap->n,&hdnnz);CHKERRQ(ierr);
    for (i=0;i<A->rmap->n;i++) hdnnz[i] = dnz;
  } else {
    hdnnz = (HYPRE_Int*)dnnz;
  }
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size > 1) {
    hypre_AuxParCSRMatrix *aux_matrix;
    if (!onnz) {
      ierr = PetscMalloc1(A->rmap->n,&honnz);CHKERRQ(ierr);
      for (i=0;i<A->rmap->n;i++) honnz[i] = onz;
    } else honnz = (HYPRE_Int*)onnz;
    /* SetDiagOffdSizes sets hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0, since it seems
       they assume the user will input the entire row values, properly sorted
       In PETSc, we don't make such an assumption and set this flag to 1,
       unless the option MAT_SORTED_FULL is set to true.
       Also, to avoid possible memory leaks, we destroy and recreate the translator
       This has to be done here, as HYPRE_IJMatrixInitialize will properly initialize
       the IJ matrix for us */
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(hA->ij);
    hypre_AuxParCSRMatrixDestroy(aux_matrix);
    hypre_IJMatrixTranslator(hA->ij) = NULL;
    PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,(hA->ij,hdnnz,honnz));
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(hA->ij);
    hypre_AuxParCSRMatrixNeedAux(aux_matrix) = !hA->sorted_full;
  } else {
    honnz = NULL;
    PetscStackCallStandard(HYPRE_IJMatrixSetRowSizes,(hA->ij,hdnnz));
  }

  /* reset assembled flag and call the initialize method */
  hypre_IJMatrixAssembleFlag(hA->ij) = 0;
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(hA->ij));
#else
  PetscStackCallStandard(HYPRE_IJMatrixInitialize_v2,(hA->ij,HYPRE_MEMORY_HOST));
#endif
  if (!dnnz) {
    ierr = PetscFree(hdnnz);CHKERRQ(ierr);
  }
  if (!onnz && honnz) {
    ierr = PetscFree(honnz);CHKERRQ(ierr);
  }
  /* Match AIJ logic */
  A->preallocated = PETSC_TRUE;
  A->assembled    = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   MatHYPRESetPreallocation - Preallocates memory for a sparse parallel matrix in HYPRE IJ format

   Collective on Mat

   Input Parameters:
+  A - the matrix
.  dnz  - number of nonzeros per row in DIAGONAL portion of local submatrix
          (same value is used for all local rows)
.  dnnz - array containing the number of nonzeros in the various rows of the
          DIAGONAL portion of the local submatrix (possibly different for each row)
          or NULL (PETSC_NULL_INTEGER in Fortran), if d_nz is used to specify the nonzero structure.
          The size of this array is equal to the number of local rows, i.e 'm'.
          For matrices that will be factored, you must leave room for (and set)
          the diagonal entry even if it is zero.
.  onz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
          submatrix (same value is used for all local rows).
-  onnz - array containing the number of nonzeros in the various rows of the
          OFF-DIAGONAL portion of the local submatrix (possibly different for
          each row) or NULL (PETSC_NULL_INTEGER in Fortran), if o_nz is used to specify the nonzero
          structure. The size of this array is equal to the number
          of local rows, i.e 'm'.

   Notes:
    If the *nnz parameter is given then the *nz parameter is ignored; for sequential matrices, onz and onnz are ignored.

   Level: intermediate

.seealso: MatCreate(), MatMPIAIJSetPreallocation(), MATHYPRE
@*/
PetscErrorCode MatHYPRESetPreallocation(Mat A, PetscInt dnz, const PetscInt dnnz[], PetscInt onz, const PetscInt onnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscTryMethod(A,"MatHYPRESetPreallocation_C",(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(A,dnz,dnnz,onz,onnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MatCreateFromParCSR - Creates a matrix from a hypre_ParCSRMatrix

   Collective

   Input Parameters:
+  parcsr   - the pointer to the hypre_ParCSRMatrix
.  mtype    - matrix type to be created. Currently MATAIJ, MATIS and MATHYPRE are supported.
-  copymode - PETSc copying options

   Output Parameter:
.  A  - the matrix

   Level: intermediate

.seealso: MatHYPRE, PetscCopyMode
*/
PETSC_EXTERN PetscErrorCode MatCreateFromParCSR(hypre_ParCSRMatrix *parcsr, MatType mtype, PetscCopyMode copymode, Mat* A)
{
  Mat            T;
  Mat_HYPRE      *hA;
  MPI_Comm       comm;
  PetscInt       rstart,rend,cstart,cend,M,N;
  PetscBool      isseqaij,isseqaijmkl,ismpiaij,isaij,ishyp,isis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm  = hypre_ParCSRMatrixComm(parcsr);
  ierr  = PetscStrcmp(mtype,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  ierr  = PetscStrcmp(mtype,MATSEQAIJMKL,&isseqaijmkl);CHKERRQ(ierr);
  ierr  = PetscStrcmp(mtype,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr  = PetscStrcmp(mtype,MATAIJ,&isaij);CHKERRQ(ierr);
  ierr  = PetscStrcmp(mtype,MATHYPRE,&ishyp);CHKERRQ(ierr);
  ierr  = PetscStrcmp(mtype,MATIS,&isis);CHKERRQ(ierr);
  isaij = (PetscBool)(isseqaij || isseqaijmkl || ismpiaij || isaij);
  /* TODO */
  if (!isaij && !ishyp && !isis) SETERRQ7(comm,PETSC_ERR_SUP,"Unsupported MatType %s! Supported types are %s, %s, %s, %s, %s, and %s",mtype,MATAIJ,MATSEQAIJ,MATSEQAIJMKL,MATMPIAIJ,MATIS,MATHYPRE);
  /* access ParCSRMatrix */
  rstart = hypre_ParCSRMatrixFirstRowIndex(parcsr);
  rend   = hypre_ParCSRMatrixLastRowIndex(parcsr);
  cstart = hypre_ParCSRMatrixFirstColDiag(parcsr);
  cend   = hypre_ParCSRMatrixLastColDiag(parcsr);
  M      = hypre_ParCSRMatrixGlobalNumRows(parcsr);
  N      = hypre_ParCSRMatrixGlobalNumCols(parcsr);

  /* fix for empty local rows/columns */
  if (rend < rstart) rend = rstart;
  if (cend < cstart) cend = cstart;

  /* PETSc convention */
  rend++;
  cend++;
  rend = PetscMin(rend,M);
  cend = PetscMin(cend,N);

  /* create PETSc matrix with MatHYPRE */
  ierr = MatCreate(comm,&T);CHKERRQ(ierr);
  ierr = MatSetSizes(T,rend-rstart,cend-cstart,M,N);CHKERRQ(ierr);
  ierr = MatSetType(T,MATHYPRE);CHKERRQ(ierr);
  hA   = (Mat_HYPRE*)(T->data);

  /* create HYPRE_IJMatrix */
  PetscStackCallStandard(HYPRE_IJMatrixCreate,(hA->comm,rstart,rend-1,cstart,cend-1,&hA->ij));
  PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,(hA->ij,HYPRE_PARCSR));

// TODO DEV
  /* create new ParCSR object if needed */
  if (ishyp && copymode == PETSC_COPY_VALUES) {
    hypre_ParCSRMatrix *new_parcsr;
#if PETSC_PKG_HYPRE_VERSION_LT(2,18,0)
    hypre_CSRMatrix    *hdiag,*hoffd,*ndiag,*noffd;

    new_parcsr = hypre_ParCSRMatrixClone(parcsr,0);
    hdiag      = hypre_ParCSRMatrixDiag(parcsr);
    hoffd      = hypre_ParCSRMatrixOffd(parcsr);
    ndiag      = hypre_ParCSRMatrixDiag(new_parcsr);
    noffd      = hypre_ParCSRMatrixOffd(new_parcsr);
    ierr       = PetscArraycpy(hypre_CSRMatrixData(ndiag),hypre_CSRMatrixData(hdiag),hypre_CSRMatrixNumNonzeros(hdiag));CHKERRQ(ierr);
    ierr       = PetscArraycpy(hypre_CSRMatrixData(noffd),hypre_CSRMatrixData(hoffd),hypre_CSRMatrixNumNonzeros(hoffd));CHKERRQ(ierr);
#else
    new_parcsr = hypre_ParCSRMatrixClone(parcsr,1);
#endif
    parcsr     = new_parcsr;
    copymode   = PETSC_OWN_POINTER;
  }

  /* set ParCSR object */
  hypre_IJMatrixObject(hA->ij) = parcsr;
  T->preallocated = PETSC_TRUE;

  /* set assembled flag */
  hypre_IJMatrixAssembleFlag(hA->ij) = 1;
#if 0
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(hA->ij));
#endif
  if (ishyp) {
    PetscMPIInt myid = 0;

    /* make sure we always have row_starts and col_starts available */
    if (HYPRE_AssumedPartitionCheck()) {
      ierr = MPI_Comm_rank(comm,&myid);CHKERRMPI(ierr);
    }
#if defined(hypre_ParCSRMatrixOwnsRowStarts)
    if (!hypre_ParCSRMatrixOwnsColStarts(parcsr)) {
      PetscLayout map;

      ierr = MatGetLayouts(T,NULL,&map);CHKERRQ(ierr);
      ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
      hypre_ParCSRMatrixColStarts(parcsr) = (HYPRE_BigInt*)(map->range + myid);
    }
    if (!hypre_ParCSRMatrixOwnsRowStarts(parcsr)) {
      PetscLayout map;

      ierr = MatGetLayouts(T,&map,NULL);CHKERRQ(ierr);
      ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
      hypre_ParCSRMatrixRowStarts(parcsr) = (HYPRE_BigInt*)(map->range + myid);
    }
#endif
    /* prevent from freeing the pointer */
    if (copymode == PETSC_USE_POINTER) hA->inner_free = PETSC_FALSE;
    *A   = T;
    ierr = MatSetOption(*A,MAT_SORTED_FULL,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else if (isaij) {
    if (copymode != PETSC_OWN_POINTER) {
      /* prevent from freeing the pointer */
      hA->inner_free = PETSC_FALSE;
      ierr = MatConvert_HYPRE_AIJ(T,MATAIJ,MAT_INITIAL_MATRIX,A);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    } else { /* AIJ return type with PETSC_OWN_POINTER */
      ierr = MatConvert_HYPRE_AIJ(T,MATAIJ,MAT_INPLACE_MATRIX,&T);CHKERRQ(ierr);
      *A   = T;
    }
  } else if (isis) {
    ierr = MatConvert_HYPRE_IS(T,MATIS,MAT_INITIAL_MATRIX,A);CHKERRQ(ierr);
    if (copymode != PETSC_OWN_POINTER) hA->inner_free = PETSC_FALSE;
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHYPREGetParCSR_HYPRE(Mat A, hypre_ParCSRMatrix **parcsr)
{
  Mat_HYPRE *hA = (Mat_HYPRE*)A->data;
  HYPRE_Int type;

  PetscFunctionBegin;
  if (!hA->ij) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"HYPRE_IJMatrix not present");
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"HYPRE_IJMatrix is not of type HYPRE_PARCSR");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)parcsr));
  PetscFunctionReturn(0);
}

/*
   MatHYPREGetParCSR - Gets the pointer to the ParCSR matrix

   Not collective

   Input Parameters:
+  A  - the MATHYPRE object

   Output Parameter:
.  parcsr  - the pointer to the hypre_ParCSRMatrix

   Level: intermediate

.seealso: MatHYPRE, PetscCopyMode
*/
PetscErrorCode MatHYPREGetParCSR(Mat A, hypre_ParCSRMatrix **parcsr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscUseMethod(A,"MatHYPREGetParCSR_C",(Mat,hypre_ParCSRMatrix**),(A,parcsr));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_HYPRE(Mat A, PetscBool *missing, PetscInt *dd)
{
  hypre_ParCSRMatrix *parcsr;
  hypre_CSRMatrix    *ha;
  PetscInt           rst;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented with non-square diagonal blocks");
  ierr = MatGetOwnershipRange(A,&rst,NULL);CHKERRQ(ierr);
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  if (missing) *missing = PETSC_FALSE;
  if (dd) *dd = -1;
  ha = hypre_ParCSRMatrixDiag(parcsr);
  if (ha) {
    PetscInt  size,i;
    HYPRE_Int *ii,*jj;

    size = hypre_CSRMatrixNumRows(ha);
    ii   = hypre_CSRMatrixI(ha);
    jj   = hypre_CSRMatrixJ(ha);
    for (i = 0; i < size; i++) {
      PetscInt  j;
      PetscBool found = PETSC_FALSE;

      for (j = ii[i]; j < ii[i+1] && !found; j++)
        found = (jj[j] == i) ? PETSC_TRUE : PETSC_FALSE;

      if (!found) {
        PetscInfo1(A,"Matrix is missing local diagonal entry %" PetscInt_FMT "\n",i);
        if (missing) *missing = PETSC_TRUE;
        if (dd) *dd = i+rst;
        PetscFunctionReturn(0);
      }
    }
    if (!size) {
      PetscInfo(A,"Matrix has no diagonal entries therefore is missing diagonal\n");
      if (missing) *missing = PETSC_TRUE;
      if (dd) *dd = rst;
    }
  } else {
    PetscInfo(A,"Matrix has no diagonal entries therefore is missing diagonal\n");
    if (missing) *missing = PETSC_TRUE;
    if (dd) *dd = rst;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_HYPRE(Mat A, PetscScalar s)
{
  hypre_ParCSRMatrix *parcsr;
#if PETSC_PKG_HYPRE_VERSION_LT(2,19,0)
  hypre_CSRMatrix    *ha;
#endif
  PetscErrorCode     ierr;
  HYPRE_Complex      hs;

  PetscFunctionBegin;
  ierr = PetscHYPREScalarCast(s,&hs);CHKERRQ(ierr);
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
#if PETSC_PKG_HYPRE_VERSION_GE(2,19,0)
  PetscStackCallStandard(hypre_ParCSRMatrixScale,(parcsr,hs));
#else  /* diagonal part */
  ha = hypre_ParCSRMatrixDiag(parcsr);
  if (ha) {
    PetscInt      size,i;
    HYPRE_Int     *ii;
    HYPRE_Complex *a;

    size = hypre_CSRMatrixNumRows(ha);
    a    = hypre_CSRMatrixData(ha);
    ii   = hypre_CSRMatrixI(ha);
    for (i = 0; i < ii[size]; i++) a[i] *= hs;
  }
  /* offdiagonal part */
  ha = hypre_ParCSRMatrixOffd(parcsr);
  if (ha) {
    PetscInt      size,i;
    HYPRE_Int     *ii;
    HYPRE_Complex *a;

    size = hypre_CSRMatrixNumRows(ha);
    a    = hypre_CSRMatrixData(ha);
    ii   = hypre_CSRMatrixI(ha);
    for (i = 0; i < ii[size]; i++) a[i] *= hs;
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRowsColumns_HYPRE(Mat A, PetscInt numRows, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  hypre_ParCSRMatrix *parcsr;
  HYPRE_Int          *lrows;
  PetscInt           rst,ren,i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (x || b) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"To be implemented");
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  ierr = PetscMalloc1(numRows,&lrows);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
  for (i=0;i<numRows;i++) {
    if (rows[i] < rst || rows[i] >= ren)
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Non-local rows not yet supported");
    lrows[i] = rows[i] - rst;
  }
  PetscStackCallStandard(hypre_ParCSRMatrixEliminateRowsCols,(parcsr,numRows,lrows));
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_HYPRE_CSRMatrix(hypre_CSRMatrix *ha)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (ha) {
    HYPRE_Int     *ii, size;
    HYPRE_Complex *a;

    size = hypre_CSRMatrixNumRows(ha);
    a    = hypre_CSRMatrixData(ha);
    ii   = hypre_CSRMatrixI(ha);

    if (a) {ierr = PetscArrayzero(a,ii[size]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_HYPRE(Mat A)
{
  Mat_HYPRE *hA = (Mat_HYPRE*)A->data;

  PetscFunctionBegin;
  if (HYPRE_MEMORY_DEVICE == hypre_IJMatrixMemoryLocation(hA->ij)) {
    PetscStackCallStandard(HYPRE_IJMatrixSetConstantValues,(hA->ij,0.0));
  } else {
    hypre_ParCSRMatrix *parcsr;
    PetscErrorCode     ierr;

    ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
    ierr = MatZeroEntries_HYPRE_CSRMatrix(hypre_ParCSRMatrixDiag(parcsr));CHKERRQ(ierr);
    ierr = MatZeroEntries_HYPRE_CSRMatrix(hypre_ParCSRMatrixOffd(parcsr));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_HYPRE_CSRMatrix(hypre_CSRMatrix *hA,PetscInt N,const PetscInt rows[],HYPRE_Complex diag)
{
  PetscInt        ii;
  HYPRE_Int       *i, *j;
  HYPRE_Complex   *a;

  PetscFunctionBegin;
  if (!hA) PetscFunctionReturn(0);

  i = hypre_CSRMatrixI(hA);
  j = hypre_CSRMatrixJ(hA);
  a = hypre_CSRMatrixData(hA);

  for (ii = 0; ii < N; ii++) {
    HYPRE_Int jj, ibeg, iend, irow;

    irow = rows[ii];
    ibeg = i[irow];
    iend = i[irow+1];
    for (jj = ibeg; jj < iend; jj++)
      if (j[jj] == irow) a[jj] = diag;
      else a[jj] = 0.0;
   }
   PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_HYPRE(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  hypre_ParCSRMatrix  *parcsr;
  PetscInt            *lrows,len;
  HYPRE_Complex       hdiag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (x || b) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Does not support to modify the solution and the right hand size");
  ierr = PetscHYPREScalarCast(diag,&hdiag);CHKERRQ(ierr);
  /* retrieve the internal matrix */
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  /* get locally owned rows */
  ierr = MatZeroRowsMapLocal_Private(A,N,rows,&len,&lrows);CHKERRQ(ierr);
  /* zero diagonal part */
  ierr = MatZeroRows_HYPRE_CSRMatrix(hypre_ParCSRMatrixDiag(parcsr),len,lrows,hdiag);CHKERRQ(ierr);
  /* zero off-diagonal part */
  ierr = MatZeroRows_HYPRE_CSRMatrix(hypre_ParCSRMatrixOffd(parcsr),len,lrows,0.0);CHKERRQ(ierr);

  ierr = PetscFree(lrows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_HYPRE(Mat mat,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mat->nooffprocentries) PetscFunctionReturn(0);

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRow_HYPRE(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  hypre_ParCSRMatrix  *parcsr;
  HYPRE_Int           hnz;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* retrieve the internal matrix */
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  /* call HYPRE API */
  PetscStackCallStandard(HYPRE_ParCSRMatrixGetRow,(parcsr,row,&hnz,(HYPRE_BigInt**)idx,(HYPRE_Complex**)v));
  if (nz) *nz = (PetscInt)hnz;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_HYPRE(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  hypre_ParCSRMatrix  *parcsr;
  HYPRE_Int           hnz;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* retrieve the internal matrix */
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  /* call HYPRE API */
  hnz = nz ? (HYPRE_Int)(*nz) : 0;
  PetscStackCallStandard(HYPRE_ParCSRMatrixRestoreRow,(parcsr,row,&hnz,(HYPRE_BigInt**)idx,(HYPRE_Complex**)v));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetValues_HYPRE(Mat A,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_HYPRE *hA = (Mat_HYPRE*)A->data;
  PetscInt  i;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0);
  /* Ignore negative row indices
   * And negative column indices should be automatically ignored in hypre
   * */
  for (i=0; i<m; i++) {
    if (idxm[i] >= 0) {
      HYPRE_Int hn = (HYPRE_Int)n;
      PetscStackCallStandard(HYPRE_IJMatrixGetValues,(hA->ij,1,&hn,(HYPRE_BigInt*)&idxm[i],(HYPRE_BigInt*)idxn,(HYPRE_Complex*)(v + i*n)));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_HYPRE(Mat A,MatOption op,PetscBool flg)
{
  Mat_HYPRE *hA = (Mat_HYPRE*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NO_OFF_PROC_ENTRIES:
    if (flg) {
      PetscStackCallStandard(HYPRE_IJMatrixSetMaxOffProcElmts,(hA->ij,0));
    }
    break;
  case MAT_SORTED_FULL:
    hA->sorted_full = flg;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_HYPRE(Mat A, PetscViewer view)
{
  PetscErrorCode     ierr;
  PetscViewerFormat  format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(view,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);
  if (format != PETSC_VIEWER_NATIVE) {
    Mat                B;
    hypre_ParCSRMatrix *parcsr;
    PetscErrorCode     (*mview)(Mat,PetscViewer) = NULL;

    ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
    ierr = MatCreateFromParCSR(parcsr,MATAIJ,PETSC_USE_POINTER,&B);CHKERRQ(ierr);
    ierr = MatGetOperation(B,MATOP_VIEW,(void(**)(void))&mview);CHKERRQ(ierr);
    if (!mview) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing view operation");
    ierr = (*mview)(B,view);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    Mat_HYPRE  *hA = (Mat_HYPRE*)A->data;
    PetscMPIInt size;
    PetscBool   isascii;
    const char *filename;

    /* HYPRE uses only text files */
    ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
    if (!isascii) SETERRQ1(PetscObjectComm((PetscObject)view),PETSC_ERR_SUP,"PetscViewerType %s: native HYPRE format needs PETSCVIEWERASCII",((PetscObject)view)->type_name);
    ierr = PetscViewerFileGetName(view,&filename);CHKERRQ(ierr);
    PetscStackCallStandard(HYPRE_IJMatrixPrint,(hA->ij,filename));
    ierr = MPI_Comm_size(hA->comm,&size);CHKERRMPI(ierr);
    if (size > 1) {
      ierr = PetscViewerASCIIPrintf(view,"Matrix files: %s.%05d ... %s.%05d\n",filename,0,filename,size-1);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(view,"Matrix file: %s.%05d\n",filename,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_HYPRE(Mat A,MatDuplicateOption op, Mat *B)
{
  hypre_ParCSRMatrix *parcsr = NULL;
  PetscErrorCode     ierr;
  PetscCopyMode      cpmode;

  PetscFunctionBegin;
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  if (op == MAT_DO_NOT_COPY_VALUES || op == MAT_SHARE_NONZERO_PATTERN) {
    parcsr = hypre_ParCSRMatrixClone(parcsr,0);
    cpmode = PETSC_OWN_POINTER;
  } else {
    cpmode = PETSC_COPY_VALUES;
  }
  ierr = MatCreateFromParCSR(parcsr,MATHYPRE,cpmode,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_HYPRE(Mat A, Mat B, MatStructure str)
{
  hypre_ParCSRMatrix *acsr,*bcsr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN && A->ops->copy == B->ops->copy) {
    ierr = MatHYPREGetParCSR_HYPRE(A,&acsr);CHKERRQ(ierr);
    ierr = MatHYPREGetParCSR_HYPRE(B,&bcsr);CHKERRQ(ierr);
    PetscStackCallStandard(hypre_ParCSRMatrixCopy,(acsr,bcsr,1));
    ierr = MatSetOption(B,MAT_SORTED_FULL,PETSC_TRUE);CHKERRQ(ierr); /* "perfect" preallocation, so no need for hypre_AuxParCSRMatrixNeedAux */
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_HYPRE(Mat A, Vec d)
{
  hypre_ParCSRMatrix *parcsr;
  hypre_CSRMatrix    *dmat;
  HYPRE_Complex      *a;
  HYPRE_Complex      *data = NULL;
  HYPRE_Int          *diag = NULL;
  PetscInt           i;
  PetscBool          cong;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatHasCongruentLayouts(A,&cong);CHKERRQ(ierr);
  if (!cong) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only for square matrices with same local distributions of rows and columns");
  if (PetscDefined(USE_DEBUG)) {
    PetscBool miss;
    ierr = MatMissingDiagonal(A,&miss,NULL);CHKERRQ(ierr);
    if (miss && A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented when diagonal entries are missing");
  }
  ierr = MatHYPREGetParCSR_HYPRE(A,&parcsr);CHKERRQ(ierr);
  dmat = hypre_ParCSRMatrixDiag(parcsr);
  if (dmat) {
    /* this cast fixes the clang error: implicit conversion from 'HYPRE_Complex' (aka '_Complex double') to 'double' is not permitted in C++ */
    ierr = VecGetArray(d,(PetscScalar**)&a);CHKERRQ(ierr);
    diag = hypre_CSRMatrixI(dmat);
    data = hypre_CSRMatrixData(dmat);
    for (i=0;i<A->rmap->n;i++) a[i] = data[diag[i]];
    ierr = VecRestoreArray(d,(PetscScalar**)&a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscblaslapack.h>

static PetscErrorCode MatAXPY_HYPRE(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  {
    Mat                B;
    hypre_ParCSRMatrix *x,*y,*z;

    ierr = MatHYPREGetParCSR(Y,&y);CHKERRQ(ierr);
    ierr = MatHYPREGetParCSR(X,&x);CHKERRQ(ierr);
    PetscStackCallStandard(hypre_ParCSRMatrixAdd,(1.0,y,1.0,x,&z));
    ierr = MatCreateFromParCSR(z,MATHYPRE,PETSC_OWN_POINTER,&B);CHKERRQ(ierr);
    ierr = MatHeaderMerge(Y,&B);CHKERRQ(ierr);
  }
#else
  if (str == SAME_NONZERO_PATTERN) {
    hypre_ParCSRMatrix *x,*y;
    hypre_CSRMatrix    *xloc,*yloc;
    PetscInt           xnnz,ynnz;
    HYPRE_Complex      *xarr,*yarr;
    PetscBLASInt       one=1,bnz;

    ierr = MatHYPREGetParCSR(Y,&y);CHKERRQ(ierr);
    ierr = MatHYPREGetParCSR(X,&x);CHKERRQ(ierr);

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
    if (xnnz != ynnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Different number of nonzeros in diagonal block %" PetscInt_FMT " != %" PetscInt_FMT,xnnz,ynnz);
    ierr = PetscBLASIntCast(xnnz,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&a,(PetscScalar*)xarr,&one,(PetscScalar*)yarr,&one));

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
    if (xnnz != ynnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Different number of nonzeros in off-diagonal block %" PetscInt_FMT " != %" PetscInt_FMT,xnnz,ynnz);
    ierr = PetscBLASIntCast(xnnz,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&a,(PetscScalar*)xarr,&one,(PetscScalar*)yarr,&one));
  } else if (str == SUBSET_NONZERO_PATTERN) {
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  } else {
    Mat B;

    ierr = MatAXPY_Basic_Preallocate(Y,X,&B);CHKERRQ(ierr);
    ierr = MatAXPY_BasicWithPreallocation(B,Y,a,X,str);CHKERRQ(ierr);
    ierr = MatHeaderReplace(Y,&B);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*MC
   MATHYPRE - MATHYPRE = "hypre" - A matrix type to be used for sequential and parallel sparse matrices
          based on the hypre IJ interface.

   Level: intermediate

.seealso: MatCreate()
M*/

PETSC_EXTERN PetscErrorCode MatCreate_HYPRE(Mat B)
{
  Mat_HYPRE      *hB;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(B,&hB);CHKERRQ(ierr);

  hB->inner_free  = PETSC_TRUE;
  hB->available   = PETSC_TRUE;
  hB->sorted_full = PETSC_FALSE; /* no assumption whether column indices are sorted or not */
  hB->size        = 0;
  hB->array       = NULL;

  B->data       = (void*)hB;
  B->rmap->bs   = 1;
  B->assembled  = PETSC_FALSE;

  ierr = PetscMemzero(B->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
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
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  B->ops->bindtocpu             = MatBindToCPU_HYPRE;
  B->boundtocpu                 = PETSC_FALSE;
#endif

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),1,&B->stash);CHKERRQ(ierr);

  ierr = PetscCommGetComm(PetscObjectComm((PetscObject)B),&hB->comm);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATHYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_hypre_aij_C",MatConvert_HYPRE_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_hypre_is_C",MatConvert_HYPRE_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_hypre_C",MatProductSetFromOptions_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaij_hypre_C",MatProductSetFromOptions_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatHYPRESetPreallocation_C",MatHYPRESetPreallocation_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatHYPREGetParCSR_C",MatHYPREGetParCSR_HYPRE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE_DEVICE)
#if defined(HYPRE_USING_HIP)
  ierr = PetscDeviceInitialize(PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = MatSetVecType(B,VECHIP);CHKERRQ(ierr);
#endif
#if defined(HYPRE_USING_CUDA)
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  ierr = MatSetVecType(B,VECCUDA);CHKERRQ(ierr);
#endif
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode hypre_array_destroy(void *ptr)
{
   PetscFunctionBegin;
   hypre_TFree(ptr,HYPRE_MEMORY_HOST);
   PetscFunctionReturn(0);
}
