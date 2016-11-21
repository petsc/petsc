
/*
    Creates hypre ijmatrix from PETSc matrix
*/

/*MC
   MATHYPRE - MATHYPRE = "hypre" - A matrix type to be used for sequential and parallel sparse matrices
          based on the hypre IJ interface.

   Level: intermediate

.seealso: MatCreate()
M*/

#include <petscmathypre.h>
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/hypre/mhypre.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/vec/vec/impls/hypre/vhyp.h>
#include <HYPRE.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>

static PetscErrorCode MatHYPRE_CreateFromMat(Mat,Mat_HYPRE*);
static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat,Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat,Vec,Vec,PetscBool);
static PetscErrorCode hypre_array_destroy(void*);

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixPreallocate"
static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat A_d, Mat A_o, HYPRE_IJMatrix ij)
{
  PetscErrorCode ierr;
  PetscInt       i,n_d,n_o;
  const PetscInt *ia_d,*ia_o;
  PetscBool      done_d=PETSC_FALSE,done_o=PETSC_FALSE;
  PetscInt       *nnz_d=NULL,*nnz_o=NULL;

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
      ierr = PetscMalloc1(n_d,&nnz_o);CHKERRQ(ierr);
      for (i=0; i<n_d; i++) {
        nnz_o[i] = 0;
      }
    }
    PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,(ij,(HYPRE_Int *)nnz_d,(HYPRE_Int *)nnz_o));
    ierr = PetscFree(nnz_d);CHKERRQ(ierr);
    ierr = PetscFree(nnz_o);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_CreateFromMat"
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
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIAIJGetSeqAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIBAIJGetSeqBAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,NULL,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,NULL,hA->ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixCopy"
static PetscErrorCode MatHYPRE_IJMatrixCopy(Mat A, HYPRE_IJMatrix ij)
{
  PetscErrorCode    ierr;
  PetscInt          i,rstart,rend,ncols,nr,nc;
  const PetscScalar *values;
  const PetscInt    *cols;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  ierr = MatGetSize(A,&nr,&nc);CHKERRQ(ierr);
  if (flg && nr == nc) {
    ierr = MatHYPRE_IJMatrixFastCopy_MPIAIJ(A,ij);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatHYPRE_IJMatrixFastCopy_SeqAIJ(A,ij);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(ij));
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
    PetscStackCallStandard(HYPRE_IJMatrixSetValues,(ij,1,(HYPRE_Int *)&ncols,(HYPRE_Int *)&i,(HYPRE_Int *)cols,values));
    ierr = MatRestoreRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixFastCopy_SeqAIJ"
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat A, HYPRE_IJMatrix ij)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJ            *pdiag = (Mat_SeqAIJ*)A->data;
  HYPRE_Int             type;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(ij));
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(ij,(void**)&par_matrix));
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  /*
       this is the Hack part where we monkey directly with the hypre datastructures
  */
  ierr = PetscMemcpy(hdiag->i,pdiag->i,(A->rmap->n + 1)*sizeof(PetscInt));
  ierr = PetscMemcpy(hdiag->j,pdiag->j,pdiag->nz*sizeof(PetscInt));
  ierr = PetscMemcpy(hdiag->data,pdiag->a,pdiag->nz*sizeof(PetscScalar));

  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixFastCopy_MPIAIJ"
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat A, HYPRE_IJMatrix ij)
{
  PetscErrorCode        ierr;
  Mat_MPIAIJ            *pA = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ            *pdiag,*poffd;
  PetscInt              i,*garray = pA->garray,*jj,cstart,*pjj;
  HYPRE_Int             type;
  hypre_ParCSRMatrix    *par_matrix;
  hypre_AuxParCSRMatrix *aux_matrix;
  hypre_CSRMatrix       *hdiag,*hoffd;

  PetscFunctionBegin;
  pdiag = (Mat_SeqAIJ*) pA->A->data;
  poffd = (Mat_SeqAIJ*) pA->B->data;
  /* cstart is only valid for square MPIAIJ layed out in the usual way */
  ierr = MatGetOwnershipRange(A,&cstart,NULL);CHKERRQ(ierr);

  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(ij));
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(ij,(void**)&par_matrix));
  hdiag = hypre_ParCSRMatrixDiag(par_matrix);
  hoffd = hypre_ParCSRMatrixOffd(par_matrix);

  /*
       this is the Hack part where we monkey directly with the hypre datastructures
  */
  ierr = PetscMemcpy(hdiag->i,pdiag->i,(pA->A->rmap->n + 1)*sizeof(PetscInt));
  /* need to shift the diag column indices (hdiag->j) back to global numbering since hypre is expecting this */
  jj  = (PetscInt*)hdiag->j;
  pjj = (PetscInt*)pdiag->j;
  for (i=0; i<pdiag->nz; i++) jj[i] = cstart + pjj[i];
  ierr = PetscMemcpy(hdiag->data,pdiag->a,pdiag->nz*sizeof(PetscScalar));
  ierr = PetscMemcpy(hoffd->i,poffd->i,(pA->A->rmap->n + 1)*sizeof(PetscInt));
  /* need to move the offd column indices (hoffd->j) back to global numbering since hypre is expecting this
     If we hacked a hypre a bit more we might be able to avoid this step */
  jj  = (PetscInt*) hoffd->j;
  pjj = (PetscInt*) poffd->j;
  for (i=0; i<poffd->nz; i++) jj[i] = garray[pjj[i]];
  ierr = PetscMemcpy(hoffd->data,poffd->a,poffd->nz*sizeof(PetscScalar));

  aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(ij);
  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_HYPRE_IS"
static PetscErrorCode MatConvert_HYPRE_IS(Mat A, MatType mtype, MatReuse reuse, Mat* B)
{
  Mat_HYPRE*             mhA = (Mat_HYPRE*)(A->data);
  Mat                    lA;
  ISLocalToGlobalMapping rl2g,cl2g;
  IS                     is;
  hypre_ParCSRMatrix     *hA;
  hypre_CSRMatrix        *hdiag,*hoffd;
  MPI_Comm               comm;
  PetscScalar            *hdd,*hod,*aa,*data;
  HYPRE_Int              *col_map_offd,*hdi,*hdj,*hoi,*hoj;
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
    if (nr != dr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of rows in local mat! %D != %D",nr,dr);
    if (iptr[nr] < nnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in local mat! reuse %D requested %D",iptr[nr],nnz);
    ierr = MatSeqAIJGetArray(lA,&data);CHKERRQ(ierr);
  }
  /* merge local matrices */
  ii   = iptr;
  jj   = jptr;
  aa   = data;
  *ii  = *(hdi++) + *(hoi++);
  for (jd=0,jo=0,cum=0; *ii<nnz; cum++) {
    PetscScalar *aold = aa;
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

#undef __FUNCT__
#define __FUNCT__ "MatConvert_AIJ_HYPRE"
PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat A, MatType type, MatReuse reuse, Mat *B)
{
  Mat_HYPRE      *hB;
  MPI_Comm       comm = PetscObjectComm((PetscObject)A);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INPLACE_MATRIX) SETERRQ(comm,PETSC_ERR_SUP,"Unsupported MAT_INPLACE_MATRIX");
  if (reuse == MAT_REUSE_MATRIX) {
    /* always destroy the old matrix and create a new memory;
       hope this does not churn the memory too much. The problem
       is I do not know if it is possible to put the matrix back to
       its initial state so that we can directly copy the values
       the second time through. */
    hB = (Mat_HYPRE*)((*B)->data);
    PetscStackCallStandard(HYPRE_IJMatrixDestroy,(hB->ij));
  } else {
    ierr = MatCreate(comm,B);CHKERRQ(ierr);
    ierr = MatSetType(*B,MATHYPRE);CHKERRQ(ierr);
    ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetUp(*B);CHKERRQ(ierr);
    hB   = (Mat_HYPRE*)((*B)->data);
  }
  ierr = MatHYPRE_CreateFromMat(A,hB);CHKERRQ(ierr);
  ierr = MatHYPRE_IJMatrixCopy(A,hB->ij);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_HYPRE_AIJ"
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)A);
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(comm,PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool ismpiaij,isseqaij;
    ierr = PetscObjectTypeCompare((PetscObject)*B,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)*B,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    if (!ismpiaij && !isseqaij) SETERRQ(comm,PETSC_ERR_SUP,"Only MATMPIAIJ or MATSEQAIJ are supported");
  }
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

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
      if (nr != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows in diag part! %D != %D",nr,m);
      if (dii[nr] < dnnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in diag part! reuse %D hypre %D",dii[nr],dnnz);
      ierr = MatSeqAIJGetArray(b->A,&da);CHKERRQ(ierr);
    } else {
      ierr = MatGetRowIJ(*B,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&dii,(const PetscInt**)&djj,&done);CHKERRQ(ierr);
      if (nr != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows! %D != %D",nr,m);
      if (dii[nr] < dnnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros! reuse %D hypre %D",dii[nr],dnnz);
      ierr = MatSeqAIJGetArray(*B,&da);CHKERRQ(ierr);
    }
  } else { /* MAT_INPLACE_MATRIX */
    dii = (PetscInt*)hypre_CSRMatrixI(hdiag);
    djj = (PetscInt*)hypre_CSRMatrixJ(hdiag);
    da  = hypre_CSRMatrixData(hdiag);
  }
  ierr = PetscMemcpy(dii,hypre_CSRMatrixI(hdiag),(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(djj,hypre_CSRMatrixJ(hdiag),dnnz*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(da,hypre_CSRMatrixData(hdiag),dnnz*sizeof(PetscScalar));CHKERRQ(ierr);
  iptr = djj;
  aptr = da;
  for (i=0; i<m; i++) {
    PetscInt nc = dii[i+1]-dii[i];
    ierr = PetscSortIntWithScalarArray(nc,iptr,aptr);CHKERRQ(ierr);
    iptr += nc;
    aptr += nc;
  }
  if (size > 1) {
    HYPRE_Int *offdj,*coffd;

    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = PetscMalloc1(m+1,&oii);CHKERRQ(ierr);
      ierr = PetscMalloc1(onnz,&ojj);CHKERRQ(ierr);
      ierr = PetscMalloc1(onnz,&oa);CHKERRQ(ierr);
    } else if (reuse == MAT_REUSE_MATRIX) {
      Mat_MPIAIJ *b = (Mat_MPIAIJ*)((*B)->data);
      PetscInt   nr,hr = hypre_CSRMatrixNumRows(hoffd);
      PetscBool  done;

      ierr = MatGetRowIJ(b->B,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&oii,(const PetscInt**)&ojj,&done);CHKERRQ(ierr);
      if (nr != hr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows in offdiag part! %D != %D",nr,hr);
      if (oii[nr] < onnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in offdiag part! reuse %D hypre %D",oii[nr],onnz);
      ierr = MatSeqAIJGetArray(b->B,&oa);CHKERRQ(ierr);
    } else { /* MAT_INPLACE_MATRIX */
      oii = (PetscInt*)hypre_CSRMatrixI(hoffd);
      ojj = (PetscInt*)hypre_CSRMatrixJ(hoffd);
      oa  = hypre_CSRMatrixData(hoffd);
    }
    ierr  = PetscMemcpy(oii,hypre_CSRMatrixI(hoffd),(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
    offdj = hypre_CSRMatrixJ(hoffd);
    coffd = hypre_ParCSRMatrixColMapOffd(parcsr);
    for (i=0; i<onnz; i++) ojj[i] = coffd[offdj[i]];
    ierr = PetscMemcpy(oa,hypre_CSRMatrixData(hoffd),onnz*sizeof(PetscScalar));CHKERRQ(ierr);
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
      hypre_CSRMatrixI(hdiag)    = NULL;
      hypre_CSRMatrixJ(hdiag)    = NULL;
      hypre_CSRMatrixData(hdiag) = NULL;
      hypre_CSRMatrixI(hoffd)    = NULL;
      hypre_CSRMatrixJ(hoffd)    = NULL;
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
      hypre_CSRMatrixI(hdiag)    = NULL;
      hypre_CSRMatrixJ(hdiag)    = NULL;
      hypre_CSRMatrixData(hdiag) = NULL;
      ierr = MatHeaderReplace(A,&T);CHKERRQ(ierr);
    }
  }

  /* we have to use hypre_Tfree to free the arrays */
  if (reuse == MAT_INPLACE_MATRIX) {
    void *ptrs[6] = {dii,djj,da,oii,ojj,oa};
    const char *names[6] = {"_hypre_csr_dii",
                            "_hypre_csr_djj",
                            "_hypre_csr_da",
                            "_hypre_csr_oii",
                            "_hypre_csr_ojj",
                            "_hypre_csr_oa"};
    for (i=0; i<6; i++) {
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

#undef __FUNCT__
#define __FUNCT__ "MatPtAP_AIJ_HYPRE"
static PetscErrorCode MatPtAP_AIJ_HYPRE(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)
{
  Mat_HYPRE          *hP = (Mat_HYPRE*)P->data;
  hypre_ParCSRMatrix *tA,*Pparcsr,*ptapparcsr;
  hypre_CSRMatrix    *hdiag,*hoffd;
  Mat_SeqAIJ         *diag,*offd;
  PetscInt           *garray,noffd,dnnz,onnz,*row_starts,*col_starts;
  HYPRE_Int          type,P_owns_col_starts;
  PetscBool          ismpiaij,isseqaij;
  MPI_Comm           comm = PetscObjectComm((PetscObject)A);
  char               mtype[256];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hP->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(comm,PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  if (scall == MAT_REUSE_MATRIX) SETERRQ(comm,PETSC_ERR_SUP,"Unsupported MAT_REUSE_MATRIX");
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  if (!ismpiaij && !isseqaij) SETERRQ1(comm,PETSC_ERR_SUP,"Unsupported type %s",((PetscObject)A)->type);

  /* It looks like we don't need to have the diagonal entries
     ordered first in the rows of the diagonal part
     for boomerAMGBuildCoarseOperator to work */
  if (ismpiaij) {
    Mat_MPIAIJ *a = (Mat_MPIAIJ*)(A->data);

    diag   = (Mat_SeqAIJ*)a->A->data;
    offd   = (Mat_SeqAIJ*)a->B->data;
    garray = a->garray;
    noffd  = a->B->cmap->N;
    dnnz   = diag->nz;
    onnz   = offd->nz;
  } else {
    diag    = (Mat_SeqAIJ*)A->data;
    offd    = NULL;
    garray  = NULL;
    noffd   = 0;
    dnnz    = diag->nz;
    onnz    = 0;
  }

  /* create a temporary ParCSR */
  if (HYPRE_AssumedPartitionCheck()) {
   PetscMPIInt myid;

   ierr       = MPI_Comm_rank(comm,&myid);CHKERRQ(ierr);
   row_starts = A->rmap->range + myid;
   col_starts = A->cmap->range + myid;
  } else {
   row_starts = A->rmap->range;
   col_starts = A->cmap->range;
  }
  tA = hypre_ParCSRMatrixCreate(comm,A->rmap->N,A->cmap->N,(HYPRE_Int*)row_starts,(HYPRE_Int*)col_starts,noffd,dnnz,onnz);
  hypre_ParCSRMatrixSetRowStartsOwner(tA,0);
  hypre_ParCSRMatrixSetColStartsOwner(tA,0);

  /* set diagonal part */
  hdiag = hypre_ParCSRMatrixDiag(tA);
  hypre_CSRMatrixI(hdiag)           = (HYPRE_Int*)diag->i;
  hypre_CSRMatrixJ(hdiag)           = (HYPRE_Int*)diag->j;
  hypre_CSRMatrixData(hdiag)        = diag->a;
  hypre_CSRMatrixNumNonzeros(hdiag) = diag->nz;
  hypre_CSRMatrixSetRownnz(hdiag);
  hypre_CSRMatrixSetDataOwner(hdiag,0);

  /* set offdiagonal part */
  hoffd = hypre_ParCSRMatrixOffd(tA);
  if (offd) {
    hypre_CSRMatrixI(hoffd)           = (HYPRE_Int*)offd->i;
    hypre_CSRMatrixJ(hoffd)           = (HYPRE_Int*)offd->j;
    hypre_CSRMatrixData(hoffd)        = offd->a;
    hypre_CSRMatrixNumNonzeros(hoffd) = offd->nz;
    hypre_CSRMatrixSetRownnz(hoffd);
    hypre_CSRMatrixSetDataOwner(hoffd,0);
    hypre_ParCSRMatrixSetNumNonzeros(tA);
    hypre_ParCSRMatrixColMapOffd(tA) = (HYPRE_Int*)garray;
  }

  /* call RAP from BoomerAMG */
  /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
     from Pparcsr (even if it does not own them)! */
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hP->ij,(void**)&Pparcsr));
  P_owns_col_starts = hypre_ParCSRMatrixOwnsColStarts(Pparcsr);
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  PetscStackCallStandard(hypre_BoomerAMGBuildCoarseOperator,(Pparcsr,tA,Pparcsr,&ptapparcsr));
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  hypre_ParCSRMatrixSetRowStartsOwner(ptapparcsr,0);
  hypre_ParCSRMatrixSetColStartsOwner(ptapparcsr,0);
  if (P_owns_col_starts) hypre_ParCSRMatrixSetColStartsOwner(Pparcsr, 1);

  /* set pointers to NULL before destroying tA */
  hypre_CSRMatrixI(hdiag)          = NULL;
  hypre_CSRMatrixJ(hdiag)          = NULL;
  hypre_CSRMatrixData(hdiag)       = NULL;
  hypre_CSRMatrixI(hoffd)          = NULL;
  hypre_CSRMatrixJ(hoffd)          = NULL;
  hypre_CSRMatrixData(hoffd)       = NULL;
  hypre_ParCSRMatrixColMapOffd(tA) = NULL;
  hypre_ParCSRMatrixDestroy(tA);

  /* create C depending on mtype */
  sprintf(mtype,MATAIJ);
  ierr = PetscOptionsGetString(((PetscObject)A)->options,((PetscObject)A)->prefix,"-matptap_hypre_outtype",mtype,256,NULL);CHKERRQ(ierr);
  ierr = MatCreateFromParCSR(ptapparcsr,mtype,PETSC_OWN_POINTER,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAP_HYPRE_HYPRE"
static PetscErrorCode MatPtAP_HYPRE_HYPRE(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)
{
  hypre_ParCSRMatrix *Aparcsr,*Pparcsr,*ptapparcsr;
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data, *hP = (Mat_HYPRE*)P->data;
  HYPRE_Int          type,P_owns_col_starts;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported MAT_REUSE_MATRIX");
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hA->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObjectType,(hP->ij,&type));
  if (type != HYPRE_PARCSR) SETERRQ(PetscObjectComm((PetscObject)P),PETSC_ERR_SUP,"Only HYPRE_PARCSR is supported");
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&Aparcsr));
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hP->ij,(void**)&Pparcsr));

  /* call RAP from BoomerAMG */
  /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
     from Pparcsr (even if it does not own them)! */
  P_owns_col_starts = hypre_ParCSRMatrixOwnsColStarts(Pparcsr);
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  PetscStackCallStandard(hypre_BoomerAMGBuildCoarseOperator,(Pparcsr,Aparcsr,Pparcsr,&ptapparcsr));
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  hypre_ParCSRMatrixSetRowStartsOwner(ptapparcsr,0);
  hypre_ParCSRMatrixSetColStartsOwner(ptapparcsr,0);
  if (P_owns_col_starts) hypre_ParCSRMatrixSetColStartsOwner(Pparcsr, 1);

  /* create MatHYPRE */
  ierr = MatCreateFromParCSR(ptapparcsr,MATHYPRE,PETSC_OWN_POINTER,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_HYPRE"
static PetscErrorCode MatMultTranspose_HYPRE(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHYPRE_MultKernel_Private(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_HYPRE"
static PetscErrorCode MatMult_HYPRE(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHYPRE_MultKernel_Private(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_MultKernel_Private"
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat A, Vec x, Vec y, PetscBool trans)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  hypre_ParCSRMatrix *parcsr;
  hypre_ParVector    *hx,*hy;
  PetscScalar        *ax,*ay,*sax,*say;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(hA->ij,(void**)&parcsr));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(hA->x,(void**)&hx));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(hA->b,(void**)&hy));
  ierr = VecGetArrayRead(x,(const PetscScalar**)&ax);CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay);CHKERRQ(ierr);
  if (trans) {
    VecHYPRE_ParVectorReplacePointer(hA->x,ay,say);
    VecHYPRE_ParVectorReplacePointer(hA->b,ax,sax);
    hypre_ParCSRMatrixMatvecT(1.,parcsr,hy,0.,hx);
    VecHYPRE_ParVectorReplacePointer(hA->x,say,ay);
    VecHYPRE_ParVectorReplacePointer(hA->b,sax,ax);
  } else {
    VecHYPRE_ParVectorReplacePointer(hA->x,ax,sax);
    VecHYPRE_ParVectorReplacePointer(hA->b,ay,say);
    hypre_ParCSRMatrixMatvec(1.,parcsr,hx,0.,hy);
    VecHYPRE_ParVectorReplacePointer(hA->x,sax,ax);
    VecHYPRE_ParVectorReplacePointer(hA->b,say,ay);
  }
  ierr = VecRestoreArrayRead(x,(const PetscScalar**)&ax);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_HYPRE"
static PetscErrorCode MatDestroy_HYPRE(Mat A)
{
  Mat_HYPRE      *hA = (Mat_HYPRE*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (hA->x) PetscStackCallStandard(HYPRE_IJVectorDestroy,(hA->x));
  if (hA->b) PetscStackCallStandard(HYPRE_IJVectorDestroy,(hA->b));
  if (hA->ij) {
    if (!hA->inner_free) hypre_IJMatrixObject(hA->ij) = NULL;
    PetscStackCallStandard(HYPRE_IJMatrixDestroy,(hA->ij));
  }
  if (hA->comm) { ierr = MPI_Comm_free(&hA->comm);CHKERRQ(ierr); }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_hypre_aij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_hypre_is_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatPtAP_seqaij_hypre_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatPtAP_mpiaij_hypre_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHYPRESetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHYPREGetParCSR_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_HYPRE"
static PetscErrorCode MatSetUp_HYPRE(Mat A)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  Vec                x,b;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (hA->x) PetscFunctionReturn(0);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),1,A->cmap->n,A->cmap->N,NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),1,A->rmap->n,A->rmap->N,NULL,&b);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorCreate(x,&hA->x);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorCreate(b,&hA->b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_HYPRE"
static PetscErrorCode MatAssemblyEnd_HYPRE(Mat A, MatAssemblyType mode)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MAT_FLUSH_ASSEMBLY currently not supported with MATHYPRE");
  PetscStackCallStandard(HYPRE_IJMatrixAssemble,(hA->ij));
  PetscFunctionReturn(0);
}

#define MATHYPRE_SCRATCH 2048

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_HYPRE"
PetscErrorCode MatSetValues_HYPRE(Mat A, PetscInt nr, const PetscInt rows[], PetscInt nc, const PetscInt cols[], const PetscScalar v[], InsertMode ins)
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  PetscScalar        *vals = (PetscScalar *)v;
  PetscScalar        sscr[MATHYPRE_SCRATCH];
  HYPRE_Int          cscr[2][MATHYPRE_SCRATCH];
  HYPRE_Int          i,nzc;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  for (i=0,nzc=0;i<nc;i++) {
    if (cols[i] >= 0) {
      cscr[0][nzc  ] = cols[i];
      cscr[1][nzc++] = i;
    }
  }
  if (!nzc) PetscFunctionReturn(0);

  if (ins == ADD_VALUES) {
    for (i=0;i<nr;i++) {
      if (rows[i] >= 0) {
        PetscInt j;
        for (j=0;j<nzc;j++) sscr[j] = vals[cscr[1][j]];
        PetscStackCallStandard(HYPRE_IJMatrixAddToValues,(hA->ij,1,&nzc,(HYPRE_Int*)(rows+i),cscr[0],sscr));
      }
      vals += nc;
    }
  } else { /* INSERT_VALUES */
#if defined(PETSC_USE_DEBUG)
    /* Insert values cannot be used to insert offproc entries */
    PetscInt rst,ren;
    ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
    for (i=0;i<nr;i++)
      if (rows[i] >= 0 && (rows[i] < rst || rows[i] >= ren)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use INSERT_VALUES for off-proc entries with MatHYPRE. Use ADD_VALUES instead");
#endif
    for (i=0;i<nr;i++) {
      if (rows[i] >= 0) {
        PetscInt j;
        for (j=0;j<nzc;j++) sscr[j] = vals[cscr[1][j]];
        PetscStackCallStandard(HYPRE_IJMatrixSetValues,(hA->ij,1,&nzc,(HYPRE_Int*)(rows+i),cscr[0],sscr));
      }
      vals += nc;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRESetPreallocation_HYPRE"
static PetscErrorCode MatHYPRESetPreallocation_HYPRE(Mat A, PetscInt dnz, const PetscInt dnnz[], PetscInt onz, const PetscInt onnz[])
{
  Mat_HYPRE          *hA = (Mat_HYPRE*)A->data;
  HYPRE_Int          *hdnnz,*honnz;
  PetscInt           i,rs,re,cs,ce;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
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
    HYPRE_Int hrs,hre,hcs,hce;
    PetscStackCallStandard(HYPRE_IJMatrixGetLocalRange,(hA->ij,&hrs,&hre,&hcs,&hce));
    if (hre-hrs+1 != re -rs) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent local rows: IJMatrix [%D,%D), PETSc [%D,%d)",hrs,hre+1,rs,re);
    if (hce-hcs+1 != ce -cs) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent local cols: IJMatrix [%D,%D), PETSc [%D,%d)",hcs,hce+1,cs,ce);
  }
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(hA->ij));

  if (!dnnz) {
    ierr = PetscMalloc1(A->rmap->n,&hdnnz);CHKERRQ(ierr);
    for (i=0;i<A->rmap->n;i++) hdnnz[i] = dnz;
  } else {
    hdnnz = (HYPRE_Int*)dnnz;
  }
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1) {
    if (!onnz) {
      ierr = PetscMalloc1(A->rmap->n,&honnz);CHKERRQ(ierr);
      for (i=0;i<A->rmap->n;i++) honnz[i] = onz;
    } else {
      honnz = (HYPRE_Int*)onnz;
    }
    PetscStackCallStandard(HYPRE_IJMatrixSetDiagOffdSizes,(hA->ij,hdnnz,honnz));
  } else {
    honnz = NULL;
    PetscStackCallStandard(HYPRE_IJMatrixSetRowSizes,(hA->ij,hdnnz));
  }
  if (!dnnz) {
    ierr = PetscFree(hdnnz);CHKERRQ(ierr);
  }
  if (!onnz && honnz) {
    ierr = PetscFree(honnz);CHKERRQ(ierr);
  }
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* SetDiagOffdSizes sets hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0 */
  {
    hypre_AuxParCSRMatrix *aux_matrix;
    aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(hA->ij);
    hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 1;
  }
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

   Notes: If the *nnz parameter is given then the *nz parameter is ignored; for sequential matrices, onz and onnz are ignored.

   Level: intermediate

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatMPIAIJSetPreallocation, MATHYPRE
@*/
#undef __FUNCT__
#define __FUNCT__ "MatHYPRESetPreallocation"
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
+  vparcsr  - the pointer to the hypre_ParCSRMatrix
.  mtype    - matrix type to be created. Currently MATAIJ, MATIS and MATHYPRE are supported.
-  copymode - PETSc copying options

   Output Parameter:
.  A  - the matrix

   Level: intermediate

.seealso: MatHYPRE, PetscCopyMode
*/
#undef __FUNCT__
#define __FUNCT__ "MatCreateFromParCSR"
PETSC_EXTERN PetscErrorCode MatCreateFromParCSR(hypre_ParCSRMatrix *vparcsr, MatType mtype, PetscCopyMode copymode, Mat* A)
{
  Mat                   T;
  Mat_HYPRE             *hA;
  hypre_ParCSRMatrix    *parcsr;
  MPI_Comm              comm;
  PetscInt              rstart,rend,cstart,cend,M,N;
  PetscBool             isseqaij,ismpiaij,isaij,ishyp,isis;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  parcsr = (hypre_ParCSRMatrix *)vparcsr;
  comm   = hypre_ParCSRMatrixComm(parcsr);
  ierr   = PetscStrcmp(mtype,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  ierr   = PetscStrcmp(mtype,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr   = PetscStrcmp(mtype,MATAIJ,&isaij);CHKERRQ(ierr);
  ierr   = PetscStrcmp(mtype,MATHYPRE,&ishyp);CHKERRQ(ierr);
  ierr   = PetscStrcmp(mtype,MATIS,&isis);CHKERRQ(ierr);
  isaij  = (PetscBool)(isseqaij || ismpiaij || isaij);
  if (!isaij && !ishyp && !isis) SETERRQ6(comm,PETSC_ERR_SUP,"Unsupported MatType %s! Supported types are %s, %s, %s, %s, and %s",mtype,MATAIJ,MATSEQAIJ,MATMPIAIJ,MATIS,MATHYPRE);
  if (ishyp && copymode == PETSC_COPY_VALUES) SETERRQ(comm,PETSC_ERR_SUP,"Unsupported copymode PETSC_COPY_VALUES");

  /* access ParCSRMatrix */
  rstart = hypre_ParCSRMatrixFirstRowIndex(parcsr);
  rend   = hypre_ParCSRMatrixLastRowIndex(parcsr);
  cstart = hypre_ParCSRMatrixFirstColDiag(parcsr);
  cend   = hypre_ParCSRMatrixLastColDiag(parcsr);
  M      = hypre_ParCSRMatrixGlobalNumRows(parcsr);
  N      = hypre_ParCSRMatrixGlobalNumCols(parcsr);

  /* create PETSc matrix with MatHYPRE */
  ierr = MatCreate(comm,&T);CHKERRQ(ierr);
  ierr = MatSetSizes(T,rend-rstart+1,cend-cstart+1,M,N);CHKERRQ(ierr);
  ierr = MatSetType(T,MATHYPRE);CHKERRQ(ierr);
  ierr = MatSetUp(T);CHKERRQ(ierr);
  hA   = (Mat_HYPRE*)(T->data);

  /* create HYPRE_IJMatrix */
  PetscStackCallStandard(HYPRE_IJMatrixCreate,(hA->comm,rstart,rend-1,cstart,cend-1,&hA->ij));

  /* set ParCSR object */
  PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,(hA->ij,HYPRE_PARCSR));
  hypre_IJMatrixObject(hA->ij) = parcsr;

  /* set assembled flag */
  hypre_IJMatrixAssembleFlag(hA->ij) = 1;
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(hA->ij));
  if (ishyp) {
    /* prevent from freeing the pointer */
    if (copymode == PETSC_USE_POINTER) hA->inner_free = PETSC_FALSE;
    *A = T;
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

#undef __FUNCT__
#define __FUNCT__ "MatHYPREGetParCSR_HYPRE"
PetscErrorCode MatHYPREGetParCSR_HYPRE(Mat A, hypre_ParCSRMatrix **parcsr)
{
  Mat_HYPRE*            hA = (Mat_HYPRE*)A->data;
  HYPRE_Int             type;
  PetscErrorCode        ierr;

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
#undef __FUNCT__
#define __FUNCT__ "MatHYPREGetParCSR"
PetscErrorCode MatHYPREGetParCSR(Mat A, hypre_ParCSRMatrix **parcsr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscUseMethod(A,"MatHYPREGetParCSR_C",(Mat,hypre_ParCSRMatrix**),(A,parcsr));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_HYPRE"
PETSC_EXTERN PetscErrorCode MatCreate_HYPRE(Mat B)
{
  Mat_HYPRE      *hB;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr           = PetscNewLog(B,&hB);CHKERRQ(ierr);
  hB->inner_free = PETSC_TRUE;

  B->data       = (void*)hB;
  B->rmap->bs   = 1;
  B->assembled  = PETSC_FALSE;

  B->ops->mult          = MatMult_HYPRE;
  B->ops->multtranspose = MatMultTranspose_HYPRE;
  B->ops->setup         = MatSetUp_HYPRE;
  B->ops->destroy       = MatDestroy_HYPRE;
  B->ops->assemblyend   = MatAssemblyEnd_HYPRE;
  B->ops->ptap          = MatPtAP_HYPRE_HYPRE;
  B->ops->setvalues     = MatSetValues_HYPRE;

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)B),&hB->comm);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATHYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_hypre_aij_C",MatConvert_HYPRE_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_hypre_is_C",MatConvert_HYPRE_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_seqaij_hypre_C",MatPtAP_AIJ_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_mpiaij_hypre_C",MatPtAP_AIJ_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatHYPRESetPreallocation_C",MatHYPRESetPreallocation_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatHYPREGetParCSR_C",MatHYPREGetParCSR_HYPRE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "hypre_array_destroy"
static PetscErrorCode hypre_array_destroy(void *ptr)
{
   PetscFunctionBegin;
   hypre_TFree(ptr);
   PetscFunctionReturn(0);
}

#if 0
/*
    Does NOT copy the data over, instead uses DIRECTLY the pointers from the PETSc MPIAIJ format

    This is UNFINISHED and does NOT work! The problem is that hypre puts the diagonal entry first
    which will corrupt the PETSc data structure if we did this. Need a work around to this problem.
*/
#include <_hypre_IJ_mv.h>
#include <HYPRE_IJ_mv.h>

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixLink"
PetscErrorCode MatHYPRE_IJMatrixLink(Mat A, HYPRE_IJMatrix *ij)
{
  PetscErrorCode        ierr;
  PetscInt              rstart,rend,cstart,cend;
  PetscBool             flg;
  hypre_AuxParCSRMatrix *aux_matrix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Can only use with PETSc MPIAIJ matrices");
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr   = PetscLogEventBegin(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  rstart = A->rmap->rstart;
  rend   = A->rmap->rend;
  cstart = A->cmap->rstart;
  cend   = A->cmap->rend;
  PetscStackCallStandard(HYPRE_IJMatrixCreate,(PetscObjectComm((PetscObject)A),rstart,rend-1,cstart,cend-1,ij));
  PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,(*ij,HYPRE_PARCSR));

  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(*ij));
  PetscStackCall("hypre_IJMatrixTranslator",aux_matrix = (hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(*ij));

  hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;

  /* this is the Hack part where we monkey directly with the hypre datastructures */

  PetscStackCallStandard(HYPRE_IJMatrixAssemble,(*ij));
  ierr = PetscLogEventEnd(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
