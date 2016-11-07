
/*
    Creates hypre ijmatrix from PETSc matrix
*/
#include <petscsys.h>
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/hypre/mhypre.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <HYPRE_struct_mv.h>
#include <HYPRE_struct_ls.h>
#include <_hypre_struct_mv.h>

static PetscErrorCode MatHYPRE_CreateFromMat(Mat,Mat_HYPRE*);
static PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat,Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_MPIAIJ(Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_IJMatrixFastCopy_SeqAIJ(Mat,HYPRE_IJMatrix);
static PetscErrorCode MatHYPRE_MultKernel_Private(Mat,Vec,Vec,PetscBool);

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
  ierr   = MatSetUp(A);CHKERRQ(ierr);
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
  int                   type;
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
  int                   type;
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
  PetscInt               *col_map_offd,*hdi,*hdj,*hoi,*hoj;
  PetscInt               *ii,*jj,*iptr,*jptr;
  PetscInt               cum,dr,dc,oc,str,stc,nnz,i,jd,jo,M,N;
  int                    type;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
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
    comm = PetscObjectComm((PetscObject)A);
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
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,dr,dc+oc,iptr,jptr,data,&lA);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(*B,lA);CHKERRQ(ierr);
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
  int                type;
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
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc1(m+1,&dii);CHKERRQ(ierr);
    ierr = PetscMalloc1(dnnz,&djj);CHKERRQ(ierr);
    ierr = PetscMalloc1(dnnz,&da);CHKERRQ(ierr);
  } else {
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
    PetscInt *offdj,*coffd;

    if (reuse != MAT_REUSE_MATRIX) {
      ierr = PetscMalloc1(m+1,&oii);CHKERRQ(ierr);
      ierr = PetscMalloc1(onnz,&ojj);CHKERRQ(ierr);
      ierr = PetscMalloc1(onnz,&oa);CHKERRQ(ierr);
    } else {
      Mat_MPIAIJ *b = (Mat_MPIAIJ*)((*B)->data);
      PetscInt   nr,hr = hypre_CSRMatrixNumRows(hoffd);
      PetscBool  done;

      ierr = MatGetRowIJ(b->B,0,PETSC_FALSE,PETSC_FALSE,&nr,(const PetscInt**)&oii,(const PetscInt**)&ojj,&done);CHKERRQ(ierr);
      if (nr != hr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of local rows in offdiag part! %D != %D",nr,hr);
      if (oii[nr] < onnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot reuse mat: invalid number of nonzeros in offdiag part! reuse %D hypre %D",oii[nr],onnz);
      ierr = MatSeqAIJGetArray(b->B,&oa);CHKERRQ(ierr);
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
    if (reuse != MAT_REUSE_MATRIX) {
      Mat_MPIAIJ *b;
      Mat_SeqAIJ *d,*o;
      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,
                                            djj,da,oii,ojj,oa,B);CHKERRQ(ierr);
      /* hack MPIAIJ */
      b          = (Mat_MPIAIJ*)((*B)->data);
      d          = (Mat_SeqAIJ*)b->A->data;
      o          = (Mat_SeqAIJ*)b->B->data;
      d->free_a  = PETSC_TRUE;
      d->free_ij = PETSC_TRUE;
      o->free_a  = PETSC_TRUE;
      o->free_ij = PETSC_TRUE;
    }
  } else if (reuse != MAT_REUSE_MATRIX) {
    Mat_SeqAIJ* b;
    ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,B);CHKERRQ(ierr);
    /* hack SeqAIJ */
    b          = (Mat_SeqAIJ*)((*B)->data);
    b->free_a  = PETSC_TRUE;
    b->free_ij = PETSC_TRUE;
  }
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  }
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
    HYPREReplacePointer(hA->x,ay,say);
    HYPREReplacePointer(hA->b,ax,sax);
    hypre_ParCSRMatrixMatvecT(1.,parcsr,hy,0.,hx);
    HYPREReplacePointer(hA->x,say,ay);
    HYPREReplacePointer(hA->b,sax,ax);
  } else {
    HYPREReplacePointer(hA->x,ax,sax);
    HYPREReplacePointer(hA->b,ay,say);
    hypre_ParCSRMatrixMatvec(1.,parcsr,hx,0.,hy);
    HYPREReplacePointer(hA->x,sax,ax);
    HYPREReplacePointer(hA->b,say,ay);
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
  PetscStackCallStandard(HYPRE_IJMatrixAssemble,(hA->ij));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHYPRECreateFromParCSR"
PETSC_EXTERN PetscErrorCode MatHYPRECreateFromParCSR(void *vparcsr, PetscCopyMode copymode, Mat* A)
{
  Mat_HYPRE             *hA;
  hypre_ParCSRMatrix    *parcsr;
  MPI_Comm              comm;
  PetscInt              rstart,rend,cstart,cend,M,N;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  parcsr = (hypre_ParCSRMatrix *)vparcsr;
  comm   = hypre_ParCSRMatrixComm(parcsr);
  if (copymode == PETSC_COPY_VALUES) SETERRQ(comm,PETSC_ERR_SUP,"Unsupported copymode PETSC_COPY_VALUES");

  /* access ParCSRMatrix */
  rstart = hypre_ParCSRMatrixFirstRowIndex(parcsr);
  rend   = hypre_ParCSRMatrixLastRowIndex(parcsr);
  cstart = hypre_ParCSRMatrixFirstColDiag(parcsr);
  cend   = hypre_ParCSRMatrixLastColDiag(parcsr);
  M      = hypre_ParCSRMatrixGlobalNumRows(parcsr);
  N      = hypre_ParCSRMatrixGlobalNumCols(parcsr);

  /* create PETSc matrix with MatHYPRE */
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,rend-rstart+1,cend-cstart+1,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATHYPRE);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  hA   = (Mat_HYPRE*)((*A)->data);

  /* create HYPRE_IJMatrix */
  PetscStackCallStandard(HYPRE_IJMatrixCreate,(hA->comm,rstart,rend-1,cstart,cend-1,&hA->ij));

  /* set ParCSR object */
  PetscStackCallStandard(HYPRE_IJMatrixSetObjectType,(hA->ij,HYPRE_PARCSR));
  hypre_IJMatrixObject(hA->ij) = parcsr;

  /* set assembled flag */
  hypre_IJMatrixAssembleFlag(hA->ij) = 1;
  PetscStackCallStandard(HYPRE_IJMatrixInitialize,(hA->ij));

  /* prevent from freeing the pointer */
  if (copymode == PETSC_USE_POINTER) hA->inner_free = PETSC_FALSE;

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)B),&hB->comm);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATHYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_hypre_aij_C",MatConvert_HYPRE_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_hypre_is_C",MatConvert_HYPRE_IS);CHKERRQ(ierr);
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
