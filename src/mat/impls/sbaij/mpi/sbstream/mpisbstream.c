#define PETSCMAT_DLL

#include "../src/mat/impls/sbaij/mpi/mpisbaij.h"
#include "../src/mat/impls/sbaij/seq/sbstream/sbstream.h"

extern PetscErrorCode MatMult_SeqSBSTRM_4(Mat,Vec,Vec);
extern PetscErrorCode MatMult_SeqSBSTRM_5(Mat,Vec,Vec);
extern PetscErrorCode MatMultTranspose_SeqBSTRM_4(Mat,Vec,Vec);
extern PetscErrorCode MatMultTranspose_SeqBSTRM_5(Mat,Vec,Vec);
extern PetscErrorCode MatMultAdd_SeqSBSTRM_4(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatMultAdd_SeqSBSTRM_5(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatMultAdd_SeqBSTRM_4(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatMultAdd_SeqBSTRM_5(Mat,Vec,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "MPISBSTRM_create_sbstrm"
PetscErrorCode MPISBSTRM_create_sbstrm(Mat A)
{
  Mat_MPISBAIJ     *a = (Mat_MPISBAIJ *)A->data;
  Mat_SeqSBAIJ     *Aij = (Mat_SeqSBAIJ*)(a->A->data), *Bij = (Mat_SeqSBAIJ*)(a->B->data);
  /* 
  */
  Mat_SeqSBSTRM   *sbstrmA, *sbstrmB;
  PetscInt       MROW = Aij->mbs, bs = a->A->rmap->bs;

  /* PetscInt       m = A->rmap->n;*/  /* Number of rows in the matrix. */
  /* PetscInt       nd = a->A->cmap->n;*/ /* number of columns in diagonal portion */
  PetscInt       *ai = Aij->i, *bi = Bij->i;  /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i,j,k;
  PetscScalar    *aa = Aij->a,*ba = Bij->a;

  PetscInt      bs2,  rbs, cbs, slen, blen;
  PetscErrorCode ierr;
  PetscScalar **asp;
  PetscScalar **bsp;

  PetscFunctionBegin;
  /* printf(" --- in MPISBSTRM_create_sbstrm, m=%d, nd=%d, bs=%d, MROW=%d\n", m,nd,bs,MROW); */

  rbs = cbs = bs;
  bs2 = bs*bs;
  blen = ai[MROW]-ai[0];
  slen = blen*bs;
  
  /* printf(" --- blen=%d, slen=%d\n", blen, slen);  */

  ierr = PetscNewLog(a->A,Mat_SeqSBSTRM,&sbstrmA);CHKERRQ(ierr);
  a->A->spptr = (void *) sbstrmA;
  sbstrmA = (Mat_SeqSBSTRM*) a->A->spptr;
  sbstrmA->rbs = sbstrmA->cbs = bs;
  ierr  = PetscMalloc(bs2*blen*sizeof(PetscScalar), &sbstrmA->as);CHKERRQ(ierr);

  ierr  = PetscMalloc(rbs*sizeof(PetscScalar *), &asp);CHKERRQ(ierr);

  for(i=0;i<rbs;i++) asp[i] = sbstrmA->as + i*slen;

  for (k=0; k<blen; k++) {
    for (j=0; j<cbs; j++)
    for (i=0; i<rbs; i++)
        asp[i][k*cbs+j] = aa[k*bs2+j*rbs+i];
  }
  switch (bs){
    case 4:
      a->A->ops->mult          = MatMult_SeqSBSTRM_4;
      a->A->ops->multtranspose = MatMult_SeqSBSTRM_4;
      /* a->A->ops->sor   = MatSOR_SeqSBSTRM_4;  */
      break;
    case 5:
      a->A->ops->mult          = MatMult_SeqSBSTRM_5;
      a->A->ops->multtranspose = MatMult_SeqSBSTRM_5;
      /* a->A->ops->sor   = MatSOR_SeqSBSTRM_5;  */
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D yet",bs);
  }
  ierr = PetscFree(asp);CHKERRQ(ierr);


/*.....*/
  blen = bi[MROW]-bi[0];
  slen = blen*bs;
  ierr = PetscNewLog(a->B,Mat_SeqSBSTRM,&sbstrmB);CHKERRQ(ierr);
  a->B->spptr = (void *) sbstrmB;
  sbstrmB = (Mat_SeqSBSTRM*) a->B->spptr;
  sbstrmB->rbs = sbstrmB->cbs = bs;
  ierr  = PetscMalloc(bs2*blen*sizeof(PetscScalar), &sbstrmB->as);CHKERRQ(ierr);

  ierr  = PetscMalloc(rbs*sizeof(PetscScalar *), &bsp);CHKERRQ(ierr);

  for(i=0;i<rbs;i++) bsp[i] = sbstrmB->as + i*slen;

  for (k=0; k<blen; k++) {
    for (j=0; j<cbs; j++)
    for (i=0; i<rbs; i++)
        bsp[i][k*cbs+j] = ba[k*bs2+j*rbs+i];
  }
  switch (bs){
    case 4:
      /* a->B->ops->mult             = MatMult_SeqSBSTRM_4; */
      a->B->ops->multtranspose    = MatMultTranspose_SeqBSTRM_4;
      a->B->ops->multadd          = MatMultAdd_SeqBSTRM_4;
      /* a->B->ops->multtransposeadd = MatMultAdd_SeqSBSTRM_4; */
      break;
    case 5:
      /* a->B->ops->mult             = MatMult_SeqSBSTRM_5; */
      a->B->ops->multtranspose    = MatMultTranspose_SeqBSTRM_5;
      a->B->ops->multadd          = MatMultAdd_SeqBSTRM_5;
      /* a->B->ops->multtransposeadd = MatMultAdd_SeqSBSTRM_5; */
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D yet",bs);
  }
  ierr = PetscFree(bsp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


extern PetscErrorCode MatAssemblyEnd_MPISBAIJ(Mat,MatAssemblyType);
#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_MPISBSTRM"
PetscErrorCode MatAssemblyEnd_MPISBSTRM(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
    Aij->inode.use = PETSC_FALSE;
    Bij->inode.use = PETSC_FALSE;
  */ 
  ierr = MatAssemblyEnd_MPISBAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Now calculate the permutation and grouping information. */
  ierr = MPISBSTRM_create_sbstrm(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPISBSTRM"
PetscErrorCode MatCreateMPISBSTRM(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPISBSTRM);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQSBSTRM);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode MatConvert_SeqSBAIJ_SeqSBSTRM(Mat,const MatType,MatReuse,Mat*);
extern PetscErrorCode MatMPISBAIJSetPreallocation_MPISBAIJ(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPISBAIJSetPreallocation_MPISBSTRM"
PetscErrorCode   MatMPISBAIJSetPreallocation_MPISBSTRM(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPISBAIJSetPreallocation_MPISBAIJ(B,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
/*
  ierr = MatConvert_SeqSBAIJ_SeqSBSTRM(b->A, MATSEQSBSTRM, MAT_REUSE_MATRIX, &b->A);CHKERRQ(ierr);
  ierr = MatConvert_SeqSBAIJ_SeqSBSTRM(b->B, MATSEQSBSTRM, MAT_REUSE_MATRIX, &b->B);CHKERRQ(ierr);
*/
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPISBAIJ_MPISBSTRM"
PetscErrorCode   MatConvert_MPISBAIJ_MPISBSTRM(Mat A,const MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqSBSTRM  *sbstrm;

  PetscFunctionBegin;

  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  /* printf(" --- in MatConvert_MPISBAIJ_MPISBSTRM  -- 1 \n"); */

  ierr = PetscNewLog(B,   Mat_SeqSBSTRM,&sbstrm);CHKERRQ(ierr);
  B->spptr    = (void *) sbstrm;

  /* Set function pointers for methods that we inherit from AIJ but override.
     B->ops->duplicate   = MatDuplicate_SBSTRM;
     B->ops->mult        = MatMult_SBSTRM;
     B->ops->destroy     = MatDestroy_MPISBSTRM;
  */
  B->ops->assemblyend = MatAssemblyEnd_MPISBSTRM;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    ierr = MPISBSTRM_create_sbstrm(B);CHKERRQ(ierr);
  }

  ierr = PetscObjectChangeTypeName( (PetscObject) B, MATMPISBSTRM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPISBAIJSetPreallocation_C",
				     "MatMPISBAIJSetPreallocation_MPISBSTRM",
				     MatMPISBAIJSetPreallocation_MPISBSTRM);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPISBSTRM"
PetscErrorCode   MatCreate_MPISBSTRM(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatConvert_MPISBAIJ_MPISBSTRM(A,MATMPISBSTRM,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SBSTRM"
PetscErrorCode   MatCreate_SBSTRM(Mat A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQSBSTRM);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPISBSTRM);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

