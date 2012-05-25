#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation_MPIAIJCUSPARSE"
PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ     *b;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (d_nz == PETSC_DEFAULT || d_nz == PETSC_DECIDE) d_nz = 5;
  if (o_nz == PETSC_DEFAULT || o_nz == PETSC_DECIDE) o_nz = 2;
  if (d_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %D",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %D",o_nz);

  ierr = PetscLayoutSetBlockSize(B->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  b = (Mat_MPIAIJ*)B->data;

  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQAIJ matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);
  }

  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  ierr=MatSetOption_SeqAIJCUSPARSE(b->A,b->diagGPUMatFormat,PETSC_TRUE);CHKERRQ(ierr);
  ierr=MatSetOption_SeqAIJCUSPARSE(b->B,b->offdiagGPUMatFormat,PETSC_TRUE);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "MatGetVecs_MPIAIJCUSPARSE"
PetscErrorCode  MatGetVecs_MPIAIJCUSPARSE(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (right) {
    ierr = VecCreate(((PetscObject)mat)->comm,right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecCreate(((PetscObject)mat)->comm,left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMult_MPIAIJCUSPARSE"
PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  // This multiplication sequence is different sequence
  // than the CPU version. In particular, the diagonal block
  // multiplication kernel is launched in one stream. Then,
  // in a separate stream, the data transfers from DeviceToHost
  // (with MPI messaging in between), then HostToDevice are 
  // launched. Once the data transfer stream is synchronized,
  // to ensure messaging is complete, the MatMultAdd kernel
  // is launched in the original (MatMult) stream to protect
  // against race conditions.
  //
  // This sequence should only be called for GPU computation.
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);        
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPIAIJCUSPARSE"
PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  // This multiplication sequence is different sequence
  // than the CPU version. In particular, the diagonal block
  // multiplication kernel is launched in one stream. Then,
  // in a separate stream, the data transfers from DeviceToHost
  // (with MPI messaging in between), then HostToDevice are 
  // launched. Once the data transfer stream is synchronized,
  // to ensure messaging is complete, the MatMultAdd kernel
  // is launched in the original (MatMult) stream to protect
  // against race conditions.
  //
  // This sequence should only be called for GPU computation.
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multtransposeadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);        
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode  MatCreate_MPIAIJ(Mat);
EXTERN_C_END
//PetscErrorCode MatSetValuesBatch_MPIAIJCUSPARSE(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, const PetscScalar *elemMats);


#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIAIJCUSPARSE"
PetscErrorCode MatSetOption_MPIAIJCUSPARSE(Mat A,MatOption op,PetscBool  flg)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatSetOption_MPIAIJ(A,op,flg);CHKERRQ(ierr);
  switch (op) {
  case DIAGBLOCK_MAT_CSR:
    a->diagGPUMatFormat = DIAGBLOCK_MAT_CSR;
    break;
  case OFFDIAGBLOCK_MAT_CSR:
    a->offdiagGPUMatFormat = OFFDIAGBLOCK_MAT_CSR;
    break;
  case DIAGBLOCK_MAT_DIA:
  case OFFDIAGBLOCK_MAT_DIA:
  case MAT_DIA:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported GPU matrix storage format DIA for (MPI,SEQ)AIJCUSPARSE matrix type.");
  case DIAGBLOCK_MAT_ELL:
    a->diagGPUMatFormat = DIAGBLOCK_MAT_ELL;
    break;
  case OFFDIAGBLOCK_MAT_ELL:
    a->offdiagGPUMatFormat = OFFDIAGBLOCK_MAT_ELL;
    break;
  case DIAGBLOCK_MAT_HYB:
    a->diagGPUMatFormat = DIAGBLOCK_MAT_HYB;
    break;
  case OFFDIAGBLOCK_MAT_HYB:
    a->offdiagGPUMatFormat = OFFDIAGBLOCK_MAT_HYB;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSetFromOptions_MPIAIJCUSPARSE"
PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode     ierr;
  PetscInt       idxDiag=0,idxOffDiag=0;
  char * formats[]={CSR,ELL,HYB};
  MatOption diagFormat, offdiagFormat;
  PetscBool      flg;
  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"When using TxPETSCGPU, MPIAIJCUSPARSE Options","Mat");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEList("-mat_mult_cusparse_diag_storage_format",
			     "Set the storage format of (mpi)aijcusparse gpu matrices for SpMV",
			     "None",formats,3,formats[0],&idxDiag,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mat_mult_cusparse_offdiag_storage_format",
			     "Set the storage format of (mpi)aijcusparse gpu matrices for SpMV",
			     "None",formats,3,formats[0],&idxOffDiag,&flg);CHKERRQ(ierr);

    //printf("MatSetFromOptions_MPIAIJCUSPARSE : %s %s\n",formats[idxDiag],formats[idxOffDiag]);
    if (formats[idxDiag] == CSR)
      diagFormat=MAT_CSR;
    else if (formats[idxDiag] == ELL)
      diagFormat=MAT_ELL;
    else if (formats[idxDiag] == HYB)
      diagFormat=MAT_HYB;

    if (formats[idxOffDiag] == CSR)
      offdiagFormat=MAT_CSR;
    else if (formats[idxOffDiag] == ELL)
      offdiagFormat=MAT_ELL;
    else if (formats[idxOffDiag] == HYB)
      offdiagFormat=MAT_HYB;

    a->diagGPUMatFormat = diagFormat;
    a->offdiagGPUMatFormat = offdiagFormat;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIAIJCUSPARSE"
PetscErrorCode  MatCreate_MPIAIJCUSPARSE(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_MPIAIJ(B);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAIJSetPreallocation_C",
                                     "MatMPIAIJSetPreallocation_MPIAIJCUSPARSE",
                                      MatMPIAIJSetPreallocation_MPIAIJCUSPARSE);CHKERRQ(ierr);
  B->ops->getvecs        = MatGetVecs_MPIAIJCUSPARSE;
  B->ops->mult           = MatMult_MPIAIJCUSPARSE;
  B->ops->multtranspose  = MatMultTranspose_MPIAIJCUSPARSE;
  B->ops->setfromoptions = MatSetFromOptions_MPIAIJCUSPARSE;	
  B->ops->setoption      = MatSetOption_MPIAIJCUSPARSE;	
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "MatCreateAIJCUSPARSE"
PetscErrorCode  MatCreateAIJCUSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJCUSPARSE - MATAIJCUSPARSE = "aijcusparse" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJCUSPARSE when constructed with a single process communicator,
   and MATMPIAIJCUSPARSE otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type mpiaijcusparse - sets the matrix type to "mpiaijcusparse" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJ,MATSEQAIJ,MATMPIAIJ, MATMPIAIJCUSPARSE, MATSEQAIJCUSPARSE
M*/

