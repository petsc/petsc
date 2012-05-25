#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/

//#ifdef PETSC_HAVE_TXPETSCGPU
//extern PetscErrorCode MatSetOption_SeqAIJCUSP(Mat,MatOption,PetscBool);
//#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation_MPIAIJCUSP"
PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJCUSP(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ     *b;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (d_nz == PETSC_DEFAULT || d_nz == PETSC_DECIDE) d_nz = 5;
  if (o_nz == PETSC_DEFAULT || o_nz == PETSC_DECIDE) o_nz = 2;
  if (d_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %D",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %D",o_nz);

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
    ierr = MatSetType(b->A,MATSEQAIJCUSP);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQAIJCUSP);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);
  }

  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
#ifdef PETSC_HAVE_TXPETSCGPU
  ierr=MatSetOption_SeqAIJCUSP(b->A,b->diagGPUMatFormat,PETSC_TRUE);CHKERRQ(ierr);
  ierr=MatSetOption_SeqAIJCUSP(b->B,b->offdiagGPUMatFormat,PETSC_TRUE);CHKERRQ(ierr);
#endif
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "MatGetVecs_MPIAIJCUSP"
PetscErrorCode  MatGetVecs_MPIAIJCUSP(Mat mat,Vec *right,Vec *left)
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


#ifdef PETSC_HAVE_TXPETSCGPU
#undef __FUNCT__
#define __FUNCT__ "MatMult_MPIAIJCUSP"
PetscErrorCode MatMult_MPIAIJCUSP(Mat A,Vec xx,Vec yy)
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
#endif 

EXTERN_C_BEGIN
PetscErrorCode  MatCreate_MPIAIJ(Mat);
EXTERN_C_END
PetscErrorCode MatSetValuesBatch_MPIAIJCUSP(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, const PetscScalar *elemMats);

#ifdef PETSC_HAVE_TXPETSCGPU

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIAIJCUSP"
PetscErrorCode MatSetOption_MPIAIJCUSP(Mat A,MatOption op,PetscBool  flg)
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
    a->diagGPUMatFormat = DIAGBLOCK_MAT_DIA;
    break;
  case OFFDIAGBLOCK_MAT_DIA:
    a->offdiagGPUMatFormat = OFFDIAGBLOCK_MAT_DIA;
    break;
  case DIAGBLOCK_MAT_ELL:
    a->diagGPUMatFormat = DIAGBLOCK_MAT_ELL;
    break;
  case OFFDIAGBLOCK_MAT_ELL:
    a->offdiagGPUMatFormat = OFFDIAGBLOCK_MAT_ELL;
    break;
  case DIAGBLOCK_MAT_HYB:
  case OFFDIAGBLOCK_MAT_HYB:
  case MAT_HYB:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported GPU matrix storage format HYB for (MPI,SEQ)AIJCUSP matrix type.");
  default:
    break;
  }
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSetFromOptions_MPIAIJCUSP"
PetscErrorCode MatSetFromOptions_MPIAIJCUSP(Mat A)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode     ierr;
  PetscInt       idxDiag=0,idxOffDiag=0;
  char * formats[]={CSR,DIA,ELL};
  MatOption diagFormat, offdiagFormat;
  PetscBool      flg;
  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"When using TxPETSCGPU, MPIAIJCUSP Options","Mat");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEList("-mat_mult_cusp_diag_storage_format",
			     "Set the storage format of (mpi)aijcusp gpu matrices for SpMV",
			     "None",formats,3,formats[0],&idxDiag,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mat_mult_cusp_offdiag_storage_format",
			     "Set the storage format of (mpi)aijcusp gpu matrices for SpMV",
			     "None",formats,3,formats[0],&idxOffDiag,&flg);CHKERRQ(ierr);

    //printf("MatSetFromOptions_MPIAIJCUSP : %s %s\n",formats[idxDiag],formats[idxOffDiag]);
    if (formats[idxDiag] == CSR)
      diagFormat=MAT_CSR;
    else if (formats[idxDiag] == DIA)
      diagFormat=MAT_DIA;
    else if (formats[idxDiag] == ELL)
      diagFormat=MAT_ELL;

    if (formats[idxOffDiag] == CSR)
      offdiagFormat=MAT_CSR;
    else if (formats[idxOffDiag] == DIA)
      offdiagFormat=MAT_DIA;
    else if (formats[idxOffDiag] == ELL)
      offdiagFormat=MAT_ELL;

    a->diagGPUMatFormat = diagFormat;
    a->offdiagGPUMatFormat = offdiagFormat;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIAIJCUSP"
PetscErrorCode  MatCreate_MPIAIJCUSP(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_MPIAIJ(B);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAIJSetPreallocation_C",
                                     "MatMPIAIJSetPreallocation_MPIAIJCUSP",
                                      MatMPIAIJSetPreallocation_MPIAIJCUSP);CHKERRQ(ierr);
  B->ops->getvecs        = MatGetVecs_MPIAIJCUSP;
  B->ops->setvaluesbatch = MatSetValuesBatch_MPIAIJCUSP;
#ifdef PETSC_HAVE_TXPETSCGPU
  B->ops->mult           = MatMult_MPIAIJCUSP; 
  B->ops->setfromoptions = MatSetFromOptions_MPIAIJCUSP;	
  B->ops->setoption      = MatSetOption_MPIAIJCUSP;	
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "MatCreateAIJCUSP"
PetscErrorCode  MatCreateAIJCUSP(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJCUSP);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJCUSP);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJCUSP - MATAIJCUSP = "aijcusp" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJCUSP when constructed with a single process communicator,
   and MATMPIAIJCUSP otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type mpiaijcusp - sets the matrix type to "mpiaijcusp" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJ,MATSEQAIJ,MATMPIAIJ, MATMPIAIJCUSP, MATSEQAIJCUSP
M*/

