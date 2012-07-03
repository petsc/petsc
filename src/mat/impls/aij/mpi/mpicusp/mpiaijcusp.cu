#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
PETSC_CUDA_EXTERN_C_END
#include "mpicuspmatimpl.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation_MPIAIJCUSP"
PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJCUSP(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ*)B->data;
#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_MPIAIJCUSP * cuspStruct = (Mat_MPIAIJCUSP*)b->spptr;
#endif
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
  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQAIJCUSP matrices. */
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
  ierr=MatSetOption_SeqAIJCUSP(b->A,cuspStruct->diagGPUMatFormat,PETSC_TRUE);CHKERRQ(ierr);
  ierr=MatSetOption_SeqAIJCUSP(b->B,cuspStruct->offdiagGPUMatFormat,PETSC_TRUE);CHKERRQ(ierr);
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

PetscErrorCode MatSetValuesBatch_MPIAIJCUSP(Mat J, PetscInt Ne, PetscInt Nl, PetscInt *elemRows, const PetscScalar *elemMats);

#ifdef PETSC_HAVE_TXPETSCGPU
#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIAIJCUSP"
PetscErrorCode MatSetOption_MPIAIJCUSP(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSP * cuspStruct  = (Mat_MPIAIJCUSP*)a->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatSetOption_MPIAIJ(A,op,flg);CHKERRQ(ierr);
  switch (op) {
  case MAT_DIAGBLOCK_CSR:
    cuspStruct->diagGPUMatFormat = MAT_DIAGBLOCK_CSR;
    break;
  case MAT_OFFDIAGBLOCK_CSR:
    cuspStruct->offdiagGPUMatFormat = MAT_OFFDIAGBLOCK_CSR;
    break;
  case MAT_DIAGBLOCK_DIA:
    cuspStruct->diagGPUMatFormat = MAT_DIAGBLOCK_DIA;
    break;
  case MAT_OFFDIAGBLOCK_DIA:
    cuspStruct->offdiagGPUMatFormat = MAT_OFFDIAGBLOCK_DIA;
    break;
  case MAT_DIAGBLOCK_ELL:
    cuspStruct->diagGPUMatFormat = MAT_DIAGBLOCK_ELL;
    break;
  case MAT_OFFDIAGBLOCK_ELL:
    cuspStruct->offdiagGPUMatFormat = MAT_OFFDIAGBLOCK_ELL;
    break;
  case MAT_DIAGBLOCK_HYB:
  case MAT_OFFDIAGBLOCK_HYB:
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
  Mat_MPIAIJCUSP * cuspStruct  = (Mat_MPIAIJCUSP*)a->spptr;
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

    switch (idxDiag)
      {
      case 0:
	diagFormat=MAT_CSR;
	break;
      case 2:
	diagFormat=MAT_DIA;
	break;
      case 3:
	diagFormat=MAT_ELL;
	break;      
      }
    
    switch (idxOffDiag)
      {
      case 0:
	offdiagFormat=MAT_CSR;
	break;
      case 2:
	offdiagFormat=MAT_DIA;
	break;
      case 3:
	offdiagFormat=MAT_ELL;
	break;      
      }
    cuspStruct->diagGPUMatFormat = diagFormat;
    cuspStruct->offdiagGPUMatFormat = offdiagFormat;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIAIJCUSP"
PetscErrorCode MatDestroy_MPIAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_MPIAIJ *a  = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSP * cuspStruct  = (Mat_MPIAIJCUSP*)a->spptr;
#endif
  PetscFunctionBegin;
#ifdef PETSC_HAVE_TXPETSCGPU
  try {
    delete cuspStruct;
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJCUSP error: %s", ex);
  } 
  cuspStruct = 0;
#endif
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIAIJCUSP"
PetscErrorCode  MatCreate_MPIAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_MPIAIJ *a;
  Mat_MPIAIJCUSP * cuspStruct;
#endif
  PetscFunctionBegin;
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMPIAIJSetPreallocation_C",
                                     "MatMPIAIJSetPreallocation_MPIAIJCUSP",
                                      MatMPIAIJSetPreallocation_MPIAIJCUSP);CHKERRQ(ierr);
  A->ops->getvecs        = MatGetVecs_MPIAIJCUSP;
  A->ops->setvaluesbatch = MatSetValuesBatch_MPIAIJCUSP;

#ifdef PETSC_HAVE_TXPETSCGPU
  a               = (Mat_MPIAIJ*)A->data;
  a->spptr        = new Mat_MPIAIJCUSP;
  cuspStruct  = (Mat_MPIAIJCUSP*)a->spptr;

  cuspStruct->diagGPUMatFormat    = MAT_DIAGBLOCK_CSR;
  cuspStruct->offdiagGPUMatFormat = MAT_OFFDIAGBLOCK_CSR;

  A->ops->mult           = MatMult_MPIAIJCUSP; 
  A->ops->setfromoptions = MatSetFromOptions_MPIAIJCUSP;	
  A->ops->setoption      = MatSetOption_MPIAIJCUSP;	
  A->ops->destroy        = MatDestroy_MPIAIJCUSP;
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSP);CHKERRQ(ierr);
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

