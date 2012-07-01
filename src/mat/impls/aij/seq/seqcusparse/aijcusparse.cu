/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
//#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include "cusparsematimpl.h"

// this is such a hack ... but I don't know of another way to pass this variable
// from one GPU_Matrix_Ifc class to another. This is necessary for the parallel
//  SpMV. Essentially, I need to use the same stream variable in two different
//  data structures. I do this by creating a single instance of that stream
//  and reuse it.
cudaStream_t theBodyStream=0;

EXTERN_C_BEGIN
PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSE(Mat,Mat,const MatFactorInfo *);
PetscErrorCode MatSolve_SeqAIJCUSPARSE(Mat,Vec,Vec);
PetscErrorCode MatSolve_SeqAIJCUSPARSE_NaturalOrdering(Mat,Vec,Vec);
PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(Mat);
PetscErrorCode MatCUSPARSEAnalysisAndCopyToGPU(Mat);
PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat,Vec,Vec);
PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);
PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat,Vec,Vec);
PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat,Vec,Vec,Vec);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_cusparse"
PetscErrorCode MatFactorGetSolverPackage_seqaij_cusparse(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCUSPARSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode MatGetFactor_seqaij_petsc(Mat,MatFactorType,Mat*);
EXTERN_C_END



/*MC
  MATSOLVERCUSPARSE = "cusparse" - A matrix type providing triangular solvers (ILU) for seq matrices 
  on the GPU of type, seqaijcusparse, aijcusparse, or seqaijcusp, aijcusp
5~
  ./configure --download-txpetscgpu to install PETSc to use CUSPARSE solves

  Options Database Keys:
+ -mat_mult_cusparse_storage_format <CSR>         - (choose one of) CSR, ELL, HYB
+ -mat_solve_cusparse_storage_format <CSR>        - (choose one of) CSR, ELL, HYB

   Level: beginner
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_cusparse"
PetscErrorCode MatGetFactor_seqaij_cusparse(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatGetFactor_seqaij_petsc(A,ftype,B);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT){
    ierr = MatSetType(*B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatSetFromOptions_SeqAIJCUSPARSE(*B);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)(*B),"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqaij_cusparse",MatFactorGetSolverPackage_seqaij_cusparse);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSPARSE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSPARSE Matrix Types");
  (*B)->factortype = ftype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatAIJCUSPARSESetGPUStorageFormatForMatSolve"
PetscErrorCode MatAIJCUSPARSESetGPUStorageFormatForMatSolve(Mat A,GPUStorageFormat format)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc* cusparseMatLo  = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc* cusparseMatUp  = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  PetscFunctionBegin;
  cusparseTriFactors->format = format;  
  if (cusparseMatLo && cusparseMatUp) {
    // If data exists already delete
    if (cusparseMatLo)
      delete (GPU_Matrix_Ifc *)cusparseMatLo;
    if (cusparseMatUp)
      delete (GPU_Matrix_Ifc *)cusparseMatUp;	 

    // key data was deleted so reset the flag.
    A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;

    // rebuild matrix
    ierr=MatCUSPARSEAnalysisAndCopyToGPU(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

//EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqAIJCUSPARSE"
PetscErrorCode MatSetOption_SeqAIJCUSPARSE(Mat A,MatOption op,PetscBool  flg)
{
  Mat_SeqAIJCUSPARSE *cusparseMat  = (Mat_SeqAIJCUSPARSE*)A->spptr; 
  PetscErrorCode ierr;
  PetscFunctionBegin;  
  ierr = MatSetOption_SeqAIJ(A,op,flg);CHKERRQ(ierr);
  switch (op) {
  case MAT_DIAGBLOCK_CSR:
  case MAT_OFFDIAGBLOCK_CSR:
  case MAT_CSR:
    cusparseMat->format = CSR;    
    break;
  case MAT_DIAGBLOCK_DIA:
  case MAT_OFFDIAGBLOCK_DIA:
  case MAT_DIA:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported GPU matrix storage format DIA for (MPI,SEQ)AIJCUSPARSE matrix type.");
  case MAT_DIAGBLOCK_ELL:
  case MAT_OFFDIAGBLOCK_ELL:
  case MAT_ELL:
    cusparseMat->format = ELL;    
    break;
  case MAT_DIAGBLOCK_HYB:
  case MAT_OFFDIAGBLOCK_HYB:
  case MAT_HYB:
    cusparseMat->format = HYB;    
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
//EXTERN_C_END

//EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSetFromOptions_SeqAIJCUSPARSE"
PetscErrorCode MatSetFromOptions_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  PetscInt       idx=0;
  char * formats[] ={CSR,ELL,HYB};
  MatOption format;
  PetscBool      flg;
  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"When using TxPETSCGPU, AIJCUSPARSE Options","Mat");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEList("-mat_mult_cusparse_storage_format",
			     "Set the storage format of (seq)aijcusparse gpu matrices for SpMV",
			     "None",formats,3,formats[0],&idx,&flg);CHKERRQ(ierr);
    switch (idx)
      {	
      case 0:
	format=MAT_CSR;
	break;
      case 2:
	format=MAT_ELL;
	break;
      case 3:
	format=MAT_HYB;
	break;      
      }
    ierr=MatSetOption_SeqAIJCUSPARSE(A,format,PETSC_TRUE);CHKERRQ(ierr);
  }
  else { 
    ierr = PetscOptionsEList("-mat_solve_cusparse_storage_format",
			     "Set the storage format of (seq)aijcusparse gpu matrices for TriSolve",
			     "None",formats,3,formats[0],&idx,&flg);CHKERRQ(ierr);
    ierr=MatAIJCUSPARSESetGPUStorageFormatForMatSolve(A,formats[idx]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
//EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_SeqAIJCUSPARSE"
PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJCUSPARSE"
PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCUSPARSEBuildLowerTriMatrix"
PetscErrorCode MatCUSPARSEBuildLowerTriMatrix(Mat A)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscInt          n = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc* cusparseMat  = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  cusparseStatus_t stat;
  const PetscInt    *ai = a->i,*aj = a->j,*vi;
  const MatScalar   *aa = a->a,*v;
  PetscErrorCode     ierr;
  PetscInt *AiLo, *AjLo;
  PetscScalar *AALo;
  PetscInt i,nz, nzLower, offset, rowOffset;
  
  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){	
    try {	
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower=n+ai[n]-ai[1];
      
      /* Allocate Space for the lower triangular matrix */	
      ierr = cudaMallocHost((void **) &AiLo, (n+1)*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void **) &AjLo, nzLower*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void **) &AALo, nzLower*sizeof(PetscScalar));CHKERRCUSP(ierr);
      
      /* Fill the lower triangular matrix */
      AiLo[0]=(PetscInt) 0;
      AiLo[n]=nzLower;
      AjLo[0]=(PetscInt) 0;
      AALo[0]=(MatScalar) 1.0;
      v    = aa;
      vi   = aj;
      offset=1;
      rowOffset=1;
      for (i=1; i<n; i++) {
	nz  = ai[i+1] - ai[i];
	// additional 1 for the term on the diagonal
	AiLo[i]=rowOffset;
	rowOffset+=nz+1;

	ierr = PetscMemcpy(&(AjLo[offset]), vi, nz*sizeof(PetscInt));CHKERRQ(ierr);
	ierr = PetscMemcpy(&(AALo[offset]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);
	
	offset+=nz;
	AjLo[offset]=(PetscInt) i;
	AALo[offset]=(MatScalar) 1.0;
	offset+=1;
	
	v  += nz;
	vi += nz;
      }    
      cusparseMat = GPU_Matrix_Factory::getNew(cusparseTriFactors->format);
      stat = cusparseMat->initializeCusparse(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_FILL_MODE_LOWER);CHKERRCUSP(stat);
      ierr = cusparseMat->setMatrix(n, n, nzLower, AiLo, AjLo, AALo);CHKERRCUSP(ierr);
      stat = cusparseMat->solveAnalysis();CHKERRCUSP(stat);
      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->loTriFactorPtr = cusparseMat;
      // free temporary data
      ierr = cudaFreeHost(AiLo);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AjLo);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AALo);CHKERRCUSP(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);	  
}

#undef __FUNCT__  
#define __FUNCT__ "MatCUSPARSEBuildUpperTriMatrix"
PetscErrorCode MatCUSPARSEBuildUpperTriMatrix(Mat A)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscInt          n = A->rmap->n;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc* cusparseMat  = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t stat;
  const PetscInt    *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar   *aa = a->a,*v;
  PetscInt *AiUp, *AjUp;
  PetscScalar *AAUp;
  PetscInt i,nz, nzUpper, offset;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;

  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){	
    try {	
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0]-adiag[n];
      
      /* Allocate Space for the upper triangular matrix */
      ierr = cudaMallocHost((void **) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void **) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUSP(ierr);
      ierr = cudaMallocHost((void **) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUSP(ierr);
      
      /* Fill the upper triangular matrix */
      AiUp[0]=(PetscInt) 0;
      AiUp[n]=nzUpper;
      offset = nzUpper;
      for (i=n-1; i>=0; i--){
	v   = aa + adiag[i+1] + 1;
	vi  = aj + adiag[i+1] + 1;
	
	// number of elements NOT on the diagonal
	nz = adiag[i] - adiag[i+1]-1;
	
	// decrement the offset
	offset -= (nz+1);
	
	// first, set the diagonal elements
	AjUp[offset] = (PetscInt) i;
	AAUp[offset] = 1./v[nz];
	AiUp[i] = AiUp[i+1] - (nz+1);
	
	ierr = PetscMemcpy(&(AjUp[offset+1]), vi, nz*sizeof(PetscInt));CHKERRQ(ierr);
	ierr = PetscMemcpy(&(AAUp[offset+1]), v, nz*sizeof(PetscScalar));CHKERRQ(ierr);
      }      
      cusparseMat = GPU_Matrix_Factory::getNew(cusparseTriFactors->format);
      stat = cusparseMat->initializeCusparse(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_FILL_MODE_UPPER);CHKERRCUSP(stat);
      ierr = cusparseMat->setMatrix(n, n, nzUpper, AiUp, AjUp, AAUp);CHKERRCUSP(ierr);
      stat = cusparseMat->solveAnalysis();CHKERRCUSP(stat);
      ((Mat_SeqAIJCUSPARSETriFactors*)A->spptr)->upTriFactorPtr = cusparseMat;
      // free temporary data
      ierr = cudaFreeHost(AiUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AjUp);CHKERRCUSP(ierr);
      ierr = cudaFreeHost(AAUp);CHKERRCUSP(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);	  
}

#undef __FUNCT__  
#define __FUNCT__ "MatCUSPARSEAnalysiAndCopyToGPU"
PetscErrorCode MatCUSPARSEAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ *a=(Mat_SeqAIJ *)A->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  IS               isrow = a->row,iscol = a->icol;
  PetscBool        row_identity,col_identity;
  const PetscInt   *r,*c;
  PetscInt          n = A->rmap->n;

  PetscFunctionBegin;       
  ierr = MatCUSPARSEBuildLowerTriMatrix(A);CHKERRQ(ierr);
  ierr = MatCUSPARSEBuildUpperTriMatrix(A);CHKERRQ(ierr);
  cusparseTriFactors->tempvec = new CUSPARRAY;
  cusparseTriFactors->tempvec->resize(n);

  A->valid_GPU_matrix = PETSC_CUSP_BOTH;
  //lower triangular indices
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  if (!row_identity)     
    ierr = cusparseTriFactors->loTriFactorPtr->setOrdIndices(r, n);CHKERRCUSP(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  //upper triangular indices
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!col_identity)
    ierr = cusparseTriFactors->upTriFactorPtr->setOrdIndices(c, n);CHKERRCUSP(ierr);
  ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);	  
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJCUSPARSE"
PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *b=(Mat_SeqAIJ *)B->data;
  IS               isrow = b->row,iscol = b->col;
  PetscBool        row_identity,col_identity;

  PetscFunctionBegin;
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  // determine which version of MatSolve needs to be used.
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) B->ops->solve = MatSolve_SeqAIJCUSPARSE_NaturalOrdering;    
  else                              B->ops->solve = MatSolve_SeqAIJCUSPARSE; 

  //if (!row_identity) printf("Row permutations exist!");
  //if (!col_identity) printf("Col permutations exist!");

  // get the triangular factors
  ierr = MatCUSPARSEAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJCUSPARSE"
PetscErrorCode MatSolve_SeqAIJCUSPARSE(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  CUSPARRAY      *xGPU, *bGPU;
  cusparseStatus_t stat;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc *cusparseMatLo  = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc *cusparseMatUp  = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  CUSPARRAY * tempGPU = (CUSPARRAY*) cusparseTriFactors->tempvec;

  PetscFunctionBegin;
  // Get the GPU pointers
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  // solve with reordering
  ierr = cusparseMatLo->reorderIn(xGPU, bGPU);CHKERRCUSP(ierr);
  stat = cusparseMatLo->solve(xGPU, tempGPU);CHKERRCUSP(stat);
  stat = cusparseMatUp->solve(tempGPU, xGPU);CHKERRCUSP(stat);
  ierr = cusparseMatUp->reorderOut(xGPU);CHKERRCUSP(ierr);
  	  	  
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJCUSPARSE_NaturalOrdering"
PetscErrorCode MatSolve_SeqAIJCUSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  CUSPARRAY         *xGPU, *bGPU;
  cusparseStatus_t stat;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  GPU_Matrix_Ifc *cusparseMatLo  = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
  GPU_Matrix_Ifc *cusparseMatUp  = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
  CUSPARRAY * tempGPU = (CUSPARRAY*) cusparseTriFactors->tempvec;

  PetscFunctionBegin;
  // Get the GPU pointers
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

  // solve
  stat = cusparseMatLo->solve(bGPU, tempGPU);CHKERRCUSP(stat);
  stat = cusparseMatUp->solve(tempGPU, xGPU);CHKERRCUSP(stat);

  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCUSPARSECopyToGPU"
PetscErrorCode MatCUSPARSECopyToGPU(Mat A)
{

  Mat_SeqAIJCUSPARSE *cusparseMat  = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJ      *a          = (Mat_SeqAIJ*)A->data;
  PetscInt        m           = A->rmap->n,*ii,*ridx;
  PetscErrorCode  ierr;


  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){    
    ierr = PetscLogEventBegin(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
    /*
      It may be possible to reuse nonzero structure with new matrix values but 
      for simplicity and insured correctness we delete and build a new matrix on
      the GPU. Likely a very small performance hit.
    */
    if (cusparseMat->mat){
      try {
	delete cusparseMat->mat;
	if (cusparseMat->tempvec)
	  delete cusparseMat->tempvec;
	
      } catch(char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
      } 
    }
    try {
      cusparseMat->nonzerorow=0;
      for (int j = 0; j<m; j++)
	cusparseMat->nonzerorow += ((a->i[j+1]-a->i[j])>0);

      if (a->compressedrow.use) {	
	m    = a->compressedrow.nrows;
	ii   = a->compressedrow.i;
	ridx = a->compressedrow.rindex;
      } else {
	// Forcing compressed row on the GPU ... only relevant for CSR storage
	int k=0;
	ierr = PetscMalloc((cusparseMat->nonzerorow+1)*sizeof(PetscInt), &ii);CHKERRQ(ierr);
	ierr = PetscMalloc((cusparseMat->nonzerorow)*sizeof(PetscInt), &ridx);CHKERRQ(ierr);
	ii[0]=0;
	for (int j = 0; j<m; j++) {
	  if ((a->i[j+1]-a->i[j])>0) {
	    ii[k] = a->i[j];
	    ridx[k]= j;
	    k++;
	  }
	}
	ii[cusparseMat->nonzerorow] = a->nz;
	m = cusparseMat->nonzerorow;
      }

      // Build our matrix ... first determine the GPU storage type
      cusparseMat->mat = GPU_Matrix_Factory::getNew(cusparseMat->format);

      // Create the streams and events (if desired). 
      PetscMPIInt    size;
      ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
      ierr = cusparseMat->mat->buildStreamsAndEvents(size, &theBodyStream);CHKERRCUSP(ierr);	

      // FILL MODE UPPER is irrelevant
      cusparseStatus_t stat = cusparseMat->mat->initializeCusparse(MAT_cusparseHandle, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER);CHKERRCUSP(stat);
      
      // lastly, build the matrix
      ierr = cusparseMat->mat->setMatrix(m, A->cmap->n, a->nz, ii, a->j, a->a);CHKERRCUSP(ierr);
      cusparseMat->mat->setCPRowIndices(ridx, m);
	//      }
      if (!a->compressedrow.use) {	
	// free data
	ierr = PetscFree(ii);CHKERRQ(ierr);
	ierr = PetscFree(ridx);CHKERRQ(ierr);
      }
      cusparseMat->tempvec = new CUSPARRAY;
      cusparseMat->tempvec->resize(m);
      //A->spptr = cusparseMat;     
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    } 
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
    ierr = PetscLogEventEnd(MAT_CUSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// #undef __FUNCT__
// #define __FUNCT__ "MatCUSPARSECopyFromGPU"
// PetscErrorCode MatCUSPARSECopyFromGPU(Mat A, CUSPMATRIX *Agpu)
// {
//   Mat_SeqAIJCUSPARSE *cuspstruct = (Mat_SeqAIJCUSPARSE *) A->spptr;
//   Mat_SeqAIJ     *a          = (Mat_SeqAIJ *) A->data;
//   PetscInt        m          = A->rmap->n;
//   PetscErrorCode  ierr;

//   PetscFunctionBegin;
//   if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
//     if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
//       try {
//         cuspstruct->mat = Agpu;
//         if (a->compressedrow.use) {
//           //PetscInt *ii, *ridx;
//           SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Cannot handle row compression for GPU matrices");
//         } else {
//           PetscInt i;

//           if (m+1 != (PetscInt) cuspstruct->mat->row_offsets.size()) {SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "GPU matrix has %d rows, should be %d", cuspstruct->mat->row_offsets.size()-1, m);}
//           a->nz    = cuspstruct->mat->values.size();
//           a->maxnz = a->nz; /* Since we allocate exactly the right amount */
//           A->preallocated = PETSC_TRUE;
//           // Copy ai, aj, aa
//           if (a->singlemalloc) {
//             if (a->a) {ierr = PetscFree3(a->a,a->j,a->i);CHKERRQ(ierr);}
//           } else {
//             if (a->i) {ierr = PetscFree(a->i);CHKERRQ(ierr);}
//             if (a->j) {ierr = PetscFree(a->j);CHKERRQ(ierr);}
//             if (a->a) {ierr = PetscFree(a->a);CHKERRQ(ierr);}
//           }
//           ierr = PetscMalloc3(a->nz,PetscScalar,&a->a,a->nz,PetscInt,&a->j,m+1,PetscInt,&a->i);CHKERRQ(ierr);
//           ierr = PetscLogObjectMemory(A, a->nz*(sizeof(PetscScalar)+sizeof(PetscInt))+(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
//           a->singlemalloc = PETSC_TRUE;
//           thrust::copy(cuspstruct->mat->row_offsets.begin(), cuspstruct->mat->row_offsets.end(), a->i);
//           thrust::copy(cuspstruct->mat->column_indices.begin(), cuspstruct->mat->column_indices.end(), a->j);
//           thrust::copy(cuspstruct->mat->values.begin(), cuspstruct->mat->values.end(), a->a);
//           // Setup row lengths
//           if (a->imax) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
//           ierr = PetscMalloc2(m,PetscInt,&a->imax,m,PetscInt,&a->ilen);CHKERRQ(ierr);
//           ierr = PetscLogObjectMemory(A, 2*m*sizeof(PetscInt));CHKERRQ(ierr);
//           for(i = 0; i < m; ++i) {
//             a->imax[i] = a->ilen[i] = a->i[i+1] - a->i[i];
//           }
//           // a->diag?
//         }
//         cuspstruct->tempvec = new CUSPARRAY;
//         cuspstruct->tempvec->resize(m);
//       } catch(char *ex) {
//         SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "CUSP error: %s", ex);
//       }
//     }
//     // This assembly prevents resetting the flag to PETSC_CUSP_CPU and recopying
//     ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     A->valid_GPU_matrix = PETSC_CUSP_BOTH;
//   } else {
//     SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only valid for unallocated GPU matrices");
//   }
//   PetscFunctionReturn(0);
// }

#undef __FUNCT__
#define __FUNCT__ "MatGetVecs_SeqAIJCUSPARSE"
PetscErrorCode MatGetVecs_SeqAIJCUSPARSE(Mat mat, Vec *right, Vec *left)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (right) {
    ierr = VecCreate(((PetscObject)mat)->comm,right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecCreate(((PetscObject)mat)->comm,left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECSEQCUSP);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJCUSPARSE"
PetscErrorCode MatMult_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE *)A->spptr;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  // The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
  // ierr = MatCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
    ierr = cusparseMat->mat->multiply(xarray, yarray);CHKERRCUSP(ierr);
  } catch (char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (!cusparseMat->mat->hasNonZeroStream()) {
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  }
  ierr = PetscLogFlops(2.0*a->nz - cusparseMat->nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqAIJCUSPARSE"
PetscErrorCode MatMultTranspose_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE *)A->spptr;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  // The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
  // ierr = MatCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
#if !defined(PETSC_USE_COMPLEX)
    ierr = cusparseMat->mat->multiply(xarray, yarray, TRANSPOSE);CHKERRCUSP(ierr);
#else
    ierr = cusparseMat->mat->multiply(xarray, yarray, HERMITIAN);CHKERRCUSP(ierr);
#endif
  } catch (char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (!cusparseMat->mat->hasNonZeroStream()) {
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  }
  ierr = PetscLogFlops(2.0*a->nz - cusparseMat->nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJCUSPARSE"
PetscErrorCode MatMultAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE *)A->spptr;
  CUSPARRAY      *xarray,*yarray,*zarray;
  PetscFunctionBegin;
  // The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
  // ierr = MatCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  try {      
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);

    // multiply add
    ierr = cusparseMat->mat->multiplyAdd(xarray, zarray);CHKERRCUSP(ierr);

    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
    
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJCUSPARSE"
PetscErrorCode MatMultTransposeAdd_SeqAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  Mat_SeqAIJCUSPARSE *cusparseMat = (Mat_SeqAIJCUSPARSE *)A->spptr;
  CUSPARRAY      *xarray,*yarray,*zarray;
  PetscFunctionBegin;
  // The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSPARSE
  // ierr = MatCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  try {      
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);

    // multiply add with matrix transpose
#if !defined(PETSC_USE_COMPLEX)
    ierr = cusparseMat->mat->multiplyAdd(xarray, yarray, TRANSPOSE);CHKERRCUSP(ierr);
#else
    ierr = cusparseMat->mat->multiplyAdd(xarray, yarray, HERMITIAN);CHKERRCUSP(ierr);
#endif

    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
    
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUSPARSE"
PetscErrorCode MatAssemblyEnd_SeqAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode  ierr;  
  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  ierr = MatCUSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  // this is not necessary since MatCUSPARSECopyToGPU has been called.
  //if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
  //  A->valid_GPU_matrix = PETSC_CUSP_CPU;
  //}
  A->ops->mult             = MatMult_SeqAIJCUSPARSE;
  A->ops->multadd          = MatMultAdd_SeqAIJCUSPARSE;
  A->ops->multtranspose    = MatMultTranspose_SeqAIJCUSPARSE;
  A->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJCUSPARSE;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJCUSPARSE"
/*@C
   MatCreateSeqAIJCUSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ()

@*/
PetscErrorCode  MatCreateSeqAIJCUSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSPARSE"
PetscErrorCode MatDestroy_SeqAIJCUSPARSE(Mat A)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJCUSPARSE      *cusparseMat = (Mat_SeqAIJCUSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    // The regular matrices
    try {
      if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
	delete (GPU_Matrix_Ifc *)(cusparseMat->mat);
      }
      if (cusparseMat->tempvec!=0)
	delete cusparseMat->tempvec;
      delete cusparseMat;
      A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    } 
  } else {
    // The triangular factors
    try {
      Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors  = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
      GPU_Matrix_Ifc *cusparseMatLo  = (GPU_Matrix_Ifc*)cusparseTriFactors->loTriFactorPtr;
      GPU_Matrix_Ifc *cusparseMatUp  = (GPU_Matrix_Ifc*)cusparseTriFactors->upTriFactorPtr;
      // destroy the matrix and the container
      delete (GPU_Matrix_Ifc *)cusparseMatLo;
      delete (GPU_Matrix_Ifc *)cusparseMatUp;	  
      delete (CUSPARRAY*) cusparseTriFactors->tempvec;
      delete cusparseTriFactors;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    } 
  }
  if (MAT_cusparseHandle) {
    cusparseStatus_t stat;
    stat = cusparseDestroy(MAT_cusparseHandle);CHKERRCUSP(stat);  
    MAT_cusparseHandle=0;
  }
  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;

  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJCUSPARSE"
PetscErrorCode  MatCreate_SeqAIJCUSPARSE(Mat B)
{
  PetscErrorCode ierr;
  GPUStorageFormat format = CSR;
    
  PetscFunctionBegin;
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  if (B->factortype==MAT_FACTOR_NONE) { 
    /* you cannot check the inode.use flag here since the matrix was just created.*/
    // now build a GPU matrix data structure
    B->spptr        = new Mat_SeqAIJCUSPARSE;
    ((Mat_SeqAIJCUSPARSE *)B->spptr)->mat = 0;
    ((Mat_SeqAIJCUSPARSE *)B->spptr)->tempvec = 0;
    ((Mat_SeqAIJCUSPARSE *)B->spptr)->format = format;
  } else {
    /* NEXT, set the pointers to the triangular factors */
    B->spptr        = new Mat_SeqAIJCUSPARSETriFactors;
    ((Mat_SeqAIJCUSPARSETriFactors *)B->spptr)->loTriFactorPtr = 0;
    ((Mat_SeqAIJCUSPARSETriFactors *)B->spptr)->upTriFactorPtr = 0;
    ((Mat_SeqAIJCUSPARSETriFactors *)B->spptr)->tempvec = 0;
    ((Mat_SeqAIJCUSPARSETriFactors *)B->spptr)->format = format;
  }
  // Create a single instance of the MAT_cusparseHandle for any matrix (matMult, TriSolve, ...)
  if (!MAT_cusparseHandle) {
    cusparseStatus_t stat;  
    stat = cusparseCreate(&MAT_cusparseHandle);CHKERRCUSP(stat);  
  }
  // Here we overload MatGetFactor_petsc_C which enables -mat_type aijcusparse to use the 
  // default cusparse tri solve. Note the difference with the implementation in 
  // MatCreate_SeqAIJCUSP in ../seqcusp/aijcusp.cu
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_petsc_C","MatGetFactor_seqaij_cusparse",MatGetFactor_seqaij_cusparse);CHKERRQ(ierr);
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJCUSPARSE;
  B->ops->destroy          = MatDestroy_SeqAIJCUSPARSE;
  B->ops->getvecs          = MatGetVecs_SeqAIJCUSPARSE;
  B->ops->setfromoptions   = MatSetFromOptions_SeqAIJCUSPARSE;
  B->ops->setoption        = MatSetOption_SeqAIJCUSPARSE;
  B->ops->mult             = MatMult_SeqAIJCUSPARSE;
  B->ops->multadd          = MatMultAdd_SeqAIJCUSPARSE;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJCUSPARSE;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJCUSPARSE;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  PetscFunctionReturn(0);
}
EXTERN_C_END

