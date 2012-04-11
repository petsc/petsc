

/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "petsc-private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include "../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h"


#ifdef PETSC_HAVE_TXPETSCGPU
// this is such a hack ... but I don't know of another way to pass this variable
// from one GPU_Matrix_Ifc class to another. This is necessary for the parallel
//  SpMV. Essentially, I need to use the same stream variable in two different
//  data structures. I do this by creating a single instance of that stream
//  and reuse it.
cudaStream_t theBodyStream=0;

struct Mat_SeqAIJGPU {
  GPU_Matrix_Ifc*   mat; /* pointer to the matrix on the GPU */
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt          nonzerorow; /* number of nonzero rows ... used in the flop calculations */
};
#endif // PETSC_HAVE_TXPETSCGPU

#include <algorithm>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <cusparse.h>

// Hanlde to the CUSPARSE library. Only need one
cusparseHandle_t Mat_CUSPARSE_Handle;

struct Mat_SeqAIJTriFactors {
  void *loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  void *upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
};

struct Mat_SeqAIJCUSPARSETriFactor {
  CUSPMATRIX*                 mat;     /* pointer to the matrix on the GPU */
  cusparseMatDescr_t          Mat_CUSPARSE_description;
  cusparseSolveAnalysisInfo_t Mat_CUSPARSE_solveAnalysisInfo; 
  CUSPARRAY*                  tempvecGPU; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  CUSPINTARRAYGPU*            ordIndicesGPU; /* For Lower triangular, this is the row permutation. For Upper triangular, this the column permutation */
};


EXTERN_C_BEGIN
PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSP(Mat,Mat,IS,IS,const MatFactorInfo*);
PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSP(Mat,Mat,IS,IS,const MatFactorInfo*);
PetscErrorCode MatLUFactorNumeric_SeqAIJCUSP(Mat,Mat,const MatFactorInfo *);
PetscErrorCode MatSolve_SeqAIJCUSP(Mat,Vec,Vec);
PetscErrorCode MatSolve_SeqAIJCUSP_NaturalOrdering(Mat,Vec,Vec);
EXTERN_C_END


EXTERN_C_BEGIN
extern PetscErrorCode MatGetFactor_seqaij_petsc(Mat,MatFactorType,Mat*);
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_petsccusp"
PetscErrorCode MatGetFactor_seqaij_petsccusp(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatGetFactor_seqaij_petsc(A,ftype,B);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT){
    ierr = MatSetType(*B,MATSEQAIJCUSP);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJCUSP;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSP;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSP Matrix Types");
  (*B)->factortype = ftype;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_SeqAIJCUSP"
PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSP(Mat fact,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatILUFactorSymbolic_SeqAIJ(fact,A,isrow,iscol,info);CHKERRQ(ierr);
  (fact)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJCUSP"
PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSP(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSP;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCUSPARSEInitialize"
PetscErrorCode MatCUSPARSEInitialize(Mat A)
{
  Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactor* cusparsestructLo  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactor* cusparsestructUp  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->upTriFactorPtr;
  cusparseStatus_t stat;

  PetscFunctionBegin;
  try {	
    // cusparse handle creation
    stat = cusparseCreate(&Mat_CUSPARSE_Handle);CHKERRCUSP(stat);

    //  LOWER TRIANGULAR MATRIX
    stat = cusparseCreateMatDescr(&(cusparsestructLo->Mat_CUSPARSE_description));CHKERRCUSP(stat);
    stat = cusparseSetMatType(cusparsestructLo->Mat_CUSPARSE_description, CUSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRCUSP(stat);
    stat = cusparseSetMatFillMode(cusparsestructLo->Mat_CUSPARSE_description, CUSPARSE_FILL_MODE_LOWER);CHKERRCUSP(stat);    
    // cusparse set matrix diag type ... this doesn't seem to work right now??
    //stat = cusparseSetMatDiagType(cusparsestructLo->Mat_CUSPARSE_description, CUSPARSE_DIAG_TYPE_UNIT);CHKERRCUSP(stat);
    stat = cusparseSetMatIndexBase(cusparsestructLo->Mat_CUSPARSE_description, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSP(stat);
    stat = cusparseCreateSolveAnalysisInfo(&(cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo));CHKERRCUSP(stat);
  
    //  UPPER TRIANGULAR MATRIX    
    stat = cusparseCreateMatDescr(&(cusparsestructUp->Mat_CUSPARSE_description));CHKERRCUSP(stat);
    stat = cusparseSetMatType(cusparsestructUp->Mat_CUSPARSE_description, CUSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRCUSP(stat);
    stat = cusparseSetMatFillMode(cusparsestructUp->Mat_CUSPARSE_description, CUSPARSE_FILL_MODE_UPPER);CHKERRCUSP(stat);
    stat = cusparseSetMatDiagType(cusparsestructUp->Mat_CUSPARSE_description, CUSPARSE_DIAG_TYPE_NON_UNIT);CHKERRCUSP(stat);
    stat = cusparseSetMatIndexBase(cusparsestructUp->Mat_CUSPARSE_description, CUSPARSE_INDEX_BASE_ZERO);CHKERRCUSP(stat);
    stat = cusparseCreateSolveAnalysisInfo(&(cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo));CHKERRCUSP(stat);  
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
  }
  PetscFunctionReturn(0);	  
}

#undef __FUNCT__  
#define __FUNCT__ "MatCUSPARSEBuildLowerTriMatrix"
PetscErrorCode MatCUSPARSEBuildLowerTriMatrix(Mat A)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscInt          n = A->rmap->n;
  Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactor* cusparsestructLo  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->loTriFactorPtr;
  const PetscInt    *ai = a->i,*aj = a->j,*vi;
  const MatScalar   *aa = a->a,*v;
  cusparseStatus_t stat;
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
	ierr = PetscMemcpy(&(AALo[offset]), v, nz*sizeof(MatScalar));CHKERRQ(ierr);
	
	offset+=nz;
	AjLo[offset]=(PetscInt) i;
	AALo[offset]=(MatScalar) 1.0;
	offset+=1;
	
	v  += nz;
	vi += nz;
      }    
      /* The Lower triangular matrix */
      cusparsestructLo->mat = new CUSPMATRIX;
      cusparsestructLo->mat->resize(n,n,nzLower);
      cusparsestructLo->mat->row_offsets.assign(AiLo,AiLo+n+1);
      cusparsestructLo->mat->column_indices.assign(AjLo,AjLo+nzLower);
      cusparsestructLo->mat->values.assign(AALo,AALo+nzLower);
      
      // work vector allocation
      cusparsestructLo->tempvecGPU = new CUSPARRAY;
      (cusparsestructLo->tempvecGPU)->resize(n);
      
      // cusparse analysis 
#if defined(PETSC_USE_REAL_SINGLE)  
      stat = cusparseScsrsv_analysis(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     n, cusparsestructLo->Mat_CUSPARSE_description,
				     thrust::raw_pointer_cast(&(cusparsestructLo->mat)->values[0]),
				     thrust::raw_pointer_cast(&(cusparsestructLo->mat)->row_offsets[0]),
				     thrust::raw_pointer_cast(&(cusparsestructLo->mat)->column_indices[0]),
				     cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo);CHKERRCUSP(stat);
#elif defined(PETSC_USE_REAL_DOUBLE)
      stat = cusparseDcsrsv_analysis(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     n, cusparsestructLo->Mat_CUSPARSE_description,
				     thrust::raw_pointer_cast(&(cusparsestructLo->mat)->values[0]),
				     thrust::raw_pointer_cast(&(cusparsestructLo->mat)->row_offsets[0]),
				     thrust::raw_pointer_cast(&(cusparsestructLo->mat)->column_indices[0]),
				     cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo);CHKERRCUSP(stat);
#endif      
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
  Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactor* cusparsestructUp  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->upTriFactorPtr;
  const PetscInt    *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar   *aa = a->a,*v;
  cusparseStatus_t stat;
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
	ierr = PetscMemcpy(&(AAUp[offset+1]), v, nz*sizeof(MatScalar));CHKERRQ(ierr);
      }      
      /* The Upper triangular matrix */
      cusparsestructUp->mat = new CUSPMATRIX;
      cusparsestructUp->mat->resize(n,n,nzUpper);
      cusparsestructUp->mat->row_offsets.assign(AiUp,AiUp+n+1);
      cusparsestructUp->mat->column_indices.assign(AjUp,AjUp+nzUpper);
      cusparsestructUp->mat->values.assign(AAUp,AAUp+nzUpper);
      
      // work vector allocation
      cusparsestructUp->tempvecGPU = new CUSPARRAY;
      (cusparsestructUp->tempvecGPU)->resize(n);
    
      // cusparse analysis 
#if defined(PETSC_USE_REAL_SINGLE)  
      stat = cusparseScsrsv_analysis(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     n, cusparsestructUp->Mat_CUSPARSE_description,
				     thrust::raw_pointer_cast(&(cusparsestructUp->mat)->values[0]),
				     thrust::raw_pointer_cast(&(cusparsestructUp->mat)->row_offsets[0]),
				     thrust::raw_pointer_cast(&(cusparsestructUp->mat)->column_indices[0]),
				     cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo);CHKERRCUSP(stat);
#elif defined(PETSC_USE_REAL_DOUBLE)
      stat = cusparseDcsrsv_analysis(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     n, cusparsestructUp->Mat_CUSPARSE_description,
				     thrust::raw_pointer_cast(&(cusparsestructUp->mat)->values[0]),
				     thrust::raw_pointer_cast(&(cusparsestructUp->mat)->row_offsets[0]),
				     thrust::raw_pointer_cast(&(cusparsestructUp->mat)->column_indices[0]),
				     cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo);CHKERRCUSP(stat);
#endif
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
  Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactor* cusparsestructLo  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactor* cusparsestructUp  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->upTriFactorPtr;
  IS               isrow = a->row,iscol = a->icol;
  PetscBool        row_identity,col_identity;
  const PetscInt   *r,*c;
  PetscFunctionBegin;      
        
  ierr = MatCUSPARSEInitialize(A);CHKERRQ(ierr);
  ierr = MatCUSPARSEBuildLowerTriMatrix(A);CHKERRQ(ierr);
  ierr = MatCUSPARSEBuildUpperTriMatrix(A);CHKERRQ(ierr);
  A->valid_GPU_matrix = PETSC_CUSP_BOTH;

  //lower triangular indices
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  cusparsestructLo->ordIndicesGPU = new CUSPINTARRAYGPU;
  (cusparsestructLo->ordIndicesGPU)->assign(&r[0], &r[0]+A->rmap->n);

  //upper triangular indices
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  cusparsestructUp->ordIndicesGPU = new CUSPINTARRAYGPU;
  (cusparsestructUp->ordIndicesGPU)->assign(&c[0], &c[0]+A->rmap->n);

  PetscFunctionReturn(0);	  
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJCUSP"
PetscErrorCode MatLUFactorNumeric_SeqAIJCUSP(Mat B,Mat A,const MatFactorInfo *info)
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
  if (row_identity && col_identity) B->ops->solve = MatSolve_SeqAIJCUSP_NaturalOrdering;    
  else                              B->ops->solve = MatSolve_SeqAIJCUSP; 

  // get the triangular factors
  ierr = MatCUSPARSEAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJCUSP"
PetscErrorCode MatSolve_SeqAIJCUSP(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  CUSPARRAY      *xGPU, *bGPU;
  PetscInt          n = A->rmap->n;
  cusparseStatus_t stat;
  Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactor *cusparsestructLo  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactor *cusparsestructUp  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->upTriFactorPtr;

  PetscFunctionBegin;

  // Get the GPU pointers
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);
  
  // Copy the right hand side vector, bGPU, into xGPU with the row permutation
  thrust::copy(thrust::make_permutation_iterator(bGPU->begin(), (cusparsestructLo->ordIndicesGPU)->begin()),
	       thrust::make_permutation_iterator(bGPU->end(),   (cusparsestructLo->ordIndicesGPU)->end()),
	       xGPU->begin());


#if defined(PETSC_USE_REAL_SINGLE)  
  stat = cusparseScsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0f, cusparsestructLo->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->column_indices[0]),
			      cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast(xGPU->data()),
			      (PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()));CHKERRCUSP(stat);
  
  stat = cusparseScsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0f, cusparsestructUp->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->column_indices[0]),
			      cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()),
			      (PetscScalar *) thrust::raw_pointer_cast(xGPU->data()));CHKERRCUSP(stat);

#elif defined(PETSC_USE_REAL_DOUBLE)
  stat = cusparseDcsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0, cusparsestructLo->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->column_indices[0]),
			      cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast(xGPU->data()),
			      (PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()));CHKERRCUSP(stat);
  
  stat = cusparseDcsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0, cusparsestructUp->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->column_indices[0]),
			      cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()),
			      (PetscScalar *) thrust::raw_pointer_cast(xGPU->data()));CHKERRCUSP(stat);
 #endif
  
  // Copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place.
  thrust::copy(thrust::make_permutation_iterator(xGPU->begin(),   (cusparsestructUp->ordIndicesGPU)->begin()),
	       thrust::make_permutation_iterator(xGPU->end(), (cusparsestructUp->ordIndicesGPU)->end()),
	       (cusparsestructLo->tempvecGPU)->begin());
  
  // Copy the temporary to the full solution.
  thrust::copy((cusparsestructLo->tempvecGPU)->begin(), (cusparsestructLo->tempvecGPU)->end(), xGPU->begin());
	  	  
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJCUSP_NaturalOrdering"
PetscErrorCode MatSolve_SeqAIJCUSP_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  CUSPARRAY         *xGPU, *bGPU;
  PetscInt          n = A->rmap->n;
  cusparseStatus_t stat;
  Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
  Mat_SeqAIJCUSPARSETriFactor *cusparsestructLo  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPARSETriFactor *cusparsestructUp  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->upTriFactorPtr;
  
  PetscFunctionBegin;
  ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);

#if defined(PETSC_USE_REAL_SINGLE)  
  stat = cusparseScsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0f, cusparsestructLo->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->column_indices[0]),
			      cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast(bGPU->data()),
			      (PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()));CHKERRCUSP(stat);
    
  stat = cusparseScsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0f, cusparsestructUp->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->column_indices[0]),
			      cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()),
			      (PetscScalar *) thrust::raw_pointer_cast(xGPU->data()));CHKERRCUSP(stat);
  
#elif defined(PETSC_USE_REAL_DOUBLE)	
  stat = cusparseDcsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0, cusparsestructLo->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructLo->mat)->column_indices[0]),
			      cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast(bGPU->data()),
			      (PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()));CHKERRCUSP(stat);
  
  stat = cusparseDcsrsv_solve(Mat_CUSPARSE_Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      n, 1.0, cusparsestructUp->Mat_CUSPARSE_description,
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->values[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->row_offsets[0]),
			      thrust::raw_pointer_cast(&(cusparsestructUp->mat)->column_indices[0]),
			      cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo,
			      (const PetscScalar *) thrust::raw_pointer_cast((cusparsestructLo->tempvecGPU)->data()),
			      (PetscScalar *) thrust::raw_pointer_cast(xGPU->data()));CHKERRCUSP(stat);
#endif
  ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyToGPU"
PetscErrorCode MatCUSPCopyToGPU(Mat A)
{

#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_SeqAIJGPU *cuspstruct  = (Mat_SeqAIJGPU*)A->spptr;
#else
  Mat_SeqAIJCUSP *cuspstruct  = (Mat_SeqAIJCUSP*)A->spptr;
#endif
  Mat_SeqAIJ      *a          = (Mat_SeqAIJ*)A->data;
  PetscInt        m           = A->rmap->n,*ii,*ridx;
  PetscErrorCode  ierr;


  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){
    ierr = PetscLogEventBegin(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    /*
      It may be possible to reuse nonzero structure with new matrix values but 
      for simplicity and insured correctness we delete and build a new matrix on
      the GPU. Likely a very small performance hit.
    */
    if (cuspstruct->mat){
      try {
	delete cuspstruct->mat;
	if (cuspstruct->tempvec)
	  delete cuspstruct->tempvec;
	
      } catch(char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      } 
    }
    try {
      cuspstruct->nonzerorow=0;
      for (int j = 0; j<m; j++)
	cuspstruct->nonzerorow += ((a->i[j+1]-a->i[j])>0);

#ifdef PETSC_HAVE_TXPETSCGPU
      if (a->compressedrow.use) {	
	m    = a->compressedrow.nrows;
	ii   = a->compressedrow.i;
	ridx = a->compressedrow.rindex;
      } else {
	// Forcing compressed row on the GPU ... only relevant for CSR storage
	int k=0;
	ierr = PetscMalloc((cuspstruct->nonzerorow+1)*sizeof(PetscInt), &ii);CHKERRQ(ierr);
	ierr = PetscMalloc((cuspstruct->nonzerorow)*sizeof(PetscInt), &ridx);CHKERRQ(ierr);
	ii[0]=0;
	for (int j = 0; j<m; j++) {
	  if ((a->i[j+1]-a->i[j])>0) {
	    ii[k] = a->i[j];
	    ridx[k]= j;
	    k++;
	  }
	}
	ii[cuspstruct->nonzerorow] = a->nz;
	m = cuspstruct->nonzerorow;
      }

      // Build our matrix ... first determine the GPU storage type
      PetscBool found;
      char      input[4] = "csr";
      std::string storageFormat;
      ierr = PetscOptionsGetString(PETSC_NULL, "-cusp_storage_format", input, 4, &found);CHKERRQ(ierr);
      storageFormat.assign(input);
      if(storageFormat!="csr" && storageFormat!="dia"&& storageFormat!="ell"&& storageFormat!="coo")
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Bad argument to -cusp_storage_format. Must be either 'csr', 'dia' (diagonal), 'ell' (ellpack), or 'coo' (coordinate)\n");
      else {
	cuspstruct->mat = GPU_Matrix_Factory::getNew(storageFormat);

	// Create the streams and events (if desired). 
	PetscMPIInt    size;
	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
	ierr = cuspstruct->mat->buildStreamsAndEvents(size, &theBodyStream);CHKERRCUSP(ierr);	

	// Determine which library to use : cusp or cusparse
	PetscBool flg = PETSC_FALSE;
	ierr = PetscOptionsGetBool(PETSC_NULL,"-use_cusparse",&flg,PETSC_NULL);CHKERRQ(ierr);
	if (flg)
	  cuspstruct->mat->initializeCusparse();
	
	// lastly, build the matrix
	cuspstruct->mat->setMatrix(m, A->cmap->n, a->nz, ii, a->j, a->a);
	cuspstruct->mat->setCPRowIndices(ridx, m);
	
	///
	// INODES : Determine the inode data structure for the GPU.
	//   This only really matters for the CSR format.
	//
	if (a->inode.use) {
	  PetscInt * temp;
	  ierr = PetscMalloc((a->inode.node_count+1)*sizeof(PetscInt), &temp);CHKERRQ(ierr);
	  temp[0]=0;
	  PetscInt nodeMax=0, nnzPerRowMax=0;
	  for (int i = 0; i<a->inode.node_count; i++) {
	    temp[i+1]= a->inode.size[i]+temp[i];
	    if (a->inode.size[i] > nodeMax)
	      nodeMax = a->inode.size[i];
	  }
	  // Determine the maximum number of nonzeros in a row.
	  cuspstruct->nonzerorow = 0;
	  for (int j = 0; j<A->rmap->n; j++) {
	    cuspstruct->nonzerorow += ((a->i[j+1]-a->i[j])>0);
	    if (a->i[j+1]-a->i[j] > nnzPerRowMax) {
	      nnzPerRowMax = a->i[j+1]-a->i[j];
	    }
	  }
	  // Set the Inode data ... only relevant for CSR really
	  cuspstruct->mat->setInodeData(temp, a->inode.node_count+1, nnzPerRowMax, nodeMax, a->inode.node_count);
	  ierr = PetscFree(temp);CHKERRQ(ierr);
	}

      }
      if (!a->compressedrow.use) {	
	// free data
	ierr = PetscFree(ii);CHKERRQ(ierr);
	ierr = PetscFree(ridx);CHKERRQ(ierr);
      }
     
#else

      cuspstruct->mat = new CUSPMATRIX;
      if (a->compressedrow.use) {
	m    = a->compressedrow.nrows;
	ii   = a->compressedrow.i;
	ridx = a->compressedrow.rindex;
	cuspstruct->mat->resize(m,A->cmap->n,a->nz);
	cuspstruct->mat->row_offsets.assign(ii,ii+m+1);
	cuspstruct->mat->column_indices.assign(a->j,a->j+a->nz);
	cuspstruct->mat->values.assign(a->a,a->a+a->nz);
	cuspstruct->indices = new CUSPINTARRAYGPU;
	cuspstruct->indices->assign(ridx,ridx+m);
      } else {
	cuspstruct->mat->resize(m,A->cmap->n,a->nz);
	cuspstruct->mat->row_offsets.assign(a->i,a->i+m+1);
	cuspstruct->mat->column_indices.assign(a->j,a->j+a->nz);
	cuspstruct->mat->values.assign(a->a,a->a+a->nz);
      }
#endif
      cuspstruct->tempvec = new CUSPARRAY;
      cuspstruct->tempvec->resize(m);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
    ierr = PetscLogEventEnd(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyFromGPU"
PetscErrorCode MatCUSPCopyFromGPU(Mat A, CUSPMATRIX *Agpu)
{
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *) A->spptr;
  Mat_SeqAIJ     *a          = (Mat_SeqAIJ *) A->data;
  PetscInt        m          = A->rmap->n;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED) {
      try {
        cuspstruct->mat = Agpu;
        if (a->compressedrow.use) {
          //PetscInt *ii, *ridx;
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Cannot handle row compression for GPU matrices");
        } else {
          PetscInt i;

          if (m+1 != (PetscInt) cuspstruct->mat->row_offsets.size()) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "GPU matrix has %d rows, should be %d", cuspstruct->mat->row_offsets.size()-1, m);
          a->nz    = cuspstruct->mat->values.size();
          a->maxnz = a->nz; /* Since we allocate exactly the right amount */
          A->preallocated = PETSC_TRUE;
          // Copy ai, aj, aa
          if (a->singlemalloc) {
            if (a->a) {ierr = PetscFree3(a->a,a->j,a->i);CHKERRQ(ierr);}
          } else {
            if (a->i) {ierr = PetscFree(a->i);CHKERRQ(ierr);}
            if (a->j) {ierr = PetscFree(a->j);CHKERRQ(ierr);}
            if (a->a) {ierr = PetscFree(a->a);CHKERRQ(ierr);}
          }
          ierr = PetscMalloc3(a->nz,PetscScalar,&a->a,a->nz,PetscInt,&a->j,m+1,PetscInt,&a->i);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory(A, a->nz*(sizeof(PetscScalar)+sizeof(PetscInt))+(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
          a->singlemalloc = PETSC_TRUE;
          thrust::copy(cuspstruct->mat->row_offsets.begin(), cuspstruct->mat->row_offsets.end(), a->i);
          thrust::copy(cuspstruct->mat->column_indices.begin(), cuspstruct->mat->column_indices.end(), a->j);
          thrust::copy(cuspstruct->mat->values.begin(), cuspstruct->mat->values.end(), a->a);
          // Setup row lengths
          if (a->imax) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
          ierr = PetscMalloc2(m,PetscInt,&a->imax,m,PetscInt,&a->ilen);CHKERRQ(ierr);
          ierr = PetscLogObjectMemory(A, 2*m*sizeof(PetscInt));CHKERRQ(ierr);
          for(i = 0; i < m; ++i) {
            a->imax[i] = a->ilen[i] = a->i[i+1] - a->i[i];
          }
          // a->diag?
        }
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "CUSP error: %s", ex);
      }
    }
    // This assembly prevents resetting the flag to PETSC_CUSP_CPU and recopying
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only valid for unallocated GPU matrices");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetVecs_SeqAIJCUSP"
PetscErrorCode MatGetVecs_SeqAIJCUSP(Mat mat, Vec *right, Vec *left)
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
#define __FUNCT__ "MatMult_SeqAIJCUSP"
PetscErrorCode MatMult_SeqAIJCUSP(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_SeqAIJGPU *cuspstruct = (Mat_SeqAIJGPU *)A->spptr;
#else
  PetscBool      usecprow    = a->compressedrow.use;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *)A->spptr;
#endif
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  // The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSP
  // ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  try {
#ifdef PETSC_HAVE_TXPETSCGPU
    ierr = cuspstruct->mat->multiply(xarray, yarray);CHKERRCUSP(ierr);
    //ierr = cuspstruct->mat->multiply(thrust::raw_pointer_cast(xarray->data()),
    //				     thrust::raw_pointer_cast(yarray->data()));CHKERRCUSP(ierr);
#else
    if (usecprow){ /* use compressed row format */
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      ierr = VecSet_SeqCUSP(yy,0.0);CHKERRQ(ierr);
      thrust::copy(cuspstruct->tempvec->begin(),cuspstruct->tempvec->end(),thrust::make_permutation_iterator(yarray->begin(),cuspstruct->indices->begin()));
    } else { /* do not use compressed row format */
      cusp::multiply(*cuspstruct->mat,*xarray,*yarray);
    }
#endif

  } catch (char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
#ifdef PETSC_HAVE_TXPETSCGPU
  if (!cuspstruct->mat->hasNonZeroStream())
    ierr = WaitForGPU();CHKERRCUSP(ierr);
#else
  ierr = WaitForGPU();CHKERRCUSP(ierr);
#endif
  ierr = PetscLogFlops(2.0*a->nz - cuspstruct->nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


struct VecCUSPPlusEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJCUSP"
PetscErrorCode MatMultAdd_SeqAIJCUSP(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_SeqAIJGPU *cuspstruct = (Mat_SeqAIJGPU *)A->spptr;
#else
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *)A->spptr;
#endif
  CUSPARRAY      *xarray,*yarray,*zarray;
  PetscFunctionBegin;
  // The line below should not be necessary as it has been moved to MatAssemblyEnd_SeqAIJCUSP
  // ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  try {      
    ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
#ifdef PETSC_HAVE_TXPETSCGPU
    ierr = cuspstruct->mat->multiplyAdd(xarray, zarray);CHKERRCUSP(ierr);
#else
    if (a->compressedrow.use) {
      cusp::multiply(*cuspstruct->mat,*xarray, *cuspstruct->tempvec);
      thrust::for_each(
           thrust::make_zip_iterator(
                 thrust::make_tuple(cuspstruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(zarray->begin(), cuspstruct->indices->begin()))),
           thrust::make_zip_iterator(
                 thrust::make_tuple(cuspstruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(zarray->begin(),cuspstruct->indices->begin()))) + cuspstruct->tempvec->size(),
           VecCUSPPlusEquals());
    } else {
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      thrust::for_each(
         thrust::make_zip_iterator(thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    zarray->begin())),
         thrust::make_zip_iterator(thrust::make_tuple(
                                    cuspstruct->tempvec->end(),
                                    zarray->end())),
         VecCUSPPlusEquals());
    }
#endif
    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
    
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUSP"
PetscErrorCode MatAssemblyEnd_SeqAIJCUSP(Mat A,MatAssemblyType mode)
{
  PetscErrorCode  ierr;  
  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  // this is not necessary since MatCUSPCopyToGPU has been called.
  //if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
  //  A->valid_GPU_matrix = PETSC_CUSP_CPU;
  //}
  A->ops->mult    = MatMult_SeqAIJCUSP;
  A->ops->multadd    = MatMultAdd_SeqAIJCUSP;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJCUSP"
/*@C
   MatCreateSeqAIJCUSP - Creates a sparse matrix in AIJ (compressed row) format
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
PetscErrorCode  MatCreateSeqAIJCUSP(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCUSP);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_TXPETSCGPU

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSP"
PetscErrorCode MatDestroy_SeqAIJCUSP(Mat A)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJGPU      *cuspstruct = (Mat_SeqAIJGPU*)A->spptr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    // The regular matrices
    try {
      if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
	delete (GPU_Matrix_Ifc *)(cuspstruct->mat);
      }
      if (cuspstruct->tempvec!=0)
	delete cuspstruct->tempvec;
      delete cuspstruct;
      A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
  } else {
    // The triangular factors
    try {
      Mat_SeqAIJTriFactors *cusparseTriFactors  = (Mat_SeqAIJTriFactors*)A->spptr;
      Mat_SeqAIJCUSPARSETriFactor *cusparsestructLo  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->loTriFactorPtr;
      Mat_SeqAIJCUSPARSETriFactor *cusparsestructUp  = (Mat_SeqAIJCUSPARSETriFactor*)cusparseTriFactors->upTriFactorPtr;
      cusparseStatus_t stat;
      
      stat = cusparseDestroy(Mat_CUSPARSE_Handle);CHKERRCUSP(stat);
      stat = cusparseDestroyMatDescr(cusparsestructLo->Mat_CUSPARSE_description);CHKERRCUSP(stat);
      stat = cusparseDestroySolveAnalysisInfo(cusparsestructLo->Mat_CUSPARSE_solveAnalysisInfo);CHKERRCUSP(stat);

      // destroy the matrix and the container
      if (cusparsestructLo->mat!=0)
	delete cusparsestructLo->mat;
      if (cusparsestructLo->tempvecGPU!=0)
	delete cusparsestructLo->tempvecGPU;
      delete cusparsestructLo;
      
      stat = cusparseDestroyMatDescr(cusparsestructUp->Mat_CUSPARSE_description);CHKERRCUSP(stat);
      stat = cusparseDestroySolveAnalysisInfo(cusparsestructUp->Mat_CUSPARSE_solveAnalysisInfo);CHKERRCUSP(stat);

      // destroy the matrix and the container
      if (cusparsestructUp->mat!=0)
	delete cusparsestructUp->mat;
      if (cusparsestructUp->tempvecGPU!=0)
	delete cusparsestructUp->tempvecGPU;
      delete cusparsestructUp;	  
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSPARSE error: %s", ex);
    } 
  }

  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;

  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else // if PETSC_HAVE_TXPETSCGPU is 0

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJCUSP"
PetscErrorCode MatDestroy_SeqAIJCUSP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *cuspcontainer = (Mat_SeqAIJCUSP*)A->spptr;

  PetscFunctionBegin;
  try {
    if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
      delete (CUSPMATRIX *)(cuspcontainer->mat);
    }
    delete cuspcontainer;
    A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  } 
  /*this next line is because MatDestroy tries to PetscFree spptr if it is not zero, and PetscFree only works if the memory was allocated with PetscNew or PetscMalloc, which don't call the constructor */
  A->spptr = 0;
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif // PETSC_HAVE_TXPETSCGPU
/*
#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqAIJCUSPFromTriple"
PetscErrorCode MatCreateSeqAIJCUSPFromTriple(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt* i, PetscInt* j, PetscScalar*a, Mat *mat, PetscInt nz, PetscBool idx)
{
  CUSPMATRIX *gpucsr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (idx){
  }
  PetscFunctionReturn(0);
}*/

extern PetscErrorCode MatSetValuesBatch_SeqAIJCUSP(Mat, PetscInt, PetscInt, PetscInt *,const PetscScalar*);

#ifdef PETSC_HAVE_TXPETSCGPU

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJCUSP"
PetscErrorCode  MatCreate_SeqAIJCUSP(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  B->ops->mult    = MatMult_SeqAIJCUSP;
  B->ops->multadd = MatMultAdd_SeqAIJCUSP;

  if (B->factortype==MAT_FACTOR_NONE) {
    /* you cannot check the inode.use flag here since the matrix was just created.*/
    B->spptr        = new Mat_SeqAIJGPU;
    ((Mat_SeqAIJGPU *)B->spptr)->mat = 0;
    ((Mat_SeqAIJGPU *)B->spptr)->tempvec = 0;
  } else {
    Mat_SeqAIJTriFactors *triFactors;
    /* NEXT, set the pointers to the triangular factors */
    B->spptr = new Mat_SeqAIJTriFactors;
    triFactors  = (Mat_SeqAIJTriFactors*)B->spptr;
    triFactors->loTriFactorPtr = 0;
    triFactors->upTriFactorPtr = 0;
    
    triFactors->loTriFactorPtr        = new Mat_SeqAIJCUSPARSETriFactor;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->loTriFactorPtr)->mat = 0;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->loTriFactorPtr)->Mat_CUSPARSE_description = 0;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->loTriFactorPtr)->Mat_CUSPARSE_solveAnalysisInfo = 0;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->loTriFactorPtr)->tempvecGPU = 0;
    
    triFactors->upTriFactorPtr        = new Mat_SeqAIJCUSPARSETriFactor;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->upTriFactorPtr)->mat = 0;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->upTriFactorPtr)->Mat_CUSPARSE_description = 0;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->upTriFactorPtr)->Mat_CUSPARSE_solveAnalysisInfo = 0;
    ((Mat_SeqAIJCUSPARSETriFactor *)triFactors->upTriFactorPtr)->tempvecGPU = 0;
  }

  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSP;
  B->ops->destroy        = MatDestroy_SeqAIJCUSP;
  B->ops->getvecs        = MatGetVecs_SeqAIJCUSP;
  B->ops->setvaluesbatch = MatSetValuesBatch_SeqAIJCUSP;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSP);CHKERRQ(ierr);
  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_petsc_C","MatGetFactor_seqaij_petsccusp",MatGetFactor_seqaij_petsccusp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#else // if PETSC_HAVE_TXPETSCGPU is 0

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJCUSP"
PetscErrorCode  MatCreate_SeqAIJCUSP(Mat B)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  aij             = (Mat_SeqAIJ*)B->data;
  aij->inode.use  = PETSC_FALSE;
  B->ops->mult    = MatMult_SeqAIJCUSP;
  B->ops->multadd = MatMultAdd_SeqAIJCUSP;
  B->spptr        = new Mat_SeqAIJCUSP;
  ((Mat_SeqAIJCUSP *)B->spptr)->mat = 0;
  ((Mat_SeqAIJCUSP *)B->spptr)->tempvec = 0;
  ((Mat_SeqAIJCUSP *)B->spptr)->indices = 0;
  
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUSP;
  B->ops->destroy        = MatDestroy_SeqAIJCUSP;
  B->ops->getvecs        = MatGetVecs_SeqAIJCUSP;
  B->ops->setvaluesbatch = MatSetValuesBatch_SeqAIJCUSP;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUSP);CHKERRQ(ierr);
  B->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif // PETSC_HAVE_TXPETSCGPU
