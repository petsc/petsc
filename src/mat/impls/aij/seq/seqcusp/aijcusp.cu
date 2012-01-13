

/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "petscbt.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "private/vecimpl.h"
PETSC_CUDA_EXTERN_C_END
#undef VecType
#include "../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h"


#ifdef PETSC_HAVE_TXPETSCGPU

#include "csr_matrix_data.h"
#include "csr_matrix_data_gpu.h"
#include "csr_tri_solve_gpu.h"
#include "csr_tri_solve_gpu_level_scheduler.h"
#include "csr_spmv_inode.h"
#include <algorithm>
#include <vector>
#include <string>
#include <thrust/sort.h>
#include <thrust/fill.h>

#define CSRMATRIXCPU csr_matrix_data<PetscInt,PetscScalar>
#define CSRMATRIXGPU csr_matrix_data_gpu<PetscInt,PetscScalar>

static std::string GPU_TRI_SOLVE_ALGORITHM="none";

struct Mat_SeqAIJCUSPTriFactors {
  void *loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  void *upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
};

struct Mat_SeqAIJCUSPInode {
  CSRMATRIXGPU*       mat; /* pointer to the matrix on the GPU */
  CUSPARRAY*        tempvec; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  CUSPINTARRAYGPU*  inodes; /*pointer to an array containing the inode data structure should use inode be true*/
  PetscInt nnzPerRowMax; /* maximum number of nonzeros in a row ... for shared memory vector size */
  PetscInt nodeMax; /* maximum number of nonzeros in a row ... for shared memory vector size */
};

struct Mat_SeqAIJCUSPTriFactorHybrid {
  CSRMATRIXCPU*    cpuMat; /* pointer to the matrix on the CPU */
  CSRMATRIXGPU*    gpuMat; /* pointer to the matrix on the GPU */
  CUSPARRAY*       tempvecGPU; /*pointer to a workvector for storing temporary results on the GPU */
  PetscInt *       nnzPerRowInDiagBlock; /* pointer to a cpu vector defining nnz in diagonal block */
  PetscScalar*     tempvecCPU1; /*pointer to a workvector for storing temporary results on the CPU */
  PetscScalar*     tempvecCPU2; /*pointer to a workvector for storing temporary results on the CPU */
  PetscInt   nnz; /* Number of nonzeros in the triangular factor */
  PetscInt   block_size; /* block size */
};

struct Mat_SeqAIJCUSPTriFactorLevelScheduler {
  CSRMATRIXGPU*    gpuMat;     /* pointer to the matrix on the GPU */
  CUSPARRAY*       tempvecGPU; /* pointer to a workvector for storing temporary results on the GPU */
  CUSPINTARRAYGPU*  perms;     /* pointer to an array containing the permutation array*/
  CUSPINTARRAYGPU*  levels;    /* pointer to an array containing the levels data*/
  CUSPINTARRAYGPU*  ordIndicesGPU; /* For Lower triangular, this is the row permutation. For Upper triangular, this the column permutation */
  PetscInt * levelsCPU;        /* pointer to an array containing the levels data*/
  PetscInt   nLevels; /* number of levels */
  PetscInt   levelSum; /* number of levels */
  PetscInt   maxNumUnknownsAtSameLevel; /* maximum number of unkowns that can be computed simultaneously */
};


EXTERN_C_BEGIN
PetscErrorCode MatILUFactorSymbolic_SeqAIJCUSP(Mat,Mat,IS,IS,const MatFactorInfo*);
PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSP(Mat,Mat,IS,IS,const MatFactorInfo*);
PetscErrorCode MatLUFactorNumeric_SeqAIJCUSP(Mat,Mat,const MatFactorInfo *);
PetscErrorCode MatSolve_SeqAIJCUSP(Mat,Vec,Vec);
PetscErrorCode MatSolve_SeqAIJCUSP_NaturalOrdering(Mat,Vec,Vec);
PetscErrorCode MatMult_SeqAIJCUSP_Inode(Mat,Vec,Vec);
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
  ierr = MatILUFactorSymbolic_SeqAIJ(fact,A,isrow,iscol,info); CHKERRQ(ierr);
  (fact)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJCUSP"
PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSP(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info); CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCUSPUnravelOrderingAndCopyToGPU"
PetscErrorCode MatCUSPUnravelOrderingAndCopyToGPU(Mat A)
{
  Mat_SeqAIJCUSPTriFactors *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
  Mat_SeqAIJCUSPTriFactorHybrid *cuspstructLo  = (Mat_SeqAIJCUSPTriFactorHybrid*)cuspTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPTriFactorHybrid *cuspstructUp  = (Mat_SeqAIJCUSPTriFactorHybrid*)cuspTriFactors->upTriFactorPtr;

  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscInt          n = A->rmap->n;
  const PetscInt    *ai = a->i,*aj = a->j,*adiag = a->diag,*vi;
  const MatScalar   *aa = a->a,*v;
  PetscInt *AiLo, *AjLo, *AiUp, *AjUp;
  PetscScalar *AALo, *AAUp;
  PetscInt          i,nz, nzLower, nzUpper, offset, rowOffset, j, block_size_counter, nnzBlockLower=0, nnzBlockUpper=0;
  PetscErrorCode    ierr;
  bool success;

  PetscFunctionBegin;

  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){
    // Get the block size from the command line
    PetscInt int_value=0;
    PetscBool found;
    PetscInt block_size=1;
    ierr = PetscOptionsGetInt(((PetscObject)A)->prefix, "-gpu_LU_block_size", &int_value, &found); CHKERRQ(ierr);
    if(found == PETSC_TRUE) {
      if(int_value > 0)
	block_size = int_value;
      else
	printf("Bad argument to -gpu_LU_block_size.  Must be positive.\n");
    }
    else {
      printf("-gpu_LU_block_size positive_int not found.  Use internal formula.\n");
      block_size = 1000; //get_gpu_LU_block_size(); // something like that for now
    }
	
    /*************************************************************************/
    /* To Unravel the factored matrix into 2 CSR matrices, do the following  */
    /* - Calculate the number of nonzeros in the lower triangular sparse     */
    /*   including 1's on the diagonal.                                      */
    /* - Calculate the number of nonzeros in the upper triangular sparse     */
    /*   including arbitrary values on the diagonal.                         */
    /* - Fill the Lower triangular portion from the matrix A                 */
    /* - Fill the Upper triangular portion from the matrix A                 */
    /* - Assign each to a separate cusp data structure                       */
    /*************************************************************************/

    /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
    nzLower=n+ai[n]-ai[1];
    /* next, figure out the number of nonzeros in the upper triangular matrix. */
    nzUpper = adiag[0]-adiag[n];

    cudaError_t err;
    /* Allocate Space for the lower triangular matrix */	
    ierr = PetscMalloc(n*sizeof(PetscInt), &(cuspstructLo->nnzPerRowInDiagBlock));CHKERRQ(ierr);
    err = cudaMallocHost((void **) &AiLo, (n+1)*sizeof(PetscInt)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AjLo, nzLower*sizeof(PetscInt)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AALo, nzLower*sizeof(PetscScalar)); CHKERRCUSP(err);

    /* set the number of nonzeros */
    cuspstructLo->nnz=nzLower;
    cuspstructLo->block_size=block_size;

    /* Fill the lower triangular matrix */
    AiLo[0]=(PetscInt) 0;
    AiLo[n]=nzLower;
    AjLo[0]=(PetscInt) 0;
    AALo[0]=(MatScalar) 1.0;
    v    = aa;
    vi   = aj;
    offset=1;
    rowOffset=1;
    cuspstructLo->nnzPerRowInDiagBlock[0]=1;
    nnzBlockLower+=1;
    block_size_counter=0;
    for (i=1; i<n; i++) {
      nz  = ai[i+1] - ai[i];
      // additional 1 for the term on the diagonal
      AiLo[i]=rowOffset;
      rowOffset+=nz+1;
      
      memcpy(&(AjLo[offset]), vi, nz*sizeof(PetscInt));
      memcpy(&(AALo[offset]), v, nz*sizeof(MatScalar));
      
      offset+=nz;
      AjLo[offset]=(PetscInt) i;
      AALo[offset]=(MatScalar) 1.0;
      offset+=1;

      // Count the number of nnz per row in the diagonal blocks.
      offset-=nz+1;
      if (i%block_size==0 && i>0)
	block_size_counter++;
      j=0;       
      while (AjLo[offset+j]<block_size_counter*block_size) j++;
      cuspstructLo->nnzPerRowInDiagBlock[i]=nz+1-j;
      nnzBlockLower+=cuspstructLo->nnzPerRowInDiagBlock[i];
      offset+=nz+1;

      v  += nz;
      vi += nz;
    }

    /* set the number of nonzeros */
    cuspstructUp->nnz=nzUpper;
    cuspstructUp->block_size=block_size;

    /* Allocate Space for the upper triangular matrix */
    ierr = PetscMalloc(n*sizeof(PetscInt), &(cuspstructUp->nnzPerRowInDiagBlock));CHKERRQ(ierr);
    err = cudaMallocHost((void **) &AiUp, (n+1)*sizeof(PetscInt)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AjUp, nzUpper*sizeof(PetscInt)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AAUp, nzUpper*sizeof(PetscScalar)); CHKERRCUSP(err);
    
    /* Fill the upper triangular matrix */
    AiUp[0]=(PetscInt) 0;
    AiUp[n]=nzUpper;
    offset = nzUpper;
    block_size_counter=-1;
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
      
      // copy the off diagonal elements
      memcpy(&(AjUp[offset+1]), vi, nz*sizeof(PetscInt));
      memcpy(&(AAUp[offset+1]), v, nz*sizeof(MatScalar));
    }

    // Count the number of nnz per row in the diagonal blocks.
    // need to do this by working from the top of the matrix
    block_size_counter=0;
    for (i=0; i<n; i++){
      if (i%block_size==0 && i>0)
	block_size_counter++;
      nz = AiUp[i+1]-AiUp[i];
      j=0;
      while (AjUp[AiUp[i]+j]<(block_size_counter+1)*block_size && j<nz) j++;
      cuspstructUp->nnzPerRowInDiagBlock[i]=j;
      nnzBlockUpper+=cuspstructUp->nnzPerRowInDiagBlock[i];
    }

    try {	
      /* The Lower triangular matrix */
      cuspstructLo->cpuMat = new CSRMATRIXCPU(n,n,nzLower,AiLo,AjLo,AALo);      
      cuspstructLo->gpuMat = new CSRMATRIXGPU;
      success = (cuspstructLo->gpuMat)->copy_from_host(*(cuspstructLo->cpuMat));
      if (!success) {
	printf("Failed in cuspstructLo->gpuMat->copy_from_host\n");
	CHKERRCUSP(1);
      }
      // allocate temporary vectors using pinned memory
      err = cudaMallocHost((void **) &(cuspstructLo->tempvecCPU1), 
			   (size_t) n*sizeof(PetscScalar)); CHKERRCUSP(err);
      err = cudaMallocHost((void **) &(cuspstructLo->tempvecCPU2), 
			   (size_t) n*sizeof(PetscScalar)); CHKERRCUSP(err);

      cuspstructLo->tempvecGPU = new CUSPARRAY;
      (cuspstructLo->tempvecGPU)->resize(n);
      

      /* The Upper triangular matrix */
      cuspstructUp->cpuMat = new CSRMATRIXCPU(n,n,nzUpper,AiUp,AjUp,AAUp);
      cuspstructUp->gpuMat = new CSRMATRIXGPU;
      success = (cuspstructUp->gpuMat)->copy_from_host(*(cuspstructUp->cpuMat));
      if (!success) {
	printf("Failed in cuspstructUp->gpuMat->copy_from_host\n");
	CHKERRCUSP(1);
      }
      // allocate temporary vectors using pinned memory
      err = cudaMallocHost((void **) &(cuspstructUp->tempvecCPU1), 
			   (size_t) n*sizeof(PetscScalar)); CHKERRCUSP(err);
      err = cudaMallocHost((void **) &(cuspstructUp->tempvecCPU2), 
			   (size_t) n*sizeof(PetscScalar)); CHKERRCUSP(err);

      cuspstructUp->tempvecGPU = new CUSPARRAY;
      (cuspstructUp->tempvecGPU)->resize(n);
      
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
  }
  PetscFunctionReturn(0);	
}


#undef __FUNCT__  
#define __FUNCT__ "MatCUSPUnravelOrderingToLevelSchedulerAndCopyToGPU"
PetscErrorCode MatCUSPUnravelOrderingToLevelSchedulerAndCopyToGPU(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJCUSPTriFactors *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructLo  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->loTriFactorPtr;
  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructUp  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->upTriFactorPtr;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscInt          n = A->rmap->n;
  const PetscInt    *ai = a->i,*aj = a->j,*adiag = a->diag,*vi;
  const MatScalar   *aa = a->a,*v;
  PetscInt i, j, max, nz, nzLower, nzUpper, offset;
  PetscInt *AiLo, *AjLo, *AiUp, *AjUp, *levelsCPULo, *levelsCPUUp;
  PetscScalar *AALo, *AAUp, *AADiag;
  bool success;
  std::vector<PetscInt> lLo(n);
  std::vector<PetscInt> lUp(n);
  std::vector<PetscInt> qLo(n);
  std::vector<PetscInt> qUp(n);
  std::vector<PetscInt> lLoBin(0);
  std::vector<PetscInt> lUpBin(0);

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){

    PetscBool diagFlag = PETSC_FALSE;
    PetscBool diagFlagFull = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL, "-level_scheduler_diagnostics_view", &diagFlag, PETSC_NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(PETSC_NULL, "-level_scheduler_diagnostics_view_full", &diagFlagFull, PETSC_NULL); CHKERRQ(ierr);

    /* initialize to lLo/lUp to 0 ... qLo/qUp to the index */
    for (i=0; i<n; i++) { lLo[i] = 0; qLo[i] = i; lUp[i] = 0; qUp[i] = i; }

    /* Compute the lower triangular levels */
    lLo[0] = 1;
    lLoBin.push_back(1);
    for (i=1; i<n; i++) {
      nz  = ai[i+1] - ai[i];
      lLo[i] = 1; max = 1;

      for (j=0; j<nz; j++)
	max = lLo[ aj[ ai[i]+j ] ]>max ? lLo[ aj[ ai[i]+j ] ] : max;      

      lLo[i] = 1+max;

      if (max>= static_cast<PetscInt>(lLoBin.size()))
	lLoBin.push_back(1);
      else
	lLoBin[max]++;
    }
    
    /* compute the number of levels */
    cuspstructLo->nLevels=lLoBin.size();

    /* set the maximum number of unknowns at the same level */
    cuspstructLo->maxNumUnknownsAtSameLevel=*(std::max_element(&lLoBin[0], &lLoBin[0]+cuspstructLo->nLevels));

    /* compute the sum of all the levels */
    cuspstructLo->levelSum=0;
    for (i=0; i< static_cast<PetscInt>(lLoBin.size()); i++)
      cuspstructLo->levelSum+=lLoBin[i];

    /* Determine the permutation array through a keyed sort ... easy to do in thrust */
    thrust::sort_by_key(&lLo[0], &lLo[0]+n, &qLo[0]);

    /* print out level scheduler diagnostics for the lower triangular matrix */
    if (diagFlag || diagFlagFull) {
      std::cout << std::endl;
      std::cout << "nlevels in lower triangular factor="<<cuspstructLo->nLevels<<std::endl;
      std::cout << "maxNumUnknownsAtSameLevel in lower triangular factor="<<cuspstructLo->maxNumUnknownsAtSameLevel<<std::endl;
      std::cout << "levelSum (should be equal to number of unknowns)="<<cuspstructLo->levelSum<<std::endl;
      std::cout << "number of unknowns="<<n<<std::endl;

      if (diagFlagFull) {
	std::cout << "Ordering of unknowns in the lower triangular matrix"<<std::endl;
	std::cout << "==================================================="<<std::endl;
	int index = 0;
	std::cout << "Level # : number of unknowns at this level :  (level #,  unknown index)" << std::endl;
	for (i=0; i<static_cast<PetscInt>(lLoBin.size()); i++) {
	  std::cout << "Level " << i+1 << " : " << lLoBin[i] << " : ";
	  for (j=0; j<lLoBin[i]; j++)
         std::cout << "  (" << lLo[index + j] << "," << qLo[index + j] <<")";
	  std::cout << std::endl;
	  index+=lLoBin[i];
	}
      }
      std::cout << std::endl;
    }

    /* Compute the upper triangular levels */
    lUp[n-1] = 1;
    lUpBin.push_back(1);

    // set the pointers
    v    = aa+ai[n]-ai[1];
    vi   = aj+ai[n]-ai[1];

    for (i=n-2; i>=0; i--){
      // set the pointers
      v   = aa + adiag[i+1] + 1;
      vi  = aj + adiag[i+1] + 1;
      
      // number of elements NOT on the diagonal
      nz = adiag[i] - adiag[i+1] -1;

      lUp[i] = 1; max = 0;

      for (j=0; j<nz; j++)
	max = lUp[ vi[ j ] ]>max ? lUp[ vi[ j ] ] : max;      

      lUp[i] = 1+max;
      if (max>= static_cast<PetscInt>(lUpBin.size()))
	lUpBin.push_back(1);
      else
	lUpBin[max]++;
    }

    /* compute the number of levels */
    cuspstructUp->nLevels=lUpBin.size();

    /* set the maximum number of unknowns at the same level */
    cuspstructUp->maxNumUnknownsAtSameLevel=*(std::max_element(&lUpBin[0], &lUpBin[0]+cuspstructUp->nLevels));

    /* compute the sum of all the levels */
    cuspstructUp->levelSum=0;
    for (i=0; i< static_cast<PetscInt>(lUpBin.size()); i++)
      cuspstructUp->levelSum+=lUpBin[i];

    /* Determine the permutation array through a keyed sort ... easy to do in thrust */
    thrust::sort_by_key(&lUp[0], &lUp[0]+n, &qUp[0]);

    /* print out level scheduler diagnostics for the upper triangular matrix */
    if (diagFlag || diagFlagFull) {
      std::cout << std::endl;
      std::cout << "nlevels in upper triangular factor="<<cuspstructUp->nLevels<<std::endl;
      std::cout << "maxNumUnknownsAtSameLevel in upper triangular factor="<<cuspstructUp->maxNumUnknownsAtSameLevel<<std::endl;
      std::cout << "levelSum (should be equal to number of unknowns)="<<cuspstructUp->levelSum<<std::endl;
      std::cout << "number of unknowns="<<n<<std::endl;

      if (diagFlagFull) {
	std::cout << "Ordering of unknowns in the upper triangular matrix"<<std::endl;
	std::cout << "==================================================="<<std::endl;
	int index = 0;
	std::cout << "Level # : number of unknowns at this level :  (level #,  unknown index)" << std::endl;
	for (i=0; i<static_cast<PetscInt>(lUpBin.size()); i++) {
	  std::cout << "Level " << i+1 << " : " << lUpBin[i] << " : ";
	  for (j=0; j<lUpBin[i]; j++)
	    std::cout << "  (" << lUp[index + j] << "," << qUp[index + j] <<")";
	  std::cout << std::endl;
	  index+=lUpBin[i];
	}
      }
      std::cout << std::endl;
    }

    ierr = PetscMalloc(lLoBin.size()*sizeof(PetscInt), &levelsCPULo);CHKERRQ(ierr);
    ierr = PetscMalloc(lUpBin.size()*sizeof(PetscInt), &levelsCPUUp);CHKERRQ(ierr);

    memcpy(&levelsCPULo[0], &lLoBin[0], lLoBin.size()*sizeof(PetscInt));
    memcpy(&levelsCPUUp[0], &lUpBin[0], lUpBin.size()*sizeof(PetscInt));

    /*************************************************************************/
    /* To Unravel the factored matrix into 2 CSR matrices, do the following  */
    /* - Calculate the number of nonzeros in the lower triangular sparse     */
    /*   including 1's on the diagonal.                                      */
    /* - Calculate the number of nonzeros in the upper triangular sparse     */
    /*   including arbitrary values on the diagonal.                         */
    /* - Fill the Lower triangular portion from the matrix A                 */
    /* - Fill the Upper triangular portion from the matrix A                 */
    /* - Assign each to a separate cusp data structure                       */
    /*************************************************************************/

    /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
    nzLower=ai[n]-ai[1];
    /* next, figure out the number of nonzeros in the upper triangular matrix ... excluding the diagonal. */
    nzUpper = adiag[0]-adiag[n]-n;

    /* Set pointers for lower triangular matrices */
    AiLo = const_cast<PetscInt *>(ai);
    AjLo = const_cast<PetscInt *>(aj);
    AALo = const_cast<PetscScalar *>(aa);

    /* Allocate Space for the upper triangular matrix */
    cudaError_t err = cudaMallocHost((void **) &AiUp, (n+1)*sizeof(PetscInt)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AjUp, nzUpper*sizeof(PetscInt)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AAUp, nzUpper*sizeof(PetscScalar)); CHKERRCUSP(err);
    err = cudaMallocHost((void **) &AADiag, n*sizeof(PetscScalar)); CHKERRCUSP(err);
    
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
      offset -= nz;
      
      // first, set the diagonal elements
      // this is actually the inverse of the diagonal.
      AADiag[i] = v[nz];
      AiUp[i] = AiUp[i+1] - nz;
      
      // copy the off diagonal elements
      memcpy(&(AjUp[offset]), vi, nz*sizeof(PetscInt));
      memcpy(&(AAUp[offset]), v, nz*sizeof(MatScalar));
      // scale the rest of the matrix by the inverse of the diagonal
      for (j=0; j<nz; j++) AAUp[offset+j]*=v[nz];      
    }

    try {	
      Mat_SeqAIJ *b=(Mat_SeqAIJ *)A->data;
      IS               isrow = b->row,iscol = b->icol;
      PetscBool        row_identity,col_identity;
      const PetscInt   *r,*c;

      ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
      ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
      ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
      ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);

      cuspstructLo->ordIndicesGPU = new CUSPINTARRAYGPU;
      (cuspstructLo->ordIndicesGPU)->assign(&r[0], &r[0]+A->rmap->n);
      
      cuspstructUp->ordIndicesGPU = new CUSPINTARRAYGPU;
      (cuspstructUp->ordIndicesGPU)->assign(&c[0], &c[0]+A->rmap->n);
      
      /* The Lower triangular matrix */
      CSRMATRIXCPU * cpuMat = new CSRMATRIXCPU(n,n,nzLower,AiLo,AjLo,AALo);      
      cuspstructLo->gpuMat = new CSRMATRIXGPU;
      success = (cuspstructLo->gpuMat)->copy_from_host(*cpuMat);
      if (!success) { printf("Failed in cuspstructLo->gpuMat->copy_from_host\n"); CHKERRCUSP(1); }
      delete cpuMat;

      cuspstructLo->tempvecGPU = new CUSPARRAY;
      (cuspstructLo->tempvecGPU)->resize(n);
      thrust::fill(cuspstructLo->tempvecGPU->begin(), cuspstructLo->tempvecGPU->end(), (PetscScalar) 1.0);
      
      cuspstructLo->levels = new CUSPINTARRAYGPU;
      (cuspstructLo->levels)->assign(&lLoBin[0], &lLoBin[0]+cuspstructLo->nLevels);

      cuspstructLo->levelsCPU = levelsCPULo;

      cuspstructLo->perms = new CUSPINTARRAYGPU;
      (cuspstructLo->perms)->assign(&qLo[0], &qLo[0]+n);

      /* The Upper triangular matrix */
      cpuMat = new CSRMATRIXCPU(n,n,nzUpper,AiUp,AjUp,AAUp);
      cuspstructUp->gpuMat = new CSRMATRIXGPU;
      success = (cuspstructUp->gpuMat)->copy_from_host(*cpuMat);
      if (!success) { printf("Failed in cuspstructUp->gpuMat->copy_from_host\n"); CHKERRCUSP(1); }
      delete cpuMat;

      // will use this vector to contain the inverse of the diagonal
      cuspstructUp->tempvecGPU = new CUSPARRAY;
      (cuspstructUp->tempvecGPU)->assign(&AADiag[0], &AADiag[0]+n);

      cuspstructUp->levels = new CUSPINTARRAYGPU;
      (cuspstructUp->levels)->assign(&lUpBin[0], &lUpBin[0]+cuspstructUp->nLevels);

      cuspstructUp->levelsCPU = levelsCPUUp;

      cuspstructUp->perms = new CUSPINTARRAYGPU;
      (cuspstructUp->perms)->assign(&qUp[0], &qUp[0]+n);
      
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }

    // Free CPU space
    err = cudaFreeHost(AiUp); CHKERRCUSP(err);
    err = cudaFreeHost(AjUp); CHKERRCUSP(err);
    err = cudaFreeHost(AAUp); CHKERRCUSP(err);
    err = cudaFreeHost(AADiag); CHKERRCUSP(err);

    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
  }
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
  
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info); CHKERRQ(ierr);
  
  // determine which version of MatSolve needs to be used.
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) B->ops->solve = MatSolve_SeqAIJCUSP_NaturalOrdering;    
  else                              B->ops->solve = MatSolve_SeqAIJCUSP; 

  // get the triangular factors
  if (GPU_TRI_SOLVE_ALGORITHM!="none") {
    if (GPU_TRI_SOLVE_ALGORITHM=="levelScheduler") {
      ierr = MatCUSPUnravelOrderingToLevelSchedulerAndCopyToGPU(B);CHKERRQ(ierr);
    } else {
      ierr = MatCUSPUnravelOrderingAndCopyToGPU(B);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJCUSP"
PetscErrorCode MatSolve_SeqAIJCUSP(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscBool      usecprow    = a->compressedrow.use;
  CUSPARRAY      *xGPU, *bGPU;

  PetscFunctionBegin;

  if (GPU_TRI_SOLVE_ALGORITHM!="none") {
    // Get the GPU pointers
    ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);
    if (usecprow){ /* use compressed row format */
      try {
	;
	// Have no idea what to do here!
	
      } catch (char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    } else { /* do not use compressed row format */
      try {
	
	if (GPU_TRI_SOLVE_ALGORITHM=="levelScheduler") {
	  Mat_SeqAIJCUSPTriFactors *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
	  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructLo  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->loTriFactorPtr;
	  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructUp  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->upTriFactorPtr;
	  
	  // Copy the right hand side vector, bGPU, into xGPU with the row permutation
	  thrust::copy(thrust::make_permutation_iterator(bGPU->begin(), (cuspstructLo->ordIndicesGPU)->begin()), 
		       thrust::make_permutation_iterator(bGPU->end(),   (cuspstructLo->ordIndicesGPU)->end()),
		       xGPU->begin());
	  
	  // Lower solve
	  ierr = csr_tri_solve_level_scheduler<PetscInt, PetscScalar>(cuspstructLo->gpuMat,
								      cuspstructLo->nLevels,
								      cuspstructLo->maxNumUnknownsAtSameLevel,
								      cuspstructLo->levelSum,
								      thrust::raw_pointer_cast((cuspstructLo->levels)->data()),
								      cuspstructLo->levelsCPU,
								      thrust::raw_pointer_cast((cuspstructLo->perms)->data()),
								      thrust::raw_pointer_cast(xGPU->data())); CHKERRCUSP(ierr);
	  
	  // Scale the result of the lower solve by diagonal vector stored in cuspstructUp->tempvecGPU.
	  // ALL off-diagonal terms in the upper triangular matrix are already normalized by the diagonal factor
	  thrust::transform((cuspstructUp->tempvecGPU)->begin(), (cuspstructUp->tempvecGPU)->end(), 
			    xGPU->begin(), xGPU->begin(), thrust::multiplies<PetscScalar>());

	  // Upper solve
	  ierr = csr_tri_solve_level_scheduler<PetscInt, PetscScalar>(cuspstructUp->gpuMat,
								      cuspstructUp->nLevels,
								      cuspstructUp->maxNumUnknownsAtSameLevel,
								      cuspstructUp->levelSum,
								      thrust::raw_pointer_cast((cuspstructUp->levels)->data()),
								      cuspstructUp->levelsCPU,
								      thrust::raw_pointer_cast((cuspstructUp->perms)->data()),
								      thrust::raw_pointer_cast(xGPU->data())); CHKERRCUSP(ierr);
	  
	  
	  // Copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place.
	  thrust::copy(thrust::make_permutation_iterator(xGPU->begin(),   (cuspstructUp->ordIndicesGPU)->begin()),
		       thrust::make_permutation_iterator(xGPU->end(), (cuspstructUp->ordIndicesGPU)->end()),
		       (cuspstructLo->tempvecGPU)->begin());
	  
	  // Copy the temporary to the full solution.
	  thrust::copy((cuspstructLo->tempvecGPU)->begin(), (cuspstructLo->tempvecGPU)->end(), xGPU->begin());
	  
	}
	else {
	  std::cout << "Error in MatSolve_SeqAIJCUSP : Currently, only levelScheduler is supported for GPU tri-solve when using matrix reordering." << std::endl;
	  CHKERRCUSP(1);
	}
      } catch(char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      } 
    }
    ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  } else { 

    // Revert to the CPU solve if a GPU algorithm is not found!
    ierr = MatSolve_SeqAIJ(A,bb,xx); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJCUSP_NaturalOrdering"
PetscErrorCode MatSolve_SeqAIJCUSP_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscBool         usecprow    = a->compressedrow.use;
  PetscScalar       *x;
  const PetscScalar *b;
  CUSPARRAY         *xGPU, *bGPU;

  PetscFunctionBegin;
  if (GPU_TRI_SOLVE_ALGORITHM!="none") {
    // Get the GPU pointers
    ierr = VecCUSPGetArrayWrite(xx,&xGPU);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(bb,&bGPU);CHKERRQ(ierr);
    if (usecprow){ /* use compressed row format */
      try {
	;
	// Have no idea what to do here!
	
      } catch (char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    } else { /* do not use compressed row format */
      try {
	
	if (GPU_TRI_SOLVE_ALGORITHM=="levelScheduler") {
	  Mat_SeqAIJCUSPTriFactors *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
	  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructLo  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->loTriFactorPtr;
	  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructUp  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->upTriFactorPtr;
	  
	  // Copy bGPU to another temporary on the GPU so that the temporary can be overwritten.
	  // This should be wrapped in a VecCUSPCopyGpuToGpu function with safety mechanisms.
	  thrust::copy(bGPU->begin(),bGPU->end(), xGPU->begin());
	  
	  // Lower solve
	  ierr = csr_tri_solve_level_scheduler<PetscInt, PetscScalar>(cuspstructLo->gpuMat,
								      cuspstructLo->nLevels,
								      cuspstructLo->maxNumUnknownsAtSameLevel,
								      cuspstructLo->levelSum,
								      thrust::raw_pointer_cast((cuspstructLo->levels)->data()),
								      cuspstructLo->levelsCPU,
								      thrust::raw_pointer_cast((cuspstructLo->perms)->data()),
								      thrust::raw_pointer_cast(xGPU->data())); CHKERRCUSP(ierr);
	  
	  
	  // Scale the result of the lower solve by diagonal vector stored in 
	  // the remainder off diagonal terms in the upper triangular matrix are already normalized
	  thrust::transform((cuspstructUp->tempvecGPU)->begin(), (cuspstructUp->tempvecGPU)->end(), 
			    xGPU->begin(), xGPU->begin(), thrust::multiplies<PetscScalar>());
	  
	  // Upper solve
	  ierr = csr_tri_solve_level_scheduler<PetscInt, PetscScalar>(cuspstructUp->gpuMat,
								      cuspstructUp->nLevels,
								      cuspstructUp->maxNumUnknownsAtSameLevel,
								      cuspstructUp->levelSum,
								      thrust::raw_pointer_cast((cuspstructUp->levels)->data()),
								      cuspstructUp->levelsCPU,
								      thrust::raw_pointer_cast((cuspstructUp->perms)->data()),								       
								      thrust::raw_pointer_cast(xGPU->data())); CHKERRCUSP(ierr);
	  
	  
	} else {
	  // Get the CPU pointers
	  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
	  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
	  
	  Mat_SeqAIJCUSPTriFactors *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
	  Mat_SeqAIJCUSPTriFactorHybrid *cuspstructLo = (Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr;
	  Mat_SeqAIJCUSPTriFactorHybrid *cuspstructUp = (Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr;
	  
	  thrust::copy(bGPU->begin(),bGPU->end(),(cuspstructUp->tempvecGPU)->begin());
	  memcpy(&(cuspstructLo->tempvecCPU2[0]), b, A->rmap->n*sizeof(PetscScalar));
	  
	  // Lower solve
	  ierr = csr_tri_solve_gpu_hybrid<PetscInt, PetscScalar>(cuspstructLo->cpuMat,
								 cuspstructLo->nnzPerRowInDiagBlock,
								 uplo_lo_only, cuspstructLo->tempvecCPU2, cuspstructLo->tempvecCPU1,
								 cuspstructLo->gpuMat,
								 thrust::raw_pointer_cast((cuspstructUp->tempvecGPU)->data()),
								 thrust::raw_pointer_cast((cuspstructLo->tempvecGPU)->data()),
								 cuspstructLo->block_size, 1, 0); CHKERRCUSP(ierr);
	  
	  // Upper solve
	  ierr = csr_tri_solve_gpu_hybrid<PetscInt, PetscScalar> (cuspstructUp->cpuMat,
								  cuspstructUp->nnzPerRowInDiagBlock,
								  uplo_up_only, cuspstructLo->tempvecCPU1, x,
								  cuspstructUp->gpuMat,
								  thrust::raw_pointer_cast((cuspstructLo->tempvecGPU)->data()),
								  thrust::raw_pointer_cast(xGPU->data()),
								  cuspstructUp->block_size, 1 , 0); CHKERRCUSP(ierr);
	}
      } catch(char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      } 
    }
    ierr = VecCUSPRestoreArrayRead(bb,&bGPU);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(xx,&xGPU);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  } else { 

    // Revert to the CPU solve if a GPU algorithm is not found!
    ierr = MatSolve_SeqAIJ_NaturalOrdering(A,bb,xx); CHKERRQ(ierr);	
  }
  
  PetscFunctionReturn(0);
}


#endif // PETSC_HAVE_TXPETSCGPU

#undef __FUNCT__
#define __FUNCT__ "MatCUSPCopyToGPU"
PetscErrorCode MatCUSPCopyToGPU(Mat A)
{
  Mat_SeqAIJCUSP *cuspstruct  = (Mat_SeqAIJCUSP*)A->spptr;
  Mat_SeqAIJ      *a          = (Mat_SeqAIJ*)A->data;
  PetscInt        m           = A->rmap->n,*ii,*ridx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){
    ierr = PetscLogEventBegin(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED){
      try {
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
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    } else if (A->valid_GPU_matrix == PETSC_CUSP_CPU) {
      /*
       It may be possible to reuse nonzero structure with new matrix values but 
       for simplicity and insured correctness we delete and build a new matrix on
       the GPU. Likely a very small performance hit.
       */
      if (cuspstruct->mat){
        try {
          delete (cuspstruct->mat);
          if (cuspstruct->tempvec) {
            delete (cuspstruct->tempvec);
          }
          if (cuspstruct->indices) {
            delete (cuspstruct->indices);
          }
        } catch(char* ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
        } 
      }
      try {
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
        cuspstruct->tempvec = new CUSPARRAY;
        cuspstruct->tempvec->resize(m);
      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      } 
    }
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

          if (m+1 != (PetscInt) cuspstruct->mat->row_offsets.size()) {SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "GPU matrix has %d rows, should be %d", cuspstruct->mat->row_offsets.size()-1, m);}
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
  PetscInt       nonzerorow=0;
  PetscBool      usecprow    = a->compressedrow.use;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *)A->spptr;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (usecprow){ /* use compressed row format */
    try {
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      ierr = VecSet_SeqCUSP(yy,0.0);CHKERRQ(ierr);
      thrust::copy(cuspstruct->tempvec->begin(),cuspstruct->tempvec->end(),thrust::make_permutation_iterator(yarray->begin(),cuspstruct->indices->begin()));
    } catch (char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  } else { /* do not use compressed row format */
    try {
      cusp::multiply(*cuspstruct->mat,*xarray,*yarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#ifdef PETSC_HAVE_TXPETSCGPU

#undef __FUNCT__
#define __FUNCT__ "MatInodeCUSPCopyToGPU"
PetscErrorCode MatInodeCUSPCopyToGPU(Mat A)
{
  Mat_SeqAIJCUSPInode *cuspstruct  = (Mat_SeqAIJCUSPInode*)A->spptr;
  Mat_SeqAIJ          *a          = (Mat_SeqAIJ*)A->data;
  PetscErrorCode      ierr;
  bool                success=0;

  PetscFunctionBegin;
  if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED || A->valid_GPU_matrix == PETSC_CUSP_CPU){
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogEventBegin(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (A->valid_GPU_matrix == PETSC_CUSP_UNALLOCATED){
      try {
	// Construct the GPU csr matrix
	CSRMATRIXCPU * cpuMat = new CSRMATRIXCPU(A->rmap->n, A->cmap->n, a->nz, a->i, a->j, a->a);      
	cuspstruct->mat = new CSRMATRIXGPU;
	success = (cuspstruct->mat)->copy_from_host(*cpuMat);
	if (!success) {
	  printf("Failed in cuspstructLo->gpuMat->copy_from_host\n");
	  CHKERRCUSP(1);
	}
	delete cpuMat;
	
	cuspstruct->tempvec = new CUSPARRAY;
	cuspstruct->tempvec->resize(A->rmap->n);

	// Determine the inode data structure for the GPU
	PetscInt * temp;
	ierr = PetscMalloc((a->inode.node_count+1)*sizeof(PetscInt), &temp);CHKERRQ(ierr);
	temp[0]=0;
	cuspstruct->nodeMax = 0;
	for (int i = 0; i<a->inode.node_count; i++) {
	  temp[i+1]= a->inode.size[i]+temp[i];
	  if (a->inode.size[i] > cuspstruct->nodeMax)
	    cuspstruct->nodeMax = a->inode.size[i];
	}
	cuspstruct->inodes = new CUSPINTARRAYGPU;
	cuspstruct->inodes->assign(temp, temp+a->inode.node_count+1);
	ierr = PetscFree(temp); CHKERRQ(ierr);

	// Determine the maximum number of nonzeros in a row.
        cuspstruct->nnzPerRowMax=0;
	for (int j = 0; j<A->rmap->n; j++) {
	  if (a->i[j+1]-a->i[j] > cuspstruct->nnzPerRowMax) {
	    cuspstruct->nnzPerRowMax = a->i[j+1]-a->i[j];
	  }
	}

      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
    } else if (A->valid_GPU_matrix == PETSC_CUSP_CPU) {
      /*
       It may be possible to reuse nonzero structure with new matrix values but 
       for simplicity and insured correctness we delete and build a new matrix on
       the GPU. Likely a very small performance hit.
       */
      if (cuspstruct->mat){
        try {
          delete (cuspstruct->mat);
          if (cuspstruct->tempvec) {
            delete (cuspstruct->tempvec);
          }
          if (cuspstruct->inodes) {
            delete (cuspstruct->inodes);
          }
        } catch(char* ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
        } 
      }
      try {
	// Construct the GPU csr matrix
	CSRMATRIXCPU * cpuMat = new CSRMATRIXCPU(A->rmap->n, A->cmap->n, a->nz, a->i, a->j, a->a);      
	cuspstruct->mat = new CSRMATRIXGPU;
	success = (cuspstruct->mat)->copy_from_host(*cpuMat);
	if (!success) {
	  printf("Failed in cuspstructLo->gpuMat->copy_from_host\n");
	  CHKERRCUSP(1);
	}
	delete cpuMat;

	cuspstruct->tempvec = new CUSPARRAY;
	cuspstruct->tempvec->resize(A->rmap->n);

	// Determine the inode data structure for the GPU
	PetscInt * temp;
	ierr = PetscMalloc((a->inode.node_count+1)*sizeof(PetscInt), &temp);CHKERRQ(ierr);
	temp[0]=0;
	cuspstruct->nodeMax = 0;
	for (int i = 0; i<a->inode.node_count; i++) {
	  temp[i+1]= a->inode.size[i]+temp[i];
	  if (a->inode.size[i] > cuspstruct->nodeMax)
	    cuspstruct->nodeMax = a->inode.size[i];
	}
	cuspstruct->inodes = new CUSPINTARRAYGPU;
	cuspstruct->inodes->assign(temp, temp+a->inode.node_count+1);
	ierr = PetscFree(temp); CHKERRQ(ierr);

	// Determine the maximum number of nonzeros in a row.
        cuspstruct->nnzPerRowMax=0;
	for (int j = 0; j<A->rmap->n+1; j++)
	  if (a->i[j+1]-a->i[j] > cuspstruct->nnzPerRowMax)
	    cuspstruct->nnzPerRowMax = a->i[j+1]-a->i[j];

      } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      } 
    }
    A->valid_GPU_matrix = PETSC_CUSP_BOTH;
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogEventEnd(MAT_CUSPCopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJCUSP_Inode"
PetscErrorCode MatMult_SeqAIJCUSP_Inode(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ                *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode            ierr;
  PetscInt                  nonzerorow=0;
  PetscBool                 usecprow    = a->compressedrow.use;
  const Mat_SeqAIJCUSPInode *cuspstruct = (Mat_SeqAIJCUSPInode *)A->spptr;
  CUSPARRAY                 *xarray, *yarray;

  PetscFunctionBegin;
  if (!a->inode.size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Inode Structure");

  ierr = MatInodeCUSPCopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUSPCopyToGPU(xx);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (usecprow){ /* use compressed row format */
    try {
      // not sure what to do here
      ; 
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  } else { /* do not use compressed row format */
    try {
      ierr = csr_spmv_inode<PetscInt, PetscScalar>(cuspstruct->mat, 
						   a->inode.node_count, cuspstruct->nodeMax, cuspstruct->nnzPerRowMax,
      						   thrust::raw_pointer_cast((cuspstruct->inodes)->data()),
      						   thrust::raw_pointer_cast(xarray->data()),
      						   thrust::raw_pointer_cast(yarray->data())); CHKERRCUSP(ierr);

      
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
  }
  ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#endif // PETSC_HAVE_TXPETSCGPU

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
  PetscBool      usecprow=a->compressedrow.use;
  Mat_SeqAIJCUSP *cuspstruct = (Mat_SeqAIJCUSP *)A->spptr;
  CUSPARRAY      *xarray,*yarray,*zarray;

  PetscFunctionBegin;
  ierr = MatCUSPCopyToGPU(A);CHKERRQ(ierr);
  if (usecprow) {
    try {
      ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
      if (a->compressedrow.nrows) {
        cusp::multiply(*cuspstruct->mat,*xarray, *cuspstruct->tempvec);
        thrust::for_each(
           thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(zarray->begin(), cuspstruct->indices->begin()))),
           thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    thrust::make_permutation_iterator(zarray->begin(),cuspstruct->indices->begin()))) + cuspstruct->tempvec->size(),
           VecCUSPPlusEquals());
      }
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
  } else {
    try {
      ierr = VecCopy_SeqCUSP(yy,zz);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yy,&yarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
      cusp::multiply(*cuspstruct->mat,*xarray,*cuspstruct->tempvec);
      thrust::for_each(
         thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->begin(),
                                    zarray->begin())),
         thrust::make_zip_iterator(
                 thrust::make_tuple(
                                    cuspstruct->tempvec->end(),
                                   zarray->end())),
         VecCUSPPlusEquals());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
    ierr = VecCUSPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yy,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJCUSP"
PetscErrorCode MatAssemblyEnd_SeqAIJCUSP(Mat A,MatAssemblyType mode)
{
  PetscErrorCode  ierr;
#ifdef PETSC_HAVE_TXPETSCGPU
  Mat_SeqAIJ      *aij = (Mat_SeqAIJ*)A->data;
#endif // PETSC_HAVE_TXPETSCGPU
  
  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
    A->valid_GPU_matrix = PETSC_CUSP_CPU;
  }

#ifdef PETSC_HAVE_TXPETSCGPU
  if (aij->inode.use)  A->ops->mult    = MatMult_SeqAIJCUSP_Inode;
  else                 A->ops->mult    = MatMult_SeqAIJCUSP;
#endif // PETSC_HAVE_TXPETSCGPU

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

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateMPIAIJ()

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
  PetscErrorCode      ierr;
  Mat_SeqAIJ          *a          = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUSP      *cuspstruct = (Mat_SeqAIJCUSP*)A->spptr;
  Mat_SeqAIJCUSPInode *cuspstructInode = (Mat_SeqAIJCUSPInode*)A->spptr;
  cudaError_t         err;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    // The regular matrices
    try {
      if (A->valid_GPU_matrix != PETSC_CUSP_UNALLOCATED){
	if (!a->inode.use) 
	  delete (CUSPMATRIX *)(cuspstruct->mat);
	else
	  delete (CSRMATRIXGPU *)(cuspstructInode->mat);
      }
      if (!a->inode.use) {
	if (cuspstruct->tempvec!=0)
	  delete cuspstruct->tempvec;
	if (cuspstruct->indices!=0)
	  delete cuspstruct->indices;
	delete cuspstruct;
      } else {
	if (cuspstructInode->tempvec!=0)
	  delete cuspstructInode->tempvec;
	if (cuspstructInode->inodes!=0)
	  delete cuspstructInode->inodes;
	delete cuspstructInode;
      }
      A->valid_GPU_matrix = PETSC_CUSP_UNALLOCATED;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
  } else {
    // The triangular factors
    if (GPU_TRI_SOLVE_ALGORITHM!="none") {
      try {
	if (GPU_TRI_SOLVE_ALGORITHM=="levelScheduler") {
	  
	  Mat_SeqAIJCUSPTriFactors *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
	  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructLo  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->loTriFactorPtr;
	  Mat_SeqAIJCUSPTriFactorLevelScheduler *cuspstructUp  = (Mat_SeqAIJCUSPTriFactorLevelScheduler*)cuspTriFactors->upTriFactorPtr;
	  
	  // the Lower factor
	  if (cuspstructLo->gpuMat!=0)
	    delete (CSRMATRIXGPU *)(cuspstructLo->gpuMat);
	  if (cuspstructLo->tempvecGPU!=0)
	    delete cuspstructLo->tempvecGPU;
	  if (cuspstructLo->levels!=0)
	    delete cuspstructLo->levels;
	  if (cuspstructLo->levelsCPU!=0) {
	    ierr = PetscFree(cuspstructLo->levelsCPU); CHKERRQ(ierr); }
	  if (cuspstructLo->perms!=0)
	    delete cuspstructLo->perms;
	  if (cuspstructLo->ordIndicesGPU!=0)
	    delete cuspstructLo->ordIndicesGPU;
	  delete cuspstructLo;
	  
	  // the Upper factor
	  if (cuspstructUp->gpuMat!=0)
	    delete (CSRMATRIXGPU *)(cuspstructUp->gpuMat);
	  if (cuspstructUp->tempvecGPU!=0)
	    delete cuspstructUp->tempvecGPU;
	  if (cuspstructUp->levels!=0)
	    delete cuspstructUp->levels;
	  if (cuspstructUp->levelsCPU!=0) {
	    ierr = PetscFree(cuspstructUp->levelsCPU); CHKERRQ(ierr); }
	  if (cuspstructUp->perms!=0)
	    delete cuspstructUp->perms;
	  if (cuspstructUp->ordIndicesGPU!=0)
	    delete cuspstructUp->ordIndicesGPU;
	  
	  delete cuspstructUp;
	  
	  /* Set the pointers to 0 */
	  cuspTriFactors->loTriFactorPtr = 0;
	  cuspTriFactors->upTriFactorPtr = 0;    
	  
	} else {
	  
	  Mat_SeqAIJCUSPTriFactors      *cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)A->spptr;
	  Mat_SeqAIJCUSPTriFactorHybrid *cuspstructLo = (Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr;
	  Mat_SeqAIJCUSPTriFactorHybrid *cuspstructUp = (Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr;
	  
	  // the Lower factor
	  if (cuspstructLo->cpuMat) {
	    ierr = PetscFree(cuspstructLo->nnzPerRowInDiagBlock); CHKERRQ(ierr);
	    err = cudaFreeHost(cuspstructLo->tempvecCPU1); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructLo->tempvecCPU2); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructLo->cpuMat->row_offsets); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructLo->cpuMat->column_indices); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructLo->cpuMat->values); CHKERRCUSP(err);
	    delete (CSRMATRIXCPU *)(cuspstructLo->cpuMat);
	  }
	  if (cuspstructLo->gpuMat)
	    delete (CSRMATRIXGPU *)(cuspstructLo->gpuMat);
	  if (cuspstructLo->tempvecGPU)
	    delete cuspstructLo->tempvecGPU;
	  delete cuspstructLo;
	  
	  // the Upper factor
	  if (cuspstructUp->cpuMat) {
	    ierr = PetscFree(cuspstructUp->nnzPerRowInDiagBlock); CHKERRQ(ierr);
	    err = cudaFreeHost(cuspstructUp->tempvecCPU1); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructUp->tempvecCPU2); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructUp->cpuMat->row_offsets); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructUp->cpuMat->column_indices); CHKERRCUSP(err);
	    err = cudaFreeHost(cuspstructUp->cpuMat->values); CHKERRCUSP(err);
	    delete (CSRMATRIXCPU *)(cuspstructUp->cpuMat);
	  }
	  if (cuspstructUp->gpuMat)
	    delete (CSRMATRIXGPU *)(cuspstructUp->gpuMat);
	  if (cuspstructUp->tempvecGPU)
	    delete cuspstructUp->tempvecGPU;
	  delete cuspstructUp;
	  
	  /* Set the pointers to 0 */
	  cuspTriFactors->loTriFactorPtr = 0;
	  cuspTriFactors->upTriFactorPtr = 0;    
	}
      } catch(char* ex) {
	SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      } 
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
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr            = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  b = (Mat_SeqAIJ*)B->data;
  B->ops->mult    = MatMult_SeqAIJCUSP;
  B->ops->multadd = MatMultAdd_SeqAIJCUSP;

  if (B->factortype==MAT_FACTOR_NONE) {
    /* you cannot check the inode.use flag here since the matrix was just created.*/
    if (!b->inode.use) {
      B->spptr        = new Mat_SeqAIJCUSP;
      ((Mat_SeqAIJCUSP *)B->spptr)->mat = 0;
      ((Mat_SeqAIJCUSP *)B->spptr)->tempvec = 0;
      ((Mat_SeqAIJCUSP *)B->spptr)->indices = 0;
    } else {
      B->spptr        = new Mat_SeqAIJCUSPInode;
      ((Mat_SeqAIJCUSPInode *)B->spptr)->mat = 0;
      ((Mat_SeqAIJCUSPInode *)B->spptr)->tempvec = 0;
      ((Mat_SeqAIJCUSPInode *)B->spptr)->inodes = 0;
      ((Mat_SeqAIJCUSPInode *)B->spptr)->nnzPerRowMax = 0;
      ((Mat_SeqAIJCUSPInode *)B->spptr)->nodeMax = 0;
    }
  } else {
    // Get the tri solve algorithm
    PetscBool found;
    char      input[20] = "hybrid";

    ierr = PetscOptionsGetString(PETSC_NULL, "-gpu_tri_solve_algorithm", input, 20, &found);CHKERRQ(ierr);
    GPU_TRI_SOLVE_ALGORITHM.assign(input);
    if(GPU_TRI_SOLVE_ALGORITHM!="levelScheduler" && GPU_TRI_SOLVE_ALGORITHM!="hybrid") SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Bad argument to -gpu_tri_solve_algorithm. Must be either 'hybrid' or 'levelScheduler'\n");

    
    if (GPU_TRI_SOLVE_ALGORITHM!="none") {    
      Mat_SeqAIJCUSPTriFactors *cuspTriFactors;
      /* NEXT, set the pointers to the triangular factors */
      B->spptr = new Mat_SeqAIJCUSPTriFactors;
      cuspTriFactors  = (Mat_SeqAIJCUSPTriFactors*)B->spptr;
      cuspTriFactors->loTriFactorPtr = 0;
      cuspTriFactors->upTriFactorPtr = 0;
      
      if (GPU_TRI_SOLVE_ALGORITHM=="levelScheduler") {
	cuspTriFactors->loTriFactorPtr        = new Mat_SeqAIJCUSPTriFactorLevelScheduler;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->gpuMat = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->tempvecGPU = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->levels = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->ordIndicesGPU = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->levelsCPU = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->perms = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->nLevels = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->maxNumUnknownsAtSameLevel = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->loTriFactorPtr)->levelSum = 0;
	
	cuspTriFactors->upTriFactorPtr        = new Mat_SeqAIJCUSPTriFactorLevelScheduler;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->gpuMat = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->tempvecGPU = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->levels = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->ordIndicesGPU = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->levelsCPU = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->perms = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->nLevels = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->maxNumUnknownsAtSameLevel = 0;
	((Mat_SeqAIJCUSPTriFactorLevelScheduler *)cuspTriFactors->upTriFactorPtr)->levelSum = 0;
      } else {
	cuspTriFactors->loTriFactorPtr        = new Mat_SeqAIJCUSPTriFactorHybrid;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->cpuMat = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->gpuMat = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->nnzPerRowInDiagBlock = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->tempvecGPU = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->tempvecCPU1 = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->tempvecCPU2 = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->nnz = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->loTriFactorPtr)->block_size = 0;
	
	cuspTriFactors->upTriFactorPtr        = new Mat_SeqAIJCUSPTriFactorHybrid;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->cpuMat = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->gpuMat = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->nnzPerRowInDiagBlock = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->tempvecGPU = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->tempvecCPU1 = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->tempvecCPU2 = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->nnz = 0;
	((Mat_SeqAIJCUSPTriFactorHybrid *)cuspTriFactors->upTriFactorPtr)->block_size = 0;
      }
    }
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
