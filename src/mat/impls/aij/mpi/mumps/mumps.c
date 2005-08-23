#define PETSCMAT_DLL

/* 
    Provides an interface to the MUMPS_4.3.1 sparse solver
*/
#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"

EXTERN_C_BEGIN 
#if defined(PETSC_USE_COMPLEX)
#include "zmumps_c.h"
#else
#include "dmumps_c.h" 
#endif
EXTERN_C_END 
#define JOB_INIT -1
#define JOB_END -2
/* macros s.t. indices match MUMPS documentation */
#define ICNTL(I) icntl[(I)-1] 
#define CNTL(I) cntl[(I)-1] 
#define INFOG(I) infog[(I)-1]
#define INFO(I) info[(I)-1]
#define RINFOG(I) rinfog[(I)-1]
#define RINFO(I) rinfo[(I)-1]

typedef struct {
#if defined(PETSC_USE_COMPLEX)
  ZMUMPS_STRUC_C id;
#else
  DMUMPS_STRUC_C id;
#endif
  MatStructure   matstruc;
  int            myid,size,*irn,*jcn,sym;
  PetscScalar    *val;
  MPI_Comm       comm_mumps;

  PetscTruth     isAIJ,CleanUpMUMPS;
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*MatView)(Mat,PetscViewer);
  PetscErrorCode (*MatAssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*MatCholeskyFactorSymbolic)(Mat,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*specialdestroy)(Mat);
  PetscErrorCode (*MatPreallocate)(Mat,int,int,int*,int,int*);
} Mat_MUMPS;

EXTERN PetscErrorCode MatDuplicate_MUMPS(Mat,MatDuplicateOption,Mat*);
EXTERN_C_BEGIN
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SBAIJ_SBAIJMUMPS(Mat,MatType,MatReuse,Mat*);
EXTERN_C_END
/* convert Petsc mpiaij matrix to triples: row[nz], col[nz], val[nz] */
/*
  input: 
    A       - matrix in mpiaij or mpisbaij (bs=1) format
    shift   - 0: C style output triple; 1: Fortran style output triple.
    valOnly - FALSE: spaces are allocated and values are set for the triple  
              TRUE:  only the values in v array are updated
  output:     
    nnz     - dim of r, c, and v (number of local nonzero entries of A)
    r, c, v - row and col index, matrix values (matrix triples) 
 */
PetscErrorCode MatConvertToTriples(Mat A,int shift,PetscTruth valOnly,int *nnz,int **r, int **c, PetscScalar **v) {
  int         *ai, *aj, *bi, *bj, rstart,nz, *garray;
  PetscErrorCode ierr;
  int         i,j,jj,jB,irow,m=A->m,*ajj,*bjj,countA,countB,colA_start,jcol;
  int         *row,*col;
  PetscScalar *av, *bv,*val;
  Mat_MUMPS   *mumps=(Mat_MUMPS*)A->spptr;

  PetscFunctionBegin;
  if (mumps->isAIJ){
    Mat_MPIAIJ    *mat =  (Mat_MPIAIJ*)A->data;
    Mat_SeqAIJ    *aa=(Mat_SeqAIJ*)(mat->A)->data;
    Mat_SeqAIJ    *bb=(Mat_SeqAIJ*)(mat->B)->data;
    nz = aa->nz + bb->nz;
    ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= mat->rstart;
    garray = mat->garray;
    av=aa->a; bv=bb->a;
   
  } else {
    Mat_MPISBAIJ  *mat =  (Mat_MPISBAIJ*)A->data;
    Mat_SeqSBAIJ  *aa=(Mat_SeqSBAIJ*)(mat->A)->data;
    Mat_SeqBAIJ    *bb=(Mat_SeqBAIJ*)(mat->B)->data;
    if (A->bs > 1) SETERRQ1(PETSC_ERR_SUP," bs=%d is not supported yet\n", A->bs);
    nz = aa->nz + bb->nz;
    ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= mat->rstart;
    garray = mat->garray;
    av=aa->a; bv=bb->a;
  }

  if (!valOnly){ 
    ierr = PetscMalloc(nz*sizeof(PetscInt) ,&row);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscInt),&col);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&val);CHKERRQ(ierr);
    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v; 
  }
  *nnz = nz; 

  jj = 0; irow = rstart;   
  for ( i=0; i<m; i++ ) {
    ajj = aj + ai[i];                 /* ptr to the beginning of this row */      
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj = bj + bi[i];  

    /* get jB, the starting local col index for the 2nd B-part */
    colA_start = rstart + ajj[0]; /* the smallest col index for A */                      
    j=-1;
    do {
      j++;
      if (j == countB) break;
      jcol = garray[bjj[j]];
    } while (jcol < colA_start);
    jB = j;
  
    /* B-part, smaller col index */   
    colA_start = rstart + ajj[0]; /* the smallest col index for A */  
    for (j=0; j<jB; j++){
      jcol = garray[bjj[j]];
      if (!valOnly){ 
        row[jj] = irow + shift; col[jj] = jcol + shift; 

      }
      val[jj++] = *bv++;
    }
    /* A-part */
    for (j=0; j<countA; j++){
      if (!valOnly){
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift; 
      }
      val[jj++] = *av++;
    }
    /* B-part, larger col index */      
    for (j=jB; j<countB; j++){
      if (!valOnly){
        row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      }
      val[jj++] = *bv++;
    }
    irow++;
  } 
  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MUMPS_Base"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MUMPS_Base(Mat A,MatType type,MatReuse reuse,Mat *newmat) \
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_MUMPS      *mumps=(Mat_MUMPS*)A->spptr;
  void           (*f)(void);

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  B->ops->duplicate              = mumps->MatDuplicate;
  B->ops->view                   = mumps->MatView;
  B->ops->assemblyend            = mumps->MatAssemblyEnd;
  B->ops->lufactorsymbolic       = mumps->MatLUFactorSymbolic;
  B->ops->choleskyfactorsymbolic = mumps->MatCholeskyFactorSymbolic;
  B->ops->destroy                = mumps->MatDestroy;

  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C","",(FCNVOID)mumps->MatPreallocate);CHKERRQ(ierr);
  }
  ierr = PetscFree(mumps);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_aijmumps_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_aijmumps_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpiaij_aijmumps_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_aijmumps_mpiaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaij_sbaijmumps_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_sbaijmumps_seqsbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_sbaijmumps_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_sbaijmumps_mpisbaij_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MUMPS"
PetscErrorCode MatDestroy_MUMPS(Mat A)
{
  Mat_MUMPS *lu=(Mat_MUMPS*)A->spptr; 
  PetscErrorCode ierr;
  int       size=lu->size;
  PetscErrorCode (*specialdestroy)(Mat);
  PetscFunctionBegin;
  if (lu->CleanUpMUMPS) {
    /* Terminate instance, deallocate memories */
    lu->id.job=JOB_END; 
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id); 
#else
    dmumps_c(&lu->id); 
#endif
    if (lu->irn) {
      ierr = PetscFree(lu->irn);CHKERRQ(ierr);
    }
    if (lu->jcn) { 
      ierr = PetscFree(lu->jcn);CHKERRQ(ierr);
    }
    if (size>1 && lu->val) {
      ierr = PetscFree(lu->val);CHKERRQ(ierr);
    }
    ierr = MPI_Comm_free(&(lu->comm_mumps));CHKERRQ(ierr);
  }
  specialdestroy = lu->specialdestroy;
  ierr = (*specialdestroy)(A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_AIJMUMPS"
PetscErrorCode MatDestroy_AIJMUMPS(Mat A) 
{
  PetscErrorCode ierr;
  int  size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size==1) {
    ierr = MatConvert_MUMPS_Base(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr = MatConvert_MUMPS_Base(A,MATMPIAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SBAIJMUMPS"
PetscErrorCode MatDestroy_SBAIJMUMPS(Mat A) 
{
  PetscErrorCode ierr;
  int  size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size==1) {
    ierr = MatConvert_MUMPS_Base(A,MATSEQSBAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr = MatConvert_MUMPS_Base(A,MATMPISBAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_MUMPS"
PetscErrorCode MatFactorInfo_MUMPS(Mat A,PetscViewer viewer) {
  Mat_MUMPS *lu=(Mat_MUMPS*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"MUMPS run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  SYM (matrix type):                  %d \n",lu->id.sym);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  PAR (host participation):           %d \n",lu->id.par);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(4) (level of printing):       %d \n",lu->id.ICNTL(4));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(5) (input mat struct):        %d \n",lu->id.ICNTL(5));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(6) (matrix prescaling):       %d \n",lu->id.ICNTL(6));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(7) (matrix ordering):         %d \n",lu->id.ICNTL(7));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(9) (A/A^T x=b is solved):     %d \n",lu->id.ICNTL(9));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(10) (max num of refinements): %d \n",lu->id.ICNTL(10));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(11) (error analysis):         %d \n",lu->id.ICNTL(11));CHKERRQ(ierr);  
  if (!lu->myid && lu->id.ICNTL(11)>0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(4) (inf norm of input mat):        %g\n",lu->id.RINFOG(4));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(5) (inf norm of solution):         %g\n",lu->id.RINFOG(5));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(6) (inf norm of residual):         %g\n",lu->id.RINFOG(6));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(7),RINFOG(8) (backward error est): %g, %g\n",lu->id.RINFOG(7),lu->id.RINFOG(8));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(9) (error estimate):               %g \n",lu->id.RINFOG(9));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n",lu->id.RINFOG(10),lu->id.RINFOG(11));CHKERRQ(ierr);
  
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(12) (efficiency control):                         %d \n",lu->id.ICNTL(12));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(13) (efficiency control):                         %d \n",lu->id.ICNTL(13));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(14) (percentage of estimated workspace increase): %d \n",lu->id.ICNTL(14));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(15) (efficiency control):                         %d \n",lu->id.ICNTL(15));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(18) (input mat struct):                           %d \n",lu->id.ICNTL(18));CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(1) (relative pivoting threshold):      %g \n",lu->id.CNTL(1));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(2) (stopping criterion of refinement): %g \n",lu->id.CNTL(2));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(3) (absolute pivoting threshold):      %g \n",lu->id.CNTL(3));CHKERRQ(ierr);

  /* infomation local to each processor */
  if (!lu->myid) ierr = PetscPrintf(PETSC_COMM_SELF, "      RINFO(1) (local estimated flops for the elimination after analysis): \n");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(A->comm,"             [%d] %g \n",lu->myid,lu->id.RINFO(1));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(A->comm);
  if (!lu->myid) ierr = PetscPrintf(PETSC_COMM_SELF, "      RINFO(2) (local estimated flops for the assembly after factorization): \n");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(A->comm,"             [%d]  %g \n",lu->myid,lu->id.RINFO(2));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(A->comm);
  if (!lu->myid) ierr = PetscPrintf(PETSC_COMM_SELF, "      RINFO(3) (local estimated flops for the elimination after factorization): \n");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(A->comm,"             [%d]  %g \n",lu->myid,lu->id.RINFO(3));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(A->comm);

  if (!lu->myid){ /* information from the host */
    ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(1) (global estimated flops for the elimination after analysis): %g \n",lu->id.RINFOG(1));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(2) (global estimated flops for the assembly after factorization): %g \n",lu->id.RINFOG(2));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(3) (global estimated flops for the elimination after factorization): %g \n",lu->id.RINFOG(3));CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d \n",lu->id.INFOG(3));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d \n",lu->id.INFOG(4));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(5) (estimated maximum front size in the complete tree): %d \n",lu->id.INFOG(5));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(6) (number of nodes in the complete tree): %d \n",lu->id.INFOG(6));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(7) (ordering option effectively uese after analysis): %d \n",lu->id.INFOG(7));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(8) (structural symmetry in percent of the permuted matrix after analysis): %d \n",lu->id.INFOG(8));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(9) (total real space store the matrix factors after analysis): %d \n",lu->id.INFOG(9));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(10) (total integer space store the matrix factors after analysis): %d \n",lu->id.INFOG(10));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(11) (order of largest frontal matrix): %d \n",lu->id.INFOG(11));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(12) (number of off-diagonal pivots): %d \n",lu->id.INFOG(12));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(13) (number of delayed pivots after factorization): %d \n",lu->id.INFOG(13));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(14) (number of memory compress after factorization): %d \n",lu->id.INFOG(14));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(15) (number of steps of iterative refinement after solution): %d \n",lu->id.INFOG(15));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(16) (estimated size (in million of bytes) of all MUMPS internal data for factorization after analysis: value on the most memory consuming processor): %d \n",lu->id.INFOG(16));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(17) (estimated size of all MUMPS internal data for factorization after analysis: sum over all processors): %d \n",lu->id.INFOG(17));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(18) (size of all MUMPS internal data allocated during factorization: value on the most memory consuming processor): %d \n",lu->id.INFOG(18));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(19) (size of all MUMPS internal data allocated during factorization: sum over all processors): %d \n",lu->id.INFOG(19));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(20) (estimated number of entries in the factors): %d \n",lu->id.INFOG(20));CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_AIJMUMPS"
PetscErrorCode MatView_AIJMUMPS(Mat A,PetscViewer viewer) {
  PetscErrorCode ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;
  Mat_MUMPS         *mumps=(Mat_MUMPS*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*mumps->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_MUMPS(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_AIJMUMPS"
PetscErrorCode MatSolve_AIJMUMPS(Mat A,Vec b,Vec x) {
  Mat_MUMPS   *lu=(Mat_MUMPS*)A->spptr; 
  PetscScalar *array;
  Vec         x_seq;
  IS          iden;
  VecScatter  scat;
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  if (lu->size > 1){
    if (!lu->myid){
      ierr = VecCreateSeq(PETSC_COMM_SELF,A->N,&x_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,A->N,0,1,&iden);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&x_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&iden);CHKERRQ(ierr);
    }
    ierr = VecScatterCreate(b,iden,x_seq,iden,&scat);CHKERRQ(ierr);
    ierr = ISDestroy(iden);CHKERRQ(ierr);

    ierr = VecScatterBegin(b,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(b,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    if (!lu->myid) {ierr = VecGetArray(x_seq,&array);CHKERRQ(ierr);}
  } else {  /* size == 1 */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  }
  if (!lu->myid) { /* define rhs on the host */
#if defined(PETSC_USE_COMPLEX)
    lu->id.rhs = (mumps_double_complex*)array;
#else
    lu->id.rhs = array;
#endif
  }

  /* solve phase */
  lu->id.job=3;
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) {   
    SETERRQ1(PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",lu->id.INFOG(1));
  }

  /* convert mumps solution x_seq to petsc mpi x */
  if (lu->size > 1) {
    if (!lu->myid){
      ierr = VecRestoreArray(x_seq,&array);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat);CHKERRQ(ierr);
    ierr = VecDestroy(x_seq);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  } 
   
  PetscFunctionReturn(0);
}

/* 
  input:
   F:        numeric factor
  output:
   nneg:     total number of negative pivots
   nzero:    0
   npos:     (global dimension of F) - nneg
*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia_SBAIJMUMPS"
PetscErrorCode MatGetInertia_SBAIJMUMPS(Mat F,int *nneg,int *nzero,int *npos)
{ 
  Mat_MUMPS  *lu =(Mat_MUMPS*)F->spptr; 
  PetscErrorCode ierr;
  int        size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(F->comm,&size);CHKERRQ(ierr);
  /* MUMPS 4.3.1 calls ScaLAPACK when ICNTL(13)=0 (default), which does not offer the possibility to compute the inertia of a dense matrix. Set ICNTL(13)=1 to skip ScaLAPACK */
  if (size > 1 && lu->id.ICNTL(13) != 1){
    SETERRQ1(PETSC_ERR_ARG_WRONG,"ICNTL(13)=%d. -mat_mumps_icntl_13 must be set as 1 for correct global matrix inertia\n",lu->id.INFOG(13));
  }
  if (nneg){  
    if (!lu->myid){
      *nneg = lu->id.INFOG(12);
    } 
    ierr = MPI_Bcast(nneg,1,MPI_INT,0,lu->comm_mumps);CHKERRQ(ierr);
  }
  if (nzero) *nzero = 0;  
  if (npos)  *npos  = F->M - (*nneg);
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatFactorNumeric_AIJMUMPS"
PetscErrorCode MatFactorNumeric_AIJMUMPS(Mat A,MatFactorInfo *info,Mat *F) 
{
  Mat_MUMPS      *lu =(Mat_MUMPS*)(*F)->spptr; 
  Mat_MUMPS      *lua=(Mat_MUMPS*)(A)->spptr; 
  PetscErrorCode ierr;
  PetscInt       rnz,nnz,nz,i,M=A->M,*ai,*aj,icntl;
  PetscTruth     valOnly,flg;
  Mat            F_diag;

  PetscFunctionBegin; 	
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ 
    (*F)->ops->solve    = MatSolve_AIJMUMPS;

    /* Initialize a MUMPS instance */
    ierr = MPI_Comm_rank(A->comm, &lu->myid);
    ierr = MPI_Comm_size(A->comm,&lu->size);CHKERRQ(ierr);
    lua->myid = lu->myid; lua->size = lu->size;
    lu->id.job = JOB_INIT; 
    ierr = MPI_Comm_dup(A->comm,&(lu->comm_mumps));CHKERRQ(ierr);
    lu->id.comm_fortran = lu->comm_mumps;

    /* Set mumps options */
    ierr = PetscOptionsBegin(A->comm,A->prefix,"MUMPS Options","Mat");CHKERRQ(ierr);
    lu->id.par=1;  /* host participates factorizaton and solve */
    lu->id.sym=lu->sym; 
    if (lu->sym == 2){
      ierr = PetscOptionsInt("-mat_mumps_sym","SYM: (1,2)","None",lu->id.sym,&icntl,&flg);CHKERRQ(ierr); 
      if (flg && icntl == 1) lu->id.sym=icntl;  /* matrix is spd */
    }
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id); 
#else
    dmumps_c(&lu->id); 
#endif
 
    if (lu->size == 1){
      lu->id.ICNTL(18) = 0;   /* centralized assembled matrix input */
    } else {
      lu->id.ICNTL(18) = 3;   /* distributed assembled matrix input */
    }

    icntl=-1;
    ierr = PetscOptionsInt("-mat_mumps_icntl_4","ICNTL(4): level of printing (0 to 4)","None",lu->id.ICNTL(4),&icntl,&flg);CHKERRQ(ierr);
    if ((flg && icntl > 0) || PetscLogPrintInfo) {
      lu->id.ICNTL(4)=icntl; /* and use mumps default icntl(i), i=1,2,3 */
    } else { /* no output */
      lu->id.ICNTL(1) = 0;  /* error message, default= 6 */
      lu->id.ICNTL(2) = -1; /* output stream for diagnostic printing, statistics, and warning. default=0 */
      lu->id.ICNTL(3) = -1; /* output stream for global information, default=6 */
      lu->id.ICNTL(4) = 0;  /* level of printing, 0,1,2,3,4, default=2 */
    }
    ierr = PetscOptionsInt("-mat_mumps_icntl_6","ICNTL(6): matrix prescaling (0 to 7)","None",lu->id.ICNTL(6),&lu->id.ICNTL(6),PETSC_NULL);CHKERRQ(ierr);
    icntl=-1;
    ierr = PetscOptionsInt("-mat_mumps_icntl_7","ICNTL(7): matrix ordering (0 to 7)","None",lu->id.ICNTL(7),&icntl,&flg);CHKERRQ(ierr);
    if (flg) {
      if (icntl== 1){
        SETERRQ(PETSC_ERR_SUP,"pivot order be set by the user in PERM_IN -- not supported by the PETSc/MUMPS interface\n");
      } else {
        lu->id.ICNTL(7) = icntl;
      }
    } 
    ierr = PetscOptionsInt("-mat_mumps_icntl_9","ICNTL(9): A or A^T x=b to be solved. 1: A; otherwise: A^T","None",lu->id.ICNTL(9),&lu->id.ICNTL(9),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_10","ICNTL(10): max num of refinements","None",lu->id.ICNTL(10),&lu->id.ICNTL(10),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_11","ICNTL(11): error analysis, a positive value returns statistics (by -ksp_view)","None",lu->id.ICNTL(11),&lu->id.ICNTL(11),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_12","ICNTL(12): efficiency control","None",lu->id.ICNTL(12),&lu->id.ICNTL(12),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_13","ICNTL(13): efficiency control","None",lu->id.ICNTL(13),&lu->id.ICNTL(13),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_14","ICNTL(14): percentage of estimated workspace increase","None",lu->id.ICNTL(14),&lu->id.ICNTL(14),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_15","ICNTL(15): efficiency control","None",lu->id.ICNTL(15),&lu->id.ICNTL(15),PETSC_NULL);CHKERRQ(ierr);

    /* 
    ierr = PetscOptionsInt("-mat_mumps_icntl_16","ICNTL(16): 1: rank detection; 2: rank detection and nullspace","None",lu->id.ICNTL(16),&icntl,&flg);CHKERRQ(ierr);
    if (flg){
      if (icntl >-1 && icntl <3 ){
        if (lu->myid==0) lu->id.ICNTL(16) = icntl;
      } else {
        SETERRQ1(PETSC_ERR_SUP,"ICNTL(16)=%d -- not supported\n",icntl);
      }
    }
    */

    ierr = PetscOptionsReal("-mat_mumps_cntl_1","CNTL(1): relative pivoting threshold","None",lu->id.CNTL(1),&lu->id.CNTL(1),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_mumps_cntl_2","CNTL(2): stopping criterion of refinement","None",lu->id.CNTL(2),&lu->id.CNTL(2),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_mumps_cntl_3","CNTL(3): absolute pivoting threshold","None",lu->id.CNTL(3),&lu->id.CNTL(3),PETSC_NULL);CHKERRQ(ierr);
    PetscOptionsEnd();
  }

  /* define matrix A */
  switch (lu->id.ICNTL(18)){
  case 0:  /* centralized assembled matrix input (size=1) */
    if (!lu->myid) {
      if (lua->isAIJ){
        Mat_SeqAIJ   *aa = (Mat_SeqAIJ*)A->data;
        nz               = aa->nz;
        ai = aa->i; aj = aa->j; lu->val = aa->a;
      } else {
        Mat_SeqSBAIJ *aa = (Mat_SeqSBAIJ*)A->data;
        nz                  =  aa->nz;
        ai = aa->i; aj = aa->j; lu->val = aa->a;
      }
      if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization, get irn and jcn */
        ierr = PetscMalloc(nz*sizeof(PetscInt),&lu->irn);CHKERRQ(ierr);
        ierr = PetscMalloc(nz*sizeof(PetscInt),&lu->jcn);CHKERRQ(ierr); 
        nz = 0;
        for (i=0; i<M; i++){
          rnz = ai[i+1] - ai[i];
          while (rnz--) {  /* Fortran row/col index! */
            lu->irn[nz] = i+1; lu->jcn[nz] = (*aj)+1; aj++; nz++; 
          }
        }
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){
      valOnly = PETSC_FALSE; 
    } else {
      valOnly = PETSC_TRUE; /* only update mat values, not row and col index */
    }
    ierr = MatConvertToTriples(A,1,valOnly, &nnz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);
    break;
  default: SETERRQ(PETSC_ERR_SUP,"Matrix input format is not supported by MUMPS.");
  }

  /* analysis phase */
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ 
     lu->id.n = M;
    switch (lu->id.ICNTL(18)){
    case 0:  /* centralized assembled matrix input */
      if (!lu->myid) {
        lu->id.nz =nz; lu->id.irn=lu->irn; lu->id.jcn=lu->jcn;
        if (lu->id.ICNTL(6)>1){
#if defined(PETSC_USE_COMPLEX)
          lu->id.a = (mumps_double_complex*)lu->val; 
#else
          lu->id.a = lu->val; 
#endif
        }
      }
      break;
    case 3:  /* distributed assembled matrix input (size>1) */ 
      lu->id.nz_loc = nnz; 
      lu->id.irn_loc=lu->irn; lu->id.jcn_loc=lu->jcn;
      if (lu->id.ICNTL(6)>1) {
#if defined(PETSC_USE_COMPLEX)
        lu->id.a_loc = (mumps_double_complex*)lu->val;
#else
        lu->id.a_loc = lu->val;
#endif
      }
      break;
    }    
    lu->id.job=1;
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id); 
#else
    dmumps_c(&lu->id); 
#endif
    if (lu->id.INFOG(1) < 0) { 
      SETERRQ1(PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 
    }
  }

  /* numerical factorization phase */
  if(!lu->id.ICNTL(18)) { 
    if (!lu->myid) {
#if defined(PETSC_USE_COMPLEX)
      lu->id.a = (mumps_double_complex*)lu->val; 
#else
      lu->id.a = lu->val; 
#endif
    }
  } else {
#if defined(PETSC_USE_COMPLEX)
    lu->id.a_loc = (mumps_double_complex*)lu->val; 
#else
    lu->id.a_loc = lu->val; 
#endif
  }
  lu->id.job=2;
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) {
    if (lu->id.INFO(1) == -13) {
      SETERRQ1(PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: Cannot allocate required memory %d megabytes\n",lu->id.INFO(2)); 
    } else {
      SETERRQ2(PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: INFO(1)=%d, INFO(2)=%d\n",lu->id.INFO(1),lu->id.INFO(2)); 
    }
  }

  if (!lu->myid && lu->id.ICNTL(16) > 0){
    SETERRQ1(PETSC_ERR_LIB,"  lu->id.ICNTL(16):=%d\n",lu->id.INFOG(16)); 
  }

  if ((*F)->factor == FACTOR_LU){
    F_diag = ((Mat_MPIAIJ *)(*F)->data)->A;
  } else {
    F_diag = ((Mat_MPISBAIJ *)(*F)->data)->A;
  }
  F_diag->assembled = PETSC_TRUE; 
  (*F)->assembled   = PETSC_TRUE;
  lu->matstruc      = SAME_NONZERO_PATTERN;
  lu->CleanUpMUMPS  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_AIJMUMPS"
PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F) {
  Mat            B;
  Mat_MUMPS      *lu;   
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Create the factorization matrix */
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->m,A->n,A->M,A->N);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric = MatFactorNumeric_AIJMUMPS;
  B->factor               = FACTOR_LU;  
  lu                      = (Mat_MUMPS*)B->spptr;
  lu->sym                 = 0;
  lu->matstruc            = DIFFERENT_NONZERO_PATTERN;

  *F = B;
  PetscFunctionReturn(0); 
}

/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SBAIJMUMPS"
PetscErrorCode MatCholeskyFactorSymbolic_SBAIJMUMPS(Mat A,IS r,MatFactorInfo *info,Mat *F) {
  Mat            B;
  Mat_MUMPS      *lu;   
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create the factorization matrix */ 
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->m,A->n,A->M,A->N);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactornumeric = MatFactorNumeric_AIJMUMPS;
  B->ops->getinertia            = MatGetInertia_SBAIJMUMPS;
  B->factor                     = FACTOR_CHOLESKY;
  lu                            = (Mat_MUMPS*)B->spptr;
  lu->sym                       = 2;
  lu->matstruc                  = DIFFERENT_NONZERO_PATTERN;

  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_AIJMUMPS"
PetscErrorCode MatAssemblyEnd_AIJMUMPS(Mat A,MatAssemblyType mode) {
  PetscErrorCode ierr;
  Mat_MUMPS *mumps=(Mat_MUMPS*)A->spptr;

  PetscFunctionBegin;
  ierr = (*mumps->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);

  mumps->MatLUFactorSymbolic       = A->ops->lufactorsymbolic;
  mumps->MatCholeskyFactorSymbolic = A->ops->choleskyfactorsymbolic;
  A->ops->lufactorsymbolic         = MatLUFactorSymbolic_AIJMUMPS;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_AIJ_AIJMUMPS"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_AIJ_AIJMUMPS(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  MPI_Comm       comm;
  Mat            B=*newmat;
  Mat_MUMPS      *mumps;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscNew(Mat_MUMPS,&mumps);CHKERRQ(ierr);

  mumps->MatDuplicate              = A->ops->duplicate;
  mumps->MatView                   = A->ops->view;
  mumps->MatAssemblyEnd            = A->ops->assemblyend;
  mumps->MatLUFactorSymbolic       = A->ops->lufactorsymbolic;
  mumps->MatCholeskyFactorSymbolic = A->ops->choleskyfactorsymbolic;
  mumps->MatDestroy                = A->ops->destroy;
  mumps->specialdestroy            = MatDestroy_AIJMUMPS;
  mumps->CleanUpMUMPS              = PETSC_FALSE;
  mumps->isAIJ                     = PETSC_TRUE;

  B->spptr                         = (void*)mumps;
  B->ops->duplicate                = MatDuplicate_MUMPS;
  B->ops->view                     = MatView_AIJMUMPS;
  B->ops->assemblyend              = MatAssemblyEnd_AIJMUMPS;
  B->ops->lufactorsymbolic         = MatLUFactorSymbolic_AIJMUMPS;
  B->ops->destroy                  = MatDestroy_MUMPS;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_aijmumps_C",
                                             "MatConvert_AIJ_AIJMUMPS",MatConvert_AIJ_AIJMUMPS);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_aijmumps_seqaij_C",
                                             "MatConvert_MUMPS_Base",MatConvert_MUMPS_Base);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpiaij_aijmumps_C",
                                             "MatConvert_AIJ_AIJMUMPS",MatConvert_AIJ_AIJMUMPS);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_aijmumps_mpiaij_C",
                                             "MatConvert_MUMPS_Base",MatConvert_MUMPS_Base);CHKERRQ(ierr);
  }

  ierr = PetscLogInfo((0,"MatConvert_AIJ_AIJMUMPS:Using MUMPS for LU factorization and solves.\n"));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,newtype);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATAIJMUMPS - MATAIJMUMPS = "aijmumps" - A matrix type providing direct solvers (LU) for distributed
  and sequential matrices via the external package MUMPS.

  If MUMPS is installed (see the manual for instructions
  on how to declare the existence of external packages),
  a matrix type can be constructed which invokes MUMPS solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATAIJMUMPS).

  If created with a single process communicator, this matrix type inherits from MATSEQAIJ.
  Otherwise, this matrix type inherits from MATMPIAIJ.  Hence for single process communicators,
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.  One can also call MatConvert for an inplace
  conversion to or from the MATSEQAIJ or MATMPIAIJ type (depending on the communicator size)
  without data copy.

  Options Database Keys:
+ -mat_type aijmumps - sets the matrix type to "aijmumps" during a call to MatSetFromOptions()
. -mat_mumps_sym <0,1,2> - 0 the matrix is unsymmetric, 1 symmetric positive definite, 2 symmetric
. -mat_mumps_icntl_4 <0,1,2,3,4> - print level
. -mat_mumps_icntl_6 <0,...,7> - matrix prescaling options (see MUMPS User's Guide)
. -mat_mumps_icntl_7 <0,...,7> - matrix orderings (see MUMPS User's Guide)
. -mat_mumps_icntl_9 <1,2> - A or A^T x=b to be solved: 1 denotes A, 2 denotes A^T
. -mat_mumps_icntl_10 <n> - maximum number of iterative refinements
. -mat_mumps_icntl_11 <n> - error analysis, a positive value returns statistics during -ksp_view
. -mat_mumps_icntl_12 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_13 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_14 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_15 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_cntl_1 <delta> - relative pivoting threshold
. -mat_mumps_cntl_2 <tol> - stopping criterion for refinement
- -mat_mumps_cntl_3 <adelta> - absolute pivoting threshold

  Level: beginner

.seealso: MATSBAIJMUMPS
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJMUMPS"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJMUMPS(Mat A) 
{
  PetscErrorCode ierr;
  int      size;
  Mat      A_diag;
  MPI_Comm comm;
  
  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ or MPIAIJ */
  /*   and AIJMUMPS types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATAIJMUMPS);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  } else {
    ierr   = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    A_diag = ((Mat_MPIAIJ *)A->data)->A;
    ierr   = MatConvert_AIJ_AIJMUMPS(A_diag,MATAIJMUMPS,MAT_REUSE_MATRIX,&A_diag);CHKERRQ(ierr);
  }
  ierr = MatConvert_AIJ_AIJMUMPS(A,MATAIJMUMPS,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SBAIJMUMPS"
PetscErrorCode MatAssemblyEnd_SBAIJMUMPS(Mat A,MatAssemblyType mode) 
{
  PetscErrorCode ierr;
  Mat_MUMPS *mumps=(Mat_MUMPS*)A->spptr;

  PetscFunctionBegin;
  ierr = (*mumps->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  mumps->MatLUFactorSymbolic       = A->ops->lufactorsymbolic;
  mumps->MatCholeskyFactorSymbolic = A->ops->choleskyfactorsymbolic;
  A->ops->choleskyfactorsymbolic   = MatCholeskyFactorSymbolic_SBAIJMUMPS;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPISBAIJSetPreallocation_MPISBAIJMUMPS"
PetscErrorCode PETSCMAT_DLLEXPORT MatMPISBAIJSetPreallocation_MPISBAIJMUMPS(Mat  B,int bs,int d_nz,int *d_nnz,int o_nz,int *o_nnz)
{
  Mat       A;
  Mat_MUMPS *mumps=(Mat_MUMPS*)B->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
    After performing the MPISBAIJ Preallocation, we need to convert the local diagonal block matrix
    into MUMPS type so that the block jacobi preconditioner (for example) can use MUMPS.  I would
    like this to be done in the MatCreate routine, but the creation of this inner matrix requires
    block size info so that PETSc can determine the local size properly.  The block size info is set
    in the preallocation routine.
  */
  ierr = (*mumps->MatPreallocate)(B,bs,d_nz,d_nnz,o_nz,o_nnz);
  A    = ((Mat_MPISBAIJ *)B->data)->A;
  ierr = MatConvert_SBAIJ_SBAIJMUMPS(A,MATSBAIJMUMPS,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SBAIJ_SBAIJMUMPS"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SBAIJ_SBAIJMUMPS(Mat A,MatType newtype,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  MPI_Comm       comm;
  Mat            B=*newmat;
  Mat_MUMPS      *mumps;  
  void           (*f)(void);

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr); 
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscNew(Mat_MUMPS,&mumps);CHKERRQ(ierr);

  mumps->MatDuplicate              = A->ops->duplicate;
  mumps->MatView                   = A->ops->view;
  mumps->MatAssemblyEnd            = A->ops->assemblyend;
  mumps->MatLUFactorSymbolic       = A->ops->lufactorsymbolic;
  mumps->MatCholeskyFactorSymbolic = A->ops->choleskyfactorsymbolic;
  mumps->MatDestroy                = A->ops->destroy;
  mumps->specialdestroy            = MatDestroy_SBAIJMUMPS;
  mumps->CleanUpMUMPS              = PETSC_FALSE;
  mumps->isAIJ                     = PETSC_FALSE;
  
  B->spptr                         = (void*)mumps;
  B->ops->duplicate                = MatDuplicate_MUMPS;
  B->ops->view                     = MatView_AIJMUMPS;
  B->ops->assemblyend              = MatAssemblyEnd_SBAIJMUMPS;
  B->ops->choleskyfactorsymbolic   = MatCholeskyFactorSymbolic_SBAIJMUMPS;
  B->ops->destroy                  = MatDestroy_MUMPS;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqsbaij_sbaijmumps_C",
                                             "MatConvert_SBAIJ_SBAIJMUMPS",MatConvert_SBAIJ_SBAIJMUMPS);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_sbaijmumps_seqsbaij_C",
                                             "MatConvert_MUMPS_Base",MatConvert_MUMPS_Base);CHKERRQ(ierr);
  } else {
  /* I really don't like needing to know the tag: MatMPISBAIJSetPreallocation_C */
    ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);
    if (f) { /* This case should always be true when this routine is called */
      mumps->MatPreallocate = (PetscErrorCode (*)(Mat,int,int,int*,int,int*))f;
      ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPISBAIJSetPreallocation_C",
                                               "MatMPISBAIJSetPreallocation_MPISBAIJMUMPS",
                                               MatMPISBAIJSetPreallocation_MPISBAIJMUMPS);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpisbaij_sbaijmumps_C",
                                             "MatConvert_SBAIJ_SBAIJMUMPS",MatConvert_SBAIJ_SBAIJMUMPS);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_sbaijmumps_mpisbaij_C",
                                             "MatConvert_MUMPS_Base",MatConvert_MUMPS_Base);CHKERRQ(ierr);
  }

  ierr = PetscLogInfo((0,"MatConvert_AIJ_AIJMUMPS:Using MUMPS for Cholesky factorization and solves.\n"));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,newtype);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_MUMPS"
PetscErrorCode MatDuplicate_MUMPS(Mat A, MatDuplicateOption op, Mat *M) {
  PetscErrorCode ierr;
  Mat_MUMPS   *lu=(Mat_MUMPS *)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr); 
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_MUMPS));CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

/*MC
  MATSBAIJMUMPS - MATSBAIJMUMPS = "sbaijmumps" - A symmetric matrix type providing direct solvers (Cholesky) for
  distributed and sequential matrices via the external package MUMPS.

  If MUMPS is installed (see the manual for instructions
  on how to declare the existence of external packages),
  a matrix type can be constructed which invokes MUMPS solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSBAIJMUMPS).

  If created with a single process communicator, this matrix type inherits from MATSEQSBAIJ.
  Otherwise, this matrix type inherits from MATMPISBAIJ.  Hence for single process communicators,
  MatSeqSBAIJSetPreallocation is supported, and similarly MatMPISBAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.  One can also call MatConvert for an inplace
  conversion to or from the MATSEQSBAIJ or MATMPISBAIJ type (depending on the communicator size)
  without data copy.

  Options Database Keys:
+ -mat_type sbaijmumps - sets the matrix type to "sbaijmumps" during a call to MatSetFromOptions()
. -mat_mumps_sym <0,1,2> - 0 the matrix is unsymmetric, 1 symmetric positive definite, 2 symmetric
. -mat_mumps_icntl_4 <0,...,4> - print level
. -mat_mumps_icntl_6 <0,...,7> - matrix prescaling options (see MUMPS User's Guide)
. -mat_mumps_icntl_7 <0,...,7> - matrix orderings (see MUMPS User's Guide)
. -mat_mumps_icntl_9 <1,2> - A or A^T x=b to be solved: 1 denotes A, 2 denotes A^T
. -mat_mumps_icntl_10 <n> - maximum number of iterative refinements
. -mat_mumps_icntl_11 <n> - error analysis, a positive value returns statistics during -ksp_view
. -mat_mumps_icntl_12 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_13 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_14 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_15 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_cntl_1 <delta> - relative pivoting threshold
. -mat_mumps_cntl_2 <tol> - stopping criterion for refinement
- -mat_mumps_cntl_3 <adelta> - absolute pivoting threshold

  Level: beginner

.seealso: MATAIJMUMPS
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SBAIJMUMPS"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SBAIJMUMPS(Mat A) 
{
  PetscErrorCode ierr;
  int size;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqSBAIJ or MPISBAIJ */
  /*   and SBAIJMUMPS types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSBAIJMUMPS);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
  } else {
    ierr   = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
  }
  ierr = MatConvert_SBAIJ_SBAIJMUMPS(A,MATSBAIJMUMPS,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
