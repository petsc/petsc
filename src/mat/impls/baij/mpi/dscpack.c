/*$Id: dscpack.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
        Provides an interface to the DSCPACK (Domain-Separator Codes) sparse direct solver
*/

#include "src/mat/impls/baij/seq/baij.h"
#include "src/mat/impls/baij/mpi/mpibaij.h"

#if defined(PETSC_HAVE_DSCPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)

EXTERN_C_BEGIN
#include "/sandbox/hzhang/DSCPACK1.0/DSC_LIB/dscmain.h"
EXTERN_C_END

#define MAX_MEM_ALLOWED  100
typedef struct {
  DSC_Solver	My_DSC_Solver;  
  int	        *replication;
  int           num_local_strucs, *local_struc_old_num,
                num_local_cols, num_local_nonz, 
                *global_struc_new_col_num,
                *global_struc_new_num, *global_struc_owner;               
  int           order_code,scheme_code,factor_type,
                LBLASLevel, DBLASLevel;             /* runtime options */
  int           rank,bs,*local_cols_old_num; 
  MatStructure  flg;
  IS            my_cols,iden;
} Mat_MPIBAIJ_DSC;

extern int MatDestroy_MPIBAIJ(Mat);
extern int MatDestroy_SeqBAIJ(Mat);

/* DSC function */
#undef __FUNCT__  
#define __FUNCT__ "isort2"
void isort2(int size, int *list, int *index)
{
                /* in increasing order */
                /* index will contain indices such that */
                /* list can be accessed in sorted order */
   int i, j, x, y;

   for (i=0; i<size; i++) index[i] =i;

   for (i=1; i<size; i++){
      y= index[i];
      x=list[index[i]];
      for (j=i-1; ((j>=0) && (x<list[index[j]])); j--)
                index[j+1]=index[j];
      index[j+1]=y;
   }
}/*end isort2*/

#undef __FUNCT__  
#define __FUNCT__ "BAIJtoMyANonz"
int  BAIJtoMyANonz( int *AIndex, int *AStruct, int bs,
		    RealNumberType *ANonz, int NumLocalStructs, 
                    int NumLocalNonz,  int *GlobalStructNewColNum,                
		    int *LocalStructOldNum,
                    int *LocalStructLocalNum,
		    RealNumberType **adr_MyANonz)
/* 
   Extract non-zero values of lower triangular part
   of the permuted matrix that belong to this processor.

   Only output parameter is adr_MyANonz -- is malloced and changed.
   Rest are input parameters left unchanged.

   When LocalStructLocalNum == PETSC_NULL,
        AIndex, AStruct, and ANonz contain entire original matrix A 
        in PETSc SeqBAIJ format,
        otherwise,
        AIndex, AStruct, and ANonz are indeces for the submatrix
        of A whose colomns (in increasing order) belong to this processor.

   Other variables supply information on ownership of columns
   and the new numbering in a fill-reducing permutation

   This information is used to setup lower half of A nonzeroes
   for columns owned by this processor
 */ 
{  
  int            i, j, k, iold,inew, jj, kk,ierr, bs2=bs*bs,
                 *idx, *NewColNum,
                 MyANonz_last, max_struct=0, struct_size;
  RealNumberType *MyANonz;             

  PetscFunctionBegin;

  /* loop: to find maximum number of subscripts over columns
     assigned to this processor */
  for (i=0; i <NumLocalStructs; i++) {
    /* for each struct i (local) assigned to this processor */
    if (LocalStructLocalNum){
      iold = LocalStructLocalNum[i];
    } else {
      iold = LocalStructOldNum[i];
    }
    
    struct_size = AIndex[iold+1] - AIndex[iold];
    if ( max_struct <= struct_size) max_struct = struct_size; 
  }

  /* allocate tmp arrays large enough to hold densest struct */
  ierr = PetscMalloc((2*max_struct+1)*sizeof(int),&NewColNum);CHKERRQ(ierr);
  idx = NewColNum + max_struct;
  
  ierr = PetscMalloc(NumLocalNonz*sizeof(RealNumberType),&MyANonz);CHKERRQ(ierr);  
  *adr_MyANonz = MyANonz;

  /* loop to set up nonzeroes in MyANonz */  
  MyANonz_last = 0 ; /* points to first empty space in MyANonz */
  for (i=0; i <NumLocalStructs; i++) {

    /* for each struct i (local) assigned to this processor */		
    if (LocalStructLocalNum){
      iold = LocalStructLocalNum[i];
    } else {
      iold = LocalStructOldNum[i];
    }

    struct_size = AIndex[iold+1] - AIndex[iold];    
    for (k=0, j=AIndex[iold]; j<AIndex[iold+1]; j++){
      NewColNum[k] = GlobalStructNewColNum[AStruct[j]];
      k++;
    }
    isort2(struct_size, NewColNum, idx);
                      
    kk = AIndex[iold]*bs2; /* points to 1st element of iold block col in ANonz */  
    inew = GlobalStructNewColNum[LocalStructOldNum[i]];

    for (jj = 0; jj < bs; jj++) {
      for (j=0; j<struct_size; j++){
        for ( k = 0; k<bs; k++){      
          if (NewColNum[idx[j]] + k >= inew)
            MyANonz[MyANonz_last++] = ANonz[kk + idx[j]*bs2 + k*bs + jj];
        }
      }
      inew++;
    }
  } /* end outer loop for i */

  ierr = PetscFree(NewColNum); 
  if (MyANonz_last != NumLocalNonz) 
    SETERRQ2(1,"MyANonz_last %d != NumLocalNonz %d\n",MyANonz_last, NumLocalNonz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIBAIJ_DSCPACK"
int MatDestroy_MPIBAIJ_DSCPACK(Mat A)
{
  Mat_MPIBAIJ         *a; 
  Mat_SeqBAIJ         *a_seq; 
  Mat_MPIBAIJ_DSC     *lu;  
  int                 ierr, size;
    
  PetscFunctionBegin;
  /* printf(" Destroy is called ...\n"); */
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
 
  if ( size == 1){
    a_seq = (Mat_SeqBAIJ*)(A)->data;  
    lu = (Mat_MPIBAIJ_DSC*)a_seq->spptr;   
  } else {
    a  = (Mat_MPIBAIJ*)(A)->data;  
    lu = (Mat_MPIBAIJ_DSC*)a->spptr; 
  }

  if (lu->rank == -1) {
      PetscPrintf(PETSC_COMM_SELF," lu->rank = -1\n");
      PetscFunctionReturn(0);
  }

  /* Deallocate DSCPACK storage */   
  DSC_DoStats(lu->My_DSC_Solver);    
  DSC_FreeAll(lu->My_DSC_Solver);           
  DSC_Close0(lu->My_DSC_Solver);
  DSC_End(lu->My_DSC_Solver); 
    
  ierr = PetscFree(lu->local_cols_old_num);CHKERRQ(ierr);
  ierr = PetscFree(lu->replication);CHKERRQ(ierr);
  ierr = ISDestroy(lu->my_cols);CHKERRQ(ierr);
  if (size >1) ierr = ISDestroy(lu->iden);CHKERRQ(ierr);
  ierr = PetscFree(lu);CHKERRQ(ierr);  

  if (size == 1){
    ierr = MatDestroy_SeqBAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIBAIJ(A);CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIBAIJ_DSCPACK"
int MatSolve_MPIBAIJ_DSCPACK(Mat A,Vec b_mpi,Vec x)
{
  Mat_MPIBAIJ       *a;
  Mat_SeqBAIJ       *a_seq;
  Mat_MPIBAIJ_DSC   *lu;
  int               ierr, size;
  RealNumberType    *solution_vec, *rhs_vec; 
  PetscScalar       *array;
  Vec               vec_dsc;
  IS                iden;
  VecScatter        scat;
  int               bs,i;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
 
  if ( size == 1){
    a_seq = (Mat_SeqBAIJ*)(A)->data;  
    lu = (Mat_MPIBAIJ_DSC*)a_seq->spptr;    
    bs = a_seq->bs;
  } else {
    a  = (Mat_MPIBAIJ*)(A)->data;  
    lu = (Mat_MPIBAIJ_DSC*)a->spptr;  
    bs = a->bs;
  }
  printf(" Solve is called ..., bs: %d\n", bs);
  if (lu->rank == -1) {
      PetscPrintf(PETSC_COMM_SELF," lu->rank = -1\n");
      PetscFunctionReturn(0);
  }

  /* create seq vec_dsc */
  ierr = VecCreateSeq(PETSC_COMM_SELF,lu->num_local_cols,&vec_dsc);CHKERRQ(ierr);

  if (size == 1 ){
    ierr = VecGetArray(vec_dsc,&rhs_vec);CHKERRQ(ierr);  
    ierr = VecGetArray(b_mpi,&array);CHKERRQ(ierr);
    for (i=0; i<lu->num_local_cols; i++) 
      rhs_vec[i] = array[lu->local_cols_old_num[i]];

    ierr = VecRestoreArray(b_mpi,&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec_dsc,&rhs_vec);CHKERRQ(ierr); 
  } else {
    ierr = ISCreateStride(PETSC_COMM_SELF,lu->num_local_cols,0,1,&iden);CHKERRQ(ierr); /*is_dsc*/
    
    /* scatter b_mpi into seq vec_dsc */
    ierr = VecScatterCreate(b_mpi,lu->my_cols,vec_dsc,iden,&scat);CHKERRQ(ierr);
    ierr = VecScatterBegin(b_mpi,vec_dsc,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(b_mpi,vec_dsc,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
  }

  ierr = VecGetArray(vec_dsc,&rhs_vec);CHKERRQ(ierr);    
  DSC_InputRhsLocalVec(lu->My_DSC_Solver, rhs_vec, lu->num_local_cols);
  ierr = VecRestoreArray(vec_dsc,&rhs_vec);CHKERRQ(ierr); 
 
  ierr = DSC_Solve(lu->My_DSC_Solver);
  if (ierr !=  DSC_NO_ERROR) {
    DSC_ErrorDisplay(lu->My_DSC_Solver);
    SETERRQ(1,"Error in calling DSC_Solve");
  }

  /* get the permuted local solution */
  ierr = VecGetArray(vec_dsc,&solution_vec);CHKERRQ(ierr);  

  ierr = DSC_GetLocalSolution(lu->My_DSC_Solver,solution_vec, lu->num_local_cols);

  /* put permuted local solution solution_vec into x_mpi in the original order */
  ierr = VecSetValues(x,lu->num_local_cols,lu->local_cols_old_num,solution_vec,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec_dsc,&rhs_vec);CHKERRQ(ierr); 
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* free spaces */
  if (size > 1) {
    ierr = ISDestroy(iden);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat);CHKERRQ(ierr);
  }

  ierr = VecDestroy(vec_dsc);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_MPIBAIJ_DSCPACK"
int MatLUFactorNumeric_MPIBAIJ_DSCPACK(Mat A,Mat *F)
{
  Mat_MPIBAIJ       *fac;
  Mat_SeqBAIJ       *fac_seq,*a_seq;
  Mat_MPIBAIJ_DSC   *lu; 
  Mat               *tseq,A_seq;

  RealNumberType    *my_a_nonz;
  int               ierr,M=A->M,Mbs,size,i;
  int               max_mem_estimate, max_single_malloc_blk,
                    number_of_procs;
  int               rank;
  int               j,next,iold;
	
  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); /* to be removed */

  if ( size == 1){    
    fac_seq = (Mat_SeqBAIJ*)(*F)->data;  
    lu      = (Mat_MPIBAIJ_DSC*)fac_seq->spptr; 
  } else {    
    fac  = (Mat_MPIBAIJ*)(*F)->data;  
    lu   = (Mat_MPIBAIJ_DSC*)fac->spptr; 
  }

  Mbs = M/lu->bs;
  printf(" Num_Factor is called ..., bs: %d, Mbs: %d, size: %d\n",lu->bs,Mbs,size);
  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */

    /* convert A to A_seq */
    if (size > 1) { 
      ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&lu->iden); CHKERRQ(ierr);
      ierr = MatGetSubMatrices(A,1,&lu->iden,&lu->iden,MAT_INITIAL_MATRIX,&tseq); CHKERRQ(ierr);  
      /* ierr = ISDestroy(iden);CHKERRQ(ierr); */
   
      A_seq = *tseq;
      ierr = PetscFree(tseq);CHKERRQ(ierr); 
      a_seq = (Mat_SeqBAIJ*)A_seq->data;
    } else {
      a_seq = (Mat_SeqBAIJ*)A->data;
    }
   
    ierr = PetscMalloc(Mbs*sizeof(int),&lu->replication);CHKERRQ(ierr);
    for (i=0; i<Mbs; i++) lu->replication[i] = lu->bs;

    number_of_procs = DSC_Analyze(Mbs, a_seq->i, a_seq->j, lu->replication);
    if (number_of_procs < size) 
      SETERRQ1(1,"... max processor possible is %d \n", number_of_procs);

    /* DSC_Solver starts */
    lu->My_DSC_Solver = DSC_Begin();
    DSC_Open0( lu->My_DSC_Solver, size, &lu->rank, PETSC_COMM_WORLD ); 
    /* fail to work for size != power of 2! */
    if (rank != lu->rank || size > 512) {
      PetscPrintf(PETSC_COMM_SELF,"[%d] my_pid = %d\n", rank,lu->rank);
      SETERRQ(1,"... num of processors must be a power of 2 and no more than 512\n");
    }

    ierr = DSC_Order(lu->My_DSC_Solver,lu->order_code,Mbs,a_seq->i,a_seq->j,lu->replication,
                   &M, 
                   &lu->num_local_strucs, 
                   &lu->num_local_cols, &lu->num_local_nonz,  &lu->global_struc_new_col_num, 
                   &lu->global_struc_new_num, &lu->global_struc_owner, 
                   &lu->local_struc_old_num);
    if (ierr !=  DSC_NO_ERROR) {
      DSC_ErrorDisplay(lu->My_DSC_Solver);
      SETERRQ(1,"Error when use DSC_Order()");
    }

    ierr = DSC_SFactor(lu->My_DSC_Solver,&max_mem_estimate,&max_single_malloc_blk,
                     MAX_MEM_ALLOWED, lu->LBLASLevel, lu->DBLASLevel);
    if (ierr !=  DSC_NO_ERROR) {
      DSC_ErrorDisplay(lu->My_DSC_Solver);
      SETERRQ(1,"Error when use DSC_Order");
    }

    /* get local_cols_old_num and IS my_cols to be used later */
    ierr = PetscMalloc(lu->num_local_cols*sizeof(int),&lu->local_cols_old_num);CHKERRQ(ierr);  
    for (next = 0, i=0; i<lu->num_local_strucs; i++){
      iold = lu->bs*lu->local_struc_old_num[i];
      for (j=0; j<lu->bs; j++)
        lu->local_cols_old_num[next++] = iold++;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->num_local_cols,lu->local_cols_old_num,&lu->my_cols);CHKERRQ(ierr);

    ierr = BAIJtoMyANonz(a_seq->i, a_seq->j, lu->bs, a_seq->a,
                       lu->num_local_strucs, lu->num_local_nonz,  
                       lu->global_struc_new_col_num, 
                       lu->local_struc_old_num,
                       PETSC_NULL,
                       &my_a_nonz);

  } else { /* use previously computed symbolic factor */
    DSC_ReFactorInitialize(lu->My_DSC_Solver);

    /* convert A to my A_seq */
    if (size > 1) { 
      int *idx,*i_idx,*tmp;
      IS  my_cols_sorted;
      
      ierr = PetscMalloc(2*lu->num_local_strucs*sizeof(int),&idx);CHKERRQ(ierr);
      /* ierr = PetscMalloc(lu->num_local_strucs*sizeof(int),&i_idx);CHKERRQ(ierr); */
      i_idx = idx + lu->num_local_strucs;
      ierr = PetscMalloc(lu->num_local_cols*sizeof(int),&tmp);CHKERRQ(ierr); 
      
      isort2(lu->num_local_strucs, lu->local_struc_old_num, idx);
      for (next=0, i=0; i< lu->num_local_strucs; i++) {
        iold = lu->bs*lu->local_struc_old_num[idx[i]]; 
        for (j=0; j<lu->bs; j++){
          tmp[next++] = iold++; /* sorted local_cols_old_num */
        }
      }
      for (i=0; i< lu->num_local_strucs; i++) {       
        i_idx[idx[i]] = i;  /* inverse of idx */
      }

      ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->num_local_cols,tmp,&my_cols_sorted);CHKERRQ(ierr);
      
      ierr = MatGetSubMatrices(A,1,&my_cols_sorted,&lu->iden,MAT_INITIAL_MATRIX,&tseq); CHKERRQ(ierr);  
      /* ierr = ISDestroy(iden);CHKERRQ(ierr); */
      ierr = ISDestroy(my_cols_sorted);CHKERRQ(ierr);
   
      A_seq = *tseq;
      ierr = PetscFree(tseq);CHKERRQ(ierr); 
      a_seq = (Mat_SeqBAIJ*)A_seq->data;
      
      ierr = BAIJtoMyANonz(a_seq->i, a_seq->j, lu->bs, a_seq->a,
                       lu->num_local_strucs, lu->num_local_nonz,  
                       lu->global_struc_new_col_num, 
                       lu->local_struc_old_num,
                       i_idx,
                       &my_a_nonz);

      ierr = PetscFree(idx);
      ierr = PetscFree(tmp);
     
    } else {
      a_seq = (Mat_SeqBAIJ*)A->data;
    
      ierr = BAIJtoMyANonz(a_seq->i, a_seq->j, lu->bs, a_seq->a,
                       lu->num_local_strucs, lu->num_local_nonz,  
                       lu->global_struc_new_col_num, 
                       lu->local_struc_old_num,
                       PETSC_NULL,
                       &my_a_nonz);
    }
  }
  
  if (ierr <0) {
    DSC_ErrorDisplay(lu->My_DSC_Solver);
    SETERRQ1(1,"Error seeting local nonzeroes at processor %d \n", lu->rank);
  }
  
  /* if( !lu->rank ) printf("scheme_code: %d, factor_type: %d\n",lu->scheme_code, lu->factor_type); */
  ierr = DSC_NFactor(lu->My_DSC_Solver, lu->scheme_code, my_a_nonz, lu->factor_type, lu->LBLASLevel, lu->DBLASLevel);    
  ierr = PetscFree(my_a_nonz);CHKERRQ(ierr);

  if ( size>1 ) {ierr = MatDestroy(A_seq);CHKERRQ(ierr); }
  
  (*F)->assembled = PETSC_TRUE; 
  lu->flg         = SAME_NONZERO_PATTERN;

  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIBAIJ_DSCPACK"
int MatLUFactorSymbolic_MPIBAIJ_DSCPACK(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_MPIBAIJ             *fac;
  Mat_SeqBAIJ             *fac_seq;
  Mat_MPIBAIJ_DSC         *lu;   
  int                     ierr,M=A->M,size; 
  char                    buff[32];
  PetscTruth              flg;
  char                    *ftype[] = {"LLT","LDLT"}; 
  char                    *ltype[] = {"LBLAS1","LBLAS2","LBLAS3"}; 
  char                    *dtype[] = {"DBLAS1","DBLAS2"}; 

  PetscFunctionBegin; 
  printf(" Sym_Factor is called ..., M: %d\n",M);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  
  ierr = PetscNew(Mat_MPIBAIJ_DSC,&lu);CHKERRQ(ierr); 

  /* Create the factorization matrix F */ 
  ierr = MatGetBlockSize(A,&lu->bs);
  if (size == 1){
    ierr = MatCreateSeqBAIJ(A->comm,lu->bs,M,M,0,PETSC_NULL,F);CHKERRQ(ierr);
    fac_seq = (Mat_SeqBAIJ*)(*F)->data; 
    fac_seq->spptr = (void*)lu;
  } else {
    ierr = MatCreateMPIBAIJ(A->comm,lu->bs,PETSC_DECIDE,PETSC_DECIDE,M,M,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
    fac     = (Mat_MPIBAIJ*)(*F)->data; 
    fac->spptr = (void*)lu;
  } 
  (*F)->ops->lufactornumeric  = MatLUFactorNumeric_MPIBAIJ_DSCPACK;
  (*F)->ops->solve            = MatSolve_MPIBAIJ_DSCPACK;
  (*F)->ops->destroy          = MatDestroy_MPIBAIJ_DSCPACK;  
  (*F)->factor                = FACTOR_LU;  

  /* Set the input options */
  lu->order_code  = 1; 
  lu->scheme_code = 1;
  lu->factor_type = 1;

  ierr = PetscOptionsBegin(A->comm,A->prefix,"DSCPACK Options","Mat");CHKERRQ(ierr); 

  ierr = PetscOptionsInt("-mat_baij_dscpack_order","order_code: 
         1 = ND,  2 = hybrid with Minimum Degree,  3 = Deficiency",
         "None",
         lu->order_code,&lu->order_code,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-mat_baij_dscpack_scheme","scheme_code: 
         1 = standard factorization,  2 = factor + selectively invert",
         "None",
         lu->scheme_code,&lu->scheme_code,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsEList("-mat_baij_dscpack_factor","factor_type","None",
             ftype,2,ftype[0],buff,32,&flg);CHKERRQ(ierr);
  while (flg) {
    ierr = PetscStrcmp(buff,"LLT",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->factor_type = DSC_LLT;
      break;
    }
    ierr = PetscStrcmp(buff,"LDLT",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->factor_type = DSC_LDLT;
      break;
    }
    SETERRQ1(1,"Unknown factor type %s",buff);
  }
  
  ierr = PetscOptionsEList("-mat_baij_dscpack_LBLAS","BLAS level used in the local phase","None",
             ltype,3,ltype[1],buff,32,&flg);CHKERRQ(ierr);
  while (flg) {
    ierr = PetscStrcmp(buff,"LBLAS1",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->LBLASLevel = DSC_LBLAS1;
      break;
    }
    ierr = PetscStrcmp(buff,"LBLAS2",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->LBLASLevel = DSC_LBLAS2;
      break;
    }
    ierr = PetscStrcmp(buff,"LBLAS3",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->LBLASLevel = DSC_LBLAS3;
      break;
    }
    SETERRQ1(1,"Unknown local phase BLAS level %s",buff);
  }

  ierr = PetscOptionsEList("-mat_baij_dscpack_DBLAS","BLAS level used in the distributed phase","None",
             dtype,2,dtype[1],buff,32,&flg);CHKERRQ(ierr);
  while (flg) {
    ierr = PetscStrcmp(buff,"DBLAS1",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->DBLASLevel = DSC_DBLAS1;
      break;
    }
    ierr = PetscStrcmp(buff,"DBLAS2",&flg);CHKERRQ(ierr);
    if (flg) {
      lu->DBLASLevel = DSC_DBLAS2;
      break;
    }
    SETERRQ1(1,"Unknown distributed phase BLAS level %s",buff);
  }

  
  PetscOptionsEnd();
  

  lu->flg = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseDSCPACK_MPIBAIJ"
int MatUseDSCPACK_MPIBAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIBAIJ_DSCPACK;
  A->ops->lufactornumeric  = MatLUFactorNumeric_MPIBAIJ_DSCPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMPIBAIJFactorInfo_DSCPACK"
int MatMPIBAIJFactorInfo_DSCPACK(Mat A,PetscViewer viewer)
{
  Mat_MPIBAIJ             *fac;
  Mat_SeqBAIJ             *fac_seq;
  Mat_MPIBAIJ_DSC         *lu;  
  int                     size,ierr;
  char                    *factor_type,*lblas,*dblas;
  
  PetscFunctionBegin;
  /* check if matrix is dscpack type */
  if (A->ops->solve != MatSolve_MPIBAIJ_DSCPACK) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if ( size == 1){    
    fac_seq = (Mat_SeqBAIJ*)A->data;  
    lu      = (Mat_MPIBAIJ_DSC*)fac_seq->spptr;    
  } else {    
    fac  = (Mat_MPIBAIJ*)A->data;  
    lu   = (Mat_MPIBAIJ_DSC*)fac->spptr; 
  }
  
  
  ierr = PetscViewerASCIIPrintf(viewer,"DSCPACK run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  order_code: %d \n",lu->order_code);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  scheme_code: %d \n",lu->scheme_code);CHKERRQ(ierr);

  if ( lu->factor_type == DSC_LLT) {
    factor_type = "LLT";
  } else if ( lu->factor_type == DSC_LDLT){
    factor_type = "LDLT";
  } else {
    SETERRQ(1,"Unknown factor type");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  factor type: %s \n",factor_type);CHKERRQ(ierr);
  
  if ( lu->LBLASLevel == DSC_LBLAS1) {    
    lblas = "BLAS1";
  } else if ( lu->LBLASLevel == DSC_LBLAS2){
    lblas = "BLAS2";
  } else if ( lu->LBLASLevel == DSC_LBLAS3){
    lblas = "BLAS3";
  } else {
    SETERRQ(1,"Unknown local phase BLAS level");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  local phase BLAS level: %s \n",lblas);CHKERRQ(ierr);

  if ( lu->DBLASLevel == DSC_DBLAS1) {
    dblas = "BLAS1";
  } else if ( lu->DBLASLevel == DSC_DBLAS2){
    dblas = "BLAS2";
  } else {
    SETERRQ(1,"Unknown distributed phase BLAS level");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  distributed phase BLAS level: %s \n",dblas);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseDSCPACK_MPIBAIJ"
int MatUseDSCPACK_MPIBAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


