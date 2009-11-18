#define PETSCMAT_DLL

/* 
        Provides an interface to the DSCPACK (Domain-Separator Codes) sparse direct solver
*/

#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/impls/baij/mpi/mpibaij.h"

EXTERN_C_BEGIN
#include "dscmain.h"
EXTERN_C_END

typedef struct {
  DSC_Solver	My_DSC_Solver;  
  PetscInt      num_local_strucs, *local_struc_old_num,
                num_local_cols, num_local_nonz, 
                *global_struc_new_col_num,
                *global_struc_new_num, *global_struc_owner,  
                dsc_id,bs,*local_cols_old_num,*replication; 
  PetscInt      order_code,scheme_code,factor_type, stat, 
                LBLASLevel,DBLASLevel,max_mem_allowed;             
  MatStructure  flg;
  IS            my_cols,iden,iden_dsc;
  Vec           vec_dsc;
  VecScatter    scat;
  MPI_Comm      comm_dsc;

  /* A few inheritance details */
  PetscMPIInt    size;

  PetscTruth CleanUpDSCPACK;
} Mat_DSCPACK;


/* DSC function */
#undef __FUNCT__  
#define __FUNCT__ "isort2"
void isort2(PetscInt size, PetscInt *list, PetscInt *idx_dsc) {
  /* in increasing order */
  /* idx_dsc will contain indices such that */
  /* list can be accessed in sorted order */
  PetscInt i, j, x, y;
  
  for (i=0; i<size; i++) idx_dsc[i] =i;

  for (i=1; i<size; i++){
    y= idx_dsc[i];
    x=list[idx_dsc[i]];
    for (j=i-1; ((j>=0) && (x<list[idx_dsc[j]])); j--)
      idx_dsc[j+1]=idx_dsc[j];
    idx_dsc[j+1]=y;
  }
}/*end isort2*/

#undef __FUNCT__  
#define __FUNCT__ "BAIJtoMyANonz"
PetscErrorCode  BAIJtoMyANonz( PetscInt *AIndex, PetscInt *AStruct, PetscInt bs,
		    RealNumberType *ANonz, PetscInt NumLocalStructs, 
                    PetscInt NumLocalNonz,  PetscInt *GlobalStructNewColNum,                
		    PetscInt *LocalStructOldNum,
                    PetscInt *LocalStructLocalNum,
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
  PetscErrorCode ierr;
  PetscInt       i, j, k, iold,inew, jj, kk, bs2=bs*bs,*idx, *NewColNum, MyANonz_last, max_struct=0, struct_size;
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
  ierr = PetscMalloc2(max_struct,PetscInt,&NewColNum,max_struct,PetscInt,&idx);CHKERRQ(ierr);
  
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

  ierr = PetscFree2(NewColNum); 
  if (MyANonz_last != NumLocalNonz) SETERRQ2(PETSC_ERR_PLIB,"MyANonz_last %d != NumLocalNonz %d\n",MyANonz_last, NumLocalNonz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_DSCPACK"
PetscErrorCode MatDestroy_DSCPACK(Mat A) 
{
  Mat_DSCPACK    *lu=(Mat_DSCPACK*)A->spptr;  
  PetscErrorCode ierr;
    
  PetscFunctionBegin;
  if (lu->CleanUpDSCPACK) {
    if (lu->dsc_id != -1) {  
      if(lu->stat) DSC_DoStats(lu->My_DSC_Solver);    
      DSC_FreeAll(lu->My_DSC_Solver);   
      DSC_Close0(lu->My_DSC_Solver);
      
      ierr = PetscFree(lu->local_cols_old_num);CHKERRQ(ierr); 
    } 
    DSC_End(lu->My_DSC_Solver); 
 
    ierr = MPI_Comm_free(&lu->comm_dsc);CHKERRQ(ierr);
    ierr = ISDestroy(lu->my_cols);CHKERRQ(ierr);  
    ierr = PetscFree(lu->replication);CHKERRQ(ierr);
    ierr = VecDestroy(lu->vec_dsc);CHKERRQ(ierr); 
    ierr = ISDestroy(lu->iden_dsc);CHKERRQ(ierr);
    ierr = VecScatterDestroy(lu->scat);CHKERRQ(ierr);
    if (lu->size >1 && lu->iden) {ierr = ISDestroy(lu->iden);CHKERRQ(ierr);}
  }
  if (lu->size == 1) {
    ierr = MatDestroy_SeqBAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIBAIJ(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_DSCPACK"
PetscErrorCode MatSolve_DSCPACK(Mat A,Vec b,Vec x) 
{
  Mat_DSCPACK    *lu= (Mat_DSCPACK*)A->spptr;
  PetscErrorCode ierr;
  RealNumberType *solution_vec,*rhs_vec; 

  PetscFunctionBegin;
  /* scatter b into seq vec_dsc */  
  if ( !lu->scat ) {
    ierr = VecScatterCreate(b,lu->my_cols,lu->vec_dsc,lu->iden_dsc,&lu->scat);CHKERRQ(ierr); 
  }    
  ierr = VecScatterBegin(lu->scat,b,lu->vec_dsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(lu->scat,b,lu->vec_dsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  if (lu->dsc_id != -1){
    ierr = VecGetArray(lu->vec_dsc,&rhs_vec);CHKERRQ(ierr);    
    DSC_InputRhsLocalVec(lu->My_DSC_Solver, rhs_vec, lu->num_local_cols);
    ierr = VecRestoreArray(lu->vec_dsc,&rhs_vec);CHKERRQ(ierr); 
 
    ierr = DSC_Solve(lu->My_DSC_Solver);
    if (ierr !=  DSC_NO_ERROR) {
      DSC_ErrorDisplay(lu->My_DSC_Solver);
      SETERRQ(PETSC_ERR_LIB,"Error in calling DSC_Solve");
    }

    /* get the permuted local solution */
    ierr = VecGetArray(lu->vec_dsc,&solution_vec);CHKERRQ(ierr);  
    ierr = DSC_GetLocalSolution(lu->My_DSC_Solver,solution_vec, lu->num_local_cols);
    ierr = VecRestoreArray(lu->vec_dsc,&solution_vec);CHKERRQ(ierr); 

  } /* end of if (lu->dsc_id != -1) */

  /* put permuted local solution solution_vec into x in the original order */
  ierr = VecScatterBegin(lu->scat,lu->vec_dsc,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(lu->scat,lu->vec_dsc,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatCholeskyFactorNumeric_DSCPACK"
PetscErrorCode MatCholeskyFactorNumeric_DSCPACK(Mat F,Mat A,const MatFactorInfo *info) 
{
  Mat_SeqBAIJ    *a_seq;
  Mat_DSCPACK    *lu=(Mat_DSCPACK*)(F)->spptr; 
  Mat            *tseq,A_seq=PETSC_NULL;
  RealNumberType *my_a_nonz;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       M=A->rmap->N,Mbs=M/lu->bs,max_mem_estimate,max_single_malloc_blk,
                 number_of_procs,i,j,next,iold,*idx,*iidx=0,*itmp;
  IS             my_cols_sorted;
  Mat            F_diag;
	
  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */
    /* convert A to A_seq */
    if (size > 1) { 
      if (!lu->iden){
        ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&lu->iden);CHKERRQ(ierr);
      }
      ierr = MatGetSubMatrices(A,1,&lu->iden,&lu->iden,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr); 
      A_seq = tseq[0];
      a_seq = (Mat_SeqBAIJ*)A_seq->data;
    } else {
      a_seq = (Mat_SeqBAIJ*)A->data;
    }
   
    ierr = PetscMalloc(Mbs*sizeof(PetscInt),&lu->replication);CHKERRQ(ierr);
    for (i=0; i<Mbs; i++) lu->replication[i] = lu->bs;

    number_of_procs = DSC_Analyze(Mbs, a_seq->i, a_seq->j, lu->replication);
    
    i = size;
    if ( number_of_procs < i ) i = number_of_procs;
    number_of_procs = 1;   
    while ( i > 1 ){
      number_of_procs  *= 2; i /= 2; 
    }

    /* DSC_Solver starts */
    DSC_Open0( lu->My_DSC_Solver, number_of_procs, &lu->dsc_id, lu->comm_dsc ); 

    if (lu->dsc_id != -1) {
      ierr = DSC_Order(lu->My_DSC_Solver,lu->order_code,Mbs,a_seq->i,a_seq->j,lu->replication,
                   &M,&lu->num_local_strucs, 
                   &lu->num_local_cols, &lu->num_local_nonz,  &lu->global_struc_new_col_num, 
                   &lu->global_struc_new_num, &lu->global_struc_owner, 
                   &lu->local_struc_old_num);
      if (ierr !=  DSC_NO_ERROR) {
        DSC_ErrorDisplay(lu->My_DSC_Solver);
        SETERRQ(PETSC_ERR_LIB,"Error when use DSC_Order()");
      }

      ierr = DSC_SFactor(lu->My_DSC_Solver,&max_mem_estimate,&max_single_malloc_blk,
                     lu->max_mem_allowed, lu->LBLASLevel, lu->DBLASLevel);
      if (ierr !=  DSC_NO_ERROR) {
        DSC_ErrorDisplay(lu->My_DSC_Solver);
        SETERRQ(PETSC_ERR_LIB,"Error when use DSC_Order"); 
      }

      ierr = BAIJtoMyANonz(a_seq->i, a_seq->j, lu->bs, a_seq->a,
                       lu->num_local_strucs, lu->num_local_nonz,  
                       lu->global_struc_new_col_num, 
                       lu->local_struc_old_num,
                       PETSC_NULL,
                       &my_a_nonz);
      if (ierr <0) {
          DSC_ErrorDisplay(lu->My_DSC_Solver);
          SETERRQ1(PETSC_ERR_LIB,"Error setting local nonzeroes at processor %d \n", lu->dsc_id);
      }

      /* get local_cols_old_num and IS my_cols to be used later */
      ierr = PetscMalloc(lu->num_local_cols*sizeof(PetscInt),&lu->local_cols_old_num);CHKERRQ(ierr);  
      for (next = 0, i=0; i<lu->num_local_strucs; i++){
        iold = lu->bs*lu->local_struc_old_num[i];
        for (j=0; j<lu->bs; j++)
          lu->local_cols_old_num[next++] = iold++;
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->num_local_cols,lu->local_cols_old_num,&lu->my_cols);CHKERRQ(ierr);
      
    } else {    /* lu->dsc_id == -1 */  
      lu->num_local_cols = 0; 
      lu->local_cols_old_num = 0; 
      ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->num_local_cols,lu->local_cols_old_num,&lu->my_cols);CHKERRQ(ierr);
    } 
    /* generate vec_dsc and iden_dsc to be used later */
    ierr = VecCreateSeq(PETSC_COMM_SELF,lu->num_local_cols,&lu->vec_dsc);CHKERRQ(ierr);  
    ierr = ISCreateStride(PETSC_COMM_SELF,lu->num_local_cols,0,1,&lu->iden_dsc);CHKERRQ(ierr); 
    lu->scat = PETSC_NULL;

    if ( size>1 ) {
      ierr = MatDestroyMatrices(1,&tseq);CHKERRQ(ierr); 
    }
  } else { /* use previously computed symbolic factor */
    /* convert A to my A_seq */
    if (size > 1) { 
      if (lu->dsc_id == -1) {
        itmp = 0;
      } else {     
        ierr = PetscMalloc2(lu->num_local_strucs,PetscInt,&idx,lu->num_local_strucs,PetscInt,&iidx);CHKERRQ(ierr);
        ierr = PetscMalloc(lu->num_local_cols*sizeof(PetscInt),&itmp);CHKERRQ(ierr); 
      
        isort2(lu->num_local_strucs, lu->local_struc_old_num, idx);
        for (next=0, i=0; i< lu->num_local_strucs; i++) {
          iold = lu->bs*lu->local_struc_old_num[idx[i]]; 
          for (j=0; j<lu->bs; j++){
            itmp[next++] = iold++; /* sorted local_cols_old_num */
          }
        }
        for (i=0; i< lu->num_local_strucs; i++) {       
          iidx[idx[i]] = i;       /* inverse of idx */
        }
      } /* end of (lu->dsc_id == -1) */
      ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->num_local_cols,itmp,&my_cols_sorted);CHKERRQ(ierr); 
      ierr = MatGetSubMatrices(A,1,&my_cols_sorted,&lu->iden,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr); 
      ierr = ISDestroy(my_cols_sorted);CHKERRQ(ierr);
      A_seq = tseq[0];
    
      if (lu->dsc_id != -1) {
        DSC_ReFactorInitialize(lu->My_DSC_Solver);

        a_seq = (Mat_SeqBAIJ*)A_seq->data;      
        ierr = BAIJtoMyANonz(a_seq->i, a_seq->j, lu->bs, a_seq->a,
                       lu->num_local_strucs, lu->num_local_nonz,  
                       lu->global_struc_new_col_num, 
                       lu->local_struc_old_num,
                       iidx,
                       &my_a_nonz);
        if (ierr <0) {
          DSC_ErrorDisplay(lu->My_DSC_Solver);
          SETERRQ1(PETSC_ERR_LIB,"Error setting local nonzeroes at processor %d \n", lu->dsc_id);
        }
        ierr = PetscFree2(idx,iidex);CHKERRQ(ierr);
        ierr = PetscFree(itmp);CHKERRQ(ierr);
      } /* end of if(lu->dsc_id != -1)  */
    } else { /* size == 1 */
      a_seq = (Mat_SeqBAIJ*)A->data;
    
      ierr = BAIJtoMyANonz(a_seq->i, a_seq->j, lu->bs, a_seq->a,
                       lu->num_local_strucs, lu->num_local_nonz,  
                       lu->global_struc_new_col_num, 
                       lu->local_struc_old_num,
                       PETSC_NULL,
                       &my_a_nonz);
      if (ierr <0) {
        DSC_ErrorDisplay(lu->My_DSC_Solver);
        SETERRQ1(PETSC_ERR_LIB,"Error setting local nonzeroes at processor %d \n", lu->dsc_id);
      }
    }
    if ( size>1 ) {ierr = MatDestroyMatrices(1,&tseq);CHKERRQ(ierr); }   
  }
  
  if (lu->dsc_id != -1) {
    ierr = DSC_NFactor(lu->My_DSC_Solver, lu->scheme_code, my_a_nonz, lu->factor_type, lu->LBLASLevel, lu->DBLASLevel);    
    ierr = PetscFree(my_a_nonz);CHKERRQ(ierr);
  }  
  
  if (size > 1) {
    F_diag = ((Mat_MPIBAIJ *)(F)->data)->A;
    F_diag->assembled = PETSC_TRUE;
  }
  F->assembled   = PETSC_TRUE; 
  lu->flg           = SAME_NONZERO_PATTERN;
  F->ops->solve                  = MatSolve_DSCPACK;

  PetscFunctionReturn(0);
}

/* Note the Petsc permutation r is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_DSCPACK"
PetscErrorCode MatCholeskyFactorSymbolic_DSCPACK(Mat F,Mat A,IS r,const MatFactorInfo *info) 
{
  Mat_DSCPACK    *lu = (Mat_DSCPACK*)(F)->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  lu->My_DSC_Solver = DSC_Begin();
  lu->CleanUpDSCPACK = PETSC_TRUE;
  (F)->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_DSCPACK;
  PetscFunctionReturn(0); 
}


EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_dscpack"
PetscErrorCode MatFactorGetSolverPackage_seqaij_dscpack(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MAT_SOLVER_DSCPACK;
  PetscFunctionReturn(0);
}
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqbaij_dscpack"
PetscErrorCode MatGetFactor_seqbaij_dscpack(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  Mat_DSCPACK    *lu;   
  PetscErrorCode ierr;
  PetscInt       bs,indx; 
  PetscTruth     flg;
  const char     *ftype[]={"LDLT","LLT"},*ltype[]={"LBLAS1","LBLAS2","LBLAS3"},*dtype[]={"DBLAS1","DBLAS2"}; 

  PetscFunctionBegin; 

  /* Create the factorization matrix F */ 
  ierr = MatGetBlockSize(A,&bs);
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(B,bs,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(B,bs,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_DSCPACKPACK,&lu);CHKERRQ(ierr);    

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_DSCPACK;
  B->ops->destroy                = MatDestroy_DSCPACK;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqaij_dscpack",MatFactorGetSolverPackage_seqaij_dscpack);CHKERRQ(ierr);
  B->factor                      = MAT_FACTOR_CHOLESKY;  

  /* Set the default input options */
  lu->order_code  = 2; 
  lu->scheme_code = 1;
  lu->factor_type = 2;
  lu->stat        = 0; /* do not display stats */
  lu->LBLASLevel  = DSC_LBLAS3;
  lu->DBLASLevel  = DSC_DBLAS2;
  lu->max_mem_allowed = 256;
  ierr = MPI_Comm_dup(((PetscObject)A)->comm,&lu->comm_dsc);CHKERRQ(ierr);
  /* Get the runtime input options */
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"DSCPACK Options","Mat");CHKERRQ(ierr); 

  ierr = PetscOptionsInt("-mat_dscpack_order","order_code: \n\
         1 = ND, 2 = Hybrid with Minimum Degree, 3 = Hybrid with Minimum Deficiency", \
         "None",
         lu->order_code,&lu->order_code,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-mat_dscpack_scheme","scheme_code: \n\
         1 = standard factorization,  2 = factorization + selective inversion", \
         "None",
         lu->scheme_code,&lu->scheme_code,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsEList("-mat_dscpack_factor","factor_type","None",ftype,2,ftype[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      lu->factor_type = DSC_LDLT;
      break;
    case 1:
      lu->factor_type = DSC_LLT;
      break;
    }
  }
  ierr = PetscOptionsInt("-mat_dscpack_MaxMemAllowed","in Mbytes","None",
         lu->max_mem_allowed,&lu->max_mem_allowed,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-mat_dscpack_stats","display stats: 0 = no display,  1 = display",
         "None", lu->stat,&lu->stat,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsEList("-mat_dscpack_LBLAS","BLAS level used in the local phase","None",ltype,3,ltype[2],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      lu->LBLASLevel = DSC_LBLAS1;
      break;
    case 1:
      lu->LBLASLevel = DSC_LBLAS2;
      break;
    case 2:
      lu->LBLASLevel = DSC_LBLAS3;
      break;
    }
  }

  ierr = PetscOptionsEList("-mat_dscpack_DBLAS","BLAS level used in the distributed phase","None",dtype,2,dtype[1],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      lu->DBLASLevel = DSC_DBLAS1;
      break;
    case 1:
      lu->DBLASLevel = DSC_DBLAS2;
      break;
    }
  }
  PetscOptionsEnd();
  lu->flg = DIFFERENT_NONZERO_PATTERN;
  *F = B;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_DSCPACK"
PetscErrorCode MatFactorInfo_DSCPACK(Mat A,PetscViewer viewer)
{
  Mat_DSCPACK    *lu=(Mat_DSCPACK*)A->spptr;  
  PetscErrorCode ierr;
  const char     *s=0;
  
  PetscFunctionBegin;   
  ierr = PetscViewerASCIIPrintf(viewer,"DSCPACK run parameters:\n");CHKERRQ(ierr);

  switch (lu->order_code) {
  case 1: s = "ND"; break;
  case 2: s = "Hybrid with Minimum Degree"; break;
  case 3: s = "Hybrid with Minimum Deficiency"; break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  order_code: %s \n",s);CHKERRQ(ierr);

  switch (lu->scheme_code) {
  case 1: s = "standard factorization"; break;
  case 2: s = "factorization + selective inversion"; break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  scheme_code: %s \n",s);CHKERRQ(ierr);

  switch (lu->stat) {
  case 0: s = "NO"; break;
  case 1: s = "YES"; break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  display stats: %s \n",s);CHKERRQ(ierr);
  
  if ( lu->factor_type == DSC_LLT) {
    s = "LLT";
  } else if ( lu->factor_type == DSC_LDLT){
    s = "LDLT";
  } else if (lu->factor_type == 0) {
    s = "None";
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Unknown factor type");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  factor type: %s \n",s);CHKERRQ(ierr);

  if ( lu->LBLASLevel == DSC_LBLAS1) {    
    s = "BLAS1";
  } else if ( lu->LBLASLevel == DSC_LBLAS2){
    s = "BLAS2";
  } else if ( lu->LBLASLevel == DSC_LBLAS3){
    s = "BLAS3";
  } else if (lu->LBLASLevel == 0) {
    s = "None";
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Unknown local phase BLAS level");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  local phase BLAS level: %s \n",s);CHKERRQ(ierr);
  
  if ( lu->DBLASLevel == DSC_DBLAS1) {
    s = "BLAS1";
  } else if ( lu->DBLASLevel == DSC_DBLAS2){
    s = "BLAS2";
  } else if (lu->DBLASLevel == 0) {
    s = "None";
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Unknown distributed phase BLAS level");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  distributed phase BLAS level: %s \n",s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatView_SeqBAIJ(Mat,PetscViewer);
EXTERN PetscErrorCode MatView_MPIBAIJ(Mat,PetscViewer);


#undef __FUNCT__
#define __FUNCT__ "MatView_DSCPACK"
PetscErrorCode MatView_DSCPACK(Mat A,PetscViewer viewer) 
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscTruth        iascii;
  PetscViewerFormat format;
  Mat_DSCPACK       *lu=(Mat_DSCPACK*)A->spptr;

  PetscFunctionBegin;

  /* This convertion ugliness is because MatView for BAIJ types calls MatConvert to AIJ */ 
  size = lu->size;
  if (size==1) {
    ierr = MatView_SeqBAIJ(A,viewer);CHKERRQ(ierr);
  } else {
    ierr = MatView_MPIBAIJ(A,viewer);CHKERRQ(ierr);
  }    

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_DSCPACK(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
  MAT_SOLVER_DSCPACK -  "dscpack" - Provides direct solvers (Cholesky) for sequential 
  or distributed matrices via the external package DSCPACK.


  Options Database Keys:
+ -mat_dscpack_order <1,2,3> - DSCPACK ordering, 1:ND, 2:Hybrid with Minimum Degree, 3:Hybrid with Minimum Deficiency
. -mat_dscpack_scheme <1,2> - factorization scheme, 1:standard factorization,  2: factorization with selective inversion
. -mat_dscpack_factor <LLT,LDLT> - the type of factorization to be performed.
. -mat_dscpack_MaxMemAllowed <n> - the maximum memory to be used during factorization
. -mat_dscpack_stats <0,1> - display stats of the factorization and solves during MatDestroy(), 0: no display,  1: display
. -mat_dscpack_LBLAS <LBLAS1,LBLAS2,LBLAS3> - BLAS level used in the local phase
- -mat_dscpack_DBLAS <DBLAS1,DBLAS2> - BLAS level used in the distributed phase

   Level: beginner

.seealso: PCCHOLESKY, PCFactorSetMatSolverPackage(), MatSolverPackage

M*/
