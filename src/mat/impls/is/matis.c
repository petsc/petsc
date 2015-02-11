
/*
    Creates a matrix class for using the Neumann-Neumann type preconditioners.
   This stores the matrices in globally unassembled form. Each processor
   assembles only its local Neumann problem and the parallel matrix vector
   product is handled "implicitly".

     We provide:
         MatMult()

    Currently this allows for only one subdomain per processor.

*/

#include <../src/mat/impls/is/matis.h>      /*I "petscmat.h" I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "MatISSetPreallocation"
/*
   MatISSetPreallocation - Preallocates memory for a MATIS parallel matrix.

   Collective on MPI_Comm

   Input Parameters:
+  B - the matrix
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or NULL, if d_nz is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e 'm'.
           For matrices that will be factored, you must leave room for (and set)
           the diagonal entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or NULL, if o_nz is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e 'm'.

   If the *_nnz parameter is given then the *_nz parameter is ignored

   Level: intermediate

   Notes: This function has the same interface as the MPIAIJ preallocation routine in order to simplify the transition
          from the asssembled format to the unassembled one. It overestimates the preallocation of MATIS local
          matrices; for exact preallocation, the user should set the preallocation directly on local matrix objects.

.keywords: matrix

.seealso: MatCreate(), MatCreateIS(), MatMPIAIJSetPreallocation(), MatISGetLocalMat()
@*/
PetscErrorCode  MatISSetPreallocation(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscTryMethod(B,"MatISSetPreallocation_C",(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISSetPreallocation_IS"
PetscErrorCode  MatISSetPreallocation_IS(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_IS         *matis = (Mat_IS*)(B->data);
  PetscSF        sf;
  PetscInt       bs,i,nroots,*rootdata,nleaves,*leafdata,nlocalcols;
  const PetscInt *gidxs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!matis->A) {
    SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"You should first call MatSetLocalToGlobalMapping");
  }
  ierr = MatGetLocalSize(B,&nroots,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&nleaves,&nlocalcols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
  ierr = PetscCalloc2(nroots,&rootdata,nleaves,&leafdata);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)B),&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(matis->mapping,&gidxs);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf,B->rmap,nleaves,NULL,PETSC_COPY_VALUES,gidxs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(matis->mapping,&gidxs);CHKERRQ(ierr);
  if (!d_nnz) {
    for (i=0;i<nroots;i++) rootdata[i] += d_nz;
  } else {
    for (i=0;i<nroots;i++) rootdata[i] += d_nnz[i];
  }
  if (!o_nnz) {
    for (i=0;i<nroots;i++) rootdata[i] += o_nz;
  } else {
    for (i=0;i<nroots;i++) rootdata[i] += o_nnz[i];
  }
  ierr = PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);
  for (i=0;i<nleaves;i++) {
    leafdata[i] = PetscMin(leafdata[i],nlocalcols);
  }
  ierr = MatSeqAIJSetPreallocation(matis->A,0,leafdata);CHKERRQ(ierr);
  for (i=0;i<nleaves/bs;i++) {
    leafdata[i] = leafdata[i*bs]/bs;
  }
  ierr = MatSeqBAIJSetPreallocation(matis->A,bs,0,leafdata);CHKERRQ(ierr);
  for (i=0;i<nleaves/bs;i++) {
    leafdata[i] = leafdata[i]-i;
  }
  ierr = MatSeqSBAIJSetPreallocation(matis->A,bs,0,leafdata);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISGetMPIXAIJ_IS"
PetscErrorCode MatISGetMPIXAIJ_IS(Mat mat, MatReuse reuse, Mat *M)
{
  Mat_IS                 *matis = (Mat_IS*)(mat->data);
  /* info on mat */
  /* ISLocalToGlobalMapping rmapping,cmapping; */
  PetscInt               bs,rows,cols;
  PetscInt               lrows,lcols;
  PetscInt               local_rows,local_cols;
  PetscBool              isdense,issbaij,issbaij_red;
  PetscMPIInt            nsubdomains;
  /* values insertion */
  PetscScalar            *array;
  PetscInt               *local_indices,*global_indices;
  /* work */
  PetscInt               i,j,index_row;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* MISSING CHECKS
    - rectangular case not covered (it is not allowed by MATIS)
  */
  /* get info from mat */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&nsubdomains);CHKERRQ(ierr);
  if (nsubdomains == 1) {
    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatDuplicate(matis->A,MAT_COPY_VALUES,&(*M));CHKERRQ(ierr);
    } else {
      ierr = MatCopy(matis->A,*M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  /* ierr = MatGetLocalToGlobalMapping(mat,&rmapping,&cmapping);CHKERRQ(ierr); */
  ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(matis->A,&local_rows,&local_cols);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);

  /* work */
  ierr = PetscMalloc1(local_rows,&local_indices);CHKERRQ(ierr);
  for (i=0;i<local_rows;i++) local_indices[i]=i;
  /* map indices of local mat to global values */
  ierr = PetscMalloc1(PetscMax(local_cols,local_rows),&global_indices);CHKERRQ(ierr);
  /* ierr = ISLocalToGlobalMappingApply(rmapping,local_rows,local_indices,global_indices);CHKERRQ(ierr); */
  ierr = ISLocalToGlobalMappingApply(matis->mapping,local_rows,local_indices,global_indices);CHKERRQ(ierr);

  if (issbaij) {
    ierr = MatGetRowUpperTriangular(matis->A);CHKERRQ(ierr);
  }

  if (reuse == MAT_INITIAL_MATRIX) {
    Mat         new_mat;
    MatType     new_mat_type;
    Vec         vec_dnz,vec_onz;
    PetscScalar *my_dnz,*my_onz;
    PetscInt    *dnz,*onz,*mat_ranges,*row_ownership;
    PetscInt    index_col,owner;

    /* determining new matrix type */
    ierr = MPI_Allreduce(&issbaij,&issbaij_red,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
    if (issbaij_red) {
      new_mat_type = MATSBAIJ;
    } else {
      if (bs>1) {
        new_mat_type = MATBAIJ;
      } else {
        new_mat_type = MATAIJ;
      }
    }

    ierr = MatCreate(PetscObjectComm((PetscObject)mat),&new_mat);CHKERRQ(ierr);
    ierr = MatSetSizes(new_mat,PETSC_DECIDE,PETSC_DECIDE,rows,cols);CHKERRQ(ierr);
    ierr = MatSetBlockSize(new_mat,bs);CHKERRQ(ierr);
    ierr = MatSetType(new_mat,new_mat_type);CHKERRQ(ierr);
    ierr = MatSetUp(new_mat);CHKERRQ(ierr);
    ierr = MatGetLocalSize(new_mat,&lrows,&lcols);CHKERRQ(ierr);

    /*
      preallocation
    */

    ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)new_mat),lrows,lcols,dnz,onz);CHKERRQ(ierr);
    /*
       Some vectors are needed to sum up properly on shared interface dofs.
       Preallocation macros cannot do the job.
       Note that preallocation is not exact, since it overestimates nonzeros
    */
    ierr = MatCreateVecs(new_mat,NULL,&vec_dnz);CHKERRQ(ierr);
    /* ierr = VecSetLocalToGlobalMapping(vec_dnz,rmapping);CHKERRQ(ierr); */
    ierr = VecSetLocalToGlobalMapping(vec_dnz,matis->mapping);CHKERRQ(ierr);
    ierr = VecDuplicate(vec_dnz,&vec_onz);CHKERRQ(ierr);
    /* All processes need to compute entire row ownership */
    ierr = PetscMalloc1(rows,&row_ownership);CHKERRQ(ierr);
    ierr = MatGetOwnershipRanges(new_mat,(const PetscInt**)&mat_ranges);CHKERRQ(ierr);
    for (i=0;i<nsubdomains;i++) {
      for (j=mat_ranges[i];j<mat_ranges[i+1];j++) {
        row_ownership[j]=i;
      }
    }

    /*
       my_dnz and my_onz contains exact contribution to preallocation from each local mat
       then, they will be summed up properly. This way, preallocation is always sufficient
    */
    ierr = PetscMalloc1(local_rows,&my_dnz);CHKERRQ(ierr);
    ierr = PetscMalloc1(local_rows,&my_onz);CHKERRQ(ierr);
    ierr = PetscMemzero(my_dnz,local_rows*sizeof(*my_dnz));CHKERRQ(ierr);
    ierr = PetscMemzero(my_onz,local_rows*sizeof(*my_onz));CHKERRQ(ierr);
    /* preallocation as a MATAIJ */
    if (isdense) { /* special case for dense local matrices */
      for (i=0;i<local_rows;i++) {
        index_row = global_indices[i];
        for (j=i;j<local_rows;j++) {
          owner = row_ownership[index_row];
          index_col = global_indices[j];
          if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1] ) { /* diag block */
            my_dnz[i] += 1.0;
          } else { /* offdiag block */
            my_onz[i] += 1.0;
          }
          /* same as before, interchanging rows and cols */
          if (i != j) {
            owner = row_ownership[index_col];
            if (index_row > mat_ranges[owner]-1 && index_row < mat_ranges[owner+1] ) {
              my_dnz[j] += 1.0;
            } else {
              my_onz[j] += 1.0;
            }
          }
        }
      }
    } else {
      for (i=0;i<local_rows;i++) {
        PetscInt ncols;
        const PetscInt *cols;
        index_row = global_indices[i];
        ierr = MatGetRow(matis->A,i,&ncols,&cols,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          owner = row_ownership[index_row];
          index_col = global_indices[cols[j]];
          if (index_col > mat_ranges[owner]-1 && index_col < mat_ranges[owner+1] ) { /* diag block */
            my_dnz[i] += 1.0;
          } else { /* offdiag block */
            my_onz[i] += 1.0;
          }
          /* same as before, interchanging rows and cols */
          if (issbaij) {
            owner = row_ownership[index_col];
            if (index_row > mat_ranges[owner]-1 && index_row < mat_ranges[owner+1] ) {
              my_dnz[j] += 1.0;
            } else {
              my_onz[j] += 1.0;
            }
          }
        }
        ierr = MatRestoreRow(matis->A,i,&ncols,&cols,NULL);CHKERRQ(ierr);
      }
    }
    ierr = VecSet(vec_dnz,0.0);CHKERRQ(ierr);
    ierr = VecSet(vec_onz,0.0);CHKERRQ(ierr);
    if (local_rows) { /* multilevel guard */
      ierr = VecSetValuesLocal(vec_dnz,local_rows,local_indices,my_dnz,ADD_VALUES);CHKERRQ(ierr);
      ierr = VecSetValuesLocal(vec_onz,local_rows,local_indices,my_onz,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(vec_dnz);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vec_onz);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(vec_dnz);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(vec_onz);CHKERRQ(ierr);
    ierr = PetscFree(my_dnz);CHKERRQ(ierr);
    ierr = PetscFree(my_onz);CHKERRQ(ierr);
    ierr = PetscFree(row_ownership);CHKERRQ(ierr);

    /* set computed preallocation in dnz and onz */
    ierr = VecGetArray(vec_dnz,&array);CHKERRQ(ierr);
    for (i=0; i<lrows; i++) dnz[i] = (PetscInt)PetscRealPart(array[i]);
    ierr = VecRestoreArray(vec_dnz,&array);CHKERRQ(ierr);
    ierr = VecGetArray(vec_onz,&array);CHKERRQ(ierr);
    for (i=0;i<lrows;i++) onz[i] = (PetscInt)PetscRealPart(array[i]);
    ierr = VecRestoreArray(vec_onz,&array);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_dnz);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_onz);CHKERRQ(ierr);

    /* Resize preallocation if overestimated */
    for (i=0;i<lrows;i++) {
      dnz[i] = PetscMin(dnz[i],lcols);
      onz[i] = PetscMin(onz[i],cols-lcols);
    }
    /* set preallocation */
    ierr = MatMPIAIJSetPreallocation(new_mat,0,dnz,0,onz);CHKERRQ(ierr);
    for (i=0;i<lrows/bs;i++) {
      dnz[i] = dnz[i*bs]/bs;
      onz[i] = onz[i*bs]/bs;
    }
    ierr = MatMPIBAIJSetPreallocation(new_mat,bs,0,dnz,0,onz);CHKERRQ(ierr);
    for (i=0;i<lrows/bs;i++) {
      dnz[i] = dnz[i]-i;
    }
    ierr = MatMPISBAIJSetPreallocation(new_mat,bs,0,dnz,0,onz);CHKERRQ(ierr);
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    *M = new_mat;
  } else {
    PetscInt mbs,mrows,mcols;
    /* some checks */
    ierr = MatGetBlockSize(*M,&mbs);CHKERRQ(ierr);
    ierr = MatGetSize(*M,&mrows,&mcols);CHKERRQ(ierr);
    if (mrows != rows) {
      SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of rows (%d != %d)",rows,mrows);
    }
    if (mrows != rows) {
      SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong number of cols (%d != %d)",cols,mcols);
    }
    if (mbs != bs) {
      SETERRQ2(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix. Wrong block size (%d != %d)",bs,mbs);
    }
    ierr = MatZeroEntries(*M);CHKERRQ(ierr);
  }
  /* set local to global mappings */
  /* ierr = MatSetLocalToGlobalMapping(*M,rmapping,cmapping);CHKERRQ(ierr); */
  /* Set values */
  if (isdense) { /* special case for dense local matrices */
    ierr = MatSetOption(*M,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatDenseGetArray(matis->A,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*M,local_rows,global_indices,local_cols,global_indices,array,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(matis->A,&array);CHKERRQ(ierr);
    ierr = PetscFree(local_indices);CHKERRQ(ierr);
    ierr = PetscFree(global_indices);CHKERRQ(ierr);
  } else { /* very basic values insertion for all other matrix types */
    ierr = PetscFree(local_indices);CHKERRQ(ierr);
    for (i=0;i<local_rows;i++) {
      ierr = MatGetRow(matis->A,i,&j,(const PetscInt**)&local_indices,(const PetscScalar**)&array);CHKERRQ(ierr);
      /* ierr = MatSetValuesLocal(*M,1,&i,j,local_indices,array,ADD_VALUES);CHKERRQ(ierr); */
      ierr = ISLocalToGlobalMappingApply(matis->mapping,j,local_indices,global_indices);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(matis->mapping,1,&i,&index_row);CHKERRQ(ierr);
      ierr = MatSetValues(*M,1,&index_row,j,global_indices,array,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(matis->A,i,&j,(const PetscInt**)&local_indices,(const PetscScalar**)&array);CHKERRQ(ierr);
    }
    ierr = PetscFree(global_indices);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (isdense) {
    ierr = MatSetOption(*M,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (issbaij) {
    ierr = MatRestoreRowUpperTriangular(matis->A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISGetMPIXAIJ"
/*@
    MatISGetMPIXAIJ - Converts MATIS matrix into a parallel AIJ format

  Input Parameter:
.  mat - the matrix (should be of type MATIS)
.  reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

  Output Parameter:
.  newmat - the matrix in AIJ format

  Level: developer

  Notes:

.seealso: MATIS
@*/
PetscErrorCode MatISGetMPIXAIJ(Mat mat, MatReuse reuse, Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,reuse,2);
  PetscValidPointer(newmat,3);
  if (reuse != MAT_INITIAL_MATRIX) {
    PetscValidHeaderSpecific(*newmat,MAT_CLASSID,3);
    PetscCheckSameComm(mat,1,*newmat,3);
  }
  ierr = PetscUseMethod(mat,"MatISGetMPIXAIJ_C",(Mat,MatReuse,Mat*),(mat,reuse,newmat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_IS"
PetscErrorCode MatDuplicate_IS(Mat mat,MatDuplicateOption op,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)(mat->data);
  PetscInt       bs,m,n,M,N;
  Mat            B,localmat;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatCreateIS(PetscObjectComm((PetscObject)mat),bs,m,n,M,N,matis->mapping,&B);CHKERRQ(ierr);
  ierr = MatDuplicate(matis->A,op,&localmat);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(B,localmat);CHKERRQ(ierr);
  ierr = MatDestroy(&localmat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsHermitian_IS"
PetscErrorCode MatIsHermitian_IS(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  ierr = MatIsHermitian(matis->A,tol,&local_sym);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsSymmetric_IS"
PetscErrorCode MatIsSymmetric_IS(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Mat_IS         *matis = (Mat_IS*)A->data;
  PetscBool      local_sym;

  PetscFunctionBegin;
  ierr = MatIsSymmetric(matis->A,tol,&local_sym);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&local_sym,flg,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_IS"
PetscErrorCode MatDestroy_IS(Mat A)
{
  PetscErrorCode ierr;
  Mat_IS         *b = (Mat_IS*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->x);CHKERRQ(ierr);
  ierr = VecDestroy(&b->y);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&b->mapping);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_IS"
PetscErrorCode MatMult_IS(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_IS         *is  = (Mat_IS*)A->data;
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMult(is->A,is->x,is->y);CHKERRQ(ierr);

  /* scatter product back into global memory */
  ierr = VecSet(y,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_IS"
PetscErrorCode MatMultAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Vec            temp_vec;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  v3 = v2 + A * v1.*/
  if (v3 != v2) {
    ierr = MatMult(A,v1,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(v2,&temp_vec);CHKERRQ(ierr);
    ierr = MatMult(A,v1,temp_vec);CHKERRQ(ierr);
    ierr = VecAXPY(temp_vec,1.0,v2);CHKERRQ(ierr);
    ierr = VecCopy(temp_vec,v3);CHKERRQ(ierr);
    ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_IS"
PetscErrorCode MatMultTranspose_IS(Mat A,Vec x,Vec y)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  y = A' * x */
  /*  scatter the global vector x into the local work vector */
  ierr = VecScatterBegin(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,x,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* multiply the local matrix */
  ierr = MatMultTranspose(is->A,is->x,is->y);CHKERRQ(ierr);

  /* scatter product back into global vector */
  ierr = VecSet(y,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(is->ctx,is->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_IS"
PetscErrorCode MatMultTransposeAdd_IS(Mat A,Vec v1,Vec v2,Vec v3)
{
  Vec            temp_vec;
  PetscErrorCode ierr;

  PetscFunctionBegin; /*  v3 = v2 + A' * v1.*/
  if (v3 != v2) {
    ierr = MatMultTranspose(A,v1,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(v2,&temp_vec);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,v1,temp_vec);CHKERRQ(ierr);
    ierr = VecAXPY(temp_vec,1.0,v2);CHKERRQ(ierr);
    ierr = VecCopy(temp_vec,v3);CHKERRQ(ierr);
    ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_IS"
PetscErrorCode MatView_IS(Mat A,PetscViewer viewer)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
  ierr = MatView(a->A,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetLocalToGlobalMapping_IS"
PetscErrorCode MatSetLocalToGlobalMapping_IS(Mat A,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscErrorCode ierr;
  PetscInt       n,bs;
  Mat_IS         *is = (Mat_IS*)A->data;
  IS             from,to;
  Vec            global;

  PetscFunctionBegin;
  PetscCheckSameComm(A,1,rmapping,2);
  if (rmapping != cmapping) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"MATIS requires the row and column mappings to be identical");
  if (is->mapping) { /* Currenly destroys the objects that will be created by this routine. Is there anything else that should be checked? */
    ierr = ISLocalToGlobalMappingDestroy(&is->mapping);CHKERRQ(ierr);
    ierr = VecDestroy(&is->x);CHKERRQ(ierr);
    ierr = VecDestroy(&is->y);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&is->ctx);CHKERRQ(ierr);
    ierr = MatDestroy(&is->A);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)rmapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&is->mapping);CHKERRQ(ierr);
  is->mapping = rmapping;
/*
  ierr = PetscLayoutSetISLocalToGlobalMapping(A->rmap,rmapping);CHKERRQ(ierr);
  ierr = PetscLayoutSetISLocalToGlobalMapping(A->cmap,cmapping);CHKERRQ(ierr);
*/

  /* Create the local matrix A */
  ierr = ISLocalToGlobalMappingGetSize(rmapping,&n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(rmapping,&bs);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&is->A);CHKERRQ(ierr);
  if (bs > 1) {
    ierr = MatSetType(is->A,MATSEQBAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(is->A,MATSEQAIJ);CHKERRQ(ierr);
  }
  ierr = MatSetSizes(is->A,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetBlockSize(is->A,bs);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(is->A,((PetscObject)A)->prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(is->A,"is_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(is->A);CHKERRQ(ierr);

  /* Create the local work vectors */
  ierr = VecCreate(PETSC_COMM_SELF,&is->x);CHKERRQ(ierr);
  ierr = VecSetBlockSize(is->x,bs);CHKERRQ(ierr);
  ierr = VecSetSizes(is->x,n,n);CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(is->x,((PetscObject)A)->prefix);CHKERRQ(ierr);
  ierr = VecAppendOptionsPrefix(is->x,"is_");CHKERRQ(ierr);
  ierr = VecSetFromOptions(is->x);CHKERRQ(ierr);
  ierr = VecDuplicate(is->x,&is->y);CHKERRQ(ierr);

  /* setup the global to local scatter */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&to);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(rmapping,to,&from);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&global,NULL);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,is->x,to,&is->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_IS"
PetscErrorCode MatSetValues_IS(Mat mat, PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       rows_l[2048],cols_l[2048];
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (m > 2048 || n > 2048) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of row/column indices must be <= 2048: they are %D %D",m,n);
#endif
  ierr = ISG2LMapApply(is->mapping,m,rows,rows_l);CHKERRQ(ierr);
  ierr = ISG2LMapApply(is->mapping,n,cols,cols_l);CHKERRQ(ierr);
  ierr = MatSetValues(is->A,m,rows_l,n,cols_l,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef ISG2LMapSetUp
#undef ISG2LMapApply

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesLocal_IS"
PetscErrorCode MatSetValuesLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  ierr = MatSetValues(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlockedLocal_IS"
PetscErrorCode MatSetValuesBlockedLocal_IS(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscErrorCode ierr;
  Mat_IS         *is = (Mat_IS*)A->data;

  PetscFunctionBegin;
  ierr = MatSetValuesBlocked(is->A,m,rows,n,cols,values,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroRows_IS"
PetscErrorCode MatZeroRows_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscInt       n_l = 0, *rows_l = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x && b) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support");
  if (n) {
    ierr = PetscMalloc1(n,&rows_l);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(is->mapping,IS_GTOLM_DROP,n,rows,&n_l,rows_l);CHKERRQ(ierr);
  }
  ierr = MatZeroRowsLocal(A,n_l,rows_l,diag,x,b);CHKERRQ(ierr);
  ierr = PetscFree(rows_l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsLocal_IS"
PetscErrorCode MatZeroRowsLocal_IS(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (x && b) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support");
  {
    /*
       Set up is->x as a "counting vector". This is in order to MatMult_IS
       work properly in the interface nodes.
    */
    Vec         counter;
    PetscScalar one=1.0, zero=0.0;
    ierr = MatCreateVecs(A,&counter,NULL);CHKERRQ(ierr);
    ierr = VecSet(counter,zero);CHKERRQ(ierr);
    ierr = VecSet(is->x,one);CHKERRQ(ierr);
    ierr = VecScatterBegin(is->ctx,is->x,counter,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (is->ctx,is->x,counter,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(is->ctx,counter,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (is->ctx,counter,is->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecDestroy(&counter);CHKERRQ(ierr);
  }
  if (!n) {
    is->pure_neumann = PETSC_TRUE;
  } else {
    is->pure_neumann = PETSC_FALSE;

    ierr = VecGetArray(is->x,&array);CHKERRQ(ierr);
    ierr = MatZeroRows(is->A,n,rows,diag,0,0);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = MatSetValue(is->A,rows[i],rows[i],diag/(array[rows[i]]),INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(is->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (is->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecRestoreArray(is->x,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_IS"
PetscErrorCode MatAssemblyBegin_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyBegin(is->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_IS"
PetscErrorCode MatAssemblyEnd_IS(Mat A,MatAssemblyType type)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd(is->A,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISGetLocalMat_IS"
PetscErrorCode MatISGetLocalMat_IS(Mat mat,Mat *local)
{
  Mat_IS *is = (Mat_IS*)mat->data;

  PetscFunctionBegin;
  *local = is->A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISGetLocalMat"
/*@
    MatISGetLocalMat - Gets the local matrix stored inside a MATIS matrix.

  Input Parameter:
.  mat - the matrix

  Output Parameter:
.  local - the local matrix usually MATSEQAIJ

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the
  matrix and want to provide it to the inner matrix object to improve the performance
  of the MatSetValues() operation.

.seealso: MATIS
@*/
PetscErrorCode MatISGetLocalMat(Mat mat,Mat *local)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  ierr = PetscUseMethod(mat,"MatISGetLocalMat_C",(Mat,Mat*),(mat,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISSetLocalMat_IS"
PetscErrorCode MatISSetLocalMat_IS(Mat mat,Mat local)
{
  Mat_IS         *is = (Mat_IS*)mat->data;
  PetscInt       nrows,ncols,orows,ocols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is->A) {
    ierr = MatGetSize(is->A,&orows,&ocols);CHKERRQ(ierr);
    ierr = MatGetSize(local,&nrows,&ncols);CHKERRQ(ierr);
    if (orows != nrows || ocols != ncols) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local MATIS matrix should be of size %dx%d (you passed a %dx%d matrix)\n",orows,ocols,nrows,ncols);
  }
  ierr  = PetscObjectReference((PetscObject)local);CHKERRQ(ierr);
  ierr  = MatDestroy(&is->A);CHKERRQ(ierr);
  is->A = local;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatISSetLocalMat"
/*@
    MatISSetLocalMat - Set the local matrix stored inside a MATIS matrix.

  Input Parameter:
.  mat - the matrix
.  local - the local matrix usually MATSEQAIJ

  Output Parameter:

  Level: advanced

  Notes:
    This can be called if you have precomputed the local matrix and
  want to provide it to the matrix object MATIS.

.seealso: MATIS
@*/
PetscErrorCode MatISSetLocalMat(Mat mat,Mat local)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(local,MAT_CLASSID,2);
  ierr = PetscUseMethod(mat,"MatISSetLocalMat_C",(Mat,Mat),(mat,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_IS"
PetscErrorCode MatZeroEntries_IS(Mat A)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_IS"
PetscErrorCode MatScale_IS(Mat A,PetscScalar a)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(is->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_IS"
PetscErrorCode MatGetDiagonal_IS(Mat A, Vec v)
{
  Mat_IS         *is = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get diagonal of the local matrix */
  ierr = MatGetDiagonal(is->A,is->x);CHKERRQ(ierr);

  /* scatter diagonal back into global vector */
  ierr = VecSet(v,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(is->ctx,is->x,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (is->ctx,is->x,v,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_IS"
PetscErrorCode MatSetOption_IS(Mat A,MatOption op,PetscBool flg)
{
  Mat_IS         *a = (Mat_IS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateIS"
/*@
    MatCreateIS - Creates a "process" unassmembled matrix, it is assembled on each
       process but not across processes.

   Input Parameters:
+     comm - MPI communicator that will share the matrix
.     bs - local and global block size of the matrix
.     m,n,M,N - local and/or global sizes of the left and right vector used in matrix vector products
-     map - mapping that defines the global number for each local number

   Output Parameter:
.    A - the resulting matrix

   Level: advanced

   Notes: See MATIS for more details
          m and n are NOT related to the size of the map, they are the size of the part of the vector owned
          by that process. m + nghosts (or n + nghosts) is the length of map since map maps all local points
          plus the ghost points to global indices.

.seealso: MATIS, MatSetLocalToGlobalMapping()
@*/
PetscErrorCode  MatCreateIS(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,ISLocalToGlobalMapping map,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*A,bs);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATIS);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*A,map,map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATIS - MATIS = "is" - A matrix type to be used for using the Neumann-Neumann type preconditioners, see PCBDDC.
   This stores the matrices in globally unassembled form. Each processor
   assembles only its local Neumann problem and the parallel matrix vector
   product is handled "implicitly".

   Operations Provided:
+  MatMult()
.  MatMultAdd()
.  MatMultTranspose()
.  MatMultTransposeAdd()
.  MatZeroEntries()
.  MatSetOption()
.  MatZeroRows()
.  MatZeroRowsLocal()
.  MatSetValues()
.  MatSetValuesLocal()
.  MatScale()
.  MatGetDiagonal()
-  MatSetLocalToGlobalMapping()

   Options Database Keys:
. -mat_type is - sets the matrix type to "is" during a call to MatSetFromOptions()

   Notes: Options prefix for the inner matrix are given by -is_mat_xxx

          You must call MatSetLocalToGlobalMapping() before using this matrix type.

          You can do matrix preallocation on the local matrix after you obtain it with
          MatISGetLocalMat()

  Level: advanced

.seealso: PC, MatISGetLocalMat(), MatSetLocalToGlobalMapping(), PCBDDC

M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_IS"
PETSC_EXTERN PetscErrorCode MatCreate_IS(Mat A)
{
  PetscErrorCode ierr;
  Mat_IS         *b;

  PetscFunctionBegin;
  ierr    = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data = (void*)b;
  ierr    = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->mult                    = MatMult_IS;
  A->ops->multadd                 = MatMultAdd_IS;
  A->ops->multtranspose           = MatMultTranspose_IS;
  A->ops->multtransposeadd        = MatMultTransposeAdd_IS;
  A->ops->destroy                 = MatDestroy_IS;
  A->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_IS;
  A->ops->setvalues               = MatSetValues_IS;
  A->ops->setvalueslocal          = MatSetValuesLocal_IS;
  A->ops->setvaluesblockedlocal   = MatSetValuesBlockedLocal_IS;
  A->ops->zerorows                = MatZeroRows_IS;
  A->ops->zerorowslocal           = MatZeroRowsLocal_IS;
  A->ops->assemblybegin           = MatAssemblyBegin_IS;
  A->ops->assemblyend             = MatAssemblyEnd_IS;
  A->ops->view                    = MatView_IS;
  A->ops->zeroentries             = MatZeroEntries_IS;
  A->ops->scale                   = MatScale_IS;
  A->ops->getdiagonal             = MatGetDiagonal_IS;
  A->ops->setoption               = MatSetOption_IS;
  A->ops->ishermitian             = MatIsHermitian_IS;
  A->ops->issymmetric             = MatIsSymmetric_IS;
  A->ops->duplicate               = MatDuplicate_IS;

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  /* zeros pointer */
  b->A       = 0;
  b->ctx     = 0;
  b->x       = 0;
  b->y       = 0;
  b->mapping = 0;

  /* special MATIS functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetLocalMat_C",MatISGetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetLocalMat_C",MatISSetLocalMat_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISGetMPIXAIJ_C",MatISGetMPIXAIJ_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatISSetPreallocation_C",MatISSetPreallocation_IS);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
