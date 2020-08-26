#include <../src/mat/impls/baij/mpi/mpibaij.h>

PetscErrorCode  MatGetMultiProcBlock_MPIBAIJ(Mat mat, MPI_Comm subComm, MatReuse scall,Mat *subMat)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *aij  = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *aijB = (Mat_SeqBAIJ*)aij->B->data;
  PetscMPIInt    commRank,subCommSize,subCommRank;
  PetscMPIInt    *commRankMap,subRank,rank,commsize;
  PetscInt       *garrayCMap,col,i,j,*nnz,newRow,newCol,*newbRow,*newbCol,k,k1;
  PetscInt       bs=mat->rmap->bs;
  PetscScalar    *vals,*aijBvals;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&commsize);CHKERRQ(ierr);
  ierr = MPI_Comm_size(subComm,&subCommSize);CHKERRQ(ierr);

  /* create subMat object with the relavent layout */
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = MatCreate(subComm,subMat);CHKERRQ(ierr);
    ierr = MatSetType(*subMat,MATMPIBAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(*subMat,mat->rmap->n,mat->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(*subMat,mat->rmap->bs,mat->cmap->bs);CHKERRQ(ierr);

    /* need to setup rmap and cmap before Preallocation */
    ierr = PetscLayoutSetBlockSize((*subMat)->rmap,mat->rmap->bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize((*subMat)->cmap,mat->cmap->bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*subMat)->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*subMat)->cmap);CHKERRQ(ierr);
  }

  /* create a map of comm_rank from subComm to comm - should commRankMap and garrayCMap be kept for reused? */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&commRank);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(subComm,&subCommRank);CHKERRQ(ierr);
  ierr = PetscMalloc1(subCommSize,&commRankMap);CHKERRQ(ierr);
  ierr = MPI_Allgather(&commRank,1,MPI_INT,commRankMap,1,MPI_INT,subComm);CHKERRQ(ierr);

  /* Traverse garray and identify blocked column indices [of offdiag mat] that
   should be discarded. For the ones not discarded, store the newCol+1
   value in garrayCMap */
  ierr = PetscCalloc1(aij->B->cmap->n/bs,&garrayCMap);CHKERRQ(ierr);
  for (i=0; i<aij->B->cmap->n/bs; i++) {
    col = aij->garray[i]; /* blocked column index */
    for (subRank=0; subRank<subCommSize; subRank++) {
      rank = commRankMap[subRank];
      if ((col >= mat->cmap->range[rank]/bs) && (col < mat->cmap->range[rank+1]/bs)) {
        garrayCMap[i] = (((*subMat)->cmap->range[subRank]- mat->cmap->range[rank])/bs + col + 1);
        break;
      }
    }
  }

  if (scall == MAT_INITIAL_MATRIX) {
    /* Now compute preallocation for the offdiag mat */
    ierr = PetscCalloc1(aij->B->rmap->n/bs,&nnz);CHKERRQ(ierr);
    for (i=0; i<aij->B->rmap->n/bs; i++) {
      for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
        if (garrayCMap[aijB->j[j]]) nnz[i]++;
      }
    }
    ierr = MatMPIBAIJSetPreallocation(*(subMat),bs,0,NULL,0,nnz);CHKERRQ(ierr);

    /* reuse diag block with the new submat */
    ierr = MatDestroy(&((Mat_MPIBAIJ*)((*subMat)->data))->A);CHKERRQ(ierr);

    ((Mat_MPIBAIJ*)((*subMat)->data))->A = aij->A;

    ierr = PetscObjectReference((PetscObject)aij->A);CHKERRQ(ierr);
  } else if (((Mat_MPIBAIJ*)(*subMat)->data)->A != aij->A) {
    PetscObject obj = (PetscObject)((Mat_MPIBAIJ*)((*subMat)->data))->A;

    ierr = PetscObjectReference((PetscObject)obj);CHKERRQ(ierr);

    ((Mat_MPIBAIJ*)((*subMat)->data))->A = aij->A;

    ierr = PetscObjectReference((PetscObject)aij->A);CHKERRQ(ierr);
  }

  /* Now traverse aij->B and insert values into subMat */
  ierr = PetscMalloc3(bs,&newbRow,bs,&newbCol,bs*bs,&vals);CHKERRQ(ierr);
  for (i=0; i<aij->B->rmap->n/bs; i++) {
    newRow = (*subMat)->rmap->range[subCommRank] + i*bs;
    for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
      newCol = garrayCMap[aijB->j[j]];
      if (newCol) {
        newCol--; /* remove the increment */
        newCol *= bs;
        for (k=0; k<bs; k++) {
          newbRow[k] = newRow + k;
          newbCol[k] = newCol + k;
        }
        /* copy column-oriented aijB->a into row-oriented vals */
        aijBvals = aijB->a + j*bs*bs;
        for (k1=0; k1<bs; k1++) {
          for (k=0; k<bs; k++) {
            vals[k1+k*bs] = *aijBvals++;
          }
        }
        ierr = MatSetValues(*subMat,bs,newbRow,bs,newbCol,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(*subMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*subMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* deallocate temporary data */
  ierr = PetscFree3(newbRow,newbCol,vals);CHKERRQ(ierr);
  ierr = PetscFree(commRankMap);CHKERRQ(ierr);
  ierr = PetscFree(garrayCMap);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscFree(nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
