#include <../src/mat/impls/baij/mpi/mpibaij.h>

PetscErrorCode  MatGetMultiProcBlock_MPIBAIJ(Mat mat, MPI_Comm subComm, MatReuse scall,Mat *subMat)
{
  Mat_MPIBAIJ    *aij  = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *aijB = (Mat_SeqBAIJ*)aij->B->data;
  PetscMPIInt    commRank,subCommSize,subCommRank;
  PetscMPIInt    *commRankMap,subRank,rank,commsize;
  PetscInt       *garrayCMap,col,i,j,*nnz,newRow,newCol,*newbRow,*newbCol,k,k1;
  PetscInt       bs=mat->rmap->bs;
  PetscScalar    *vals,*aijBvals;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&commsize));
  CHKERRMPI(MPI_Comm_size(subComm,&subCommSize));

  /* create subMat object with the relevant layout */
  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatCreate(subComm,subMat));
    CHKERRQ(MatSetType(*subMat,MATMPIBAIJ));
    CHKERRQ(MatSetSizes(*subMat,mat->rmap->n,mat->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(MatSetBlockSizes(*subMat,mat->rmap->bs,mat->cmap->bs));

    /* need to setup rmap and cmap before Preallocation */
    CHKERRQ(PetscLayoutSetBlockSize((*subMat)->rmap,mat->rmap->bs));
    CHKERRQ(PetscLayoutSetBlockSize((*subMat)->cmap,mat->cmap->bs));
    CHKERRQ(PetscLayoutSetUp((*subMat)->rmap));
    CHKERRQ(PetscLayoutSetUp((*subMat)->cmap));
  }

  /* create a map of comm_rank from subComm to comm - should commRankMap and garrayCMap be kept for reused? */
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&commRank));
  CHKERRMPI(MPI_Comm_rank(subComm,&subCommRank));
  CHKERRQ(PetscMalloc1(subCommSize,&commRankMap));
  CHKERRMPI(MPI_Allgather(&commRank,1,MPI_INT,commRankMap,1,MPI_INT,subComm));

  /* Traverse garray and identify blocked column indices [of offdiag mat] that
   should be discarded. For the ones not discarded, store the newCol+1
   value in garrayCMap */
  CHKERRQ(PetscCalloc1(aij->B->cmap->n/bs,&garrayCMap));
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
    CHKERRQ(PetscCalloc1(aij->B->rmap->n/bs,&nnz));
    for (i=0; i<aij->B->rmap->n/bs; i++) {
      for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
        if (garrayCMap[aijB->j[j]]) nnz[i]++;
      }
    }
    CHKERRQ(MatMPIBAIJSetPreallocation(*(subMat),bs,0,NULL,0,nnz));

    /* reuse diag block with the new submat */
    CHKERRQ(MatDestroy(&((Mat_MPIBAIJ*)((*subMat)->data))->A));

    ((Mat_MPIBAIJ*)((*subMat)->data))->A = aij->A;

    CHKERRQ(PetscObjectReference((PetscObject)aij->A));
  } else if (((Mat_MPIBAIJ*)(*subMat)->data)->A != aij->A) {
    PetscObject obj = (PetscObject)((Mat_MPIBAIJ*)((*subMat)->data))->A;

    CHKERRQ(PetscObjectReference((PetscObject)obj));

    ((Mat_MPIBAIJ*)((*subMat)->data))->A = aij->A;

    CHKERRQ(PetscObjectReference((PetscObject)aij->A));
  }

  /* Now traverse aij->B and insert values into subMat */
  CHKERRQ(PetscMalloc3(bs,&newbRow,bs,&newbCol,bs*bs,&vals));
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
        CHKERRQ(MatSetValues(*subMat,bs,newbRow,bs,newbCol,vals,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(*subMat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*subMat,MAT_FINAL_ASSEMBLY));

  /* deallocate temporary data */
  CHKERRQ(PetscFree3(newbRow,newbCol,vals));
  CHKERRQ(PetscFree(commRankMap));
  CHKERRQ(PetscFree(garrayCMap));
  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(PetscFree(nnz));
  }
  PetscFunctionReturn(0);
}
