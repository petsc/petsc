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
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&commsize));
  PetscCallMPI(MPI_Comm_size(subComm,&subCommSize));

  /* create subMat object with the relevant layout */
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(subComm,subMat));
    PetscCall(MatSetType(*subMat,MATMPIBAIJ));
    PetscCall(MatSetSizes(*subMat,mat->rmap->n,mat->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetBlockSizes(*subMat,mat->rmap->bs,mat->cmap->bs));

    /* need to setup rmap and cmap before Preallocation */
    PetscCall(PetscLayoutSetBlockSize((*subMat)->rmap,mat->rmap->bs));
    PetscCall(PetscLayoutSetBlockSize((*subMat)->cmap,mat->cmap->bs));
    PetscCall(PetscLayoutSetUp((*subMat)->rmap));
    PetscCall(PetscLayoutSetUp((*subMat)->cmap));
  }

  /* create a map of comm_rank from subComm to comm - should commRankMap and garrayCMap be kept for reused? */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&commRank));
  PetscCallMPI(MPI_Comm_rank(subComm,&subCommRank));
  PetscCall(PetscMalloc1(subCommSize,&commRankMap));
  PetscCallMPI(MPI_Allgather(&commRank,1,MPI_INT,commRankMap,1,MPI_INT,subComm));

  /* Traverse garray and identify blocked column indices [of offdiag mat] that
   should be discarded. For the ones not discarded, store the newCol+1
   value in garrayCMap */
  PetscCall(PetscCalloc1(aij->B->cmap->n/bs,&garrayCMap));
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
    PetscCall(PetscCalloc1(aij->B->rmap->n/bs,&nnz));
    for (i=0; i<aij->B->rmap->n/bs; i++) {
      for (j=aijB->i[i]; j<aijB->i[i+1]; j++) {
        if (garrayCMap[aijB->j[j]]) nnz[i]++;
      }
    }
    PetscCall(MatMPIBAIJSetPreallocation(*(subMat),bs,0,NULL,0,nnz));

    /* reuse diag block with the new submat */
    PetscCall(MatDestroy(&((Mat_MPIBAIJ*)((*subMat)->data))->A));

    ((Mat_MPIBAIJ*)((*subMat)->data))->A = aij->A;

    PetscCall(PetscObjectReference((PetscObject)aij->A));
  } else if (((Mat_MPIBAIJ*)(*subMat)->data)->A != aij->A) {
    PetscObject obj = (PetscObject)((Mat_MPIBAIJ*)((*subMat)->data))->A;

    PetscCall(PetscObjectReference((PetscObject)obj));

    ((Mat_MPIBAIJ*)((*subMat)->data))->A = aij->A;

    PetscCall(PetscObjectReference((PetscObject)aij->A));
  }

  /* Now traverse aij->B and insert values into subMat */
  PetscCall(PetscMalloc3(bs,&newbRow,bs,&newbCol,bs*bs,&vals));
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
        PetscCall(MatSetValues(*subMat,bs,newbRow,bs,newbCol,vals,INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(*subMat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*subMat,MAT_FINAL_ASSEMBLY));

  /* deallocate temporary data */
  PetscCall(PetscFree3(newbRow,newbCol,vals));
  PetscCall(PetscFree(commRankMap));
  PetscCall(PetscFree(garrayCMap));
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscFree(nnz));
  }
  PetscFunctionReturn(0);
}
