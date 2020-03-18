
static char help[] = "Tests MatCreateSubmatrix() in parallel.";

#include <petscmat.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode ISGetSeqIS_SameColDist_Private(Mat mat,IS isrow,IS iscol,IS *isrow_d,IS *iscol_d,IS *iscol_o,const PetscInt *garray[])
{
  PetscErrorCode ierr;
  Vec            x,cmap;
  const PetscInt *is_idx;
  PetscScalar    *xarray,*cmaparray;
  PetscInt       ncols,isstart,*idx,m,rstart,count;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)mat->data;
  Mat            B=a->B;
  Vec            lvec=a->lvec,lcmap;
  PetscInt       i,cstart,cend,Bn=B->cmap->N;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  VecScatter     Mvctx;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
  ierr = MatCreateVecs(mat,&x,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&cmap);CHKERRQ(ierr);
  ierr = VecSet(x,-1.0);CHKERRQ(ierr);
  ierr = VecSet(cmap,-1.0);CHKERRQ(ierr);

  ierr = VecDuplicate(lvec,&lcmap);CHKERRQ(ierr);

  /* Get start indices */
  ierr = MPI_Scan(&ncols,&isstart,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  isstart -= ncols;
  ierr = MatGetOwnershipRangeColumn(mat,&cstart,&cend);CHKERRQ(ierr);

  ierr = ISGetIndices(iscol,&is_idx);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(cmap,&cmaparray);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncols,&idx);CHKERRQ(ierr);
  for (i=0; i<ncols; i++) {
    xarray[is_idx[i]-cstart]    = (PetscScalar)is_idx[i];
    cmaparray[is_idx[i]-cstart] = (PetscScalar)(i + isstart);      /* global index of iscol[i] */
    idx[i]                      = is_idx[i]-cstart; /* local index of iscol[i]  */
  }
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(cmap,&cmaparray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&is_idx);CHKERRQ(ierr);

  /* Get iscol_d */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,PETSC_OWN_POINTER,iscol_d);CHKERRQ(ierr);
  ierr = ISGetBlockSize(iscol,&i);CHKERRQ(ierr);
  ierr = ISSetBlockSize(*iscol_d,i);CHKERRQ(ierr);

  /* Get isrow_d */
  ierr = ISGetLocalSize(isrow,&m);CHKERRQ(ierr);
  rstart = mat->rmap->rstart;
  ierr = PetscMalloc1(m,&idx);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&is_idx);CHKERRQ(ierr);
  for (i=0; i<m; i++) idx[i] = is_idx[i]-rstart;
  ierr = ISRestoreIndices(isrow,&is_idx);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,isrow_d);CHKERRQ(ierr);
  ierr = ISGetBlockSize(isrow,&i);CHKERRQ(ierr);
  ierr = ISSetBlockSize(*isrow_d,i);CHKERRQ(ierr);

  /* (2) Scatter x and cmap using aij->Mvctx to get their off-process portions (see MatMult_MPIAIJ) */
#if 0
  if (!a->Mvctx_mpi1) {
    /* a->Mvctx causes random 'count' in o-build? See src/mat/tests/runex59_2 */
    a->Mvctx_mpi1_flg = PETSC_TRUE;
    ierr = MatSetUpMultiply_MPIAIJ(mat);CHKERRQ(ierr);
  }
  Mvctx = a->Mvctx_mpi1;
#endif
  Mvctx = a->Mvctx;
  ierr = VecScatterBegin(Mvctx,x,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Mvctx,x,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterBegin(Mvctx,cmap,lcmap,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Mvctx,cmap,lcmap,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* (3) create sequential iscol_o (a subset of iscol) and isgarray */
  /* off-process column indices */
  count = 0;
  PetscInt *cmap1;
  ierr = PetscMalloc1(Bn,&idx);CHKERRQ(ierr);
  ierr = PetscMalloc1(Bn,&cmap1);CHKERRQ(ierr);

  ierr = VecGetArray(lvec,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(lcmap,&cmaparray);CHKERRQ(ierr);
  for (i=0; i<Bn; i++) {
    if (PetscRealPart(xarray[i]) > -1.0) {
      idx[count]   = i;                   /* local column index in off-diagonal part B */
      cmap1[count] = (PetscInt)(PetscRealPart(cmaparray[i]));  /* column index in submat */
      count++;
    }
  }
  printf("[%d] Bn %d, count %d\n",rank,Bn,count);
  ierr = VecRestoreArray(lvec,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(lcmap,&cmaparray);CHKERRQ(ierr);
  if (count != 6) {
    printf("[%d] count %d != 6 lvec:\n",rank,count);
    ierr = VecView(lvec,0);CHKERRQ(ierr);

    printf("[%d] count %d != 6 lcmap:\n",rank,count);
    ierr = VecView(lcmap,0);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"count %d != 6",count);
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF,count,idx,PETSC_COPY_VALUES,iscol_o);CHKERRQ(ierr);
  /* cannot ensure iscol_o has same blocksize as iscol! */

  ierr = PetscFree(idx);CHKERRQ(ierr);

  *garray = cmap1;

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&cmap);CHKERRQ(ierr);
  ierr = VecDestroy(&lcmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 3,n = 2,rstart,rend;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             isrow,iscol;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n    = 2*size;

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<m*n; j++) {
      v    = i + j + 1;
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Generate a new matrix consisting of every second row and column of
   the original matrix
  */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  /* Create parallel IS with the rows we want on THIS processor */
  ierr = ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&isrow);CHKERRQ(ierr);
  /* Create parallel IS with the rows we want on THIS processor (same as rows for now) */
  ierr = ISCreateStride(PETSC_COMM_WORLD,(rend-rstart)/2,rstart,2,&iscol);CHKERRQ(ierr);

  IS             iscol_d,isrow_d,iscol_o;
  const PetscInt *garray;
  ierr = ISGetSeqIS_SameColDist_Private(C,isrow,iscol,&isrow_d,&iscol_d,&iscol_o,&garray);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow_d);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol_d);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol_o);CHKERRQ(ierr);
  ierr = PetscFree(garray);CHKERRQ(ierr);

  ierr = MatCreateSubMatrix(C,isrow,iscol,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(C,isrow,iscol,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
