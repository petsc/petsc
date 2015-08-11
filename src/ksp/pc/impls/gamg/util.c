/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include <petsc/private/matimpl.h>
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatCollapseRow"
/*
   Produces a set of block column indices of the matrix row, one for each block represented in the original row

   n - the number of block indices in cc[]
   cc - the block indices (must be large enough to contain the indices)
*/
PETSC_STATIC_INLINE PetscErrorCode MatCollapseRow(Mat Amat,PetscInt row,PetscInt bs,PetscInt *n,PetscInt *cc)
{
  PetscInt       cnt = -1,nidx,j;
  const PetscInt *idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetRow(Amat,row,&nidx,&idx,NULL);CHKERRQ(ierr);
  if (nidx) {
    cnt = 0;
    cc[cnt] = idx[0]/bs;
    for (j=1; j<nidx; j++) {
      if (cc[cnt] < idx[j]/bs) cc[++cnt] = idx[j]/bs;
    }
  }
  ierr = MatRestoreRow(Amat,row,&nidx,&idx,NULL);CHKERRQ(ierr);
  *n = cnt+1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCollapseRows"
/*
    Produces a set of block column indices of the matrix block row, one for each block represented in the original set of rows

    ncollapsed - the number of block indices
    collapsed - the block indices (must be large enough to contain the indices)
*/
PETSC_STATIC_INLINE PetscErrorCode MatCollapseRows(Mat Amat,PetscInt start,PetscInt bs,PetscInt *w0,PetscInt *w1,PetscInt *w2,PetscInt *ncollapsed,PetscInt **collapsed)
{
  PetscInt       i,nprev,*cprev = w0,ncur = 0,*ccur = w1,*merged = w2,*cprevtmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCollapseRow(Amat,start,bs,&nprev,cprev);CHKERRQ(ierr);
  for (i=start+1; i<start+bs; i++) {
    ierr  = MatCollapseRow(Amat,i,bs,&ncur,ccur);CHKERRQ(ierr);
    ierr  = PetscMergeIntArray(nprev,cprev,ncur,ccur,&nprev,&merged);CHKERRQ(ierr);
    cprevtmp = cprev; cprev = merged; merged = cprevtmp;
  }
  *ncollapsed = nprev;
  if (collapsed) *collapsed  = cprev;
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCGAMGCreateGraph - create simple scaled scalar graph from matrix

 Input Parameter:
 . Amat - matrix
 Output Parameter:
 . a_Gmaat - eoutput scalar graph (symmetric?)
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCreateGraph"
PetscErrorCode PCGAMGCreateGraph(Mat Amat, Mat *a_Gmat)
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,i,jj,kk,ncols,nloc,NN,MM,bs;
  MPI_Comm       comm;
  Mat            Gmat;
  MatType        mtype;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat, &Istart, &Iend);CHKERRQ(ierr);
  ierr = MatGetSize(Amat, &MM, &NN);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Amat, &bs);CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs;

#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif

  if (bs > 1) {
    const PetscScalar *vals;
    const PetscInt    *idx;
    PetscInt          *d_nnz, *o_nnz,*blockmask = NULL,maskcnt,*w0,*w1,*w2;
    PetscBool         ismpiaij,isseqaij;

    /*
       Determine the preallocation needed for the scalar matrix derived from the vector matrix.
    */

    ierr = PetscObjectTypeCompare((PetscObject)Amat,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)Amat,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);

    ierr = PetscMalloc2(nloc, &d_nnz,isseqaij ? 0 : nloc, &o_nnz);CHKERRQ(ierr);

    if (isseqaij) {
      PetscInt       max_d_nnz;

      /*
          Determine exact preallocation count for (sequential) scalar matrix
      */
      ierr = MatSeqAIJGetMaxRowNonzeros(Amat,&max_d_nnz);CHKERRQ(ierr);
      max_d_nnz = PetscMin(nloc,bs*max_d_nnz);CHKERRQ(ierr);
      ierr = PetscMalloc3(max_d_nnz, &w0,max_d_nnz, &w1,max_d_nnz, &w2);CHKERRQ(ierr);
      for (Ii = 0, jj = 0; Ii < Iend; Ii += bs, jj++) {
        ierr = MatCollapseRows(Amat,Ii,bs,w0,w1,w2,&d_nnz[jj],NULL);CHKERRQ(ierr);
      }
      ierr = PetscFree3(w0,w1,w2);CHKERRQ(ierr);

    } else if (ismpiaij) {
      Mat            Daij,Oaij;
      const PetscInt *garray;
      PetscInt       max_d_nnz;

      ierr = MatMPIAIJGetSeqAIJ(Amat,&Daij,&Oaij,&garray);CHKERRQ(ierr);

      /*
          Determine exact preallocation count for diagonal block portion of scalar matrix
      */
      ierr = MatSeqAIJGetMaxRowNonzeros(Daij,&max_d_nnz);CHKERRQ(ierr);
      max_d_nnz = PetscMin(nloc,bs*max_d_nnz);CHKERRQ(ierr);
      ierr = PetscMalloc3(max_d_nnz, &w0,max_d_nnz, &w1,max_d_nnz, &w2);CHKERRQ(ierr);
      for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
        ierr = MatCollapseRows(Daij,Ii,bs,w0,w1,w2,&d_nnz[jj],NULL);CHKERRQ(ierr);
      }
      ierr = PetscFree3(w0,w1,w2);CHKERRQ(ierr);

      /*
         Over estimate (usually grossly over), preallocation count for off-diagonal portion of scalar matrix
      */
      for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
        o_nnz[jj] = 0;
        for (kk=0; kk<bs; kk++) { /* rows that get collapsed to a single row */
          ierr = MatGetRow(Oaij,Ii+kk,&ncols,0,0);CHKERRQ(ierr);
          o_nnz[jj] += ncols;
          ierr = MatRestoreRow(Oaij,Ii+kk,&ncols,0,0);CHKERRQ(ierr);
        }
        if (o_nnz[jj] > (NN/bs-nloc)) o_nnz[jj] = NN/bs-nloc;
      }

    } else {
      /*

       This is O(nloc*nloc/bs) work!

       This is accurate for the "diagonal" block of the matrix but will be grossly high for the
       off diagonal block most of the time but could be too low for the off-diagonal.

       This should be fixed to be accurate for the off-diagonal portion. Cannot just use a mask
       for the off-diagonal portion since for huge matrices that would require too much memory per
       MPI process.
      */
      ierr = PetscMalloc1(nloc, &blockmask);CHKERRQ(ierr);
      for (Ii = Istart, jj = 0; Ii < Iend; Ii += bs, jj++) {
        o_nnz[jj] = 0;
        ierr = PetscMemzero(blockmask,nloc*sizeof(PetscInt));CHKERRQ(ierr);
        for (kk=0; kk<bs; kk++) { /* rows that get collapsed to a single row */
          ierr = MatGetRow(Amat,Ii+kk,&ncols,&idx,0);CHKERRQ(ierr);
          for (i=0; i<ncols; i++) {
            if (idx[i] >= Istart && idx[i] < Iend) {
              blockmask[(idx[i] - Istart)/bs] = 1;
            }
          }
          if (ncols > o_nnz[jj]) {
            o_nnz[jj] = ncols;
            if (o_nnz[jj] > (NN/bs-nloc)) o_nnz[jj] = NN/bs-nloc;
          }
          ierr = MatRestoreRow(Amat,Ii+kk,&ncols,&idx,0);CHKERRQ(ierr);
        }
        maskcnt = 0;
        for (i=0; i<nloc; i++) {
          if (blockmask[i]) maskcnt++;
        }
        d_nnz[jj] = maskcnt;
      }
      ierr = PetscFree(blockmask);CHKERRQ(ierr);
    }

    /* get scalar copy (norms) of matrix -- AIJ specific!!! */
    ierr = MatGetType(Amat,&mtype);CHKERRQ(ierr);
    ierr = MatCreate(comm, &Gmat);CHKERRQ(ierr);
    ierr = MatSetSizes(Gmat,nloc,nloc,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(Gmat, 1, 1);CHKERRQ(ierr);
    ierr = MatSetType(Gmat, mtype);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Gmat,0,d_nnz);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(Gmat,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
    ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

    for (Ii = Istart; Ii < Iend; Ii++) {
      PetscInt dest_row = Ii/bs;
      ierr = MatGetRow(Amat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
      for (jj=0; jj<ncols; jj++) {
        PetscInt    dest_col = idx[jj]/bs;
        PetscScalar sv       = PetscAbs(PetscRealPart(vals[jj]));
        ierr = MatSetValues(Gmat,1,&dest_row,1,&dest_col,&sv,ADD_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(Amat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Gmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Gmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    /* just copy scalar matrix - abs() not taken here but scaled later */
    ierr = MatDuplicate(Amat, MAT_COPY_VALUES, &Gmat);CHKERRQ(ierr);
  }

#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif

  *a_Gmat = Gmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGFilterGraph - filter (remove zero and possibly small values from the) graph and symetrize if needed

 Input Parameter:
 . vfilter - threshold paramter [0,1)
 . symm - symetrize?
 In/Output Parameter:
 . a_Gmat - original graph
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGFilterGraph"
PetscErrorCode PCGAMGFilterGraph(Mat *a_Gmat,PetscReal vfilter,PetscBool symm)
{
  PetscErrorCode    ierr;
  PetscInt          Istart,Iend,Ii,jj,ncols,nnz0,nnz1, NN, MM, nloc;
  PetscMPIInt       rank;
  Mat               Gmat  = *a_Gmat, tGmat, matTrans;
  MPI_Comm          comm;
  const PetscScalar *vals;
  const PetscInt    *idx;
  PetscInt          *d_nnz, *o_nnz;
  Vec               diag;
  MatType           mtype;

  PetscFunctionBegin;
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif
  /* scale Gmat for all values between -1 and 1 */
  ierr = MatCreateVecs(Gmat, &diag, 0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(Gmat, diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  ierr = VecSqrtAbs(diag);CHKERRQ(ierr);
  ierr = MatDiagonalScale(Gmat, diag, diag);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);

  if (vfilter < 0.0 && !symm) {
    /* Just use the provided matrix as the graph but make all values positive */
    Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)Gmat->data;
    MatInfo     info;
    PetscScalar *avals;
    PetscMPIInt size;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)Gmat),&size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatGetInfo(Gmat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(Gmat,&avals);CHKERRQ(ierr);
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      ierr = MatSeqAIJRestoreArray(Gmat,&avals);CHKERRQ(ierr);
    } else {
      ierr = MatGetInfo(aij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(aij->A,&avals);CHKERRQ(ierr);
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      ierr = MatSeqAIJRestoreArray(aij->A,&avals);CHKERRQ(ierr);
      ierr = MatGetInfo(aij->B,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(aij->B,&avals);CHKERRQ(ierr);
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      ierr = MatSeqAIJRestoreArray(aij->B,&avals);CHKERRQ(ierr);
    }
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectGetComm((PetscObject)Gmat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat, &Istart, &Iend);CHKERRQ(ierr);
  nloc = Iend - Istart;
  ierr = MatGetSize(Gmat, &MM, &NN);CHKERRQ(ierr);

  if (symm) {
    ierr = MatTranspose(Gmat, MAT_INITIAL_MATRIX, &matTrans);CHKERRQ(ierr);
  }

  /* Determine upper bound on nonzeros needed in new filtered matrix */
  ierr = PetscMalloc2(nloc, &d_nnz,nloc, &o_nnz);CHKERRQ(ierr);
  for (Ii = Istart, jj = 0; Ii < Iend; Ii++, jj++) {
    ierr      = MatGetRow(Gmat,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
    d_nnz[jj] = ncols;
    o_nnz[jj] = ncols;
    ierr      = MatRestoreRow(Gmat,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
    if (symm) {
      ierr       = MatGetRow(matTrans,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
      d_nnz[jj] += ncols;
      o_nnz[jj] += ncols;
      ierr       = MatRestoreRow(matTrans,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
    if (d_nnz[jj] > nloc) d_nnz[jj] = nloc;
    if (o_nnz[jj] > (MM-nloc)) o_nnz[jj] = MM - nloc;
  }
  ierr = MatGetType(Gmat,&mtype);CHKERRQ(ierr);
  ierr = MatCreate(comm, &tGmat);CHKERRQ(ierr);
  ierr = MatSetSizes(tGmat,nloc,nloc,MM,MM);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(tGmat, 1, 1);CHKERRQ(ierr);
  ierr = MatSetType(tGmat, mtype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(tGmat,0,d_nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(tGmat,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  if (symm) {
    ierr = MatDestroy(&matTrans);CHKERRQ(ierr);
  } else {
    /* all entries are generated locally so MatAssembly will be slightly faster for large process counts */
    ierr = MatSetOption(tGmat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  }

  for (Ii = Istart, nnz0 = nnz1 = 0; Ii < Iend; Ii++) {
    ierr = MatGetRow(Gmat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
    for (jj=0; jj<ncols; jj++,nnz0++) {
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      if (PetscRealPart(sv) > vfilter) {
        nnz1++;
        if (symm) {
          sv  *= 0.5;
          ierr = MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,ADD_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(tGmat,1,&idx[jj],1,&Ii,&sv,ADD_VALUES);CHKERRQ(ierr);
        } else {
          ierr = MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(tGmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tGmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif

#if defined(PETSC_USE_INFO)
  {
    double t1 = (!nnz0) ? 1. : 100.*(double)nnz1/(double)nnz0, t2 = (!nloc) ? 1. : (double)nnz0/(double)nloc;
    ierr = PetscInfo4(*a_Gmat,"\t %g%% nnz after filtering, with threshold %g, %g nnz ave. (N=%D)\n",t1,vfilter,t2,MM);CHKERRQ(ierr);
  }
#endif
  ierr    = MatDestroy(&Gmat);CHKERRQ(ierr);
  *a_Gmat = tGmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGGetDataWithGhosts - hacks into Mat MPIAIJ so this must have > 1 pe

   Input Parameter:
   . Gmat - MPIAIJ matrix for scattters
   . data_sz - number of data terms per node (# cols in output)
   . data_in[nloc*data_sz] - column oriented data
   Output Parameter:
   . a_stride - numbrt of rows of output
   . a_data_out[stride*data_sz] - output data with ghosts
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGGetDataWithGhosts"
PetscErrorCode PCGAMGGetDataWithGhosts(Mat Gmat,PetscInt data_sz,PetscReal data_in[],PetscInt *a_stride,PetscReal **a_data_out)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Vec            tmp_crds;
  Mat_MPIAIJ     *mpimat = (Mat_MPIAIJ*)Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar    *data_arr;
  PetscReal      *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Gmat,&comm);CHKERRQ(ierr);
  ierr      = PetscObjectTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ);CHKERRQ(ierr);
  ierr      = MatGetOwnershipRange(Gmat, &my0, &Iend);CHKERRQ(ierr);
  nloc      = Iend - my0;
  ierr      = VecGetLocalSize(mpimat->lvec, &num_ghosts);CHKERRQ(ierr);
  nnodes    = num_ghosts + nloc;
  *a_stride = nnodes;
  ierr      = MatCreateVecs(Gmat, &tmp_crds, 0);CHKERRQ(ierr);

  ierr = PetscMalloc1(data_sz*nnodes, &datas);CHKERRQ(ierr);
  for (dir=0; dir<data_sz; dir++) {
    /* set local, and global */
    for (kk=0; kk<nloc; kk++) {
      PetscInt    gid = my0 + kk;
      PetscScalar crd = (PetscScalar)data_in[dir*nloc + kk]; /* col oriented */
      datas[dir*nnodes + kk] = PetscRealPart(crd);

      ierr = VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tmp_crds);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tmp_crds);CHKERRQ(ierr);
    /* get ghost datas */
    ierr = VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(mpimat->lvec, &data_arr);CHKERRQ(ierr);
    for (kk=nloc,jj=0;jj<num_ghosts;kk++,jj++) datas[dir*nnodes + kk] = PetscRealPart(data_arr[jj]);
    ierr = VecRestoreArray(mpimat->lvec, &data_arr);CHKERRQ(ierr);
  }
  ierr        = VecDestroy(&tmp_crds);CHKERRQ(ierr);
  *a_data_out = datas;
  PetscFunctionReturn(0);
}


/*
 *
 *  GAMGTableCreate
 */

#undef __FUNCT__
#define __FUNCT__ "GAMGTableCreate"
PetscErrorCode GAMGTableCreate(PetscInt a_size, GAMGHashTable *a_tab)
{
  PetscErrorCode ierr;
  PetscInt       kk;

  PetscFunctionBegin;
  a_tab->size = a_size;
  ierr = PetscMalloc1(a_size, &a_tab->table);CHKERRQ(ierr);
  ierr = PetscMalloc1(a_size, &a_tab->data);CHKERRQ(ierr);
  for (kk=0; kk<a_size; kk++) a_tab->table[kk] = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GAMGTableDestroy"
PetscErrorCode GAMGTableDestroy(GAMGHashTable *a_tab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(a_tab->table);CHKERRQ(ierr);
  ierr = PetscFree(a_tab->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GAMGTableAdd"
PetscErrorCode GAMGTableAdd(GAMGHashTable *a_tab, PetscInt a_key, PetscInt a_data)
{
  PetscInt kk,idx;

  PetscFunctionBegin;
  if (a_key<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Negative key %d.",a_key);
  for (kk = 0, idx = GAMG_HASH(a_key);
       kk < a_tab->size;
       kk++, idx = (idx==(a_tab->size-1)) ? 0 : idx + 1) {

    if (a_tab->table[idx] == a_key) {
      /* exists */
      a_tab->data[idx] = a_data;
      break;
    } else if (a_tab->table[idx] == -1) {
      /* add */
      a_tab->table[idx] = a_key;
      a_tab->data[idx]  = a_data;
      break;
    }
  }
  if (kk==a_tab->size) {
    /* this is not to efficient, waiting until completely full */
    PetscInt       oldsize = a_tab->size, new_size = 2*a_tab->size + 5, *oldtable = a_tab->table, *olddata = a_tab->data;
    PetscErrorCode ierr;

    a_tab->size = new_size;

    ierr = PetscMalloc1(a_tab->size, &a_tab->table);CHKERRQ(ierr);
    ierr = PetscMalloc1(a_tab->size, &a_tab->data);CHKERRQ(ierr);

    for (kk=0;kk<a_tab->size;kk++) a_tab->table[kk] = -1;
    for (kk=0;kk<oldsize;kk++) {
      if (oldtable[kk] != -1) {
        ierr = GAMGTableAdd(a_tab, oldtable[kk], olddata[kk]);CHKERRQ(ierr);
       }
    }
    ierr = PetscFree(oldtable);CHKERRQ(ierr);
    ierr = PetscFree(olddata);CHKERRQ(ierr);
    ierr = GAMGTableAdd(a_tab, a_key, a_data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

