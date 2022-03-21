/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include <petsc/private/matimpl.h>
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>

/*
   Produces a set of block column indices of the matrix row, one for each block represented in the original row

   n - the number of block indices in cc[]
   cc - the block indices (must be large enough to contain the indices)
*/
static inline PetscErrorCode MatCollapseRow(Mat Amat,PetscInt row,PetscInt bs,PetscInt *n,PetscInt *cc)
{
  PetscInt       cnt = -1,nidx,j;
  const PetscInt *idx;

  PetscFunctionBegin;
  PetscCall(MatGetRow(Amat,row,&nidx,&idx,NULL));
  if (nidx) {
    cnt = 0;
    cc[cnt] = idx[0]/bs;
    for (j=1; j<nidx; j++) {
      if (cc[cnt] < idx[j]/bs) cc[++cnt] = idx[j]/bs;
    }
  }
  PetscCall(MatRestoreRow(Amat,row,&nidx,&idx,NULL));
  *n = cnt+1;
  PetscFunctionReturn(0);
}

/*
    Produces a set of block column indices of the matrix block row, one for each block represented in the original set of rows

    ncollapsed - the number of block indices
    collapsed - the block indices (must be large enough to contain the indices)
*/
static inline PetscErrorCode MatCollapseRows(Mat Amat,PetscInt start,PetscInt bs,PetscInt *w0,PetscInt *w1,PetscInt *w2,PetscInt *ncollapsed,PetscInt **collapsed)
{
  PetscInt       i,nprev,*cprev = w0,ncur = 0,*ccur = w1,*merged = w2,*cprevtmp;

  PetscFunctionBegin;
  PetscCall(MatCollapseRow(Amat,start,bs,&nprev,cprev));
  for (i=start+1; i<start+bs; i++) {
    PetscCall(MatCollapseRow(Amat,i,bs,&ncur,ccur));
    PetscCall(PetscMergeIntArray(nprev,cprev,ncur,ccur,&nprev,&merged));
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
PetscErrorCode PCGAMGCreateGraph(Mat Amat, Mat *a_Gmat)
{
  PetscInt       Istart,Iend,Ii,jj,kk,ncols,nloc,NN,MM,bs;
  MPI_Comm       comm;
  Mat            Gmat;
  PetscBool      ismpiaij,isseqaij;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat,&comm));
  PetscCall(MatGetOwnershipRange(Amat, &Istart, &Iend));
  PetscCall(MatGetSize(Amat, &MM, &NN));
  PetscCall(MatGetBlockSize(Amat, &bs));
  nloc = (Iend-Istart)/bs;

  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Amat,MATSEQAIJ,&isseqaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Amat,MATMPIAIJ,&ismpiaij));
  PetscCheck(isseqaij || ismpiaij,PETSC_COMM_WORLD,PETSC_ERR_USER,"Require (MPI)AIJ matrix type");
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0));

  /* TODO GPU: these calls are potentially expensive if matrices are large and we want to use the GPU */
  /* A solution consists in providing a new API, MatAIJGetCollapsedAIJ, and each class can provide a fast
     implementation */
  PetscCall(MatViewFromOptions(Amat, NULL, "-g_mat_view"));
  if (bs > 1 && (isseqaij || ((Mat_MPIAIJ*)Amat->data)->garray)) {
    PetscInt  *d_nnz, *o_nnz;
    Mat       a, b, c;
    MatScalar *aa,val,AA[4096];
    PetscInt  *aj,*ai,AJ[4096],nc;
    PetscCall(PetscInfo(Amat,"New bs>1 PCGAMGCreateGraph. nloc=%D\n",nloc));
    if (isseqaij) {
      a = Amat; b = NULL;
    }
    else {
      Mat_MPIAIJ *d = (Mat_MPIAIJ*)Amat->data;
      a = d->A; b = d->B;
    }
    PetscCall(PetscMalloc2(nloc, &d_nnz,isseqaij ? 0 : nloc, &o_nnz));
    for (c=a, kk=0 ; c && kk<2 ; c=b, kk++){
      PetscInt       *nnz = (c==a) ? d_nnz : o_nnz, nmax=0;
      const PetscInt *cols;
      for (PetscInt brow=0,jj,ok=1,j0; brow < nloc*bs; brow += bs) { // block rows
        PetscCall(MatGetRow(c,brow,&jj,&cols,NULL));
        nnz[brow/bs] = jj/bs;
        if (jj%bs) ok = 0;
        if (cols) j0 = cols[0];
        else j0 = -1;
        PetscCall(MatRestoreRow(c,brow,&jj,&cols,NULL));
        if (nnz[brow/bs]>nmax) nmax = nnz[brow/bs];
        for (PetscInt ii=1; ii < bs && nnz[brow/bs] ; ii++) { // check for non-dense blocks
          PetscCall(MatGetRow(c,brow+ii,&jj,&cols,NULL));
          if (jj%bs) ok = 0;
          if (j0 != cols[0]) ok = 0;
          if (nnz[brow/bs] != jj/bs) ok = 0;
          PetscCall(MatRestoreRow(c,brow+11,&jj,&cols,NULL));
        }
        if (!ok) {
          PetscCall(PetscFree2(d_nnz,o_nnz));
          goto old_bs;
        }
      }
      PetscCheck(nmax<4096,PETSC_COMM_SELF,PETSC_ERR_USER,"Buffer %D too small %D.",nmax,4096);
    }
    PetscCall(MatCreate(comm, &Gmat));
    PetscCall(MatSetSizes(Gmat,nloc,nloc,PETSC_DETERMINE,PETSC_DETERMINE));
    PetscCall(MatSetBlockSizes(Gmat, 1, 1));
    PetscCall(MatSetType(Gmat, MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(Gmat,0,d_nnz));
    PetscCall(MatMPIAIJSetPreallocation(Gmat,0,d_nnz,0,o_nnz));
    PetscCall(PetscFree2(d_nnz,o_nnz));
    // diag
    for (PetscInt brow=0,n,grow; brow < nloc*bs; brow += bs) { // block rows
      Mat_SeqAIJ *aseq  = (Mat_SeqAIJ*)a->data;
      ai = aseq->i;
      n  = ai[brow+1] - ai[brow];
      aj = aseq->j + ai[brow];
      for (int k=0; k<n; k += bs) { // block columns
        AJ[k/bs] = aj[k]/bs + Istart/bs; // diag starts at (Istart,Istart)
        val = 0;
        for (int ii=0; ii<bs; ii++) { // rows in block
          aa = aseq->a + ai[brow+ii] + k;
          for (int jj=0; jj<bs; jj++) { // columns in block
            val += PetscAbs(PetscRealPart(aa[jj])); // a sort of norm
          }
        }
        AA[k/bs] = val;
      }
      grow = Istart/bs + brow/bs;
      PetscCall(MatSetValues(Gmat,1,&grow,n/bs,AJ,AA,INSERT_VALUES));
    }
    // off-diag
    if (ismpiaij) {
      Mat_MPIAIJ        *aij = (Mat_MPIAIJ*)Amat->data;
      const PetscScalar *vals;
      const PetscInt    *cols, *garray = aij->garray;
      PetscCheck(garray,PETSC_COMM_SELF,PETSC_ERR_USER,"No garray ?");
      for (PetscInt brow=0,grow; brow < nloc*bs; brow += bs) { // block rows
        PetscCall(MatGetRow(b,brow,&ncols,&cols,NULL));
        for (int k=0,cidx=0; k<ncols; k += bs,cidx++) {
          AA[k/bs] = 0;
          AJ[cidx] = garray[cols[k]]/bs;
        }
        nc = ncols/bs;
        PetscCall(MatRestoreRow(b,brow,&ncols,&cols,NULL));
        for (int ii=0; ii<bs; ii++) { // rows in block
          PetscCall(MatGetRow(b,brow+ii,&ncols,&cols,&vals));
          for (int k=0; k<ncols; k += bs) {
            for (int jj=0; jj<bs; jj++) { // cols in block
              AA[k/bs] += PetscAbs(PetscRealPart(vals[k+jj]));
            }
          }
          PetscCall(MatRestoreRow(b,brow+ii,&ncols,&cols,&vals));
        }
        grow = Istart/bs + brow/bs;
        PetscCall(MatSetValues(Gmat,1,&grow,nc,AJ,AA,INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(Gmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Gmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(Gmat, NULL, "-g_mat_view"));
  } else if (bs > 1) {
    const PetscScalar *vals;
    const PetscInt    *idx;
    PetscInt          *d_nnz, *o_nnz,*w0,*w1,*w2;

old_bs:
    /*
       Determine the preallocation needed for the scalar matrix derived from the vector matrix.
    */

    PetscCall(PetscInfo(Amat,"OLD bs>1 PCGAMGCreateGraph\n"));
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)Amat,MATSEQAIJ,&isseqaij));
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)Amat,MATMPIAIJ,&ismpiaij));
    PetscCall(PetscMalloc2(nloc, &d_nnz,isseqaij ? 0 : nloc, &o_nnz));

    if (isseqaij) {
      PetscInt max_d_nnz;

      /*
          Determine exact preallocation count for (sequential) scalar matrix
      */
      PetscCall(MatSeqAIJGetMaxRowNonzeros(Amat,&max_d_nnz));
      max_d_nnz = PetscMin(nloc,bs*max_d_nnz);
      PetscCall(PetscMalloc3(max_d_nnz, &w0,max_d_nnz, &w1,max_d_nnz, &w2));
      for (Ii = 0, jj = 0; Ii < Iend; Ii += bs, jj++) {
        PetscCall(MatCollapseRows(Amat,Ii,bs,w0,w1,w2,&d_nnz[jj],NULL));
      }
      PetscCall(PetscFree3(w0,w1,w2));

    } else if (ismpiaij) {
      Mat            Daij,Oaij;
      const PetscInt *garray;
      PetscInt       max_d_nnz;

      PetscCall(MatMPIAIJGetSeqAIJ(Amat,&Daij,&Oaij,&garray));

      /*
          Determine exact preallocation count for diagonal block portion of scalar matrix
      */
      PetscCall(MatSeqAIJGetMaxRowNonzeros(Daij,&max_d_nnz));
      max_d_nnz = PetscMin(nloc,bs*max_d_nnz);
      PetscCall(PetscMalloc3(max_d_nnz, &w0,max_d_nnz, &w1,max_d_nnz, &w2));
      for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
        PetscCall(MatCollapseRows(Daij,Ii,bs,w0,w1,w2,&d_nnz[jj],NULL));
      }
      PetscCall(PetscFree3(w0,w1,w2));

      /*
         Over estimate (usually grossly over), preallocation count for off-diagonal portion of scalar matrix
      */
      for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
        o_nnz[jj] = 0;
        for (kk=0; kk<bs; kk++) { /* rows that get collapsed to a single row */
          PetscCall(MatGetRow(Oaij,Ii+kk,&ncols,NULL,NULL));
          o_nnz[jj] += ncols;
          PetscCall(MatRestoreRow(Oaij,Ii+kk,&ncols,NULL,NULL));
        }
        if (o_nnz[jj] > (NN/bs-nloc)) o_nnz[jj] = NN/bs-nloc;
      }

    } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require AIJ matrix type");

    /* get scalar copy (norms) of matrix */
    PetscCall(MatCreate(comm, &Gmat));
    PetscCall(MatSetSizes(Gmat,nloc,nloc,PETSC_DETERMINE,PETSC_DETERMINE));
    PetscCall(MatSetBlockSizes(Gmat, 1, 1));
    PetscCall(MatSetType(Gmat, MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(Gmat,0,d_nnz));
    PetscCall(MatMPIAIJSetPreallocation(Gmat,0,d_nnz,0,o_nnz));
    PetscCall(PetscFree2(d_nnz,o_nnz));

    for (Ii = Istart; Ii < Iend; Ii++) {
      PetscInt dest_row = Ii/bs;
      PetscCall(MatGetRow(Amat,Ii,&ncols,&idx,&vals));
      for (jj=0; jj<ncols; jj++) {
        PetscInt    dest_col = idx[jj]/bs;
        PetscScalar sv       = PetscAbs(PetscRealPart(vals[jj]));
        PetscCall(MatSetValues(Gmat,1,&dest_row,1,&dest_col,&sv,ADD_VALUES));
      }
      PetscCall(MatRestoreRow(Amat,Ii,&ncols,&idx,&vals));
    }
    PetscCall(MatAssemblyBegin(Gmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Gmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(Gmat, NULL, "-g_mat_view"));
  } else {
    /* just copy scalar matrix - abs() not taken here but scaled later */
    PetscCall(MatDuplicate(Amat, MAT_COPY_VALUES, &Gmat));
  }
  PetscCall(MatPropagateSymmetryOptions(Amat, Gmat));

  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0));

  *a_Gmat = Gmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*@C
   PCGAMGFilterGraph - filter (remove zero and possibly small values from the) graph and make it symmetric if requested

   Collective on Mat

   Input Parameters:
+   a_Gmat - the graph
.   vfilter - threshold parameter [0,1)
-   symm - make the result symmetric

   Level: developer

   Notes:
    This is called before graph coarsers are called.

.seealso: PCGAMGSetThreshold()
@*/
PetscErrorCode PCGAMGFilterGraph(Mat *a_Gmat,PetscReal vfilter,PetscBool symm)
{
  PetscInt          Istart,Iend,Ii,jj,ncols,nnz0,nnz1, NN, MM, nloc;
  PetscMPIInt       rank;
  Mat               Gmat  = *a_Gmat, tGmat;
  MPI_Comm          comm;
  const PetscScalar *vals;
  const PetscInt    *idx;
  PetscInt          *d_nnz, *o_nnz;
  Vec               diag;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[SET16],0,0,0,0));

  /* TODO GPU: optimization proposal, each class provides fast implementation of this
     procedure via MatAbs API */
  if (vfilter < 0.0 && !symm) {
    /* Just use the provided matrix as the graph but make all values positive */
    MatInfo     info;
    PetscScalar *avals;
    PetscBool isaij,ismpiaij;
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat,MATSEQAIJ,&isaij));
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat,MATMPIAIJ,&ismpiaij));
    PetscCheck(isaij || ismpiaij,PETSC_COMM_WORLD,PETSC_ERR_USER,"Require (MPI)AIJ matrix type");
    if (isaij) {
      PetscCall(MatGetInfo(Gmat,MAT_LOCAL,&info));
      PetscCall(MatSeqAIJGetArray(Gmat,&avals));
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      PetscCall(MatSeqAIJRestoreArray(Gmat,&avals));
    } else {
      Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)Gmat->data;
      PetscCall(MatGetInfo(aij->A,MAT_LOCAL,&info));
      PetscCall(MatSeqAIJGetArray(aij->A,&avals));
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      PetscCall(MatSeqAIJRestoreArray(aij->A,&avals));
      PetscCall(MatGetInfo(aij->B,MAT_LOCAL,&info));
      PetscCall(MatSeqAIJGetArray(aij->B,&avals));
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      PetscCall(MatSeqAIJRestoreArray(aij->B,&avals));
    }
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[SET16],0,0,0,0));
    PetscFunctionReturn(0);
  }

  /* TODO GPU: this can be called when filter = 0 -> Probably provide MatAIJThresholdCompress that compresses the entries below a threshold?
               Also, if the matrix is symmetric, can we skip this
               operation? It can be very expensive on large matrices. */
  PetscCall(PetscObjectGetComm((PetscObject)Gmat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(MatGetOwnershipRange(Gmat, &Istart, &Iend));
  nloc = Iend - Istart;
  PetscCall(MatGetSize(Gmat, &MM, &NN));

  if (symm) {
    Mat matTrans;
    PetscCall(MatTranspose(Gmat, MAT_INITIAL_MATRIX, &matTrans));
    PetscCall(MatAXPY(Gmat, 1.0, matTrans, Gmat->structurally_symmetric ? SAME_NONZERO_PATTERN : DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&matTrans));
  }

  /* scale Gmat for all values between -1 and 1 */
  PetscCall(MatCreateVecs(Gmat, &diag, NULL));
  PetscCall(MatGetDiagonal(Gmat, diag));
  PetscCall(VecReciprocal(diag));
  PetscCall(VecSqrtAbs(diag));
  PetscCall(MatDiagonalScale(Gmat, diag, diag));
  PetscCall(VecDestroy(&diag));

  /* Determine upper bound on nonzeros needed in new filtered matrix */
  PetscCall(PetscMalloc2(nloc, &d_nnz,nloc, &o_nnz));
  for (Ii = Istart, jj = 0; Ii < Iend; Ii++, jj++) {
    PetscCall(MatGetRow(Gmat,Ii,&ncols,NULL,NULL));
    d_nnz[jj] = ncols;
    o_nnz[jj] = ncols;
    PetscCall(MatRestoreRow(Gmat,Ii,&ncols,NULL,NULL));
    if (d_nnz[jj] > nloc) d_nnz[jj] = nloc;
    if (o_nnz[jj] > (MM-nloc)) o_nnz[jj] = MM - nloc;
  }
  PetscCall(MatCreate(comm, &tGmat));
  PetscCall(MatSetSizes(tGmat,nloc,nloc,MM,MM));
  PetscCall(MatSetBlockSizes(tGmat, 1, 1));
  PetscCall(MatSetType(tGmat, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(tGmat,0,d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(tGmat,0,d_nnz,0,o_nnz));
  PetscCall(MatSetOption(tGmat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  PetscCall(PetscFree2(d_nnz,o_nnz));

  for (Ii = Istart, nnz0 = nnz1 = 0; Ii < Iend; Ii++) {
    PetscCall(MatGetRow(Gmat,Ii,&ncols,&idx,&vals));
    for (jj=0; jj<ncols; jj++,nnz0++) {
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      if (PetscRealPart(sv) > vfilter) {
        nnz1++;
        PetscCall(MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,INSERT_VALUES));
      }
    }
    PetscCall(MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals));
  }
  PetscCall(MatAssemblyBegin(tGmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(tGmat,MAT_FINAL_ASSEMBLY));
  if (symm) {
    PetscCall(MatSetOption(tGmat,MAT_SYMMETRIC,PETSC_TRUE));
  } else {
    PetscCall(MatPropagateSymmetryOptions(Gmat,tGmat));
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[SET16],0,0,0,0));

#if defined(PETSC_USE_INFO)
  {
    double t1 = (!nnz0) ? 1. : 100.*(double)nnz1/(double)nnz0, t2 = (!nloc) ? 1. : (double)nnz0/(double)nloc;
    PetscCall(PetscInfo(*a_Gmat,"\t %g%% nnz after filtering, with threshold %g, %g nnz ave. (N=%D)\n",t1,vfilter,t2,MM));
  }
#endif
  PetscCall(MatDestroy(&Gmat));
  *a_Gmat = tGmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGGetDataWithGhosts - hacks into Mat MPIAIJ so this must have size > 1

   Input Parameter:
   . Gmat - MPIAIJ matrix for scattters
   . data_sz - number of data terms per node (# cols in output)
   . data_in[nloc*data_sz] - column oriented data
   Output Parameter:
   . a_stride - numbrt of rows of output
   . a_data_out[stride*data_sz] - output data with ghosts
*/
PetscErrorCode PCGAMGGetDataWithGhosts(Mat Gmat,PetscInt data_sz,PetscReal data_in[],PetscInt *a_stride,PetscReal **a_data_out)
{
  Vec            tmp_crds;
  Mat_MPIAIJ     *mpimat = (Mat_MPIAIJ*)Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar    *data_arr;
  PetscReal      *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ));
  PetscCall(MatGetOwnershipRange(Gmat, &my0, &Iend));
  nloc      = Iend - my0;
  PetscCall(VecGetLocalSize(mpimat->lvec, &num_ghosts));
  nnodes    = num_ghosts + nloc;
  *a_stride = nnodes;
  PetscCall(MatCreateVecs(Gmat, &tmp_crds, NULL));

  PetscCall(PetscMalloc1(data_sz*nnodes, &datas));
  for (dir=0; dir<data_sz; dir++) {
    /* set local, and global */
    for (kk=0; kk<nloc; kk++) {
      PetscInt    gid = my0 + kk;
      PetscScalar crd = (PetscScalar)data_in[dir*nloc + kk]; /* col oriented */
      datas[dir*nnodes + kk] = PetscRealPart(crd);

      PetscCall(VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(tmp_crds));
    PetscCall(VecAssemblyEnd(tmp_crds));
    /* get ghost datas */
    PetscCall(VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGetArray(mpimat->lvec, &data_arr));
    for (kk=nloc,jj=0;jj<num_ghosts;kk++,jj++) datas[dir*nnodes + kk] = PetscRealPart(data_arr[jj]);
    PetscCall(VecRestoreArray(mpimat->lvec, &data_arr));
  }
  PetscCall(VecDestroy(&tmp_crds));
  *a_data_out = datas;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableCreate(PetscInt a_size, PCGAMGHashTable *a_tab)
{
  PetscInt       kk;

  PetscFunctionBegin;
  a_tab->size = a_size;
  PetscCall(PetscMalloc2(a_size, &a_tab->table,a_size, &a_tab->data));
  for (kk=0; kk<a_size; kk++) a_tab->table[kk] = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableDestroy(PCGAMGHashTable *a_tab)
{
  PetscFunctionBegin;
  PetscCall(PetscFree2(a_tab->table,a_tab->data));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableAdd(PCGAMGHashTable *a_tab, PetscInt a_key, PetscInt a_data)
{
  PetscInt kk,idx;

  PetscFunctionBegin;
  PetscCheckFalse(a_key<0,PETSC_COMM_SELF,PETSC_ERR_USER,"Negative key %D.",a_key);
  for (kk = 0, idx = GAMG_HASH(a_key); kk < a_tab->size; kk++, idx = (idx==(a_tab->size-1)) ? 0 : idx + 1) {
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

    a_tab->size = new_size;
    PetscCall(PetscMalloc2(a_tab->size, &a_tab->table,a_tab->size, &a_tab->data));
    for (kk=0;kk<a_tab->size;kk++) a_tab->table[kk] = -1;
    for (kk=0;kk<oldsize;kk++) {
      if (oldtable[kk] != -1) {
        PetscCall(PCGAMGHashTableAdd(a_tab, oldtable[kk], olddata[kk]));
       }
    }
    PetscCall(PetscFree2(oldtable,olddata));
    PetscCall(PCGAMGHashTableAdd(a_tab, a_key, a_data));
  }
  PetscFunctionReturn(0);
}
