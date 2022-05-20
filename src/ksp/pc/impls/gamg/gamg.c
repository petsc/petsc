/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include <petsc/private/matimpl.h>
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <../src/ksp/ksp/impls/cheby/chebyshevimpl.h>    /*I "petscksp.h" I*/

#if defined(PETSC_HAVE_CUDA)
  #include <cuda_runtime.h>
#endif

#if defined(PETSC_HAVE_HIP)
  #include <hip/hip_runtime.h>
#endif

PetscLogEvent petsc_gamg_setup_events[GAMG_NUM_SET];
PetscLogEvent petsc_gamg_setup_matmat_events[PETSC_MG_MAXLEVELS][3];

// #define GAMG_STAGES
#if defined(GAMG_STAGES)
static PetscLogStage gamg_stages[PETSC_MG_MAXLEVELS];
#endif

static PetscFunctionList GAMGList = NULL;
static PetscBool PCGAMGPackageInitialized;

/* ----------------------------------------------------------------------------- */
PetscErrorCode PCReset_GAMG(PC pc)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PetscFree(pc_gamg->data));
  pc_gamg->data_sz = 0;
  PetscCall(PetscFree(pc_gamg->orig_data));
  for (PetscInt level = 0; level < PETSC_MG_MAXLEVELS ; level++) {
    mg->min_eigen_DinvA[level] = 0;
    mg->max_eigen_DinvA[level] = 0;
  }
  pc_gamg->emin = 0;
  pc_gamg->emax = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCreateLevel_GAMG: create coarse op with RAP.  repartition and/or reduce number
     of active processors.

   Input Parameter:
   . pc - parameters + side effect: coarse data in 'pc_gamg->data' and
          'pc_gamg->data_sz' are changed via repartitioning/reduction.
   . Amat_fine - matrix on this fine (k) level
   . cr_bs - coarse block size
   In/Output Parameter:
   . a_P_inout - prolongation operator to the next level (k-->k-1)
   . a_nactive_proc - number of active procs
   Output Parameter:
   . a_Amat_crs - coarse matrix that is created (k-1)
*/
static PetscErrorCode PCGAMGCreateLevel_GAMG(PC pc,Mat Amat_fine,PetscInt cr_bs,Mat *a_P_inout,Mat *a_Amat_crs,PetscMPIInt *a_nactive_proc,IS * Pcolumnperm, PetscBool is_last)
{
  PC_MG           *mg         = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg    = (PC_GAMG*)mg->innerctx;
  Mat             Cmat,Pold=*a_P_inout;
  MPI_Comm        comm;
  PetscMPIInt     rank,size,new_size,nactive=*a_nactive_proc;
  PetscInt        ncrs_eq,ncrs,f_bs;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat_fine,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetBlockSize(Amat_fine, &f_bs));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PTAP],0,0,0,0));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_matmat_events[pc_gamg->current_level][1],0,0,0,0));
  PetscCall(MatPtAP(Amat_fine, Pold, MAT_INITIAL_MATRIX, 2.0, &Cmat));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_matmat_events[pc_gamg->current_level][1],0,0,0,0));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PTAP],0,0,0,0));

  if (Pcolumnperm) *Pcolumnperm = NULL;

  /* set 'ncrs' (nodes), 'ncrs_eq' (equations)*/
  PetscCall(MatGetLocalSize(Cmat, &ncrs_eq, NULL));
  if (pc_gamg->data_cell_rows>0) {
    ncrs = pc_gamg->data_sz/pc_gamg->data_cell_cols/pc_gamg->data_cell_rows;
  } else {
    PetscInt  bs;
    PetscCall(MatGetBlockSize(Cmat, &bs));
    ncrs = ncrs_eq/bs;
  }
  /* get number of PEs to make active 'new_size', reduce, can be any integer 1-P */
  if (pc_gamg->level_reduction_factors[pc_gamg->current_level] == 0 && PetscDefined(HAVE_CUDA) && pc_gamg->current_level==0) { /* 0 turns reducing to 1 process/device on; do for HIP, etc. */
#if defined(PETSC_HAVE_CUDA)
    PetscShmComm pshmcomm;
    PetscMPIInt  locrank;
    MPI_Comm     loccomm;
    PetscInt     s_nnodes,r_nnodes, new_new_size;
    cudaError_t  cerr;
    int          devCount;
    PetscCall(PetscShmCommGet(comm,&pshmcomm));
    PetscCall(PetscShmCommGetMpiShmComm(pshmcomm,&loccomm));
    PetscCallMPI(MPI_Comm_rank(loccomm, &locrank));
    s_nnodes = !locrank;
    PetscCallMPI(MPI_Allreduce(&s_nnodes,&r_nnodes,1,MPIU_INT,MPI_SUM,comm));
    PetscCheck((size%r_nnodes) == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"odd number of nodes np=%d nnodes%" PetscInt_FMT,size,r_nnodes);
    devCount = 0;
    cerr = cudaGetDeviceCount(&devCount);
    cudaGetLastError(); /* Reset the last error */
    if (cerr == cudaSuccess && devCount >= 1) { /* There are devices, else go to heuristic */
      new_new_size = r_nnodes * devCount;
      new_size = new_new_size;
      PetscCall(PetscInfo(pc,"%s: Fine grid with Cuda. %" PetscInt_FMT " nodes. Change new active set size %d --> %d (devCount=%d #nodes=%" PetscInt_FMT ")\n",((PetscObject)pc)->prefix,r_nnodes,nactive,new_size,devCount,r_nnodes));
    } else {
      PetscCall(PetscInfo(pc,"%s: With Cuda but no device. Use heuristics.",((PetscObject)pc)->prefix));
      goto HEURISTIC;
    }
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"should not be here");
#endif
  } else if (pc_gamg->level_reduction_factors[pc_gamg->current_level] > 0) {
    PetscCheck(nactive%pc_gamg->level_reduction_factors[pc_gamg->current_level] == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"odd number of active process %d wrt reduction factor %" PetscInt_FMT,nactive,pc_gamg->level_reduction_factors[pc_gamg->current_level]);
    new_size = nactive/pc_gamg->level_reduction_factors[pc_gamg->current_level];
    PetscCall(PetscInfo(pc,"%s: Manually setting reduction to %d active processes (%d/%" PetscInt_FMT ")\n",((PetscObject)pc)->prefix,new_size,nactive,pc_gamg->level_reduction_factors[pc_gamg->current_level]));
  } else if (is_last && !pc_gamg->use_parallel_coarse_grid_solver) {
    new_size = 1;
    PetscCall(PetscInfo(pc,"%s: Force coarsest grid reduction to %d active processes\n",((PetscObject)pc)->prefix,new_size));
  } else {
    PetscInt ncrs_eq_glob;
#if defined(PETSC_HAVE_CUDA)
    HEURISTIC:
#endif
    PetscCall(MatGetSize(Cmat, &ncrs_eq_glob, NULL));
    new_size = (PetscMPIInt)((float)ncrs_eq_glob/(float)pc_gamg->min_eq_proc + 0.5); /* hardwire min. number of eq/proc */
    if (!new_size) new_size = 1; /* not likely, posible? */
    else if (new_size >= nactive) new_size = nactive; /* no change, rare */
    PetscCall(PetscInfo(pc,"%s: Coarse grid reduction from %d to %d active processes\n",((PetscObject)pc)->prefix,nactive,new_size));
  }
  if (new_size==nactive) {
    *a_Amat_crs = Cmat; /* output - no repartitioning or reduction - could bail here */
    if (new_size < size) {
      /* odd case where multiple coarse grids are on one processor or no coarsening ... */
      PetscCall(PetscInfo(pc,"%s: reduced grid using same number of processors (%d) as last grid (use larger coarse grid)\n",((PetscObject)pc)->prefix,nactive));
      if (pc_gamg->cpu_pin_coarse_grids) {
        PetscCall(MatBindToCPU(*a_Amat_crs,PETSC_TRUE));
        PetscCall(MatBindToCPU(*a_P_inout,PETSC_TRUE));
      }
    }
    /* we know that the grid structure can be reused in MatPtAP */
  } else { /* reduce active processors - we know that the grid structure can NOT be reused in MatPtAP */
    PetscInt       *counts,*newproc_idx,ii,jj,kk,strideNew,*tidx,ncrs_new,ncrs_eq_new,nloc_old,expand_factor=1,rfactor=1;
    IS             is_eq_newproc,is_eq_num,is_eq_num_prim,new_eq_indices;
    PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_REDUCE],0,0,0,0));
    nloc_old = ncrs_eq/cr_bs;
    PetscCheck(ncrs_eq % cr_bs == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ncrs_eq %" PetscInt_FMT " not divisible by cr_bs %" PetscInt_FMT,ncrs_eq,cr_bs);
    /* get new_size and rfactor */
    if (pc_gamg->layout_type==PCGAMG_LAYOUT_SPREAD || !pc_gamg->repart) {
      /* find factor */
      if (new_size == 1) rfactor = size; /* don't modify */
      else {
        PetscReal best_fact = 0.;
        jj = -1;
        for (kk = 1 ; kk <= size ; kk++) {
          if (!(size%kk)) { /* a candidate */
            PetscReal nactpe = (PetscReal)size/(PetscReal)kk, fact = nactpe/(PetscReal)new_size;
            if (fact > 1.0) fact = 1./fact; /* keep fact < 1 */
            if (fact > best_fact) {
              best_fact = fact; jj = kk;
            }
          }
        }
        if (jj != -1) rfactor = jj;
        else rfactor = 1; /* a prime */
        if (pc_gamg->layout_type == PCGAMG_LAYOUT_COMPACT) expand_factor = 1;
        else expand_factor = rfactor;
      }
      new_size = size/rfactor; /* make new size one that is factor */
      if (new_size==nactive) { /* no repartitioning or reduction, bail out because nested here (rare) */
        *a_Amat_crs = Cmat;
        PetscCall(PetscInfo(pc,"%s: Finding factorable processor set stopped reduction: new_size=%d, neq(loc)=%" PetscInt_FMT "\n",((PetscObject)pc)->prefix,new_size,ncrs_eq));
        PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_REDUCE],0,0,0,0));
        PetscFunctionReturn(0);
      }
    }
    /* make 'is_eq_newproc' */
    PetscCall(PetscMalloc1(size, &counts));
    if (pc_gamg->repart) { /* Repartition Cmat_{k} and move columns of P^{k}_{k-1} and coordinates of primal part accordingly */
      Mat      adj;
      PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_REPART],0,0,0,0));
      PetscCall(PetscInfo(pc,"%s: Repartition: size (active): %d --> %d, %" PetscInt_FMT " local equations, using %s process layout\n",((PetscObject)pc)->prefix,*a_nactive_proc, new_size, ncrs_eq, (pc_gamg->layout_type==PCGAMG_LAYOUT_COMPACT) ? "compact" : "spread"));
      /* get 'adj' */
      if (cr_bs == 1) {
        PetscCall(MatConvert(Cmat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj));
      } else {
        /* make a scalar matrix to partition (no Stokes here) */
        Mat               tMat;
        PetscInt          Istart_crs,Iend_crs,ncols,jj,Ii;
        const PetscScalar *vals;
        const PetscInt    *idx;
        PetscInt          *d_nnz, *o_nnz, M, N;
        static PetscInt   llev = 0; /* ugly but just used for debugging */
        MatType           mtype;

        PetscCall(PetscMalloc2(ncrs, &d_nnz,ncrs, &o_nnz));
        PetscCall(MatGetOwnershipRange(Cmat, &Istart_crs, &Iend_crs));
        PetscCall(MatGetSize(Cmat, &M, &N));
        for (Ii = Istart_crs, jj = 0; Ii < Iend_crs; Ii += cr_bs, jj++) {
          PetscCall(MatGetRow(Cmat,Ii,&ncols,NULL,NULL));
          d_nnz[jj] = ncols/cr_bs;
          o_nnz[jj] = ncols/cr_bs;
          PetscCall(MatRestoreRow(Cmat,Ii,&ncols,NULL,NULL));
          if (d_nnz[jj] > ncrs) d_nnz[jj] = ncrs;
          if (o_nnz[jj] > (M/cr_bs-ncrs)) o_nnz[jj] = M/cr_bs-ncrs;
        }

        PetscCall(MatGetType(Amat_fine,&mtype));
        PetscCall(MatCreate(comm, &tMat));
        PetscCall(MatSetSizes(tMat, ncrs, ncrs,PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(MatSetType(tMat,mtype));
        PetscCall(MatSeqAIJSetPreallocation(tMat,0,d_nnz));
        PetscCall(MatMPIAIJSetPreallocation(tMat,0,d_nnz,0,o_nnz));
        PetscCall(PetscFree2(d_nnz,o_nnz));

        for (ii = Istart_crs; ii < Iend_crs; ii++) {
          PetscInt dest_row = ii/cr_bs;
          PetscCall(MatGetRow(Cmat,ii,&ncols,&idx,&vals));
          for (jj = 0; jj < ncols; jj++) {
            PetscInt    dest_col = idx[jj]/cr_bs;
            PetscScalar v        = 1.0;
            PetscCall(MatSetValues(tMat,1,&dest_row,1,&dest_col,&v,ADD_VALUES));
          }
          PetscCall(MatRestoreRow(Cmat,ii,&ncols,&idx,&vals));
        }
        PetscCall(MatAssemblyBegin(tMat,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(tMat,MAT_FINAL_ASSEMBLY));

        if (llev++ == -1) {
          PetscViewer viewer; char fname[32];
          PetscCall(PetscSNPrintf(fname,sizeof(fname),"part_mat_%" PetscInt_FMT ".mat",llev));
          PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&viewer);
          PetscCall(MatView(tMat, viewer));
          PetscCall(PetscViewerDestroy(&viewer));
        }
        PetscCall(MatConvert(tMat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj));
        PetscCall(MatDestroy(&tMat));
      } /* create 'adj' */

      { /* partition: get newproc_idx */
        char            prefix[256];
        const char      *pcpre;
        const PetscInt  *is_idx;
        MatPartitioning mpart;
        IS              proc_is;

        PetscCall(MatPartitioningCreate(comm, &mpart));
        PetscCall(MatPartitioningSetAdjacency(mpart, adj));
        PetscCall(PCGetOptionsPrefix(pc, &pcpre));
        PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"%spc_gamg_",pcpre ? pcpre : ""));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix));
        PetscCall(MatPartitioningSetFromOptions(mpart));
        PetscCall(MatPartitioningSetNParts(mpart, new_size));
        PetscCall(MatPartitioningApply(mpart, &proc_is));
        PetscCall(MatPartitioningDestroy(&mpart));

        /* collect IS info */
        PetscCall(PetscMalloc1(ncrs_eq, &newproc_idx));
        PetscCall(ISGetIndices(proc_is, &is_idx));
        for (kk = jj = 0 ; kk < nloc_old ; kk++) {
          for (ii = 0 ; ii < cr_bs ; ii++, jj++) {
            newproc_idx[jj] = is_idx[kk] * expand_factor; /* distribution */
          }
        }
        PetscCall(ISRestoreIndices(proc_is, &is_idx));
        PetscCall(ISDestroy(&proc_is));
      }
      PetscCall(MatDestroy(&adj));

      PetscCall(ISCreateGeneral(comm, ncrs_eq, newproc_idx, PETSC_COPY_VALUES, &is_eq_newproc));
      PetscCall(PetscFree(newproc_idx));
      PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_REPART],0,0,0,0));
    } else { /* simple aggregation of parts -- 'is_eq_newproc' */
      PetscInt targetPE;
      PetscCheck(new_size!=nactive,PETSC_COMM_SELF,PETSC_ERR_PLIB,"new_size==nactive. Should not happen");
      PetscCall(PetscInfo(pc,"%s: Number of equations (loc) %" PetscInt_FMT " with simple aggregation\n",((PetscObject)pc)->prefix,ncrs_eq));
      targetPE = (rank/rfactor)*expand_factor;
      PetscCall(ISCreateStride(comm, ncrs_eq, targetPE, 0, &is_eq_newproc));
    } /* end simple 'is_eq_newproc' */

    /*
      Create an index set from the is_eq_newproc index set to indicate the mapping TO
    */
    PetscCall(ISPartitioningToNumbering(is_eq_newproc, &is_eq_num));
    is_eq_num_prim = is_eq_num;
    /*
      Determine how many equations/vertices are assigned to each processor
    */
    PetscCall(ISPartitioningCount(is_eq_newproc, size, counts));
    ncrs_eq_new = counts[rank];
    PetscCall(ISDestroy(&is_eq_newproc));
    ncrs_new = ncrs_eq_new/cr_bs;

    PetscCall(PetscFree(counts));
    /* data movement scope -- this could be moved to subclasses so that we don't try to cram all auxiliary data into some complex abstracted thing */
    {
      Vec            src_crd, dest_crd;
      const PetscInt *idx,ndata_rows=pc_gamg->data_cell_rows,ndata_cols=pc_gamg->data_cell_cols,node_data_sz=ndata_rows*ndata_cols;
      VecScatter     vecscat;
      PetscScalar    *array;
      IS isscat;
      /* move data (for primal equations only) */
      /* Create a vector to contain the newly ordered element information */
      PetscCall(VecCreate(comm, &dest_crd));
      PetscCall(VecSetSizes(dest_crd, node_data_sz*ncrs_new, PETSC_DECIDE));
      PetscCall(VecSetType(dest_crd,VECSTANDARD)); /* this is needed! */
      /*
        There are 'ndata_rows*ndata_cols' data items per node, (one can think of the vectors of having
        a block size of ...).  Note, ISs are expanded into equation space by 'cr_bs'.
      */
      PetscCall(PetscMalloc1(ncrs*node_data_sz, &tidx));
      PetscCall(ISGetIndices(is_eq_num_prim, &idx));
      for (ii=0,jj=0; ii<ncrs; ii++) {
        PetscInt id = idx[ii*cr_bs]/cr_bs; /* get node back */
        for (kk=0; kk<node_data_sz; kk++, jj++) tidx[jj] = id*node_data_sz + kk;
      }
      PetscCall(ISRestoreIndices(is_eq_num_prim, &idx));
      PetscCall(ISCreateGeneral(comm, node_data_sz*ncrs, tidx, PETSC_COPY_VALUES, &isscat));
      PetscCall(PetscFree(tidx));
      /*
        Create a vector to contain the original vertex information for each element
      */
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, node_data_sz*ncrs, &src_crd));
      for (jj=0; jj<ndata_cols; jj++) {
        const PetscInt stride0=ncrs*pc_gamg->data_cell_rows;
        for (ii=0; ii<ncrs; ii++) {
          for (kk=0; kk<ndata_rows; kk++) {
            PetscInt    ix = ii*ndata_rows + kk + jj*stride0, jx = ii*node_data_sz + kk*ndata_cols + jj;
            PetscScalar tt = (PetscScalar)pc_gamg->data[ix];
            PetscCall(VecSetValues(src_crd, 1, &jx, &tt, INSERT_VALUES));
          }
        }
      }
      PetscCall(VecAssemblyBegin(src_crd));
      PetscCall(VecAssemblyEnd(src_crd));
      /*
        Scatter the element vertex information (still in the original vertex ordering)
        to the correct processor
      */
      PetscCall(VecScatterCreate(src_crd, NULL, dest_crd, isscat, &vecscat));
      PetscCall(ISDestroy(&isscat));
      PetscCall(VecScatterBegin(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterDestroy(&vecscat));
      PetscCall(VecDestroy(&src_crd));
      /*
        Put the element vertex data into a new allocation of the gdata->ele
      */
      PetscCall(PetscFree(pc_gamg->data));
      PetscCall(PetscMalloc1(node_data_sz*ncrs_new, &pc_gamg->data));

      pc_gamg->data_sz = node_data_sz*ncrs_new;
      strideNew        = ncrs_new*ndata_rows;

      PetscCall(VecGetArray(dest_crd, &array));
      for (jj=0; jj<ndata_cols; jj++) {
        for (ii=0; ii<ncrs_new; ii++) {
          for (kk=0; kk<ndata_rows; kk++) {
            PetscInt ix = ii*ndata_rows + kk + jj*strideNew, jx = ii*node_data_sz + kk*ndata_cols + jj;
            pc_gamg->data[ix] = PetscRealPart(array[jx]);
          }
        }
      }
      PetscCall(VecRestoreArray(dest_crd, &array));
      PetscCall(VecDestroy(&dest_crd));
    }
    /* move A and P (columns) with new layout */
    /* PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[SET13],0,0,0,0)); */
    /*
      Invert for MatCreateSubMatrix
    */
    PetscCall(ISInvertPermutation(is_eq_num, ncrs_eq_new, &new_eq_indices));
    PetscCall(ISSort(new_eq_indices)); /* is this needed? */
    PetscCall(ISSetBlockSize(new_eq_indices, cr_bs));
    if (is_eq_num != is_eq_num_prim) {
      PetscCall(ISDestroy(&is_eq_num_prim)); /* could be same as 'is_eq_num' */
    }
    if (Pcolumnperm) {
      PetscCall(PetscObjectReference((PetscObject)new_eq_indices));
      *Pcolumnperm = new_eq_indices;
    }
    PetscCall(ISDestroy(&is_eq_num));
    /* PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[SET13],0,0,0,0)); */
    /* PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[SET14],0,0,0,0)); */

    /* 'a_Amat_crs' output */
    {
      Mat       mat;
      PetscBool flg;
      PetscCall(MatCreateSubMatrix(Cmat, new_eq_indices, new_eq_indices, MAT_INITIAL_MATRIX, &mat));
      PetscCall(MatGetOption(Cmat, MAT_SPD, &flg));
      if (flg) {
        PetscCall(MatSetOption(mat, MAT_SPD,PETSC_TRUE));
      } else {
        PetscCall(MatGetOption(Cmat, MAT_HERMITIAN, &flg));
        if (flg) {
          PetscCall(MatSetOption(mat, MAT_HERMITIAN,PETSC_TRUE));
        } else {
#if !defined(PETSC_USE_COMPLEX)
          PetscCall(MatGetOption(Cmat, MAT_SYMMETRIC, &flg));
          if (flg) {
            PetscCall(MatSetOption(mat, MAT_SYMMETRIC,PETSC_TRUE));
          }
#endif
        }
      }
      *a_Amat_crs = mat;
    }
    PetscCall(MatDestroy(&Cmat));

    /* PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[SET14],0,0,0,0)); */
    /* prolongator */
    {
      IS       findices;
      PetscInt Istart,Iend;
      Mat      Pnew;

      PetscCall(MatGetOwnershipRange(Pold, &Istart, &Iend));
      /* PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[SET15],0,0,0,0)); */
      PetscCall(ISCreateStride(comm,Iend-Istart,Istart,1,&findices));
      PetscCall(ISSetBlockSize(findices,f_bs));
      PetscCall(MatCreateSubMatrix(Pold, findices, new_eq_indices, MAT_INITIAL_MATRIX, &Pnew));
      PetscCall(ISDestroy(&findices));
      PetscCall(MatSetOption(Pnew,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));

      /* PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[SET15],0,0,0,0)); */
      PetscCall(MatDestroy(a_P_inout));

      /* output - repartitioned */
      *a_P_inout = Pnew;
    }
    PetscCall(ISDestroy(&new_eq_indices));

    *a_nactive_proc = new_size; /* output */

    /* pinning on reduced grids, not a bad heuristic and optimization gets folded into process reduction optimization */
    if (pc_gamg->cpu_pin_coarse_grids) {
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
      static PetscInt llev = 2;
      PetscCall(PetscInfo(pc,"%s: Pinning level %" PetscInt_FMT " to the CPU\n",((PetscObject)pc)->prefix,llev++));
#endif
      PetscCall(MatBindToCPU(*a_Amat_crs,PETSC_TRUE));
      PetscCall(MatBindToCPU(*a_P_inout,PETSC_TRUE));
      if (1) { /* HACK: move this to MatBindCPU_MPIAIJXXX; lvec is created, need to pin it, this is done in MatSetUpMultiply_MPIAIJ. Hack */
        Mat         A = *a_Amat_crs, P = *a_P_inout;
        PetscMPIInt size;
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
        if (size > 1) {
          Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data, *p = (Mat_MPIAIJ*)P->data;
          PetscCall(VecBindToCPU(a->lvec,PETSC_TRUE));
          PetscCall(VecBindToCPU(p->lvec,PETSC_TRUE));
        }
      }
    }
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_REDUCE],0,0,0,0));
  } // processor reduce
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGSquareGraph_GAMG(PC a_pc, Mat Gmat1, Mat* Gmat2)
{
  const char     *prefix;
  char           addp[32];
  PC_MG          *mg      = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(a_pc,&prefix));
  PetscCall(PetscInfo(a_pc,"%s: Square Graph on level %" PetscInt_FMT "\n",((PetscObject)a_pc)->prefix,pc_gamg->current_level+1));
  PetscCall(MatProductCreate(Gmat1,Gmat1,NULL,Gmat2));
  PetscCall(MatSetOptionsPrefix(*Gmat2,prefix));
  PetscCall(PetscSNPrintf(addp,sizeof(addp),"pc_gamg_square_%" PetscInt_FMT "_",pc_gamg->current_level));
  PetscCall(MatAppendOptionsPrefix(*Gmat2,addp));
  if ((*Gmat2)->structurally_symmetric) {
    PetscCall(MatProductSetType(*Gmat2,MATPRODUCT_AB));
  } else {
    PetscCall(MatSetOption(Gmat1,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));
    PetscCall(MatProductSetType(*Gmat2,MATPRODUCT_AtB));
  }
  PetscCall(MatProductSetFromOptions(*Gmat2));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_matmat_events[pc_gamg->current_level][0],0,0,0,0));
  PetscCall(MatProductSymbolic(*Gmat2));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_matmat_events[pc_gamg->current_level][0],0,0,0,0));
  PetscCall(MatProductClear(*Gmat2));
  /* we only need the sparsity, cheat and tell PETSc the matrix has been assembled */
  (*Gmat2)->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_GAMG - Prepares for the use of the GAMG preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

*/
PetscErrorCode PCSetUp_GAMG(PC pc)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  Mat            Pmat     = pc->pmat;
  PetscInt       fine_level,level,level1,bs,M,N,qq,lidx,nASMBlocksArr[PETSC_MG_MAXLEVELS];
  MPI_Comm       comm;
  PetscMPIInt    rank,size,nactivepe;
  Mat            Aarr[PETSC_MG_MAXLEVELS],Parr[PETSC_MG_MAXLEVELS];
  IS             *ASMLocalIDsArr[PETSC_MG_MAXLEVELS];
  PetscLogDouble nnz0=0.,nnztot=0.;
  MatInfo        info;
  PetscBool      is_last = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_SETUP],0,0,0,0));
  if (pc->setupcalled) {
    if (!pc_gamg->reuse_prol || pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* reset everything */
      PetscCall(PCReset_MG(pc));
      pc->setupcalled = 0;
    } else {
      PC_MG_Levels **mglevels = mg->levels;
      /* just do Galerkin grids */
      Mat          B,dA,dB;
      if (pc_gamg->Nlevels > 1) {
        PetscInt gl;
        /* currently only handle case where mat and pmat are the same on coarser levels */
        PetscCall(KSPGetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,&dA,&dB));
        /* (re)set to get dirty flag */
        PetscCall(KSPSetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,dA,dB));

        for (level=pc_gamg->Nlevels-2,gl=0; level>=0; level--,gl++) {
          MatReuse reuse = MAT_INITIAL_MATRIX ;
#if defined(GAMG_STAGES)
          PetscCall(PetscLogStagePush(gamg_stages[gl]));
#endif
          /* matrix structure can change from repartitioning or process reduction but don't know if we have process reduction here. Should fix */
          PetscCall(KSPGetOperators(mglevels[level]->smoothd,NULL,&B));
          if (B->product) {
            if (B->product->A == dB && B->product->B == mglevels[level+1]->interpolate) {
              reuse = MAT_REUSE_MATRIX;
            }
          }
          if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDestroy(&mglevels[level]->A));
          if (reuse == MAT_REUSE_MATRIX) {
            PetscCall(PetscInfo(pc,"%s: RAP after first solve, reuse matrix level %" PetscInt_FMT "\n",((PetscObject)pc)->prefix,level));
          } else {
            PetscCall(PetscInfo(pc,"%s: RAP after first solve, new matrix level %" PetscInt_FMT "\n",((PetscObject)pc)->prefix,level));
          }
          PetscCall(PetscLogEventBegin(petsc_gamg_setup_matmat_events[gl][1],0,0,0,0));
          PetscCall(MatPtAP(dB,mglevels[level+1]->interpolate,reuse,PETSC_DEFAULT,&B));
          PetscCall(PetscLogEventEnd(petsc_gamg_setup_matmat_events[gl][1],0,0,0,0));
          if (reuse == MAT_INITIAL_MATRIX) mglevels[level]->A = B;
          PetscCall(KSPSetOperators(mglevels[level]->smoothd,B,B));
          dB   = B;
#if defined(GAMG_STAGES)
          PetscCall(PetscLogStagePop());
#endif
        }
      }

      PetscCall(PCSetUp_MG(pc));
      PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_SETUP],0,0,0,0));
      PetscFunctionReturn(0);
    }
  }

  if (!pc_gamg->data) {
    if (pc_gamg->orig_data) {
      PetscCall(MatGetBlockSize(Pmat, &bs));
      PetscCall(MatGetLocalSize(Pmat, &qq, NULL));

      pc_gamg->data_sz        = (qq/bs)*pc_gamg->orig_data_cell_rows*pc_gamg->orig_data_cell_cols;
      pc_gamg->data_cell_rows = pc_gamg->orig_data_cell_rows;
      pc_gamg->data_cell_cols = pc_gamg->orig_data_cell_cols;

      PetscCall(PetscMalloc1(pc_gamg->data_sz, &pc_gamg->data));
      for (qq=0; qq<pc_gamg->data_sz; qq++) pc_gamg->data[qq] = pc_gamg->orig_data[qq];
    } else {
      PetscCheck(pc_gamg->ops->createdefaultdata,comm,PETSC_ERR_PLIB,"'createdefaultdata' not set(?) need to support NULL data");
      PetscCall(pc_gamg->ops->createdefaultdata(pc,Pmat));
    }
  }

  /* cache original data for reuse */
  if (!pc_gamg->orig_data && (PetscBool)(!pc_gamg->reuse_prol)) {
    PetscCall(PetscMalloc1(pc_gamg->data_sz, &pc_gamg->orig_data));
    for (qq=0; qq<pc_gamg->data_sz; qq++) pc_gamg->orig_data[qq] = pc_gamg->data[qq];
    pc_gamg->orig_data_cell_rows = pc_gamg->data_cell_rows;
    pc_gamg->orig_data_cell_cols = pc_gamg->data_cell_cols;
  }

  /* get basic dims */
  PetscCall(MatGetBlockSize(Pmat, &bs));
  PetscCall(MatGetSize(Pmat, &M, &N));

  PetscCall(MatGetInfo(Pmat,MAT_GLOBAL_SUM,&info)); /* global reduction */
  nnz0   = info.nz_used;
  nnztot = info.nz_used;
  PetscCall(PetscInfo(pc,"%s: level %" PetscInt_FMT ") N=0, n data rows=%" PetscInt_FMT ", n data cols=%" PetscInt_FMT ", nnz/row (ave)=%" PetscInt_FMT ", np=%d\n",((PetscObject)pc)->prefix,M,pc_gamg->data_cell_rows,pc_gamg->data_cell_cols,(PetscInt)(nnz0/(PetscReal)M+0.5),size));

  /* Get A_i and R_i */
  for (level=0, Aarr[0]=Pmat, nactivepe = size; level < (pc_gamg->Nlevels-1) && (!level || M>pc_gamg->coarse_eq_limit); level++) {
    pc_gamg->current_level = level;
    PetscCheck(level < PETSC_MG_MAXLEVELS,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Too many levels %" PetscInt_FMT,level);
    level1 = level + 1;
#if defined(GAMG_STAGES)
    if (!gamg_stages[level]) {
      char     str[32];
      sprintf(str,"GAMG Level %d",(int)level);
      PetscCall(PetscLogStageRegister(str, &gamg_stages[level]));
    }
    PetscCall(PetscLogStagePush(gamg_stages[level]));
#endif
    { /* construct prolongator */
      Mat              Gmat;
      PetscCoarsenData *agg_lists;
      Mat              Prol11;

      PetscCall(pc_gamg->ops->graph(pc,Aarr[level], &Gmat));
      PetscCall(pc_gamg->ops->coarsen(pc, &Gmat, &agg_lists));
      PetscCall(pc_gamg->ops->prolongator(pc,Aarr[level],Gmat,agg_lists,&Prol11));

      /* could have failed to create new level */
      if (Prol11) {
        const char *prefix;
        char       addp[32];

        /* get new block size of coarse matrices */
        PetscCall(MatGetBlockSizes(Prol11, NULL, &bs));

        if (pc_gamg->ops->optprolongator) {
          /* smooth */
          PetscCall(pc_gamg->ops->optprolongator(pc, Aarr[level], &Prol11));
        }

        if (pc_gamg->use_aggs_in_asm) {
          PetscInt bs;
          PetscCall(MatGetBlockSizes(Prol11, &bs, NULL)); // not timed directly, ugly, could remove, but good ASM method
          PetscCall(PetscCDGetASMBlocks(agg_lists, bs, Gmat, &nASMBlocksArr[level], &ASMLocalIDsArr[level]));
        }

        PetscCall(PCGetOptionsPrefix(pc,&prefix));
        PetscCall(MatSetOptionsPrefix(Prol11,prefix));
        PetscCall(PetscSNPrintf(addp,sizeof(addp),"pc_gamg_prolongator_%d_",(int)level));
        PetscCall(MatAppendOptionsPrefix(Prol11,addp));
        /* Always generate the transpose with CUDA
           Such behaviour can be adapted with -pc_gamg_prolongator_ prefixed options */
        PetscCall(MatSetOption(Prol11,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));
        PetscCall(MatSetFromOptions(Prol11));
        Parr[level1] = Prol11;
      } else Parr[level1] = NULL; /* failed to coarsen */

      PetscCall(MatDestroy(&Gmat));
      PetscCall(PetscCDDestroy(agg_lists));
    } /* construct prolongator scope */
    if (!level) Aarr[0] = Pmat; /* use Pmat for finest level setup */
    if (!Parr[level1]) { /* failed to coarsen */
      PetscCall(PetscInfo(pc,"%s: Stop gridding, level %" PetscInt_FMT "\n",((PetscObject)pc)->prefix,level));
#if defined(GAMG_STAGES)
      PetscCall(PetscLogStagePop());
#endif
      break;
    }
    PetscCall(MatGetSize(Parr[level1], &M, &N)); /* N is next M, a loop test variables */
    PetscCheck(!is_last,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Is last ?");
    if (N <= pc_gamg->coarse_eq_limit) is_last = PETSC_TRUE;
    if (level1 == pc_gamg->Nlevels-1) is_last = PETSC_TRUE;
    PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_LEVEL],0,0,0,0));
    PetscCall(pc_gamg->ops->createlevel(pc, Aarr[level], bs, &Parr[level1], &Aarr[level1], &nactivepe, NULL, is_last));
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_LEVEL],0,0,0,0));

    PetscCall(MatGetSize(Aarr[level1], &M, &N)); /* M is loop test variables */
    PetscCall(MatGetInfo(Aarr[level1], MAT_GLOBAL_SUM, &info));
    nnztot += info.nz_used;
    PetscCall(PetscInfo(pc,"%s: %" PetscInt_FMT ") N=%" PetscInt_FMT ", n data cols=%" PetscInt_FMT ", nnz/row (ave)=%" PetscInt_FMT ", %d active pes\n",((PetscObject)pc)->prefix,level1,M,pc_gamg->data_cell_cols,(PetscInt)(info.nz_used/(PetscReal)M),nactivepe));

#if defined(GAMG_STAGES)
    PetscCall(PetscLogStagePop());
#endif
    /* stop if one node or one proc -- could pull back for singular problems */
    if ((pc_gamg->data_cell_cols && M/pc_gamg->data_cell_cols < 2) || (!pc_gamg->data_cell_cols && M/bs < 2)) {
      PetscCall(PetscInfo(pc,"%s: HARD stop of coarsening on level %" PetscInt_FMT ".  Grid too small: %" PetscInt_FMT " block nodes\n",((PetscObject)pc)->prefix,level,M/bs));
      level++;
      break;
    }
  } /* levels */
  PetscCall(PetscFree(pc_gamg->data));

  PetscCall(PetscInfo(pc,"%s: %" PetscInt_FMT " levels, grid complexity = %g\n",((PetscObject)pc)->prefix,level+1,nnztot/nnz0));
  pc_gamg->Nlevels = level + 1;
  fine_level       = level;
  PetscCall(PCMGSetLevels(pc,pc_gamg->Nlevels,NULL));

  if (pc_gamg->Nlevels > 1) { /* don't setup MG if one level */

    /* set default smoothers & set operators */
    for (lidx = 1, level = pc_gamg->Nlevels-2; lidx <= fine_level; lidx++, level--) {
      KSP smoother;
      PC  subpc;

      PetscCall(PCMGGetSmoother(pc, lidx, &smoother));
      PetscCall(KSPGetPC(smoother, &subpc));

      PetscCall(KSPSetNormType(smoother, KSP_NORM_NONE));
      /* set ops */
      PetscCall(KSPSetOperators(smoother, Aarr[level], Aarr[level]));
      PetscCall(PCMGSetInterpolation(pc, lidx, Parr[level+1]));

      /* set defaults */
      PetscCall(KSPSetType(smoother, KSPCHEBYSHEV));

      /* set blocks for ASM smoother that uses the 'aggregates' */
      if (pc_gamg->use_aggs_in_asm) {
        PetscInt sz;
        IS       *iss;

        sz   = nASMBlocksArr[level];
        iss  = ASMLocalIDsArr[level];
        PetscCall(PCSetType(subpc, PCASM));
        PetscCall(PCASMSetOverlap(subpc, 0));
        PetscCall(PCASMSetType(subpc,PC_ASM_BASIC));
        if (!sz) {
          IS       is;
          PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_COPY_VALUES, &is));
          PetscCall(PCASMSetLocalSubdomains(subpc, 1, NULL, &is));
          PetscCall(ISDestroy(&is));
        } else {
          PetscInt kk;
          PetscCall(PCASMSetLocalSubdomains(subpc, sz, NULL, iss));
          for (kk=0; kk<sz; kk++) {
            PetscCall(ISDestroy(&iss[kk]));
          }
          PetscCall(PetscFree(iss));
        }
        ASMLocalIDsArr[level] = NULL;
        nASMBlocksArr[level]  = 0;
      } else {
        PetscCall(PCSetType(subpc, PCJACOBI));
      }
    }
    {
      /* coarse grid */
      KSP smoother,*k2; PC subpc,pc2; PetscInt ii,first;
      Mat Lmat = Aarr[(level=pc_gamg->Nlevels-1)]; lidx = 0;

      PetscCall(PCMGGetSmoother(pc, lidx, &smoother));
      PetscCall(KSPSetOperators(smoother, Lmat, Lmat));
      if (!pc_gamg->use_parallel_coarse_grid_solver) {
        PetscCall(KSPSetNormType(smoother, KSP_NORM_NONE));
        PetscCall(KSPGetPC(smoother, &subpc));
        PetscCall(PCSetType(subpc, PCBJACOBI));
        PetscCall(PCSetUp(subpc));
        PetscCall(PCBJacobiGetSubKSP(subpc,&ii,&first,&k2));
        PetscCheck(ii == 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ii %" PetscInt_FMT " is not one",ii);
        PetscCall(KSPGetPC(k2[0],&pc2));
        PetscCall(PCSetType(pc2, PCLU));
        PetscCall(PCFactorSetShiftType(pc2,MAT_SHIFT_INBLOCKS));
        PetscCall(KSPSetTolerances(k2[0],PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1));
        PetscCall(KSPSetType(k2[0], KSPPREONLY));
      }
    }

    /* should be called in PCSetFromOptions_GAMG(), but cannot be called prior to PCMGSetLevels() */
    PetscObjectOptionsBegin((PetscObject)pc);
    PetscCall(PCSetFromOptions_MG(PetscOptionsObject,pc));
    PetscOptionsEnd();
    PetscCall(PCMGSetGalerkin(pc,PC_MG_GALERKIN_EXTERNAL));

    /* setup cheby eigen estimates from SA */
    if (pc_gamg->use_sa_esteig) {
      for (lidx = 1, level = pc_gamg->Nlevels-2; level >= 0 ; lidx++, level--) {
        KSP       smoother;
        PetscBool ischeb;

        PetscCall(PCMGGetSmoother(pc, lidx, &smoother));
        PetscCall(PetscObjectTypeCompare((PetscObject)smoother,KSPCHEBYSHEV,&ischeb));
        if (ischeb) {
          KSP_Chebyshev *cheb = (KSP_Chebyshev*)smoother->data;

          // The command line will override these settings because KSPSetFromOptions is called in PCSetUp_MG
          if (mg->max_eigen_DinvA[level] > 0) {
            // SA uses Jacobi for P; we use SA estimates if the smoother is also Jacobi or if the user explicitly requested it.
            // TODO: This should test whether it's the same Jacobi variant (DIAG, ROWSUM, etc.)
            PetscReal emax,emin;

            emin = mg->min_eigen_DinvA[level];
            emax = mg->max_eigen_DinvA[level];
            PetscCall(PetscInfo(pc,"%s: PCSetUp_GAMG: call KSPChebyshevSetEigenvalues on level %" PetscInt_FMT " (N=%" PetscInt_FMT ") with emax = %g emin = %g\n",((PetscObject)pc)->prefix,level,Aarr[level]->rmap->N,(double)emax,(double)emin));
            cheb->emin_provided = emin;
            cheb->emax_provided = emax;
          }
        }
      }
    }

    PetscCall(PCSetUp_MG(pc));

    /* clean up */
    for (level=1; level<pc_gamg->Nlevels; level++) {
      PetscCall(MatDestroy(&Parr[level]));
      PetscCall(MatDestroy(&Aarr[level]));
    }
  } else {
    KSP smoother;

    PetscCall(PetscInfo(pc,"%s: One level solver used (system is seen as DD). Using default solver.\n",((PetscObject)pc)->prefix));
    PetscCall(PCMGGetSmoother(pc, 0, &smoother));
    PetscCall(KSPSetOperators(smoother, Aarr[0], Aarr[0]));
    PetscCall(KSPSetType(smoother, KSPPREONLY));
    PetscCall(PCSetUp_MG(pc));
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_SETUP],0,0,0,0));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------- */
/*
 PCDestroy_GAMG - Destroys the private context for the GAMG preconditioner
   that was created with PCCreate_GAMG().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
PetscErrorCode PCDestroy_GAMG(PC pc)
{
  PC_MG          *mg     = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg= (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PCReset_GAMG(pc));
  if (pc_gamg->ops->destroy) {
    PetscCall((*pc_gamg->ops->destroy)(pc));
  }
  PetscCall(PetscFree(pc_gamg->ops));
  PetscCall(PetscFree(pc_gamg->gamg_type_name));
  PetscCall(PetscFree(pc_gamg));
  PetscCall(PCDestroy_MG(pc));
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetProcEqLim - Set number of equations to aim for per process on the coarse grids via processor reduction.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - the number of equations

   Options Database Key:
.  -pc_gamg_process_eq_limit <limit> - set the limit

   Notes:
    GAMG will reduce the number of MPI processes used directly on the coarse grids so that there are around <limit> equations on each process
    that has degrees of freedom

   Level: intermediate

.seealso: `PCGAMGSetCoarseEqLim()`, `PCGAMGSetRankReductionFactors()`
@*/
PetscErrorCode  PCGAMGSetProcEqLim(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetProcEqLim_C",(PC,PetscInt),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetProcEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if (n>0) pc_gamg->min_eq_proc = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetCoarseEqLim - Set maximum number of equations on coarsest grid.

 Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - maximum number of equations to aim for

   Options Database Key:
.  -pc_gamg_coarse_eq_limit <limit> - set the limit

   Notes: For example -pc_gamg_coarse_eq_limit 1000 will stop coarsening once the coarse grid
     has less than 1000 unknowns.

   Level: intermediate

.seealso: `PCGAMGSetProcEqLim()`, `PCGAMGSetRankReductionFactors()`
@*/
PetscErrorCode PCGAMGSetCoarseEqLim(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetCoarseEqLim_C",(PC,PetscInt),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetCoarseEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if (n>0) pc_gamg->coarse_eq_limit = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetRepartition - Repartition the degrees of freedom across the processors on the coarser grids

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -pc_gamg_repartition <true,false> - turn on the repartitioning

   Notes:
    this will generally improve the loading balancing of the work on each level

   Level: intermediate

@*/
PetscErrorCode PCGAMGSetRepartition(PC pc, PetscBool n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetRepartition_C",(PC,PetscBool),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetRepartition_GAMG(PC pc, PetscBool n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->repart = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetUseSAEstEig - Use eigen estimate from smoothed aggregation for Chebyshev smoother

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - number of its

   Options Database Key:
.  -pc_gamg_use_sa_esteig <true,false> - use the eigen estimate

   Notes:
   Smoothed aggregation constructs the smoothed prolongator $P = (I - \omega D^{-1} A) T$ where $T$ is the tentative prolongator and $D$ is the diagonal of $A$.
   Eigenvalue estimates (based on a few CG or GMRES iterations) are computed to choose $\omega$ so that this is a stable smoothing operation.
   If Chebyshev with Jacobi (diagonal) preconditioning is used for smoothing, then the eigenvalue estimates can be reused.
   This option is only used when the smoother uses Jacobi, and should be turned off if a different PCJacobiType is used.
   It became default in PETSc 3.17.

   Level: advanced

.seealso: `KSPChebyshevSetEigenvalues()`, `KSPChebyshevEstEigSet()`
@*/
PetscErrorCode PCGAMGSetUseSAEstEig(PC pc, PetscBool n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetUseSAEstEig_C",(PC,PetscBool),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetUseSAEstEig_GAMG(PC pc, PetscBool n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->use_sa_esteig = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetEigenvalues - Set eigenvalues

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  emax - max eigenvalue
.  emin - min eigenvalue

   Options Database Key:
.  -pc_gamg_eigenvalues <emin,emax> - estimates of the eigenvalues

   Level: intermediate

.seealso: `PCGAMGSetUseSAEstEig()`
@*/
PetscErrorCode PCGAMGSetEigenvalues(PC pc, PetscReal emax,PetscReal emin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetEigenvalues_C",(PC,PetscReal,PetscReal),(pc,emax,emin));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetEigenvalues_GAMG(PC pc,PetscReal emax,PetscReal emin)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  PetscCheck(emax > emin,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Maximum eigenvalue must be larger than minimum: max %g min %g",(double)emax,(double)emin);
  PetscCheck(emax*emin > 0.0,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Both eigenvalues must be of the same sign: max %g min %g",(double)emax,(double)emin);
  pc_gamg->emax = emax;
  pc_gamg->emin = emin;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetReuseInterpolation - Reuse prolongation when rebuilding algebraic multigrid preconditioner

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -pc_gamg_reuse_interpolation <true,false> - reuse the previous interpolation

   Level: intermediate

   Notes:
    May negatively affect the convergence rate of the method on new matrices if the matrix entries change a great deal, but allows
    rebuilding the preconditioner quicker.

@*/
PetscErrorCode PCGAMGSetReuseInterpolation(PC pc, PetscBool n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetReuseInterpolation_C",(PC,PetscBool),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetReuseInterpolation_GAMG(PC pc, PetscBool n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->reuse_prol = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGASMSetUseAggs - Have the PCGAMG smoother on each level use the aggregates defined by the coarsening process as the subdomains for the additive Schwarz preconditioner.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - PETSC_TRUE to use aggregates, PETSC_FALSE to not

   Options Database Key:
.  -pc_gamg_asm_use_agg <true,false> - use aggregates to define the additive Schwarz subdomains

   Level: intermediate

@*/
PetscErrorCode PCGAMGASMSetUseAggs(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGASMSetUseAggs_C",(PC,PetscBool),(pc,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGASMSetUseAggs_GAMG(PC pc, PetscBool flg)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->use_aggs_in_asm = flg;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetUseParallelCoarseGridSolve - allow a parallel coarse grid solver

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - PETSC_TRUE to not force coarse grid onto one processor

   Options Database Key:
.  -pc_gamg_use_parallel_coarse_grid_solver - use a parallel coarse grid direct solver

   Level: intermediate

.seealso: `PCGAMGSetCoarseGridLayoutType()`, `PCGAMGSetCpuPinCoarseGrids()`
@*/
PetscErrorCode PCGAMGSetUseParallelCoarseGridSolve(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetUseParallelCoarseGridSolve_C",(PC,PetscBool),(pc,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetUseParallelCoarseGridSolve_GAMG(PC pc, PetscBool flg)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->use_parallel_coarse_grid_solver = flg;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetCpuPinCoarseGrids - pin reduced grids to CPU

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - PETSC_TRUE to pin coarse grids to CPU

   Options Database Key:
.  -pc_gamg_cpu_pin_coarse_grids - pin the coarse grids to the CPU

   Level: intermediate

.seealso: `PCGAMGSetCoarseGridLayoutType()`, `PCGAMGSetUseParallelCoarseGridSolve()`
@*/
PetscErrorCode PCGAMGSetCpuPinCoarseGrids(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetCpuPinCoarseGrids_C",(PC,PetscBool),(pc,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetCpuPinCoarseGrids_GAMG(PC pc, PetscBool flg)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->cpu_pin_coarse_grids = flg;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetCoarseGridLayoutType - place coarse grids on processors with natural order (compact type)

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - Layout type

   Options Database Key:
.  -pc_gamg_coarse_grid_layout_type - place the coarse grids with natural ordering

   Level: intermediate

.seealso: `PCGAMGSetUseParallelCoarseGridSolve()`, `PCGAMGSetCpuPinCoarseGrids()`
@*/
PetscErrorCode PCGAMGSetCoarseGridLayoutType(PC pc, PCGAMGLayoutType flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetCoarseGridLayoutType_C",(PC,PCGAMGLayoutType),(pc,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetCoarseGridLayoutType_GAMG(PC pc, PCGAMGLayoutType flg)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->layout_type = flg;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetNlevels -  Sets the maximum number of levels PCGAMG will use

   Not collective on PC

   Input Parameters:
+  pc - the preconditioner
-  n - the maximum number of levels to use

   Options Database Key:
.  -pc_mg_levels <n> - set the maximum number of levels to allow

   Level: intermediate

@*/
PetscErrorCode PCGAMGSetNlevels(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetNlevels_C",(PC,PetscInt),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetNlevels_GAMG(PC pc, PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->Nlevels = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetThreshold - Relative threshold to use for dropping edges in aggregation graph

   Not collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  v - array of threshold values for finest n levels; 0.0 means keep all nonzero entries in the graph; negative means keep even zero entries in the graph
-  n - number of threshold values provided in array

   Options Database Key:
.  -pc_gamg_threshold <threshold> - the threshold to drop edges

   Notes:
    Increasing the threshold decreases the rate of coarsening. Conversely reducing the threshold increases the rate of coarsening (aggressive coarsening) and thereby reduces the complexity of the coarse grids, and generally results in slower solver converge rates. Reducing coarse grid complexity reduced the complexity of Galerkin coarse grid construction considerably.
    Before coarsening or aggregating the graph, GAMG removes small values from the graph with this threshold, and thus reducing the coupling in the graph and a different (perhaps better) coarser set of points.

    If n is less than the total number of coarsenings (see PCGAMGSetNlevels()), then threshold scaling (see PCGAMGSetThresholdScale()) is used for each successive coarsening.
    In this case, PCGAMGSetThresholdScale() must be called before PCGAMGSetThreshold().
    If n is greater than the total number of levels, the excess entries in threshold will not be used.

   Level: intermediate

.seealso: `PCGAMGFilterGraph()`, `PCGAMGSetSquareGraph()`
@*/
PetscErrorCode PCGAMGSetThreshold(PC pc, PetscReal v[], PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidRealPointer(v,2);
  PetscTryMethod(pc,"PCGAMGSetThreshold_C",(PC,PetscReal[],PetscInt),(pc,v,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetThreshold_GAMG(PC pc, PetscReal v[], PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<PetscMin(n,PETSC_MG_MAXLEVELS); i++) pc_gamg->threshold[i] = v[i];
  for (; i<PETSC_MG_MAXLEVELS; i++) pc_gamg->threshold[i] = pc_gamg->threshold[i-1]*pc_gamg->threshold_scale;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetRankReductionFactors - Set manual schedule for process reduction on coarse grids

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  v - array of reduction factors. 0 for fist value forces a reduction to one process/device on first level in Cuda
-  n - number of values provided in array

   Options Database Key:
.  -pc_gamg_rank_reduction_factors <factors> - provide the schedule

   Level: intermediate

.seealso: `PCGAMGSetProcEqLim()`, `PCGAMGSetCoarseEqLim()`
@*/
PetscErrorCode PCGAMGSetRankReductionFactors(PC pc, PetscInt v[], PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(v,2);
  PetscTryMethod(pc,"PCGAMGSetRankReductionFactors_C",(PC,PetscInt[],PetscInt),(pc,v,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetRankReductionFactors_GAMG(PC pc, PetscInt v[], PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<PetscMin(n,PETSC_MG_MAXLEVELS); i++) pc_gamg->level_reduction_factors[i] = v[i];
  for (; i<PETSC_MG_MAXLEVELS; i++) pc_gamg->level_reduction_factors[i] = -1; /* 0 stop putting one process/device on first level */
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetThresholdScale - Relative threshold reduction at each level

   Not collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  scale - the threshold value reduction, usually < 1.0

   Options Database Key:
.  -pc_gamg_threshold_scale <v> - set the relative threshold reduction on each level

   Notes:
   The initial threshold (for an arbitrary number of levels starting from the finest) can be set with PCGAMGSetThreshold().
   This scaling is used for each subsequent coarsening, but must be called before PCGAMGSetThreshold().

   Level: advanced

.seealso: `PCGAMGSetThreshold()`
@*/
PetscErrorCode PCGAMGSetThresholdScale(PC pc, PetscReal v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetThresholdScale_C",(PC,PetscReal),(pc,v));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetThresholdScale_GAMG(PC pc, PetscReal v)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscFunctionBegin;
  pc_gamg->threshold_scale = v;
  PetscFunctionReturn(0);
}

/*@C
   PCGAMGSetType - Set solution method

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - PCGAMGAGG, PCGAMGGEO, or PCGAMGCLASSICAL

   Options Database Key:
.  -pc_gamg_type <agg,geo,classical> - type of algebraic multigrid to apply

   Level: intermediate

.seealso: `PCGAMGGetType()`, `PCGAMG`, `PCGAMGType`
@*/
PetscErrorCode PCGAMGSetType(PC pc, PCGAMGType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCGAMGSetType_C",(PC,PCGAMGType),(pc,type));
  PetscFunctionReturn(0);
}

/*@C
   PCGAMGGetType - Get solution method

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  type - the type of algorithm used

   Level: intermediate

.seealso: `PCGAMGSetType()`, `PCGAMGType`
@*/
PetscErrorCode PCGAMGGetType(PC pc, PCGAMGType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCGAMGGetType_C",(PC,PCGAMGType*),(pc,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGGetType_GAMG(PC pc, PCGAMGType *type)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  *type = pc_gamg->type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetType_GAMG(PC pc, PCGAMGType type)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode (*r)(PC);

  PetscFunctionBegin;
  pc_gamg->type = type;
  PetscCall(PetscFunctionListFind(GAMGList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown GAMG type %s given",type);
  if (pc_gamg->ops->destroy) {
    PetscCall((*pc_gamg->ops->destroy)(pc));
    PetscCall(PetscMemzero(pc_gamg->ops,sizeof(struct _PCGAMGOps)));
    pc_gamg->ops->createlevel = PCGAMGCreateLevel_GAMG;
    /* cleaning up common data in pc_gamg - this should disapear someday */
    pc_gamg->data_cell_cols = 0;
    pc_gamg->data_cell_rows = 0;
    pc_gamg->orig_data_cell_cols = 0;
    pc_gamg->orig_data_cell_rows = 0;
    PetscCall(PetscFree(pc_gamg->data));
    pc_gamg->data_sz = 0;
  }
  PetscCall(PetscFree(pc_gamg->gamg_type_name));
  PetscCall(PetscStrallocpy(type,&pc_gamg->gamg_type_name));
  PetscCall((*r)(pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_GAMG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscReal       gc=0, oc=0;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer,"    GAMG specific options\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer,"      Threshold for dropping small values in graph on each level ="));
  for (PetscInt i=0;i<mg->nlevels; i++) PetscCall(PetscViewerASCIIPrintf(viewer," %g",(double)pc_gamg->threshold[i]));
  PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer,"      Threshold scaling factor for each level not specified = %g\n",(double)pc_gamg->threshold_scale));
  if (pc_gamg->use_aggs_in_asm) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"      Using aggregates from coarsening process to define subdomains for PCASM\n"));
  }
  if (pc_gamg->use_parallel_coarse_grid_solver) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"      Using parallel coarse grid solver (all coarse grid equations not put on one process)\n"));
  }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  if (pc_gamg->cpu_pin_coarse_grids) {
    /* PetscCall(PetscViewerASCIIPrintf(viewer,"      Pinning coarse grids to the CPU)\n")); */
  }
#endif
  /* if (pc_gamg->layout_type==PCGAMG_LAYOUT_COMPACT) { */
  /*   PetscCall(PetscViewerASCIIPrintf(viewer,"      Put reduced grids on processes in natural order (ie, 0,1,2...)\n")); */
  /* } else { */
  /*   PetscCall(PetscViewerASCIIPrintf(viewer,"      Put reduced grids on whole machine (ie, 0,1*f,2*f...,np-f)\n")); */
  /* } */
  if (pc_gamg->ops->view) {
    PetscCall((*pc_gamg->ops->view)(pc,viewer));
  }
  PetscCall(PCMGGetGridComplexity(pc,&gc,&oc));
  PetscCall(PetscViewerASCIIPrintf(viewer,"      Complexity:    grid = %g    operator = %g\n",(double)gc,(double)oc));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_GAMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscBool      flag;
  MPI_Comm       comm;
  char           prefix[256],tname[32];
  PetscInt       i,n;
  const char     *pcpre;
  static const char *LayoutTypes[] = {"compact","spread","PCGAMGLayoutType","PC_GAMG_LAYOUT",NULL};
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
  PetscOptionsHeadBegin(PetscOptionsObject,"GAMG options");
  PetscCall(PetscOptionsFList("-pc_gamg_type","Type of AMG method","PCGAMGSetType",GAMGList, pc_gamg->gamg_type_name, tname, sizeof(tname), &flag));
  if (flag) {
    PetscCall(PCGAMGSetType(pc,tname));
  }
  PetscCall(PetscOptionsBool("-pc_gamg_repartition","Repartion coarse grids","PCGAMGSetRepartition",pc_gamg->repart,&pc_gamg->repart,NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_use_sa_esteig","Use eigen estimate from Smoothed aggregation for smoother","PCGAMGSetUseSAEstEig",pc_gamg->use_sa_esteig,&pc_gamg->use_sa_esteig,NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_reuse_interpolation","Reuse prolongation operator","PCGAMGReuseInterpolation",pc_gamg->reuse_prol,&pc_gamg->reuse_prol,NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_asm_use_agg","Use aggregation aggregates for ASM smoother","PCGAMGASMSetUseAggs",pc_gamg->use_aggs_in_asm,&pc_gamg->use_aggs_in_asm,NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_use_parallel_coarse_grid_solver","Use parallel coarse grid solver (otherwise put last grid on one process)","PCGAMGSetUseParallelCoarseGridSolve",pc_gamg->use_parallel_coarse_grid_solver,&pc_gamg->use_parallel_coarse_grid_solver,NULL));
  PetscCall(PetscOptionsBool("-pc_gamg_cpu_pin_coarse_grids","Pin coarse grids to the CPU","PCGAMGSetCpuPinCoarseGrids",pc_gamg->cpu_pin_coarse_grids,&pc_gamg->cpu_pin_coarse_grids,NULL));
  PetscCall(PetscOptionsEnum("-pc_gamg_coarse_grid_layout_type","compact: place reduced grids on processes in natural order; spread: distribute to whole machine for more memory bandwidth","PCGAMGSetCoarseGridLayoutType",LayoutTypes,(PetscEnum)pc_gamg->layout_type,(PetscEnum*)&pc_gamg->layout_type,NULL));
  PetscCall(PetscOptionsInt("-pc_gamg_process_eq_limit","Limit (goal) on number of equations per process on coarse grids","PCGAMGSetProcEqLim",pc_gamg->min_eq_proc,&pc_gamg->min_eq_proc,NULL));
  PetscCall(PetscOptionsInt("-pc_gamg_coarse_eq_limit","Limit on number of equations for the coarse grid","PCGAMGSetCoarseEqLim",pc_gamg->coarse_eq_limit,&pc_gamg->coarse_eq_limit,NULL));
  PetscCall(PetscOptionsReal("-pc_gamg_threshold_scale","Scaling of threshold for each level not specified","PCGAMGSetThresholdScale",pc_gamg->threshold_scale,&pc_gamg->threshold_scale,NULL));
  n = PETSC_MG_MAXLEVELS;
  PetscCall(PetscOptionsRealArray("-pc_gamg_threshold","Relative threshold to use for dropping edges in aggregation graph","PCGAMGSetThreshold",pc_gamg->threshold,&n,&flag));
  if (!flag || n < PETSC_MG_MAXLEVELS) {
    if (!flag) n = 1;
    i = n;
    do {pc_gamg->threshold[i] = pc_gamg->threshold[i-1]*pc_gamg->threshold_scale;} while (++i<PETSC_MG_MAXLEVELS);
  }
  n = PETSC_MG_MAXLEVELS;
  PetscCall(PetscOptionsIntArray("-pc_gamg_rank_reduction_factors","Manual schedule of coarse grid reduction factors that overrides internal heuristics (0 for first reduction puts one process/device)","PCGAMGSetRankReductionFactors",pc_gamg->level_reduction_factors,&n,&flag));
  if (!flag) i = 0;
  else i = n;
  do {pc_gamg->level_reduction_factors[i] = -1;} while (++i<PETSC_MG_MAXLEVELS);
  PetscCall(PetscOptionsInt("-pc_mg_levels","Set number of MG levels","PCGAMGSetNlevels",pc_gamg->Nlevels,&pc_gamg->Nlevels,NULL));
  {
    PetscReal eminmax[2] = {0., 0.};
    n = 2;
    PetscCall(PetscOptionsRealArray("-pc_gamg_eigenvalues","extreme eigenvalues for smoothed aggregation","PCGAMGSetEigenvalues",eminmax,&n,&flag));
    if (flag) {
      PetscCheck(n == 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"-pc_gamg_eigenvalues: must specify 2 parameters, min and max eigenvalues");
      PetscCall(PCGAMGSetEigenvalues(pc, eminmax[1], eminmax[0]));
    }
  }
  /* set options for subtype */
  if (pc_gamg->ops->setfromoptions) PetscCall((*pc_gamg->ops->setfromoptions)(PetscOptionsObject,pc));

  PetscCall(PCGetOptionsPrefix(pc, &pcpre));
  PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"%spc_gamg_",pcpre ? pcpre : ""));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCGAMG - Geometric algebraic multigrid (AMG) preconditioner

   Options Database Keys:
+   -pc_gamg_type <type> - one of agg, geo, or classical
.   -pc_gamg_repartition  <true,default=false> - repartition the degrees of freedom accross the coarse grids as they are determined
.   -pc_gamg_reuse_interpolation <true,default=false> - when rebuilding the algebraic multigrid preconditioner reuse the previously computed interpolations
.   -pc_gamg_asm_use_agg <true,default=false> - use the aggregates from the coasening process to defined the subdomains on each level for the PCASM smoother
.   -pc_gamg_process_eq_limit <limit, default=50> - GAMG will reduce the number of MPI processes used directly on the coarse grids so that there are around <limit>
                                        equations on each process that has degrees of freedom
.   -pc_gamg_coarse_eq_limit <limit, default=50> - Set maximum number of equations on coarsest grid to aim for.
.   -pc_gamg_threshold[] <thresh,default=0> - Before aggregating the graph GAMG will remove small values from the graph on each level
-   -pc_gamg_threshold_scale <scale,default=1> - Scaling of threshold on each coarser grid if not specified

   Options Database Keys for default Aggregation:
+  -pc_gamg_agg_nsmooths <nsmooth, default=1> - number of smoothing steps to use with smooth aggregation
.  -pc_gamg_sym_graph <true,default=false> - symmetrize the graph before computing the aggregation
-  -pc_gamg_square_graph <n,default=1> - number of levels to square the graph before aggregating it

   Multigrid options:
+  -pc_mg_cycles <v> - v or w, see PCMGSetCycleType()
.  -pc_mg_distinct_smoothup - configure the up and down (pre and post) smoothers separately, see PCMGSetDistinctSmoothUp()
.  -pc_mg_type <multiplicative> - (one of) additive multiplicative full kascade
-  -pc_mg_levels <levels> - Number of levels of multigrid to use.

  Notes:
    In order to obtain good performance for PCGAMG for vector valued problems you must
       Call MatSetBlockSize() to indicate the number of degrees of freedom per grid point
       Call MatSetNearNullSpace() (or PCSetCoordinates() if solving the equations of elasticity) to indicate the near null space of the operator
       See the Users Manual Chapter 4 for more details

  Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `MatSetBlockSize()`, `PCMGType`, `PCSetCoordinates()`, `MatSetNearNullSpace()`, `PCGAMGSetType()`, `PCGAMGAGG`, `PCGAMGGEO`, `PCGAMGCLASSICAL`, `PCGAMGSetProcEqLim()`,
          `PCGAMGSetCoarseEqLim()`, `PCGAMGSetRepartition()`, `PCGAMGRegister()`, `PCGAMGSetReuseInterpolation()`, `PCGAMGASMSetUseAggs()`, `PCGAMGSetUseParallelCoarseGridSolve()`, `PCGAMGSetNlevels()`, `PCGAMGSetThreshold()`, `PCGAMGGetType()`, `PCGAMGSetReuseInterpolation()`, `PCGAMGSetUseSAEstEig()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_GAMG(PC pc)
{
  PC_GAMG *pc_gamg;
  PC_MG   *mg;

  PetscFunctionBegin;
   /* register AMG type */
  PetscCall(PCGAMGInitializePackage());

  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PetscObjectChangeTypeName((PetscObject)pc, PCGAMG));

  /* create a supporting struct and attach it to pc */
  PetscCall(PetscNewLog(pc,&pc_gamg));
  PetscCall(PCMGSetGalerkin(pc,PC_MG_GALERKIN_EXTERNAL));
  mg           = (PC_MG*)pc->data;
  mg->innerctx = pc_gamg;

  PetscCall(PetscNewLog(pc,&pc_gamg->ops));

  /* these should be in subctx but repartitioning needs simple arrays */
  pc_gamg->data_sz = 0;
  pc_gamg->data    = NULL;

  /* overwrite the pointers of PCMG by the functions of base class PCGAMG */
  pc->ops->setfromoptions = PCSetFromOptions_GAMG;
  pc->ops->setup          = PCSetUp_GAMG;
  pc->ops->reset          = PCReset_GAMG;
  pc->ops->destroy        = PCDestroy_GAMG;
  mg->view                = PCView_GAMG;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCMGGetLevels_C",PCMGGetLevels_MG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCMGSetLevels_C",PCMGSetLevels_MG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetProcEqLim_C",PCGAMGSetProcEqLim_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCoarseEqLim_C",PCGAMGSetCoarseEqLim_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetRepartition_C",PCGAMGSetRepartition_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetEigenvalues_C",PCGAMGSetEigenvalues_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetUseSAEstEig_C",PCGAMGSetUseSAEstEig_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetReuseInterpolation_C",PCGAMGSetReuseInterpolation_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGASMSetUseAggs_C",PCGAMGASMSetUseAggs_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetUseParallelCoarseGridSolve_C",PCGAMGSetUseParallelCoarseGridSolve_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCpuPinCoarseGrids_C",PCGAMGSetCpuPinCoarseGrids_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCoarseGridLayoutType_C",PCGAMGSetCoarseGridLayoutType_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetThreshold_C",PCGAMGSetThreshold_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetRankReductionFactors_C",PCGAMGSetRankReductionFactors_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetThresholdScale_C",PCGAMGSetThresholdScale_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetType_C",PCGAMGSetType_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGGetType_C",PCGAMGGetType_GAMG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetNlevels_C",PCGAMGSetNlevels_GAMG));
  pc_gamg->repart           = PETSC_FALSE;
  pc_gamg->reuse_prol       = PETSC_FALSE;
  pc_gamg->use_aggs_in_asm  = PETSC_FALSE;
  pc_gamg->use_parallel_coarse_grid_solver = PETSC_FALSE;
  pc_gamg->cpu_pin_coarse_grids = PETSC_FALSE;
  pc_gamg->layout_type      = PCGAMG_LAYOUT_SPREAD;
  pc_gamg->min_eq_proc      = 50;
  pc_gamg->coarse_eq_limit  = 50;
  PetscCall(PetscArrayzero(pc_gamg->threshold,PETSC_MG_MAXLEVELS));
  pc_gamg->threshold_scale = 1.;
  pc_gamg->Nlevels          = PETSC_MG_MAXLEVELS;
  pc_gamg->current_level    = 0; /* don't need to init really */
  pc_gamg->use_sa_esteig    = PETSC_TRUE;
  pc_gamg->emin             = 0;
  pc_gamg->emax             = 0;

  pc_gamg->ops->createlevel = PCGAMGCreateLevel_GAMG;

  /* PCSetUp_GAMG assumes that the type has been set, so set it to the default now */
  PetscCall(PCGAMGSetType(pc,PCGAMGAGG));
  PetscFunctionReturn(0);
}

/*@C
 PCGAMGInitializePackage - This function initializes everything in the PCGAMG package. It is called
    from PCInitializePackage().

 Level: developer

 .seealso: `PetscInitialize()`
@*/
PetscErrorCode PCGAMGInitializePackage(void)
{
  PetscInt       l;

  PetscFunctionBegin;
  if (PCGAMGPackageInitialized) PetscFunctionReturn(0);
  PCGAMGPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&GAMGList,PCGAMGGEO,PCCreateGAMG_GEO));
  PetscCall(PetscFunctionListAdd(&GAMGList,PCGAMGAGG,PCCreateGAMG_AGG));
  PetscCall(PetscFunctionListAdd(&GAMGList,PCGAMGCLASSICAL,PCCreateGAMG_Classical));
  PetscCall(PetscRegisterFinalize(PCGAMGFinalizePackage));

  /* general events */
  PetscCall(PetscLogEventRegister("PCSetUp_GAMG+", PC_CLASSID, &petsc_gamg_setup_events[GAMG_SETUP]));
  PetscCall(PetscLogEventRegister(" PCGAMGCreateG", PC_CLASSID, &petsc_gamg_setup_events[GAMG_GRAPH]));
  PetscCall(PetscLogEventRegister(" PCGAMGFilter", PC_CLASSID, &petsc_gamg_setup_events[GAMG_FILTER]));
  PetscCall(PetscLogEventRegister(" GAMG Coarsen", PC_CLASSID, &petsc_gamg_setup_events[GAMG_COARSEN]));
  PetscCall(PetscLogEventRegister("  GAMG Square G", PC_CLASSID, &petsc_gamg_setup_events[GAMG_SQUARE]));
  PetscCall(PetscLogEventRegister("  GAMG MIS/Agg", PC_CLASSID, &petsc_gamg_setup_events[GAMG_MIS]));
  PetscCall(PetscLogEventRegister(" PCGAMGProl", PC_CLASSID, &petsc_gamg_setup_events[GAMG_PROL]));
  PetscCall(PetscLogEventRegister("  GAMG Prol-col", PC_CLASSID, &petsc_gamg_setup_events[GAMG_PROLA]));
  PetscCall(PetscLogEventRegister("  GAMG Prol-lift", PC_CLASSID, &petsc_gamg_setup_events[GAMG_PROLB]));
  PetscCall(PetscLogEventRegister(" PCGAMGOptProl", PC_CLASSID, &petsc_gamg_setup_events[GAMG_OPT]));
  PetscCall(PetscLogEventRegister("  GAMG smooth", PC_CLASSID, &petsc_gamg_setup_events[GAMG_OPTSM]));
  PetscCall(PetscLogEventRegister(" PCGAMGCreateL", PC_CLASSID, &petsc_gamg_setup_events[GAMG_LEVEL]));
  PetscCall(PetscLogEventRegister("  GAMG PtAP",PC_CLASSID, &petsc_gamg_setup_events[GAMG_PTAP]));
  PetscCall(PetscLogEventRegister("  GAMG Reduce",PC_CLASSID, &petsc_gamg_setup_events[GAMG_REDUCE]));
  PetscCall(PetscLogEventRegister("   GAMG Repart", PC_CLASSID, &petsc_gamg_setup_events[GAMG_REPART]));
  /* PetscCall(PetscLogEventRegister("   GAMG Inv-Srt", PC_CLASSID, &petsc_gamg_setup_events[SET13])); */
  /* PetscCall(PetscLogEventRegister("   GAMG Move A", PC_CLASSID, &petsc_gamg_setup_events[SET14])); */
  /* PetscCall(PetscLogEventRegister("   GAMG Move P", PC_CLASSID, &petsc_gamg_setup_events[SET15])); */
  for (l=0;l<PETSC_MG_MAXLEVELS;l++) {
    char ename[32];

    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCGAMG Squ l%02" PetscInt_FMT,l));
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &petsc_gamg_setup_matmat_events[l][0]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCGAMG Gal l%02" PetscInt_FMT,l));
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &petsc_gamg_setup_matmat_events[l][1]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCGAMG Opt l%02" PetscInt_FMT,l));
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &petsc_gamg_setup_matmat_events[l][2]));
  }
#if defined(GAMG_STAGES)
  {  /* create timer stages */
    char     str[32];
    sprintf(str,"GAMG Level %d",0);
    PetscCall(PetscLogStageRegister(str, &gamg_stages[0]));
  }
#endif
  PetscFunctionReturn(0);
}

/*@C
 PCGAMGFinalizePackage - This function frees everything from the PCGAMG package. It is
    called from PetscFinalize() automatically.

 Level: developer

 .seealso: `PetscFinalize()`
@*/
PetscErrorCode PCGAMGFinalizePackage(void)
{
  PetscFunctionBegin;
  PCGAMGPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&GAMGList));
  PetscFunctionReturn(0);
}

/*@C
 PCGAMGRegister - Register a PCGAMG implementation.

 Input Parameters:
 + type - string that will be used as the name of the GAMG type.
 - create - function for creating the gamg context.

  Level: advanced

 .seealso: `PCGAMGType`, `PCGAMG`, `PCGAMGSetType()`
@*/
PetscErrorCode PCGAMGRegister(PCGAMGType type, PetscErrorCode (*create)(PC))
{
  PetscFunctionBegin;
  PetscCall(PCGAMGInitializePackage());
  PetscCall(PetscFunctionListAdd(&GAMGList,type,create));
  PetscFunctionReturn(0);
}
