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

PetscLogEvent petsc_gamg_setup_events[NUM_SET];
PetscLogEvent petsc_gamg_setup_matmat_events[PETSC_MG_MAXLEVELS][3];
PetscLogEvent PC_GAMGGraph_AGG;
PetscLogEvent PC_GAMGGraph_GEO;
PetscLogEvent PC_GAMGCoarsen_AGG;
PetscLogEvent PC_GAMGCoarsen_GEO;
PetscLogEvent PC_GAMGProlongator_AGG;
PetscLogEvent PC_GAMGProlongator_GEO;
PetscLogEvent PC_GAMGOptProlongator_AGG;

/* #define GAMG_STAGES */
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
  CHKERRQ(PetscFree(pc_gamg->data));
  pc_gamg->data_sz = 0;
  CHKERRQ(PetscFree(pc_gamg->orig_data));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)Amat_fine,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(MatGetBlockSize(Amat_fine, &f_bs));
  CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_matmat_events[pc_gamg->current_level][1],0,0,0,0));
  CHKERRQ(MatPtAP(Amat_fine, Pold, MAT_INITIAL_MATRIX, 2.0, &Cmat));
  CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_matmat_events[pc_gamg->current_level][1],0,0,0,0));

  if (Pcolumnperm) *Pcolumnperm = NULL;

  /* set 'ncrs' (nodes), 'ncrs_eq' (equations)*/
  CHKERRQ(MatGetLocalSize(Cmat, &ncrs_eq, NULL));
  if (pc_gamg->data_cell_rows>0) {
    ncrs = pc_gamg->data_sz/pc_gamg->data_cell_cols/pc_gamg->data_cell_rows;
  } else {
    PetscInt  bs;
    CHKERRQ(MatGetBlockSize(Cmat, &bs));
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
    CHKERRQ(PetscShmCommGet(comm,&pshmcomm));
    CHKERRQ(PetscShmCommGetMpiShmComm(pshmcomm,&loccomm));
    CHKERRMPI(MPI_Comm_rank(loccomm, &locrank));
    s_nnodes = !locrank;
    CHKERRMPI(MPI_Allreduce(&s_nnodes,&r_nnodes,1,MPIU_INT,MPI_SUM,comm));
    PetscCheckFalse(size%r_nnodes,PETSC_COMM_SELF,PETSC_ERR_PLIB,"odd number of nodes np=%D nnodes%D",size,r_nnodes);
    devCount = 0;
    cerr = cudaGetDeviceCount(&devCount);
    cudaGetLastError(); /* Reset the last error */
    if (cerr == cudaSuccess && devCount >= 1) { /* There are devices, else go to heuristic */
      new_new_size = r_nnodes * devCount;
      new_size = new_new_size;
      CHKERRQ(PetscInfo(pc,"%s: Fine grid with Cuda. %D nodes. Change new active set size %d --> %d (devCount=%d #nodes=%D)\n",((PetscObject)pc)->prefix,r_nnodes,nactive,new_size,devCount,r_nnodes));
    } else {
      CHKERRQ(PetscInfo(pc,"%s: With Cuda but no device. Use heuristics."));
      goto HEURISTIC;
    }
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"should not be here");
#endif
  } else if (pc_gamg->level_reduction_factors[pc_gamg->current_level] > 0) {
    PetscCheckFalse(nactive%pc_gamg->level_reduction_factors[pc_gamg->current_level],PETSC_COMM_SELF,PETSC_ERR_PLIB,"odd number of active process %D wrt reduction factor %D",nactive,pc_gamg->level_reduction_factors[pc_gamg->current_level]);
    new_size = nactive/pc_gamg->level_reduction_factors[pc_gamg->current_level];
    CHKERRQ(PetscInfo(pc,"%s: Manually setting reduction to %d active processes (%d/%D)\n",((PetscObject)pc)->prefix,new_size,nactive,pc_gamg->level_reduction_factors[pc_gamg->current_level]));
  } else if (is_last && !pc_gamg->use_parallel_coarse_grid_solver) {
    new_size = 1;
    CHKERRQ(PetscInfo(pc,"%s: Force coarsest grid reduction to %d active processes\n",((PetscObject)pc)->prefix,new_size));
  } else {
    PetscInt ncrs_eq_glob;
#if defined(PETSC_HAVE_CUDA)
    HEURISTIC:
#endif
    CHKERRQ(MatGetSize(Cmat, &ncrs_eq_glob, NULL));
    new_size = (PetscMPIInt)((float)ncrs_eq_glob/(float)pc_gamg->min_eq_proc + 0.5); /* hardwire min. number of eq/proc */
    if (!new_size) new_size = 1; /* not likely, posible? */
    else if (new_size >= nactive) new_size = nactive; /* no change, rare */
    CHKERRQ(PetscInfo(pc,"%s: Coarse grid reduction from %d to %d active processes\n",((PetscObject)pc)->prefix,nactive,new_size));
  }
  if (new_size==nactive) {
    *a_Amat_crs = Cmat; /* output - no repartitioning or reduction - could bail here */
    if (new_size < size) {
      /* odd case where multiple coarse grids are on one processor or no coarsening ... */
      CHKERRQ(PetscInfo(pc,"%s: reduced grid using same number of processors (%d) as last grid (use larger coarse grid)\n",((PetscObject)pc)->prefix,nactive));
      if (pc_gamg->cpu_pin_coarse_grids) {
        CHKERRQ(MatBindToCPU(*a_Amat_crs,PETSC_TRUE));
        CHKERRQ(MatBindToCPU(*a_P_inout,PETSC_TRUE));
      }
    }
    /* we know that the grid structure can be reused in MatPtAP */
  } else { /* reduce active processors - we know that the grid structure can NOT be reused in MatPtAP */
    PetscInt       *counts,*newproc_idx,ii,jj,kk,strideNew,*tidx,ncrs_new,ncrs_eq_new,nloc_old,expand_factor=1,rfactor=1;
    IS             is_eq_newproc,is_eq_num,is_eq_num_prim,new_eq_indices;
    nloc_old = ncrs_eq/cr_bs;
    PetscCheckFalse(ncrs_eq % cr_bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ncrs_eq %D not divisible by cr_bs %D",ncrs_eq,cr_bs);
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
        CHKERRQ(PetscInfo(pc,"%s: Finding factorable processor set stopped reduction: new_size=%d, neq(loc)=%D\n",((PetscObject)pc)->prefix,new_size,ncrs_eq));
        PetscFunctionReturn(0);
      }
    }
    CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET12],0,0,0,0));
    /* make 'is_eq_newproc' */
    CHKERRQ(PetscMalloc1(size, &counts));
    if (pc_gamg->repart) {
      /* Repartition Cmat_{k} and move columns of P^{k}_{k-1} and coordinates of primal part accordingly */
      Mat      adj;
      CHKERRQ(PetscInfo(pc,"%s: Repartition: size (active): %d --> %d, %D local equations, using %s process layout\n",((PetscObject)pc)->prefix,*a_nactive_proc, new_size, ncrs_eq, (pc_gamg->layout_type==PCGAMG_LAYOUT_COMPACT) ? "compact" : "spread"));
      /* get 'adj' */
      if (cr_bs == 1) {
        CHKERRQ(MatConvert(Cmat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj));
      } else {
        /* make a scalar matrix to partition (no Stokes here) */
        Mat               tMat;
        PetscInt          Istart_crs,Iend_crs,ncols,jj,Ii;
        const PetscScalar *vals;
        const PetscInt    *idx;
        PetscInt          *d_nnz, *o_nnz, M, N;
        static PetscInt   llev = 0; /* ugly but just used for debugging */
        MatType           mtype;

        CHKERRQ(PetscMalloc2(ncrs, &d_nnz,ncrs, &o_nnz));
        CHKERRQ(MatGetOwnershipRange(Cmat, &Istart_crs, &Iend_crs));
        CHKERRQ(MatGetSize(Cmat, &M, &N));
        for (Ii = Istart_crs, jj = 0; Ii < Iend_crs; Ii += cr_bs, jj++) {
          CHKERRQ(MatGetRow(Cmat,Ii,&ncols,NULL,NULL));
          d_nnz[jj] = ncols/cr_bs;
          o_nnz[jj] = ncols/cr_bs;
          CHKERRQ(MatRestoreRow(Cmat,Ii,&ncols,NULL,NULL));
          if (d_nnz[jj] > ncrs) d_nnz[jj] = ncrs;
          if (o_nnz[jj] > (M/cr_bs-ncrs)) o_nnz[jj] = M/cr_bs-ncrs;
        }

        CHKERRQ(MatGetType(Amat_fine,&mtype));
        CHKERRQ(MatCreate(comm, &tMat));
        CHKERRQ(MatSetSizes(tMat, ncrs, ncrs,PETSC_DETERMINE, PETSC_DETERMINE));
        CHKERRQ(MatSetType(tMat,mtype));
        CHKERRQ(MatSeqAIJSetPreallocation(tMat,0,d_nnz));
        CHKERRQ(MatMPIAIJSetPreallocation(tMat,0,d_nnz,0,o_nnz));
        CHKERRQ(PetscFree2(d_nnz,o_nnz));

        for (ii = Istart_crs; ii < Iend_crs; ii++) {
          PetscInt dest_row = ii/cr_bs;
          CHKERRQ(MatGetRow(Cmat,ii,&ncols,&idx,&vals));
          for (jj = 0; jj < ncols; jj++) {
            PetscInt    dest_col = idx[jj]/cr_bs;
            PetscScalar v        = 1.0;
            CHKERRQ(MatSetValues(tMat,1,&dest_row,1,&dest_col,&v,ADD_VALUES));
          }
          CHKERRQ(MatRestoreRow(Cmat,ii,&ncols,&idx,&vals));
        }
        CHKERRQ(MatAssemblyBegin(tMat,MAT_FINAL_ASSEMBLY));
        CHKERRQ(MatAssemblyEnd(tMat,MAT_FINAL_ASSEMBLY));

        if (llev++ == -1) {
          PetscViewer viewer; char fname[32];
          CHKERRQ(PetscSNPrintf(fname,sizeof(fname),"part_mat_%D.mat",llev));
          PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&viewer);
          CHKERRQ(MatView(tMat, viewer));
          CHKERRQ(PetscViewerDestroy(&viewer));
        }
        CHKERRQ(MatConvert(tMat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj));
        CHKERRQ(MatDestroy(&tMat));
      } /* create 'adj' */

      { /* partition: get newproc_idx */
        char            prefix[256];
        const char      *pcpre;
        const PetscInt  *is_idx;
        MatPartitioning mpart;
        IS              proc_is;

        CHKERRQ(MatPartitioningCreate(comm, &mpart));
        CHKERRQ(MatPartitioningSetAdjacency(mpart, adj));
        CHKERRQ(PCGetOptionsPrefix(pc, &pcpre));
        CHKERRQ(PetscSNPrintf(prefix,sizeof(prefix),"%spc_gamg_",pcpre ? pcpre : ""));
        CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix));
        CHKERRQ(MatPartitioningSetFromOptions(mpart));
        CHKERRQ(MatPartitioningSetNParts(mpart, new_size));
        CHKERRQ(MatPartitioningApply(mpart, &proc_is));
        CHKERRQ(MatPartitioningDestroy(&mpart));

        /* collect IS info */
        CHKERRQ(PetscMalloc1(ncrs_eq, &newproc_idx));
        CHKERRQ(ISGetIndices(proc_is, &is_idx));
        for (kk = jj = 0 ; kk < nloc_old ; kk++) {
          for (ii = 0 ; ii < cr_bs ; ii++, jj++) {
            newproc_idx[jj] = is_idx[kk] * expand_factor; /* distribution */
          }
        }
        CHKERRQ(ISRestoreIndices(proc_is, &is_idx));
        CHKERRQ(ISDestroy(&proc_is));
      }
      CHKERRQ(MatDestroy(&adj));

      CHKERRQ(ISCreateGeneral(comm, ncrs_eq, newproc_idx, PETSC_COPY_VALUES, &is_eq_newproc));
      CHKERRQ(PetscFree(newproc_idx));
    } else { /* simple aggregation of parts -- 'is_eq_newproc' */
      PetscInt targetPE;
      PetscCheckFalse(new_size==nactive,PETSC_COMM_SELF,PETSC_ERR_PLIB,"new_size==nactive. Should not happen");
      CHKERRQ(PetscInfo(pc,"%s: Number of equations (loc) %D with simple aggregation\n",((PetscObject)pc)->prefix,ncrs_eq));
      targetPE = (rank/rfactor)*expand_factor;
      CHKERRQ(ISCreateStride(comm, ncrs_eq, targetPE, 0, &is_eq_newproc));
    } /* end simple 'is_eq_newproc' */

    /*
      Create an index set from the is_eq_newproc index set to indicate the mapping TO
    */
    CHKERRQ(ISPartitioningToNumbering(is_eq_newproc, &is_eq_num));
    is_eq_num_prim = is_eq_num;
    /*
      Determine how many equations/vertices are assigned to each processor
    */
    CHKERRQ(ISPartitioningCount(is_eq_newproc, size, counts));
    ncrs_eq_new = counts[rank];
    CHKERRQ(ISDestroy(&is_eq_newproc));
    ncrs_new = ncrs_eq_new/cr_bs;

    CHKERRQ(PetscFree(counts));
    /* data movement scope -- this could be moved to subclasses so that we don't try to cram all auxilary data into some complex abstracted thing */
    {
      Vec            src_crd, dest_crd;
      const PetscInt *idx,ndata_rows=pc_gamg->data_cell_rows,ndata_cols=pc_gamg->data_cell_cols,node_data_sz=ndata_rows*ndata_cols;
      VecScatter     vecscat;
      PetscScalar    *array;
      IS isscat;
      /* move data (for primal equations only) */
      /* Create a vector to contain the newly ordered element information */
      CHKERRQ(VecCreate(comm, &dest_crd));
      CHKERRQ(VecSetSizes(dest_crd, node_data_sz*ncrs_new, PETSC_DECIDE));
      CHKERRQ(VecSetType(dest_crd,VECSTANDARD)); /* this is needed! */
      /*
        There are 'ndata_rows*ndata_cols' data items per node, (one can think of the vectors of having
        a block size of ...).  Note, ISs are expanded into equation space by 'cr_bs'.
      */
      CHKERRQ(PetscMalloc1(ncrs*node_data_sz, &tidx));
      CHKERRQ(ISGetIndices(is_eq_num_prim, &idx));
      for (ii=0,jj=0; ii<ncrs; ii++) {
        PetscInt id = idx[ii*cr_bs]/cr_bs; /* get node back */
        for (kk=0; kk<node_data_sz; kk++, jj++) tidx[jj] = id*node_data_sz + kk;
      }
      CHKERRQ(ISRestoreIndices(is_eq_num_prim, &idx));
      CHKERRQ(ISCreateGeneral(comm, node_data_sz*ncrs, tidx, PETSC_COPY_VALUES, &isscat));
      CHKERRQ(PetscFree(tidx));
      /*
        Create a vector to contain the original vertex information for each element
      */
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, node_data_sz*ncrs, &src_crd));
      for (jj=0; jj<ndata_cols; jj++) {
        const PetscInt stride0=ncrs*pc_gamg->data_cell_rows;
        for (ii=0; ii<ncrs; ii++) {
          for (kk=0; kk<ndata_rows; kk++) {
            PetscInt    ix = ii*ndata_rows + kk + jj*stride0, jx = ii*node_data_sz + kk*ndata_cols + jj;
            PetscScalar tt = (PetscScalar)pc_gamg->data[ix];
            CHKERRQ(VecSetValues(src_crd, 1, &jx, &tt, INSERT_VALUES));
          }
        }
      }
      CHKERRQ(VecAssemblyBegin(src_crd));
      CHKERRQ(VecAssemblyEnd(src_crd));
      /*
        Scatter the element vertex information (still in the original vertex ordering)
        to the correct processor
      */
      CHKERRQ(VecScatterCreate(src_crd, NULL, dest_crd, isscat, &vecscat));
      CHKERRQ(ISDestroy(&isscat));
      CHKERRQ(VecScatterBegin(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(VecDestroy(&src_crd));
      /*
        Put the element vertex data into a new allocation of the gdata->ele
      */
      CHKERRQ(PetscFree(pc_gamg->data));
      CHKERRQ(PetscMalloc1(node_data_sz*ncrs_new, &pc_gamg->data));

      pc_gamg->data_sz = node_data_sz*ncrs_new;
      strideNew        = ncrs_new*ndata_rows;

      CHKERRQ(VecGetArray(dest_crd, &array));
      for (jj=0; jj<ndata_cols; jj++) {
        for (ii=0; ii<ncrs_new; ii++) {
          for (kk=0; kk<ndata_rows; kk++) {
            PetscInt ix = ii*ndata_rows + kk + jj*strideNew, jx = ii*node_data_sz + kk*ndata_cols + jj;
            pc_gamg->data[ix] = PetscRealPart(array[jx]);
          }
        }
      }
      CHKERRQ(VecRestoreArray(dest_crd, &array));
      CHKERRQ(VecDestroy(&dest_crd));
    }
    /* move A and P (columns) with new layout */
    CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET13],0,0,0,0));
    /*
      Invert for MatCreateSubMatrix
    */
    CHKERRQ(ISInvertPermutation(is_eq_num, ncrs_eq_new, &new_eq_indices));
    CHKERRQ(ISSort(new_eq_indices)); /* is this needed? */
    CHKERRQ(ISSetBlockSize(new_eq_indices, cr_bs));
    if (is_eq_num != is_eq_num_prim) {
      CHKERRQ(ISDestroy(&is_eq_num_prim)); /* could be same as 'is_eq_num' */
    }
    if (Pcolumnperm) {
      CHKERRQ(PetscObjectReference((PetscObject)new_eq_indices));
      *Pcolumnperm = new_eq_indices;
    }
    CHKERRQ(ISDestroy(&is_eq_num));
    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET13],0,0,0,0));
    CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET14],0,0,0,0));
    /* 'a_Amat_crs' output */
    {
      Mat       mat;
      PetscBool flg;
      CHKERRQ(MatCreateSubMatrix(Cmat, new_eq_indices, new_eq_indices, MAT_INITIAL_MATRIX, &mat));
      CHKERRQ(MatGetOption(Cmat, MAT_SPD, &flg));
      if (flg) {
        CHKERRQ(MatSetOption(mat, MAT_SPD,PETSC_TRUE));
      } else {
        CHKERRQ(MatGetOption(Cmat, MAT_HERMITIAN, &flg));
        if (flg) {
          CHKERRQ(MatSetOption(mat, MAT_HERMITIAN,PETSC_TRUE));
        } else {
#if !defined(PETSC_USE_COMPLEX)
          CHKERRQ(MatGetOption(Cmat, MAT_SYMMETRIC, &flg));
          if (flg) {
            CHKERRQ(MatSetOption(mat, MAT_SYMMETRIC,PETSC_TRUE));
          }
#endif
        }
      }
      *a_Amat_crs = mat;
    }
    CHKERRQ(MatDestroy(&Cmat));

    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET14],0,0,0,0));
    /* prolongator */
    {
      IS       findices;
      PetscInt Istart,Iend;
      Mat      Pnew;

      CHKERRQ(MatGetOwnershipRange(Pold, &Istart, &Iend));
      CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET15],0,0,0,0));
      CHKERRQ(ISCreateStride(comm,Iend-Istart,Istart,1,&findices));
      CHKERRQ(ISSetBlockSize(findices,f_bs));
      CHKERRQ(MatCreateSubMatrix(Pold, findices, new_eq_indices, MAT_INITIAL_MATRIX, &Pnew));
      CHKERRQ(ISDestroy(&findices));
      CHKERRQ(MatSetOption(Pnew,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));

      CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET15],0,0,0,0));
      CHKERRQ(MatDestroy(a_P_inout));

      /* output - repartitioned */
      *a_P_inout = Pnew;
    }
    CHKERRQ(ISDestroy(&new_eq_indices));

    *a_nactive_proc = new_size; /* output */

    /* pinning on reduced grids, not a bad heuristic and optimization gets folded into process reduction optimization */
    if (pc_gamg->cpu_pin_coarse_grids) {
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
      static PetscInt llev = 2;
      CHKERRQ(PetscInfo(pc,"%s: Pinning level %D to the CPU\n",((PetscObject)pc)->prefix,llev++));
#endif
      CHKERRQ(MatBindToCPU(*a_Amat_crs,PETSC_TRUE));
      CHKERRQ(MatBindToCPU(*a_P_inout,PETSC_TRUE));
      if (1) { /* HACK: move this to MatBindCPU_MPIAIJXXX; lvec is created, need to pin it, this is done in MatSetUpMultiply_MPIAIJ. Hack */
        Mat         A = *a_Amat_crs, P = *a_P_inout;
        PetscMPIInt size;
        CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
        if (size > 1) {
          Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data, *p = (Mat_MPIAIJ*)P->data;
          CHKERRQ(VecBindToCPU(a->lvec,PETSC_TRUE));
          CHKERRQ(VecBindToCPU(p->lvec,PETSC_TRUE));
        }
      }
    }
    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET12],0,0,0,0));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGSquareGraph_GAMG(PC a_pc, Mat Gmat1, Mat* Gmat2)
{
  const char     *prefix;
  char           addp[32];
  PC_MG          *mg      = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PCGetOptionsPrefix(a_pc,&prefix));
  CHKERRQ(PetscInfo(a_pc,"%s: Square Graph on level %D\n",((PetscObject)a_pc)->prefix,pc_gamg->current_level+1));
  CHKERRQ(MatProductCreate(Gmat1,Gmat1,NULL,Gmat2));
  CHKERRQ(MatSetOptionsPrefix(*Gmat2,prefix));
  CHKERRQ(PetscSNPrintf(addp,sizeof(addp),"pc_gamg_square_%d_",pc_gamg->current_level));
  CHKERRQ(MatAppendOptionsPrefix(*Gmat2,addp));
  if ((*Gmat2)->structurally_symmetric) {
    CHKERRQ(MatProductSetType(*Gmat2,MATPRODUCT_AB));
  } else {
    CHKERRQ(MatSetOption(Gmat1,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));
    CHKERRQ(MatProductSetType(*Gmat2,MATPRODUCT_AtB));
  }
  CHKERRQ(MatProductSetFromOptions(*Gmat2));
  CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_matmat_events[pc_gamg->current_level][0],0,0,0,0));
  CHKERRQ(MatProductSymbolic(*Gmat2));
  CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_matmat_events[pc_gamg->current_level][0],0,0,0,0));
  CHKERRQ(MatProductClear(*Gmat2));
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
  PetscErrorCode ierr;
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
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  if (pc->setupcalled) {
    if (!pc_gamg->reuse_prol || pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* reset everything */
      CHKERRQ(PCReset_MG(pc));
      pc->setupcalled = 0;
    } else {
      PC_MG_Levels **mglevels = mg->levels;
      /* just do Galerkin grids */
      Mat          B,dA,dB;

      if (pc_gamg->Nlevels > 1) {
        PetscInt gl;
        /* currently only handle case where mat and pmat are the same on coarser levels */
        CHKERRQ(KSPGetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,&dA,&dB));
        /* (re)set to get dirty flag */
        CHKERRQ(KSPSetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,dA,dB));

        for (level=pc_gamg->Nlevels-2,gl=0; level>=0; level--,gl++) {
          MatReuse reuse = MAT_INITIAL_MATRIX ;

          /* matrix structure can change from repartitioning or process reduction but don't know if we have process reduction here. Should fix */
          CHKERRQ(KSPGetOperators(mglevels[level]->smoothd,NULL,&B));
          if (B->product) {
            if (B->product->A == dB && B->product->B == mglevels[level+1]->interpolate) {
              reuse = MAT_REUSE_MATRIX;
            }
          }
          if (reuse == MAT_INITIAL_MATRIX) CHKERRQ(MatDestroy(&mglevels[level]->A));
          if (reuse == MAT_REUSE_MATRIX) {
            CHKERRQ(PetscInfo(pc,"%s: RAP after first solve, reuse matrix level %D\n",((PetscObject)pc)->prefix,level));
          } else {
            CHKERRQ(PetscInfo(pc,"%s: RAP after first solve, new matrix level %D\n",((PetscObject)pc)->prefix,level));
          }
          CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_matmat_events[gl][1],0,0,0,0));
          CHKERRQ(MatPtAP(dB,mglevels[level+1]->interpolate,reuse,PETSC_DEFAULT,&B));
          CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_matmat_events[gl][1],0,0,0,0));
          if (reuse == MAT_INITIAL_MATRIX) mglevels[level]->A = B;
          CHKERRQ(KSPSetOperators(mglevels[level]->smoothd,B,B));
          dB   = B;
        }
      }

      CHKERRQ(PCSetUp_MG(pc));
      PetscFunctionReturn(0);
    }
  }

  if (!pc_gamg->data) {
    if (pc_gamg->orig_data) {
      CHKERRQ(MatGetBlockSize(Pmat, &bs));
      CHKERRQ(MatGetLocalSize(Pmat, &qq, NULL));

      pc_gamg->data_sz        = (qq/bs)*pc_gamg->orig_data_cell_rows*pc_gamg->orig_data_cell_cols;
      pc_gamg->data_cell_rows = pc_gamg->orig_data_cell_rows;
      pc_gamg->data_cell_cols = pc_gamg->orig_data_cell_cols;

      CHKERRQ(PetscMalloc1(pc_gamg->data_sz, &pc_gamg->data));
      for (qq=0; qq<pc_gamg->data_sz; qq++) pc_gamg->data[qq] = pc_gamg->orig_data[qq];
    } else {
      PetscCheck(pc_gamg->ops->createdefaultdata,comm,PETSC_ERR_PLIB,"'createdefaultdata' not set(?) need to support NULL data");
      CHKERRQ(pc_gamg->ops->createdefaultdata(pc,Pmat));
    }
  }

  /* cache original data for reuse */
  if (!pc_gamg->orig_data && (PetscBool)(!pc_gamg->reuse_prol)) {
    CHKERRQ(PetscMalloc1(pc_gamg->data_sz, &pc_gamg->orig_data));
    for (qq=0; qq<pc_gamg->data_sz; qq++) pc_gamg->orig_data[qq] = pc_gamg->data[qq];
    pc_gamg->orig_data_cell_rows = pc_gamg->data_cell_rows;
    pc_gamg->orig_data_cell_cols = pc_gamg->data_cell_cols;
  }

  /* get basic dims */
  CHKERRQ(MatGetBlockSize(Pmat, &bs));
  CHKERRQ(MatGetSize(Pmat, &M, &N));

  CHKERRQ(MatGetInfo(Pmat,MAT_GLOBAL_SUM,&info)); /* global reduction */
  nnz0   = info.nz_used;
  nnztot = info.nz_used;
  CHKERRQ(PetscInfo(pc,"%s: level %D) N=%D, n data rows=%D, n data cols=%D, nnz/row (ave)=%d, np=%D\n",((PetscObject)pc)->prefix,0,M,pc_gamg->data_cell_rows,pc_gamg->data_cell_cols,(int)(nnz0/(PetscReal)M+0.5),size));

  /* Get A_i and R_i */
  for (level=0, Aarr[0]=Pmat, nactivepe = size; level < (pc_gamg->Nlevels-1) && (!level || M>pc_gamg->coarse_eq_limit); level++) {
    pc_gamg->current_level = level;
    PetscCheck(level < PETSC_MG_MAXLEVELS,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Too many levels %D",level);
    level1 = level + 1;
    CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET1],0,0,0,0));
#if defined(GAMG_STAGES)
    CHKERRQ(PetscLogStagePush(gamg_stages[level]));
#endif
    { /* construct prolongator */
      Mat              Gmat;
      PetscCoarsenData *agg_lists;
      Mat              Prol11;

      CHKERRQ(pc_gamg->ops->graph(pc,Aarr[level], &Gmat));
      CHKERRQ(pc_gamg->ops->coarsen(pc, &Gmat, &agg_lists));
      CHKERRQ(pc_gamg->ops->prolongator(pc,Aarr[level],Gmat,agg_lists,&Prol11));

      /* could have failed to create new level */
      if (Prol11) {
        const char *prefix;
        char       addp[32];

        /* get new block size of coarse matrices */
        CHKERRQ(MatGetBlockSizes(Prol11, NULL, &bs));

        if (pc_gamg->ops->optprolongator) {
          /* smooth */
          CHKERRQ(pc_gamg->ops->optprolongator(pc, Aarr[level], &Prol11));
        }

        if (pc_gamg->use_aggs_in_asm) {
          PetscInt bs;
          CHKERRQ(MatGetBlockSizes(Prol11, &bs, NULL));
          CHKERRQ(PetscCDGetASMBlocks(agg_lists, bs, Gmat, &nASMBlocksArr[level], &ASMLocalIDsArr[level]));
        }

        CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
        CHKERRQ(MatSetOptionsPrefix(Prol11,prefix));
        CHKERRQ(PetscSNPrintf(addp,sizeof(addp),"pc_gamg_prolongator_%d_",(int)level));
        CHKERRQ(MatAppendOptionsPrefix(Prol11,addp));
        /* Always generate the transpose with CUDA
           Such behaviour can be adapted with -pc_gamg_prolongator_ prefixed options */
        CHKERRQ(MatSetOption(Prol11,MAT_FORM_EXPLICIT_TRANSPOSE,PETSC_TRUE));
        CHKERRQ(MatSetFromOptions(Prol11));
        Parr[level1] = Prol11;
      } else Parr[level1] = NULL; /* failed to coarsen */

      CHKERRQ(MatDestroy(&Gmat));
      CHKERRQ(PetscCDDestroy(agg_lists));
    } /* construct prolongator scope */
    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET1],0,0,0,0));
    if (!level) Aarr[0] = Pmat; /* use Pmat for finest level setup */
    if (!Parr[level1]) { /* failed to coarsen */
      CHKERRQ(PetscInfo(pc,"%s: Stop gridding, level %D\n",((PetscObject)pc)->prefix,level));
#if defined(GAMG_STAGES)
      CHKERRQ(PetscLogStagePop());
#endif
      break;
    }
    CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[SET2],0,0,0,0));
    CHKERRQ(MatGetSize(Parr[level1], &M, &N)); /* N is next M, a loop test variables */
    PetscCheck(!is_last,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Is last ?");
    if (N <= pc_gamg->coarse_eq_limit) is_last = PETSC_TRUE;
    if (level1 == pc_gamg->Nlevels-1) is_last = PETSC_TRUE;
    CHKERRQ(pc_gamg->ops->createlevel(pc, Aarr[level], bs, &Parr[level1], &Aarr[level1], &nactivepe, NULL, is_last));

    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[SET2],0,0,0,0));
    CHKERRQ(MatGetSize(Aarr[level1], &M, &N)); /* M is loop test variables */
    CHKERRQ(MatGetInfo(Aarr[level1], MAT_GLOBAL_SUM, &info));
    nnztot += info.nz_used;
    CHKERRQ(PetscInfo(pc,"%s: %D) N=%D, n data cols=%D, nnz/row (ave)=%d, %D active pes\n",((PetscObject)pc)->prefix,level1,M,pc_gamg->data_cell_cols,(int)(info.nz_used/(PetscReal)M),nactivepe));

#if defined(GAMG_STAGES)
    CHKERRQ(PetscLogStagePop());
#endif
    /* stop if one node or one proc -- could pull back for singular problems */
    if ((pc_gamg->data_cell_cols && M/pc_gamg->data_cell_cols < 2) || (!pc_gamg->data_cell_cols && M/bs < 2)) {
      CHKERRQ(PetscInfo(pc,"%s: HARD stop of coarsening on level %D.  Grid too small: %D block nodes\n",((PetscObject)pc)->prefix,level,M/bs));
      level++;
      break;
    }
  } /* levels */
  CHKERRQ(PetscFree(pc_gamg->data));

  CHKERRQ(PetscInfo(pc,"%s: %D levels, grid complexity = %g\n",((PetscObject)pc)->prefix,level+1,nnztot/nnz0));
  pc_gamg->Nlevels = level + 1;
  fine_level       = level;
  CHKERRQ(PCMGSetLevels(pc,pc_gamg->Nlevels,NULL));

  if (pc_gamg->Nlevels > 1) { /* don't setup MG if one level */

    /* set default smoothers & set operators */
    for (lidx = 1, level = pc_gamg->Nlevels-2; lidx <= fine_level; lidx++, level--) {
      KSP smoother;
      PC  subpc;

      CHKERRQ(PCMGGetSmoother(pc, lidx, &smoother));
      CHKERRQ(KSPGetPC(smoother, &subpc));

      CHKERRQ(KSPSetNormType(smoother, KSP_NORM_NONE));
      /* set ops */
      CHKERRQ(KSPSetOperators(smoother, Aarr[level], Aarr[level]));
      CHKERRQ(PCMGSetInterpolation(pc, lidx, Parr[level+1]));

      /* set defaults */
      CHKERRQ(KSPSetType(smoother, KSPCHEBYSHEV));

      /* set blocks for ASM smoother that uses the 'aggregates' */
      if (pc_gamg->use_aggs_in_asm) {
        PetscInt sz;
        IS       *iss;

        sz   = nASMBlocksArr[level];
        iss  = ASMLocalIDsArr[level];
        CHKERRQ(PCSetType(subpc, PCASM));
        CHKERRQ(PCASMSetOverlap(subpc, 0));
        CHKERRQ(PCASMSetType(subpc,PC_ASM_BASIC));
        if (!sz) {
          IS       is;
          CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_COPY_VALUES, &is));
          CHKERRQ(PCASMSetLocalSubdomains(subpc, 1, NULL, &is));
          CHKERRQ(ISDestroy(&is));
        } else {
          PetscInt kk;
          CHKERRQ(PCASMSetLocalSubdomains(subpc, sz, NULL, iss));
          for (kk=0; kk<sz; kk++) {
            CHKERRQ(ISDestroy(&iss[kk]));
          }
          CHKERRQ(PetscFree(iss));
        }
        ASMLocalIDsArr[level] = NULL;
        nASMBlocksArr[level]  = 0;
      } else {
        CHKERRQ(PCSetType(subpc, PCJACOBI));
      }
    }
    {
      /* coarse grid */
      KSP smoother,*k2; PC subpc,pc2; PetscInt ii,first;
      Mat Lmat = Aarr[(level=pc_gamg->Nlevels-1)]; lidx = 0;

      CHKERRQ(PCMGGetSmoother(pc, lidx, &smoother));
      CHKERRQ(KSPSetOperators(smoother, Lmat, Lmat));
      if (!pc_gamg->use_parallel_coarse_grid_solver) {
        CHKERRQ(KSPSetNormType(smoother, KSP_NORM_NONE));
        CHKERRQ(KSPGetPC(smoother, &subpc));
        CHKERRQ(PCSetType(subpc, PCBJACOBI));
        CHKERRQ(PCSetUp(subpc));
        CHKERRQ(PCBJacobiGetSubKSP(subpc,&ii,&first,&k2));
        PetscCheck(ii == 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ii %D is not one",ii);
        CHKERRQ(KSPGetPC(k2[0],&pc2));
        CHKERRQ(PCSetType(pc2, PCLU));
        CHKERRQ(PCFactorSetShiftType(pc2,MAT_SHIFT_INBLOCKS));
        CHKERRQ(KSPSetTolerances(k2[0],PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1));
        CHKERRQ(KSPSetType(k2[0], KSPPREONLY));
      }
    }

    /* should be called in PCSetFromOptions_GAMG(), but cannot be called prior to PCMGSetLevels() */
    ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
    CHKERRQ(PCSetFromOptions_MG(PetscOptionsObject,pc));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    CHKERRQ(PCMGSetGalerkin(pc,PC_MG_GALERKIN_EXTERNAL));

    /* setup cheby eigen estimates from SA */
    if (pc_gamg->use_sa_esteig) {
      for (lidx = 1, level = pc_gamg->Nlevels-2; level >= 0 ; lidx++, level--) {
        KSP       smoother;
        PetscBool ischeb;

        CHKERRQ(PCMGGetSmoother(pc, lidx, &smoother));
        CHKERRQ(PetscObjectTypeCompare((PetscObject)smoother,KSPCHEBYSHEV,&ischeb));
        if (ischeb) {
          KSP_Chebyshev *cheb = (KSP_Chebyshev*)smoother->data;

          // The command line will override these settings because KSPSetFromOptions is called in PCSetUp_MG
          if (mg->max_eigen_DinvA[level] > 0) {
            // SA uses Jacobi for P; we use SA estimates if the smoother is also Jacobi or if the user explicitly requested it.
            // TODO: This should test whether it's the same Jacobi variant (DIAG, ROWSUM, etc.)
            PetscReal emax,emin;

            emin = mg->min_eigen_DinvA[level];
            emax = mg->max_eigen_DinvA[level];
            CHKERRQ(PetscInfo(pc,"%s: PCSetUp_GAMG: call KSPChebyshevSetEigenvalues on level %D (N=%D) with emax = %g emin = %g\n",((PetscObject)pc)->prefix,level,Aarr[level]->rmap->N,(double)emax,(double)emin));
            cheb->emin_provided = emin;
            cheb->emax_provided = emax;
          }
        }
      }
    }

    CHKERRQ(PCSetUp_MG(pc));

    /* clean up */
    for (level=1; level<pc_gamg->Nlevels; level++) {
      CHKERRQ(MatDestroy(&Parr[level]));
      CHKERRQ(MatDestroy(&Aarr[level]));
    }
  } else {
    KSP smoother;

    CHKERRQ(PetscInfo(pc,"%s: One level solver used (system is seen as DD). Using default solver.\n"));
    CHKERRQ(PCMGGetSmoother(pc, 0, &smoother));
    CHKERRQ(KSPSetOperators(smoother, Aarr[0], Aarr[0]));
    CHKERRQ(KSPSetType(smoother, KSPPREONLY));
    CHKERRQ(PCSetUp_MG(pc));
  }
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
  CHKERRQ(PCReset_GAMG(pc));
  if (pc_gamg->ops->destroy) {
    CHKERRQ((*pc_gamg->ops->destroy)(pc));
  }
  CHKERRQ(PetscFree(pc_gamg->ops));
  CHKERRQ(PetscFree(pc_gamg->gamg_type_name));
  CHKERRQ(PetscFree(pc_gamg));
  CHKERRQ(PCDestroy_MG(pc));
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

.seealso: PCGAMGSetCoarseEqLim(), PCGAMGSetRankReductionFactors()
@*/
PetscErrorCode  PCGAMGSetProcEqLim(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetProcEqLim_C",(PC,PetscInt),(pc,n)));
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

.seealso: PCGAMGSetProcEqLim(), PCGAMGSetRankReductionFactors()
@*/
PetscErrorCode PCGAMGSetCoarseEqLim(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetCoarseEqLim_C",(PC,PetscInt),(pc,n)));
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
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetRepartition_C",(PC,PetscBool),(pc,n)));
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

.seealso: KSPChebyshevSetEigenvalues(), KSPChebyshevEstEigSet()
@*/
PetscErrorCode PCGAMGSetUseSAEstEig(PC pc, PetscBool n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetUseSAEstEig_C",(PC,PetscBool),(pc,n)));
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

.seealso: PCGAMGSetUseSAEstEig()
@*/
PetscErrorCode PCGAMGSetEigenvalues(PC pc, PetscReal emax,PetscReal emin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetEigenvalues_C",(PC,PetscReal,PetscReal),(pc,emax,emin)));
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
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetReuseInterpolation_C",(PC,PetscBool),(pc,n)));
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
  CHKERRQ(PetscTryMethod(pc,"PCGAMGASMSetUseAggs_C",(PC,PetscBool),(pc,flg)));
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

.seealso: PCGAMGSetCoarseGridLayoutType(), PCGAMGSetCpuPinCoarseGrids()
@*/
PetscErrorCode PCGAMGSetUseParallelCoarseGridSolve(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetUseParallelCoarseGridSolve_C",(PC,PetscBool),(pc,flg)));
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

.seealso: PCGAMGSetCoarseGridLayoutType(), PCGAMGSetUseParallelCoarseGridSolve()
@*/
PetscErrorCode PCGAMGSetCpuPinCoarseGrids(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetCpuPinCoarseGrids_C",(PC,PetscBool),(pc,flg)));
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

.seealso: PCGAMGSetUseParallelCoarseGridSolve(), PCGAMGSetCpuPinCoarseGrids()
@*/
PetscErrorCode PCGAMGSetCoarseGridLayoutType(PC pc, PCGAMGLayoutType flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetCoarseGridLayoutType_C",(PC,PCGAMGLayoutType),(pc,flg)));
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
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetNlevels_C",(PC,PetscInt),(pc,n)));
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

.seealso: PCGAMGFilterGraph(), PCGAMGSetSquareGraph()
@*/
PetscErrorCode PCGAMGSetThreshold(PC pc, PetscReal v[], PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidRealPointer(v,2);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetThreshold_C",(PC,PetscReal[],PetscInt),(pc,v,n)));
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

.seealso: PCGAMGSetProcEqLim(), PCGAMGSetCoarseEqLim()
@*/
PetscErrorCode PCGAMGSetRankReductionFactors(PC pc, PetscInt v[], PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (n) PetscValidIntPointer(v,2);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetRankReductionFactors_C",(PC,PetscInt[],PetscInt),(pc,v,n)));
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

.seealso: PCGAMGSetThreshold()
@*/
PetscErrorCode PCGAMGSetThresholdScale(PC pc, PetscReal v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetThresholdScale_C",(PC,PetscReal),(pc,v)));
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

.seealso: PCGAMGGetType(), PCGAMG, PCGAMGType
@*/
PetscErrorCode PCGAMGSetType(PC pc, PCGAMGType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGSetType_C",(PC,PCGAMGType),(pc,type)));
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

.seealso: PCGAMGSetType(), PCGAMGType
@*/
PetscErrorCode PCGAMGGetType(PC pc, PCGAMGType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCGAMGGetType_C",(PC,PCGAMGType*),(pc,type)));
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
  CHKERRQ(PetscFunctionListFind(GAMGList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown GAMG type %s given",type);
  if (pc_gamg->ops->destroy) {
    CHKERRQ((*pc_gamg->ops->destroy)(pc));
    CHKERRQ(PetscMemzero(pc_gamg->ops,sizeof(struct _PCGAMGOps)));
    pc_gamg->ops->createlevel = PCGAMGCreateLevel_GAMG;
    /* cleaning up common data in pc_gamg - this should disapear someday */
    pc_gamg->data_cell_cols = 0;
    pc_gamg->data_cell_rows = 0;
    pc_gamg->orig_data_cell_cols = 0;
    pc_gamg->orig_data_cell_rows = 0;
    CHKERRQ(PetscFree(pc_gamg->data));
    pc_gamg->data_sz = 0;
  }
  CHKERRQ(PetscFree(pc_gamg->gamg_type_name));
  CHKERRQ(PetscStrallocpy(type,&pc_gamg->gamg_type_name));
  CHKERRQ((*r)(pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_GAMG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscReal       gc=0, oc=0;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"    GAMG specific options\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Threshold for dropping small values in graph on each level ="));
  for (PetscInt i=0;i<mg->nlevels; i++) CHKERRQ(PetscViewerASCIIPrintf(viewer," %g",(double)pc_gamg->threshold[i]));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Threshold scaling factor for each level not specified = %g\n",(double)pc_gamg->threshold_scale));
  if (pc_gamg->use_aggs_in_asm) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Using aggregates from coarsening process to define subdomains for PCASM\n"));
  }
  if (pc_gamg->use_parallel_coarse_grid_solver) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Using parallel coarse grid solver (all coarse grid equations not put on one process)\n"));
  }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  if (pc_gamg->cpu_pin_coarse_grids) {
    /* CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Pinning coarse grids to the CPU)\n")); */
  }
#endif
  /* if (pc_gamg->layout_type==PCGAMG_LAYOUT_COMPACT) { */
  /*   CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Put reduced grids on processes in natural order (ie, 0,1,2...)\n")); */
  /* } else { */
  /*   CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Put reduced grids on whole machine (ie, 0,1*f,2*f...,np-f)\n")); */
  /* } */
  if (pc_gamg->ops->view) {
    CHKERRQ((*pc_gamg->ops->view)(pc,viewer));
  }
  CHKERRQ(PCMGGetGridComplexity(pc,&gc,&oc));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Complexity:    grid = %g    operator = %g\n",gc,oc));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"GAMG options"));
  CHKERRQ(PetscOptionsFList("-pc_gamg_type","Type of AMG method","PCGAMGSetType",GAMGList, pc_gamg->gamg_type_name, tname, sizeof(tname), &flag));
  if (flag) {
    CHKERRQ(PCGAMGSetType(pc,tname));
  }
  CHKERRQ(PetscOptionsBool("-pc_gamg_repartition","Repartion coarse grids","PCGAMGSetRepartition",pc_gamg->repart,&pc_gamg->repart,NULL));
  CHKERRQ(PetscOptionsBool("-pc_gamg_use_sa_esteig","Use eigen estimate from Smoothed aggregation for smoother","PCGAMGSetUseSAEstEig",pc_gamg->use_sa_esteig,&pc_gamg->use_sa_esteig,NULL));
  CHKERRQ(PetscOptionsBool("-pc_gamg_reuse_interpolation","Reuse prolongation operator","PCGAMGReuseInterpolation",pc_gamg->reuse_prol,&pc_gamg->reuse_prol,NULL));
  CHKERRQ(PetscOptionsBool("-pc_gamg_asm_use_agg","Use aggregation aggregates for ASM smoother","PCGAMGASMSetUseAggs",pc_gamg->use_aggs_in_asm,&pc_gamg->use_aggs_in_asm,NULL));
  CHKERRQ(PetscOptionsBool("-pc_gamg_use_parallel_coarse_grid_solver","Use parallel coarse grid solver (otherwise put last grid on one process)","PCGAMGSetUseParallelCoarseGridSolve",pc_gamg->use_parallel_coarse_grid_solver,&pc_gamg->use_parallel_coarse_grid_solver,NULL));
  CHKERRQ(PetscOptionsBool("-pc_gamg_cpu_pin_coarse_grids","Pin coarse grids to the CPU","PCGAMGSetCpuPinCoarseGrids",pc_gamg->cpu_pin_coarse_grids,&pc_gamg->cpu_pin_coarse_grids,NULL));
  CHKERRQ(PetscOptionsEnum("-pc_gamg_coarse_grid_layout_type","compact: place reduced grids on processes in natural order; spread: distribute to whole machine for more memory bandwidth","PCGAMGSetCoarseGridLayoutType",LayoutTypes,(PetscEnum)pc_gamg->layout_type,(PetscEnum*)&pc_gamg->layout_type,NULL));
  CHKERRQ(PetscOptionsInt("-pc_gamg_process_eq_limit","Limit (goal) on number of equations per process on coarse grids","PCGAMGSetProcEqLim",pc_gamg->min_eq_proc,&pc_gamg->min_eq_proc,NULL));
  CHKERRQ(PetscOptionsInt("-pc_gamg_coarse_eq_limit","Limit on number of equations for the coarse grid","PCGAMGSetCoarseEqLim",pc_gamg->coarse_eq_limit,&pc_gamg->coarse_eq_limit,NULL));
  CHKERRQ(PetscOptionsReal("-pc_gamg_threshold_scale","Scaling of threshold for each level not specified","PCGAMGSetThresholdScale",pc_gamg->threshold_scale,&pc_gamg->threshold_scale,NULL));
  n = PETSC_MG_MAXLEVELS;
  CHKERRQ(PetscOptionsRealArray("-pc_gamg_threshold","Relative threshold to use for dropping edges in aggregation graph","PCGAMGSetThreshold",pc_gamg->threshold,&n,&flag));
  if (!flag || n < PETSC_MG_MAXLEVELS) {
    if (!flag) n = 1;
    i = n;
    do {pc_gamg->threshold[i] = pc_gamg->threshold[i-1]*pc_gamg->threshold_scale;} while (++i<PETSC_MG_MAXLEVELS);
  }
  n = PETSC_MG_MAXLEVELS;
  CHKERRQ(PetscOptionsIntArray("-pc_gamg_rank_reduction_factors","Manual schedule of coarse grid reduction factors that overrides internal heuristics (0 for first reduction puts one process/device)","PCGAMGSetRankReductionFactors",pc_gamg->level_reduction_factors,&n,&flag));
  if (!flag) i = 0;
  else i = n;
  do {pc_gamg->level_reduction_factors[i] = -1;} while (++i<PETSC_MG_MAXLEVELS);
  CHKERRQ(PetscOptionsInt("-pc_mg_levels","Set number of MG levels","PCGAMGSetNlevels",pc_gamg->Nlevels,&pc_gamg->Nlevels,NULL));
  {
    PetscReal eminmax[2] = {0., 0.};
    n = 2;
    CHKERRQ(PetscOptionsRealArray("-pc_gamg_eigenvalues","extreme eigenvalues for smoothed aggregation","PCGAMGSetEigenvalues",eminmax,&n,&flag));
    if (flag) {
      PetscCheckFalse(n != 2,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"-pc_gamg_eigenvalues: must specify 2 parameters, min and max eigenvalues");
      CHKERRQ(PCGAMGSetEigenvalues(pc, eminmax[1], eminmax[0]));
    }
  }
  /* set options for subtype */
  if (pc_gamg->ops->setfromoptions) CHKERRQ((*pc_gamg->ops->setfromoptions)(PetscOptionsObject,pc));

  CHKERRQ(PCGetOptionsPrefix(pc, &pcpre));
  CHKERRQ(PetscSNPrintf(prefix,sizeof(prefix),"%spc_gamg_",pcpre ? pcpre : ""));
  CHKERRQ(PetscOptionsTail());
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

.seealso:  PCCreate(), PCSetType(), MatSetBlockSize(), PCMGType, PCSetCoordinates(), MatSetNearNullSpace(), PCGAMGSetType(), PCGAMGAGG, PCGAMGGEO, PCGAMGCLASSICAL, PCGAMGSetProcEqLim(),
           PCGAMGSetCoarseEqLim(), PCGAMGSetRepartition(), PCGAMGRegister(), PCGAMGSetReuseInterpolation(), PCGAMGASMSetUseAggs(), PCGAMGSetUseParallelCoarseGridSolve(), PCGAMGSetNlevels(), PCGAMGSetThreshold(), PCGAMGGetType(), PCGAMGSetReuseInterpolation(), PCGAMGSetUseSAEstEig()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_GAMG(PC pc)
{
  PC_GAMG *pc_gamg;
  PC_MG   *mg;

  PetscFunctionBegin;
   /* register AMG type */
  CHKERRQ(PCGAMGInitializePackage());

  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  CHKERRQ(PCSetType(pc, PCMG));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)pc, PCGAMG));

  /* create a supporting struct and attach it to pc */
  CHKERRQ(PetscNewLog(pc,&pc_gamg));
  CHKERRQ(PCMGSetGalerkin(pc,PC_MG_GALERKIN_EXTERNAL));
  mg           = (PC_MG*)pc->data;
  mg->innerctx = pc_gamg;

  CHKERRQ(PetscNewLog(pc,&pc_gamg->ops));

  /* these should be in subctx but repartitioning needs simple arrays */
  pc_gamg->data_sz = 0;
  pc_gamg->data    = NULL;

  /* overwrite the pointers of PCMG by the functions of base class PCGAMG */
  pc->ops->setfromoptions = PCSetFromOptions_GAMG;
  pc->ops->setup          = PCSetUp_GAMG;
  pc->ops->reset          = PCReset_GAMG;
  pc->ops->destroy        = PCDestroy_GAMG;
  mg->view                = PCView_GAMG;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCMGGetLevels_C",PCMGGetLevels_MG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCMGSetLevels_C",PCMGSetLevels_MG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetProcEqLim_C",PCGAMGSetProcEqLim_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCoarseEqLim_C",PCGAMGSetCoarseEqLim_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetRepartition_C",PCGAMGSetRepartition_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetEigenvalues_C",PCGAMGSetEigenvalues_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetUseSAEstEig_C",PCGAMGSetUseSAEstEig_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetReuseInterpolation_C",PCGAMGSetReuseInterpolation_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGASMSetUseAggs_C",PCGAMGASMSetUseAggs_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetUseParallelCoarseGridSolve_C",PCGAMGSetUseParallelCoarseGridSolve_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCpuPinCoarseGrids_C",PCGAMGSetCpuPinCoarseGrids_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCoarseGridLayoutType_C",PCGAMGSetCoarseGridLayoutType_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetThreshold_C",PCGAMGSetThreshold_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetRankReductionFactors_C",PCGAMGSetRankReductionFactors_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetThresholdScale_C",PCGAMGSetThresholdScale_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetType_C",PCGAMGSetType_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGGetType_C",PCGAMGGetType_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetNlevels_C",PCGAMGSetNlevels_GAMG));
  pc_gamg->repart           = PETSC_FALSE;
  pc_gamg->reuse_prol       = PETSC_FALSE;
  pc_gamg->use_aggs_in_asm  = PETSC_FALSE;
  pc_gamg->use_parallel_coarse_grid_solver = PETSC_FALSE;
  pc_gamg->cpu_pin_coarse_grids = PETSC_FALSE;
  pc_gamg->layout_type      = PCGAMG_LAYOUT_SPREAD;
  pc_gamg->min_eq_proc      = 50;
  pc_gamg->coarse_eq_limit  = 50;
  CHKERRQ(PetscArrayzero(pc_gamg->threshold,PETSC_MG_MAXLEVELS));
  pc_gamg->threshold_scale = 1.;
  pc_gamg->Nlevels          = PETSC_MG_MAXLEVELS;
  pc_gamg->current_level    = 0; /* don't need to init really */
  pc_gamg->use_sa_esteig    = PETSC_TRUE;
  pc_gamg->emin             = 0;
  pc_gamg->emax             = 0;

  pc_gamg->ops->createlevel = PCGAMGCreateLevel_GAMG;

  /* PCSetUp_GAMG assumes that the type has been set, so set it to the default now */
  CHKERRQ(PCGAMGSetType(pc,PCGAMGAGG));
  PetscFunctionReturn(0);
}

/*@C
 PCGAMGInitializePackage - This function initializes everything in the PCGAMG package. It is called
    from PCInitializePackage().

 Level: developer

 .seealso: PetscInitialize()
@*/
PetscErrorCode PCGAMGInitializePackage(void)
{
  PetscInt       l;

  PetscFunctionBegin;
  if (PCGAMGPackageInitialized) PetscFunctionReturn(0);
  PCGAMGPackageInitialized = PETSC_TRUE;
  CHKERRQ(PetscFunctionListAdd(&GAMGList,PCGAMGGEO,PCCreateGAMG_GEO));
  CHKERRQ(PetscFunctionListAdd(&GAMGList,PCGAMGAGG,PCCreateGAMG_AGG));
  CHKERRQ(PetscFunctionListAdd(&GAMGList,PCGAMGCLASSICAL,PCCreateGAMG_Classical));
  CHKERRQ(PetscRegisterFinalize(PCGAMGFinalizePackage));

  /* general events */
  CHKERRQ(PetscLogEventRegister("PCGAMGGraph_AGG", 0, &PC_GAMGGraph_AGG));
  CHKERRQ(PetscLogEventRegister("PCGAMGGraph_GEO", PC_CLASSID, &PC_GAMGGraph_GEO));
  CHKERRQ(PetscLogEventRegister("PCGAMGCoarse_AGG", PC_CLASSID, &PC_GAMGCoarsen_AGG));
  CHKERRQ(PetscLogEventRegister("PCGAMGCoarse_GEO", PC_CLASSID, &PC_GAMGCoarsen_GEO));
  CHKERRQ(PetscLogEventRegister("PCGAMGProl_AGG", PC_CLASSID, &PC_GAMGProlongator_AGG));
  CHKERRQ(PetscLogEventRegister("PCGAMGProl_GEO", PC_CLASSID, &PC_GAMGProlongator_GEO));
  CHKERRQ(PetscLogEventRegister("PCGAMGPOpt_AGG", PC_CLASSID, &PC_GAMGOptProlongator_AGG));

  CHKERRQ(PetscLogEventRegister("GAMG: createProl", PC_CLASSID, &petsc_gamg_setup_events[SET1]));
  CHKERRQ(PetscLogEventRegister("  Create Graph", PC_CLASSID, &petsc_gamg_setup_events[GRAPH]));
  CHKERRQ(PetscLogEventRegister("  Filter Graph", PC_CLASSID, &petsc_gamg_setup_events[SET16]));
  /* PetscLogEventRegister("    G.Mat", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_MAT]); */
  /* PetscLogEventRegister("    G.Filter", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_FILTER]); */
  /* PetscLogEventRegister("    G.Square", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_SQR]); */
  CHKERRQ(PetscLogEventRegister("  MIS/Agg", PC_CLASSID, &petsc_gamg_setup_events[SET4]));
  CHKERRQ(PetscLogEventRegister("  geo: growSupp", PC_CLASSID, &petsc_gamg_setup_events[SET5]));
  CHKERRQ(PetscLogEventRegister("  geo: triangle", PC_CLASSID, &petsc_gamg_setup_events[SET6]));
  CHKERRQ(PetscLogEventRegister("    search-set", PC_CLASSID, &petsc_gamg_setup_events[FIND_V]));
  CHKERRQ(PetscLogEventRegister("  SA: col data", PC_CLASSID, &petsc_gamg_setup_events[SET7]));
  CHKERRQ(PetscLogEventRegister("  SA: frmProl0", PC_CLASSID, &petsc_gamg_setup_events[SET8]));
  CHKERRQ(PetscLogEventRegister("  SA: smooth", PC_CLASSID, &petsc_gamg_setup_events[SET9]));
  CHKERRQ(PetscLogEventRegister("GAMG: partLevel", PC_CLASSID, &petsc_gamg_setup_events[SET2]));
  CHKERRQ(PetscLogEventRegister("  repartition", PC_CLASSID, &petsc_gamg_setup_events[SET12]));
  CHKERRQ(PetscLogEventRegister("  Invert-Sort", PC_CLASSID, &petsc_gamg_setup_events[SET13]));
  CHKERRQ(PetscLogEventRegister("  Move A", PC_CLASSID, &petsc_gamg_setup_events[SET14]));
  CHKERRQ(PetscLogEventRegister("  Move P", PC_CLASSID, &petsc_gamg_setup_events[SET15]));
  for (l=0;l<PETSC_MG_MAXLEVELS;l++) {
    char ename[32];

    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCGAMG Squ l%02d",l));
    CHKERRQ(PetscLogEventRegister(ename, PC_CLASSID, &petsc_gamg_setup_matmat_events[l][0]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCGAMG Gal l%02d",l));
    CHKERRQ(PetscLogEventRegister(ename, PC_CLASSID, &petsc_gamg_setup_matmat_events[l][1]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCGAMG Opt l%02d",l));
    CHKERRQ(PetscLogEventRegister(ename, PC_CLASSID, &petsc_gamg_setup_matmat_events[l][2]));
  }
  /* PetscLogEventRegister(" PL move data", PC_CLASSID, &petsc_gamg_setup_events[SET13]); */
  /* PetscLogEventRegister("GAMG: fix", PC_CLASSID, &petsc_gamg_setup_events[SET10]); */
  /* PetscLogEventRegister("GAMG: set levels", PC_CLASSID, &petsc_gamg_setup_events[SET11]); */
  /* create timer stages */
#if defined(GAMG_STAGES)
  {
    char     str[32];
    PetscInt lidx;
    sprintf(str,"MG Level %d (finest)",0);
    CHKERRQ(PetscLogStageRegister(str, &gamg_stages[0]));
    for (lidx=1; lidx<9; lidx++) {
      sprintf(str,"MG Level %d",(int)lidx);
      CHKERRQ(PetscLogStageRegister(str, &gamg_stages[lidx]));
    }
  }
#endif
  PetscFunctionReturn(0);
}

/*@C
 PCGAMGFinalizePackage - This function frees everything from the PCGAMG package. It is
    called from PetscFinalize() automatically.

 Level: developer

 .seealso: PetscFinalize()
@*/
PetscErrorCode PCGAMGFinalizePackage(void)
{
  PetscFunctionBegin;
  PCGAMGPackageInitialized = PETSC_FALSE;
  CHKERRQ(PetscFunctionListDestroy(&GAMGList));
  PetscFunctionReturn(0);
}

/*@C
 PCGAMGRegister - Register a PCGAMG implementation.

 Input Parameters:
 + type - string that will be used as the name of the GAMG type.
 - create - function for creating the gamg context.

  Level: advanced

 .seealso: PCGAMGType, PCGAMG, PCGAMGSetType()
@*/
PetscErrorCode PCGAMGRegister(PCGAMGType type, PetscErrorCode (*create)(PC))
{
  PetscFunctionBegin;
  CHKERRQ(PCGAMGInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&GAMGList,type,create));
  PetscFunctionReturn(0);
}
