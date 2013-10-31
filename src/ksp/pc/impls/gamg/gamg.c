/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include "petsc-private/matimpl.h"
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>
#include <../src/ksp/pc/impls/bjacobi/bjacobi.h> /* Hack to access same_local_solves */

#if defined PETSC_GAMG_USE_LOG
PetscLogEvent petsc_gamg_setup_events[NUM_SET];
#endif

#if defined PETSC_USE_LOG
PetscLogEvent PC_GAMGGgraph_AGG;
PetscLogEvent PC_GAMGGgraph_GEO;
PetscLogEvent PC_GAMGCoarsen_AGG;
PetscLogEvent PC_GAMGCoarsen_GEO;
PetscLogEvent PC_GAMGProlongator_AGG;
PetscLogEvent PC_GAMGProlongator_GEO;
PetscLogEvent PC_GAMGOptprol_AGG;
#endif

#define GAMG_MAXLEVELS 30

/* #define GAMG_STAGES */
#if (defined PETSC_GAMG_USE_LOG && defined GAMG_STAGES)
static PetscLogStage gamg_stages[GAMG_MAXLEVELS];
#endif

static PetscFunctionList GAMGList = 0;
static PetscBool PCGAMGPackageInitialized;

/* ----------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCReset_GAMG"
PetscErrorCode PCReset_GAMG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if (pc_gamg->data) { /* this should not happen, cleaned up in SetUp */
    PetscPrintf(PetscObjectComm((PetscObject)pc),"***[%d]%s this should not happen, cleaned up in SetUp\n",0,__FUNCT__);
    ierr = PetscFree(pc_gamg->data);CHKERRQ(ierr);
  }
  pc_gamg->data = NULL; pc_gamg->data_sz = 0;

  if (pc_gamg->orig_data) {
    ierr = PetscFree(pc_gamg->orig_data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   createLevel: create coarse op with RAP.  repartition and/or reduce number
     of active processors.

   Input Parameter:
   . pc - parameters + side effect: coarse data in 'pc_gamg->data' and
          'pc_gamg->data_sz' are changed via repartitioning/reduction.
   . Amat_fine - matrix on this fine (k) level
   . cr_bs - coarse block size
   . isLast -
   In/Output Parameter:
   . a_P_inout - prolongation operator to the next level (k-->k-1)
   . a_nactive_proc - number of active procs
   Output Parameter:
   . a_Amat_crs - coarse matrix that is created (k-1)
*/

#undef __FUNCT__
#define __FUNCT__ "createLevel"
static PetscErrorCode createLevel(const PC pc,const Mat Amat_fine,const PetscInt cr_bs,const PetscBool isLast,
                                  Mat *a_P_inout,Mat *a_Amat_crs,PetscMPIInt *a_nactive_proc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg         = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg    = (PC_GAMG*)mg->innerctx;
  const PetscBool repart      = pc_gamg->repart;
  const PetscInt  min_eq_proc = pc_gamg->min_eq_proc, coarse_max = pc_gamg->coarse_eq_limit;
  Mat             Cmat,Pold=*a_P_inout;
  MPI_Comm        comm;
  PetscMPIInt     rank,size,new_size,nactive=*a_nactive_proc;
  PetscInt        ncrs_eq,ncrs_prim,f_bs;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat_fine,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Amat_fine, &f_bs);CHKERRQ(ierr);
  /* RAP */
  ierr = MatPtAP(Amat_fine, Pold, MAT_INITIAL_MATRIX, 2.0, &Cmat);CHKERRQ(ierr);

  /* set 'ncrs_prim' (nodes), 'ncrs_eq' (equations)*/
  ncrs_prim = pc_gamg->data_sz/pc_gamg->data_cell_cols/pc_gamg->data_cell_rows;
  if (pc_gamg->data_sz % (pc_gamg->data_cell_cols*pc_gamg->data_cell_rows)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"pc_gamg->data_sz %D not divisible by (pc_gamg->data_cell_cols %D *pc_gamg->data_cell_rows %D)",pc_gamg->data_sz,pc_gamg->data_cell_cols,pc_gamg->data_cell_rows);
  ierr = MatGetLocalSize(Cmat, &ncrs_eq, NULL);CHKERRQ(ierr);

  /* get number of PEs to make active 'new_size', reduce, can be any integer 1-P */
  {
    PetscInt ncrs_eq_glob;
    ierr     = MatGetSize(Cmat, &ncrs_eq_glob, NULL);CHKERRQ(ierr);
    new_size = (PetscMPIInt)((float)ncrs_eq_glob/(float)min_eq_proc + 0.5); /* hardwire min. number of eq/proc */
    if (new_size == 0 || ncrs_eq_glob < coarse_max) new_size = 1;
    else if (new_size >= nactive) new_size = nactive; /* no change, rare */
    if (isLast) new_size = 1;
  }

  if (!repart && new_size==nactive) *a_Amat_crs = Cmat; /* output - no repartitioning or reduction - could bail here */
  else {
    const PetscInt *idx,ndata_rows=pc_gamg->data_cell_rows,ndata_cols=pc_gamg->data_cell_cols,node_data_sz=ndata_rows*ndata_cols;
    PetscInt       *counts,*newproc_idx,ii,jj,kk,strideNew,*tidx,ncrs_prim_new,ncrs_eq_new,nloc_old;
    IS             is_eq_newproc,is_eq_num,is_eq_num_prim,isscat,new_eq_indices;
    VecScatter     vecscat;
    PetscScalar    *array;
    Vec            src_crd, dest_crd;

    nloc_old = ncrs_eq/cr_bs;
    if (ncrs_eq % cr_bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ncrs_eq %D not divisible by cr_bs %D",ncrs_eq,cr_bs);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET12],0,0,0,0);CHKERRQ(ierr);
#endif
    /* make 'is_eq_newproc' */
    ierr = PetscMalloc(size*sizeof(PetscInt), &counts);CHKERRQ(ierr);
    if (repart) {
      /* Repartition Cmat_{k} and move colums of P^{k}_{k-1} and coordinates of primal part accordingly */
      Mat adj;

      if (pc_gamg->verbose>0) {
        if (pc_gamg->verbose==1) PetscPrintf(comm,"\t[%d]%s repartition: size (active): %d --> %d, neq = %d\n",rank,__FUNCT__,*a_nactive_proc,new_size,ncrs_eq);
        else {
          PetscInt n;
          ierr = MPI_Allreduce(&ncrs_eq, &n, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
          PetscPrintf(comm,"\t[%d]%s repartition: size (active): %d --> %d, neq = %d\n",rank,__FUNCT__,*a_nactive_proc,new_size,n);
        }
      }

      /* get 'adj' */
      if (cr_bs == 1) {
        ierr = MatConvert(Cmat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj);CHKERRQ(ierr);
      } else {
        /* make a scalar matrix to partition (no Stokes here) */
        Mat               tMat;
        PetscInt          Istart_crs,Iend_crs,ncols,jj,Ii;
        const PetscScalar *vals;
        const PetscInt    *idx;
        PetscInt          *d_nnz, *o_nnz, M, N;
        static PetscInt   llev = 0;

        ierr = PetscMalloc(ncrs_prim*sizeof(PetscInt), &d_nnz);CHKERRQ(ierr);
        ierr = PetscMalloc(ncrs_prim*sizeof(PetscInt), &o_nnz);CHKERRQ(ierr);
        ierr = MatGetOwnershipRange(Cmat, &Istart_crs, &Iend_crs);CHKERRQ(ierr);
        ierr = MatGetSize(Cmat, &M, &N);CHKERRQ(ierr);
        for (Ii = Istart_crs, jj = 0; Ii < Iend_crs; Ii += cr_bs, jj++) {
          ierr      = MatGetRow(Cmat,Ii,&ncols,0,0);CHKERRQ(ierr);
          d_nnz[jj] = ncols/cr_bs;
          o_nnz[jj] = ncols/cr_bs;
          ierr      = MatRestoreRow(Cmat,Ii,&ncols,0,0);CHKERRQ(ierr);
          if (d_nnz[jj] > ncrs_prim) d_nnz[jj] = ncrs_prim;
          if (o_nnz[jj] > (M/cr_bs-ncrs_prim)) o_nnz[jj] = M/cr_bs-ncrs_prim;
        }

        ierr = MatCreate(comm, &tMat);CHKERRQ(ierr);
        ierr = MatSetSizes(tMat, ncrs_prim, ncrs_prim,PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
        ierr = MatSetType(tMat,MATAIJ);CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(tMat,0,d_nnz);CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(tMat,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
        ierr = PetscFree(d_nnz);CHKERRQ(ierr);
        ierr = PetscFree(o_nnz);CHKERRQ(ierr);

        for (ii = Istart_crs; ii < Iend_crs; ii++) {
          PetscInt dest_row = ii/cr_bs;
          ierr = MatGetRow(Cmat,ii,&ncols,&idx,&vals);CHKERRQ(ierr);
          for (jj = 0; jj < ncols; jj++) {
            PetscInt    dest_col = idx[jj]/cr_bs;
            PetscScalar v        = 1.0;
            ierr = MatSetValues(tMat,1,&dest_row,1,&dest_col,&v,ADD_VALUES);CHKERRQ(ierr);
          }
          ierr = MatRestoreRow(Cmat,ii,&ncols,&idx,&vals);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(tMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(tMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

        if (llev++ == -1) {
          PetscViewer viewer; char fname[32];
          ierr = PetscSNPrintf(fname,sizeof(fname),"part_mat_%D.mat",llev);CHKERRQ(ierr);
          PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&viewer);
          ierr = MatView(tMat, viewer);CHKERRQ(ierr);
          ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        }

        ierr = MatConvert(tMat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj);CHKERRQ(ierr);

        ierr = MatDestroy(&tMat);CHKERRQ(ierr);
      } /* create 'adj' */

      { /* partition: get newproc_idx */
        char            prefix[256];
        const char      *pcpre;
        const PetscInt  *is_idx;
        MatPartitioning mpart;
        IS              proc_is;
        PetscInt        targetPE;

        ierr = MatPartitioningCreate(comm, &mpart);CHKERRQ(ierr);
        ierr = MatPartitioningSetAdjacency(mpart, adj);CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc, &pcpre);CHKERRQ(ierr);
        ierr = PetscSNPrintf(prefix,sizeof(prefix),"%spc_gamg_",pcpre ? pcpre : "");CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix);CHKERRQ(ierr);
        ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
        ierr = MatPartitioningSetNParts(mpart, new_size);CHKERRQ(ierr);
        ierr = MatPartitioningApply(mpart, &proc_is);CHKERRQ(ierr);
        ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);

        /* collect IS info */
        ierr     = PetscMalloc(ncrs_eq*sizeof(PetscInt), &newproc_idx);CHKERRQ(ierr);
        ierr     = ISGetIndices(proc_is, &is_idx);CHKERRQ(ierr);
        targetPE = 1; /* bring to "front" of machine */
        /*targetPE = size/new_size;*/ /* spread partitioning across machine */
        for (kk = jj = 0 ; kk < nloc_old ; kk++) {
          for (ii = 0 ; ii < cr_bs ; ii++, jj++) {
            newproc_idx[jj] = is_idx[kk] * targetPE; /* distribution */
          }
        }
        ierr = ISRestoreIndices(proc_is, &is_idx);CHKERRQ(ierr);
        ierr = ISDestroy(&proc_is);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&adj);CHKERRQ(ierr);

      ierr = ISCreateGeneral(comm, ncrs_eq, newproc_idx, PETSC_COPY_VALUES, &is_eq_newproc);CHKERRQ(ierr);
      if (newproc_idx != 0) {
        ierr = PetscFree(newproc_idx);CHKERRQ(ierr);
      }
    } else { /* simple aggreagtion of parts -- 'is_eq_newproc' */

      PetscInt rfactor,targetPE;
      /* find factor */
      if (new_size == 1) rfactor = size; /* easy */
      else {
        PetscReal best_fact = 0.;
        jj = -1;
        for (kk = 1 ; kk <= size ; kk++) {
          if (size%kk==0) { /* a candidate */
            PetscReal nactpe = (PetscReal)size/(PetscReal)kk, fact = nactpe/(PetscReal)new_size;
            if (fact > 1.0) fact = 1./fact; /* keep fact < 1 */
            if (fact > best_fact) {
              best_fact = fact; jj = kk;
            }
          }
        }
        if (jj != -1) rfactor = jj;
        else rfactor = 1; /* does this happen .. a prime */
      }
      new_size = size/rfactor;

      if (new_size==nactive) {
        *a_Amat_crs = Cmat; /* output - no repartitioning or reduction, bail out because nested here */
        ierr        = PetscFree(counts);CHKERRQ(ierr);
        if (pc_gamg->verbose>0) {
          PetscPrintf(comm,"\t[%d]%s aggregate processors noop: new_size=%d, neq(loc)=%d\n",rank,__FUNCT__,new_size,ncrs_eq);
        }
        PetscFunctionReturn(0);
      }

      if (pc_gamg->verbose) PetscPrintf(comm,"\t[%d]%s number of equations (loc) %d with simple aggregation\n",rank,__FUNCT__,ncrs_eq);
      targetPE = rank/rfactor;
      ierr     = ISCreateStride(comm, ncrs_eq, targetPE, 0, &is_eq_newproc);CHKERRQ(ierr);
    } /* end simple 'is_eq_newproc' */

    /*
     Create an index set from the is_eq_newproc index set to indicate the mapping TO
     */
    ierr = ISPartitioningToNumbering(is_eq_newproc, &is_eq_num);CHKERRQ(ierr);
    is_eq_num_prim = is_eq_num;
    /*
      Determine how many equations/vertices are assigned to each processor
     */
    ierr        = ISPartitioningCount(is_eq_newproc, size, counts);CHKERRQ(ierr);
    ncrs_eq_new = counts[rank];
    ierr        = ISDestroy(&is_eq_newproc);CHKERRQ(ierr);
    ncrs_prim_new = ncrs_eq_new/cr_bs; /* eqs */

    ierr = PetscFree(counts);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET12],0,0,0,0);CHKERRQ(ierr);
#endif

    /* move data (for primal equations only) */
    /* Create a vector to contain the newly ordered element information */
    ierr = VecCreate(comm, &dest_crd);CHKERRQ(ierr);
    ierr = VecSetSizes(dest_crd, node_data_sz*ncrs_prim_new, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(dest_crd,VECSTANDARD);CHKERRQ(ierr); /* this is needed! */
    /*
     There are 'ndata_rows*ndata_cols' data items per node, (one can think of the vectors of having
     a block size of ...).  Note, ISs are expanded into equation space by 'cr_bs'.
     */
    ierr = PetscMalloc((ncrs_prim*node_data_sz)*sizeof(PetscInt), &tidx);CHKERRQ(ierr);
    ierr = ISGetIndices(is_eq_num_prim, &idx);CHKERRQ(ierr);
    for (ii=0,jj=0; ii<ncrs_prim; ii++) {
      PetscInt id = idx[ii*cr_bs]/cr_bs; /* get node back */
      for (kk=0; kk<node_data_sz; kk++, jj++) tidx[jj] = id*node_data_sz + kk;
    }
    ierr = ISRestoreIndices(is_eq_num_prim, &idx);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, node_data_sz*ncrs_prim, tidx, PETSC_COPY_VALUES, &isscat);CHKERRQ(ierr);
    ierr = PetscFree(tidx);CHKERRQ(ierr);
    /*
     Create a vector to contain the original vertex information for each element
     */
    ierr = VecCreateSeq(PETSC_COMM_SELF, node_data_sz*ncrs_prim, &src_crd);CHKERRQ(ierr);
    for (jj=0; jj<ndata_cols; jj++) {
      const PetscInt stride0=ncrs_prim*pc_gamg->data_cell_rows;
      for (ii=0; ii<ncrs_prim; ii++) {
        for (kk=0; kk<ndata_rows; kk++) {
          PetscInt    ix = ii*ndata_rows + kk + jj*stride0, jx = ii*node_data_sz + kk*ndata_cols + jj;
          PetscScalar tt = (PetscScalar)pc_gamg->data[ix];
          ierr = VecSetValues(src_crd, 1, &jx, &tt, INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = VecAssemblyBegin(src_crd);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(src_crd);CHKERRQ(ierr);
    /*
      Scatter the element vertex information (still in the original vertex ordering)
      to the correct processor
    */
    ierr = VecScatterCreate(src_crd, NULL, dest_crd, isscat, &vecscat);CHKERRQ(ierr);
    ierr = ISDestroy(&isscat);CHKERRQ(ierr);
    ierr = VecScatterBegin(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vecscat);CHKERRQ(ierr);
    ierr = VecDestroy(&src_crd);CHKERRQ(ierr);
    /*
      Put the element vertex data into a new allocation of the gdata->ele
    */
    ierr = PetscFree(pc_gamg->data);CHKERRQ(ierr);
    ierr = PetscMalloc(node_data_sz*ncrs_prim_new*sizeof(PetscReal), &pc_gamg->data);CHKERRQ(ierr);

    pc_gamg->data_sz = node_data_sz*ncrs_prim_new;
    strideNew        = ncrs_prim_new*ndata_rows;

    ierr = VecGetArray(dest_crd, &array);CHKERRQ(ierr);
    for (jj=0; jj<ndata_cols; jj++) {
      for (ii=0; ii<ncrs_prim_new; ii++) {
        for (kk=0; kk<ndata_rows; kk++) {
          PetscInt ix = ii*ndata_rows + kk + jj*strideNew, jx = ii*node_data_sz + kk*ndata_cols + jj;
          pc_gamg->data[ix] = PetscRealPart(array[jx]);
        }
      }
    }
    ierr = VecRestoreArray(dest_crd, &array);CHKERRQ(ierr);
    ierr = VecDestroy(&dest_crd);CHKERRQ(ierr);

    /* move A and P (columns) with new layout */
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET13],0,0,0,0);CHKERRQ(ierr);
#endif

    /*
      Invert for MatGetSubMatrix
    */
    ierr = ISInvertPermutation(is_eq_num, ncrs_eq_new, &new_eq_indices);CHKERRQ(ierr);
    ierr = ISSort(new_eq_indices);CHKERRQ(ierr); /* is this needed? */
    ierr = ISSetBlockSize(new_eq_indices, cr_bs);CHKERRQ(ierr);
    if (is_eq_num != is_eq_num_prim) {
      ierr = ISDestroy(&is_eq_num_prim);CHKERRQ(ierr); /* could be same as 'is_eq_num' */
    }
    ierr = ISDestroy(&is_eq_num);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET13],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET14],0,0,0,0);CHKERRQ(ierr);
#endif
    /* 'a_Amat_crs' output */
    {
      Mat mat;
      ierr        = MatGetSubMatrix(Cmat, new_eq_indices, new_eq_indices, MAT_INITIAL_MATRIX, &mat);CHKERRQ(ierr);
      *a_Amat_crs = mat;

      if (!PETSC_TRUE) {
        PetscInt cbs, rbs;
        ierr = MatGetBlockSizes(Cmat, &rbs, &cbs);CHKERRQ(ierr);
        ierr = PetscPrintf(MPI_COMM_SELF,"[%d]%s Old Mat rbs=%d cbs=%d\n",rank,__FUNCT__,rbs,cbs);CHKERRQ(ierr);
        ierr = MatGetBlockSizes(mat, &rbs, &cbs);CHKERRQ(ierr);
        ierr = PetscPrintf(MPI_COMM_SELF,"[%d]%s New Mat rbs=%d cbs=%d cr_bs=%d\n",rank,__FUNCT__,rbs,cbs,cr_bs);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&Cmat);CHKERRQ(ierr);

#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET14],0,0,0,0);CHKERRQ(ierr);
#endif
    /* prolongator */
    {
      IS       findices;
      PetscInt Istart,Iend;
      Mat      Pnew;
      ierr = MatGetOwnershipRange(Pold, &Istart, &Iend);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET15],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = ISCreateStride(comm,Iend-Istart,Istart,1,&findices);CHKERRQ(ierr);
      ierr = ISSetBlockSize(findices,f_bs);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(Pold, findices, new_eq_indices, MAT_INITIAL_MATRIX, &Pnew);CHKERRQ(ierr);
      ierr = ISDestroy(&findices);CHKERRQ(ierr);

      if (!PETSC_TRUE) {
        PetscInt cbs, rbs;
        ierr = MatGetBlockSizes(Pold, &rbs, &cbs);CHKERRQ(ierr);
        ierr = PetscPrintf(MPI_COMM_SELF,"[%d]%s Pold rbs=%d cbs=%d\n",rank,__FUNCT__,rbs,cbs);CHKERRQ(ierr);
        ierr = MatGetBlockSizes(Pnew, &rbs, &cbs);CHKERRQ(ierr);
        ierr = PetscPrintf(MPI_COMM_SELF,"[%d]%s Pnew rbs=%d cbs=%d\n",rank,__FUNCT__,rbs,cbs);CHKERRQ(ierr);
      }
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET15],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = MatDestroy(a_P_inout);CHKERRQ(ierr);

      /* output - repartitioned */
      *a_P_inout = Pnew;
    }
    ierr = ISDestroy(&new_eq_indices);CHKERRQ(ierr);

    *a_nactive_proc = new_size; /* output */
  }

  /* outout matrix data */
  if (!PETSC_TRUE) {
    PetscViewer viewer; char fname[32]; static int llev=0; Cmat = *a_Amat_crs;
    if (llev==0) {
      sprintf(fname,"Cmat_%d.m",llev++);
      PetscViewerASCIIOpen(comm,fname,&viewer);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = MatView(Amat_fine, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);
    }
    sprintf(fname,"Cmat_%d.m",llev++);
    PetscViewerASCIIOpen(comm,fname,&viewer);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(Cmat, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_GAMG - Prepares for the use of the GAMG preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_GAMG"
PetscErrorCode PCSetUp_GAMG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  Mat            Pmat     = pc->pmat;
  PetscInt       fine_level,level,level1,bs,M,qq,lidx,nASMBlocksArr[GAMG_MAXLEVELS];
  MPI_Comm       comm;
  PetscMPIInt    rank,size,nactivepe;
  Mat            Aarr[GAMG_MAXLEVELS],Parr[GAMG_MAXLEVELS];
  PetscReal      emaxs[GAMG_MAXLEVELS];
  IS             *ASMLocalIDsArr[GAMG_MAXLEVELS];
  PetscLogDouble nnz0=0.,nnztot=0.;
  MatInfo        info;
  PetscBool      redo_mesh_setup = (PetscBool)(!pc_gamg->reuse_prol);

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (pc_gamg->verbose>2) PetscPrintf(comm,"[%d]%s pc_gamg->setup_count=%d pc->setupcalled=%d\n",rank,__FUNCT__,pc_gamg->setup_count,pc->setupcalled);

  if (pc_gamg->setup_count++ > 0) {
    if (redo_mesh_setup) {
      /* reset everything */
      ierr = PCReset_MG(pc);CHKERRQ(ierr);
      pc->setupcalled = 0;
    } else {
      PC_MG_Levels **mglevels = mg->levels;
      /* just do Galerkin grids */
      Mat          B,dA,dB;

     if (!pc->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PCSetUp() has not been called yet");
      if (pc_gamg->Nlevels > 1) {
        /* currently only handle case where mat and pmat are the same on coarser levels */
        ierr = KSPGetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,&dA,&dB,NULL);CHKERRQ(ierr);
        /* (re)set to get dirty flag */
        ierr = KSPSetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,dA,dB,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

        for (level=pc_gamg->Nlevels-2; level>=0; level--) {
          /* the first time through the matrix structure has changed from repartitioning */
          if (pc_gamg->setup_count==2) {
            ierr = MatPtAP(dB,mglevels[level+1]->interpolate,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
            ierr = MatDestroy(&mglevels[level]->A);CHKERRQ(ierr);

            mglevels[level]->A = B;
          } else {
            ierr = KSPGetOperators(mglevels[level]->smoothd,NULL,&B,NULL);CHKERRQ(ierr);
            ierr = MatPtAP(dB,mglevels[level+1]->interpolate,MAT_REUSE_MATRIX,1.0,&B);CHKERRQ(ierr);
          }
          ierr = KSPSetOperators(mglevels[level]->smoothd,B,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
          dB   = B;
        }
      }

      ierr = PCSetUp_MG(pc);CHKERRQ(ierr);

      /* PCSetUp_MG seems to insists on setting this to GMRES */
      ierr = KSPSetType(mglevels[0]->smoothd, KSPPREONLY);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (!pc_gamg->data) {
    if (pc_gamg->orig_data) {
      ierr = MatGetBlockSize(Pmat, &bs);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Pmat, &qq, NULL);CHKERRQ(ierr);

      pc_gamg->data_sz        = (qq/bs)*pc_gamg->orig_data_cell_rows*pc_gamg->orig_data_cell_cols;
      pc_gamg->data_cell_rows = pc_gamg->orig_data_cell_rows;
      pc_gamg->data_cell_cols = pc_gamg->orig_data_cell_cols;

      ierr = PetscMalloc(pc_gamg->data_sz*sizeof(PetscReal), &pc_gamg->data);CHKERRQ(ierr);
      for (qq=0; qq<pc_gamg->data_sz; qq++) pc_gamg->data[qq] = pc_gamg->orig_data[qq];
    } else {
      if (!pc_gamg->ops->createdefaultdata) SETERRQ(comm,PETSC_ERR_PLIB,"'createdefaultdata' not set(?) need to support NULL data");
      ierr = pc_gamg->ops->createdefaultdata(pc,Pmat);CHKERRQ(ierr);
    }
  }

  /* cache original data for reuse */
  if (!pc_gamg->orig_data && redo_mesh_setup) {
    ierr = PetscMalloc(pc_gamg->data_sz*sizeof(PetscReal), &pc_gamg->orig_data);CHKERRQ(ierr);
    for (qq=0; qq<pc_gamg->data_sz; qq++) pc_gamg->orig_data[qq] = pc_gamg->data[qq];
    pc_gamg->orig_data_cell_rows = pc_gamg->data_cell_rows;
    pc_gamg->orig_data_cell_cols = pc_gamg->data_cell_cols;
  }

  /* get basic dims */
  ierr = MatGetBlockSize(Pmat, &bs);CHKERRQ(ierr);

  ierr = MatGetSize(Pmat, &M, &qq);CHKERRQ(ierr);
  if (pc_gamg->verbose) {
    PetscInt NN = M;
    if (pc_gamg->verbose==1) {
      ierr =  MatGetInfo(Pmat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Pmat, &NN, &qq);CHKERRQ(ierr);
    } else {
      ierr = MatGetInfo(Pmat,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
    }
    nnz0   = info.nz_used;
    nnztot = info.nz_used;
    ierr   = PetscPrintf(comm,"\t[%d]%s level %d N=%d, n data rows=%d, n data cols=%d, nnz/row (ave)=%d, np=%d\n",
                         rank,__FUNCT__,0,M,pc_gamg->data_cell_rows,pc_gamg->data_cell_cols,
                         (int)(nnz0/(PetscReal)NN),size);CHKERRQ(ierr);
  }

  /* Get A_i and R_i */
  for (level=0, Aarr[0]=Pmat, nactivepe = size; /* hard wired stopping logic */
       level < (pc_gamg->Nlevels-1) && (level==0 || M>pc_gamg->coarse_eq_limit);  /* && (size==1 || nactivepe>1); */
       level++) {
    level1 = level + 1;
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#if (defined GAMG_STAGES)
    ierr = PetscLogStagePush(gamg_stages[level]);CHKERRQ(ierr);
#endif
#endif
    { /* construct prolongator */
      Mat              Gmat;
      PetscCoarsenData *agg_lists;
      Mat              Prol11;

      ierr = pc_gamg->ops->graph(pc,Aarr[level], &Gmat);CHKERRQ(ierr);
      ierr = pc_gamg->ops->coarsen(pc, &Gmat, &agg_lists);CHKERRQ(ierr);
      ierr = pc_gamg->ops->prolongator(pc,Aarr[level],Gmat,agg_lists,&Prol11);CHKERRQ(ierr);

      /* could have failed to create new level */
      if (Prol11) {
        /* get new block size of coarse matrices */
        ierr = MatGetBlockSizes(Prol11, NULL, &bs);CHKERRQ(ierr);

        if (pc_gamg->ops->optprol) {
          /* smooth */
          ierr = pc_gamg->ops->optprol(pc, Aarr[level], &Prol11);CHKERRQ(ierr);
        }

        Parr[level1] = Prol11;
      } else Parr[level1] = NULL;

      if (pc_gamg->use_aggs_in_gasm) {
        ierr = PetscCDGetASMBlocks(agg_lists, bs, &nASMBlocksArr[level], &ASMLocalIDsArr[level]);CHKERRQ(ierr);
      }

      ierr = MatDestroy(&Gmat);CHKERRQ(ierr);
      ierr = PetscCDDestroy(agg_lists);CHKERRQ(ierr);
    } /* construct prolongator scope */
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#endif
    /* cache eigen estimate */
    if (pc_gamg->emax_id != -1) {
      PetscBool flag;
      ierr = PetscObjectComposedDataGetReal((PetscObject)Aarr[level], pc_gamg->emax_id, emaxs[level], flag);CHKERRQ(ierr);
      if (!flag) emaxs[level] = -1.;
    } else emaxs[level] = -1.;
    if (level==0) Aarr[0] = Pmat; /* use Pmat for finest level setup */
    if (!Parr[level1]) {
      if (pc_gamg->verbose) {
        ierr =  PetscPrintf(comm,"\t[%d]%s stop gridding, level %d\n",rank,__FUNCT__,level);CHKERRQ(ierr);
      }
      break;
    }
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif

    ierr = createLevel(pc, Aarr[level], bs, (PetscBool)(level==pc_gamg->Nlevels-2),
                       &Parr[level1], &Aarr[level1], &nactivepe);CHKERRQ(ierr);

#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = MatGetSize(Aarr[level1], &M, &qq);CHKERRQ(ierr);

    if (pc_gamg->verbose > 0) {
      PetscInt NN = M;
      if (pc_gamg->verbose==1) {
        ierr = MatGetInfo(Aarr[level1],MAT_LOCAL,&info);CHKERRQ(ierr);
        ierr = MatGetLocalSize(Aarr[level1], &NN, &qq);CHKERRQ(ierr);
      } else {
        ierr = MatGetInfo(Aarr[level1], MAT_GLOBAL_SUM, &info);CHKERRQ(ierr);
      }

      nnztot += info.nz_used;
      ierr    = PetscPrintf(comm,"\t\t[%d]%s %d) N=%d, n data cols=%d, nnz/row (ave)=%d, %d active pes\n",
                            rank,__FUNCT__,(int)level1,M,pc_gamg->data_cell_cols,
                            (int)(info.nz_used/(PetscReal)NN), nactivepe);CHKERRQ(ierr);
    }

    /* stop if one node -- could pull back for singular problems */
    if (M/pc_gamg->data_cell_cols < 2) {
      level++;
      break;
    }
#if (defined PETSC_GAMG_USE_LOG && defined GAMG_STAGES)
    ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif
  } /* levels */

  if (pc_gamg->data) {
    ierr          = PetscFree(pc_gamg->data);CHKERRQ(ierr);
    pc_gamg->data = NULL;
  }

  if (pc_gamg->verbose) PetscPrintf(comm,"\t[%d]%s %d levels, grid complexity = %g\n",0,__FUNCT__,level+1,nnztot/nnz0);
  pc_gamg->Nlevels = level + 1;
  fine_level       = level;
  ierr             = PCMGSetLevels(pc,pc_gamg->Nlevels,NULL);CHKERRQ(ierr);

  /* simple setup */
  if (!PETSC_TRUE) {
    PC_MG_Levels **mglevels = mg->levels;
    for (lidx=0,level=pc_gamg->Nlevels-1;
         lidx<fine_level;
         lidx++, level--) {
      ierr = PCMGSetInterpolation(pc, lidx+1, Parr[level]);CHKERRQ(ierr);
      ierr = KSPSetOperators(mglevels[lidx]->smoothd, Aarr[level], Aarr[level], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&Parr[level]);CHKERRQ(ierr);
      ierr = MatDestroy(&Aarr[level]);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(mglevels[fine_level]->smoothd, Aarr[0], Aarr[0], SAME_NONZERO_PATTERN);CHKERRQ(ierr);

    ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  } else if (pc_gamg->Nlevels > 1) { /* don't setup MG if one level */
    /* set default smoothers & set operators */
    for (lidx = 1, level = pc_gamg->Nlevels-2;
         lidx <= fine_level;
         lidx++, level--) {
      KSP smoother;
      PC  subpc;

      ierr = PCMGGetSmoother(pc, lidx, &smoother);CHKERRQ(ierr);
      ierr = KSPGetPC(smoother, &subpc);CHKERRQ(ierr);

      ierr = KSPSetNormType(smoother, KSP_NORM_NONE);CHKERRQ(ierr);
      /* set ops */
      ierr = KSPSetOperators(smoother, Aarr[level], Aarr[level], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PCMGSetInterpolation(pc, lidx, Parr[level+1]);CHKERRQ(ierr);

      /* set defaults */
      ierr = KSPSetType(smoother, KSPCHEBYSHEV);CHKERRQ(ierr);

      /* override defaults and command line args (!) */
      if (pc_gamg->use_aggs_in_gasm) {
        PetscInt sz;
        IS       *is;

        sz   = nASMBlocksArr[level];
        is   = ASMLocalIDsArr[level];
        ierr = PCSetType(subpc, PCGASM);CHKERRQ(ierr);
        if (sz==0) {
          IS       is;
          PetscInt my0,kk;
          ierr = MatGetOwnershipRange(Aarr[level], &my0, &kk);CHKERRQ(ierr);
          ierr = ISCreateGeneral(PETSC_COMM_SELF, 1, &my0, PETSC_COPY_VALUES, &is);CHKERRQ(ierr);
          ierr = PCGASMSetSubdomains(subpc, 1, &is, NULL);CHKERRQ(ierr);
          ierr = ISDestroy(&is);CHKERRQ(ierr);
        } else {
          PetscInt kk;
          ierr = PCGASMSetSubdomains(subpc, sz, is, NULL);CHKERRQ(ierr);
          for (kk=0; kk<sz; kk++) {
            ierr = ISDestroy(&is[kk]);CHKERRQ(ierr);
          }
          ierr = PetscFree(is);CHKERRQ(ierr);
        }
        ierr = PCGASMSetOverlap(subpc, 0);CHKERRQ(ierr);

        ASMLocalIDsArr[level] = NULL;
        nASMBlocksArr[level]  = 0;
        ierr                  = PCGASMSetType(subpc, PC_GASM_BASIC);CHKERRQ(ierr);
      } else {
        ierr = PCSetType(subpc, PCSOR);CHKERRQ(ierr);
      }
    }
    {
      /* coarse grid */
      KSP smoother,*k2; PC subpc,pc2; PetscInt ii,first;
      Mat Lmat = Aarr[(level=pc_gamg->Nlevels-1)]; lidx = 0;
      ierr = PCMGGetSmoother(pc, lidx, &smoother);CHKERRQ(ierr);
      ierr = KSPSetOperators(smoother, Lmat, Lmat, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetNormType(smoother, KSP_NORM_NONE);CHKERRQ(ierr);
      ierr = KSPGetPC(smoother, &subpc);CHKERRQ(ierr);
      ierr = PCSetType(subpc, PCBJACOBI);CHKERRQ(ierr);
      ierr = PCSetUp(subpc);CHKERRQ(ierr);
      ierr = PCBJacobiGetSubKSP(subpc,&ii,&first,&k2);CHKERRQ(ierr);
      if (ii != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ii %D is not one",ii);
      ierr = KSPGetPC(k2[0],&pc2);CHKERRQ(ierr);
      ierr = PCSetType(pc2, PCLU);CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(pc2,MAT_SHIFT_INBLOCKS);CHKERRQ(ierr);
      ierr = KSPSetTolerances(k2[0],PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
      /* This flag gets reset by PCBJacobiGetSubKSP(), but our BJacobi really does the same algorithm everywhere (and in
       * fact, all but one process will have zero dofs), so we reset the flag to avoid having PCView_BJacobi attempt to
       * view every subdomain as though they were different. */
      ((PC_BJacobi*)subpc->data)->same_local_solves = PETSC_TRUE;
    }

    /* should be called in PCSetFromOptions_GAMG(), but cannot be called prior to PCMGSetLevels() */
    ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
    ierr = PCSetFromOptions_MG(pc);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (mg->galerkin != 2) SETERRQ(comm,PETSC_ERR_USER,"GAMG does Galerkin manually so the -pc_mg_galerkin option must not be used.");

    /* create cheby smoothers */
    for (lidx = 1, level = pc_gamg->Nlevels-2;
         lidx <= fine_level;
         lidx++, level--) {
      KSP       smoother;
      PetscBool flag,flag2;
      PC        subpc;

      ierr = PCMGGetSmoother(pc, lidx, &smoother);CHKERRQ(ierr);
      ierr = KSPGetPC(smoother, &subpc);CHKERRQ(ierr);

      /* do my own cheby */
      ierr = PetscObjectTypeCompare((PetscObject)smoother, KSPCHEBYSHEV, &flag);CHKERRQ(ierr);
      if (flag) {
        PetscReal emax, emin;
        ierr = PetscObjectTypeCompare((PetscObject)subpc, PCJACOBI, &flag);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)subpc, PCSOR, &flag2);CHKERRQ(ierr);
        if ((flag||flag2) && emaxs[level] > 0.0) emax=emaxs[level]; /* eigen estimate only for diagnal PC but lets acccept SOR because it is close and safe (always lower) */
        else { /* eigen estimate 'emax' -- this is done in cheby */
          KSP eksp;
          Mat Lmat = Aarr[level];
          Vec bb, xx;

          ierr = MatGetVecs(Lmat, &bb, 0);CHKERRQ(ierr);
          ierr = MatGetVecs(Lmat, &xx, 0);CHKERRQ(ierr);
          {
            PetscRandom rctx;
            ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
            ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
            ierr = VecSetRandom(bb,rctx);CHKERRQ(ierr);
            ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
          }

          /* zeroing out BC rows -- needed for crazy matrices */
          {
            PetscInt    Istart,Iend,ncols,jj,Ii;
            PetscScalar zero = 0.0;
            ierr = MatGetOwnershipRange(Lmat, &Istart, &Iend);CHKERRQ(ierr);
            for (Ii = Istart, jj = 0; Ii < Iend; Ii++, jj++) {
              ierr = MatGetRow(Lmat,Ii,&ncols,0,0);CHKERRQ(ierr);
              if (ncols <= 1) {
                ierr = VecSetValues(bb, 1, &Ii, &zero, INSERT_VALUES);CHKERRQ(ierr);
              }
              ierr = MatRestoreRow(Lmat,Ii,&ncols,0,0);CHKERRQ(ierr);
            }
            ierr = VecAssemblyBegin(bb);CHKERRQ(ierr);
            ierr = VecAssemblyEnd(bb);CHKERRQ(ierr);
          }

          ierr = KSPCreate(comm, &eksp);CHKERRQ(ierr);
          ierr = KSPSetTolerances(eksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 10);CHKERRQ(ierr);
          ierr = KSPSetNormType(eksp, KSP_NORM_NONE);CHKERRQ(ierr);
          ierr = KSPSetOptionsPrefix(eksp,((PetscObject)pc)->prefix);CHKERRQ(ierr);
          ierr = KSPAppendOptionsPrefix(eksp, "gamg_est_");CHKERRQ(ierr);
          ierr = KSPSetFromOptions(eksp);CHKERRQ(ierr);

          ierr = KSPSetInitialGuessNonzero(eksp, PETSC_FALSE);CHKERRQ(ierr);
          ierr = KSPSetOperators(eksp, Lmat, Lmat, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = KSPSetComputeSingularValues(eksp,PETSC_TRUE);CHKERRQ(ierr);

          /* set PC type to be same as smoother */
          ierr = KSPSetPC(eksp, subpc);CHKERRQ(ierr);

          /* solve - keep stuff out of logging */
          ierr = PetscLogEventDeactivate(KSP_Solve);CHKERRQ(ierr);
          ierr = PetscLogEventDeactivate(PC_Apply);CHKERRQ(ierr);
          ierr = KSPSolve(eksp, bb, xx);CHKERRQ(ierr);
          ierr = PetscLogEventActivate(KSP_Solve);CHKERRQ(ierr);
          ierr = PetscLogEventActivate(PC_Apply);CHKERRQ(ierr);

          ierr = KSPComputeExtremeSingularValues(eksp, &emax, &emin);CHKERRQ(ierr);

          ierr = VecDestroy(&xx);CHKERRQ(ierr);
          ierr = VecDestroy(&bb);CHKERRQ(ierr);
          ierr = KSPDestroy(&eksp);CHKERRQ(ierr);

          if (pc_gamg->verbose > 0) {
            PetscInt N1, tt;
            ierr = MatGetSize(Aarr[level], &N1, &tt);CHKERRQ(ierr);
            PetscPrintf(comm,"\t\t\t%s PC setup max eigen=%e min=%e on level %d (N=%d)\n",__FUNCT__,emax,emin,lidx,N1);
          }
        }
        {
          PetscInt N1, N0;
          ierr = MatGetSize(Aarr[level], &N1, NULL);CHKERRQ(ierr);
          ierr = MatGetSize(Aarr[level+1], &N0, NULL);CHKERRQ(ierr);
          /* heuristic - is this crap? */
          /* emin = 1.*emax/((PetscReal)N1/(PetscReal)N0); */
          emin  = emax * pc_gamg->eigtarget[0];
          emax *= pc_gamg->eigtarget[1];
        }
        ierr = KSPChebyshevSetEigenvalues(smoother, emax, emin);CHKERRQ(ierr);
      } /* setup checby flag */
    } /* non-coarse levels */

    /* clean up */
    for (level=1; level<pc_gamg->Nlevels; level++) {
      ierr = MatDestroy(&Parr[level]);CHKERRQ(ierr);
      ierr = MatDestroy(&Aarr[level]);CHKERRQ(ierr);
    }

    ierr = PCSetUp_MG(pc);CHKERRQ(ierr);

    if (PETSC_TRUE) {
      KSP smoother;  /* PCSetUp_MG seems to insists on setting this to GMRES on coarse grid */
      ierr = PCMGGetSmoother(pc, 0, &smoother);CHKERRQ(ierr);
      ierr = KSPSetType(smoother, KSPPREONLY);CHKERRQ(ierr);
    }
  } else {
    KSP smoother;
    if (pc_gamg->verbose) PetscPrintf(comm,"\t[%d]%s one level solver used (system is seen as DD). Using default solver.\n",rank,__FUNCT__);
    ierr = PCMGGetSmoother(pc, 0, &smoother);CHKERRQ(ierr);
    ierr = KSPSetOperators(smoother, Aarr[0], Aarr[0], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetType(smoother, KSPPREONLY);CHKERRQ(ierr);
    ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_GAMG"
PetscErrorCode PCDestroy_GAMG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg     = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg= (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PCReset_GAMG(pc);CHKERRQ(ierr);
  if (pc_gamg->ops->destroy) {
    ierr = (*pc_gamg->ops->destroy)(pc);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc_gamg->ops);CHKERRQ(ierr);
  ierr = PetscFree(pc_gamg->gamg_type_name);CHKERRQ(ierr);
  ierr = PetscFree(pc_gamg);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetProcEqLim"
/*@
   PCGAMGSetProcEqLim - Set number of equations to aim for on coarse grids via
   processor reduction.

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_process_eq_limit

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode  PCGAMGSetProcEqLim(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetProcEqLim_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetProcEqLim_GAMG"
static PetscErrorCode PCGAMGSetProcEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if (n>0) pc_gamg->min_eq_proc = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetCoarseEqLim"
/*@
   PCGAMGSetCoarseEqLim - Set max number of equations on coarse grids.

 Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_coarse_eq_limit

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
 @*/
PetscErrorCode PCGAMGSetCoarseEqLim(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetCoarseEqLim_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetCoarseEqLim_GAMG"
static PetscErrorCode PCGAMGSetCoarseEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if (n>0) pc_gamg->coarse_eq_limit = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetRepartitioning"
/*@
   PCGAMGSetRepartitioning - Repartition the coarse grids

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_repartition

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetRepartitioning(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetRepartitioning_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetRepartitioning_GAMG"
static PetscErrorCode PCGAMGSetRepartitioning_GAMG(PC pc, PetscBool n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->repart = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetReuseProl"
/*@
   PCGAMGSetReuseProl - Reuse prlongation

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_reuse_interpolation

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetReuseProl(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetReuseProl_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetReuseProl_GAMG"
static PetscErrorCode PCGAMGSetReuseProl_GAMG(PC pc, PetscBool n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->reuse_prol = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetUseASMAggs"
/*@
   PCGAMGSetUseASMAggs -

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_use_agg_gasm

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetUseASMAggs(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetUseASMAggs_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetUseASMAggs_GAMG"
static PetscErrorCode PCGAMGSetUseASMAggs_GAMG(PC pc, PetscBool n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->use_aggs_in_gasm = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetNlevels"
/*@
   PCGAMGSetNlevels -

   Not collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_mg_levels

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetNlevels(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetNlevels_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetNlevels_GAMG"
static PetscErrorCode PCGAMGSetNlevels_GAMG(PC pc, PetscInt n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->Nlevels = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetThreshold"
/*@
   PCGAMGSetThreshold - Relative threshold to use for dropping edges in aggregation graph

   Not collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_threshold

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetThreshold(PC pc, PetscReal n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetThreshold_C",(PC,PetscReal),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetThreshold_GAMG"
static PetscErrorCode PCGAMGSetThreshold_GAMG(PC pc, PetscReal n)
{
  PC_MG   *mg      = (PC_MG*)pc->data;
  PC_GAMG *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  pc_gamg->threshold = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetType"
/*@
   PCGAMGSetType - Set solution method - calls sub create method

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_type

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetType(PC pc, PCGAMGType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetType_C",(PC,PCGAMGType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetType_GAMG"
static PetscErrorCode PCGAMGSetType_GAMG(PC pc, PCGAMGType type)
{
  PetscErrorCode ierr,(*r)(PC);
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFunctionListFind(GAMGList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown GAMG type %s given",type);
  if (pc_gamg->ops->destroy) {
    ierr = (*pc_gamg->ops->destroy)(pc);CHKERRQ(ierr);
    ierr = PetscMemzero(pc_gamg->ops,sizeof(struct _PCGAMGOps));CHKERRQ(ierr);
  }
  ierr = PetscFree(pc_gamg->gamg_type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&pc_gamg->gamg_type_name);CHKERRQ(ierr);
  ierr = (*r)(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GAMG"
PetscErrorCode PCSetFromOptions_GAMG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscBool      flag;
  PetscInt       two   = 2;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PetscOptionsHead("GAMG options");CHKERRQ(ierr);
  {
    /* -pc_gamg_type */
    {
      char tname[256];
      ierr = PetscOptionsFList("-pc_gamg_type","Type of AMG method","PCGAMGSetType",GAMGList, pc_gamg->gamg_type_name, tname, sizeof(tname), &flag);CHKERRQ(ierr);
      /* call PCCreateGAMG_XYZ */
      if (flag) {
        ierr = PCGAMGSetType(pc,tname);CHKERRQ(ierr);
      }
    }
    /* -pc_gamg_verbose */
    ierr = PetscOptionsInt("-pc_gamg_verbose","Verbose (debugging) output for PCGAMG",
                           "none", pc_gamg->verbose,
                           &pc_gamg->verbose, NULL);CHKERRQ(ierr);
    /* -pc_gamg_repartition */
    ierr = PetscOptionsBool("-pc_gamg_repartition",
                            "Repartion coarse grids (false)",
                            "PCGAMGRepartitioning",
                            pc_gamg->repart,
                            &pc_gamg->repart,
                            &flag);CHKERRQ(ierr);
    /* -pc_gamg_reuse_interpolation */
    ierr = PetscOptionsBool("-pc_gamg_reuse_interpolation",
                            "Reuse prolongation operator (true)",
                            "PCGAMGReuseProl",
                            pc_gamg->reuse_prol,
                            &pc_gamg->reuse_prol,
                            &flag);CHKERRQ(ierr);
    /* -pc_gamg_use_agg_gasm */
    ierr = PetscOptionsBool("-pc_gamg_use_agg_gasm",
                            "Use aggregation agragates for GASM smoother (false)",
                            "PCGAMGUseASMAggs",
                            pc_gamg->use_aggs_in_gasm,
                            &pc_gamg->use_aggs_in_gasm,
                            &flag);CHKERRQ(ierr);
    /* -pc_gamg_process_eq_limit */
    ierr = PetscOptionsInt("-pc_gamg_process_eq_limit",
                           "Limit (goal) on number of equations per process on coarse grids",
                           "PCGAMGSetProcEqLim",
                           pc_gamg->min_eq_proc,
                           &pc_gamg->min_eq_proc,
                           &flag);CHKERRQ(ierr);
    /* -pc_gamg_coarse_eq_limit */
    ierr = PetscOptionsInt("-pc_gamg_coarse_eq_limit",
                           "Limit on number of equations for the coarse grid",
                           "PCGAMGSetCoarseEqLim",
                           pc_gamg->coarse_eq_limit,
                           &pc_gamg->coarse_eq_limit,
                           &flag);CHKERRQ(ierr);
    /* -pc_gamg_threshold */
    ierr = PetscOptionsReal("-pc_gamg_threshold",
                            "Relative threshold to use for dropping edges in aggregation graph",
                            "PCGAMGSetThreshold",
                            pc_gamg->threshold,
                            &pc_gamg->threshold,
                            &flag);CHKERRQ(ierr);
    if (flag && pc_gamg->verbose) {
      ierr = PetscPrintf(comm,"\t[%d]%s threshold set %e\n",0,__FUNCT__,pc_gamg->threshold);CHKERRQ(ierr);
    }
    /* -pc_gamg_eigtarget */
    ierr = PetscOptionsRealArray("-pc_gamg_eigtarget","Target eigenvalue range as fraction of estimated maximum eigenvalue","PCGAMGSetEigTarget",pc_gamg->eigtarget,&two,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_mg_levels",
                           "Set number of MG levels",
                           "PCGAMGSetNlevels",
                           pc_gamg->Nlevels,
                           &pc_gamg->Nlevels,
                           &flag);CHKERRQ(ierr);

    /* set options for subtype */
    if (pc_gamg->ops->setfromoptions) {ierr = (*pc_gamg->ops->setfromoptions)(pc);CHKERRQ(ierr);}
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCGAMG - Geometric algebraic multigrid (AMG) preconditioning framework.
       - This is the entry point to GAMG, registered in pcregis.c

   Options Database Keys:
   Multigrid options(inherited)
+  -pc_mg_cycles <1>: 1 for V cycle, 2 for W-cycle (PCMGSetCycleType)
.  -pc_mg_smoothup <1>: Number of post-smoothing steps (PCMGSetNumberSmoothUp)
.  -pc_mg_smoothdown <1>: Number of pre-smoothing steps (PCMGSetNumberSmoothDown)
-  -pc_mg_type <multiplicative>: (one of) additive multiplicative full kascade

  Level: intermediate

  Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType,
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), PCMGSetCycleType(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCyclesOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()
M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_GAMG"
PETSC_EXTERN PetscErrorCode PCCreate_GAMG(PC pc)
{
  PetscErrorCode ierr;
  PC_GAMG        *pc_gamg;
  PC_MG          *mg;
#if defined PETSC_GAMG_USE_LOG
  static long count = 0;
#endif

  PetscFunctionBegin;
  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  ierr = PCSetType(pc, PCMG);CHKERRQ(ierr); /* calls PCCreate_MG() and MGCreate_Private() */
  ierr = PetscObjectChangeTypeName((PetscObject)pc, PCGAMG);CHKERRQ(ierr);

  /* create a supporting struct and attach it to pc */
  ierr         = PetscNewLog(pc, PC_GAMG, &pc_gamg);CHKERRQ(ierr);
  mg           = (PC_MG*)pc->data;
  mg->galerkin = 2;             /* Use Galerkin, but it is computed externally */
  mg->innerctx = pc_gamg;

  ierr = PetscNewLog(pc,struct _PCGAMGOps,&pc_gamg->ops);CHKERRQ(ierr);

  pc_gamg->setup_count = 0;
  /* these should be in subctx but repartitioning needs simple arrays */
  pc_gamg->data_sz = 0;
  pc_gamg->data    = 0;

  /* register AMG type */
  ierr = PCGAMGInitializePackage();CHKERRQ(ierr);

  /* overwrite the pointers of PCMG by the functions of base class PCGAMG */
  pc->ops->setfromoptions = PCSetFromOptions_GAMG;
  pc->ops->setup          = PCSetUp_GAMG;
  pc->ops->reset          = PCReset_GAMG;
  pc->ops->destroy        = PCDestroy_GAMG;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetProcEqLim_C",PCGAMGSetProcEqLim_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetCoarseEqLim_C",PCGAMGSetCoarseEqLim_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetRepartitioning_C",PCGAMGSetRepartitioning_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetReuseProl_C",PCGAMGSetReuseProl_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetUseASMAggs_C",PCGAMGSetUseASMAggs_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetThreshold_C",PCGAMGSetThreshold_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetType_C",PCGAMGSetType_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetNlevels_C",PCGAMGSetNlevels_GAMG);CHKERRQ(ierr);
  pc_gamg->repart           = PETSC_FALSE;
  pc_gamg->reuse_prol       = PETSC_FALSE;
  pc_gamg->use_aggs_in_gasm = PETSC_FALSE;
  pc_gamg->min_eq_proc      = 50;
  pc_gamg->coarse_eq_limit  = 800;
  pc_gamg->threshold        = 0.;
  pc_gamg->Nlevels          = GAMG_MAXLEVELS;
  pc_gamg->verbose          = 0;
  pc_gamg->emax_id          = -1;
  pc_gamg->eigtarget[0]     = 0.05;
  pc_gamg->eigtarget[1]     = 1.05;

  /* private events */
#if defined PETSC_GAMG_USE_LOG
  if (count++ == 0) {
    ierr = PetscLogEventRegister("GAMG: createProl", PC_CLASSID, &petsc_gamg_setup_events[SET1]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  Graph", PC_CLASSID, &petsc_gamg_setup_events[GRAPH]);CHKERRQ(ierr);
    /* PetscLogEventRegister("    G.Mat", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_MAT]); */
    /* PetscLogEventRegister("    G.Filter", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_FILTER]); */
    /* PetscLogEventRegister("    G.Square", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_SQR]); */
    ierr = PetscLogEventRegister("  MIS/Agg", PC_CLASSID, &petsc_gamg_setup_events[SET4]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  geo: growSupp", PC_CLASSID, &petsc_gamg_setup_events[SET5]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  geo: triangle", PC_CLASSID, &petsc_gamg_setup_events[SET6]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("    search&set", PC_CLASSID, &petsc_gamg_setup_events[FIND_V]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  SA: col data", PC_CLASSID, &petsc_gamg_setup_events[SET7]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  SA: frmProl0", PC_CLASSID, &petsc_gamg_setup_events[SET8]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  SA: smooth", PC_CLASSID, &petsc_gamg_setup_events[SET9]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("GAMG: partLevel", PC_CLASSID, &petsc_gamg_setup_events[SET2]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  repartition", PC_CLASSID, &petsc_gamg_setup_events[SET12]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  Invert-Sort", PC_CLASSID, &petsc_gamg_setup_events[SET13]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  Move A", PC_CLASSID, &petsc_gamg_setup_events[SET14]);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("  Move P", PC_CLASSID, &petsc_gamg_setup_events[SET15]);CHKERRQ(ierr);

    /* PetscLogEventRegister(" PL move data", PC_CLASSID, &petsc_gamg_setup_events[SET13]); */
    /* PetscLogEventRegister("GAMG: fix", PC_CLASSID, &petsc_gamg_setup_events[SET10]); */
    /* PetscLogEventRegister("GAMG: set levels", PC_CLASSID, &petsc_gamg_setup_events[SET11]); */
    /* create timer stages */
#if defined GAMG_STAGES
    {
      char     str[32];
      PetscInt lidx;
      sprintf(str,"MG Level %d (finest)",0);
      ierr = PetscLogStageRegister(str, &gamg_stages[0]);CHKERRQ(ierr);
      for (lidx=1; lidx<9; lidx++) {
        sprintf(str,"MG Level %d",lidx);
        ierr = PetscLogStageRegister(str, &gamg_stages[lidx]);CHKERRQ(ierr);
      }
    }
#endif
  }
#endif
  /* PCSetUp_GAMG assumes that the type has been set, so set it to the default now */
  ierr = PCGAMGSetType(pc,PCGAMGAGG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGInitializePackage"
/*@C
 PCGAMGInitializePackage - This function initializes everything in the PCGAMG package. It is called
 from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PCCreate_GAMG()
 when using static libraries.

 Level: developer

 .keywords: PC, PCGAMG, initialize, package
 .seealso: PetscInitialize()
@*/
PetscErrorCode PCGAMGInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PCGAMGPackageInitialized) PetscFunctionReturn(0);
  PCGAMGPackageInitialized = PETSC_TRUE;
  ierr = PetscFunctionListAdd(&GAMGList,PCGAMGGEO,PCCreateGAMG_GEO);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&GAMGList,PCGAMGAGG,PCCreateGAMG_AGG);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&GAMGList,PCGAMGCLASSICAL,PCCreateGAMG_Classical);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PCGAMGFinalizePackage);CHKERRQ(ierr);

  /* general events */
#if defined PETSC_USE_LOG
  ierr = PetscLogEventRegister("PCGAMGgraph_AGG", 0, &PC_GAMGGgraph_AGG);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCGAMGgraph_GEO", PC_CLASSID, &PC_GAMGGgraph_GEO);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCGAMGcoarse_AGG", PC_CLASSID, &PC_GAMGCoarsen_AGG);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCGAMGcoarse_GEO", PC_CLASSID, &PC_GAMGCoarsen_GEO);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCGAMGProl_AGG", PC_CLASSID, &PC_GAMGProlongator_AGG);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCGAMGProl_GEO", PC_CLASSID, &PC_GAMGProlongator_GEO);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCGAMGPOpt_AGG", PC_CLASSID, &PC_GAMGOptprol_AGG);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGFinalizePackage"
/*@C
 PCGAMGFinalizePackage - This function destroys everything in the PCGAMG package. It is
 called from PetscFinalize().

 Level: developer

 .keywords: Petsc, destroy, package
 .seealso: PetscFinalize()
@*/
PetscErrorCode PCGAMGFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PCGAMGPackageInitialized = PETSC_FALSE;
  ierr = PetscFunctionListDestroy(&GAMGList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
