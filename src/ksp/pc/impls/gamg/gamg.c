/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include "petsc-private/matimpl.h"
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

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

static PetscFList GAMGList = 0;

/* ----------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCReset_GAMG"
PetscErrorCode PCReset_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if( pc_gamg->data != 0 ) { /* this should not happen, cleaned up in SetUp */
    ierr = PetscFree(pc_gamg->data); CHKERRQ(ierr);
  }
  pc_gamg->data = 0; pc_gamg->data_sz = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   createLevel: create coarse op with RAP.  repartition and/or reduce number 
     of active processors.

   Input Parameter:
   . pc - parameters
   . Amat_fine - matrix on this fine (k) level
   . cbs - coarse block size
   . isLast - 
   In/Output Parameter:
   . a_P_inout - prolongation operator to the next level (k-1)
   . a_nactive_proc - number of active procs
   Output Parameter:
   . a_Amat_crs - coarse matrix that is created (k-1)
*/

#undef __FUNCT__
#define __FUNCT__ "createLevel"
static PetscErrorCode createLevel( const PC pc,
                            const Mat Amat_fine,
                            const PetscInt cbs,
                            const PetscBool isLast,
                            Mat *a_P_inout,
                            PetscMPIInt *a_nactive_proc,
                            Mat *a_Amat_crs
                            )
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscBool  repart = pc_gamg->repart;
  const PetscInt   min_eq_proc = pc_gamg->min_eq_proc, coarse_max = pc_gamg->coarse_eq_limit;
  PetscErrorCode   ierr;
  Mat              Cmat,Pnew,Pold=*a_P_inout;
  IS               new_indices,isnum;
  MPI_Comm         wcomm = ((PetscObject)Amat_fine)->comm;
  PetscMPIInt      mype,npe,new_npe,nactive = *a_nactive_proc;
  PetscInt         neq,NN,Istart,Iend,Istart0,Iend0,ncrs_new,ncrs0,rfactor;
 
  PetscFunctionBegin;  
  ierr = MPI_Comm_rank( wcomm, &mype ); CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );  CHKERRQ(ierr);

  /* RAP */
#ifdef USE_R
  /* make R wih brute force for now */
  ierr = MatTranspose( Pold, Pnew );     
  ierr = MatDestroy( &Pold );  CHKERRQ(ierr);
  ierr = MatRARt( Amat_fine, Pnew, MAT_INITIAL_MATRIX, 2.0, &Cmat ); CHKERRQ(ierr);
  Pold = Pnew;
#else
  ierr = MatPtAP( Amat_fine, Pold, MAT_INITIAL_MATRIX, 2.0, &Cmat ); CHKERRQ(ierr);
#endif
  ierr = MatSetBlockSize( Cmat, cbs );      CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Cmat, &Istart0, &Iend0 ); CHKERRQ(ierr);
  ncrs0 = (Iend0-Istart0)/cbs; assert((Iend0-Istart0)%cbs == 0);

  /* get number of PEs to make active, reduce */
  ierr = MatGetSize( Cmat, &neq, &NN );  CHKERRQ(ierr);
  new_npe = (PetscMPIInt)((float)neq/(float)min_eq_proc + 0.5); /* hardwire min. number of eq/proc */
  if( new_npe == 0 || neq < coarse_max ) new_npe = 1; 
  else if (new_npe >= nactive ) new_npe = nactive; /* no change, rare */
  if( isLast ) new_npe = 1;
  
  if( !repart && new_npe==nactive ) { 
    *a_Amat_crs = Cmat; /* output - no repartitioning or reduction */
  }
  else {
    /* Repartition Cmat_{k} and move colums of P^{k}_{k-1} and coordinates accordingly */
    const PetscInt *idx,ndata_rows=pc_gamg->data_cell_rows,ndata_cols=pc_gamg->data_cell_cols,data_sz=ndata_rows*ndata_cols;
    const PetscInt  stride0=ncrs0*pc_gamg->data_cell_rows;
    PetscInt        *counts,is_sz,*newproc_idx,ii,jj,kk,strideNew,*tidx,inpe,targetPE;
    IS              isnewproc;
    VecScatter      vecscat;
    PetscScalar    *array;
    Vec             src_crd, dest_crd;
    IS              isscat;

    ierr = PetscMalloc( npe*sizeof(PetscInt), &counts ); CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET12],0,0,0,0);CHKERRQ(ierr);
#endif
    if( repart ) {
      /* create sub communicator  */
      Mat             adj;

      if (pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s repartition: npe (active): %d --> %d, neq = %d\n",mype,__FUNCT__,*a_nactive_proc,new_npe,neq);

      /* MatPartitioningApply call MatConvert, which is collective */
      if( cbs == 1 ) { 
	ierr = MatConvert( Cmat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj );   CHKERRQ(ierr);
      }
      else{
	/* make a scalar matrix to partition */
	Mat tMat;
	PetscInt ncols,jj,Ii; 
	const PetscScalar *vals; 
	const PetscInt *idx;
	PetscInt *d_nnz, *o_nnz;
	static PetscInt llev = 0;
	
	ierr = PetscMalloc( ncrs0*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
	ierr = PetscMalloc( ncrs0*sizeof(PetscInt), &o_nnz ); CHKERRQ(ierr);
	for ( Ii = Istart0, jj = 0 ; Ii < Iend0 ; Ii += cbs, jj++ ) {
	  ierr = MatGetRow(Cmat,Ii,&ncols,0,0); CHKERRQ(ierr);
	  d_nnz[jj] = ncols/cbs;
	  o_nnz[jj] = ncols/cbs;
	  ierr = MatRestoreRow(Cmat,Ii,&ncols,0,0); CHKERRQ(ierr);
	  if( d_nnz[jj] > ncrs0 ) d_nnz[jj] = ncrs0;
	  if( o_nnz[jj] > (neq/cbs-ncrs0) ) o_nnz[jj] = neq/cbs-ncrs0;
	}
	
	ierr = MatCreateAIJ( wcomm, ncrs0, ncrs0,
                             PETSC_DETERMINE, PETSC_DETERMINE,
                             0, d_nnz, 0, o_nnz,
                             &tMat );
	CHKERRQ(ierr);
	ierr = PetscFree( d_nnz ); CHKERRQ(ierr); 
	ierr = PetscFree( o_nnz ); CHKERRQ(ierr); 
	
	for ( ii = Istart0; ii < Iend0; ii++ ) {
	  PetscInt dest_row = ii/cbs;
	  ierr = MatGetRow(Cmat,ii,&ncols,&idx,&vals); CHKERRQ(ierr);
	  for( jj = 0 ; jj < ncols ; jj++ ){
	    PetscInt dest_col = idx[jj]/cbs;
	    PetscScalar v = 1.0;
	    ierr = MatSetValues(tMat,1,&dest_row,1,&dest_col,&v,ADD_VALUES); CHKERRQ(ierr);
	  }
	  ierr = MatRestoreRow(Cmat,ii,&ncols,&idx,&vals); CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(tMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(tMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	
	if( llev++ == -1 ) {
	  PetscViewer viewer; char fname[32];
	  ierr = PetscSNPrintf(fname,sizeof fname,"part_mat_%D.mat",llev);CHKERRQ(ierr);
	  PetscViewerBinaryOpen(wcomm,fname,FILE_MODE_WRITE,&viewer);
	  ierr = MatView( tMat, viewer ); CHKERRQ(ierr);
	  ierr = PetscViewerDestroy( &viewer );
	}
	
	ierr = MatConvert( tMat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj );   CHKERRQ(ierr);
	
	ierr = MatDestroy( &tMat );  CHKERRQ(ierr);
      }

      { /* partition: get newproc_idx, set is_sz */
	char prefix[256];
	const char *pcpre;
	const PetscInt *is_idx;
	MatPartitioning  mpart;
	IS proc_is;
	
	ierr = MatPartitioningCreate( wcomm, &mpart ); CHKERRQ(ierr);
	ierr = MatPartitioningSetAdjacency( mpart, adj ); CHKERRQ(ierr);
	ierr = PCGetOptionsPrefix( pc, &pcpre );CHKERRQ(ierr);
	ierr = PetscSNPrintf(prefix,sizeof prefix,"%spc_gamg_",pcpre?pcpre:"");CHKERRQ(ierr);
	ierr = PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix);CHKERRQ(ierr);
	ierr = MatPartitioningSetFromOptions( mpart );    CHKERRQ(ierr);
	ierr = MatPartitioningSetNParts( mpart, new_npe );CHKERRQ(ierr);
	ierr = MatPartitioningApply( mpart, &proc_is ); CHKERRQ(ierr);
	ierr = MatPartitioningDestroy( &mpart );          CHKERRQ(ierr);
      
	/* collect IS info */
	ierr = ISGetLocalSize( proc_is, &is_sz );       CHKERRQ(ierr);
	ierr = PetscMalloc( cbs*is_sz*sizeof(PetscInt), &newproc_idx ); CHKERRQ(ierr);
	ierr = ISGetIndices( proc_is, &is_idx );        CHKERRQ(ierr);
	NN = 1; /* bring to "front" of machine */
	/*NN = npe/new_npe;*/ /* spread partitioning across machine */
	for( kk = jj = 0 ; kk < is_sz ; kk++ ){
	  for( ii = 0 ; ii < cbs ; ii++, jj++ ){
	    newproc_idx[jj] = is_idx[kk] * NN; /* distribution */
	  }
	}
	ierr = ISRestoreIndices( proc_is, &is_idx );     CHKERRQ(ierr);
	ierr = ISDestroy( &proc_is );                    CHKERRQ(ierr);

	is_sz *= cbs;
      }
      ierr = MatDestroy( &adj );                       CHKERRQ(ierr);

      ierr = ISCreateGeneral( wcomm, is_sz, newproc_idx, PETSC_COPY_VALUES, &isnewproc );
      CHKERRQ(ierr);
      if( newproc_idx != 0 ) {
	ierr = PetscFree( newproc_idx );  CHKERRQ(ierr);
      }
    }
    else { /* simple aggreagtion of parts */
      /* find factor */
      if( new_npe == 1 ) rfactor = npe; /* easy */
      else {
	PetscReal best_fact = 0.;
	jj = -1;
	for( kk = 1 ; kk <= npe ; kk++ ){
	  if( npe%kk==0 ) { /* a candidate */
	    PetscReal nactpe = (PetscReal)npe/(PetscReal)kk, fact = nactpe/(PetscReal)new_npe;
	    if(fact > 1.0) fact = 1./fact; /* keep fact < 1 */
	    if( fact > best_fact ) {
	      best_fact = fact; jj = kk;
	    }
	  }
	}
	if(jj!=-1) rfactor = jj;
	else rfactor = 1; /* prime? */
      }
      new_npe = npe/rfactor;
      
      if( new_npe==nactive ) { 
	*a_Amat_crs = Cmat; /* output - no repartitioning or reduction */
	ierr = PetscFree( counts );  CHKERRQ(ierr);
	*a_nactive_proc = new_npe; /* output */
	if (pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s aggregate processors noop: new_npe=%d, neq=%d\n",mype,__FUNCT__,new_npe,neq);
	PetscFunctionReturn(0);
      }

      /* reduce - isnewproc */
      targetPE = mype/rfactor;

      if (pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s aggregate processors: npe: %d --> %d, neq=%d\n",mype,__FUNCT__,*a_nactive_proc,new_npe,neq);
      is_sz = ncrs0*cbs;
      ierr = ISCreateStride( wcomm, is_sz, targetPE, 0, &isnewproc );
      CHKERRQ(ierr);
    }

    /*
      Create an index set from the isnewproc index set to indicate the mapping TO
    */
    ierr = ISPartitioningToNumbering( isnewproc, &isnum ); CHKERRQ(ierr);
    /*
      Determine how many elements are assigned to each processor
    */
    inpe = npe;
    ierr = ISPartitioningCount( isnewproc, inpe, counts ); CHKERRQ(ierr);
    ierr = ISDestroy( &isnewproc );                       CHKERRQ(ierr);
    ncrs_new = counts[mype]/cbs;
    strideNew = ncrs_new*ndata_rows;
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET12],0,0,0,0);   CHKERRQ(ierr);
#endif
    /* Create a vector to contain the newly ordered element information */
    ierr = VecCreate( wcomm, &dest_crd );
    ierr = VecSetSizes( dest_crd, data_sz*ncrs_new, PETSC_DECIDE ); CHKERRQ(ierr);
    ierr = VecSetFromOptions( dest_crd ); CHKERRQ(ierr); /* this is needed! */
    /*
     There are 'ndata_rows*ndata_cols' data items per node, (one can think of the vectors of having 
     a block size of ...).  Note, ISs are expanded into equation space by 'cbs'.
     */
    ierr = PetscMalloc( (ncrs0*data_sz)*sizeof(PetscInt), &tidx ); CHKERRQ(ierr); 
    ierr = ISGetIndices( isnum, &idx ); CHKERRQ(ierr);
    for(ii=0,jj=0; ii<ncrs0 ; ii++) {
      PetscInt id = idx[ii*cbs]/cbs; /* get node back */
      for( kk=0; kk<data_sz ; kk++, jj++) tidx[jj] = id*data_sz + kk;
    }
    ierr = ISRestoreIndices( isnum, &idx ); CHKERRQ(ierr);
    ierr = ISCreateGeneral( wcomm, data_sz*ncrs0, tidx, PETSC_COPY_VALUES, &isscat );
    CHKERRQ(ierr);
    ierr = PetscFree( tidx );  CHKERRQ(ierr);
    /*
     Create a vector to contain the original vertex information for each element
     */
    ierr = VecCreateSeq( PETSC_COMM_SELF, data_sz*ncrs0, &src_crd ); CHKERRQ(ierr);
    for( jj=0; jj<ndata_cols ; jj++ ) {
      for( ii=0 ; ii<ncrs0 ; ii++) {
	for( kk=0; kk<ndata_rows ; kk++ ) {
	  PetscInt ix = ii*ndata_rows + kk + jj*stride0, jx = ii*data_sz + kk*ndata_cols + jj;
          PetscScalar tt = (PetscScalar)pc_gamg->data[ix];
	  ierr = VecSetValues( src_crd, 1, &jx, &tt, INSERT_VALUES );  CHKERRQ(ierr);
	}
      }
    }
    ierr = VecAssemblyBegin(src_crd); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(src_crd); CHKERRQ(ierr);
    /*
      Scatter the element vertex information (still in the original vertex ordering)
      to the correct processor
    */
    ierr = VecScatterCreate( src_crd, PETSC_NULL, dest_crd, isscat, &vecscat);
    CHKERRQ(ierr);
    ierr = ISDestroy( &isscat );       CHKERRQ(ierr);
    ierr = VecScatterBegin(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vecscat,src_crd,dest_crd,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy( &vecscat );       CHKERRQ(ierr);
    ierr = VecDestroy( &src_crd );       CHKERRQ(ierr);
    /*
      Put the element vertex data into a new allocation of the gdata->ele
    */
    ierr = PetscFree( pc_gamg->data );    CHKERRQ(ierr);
    ierr = PetscMalloc( data_sz*ncrs_new*sizeof(PetscReal), &pc_gamg->data );    CHKERRQ(ierr);
    pc_gamg->data_sz = data_sz*ncrs_new;
    ierr = VecGetArray( dest_crd, &array );    CHKERRQ(ierr);
    for( jj=0; jj<ndata_cols ; jj++ ) {
      for( ii=0 ; ii<ncrs_new ; ii++) {
	for( kk=0; kk<ndata_rows ; kk++ ) {
	  PetscInt ix = ii*ndata_rows + kk + jj*strideNew, jx = ii*data_sz + kk*ndata_cols + jj;
	  pc_gamg->data[ix] = PetscRealPart(array[jx]);
	  array[jx] = 1.e300;
	}
      }
    }
    ierr = VecRestoreArray( dest_crd, &array );    CHKERRQ(ierr);
    ierr = VecDestroy( &dest_crd );    CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET13],0,0,0,0);CHKERRQ(ierr);
#endif
    /*
      Invert for MatGetSubMatrix
    */
    ierr = ISInvertPermutation( isnum, ncrs_new*cbs, &new_indices ); CHKERRQ(ierr);
    ierr = ISSort( new_indices ); CHKERRQ(ierr); /* is this needed? */
    ierr = ISDestroy( &isnum ); CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET13],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET14],0,0,0,0);CHKERRQ(ierr);
#endif
    /* A_crs output */
    ierr = MatGetSubMatrix( Cmat, new_indices, new_indices, MAT_INITIAL_MATRIX, a_Amat_crs );
    CHKERRQ(ierr);

    ierr = MatDestroy( &Cmat ); CHKERRQ(ierr);
    Cmat = *a_Amat_crs; /* output */
    ierr = MatSetBlockSize( Cmat, cbs );      CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET14],0,0,0,0);CHKERRQ(ierr);
#endif
    /* prolongator */
    ierr = MatGetOwnershipRange( Pold, &Istart, &Iend );    CHKERRQ(ierr);
    {
      IS findices;
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET15],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = ISCreateStride(wcomm,Iend-Istart,Istart,1,&findices);   CHKERRQ(ierr);
#ifdef USE_R
      ierr = MatGetSubMatrix( Pold, new_indices, findices, MAT_INITIAL_MATRIX, &Pnew );
#else
      ierr = MatGetSubMatrix( Pold, findices, new_indices, MAT_INITIAL_MATRIX, &Pnew );
#endif
      CHKERRQ(ierr);
      ierr = ISDestroy( &findices ); CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
      ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET15],0,0,0,0);CHKERRQ(ierr);
#endif
    }
    ierr = MatDestroy( a_P_inout ); CHKERRQ(ierr);
    *a_P_inout = Pnew; /* output - repartitioned */
    ierr = ISDestroy( &new_indices ); CHKERRQ(ierr);
    ierr = PetscFree( counts );  CHKERRQ(ierr);
  }

  *a_nactive_proc = new_npe; /* output */

  if( !PETSC_TRUE ) {
    PetscViewer viewer; char fname[32]; static int llev=0; Cmat = *a_Amat_crs;
    if(llev==0) {
      sprintf(fname,"Cmat_%d.m",llev++);
      PetscViewerASCIIOpen(wcomm,fname,&viewer);
      ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
      ierr = MatView(Amat_fine, viewer ); CHKERRQ(ierr);
      ierr = PetscViewerDestroy( &viewer );
    }
    sprintf(fname,"Cmat_%d.m",llev++);
    PetscViewerASCIIOpen(wcomm,fname,&viewer);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Cmat, viewer ); CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
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
PetscErrorCode PCSetUp_GAMG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  Mat              Pmat = pc->pmat;
  PetscInt         fine_level,level,level1,M,N,bs,nloc,lidx,Istart,Iend,nASMBlocksArr[GAMG_MAXLEVELS];
  MPI_Comm         wcomm = ((PetscObject)pc)->comm;
  PetscMPIInt      mype,npe,nactivepe;
  Mat              Aarr[GAMG_MAXLEVELS], Parr[GAMG_MAXLEVELS];
  PetscReal        emaxs[GAMG_MAXLEVELS];
  IS              *ASMLocalIDsArr[GAMG_MAXLEVELS],removedEqs[GAMG_MAXLEVELS];
  PetscInt         level_bs[GAMG_MAXLEVELS];
  PetscLogDouble   nnz0=0,nnztot=0;
  MatInfo          info;
  PetscBool        stokes = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  if( pc_gamg->setup_count++ > 0 ) {
    PC_MG_Levels   **mglevels = mg->levels;
    /* just do Galerkin grids */
    Mat B,dA,dB;
    assert(pc->setupcalled);

    if( pc_gamg->Nlevels > 1 ) {
      /* currently only handle case where mat and pmat are the same on coarser levels */
      ierr = KSPGetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,&dA,&dB,PETSC_NULL);CHKERRQ(ierr);
      /* (re)set to get dirty flag */
      ierr = KSPSetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,dA,dB,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
       
      for (level=pc_gamg->Nlevels-2; level>-1; level--) {
        /* the first time through the matrix structure has changed from repartitioning */
        if( pc_gamg->setup_count==2 /*&& (pc_gamg->repart || level==0)*/) {
          ierr = MatPtAP(dB,mglevels[level+1]->interpolate,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
          ierr = MatDestroy(&mglevels[level]->A);CHKERRQ(ierr);
          mglevels[level]->A = B;
        }
        else {
          ierr = KSPGetOperators(mglevels[level]->smoothd,PETSC_NULL,&B,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatPtAP(dB,mglevels[level+1]->interpolate,MAT_REUSE_MATRIX,1.0,&B);CHKERRQ(ierr);
        }
        ierr = KSPSetOperators(mglevels[level]->smoothd,B,B,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        dB = B;
        }
    }

    ierr = PCSetUp_MG( pc );CHKERRQ( ierr );

    /* PCSetUp_MG seems to insists on setting this to GMRES */
    ierr = KSPSetType( mglevels[0]->smoothd, KSPPREONLY ); CHKERRQ(ierr);

    PetscFunctionReturn(0);
  }
  assert(pc->setupcalled == 0);

  /* deal with Stokes */ 
  ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_fieldsplit_detect_saddle_point",&stokes,PETSC_NULL);CHKERRQ(ierr);

  if( pc_gamg->data == 0 ) {
    if( !pc_gamg->createdefaultdata ){
      SETERRQ(wcomm,PETSC_ERR_LIB,"'createdefaultdata' not set?!?! need to support NULL data!!!");
    }
    if( stokes ) {
      SETERRQ(wcomm,PETSC_ERR_LIB,"Need data (eg, PCSetCoordinates) for Stokes problems");
    }
    ierr = pc_gamg->createdefaultdata( pc, Pmat ); CHKERRQ(ierr);
  }

  /* get basic dims */
  if( stokes ) {
    bs = pc_gamg->data_cell_rows;
    nloc = pc_gamg->data_sz/pc_gamg->data_cell_cols/bs;
    /* M,N,Iend,Istart,Pmat -- TODO */
    ierr = MatGetBlockSize( Pmat, &M ); CHKERRQ(ierr); assert(M==bs);
    ierr = MatGetOwnershipRange( Pmat, &Istart, &Iend ); CHKERRQ(ierr); assert((Iend-Istart)/bs == nloc);
    ierr = MatGetSize( Pmat, &M, &N );CHKERRQ(ierr);
  }
  else {
    ierr = MatGetBlockSize( Pmat, &bs ); CHKERRQ(ierr);
    ierr = MatGetSize( Pmat, &M, &N );CHKERRQ(ierr);
    ierr = MatGetOwnershipRange( Pmat, &Istart, &Iend ); CHKERRQ(ierr);
    nloc = (Iend-Istart)/bs; assert((Iend-Istart)%bs == 0);
  }
  /* Get A_i and R_i */
  if (pc_gamg->verbose) {
    if(pc_gamg->verbose==1) ierr =  MatGetInfo(Pmat,MAT_LOCAL,&info); 
    else ierr =  MatGetInfo(Pmat,MAT_GLOBAL_SUM,&info); 
    CHKERRQ(ierr);
    nnz0 = info.nz_used;
    nnztot = info.nz_used;
    PetscPrintf(wcomm,"\t[%d]%s level %d N=%d, n data rows=%d, n data cols=%d, nnz/row (ave)=%d, np=%d\n",
                mype,__FUNCT__,0,N,pc_gamg->data_cell_rows,pc_gamg->data_cell_cols,
                (int)(nnz0/(PetscReal)N),npe);
  }

  for ( level=0, Aarr[0] = Pmat, nactivepe = npe; /* hard wired stopping logic */
        level < (pc_gamg->Nlevels-1) && (level==0 || M>pc_gamg->coarse_eq_limit); /* && (npe==1 || nactivepe>1); */
        level++ ){
    level1 = level + 1;
#if (defined PETSC_GAMG_USE_LOG && defined GAMG_STAGES)
    ierr = PetscLogStagePush(gamg_stages[level]); CHKERRQ( ierr );
#endif
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET1],0,0,0,0); CHKERRQ(ierr);
#endif
    { /* construct prolongator */
      Mat Gmat;
      PetscCoarsenData *agg_lists;

      level_bs[level] = bs;
      ierr = pc_gamg->graph( pc, Aarr[level], &Gmat ); CHKERRQ(ierr);
      ierr = pc_gamg->coarsen( pc, &Gmat, &agg_lists ); CHKERRQ(ierr);
      ierr = pc_gamg->prolongator( pc, Aarr[level], Gmat, agg_lists, &Parr[level1] );
      CHKERRQ(ierr);

      if( Parr[level1] ){
        /* get new block size of coarse matrices */    
        if( pc_gamg->col_bs_id != -1 ){
          PetscBool flag;
          ierr = PetscObjectComposedDataGetInt( (PetscObject)Parr[level1], pc_gamg->col_bs_id, bs, flag );
          CHKERRQ( ierr ); assert(flag);
        }
      }

      if( pc_gamg->optprol && Parr[level1] ){
        /* smooth */
        ierr = pc_gamg->optprol( pc, Aarr[level], &Parr[level1] ); CHKERRQ(ierr);
      }

      ierr = MatDestroy( &Gmat );      CHKERRQ(ierr); 
      
      if ( pc_gamg->use_aggs_in_gasm ) {
        ierr = PetscCDGetASMBlocks(agg_lists, level_bs[level], &nASMBlocksArr[level], &ASMLocalIDsArr[level]);  CHKERRQ(ierr);
      }

      ierr = PetscCDGetRemovedIS( agg_lists, &removedEqs[level] );  CHKERRQ(ierr);

      ierr = PetscCDDestroy( agg_lists );  CHKERRQ(ierr);
    }
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#endif
    /* cache eigen estimate */
    if( pc_gamg->emax_id != -1 ){
      PetscBool flag;
      ierr = PetscObjectComposedDataGetReal( (PetscObject)Aarr[level], pc_gamg->emax_id, emaxs[level], flag );
      CHKERRQ( ierr );
      if( !flag ) emaxs[level] = -1.;
    }
    else emaxs[level] = -1.;
    if(level==0) Aarr[0] = Pmat; /* use Pmat for finest level setup */
    if( !Parr[level1] ) {
      if (pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s stop gridding, level %d\n",mype,__FUNCT__,level);
      break;
    }
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = createLevel( pc, Aarr[level], bs,
                        (PetscBool)(level==pc_gamg->Nlevels-2),
                        &Parr[level1], &nactivepe, &Aarr[level1] );
    CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = MatGetSize( Aarr[level1], &M, &N );CHKERRQ(ierr);
    if (pc_gamg->verbose){
      PetscInt NN = M;
      if(pc_gamg->verbose==1) {
        ierr = MatGetInfo(Aarr[level1],MAT_LOCAL,&info); 
        ierr = MatGetLocalSize( Aarr[level1], &NN, &N );CHKERRQ(ierr);
      }
      else ierr = MatGetInfo(Aarr[level1],MAT_GLOBAL_SUM,&info); 
      CHKERRQ(ierr);
      nnztot += info.nz_used;
      PetscPrintf(wcomm,"\t\t[%d]%s %d) N=%d, n data cols=%d, nnz/row (ave)=%d, %d active pes\n",
                  mype,__FUNCT__,(int)level1,N,pc_gamg->data_cell_cols,
                  (int)(info.nz_used/(PetscReal)NN), nactivepe );
      CHKERRQ(ierr);
    }
    /* stop if one node */
    if( M/pc_gamg->data_cell_cols < 2 ) {
      level++;
      break;
    }
    /* Vec diag; PetscScalar *data_arr,v; PetscInt Istart,Iend,kk,nloceq,id; */
    /* v = 1.e-10; /\* LU factor has hard wired numbers for small diags so this needs to match (yuk) *\/ */
    /* ierr = MatGetOwnershipRange(Aarr[level1], &Istart, &Iend); CHKERRQ(ierr); */
    /* nloceq = Iend-Istart; */
    /* ierr = MatGetVecs( Aarr[level1], &diag, 0 );    CHKERRQ(ierr); */
    /* ierr = MatGetDiagonal( Aarr[level1], diag );    CHKERRQ(ierr); */
    /* ierr = VecGetArray( diag, &data_arr );   CHKERRQ(ierr); */
    /* for(kk=0;kk<nloceq;kk++){ */
    /*   if(data_arr[kk]==0.0) { */
    /*     id = kk + Istart; */
    /*     ierr = MatSetValues(Aarr[level1],1,&id,1,&id,&v,INSERT_VALUES); */
    /*     CHKERRQ(ierr); */
    /*     PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s warning: added zero to diag (%d) on level %d \n",mype,__FUNCT__,id,level1); */
    /*   } */
    /* } */
    /* ierr = VecRestoreArray( diag, &data_arr ); CHKERRQ(ierr); */
    /* ierr = VecDestroy( &diag );                CHKERRQ(ierr); */
    /* ierr = MatAssemblyBegin(Aarr[level1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
    /* ierr = MatAssemblyEnd(Aarr[level1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); */
    
#if (defined PETSC_GAMG_USE_LOG && defined GAMG_STAGES)
    ierr = PetscLogStagePop(); CHKERRQ( ierr );
#endif
  } /* levels */
  
  if( pc_gamg->data ) {
    ierr = PetscFree( pc_gamg->data ); CHKERRQ( ierr );
    pc_gamg->data = 0;
  }
  
  if (pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s %d levels, grid complexity = %g\n",0,__FUNCT__,level+1,nnztot/nnz0);
  pc_gamg->Nlevels = level + 1;
  fine_level = level;
  ierr = PCMGSetLevels(pc,pc_gamg->Nlevels,PETSC_NULL);CHKERRQ(ierr);

  /* simple setup */
  if( !PETSC_TRUE ){
    PC_MG_Levels **mglevels = mg->levels;
    for (lidx=0,level=pc_gamg->Nlevels-1;
         lidx<fine_level;
         lidx++, level--){
      ierr = PCMGSetInterpolation( pc, lidx+1, Parr[level] );CHKERRQ(ierr);
      ierr = KSPSetOperators( mglevels[lidx]->smoothd, Aarr[level], Aarr[level], SAME_NONZERO_PATTERN );CHKERRQ(ierr);
      ierr = MatDestroy( &Parr[level] );  CHKERRQ(ierr);
      ierr = MatDestroy( &Aarr[level] );  CHKERRQ(ierr);
    }
    ierr = KSPSetOperators( mglevels[fine_level]->smoothd, Aarr[0], Aarr[0], SAME_NONZERO_PATTERN );   CHKERRQ(ierr);
    
    ierr = PCSetUp_MG( pc );  CHKERRQ( ierr );
  }
  else if( pc_gamg->Nlevels > 1 ) { /* don't setup MG if one level */
    /* set default smoothers & set operators */
    for ( lidx = 1, level = pc_gamg->Nlevels-2;
          lidx <= fine_level;
          lidx++, level--) {
      KSP smoother; 
      PC subpc; 
      /* set defaults */
      ierr = PCMGGetSmoother( pc, lidx, &smoother ); CHKERRQ(ierr);
      ierr = KSPSetType( smoother, KSPCHEBYCHEV );CHKERRQ(ierr);
      ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);

      /* override defaults and command line args (!) */
      if ( pc_gamg->use_aggs_in_gasm ) {
        PetscInt sz;
        IS *is;
        if( PETSC_FALSE ){
          PetscInt ii;
          PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s Nblocks=%d\n",mype,__FUNCT__,nASMBlocksArr[level]); 
          for(ii=0;ii<nASMBlocksArr[level];ii++){
            ISView(ASMLocalIDsArr[level][ii],PETSC_VIEWER_STDOUT_SELF);
          }
        }
        sz = nASMBlocksArr[level];
        is = ASMLocalIDsArr[level];
        ierr = PCSetType( subpc, PCGASM ); CHKERRQ(ierr);
        if(sz==0){
          IS is;
          PetscInt my0,kk;
          ierr = MatGetOwnershipRange( Aarr[level], &my0, &kk ); CHKERRQ(ierr);
          ierr = ISCreateGeneral(PETSC_COMM_SELF, 1, &my0, PETSC_COPY_VALUES, &is ); CHKERRQ(ierr);
          ierr = PCGASMSetLocalSubdomains( subpc, 1, &is, PETSC_NULL ); CHKERRQ(ierr);
          ierr = ISDestroy( &is ); CHKERRQ(ierr);
        }
        else {
          PetscInt kk;
          ierr = PCGASMSetLocalSubdomains( subpc, sz, is, PETSC_NULL ); CHKERRQ(ierr);
          for(kk=0;kk<sz;kk++){
            ierr = ISDestroy( &is[kk] ); CHKERRQ(ierr);
          }
          ierr = PetscFree( is ); CHKERRQ(ierr);
        }
        ierr = PCGASMSetOverlap( subpc, 0 ); CHKERRQ(ierr);

        ASMLocalIDsArr[level] = PETSC_NULL;
        nASMBlocksArr[level] = 0;
        ierr = PCGASMSetType( subpc, PC_GASM_BASIC ); CHKERRQ(ierr);
      }
      else {
        ierr = PCSetType( subpc, PCJACOBI ); CHKERRQ(ierr);
      }
      ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
      /* set ops */
      ierr = KSPSetOperators( smoother, Aarr[level], Aarr[level], SAME_NONZERO_PATTERN );   CHKERRQ(ierr);
      ierr = PCMGSetInterpolation( pc, lidx, Parr[level+1] );CHKERRQ(ierr);
    }
    {
      /* coarse grid */
      KSP smoother,*k2; PC subpc,pc2; PetscInt ii,first;
      Mat Lmat = Aarr[(level=pc_gamg->Nlevels-1)]; lidx = 0;
      ierr = PCMGGetSmoother( pc, lidx, &smoother ); CHKERRQ(ierr);
      ierr = KSPSetOperators( smoother, Lmat, Lmat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
      ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
      ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);
      ierr = PCSetType( subpc, PCBJACOBI ); CHKERRQ(ierr);
      ierr = PCSetUp( subpc ); CHKERRQ(ierr);
      ierr = PCBJacobiGetSubKSP(subpc,&ii,&first,&k2);CHKERRQ(ierr);      assert(ii==1);
      ierr = KSPGetPC(k2[0],&pc2);CHKERRQ(ierr);
      ierr = PCSetType( pc2, PCLU ); CHKERRQ(ierr);
    }

    /* should be called in PCSetFromOptions_GAMG(), but cannot be called prior to PCMGSetLevels() */
    ierr = PetscObjectOptionsBegin( (PetscObject)pc );CHKERRQ(ierr);
    ierr = PCSetFromOptions_MG( pc ); CHKERRQ(ierr);
    ierr = PetscOptionsEnd();  CHKERRQ(ierr);
    if (mg->galerkin != 2) SETERRQ(wcomm,PETSC_ERR_USER,"GAMG does Galerkin manually so the -pc_mg_galerkin option must not be used.");

    /* create cheby smoothers */
    for ( lidx = 1, level = pc_gamg->Nlevels-2;
          lidx <= fine_level;
          lidx++, level--) {
      KSP smoother; 
      PetscBool flag;
      PC subpc; 

      ierr = PCMGGetSmoother( pc, lidx, &smoother ); CHKERRQ(ierr);

      /* do my own cheby */
      ierr = PetscTypeCompare( (PetscObject)smoother, KSPCHEBYCHEV, &flag ); CHKERRQ(ierr);
      if( flag ) {
        PetscReal emax, emin;
        ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);
        ierr = PetscTypeCompare( (PetscObject)subpc, PCJACOBI, &flag ); CHKERRQ(ierr);
        if( flag && emaxs[level] > 0.0 ) emax=emaxs[level]; /* eigen estimate only for diagnal PC */
        else{ /* eigen estimate 'emax' */
          KSP eksp; Mat Lmat = Aarr[level];
          Vec bb, xx; 

          ierr = MatGetVecs( Lmat, &bb, 0 );         CHKERRQ(ierr);
          ierr = MatGetVecs( Lmat, &xx, 0 );         CHKERRQ(ierr);
          {
            PetscRandom    rctx;
            ierr = PetscRandomCreate(wcomm,&rctx);CHKERRQ(ierr);
            ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
            ierr = VecSetRandom(bb,rctx);CHKERRQ(ierr);
            ierr = PetscRandomDestroy( &rctx ); CHKERRQ(ierr);
          }
          if( removedEqs[level] ) {
            /* being very careful - zeroing out BC rows (this is not done in agg.c estimates) */
            PetscScalar *zeros; 
            PetscInt ii,jj, *idx_bs, sz, bs=level_bs[level];
            const PetscInt *idx;
            ierr = ISGetLocalSize( removedEqs[level], &sz ); CHKERRQ(ierr);
            ierr = PetscMalloc( bs*sz*sizeof(PetscScalar), &zeros ); CHKERRQ(ierr);
            for(ii=0;ii<bs*sz;ii++) zeros[ii] = 0.;
            ierr = PetscMalloc( bs*sz*sizeof(PetscInt), &idx_bs ); CHKERRQ(ierr);
            ierr = ISGetIndices( removedEqs[level], &idx); CHKERRQ(ierr);
            for(ii=0;ii<sz;ii++) {
              for(jj=0;jj<bs;jj++) {
                idx_bs[ii] = bs*idx[ii]+jj;
              }
            }
            ierr = ISRestoreIndices( removedEqs[level], &idx ); CHKERRQ(ierr);
            if( sz > 0 ) {
              ierr = VecSetValues( bb, sz, idx_bs, zeros, INSERT_VALUES );  CHKERRQ(ierr);
            }
            ierr = PetscFree( idx_bs );  CHKERRQ(ierr);
            ierr = PetscFree( zeros );  CHKERRQ(ierr);            
            ierr = VecAssemblyBegin(bb); CHKERRQ(ierr);
            ierr = VecAssemblyEnd(bb); CHKERRQ(ierr);
            ierr = ISDestroy( &removedEqs[level] );                    CHKERRQ(ierr);
          }
          ierr = KSPCreate( wcomm, &eksp );CHKERRQ(ierr);
          ierr = KSPSetTolerances( eksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 10 );
          CHKERRQ(ierr);
          ierr = KSPSetNormType( eksp, KSP_NORM_NONE );                 CHKERRQ(ierr);
          ierr = KSPSetOptionsPrefix(eksp,((PetscObject)pc)->prefix);CHKERRQ(ierr);
          ierr = KSPAppendOptionsPrefix( eksp, "gamg_est_");         CHKERRQ(ierr);
          ierr = KSPSetFromOptions( eksp );    CHKERRQ(ierr);

          ierr = KSPSetInitialGuessNonzero( eksp, PETSC_FALSE ); CHKERRQ(ierr);
          ierr = KSPSetOperators( eksp, Lmat, Lmat, SAME_NONZERO_PATTERN ); CHKERRQ( ierr );
          ierr = KSPSetComputeSingularValues( eksp,PETSC_TRUE ); CHKERRQ(ierr);

          /* set PC type to be same as smoother */
          ierr = KSPSetPC( eksp, subpc ); CHKERRQ( ierr );

          /* solve - keep stuff out of logging */
          ierr = PetscLogEventDeactivate(KSP_Solve);CHKERRQ(ierr);
          ierr = PetscLogEventDeactivate(PC_Apply);CHKERRQ(ierr);
          ierr = KSPSolve( eksp, bb, xx ); CHKERRQ(ierr);
          ierr = PetscLogEventActivate(KSP_Solve);CHKERRQ(ierr);
          ierr = PetscLogEventActivate(PC_Apply);CHKERRQ(ierr);
          
          ierr = KSPComputeExtremeSingularValues( eksp, &emax, &emin ); CHKERRQ(ierr);
          
          ierr = VecDestroy( &xx );       CHKERRQ(ierr);
          ierr = VecDestroy( &bb );       CHKERRQ(ierr);
          ierr = KSPDestroy( &eksp );       CHKERRQ(ierr);
          
          if( pc_gamg->verbose > 0 ) {
            PetscInt N1, tt;
            ierr = MatGetSize( Aarr[level], &N1, &tt );         CHKERRQ(ierr);
            PetscPrintf(wcomm,"\t\t\t%s PC setup max eigen=%e min=%e on level %d (N=%d)\n",__FUNCT__,emax,emin,lidx,N1);
          }
        }
        { 
          PetscInt N1, N0, tt;
          ierr = MatGetSize( Aarr[level], &N1, &tt );         CHKERRQ(ierr);
          ierr = MatGetSize( Aarr[level+1], &N0, &tt );       CHKERRQ(ierr);
          /* heuristic - is this crap? */
          emin = 1.*emax/((PetscReal)N1/(PetscReal)N0); 
          emax *= 1.05;
        }
        ierr = KSPChebychevSetEigenvalues( smoother, emax, emin );CHKERRQ(ierr);
      } /* setup checby flag */
    } /* non-coarse levels */
    
    /* clean up */
    for(level=1;level<pc_gamg->Nlevels;level++){
      ierr = MatDestroy( &Parr[level] );  CHKERRQ(ierr);
      ierr = MatDestroy( &Aarr[level] );  CHKERRQ(ierr);
    }

    ierr = PCSetUp_MG( pc );CHKERRQ( ierr );
    
    if( PETSC_FALSE ){
      KSP smoother;  /* PCSetUp_MG seems to insists on setting this to GMRES on coarse grid */
      ierr = PCMGGetSmoother( pc, 0, &smoother ); CHKERRQ(ierr);
      ierr = KSPSetType( smoother, KSPPREONLY ); CHKERRQ(ierr);
    }
  }
  else {
    KSP smoother;
    if (pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s one level solver used (system is seen as DD). Using default solver.\n",mype,__FUNCT__);
    ierr = PCMGGetSmoother( pc, 0, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetOperators( smoother, Aarr[0], Aarr[0], SAME_NONZERO_PATTERN );   CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPPREONLY ); CHKERRQ(ierr);
    ierr = PCSetUp_MG( pc );CHKERRQ( ierr );
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
PetscErrorCode PCDestroy_GAMG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg= (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PCReset_GAMG( pc );CHKERRQ(ierr);
  ierr = PetscFree( pc_gamg );CHKERRQ(ierr);
  ierr = PCDestroy_MG( pc );CHKERRQ(ierr);
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetProcEqLim_GAMG"
PetscErrorCode PCGAMGSetProcEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  if(n>0) pc_gamg->min_eq_proc = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetCoarseEqLim_GAMG"
PetscErrorCode PCGAMGSetCoarseEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  if(n>0) pc_gamg->coarse_eq_limit = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetRepartitioning_GAMG"
PetscErrorCode PCGAMGSetRepartitioning_GAMG(PC pc, PetscBool n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  pc_gamg->repart = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetUseASMAggs_GAMG"
PetscErrorCode PCGAMGSetUseASMAggs_GAMG(PC pc, PetscBool n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  pc_gamg->use_aggs_in_gasm = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetNlevels_GAMG"
PetscErrorCode PCGAMGSetNlevels_GAMG(PC pc, PetscInt n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  pc_gamg->Nlevels = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetThreshold_GAMG"
PetscErrorCode PCGAMGSetThreshold_GAMG(PC pc, PetscReal n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  pc_gamg->threshold = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
PetscErrorCode PCGAMGSetType( PC pc, const PCGAMGType type )
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetType_C",(PC,const PCGAMGType),(pc,type));
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetType_GAMG"
PetscErrorCode PCGAMGSetType_GAMG( PC pc, const PCGAMGType type )
{
  PetscErrorCode ierr,(*r)(PC);
  
  PetscFunctionBegin;
  ierr = PetscFListFind(GAMGList,((PetscObject)pc)->comm,type,PETSC_FALSE,(PetscVoidStarFunction)&r); 
  CHKERRQ(ierr);

  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown GAMG type %s given",type);

  /* call sub create method */
  ierr = (*r)(pc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GAMG"
PetscErrorCode PCSetFromOptions_GAMG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscBool        flag;
  MPI_Comm         wcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG options"); CHKERRQ(ierr);
  {
    /* -pc_gamg_verbose */
    ierr = PetscOptionsInt("-pc_gamg_verbose","Verbose (debugging) output for PCGAMG",
                           "none", pc_gamg->verbose,
                           &pc_gamg->verbose, PETSC_NULL );
    CHKERRQ(ierr);
    
    /* -pc_gamg_repartition */
    ierr = PetscOptionsBool("-pc_gamg_repartition",
                            "Repartion coarse grids (false)",
                            "PCGAMGRepartitioning",
                            pc_gamg->repart,
                            &pc_gamg->repart, 
                            &flag); 
    CHKERRQ(ierr);
   
    /* -pc_gamg_use_agg_gasm */
    ierr = PetscOptionsBool("-pc_gamg_use_agg_gasm",
                            "Use aggregation agragates for GASM smoother (false)",
                            "PCGAMGUseASMAggs",
                            pc_gamg->use_aggs_in_gasm,
                            &pc_gamg->use_aggs_in_gasm, 
                            &flag); 
    CHKERRQ(ierr);
    
    /* -pc_gamg_process_eq_limit */
    ierr = PetscOptionsInt("-pc_gamg_process_eq_limit",
                           "Limit (goal) on number of equations per process on coarse grids",
                           "PCGAMGSetProcEqLim",
                           pc_gamg->min_eq_proc,
                           &pc_gamg->min_eq_proc, 
                           &flag ); 
    CHKERRQ(ierr);
  
    /* -pc_gamg_coarse_eq_limit */
    ierr = PetscOptionsInt("-pc_gamg_coarse_eq_limit",
                           "Limit on number of equations for the coarse grid",
                           "PCGAMGSetCoarseEqLim",
                           pc_gamg->coarse_eq_limit,
                           &pc_gamg->coarse_eq_limit, 
                           &flag );
    CHKERRQ(ierr);

    /* -pc_gamg_threshold */
    ierr = PetscOptionsReal("-pc_gamg_threshold",
                            "Relative threshold to use for dropping edges in aggregation graph",
                            "PCGAMGSetThreshold",
                            pc_gamg->threshold,
                            &pc_gamg->threshold, 
                            &flag ); 
    CHKERRQ(ierr);
    if(flag && pc_gamg->verbose) PetscPrintf(wcomm,"\t[%d]%s threshold set %e\n",0,__FUNCT__,pc_gamg->threshold);

    ierr = PetscOptionsInt("-pc_mg_levels",
                           "Set number of MG levels",
                           "PCGAMGSetNlevels",
                           pc_gamg->Nlevels,
                           &pc_gamg->Nlevels, 
                           &flag ); 
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
-  -pc_mg_type <multiplicative>: (one of) additive multiplicative full cascade kascade

  Level: intermediate

  Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType,
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), PCMGSetCycleType(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCyclesOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_GAMG"
PetscErrorCode  PCCreate_GAMG( PC pc )
{
  PetscErrorCode  ierr;
  PC_GAMG         *pc_gamg;
  PC_MG           *mg;
#if defined PETSC_GAMG_USE_LOG
  static long count = 0;
#endif

  PetscFunctionBegin;

  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  ierr = PCSetType( pc, PCMG );  CHKERRQ(ierr); /* calls PCCreate_MG() and MGCreate_Private() */
  ierr = PetscObjectChangeTypeName( (PetscObject)pc, PCGAMG ); CHKERRQ(ierr);

  /* create a supporting struct and attach it to pc */
  ierr = PetscNewLog( pc, PC_GAMG, &pc_gamg ); CHKERRQ(ierr);
  mg = (PC_MG*)pc->data;
  mg->galerkin = 2;             /* Use Galerkin, but it is computed externally */
  mg->innerctx = pc_gamg;

  pc_gamg->setup_count = 0;
  /* these should be in subctx but repartitioning needs simple arrays */
  pc_gamg->data_sz = 0; 
  pc_gamg->data = 0; 

  /* register AMG type */
  if( !GAMGList ){
    ierr = PetscFListAdd(&GAMGList,GAMGGEO,"PCCreateGAMG_GEO",(void(*)(void))PCCreateGAMG_GEO);CHKERRQ(ierr);
    ierr = PetscFListAdd(&GAMGList,GAMGAGG,"PCCreateGAMG_AGG",(void(*)(void))PCCreateGAMG_AGG);CHKERRQ(ierr);
  }

  /* overwrite the pointers of PCMG by the functions of base class PCGAMG */
  pc->ops->setfromoptions = PCSetFromOptions_GAMG;
  pc->ops->setup          = PCSetUp_GAMG;
  pc->ops->reset          = PCReset_GAMG;
  pc->ops->destroy        = PCDestroy_GAMG;

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetProcEqLim_C",
					    "PCGAMGSetProcEqLim_GAMG",
					    PCGAMGSetProcEqLim_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetCoarseEqLim_C",
					    "PCGAMGSetCoarseEqLim_GAMG",
					    PCGAMGSetCoarseEqLim_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetRepartitioning_C",
					    "PCGAMGSetRepartitioning_GAMG",
					    PCGAMGSetRepartitioning_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetUseASMAggs_C",
					    "PCGAMGSetUseASMAggs_GAMG",
					    PCGAMGSetUseASMAggs_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetThreshold_C",
					    "PCGAMGSetThreshold_GAMG",
					    PCGAMGSetThreshold_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetType_C",
					    "PCGAMGSetType_GAMG",
					    PCGAMGSetType_GAMG);
  CHKERRQ(ierr);

  pc_gamg->repart = PETSC_FALSE;
  pc_gamg->use_aggs_in_gasm = PETSC_FALSE;
  pc_gamg->min_eq_proc = 100;
  pc_gamg->coarse_eq_limit = 800;
  pc_gamg->threshold = 0.05;
  pc_gamg->Nlevels = GAMG_MAXLEVELS;
  pc_gamg->verbose = 0;
  pc_gamg->emax_id = -1;
  pc_gamg->col_bs_id = -1;

  /* private events */
#if defined PETSC_GAMG_USE_LOG
  if( count++ == 0 ) {
    PetscLogEventRegister("GAMG: createProl", PC_CLASSID, &petsc_gamg_setup_events[SET1]);
    PetscLogEventRegister("  Graph", PC_CLASSID, &petsc_gamg_setup_events[GRAPH]);
    /* PetscLogEventRegister("    G.Mat", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_MAT]); */
    /* PetscLogEventRegister("    G.Filter", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_FILTER]); */
    /* PetscLogEventRegister("    G.Square", PC_CLASSID, &petsc_gamg_setup_events[GRAPH_SQR]); */
    PetscLogEventRegister("  MIS/Agg", PC_CLASSID, &petsc_gamg_setup_events[SET4]);
    PetscLogEventRegister("  geo: growSupp", PC_CLASSID, &petsc_gamg_setup_events[SET5]);
    PetscLogEventRegister("  geo: triangle", PC_CLASSID, &petsc_gamg_setup_events[SET6]);
    PetscLogEventRegister("    search&set", PC_CLASSID, &petsc_gamg_setup_events[FIND_V]);
    PetscLogEventRegister("  SA: col data", PC_CLASSID, &petsc_gamg_setup_events[SET7]);
    PetscLogEventRegister("  SA: frmProl0", PC_CLASSID, &petsc_gamg_setup_events[SET8]);
    PetscLogEventRegister("  SA: smooth", PC_CLASSID, &petsc_gamg_setup_events[SET9]);
    PetscLogEventRegister("GAMG: partLevel", PC_CLASSID, &petsc_gamg_setup_events[SET2]);
    PetscLogEventRegister("  repartition", PC_CLASSID, &petsc_gamg_setup_events[SET12]);
    PetscLogEventRegister("  Invert-Sort", PC_CLASSID, &petsc_gamg_setup_events[SET13]);
    PetscLogEventRegister("  Move A", PC_CLASSID, &petsc_gamg_setup_events[SET14]); 
    PetscLogEventRegister("  Move P", PC_CLASSID, &petsc_gamg_setup_events[SET15]); 

    /* PetscLogEventRegister(" PL move data", PC_CLASSID, &petsc_gamg_setup_events[SET13]); */
    /* PetscLogEventRegister("GAMG: fix", PC_CLASSID, &petsc_gamg_setup_events[SET10]); */
    /* PetscLogEventRegister("GAMG: set levels", PC_CLASSID, &petsc_gamg_setup_events[SET11]); */
    /* create timer stages */
#if defined GAMG_STAGES
    {
      char str[32];
      sprintf(str,"MG Level %d (finest)",0);
      PetscLogStageRegister(str, &gamg_stages[0]);
      PetscInt lidx;
      for (lidx=1;lidx<9;lidx++){
	sprintf(str,"MG Level %d",lidx);
	PetscLogStageRegister(str, &gamg_stages[lidx]);
      }
    }
#endif
  }
#endif
  /* general events */
#if defined PETSC_USE_LOG
  PetscLogEventRegister("PCGAMGgraph_AGG", 0, &PC_GAMGGgraph_AGG);
  PetscLogEventRegister("PCGAMGgraph_GEO", PC_CLASSID, &PC_GAMGGgraph_GEO);
  PetscLogEventRegister("PCGAMGcoarse_AGG", PC_CLASSID, &PC_GAMGCoarsen_AGG);
  PetscLogEventRegister("PCGAMGcoarse_GEO", PC_CLASSID, &PC_GAMGCoarsen_GEO);
  PetscLogEventRegister("PCGAMGProl_AGG", PC_CLASSID, &PC_GAMGProlongator_AGG);
  PetscLogEventRegister("PCGAMGProl_GEO", PC_CLASSID, &PC_GAMGProlongator_GEO);
  PetscLogEventRegister("PCGAMGPOpt_AGG", PC_CLASSID, &PC_GAMGOptprol_AGG);
#endif

  /* instantiate derived type */
  ierr = PetscOptionsHead("GAMG options"); CHKERRQ(ierr);
  {
    char tname[256] = GAMGAGG;
    ierr = PetscOptionsList("-pc_gamg_type","Type of GAMG method","PCGAMGSetType",
                            GAMGList, tname, tname, sizeof(tname), PETSC_NULL );
    CHKERRQ(ierr);
    ierr = PCGAMGSetType( pc, tname ); CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
