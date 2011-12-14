/*
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */
#include "private/matimpl.h"
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/

#if defined PETSC_USE_LOG 
PetscLogEvent gamg_setup_events[NUM_SET];
#endif
#define GAMG_MAXLEVELS 30

/*#define GAMG_STAGES*/
#if (defined PETSC_USE_LOG && defined GAMG_STAGES)
static PetscLogStage gamg_stages[GAMG_MAXLEVELS];
#endif

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_GAMG

   Input Parameter:
   .  pc - the preconditioner context
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_GAMG"
PetscErrorCode PCSetCoordinates_GAMG( PC a_pc, PetscInt a_ndm, PetscReal *a_coords )
{
  PC_MG          *mg = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;
  PetscInt       arrsz,bs,my0,kk,ii,jj,nloc,Iend;
  Mat            Amat = a_pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific( Amat, MAT_CLASSID, 1 );
  ierr  = MatGetBlockSize( Amat, &bs );               CHKERRQ( ierr );
  ierr  = MatGetOwnershipRange( Amat, &my0, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-my0)/bs; 
  if((Iend-my0)%bs!=0) SETERRQ1(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Bad local size %d.",nloc);
 
  pc_gamg->m_data_rows = 1;
  if(a_coords==0 && pc_gamg->m_method==0) {
    SETERRQ(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Need coordinates for pc_gamg_type 'geo'.");
  }
  if( pc_gamg->m_method==0 ) pc_gamg->m_data_cols = a_ndm; /* coordinates */
  else{ /* SA: null space vectors */
    if(a_coords != 0 && bs==1 ) pc_gamg->m_data_cols = 1; /* scalar w/ coords and SA (not needed) */
    else if(a_coords != 0 ) pc_gamg->m_data_cols = (a_ndm==2 ? 3 : 6); /* elasticity */
    else pc_gamg->m_data_cols = bs; /* no data, force SA with constant null space vectors */
    pc_gamg->m_data_rows = bs; 
  }
  arrsz = nloc*pc_gamg->m_data_rows*pc_gamg->m_data_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (pc_gamg->m_data==0 || (pc_gamg->m_data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->m_data );  CHKERRQ(ierr);
    ierr = PetscMalloc((arrsz+1)*sizeof(PetscReal), &pc_gamg->m_data ); CHKERRQ(ierr);
  }
  for(kk=0;kk<arrsz;kk++)pc_gamg->m_data[kk] = -999.;
  pc_gamg->m_data[arrsz] = -99.;
  /* copy data in - column oriented */
  if( pc_gamg->m_method != 0 ) {
    const PetscInt M = Iend - my0;
    for(kk=0;kk<nloc;kk++){
      PetscReal *data = &pc_gamg->m_data[kk*bs];
      if( pc_gamg->m_data_cols==1 ) *data = 1.0;
      else {
        for(ii=0;ii<bs;ii++)
	  for(jj=0;jj<bs;jj++)
	    if(ii==jj)data[ii*M + jj] = 1.0; /* translational modes */
	    else data[ii*M + jj] = 0.0;
        if( a_coords != 0 ) {
          if( a_ndm == 2 ){ /* rotational modes */
            data += 2*M;
            data[0] = -a_coords[2*kk+1];
            data[1] =  a_coords[2*kk];
          }
          else {
            data += 3*M;
            data[0] = 0.0;               data[M+0] =  a_coords[3*kk+2]; data[2*M+0] = -a_coords[3*kk+1];
            data[1] = -a_coords[3*kk+2]; data[M+1] = 0.0;               data[2*M+1] =  a_coords[3*kk];
            data[2] =  a_coords[3*kk+1]; data[M+2] = -a_coords[3*kk];   data[2*M+2] = 0.0;
          }          
        }
      }
    }
  }
  else {
    for( kk = 0 ; kk < nloc ; kk++ ){
      for( ii = 0 ; ii < a_ndm ; ii++ ) {
        pc_gamg->m_data[ii*nloc + kk] =  a_coords[kk*a_ndm + ii];
      }
    }
  }
  assert(pc_gamg->m_data[arrsz] == -99.);
    
  pc_gamg->m_data_sz = arrsz;
  pc_gamg->m_dim = a_ndm;

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* ----------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCReset_GAMG"
PetscErrorCode PCReset_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  if( pc_gamg->m_data != 0 ) { /* this should not happen, cleaned up in SetUp */
    ierr = PetscFree(pc_gamg->m_data); CHKERRQ(ierr);
  }
  pc_gamg->m_data = 0; pc_gamg->m_data_sz = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGPartitionLevel

   Input Parameter:
   . a_Amat_fine - matrix on this fine (k) level
   . a_ndata_rows - size of data to move (coarse grid)
   . a_ndata_cols - size of data to move (coarse grid)
   . a_pc_gamg - parameters
   In/Output Parameter:
   . a_P_inout - prolongation operator to the next level (k-1)
   . a_coarse_data - data that need to be moved
   . a_nactive_proc - number of active procs
   Output Parameter:
   . a_Amat_crs - coarse matrix that is created (k-1)
*/

#undef __FUNCT__
#define __FUNCT__ "PCGAMGPartitionLevel"
PetscErrorCode PCGAMGPartitionLevel(PC pc, Mat a_Amat_fine,
                                    PetscInt a_ndata_rows,
                                    PetscInt a_ndata_cols,
                                    PetscInt a_cbs,
                                    Mat *a_P_inout,
                                    PetscReal **a_coarse_data,
                                    PetscMPIInt *a_nactive_proc,
                                    Mat *a_Amat_crs
                                    )
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscBool  avoid_repart = pc_gamg->m_avoid_repart;
  const PetscInt   min_eq_proc = pc_gamg->m_min_eq_proc, coarse_max = pc_gamg->m_coarse_eq_limit;
  PetscErrorCode   ierr;
  Mat              Cmat,Pnew,Pold=*a_P_inout;
  IS               new_indices,isnum;
  MPI_Comm         wcomm = ((PetscObject)a_Amat_fine)->comm;
  PetscMPIInt      mype,npe,new_npe,nactive;
  PetscInt         neq,NN,Istart,Iend,Istart0,Iend0,ncrs_new,ncrs0;
 
  PetscFunctionBegin;  
  ierr = MPI_Comm_rank( wcomm, &mype ); CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );  CHKERRQ(ierr);

  /* RAP */
#ifdef USE_R
  /* make R wih brute force for now */
  ierr = MatTranspose( Pold, Pnew );     
  ierr = MatDestroy( &Pold );  CHKERRQ(ierr);
  ierr = MatRARt( a_Amat_fine, Pnew, MAT_INITIAL_MATRIX, 2.0, &Cmat ); CHKERRQ(ierr);
  Pold = Pnew;
#else
  ierr = MatPtAP( a_Amat_fine, Pold, MAT_INITIAL_MATRIX, 2.0, &Cmat ); CHKERRQ(ierr);
#endif
  ierr = MatSetBlockSize( Cmat, a_cbs );      CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Cmat, &Istart0, &Iend0 ); CHKERRQ(ierr);
  ncrs0 = (Iend0-Istart0)/a_cbs; assert((Iend0-Istart0)%a_cbs == 0);

  /* get number of PEs to make active, reduce */
  ierr = MatGetSize( Cmat, &neq, &NN );  CHKERRQ(ierr);
  new_npe = neq/min_eq_proc; /* hardwire min. number of eq/proc */
  if( new_npe == 0 || neq < coarse_max ) new_npe = 1; 
  else if (new_npe >= *a_nactive_proc ) new_npe = *a_nactive_proc; /* no change, rare */

  if( avoid_repart && !(new_npe == 1 && *a_nactive_proc != 1) ) { 
    *a_Amat_crs = Cmat; /* output */
  }
  else {
    /* Repartition Cmat_{k} and move colums of P^{k}_{k-1} and coordinates accordingly */
    Mat              adj;
    const PetscInt *idx,data_sz=a_ndata_rows*a_ndata_cols;
    const PetscInt  stride0=ncrs0*a_ndata_rows;
    PetscInt        is_sz,*isnewproc_idx,ii,jj,kk,strideNew,*tidx;
    /* create sub communicator  */
    MPI_Comm        cm;
    MPI_Group       wg, g2;
    PetscInt       *counts,inpe;
    PetscMPIInt    *ranks;
    IS              isscat;
    PetscScalar    *array;
    Vec             src_crd, dest_crd;
    PetscReal      *data = *a_coarse_data;
    VecScatter      vecscat;
    IS  isnewproc;

    ierr = PetscMalloc( npe*sizeof(PetscMPIInt), &ranks ); CHKERRQ(ierr); 
    ierr = PetscMalloc( npe*sizeof(PetscInt), &counts ); CHKERRQ(ierr); 
    
    ierr = MPI_Allgather( &ncrs0, 1, MPIU_INT, counts, 1, MPIU_INT, wcomm ); CHKERRQ(ierr); 
    assert(counts[mype]==ncrs0);
    /* count real active pes */
    for( nactive = jj = 0 ; jj < npe ; jj++) {
      if( counts[jj] != 0 ) {
	ranks[nactive++] = jj;
        }
    }

    if (nactive < new_npe) new_npe = nactive; /* this can happen with empty input procs */

    if (pc_gamg->m_verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s npe (active): %d --> %d. new npe = %d, neq = %d\n",mype,__FUNCT__,*a_nactive_proc,nactive,new_npe,neq);

    *a_nactive_proc = new_npe; /* output */
    
    ierr = MPI_Comm_group( wcomm, &wg ); CHKERRQ(ierr); 
    ierr = MPI_Group_incl( wg, nactive, ranks, &g2 ); CHKERRQ(ierr); 
    ierr = MPI_Comm_create( wcomm, g2, &cm ); CHKERRQ(ierr); 

    ierr = MPI_Group_free( &wg );                            CHKERRQ(ierr);
    ierr = MPI_Group_free( &g2 );                            CHKERRQ(ierr);

    /* MatPartitioningApply call MatConvert, which is collective */
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET12],0,0,0,0);CHKERRQ(ierr);
#endif
    if( a_cbs == 1) { 
      ierr = MatConvert( Cmat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj );   CHKERRQ(ierr);
    }
    else{
      /* make a scalar matrix to partition */
      Mat tMat;
      PetscInt ncols,jj,Ii; 
      const PetscScalar *vals; 
      const PetscInt *idx;
      PetscInt *d_nnz;
      static int llev = 0;

      ierr = PetscMalloc( ncrs0*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
      for ( Ii = Istart0, jj = 0 ; Ii < Iend0 ; Ii += a_cbs, jj++ ) {
        ierr = MatGetRow(Cmat,Ii,&ncols,0,0); CHKERRQ(ierr);
        d_nnz[jj] = ncols/a_cbs;
        if( d_nnz[jj] > ncrs0 ) d_nnz[jj] = ncrs0; 
        ierr = MatRestoreRow(Cmat,Ii,&ncols,0,0); CHKERRQ(ierr);    
      }
      
      ierr = MatCreateMPIAIJ( wcomm, ncrs0, ncrs0,
                              PETSC_DETERMINE, PETSC_DETERMINE,
                              0, d_nnz, 0, d_nnz,
                              &tMat );
      CHKERRQ(ierr);
      ierr = PetscFree( d_nnz ); CHKERRQ(ierr);
      
      for ( ii = Istart0; ii < Iend0; ii++ ) {
        PetscInt dest_row = ii/a_cbs;
        ierr = MatGetRow(Cmat,ii,&ncols,&idx,&vals); CHKERRQ(ierr);
        for( jj = 0 ; jj < ncols ; jj++ ){
          PetscInt dest_col = idx[jj]/a_cbs;
          PetscScalar v = 1.0;
          ierr = MatSetValues(tMat,1,&dest_row,1,&dest_col,&v,ADD_VALUES); CHKERRQ(ierr);
        }
        ierr = MatRestoreRow(Cmat,ii,&ncols,&idx,&vals); CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(tMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(tMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      if( llev++ == -1 ) {
	PetscViewer viewer; char fname[32];
	sprintf(fname,"part_mat_%d.mat",llev);
	PetscViewerBinaryOpen(wcomm,fname,FILE_MODE_WRITE,&viewer);
	ierr = MatView( tMat, viewer ); CHKERRQ(ierr);
	ierr = PetscViewerDestroy( &viewer );
      }
      
      ierr = MatConvert( tMat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj );   CHKERRQ(ierr);

      ierr = MatDestroy( &tMat );  CHKERRQ(ierr);
    }

    if( ncrs0 != 0 ){
      const PetscInt *is_idx;
      MatPartitioning  mpart;
      /* hack to fix global data that pmetis.c uses in 'adj' */
      for( nactive = jj = 0 ; jj < npe ; jj++ ) {
	if( counts[jj] != 0 ) {
	  adj->rmap->range[nactive++] = adj->rmap->range[jj];
	}
      }
      adj->rmap->range[nactive] = adj->rmap->range[npe];
      
      ierr = MatPartitioningCreate( cm, &mpart ); CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency( mpart, adj ); CHKERRQ(ierr);
      ierr = MatPartitioningSetFromOptions( mpart );    CHKERRQ(ierr);
      ierr = MatPartitioningSetNParts( mpart, new_npe );CHKERRQ(ierr);
      ierr = MatPartitioningApply( mpart, &isnewproc ); CHKERRQ(ierr);
      ierr = MatPartitioningDestroy( &mpart );          CHKERRQ(ierr);

      /* collect IS info */
      ierr = ISGetLocalSize( isnewproc, &is_sz );       CHKERRQ(ierr);
      ierr = PetscMalloc( a_cbs*is_sz*sizeof(PetscInt), &isnewproc_idx ); CHKERRQ(ierr);
      ierr = ISGetIndices( isnewproc, &is_idx );        CHKERRQ(ierr);
      /* spread partitioning across machine - best way ??? */
      NN = 1; /*npe/new_npe;*/
      for( kk = jj = 0 ; kk < is_sz ; kk++ ){
        for( ii = 0 ; ii < a_cbs ; ii++, jj++ ) {
          isnewproc_idx[jj] = is_idx[kk] * NN; /* distribution */
        }
      }
      ierr = ISRestoreIndices( isnewproc, &is_idx );     CHKERRQ(ierr);
      ierr = ISDestroy( &isnewproc );                    CHKERRQ(ierr);
      ierr = MPI_Comm_free( &cm );              CHKERRQ(ierr);  

      is_sz *= a_cbs;
    }
    else{
      isnewproc_idx = 0;
      is_sz = 0;
    }

    ierr = MatDestroy( &adj );                       CHKERRQ(ierr);
    ierr = ISCreateGeneral( wcomm, is_sz, isnewproc_idx, PETSC_COPY_VALUES, &isnewproc );
    if( isnewproc_idx != 0 ) {
      ierr = PetscFree( isnewproc_idx );  CHKERRQ(ierr);
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
    ncrs_new = counts[mype]/a_cbs;
    strideNew = ncrs_new*a_ndata_rows;
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET12],0,0,0,0);   CHKERRQ(ierr);
#endif
    /* Create a vector to contain the newly ordered element information */
    ierr = VecCreate( wcomm, &dest_crd );
    ierr = VecSetSizes( dest_crd, data_sz*ncrs_new, PETSC_DECIDE ); CHKERRQ(ierr);
    ierr = VecSetFromOptions( dest_crd ); CHKERRQ(ierr); /* this is needed! */
    /*
     There are 'a_ndata_rows*a_ndata_cols' data items per node, (one can think of the vectors of having 
     a block size of ...).  Note, ISs are expanded into equation space by 'a_cbs'.
     */
    ierr = PetscMalloc( (ncrs0*data_sz)*sizeof(PetscInt), &tidx ); CHKERRQ(ierr); 
    ierr = ISGetIndices( isnum, &idx ); CHKERRQ(ierr);
    for(ii=0,jj=0; ii<ncrs0 ; ii++) {
      PetscInt id = idx[ii*a_cbs]/a_cbs; /* get node back */
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
    for( jj=0; jj<a_ndata_cols ; jj++ ) {
      for( ii=0 ; ii<ncrs0 ; ii++) {
	for( kk=0; kk<a_ndata_rows ; kk++ ) {
	  PetscInt ix = ii*a_ndata_rows + kk + jj*stride0, jx = ii*data_sz + kk*a_ndata_cols + jj;
          PetscScalar tt = (PetscScalar)data[ix];
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
    ierr = PetscFree( *a_coarse_data );    CHKERRQ(ierr);
    ierr = PetscMalloc( data_sz*ncrs_new*sizeof(PetscReal), a_coarse_data );    CHKERRQ(ierr);
    
    ierr = VecGetArray( dest_crd, &array );    CHKERRQ(ierr);
    data = *a_coarse_data;
    for( jj=0; jj<a_ndata_cols ; jj++ ) {
      for( ii=0 ; ii<ncrs_new ; ii++) {
	for( kk=0; kk<a_ndata_rows ; kk++ ) {
	  PetscInt ix = ii*a_ndata_rows + kk + jj*strideNew, jx = ii*data_sz + kk*a_ndata_cols + jj;
	  data[ix] = PetscRealPart(array[jx]);
	  array[jx] = 1.e300;
	}
      }
    }
    ierr = VecRestoreArray( dest_crd, &array );    CHKERRQ(ierr);
    ierr = VecDestroy( &dest_crd );    CHKERRQ(ierr);
    /*
      Invert for MatGetSubMatrix
    */
    ierr = ISInvertPermutation( isnum, ncrs_new*a_cbs, &new_indices ); CHKERRQ(ierr);
    ierr = ISSort( new_indices ); CHKERRQ(ierr); /* is this needed? */
    ierr = ISDestroy( &isnum ); CHKERRQ(ierr);
    /* A_crs output */
    ierr = MatGetSubMatrix( Cmat, new_indices, new_indices, MAT_INITIAL_MATRIX, a_Amat_crs );
    CHKERRQ(ierr);

    ierr = MatDestroy( &Cmat ); CHKERRQ(ierr);
    Cmat = *a_Amat_crs; /* output */
    ierr = MatSetBlockSize( Cmat, a_cbs );      CHKERRQ(ierr);

    /* prolongator */
    ierr = MatGetOwnershipRange( Pold, &Istart, &Iend );    CHKERRQ(ierr);
    {
      IS findices;
      ierr = ISCreateStride(wcomm,Iend-Istart,Istart,1,&findices);   CHKERRQ(ierr);
#ifdef USE_R
      ierr = MatGetSubMatrix( Pold, new_indices, findices, MAT_INITIAL_MATRIX, &Pnew );
#else
      ierr = MatGetSubMatrix( Pold, findices, new_indices, MAT_INITIAL_MATRIX, &Pnew );
#endif
      CHKERRQ(ierr);

      ierr = ISDestroy( &findices ); CHKERRQ(ierr);
    }

    ierr = MatDestroy( a_P_inout ); CHKERRQ(ierr);
    *a_P_inout = Pnew; /* output */

    ierr = ISDestroy( &new_indices ); CHKERRQ(ierr);
    ierr = PetscFree( counts );  CHKERRQ(ierr);
    ierr = PetscFree( ranks );  CHKERRQ(ierr);
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
PetscErrorCode PCSetUp_GAMG( PC a_pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)a_pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_MG_Levels   **mglevels = mg->levels;
  Mat              Amat = a_pc->mat, Pmat = a_pc->pmat;
  PetscInt         fine_level, level, level1, M, N, bs, nloc, lidx, Istart, Iend;
  MPI_Comm         wcomm = ((PetscObject)a_pc)->comm;
  PetscMPIInt      mype,npe,nactivepe;
  PetscBool        isOK;
  Mat              Aarr[GAMG_MAXLEVELS], Parr[GAMG_MAXLEVELS];
  PetscReal       *coarse_data = 0, *data, emaxs[GAMG_MAXLEVELS];
  MatInfo          info;

  PetscFunctionBegin;
  pc_gamg->m_count++;

  if( a_pc->setupcalled > 0 ) {
    /* just do Galerkin grids */
    Mat B,dA,dB;
    
    /* PCSetUp_MG seems to insists on setting this to GMRES */
    ierr = KSPSetType( mglevels[0]->smoothd, KSPPREONLY ); CHKERRQ(ierr);

    /* currently only handle case where mat and pmat are the same on coarser levels */
    ierr = KSPGetOperators(mglevels[pc_gamg->m_Nlevels-1]->smoothd,&dA,&dB,PETSC_NULL);CHKERRQ(ierr);
    /* (re)set to get dirty flag */
    ierr = KSPSetOperators(mglevels[pc_gamg->m_Nlevels-1]->smoothd,dA,dB,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetUp( mglevels[pc_gamg->m_Nlevels-1]->smoothd ); CHKERRQ(ierr);

    for (level=pc_gamg->m_Nlevels-2; level>-1; level--) {
      ierr = KSPGetOperators(mglevels[level]->smoothd,PETSC_NULL,&B,PETSC_NULL);CHKERRQ(ierr);
      /* the first time through the matrix structure has changed from repartitioning */
      if( pc_gamg->m_count == 2 ) {
        ierr = MatDestroy( &B );  CHKERRQ(ierr);
        ierr = MatPtAP(dB,mglevels[level+1]->interpolate,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
        mglevels[level]->A = B;
      }
      else {
        ierr = MatPtAP(dB,mglevels[level+1]->interpolate,MAT_REUSE_MATRIX,1.0,&B);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(mglevels[level]->smoothd,B,B,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
      dB = B;
      /* setup KSP/PC */
      ierr = KSPSetUp( mglevels[level]->smoothd ); CHKERRQ(ierr);
    }

#define PRINT_MATS PETSC_FALSE
    /* plot levels - A */
    if( PRINT_MATS ) {
      for (lidx=0, level=pc_gamg->m_Nlevels-1; level>0 ; level--,lidx++){
        PetscViewer viewer; 
        char fname[32]; KSP smoother; Mat Tmat, TTm;
        ierr = PCMGGetSmoother( a_pc, lidx, &smoother ); CHKERRQ(ierr);
        ierr = KSPGetOperators( smoother, &Tmat, &TTm, 0 ); CHKERRQ(ierr);
        sprintf(fname,"Amat_%d_%d.m",(int)pc_gamg->m_count,(int)level);
        ierr = PetscViewerASCIIOpen( wcomm, fname, &viewer );  CHKERRQ(ierr);
        ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
        ierr = MatView( Tmat, viewer ); CHKERRQ(ierr);
        ierr = PetscViewerDestroy( &viewer );
      }
    }

    a_pc->setupcalled = 2;

    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  /* GAMG requires input of fine-grid matrix. It determines nlevels. */
  ierr = MatGetBlockSize( Amat, &bs ); CHKERRQ(ierr);
  ierr = MatGetSize( Amat, &M, &N );CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; assert((Iend-Istart)%bs == 0);

  /* get data of not around */
  if( pc_gamg->m_data == 0 && nloc > 0 ) {
    ierr  = PCSetCoordinates_GAMG( a_pc, -1, 0 );    CHKERRQ( ierr );
  }
  data = pc_gamg->m_data;

  /* Get A_i and R_i */
  ierr = MatGetInfo(Amat,MAT_GLOBAL_SUM,&info); CHKERRQ(ierr);
  if (pc_gamg->m_verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s level %d N=%d, n data rows=%d, n data cols=%d, nnz/row (ave)=%d, np=%d\n",
	      mype,__FUNCT__,0,N,pc_gamg->m_data_rows,pc_gamg->m_data_cols,
	      (int)(info.nz_used/(PetscReal)N),npe);
  for ( level=0, Aarr[0] = Pmat, nactivepe = npe; /* hard wired stopping logic */
        level < (GAMG_MAXLEVELS-1) && (level==0 || M>pc_gamg->m_coarse_eq_limit); /* && (npe==1 || nactivepe>1); */
        level++ ){
    level1 = level + 1;
#if (defined PETSC_USE_LOG && defined GAMG_STAGES)
    ierr = PetscLogStagePush(gamg_stages[level]); CHKERRQ( ierr );
#endif
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = createProlongation(Aarr[level], data, level, &bs, &Parr[level1], &coarse_data, &isOK, &emaxs[level], pc_gamg );
    CHKERRQ(ierr);
    ierr = PetscFree( data ); CHKERRQ( ierr );
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#endif
    if(level==0) Aarr[0] = Amat; /* use Pmat for finest level setup, but use mat for solver */
    if( isOK ) {
#if defined PETSC_USE_LOG
      ierr = PetscLogEventBegin(gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = PCGAMGPartitionLevel(a_pc, Aarr[level], (pc_gamg->m_method != 0) ? bs : 1, pc_gamg->m_data_cols, bs,
                             &Parr[level1], &coarse_data, &nactivepe, &Aarr[level1] );
      CHKERRQ(ierr);
#if defined PETSC_USE_LOG
      ierr = PetscLogEventEnd(gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = MatGetSize( Aarr[level1], &M, &N );CHKERRQ(ierr);
      ierr = MatGetInfo(Aarr[level1],MAT_GLOBAL_SUM,&info); CHKERRQ(ierr);
      if (pc_gamg->m_verbose) PetscPrintf(PETSC_COMM_WORLD,"\t\t[%d]%s %d) N=%d, n data cols=%d, nnz/row (ave)=%d, %d active pes\n",
		  mype,__FUNCT__,(int)level1,N,pc_gamg->m_data_cols,
		  (int)(info.nz_used/(PetscReal)N),nactivepe);
      /* coarse grids with SA can have zero row/cols from singleton aggregates */
      /* aggregation method should gaurrentee this does not happen! */

      if (pc_gamg->m_verbose) {
        Vec diag; PetscScalar *data_arr,v; PetscInt Istart,Iend,kk,nloceq,id;
        v = 1.e-10; /* LU factor has hard wired numbers for small diags so this needs to match (yuk) */
        ierr = MatGetOwnershipRange(Aarr[level1], &Istart, &Iend); CHKERRQ(ierr);
        nloceq = Iend-Istart;
        ierr = MatGetVecs( Aarr[level1], &diag, 0 );    CHKERRQ(ierr);
        ierr = MatGetDiagonal( Aarr[level1], diag );    CHKERRQ(ierr);
        ierr = VecGetArray( diag, &data_arr );   CHKERRQ(ierr);
        for(kk=0;kk<nloceq;kk++){
          if(data_arr[kk]==0.0) {
            id = kk + Istart; 
            ierr = MatSetValues(Aarr[level1],1,&id,1,&id,&v,INSERT_VALUES);
            CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s warning: added zero to diag (%d) on level %d \n",mype,__FUNCT__,id,level1);
          }
        }
        ierr = VecRestoreArray( diag, &data_arr ); CHKERRQ(ierr);
        ierr = VecDestroy( &diag );                CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Aarr[level1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Aarr[level1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
    } else {
      coarse_data = 0;
      break;
    }
    data = coarse_data;

#if (defined PETSC_USE_LOG && defined GAMG_STAGES)
    ierr = PetscLogStagePop(); CHKERRQ( ierr );
#endif
  }
  if( coarse_data ) {
    ierr = PetscFree( coarse_data ); CHKERRQ( ierr );
  }
  if (pc_gamg->m_verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s %d levels\n",0,__FUNCT__,level + 1);
  pc_gamg->m_data = 0; /* destroyed coordinate data */
  pc_gamg->m_Nlevels = level + 1;
  fine_level = level;
  ierr = PCMGSetLevels(a_pc,pc_gamg->m_Nlevels,PETSC_NULL);CHKERRQ(ierr);

  /* set default smoothers */
  for ( lidx=1, level = pc_gamg->m_Nlevels-2;
        lidx <= fine_level;
        lidx++, level--) {
    PetscReal emax, emin;
    KSP smoother; PC subpc; 
    PetscBool isCheb;
    /* set defaults */
    ierr = PCMGGetSmoother( a_pc, lidx, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPCHEBYCHEV );CHKERRQ(ierr);
    ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);
    /* ierr = KSPSetTolerances(smoother,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,2); CHKERRQ(ierr); */
    ierr = PCSetType( subpc, PETSC_GAMG_SMOOTHER ); CHKERRQ(ierr);
    ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
    /* overide defaults with input parameters */
    ierr = KSPSetFromOptions( smoother ); CHKERRQ(ierr);

    ierr = KSPSetOperators( smoother, Aarr[level], Aarr[level], SAME_NONZERO_PATTERN );   CHKERRQ(ierr);
    /* do my own cheby */
    ierr = PetscTypeCompare( (PetscObject)smoother, KSPCHEBYCHEV, &isCheb ); CHKERRQ(ierr);
    if( isCheb ) {
      ierr = PetscTypeCompare( (PetscObject)subpc, PETSC_GAMG_SMOOTHER, &isCheb ); CHKERRQ(ierr);
      if( isCheb && emaxs[level] > 0.0 ) emax=emaxs[level]; /* eigen estimate only for diagnal PC */
      else{ /* eigen estimate 'emax' */
        KSP eksp; Mat Lmat = Aarr[level];
        Vec bb, xx; PC pc;
        const PCType type;
        
        ierr = PCGetType( subpc, &type );   CHKERRQ(ierr); 
        ierr = MatGetVecs( Lmat, &bb, 0 );         CHKERRQ(ierr);
        ierr = MatGetVecs( Lmat, &xx, 0 );         CHKERRQ(ierr);
        {
          PetscRandom    rctx;
          ierr = PetscRandomCreate(wcomm,&rctx);CHKERRQ(ierr);
          ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
          ierr = VecSetRandom(bb,rctx);CHKERRQ(ierr);
          ierr = PetscRandomDestroy( &rctx ); CHKERRQ(ierr);
        }
        ierr = KSPCreate(wcomm,&eksp);CHKERRQ(ierr);
        ierr = KSPSetType( eksp, KSPCG );                      CHKERRQ(ierr);
        ierr = KSPSetTolerances( eksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 10 );
        CHKERRQ(ierr);
        ierr = KSPSetNormType( eksp, KSP_NORM_NONE );                 CHKERRQ(ierr);
        
        ierr = KSPAppendOptionsPrefix( eksp, "est_");         CHKERRQ(ierr);
        ierr = KSPSetFromOptions( eksp );    CHKERRQ(ierr);
        
        ierr = KSPSetInitialGuessNonzero( eksp, PETSC_FALSE ); CHKERRQ(ierr);
        ierr = KSPSetOperators( eksp, Lmat, Lmat, SAME_NONZERO_PATTERN ); CHKERRQ( ierr );
        ierr = KSPGetPC( eksp, &pc );CHKERRQ( ierr );
        ierr = PCSetType( pc, type ); CHKERRQ(ierr); /* should be same as eigen estimates op. */
        
        ierr = KSPSetComputeSingularValues( eksp,PETSC_TRUE ); CHKERRQ(ierr);
        ierr = KSPSolve( eksp, bb, xx ); CHKERRQ(ierr);
        ierr = KSPComputeExtremeSingularValues( eksp, &emax, &emin ); CHKERRQ(ierr);
        ierr = VecDestroy( &xx );       CHKERRQ(ierr);
        ierr = VecDestroy( &bb );       CHKERRQ(ierr); 
        ierr = KSPDestroy( &eksp );       CHKERRQ(ierr);

        if (pc_gamg->m_verbose) {
          PetscPrintf(PETSC_COMM_WORLD,"\t\t\t%s PC setup max eigen=%e min=%e PC=%s\n",
                      __FUNCT__,emax,emin,PETSC_GAMG_SMOOTHER);
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
    }
  }
  {
    /* coarse grid */
    KSP smoother,*k2; PC subpc,pc2; PetscInt ii,first;
    Mat Lmat = Aarr[pc_gamg->m_Nlevels-1];
    ierr = PCMGGetSmoother( a_pc, 0, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetOperators( smoother, Lmat, Lmat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
    ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
    ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);
    ierr = PCSetType( subpc, PCBJACOBI ); CHKERRQ(ierr);
    ierr = PCSetUp( subpc ); CHKERRQ(ierr);
    ierr = PCBJacobiGetSubKSP(subpc,&ii,&first,&k2);CHKERRQ(ierr);
    assert(ii==1); 
    ierr = KSPGetPC(k2[0],&pc2);CHKERRQ(ierr); 
    ierr = PCSetType( pc2, PCLU ); CHKERRQ(ierr);
  }
 
  /* should be called in PCSetFromOptions_GAMG(), but cannot be called prior to PCMGSetLevels() */
  ierr = PCSetFromOptions_MG(a_pc); CHKERRQ(ierr);
  {
    PetscBool galerkin;
    ierr = PCMGGetGalerkin( a_pc,  &galerkin); CHKERRQ(ierr);
    if(galerkin){
      SETERRQ(wcomm,PETSC_ERR_ARG_WRONG, "GAMG does galerkin manually so it must not be used in PC_MG.");
    }
  }
  
  /* plot levels - R/P */
  if( PRINT_MATS ) {
    for (level=pc_gamg->m_Nlevels-1;level>0;level--){
      PetscViewer viewer;
      char fname[32];
      sprintf(fname,"Pmat_%d_%d.m",(int)pc_gamg->m_count,(int)level);
      ierr = PetscViewerASCIIOpen( wcomm, fname, &viewer );  CHKERRQ(ierr);
      ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
      ierr = MatView( Parr[level], viewer ); CHKERRQ(ierr);
      ierr = PetscViewerDestroy( &viewer );
      sprintf(fname,"Amat_%d_%d.m",(int)pc_gamg->m_count,(int)level);
      ierr = PetscViewerASCIIOpen( wcomm, fname, &viewer );  CHKERRQ(ierr);
      ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
      ierr = MatView( Aarr[level], viewer ); CHKERRQ(ierr);
      ierr = PetscViewerDestroy( &viewer );
    }
  }

  /* set interpolation between the levels, clean up */  
  for (lidx=0,level=pc_gamg->m_Nlevels-1;
       lidx<fine_level;
       lidx++, level--){
    ierr = PCMGSetInterpolation( a_pc, lidx+1, Parr[level] );CHKERRQ(ierr);
    ierr = MatDestroy( &Parr[level] );  CHKERRQ(ierr);
    ierr = MatDestroy( &Aarr[level] );  CHKERRQ(ierr);
  }
  
  /* setupcalled is set to 0 so that MG is setup from scratch */
  a_pc->setupcalled = 0;
  ierr = PCSetUp_MG( a_pc );CHKERRQ( ierr );
  a_pc->setupcalled = 1; /* use 1 as signal that this has not been re-setup */
  
  {
    KSP smoother;  /* PCSetUp_MG seems to insists on setting this to GMRES on coarse grid */
    ierr = PCMGGetSmoother( a_pc, 0, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPPREONLY ); CHKERRQ(ierr);
    ierr = KSPSetUp( smoother ); CHKERRQ(ierr);
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
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg= (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PCReset_GAMG(pc);CHKERRQ(ierr);
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetProcEqLim_GAMG"
PetscErrorCode PCGAMGSetProcEqLim_GAMG(PC pc, PetscInt n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  if(n>0) pc_gamg->m_min_eq_proc = n;
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
  if(n>0) pc_gamg->m_coarse_eq_limit = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCGAMGAvoidRepartitioning"
/*@
   PCGAMGAvoidRepartitioning - Do not repartition the coarse grids

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_avoid_repartitioning

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGAvoidRepartitioning(PC pc, PetscBool n)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGAvoidRepartitioning_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGAvoidRepartitioning_GAMG"
PetscErrorCode PCGAMGAvoidRepartitioning_GAMG(PC pc, PetscBool n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  
  PetscFunctionBegin;
  pc_gamg->m_avoid_repart = n;
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
  pc_gamg->m_threshold = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetSolverType"
/*@
   PCGAMGSetSolverType - Set solution method.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context


   Options Database Key:
.  -pc_gamg_type

   Level: intermediate

   Concepts: Unstructured multrigrid preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetSolverType(PC pc, char str[], PetscInt sz )
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSolverType_C",(PC,char[],PetscInt),(pc,str,sz));
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetSolverType_GAMG"
PetscErrorCode PCGAMGSetSolverType_GAMG(PC pc, char str[], PetscInt sz )
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx; 
  
  PetscFunctionBegin;
  if(sz < 64) strcpy(pc_gamg->m_type,str);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GAMG"
PetscErrorCode PCSetFromOptions_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscBool flag;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG options"); CHKERRQ(ierr);
  {
    ierr = PetscOptionsString("-pc_gamg_type",
                              "Solver type: plane aggregation ('pa'), smoothed aggregation ('sa') or geometric multigrid (default)",
                              "PCGAMGSetSolverType",
                              pc_gamg->m_type,
                              pc_gamg->m_type, 
                              64, 
                              &flag ); 
    CHKERRQ(ierr);
    if( flag && pc_gamg->m_data != 0 ) {
      if( (strcmp(pc_gamg->m_type,"sa")==0 && pc_gamg->m_method != 2) ||
          (strcmp(pc_gamg->m_type,"pa")==0 && pc_gamg->m_method != 1) ||
          (strcmp(pc_gamg->m_type,"geo")==0 && pc_gamg->m_method != 0) ) { 
        SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG, "PCSetFromOptions called of PCSetCoordinates (with new method, after data was created)"); 
      }
    }

    /* -pc_gamg_verbose */
    ierr = PetscOptionsBool("-pc_gamg_verbose","Verbose (debugging) output for PCGAMG","none",pc_gamg->m_verbose,&pc_gamg->m_verbose,PETSC_NULL);CHKERRQ(ierr);
    
    pc_gamg->m_method = 1; /* default to plane aggregation */
    if (flag ) {
      if( strcmp(pc_gamg->m_type,"sa") == 0) pc_gamg->m_method = 2;
      else if( strcmp(pc_gamg->m_type,"pa") == 0) pc_gamg->m_method = 1;
      else if( strcmp(pc_gamg->m_type,"geo") == 0) pc_gamg->m_method = 0;
      else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG, "Invalid gamg type: %s",pc_gamg->m_type); 
    }
    /* -pc_gamg_avoid_repartitioning */
    pc_gamg->m_avoid_repart = PETSC_FALSE;
    ierr = PetscOptionsBool("-pc_gamg_avoid_repartitioning",
                            "Do not repartion coarse grids (false)",
                            "PCGAMGAvoidRepartitioning",
                            pc_gamg->m_avoid_repart,
                            &pc_gamg->m_avoid_repart, 
                            &flag); 
    CHKERRQ(ierr);
    
    /* -pc_gamg_process_eq_limit */
    pc_gamg->m_min_eq_proc = 600;
    ierr = PetscOptionsInt("-pc_gamg_process_eq_limit",
                           "Limit (goal) on number of equations per process on coarse grids",
                           "PCGAMGSetProcEqLim",
                           pc_gamg->m_min_eq_proc,
                           &pc_gamg->m_min_eq_proc, 
                           &flag ); 
    CHKERRQ(ierr);
  
    /* -pc_gamg_coarse_eq_limit */
    pc_gamg->m_coarse_eq_limit = 1500;
    ierr = PetscOptionsInt("-pc_gamg_coarse_eq_limit",
                           "Limit on number of equations for the coarse grid",
                           "PCGAMGSetCoarseEqLim",
                           pc_gamg->m_coarse_eq_limit,
                           &pc_gamg->m_coarse_eq_limit, 
                           &flag );
    CHKERRQ(ierr);

    /* -pc_gamg_threshold */
    pc_gamg->m_threshold = 0.05;
    ierr = PetscOptionsReal("-pc_gamg_threshold",
                            "Relative threshold to use for dropping edges in aggregation graph",
                            "PCGAMGSetThreshold",
                            pc_gamg->m_threshold,
                            &pc_gamg->m_threshold, 
                            &flag ); 
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCGAMG - Geometric algebraic multigrid (AMG) preconditioning. This preconditioner currently has two 
           AMG methods: 1) an unstructured geometric method, which requires that you provide coordinates for each 
           vertex, and 2) smoothed aggregation.  Smoothed aggregation (SA) can work without coordinates but it 
           will generate some common non-trivial null spaces if coordinates are provided.  The input fine grid matrix  
           must have the block size set for 'system' problems (with multiple dof per vertex/cell) to work properly.  
           SA will generate rotational rigid body mode null space vectors, in addition to the trivial translational 
           modes, when coordinates are provide in 2D and 3D.

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
PetscErrorCode  PCCreate_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_GAMG         *pc_gamg;
  PC_MG           *mg;
  PetscClassId     cookie;
#if defined PETSC_USE_LOG
  static int count = 0;
#endif

  PetscFunctionBegin;
  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr); /* calls PCCreate_MG() and MGCreate_Private() */
  ierr = PetscObjectChangeTypeName((PetscObject)pc,PCGAMG);CHKERRQ(ierr);

  /* create a supporting struct and attach it to pc */
  ierr = PetscNewLog(pc,PC_GAMG,&pc_gamg);CHKERRQ(ierr);
  pc_gamg->m_data_sz = 0; pc_gamg->m_data = 0; pc_gamg->m_count = 0;
  mg = (PC_MG*)pc->data;
  mg->innerctx = pc_gamg;

  pc_gamg->m_Nlevels    = -1;

  /* overwrite the pointers of PCMG by the functions of PCGAMG */
  pc->ops->setfromoptions = PCSetFromOptions_GAMG;
  pc->ops->setup          = PCSetUp_GAMG;
  pc->ops->reset          = PCReset_GAMG;
  pc->ops->destroy        = PCDestroy_GAMG;

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCSetCoordinates_C",
					    "PCSetCoordinates_GAMG",
					    PCSetCoordinates_GAMG);
  CHKERRQ(ierr);

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
					    "PCGAMGAvoidRepartitioning_C",
					    "PCGAMGAvoidRepartitioning_GAMG",
					    PCGAMGAvoidRepartitioning_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetThreshold_C",
					    "PCGAMGSetThreshold_GAMG",
					    PCGAMGSetThreshold_GAMG);
  CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCGAMGSetSolverType_C",
					    "PCGAMGSetSolverType_GAMG",
					    PCGAMGSetSolverType_GAMG);
  CHKERRQ(ierr);

#if defined PETSC_USE_LOG
  if( count++ == 0 ) {
    PetscClassIdRegister("GAMG Setup",&cookie);
    PetscLogEventRegister("GAMG: createProl", cookie, &gamg_setup_events[SET1]);
    PetscLogEventRegister("  Graph", cookie, &gamg_setup_events[GRAPH]);
    PetscLogEventRegister("    G.Mat", cookie, &gamg_setup_events[GRAPH_MAT]);
    PetscLogEventRegister("    G.Filter", cookie, &gamg_setup_events[GRAPH_FILTER]);
    PetscLogEventRegister("    G.Square", cookie, &gamg_setup_events[GRAPH_SQR]);
    PetscLogEventRegister("  MIS/Agg", cookie, &gamg_setup_events[SET4]);
    PetscLogEventRegister("  geo: growSupp", cookie, &gamg_setup_events[SET5]);
    PetscLogEventRegister("  geo: triangle", cookie, &gamg_setup_events[SET6]);
    PetscLogEventRegister("    search&set", cookie, &gamg_setup_events[FIND_V]);
    PetscLogEventRegister("  SA: init", cookie, &gamg_setup_events[SET7]);
    /* PetscLogEventRegister("  SA: frmProl0", cookie, &gamg_setup_events[SET8]); */
    PetscLogEventRegister("  SA: smooth", cookie, &gamg_setup_events[SET9]);
    PetscLogEventRegister("GAMG: partLevel", cookie, &gamg_setup_events[SET2]);
    PetscLogEventRegister("  PL repartition", cookie, &gamg_setup_events[SET12]);

    /* PetscLogEventRegister("  PL 1", cookie, &gamg_setup_events[SET13]); */
    /* PetscLogEventRegister("  PL 2", cookie, &gamg_setup_events[SET14]); */
    /* PetscLogEventRegister("  PL 3", cookie, &gamg_setup_events[SET15]); */

    /* PetscLogEventRegister(" PL move data", cookie, &gamg_setup_events[SET13]); */
    /* PetscLogEventRegister("GAMG: fix", cookie, &gamg_setup_events[SET10]); */
    /* PetscLogEventRegister("GAMG: set levels", cookie, &gamg_setup_events[SET11]); */
    
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
  PetscFunctionReturn(0);
}
EXTERN_C_END
