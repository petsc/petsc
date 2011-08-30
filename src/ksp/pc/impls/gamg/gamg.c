/*
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */
#include <../src/ksp/pc/impls/gamg/gamg.h>

PetscLogEvent gamg_setup_stages[NUM_SET];

/* Private context for the GAMG preconditioner */
typedef struct gamg_TAG {
  PetscInt       m_dim;
  PetscInt       m_Nlevels;
  PetscInt       m_data_sz;
  PetscInt       m_data_rows;
  PetscInt       m_data_cols;
  PetscBool      m_useSA;
  PetscReal     *m_data; /* blocked vector of vertex data on fine grid (coordinates) */
} PC_GAMG;

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
  PetscBool      flag;
  char           str[64];

  PetscFunctionBegin;
  ierr  = MatGetBlockSize( Amat, &bs );               CHKERRQ( ierr );
  ierr  = MatGetOwnershipRange( Amat, &my0, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-my0)/bs; 
  if((Iend-my0)%bs!=0) SETERRQ1(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Bad local size %d.",nloc);

  ierr  = PetscOptionsGetString(PETSC_NULL,"-pc_gamg_type",str,64,&flag);    CHKERRQ( ierr );
  pc_gamg->m_useSA = (PetscBool)(flag && strcmp(str,"sa") == 0);

  pc_gamg->m_data_rows = 1; 
  if(a_coords == 0) pc_gamg->m_useSA = PETSC_TRUE; /* use SA if no data */
  if( !pc_gamg->m_useSA ) pc_gamg->m_data_cols = a_ndm; /* coordinates */
  else{ /* SA: null space vectors */
    if(a_coords != 0 && bs==1 ) pc_gamg->m_data_cols = 1; /* scalar w/ coords and SA (not needed) */
    else if(a_coords != 0) pc_gamg->m_data_cols = (a_ndm==2 ? 3 : 6); /* elasticity */
    else pc_gamg->m_data_cols = bs; /* no data, force SA with constant null space vectors */
    pc_gamg->m_data_rows = bs; 
  }
  arrsz = nloc*pc_gamg->m_data_rows*pc_gamg->m_data_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (!pc_gamg->m_data || (pc_gamg->m_data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->m_data );  CHKERRQ(ierr);
    ierr = PetscMalloc((arrsz+1)*sizeof(double), &pc_gamg->m_data ); CHKERRQ(ierr);
  }
  for(kk=0;kk<arrsz;kk++)pc_gamg->m_data[kk] = -999.;
  pc_gamg->m_data[arrsz] = -99.;
  /* copy data in - column oriented */
  if( pc_gamg->m_useSA ) {
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


/* -----------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCReset_GAMG"
PetscErrorCode PCReset_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->m_data);CHKERRQ(ierr);
  pc_gamg->m_data = 0; pc_gamg->m_data_sz = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   partitionLevel

   Input Parameter:
   . a_Amat_fine - matrix on this fine (k) level
   . a_ndata_rows - size of data to move (coarse grid)
   . a_ndata_cols - size of data to move (coarse grid)
   In/Output Parameter:
   . a_P_inout - prolongation operator to the next level (k-1)
   . a_coarse_data - data that need to be moved
   . a_nactive_proc - number of active procs
   Output Parameter:
   . a_Amat_crs - coarse matrix that is created (k-1)
*/

#define MIN_EQ_PROC 300
#define TOP_GRID_LIM 1000

#undef __FUNCT__
#define __FUNCT__ "partitionLevel"
PetscErrorCode partitionLevel( Mat a_Amat_fine,
                               PetscInt a_ndata_rows,
                               PetscInt a_ndata_cols,
			       PetscInt a_cbs,
                               Mat *a_P_inout,
                               PetscReal **a_coarse_data,
                               PetscMPIInt *a_nactive_proc,
                               Mat *a_Amat_crs
                               )
{
  PetscErrorCode   ierr;
  Mat              Cmat,Pnew,Pold=*a_P_inout;
  IS               new_indices,isnum;
  MPI_Comm         wcomm = ((PetscObject)a_Amat_fine)->comm;
  PetscMPIInt      mype,npe;
  PetscInt         neq,NN,Istart,Iend,Istart0,Iend0,ncrs_new;
  PetscMPIInt      new_npe,nactive,ncrs0;
  PetscBool        flag = PETSC_FALSE;
 
  PetscFunctionBegin;  
  ierr = MPI_Comm_rank( wcomm, &mype ); CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );  CHKERRQ(ierr);
  /* RAP */
  ierr = MatPtAP( a_Amat_fine, Pold, MAT_INITIAL_MATRIX, 2.0, &Cmat ); CHKERRQ(ierr);
  
  ierr = MatSetBlockSize( Cmat, a_cbs );      CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Cmat, &Istart0, &Iend0 ); CHKERRQ(ierr);
  ncrs0 = (Iend0-Istart0)/a_cbs; assert((Iend0-Istart0)%a_cbs == 0);
  
  ierr  = PetscOptionsHasName(PETSC_NULL,"-pc_gamg_avoid_repartitioning",&flag);
  CHKERRQ( ierr );
  if( flag ) { 
    *a_Amat_crs = Cmat; /* output */
  }
  else {
    /* Repartition Cmat_{k} and move colums of P^{k}_{k-1} and coordinates accordingly */
    MatPartitioning  mpart;
    Mat              adj;
    const PetscInt *idx, data_sz=a_ndata_rows*a_ndata_cols;
    const PetscInt  stride0=ncrs0*a_ndata_rows,*is_idx;
    PetscInt         is_sz,*isnewproc_idx,ii,jj,kk,strideNew,tidx[ncrs0*data_sz];;
    /* create sub communicator  */
    MPI_Comm         cm,new_comm;
    IS               isnewproc;
    MPI_Group        wg, g2;
    PetscMPIInt      ranks[npe],counts[npe];
    IS               isscat;
    PetscScalar    *array;
    Vec             src_crd, dest_crd;
    PetscReal      *data = *a_coarse_data;
    VecScatter      vecscat;
    PetscInt        

    /* get number of PEs to make active, reduce */
    ierr = MatGetSize( Cmat, &neq, &NN );CHKERRQ(ierr);
    new_npe = neq/MIN_EQ_PROC; /* hardwire min. number of eq/proc */
    if( new_npe == 0 || neq < TOP_GRID_LIM ) new_npe = 1; 
    else if (new_npe >= *a_nactive_proc ) new_npe = *a_nactive_proc; /* no change, rare */

    ierr = MPI_Allgather( &ncrs0, 1, MPI_INT, counts, 1, MPI_INT, wcomm ); CHKERRQ(ierr); 
    assert(counts[mype]==ncrs0);
    /* count real active pes */
    for( nactive = jj = 0 ; jj < npe ; jj++) {
      if( counts[jj] != 0 ) {
	ranks[nactive++] = jj;
      }
    }
    assert(nactive>=new_npe);

    PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s npe (active): %d --> %d. new npe = %d, neq = %d\n",mype,__FUNCT__,*a_nactive_proc,nactive,new_npe,neq);
    *a_nactive_proc = new_npe; /* output */
    
    ierr = MPI_Comm_group( wcomm, &wg ); CHKERRQ(ierr); 
    ierr = MPI_Group_incl( wg, nactive, ranks, &g2 ); CHKERRQ(ierr); 
    ierr = MPI_Comm_create( wcomm, g2, &cm ); CHKERRQ(ierr); 
    if( cm != MPI_COMM_NULL ) {
      ierr = PetscCommDuplicate( cm, &new_comm, PETSC_NULL ); CHKERRQ(ierr);
      ierr = MPI_Comm_free( &cm );                             CHKERRQ(ierr);
    }
    ierr = MPI_Group_free( &wg );                            CHKERRQ(ierr);
    ierr = MPI_Group_free( &g2 );                            CHKERRQ(ierr);

    /* MatPartitioningApply call MatConvert, which is collective */
    ierr = PetscLogEventBegin(gamg_setup_stages[SET12],0,0,0,0);CHKERRQ(ierr);
    if( a_cbs == 1) {
      ierr = MatConvert( Cmat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj );   CHKERRQ(ierr);
    }
    else{
      /* make a scalar matrix to partition */
      Mat tMat;
      PetscInt ncols; const PetscScalar *vals; const PetscInt *idx;
      MatInfo info;
      ierr = MatGetInfo(Cmat,MAT_LOCAL,&info); CHKERRQ(ierr);
      ncols = (PetscInt)info.nz_used/((ncrs0+1)*a_cbs*a_cbs)+1;
      
      ierr = MatCreateMPIAIJ( wcomm, ncrs0, ncrs0,
                              PETSC_DETERMINE, PETSC_DETERMINE,
                              2*ncols, PETSC_NULL, ncols, PETSC_NULL,
                              &tMat );
      CHKERRQ(ierr);

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

      ierr = MatConvert( tMat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj );   CHKERRQ(ierr);

      ierr = MatDestroy( &tMat );  CHKERRQ(ierr);
    }

    if( ncrs0 != 0 ){
      /* hack to fix global data that pmetis.c uses in 'adj' */
      for( nactive = jj = 0 ; jj < npe ; jj++ ) {
	if( counts[jj] != 0 ) {
	  adj->rmap->range[nactive++] = adj->rmap->range[jj];
	}
      }
      adj->rmap->range[nactive] = adj->rmap->range[npe];

      ierr = MatPartitioningCreate( new_comm, &mpart ); CHKERRQ(ierr);
      ierr = MatPartitioningSetAdjacency( mpart, adj ); CHKERRQ(ierr);
      ierr = MatPartitioningSetFromOptions( mpart );    CHKERRQ(ierr);
      ierr = MatPartitioningSetNParts( mpart, new_npe );CHKERRQ(ierr);
      ierr = MatPartitioningApply( mpart, &isnewproc ); CHKERRQ(ierr);
      ierr = MatPartitioningDestroy( &mpart );          CHKERRQ(ierr);

      /* collect IS info */
      ierr = ISGetLocalSize( isnewproc, &is_sz );        CHKERRQ(ierr);
      ierr = PetscMalloc( a_cbs*is_sz*sizeof(PetscInt), &isnewproc_idx ); CHKERRQ(ierr);
      ierr = ISGetIndices( isnewproc, &is_idx );     CHKERRQ(ierr);
      /* spread partitioning across machine - probably the right thing to do but machine spec. */
      NN = npe/new_npe;
      for(kk=0,jj=0;kk<is_sz;kk++){
        for(ii=0 ; ii<a_cbs ; ii++, jj++ ) {
          isnewproc_idx[jj] = is_idx[kk] * NN; /* distribution */
        }
      }
      ierr = ISRestoreIndices( isnewproc, &is_idx );     CHKERRQ(ierr);
      ierr = ISDestroy( &isnewproc );                    CHKERRQ(ierr);
      is_sz *= a_cbs;

      ierr = MPI_Comm_free( &new_comm );    CHKERRQ(ierr);  
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
    ierr = ISPartitioningCount( isnewproc, npe, counts ); CHKERRQ(ierr);
    ierr = ISDestroy( &isnewproc );                       CHKERRQ(ierr);
    ncrs_new = counts[mype]/a_cbs;
    strideNew = ncrs_new*a_ndata_rows;
    ierr = PetscLogEventEnd(gamg_setup_stages[SET12],0,0,0,0);   CHKERRQ(ierr);

    /* Create a vector to contain the newly ordered element information */
    ierr = VecCreate( wcomm, &dest_crd );
    ierr = VecSetSizes( dest_crd, data_sz*ncrs_new, PETSC_DECIDE ); CHKERRQ(ierr);
    ierr = VecSetFromOptions( dest_crd ); CHKERRQ(ierr); /*funny vector-get global options?*/
    /*
      There are 'a_ndata_rows*a_ndata_cols' data items per node, (one can think of the vectors of having 
      a block size of ...).  Note, ISs are expanded into equation space by 'a_cbs'.
    */
    ierr = ISGetIndices( isnum, &idx ); CHKERRQ(ierr);
    for(ii=0,jj=0; ii<ncrs0 ; ii++) {
      PetscInt id = idx[ii*a_cbs]/a_cbs; /* get node back */
      for( kk=0; kk<data_sz ; kk++, jj++) tidx[jj] = id*data_sz + kk;
    }
    ierr = ISRestoreIndices( isnum, &idx ); CHKERRQ(ierr);
    ierr = ISCreateGeneral( wcomm, data_sz*ncrs0, tidx, PETSC_COPY_VALUES, &isscat );
    CHKERRQ(ierr);
    /*
      Create a vector to contain the original vertex information for each element
    */
    ierr = VecCreateSeq( PETSC_COMM_SELF, data_sz*ncrs0, &src_crd ); CHKERRQ(ierr);
    for( jj=0; jj<a_ndata_cols ; jj++ ) {
      for( ii=0 ; ii<ncrs0 ; ii++) {
	for( kk=0; kk<a_ndata_rows ; kk++ ) {
	  PetscInt ix = ii*a_ndata_rows + kk + jj*stride0, jx = ii*data_sz + kk*a_ndata_cols + jj;
	  ierr = VecSetValues( src_crd, 1, &jx, &data[ix], INSERT_VALUES );  CHKERRQ(ierr);
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
      ierr = MatGetSubMatrix( Pold, findices, new_indices, MAT_INITIAL_MATRIX, &Pnew );
      CHKERRQ(ierr);
      ierr = ISDestroy( &findices ); CHKERRQ(ierr);
    }
    ierr = MatDestroy( a_P_inout ); CHKERRQ(ierr);
    *a_P_inout = Pnew; /* output */
    ierr = ISDestroy( &new_indices ); CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

#define GAMG_MAXLEVELS 30
#if defined(PETSC_USE_LOG)
PetscLogStage  gamg_stages[20];
#endif
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
  Mat              Amat = a_pc->mat, Pmat = a_pc->pmat;
  PetscInt         fine_level, level, level1, M, N, bs, nloc, lidx, Istart, Iend;
  MPI_Comm         wcomm = ((PetscObject)a_pc)->comm;
  PetscMPIInt      mype,npe,nactivepe;
  PetscBool        isOK;
  Mat              Aarr[GAMG_MAXLEVELS], Parr[GAMG_MAXLEVELS];
  PetscReal       *coarse_data = 0, *data, emaxs[GAMG_MAXLEVELS];
  MatInfo          info;

  PetscFunctionBegin;
  if( a_pc->setupcalled ) {
    /* no state data in GAMG to destroy */
    ierr = PCReset_MG( a_pc ); CHKERRQ(ierr);
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
  PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s level %d N=%d, n data rows=%d, n data cols=%d, nnz/row (ave)=%d, np=%d\n",
	      mype,__FUNCT__,0,N,pc_gamg->m_data_rows,pc_gamg->m_data_cols,(PetscInt)(info.nz_used/(PetscReal)N),npe);
  for ( level=0, Aarr[0] = Pmat, nactivepe = npe; /* hard wired stopping logic */
        level < GAMG_MAXLEVELS-1 && (level==0 || M>TOP_GRID_LIM) && (npe==1 || nactivepe>1); 
        level++ ){
    level1 = level + 1;
    ierr = PetscLogEventBegin(gamg_setup_stages[SET1],0,0,0,0);CHKERRQ(ierr);
    ierr = createProlongation(Aarr[level], data, pc_gamg->m_dim, pc_gamg->m_data_cols, pc_gamg->m_useSA,
                              level, &bs, &Parr[level1], &coarse_data, &isOK, &emaxs[level] );
    CHKERRQ(ierr);
    ierr = PetscFree( data ); CHKERRQ( ierr );
    ierr = PetscLogEventEnd(gamg_setup_stages[SET1],0,0,0,0);CHKERRQ(ierr);
    
    if(level==0) Aarr[0] = Amat; /* use Pmat for finest level setup, but use mat for solver */

    if( isOK ) {
      ierr = PetscLogEventBegin(gamg_setup_stages[SET2],0,0,0,0);CHKERRQ(ierr);
      ierr = partitionLevel( Aarr[level], pc_gamg->m_useSA ? bs : 1, pc_gamg->m_data_cols, bs,
                             &Parr[level1], &coarse_data, &nactivepe, &Aarr[level1] );
      CHKERRQ(ierr);
      ierr = PetscLogEventEnd(gamg_setup_stages[SET2],0,0,0,0);CHKERRQ(ierr);
      ierr = MatGetSize( Aarr[level1], &M, &N );CHKERRQ(ierr);
      ierr = MatGetInfo(Aarr[level1],MAT_GLOBAL_SUM,&info); CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"\t\t[%d]%s %d) N=%d, bs=%d, n data cols=%d, nnz/row (ave)=%d, %d active pes\n",
		  mype,__FUNCT__,level1,N,bs,pc_gamg->m_data_cols,(PetscInt)(info.nz_used/(PetscReal)N),nactivepe);
      /* coarse grids with SA can have zero row/cols from singleton aggregates */
      /* aggregation method can probably gaurrentee this does not happen! - be safe for now */
 
      if( PETSC_TRUE ){
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
            PetscPrintf(PETSC_COMM_SELF,"\t[%d]%s warning: added diag to zero (%d) on level %d \n",mype,__FUNCT__,id,level);
          }
        }
        ierr = VecRestoreArray( diag, &data_arr ); CHKERRQ(ierr);
        ierr = VecDestroy( &diag );                CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Aarr[level1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Aarr[level1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
    }
    else{
      coarse_data = 0;
      break;
    }
    data = coarse_data;
  }
  if( coarse_data ) {
    ierr = PetscFree( coarse_data ); CHKERRQ( ierr );
  }
  PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s %d levels\n",0,__FUNCT__,level + 1);
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
    ierr = PCMGGetSmoother( a_pc, lidx, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPCHEBYCHEV );CHKERRQ(ierr);
    if( emaxs[level] > 0.0 ) emax=emaxs[level];
    else{ /* eigen estimate 'emax' */
      KSP eksp; Mat Lmat = Aarr[level];
      Vec bb, xx; PC pc;

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
      ierr = KSPSetInitialGuessNonzero( eksp, PETSC_FALSE ); CHKERRQ(ierr);
      ierr = KSPSetOperators( eksp, Lmat, Lmat, DIFFERENT_NONZERO_PATTERN ); CHKERRQ( ierr );
      ierr = KSPGetPC( eksp, &pc );CHKERRQ( ierr );
      ierr = PCSetType( pc, PCPBJACOBI ); CHKERRQ(ierr); /* should be same as above */
      ierr = KSPSetTolerances( eksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 10 );
      CHKERRQ(ierr);
      //ierr = KSPSetConvergenceTest( eksp, KSPSkipConverged, 0, 0 ); CHKERRQ(ierr);
      ierr = KSPSetNormType( eksp, KSP_NORM_NONE );                 CHKERRQ(ierr);

      ierr = KSPSetComputeSingularValues( eksp,PETSC_TRUE ); CHKERRQ(ierr);
      ierr = KSPSolve( eksp, bb, xx ); CHKERRQ(ierr);
      ierr = KSPComputeExtremeSingularValues( eksp, &emax, &emin ); CHKERRQ(ierr);
      ierr = VecDestroy( &xx );       CHKERRQ(ierr);
      ierr = VecDestroy( &bb );       CHKERRQ(ierr); 
      ierr = KSPDestroy( &eksp );       CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"\t\t\t%s max eigen=%e min=%e PC=%s\n",__FUNCT__,emax,emin,PETSC_GAMG_SMOOTHER);
    }
    {
      PetscInt N1, N0, tt;
      ierr = MatGetSize( Aarr[level], &N1, &tt );         CHKERRQ(ierr);
      ierr = MatGetSize( Aarr[level+1], &N0, &tt );       CHKERRQ(ierr);
      emin = 1.*emax/((PetscReal)N1/(PetscReal)N0); /* this should be about the coarsening rate */
      emax *= 1.05;

    }

    ierr = KSPSetOperators( smoother, Aarr[level], Aarr[level], DIFFERENT_NONZERO_PATTERN );
    ierr = KSPChebychevSetEigenvalues( smoother, emax, emin );CHKERRQ(ierr);
    /*ierr = KSPSetTolerances(smoother,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,2); CHKERRQ(ierr);*/
    ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);
    ierr = PCSetType( subpc, PETSC_GAMG_SMOOTHER ); CHKERRQ(ierr);
    ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
  }
  {
    KSP smoother; /* coarse grid */
    Mat Lmat = Aarr[pc_gamg->m_Nlevels-1];
    ierr = PCMGGetSmoother( a_pc, 0, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetOperators( smoother, Lmat, Lmat, DIFFERENT_NONZERO_PATTERN );
    CHKERRQ(ierr);
    ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
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

  /* set interpolation between the levels, create timer stages, clean up */
  if( PETSC_FALSE ) {
    char str[32];
    sprintf(str,"MG Level %d (%d)",0,pc_gamg->m_Nlevels-1);
    PetscLogStageRegister(str, &gamg_stages[fine_level]);
  }
  for (lidx=0,level=pc_gamg->m_Nlevels-1;
       lidx<fine_level;
       lidx++, level--){
    ierr = PCMGSetInterpolation( a_pc, lidx+1, Parr[level] );CHKERRQ(ierr);
    if( !PETSC_TRUE ) {
      PetscViewer viewer; char fname[32];
      sprintf(fname,"Pmat_%d.m",level);
      ierr = PetscViewerASCIIOpen( wcomm, fname, &viewer );  CHKERRQ(ierr);
      ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
      ierr = MatView( Parr[level], viewer ); CHKERRQ(ierr);
      ierr = PetscViewerDestroy( &viewer );
      sprintf(fname,"Amat_%d.m",level);
      ierr = PetscViewerASCIIOpen( wcomm, fname, &viewer );  CHKERRQ(ierr);
      ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
      ierr = MatView( Aarr[level], viewer ); CHKERRQ(ierr);
      ierr = PetscViewerDestroy( &viewer );
    }
    ierr = MatDestroy( &Parr[level] );  CHKERRQ(ierr);
    ierr = MatDestroy( &Aarr[level] );  CHKERRQ(ierr);
    if( PETSC_FALSE ) {
      char str[32];
      sprintf(str,"MG Level %d (%d)",lidx+1,level-1);
      PetscLogStageRegister(str, &gamg_stages[level-1]);
    }
  }

  /* setupcalled is set to 0 so that MG is setup from scratch */
  a_pc->setupcalled = 0;
  ierr = PCSetUp_MG(a_pc);CHKERRQ(ierr);

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
#define __FUNCT__ "PCSetFromOptions_GAMG"
PetscErrorCode PCSetFromOptions_GAMG(PC pc)
{
  /* PetscErrorCode  ierr; */
  /* PC_MG           *mg = (PC_MG*)pc->data; */
  /* PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx; */
  /* MPI_Comm        comm = ((PetscObject)pc)->comm; */

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCCreate_GAMG - Creates a GAMG preconditioner context, PC_GAMG

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()

  */
 /* MC
     PCGAMG - Use algebraic multigrid preconditioning. This preconditioner requires you provide
       fine grid discretization matrix and coordinates on the fine grid.

   Options Database Key:
   Multigrid options(inherited)
+  -pc_mg_cycles <1>: 1 for V cycle, 2 for W-cycle (MGSetCycles)
.  -pc_mg_smoothup <1>: Number of post-smoothing steps (MGSetNumberSmoothUp)
.  -pc_mg_smoothdown <1>: Number of pre-smoothing steps (MGSetNumberSmoothDown)
   -pc_mg_type <multiplicative>: (one of) additive multiplicative full cascade kascade
   GAMG options:

   Level: intermediate
  Concepts: multigrid

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType,
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), MPSetCycles(), PCMGSetNumberSmoothDown(),
           PCMGSetNumberSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCyclesOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()
M */

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_GAMG"
PetscErrorCode  PCCreate_GAMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_GAMG         *pc_gamg;
  PC_MG           *mg;
  PetscClassId     cookie;

  PetscFunctionBegin;
  /* PCGAMG is an inherited class of PCMG. Initialize pc as PCMG */
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr); /* calls PCCreate_MG() and MGCreate_Private() */
  ierr = PetscObjectChangeTypeName((PetscObject)pc,PCGAMG);CHKERRQ(ierr);

  /* create a supporting struct and attach it to pc */
  ierr = PetscNewLog(pc,PC_GAMG,&pc_gamg);CHKERRQ(ierr);
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
					    PCSetCoordinates_GAMG);CHKERRQ(ierr);
  static int count = 0;
  if( count++ == 0 ) {
    PetscClassIdRegister("GAMG Setup",&cookie);
    PetscLogEventRegister("GAMG: createProl", cookie, &gamg_setup_stages[SET1]);
    PetscLogEventRegister(" make graph", cookie, &gamg_setup_stages[SET3]);
    PetscLogEventRegister(" MIS/Agg", cookie, &gamg_setup_stages[SET4]);
    PetscLogEventRegister("  geo: growSupp", cookie, &gamg_setup_stages[SET5]);
    PetscLogEventRegister("  geo: triangle", cookie, &gamg_setup_stages[SET6]);
    PetscLogEventRegister("   search & set", cookie, &gamg_setup_stages[FIND_V]);
    PetscLogEventRegister("  SA: init", cookie, &gamg_setup_stages[SET7]);
    /* PetscLogEventRegister("  SA: frmProl0", cookie, &gamg_setup_stages[SET8]); */
    PetscLogEventRegister("  SA: smooth", cookie, &gamg_setup_stages[SET9]);
    PetscLogEventRegister("GAMG: partLevel", cookie, &gamg_setup_stages[SET2]);
    PetscLogEventRegister(" PL repartition", cookie, &gamg_setup_stages[SET12]);
    /* PetscLogEventRegister(" PL move data", cookie, &gamg_setup_stages[SET13]); */
    /* PetscLogEventRegister("GAMG: fix", cookie, &gamg_setup_stages[SET10]); */
    /* PetscLogEventRegister("GAMG: set levels", cookie, &gamg_setup_stages[SET11]); */
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END
