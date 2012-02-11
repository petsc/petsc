/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include "private/matimpl.h"
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <private/kspimpl.h>

#if defined PETSC_USE_LOG 
PetscLogEvent gamg_setup_events[NUM_SET];
#endif
#define GAMG_MAXLEVELS 30

/*#define GAMG_STAGES*/
#if (defined PETSC_USE_LOG && defined GAMG_STAGES)
static PetscLogStage gamg_stages[GAMG_MAXLEVELS];
#endif

/* typedef enum { NOT_DONE=-2, DELETED=-1, REMOVED=-3 } NState; */
/* use int instead of enum to facilitate passing them via Scatters */
typedef int NState;
static const NState NOT_DONE=-2;
static const NState DELETED=-1;
static const NState REMOVED=-3;

#define  IS_SELECTED(s) (s!=DELETED && s!=NOT_DONE && s!=REMOVED)

int compare (const void *a, const void *b)
{
  return (((GAMGNode*)a)->degree - ((GAMGNode*)b)->degree);
}

static PetscFList GAMGList = 0;

/* -------------------------------------------------------------------------- */
/*
   createGraph
 
 Input Parameter:
 . pc - this
 . Amat - original graph
 Output Parameter:
 . Gmat - output, filter Graph
 . AuxMat - filter matrix for when 'square'
 . permIS - perumutation IS (this should not be here)
 */
#undef __FUNCT__
#define __FUNCT__ "createGraph"
PetscErrorCode createGraph(PC pc, const Mat Amat, Mat *a_Gmat, Mat *a_AuxMat, IS *a_permIS )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;  
  const PetscInt verbose = pc_gamg->verbose;
  const PetscReal vfilter = pc_gamg->threshold;
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,jj,ncols,nloc,kk,nnz0,nnz1,NN,MM,bs;
  PetscMPIInt    mype, npe;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  PetscInt       *d_nnz, *o_nnz;
  Mat            Gmat;
  const PetscScalar    *vals;
  const PetscInt *idx;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  ierr = MatGetSize( Amat, &MM, &NN ); CHKERRQ(ierr);
  ierr = MatGetBlockSize( Amat, &bs ); CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; 
 
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH_MAT],0,0,0,0);CHKERRQ(ierr);
#endif
  /* count nnz, there is sparcity in here so this might not be enough */
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &o_nnz ); CHKERRQ(ierr);
  for ( Ii = Istart, nnz0 = jj = 0 ; Ii < Iend ; Ii += bs, jj++ ) {
    ierr = MatGetRow(Amat,Ii,&ncols,0,0); CHKERRQ(ierr);
    d_nnz[jj] = ncols; /* very pessimistic */
    o_nnz[jj] = ncols;
    if( d_nnz[jj] > nloc ) d_nnz[jj] = nloc;
    if( o_nnz[jj] > (NN/bs-nloc) ) o_nnz[jj] = NN/bs-nloc;
    nnz0 += ncols;
    ierr = MatRestoreRow(Amat,Ii,&ncols,0,0); CHKERRQ(ierr);
  }
  nnz0 /= (nloc+1);

  /* get scalar copy (norms) of matrix */
  ierr = MatCreateMPIAIJ( wcomm, nloc, nloc,
                          PETSC_DETERMINE, PETSC_DETERMINE,
                          0, d_nnz, 0, o_nnz, &Gmat );

  for( Ii = Istart; Ii < Iend ; Ii++ ) {
    PetscInt dest_row = Ii/bs; 
    ierr = MatGetRow(Amat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
    for(jj=0;jj<ncols;jj++){
      PetscInt dest_col = idx[jj]/bs;
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      ierr = MatSetValues(Gmat,1,&dest_row,1,&dest_col,&sv,ADD_VALUES);  CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(Amat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Gmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Gmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* scale Gmat so filter works */
  {
    Vec diag;
    ierr = MatGetVecs( Gmat, &diag, 0 );    CHKERRQ(ierr);
    ierr = MatGetDiagonal( Gmat, diag );    CHKERRQ(ierr);
    ierr = VecReciprocal( diag );           CHKERRQ(ierr);
    ierr = VecSqrtAbs( diag );              CHKERRQ(ierr);
    ierr = MatDiagonalScale( Gmat, diag, diag );CHKERRQ(ierr);
    ierr = VecDestroy( &diag );           CHKERRQ(ierr);
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH_MAT],0,0,0,0);   CHKERRQ(ierr);
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH_FILTER],0,0,0,0);CHKERRQ(ierr);
#endif

  ierr = MatGetOwnershipRange(Gmat,&Istart,&Iend);CHKERRQ(ierr); /* use AIJ from here */
  {
    Mat Gmat2; 
    ierr = MatCreateMPIAIJ(wcomm,nloc,nloc,PETSC_DECIDE,PETSC_DECIDE,0,d_nnz,0,o_nnz,&Gmat2);
    CHKERRQ(ierr);
    for( Ii = Istart, nnz1 = 0 ; Ii < Iend; Ii++ ){
      ierr = MatGetRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
      for(jj=0;jj<ncols;jj++){
        PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
        if( PetscRealPart(sv) > vfilter ) {
          ierr = MatSetValues(Gmat2,1,&Ii,1,&idx[jj],&sv,INSERT_VALUES); CHKERRQ(ierr);
	  nnz1++;
        }
      }
      ierr = MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
    }
    nnz1 /= (nloc+1);
    ierr = MatAssemblyBegin(Gmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Gmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
    Gmat = Gmat2;
    if( verbose ) {
      PetscPrintf(PETSC_COMM_WORLD,"\t%s ave nnz/row %d --> %d\n",__FUNCT__,nnz0,nnz1); 
    }
  }
  ierr = PetscFree( d_nnz ); CHKERRQ(ierr);
  ierr = PetscFree( o_nnz ); CHKERRQ(ierr);

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH_FILTER],0,0,0,0);   CHKERRQ(ierr);
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH_SQR],0,0,0,0);CHKERRQ(ierr);
#endif
  /* square matrix - SA */  
  if( a_AuxMat ){
    Mat Gmat2;
    ierr = MatTransposeMatMult( Gmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );
    CHKERRQ(ierr);
    *a_AuxMat = Gmat;
    /* force compressed row storage for B matrix in AuxMat */
    if (npe > 1) {
      Mat_MPIAIJ *mpimat = (Mat_MPIAIJ*)Gmat->data;
      Mat_SeqAIJ *Bmat = (Mat_SeqAIJ*) mpimat->B->data;
      Bmat->compressedrow.check = PETSC_TRUE;
      ierr = MatCheckCompressedRow(mpimat->B,&Bmat->compressedrow,Bmat->i,Gmat->rmap->n,-1.0);
      CHKERRQ(ierr);
      assert( Bmat->compressedrow.use );
    }
    Gmat = Gmat2; /* now work with the squared matrix (get forced soon) */
  }

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH_SQR],0,0,0,0);   CHKERRQ(ierr);
#endif
  if (npe > 1) {
    /* force compressed row storage for B matrix */
    Mat_MPIAIJ *mpimat = (Mat_MPIAIJ*)Gmat->data;
    Mat_SeqAIJ *Bmat = (Mat_SeqAIJ*) mpimat->B->data;
    Bmat->compressedrow.check = PETSC_TRUE;
    ierr = MatCheckCompressedRow(mpimat->B,&Bmat->compressedrow,Bmat->i,Gmat->rmap->n,-1.0);
    CHKERRQ(ierr);
    assert( Bmat->compressedrow.use );
  }

  /* create random permutation with sort for geo-mg -- this should be refactored, its sort of geo-mg specific */
  {
    GAMGNode *gnodes;
    PetscInt *permute;
    
    ierr = PetscMalloc( nloc*sizeof(GAMGNode), &gnodes ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &permute ); CHKERRQ(ierr);

    for (Ii=Istart; Ii<Iend; Ii++) { /* locals only? */
      ierr = MatGetRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
      {
        PetscInt lid = Ii - Istart;
        gnodes[lid].lid = lid;
        gnodes[lid].degree = ncols;
      }
      ierr = MatRestoreRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
    }
    /* randomize */
    srand(1); /* make deterministic */
    if( PETSC_TRUE ) {
      PetscBool *bIndexSet;
      ierr = PetscMalloc( nloc*sizeof(PetscBool), &bIndexSet ); CHKERRQ(ierr);
      for ( Ii = 0; Ii < nloc ; Ii++) bIndexSet[Ii] = PETSC_FALSE;
      for ( Ii = 0; Ii < nloc ; Ii++)
      {
        PetscInt iSwapIndex = rand()%nloc;
        if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii)
        {
          GAMGNode iTemp = gnodes[iSwapIndex];
          gnodes[iSwapIndex] = gnodes[Ii];
          gnodes[Ii] = iTemp;
          bIndexSet[Ii] = PETSC_TRUE;
          bIndexSet[iSwapIndex] = PETSC_TRUE;
        }
      }
      ierr = PetscFree( bIndexSet );  CHKERRQ(ierr);
    }
    /* only sort locals */
    qsort( gnodes, nloc, sizeof(GAMGNode), compare );
    /* create IS of permutation */
    for(kk=0;kk<nloc;kk++) { /* locals only */
      permute[kk] = gnodes[kk].lid;
    }
    ierr = ISCreateGeneral( PETSC_COMM_SELF, (Iend-Istart), permute, PETSC_COPY_VALUES, a_permIS );
    CHKERRQ(ierr);

    ierr = PetscFree( gnodes );  CHKERRQ(ierr);
    ierr = PetscFree( permute );  CHKERRQ(ierr);
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH],0,0,0,0);   CHKERRQ(ierr);
#endif
  *a_Gmat = Gmat; 

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   getDataWithGhosts - hacks into Mat MPIAIJ so this must have > 1 pe

   Input Parameter:
   . Gmat - MPIAIJ matrix for scattters
   . data_sz - number of data terms per node (# cols in output)
   . data_in[nloc*data_sz] - column oriented data
   Output Parameter:
   . stride - numbrt of rows of output
   . data_out[stride*data_sz] - output data with ghosts
*/
#undef __FUNCT__
#define __FUNCT__ "getDataWithGhosts"
PetscErrorCode getDataWithGhosts( const Mat Gmat,
                                  const PetscInt data_sz,
                                  const PetscReal data_in[],
                                  PetscInt *a_stride,
                                  PetscReal **a_data_out
                                  )
{
  PetscErrorCode ierr;
  PetscMPIInt    mype,npe;
  MPI_Comm       wcomm = ((PetscObject)Gmat)->comm;
  Vec            tmp_crds;
  Mat_MPIAIJ    *mpimat = (Mat_MPIAIJ*)Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar   *data_arr;
  PetscReal     *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscTypeCompare( (PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  assert(isMPIAIJ);
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);                      assert(npe>1);
  ierr = MatGetOwnershipRange( Gmat, &my0, &Iend );    CHKERRQ(ierr);
  nloc = Iend - my0;
  ierr = VecGetLocalSize( mpimat->lvec, &num_ghosts );   CHKERRQ(ierr);
  nnodes = num_ghosts + nloc;
  *a_stride = nnodes;
  ierr = MatGetVecs( Gmat, &tmp_crds, 0 );    CHKERRQ(ierr);

  ierr = PetscMalloc( data_sz*nnodes*sizeof(PetscReal), &datas); CHKERRQ(ierr);
  for(dir=0; dir<data_sz; dir++) {
    /* set local, and global */
    for(kk=0; kk<nloc; kk++) {
      PetscInt gid = my0 + kk;
      PetscScalar crd = (PetscScalar)data_in[dir*nloc + kk]; /* col oriented */
      datas[dir*nnodes + kk] = PetscRealPart(crd);
      ierr = VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES ); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin( tmp_crds ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tmp_crds ); CHKERRQ(ierr);
    /* get ghost datas */
    ierr = VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( mpimat->lvec, &data_arr );   CHKERRQ(ierr);
    for(kk=nloc,jj=0;jj<num_ghosts;kk++,jj++){
      datas[dir*nnodes + kk] = PetscRealPart(data_arr[jj]);
    }
    ierr = VecRestoreArray( mpimat->lvec, &data_arr ); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&tmp_crds); CHKERRQ(ierr);

  *a_data_out = datas;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   smoothAggs - 

   Input Parameter:
   . Gmat_2 - glabal matrix of graph (data not defined)
   . Gmat_1
   . lid_state
   Input/Output Parameter:
   . id_llist - linked list rooted at selected node (size is nloc + nghosts_2)
   . deleted_parent_gid
*/
#undef __FUNCT__
#define __FUNCT__ "smoothAggs"
PetscErrorCode smoothAggs( const Mat Gmat_2, /* base (squared) graph */
                           const Mat Gmat_1, /* base graph, could be unsymmetic */
                           const PetscScalar lid_state[], /* [nloc], states */
                           PetscInt id_llist[], /* [nloc+nghost_2], aggragate list */
                           PetscScalar deleted_parent_gid[] /* [nloc], which pe owns my deleted */

                           )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA_1, *matB_1=0;
  MPI_Comm       wcomm = ((PetscObject)Gmat_2)->comm;
  PetscMPIInt    mype;
  PetscInt       nghosts_2,nghosts_1,lid,*ii,n,*idx,j,ix,Iend,my0;
  Mat_MPIAIJ    *mpimat_2 = 0, *mpimat_1=0;
  const PetscInt nloc = Gmat_2->rmap->n;
  PetscScalar   *cpcol_1_state;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat_1,&my0,&Iend);  CHKERRQ(ierr);

  if( !PETSC_TRUE ) {
    PetscViewer viewer; char fname[32]; static int llev=0;
    sprintf(fname,"Gmat1_%d.mat",llev++);
    PetscViewerASCIIOpen(wcomm,fname,&viewer);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Gmat_1, viewer ); CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }
  
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)Gmat_1, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);

  if (isMPI) {
    PetscInt    *gids, gid;
    PetscScalar *real_gids;
    Vec          tempVec;
    
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &gids ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscScalar), &real_gids ); CHKERRQ(ierr);

    for(lid=0,gid=my0;lid<nloc;lid++,gid++){
      gids[lid] = gid;
      real_gids[lid] = (PetscScalar)gid;
    }
    /* grab matrix objects */
    mpimat_2 = (Mat_MPIAIJ*)Gmat_2->data;
    mpimat_1 = (Mat_MPIAIJ*)Gmat_1->data;
    matA_1 = (Mat_SeqAIJ*)mpimat_1->A->data;
    matB_1 = (Mat_SeqAIJ*)mpimat_1->B->data;
    /* get ghost sizes */
    ierr = VecGetLocalSize( mpimat_1->lvec, &nghosts_1 ); CHKERRQ(ierr);
    ierr = VecGetLocalSize( mpimat_2->lvec, &nghosts_2 ); CHKERRQ(ierr);
    /* get 'cpcol_1_state' */
    ierr = MatGetVecs( Gmat_1, &tempVec, 0 );         CHKERRQ(ierr);
    if(nloc>0){
      ierr = VecSetValues( tempVec, nloc, gids, lid_state, INSERT_VALUES );  CHKERRQ(ierr); 
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( mpimat_1->lvec, &cpcol_1_state ); CHKERRQ(ierr);
    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr); 

    ierr = PetscFree( gids );  CHKERRQ(ierr);
    ierr = PetscFree( real_gids );  CHKERRQ(ierr);
  } else {
    PetscBool      isAIJ;
    ierr = PetscTypeCompare( (PetscObject)Gmat_2, MATSEQAIJ, &isAIJ ); CHKERRQ(ierr);
    assert(isAIJ);
    matA_1 = (Mat_SeqAIJ*)Gmat_1->data;
    nghosts_2 = nghosts_1 = 0;
  }
  assert( matA_1 && !matA_1->compressedrow.use );
  assert( matB_1==0 || matB_1->compressedrow.use );

  {
    PetscInt *lid_cprowID_1;
    PetscInt *lid_sel_lid;

    ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID_1 ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_sel_lid ); CHKERRQ(ierr);

    /*reverse map to selelected node */
    for(lid=0;lid<nloc;lid++) lid_cprowID_1[lid] = lid_sel_lid[lid] = -1;
    for(lid=0;lid<nloc;lid++){
      NState state = (NState)PetscRealPart(lid_state[lid]);
      if( IS_SELECTED(state) ){
        PetscInt flid = lid;
        do{
          lid_sel_lid[flid] = lid; assert(flid<nloc); 
        } while( (flid=id_llist[flid]) != -1 );
      }
    }

    /* set index into compressed row 'lid_cprowID' */
    if( matB_1 ) {
      ii = matB_1->compressedrow.i;
      for (ix=0; ix<matB_1->compressedrow.nrows; ix++) {
        PetscInt lid = matB_1->compressedrow.rindex[ix];
        lid_cprowID_1[lid] = ix;
      }
    }
    
    for(lid=0;lid<nloc;lid++){
      NState state = (NState)PetscRealPart(lid_state[lid]);
      if( IS_SELECTED(state) ) {
        /* steal locals */
        ii = matA_1->i; n = ii[lid+1] - ii[lid]; 
        idx = matA_1->j + ii[lid];
        for (j=0; j<n; j++) {
          PetscInt flid, lidj = idx[j];
          NState statej = (NState)PetscRealPart(lid_state[lidj]);
          if( statej==DELETED && lid_sel_lid[lidj] != lid ){ /* steal it */
	    if( lid_sel_lid[lidj] != -1 ){
	      /* I'm stealing this local */
	      PetscInt hav=0, flid2 = lid_sel_lid[lidj], lastid; assert(flid2!=-1);
	      for( lastid=flid2, flid=id_llist[flid2] ; flid!=-1 ; flid=id_llist[flid] ) {
		if( flid == lidj ) {
		  id_llist[lastid] = id_llist[lidj];                    /* remove lidj from list */
		  id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
		  hav++;
		  /* break; */
		}
		lastid = flid;
	      }
	      if(hav!=1){
                flid2 = lid_sel_lid[lidj];
		if(hav!=0){
		  SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,
			   "found %d vertices.  (if %d==0) failed to find self in 'selected' lists.  probably structurally unsymmetric matrix",
			   hav,hav);
		}
	      }
	    }
	    else{
	      /* I'm stealing this local, owned by a ghost */
	      deleted_parent_gid[lidj] = (PetscScalar)(lid+my0); /* this makes it a no-op later */
	      lid_sel_lid[lidj] = lid; /* not needed */
	      id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
	    }
	  }
        }
        /* ghosts are done by 'DELETED' branch */
      }
      else if( state == DELETED ) {
        /* see if I have a selected ghost neighbors */
        if( (ix=lid_cprowID_1[lid]) != -1 ) { 
          PetscInt hav = 0, old_sel_lid = lid_sel_lid[lid], lastid; assert(old_sel_lid<nloc);
          ii = matB_1->compressedrow.i; n = ii[ix+1] - ii[ix];
          idx = matB_1->j + ii[ix];
          for( j=0 ; j<n ; j++ ) {
            PetscInt cpid = idx[j];
            NState statej = (NState)PetscRealPart(cpcol_1_state[cpid]);
            if( IS_SELECTED(statej) ) {
              PetscInt new_sel_gid = (PetscInt)statej, hv=0, flid;
              hav++;
              /* remove from list */
	      if( old_sel_lid != -1 ) {
		/* assert(deleted_parent_gid[lid]==-1.0 ); */
		for( lastid=old_sel_lid, flid=id_llist[old_sel_lid] ; flid != -1 ; flid=id_llist[flid] ) {
		  if( flid == lid ) {
		    id_llist[lastid] = id_llist[lid];   /* remove lid from 'old_sel_lid' list */
		    hv++;
		    break;
		  }
		  lastid = flid;
		}
		/* assert(hv==1); */
	      }
	      else {
		assert(deleted_parent_gid[lid] != -1.0); /* stealing from one ghost, giving to another */
	      }

	      /* this will get other proc to add this */
              deleted_parent_gid[lid] = (PetscScalar)new_sel_gid; 
	    }
          }
          assert(hav <= 1);
        }
      }
    }
 
    ierr = PetscFree( lid_cprowID_1 );  CHKERRQ(ierr);
    ierr = PetscFree( lid_sel_lid );  CHKERRQ(ierr);
  }
  
  if(isMPI) {
    ierr = VecRestoreArray( mpimat_1->lvec, &cpcol_1_state ); CHKERRQ(ierr); 
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info.

   Input Parameter:
   . perm - serial permutation of rows of local to process in MIS
   . Gmat - glabal matrix of graph (data not defined)
   . Auxmat - non-squared matrix
   . strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "maxIndSetAgg"
PetscErrorCode maxIndSetAgg( const IS perm,
                             const Mat Gmat,
                             const Mat Auxmat,
			     const PetscBool strict_aggs,
                             IS *a_selected,
                             IS *a_locals_llist
                             )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA, *matB = 0;
  MPI_Comm       wcomm = ((PetscObject)Gmat)->comm;
  Vec            locState, ghostState;
  PetscInt       num_fine_ghosts,kk,n,ix,j,*idx,*ii,iter,Iend,my0;
  Mat_MPIAIJ    *mpimat = 0;
  PetscScalar   *cpcol_gid,*cpcol_state;
  PetscMPIInt    mype;
  const PetscInt *perm_ix;
  PetscInt nDone = 0, nselected = 0;
  const PetscInt nloc = Gmat->rmap->n;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)Gmat, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)Gmat->data;
    matA = (Mat_SeqAIJ*)mpimat->A->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
  } else {
    PetscBool      isAIJ;
    ierr = PetscTypeCompare( (PetscObject)Gmat, MATSEQAIJ, &isAIJ ); CHKERRQ(ierr);
    assert(isAIJ);
    matA = (Mat_SeqAIJ*)Gmat->data;
  }
  assert( matA && !matA->compressedrow.use );
  assert( matB==0 || matB->compressedrow.use );
  /* get vector */
  ierr = MatGetVecs( Gmat, &locState, 0 );         CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(Gmat,&my0,&Iend);  CHKERRQ(ierr);

  if( mpimat ) {
    PetscInt gid;
    for(kk=0,gid=my0;kk<nloc;kk++,gid++) {
      PetscScalar v = (PetscScalar)(gid);
      ierr = VecSetValues( locState, 1, &gid, &v, INSERT_VALUES );  CHKERRQ(ierr); /* set with GID */
    }
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr); /* get proc ID in 'cpcol_gid' */
    ierr = VecDuplicate( mpimat->lvec, &ghostState ); CHKERRQ(ierr); /* need 2nd compressed col. of off proc data */
    ierr = VecGetLocalSize( mpimat->lvec, &num_fine_ghosts ); CHKERRQ(ierr);
    ierr = VecSet( ghostState, (PetscScalar)((PetscReal)NOT_DONE) );  CHKERRQ(ierr); /* set with UNKNOWN state */
  }
  else num_fine_ghosts = 0;

  {  /* need an inverse map - locals */
    PetscInt *lid_cprowID, *lid_gid;
    PetscScalar *deleted_parent_gid; /* only used for strict aggs */
    PetscInt *id_llist; /* linked list with locality info - output */
    PetscScalar *lid_state;

    ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID ); CHKERRQ(ierr);
    ierr = PetscMalloc( (nloc+1)*sizeof(PetscInt), &lid_gid ); CHKERRQ(ierr);
    ierr = PetscMalloc( (nloc+1)*sizeof(PetscScalar), &deleted_parent_gid ); CHKERRQ(ierr);
    ierr = PetscMalloc( (nloc+num_fine_ghosts)*sizeof(PetscInt), &id_llist ); CHKERRQ(ierr);
    ierr = PetscMalloc( (nloc+1)*sizeof(PetscScalar), &lid_state ); CHKERRQ(ierr);

    for(kk=0;kk<nloc;kk++) {
      id_llist[kk] = -1; /* terminates linked lists */
      lid_cprowID[kk] = -1;
      deleted_parent_gid[kk] = -1.0;
      lid_gid[kk] = kk + my0;
      lid_state[kk] =  (PetscScalar)((PetscReal)NOT_DONE);
    }
    for(ix=0;kk<nloc+num_fine_ghosts;kk++,ix++) {
      id_llist[kk] = -1; /* terminates linked lists */
    }
    /* set index into cmpressed row 'lid_cprowID' */
    if( matB ) {
      ii = matB->compressedrow.i;
      for (ix=0; ix<matB->compressedrow.nrows; ix++) {
        PetscInt lid = matB->compressedrow.rindex[ix];
        lid_cprowID[lid] = ix;
      }
    }
    /* MIS */
    ierr = ISGetIndices( perm, &perm_ix );     CHKERRQ(ierr);
    iter = 0;
    while ( nDone < nloc || PETSC_TRUE ) { /* asyncronous not implemented */
      iter++;
      if( mpimat ) {
        ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
      }
      /* check all vertices */
      for(kk=0;kk<nloc;kk++){
        PetscInt lid = perm_ix[kk];
        NState state = (NState)PetscRealPart(lid_state[lid]);
        if( state == NOT_DONE ) {
          /* parallel test, delete if selected ghost */
          PetscBool isOK = PETSC_TRUE;
          if( (ix=lid_cprowID[lid]) != -1 ) { /* if I have any ghost neighbors */
            ii = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
            idx = matB->j + ii[ix];
            for( j=0 ; j<n ; j++ ) {
              PetscInt cpid = idx[j]; /* compressed row ID in B mat */
              PetscInt gid = (PetscInt)PetscRealPart(cpcol_gid[cpid]);
              NState statej = (NState)PetscRealPart(cpcol_state[cpid]);
              if( statej == NOT_DONE && gid >= Iend ) { /* should be (pe>mype), use gid as pe proxy */
                isOK = PETSC_FALSE; /* can not delete */
              }
              else if( IS_SELECTED(statej) ) { /* lid is now deleted, do it */
		assert(0);
              }
            }
          } /* parallel test */
          if( isOK ){ /* select or remove this vertex */
            nDone++;
            /* check for singleton */
            ii = matA->i; n = ii[lid+1] - ii[lid]; 
            if( n < 2 ) {
              /* if I have any ghost adj then not a sing */
              ix = lid_cprowID[lid];
              if( ix==-1 || (matB->compressedrow.i[ix+1]-matB->compressedrow.i[ix])==0 ){
                lid_state[lid] =  (PetscScalar)((PetscReal)REMOVED);
                continue; /* one local adj (me) and no ghost - singleton - flag and continue */
              }
            }
            /* SELECTED state encoded with global index */
            lid_state[lid] =  (PetscScalar)(lid+my0);
            nselected++;
	    /* delete local adj */
	    idx = matA->j + ii[lid];
	    for (j=0; j<n; j++) {
              PetscInt lidj = idx[j];
              NState statej = (NState)PetscRealPart(lid_state[lidj]);
              if( statej == NOT_DONE ){
                nDone++; 
                id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
                lid_state[lidj] = (PetscScalar)(PetscReal)DELETED;  /* delete this */
              }
            }

            /* delete ghost adj - deleted ghost done later for aggregation */
            if( !strict_aggs ) {
              if( (ix=lid_cprowID[lid]) != -1 ) { /* if I have any ghost neighbors */
                ii = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
                idx = matB->j + ii[ix];
                for( j=0 ; j<n ; j++ ) {
                  PetscInt cpid = idx[j]; /* compressed row ID in B mat */
                  NState statej = (NState)PetscRealPart(cpcol_state[cpid]); 
                  assert( !IS_SELECTED(statej) );
                  
		  if( statej == NOT_DONE ) {
		    PetscInt lidj = nloc + cpid;
		    /* cpcol_state[cpid] = (PetscScalar)DELETED; this should happen later ... */
		    id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
		  }
		}
	      }
	    }

          } /* selected */
        } /* not done vertex */
      } /* vertex loop */

      /* update ghost states and count todos */
      if( mpimat ) {
        PetscInt t1, t2;
        ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
        /* put lid state in 'locState' */
        ierr = VecSetValues( locState, nloc, lid_gid, lid_state, INSERT_VALUES ); CHKERRQ(ierr);
        ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
        ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
        /* scatter states, check for done */
        ierr = VecScatterBegin(mpimat->Mvctx,locState,ghostState,INSERT_VALUES,SCATTER_FORWARD);
        CHKERRQ(ierr);
        ierr =   VecScatterEnd(mpimat->Mvctx,locState,ghostState,INSERT_VALUES,SCATTER_FORWARD);
        CHKERRQ(ierr);
	/* delete locals from selected ghosts */
        ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
	ii = matB->compressedrow.i;        
	for (ix=0; ix<matB->compressedrow.nrows; ix++) {
	  PetscInt lid = matB->compressedrow.rindex[ix];
	  NState state = (NState)PetscRealPart(lid_state[lid]);
	  if( state == NOT_DONE ) {
	    /* look at ghosts */
	    n = ii[ix+1] - ii[ix];
	    idx = matB->j + ii[ix];
            for( j=0 ; j<n ; j++ ) {
              PetscInt cpid = idx[j]; /* compressed row ID in B mat */
              NState statej = (NState)PetscRealPart(cpcol_state[cpid]);
              if( IS_SELECTED(statej) ) { /* lid is now deleted, do it */
                PetscInt lidj = nloc + cpid;
                nDone++;
		lid_state[lid] = (PetscScalar)(PetscReal)DELETED; /* delete this */
		if( !strict_aggs ) {	
		  id_llist[lid] = id_llist[lidj]; id_llist[lidj] = lid; /* insert 'lid' into head of ghost llist */
		}
		else {
                  PetscInt gid = (PetscInt)PetscRealPart(cpcol_gid[cpid]);  
		  deleted_parent_gid[lid] = (PetscScalar)gid; /* keep track of proc that I belong to */
		}
		break;
	      }
	    }
	  }
	}
        ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);

	/* all done? */
        t1 = nloc - nDone; assert(t1>=0);
        ierr = MPI_Allreduce ( &t1, &t2, 1, MPIU_INT, MPIU_SUM, wcomm ); /* synchronous version */
        if( t2 == 0 ) break;
      }
      else break; /* all done */
    } /* outer parallel MIS loop */
    ierr = ISRestoreIndices(perm,&perm_ix);     CHKERRQ(ierr);

    if( mpimat ){ /* free this buffer up (not really needed here) */
      ierr = VecRestoreArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr);
    }
    
    /* adjust aggregates */
    if( strict_aggs ) {
      ierr = smoothAggs(Gmat, Auxmat, lid_state, id_llist, deleted_parent_gid); 
      CHKERRQ(ierr);
    }

    /* tell adj who my deleted vertices belong to */
    if( strict_aggs && matB ) {
      PetscScalar *cpcol_sel_gid; 
      PetscInt cpid;
      /* get proc of deleted ghost */
      ierr = VecSetValues(locState, nloc, lid_gid, deleted_parent_gid, INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecAssemblyBegin(locState); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(locState); CHKERRQ(ierr);
      ierr = VecScatterBegin(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr =   VecScatterEnd(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray( mpimat->lvec, &cpcol_sel_gid ); CHKERRQ(ierr); /* has pe that owns ghost */
      for(cpid=0; cpid<num_fine_ghosts; cpid++) {
        PetscInt gid = (PetscInt)PetscRealPart(cpcol_sel_gid[cpid]);
	if( gid >= my0 && gid < Iend ) { /* I own this deleted */
	  PetscInt lidj = nloc + cpid;
	  PetscInt lid = gid - my0;
	  id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
	  assert(IS_SELECTED( (NState)PetscRealPart(lid_state[lid]) ));
	}
      }
      ierr = VecRestoreArray( mpimat->lvec, &cpcol_sel_gid ); CHKERRQ(ierr);
    }

    /* create output IS of aggregates in linked list */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nloc+num_fine_ghosts,id_llist,PETSC_COPY_VALUES,a_locals_llist);
    CHKERRQ(ierr);

    /* make 'a_selected' - output */
    if( mpimat ) {
      ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    }
    for (j=0; j<num_fine_ghosts; j++) {
      if( IS_SELECTED( (NState)PetscRealPart(cpcol_state[j]) ) ) nselected++;
    }
    {
      PetscInt *selected_set;
      ierr = PetscMalloc( nselected*sizeof(PetscInt), &selected_set ); CHKERRQ(ierr); 
      for(kk=0,j=0;kk<nloc;kk++){
        NState state = (NState)PetscRealPart(lid_state[kk]);
        if( IS_SELECTED(state) ) {
          selected_set[j++] = kk;
        }
      }
      for (kk=0; kk<num_fine_ghosts; kk++) {
        if( IS_SELECTED( (NState)PetscRealPart(cpcol_state[kk]) ) ) {
          selected_set[j++] = nloc + kk;
        }
      }
      assert(j==nselected);
      ierr = ISCreateGeneral(PETSC_COMM_SELF, nselected, selected_set, PETSC_COPY_VALUES, a_selected );
      CHKERRQ(ierr);
      ierr = PetscFree( selected_set );  CHKERRQ(ierr);
    }
    if( mpimat ) {
      ierr = VecRestoreArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    }

    ierr = PetscFree( lid_cprowID );  CHKERRQ(ierr);
    ierr = PetscFree( lid_gid );  CHKERRQ(ierr);
    ierr = PetscFree( deleted_parent_gid );  CHKERRQ(ierr);
    ierr = PetscFree( id_llist );  CHKERRQ(ierr);
    ierr = PetscFree( lid_state );  CHKERRQ(ierr);
  } /* scoping */

  if(mpimat){
    ierr = VecDestroy( &ghostState ); CHKERRQ(ierr);
  }

  ierr = VecDestroy( &locState );                    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
   . ndata_rows - size of data to move (coarse grid)
   . ndata_cols - size of data to move (coarse grid)
   . cbs - coarse block size
   . isLast - 
   In/Output Parameter:
   . a_P_inout - prolongation operator to the next level (k-1)
   . a_coarse_data - data that need to be moved
   . a_nactive_proc - number of active procs
   Output Parameter:
   . a_Amat_crs - coarse matrix that is created (k-1)
*/

#undef __FUNCT__
#define __FUNCT__ "createLevel"
PetscErrorCode createLevel( const PC pc,
                            const Mat Amat_fine,
                            const PetscInt ndata_rows,
                            const PetscInt ndata_cols,
                            const PetscInt cbs,
                            const PetscBool isLast,
                            Mat *a_P_inout,
                            PetscReal **a_coarse_data,
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
    const PetscInt *idx,data_sz=ndata_rows*ndata_cols;
    const PetscInt  stride0=ncrs0*ndata_rows;
    PetscInt        *counts,is_sz,*newproc_idx,ii,jj,kk,strideNew,*tidx,inpe,targetPE;
    IS              isnewproc;
    VecScatter      vecscat;
    PetscScalar    *array;
    Vec             src_crd, dest_crd;
    PetscReal      *data = *a_coarse_data;
    IS              isscat;

    ierr = PetscMalloc( npe*sizeof(PetscInt), &counts ); CHKERRQ(ierr);

#if defined PETSC_USE_LOG
      ierr = PetscLogEventBegin(gamg_setup_events[SET12],0,0,0,0);CHKERRQ(ierr);
#endif
    if( repart ) {
      /* create sub communicator  */
      Mat             adj;

      if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s repartition: npe (active): %d --> %d, neq = %d\n",mype,__FUNCT__,*a_nactive_proc,new_npe,neq);

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
	static int llev = 0;
	
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
	
	ierr = MatCreateMPIAIJ( wcomm, ncrs0, ncrs0,
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
	  sprintf(fname,"part_mat_%d.mat",llev);
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
	ierr = PCGetOptionsPrefix(pc,&pcpre);CHKERRQ(ierr);
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
	if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s aggregate processors noop: new_npe=%d, neq=%d\n",mype,__FUNCT__,new_npe,neq);
	PetscFunctionReturn(0);
      }

      /* reduce - isnewproc */
      targetPE = mype/rfactor;

      if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s aggregate processors: npe: %d --> %d, neq=%d\n",mype,__FUNCT__,*a_nactive_proc,new_npe,neq);
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
#if defined PETSC_USE_LOG
      ierr = PetscLogEventEnd(gamg_setup_events[SET12],0,0,0,0);   CHKERRQ(ierr);
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
    for( jj=0; jj<ndata_cols ; jj++ ) {
      for( ii=0 ; ii<ncrs_new ; ii++) {
	for( kk=0; kk<ndata_rows ; kk++ ) {
	  PetscInt ix = ii*ndata_rows + kk + jj*strideNew, jx = ii*data_sz + kk*ndata_cols + jj;
	  data[ix] = PetscRealPart(array[jx]);
	  array[jx] = 1.e300;
	}
      }
    }
    ierr = VecRestoreArray( dest_crd, &array );    CHKERRQ(ierr);
    ierr = VecDestroy( &dest_crd );    CHKERRQ(ierr);
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET13],0,0,0,0);CHKERRQ(ierr);
#endif
    /*
      Invert for MatGetSubMatrix
    */
    ierr = ISInvertPermutation( isnum, ncrs_new*cbs, &new_indices ); CHKERRQ(ierr);
    ierr = ISSort( new_indices ); CHKERRQ(ierr); /* is this needed? */
    ierr = ISDestroy( &isnum ); CHKERRQ(ierr);
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET13],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(gamg_setup_events[SET14],0,0,0,0);CHKERRQ(ierr);
#endif
    /* A_crs output */
    ierr = MatGetSubMatrix( Cmat, new_indices, new_indices, MAT_INITIAL_MATRIX, a_Amat_crs );
    CHKERRQ(ierr);

    ierr = MatDestroy( &Cmat ); CHKERRQ(ierr);
    Cmat = *a_Amat_crs; /* output */
    ierr = MatSetBlockSize( Cmat, cbs );      CHKERRQ(ierr);
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET14],0,0,0,0);CHKERRQ(ierr);
#endif
    /* prolongator */
    ierr = MatGetOwnershipRange( Pold, &Istart, &Iend );    CHKERRQ(ierr);
    {
      IS findices;
#if defined PETSC_USE_LOG
      ierr = PetscLogEventBegin(gamg_setup_events[SET15],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = ISCreateStride(wcomm,Iend-Istart,Istart,1,&findices);   CHKERRQ(ierr);
#ifdef USE_R
      ierr = MatGetSubMatrix( Pold, new_indices, findices, MAT_INITIAL_MATRIX, &Pnew );
#else
      ierr = MatGetSubMatrix( Pold, findices, new_indices, MAT_INITIAL_MATRIX, &Pnew );
#endif
      CHKERRQ(ierr);
      ierr = ISDestroy( &findices ); CHKERRQ(ierr);
#if defined PETSC_USE_LOG
      ierr = PetscLogEventEnd(gamg_setup_events[SET15],0,0,0,0);CHKERRQ(ierr);
#endif
    }
    ierr = MatDestroy( a_P_inout ); CHKERRQ(ierr);
    *a_P_inout = Pnew; /* output - repartitioned */
    ierr = ISDestroy( &new_indices ); CHKERRQ(ierr);
    ierr = PetscFree( counts );  CHKERRQ(ierr);
  }

  *a_nactive_proc = new_npe; /* output */

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
  PC_MG_Levels   **mglevels = mg->levels;
  Mat              Amat = pc->mat, Pmat = pc->pmat;
  PetscInt         fine_level, level, level1, M, N, bs, nloc, lidx, Istart, Iend;
  MPI_Comm         wcomm = ((PetscObject)pc)->comm;
  PetscMPIInt      mype,npe,nactivepe;
  Mat              Aarr[GAMG_MAXLEVELS], Parr[GAMG_MAXLEVELS];
  PetscReal       *coarse_data = 0, *data, emaxs[GAMG_MAXLEVELS];
  MatInfo          info;

  PetscFunctionBegin;
  pc_gamg->setup_count++;
  assert(pc_gamg->createprolongator);

  if( pc->setupcalled > 0 ) {
    /* just do Galerkin grids */
    Mat B,dA,dB;
    
    /* PCSetUp_MG seems to insists on setting this to GMRES */
    ierr = KSPSetType( mglevels[0]->smoothd, KSPPREONLY ); CHKERRQ(ierr);
    
    if( pc_gamg->Nlevels > 1 ) {
      /* currently only handle case where mat and pmat are the same on coarser levels */
      ierr = KSPGetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,&dA,&dB,PETSC_NULL);CHKERRQ(ierr);
      /* (re)set to get dirty flag */
      ierr = KSPSetOperators(mglevels[pc_gamg->Nlevels-1]->smoothd,dA,dB,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSetUp( mglevels[pc_gamg->Nlevels-1]->smoothd ); CHKERRQ(ierr);
      
      for (level=pc_gamg->Nlevels-2; level>-1; level--) {
        ierr = KSPGetOperators(mglevels[level]->smoothd,PETSC_NULL,&B,PETSC_NULL);CHKERRQ(ierr);
        /* the first time through the matrix structure has changed from repartitioning */
        if( pc_gamg->setup_count == 2 ) {
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
    }

    pc->setupcalled = 2;

    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  /* GAMG requires input of fine-grid matrix. It determines nlevels. */
  ierr = MatGetBlockSize( Amat, &bs ); CHKERRQ(ierr);
  ierr = MatGetSize( Amat, &M, &N );CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; assert((Iend-Istart)%bs == 0);
  
  if( pc_gamg->data == 0 && nloc > 0 ) {
    if(!pc_gamg->createdefaultdata){
      SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_LIB,"GEO MG needs coordinates");
    }
    ierr = pc_gamg->createdefaultdata( pc ); CHKERRQ(ierr);
  }
  data = pc_gamg->data;
  
  /* Get A_i and R_i */
  ierr = MatGetInfo(Amat,MAT_GLOBAL_SUM,&info); CHKERRQ(ierr);
  if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s level %d N=%d, n data rows=%d, n data cols=%d, nnz/row (ave)=%d, np=%d\n",
	      mype,__FUNCT__,0,N,pc_gamg->data_rows,pc_gamg->data_cols,
	      (int)(info.nz_used/(PetscReal)N),npe);
  for ( level=0, Aarr[0] = Pmat, nactivepe = npe; /* hard wired stopping logic */
        level < (pc_gamg->Nlevels-1) && (level==0 || M>pc_gamg->coarse_eq_limit); /* && (npe==1 || nactivepe>1); */
        level++ ){
    level1 = level + 1;
#if (defined PETSC_USE_LOG && defined GAMG_STAGES)
    ierr = PetscLogStagePush(gamg_stages[level]); CHKERRQ( ierr );
#endif
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = pc_gamg->createprolongator( pc, Aarr[level], data, &Parr[level1], &coarse_data );
    CHKERRQ(ierr);
    /* get new block size of coarse matrices */    
    if( pc_gamg->col_bs_id != -1 && Parr[level1] ){
      PetscBool flag;
      ierr = PetscObjectComposedDataGetInt( (PetscObject)Parr[level1], pc_gamg->col_bs_id, bs, flag );
      CHKERRQ( ierr );
    }
    /* cache eigen estimate */
    if( pc_gamg->emax_id != -1 ){
      PetscBool flag;
      ierr = PetscObjectComposedDataGetReal( (PetscObject)Aarr[level], pc_gamg->emax_id, emaxs[level], flag );
      CHKERRQ( ierr );
      if( !flag ) emaxs[level] = -1.;
    }
    else emaxs[level] = -1.;

    ierr = PetscFree( data ); CHKERRQ( ierr );
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET1],0,0,0,0);CHKERRQ(ierr);
#endif
    if(level==0) Aarr[0] = Amat; /* use Pmat for finest level setup, but use mat for solver */
    if( Parr[level1] ) {
#if defined PETSC_USE_LOG
      ierr = PetscLogEventBegin(gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = createLevel( pc, Aarr[level], 
                          pc_gamg->data_rows, 
                          pc_gamg->data_cols, bs,
                          (PetscBool)(level==pc_gamg->Nlevels-2),
                          &Parr[level1], &coarse_data, &nactivepe, &Aarr[level1] );
      CHKERRQ(ierr);
#if defined PETSC_USE_LOG
      ierr = PetscLogEventEnd(gamg_setup_events[SET2],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = MatGetSize( Aarr[level1], &M, &N );CHKERRQ(ierr);
      ierr = MatGetInfo( Aarr[level1], MAT_GLOBAL_SUM, &info ); CHKERRQ(ierr);
      if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t\t[%d]%s %d) N=%d, n data cols=%d, nnz/row (ave)=%d, %d active pes\n",
		  mype,__FUNCT__,(int)level1,N,pc_gamg->data_cols,
		  (int)(info.nz_used/(PetscReal)N),nactivepe);
      /* coarse grids with SA can have zero row/cols from singleton aggregates */

      /* stop if one node */
      if( M/pc_gamg->data_cols < 2 ) {
        level++;
        break;
      }

      if (PETSC_FALSE) {
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
  if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s %d levels\n",0,__FUNCT__,level + 1);
  pc_gamg->data = 0; /* destroyed coordinate data */
  pc_gamg->Nlevels = level + 1;
  fine_level = level;
  ierr = PCMGSetLevels(pc,pc_gamg->Nlevels,PETSC_NULL);CHKERRQ(ierr);
  
  if( pc_gamg->Nlevels > 1 ) { /* don't setup MG if  */
  /* set default smoothers */
  for ( lidx = 1, level = pc_gamg->Nlevels-2;
        lidx <= fine_level;
        lidx++, level--) {
    PetscReal emax, emin;
    KSP smoother; PC subpc; 
    PetscBool isCheb;
    /* set defaults */
    ierr = PCMGGetSmoother( pc, lidx, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPCHEBYCHEV );CHKERRQ(ierr);
    ierr = KSPGetPC( smoother, &subpc ); CHKERRQ(ierr);
    /* ierr = KSPSetTolerances(smoother,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,2); CHKERRQ(ierr); */
    ierr = PCSetType( subpc, PCJACOBI ); CHKERRQ(ierr);
    ierr = KSPSetNormType( smoother, KSP_NORM_NONE ); CHKERRQ(ierr);
    /* overide defaults with input parameters */
    ierr = KSPSetFromOptions( smoother ); CHKERRQ(ierr);

    ierr = KSPSetOperators( smoother, Aarr[level], Aarr[level], SAME_NONZERO_PATTERN );   CHKERRQ(ierr);
    /* do my own cheby */
    ierr = PetscTypeCompare( (PetscObject)smoother, KSPCHEBYCHEV, &isCheb ); CHKERRQ(ierr);

    if( isCheb ) {
      ierr = PetscTypeCompare( (PetscObject)subpc, PCJACOBI, &isCheb ); CHKERRQ(ierr);
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

        if (pc_gamg->verbose) {
          PetscPrintf(PETSC_COMM_WORLD,"\t\t\t%s PC setup max eigen=%e min=%e\n",
                      __FUNCT__,emax,emin);
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
    Mat Lmat = Aarr[pc_gamg->Nlevels-1];
    ierr = PCMGGetSmoother( pc, 0, &smoother ); CHKERRQ(ierr);
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
  ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCSetFromOptions_MG(pc); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (mg->galerkin != 2) SETERRQ(wcomm,PETSC_ERR_USER,"GAMG does Galerkin manually so the -pc_mg_galerkin option must not be used.");

  /* set interpolation between the levels, clean up */  
  for (lidx=0,level=pc_gamg->Nlevels-1;
       lidx<fine_level;
       lidx++, level--){
    ierr = PCMGSetInterpolation( pc, lidx+1, Parr[level] );CHKERRQ(ierr);
    ierr = MatDestroy( &Parr[level] );  CHKERRQ(ierr);
    ierr = MatDestroy( &Aarr[level] );  CHKERRQ(ierr);
  }

  /* setupcalled is set to 0 so that MG is setup from scratch */
  pc->setupcalled = 0;
  ierr = PCSetUp_MG( pc );CHKERRQ( ierr );
  pc->setupcalled = 1; /* use 1 as signal that this has not been re-setup */
  
  {
    KSP smoother;  /* PCSetUp_MG seems to insists on setting this to GMRES on coarse grid */
    ierr = PCMGGetSmoother( pc, 0, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPPREONLY ); CHKERRQ(ierr);
    ierr = KSPSetUp( smoother ); CHKERRQ(ierr);
  }
  }
  else {
    KSP smoother;
    if (pc_gamg->verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s one level solver used (system is seen as DD). Using default solver.\n",mype,__FUNCT__);
    ierr = PCMGGetSmoother( pc, 0, &smoother ); CHKERRQ(ierr);
    ierr = KSPSetOperators( smoother, Aarr[0], Aarr[0], SAME_NONZERO_PATTERN );   CHKERRQ(ierr);
    ierr = KSPSetType( smoother, KSPPREONLY ); CHKERRQ(ierr);
    /* setupcalled is set to 0 so that MG is setup from scratch */
    pc->setupcalled = 0;
    ierr = PCSetUp_MG( pc );CHKERRQ( ierr );
    pc->setupcalled = 1; /* use 1 as signal that this has not been re-setup */
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
     PCCreate_GAMG - Geometric algebraic multigrid (AMG) preconditioning framework. 
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
  PetscClassId     cookie;

#if defined PETSC_USE_LOG
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
  pc_gamg->min_eq_proc = 100;
  pc_gamg->coarse_eq_limit = 1000;
  pc_gamg->threshold = 0.05;
  pc_gamg->Nlevels = GAMG_MAXLEVELS;
  pc_gamg->verbose = 0;
  pc_gamg->emax_id = -1;
  pc_gamg->col_bs_id = -1;

  /* instantiate derived type */
  ierr = PetscOptionsHead("GAMG options"); CHKERRQ(ierr);
  {
    char tname[256] = GAMGAGG;
    ierr = PetscOptionsList("-pc_gamg_type","Type of GAMG method","PCGAMGSetType",
                            GAMGList, tname, tname, sizeof(tname), PETSC_NULL );
    CHKERRQ(ierr);
    ierr = PCGAMGSetType( pc, tname ); CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);

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
    PetscLogEventRegister("  repartition", cookie, &gamg_setup_events[SET12]);
    PetscLogEventRegister("  Invert-Sort", cookie, &gamg_setup_events[SET13]);
    PetscLogEventRegister("  Move A", cookie, &gamg_setup_events[SET14]); 
    PetscLogEventRegister("  Move P", cookie, &gamg_setup_events[SET15]); 

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
