/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>

#if defined(PETSC_HAVE_TRIANGLE) 
#define REAL PetscReal
#include <triangle.h>
#endif

#include <assert.h>
#include <petscblaslapack.h>

/* typedef enum { NOT_DONE=-2, DELETED=-1, REMOVED=-3 } NState; */
/* use int instead of enum to facilitate passing them via Scatters */
typedef int NState;
static const NState NOT_DONE=-2;
static const NState DELETED=-1;
static const NState REMOVED=-3;

#define  IS_SELECTED(s) (s!=DELETED && s!=NOT_DONE && s!=REMOVED)

/* -------------------------------------------------------------------------- */
/*
   getDataWithGhosts - hacks into Mat MPIAIJ so this must have > 1 pe

   Input Parameter:
   . a_Gmat - MPIAIJ matrix for scattters
   . a_data_sz - number of data terms per node (# cols in output)
   . a_data_in[a_nloc*a_data_sz] - column oriented data
   Output Parameter:
   . a_stride - numbrt of rows of output
   . a_data_out[a_stride*a_data_sz] - output data with ghosts
*/
#undef __FUNCT__
#define __FUNCT__ "getDataWithGhosts"
PetscErrorCode getDataWithGhosts( const Mat a_Gmat,
                                  const PetscInt a_data_sz,
                                  const PetscReal a_data_in[],
                                  PetscInt *a_stride,
                                  PetscReal **a_data_out
                                  )
{
  PetscErrorCode ierr;
  PetscMPIInt    mype,npe;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat)->comm;
  Vec            tmp_crds;
  Mat_MPIAIJ    *mpimat = (Mat_MPIAIJ*)a_Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar   *data_arr;
  PetscReal     *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscTypeCompare( (PetscObject)a_Gmat, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  assert(isMPIAIJ);
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);                      assert(npe>1);
  ierr = MatGetOwnershipRange( a_Gmat, &my0, &Iend );    CHKERRQ(ierr);
  nloc = Iend - my0;
  ierr = VecGetLocalSize( mpimat->lvec, &num_ghosts );   CHKERRQ(ierr);
  nnodes = num_ghosts + nloc;
  *a_stride = nnodes;
  ierr = MatGetVecs( a_Gmat, &tmp_crds, 0 );    CHKERRQ(ierr);

  ierr = PetscMalloc( a_data_sz*nnodes*sizeof(PetscReal), &datas); CHKERRQ(ierr);
  for(dir=0; dir<a_data_sz; dir++) {
    /* set local, and global */
    for(kk=0; kk<nloc; kk++) {
      PetscInt gid = my0 + kk;
      PetscScalar crd = (PetscScalar)a_data_in[dir*nloc + kk]; /* col oriented */
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
   . a_Gmat_2 - glabal matrix of graph (data not defined)
   . a_Gmat_1
   . a_lid_state
   Input/Output Parameter:
   . a_id_llist - linked list rooted at selected node (size is nloc + nghosts_2)
   . a_deleted_parent_gid
*/
#undef __FUNCT__
#define __FUNCT__ "smoothAggs"
PetscErrorCode smoothAggs( const Mat a_Gmat_2, /* base (squared) graph */
                           const Mat a_Gmat_1, /* base graph */
                           const PetscScalar a_lid_state[], /* [nloc], states */
                           PetscInt a_id_llist[], /* [nloc+nghost_2], aggragate list */
                           PetscScalar a_deleted_parent_gid[] /* [nloc], which pe owns my deleted */

                           )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA_1, *matB_1=0;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat_2)->comm;
  PetscMPIInt    mype;
  PetscInt       nghosts_2,nghosts_1,lid,*ii,n,*idx,j,ix,Iend,my0;
  Mat_MPIAIJ    *mpimat_2 = 0, *mpimat_1=0;
  const PetscInt nloc = a_Gmat_2->rmap->n;
  PetscScalar   *cpcol_1_state;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Gmat_1,&my0,&Iend);  CHKERRQ(ierr);
  
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)a_Gmat_2, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
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
    mpimat_2 = (Mat_MPIAIJ*)a_Gmat_2->data;
    mpimat_1 = (Mat_MPIAIJ*)a_Gmat_1->data;
    matA_1 = (Mat_SeqAIJ*)mpimat_1->A->data;
    matB_1 = (Mat_SeqAIJ*)mpimat_1->B->data;
    /* get ghost sizes */
    ierr = VecGetLocalSize( mpimat_1->lvec, &nghosts_1 ); CHKERRQ(ierr);
    ierr = VecGetLocalSize( mpimat_2->lvec, &nghosts_2 ); CHKERRQ(ierr);
    /* get 'cpcol_1_state' */
    ierr = MatGetVecs( a_Gmat_1, &tempVec, 0 );         CHKERRQ(ierr);
    if(nloc>0){
      ierr = VecSetValues( tempVec, nloc, gids, a_lid_state, INSERT_VALUES );  CHKERRQ(ierr); 
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray( mpimat_1->lvec, &cpcol_1_state ); CHKERRQ(ierr);
    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr); 

    ierr = PetscFree( gids );  CHKERRQ(ierr);
    ierr = PetscFree( real_gids );  CHKERRQ(ierr);
  } else {
    PetscBool      isAIJ;
    ierr = PetscTypeCompare( (PetscObject)a_Gmat_2, MATSEQAIJ, &isAIJ ); CHKERRQ(ierr);
    assert(isAIJ);
    matA_1 = (Mat_SeqAIJ*)a_Gmat_1->data;
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
      NState state = (NState)PetscRealPart(a_lid_state[lid]);
      if( IS_SELECTED(state) ){
        PetscInt flid = lid;
        do{
          lid_sel_lid[flid] = lid; assert(flid<nloc); 
        } while( (flid=a_id_llist[flid]) != -1 );
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
      NState state = (NState)PetscRealPart(a_lid_state[lid]);
      if( IS_SELECTED(state) ) {
        /* steal locals */
        ii = matA_1->i; n = ii[lid+1] - ii[lid]; 
        idx = matA_1->j + ii[lid];
        for (j=0; j<n; j++) {
          PetscInt flid, lidj = idx[j];
          NState statej = (NState)PetscRealPart(a_lid_state[lidj]);
          if( statej==DELETED && lid_sel_lid[lidj] != lid ){ /* steal it */
	    if( lid_sel_lid[lidj] != -1 ){
	      /* I'm stealing this local */
	      PetscInt hav=0, flid2 = lid_sel_lid[lidj], lastid; assert(flid2!=-1);
	      for( lastid=flid2, flid=a_id_llist[flid2] ; flid!=-1 ; flid=a_id_llist[flid] ) {
		if( flid == lidj ) {
		  a_id_llist[lastid] = a_id_llist[lidj];                    /* remove lidj from list */
		  a_id_llist[lidj] = a_id_llist[lid]; a_id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
		  hav++;
		  /* break; */
		}
		lastid = flid;
	      }
	      assert(hav==1);
	    }
	    else{
	      /* I'm stealing this local, owned by a ghost */
	      a_deleted_parent_gid[lidj] = (PetscScalar)(lid+my0); /* this makes it a no-op later */
	      lid_sel_lid[lidj] = lid; /* not needed */
	      a_id_llist[lidj] = a_id_llist[lid]; a_id_llist[lid] = lidj; /* insert 'lidj' into head of llist */
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
	      if( old_sel_lid != -1 ){ 
		assert(a_deleted_parent_gid[lid]==-1.0); 
		for( lastid=old_sel_lid, flid=a_id_llist[old_sel_lid] ; flid != -1 ; flid=a_id_llist[flid] ) {
		  if( flid == lid ) {
		    a_id_llist[lastid] = a_id_llist[lid];   /* remove lid from 'old_sel_lid' list */
		    hv++;
		    /*break;*/
		  }
		  lastid = flid;
		}
		assert(hv==1);
	      }
	      else {
		assert(a_deleted_parent_gid[lid] != -1.0); /* stealing from one ghost, giving to another */
	      }

	      /* this will get other proc to add this */
              a_deleted_parent_gid[lid] = (PetscScalar)new_sel_gid; 
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

static const PetscMPIInt target = -2;
/* -------------------------------------------------------------------------- */
/*
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info.

   Input Parameter:
   . a_perm - serial permutation of rows of local to process in MIS
   . a_rank - serial array of rank (priority) for MIS, lower can not be deleted by higher
   . a_Gmat - glabal matrix of graph (data not defined)
   . a_Auxmat - non-squared matrix
   . a_strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "maxIndSetAgg"
PetscErrorCode maxIndSetAgg( const IS a_perm,
                             const IS a_ranks,
                             const Mat a_Gmat,
                             const Mat a_Auxmat,
			     const PetscBool a_strict_aggs,
                             IS *a_selected,
                             IS *a_locals_llist
                             )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA, *matB = 0;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat)->comm;
  Vec            locState, ghostState;
  PetscInt       num_fine_ghosts,kk,n,ix,j,*idx,*ii,iter,Iend,my0;
  Mat_MPIAIJ    *mpimat = 0;
  PetscScalar   *cpcol_gid,*cpcol_state;
  PetscMPIInt    mype;
  const PetscInt *perm_ix;
  PetscInt nDone = 0, nselected = 0;
  const PetscInt nloc = a_Gmat->rmap->n;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)a_Gmat, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)a_Gmat->data;
    matA = (Mat_SeqAIJ*)mpimat->A->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
  } else {
    PetscBool      isAIJ;
    ierr = PetscTypeCompare( (PetscObject)a_Gmat, MATSEQAIJ, &isAIJ ); CHKERRQ(ierr);
    assert(isAIJ);
    matA = (Mat_SeqAIJ*)a_Gmat->data;
  }
  assert( matA && !matA->compressedrow.use );
  assert( matB==0 || matB->compressedrow.use );
  /* get vector */
  ierr = MatGetVecs( a_Gmat, &locState, 0 );         CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(a_Gmat,&my0,&Iend);  CHKERRQ(ierr);

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
    ierr = ISGetIndices( a_perm, &perm_ix );     CHKERRQ(ierr);
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
            if( !a_strict_aggs ) {
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
		if( !a_strict_aggs ) {	
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
    ierr = ISRestoreIndices(a_perm,&perm_ix);     CHKERRQ(ierr);

    if( mpimat ){ /* free this buffer up (not really needed here) */
      ierr = VecRestoreArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr);
    }
    
    /* adjust aggregates */
    if( a_strict_aggs ) {
      ierr = smoothAggs(a_Gmat, a_Auxmat, lid_state, id_llist, deleted_parent_gid); 
      CHKERRQ(ierr);
    }

    /* tell adj who my deleted vertices belong to */
    if( a_strict_aggs && matB ) {
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

/* -------------------------------------------------------------------------- */
/*
 formProl0

   Input Parameter:
   . a_selected - list of selected local ID, includes selected ghosts
   . a_locals_llist - linked list with aggregates
   . a_bs - block size
   . a_nSAvec - num columns of new P
   . a_my0crs - global index of locals
   . a_data_stride - a_bs*(nloc nodes + ghost nodes)
   . a_data_in[a_data_stride*a_nSAvec] - local data on fine grid
   . a_flid_fgid[a_data_stride/a_bs] - make local to global IDs, includes ghosts in 'a_locals_llist'
  Output Parameter:
   . a_data_out - in with fine grid data (w/ghosts), out with coarse grid data
   . a_Prol - prolongation operator
*/
#undef __FUNCT__
#define __FUNCT__ "formProl0"
PetscErrorCode formProl0(IS a_selected, /* list of selected local ID, includes selected ghosts */
                         IS a_locals_llist, /* linked list from selected vertices of aggregate unselected vertices */
                         const PetscInt a_bs,
                         const PetscInt a_nSAvec,
                         const PetscInt a_my0crs,
			 const PetscInt a_data_stride,
			 PetscReal a_data_in[],
                         const PetscInt a_flid_fgid[],
                         PetscReal **a_data_out,
                         Mat a_Prol /* prolongation operator (output)*/
                         )
{
  PetscErrorCode ierr;
  PetscInt  Istart,Iend,nFineLoc,clid,flid,aggID,kk,jj,ii,nLocalSelected,ndone,nSelected;
  MPI_Comm       wcomm = ((PetscObject)a_Prol)->comm;
  PetscMPIInt    mype, npe;
  const PetscInt *selected_idx,*llist_idx;
  PetscReal      *out_data;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( a_Prol, &Istart, &Iend );    CHKERRQ(ierr);
  nFineLoc = (Iend-Istart)/a_bs; assert((Iend-Istart)%a_bs==0);
  
  ierr = ISGetLocalSize( a_selected, &nSelected );        CHKERRQ(ierr);
  ierr = ISGetIndices( a_selected, &selected_idx );       CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<nSelected;kk++){
    PetscInt lid = selected_idx[kk];
    if(lid<nFineLoc) nLocalSelected++;
  }

  /* aloc space for coarse point data (output) */
#define DATA_OUT_STRIDE (nLocalSelected*a_nSAvec)
  ierr = PetscMalloc( (DATA_OUT_STRIDE*a_nSAvec+1)*sizeof(PetscReal), &out_data ); CHKERRQ(ierr);
  for(ii=0;ii<DATA_OUT_STRIDE*a_nSAvec+1;ii++) out_data[ii]=1.e300;
  *a_data_out = out_data; /* output - stride nLocalSelected*a_nSAvec */

  /* find points and set prolongation */
  ndone = 0;
  ierr = ISGetIndices( a_locals_llist, &llist_idx );      CHKERRQ(ierr);
  for( clid = 0 ; clid < nLocalSelected ; clid++ ){
    PetscInt cgid = a_my0crs + clid, cids[100];

    /* count agg */
    aggID = 0;
    flid = selected_idx[clid]; assert(flid != -1);
    do{
      aggID++;
    } while( (flid=llist_idx[flid]) != -1 );

    /* get block */
    {
      PetscBLASInt   asz=aggID,M=asz*a_bs,N=a_nSAvec,INFO;
      PetscBLASInt   Mdata=M+((N-M>0)?N-M:0),LDA=Mdata,LWORK=N*a_bs;
      PetscScalar    *qqc,*qqr,*TAU,*WORK;
      PetscInt       *fids;
      
      ierr = PetscMalloc( (Mdata*N)*sizeof(PetscScalar), &qqc ); CHKERRQ(ierr); 
      ierr = PetscMalloc( (M*N)*sizeof(PetscScalar), &qqr ); CHKERRQ(ierr); 
      ierr = PetscMalloc( N*sizeof(PetscScalar), &TAU ); CHKERRQ(ierr); 
      ierr = PetscMalloc( LWORK*sizeof(PetscScalar), &WORK ); CHKERRQ(ierr); 
      ierr = PetscMalloc( M*sizeof(PetscInt), &fids ); CHKERRQ(ierr); 

      flid = selected_idx[clid];
      aggID = 0;
      do{
        /* copy in B_i matrix - column oriented */
        PetscReal *data = &a_data_in[flid*a_bs];
        for( kk = ii = 0; ii < a_bs ; ii++ ) {
          for( jj = 0; jj < N ; jj++ ) {
            qqc[jj*Mdata + aggID*a_bs + ii] = data[jj*a_data_stride + ii];
          }
        }

        /* set fine IDs */
        for(kk=0;kk<a_bs;kk++) fids[aggID*a_bs + kk] = a_flid_fgid[flid]*a_bs + kk;
        
        aggID++;
      }while( (flid=llist_idx[flid]) != -1 );

      /* pad with zeros */
      for( ii = asz*a_bs; ii < Mdata ; ii++ ) {
	for( jj = 0; jj < N ; jj++, kk++ ) {
	  qqc[jj*Mdata + ii] = .0;
	}
      }

      ndone += aggID;
      /* QR */
      LAPACKgeqrf_( &Mdata, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO );
      if( INFO != 0 ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRS error");
      /* get R - column oriented - output B_{i+1} */
      {
        PetscReal *data = &out_data[clid*a_nSAvec];
        for( jj = 0; jj < a_nSAvec ; jj++ ) {
          for( ii = 0; ii < a_nSAvec ; ii++ ) {
            assert(data[jj*DATA_OUT_STRIDE + ii] == 1.e300);
            if( ii <= jj ) data[jj*DATA_OUT_STRIDE + ii] = PetscRealPart(qqc[jj*Mdata + ii]);
	    else data[jj*DATA_OUT_STRIDE + ii] = 0.;
          }
        }
      }

      /* get Q - row oriented */
      LAPACKungqr_( &Mdata, &N, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO );
      if( INFO != 0 ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR error arg %d",-INFO);

      for( ii = 0 ; ii < M ; ii++ ){
        for( jj = 0 ; jj < N ; jj++ ) {
          qqr[N*ii + jj] = qqc[jj*Mdata + ii];
        }
      }

      /* add diagonal block of P0 */
      for(kk=0;kk<N;kk++) cids[kk] = N*cgid + kk; /* global col IDs in P0 */
      ierr = MatSetValues(a_Prol,M,fids,N,cids,qqr,INSERT_VALUES); CHKERRQ(ierr);

      ierr = PetscFree( qqc );  CHKERRQ(ierr);
      ierr = PetscFree( qqr );  CHKERRQ(ierr);
      ierr = PetscFree( TAU );  CHKERRQ(ierr);
      ierr = PetscFree( WORK );  CHKERRQ(ierr);
      ierr = PetscFree( fids );  CHKERRQ(ierr);
    } /* scoping */
  } /* for all coarse nodes */
  assert(out_data[a_nSAvec*DATA_OUT_STRIDE]==1.e300);

/* ierr = MPI_Allreduce( &ndone, &ii, 1, MPIU_INT, MPIU_SUM, wcomm ); /\* synchronous version *\/ */
/* MatGetSize( a_Prol, &kk, &jj ); */
/* PetscPrintf(PETSC_COMM_WORLD," **** [%d]%s %d total done, N=%d (%d local done)\n",mype,__FUNCT__,ii,kk/a_bs,ndone); */

  ierr = ISRestoreIndices( a_selected, &selected_idx );     CHKERRQ(ierr);
  ierr = ISRestoreIndices( a_locals_llist, &llist_idx );     CHKERRQ(ierr);
  ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 triangulateAndFormProl

   Input Parameter:
   . a_selected_2 - list of selected local ID, includes selected ghosts
   . a_nnodes -
   . a_coords[2*a_nnodes] - column vector of local coordinates w/ ghosts
   . a_selected_1 - selected IDs that go with base (1) graph
   . a_locals_llist - linked list with (some) locality info of base graph
   . a_crsGID[a_selected.size()] - global index for prolongation operator
   . a_bs - block size
  Output Parameter:
   . a_Prol - prolongation operator
   . a_worst_best - measure of worst missed fine vertex, 0 is no misses
*/
#undef __FUNCT__
#define __FUNCT__ "triangulateAndFormProl"
PetscErrorCode triangulateAndFormProl( IS  a_selected_2, /* list of selected local ID, includes selected ghosts */
				       const PetscInt a_nnodes,
                                       const PetscReal a_coords[], /* column vector of local coordinates w/ ghosts */
                                       IS  a_selected_1, /* list of selected local ID, includes selected ghosts */
                                       IS  a_locals_llist, /* linked list from selected vertices of aggregate unselected vertices */
				       const PetscInt a_crsGID[],
                                       const PetscInt a_bs,
                                       Mat a_Prol, /* prolongation operator (output) */
                                       PetscReal *a_worst_best /* measure of worst missed fine vertex, 0 is no misses */
                                       )
{
#if defined(PETSC_HAVE_TRIANGLE) 
  PetscErrorCode       ierr;
  PetscInt             jj,tid,tt,idx,nselected_1,nselected_2;
  struct triangulateio in,mid;
  const PetscInt      *selected_idx_1,*selected_idx_2,*llist_idx;
  PetscMPIInt          mype,npe;
  PetscInt             Istart,Iend,nFineLoc,myFine0;
  int kk,nPlotPts,sid;

  PetscFunctionBegin;
  *a_worst_best = 0.0;
  ierr = MPI_Comm_rank(((PetscObject)a_Prol)->comm,&mype);    CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)a_Prol)->comm,&npe);     CHKERRQ(ierr);
  ierr = ISGetLocalSize( a_selected_1, &nselected_1 );        CHKERRQ(ierr);
  ierr = ISGetLocalSize( a_selected_2, &nselected_2 );        CHKERRQ(ierr);
  if(nselected_2 == 1 || nselected_2 == 2 ){ /* 0 happens on idle processors */
    /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Not enough points - error in stopping logic",nselected_2); */
    *a_worst_best = 100.0; /* this will cause a stop, but not globalized (should not happen) */
#ifdef VERBOSE
    PetscPrintf(PETSC_COMM_SELF,"[%d]%s %d selected point - bailing out\n",mype,__FUNCT__,nselected_2);
#endif
    PetscFunctionReturn(0);
  }
  ierr = MatGetOwnershipRange( a_Prol, &Istart, &Iend );  CHKERRQ(ierr);
  nFineLoc = (Iend-Istart)/a_bs; myFine0 = Istart/a_bs;
  nPlotPts = nFineLoc; /* locals */
  /* traingle */
  /* Define input points - in*/
  in.numberofpoints = nselected_2;
  in.numberofpointattributes = 0;
  /* get nselected points */
  ierr = PetscMalloc( 2*(nselected_2)*sizeof(REAL), &in.pointlist ); CHKERRQ(ierr);
  ierr = ISGetIndices( a_selected_2, &selected_idx_2 );     CHKERRQ(ierr);

  for(kk=0,sid=0;kk<nselected_2;kk++,sid += 2){
    PetscInt lid = selected_idx_2[kk];
    in.pointlist[sid] = a_coords[lid];
    in.pointlist[sid+1] = a_coords[a_nnodes + lid];
    if(lid>=nFineLoc) nPlotPts++;
  }
  assert(sid==2*nselected_2);

  in.numberofsegments = 0;
  in.numberofedges = 0;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.trianglelist = 0;
  in.segmentmarkerlist = 0;
  in.pointattributelist = 0;
  in.pointmarkerlist = 0;
  in.triangleattributelist = 0;
  in.trianglearealist = 0;
  in.segmentlist = 0;
  in.holelist = 0;
  in.regionlist = 0;
  in.edgelist = 0;
  in.edgemarkerlist = 0;
  in.normlist = 0;
  /* triangulate */
  mid.pointlist = 0;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = 0;
  mid.pointmarkerlist = 0; /* Not needed if -N or -B switch used. */
  mid.trianglelist = 0;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = 0;
  mid.neighborlist = 0;         /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = 0;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = 0;
  mid.edgelist = 0;             /* Needed only if -e switch used. */
  mid.edgemarkerlist = 0;   /* Needed if -e used and -B not used. */
  mid.numberoftriangles = 0;

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                            */
  if(nselected_2 != 0){ /* inactive processor */
    char args[] = "npczQ"; /* c is needed ? */
    triangulate(args, &in, &mid, (struct triangulateio *) NULL );
    /* output .poly files for 'showme' */
    if( !PETSC_TRUE ) {
      static int level = 1;
      FILE *file; char fname[32];

      sprintf(fname,"C%d_%d.poly",level,mype); file = fopen(fname, "w");
      /*First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>*/
      fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++,sid += 2){
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
      }
      /*One line: <# of segments> <# of boundary markers (0 or 1)> */
      fprintf(file, "%d  %d\n",0,0);
      /*Following lines: <segment #> <endpoint> <endpoint> [boundary marker] */
      /* One line: <# of holes> */
      fprintf(file, "%d\n",0);
      /* Following lines: <hole #> <x> <y> */
      /* Optional line: <# of regional attributes and/or area constraints> */
      /* Optional following lines: <region #> <x> <y> <attribute> <maximum area> */
      fclose(file);

      /* elems */
      sprintf(fname,"C%d_%d.ele",level,mype); file = fopen(fname, "w");
      /* First line: <# of triangles> <nodes per triangle> <# of attributes> */
      fprintf(file, "%d %d %d\n",mid.numberoftriangles,3,0);
      /* Remaining lines: <triangle #> <node> <node> <node> ... [attributes] */
      for(kk=0,sid=0;kk<mid.numberoftriangles;kk++,sid += 3){
        fprintf(file, "%d %d %d %d\n",kk,mid.trianglelist[sid],mid.trianglelist[sid+1],mid.trianglelist[sid+2]);
      }
      fclose(file);

      sprintf(fname,"C%d_%d.node",level,mype); file = fopen(fname, "w");
      /* First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)> */
      /* fprintf(file, "%d  %d  %d  %d\n",in.numberofpoints,2,0,0); */
      fprintf(file, "%d  %d  %d  %d\n",nPlotPts,2,0,0);
      /*Following lines: <vertex #> <x> <y> */
      for(kk=0,sid=0;kk<in.numberofpoints;kk++,sid+=2){
        fprintf(file, "%d %e %e\n",kk,in.pointlist[sid],in.pointlist[sid+1]);
      }

      sid /= 2;
      for(jj=0;jj<nFineLoc;jj++){
        PetscBool sel = PETSC_TRUE;
        for( kk=0 ; kk<nselected_2 && sel ; kk++ ){
          PetscInt lid = selected_idx_2[kk];
          if( lid == jj ) sel = PETSC_FALSE;
        }
        if( sel ) {
          fprintf(file, "%d %e %e\n",sid++,a_coords[jj],a_coords[a_nnodes + jj]);
        }
      }
      fclose(file);
      assert(sid==nPlotPts);
      level++;
    }
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[FIND_V],0,0,0,0);CHKERRQ(ierr);
#endif
  { /* form P - setup some maps */
    PetscInt clid_iterator;
    PetscInt *nTri, *node_tri;
    
    ierr = PetscMalloc( nselected_2*sizeof(PetscInt), &node_tri ); CHKERRQ(ierr); 
    ierr = PetscMalloc( nselected_2*sizeof(PetscInt), &nTri ); CHKERRQ(ierr); 

    /* need list of triangles on node */
    for(kk=0;kk<nselected_2;kk++) nTri[kk] = 0;
    for(tid=0,kk=0;tid<mid.numberoftriangles;tid++){
      for(jj=0;jj<3;jj++) {
        PetscInt cid = mid.trianglelist[kk++];
        if( nTri[cid] == 0 ) node_tri[cid] = tid;
        nTri[cid]++;
      }
    } 
#define EPS 1.e-12
    /* find points and set prolongation */
    ierr = ISGetIndices( a_selected_1, &selected_idx_1 );     CHKERRQ(ierr); 
    ierr = ISGetIndices( a_locals_llist, &llist_idx );     CHKERRQ(ierr);
    for( clid_iterator = 0 ; clid_iterator < nselected_1 ; clid_iterator++ ){
      PetscInt flid = selected_idx_1[clid_iterator]; assert(flid != -1);
      PetscScalar AA[3][3];
      PetscBLASInt N=3,NRHS=1,LDA=3,IPIV[3],LDB=3,INFO;
      do{
        if( flid < nFineLoc ) {  /* could be a ghost */
          PetscInt bestTID = -1; PetscReal best_alpha = 1.e10; 
          const PetscInt fgid = flid + myFine0;
          /* compute shape function for gid */
          const PetscReal fcoord[3] = { a_coords[flid], a_coords[a_nnodes + flid], 1.0 };
          PetscBool haveit = PETSC_FALSE; PetscScalar alpha[3]; PetscInt clids[3];
          /* look for it */
          for( tid = node_tri[clid_iterator], jj=0;
               jj < 5 && !haveit && tid != -1;
               jj++ ){
            for(tt=0;tt<3;tt++){
              PetscInt cid2 = mid.trianglelist[3*tid + tt];
              PetscInt lid2 = selected_idx_2[cid2];
              AA[tt][0] = a_coords[lid2]; AA[tt][1] = a_coords[a_nnodes + lid2]; AA[tt][2] = 1.0;
              clids[tt] = cid2; /* store for interp */
            }

            for(tt=0;tt<3;tt++) alpha[tt] = (PetscScalar)fcoord[tt];

            /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
            LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
            {
              PetscBool have=PETSC_TRUE;  PetscReal lowest=1.e10;
              for( tt = 0, idx = 0 ; tt < 3 ; tt++ ) {
                if( PetscRealPart(alpha[tt]) > (1.0+EPS) || PetscRealPart(alpha[tt]) < -EPS ) have = PETSC_FALSE;
                if( PetscRealPart(alpha[tt]) < lowest ){
                  lowest = PetscRealPart(alpha[tt]);
                  idx = tt;
                }
              }
              haveit = have;
            }
            tid = mid.neighborlist[3*tid + idx];
          }

          if( !haveit ) {
            /* brute force */
            for(tid=0 ; tid<mid.numberoftriangles && !haveit ; tid++ ){
              for(tt=0;tt<3;tt++){
                PetscInt cid2 = mid.trianglelist[3*tid + tt];
                PetscInt lid2 = selected_idx_2[cid2];
                AA[tt][0] = a_coords[lid2]; AA[tt][1] = a_coords[a_nnodes + lid2]; AA[tt][2] = 1.0;
                clids[tt] = cid2; /* store for interp */
              }
              for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
              /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
              LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
              {
                PetscBool have=PETSC_TRUE;  PetscReal worst=0.0, v;
                for(tt=0; tt<3 && have ;tt++) {
                  if( PetscRealPart(alpha[tt]) > 1.0+EPS || PetscRealPart(alpha[tt]) < -EPS ) have=PETSC_FALSE;
                  if( (v=PetscAbs(PetscRealPart(alpha[tt])-0.5)) > worst ) worst = v;
                }
                if( worst < best_alpha ) {
                  best_alpha = worst; bestTID = tid;
                }
                haveit = have;
              }
            }
          }
          if( !haveit ) {
            if( best_alpha > *a_worst_best ) *a_worst_best = best_alpha;
            /* use best one */
            for(tt=0;tt<3;tt++){
              PetscInt cid2 = mid.trianglelist[3*bestTID + tt];
              PetscInt lid2 = selected_idx_2[cid2];
              AA[tt][0] = a_coords[lid2]; AA[tt][1] = a_coords[a_nnodes + lid2]; AA[tt][2] = 1.0;
              clids[tt] = cid2; /* store for interp */
            }
            for(tt=0;tt<3;tt++) alpha[tt] = fcoord[tt];
            /* SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
            LAPACKgesv_(&N, &NRHS, (PetscScalar*)AA, &LDA, IPIV, alpha, &LDB, &INFO);
          }

          /* put in row of P */
          for(idx=0;idx<3;idx++){
            PetscScalar shp = alpha[idx];
            if( PetscAbs(PetscRealPart(shp)) > 1.e-6 ) {
              PetscInt cgid = a_crsGID[clids[idx]];
              PetscInt jj = cgid*a_bs, ii = fgid*a_bs; /* need to gloalize */
              for(tt=0 ; tt < a_bs ; tt++, ii++, jj++ ){
                ierr = MatSetValues(a_Prol,1,&ii,1,&jj,&shp,INSERT_VALUES); CHKERRQ(ierr);
              }
            }
          }
        } /* local vertex test */
      } while( (flid=llist_idx[flid]) != -1 );
    }
    ierr = ISRestoreIndices( a_selected_2, &selected_idx_2 );     CHKERRQ(ierr);
    ierr = ISRestoreIndices( a_selected_1, &selected_idx_1 );     CHKERRQ(ierr);
    ierr = ISRestoreIndices( a_locals_llist, &llist_idx );     CHKERRQ(ierr);
    ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PetscFree( node_tri );  CHKERRQ(ierr);
    ierr = PetscFree( nTri );  CHKERRQ(ierr);
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[FIND_V],0,0,0,0);CHKERRQ(ierr);
#endif
  free( mid.trianglelist );
  free( mid.neighborlist );
  ierr = PetscFree( in.pointlist );  CHKERRQ(ierr);

  PetscFunctionReturn(0);
#else
    SETERRQ(((PetscObject)a_Prol)->comm,PETSC_ERR_LIB,"configure with TRIANGLE to use geometric MG");
#endif
}
/* -------------------------------------------------------------------------- */
/*
   getGIDsOnSquareGraph - square graph, get

   Input Parameter:
   . a_selected_1 - selected local indices (includes ghosts in input a_Gmat_1)
   . a_Gmat1 - graph that goes with 'a_selected_1'
   Output Parameter:
   . a_selected_2 - selected local indices (includes ghosts in output a_Gmat_2)
   . a_Gmat_2 - graph that is squared of 'a_Gmat_1'
   . a_crsGID[a_selected_2.size()] - map of global IDs of coarse grid nodes
*/
#undef __FUNCT__
#define __FUNCT__ "getGIDsOnSquareGraph"
PetscErrorCode getGIDsOnSquareGraph( const IS a_selected_1,
                                     const Mat a_Gmat1,
                                     IS *a_selected_2,
                                     Mat *a_Gmat_2,
                                     PetscInt **a_crsGID
                                     )
{
  PetscErrorCode ierr;
  PetscMPIInt    mype,npe;
  PetscInt       *crsGID, kk,my0,Iend,nloc,nSelected_1;
  const PetscInt *selected_idx;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat1)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Gmat1,&my0,&Iend); CHKERRQ(ierr); /* AIJ */
  nloc = Iend - my0; /* this does not change */
  ierr = ISGetLocalSize( a_selected_1, &nSelected_1 );        CHKERRQ(ierr);

  if (npe == 1) { /* not much to do in serial */
    ierr = PetscMalloc( nSelected_1*sizeof(PetscInt), &crsGID ); CHKERRQ(ierr);
    for(kk=0;kk<nSelected_1;kk++) crsGID[kk] = kk;
    *a_Gmat_2 = 0;
    *a_selected_2 = a_selected_1; /* needed? */
  }
  else {
    PetscInt      idx,num_fine_ghosts,num_crs_ghost,nLocalSelected,myCrs0;
    Mat_MPIAIJ   *mpimat2; 
    Mat           Gmat2;
    Vec           locState;
    PetscScalar   *cpcol_state;

    /* get 'nLocalSelected' */
    ierr = ISGetIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
    for(kk=0,nLocalSelected=0;kk<nSelected_1;kk++){
      PetscInt lid = selected_idx[kk];
      if(lid<nloc) nLocalSelected++;
    }
    ierr = ISRestoreIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
    /* scan my coarse zero gid, set 'lid_state' with coarse GID */
    MPI_Scan( &nLocalSelected, &myCrs0, 1, MPIU_INT, MPIU_SUM, wcomm );
    myCrs0 -= nLocalSelected;

    if( a_Gmat_2 != 0 ) { /* output */
      /* grow graph to get wider set of selected vertices to cover fine grid, invalidates 'llist' */
      if( npe==1 ){
        ierr = MatMatTransposeMult(a_Gmat1, a_Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );   CHKERRQ(ierr);
      }
      else {
        ierr = MatMatMult(a_Gmat1, a_Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );   CHKERRQ(ierr);
      }
      *a_Gmat_2 = Gmat2; /* output */
    }
    else Gmat2 = a_Gmat1;  /* use local to get crsGIDs at least */
    /* get coarse grid GIDS for selected (locals and ghosts) */
    mpimat2 = (Mat_MPIAIJ*)Gmat2->data;
    ierr = MatGetVecs( Gmat2, &locState, 0 );         CHKERRQ(ierr);
    ierr = VecSet( locState, (PetscScalar)(PetscReal)(NOT_DONE) );  CHKERRQ(ierr); /* set with UNKNOWN state */
    ierr = ISGetIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
    for(kk=0;kk<nLocalSelected;kk++){
      PetscInt fgid = selected_idx[kk] + my0;
      PetscScalar v = (PetscScalar)(kk+myCrs0);
      ierr = VecSetValues( locState, 1, &fgid, &v, INSERT_VALUES );  CHKERRQ(ierr); /* set with PID */
    }
    ierr = ISRestoreIndices( a_selected_1, &selected_idx );     CHKERRQ(ierr);
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat2->Mvctx,locState,mpimat2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetLocalSize( mpimat2->lvec, &num_fine_ghosts ); CHKERRQ(ierr);
    ierr = VecGetArray( mpimat2->lvec, &cpcol_state ); CHKERRQ(ierr); 
    for(kk=0,num_crs_ghost=0;kk<num_fine_ghosts;kk++){
      if( (NState)PetscRealPart(cpcol_state[kk]) != NOT_DONE ) num_crs_ghost++;
    }
    ierr = PetscMalloc( (nLocalSelected+num_crs_ghost)*sizeof(PetscInt), &crsGID ); CHKERRQ(ierr); /* output */
    {
      PetscInt *selected_set;
      ierr = PetscMalloc( (nLocalSelected+num_crs_ghost)*sizeof(PetscInt), &selected_set ); CHKERRQ(ierr);
      /* do ghost of 'crsGID' */
      for(kk=0,idx=nLocalSelected;kk<num_fine_ghosts;kk++){
        if( (NState)PetscRealPart(cpcol_state[kk]) != NOT_DONE ){
          PetscInt cgid = (PetscInt)PetscRealPart(cpcol_state[kk]);
          selected_set[idx] = nloc + kk;
          crsGID[idx++] = cgid;
        }
      }
      assert(idx==(nLocalSelected+num_crs_ghost));
      ierr = VecRestoreArray( mpimat2->lvec, &cpcol_state ); CHKERRQ(ierr);
      /* do locals in 'crsGID' */
      ierr = VecGetArray( locState, &cpcol_state ); CHKERRQ(ierr);
      for(kk=0,idx=0;kk<nloc;kk++){
        if( (NState)PetscRealPart(cpcol_state[kk]) != NOT_DONE ){
          PetscInt cgid = (PetscInt)PetscRealPart(cpcol_state[kk]);
          selected_set[idx] = kk;
          crsGID[idx++] = cgid;
        }
      }
      assert(idx==nLocalSelected);
      ierr = VecRestoreArray( locState, &cpcol_state ); CHKERRQ(ierr);

      if( a_selected_2 != 0 ) { /* output */
        ierr = ISCreateGeneral(PETSC_COMM_SELF,(nLocalSelected+num_crs_ghost),selected_set,PETSC_COPY_VALUES,a_selected_2);
        CHKERRQ(ierr);
      }
      ierr = PetscFree( selected_set );  CHKERRQ(ierr);
    }
    ierr = VecDestroy( &locState );                    CHKERRQ(ierr);
  }
  *a_crsGID = crsGID; /* output */

  PetscFunctionReturn(0);
}

/* Private context for the GAMG preconditioner */
typedef struct{
  PetscInt       m_lid;      /* local vertex index */
  PetscInt       m_degree;   /* vertex degree */
} GNode;
int compare (const void *a, const void *b)
{
  return (((GNode*)a)->m_degree - ((GNode*)b)->m_degree);
}

/* -------------------------------------------------------------------------- */
/*
   createProlongation

  Input Parameter:
   . a_Amat - matrix on this fine level
   . a_data[nloc*a_data_sz(in)]
   . a_dim - dimention
   . a_data_cols - number of colums in data (rows is infered from 
   . a_method - flag for: smoothed aggregation (2), plain agg (1) or geometric (0)
   . a_level - 0 for finest, +L-1 for coarsest
   . a_threshold - Relative threshold to use for dropping edges in aggregation graph
  Input/Output Parameter:
   . a_bs - block size of fine grid (in) and coarse grid (out)
  Output Parameter:
   . a_P_out - prolongation operator to the next level
   . a_data_out - data of coarse grid points (num local columns in 'a_P_out')
   . a_isOK - flag for if this grid is usable
   . a_emax - max iegen value
*/
#undef __FUNCT__
#define __FUNCT__ "createProlongation"
PetscErrorCode createProlongation( const Mat a_Amat,
                                   const PetscReal a_data[],
                                   const PetscInt a_dim,
                                   const PetscInt a_data_cols,
                                   const PetscInt a_method,
                                   const PetscInt a_level,
                                   const PetscReal a_threshold,
				   PetscInt *a_bs,
                                   Mat *a_P_out,
                                   PetscReal **a_data_out,
                                   PetscBool *a_isOK,
                                   PetscReal *a_emax
                                   )
{
  PetscErrorCode ierr;
  PetscInt       ncols,Istart,Iend,Ii,nloc,jj,kk,my0,nLocalSelected,nnz0,nnz1;
  Mat            Prol, Gmat, AuxMat;
  PetscMPIInt    mype, npe;
  MPI_Comm       wcomm = ((PetscObject)a_Amat)->comm;
  IS             rankIS, permIS, llist_1, selected_1, selected_2;
  const PetscInt *selected_idx, *idx,bs_in=*a_bs,col_bs=(a_method!=0 ? a_data_cols : bs_in);
  const PetscScalar *vals;
  PetscReal       vfilter;
  PetscInt       *d_nnz;

  PetscFunctionBegin;
  *a_isOK = PETSC_TRUE;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( a_Amat, &Istart, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs_in; my0 = Istart/bs_in; assert((Iend-Istart)%bs_in==0);

#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH_MAT],0,0,0,0);CHKERRQ(ierr);
#endif

  /* count nnz, there is sparcity in here so this might not be enough! */
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
  for ( Ii = Istart, nnz0 = jj = 0 ; Ii < Iend ; Ii += bs_in, jj++ ) {
    ierr = MatGetRow(a_Amat,Ii,&ncols,0,0); CHKERRQ(ierr);
    d_nnz[jj] = ncols;
    if( d_nnz[jj] > nloc ) d_nnz[jj] = nloc; 
    nnz0 += d_nnz[jj];
    ierr = MatRestoreRow(a_Amat,Ii,&ncols,0,0); CHKERRQ(ierr);    
  }
  nnz0 /= (nloc+1);

  /* get scalar copy (norms) of matrix */
  ierr = MatCreateMPIAIJ( wcomm, nloc, nloc,
                          PETSC_DETERMINE, PETSC_DETERMINE,
                          0, d_nnz, 0, d_nnz, &Gmat );

  for( Ii = Istart; Ii < Iend ; Ii++ ) {
    PetscInt dest_row = Ii/bs_in;
    ierr = MatGetRow(a_Amat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
    for(jj=0;jj<ncols;jj++){
      PetscInt dest_col = idx[jj]/bs_in;
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      ierr = MatSetValues(Gmat,1,&dest_row,1,&dest_col,&sv,ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(a_Amat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
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

  /* filter Gmat */
  vfilter = a_threshold;
  /* for(jj=0;jj<a_level;jj++) vfilter *= .01; very fast decay for SA */

  ierr = MatGetOwnershipRange(Gmat,&Istart,&Iend);CHKERRQ(ierr); /* use AIJ from here */
  {
    Mat Gmat2; 
    ierr = MatCreateMPIAIJ(wcomm,nloc,nloc,PETSC_DECIDE,PETSC_DECIDE,0,d_nnz,0,d_nnz,&Gmat2);
    CHKERRQ(ierr);
    for( Ii = Istart, nnz1 = 0 ; Ii < Iend; Ii++ ){
      ierr = MatGetRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
      for(jj=0;jj<ncols;jj++){
        PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
        if( PetscRealPart(sv) > vfilter ) {
          ierr = MatSetValues(Gmat2,1,&Ii,1,&idx[jj],&sv,INSERT_VALUES); CHKERRQ(ierr);
	  nnz1++;
        }
        /* else PetscPrintf(PETSC_COMM_SELF,"\t%s filtered %d, v=%e\n",__FUNCT__,Ii,vals[jj]); */
      }
      ierr = MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
    }
    nnz1 /= (nloc+1);
    ierr = MatAssemblyBegin(Gmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Gmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
    Gmat = Gmat2;
#ifdef VERBOSE 
    PetscPrintf(PETSC_COMM_WORLD,"\t%s ave nnz/row %d --> %d\n",__FUNCT__,nnz0,nnz1); 
#endif
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH_FILTER],0,0,0,0);   CHKERRQ(ierr);
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH_SQR],0,0,0,0);CHKERRQ(ierr);
#endif
  /* square matrix - SA */  
  if( a_method != 0 ){
    Mat Gmat2;
    if( npe==1 ){
      ierr = MatMatTransposeMult( Gmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );
    }
    else {
      ierr = MatMatMult( Gmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );
    }
    CHKERRQ(ierr);
    AuxMat = Gmat;
    Gmat = Gmat2;
    /* force compressed row storage for B matrix in AuxMat */
    if (npe > 1) {
      Mat_MPIAIJ *mpimat = (Mat_MPIAIJ*)AuxMat->data;
      Mat_SeqAIJ *Bmat = (Mat_SeqAIJ*) mpimat->B->data;
      Bmat->compressedrow.check = PETSC_TRUE;
      ierr = MatCheckCompressedRow(mpimat->B,&Bmat->compressedrow,Bmat->i,AuxMat->rmap->n,-1.0);
      CHKERRQ(ierr);
      assert( Bmat->compressedrow.use );
    }
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH_SQR],0,0,0,0);   CHKERRQ(ierr);
#endif
  /* force compressed row storage for B matrix */
  if (npe > 1) {
    Mat_MPIAIJ *mpimat = (Mat_MPIAIJ*)Gmat->data;
    Mat_SeqAIJ *Bmat = (Mat_SeqAIJ*) mpimat->B->data;
    Bmat->compressedrow.check = PETSC_TRUE;
    ierr = MatCheckCompressedRow(mpimat->B,&Bmat->compressedrow,Bmat->i,Gmat->rmap->n,-1.0);
    CHKERRQ(ierr);
    assert( Bmat->compressedrow.use );
  }

  /* view */
  if( PETSC_FALSE ) {
    PetscViewer        viewer;
    ierr = PetscViewerASCIIOpen(wcomm, "Gmat_2.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
    ierr = MatView(Gmat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
    
    ierr = PetscViewerASCIIOpen(wcomm, "Gmat_1.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
    ierr = MatView(AuxMat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  /* Mat subMat = Gmat -- get degree of vertices */
  {
    GNode *gnodes;
    PetscInt *permute,*ranks;
    
    ierr = PetscMalloc( nloc*sizeof(GNode), &gnodes ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &permute ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &ranks ); CHKERRQ(ierr);

    for (Ii=Istart; Ii<Iend; Ii++) { /* locals only? */
      ierr = MatGetRow(Gmat,Ii,&ncols,0,0); CHKERRQ(ierr);
      {
        PetscInt lid = Ii - Istart;
        gnodes[lid].m_lid = lid;
        gnodes[lid].m_degree = ncols;
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
          GNode iTemp = gnodes[iSwapIndex];
          gnodes[iSwapIndex] = gnodes[Ii];
          gnodes[Ii] = iTemp;
          bIndexSet[Ii] = PETSC_TRUE;
          bIndexSet[iSwapIndex] = PETSC_TRUE;
        }
      }
      ierr = PetscFree( bIndexSet );  CHKERRQ(ierr);
    }
    /* only sort locals */
    qsort( gnodes, nloc, sizeof(GNode), compare );
    /* create IS of permutation */
    for(kk=0;kk<nloc;kk++) { /* locals only */
      permute[kk] = gnodes[kk].m_lid;
      ranks[kk] = gnodes[kk].m_degree;
    }
    ierr = ISCreateGeneral( PETSC_COMM_SELF, (Iend-Istart), permute, PETSC_COPY_VALUES, &permIS );
    CHKERRQ(ierr);
    ierr = ISCreateGeneral( PETSC_COMM_SELF, (Iend-Istart), ranks, PETSC_COPY_VALUES, &rankIS );
    CHKERRQ(ierr);

    ierr = PetscFree( gnodes );  CHKERRQ(ierr);
    ierr = PetscFree( permute );  CHKERRQ(ierr);
    ierr = PetscFree( ranks );  CHKERRQ(ierr);
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH],0,0,0,0);   CHKERRQ(ierr);
#endif
  /* SELECT COARSE POINTS */
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif

  ierr = maxIndSetAgg( permIS, rankIS, Gmat, AuxMat, (PetscBool)(a_method != 0), &selected_1, &llist_1 );
  CHKERRQ(ierr);
  if( a_method != 0 ) {
    ierr = MatDestroy( &AuxMat );  CHKERRQ(ierr); 
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = ISDestroy(&permIS); CHKERRQ(ierr);
  ierr = ISDestroy(&rankIS); CHKERRQ(ierr);

  /* get 'nLocalSelected' */
  ierr = ISGetLocalSize( selected_1, &ncols );        CHKERRQ(ierr);
  ierr = ISGetIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<ncols;kk++){
    PetscInt lid = selected_idx[kk];
    if(lid<nloc) nLocalSelected++;
  }
  ierr = ISRestoreIndices( selected_1, &selected_idx );     CHKERRQ(ierr);

  /* create prolongator, create P matrix */
  ierr = MatCreateMPIAIJ(wcomm, 
                         nloc*bs_in, nLocalSelected*col_bs,
                         PETSC_DETERMINE, PETSC_DETERMINE,
                         a_data_cols, PETSC_NULL, a_data_cols, PETSC_NULL,
                         &Prol );
  CHKERRQ(ierr);
  
  /* can get all points "removed" */
  ierr =  MatGetSize( Prol, &kk, &jj ); CHKERRQ(ierr);
  if( jj==0 ) {
    assert(0);
    *a_isOK = PETSC_FALSE;
#ifdef VERBOSE
    PetscPrintf(PETSC_COMM_WORLD,"[%d]%s no selected points on coarse grid\n",mype,__FUNCT__);
#endif
    ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
    ierr = ISDestroy( &llist_1 ); CHKERRQ(ierr);
    ierr = ISDestroy( &selected_1 ); CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);    
    ierr = PetscFree( d_nnz ); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* switch for SA or GAMG */
  if( a_method == 0 ) {
    PetscReal *coords; 
    PetscInt nnodes;
    PetscInt  *crsGID;
    Mat        Gmat2;

    assert(a_dim==a_data_cols); 
    /* grow ghost data for better coarse grid cover of fine grid */
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET5],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = getGIDsOnSquareGraph( selected_1, Gmat, &selected_2, &Gmat2, &crsGID );
    CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
    /* llist is now not valid wrt squared graph, but will work as iterator in 'triangulateAndFormProl' */
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET5],0,0,0,0);CHKERRQ(ierr);
#endif
    /* create global vector of coorindates in 'coords' */
    if (npe > 1) {
      ierr = getDataWithGhosts( Gmat2, a_dim, a_data, &nnodes, &coords );
      CHKERRQ(ierr);
    }
    else {
      coords = (PetscReal*)a_data;
      nnodes = nloc;
    }
    ierr = MatDestroy( &Gmat2 );  CHKERRQ(ierr);

    /* triangulate */
    if( a_dim == 2 ) {
      PetscReal metric=0.0;
#if defined PETSC_USE_LOG
      ierr = PetscLogEventBegin(gamg_setup_events[SET6],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = triangulateAndFormProl( selected_2, nnodes, coords,
                                     selected_1, llist_1, crsGID, bs_in, Prol, &metric );
      CHKERRQ(ierr);
#if defined PETSC_USE_LOG
      ierr = PetscLogEventEnd(gamg_setup_events[SET6],0,0,0,0); CHKERRQ(ierr);
#endif
      ierr = PetscFree( crsGID );  CHKERRQ(ierr);

      /* clean up and create coordinates for coarse grid (output) */
      if (npe > 1) ierr = PetscFree( coords ); CHKERRQ(ierr);
      
      if( metric > 1. ) { /* needs to be globalized - should not happen */
        *a_isOK = PETSC_FALSE;
#ifdef VERBOSE
        PetscPrintf(PETSC_COMM_SELF,"[%d]%s failed metric for coarse grid %e\n",mype,__FUNCT__,metric);
#endif
        ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
      }
      else if( metric > .0 ) {
#ifdef VERBOSE
        PetscPrintf(PETSC_COMM_SELF,"[%d]%s metric for coarse grid = %e\n",mype,__FUNCT__,metric);
#endif
      }
    } else {
      SETERRQ(wcomm,PETSC_ERR_LIB,"3D not implemented");
    }
    { /* create next coords - output */
      PetscReal *crs_crds;
      ierr = PetscMalloc( a_dim*nLocalSelected*sizeof(PetscReal), &crs_crds ); 
      CHKERRQ(ierr);
      ierr = ISGetIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
      for(kk=0;kk<nLocalSelected;kk++){/* grab local select nodes to promote - output */
        PetscInt lid = selected_idx[kk];
        for(jj=0;jj<a_dim;jj++) crs_crds[jj*nLocalSelected + kk] = a_data[jj*nloc + lid];
      }
      ierr = ISRestoreIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
      *a_data_out = crs_crds; /* out */
    }
    if (npe > 1) {
      ierr = ISDestroy( &selected_2 ); CHKERRQ(ierr); /* this is selected_1 in serial */
    }
    *a_emax = -1.0; /* no estimate */
  }
  else { /* SA */
    PetscReal alpha,emax,emin,*data_w_ghost;
    PetscInt  myCrs0, nbnodes=0, *flid_fgid;

    /* create global vector of data in 'data_w_ghost' */
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
#endif
    if (npe > 1) {
      PetscReal *tmp_gdata,*tmp_ldata,*tp2;

      ierr = PetscMalloc( nloc*sizeof(PetscReal), &tmp_ldata ); CHKERRQ(ierr);
      for( jj = 0 ; jj < a_data_cols ; jj++ ){
        for( kk = 0 ; kk < bs_in ; kk++) {
          PetscInt ii,nnodes;
	  const PetscReal *tp = a_data + jj*bs_in*nloc + kk;
	  for( ii = 0 ; ii < nloc ; ii++, tp += bs_in ){
	    tmp_ldata[ii] = *tp;
	  }
	  ierr = getDataWithGhosts( Gmat, 1, tmp_ldata, &nnodes, &tmp_gdata );
	  CHKERRQ(ierr);
	  if(jj==0 && kk==0) { /* now I know how many todal nodes - allocate */
	    ierr = PetscMalloc( nnodes*bs_in*a_data_cols*sizeof(PetscReal), &data_w_ghost ); CHKERRQ(ierr);
	    nbnodes = bs_in*nnodes;
	  }
	  tp2 = data_w_ghost + jj*bs_in*nnodes + kk;
	  for( ii = 0 ; ii < nnodes ; ii++, tp2 += bs_in ){
	    *tp2 = tmp_gdata[ii];
	  }
	  ierr = PetscFree( tmp_gdata ); CHKERRQ(ierr);
	}
      }
      ierr = PetscFree( tmp_ldata ); CHKERRQ(ierr);
    }
    else {
      nbnodes = bs_in*nloc;
      data_w_ghost = (PetscReal*)a_data;
    }
    
    /* scan my coarse zero gid */
    MPI_Scan( &nLocalSelected, &myCrs0, 1, MPIU_INT, MPIU_SUM, wcomm );
    myCrs0 -= nLocalSelected;

    /* get P0 */
    if( npe > 1 ){
      PetscReal *fid_glid_loc,*fiddata; 
      PetscInt nnodes;

      ierr = PetscMalloc( nloc*sizeof(PetscReal), &fid_glid_loc ); CHKERRQ(ierr);
      for(kk=0;kk<nloc;kk++) fid_glid_loc[kk] = (PetscReal)(my0+kk);
      ierr = getDataWithGhosts(Gmat, 1, fid_glid_loc, &nnodes, &fiddata);
      CHKERRQ(ierr);
      ierr = PetscMalloc( nnodes*sizeof(PetscInt), &flid_fgid ); CHKERRQ(ierr);
      for(kk=0;kk<nnodes;kk++) flid_fgid[kk] = (PetscInt)fiddata[kk];
      ierr = PetscFree( fiddata ); CHKERRQ(ierr);
      assert(nnodes==nbnodes/bs_in);
      ierr = PetscFree( fid_glid_loc ); CHKERRQ(ierr);
    }
    else {
      ierr = PetscMalloc( nloc*sizeof(PetscInt), &flid_fgid ); CHKERRQ(ierr);
      for(kk=0;kk<nloc;kk++) flid_fgid[kk] = my0 + kk;
    }
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);
#if defined PETSC_USE_LOG
    ierr = PetscLogEventEnd(gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
#endif

    ierr = formProl0(selected_1,llist_1,bs_in,a_data_cols,myCrs0,nbnodes,
		     data_w_ghost,flid_fgid,a_data_out,Prol);
    CHKERRQ(ierr);

    if (npe > 1) ierr = PetscFree( data_w_ghost );      CHKERRQ(ierr);
    ierr = PetscFree( flid_fgid ); CHKERRQ(ierr);
 
    /* smooth P0 */
    if( a_method == 2 ) {
      Mat tMat; 
      Vec diag;    
      KSP eksp; 
      Vec bb, xx; 
      PC pc;

#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif 
      ierr = MatGetVecs( a_Amat, &bb, 0 );         CHKERRQ(ierr);
      ierr = MatGetVecs( a_Amat, &xx, 0 );         CHKERRQ(ierr);
      {
	PetscRandom    rctx;
	ierr = PetscRandomCreate(wcomm,&rctx);CHKERRQ(ierr);
	ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
	ierr = VecSetRandom(bb,rctx);CHKERRQ(ierr);
	ierr = PetscRandomDestroy( &rctx ); CHKERRQ(ierr);
      }  
      ierr = KSPCreate(wcomm,&eksp);                            CHKERRQ(ierr);
      ierr = KSPSetType( eksp, KSPCG );                         CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero( eksp, PETSC_FALSE );    CHKERRQ(ierr);
      ierr = KSPSetOperators( eksp, a_Amat, a_Amat, SAME_NONZERO_PATTERN );
      CHKERRQ( ierr );
      ierr = KSPGetPC( eksp, &pc );                              CHKERRQ( ierr );
      ierr = PCSetType( pc, PETSC_GAMG_SMOOTHER ); CHKERRQ(ierr);  /* smoother */
      ierr = KSPSetTolerances(eksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10);
      CHKERRQ(ierr);
      ierr = KSPSetNormType( eksp, KSP_NORM_NONE );                 CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues( eksp,PETSC_TRUE );        CHKERRQ(ierr);
      ierr = KSPSolve( eksp, bb, xx );                              CHKERRQ(ierr);
      ierr = KSPComputeExtremeSingularValues( eksp, &emax, &emin ); CHKERRQ(ierr);
#ifdef VERBOSE
      PetscPrintf(PETSC_COMM_WORLD,"\t\t\t%s smooth P0: max eigen=%e min=%e PC=%s\n",__FUNCT__,emax,emin,PETSC_GAMG_SMOOTHER);
#endif
      ierr = VecDestroy( &xx );       CHKERRQ(ierr); 
      ierr = VecDestroy( &bb );       CHKERRQ(ierr);
      ierr = KSPDestroy( &eksp );     CHKERRQ(ierr);

      /* smooth P1 := (I - omega/lam D^{-1}A)P0 */
      ierr = MatMatMult( a_Amat, Prol, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tMat );   CHKERRQ(ierr);
      ierr = MatGetVecs( a_Amat, &diag, 0 );    CHKERRQ(ierr);
      ierr = MatGetDiagonal( a_Amat, diag );    CHKERRQ(ierr); /* effectively PCJACOBI */
      ierr = VecReciprocal( diag );         CHKERRQ(ierr);
      ierr = MatDiagonalScale( tMat, diag, 0 ); CHKERRQ(ierr);
      ierr = VecDestroy( &diag );           CHKERRQ(ierr);
      alpha = -1.5/emax;
      ierr = MatAYPX( tMat, alpha, Prol, SUBSET_NONZERO_PATTERN );           CHKERRQ(ierr);
      ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
      Prol = tMat;
#if defined PETSC_USE_LOG
      ierr = PetscLogEventEnd(gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif
      *a_emax = emax; /* estimate for cheb smoother */
    }
    else {
      *a_emax = -1.0; /* no estimate */
    }
    *a_bs = a_data_cols;
  } /* aggregation method */
  *a_P_out = Prol;  /* out */

  ierr = ISDestroy( &llist_1 ); CHKERRQ(ierr);
  ierr = ISDestroy( &selected_1 ); CHKERRQ(ierr);
  ierr = PetscFree( d_nnz ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
