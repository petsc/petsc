/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include "private/matimpl.h"
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <private/kspimpl.h>

/* -------------------------------------------------------------------------- */
/*
   createSimpleGraph - create simple scalar graph
 
 Input Parameter:
 . Amat - matrix
 Output Parameter:
 . a_Gmaat - output scalar graph (symmetric?)
 */
#undef __FUNCT__
#define __FUNCT__ "createSimpleGraph"
PetscErrorCode createSimpleGraph( const Mat Amat, Mat *a_Gmat )
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,jj,ncols,nloc,NN,MM,bs;
  PetscMPIInt    mype, npe;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  Mat            Gmat;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  ierr = MatGetSize( Amat, &MM, &NN ); CHKERRQ(ierr);
  ierr = MatGetBlockSize( Amat, &bs ); CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; 
 
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif
  if( bs > 1 ) {
    const PetscScalar *vals;
    const PetscInt *idx;
    PetscInt       *d_nnz, *o_nnz;
    /* count nnz, there is sparcity in here so this might not be enough */
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &o_nnz ); CHKERRQ(ierr);
    for ( Ii = Istart, jj = 0 ; Ii < Iend ; Ii += bs, jj++ ) {
      ierr = MatGetRow(Amat,Ii,&ncols,0,0); CHKERRQ(ierr);
      d_nnz[jj] = ncols; /* very pessimistic */
      o_nnz[jj] = ncols;
      if( d_nnz[jj] > nloc ) d_nnz[jj] = nloc;
      if( o_nnz[jj] > (NN/bs-nloc) ) o_nnz[jj] = NN/bs-nloc;
      ierr = MatRestoreRow(Amat,Ii,&ncols,0,0); CHKERRQ(ierr);
    }

    /* get scalar copy (norms) of matrix -- AIJ specific!!! */
    ierr = MatCreateMPIAIJ( wcomm, nloc, nloc,
                            PETSC_DETERMINE, PETSC_DETERMINE,
                            0, d_nnz, 0, o_nnz, &Gmat );

    ierr = PetscFree( d_nnz ); CHKERRQ(ierr);
    ierr = PetscFree( o_nnz ); CHKERRQ(ierr);

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
  }
  else {
    /* just copy scalar matrix - abs() not taken here but scaled later */
    ierr = MatDuplicate( Amat, MAT_COPY_VALUES, &Gmat ); CHKERRQ(ierr);
  }

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif

  *a_Gmat = Gmat;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   scaleFilterGraph
 
 Input Parameter:
 . vfilter - threshold paramter [0,1)
 . symm - symetrize?
 In/Output Parameter:
 . a_Gmat - original graph
 */
#undef __FUNCT__
#define __FUNCT__ "scaleFilterGraph"
PetscErrorCode scaleFilterGraph( Mat *a_Gmat, const PetscReal vfilter, const PetscBool symm, const PetscInt verbose )
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,jj,ncols,nnz0,nnz1, NN, MM, nloc;
  PetscMPIInt    mype, npe;
  Mat            Gmat = *a_Gmat, tGmat;
  MPI_Comm       wcomm = ((PetscObject)Gmat)->comm;
  const PetscScalar *vals;
  const PetscInt *idx;
  PetscInt *d_nnz, *o_nnz;
  Vec diag;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Gmat, &Istart, &Iend ); CHKERRQ(ierr);
  nloc = Iend - Istart;
  ierr = MatGetSize( Gmat, &MM, &NN ); CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif
  /* scale Gmat so filter works */
  ierr = MatGetVecs( Gmat, &diag, 0 );    CHKERRQ(ierr);
  ierr = MatGetDiagonal( Gmat, diag );    CHKERRQ(ierr);
  ierr = VecReciprocal( diag );           CHKERRQ(ierr);
  ierr = VecSqrtAbs( diag );              CHKERRQ(ierr);
  ierr = MatDiagonalScale( Gmat, diag, diag ); CHKERRQ(ierr);
  ierr = VecDestroy( &diag );           CHKERRQ(ierr);

  /* filter - dup zeros out matrix */
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &o_nnz ); CHKERRQ(ierr);
  for( Ii = Istart, jj = 0 ; Ii < Iend; Ii++, jj++ ){
    ierr = MatGetRow(Gmat,Ii,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    d_nnz[jj] = ncols;
    o_nnz[jj] = ncols;
    ierr = MatRestoreRow(Gmat,Ii,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    if( d_nnz[jj] > nloc ) d_nnz[jj] = nloc;
    if( o_nnz[jj] > (MM-nloc) ) o_nnz[jj] = MM - nloc;
  }
  ierr = MatCreateMPIAIJ( wcomm, nloc, nloc, MM, MM, 0, d_nnz, 0, o_nnz, &tGmat );
  CHKERRQ(ierr);
  ierr = PetscFree( d_nnz ); CHKERRQ(ierr); 
  ierr = PetscFree( o_nnz ); CHKERRQ(ierr); 

  for( Ii = Istart, nnz0 = nnz1 = 0 ; Ii < Iend; Ii++ ){
    ierr = MatGetRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
    for(jj=0;jj<ncols;jj++,nnz0++){
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      if( PetscRealPart(sv) > vfilter ) {
        nnz1++;
        if( symm ) {
          sv *= 0.5;
          ierr = MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,ADD_VALUES); CHKERRQ(ierr);
          ierr = MatSetValues(tGmat,1,&idx[jj],1,&Ii,&sv,ADD_VALUES);  CHKERRQ(ierr);
        }
        else {
          ierr = MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,ADD_VALUES); CHKERRQ(ierr);
        }        
      }
    }
    ierr = MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(tGmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tGmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[GRAPH],0,0,0,0);   CHKERRQ(ierr);
#endif

  if( verbose ) {
    if( verbose == 1 ) {
      PetscPrintf(wcomm,"\t[%d]%s %g%% nnz after filtering, with threshold %g, %g nnz ave.\n",mype,__FUNCT__,
                  100.*(double)nnz1/(double)nnz0,vfilter,(double)nnz0/(double)nloc);
    }
    else {
      PetscInt nnz[2] = {nnz0,nnz1},out[2];
      MPI_Allreduce( nnz, out, 2, MPIU_INT, MPI_SUM, wcomm );
      PetscPrintf(wcomm,"\t[%d]%s %g%% nnz after filtering, with threshold %g, %g nnz ave.\n",mype,__FUNCT__,
                  100.*(double)out[1]/(double)out[0],vfilter,(double)out[0]/(double)MM);
    }
  }
  
  ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);

  *a_Gmat = tGmat;

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
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . perm - serial permutation of rows of local to process in MIS
   . Gmat - glabal matrix of graph (data not defined)
   . strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';
   . verbose - 
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "maxIndSetAgg"
PetscErrorCode maxIndSetAgg( const IS perm,
                             const Mat Gmat,
			     const PetscBool strict_aggs,
                             const PetscInt verbose, 
                             IS *a_selected,
                             IS *a_locals_llist
                             )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA, *matB = 0;
  MPI_Comm       wcomm = ((PetscObject)Gmat)->comm;
  Vec            locState, ghostState;
  PetscInt       num_fine_ghosts,kk,n,ix,j,*idx,*ii,iter,Iend,my0,nremoved;
  Mat_MPIAIJ    *mpimat = 0;
  PetscScalar   *cpcol_gid,*cpcol_state;
  PetscMPIInt    mype;
  const PetscInt *perm_ix;
  PetscInt nDone = 0, nselected = 0;
  const PetscInt nloc = Gmat->rmap->n;
  PetscInt *lid_cprowID, *lid_gid;
  PetscScalar *deleted_parent_gid; /* only used for strict aggs */
  PetscInt *id_llist; /* linked list with locality info - output */
  PetscScalar *lid_state;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)Gmat, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)Gmat->data;
    matA = (Mat_SeqAIJ*)mpimat->A->data;
    matB = (Mat_SeqAIJ*)mpimat->B->data;
    /* force compressed storage of B */
    matB->compressedrow.check = PETSC_TRUE;
    ierr = MatCheckCompressedRow(mpimat->B,&matB->compressedrow,matB->i,Gmat->rmap->n,-1.0); CHKERRQ(ierr);
    assert( matB->compressedrow.use );
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

  ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID ); CHKERRQ(ierr);
  ierr = PetscMalloc( (nloc+1)*sizeof(PetscInt), &lid_gid ); CHKERRQ(ierr); /* explicit array needed */
  ierr = PetscMalloc( (nloc+1)*sizeof(PetscScalar), &deleted_parent_gid ); CHKERRQ(ierr);
  ierr = PetscMalloc( (nloc+1)*sizeof(PetscScalar), &lid_state ); CHKERRQ(ierr);
  ierr = PetscMalloc( (nloc+num_fine_ghosts)*sizeof(PetscInt), &id_llist ); CHKERRQ(ierr);
  
  /* need an inverse map - locals */
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
  iter = 0; nremoved = 0;
  ierr = ISGetIndices( perm, &perm_ix );     CHKERRQ(ierr);
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
              break;
            }
            else assert( !IS_SELECTED(statej) ); /* lid is now deleted, do it */
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
              nremoved++;
              continue; /* one local adj (me) and no ghost - singleton - flag and continue */
            }
          }
          /* SELECTED state encoded with global index */
          lid_state[lid] = (PetscScalar)(lid+my0);
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
          
          /* delete ghost adj of lid - deleted ghost done later for strict_aggs */
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
        PetscInt lid = matB->compressedrow.rindex[ix]; /* local boundary node */
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
      {
        PetscInt t1, t2;
        t1 = nloc - nDone; assert(t1>=0);
        ierr = MPI_Allreduce( &t1, &t2, 1, MPIU_INT, MPI_SUM, wcomm ); /* synchronous version */
        if( t2 == 0 ) break;
      }
    }
    else break; /* all done */
  } /* outer parallel MIS loop */
  ierr = ISRestoreIndices(perm,&perm_ix);     CHKERRQ(ierr);
  
  if( verbose ) {
    if( verbose == 1 ) {
      PetscPrintf(wcomm,"\t[%d]%s removed %d of %d vertices.\n",mype,__FUNCT__,nremoved,nloc);
    }
    else {
      MPI_Allreduce( &nremoved, &n, 1, MPIU_INT, MPI_SUM, wcomm );
      ierr = MatGetSize( Gmat, &kk, &j ); CHKERRQ(ierr);
      PetscPrintf(wcomm,"\t[%d]%s removed %d of %d vertices.\n",mype,__FUNCT__,n,kk);
    }
  }

  if( mpimat ){ /* free this buffer up (not really needed here) */
    ierr = VecRestoreArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr);
  }
  
  /* tell adj who my deleted vertices belong to - fill in id_llist[] selected ghost lists */
  if( strict_aggs && matB && a_locals_llist ) {
    PetscScalar *cpcol_sel_gid; 
    PetscInt cpid;
    /* get proc of deleted ghost */
    ierr = VecSetValues(locState, nloc, lid_gid, deleted_parent_gid, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
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
  if( a_locals_llist ) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nloc+num_fine_ghosts,id_llist,PETSC_COPY_VALUES,a_locals_llist);
    CHKERRQ(ierr);
  }
  
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

  if(mpimat){
    ierr = VecDestroy( &ghostState ); CHKERRQ(ierr);
  }
  ierr = VecDestroy( &locState );                    CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
