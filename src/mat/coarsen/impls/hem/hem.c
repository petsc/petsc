 
#include <private/matimpl.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

/* typedef enum { NOT_DONE=-2, DELETED=-1, REMOVED=-3 } NState; */
/* use int instead of enum to facilitate passing them via Scatters */
typedef PetscInt NState;
static const NState NOT_DONE=-2;
static const NState DELETED=-1;
static const NState REMOVED=-3;
#define IS_SELECTED(s) (s!=DELETED && s!=NOT_DONE && s!=REMOVED)

/* -------------------------------------------------------------------------- */
/*
   heavyEdgeMatchAgg - parallel heavy edge matching (HEM) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . Gmat - glabal matrix of graph (data not defined)
   . verbose - 
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "heavyEdgeMatchAgg"
PetscErrorCode heavyEdgeMatchAgg( const Mat Gmat,
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
  PetscInt nDone = 0, nselected = 0;
  const PetscInt nloc = Gmat->rmap->n;
  PetscInt *lid_cprowID, *lid_gid;
  PetscScalar *deleted_parent_gid; /* only used for strict aggs */
  PetscInt *id_llist; /* linked list with locality info - output */
  PetscScalar *lid_state;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);

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
  while ( nDone < nloc || PETSC_TRUE ) { /* asyncronous not implemented */
    iter++;
    if( mpimat ) {
      ierr = VecGetArray( ghostState, &cpcol_state ); CHKERRQ(ierr);
    }
    /* check all vertices */
    for(kk=0;kk<nloc;kk++){
      PetscInt lid = kk;
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
              nremoved++; /* diagnostic */
              if( PETSC_FALSE ) {
                lid_state[lid] =  (PetscScalar)((PetscReal)REMOVED);
              }
              else {
                /* select this because it is technically in the MIS */
                lid_state[lid] = (PetscScalar)(lid+my0);
                nselected++;
              }
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
              PetscInt gid = (PetscInt)PetscRealPart(cpcol_gid[cpid]);  
              nDone++;
              lid_state[lid] = (PetscScalar)(PetscReal)DELETED; /* delete this */              
              deleted_parent_gid[lid] = (PetscScalar)gid; /* keep track of proc that I belong to */
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
  
  if( verbose ) {
    if( verbose == 1 ) {
      PetscPrintf(wcomm,"\t[%d]%s removed %d of %d vertices.\n",mype,__FUNCT__,nremoved,nloc);
    }
    else {
      MPI_Allreduce( &nremoved, &n, 1, MPIU_INT, MPI_SUM, wcomm );
      kk = nremoved;
      ierr = MatGetSize( Gmat, &kk, &j ); CHKERRQ(ierr);
      PetscPrintf(wcomm,"\t[%d]%s removed %d of %d vertices.\n",mype,__FUNCT__,n,kk);
    }
  }

  if( mpimat ){ /* free this buffer up (not really needed here) */
    ierr = VecRestoreArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr);
  }
  
  /* tell adj who my deleted vertices belong to - fill in id_llist[] selected ghost lists */
  if( matB && a_locals_llist ) {
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
    ierr = ISCreateGeneral( PETSC_COMM_SELF, nselected, selected_set, PETSC_COPY_VALUES, a_selected );
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

  PetscFunctionReturn(0);
}

typedef struct {
} MatCoarsen_HEM;
/*
   HEM coarsen, simple greedy. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenApply_HEM" 
static PetscErrorCode MatCoarsenApply_HEM( MatCoarsen coarse )
{
  /* MatCoarsen_HEM *HEM = (MatCoarsen_HEM*)coarse->data; */
  PetscErrorCode  ierr;
  Mat             mat = coarse->graph;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);

  ierr = heavyEdgeMatchAgg( mat, coarse->verbose, &coarse->mis, &coarse->agg_llist );CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenView_HEM" 
PetscErrorCode MatCoarsenView_HEM(MatCoarsen coarse,PetscViewer viewer)
{
  /* MatCoarsen_HEM *HEM = (MatCoarsen_HEM *)coarse->data; */
  PetscErrorCode ierr;
  int rank;
  PetscBool    iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  ierr = MPI_Comm_rank(((PetscObject)coarse)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] HEM aggregator\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE); CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this HEM coarsener",((PetscObject)viewer)->type_name);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenDestroy_HEM" 
PetscErrorCode MatCoarsenDestroy_HEM ( MatCoarsen coarse )
{
  MatCoarsen_HEM *HEM = (MatCoarsen_HEM *)coarse->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  ierr = PetscFree(HEM);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*MC
   MATCOARSENHEM - Creates a coarsen context via the external package HEM.

   Collective on MPI_Comm

   Input Parameter:
.  coarse - the coarsen context

   Options Database Keys:
+  -mat_coarsen_HEM_xxx - 

   Level: beginner

.keywords: Coarsen, create, context

.seealso: MatCoarsenSetType(), MatCoarsenType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenCreate_HEM" 
PetscErrorCode  MatCoarsenCreate_HEM(MatCoarsen coarse)
{
  PetscErrorCode ierr;
  MatCoarsen_HEM *HEM;

  PetscFunctionBegin;
  ierr  = PetscNewLog( coarse, MatCoarsen_HEM, &HEM ); CHKERRQ(ierr);
  coarse->data                = (void*)HEM;

  coarse->ops->apply          = MatCoarsenApply_HEM;
  coarse->ops->view           = MatCoarsenView_HEM;
  coarse->ops->destroy        = MatCoarsenDestroy_HEM;
  /* coarse->ops->setfromoptions = MatCoarsenSetFromOptions_HEM; */
  PetscFunctionReturn(0);
}
EXTERN_C_END

