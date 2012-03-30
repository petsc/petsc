 
#include <petsc-private/matimpl.h>    /*I "petscmat.h" I*/
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
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . perm - serial permutation of rows of local to process in MIS
   . Gmat - glabal matrix of graph (data not defined)
   . strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';
   . verbose - 
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - array of list of nodes rooted at selected nodes
*/
#undef __FUNCT__
#define __FUNCT__ "maxIndSetAgg"
PetscErrorCode maxIndSetAgg( const IS perm,
                             const Mat Gmat,
			     const PetscBool strict_aggs,
                             const PetscInt verbose, 
                             PetscCoarsenData **a_locals_llist
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
  PetscMPIInt    mype,npe;
  const PetscInt *perm_ix;
  PetscInt       nDone, nselected = 0;
  const PetscInt nloc = Gmat->rmap->n;
  PetscInt      *lid_cprowID, *lid_gid;
  PetscBool  *lid_removed;
  PetscScalar   *lid_parent_gid = PETSC_NULL; /* only used for strict aggs */
  PetscScalar  *lid_state;
  PetscCoarsenData *agg_lists;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );   CHKERRQ(ierr);

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
  ierr = PetscMalloc( nloc*sizeof(PetscBool), &lid_removed ); CHKERRQ(ierr); /* explicit array needed */
  if( strict_aggs ) { 
    ierr = PetscMalloc( (nloc+1)*sizeof(PetscScalar), &lid_parent_gid ); CHKERRQ(ierr);
  }
  ierr = PetscMalloc( (nloc+1)*sizeof(PetscScalar), &lid_state ); CHKERRQ(ierr);

  /* has ghost nodes for !strict and uses local indexing (yuck) */
  ierr = PetscCDCreate( strict_aggs ? nloc : num_fine_ghosts+nloc, &agg_lists ); CHKERRQ(ierr); 
  if(a_locals_llist) *a_locals_llist = agg_lists;

  /* need an inverse map - locals */
  for(kk=0;kk<nloc;kk++) {
    lid_cprowID[kk] = -1; lid_removed[kk] = PETSC_FALSE;
    if( strict_aggs ) { 
      lid_parent_gid[kk] = -1.0;
    }
    lid_gid[kk] = kk + my0;
    lid_state[kk] =  (PetscScalar)((PetscReal)NOT_DONE);
  }
  /* set index into cmpressed row 'lid_cprowID' */
  if( matB ) {
    for (ix=0; ix<matB->compressedrow.nrows; ix++) {
      PetscInt lid = matB->compressedrow.rindex[ix];
      lid_cprowID[lid] = ix;
    }
  }
  /* MIS */
  iter = nremoved = nDone = 0;
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
      if(lid_removed[lid]) continue;
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
              nremoved++;
              lid_removed[lid] = PETSC_TRUE;
              /* should select this because it is technically in the MIS but lets not */
              /* lid_state[lid] = (PetscScalar)(lid+my0); */
              continue; /* one local adj (me) and no ghost - singleton */
            }
          }
          /* SELECTED state encoded with global index */
          lid_state[lid] = (PetscScalar)(lid+my0); /* needed???? */
          nselected++;
          if( strict_aggs ) {
            ierr = PetscCDAppendID( agg_lists, lid, lid+my0 ); CHKERRQ(ierr);
          }
          else {
            ierr = PetscCDAppendID( agg_lists, lid, lid ); CHKERRQ(ierr);
          }
          /* delete local adj */
          idx = matA->j + ii[lid];
          for (j=0; j<n; j++) {
            PetscInt lidj = idx[j];
            NState statej = (NState)PetscRealPart(lid_state[lidj]);
            if( statej == NOT_DONE ){
              nDone++;
              /* id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; */ /* insert 'lidj' into head of llist */
              if( strict_aggs ) {
                ierr = PetscCDAppendID( agg_lists, lid, lidj+my0 ); CHKERRQ(ierr);
              }
              else {
                ierr = PetscCDAppendID( agg_lists, lid, lidj ); CHKERRQ(ierr);
              }
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
                NState statej = (NState)PetscRealPart(cpcol_state[cpid]);        assert( !IS_SELECTED(statej) );
                if( statej == NOT_DONE ) {
                  /* cpcol_state[cpid] = (PetscScalar)DELETED; this should happen later ... */
                  /* id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; */ /* insert 'lidj' into head of llist */
                  ierr = PetscCDAppendID( agg_lists, lid, nloc+cpid ); CHKERRQ(ierr);
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
              nDone++;
              lid_state[lid] = (PetscScalar)(PetscReal)DELETED; /* delete this */
              if( !strict_aggs ) {
                PetscInt lidj = nloc + cpid;
                /* id_llist[lid] = id_llist[lidj]; id_llist[lidj] = lid; */ /* insert 'lid' into head of ghost llist */
                ierr = PetscCDAppendID( agg_lists, lidj, lid ); CHKERRQ(ierr);
              }
              else {
                PetscInt sgid = (PetscInt)PetscRealPart(cpcol_gid[cpid]);  
                lid_parent_gid[lid] = (PetscScalar)sgid; /* keep track of proc that I belong to */
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
      PetscPrintf(wcomm,"\t[%d]%s removed %d of %d vertices.  %d selected.\n",mype,__FUNCT__,nremoved,nloc,nselected);
    }
    else {
      ierr = MPI_Allreduce( &nremoved, &n, 1, MPIU_INT, MPI_SUM, wcomm );CHKERRQ(ierr);
      ierr = MatGetSize( Gmat, &kk, &j ); CHKERRQ(ierr);
      ierr = MPI_Allreduce( &nselected, &j, 1, MPIU_INT, MPI_SUM, wcomm );CHKERRQ(ierr);
      PetscPrintf(wcomm,"\t[%d]%s removed %d of %d vertices. (%d local)  %d selected.\n",mype,__FUNCT__,n,kk,nremoved,j);
    }
  }

  /* tell adj who my lid_parent_gid vertices belong to - fill in agg_lists selected ghost lists */
  if( strict_aggs && matB ) {
    PetscScalar *cpcol_sel_gid; 
    PetscInt cpid,*icpcol_gid;

    /* need to copy this to free buffer -- should do this globaly */
    ierr = PetscMalloc( num_fine_ghosts*sizeof(PetscInt), &icpcol_gid ); CHKERRQ(ierr);
    for(cpid=0; cpid<num_fine_ghosts; cpid++) icpcol_gid[cpid] = (PetscInt)PetscRealPart(cpcol_gid[cpid]);
    
    /* get proc of deleted ghost */
    ierr = VecSetValues(locState, nloc, lid_gid, lid_parent_gid, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin( locState ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locState ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat->Mvctx,locState,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray( mpimat->lvec, &cpcol_sel_gid ); CHKERRQ(ierr); /* has pe that owns ghost */
    for(cpid=0; cpid<num_fine_ghosts; cpid++) {
      PetscInt sgid = (PetscInt)PetscRealPart(cpcol_sel_gid[cpid]);
      PetscInt gid = icpcol_gid[cpid];
      if( sgid >= my0 && sgid < Iend ) { /* I own this deleted */
        PetscInt slid = sgid - my0;
        /* id_llist[lidj] = id_llist[lid]; id_llist[lid] = lidj; */ /* insert 'lidj' into head of llist */
        ierr = PetscCDAppendID( agg_lists, slid, gid ); CHKERRQ(ierr);
        assert(IS_SELECTED( (NState)PetscRealPart(lid_state[slid]) ));
      }
    }
    ierr = VecRestoreArray( mpimat->lvec, &cpcol_sel_gid ); CHKERRQ(ierr);
    ierr = PetscFree( icpcol_gid );  CHKERRQ(ierr);
  }
  else if( matB ) {
    ierr = VecRestoreArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr);
  }

  /* cache IS of removed nodes, use 'lid_gid' */
  for(kk=n=0,ix=my0;kk<nloc;kk++,ix++) {
    if( lid_removed[kk] ) lid_gid[n++] = ix;
  }
  assert(n==nremoved);
  ierr = PetscCDSetRemovedIS(agg_lists, wcomm, n, lid_gid ); CHKERRQ(ierr);

  ierr = PetscFree( lid_cprowID );  CHKERRQ(ierr);
  ierr = PetscFree( lid_gid );  CHKERRQ(ierr);
  ierr = PetscFree( lid_removed );  CHKERRQ(ierr);
  if( strict_aggs ) { 
    ierr = PetscFree( lid_parent_gid );  CHKERRQ(ierr);
  }
  ierr = PetscFree( lid_state );  CHKERRQ(ierr);

  if(mpimat){
    ierr = VecDestroy( &ghostState ); CHKERRQ(ierr);
  }
  ierr = VecDestroy( &locState );                    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

typedef struct {
  int dummy;
} MatCoarsen_MIS;
/*
   MIS coarsen, simple greedy. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenApply_MIS" 
static PetscErrorCode MatCoarsenApply_MIS( MatCoarsen coarse )
{
  /* MatCoarsen_MIS *MIS = (MatCoarsen_MIS*)coarse->; */
  PetscErrorCode  ierr;
  Mat             mat = coarse->graph;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  if(!coarse->perm) {
    IS perm;
    PetscInt n,m;
    MPI_Comm wcomm = ((PetscObject)mat)->comm;
    ierr = MatGetLocalSize( mat, &m, &n );       CHKERRQ(ierr);
    ierr = ISCreateStride( wcomm, m, 0, 1, &perm );CHKERRQ(ierr);
    ierr = maxIndSetAgg( perm, mat, coarse->strict_aggs, coarse->verbose, &coarse->agg_lists );
    CHKERRQ(ierr);
    ierr = ISDestroy( &perm );                    CHKERRQ(ierr);
  }
  else {
    ierr = maxIndSetAgg(coarse->perm, mat, coarse->strict_aggs, coarse->verbose, &coarse->agg_lists );
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenView_MIS" 
PetscErrorCode MatCoarsenView_MIS(MatCoarsen coarse,PetscViewer viewer)
{
  /* MatCoarsen_MIS *MIS = (MatCoarsen_MIS *)coarse->; */
  PetscErrorCode ierr;
  int rank;
  PetscBool    iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  ierr = MPI_Comm_rank(((PetscObject)coarse)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] MIS aggregator\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  } 
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this MIS coarsener",
                ((PetscObject)viewer)->type_name);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenDestroy_MIS" 
PetscErrorCode MatCoarsenDestroy_MIS ( MatCoarsen coarse )
{
  MatCoarsen_MIS *MIS = (MatCoarsen_MIS *)coarse->subctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  ierr = PetscFree(MIS);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*MC
   MATCOARSENMIS - Creates a coarsen context via the external package MIS.

   Collective on MPI_Comm

   Input Parameter:
.  coarse - the coarsen context

   Options Database Keys:
+  -mat_coarsen_MIS_xxx - 

   Level: beginner

.keywords: Coarsen, create, context

.seealso: MatCoarsenSetType(), MatCoarsenType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenCreate_MIS" 
PetscErrorCode  MatCoarsenCreate_MIS(MatCoarsen coarse)
{
  PetscErrorCode ierr;
  MatCoarsen_MIS *MIS;

  PetscFunctionBegin;
  ierr  = PetscNewLog( coarse, MatCoarsen_MIS, &MIS ); CHKERRQ(ierr);
  coarse->subctx                = (void*)MIS;

  coarse->ops->apply          = MatCoarsenApply_MIS;
  coarse->ops->view           = MatCoarsenView_MIS;
  coarse->ops->destroy        = MatCoarsenDestroy_MIS;
  /* coarse->ops->setfromoptions = MatCoarsenSetFromOptions_MIS; */
  PetscFunctionReturn(0);
}
EXTERN_C_END

