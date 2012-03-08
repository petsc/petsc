 
#include <private/matimpl.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

/* linked list for aggregates */
typedef struct llnode_tag{
  struct llnode_tag *next;
  union data_tag{
    PetscInt   gid;
    void      *array;
  }data;
}llNode;
typedef llNode lList;

#define POOL_CHK_SZ 10
static lList node_pool = {.next=0,.data.array=0};
static llNode *new_node = 0;
static PetscInt new_left = 0;

PetscErrorCode ll_set_id( lList *a_this, PetscInt a_gid )
{
  a_this->data.gid = a_gid;
  return 0;
}

PetscInt ll_get_id( lList *a_this )
{
  return a_this->data.gid;
}

PetscErrorCode get_new_node( llNode **a_out, PetscInt a_gid )
{
  PetscErrorCode ierr;
  if( !node_pool.data.array ){
    ierr = PetscMalloc( POOL_CHK_SZ*sizeof(llNode), &node_pool.data.array ); CHKERRQ(ierr);
    node_pool.next = 0;
    new_left = POOL_CHK_SZ;
    new_node->next = PETSC_NULL;
    new_node->data.array = node_pool.data.array;
  }
  else if( !new_left ){
    llNode *node;
    ierr = PetscMalloc(sizeof(llNode), &node); CHKERRQ(ierr);
    ierr = PetscMalloc(POOL_CHK_SZ*sizeof(llNode), &node->data.array); CHKERRQ(ierr);
    node->next = node_pool.next;
    node_pool.next = node;
    new_left = POOL_CHK_SZ;
    new_node = (llNode*)node->data.array;
  }
  new_node->data.gid = a_gid;
  new_node->next = 0;
  *a_out = new_node++; new_left--;
  return 0;
}

PetscErrorCode delete_node_pool(lList *a_node)
{
  PetscErrorCode ierr;
  llNode *n;
  if( !a_node ) a_node = &node_pool;
  if( a_node->data.array ) {
    ierr = PetscFree( a_node->data.array );  CHKERRQ(ierr);
    a_node->data.array = 0;
  }
  n = a_node->next;
  while( n ){
    llNode *lstn = n; 
    ierr = PetscFree( n->data.array );  CHKERRQ(ierr);
    n = n->next;
    ierr = PetscFree( lstn );  CHKERRQ(ierr);
  }
  a_node->next = 0; new_node = 0; new_left = 0;
  return 0;
}

PetscErrorCode ll_add_n( lList *a_this, llNode *n ) 
{
  n->next = a_this->next; 
  a_this->next = n;
  return 0;
}

PetscErrorCode ll_add_id( lList *a_this, PetscInt a_gid )
{
  PetscErrorCode ierr;
  llNode *n;
  ierr = get_new_node( &n, a_gid );  CHKERRQ(ierr);
  n->next = a_this->next; 
  a_this->next = n;
  return 0;
}

PetscErrorCode ll_add_destroy( lList *a_this, lList *a_list )
{
  PetscErrorCode ierr;
  llNode *n = a_this->next, *n1; assert(ll_get_id(a_list) != -1);
  ierr = get_new_node( &n1, ll_get_id(a_list) );  CHKERRQ(ierr);
  n1->next = a_list->next;
  ll_set_id( a_list, -1 ); /* flag for not a root */
  a_this->next = n1;
  while( n1 ){
    if( !n1->next ){
      n1->next = n;
      break;
    }
    n1 = n1->next;
  }
  return 0;
}

/* PetscErrorCode clear_node( llNode *n )  */
/* { */
/*   n->next=0; n->prev=0; n->data.gid=-1; */
/*   return 0; */
/* } */

typedef struct edge_tag{
  PetscReal   weight;
  PetscInt    lid0,gid1;
}Edge;

int hem_compare (const void *a, const void *b)
{
  return (((Edge*)a)->weight < ((Edge*)b)->weight) ? 1 : -1; /* 0 for equal */
}

/* -------------------------------------------------------------------------- */
/*
   heavyEdgeMatchAgg - parallel heavy edge matching (HEM) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . perm - permutation
   . a_Gmat - glabal matrix of graph (data not defined)
   . verbose - 
   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - linked list of local nodes rooted at selected node (size is nloc + nghosts)
*/
#undef __FUNCT__
#define __FUNCT__ "heavyEdgeMatchAgg"
PetscErrorCode heavyEdgeMatchAgg( const IS perm,
                                  const Mat a_Gmat,
                                  const PetscInt verbose, 
                                  IS *a_selected,
                                  IS *a_locals_llist
                                  )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  MPI_Comm       wcomm = ((PetscObject)a_Gmat)->comm;
  PetscInt       kk,n,ix,j,*idx,*ii,iter,Iend,my0;
  PetscMPIInt    mype;
  const PetscInt nloc = a_Gmat->rmap->n;
  PetscInt      *lid_cprowID,*lid_gid,*lid_state;
  Mat_SeqAIJ    *matA, *matB=0;
  Mat_MPIAIJ    *mpimat=0;
  PetscScalar   *lid_max_edge;
  lList         *agg_lists;
  Mat            cMat,tMat,P;
  MatScalar     *ap;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( a_Gmat, &my0, &Iend );  CHKERRQ(ierr);

  ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_gid ); CHKERRQ(ierr); /* explicit array needed */
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_state ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscScalar), &lid_max_edge ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(lList), &agg_lists ); CHKERRQ(ierr);

  /* need an inverse map - locals */
  for(kk=0;kk<nloc;kk++) {
    lid_gid[kk] = kk + my0;
    ll_set_id( &agg_lists[kk], -1 ); /* flag for not a root */
    agg_lists[kk].next = 0;
  }

  /* make a copy, we will destroy */
  ierr = MatDuplicate(a_Gmat,MAT_COPY_VALUES,&cMat);  CHKERRQ(ierr);
  ierr = PetscTypeCompare( (PetscObject)a_Gmat, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  iter = 0; 
  while( iter++ < 100 ) { 
    PetscScalar *cpcol_gid,*cpcol_me,*lid_me,one=1.0;
    Vec          locMaxEdge,ghostMaxEdge;
    PetscInt     nEdges,n_nz_row,nn;
    Edge        *Edges;
    PetscInt     lid,gid;
    const PetscInt *perm_ix;

    /* get submatrices of cMat */
    if (isMPI) {
      mpimat = (Mat_MPIAIJ*)cMat->data;
      matA = (Mat_SeqAIJ*)mpimat->A->data;
      matB = (Mat_SeqAIJ*)mpimat->B->data;
      /* force compressed storage of B */
      matB->compressedrow.check = PETSC_TRUE;
      ierr = MatCheckCompressedRow(mpimat->B,&matB->compressedrow,matB->i,cMat->rmap->n,-1.0); CHKERRQ(ierr);
      assert( matB->compressedrow.use );
    } else {
      matA = (Mat_SeqAIJ*)cMat->data;
    }
    assert( matA && !matA->compressedrow.use );
    assert( matB==0 || matB->compressedrow.use );

    /* set max edge on nodes */
    ierr = MatGetVecs( cMat, &locMaxEdge, 0 );         CHKERRQ(ierr);

    /* need an inverse map - locals */
    for(kk=0;kk<nloc;kk++) lid_cprowID[kk] = -1;
    /* set index into cmpressed row 'lid_cprowID' */
    if( matB ) {
      ii = matB->compressedrow.i;
      for (ix=0; ix<matB->compressedrow.nrows; ix++) {
        lid_cprowID[matB->compressedrow.rindex[ix]] = ix;
      }
    }
    
    /* set 'locMaxEdge' and create list of edges - count edges */
    for(nEdges=0,lid=0,gid=my0;lid<nloc;lid++,gid++){
      PetscReal max_dege = 0.;
      PetscScalar vval;
      ii = matA->i; n = ii[lid+1] - ii[lid]; idx = matA->j + ii[lid]; 
      ap = matA->a + ii[lid];
      for (j=0; j<n; j++) {
        PetscInt lidj = idx[j];
        if(lidj != lid && PetscAbsScalar(ap[j]) > max_dege ) max_dege = PetscAbsScalar(ap[j]);
        if(lidj > lid) nEdges++;
      }
      if( (ix=lid_cprowID[lid]) != -1 ) { /* if I have any ghost neighbors */
        ii = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
        ap = matB->a + ii[ix];
        idx = matB->j + ii[ix];
        for( j=0 ; j<n ; j++ ) {
          if( PetscAbsScalar(ap[j]) > max_dege ) max_dege = PetscAbsScalar(ap[j]);
          if( idx[j] > my0 ) nEdges++;
        }
      }
      vval = max_dege;
      ierr = VecSetValues( locMaxEdge, 1, &gid, &vval, INSERT_VALUES );  CHKERRQ(ierr); /* set with GID */
    }
    ierr = VecAssemblyBegin( locMaxEdge ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( locMaxEdge ); CHKERRQ(ierr);
    
    /* get 'cpcol_me' & 'cpcol_gid' using 'mpimat->lvec' */
    if( mpimat ) {
      PetscInt gid;
      Vec vec;
      ierr = MatGetVecs( cMat, &vec, 0 ); CHKERRQ(ierr);
      for(kk=0,gid=my0;kk<nloc;kk++,gid++) {
        PetscScalar v = (PetscScalar)(gid);
        ierr = VecSetValues( vec, 1, &gid, &v, INSERT_VALUES );  CHKERRQ(ierr); /* set with GID */
      }
      ierr = VecAssemblyBegin( vec ); CHKERRQ(ierr);
      ierr = VecAssemblyEnd( vec ); CHKERRQ(ierr);
      ierr = VecScatterBegin(mpimat->Mvctx,vec,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr =   VecScatterEnd(mpimat->Mvctx,vec,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecDestroy( &vec ); CHKERRQ(ierr);
      ierr = VecGetArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr); /* get proc ID in 'cpcol_gid' */
      
      ierr = VecDuplicate( mpimat->lvec, &ghostMaxEdge ); CHKERRQ(ierr); /* need 2nd compressed col. of off proc data */
      ierr = VecScatterBegin(mpimat->Mvctx,locMaxEdge,ghostMaxEdge,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr =   VecScatterEnd(mpimat->Mvctx,locMaxEdge,ghostMaxEdge,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray( ghostMaxEdge, &cpcol_me ); CHKERRQ(ierr);
    }
    ierr = VecGetArray( locMaxEdge, &lid_me ); CHKERRQ(ierr);

    /* setup sorted list of edges */
    ierr = PetscMalloc( nEdges*sizeof(Edge), &Edges ); CHKERRQ(ierr);
    ierr = ISGetIndices( perm, &perm_ix );     CHKERRQ(ierr);
    for(nEdges=n_nz_row=kk=0;kk<nloc;kk++){
      lid = perm_ix[kk];
      ii = matA->i; nn = n = ii[lid+1] - ii[lid]; idx = matA->j + ii[lid]; 
      ap = matA->a + ii[lid];
      for (j=0; j<n; j++) {
        PetscInt lidj = idx[j];        assert(PetscRealPart(ap[j])>0.);
        if(lidj > lid) {
          Edges[nEdges].lid0 = lid;
          Edges[nEdges].gid1 = lidj + my0;
          Edges[nEdges].weight = PetscRealPart(ap[j]);
          nEdges++;
        }
      }
      if( (ix=lid_cprowID[lid]) != -1 ) { /* if I have any ghost neighbors */
        ii = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
        ap = matB->a + ii[ix];
        idx = matB->j + ii[ix];
        nn += n;
        for( j=0 ; j<n ; j++ ) {
          assert(PetscRealPart(ap[j])>0.);
          if( idx[j] > my0 )  {
            Edges[nEdges].lid0 = lid;
            Edges[nEdges].gid1 = (PetscInt)PetscRealPart(cpcol_gid[idx[j]]);
            Edges[nEdges].weight = PetscRealPart(ap[j]);
            nEdges++;
          }
        }
      }
      if( nn > 1 ) n_nz_row++;
    }
    ierr = ISRestoreIndices(perm,&perm_ix);     CHKERRQ(ierr);
    qsort( Edges, nEdges, sizeof(Edge), hem_compare );

    /* projection matrix */
    ierr = MatCreateAIJ( wcomm, nloc, nloc,
                         PETSC_DETERMINE, PETSC_DETERMINE,
                         1, 0, 1, 0, &P );
    CHKERRQ(ierr);
    /* HEM */
    for(kk=0;kk<nloc;kk++) lid_state[kk] = -1;
    for(kk=0;kk<nEdges;kk++){
      Edge *e = &Edges[kk];
      PetscInt lid0=e->lid0,gid1=e->gid1,gid0=lid0+my0;      
      PetscBool isOK = PETSC_TRUE;
      if(lid_state[lid0] != -1 || ( gid1>=my0 && gid1<Iend && lid_state[gid1-my0] != -1) ) {
        continue;
      }
      /* parallel test -- see if larger edged vertex neighbor */
      if( (ix=lid_cprowID[lid0]) != -1 ) { /* if I have any ghost neighbors */
        ii = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
        idx = matB->j + ii[ix];
        for( j=0 ; j<n ; j++ ) {
          PetscInt cpid = idx[j]; /* compressed row ID in B mat */
          PetscReal jwht = PetscRealPart(cpcol_me[cpid]), iwht = PetscRealPart(lid_me[lid0]);
          if( jwht > iwht || (jwht==iwht && (PetscInt)PetscRealPart(cpcol_gid[cpid]) > my0) ) { /* use gid as pe proxy */
            isOK = PETSC_FALSE; /* can not do */
PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s skip big iwht=%e jwht=%e\n",mype,__FUNCT__,iwht,jwht);  
            break;
          }
        }
      }
      if( isOK ){
        ierr = ll_set_id( &agg_lists[lid0], gid0 );  CHKERRQ(ierr); /* this selects this */
        lid_state[lid0] = lid+my0; /* keep track of what we've done this round */
        if( gid1>=my0 && gid1<Iend ) {
          lid_state[gid1-my0] = gid1;  /* keep track of what we've done this round */
          if( ll_get_id(&agg_lists[gid1-my0]) != -1 ) {
            ll_add_destroy( &agg_lists[lid0], &agg_lists[gid1-my0] );
          }
          else {
            ierr = ll_add_id( &agg_lists[lid0], gid1 );  CHKERRQ(ierr);
          }
        }
        else {
          ierr = ll_add_id( &agg_lists[lid0], gid1 );  CHKERRQ(ierr); /* ghost??? */
        }
        ierr = MatSetValues(P,1,&gid0,1,&gid0,&one,ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValues(P,1,&gid1,1,&gid0,&one,ADD_VALUES); CHKERRQ(ierr);
      } /* matched */
    } /* edge loop */
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* clean up iteration */
    ierr = PetscFree( Edges );  CHKERRQ(ierr);
    if( mpimat ){ 
      ierr = VecRestoreArray( ghostMaxEdge, &cpcol_me ); CHKERRQ(ierr);
      ierr = VecRestoreArray( mpimat->lvec, &cpcol_gid ); CHKERRQ(ierr); 
      ierr = VecDestroy( &ghostMaxEdge ); CHKERRQ(ierr); 
    }
    ierr = VecRestoreArray( locMaxEdge, &lid_me ); CHKERRQ(ierr); 
    ierr = VecDestroy( &locMaxEdge ); CHKERRQ(ierr); 

    /* create next G if needed */
    if( iter==6 ) { /* hard wired test - need to look at full surrounded nodes or something */
      ierr = MatDestroy( &P );  CHKERRQ(ierr);
      ierr = MatDestroy( &cMat );  CHKERRQ(ierr);
      break;
    }
    else {
      ierr = MatPtAP(cMat,P,MAT_INITIAL_MATRIX,1.0,&tMat);CHKERRQ(ierr);
      ierr = MatDestroy( &P );  CHKERRQ(ierr);
      ierr = MatDestroy( &cMat );  CHKERRQ(ierr);
      cMat = tMat;
    }
  } /* coarsen iterator */

  /* create output IS of aggregates in linked list -- does not work in parallel!!!! */
  if( a_locals_llist ) {
    PetscInt *id_llist; /* linked list with locality info - output */
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &id_llist ); CHKERRQ(ierr);
    for(kk=0;kk<nloc;kk++) id_llist[kk] = -1;
    for(kk=0;kk<nloc;kk++) {
      if( ll_get_id(&agg_lists[kk]) != -1 ) {
        //if(verbose) PetscPrintf(PETSC_COMM_WORLD,"\t\t\t[%d]%s %d) root %d\n",mype,__FUNCT__,++cc,kk);
        llNode *node = agg_lists[kk].next;
        while(node){
          PetscInt lidj = ll_get_id(node)-my0;             assert(id_llist[lidj] == -1);
          id_llist[lidj] = id_llist[kk]; id_llist[kk] = lidj; /* insert 'lidj' into head of llist */
          node = node->next;
          //if(verbose) PetscPrintf(PETSC_COMM_WORLD,"\t\t\t\t[%d]%s %d) add %d\n",mype,__FUNCT__,++cc,lidj);
        }
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nloc,id_llist,PETSC_COPY_VALUES,a_locals_llist);
    CHKERRQ(ierr);
    ierr = PetscFree( id_llist );  CHKERRQ(ierr);
  }

  /* make 'a_selected' - output */
  {
    PetscInt nselected = 0, *selected_set, gid;
    for(kk=0;kk<nloc;kk++) if(ll_get_id( &agg_lists[kk] ) != -1) nselected++;    
    ierr = PetscMalloc( nselected*sizeof(PetscInt), &selected_set ); CHKERRQ(ierr); 
    for(kk=nselected=0;kk<nloc;kk++) {
      if((gid=ll_get_id(&agg_lists[kk])) != -1) {
        selected_set[nselected++] = gid-my0;
      }
    }
    ierr = ISCreateGeneral( PETSC_COMM_SELF, nselected, selected_set, PETSC_COPY_VALUES, a_selected );
    CHKERRQ(ierr);
    ierr = PetscFree( selected_set );  CHKERRQ(ierr);
    if(verbose) PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s nselected=%d nloc=%d\n",mype,__FUNCT__,nselected,nloc);  
  }

  ierr = PetscFree( lid_cprowID );  CHKERRQ(ierr);
  ierr = PetscFree( lid_gid );  CHKERRQ(ierr);
  ierr = PetscFree( lid_max_edge );  CHKERRQ(ierr);
  ierr = PetscFree( agg_lists );  CHKERRQ(ierr);
  ierr = PetscFree( lid_state );  CHKERRQ(ierr);
  ierr = delete_node_pool( 0 );  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

typedef struct {
  int dummy;
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
  if(!coarse->perm) {
    IS perm;
    PetscInt n,m;
    MPI_Comm wcomm = ((PetscObject)mat)->comm;
    ierr = MatGetLocalSize( mat, &m, &n );       CHKERRQ(ierr);
    ierr = ISCreateStride( wcomm, m, 0, 1, &perm );CHKERRQ(ierr);
    ierr = heavyEdgeMatchAgg( perm, mat, coarse->verbose, &coarse->mis, &coarse->agg_llist );CHKERRQ(ierr);
    ierr = ISDestroy( &perm );                    CHKERRQ(ierr);
  }
  else {
    ierr = heavyEdgeMatchAgg( coarse->perm, mat, coarse->verbose, &coarse->mis, &coarse->agg_llist );CHKERRQ(ierr);
  }

  
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

