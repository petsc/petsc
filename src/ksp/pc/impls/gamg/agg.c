/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <private/kspimpl.h>

#include <assert.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt nsmooths;
  Mat aux_mat;
  PetscBool sym_graph;
}PC_GAMG_AGG;

#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetNSmooths"
/*@
   PCGAMGSetNSmooths - Set number of smoothing steps (1 is typical)

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_agg_nsmooths

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetNSmooths(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetNSmooths_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetNSmooths_GAMG"
PetscErrorCode PCGAMGSetNSmooths_GAMG(PC pc, PetscInt n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  
  PetscFunctionBegin;
  pc_gamg_agg->nsmooths = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetSymGraph"
/*@
   PCGAMGSetSymGraph - 

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_sym_graph

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetSymGraph(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSymGraph_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetSymGraph_GAMG"
PetscErrorCode PCGAMGSetSymGraph_GAMG(PC pc, PetscBool n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->sym_graph = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*
   PCSetFromOptions_GAMG_AGG

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GAMG_AGG"
PetscErrorCode PCSetFromOptions_GAMG_AGG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  PetscBool        flag;
  
  PetscFunctionBegin;
  /* call base class */
  ierr = PCSetFromOptions_GAMG( pc ); CHKERRQ(ierr);

  ierr = PetscOptionsHead("GAMG-AGG options"); CHKERRQ(ierr);
  {
    /* -pc_gamg_agg_nsmooths */
    pc_gamg_agg->nsmooths = 0;
    ierr = PetscOptionsInt("-pc_gamg_agg_nsmooths",
                           "smoothing steps for smoothed aggregation, usually 1 (0)",
                           "PCGAMGSetNSmooths",
                           pc_gamg_agg->nsmooths,
                           &pc_gamg_agg->nsmooths,
                           &flag); 
    CHKERRQ(ierr);

    /* -pc_gamg_sym_graph */
    pc_gamg_agg->sym_graph = PETSC_FALSE;
    ierr = PetscOptionsBool("-pc_gamg_sym_graph",
                            "Set for asymetric matrices",
                            "PCGAMGSetSymGraph",
                            pc_gamg_agg->sym_graph,
                            &pc_gamg_agg->sym_graph,
                            &flag); 
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
  if( pc_gamg->verbose > 1 ) {
    PetscPrintf(PETSC_COMM_WORLD,"[%d]%s done\n",0,__FUNCT__);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_AGG

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_AGG"
PetscErrorCode PCDestroy_AGG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  
  PetscFunctionBegin;
  if( pc_gamg_agg ) {
    ierr = PetscFree(pc_gamg_agg);CHKERRQ(ierr);
    pc_gamg_agg = 0;
  }

  /* call base class */
  ierr = PCDestroy_GAMG( pc );CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_AGG

   Input Parameter:
   .  pc - the preconditioner context
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_AGG"
PetscErrorCode PCSetCoordinates_AGG( PC pc, PetscInt ndm, PetscReal *coords )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;
  PetscInt       arrsz,bs,my0,kk,ii,jj,nloc,Iend;
  Mat            Amat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific( Amat, MAT_CLASSID, 1 );
  ierr  = MatGetBlockSize( Amat, &bs );               CHKERRQ( ierr );
  ierr  = MatGetOwnershipRange( Amat, &my0, &Iend ); CHKERRQ(ierr);
  nloc = (Iend-my0)/bs; 
  if((Iend-my0)%bs!=0) SETERRQ1(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Bad local size %d.",nloc);
 
  /* SA: null space vectors */
  if( coords && bs==1 ) pc_gamg->data_cell_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if( coords ) pc_gamg->data_cell_cols = (ndm==2 ? 3 : 6); /* elasticity */
  else pc_gamg->data_cell_cols = bs; /* no data, force SA with constant null space vectors */
  pc_gamg->data_cell_rows = bs;

  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (pc_gamg->data==0 || (pc_gamg->data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->data );  CHKERRQ(ierr);
    ierr = PetscMalloc(arrsz*sizeof(PetscReal), &pc_gamg->data ); CHKERRQ(ierr);
  }
  /* copy data in - column oriented */
  for(kk=0;kk<nloc;kk++){
    const PetscInt M = Iend - my0;
    PetscReal *data = &pc_gamg->data[kk*bs];
    if( pc_gamg->data_cell_cols==1 ) *data = 1.0;
    else {
      for(ii=0;ii<bs;ii++)
        for(jj=0;jj<bs;jj++)
          if(ii==jj)data[ii*M + jj] = 1.0; /* translational modes */
          else data[ii*M + jj] = 0.0;
      if( coords ) {
        if( ndm == 2 ){ /* rotational modes */
          data += 2*M;
          data[0] = -coords[2*kk+1];
          data[1] =  coords[2*kk];
        }
        else {
          data += 3*M;
          data[0] = 0.0;               data[M+0] =  coords[3*kk+2]; data[2*M+0] = -coords[3*kk+1];
          data[1] = -coords[3*kk+2]; data[M+1] = 0.0;               data[2*M+1] =  coords[3*kk];
          data[2] =  coords[3*kk+1]; data[M+2] = -coords[3*kk];   data[2*M+2] = 0.0;
        }          
      }
    }
  }

  pc_gamg->data_sz = arrsz;

  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscInt NState;
static const NState NOT_DONE=-2;
static const NState DELETED=-1;
static const NState REMOVED=-3;
#define IS_SELECTED(s) (s!=DELETED && s!=NOT_DONE && s!=REMOVED)

/* -------------------------------------------------------------------------- */
/*
   smoothAggs - greedy grab of with G1 (unsquared graph) -- AIJ specific
     - AGG-MG specific: clears singletons out of 'selected_2'

   Input Parameter:
   . Gmat_2 - glabal matrix of graph (data not defined)
   . Gmat_1 - base graph to grab with
   . selected_2 - 
   Input/Output Parameter:
   . llist_aggs_2 - linked list of aggs, ghost lids are based on Gmat_2 (squared graph)
*/
#undef __FUNCT__
#define __FUNCT__ "smoothAggs"
PetscErrorCode smoothAggs( const Mat Gmat_2, /* base (squared) graph */
                           const Mat Gmat_1, /* base graph, could be unsymmetic */
                           const IS selected_2, /* [nselected total] selected vertices */
                           IS llist_aggs_2 /* [nloc_nghost] global ID of aggregate */
                           )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA_1, *matB_1=0, *matA_2, *matB_2=0;
  MPI_Comm       wcomm = ((PetscObject)Gmat_2)->comm;
  PetscMPIInt    mype;
  PetscInt       lid,*ii,*idx,ix,Iend,my0,nnodes_2,kk,n,j;
  Mat_MPIAIJ    *mpimat_2 = 0, *mpimat_1=0;
  const PetscInt nloc = Gmat_2->rmap->n;
  PetscScalar   *cpcol_1_state,*cpcol_2_state,*deleted_parent_gid;
  PetscInt      *lid_cprowID_1,*id_llist_2,*lid_cprowID_2;
  NState        *lid_state;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat_1,&my0,&Iend);  CHKERRQ(ierr);

  if( !PETSC_TRUE ) {
    PetscViewer viewer; char fname[32]; static int llev=0;
    sprintf(fname,"Gmat2_%d.m",llev++);
    PetscViewerASCIIOpen(wcomm,fname,&viewer);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Gmat_2, viewer ); CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  { /* copy linked list into temp buffer - should not work directly on pointer */
    const PetscInt *llist_idx;
    ierr = ISGetSize( llist_aggs_2, &nnodes_2 );        CHKERRQ(ierr);
    ierr = PetscMalloc( nnodes_2*sizeof(PetscInt), &id_llist_2 ); CHKERRQ(ierr);
    ierr = ISGetIndices( llist_aggs_2, &llist_idx );       CHKERRQ(ierr);
    for(lid=0;lid<nnodes_2;lid++) id_llist_2[lid] = llist_idx[lid];
    ierr = ISRestoreIndices( llist_aggs_2, &llist_idx );     CHKERRQ(ierr);
  }
  
  /* get submatrices */
  ierr = PetscTypeCompare( (PetscObject)Gmat_1, MATMPIAIJ, &isMPI ); CHKERRQ(ierr);
  if(isMPI) {
    /* grab matrix objects */
    mpimat_2 = (Mat_MPIAIJ*)Gmat_2->data;
    mpimat_1 = (Mat_MPIAIJ*)Gmat_1->data;
    matA_1 = (Mat_SeqAIJ*)mpimat_1->A->data;
    matB_1 = (Mat_SeqAIJ*)mpimat_1->B->data;
    matA_2 = (Mat_SeqAIJ*)mpimat_2->A->data;
    matB_2 = (Mat_SeqAIJ*)mpimat_2->B->data;

    /* force compressed row storage for B matrix in AuxMat */
    matB_1->compressedrow.check = PETSC_TRUE;
    ierr = MatCheckCompressedRow(mpimat_1->B,&matB_1->compressedrow,matB_1->i,Gmat_1->rmap->n,-1.0);
    CHKERRQ(ierr);

    ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID_2 ); CHKERRQ(ierr);
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID_1 ); CHKERRQ(ierr);
    for(lid=0;lid<nloc;lid++) lid_cprowID_1[lid] = lid_cprowID_2[lid] = -1;
    for (ix=0; ix<matB_1->compressedrow.nrows; ix++) {
      PetscInt lid = matB_1->compressedrow.rindex[ix];
      lid_cprowID_1[lid] = ix;
    }
    for (ix=0; ix<matB_2->compressedrow.nrows; ix++) {
      PetscInt lid = matB_2->compressedrow.rindex[ix];
      lid_cprowID_2[lid] = ix;
    }
  }  
  else {
    matA_1 = (Mat_SeqAIJ*)Gmat_1->data;
    matA_2 = (Mat_SeqAIJ*)Gmat_2->data;
    lid_cprowID_2 = lid_cprowID_1 = 0;
  }
  assert( matA_1 && !matA_1->compressedrow.use );
  assert( matB_1==0 || matB_1->compressedrow.use );
  assert( matA_2 && !matA_2->compressedrow.use );
  assert( matB_2==0 || matB_2->compressedrow.use );

  /* get state of locals and selected gid for deleted */
  ierr = PetscMalloc( nloc*sizeof(NState), &lid_state ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscScalar), &deleted_parent_gid ); CHKERRQ(ierr);
  for( lid = 0 ; lid < nloc ; lid++ ) {
    deleted_parent_gid[lid] = -1.0;
    lid_state[lid] = DELETED;
  }
  /* set index into compressed row 'lid_cprowID', not -1 means its a boundary node */
  {
    PetscInt nSelected;
    const PetscInt *selected_idx;
    /* set local selected */
    ierr = ISGetSize( selected_2, &nSelected );        CHKERRQ(ierr);
    ierr = ISGetIndices( selected_2, &selected_idx );       CHKERRQ(ierr);
    for(kk=0;kk<nSelected;kk++){
      PetscInt lid = selected_idx[kk];
      if(lid<nloc) lid_state[lid] = (NState)(lid+my0); /* selected flag */
      else break;
      /* remove singletons */
      ii = matA_2->i; n = ii[lid+1] - ii[lid];
      if( n < 2 ) {
        if(!lid_cprowID_2 || (ix=lid_cprowID_2[lid])==-1 || (matB_2->compressedrow.i[ix+1]-matB_2->compressedrow.i[ix])==0){
          lid_state[lid] = REMOVED;
        }
      }
    }
    ierr = ISRestoreIndices( selected_2, &selected_idx );     CHKERRQ(ierr);
  }
  /* map local to selected local, -1 means a ghost owns it */
  for(lid=kk=0;lid<nloc;lid++){
    NState state = lid_state[lid];
    if( IS_SELECTED(state) ){
      PetscInt flid = lid;
      do{
        if(flid<nloc){
          deleted_parent_gid[flid] = (PetscScalar)(lid + my0);
        }
        kk++;
      } while( (flid=id_llist_2[flid]) != -1 );
    }
  }
  /* get 'cpcol_1_state', 'cpcol_2_state' - uses mpimat_1->lvec & mpimat_2->lvec for temp space */
  if (isMPI) {
    Vec          tempVec;

    /* get 'cpcol_1_state' */ 
    ierr = MatGetVecs( Gmat_1, &tempVec, 0 );         CHKERRQ(ierr);
    for(kk=0,j=my0;kk<nloc;kk++,j++){
      PetscScalar v = (PetscScalar)lid_state[kk];
      ierr = VecSetValues( tempVec, 1, &j, &v, INSERT_VALUES );  CHKERRQ(ierr); 
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( mpimat_1->lvec, &cpcol_1_state ); CHKERRQ(ierr);
    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr);

    /* get 'cpcol_2_state' */ 
    ierr = MatGetVecs( Gmat_2, &tempVec, 0 );         CHKERRQ(ierr);
    for(kk=0,j=my0;kk<nloc;kk++,j++){
      PetscScalar v = (PetscScalar)lid_state[kk];
      ierr = VecSetValues( tempVec, 1, &j, &v, INSERT_VALUES );  CHKERRQ(ierr); 
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( mpimat_2->lvec, &cpcol_2_state ); CHKERRQ(ierr);
    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr);
  } /* ismpi */

  /* doit */
  for(lid=0;lid<nloc;lid++){
    NState state = lid_state[lid];
    if( IS_SELECTED(state) ) {      /* steal locals */
      ii = matA_1->i; n = ii[lid+1] - ii[lid]; 
      idx = matA_1->j + ii[lid];
      for (j=0; j<n; j++) {
        PetscInt flid, lidj = idx[j], sgid;
        NState statej = lid_state[lidj];
        if (statej==DELETED && (sgid=(PetscInt)PetscRealPart(deleted_parent_gid[lidj])) != lid+my0) { /* steal local */
          deleted_parent_gid[lidj] = (PetscScalar)(lid+my0); /* send this with _2 */
          if( sgid >= my0 && sgid < my0+nloc ){       /* I'm stealing this local from a local */
            PetscInt hav=0, flid2=sgid-my0, lastid;
            /* looking for local from local so id_llist_2 works */
            for( lastid=flid2, flid=id_llist_2[flid2] ; flid!=-1 ; flid=id_llist_2[flid] ) {
              if( flid == lidj ) {
                id_llist_2[lastid] = id_llist_2[flid];                    /* remove lidj from list */             
                id_llist_2[flid] = id_llist_2[lid]; id_llist_2[lid] = flid; /* insert 'lidj' into head of llist */
                hav++;
                break;
              }
              lastid = flid;
            }
            if(hav!=1){
              if(hav==0)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"found node %d times???",hav);
            }
          }
          else{            /* I'm stealing this local, owned by a ghost - ok to use _2, local */
            assert(sgid==-1); 
            id_llist_2[lidj] = id_llist_2[lid]; id_llist_2[lid] = lidj; /* insert 'lidj' into head of llist */
            /* local remove at end, off add/rm at end */
          }
        }
      }
    }
    else if( state == DELETED && lid_cprowID_1 ) {
      PetscInt sgidold = (PetscInt)PetscRealPart(deleted_parent_gid[lid]);
      /* see if I have a selected ghost neighbor that will steal me */
      if( (ix=lid_cprowID_1[lid]) != -1 ){ 
        ii = matB_1->compressedrow.i; n = ii[ix+1] - ii[ix];
        idx = matB_1->j + ii[ix];
        for( j=0 ; j<n ; j++ ) {
          PetscInt cpid = idx[j];
          NState statej = (NState)PetscRealPart(cpcol_1_state[cpid]);
          if( IS_SELECTED(statej) && sgidold != (PetscInt)statej ) { /* ghost will steal this, remove from my list */
            deleted_parent_gid[lid] = (PetscScalar)statej; /* send who selected with _2 */
            if( sgidold>=my0 && sgidold<(my0+nloc) ) { /* this was mine */
              PetscInt lastid,hav=0,flid,oldslidj=sgidold-my0;
              /* remove from 'oldslidj' list, local so _2 is OK */
              for( lastid=oldslidj, flid=id_llist_2[oldslidj] ; flid != -1 ; flid=id_llist_2[flid] ) {
                if( flid == lid ) {
                  id_llist_2[lastid] = id_llist_2[flid];   /* remove lid from oldslidj list */
                  hav++;
                  break;
                }
                lastid = flid;
              }
              if(hav!=1){
                if(hav==0)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
                SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"found node %d times???",hav);
              }
              id_llist_2[lid] = -1; /* terminate linked list - needed? */
            }
            else assert(id_llist_2[lid] == -1);
          }
        }
      }
    } /* selected/deleted */
    else assert(state == REMOVED || !lid_cprowID_1);
  } /* node loop */

  if( isMPI ) {
    PetscScalar *cpcol_2_sel_gid;
    Vec          tempVec;
    PetscInt     cpid;

    ierr = VecRestoreArray( mpimat_1->lvec, &cpcol_1_state ); CHKERRQ(ierr); 
    ierr = VecRestoreArray( mpimat_2->lvec, &cpcol_2_state ); CHKERRQ(ierr); 

    /* get 'cpcol_2_sel_gid' */ 
    ierr = MatGetVecs( Gmat_2, &tempVec, 0 );         CHKERRQ(ierr);
    for(kk=0,j=my0;kk<nloc;kk++,j++){
      ierr = VecSetValues( tempVec, 1, &j, &deleted_parent_gid[kk], INSERT_VALUES );  CHKERRQ(ierr); 
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr);

    ierr = VecGetArray( mpimat_2->lvec, &cpcol_2_sel_gid ); CHKERRQ(ierr);

    /* look for deleted ghosts and see if they moved */
    for(lid=0;lid<nloc;lid++){
      NState state = lid_state[lid];
      if( IS_SELECTED(state) ){
        PetscInt flid,lastid,old_sgid=lid+my0;
        /* look for deleted ghosts and see if they moved */
        for( lastid=lid, flid=id_llist_2[lid] ; flid!=-1 ; flid=id_llist_2[flid] ) {
          if( flid>=nloc ) {
            PetscInt cpid = flid-nloc, sgid_new = (PetscInt)PetscRealPart(cpcol_2_sel_gid[cpid]);
            if( sgid_new != old_sgid && sgid_new != -1 ) {
              id_llist_2[lastid] = id_llist_2[flid];                    /* remove 'flid' from list */
              id_llist_2[flid] = -1; 
              flid = lastid;
            } /* if it changed parents */
            else lastid = flid;
          } /* for ghost nodes */
          else lastid = flid;
        } /* loop over list of deleted */
      } /* selected */
    }

    /* look at ghosts, see if they changed, and moved here */
    for(cpid=0;cpid<nnodes_2-nloc;cpid++){
      PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_sel_gid[cpid]);
      if( sgid_new>=my0 && sgid_new<(my0+nloc) ) { /* this is mine */
        PetscInt lastid,flid,slid_new=sgid_new-my0,flidj=nloc+cpid,hav=0;
        for( lastid=slid_new, flid=id_llist_2[slid_new] ; flid != -1 ; flid=id_llist_2[flid] ) {
          if( flid == flidj ) {
            hav++;
            break;
          }
          lastid = flid;
        }
        if( hav != 1 ){
          assert(id_llist_2[flidj] == -1);
          id_llist_2[flidj] = id_llist_2[slid_new]; id_llist_2[slid_new] = flidj; /* insert 'flidj' into head of llist */
        }
      }
    }

    ierr = VecRestoreArray( mpimat_2->lvec, &cpcol_2_sel_gid ); CHKERRQ(ierr);  
    ierr = PetscFree( lid_cprowID_1 );  CHKERRQ(ierr);
    ierr = PetscFree( lid_cprowID_2 );  CHKERRQ(ierr);
  }

  /* copy out new aggs */
  ierr = ISGeneralSetIndices(llist_aggs_2, nnodes_2, id_llist_2, PETSC_COPY_VALUES ); CHKERRQ(ierr);

  ierr = PetscFree( id_llist_2 );  CHKERRQ(ierr);
  ierr = PetscFree( deleted_parent_gid );  CHKERRQ(ierr);
  ierr = PetscFree( lid_state );  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetData_AGG

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetData_AGG"
PetscErrorCode PCSetData_AGG( PC pc )
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = PCSetCoordinates_AGG( pc, -1, PETSC_NULL ); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 formProl0

   Input Parameter:
   . selected - list of selected local ID, includes selected ghosts
   . locals_llist - linked list with aggregates
   . bs - block size
   . nSAvec - num columns of new P
   . my0crs - global index of locals
   . data_stride - bs*(nloc nodes + ghost nodes)
   . data_in[data_stride*nSAvec] - local data on fine grid
   . flid_fgid[data_stride/bs] - make local to global IDs, includes ghosts in 'locals_llist'
  Output Parameter:
   . a_data_out - in with fine grid data (w/ghosts), out with coarse grid data
   . a_Prol - prolongation operator
*/
#undef __FUNCT__
#define __FUNCT__ "formProl0"
PetscErrorCode formProl0( IS selected, /* list of selected local ID, includes selected ghosts */
                          IS locals_llist, /* linked list from selected vertices of aggregate unselected vertices */
                          const PetscInt bs,
                          const PetscInt nSAvec,
                          const PetscInt my0crs,
                          const PetscInt data_stride,
                          PetscReal data_in[],
                          const PetscInt flid_fgid[],
                          PetscReal **a_data_out,
                          Mat a_Prol /* prolongation operator (output)*/
                          )
{
  PetscErrorCode ierr;
  PetscInt  Istart,Iend,nFineLoc,clid,flid,aggID,kk,jj,ii,mm,nLocalSelected,ndone,nSelected,minsz;
  MPI_Comm       wcomm = ((PetscObject)a_Prol)->comm;
  PetscMPIInt    mype, npe;
  const PetscInt *selected_idx,*llist_idx;
  PetscReal      *out_data;
/* #define OUT_AGGS */
#ifdef OUT_AGGS
  static PetscInt llev = 0; char fname[32]; FILE *file;
  sprintf(fname,"aggs_%d.m",llev++);
  if(llev==1) {
    file = fopen(fname,"w");
    fprintf(file,"figure,\n");
  }
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( a_Prol, &Istart, &Iend );    CHKERRQ(ierr);
  nFineLoc = (Iend-Istart)/bs; assert((Iend-Istart)%bs==0);
  
  ierr = ISGetSize( selected, &nSelected );        CHKERRQ(ierr);
  ierr = ISGetIndices( selected, &selected_idx );       CHKERRQ(ierr);
  ierr = ISGetIndices( locals_llist, &llist_idx );      CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<nSelected;kk++){
    PetscInt lid = selected_idx[kk];
    if( lid<nFineLoc && llist_idx[lid] != -1 ) nLocalSelected++;
  }

  /* aloc space for coarse point data (output) */
#define DATA_OUT_STRIDE (nLocalSelected*nSAvec)
  ierr = PetscMalloc( DATA_OUT_STRIDE*nSAvec*sizeof(PetscReal), &out_data ); CHKERRQ(ierr);
  for(ii=0;ii<DATA_OUT_STRIDE*nSAvec;ii++) out_data[ii]=1.e300;
  *a_data_out = out_data; /* output - stride nLocalSelected*nSAvec */

  /* find points and set prolongation */
  minsz = 100;
  ndone = 0;
  for( mm = clid = 0 ; mm < nSelected ; mm++ ){
    PetscInt lid = selected_idx[mm];
    PetscInt cgid = my0crs + clid, cids[100];

    if( lid>=nFineLoc || llist_idx[lid]==-1 ) continue; /* skip ghost or singleton */

    /* count agg */
    aggID = 0;
    flid = selected_idx[mm]; assert(flid != -1);
    do{
      aggID++;
    } while( (flid=llist_idx[flid]) != -1 );
    if( aggID<minsz ) minsz = aggID;

    /* get block */
    {
      PetscBLASInt   asz=aggID,M=asz*bs,N=nSAvec,INFO;
      PetscBLASInt   Mdata=M+((N-M>0)?N-M:0),LDA=Mdata,LWORK=N*bs;
      PetscScalar    *qqc,*qqr,*TAU,*WORK;
      PetscInt       *fids;
      
      ierr = PetscMalloc( (Mdata*N)*sizeof(PetscScalar), &qqc ); CHKERRQ(ierr); 
      ierr = PetscMalloc( (M*N)*sizeof(PetscScalar), &qqr ); CHKERRQ(ierr); 
      ierr = PetscMalloc( N*sizeof(PetscScalar), &TAU ); CHKERRQ(ierr); 
      ierr = PetscMalloc( LWORK*sizeof(PetscScalar), &WORK ); CHKERRQ(ierr); 
      ierr = PetscMalloc( M*sizeof(PetscInt), &fids ); CHKERRQ(ierr); 

      flid = selected_idx[mm];
      aggID = 0;
      do{
        /* copy in B_i matrix - column oriented */
        PetscReal *data = &data_in[flid*bs];
        for( kk = ii = 0; ii < bs ; ii++ ) {
          for( jj = 0; jj < N ; jj++ ) {
            qqc[jj*Mdata + aggID*bs + ii] = data[jj*data_stride + ii];
          }
        }
#ifdef OUT_AGGS
        if(llev==1) {
          char str[] = "plot(%e,%e,'r*'), hold on,\n", col[] = "rgbkmc", sim[] = "*os+h>d<vx^";
          PetscInt M,pi,pj,gid=Istart+flid;
          str[12] = col[clid%6]; str[13] = sim[(clid/6)%11]; 
          M = (PetscInt)(PetscSqrtScalar((PetscScalar)nFineLoc*npe));
          pj = gid/M; pi = gid%M;
          fprintf(file,str,(double)pi,(double)pj);
          /* fprintf(file,str,data[2*data_stride+1],-data[2*data_stride]); */
        }
#endif
        /* set fine IDs */
        for(kk=0;kk<bs;kk++) fids[aggID*bs + kk] = flid_fgid[flid]*bs + kk;
        
        aggID++;
      }while( (flid=llist_idx[flid]) != -1 );

      /* pad with zeros */
      for( ii = asz*bs; ii < Mdata ; ii++ ) {
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
        PetscReal *data = &out_data[clid*nSAvec];
        for( jj = 0; jj < nSAvec ; jj++ ) {
          for( ii = 0; ii < nSAvec ; ii++ ) {
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
      for(kk=0;kk<N;kk++) {
        cids[kk] = N*cgid + kk; /* global col IDs in P0 */
      }
      ierr = MatSetValues(a_Prol,M,fids,N,cids,qqr,INSERT_VALUES); CHKERRQ(ierr);

      ierr = PetscFree( qqc );  CHKERRQ(ierr);
      ierr = PetscFree( qqr );  CHKERRQ(ierr);
      ierr = PetscFree( TAU );  CHKERRQ(ierr);
      ierr = PetscFree( WORK );  CHKERRQ(ierr);
      ierr = PetscFree( fids );  CHKERRQ(ierr);
    } /* scoping */
    clid++;
  } /* for all coarse nodes */

/* ierr = MPI_Allreduce( &ndone, &ii, 1, MPIU_INT, MPIU_SUM, wcomm ); */
/* MatGetSize( a_Prol, &kk, &jj ); */
/* ierr = MPI_Allreduce( &minsz, &jj, 1, MPIU_INT, MPIU_MIN, wcomm ); */
/* PetscPrintf(PETSC_COMM_WORLD," **** [%d]%s %d total done, N=%d (%d local done), min agg. size = %d\n",mype,__FUNCT__,ii,kk/bs,ndone,jj); */

#ifdef OUT_AGGS
  if(llev==1) fclose(file);
#endif
  ierr = ISRestoreIndices( selected, &selected_idx );     CHKERRQ(ierr);
  ierr = ISRestoreIndices( locals_llist, &llist_idx );     CHKERRQ(ierr);
  ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGgraph_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
  Output Parameter:
   . a_Gmat - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGgraph_AGG"
PetscErrorCode PCGAMGgraph_AGG( PC pc,
                                const Mat Amat,
                                Mat *a_Gmat
                                )
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscInt verbose = pc_gamg->verbose;
  const PetscReal vfilter = pc_gamg->threshold;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  PetscMPIInt    mype,npe;
  Mat            Gmat, Gmat2;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);

  ierr  = createSimpleGraph( Amat, &Gmat ); CHKERRQ( ierr );
  
  ierr  = scaleFilterGraph( &Gmat, vfilter, pc_gamg_agg->sym_graph, verbose ); CHKERRQ( ierr );
  
  ierr = MatTransposeMatMult( Gmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );
  CHKERRQ(ierr);
  
  /* attach auxilary matrix */
  pc_gamg_agg->aux_mat = Gmat;
  
  *a_Gmat = Gmat2;
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCoarsen_AGG

  Input Parameter:
   . pc - this
   . Gmat2 - matrix on this fine level
  Output Parameter:
   . a_selected - prolongation operator to the next level
   . a_llist_parent - data of coarse grid points (num local columns in 'a_P_out')
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_AGG"
PetscErrorCode PCGAMGCoarsen_AGG( PC pc,
                                  const Mat Gmat2,
                                  IS *a_selected,
                                  IS *a_llist_parent
                                  )
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat             Gmat1; /* unsquared graph (not symetrized!) */
  IS              perm, selected, llist_parent;
  PetscInt        Ii,nloc,bs,n,m;
  PetscInt *permute; 
  PetscBool *bIndexSet;
  MatCoarsen crs;
  MPI_Comm        wcomm = ((PetscObject)Gmat2)->comm;
  /* PetscMPIInt     mype,npe; */

  PetscFunctionBegin;
  /* ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr); */
  /* ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr); */
  ierr = MatGetLocalSize( Gmat2, &n, &m ); CHKERRQ(ierr);
  ierr = MatGetBlockSize( Gmat2, &bs ); CHKERRQ(ierr); assert(bs==1);
  nloc = n/bs;

  /* get unsquared graph */
  Gmat1 = pc_gamg_agg->aux_mat; pc_gamg_agg->aux_mat = 0;
  
  /* get MIS aggs */
  /* randomize */  
  ierr = PetscMalloc( nloc*sizeof(PetscInt), &permute ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscBool), &bIndexSet ); CHKERRQ(ierr);
  for ( Ii = 0; Ii < nloc ; Ii++ ){
    bIndexSet[Ii] = PETSC_FALSE;
    permute[Ii] = Ii;
  }
  srand(1); /* make deterministic */
  for ( Ii = 0; Ii < nloc ; Ii++ ) {
    PetscInt iSwapIndex = rand()%nloc;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii) {
      PetscInt iTemp = permute[iSwapIndex];
      permute[iSwapIndex] = permute[Ii];
      permute[Ii] = iTemp;
      bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
  ierr = PetscFree( bIndexSet );  CHKERRQ(ierr);
  
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_USE_POINTER, &perm);
  CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MatCoarsenCreate( wcomm, &crs ); CHKERRQ(ierr);
  /* ierr = MatCoarsenSetType( crs, MATCOARSENMIS ); CHKERRQ(ierr); */
  ierr = MatCoarsenSetFromOptions( crs ); CHKERRQ(ierr);
  ierr = MatCoarsenSetGreedyOrdering( crs, perm ); CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency( crs, Gmat2 ); CHKERRQ(ierr);
  ierr = MatCoarsenSetVerbose( crs, pc_gamg->verbose ); CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs( crs, PETSC_TRUE ); CHKERRQ(ierr);
  ierr = MatCoarsenApply( crs ); CHKERRQ(ierr);
  ierr = MatCoarsenGetMISAggLists( crs, &selected, &llist_parent ); CHKERRQ(ierr);
  ierr = MatCoarsenDestroy( &crs ); CHKERRQ(ierr);

  ierr = ISDestroy( &perm );                    CHKERRQ(ierr); 
  ierr = PetscFree( permute );  CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  /* smooth aggs */
  ierr = smoothAggs( Gmat2, Gmat1, selected, llist_parent ); CHKERRQ(ierr);

  ierr = MatDestroy( &Gmat1 );  CHKERRQ(ierr);

  *a_selected = selected;
  *a_llist_parent = llist_parent;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCGAMGprolongator_AGG
 
 Input Parameter:
 . pc - this
 . Amat - matrix on this fine level
 . Graph - used to get ghost data for nodes in 
 . selected - [nselected inc. chosts]
 . llist_parent - [nloc + Gmat.nghost] linked list 
 Output Parameter:
 . a_P_out - prolongation operator to the next level
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGprolongator_AGG"
PetscErrorCode PCGAMGprolongator_AGG( PC pc,
                                     const Mat Amat,
                                     const Mat Gmat,
                                     IS selected,
                                     IS llist_parent,
                                     Mat *a_P_out
                                     )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscInt verbose = pc_gamg->verbose;
  const PetscInt data_cols = pc_gamg->data_cell_cols;
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,nloc,ii,jj,kk,my0,nLocalSelected,bs;
  Mat            Prol;
  PetscMPIInt    mype, npe;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  const PetscInt *selected_idx,*llist_idx,col_bs=data_cols;
  PetscReal      *data_w_ghost;
  PetscInt       myCrs0, nbnodes=0, *flid_fgid;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  ierr  = MatGetBlockSize( Amat, &bs ); CHKERRQ( ierr ); 
  nloc = (Iend-Istart)/bs; my0 = Istart/bs; assert((Iend-Istart)%bs==0);
  
  /* get 'nLocalSelected' */
  ierr = ISGetSize( selected, &kk );        CHKERRQ(ierr);
  ierr = ISGetIndices( selected, &selected_idx );     CHKERRQ(ierr);
  ierr = ISGetIndices( llist_parent, &llist_idx );     CHKERRQ(ierr);
  for(ii=0,nLocalSelected=0;ii<kk;ii++){
    PetscInt lid = selected_idx[ii];
    /* filter out singletons */
    if( lid<nloc && llist_idx[lid] != -1) nLocalSelected++;
  }
  ierr = ISRestoreIndices( selected, &selected_idx );     CHKERRQ(ierr);
  ierr = ISRestoreIndices( llist_parent, &llist_idx );     CHKERRQ(ierr);

  /* create prolongator, create P matrix */
  ierr = MatCreateMPIAIJ( wcomm, 
                          nloc*bs, nLocalSelected*col_bs,
                          PETSC_DETERMINE, PETSC_DETERMINE,
                          data_cols, PETSC_NULL, data_cols, PETSC_NULL,
                          &Prol );
  CHKERRQ(ierr);
  
  /* can get all points "removed" */
  ierr =  MatGetSize( Prol, &kk, &ii ); CHKERRQ(ierr);
  if( ii==0 ) {
    if( verbose ) {
      PetscPrintf(wcomm,"[%d]%s no selected points on coarse grid\n",mype,__FUNCT__);
    }
    ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
    *a_P_out = PETSC_NULL;  /* out */
    PetscFunctionReturn(0);
  }
  if( verbose ) {
    PetscPrintf(PETSC_COMM_WORLD,"\t\t[%d]%s New grid %d nodes\n",mype,__FUNCT__,ii/bs);
  }
  ierr = MatGetOwnershipRangeColumn( Prol, &myCrs0, &kk ); CHKERRQ(ierr);
  myCrs0 = myCrs0/col_bs;

  /* create global vector of data in 'data_w_ghost' */
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
#endif
  if (npe > 1) { /*  */
    PetscReal *tmp_gdata,*tmp_ldata,*tp2;
    ierr = PetscMalloc( nloc*sizeof(PetscReal), &tmp_ldata ); CHKERRQ(ierr);
    for( jj = 0 ; jj < data_cols ; jj++ ){
      for( kk = 0 ; kk < bs ; kk++) {
        PetscInt ii,nnodes;
        const PetscReal *tp = pc_gamg->data + jj*bs*nloc + kk;
        for( ii = 0 ; ii < nloc ; ii++, tp += bs ){
          tmp_ldata[ii] = *tp;
        }
        ierr = getDataWithGhosts( Gmat, 1, tmp_ldata, &nnodes, &tmp_gdata );
        CHKERRQ(ierr);
        if(jj==0 && kk==0) { /* now I know how many todal nodes - allocate */
          ierr = PetscMalloc( nnodes*bs*data_cols*sizeof(PetscReal), &data_w_ghost ); CHKERRQ(ierr);
          nbnodes = bs*nnodes;
        }
        tp2 = data_w_ghost + jj*bs*nnodes + kk;
        for( ii = 0 ; ii < nnodes ; ii++, tp2 += bs ){
          *tp2 = tmp_gdata[ii];
        }
        ierr = PetscFree( tmp_gdata ); CHKERRQ(ierr);
      }
    }
    ierr = PetscFree( tmp_ldata ); CHKERRQ(ierr);
  }
  else {
    nbnodes = bs*nloc;
    data_w_ghost = (PetscReal*)pc_gamg->data;
  }
  
  /* get P0 */
  if( npe > 1 ){
    PetscReal *fid_glid_loc,*fiddata; 
    PetscInt nnodes;
    
    ierr = PetscMalloc( nloc*sizeof(PetscReal), &fid_glid_loc ); CHKERRQ(ierr);
    for(kk=0;kk<nloc;kk++) fid_glid_loc[kk] = (PetscReal)(my0+kk);
    ierr = getDataWithGhosts( Gmat, 1, fid_glid_loc, &nnodes, &fiddata );
    CHKERRQ(ierr);
    ierr = PetscMalloc( nnodes*sizeof(PetscInt), &flid_fgid ); CHKERRQ(ierr);
    for(kk=0;kk<nnodes;kk++) flid_fgid[kk] = (PetscInt)fiddata[kk];
    ierr = PetscFree( fiddata ); CHKERRQ(ierr);
    assert(nnodes==nbnodes/bs);
    ierr = PetscFree( fid_glid_loc ); CHKERRQ(ierr);
  }
  else {
    ierr = PetscMalloc( nloc*sizeof(PetscInt), &flid_fgid ); CHKERRQ(ierr);
    for(kk=0;kk<nloc;kk++) flid_fgid[kk] = my0 + kk;
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  {
    PetscReal *data_out;
    ierr = formProl0( selected, llist_parent, bs, data_cols, myCrs0, nbnodes,
                      data_w_ghost, flid_fgid, &data_out, Prol );
    CHKERRQ(ierr);
    ierr = PetscFree( pc_gamg->data ); CHKERRQ( ierr );
    pc_gamg->data = data_out;
    pc_gamg->data_cell_rows = data_cols; 
    pc_gamg->data_sz = data_cols*data_cols*nLocalSelected;
  }
 #if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif 
  if (npe > 1) ierr = PetscFree( data_w_ghost );      CHKERRQ(ierr);
  ierr = PetscFree( flid_fgid ); CHKERRQ(ierr);
  
  /* attach block size of columns */
  if( pc_gamg->col_bs_id == -1 ) {
    ierr = PetscObjectComposedDataRegister( &pc_gamg->col_bs_id ); assert(pc_gamg->col_bs_id != -1 );
  }
  ierr = PetscObjectComposedDataSetInt( (PetscObject)Prol, pc_gamg->col_bs_id, data_cols ); CHKERRQ(ierr);

  *a_P_out = Prol;  /* out */

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGoptprol_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
 In/Output Parameter:
   . a_P_out - prolongation operator to the next level
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGoptprol_AGG"
PetscErrorCode PCGAMGoptprol_AGG( PC pc,
                                  const Mat Amat,
                                  Mat *a_P
                                  )
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  const PetscInt verbose = pc_gamg->verbose;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;  
  PetscInt       jj;
  PetscMPIInt    mype,npe;
  Mat            Prol = *a_P;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);

  /* smooth P0 */
  for( jj = 0 ; jj < pc_gamg_agg->nsmooths ; jj++ ){
    Mat tMat; 
    Vec diag;
    PetscReal alpha, emax, emin;
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif
    if( jj == 0 ) {
      KSP eksp; 
      Vec bb, xx;  
      PC pc;
      ierr = MatGetVecs( Amat, &bb, 0 );         CHKERRQ(ierr);
      ierr = MatGetVecs( Amat, &xx, 0 );         CHKERRQ(ierr);
      {
        PetscRandom    rctx;
        ierr = PetscRandomCreate(wcomm,&rctx);CHKERRQ(ierr);
        ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
        ierr = VecSetRandom(bb,rctx);CHKERRQ(ierr);
        ierr = PetscRandomDestroy( &rctx ); CHKERRQ(ierr);
      }  
      ierr = KSPCreate(wcomm,&eksp);                            CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix( eksp, "est_");         CHKERRQ(ierr);
      ierr = KSPSetFromOptions( eksp );    CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero( eksp, PETSC_FALSE );    CHKERRQ(ierr);
      ierr = KSPSetOperators( eksp, Amat, Amat, SAME_NONZERO_PATTERN );
      CHKERRQ( ierr );
      ierr = KSPGetPC( eksp, &pc );                              CHKERRQ( ierr );
      ierr = PCSetType( pc, PCJACOBI ); CHKERRQ(ierr);  /* smoother */
      ierr = KSPSetTolerances(eksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10);
      CHKERRQ(ierr);
      ierr = KSPSetNormType( eksp, KSP_NORM_NONE );                 CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues( eksp,PETSC_TRUE );        CHKERRQ(ierr);
      
      /* solve - keep stuff out of logging */
      ierr = PetscLogEventDeactivate(KSP_Solve);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(PC_Apply);CHKERRQ(ierr);
      ierr = KSPSolve( eksp, bb, xx );                              CHKERRQ(ierr);
      ierr = PetscLogEventActivate(KSP_Solve);CHKERRQ(ierr);
      ierr = PetscLogEventActivate(PC_Apply);CHKERRQ(ierr);
      
      ierr = KSPComputeExtremeSingularValues( eksp, &emax, &emin ); CHKERRQ(ierr);
      if( verbose ) {
        PetscPrintf(wcomm,"\t\t\t%s smooth P0: max eigen=%e min=%e PC=%s\n",
                    __FUNCT__,emax,emin,PCJACOBI);
      }
      ierr = VecDestroy( &xx );       CHKERRQ(ierr);
      ierr = VecDestroy( &bb );       CHKERRQ(ierr);
      ierr = KSPDestroy( &eksp );     CHKERRQ(ierr);

      if( pc_gamg->emax_id == -1 ) {
        ierr = PetscObjectComposedDataRegister( &pc_gamg->emax_id );
        assert(pc_gamg->emax_id != -1 );
      }
      ierr = PetscObjectComposedDataSetScalar( (PetscObject)Amat, pc_gamg->emax_id, emax ); CHKERRQ(ierr);
    }

    /* smooth P1 := (I - omega/lam D^{-1}A)P0 */
    ierr = MatMatMult( Amat, Prol, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tMat );   CHKERRQ(ierr);
    ierr = MatGetVecs( Amat, &diag, 0 );    CHKERRQ(ierr);
    ierr = MatGetDiagonal( Amat, diag );    CHKERRQ(ierr); /* effectively PCJACOBI */
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
  }

  *a_P = Prol;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_AGG

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_AGG"
PetscErrorCode  PCCreateGAMG_AGG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_agg;

  PetscFunctionBegin;
  /* create sub context for SA */
  ierr = PetscNewLog( pc, PC_GAMG_AGG, &pc_gamg_agg ); CHKERRQ(ierr);
  assert(!pc_gamg->subctx);
  pc_gamg->subctx = pc_gamg_agg;
  
  pc->ops->setfromoptions = PCSetFromOptions_GAMG_AGG;
  pc->ops->destroy        = PCDestroy_AGG;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->graph = PCGAMGgraph_AGG;
  pc_gamg->coarsen = PCGAMGCoarsen_AGG;
  pc_gamg->prolongator = PCGAMGprolongator_AGG;
  pc_gamg->optprol = PCGAMGoptprol_AGG;
 
  pc_gamg->createdefaultdata = PCSetData_AGG;

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
                                            "PCSetCoordinates_C",
                                            "PCSetCoordinates_AGG",
                                            PCSetCoordinates_AGG);
  PetscFunctionReturn(0);
}
