/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

#include <assert.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt nsmooths;
  PetscBool sym_graph;
  PetscBool square_graph;
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

#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetSquareGraph"
/*@
   PCGAMGSetSquareGraph - 

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_square_graph

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetSquareGraph(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSquareGraph_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetSquareGraph_GAMG"
PetscErrorCode PCGAMGSetSquareGraph_GAMG(PC pc, PetscBool n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->square_graph = n;
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
                            "Set for asymmetric matrices",
                            "PCGAMGSetSymGraph",
                            pc_gamg_agg->sym_graph,
                            &pc_gamg_agg->sym_graph,
                            &flag); 
    CHKERRQ(ierr);

    /* -pc_gamg_square_graph */
    pc_gamg_agg->square_graph = PETSC_TRUE;
    ierr = PetscOptionsBool("-pc_gamg_square_graph",
                            "For faster coarsening and lower coarse grid complexity",
                            "PCGAMGSetSquareGraph",
                            pc_gamg_agg->square_graph,
                            &pc_gamg_agg->square_graph,
                            &flag); 
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
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
     - collective

   Input Parameter:
   .  pc - the preconditioner context
   . ndm - dimesion 
   . a_nloc - number of vertices local
   . coords - [a_nloc][ndm] - interleaved coordinate data: {x_0, y_0, z_0, x_1, y_1, ...}
*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_AGG"
PetscErrorCode PCSetCoordinates_AGG( PC pc, PetscInt ndm, PetscInt a_nloc, PetscReal *coords )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;
  PetscInt       arrsz,bs,kk,ii,jj,nloc;
  Mat            Amat = pc->pmat;
  MPI_Comm       wcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific( Amat, MAT_CLASSID, 1 );
  /* set 'bs' and 'nloc' */
  ierr = MatGetBlockSize( Amat, &bs );  CHKERRQ( ierr );
  if( a_nloc == -1 ) {
    PetscInt my0, Iend;
    /* stokes = PETSC_FALSE; */
    ierr  = MatGetOwnershipRange( Amat, &my0, &Iend ); CHKERRQ(ierr);
    nloc = (Iend-my0)/bs; 
    if((Iend-my0)%bs!=0) SETERRQ1(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Bad local size %d.",nloc);    
  }
  else {
    ierr = MPI_Allreduce( &a_nloc, &ii, 1, MPIU_INT, MPIU_SUM, wcomm ); CHKERRQ( ierr );
    ierr = MatGetSize( Amat, &kk, &jj );               CHKERRQ( ierr );    
    if( bs==1 && ii!=kk ) {
      /* stokes = PETSC_TRUE; */
      bs = ndm;
      nloc = a_nloc;
    }
    else{
      PetscInt       my0,Iend;
      /* stokes = PETSC_FALSE; */
      ierr  = MatGetOwnershipRange( Amat, &my0, &Iend ); CHKERRQ(ierr);
      nloc = (Iend-my0)/bs; 
      if((Iend-my0)%bs!=0) SETERRQ1(((PetscObject)Amat)->comm,PETSC_ERR_ARG_WRONG, "Bad local size %d.",nloc);
    }
  }

  /* SA: null space vectors */
  if( coords && bs==1 ) pc_gamg->data_cell_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if( coords ) pc_gamg->data_cell_cols = (ndm==2 ? 3 : 6); /* elasticity */
  else pc_gamg->data_cell_cols = bs; /* no data, force SA with constant null space vectors */
  pc_gamg->data_cell_rows = bs;

  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (pc_gamg->data==0 || (pc_gamg->data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->data );  CHKERRQ(ierr);
    ierr = PetscMalloc((arrsz+1)*sizeof(PetscReal), &pc_gamg->data ); CHKERRQ(ierr);
    /* !nul if if nloc==0 */
  }
  /* copy data in - column oriented */
  for(kk=0;kk<nloc;kk++){
    const PetscInt M = nloc*pc_gamg->data_cell_rows;
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
   Input/Output Parameter:
   . aggs_2 - linked list of aggs with gids )
*/
#undef __FUNCT__
#define __FUNCT__ "smoothAggs"
static PetscErrorCode smoothAggs( const Mat Gmat_2, /* base (squared) graph */
                                  const Mat Gmat_1, /* base graph */
                                  /* const IS selected_2, [nselected local] selected vertices */
                                  PetscCoarsenData *aggs_2 /* [nselected local] global ID of aggregate */
                                  )
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ    *matA_1, *matB_1=0, *matA_2, *matB_2=0;
  MPI_Comm       wcomm = ((PetscObject)Gmat_2)->comm;
  PetscMPIInt    mype,npe;
  PetscInt       lid,*ii,*idx,ix,Iend,my0,kk,n,j;
  Mat_MPIAIJ    *mpimat_2 = 0, *mpimat_1=0;
  const PetscInt nloc = Gmat_2->rmap->n;
  PetscScalar   *cpcol_1_state,*cpcol_2_state,*cpcol_2_par_orig,*lid_parent_gid;
  PetscInt      *lid_cprowID_1;
  NState        *lid_state;
  Vec            ghost_par_orig2;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat_1,&my0,&Iend);  CHKERRQ(ierr);

  if( PETSC_FALSE ) {
    PetscViewer viewer; char fname[32]; static int llev=0;
    sprintf(fname,"Gmat2_%d.m",llev++);
    PetscViewerASCIIOpen(wcomm,fname,&viewer);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Gmat_2, viewer ); CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
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

    ierr = PetscMalloc( nloc*sizeof(PetscInt), &lid_cprowID_1 ); CHKERRQ(ierr);
    for( lid = 0 ; lid < nloc ; lid++ ) lid_cprowID_1[lid] = -1;
    for (ix=0; ix<matB_1->compressedrow.nrows; ix++) {
      PetscInt lid = matB_1->compressedrow.rindex[ix];
      lid_cprowID_1[lid] = ix;
    }
  }
  else {
    matA_1 = (Mat_SeqAIJ*)Gmat_1->data;
    matA_2 = (Mat_SeqAIJ*)Gmat_2->data;
    lid_cprowID_1 = PETSC_NULL;
  }
  assert( matA_1 && !matA_1->compressedrow.use );
  assert( matB_1==0 || matB_1->compressedrow.use );
  assert( matA_2 && !matA_2->compressedrow.use );
  assert( matB_2==0 || matB_2->compressedrow.use );

  /* get state of locals and selected gid for deleted */
  ierr = PetscMalloc( nloc*sizeof(NState), &lid_state ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscScalar), &lid_parent_gid ); CHKERRQ(ierr);
  for( lid = 0 ; lid < nloc ; lid++ ) {
    lid_parent_gid[lid] = -1.0;
    lid_state[lid] = DELETED;
  }
  
  /* set lid_state */
  for( lid = 0 ; lid < nloc ; lid++ ) {
    PetscCDPos pos;
    ierr = PetscCDGetHeadPos(aggs_2,lid,&pos); CHKERRQ(ierr);
    if( pos ) {
      PetscInt gid1;
      ierr = PetscLLNGetID( pos, &gid1 ); CHKERRQ(ierr); assert(gid1==lid+my0);
      lid_state[lid] = gid1;
    }
  }

  /* map local to selected local, DELETED means a ghost owns it */
  for(lid=kk=0;lid<nloc;lid++){
    NState state = lid_state[lid];
    if( IS_SELECTED(state) ){
      PetscCDPos pos;
      ierr = PetscCDGetHeadPos(aggs_2,lid,&pos); CHKERRQ(ierr);
      while(pos){              
        PetscInt gid1; 
        ierr = PetscLLNGetID( pos, &gid1 ); CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(aggs_2,lid,&pos); CHKERRQ(ierr);
        
        if( gid1 >= my0 && gid1 < Iend ){
          lid_parent_gid[gid1-my0] = (PetscScalar)(lid + my0);
        }
      }
    }
  }
  /* get 'cpcol_1/2_state' & cpcol_2_par_orig - uses mpimat_1/2->lvec for temp space */
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
    /* get 'cpcol_2_state' */
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( mpimat_2->lvec, &cpcol_2_state ); CHKERRQ(ierr);
    /* get 'cpcol_2_par_orig' */
    for(kk=0,j=my0;kk<nloc;kk++,j++){
      PetscScalar v = (PetscScalar)lid_parent_gid[kk];
      ierr = VecSetValues( tempVec, 1, &j, &v, INSERT_VALUES );  CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecDuplicate( mpimat_2->lvec, &ghost_par_orig2 ); CHKERRQ(ierr); 
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghost_par_orig2,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, ghost_par_orig2,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( ghost_par_orig2, &cpcol_2_par_orig ); CHKERRQ(ierr);

    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr);
  } /* ismpi */

  /* doit */
  for(lid=0;lid<nloc;lid++){
    NState state = lid_state[lid];
    if( IS_SELECTED(state) ) {      
      /* steal locals */
      ii = matA_1->i; n = ii[lid+1] - ii[lid];
      idx = matA_1->j + ii[lid];
      for (j=0; j<n; j++) {
        PetscInt lidj = idx[j], sgid;
        NState statej = lid_state[lidj];
        if (statej==DELETED && (sgid=(PetscInt)PetscRealPart(lid_parent_gid[lidj])) != lid+my0) { /* steal local */
          lid_parent_gid[lidj] = (PetscScalar)(lid+my0); /* send this if sgid is not local */
          if( sgid >= my0 && sgid < Iend ){       /* I'm stealing this local from a local sgid */
            PetscInt hav=0,slid=sgid-my0,gidj=lidj+my0;
            PetscCDPos pos,last=PETSC_NULL;
            /* looking for local from local so id_llist_2 works */
            ierr = PetscCDGetHeadPos(aggs_2,slid,&pos); CHKERRQ(ierr);
            while(pos){              
              PetscInt gid; 
              ierr = PetscLLNGetID( pos, &gid ); CHKERRQ(ierr);
              if( gid == gidj ) {
                assert(last);
                ierr = PetscCDRemoveNextNode( aggs_2, slid, last ); CHKERRQ(ierr);
                ierr = PetscCDAppendNode( aggs_2, lid, pos );       CHKERRQ(ierr);
                hav = 1;
                break;
              }
              else last = pos;

              ierr = PetscCDGetNextPos(aggs_2,slid,&pos); CHKERRQ(ierr);
            }
            if(hav!=1){
              if(hav==0)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"found node %d times???",hav);
            }
          }
          else{            /* I'm stealing this local, owned by a ghost */
            assert(sgid==-1);
            ierr = PetscCDAppendID( aggs_2, lid, lidj+my0 );      CHKERRQ(ierr);
          }
        }
      } /* local neighbors */
    }
    else if( state == DELETED && lid_cprowID_1 ) {
      PetscInt sgidold = (PetscInt)PetscRealPart(lid_parent_gid[lid]);
      /* see if I have a selected ghost neighbor that will steal me */
      if( (ix=lid_cprowID_1[lid]) != -1 ){
        ii = matB_1->compressedrow.i; n = ii[ix+1] - ii[ix];
        idx = matB_1->j + ii[ix];
        for( j=0 ; j<n ; j++ ) {
          PetscInt cpid = idx[j];
          NState statej = (NState)PetscRealPart(cpcol_1_state[cpid]);
          if( IS_SELECTED(statej) && sgidold != (PetscInt)statej ) { /* ghost will steal this, remove from my list */
            lid_parent_gid[lid] = (PetscScalar)statej; /* send who selected */
            if( sgidold>=my0 && sgidold<Iend ) { /* this was mine */
              PetscInt hav=0,oldslidj=sgidold-my0;
              PetscCDPos pos,last=PETSC_NULL;
              /* remove from 'oldslidj' list */
              ierr = PetscCDGetHeadPos(aggs_2,oldslidj,&pos); CHKERRQ(ierr);
              while( pos ) {
                PetscInt gid;
                ierr = PetscLLNGetID( pos, &gid ); CHKERRQ(ierr);
                if( lid+my0 == gid ) {
                  /* id_llist_2[lastid] = id_llist_2[flid];   /\* remove lid from oldslidj list *\/ */
                  assert(last);
                  ierr = PetscCDRemoveNextNode( aggs_2, oldslidj, last ); CHKERRQ(ierr);
                  /* ghost (PetscScalar)statej will add this later */
                  hav = 1;
                  break;
                }
                else last = pos;

                ierr = PetscCDGetNextPos(aggs_2,oldslidj,&pos); CHKERRQ(ierr);
              }
              if(hav!=1){
                if(hav==0)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
                SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"found node %d times???",hav);
              }
            }
            else {
              /* ghosts remove this later */
            }
          }
        }
      }
    } /* selected/deleted */
  } /* node loop */

  if( isMPI ) {
    PetscScalar *cpcol_2_parent,*cpcol_2_gid;
    Vec          tempVec,ghostgids2,ghostparents2;
    PetscInt     cpid,nghost_2;
    GAMGHashTable gid_cpid;

    ierr = VecGetSize( mpimat_2->lvec, &nghost_2 );   CHKERRQ(ierr);
    ierr = MatGetVecs( Gmat_2, &tempVec, 0 );         CHKERRQ(ierr);

    /* get 'cpcol_2_parent' */
    for(kk=0,j=my0;kk<nloc;kk++,j++){
      ierr = VecSetValues( tempVec, 1, &j, &lid_parent_gid[kk], INSERT_VALUES );  CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecDuplicate( mpimat_2->lvec, &ghostparents2 ); CHKERRQ(ierr); 
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghostparents2,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, ghostparents2,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( ghostparents2, &cpcol_2_parent ); CHKERRQ(ierr);

    /* get 'cpcol_2_gid' */
    for(kk=0,j=my0;kk<nloc;kk++,j++){
      PetscScalar v = (PetscScalar)j;
      ierr = VecSetValues( tempVec, 1, &j, &v, INSERT_VALUES );  CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin( tempVec ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( tempVec ); CHKERRQ(ierr);
    ierr = VecDuplicate( mpimat_2->lvec, &ghostgids2 ); CHKERRQ(ierr); 
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghostgids2,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, ghostgids2,INSERT_VALUES,SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGetArray( ghostgids2, &cpcol_2_gid ); CHKERRQ(ierr);

    ierr = VecDestroy( &tempVec ); CHKERRQ(ierr);

    /* look for deleted ghosts and add to table */
    ierr = GAMGTableCreate( 2*nghost_2, &gid_cpid ); CHKERRQ(ierr);
    for( cpid = 0 ; cpid < nghost_2 ; cpid++ ) {
      NState state = (NState)PetscRealPart(cpcol_2_state[cpid]);
      if( state==DELETED ) {
        PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
        PetscInt sgid_old = (PetscInt)PetscRealPart(cpcol_2_par_orig[cpid]);
        if( sgid_old == -1 && sgid_new != -1 ) {
          PetscInt gid = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
          ierr = GAMGTableAdd( &gid_cpid, gid, cpid ); CHKERRQ(ierr);
        }
      }
    }

    /* look for deleted ghosts and see if they moved - remove it */
    for(lid=0;lid<nloc;lid++){
      NState state = lid_state[lid];
      if( IS_SELECTED(state) ){
        PetscCDPos pos,last=PETSC_NULL;        
        /* look for deleted ghosts and see if they moved */
        ierr = PetscCDGetHeadPos(aggs_2,lid,&pos); CHKERRQ(ierr);
        while(pos){              
          PetscInt gid; 
          ierr = PetscLLNGetID( pos, &gid ); CHKERRQ(ierr);

          if( gid < my0 || gid >= Iend ) {
            ierr = GAMGTableFind( &gid_cpid, gid, &cpid ); CHKERRQ(ierr);
            if( cpid != -1 ) {
              /* a moved ghost - */
              /* id_llist_2[lastid] = id_llist_2[flid];    /\* remove 'flid' from list *\/ */
              ierr = PetscCDRemoveNextNode( aggs_2, lid, last ); CHKERRQ(ierr);
            }
            else last = pos;
          }
          else last = pos;

          ierr = PetscCDGetNextPos(aggs_2,lid,&pos); CHKERRQ(ierr);
        } /* loop over list of deleted */
      } /* selected */
    }
    ierr = GAMGTableDestroy( &gid_cpid ); CHKERRQ(ierr);

    /* look at ghosts, see if they changed - and it */
    for( cpid = 0 ; cpid < nghost_2 ; cpid++ ){
      PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
      if( sgid_new >= my0 && sgid_new < Iend ) { /* this is mine */
        PetscInt gid = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
        PetscInt slid_new=sgid_new-my0,hav=0;
        PetscCDPos pos;
        /* search for this gid to see if I have it */
        ierr = PetscCDGetHeadPos(aggs_2,slid_new,&pos); CHKERRQ(ierr);
        while(pos){              
          PetscInt gidj; 
          ierr = PetscLLNGetID( pos, &gidj ); CHKERRQ(ierr);
          ierr = PetscCDGetNextPos(aggs_2,slid_new,&pos); CHKERRQ(ierr);
          
          if( gidj == gid ) { hav = 1; break; }
        }
        if( hav != 1 ){
          /* insert 'flidj' into head of llist */
          ierr = PetscCDAppendID( aggs_2, slid_new, gid );      CHKERRQ(ierr);
        }
      }
    }

    ierr = VecRestoreArray( mpimat_1->lvec, &cpcol_1_state ); CHKERRQ(ierr);
    ierr = VecRestoreArray( mpimat_2->lvec, &cpcol_2_state ); CHKERRQ(ierr);
    ierr = VecRestoreArray( ghostparents2, &cpcol_2_parent ); CHKERRQ(ierr);
    ierr = VecRestoreArray( ghostgids2, &cpcol_2_gid ); CHKERRQ(ierr);
    ierr = PetscFree( lid_cprowID_1 );  CHKERRQ(ierr);
    ierr = VecDestroy( &ghostgids2 ); CHKERRQ(ierr);
    ierr = VecDestroy( &ghostparents2 ); CHKERRQ(ierr);
    ierr = VecDestroy( &ghost_par_orig2 ); CHKERRQ(ierr);
  }

  ierr = PetscFree( lid_parent_gid );  CHKERRQ(ierr);
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
PetscErrorCode PCSetData_AGG( PC pc, Mat a_A )
{
  PetscErrorCode  ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  MatNullSpace mnull;

  PetscFunctionBegin;
  ierr = MatGetNearNullSpace( a_A, &mnull ); CHKERRQ(ierr);
  if( !mnull ) {
    ierr = PCSetCoordinates_AGG( pc, -1, -1, PETSC_NULL ); CHKERRQ(ierr);
  }
  else {
    PetscReal *nullvec;
    PetscBool has_const;
    PetscInt i,j,mlocal,nvec,bs;
    const Vec *vecs; const PetscScalar *v;
    ierr = MatGetLocalSize(a_A,&mlocal,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatNullSpaceGetVecs(mnull,&has_const,&nvec,&vecs);CHKERRQ(ierr);
     ierr  = MatGetBlockSize( a_A, &bs );               CHKERRQ( ierr );
    ierr = PetscMalloc((nvec+!!has_const)*mlocal*sizeof *nullvec,&nullvec);CHKERRQ(ierr);
    if (has_const) for (i=0; i<mlocal; i++) nullvec[i] = 1.0;
    for (i=0; i<nvec; i++) {
      ierr = VecGetArrayRead(vecs[i],&v);CHKERRQ(ierr);
      for (j=0; j<mlocal; j++) nullvec[(i+!!has_const)*mlocal + j] = PetscRealPart(v[j]);
      ierr = VecRestoreArrayRead(vecs[i],&v);CHKERRQ(ierr);
    }
    pc_gamg->data = nullvec;
    pc_gamg->data_cell_cols = (nvec+!!has_const);
    pc_gamg->data_cell_rows = bs;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 formProl0

   Input Parameter:
   . agg_llists - list of arrays with aggregates
   . bs - block size
   . nSAvec - column bs of new P
   . my0crs - global index of start of locals
   . data_stride - bs*(nloc nodes + ghost nodes)
   . data_in[data_stride*nSAvec] - local data on fine grid
   . flid_fgid[data_stride/bs] - make local to global IDs, includes ghosts in 'locals_llist'
  Output Parameter:
   . a_data_out - in with fine grid data (w/ghosts), out with coarse grid data
   . a_Prol - prolongation operator
*/
#undef __FUNCT__
#define __FUNCT__ "formProl0"
static PetscErrorCode formProl0(const PetscCoarsenData *agg_llists,/* list from selected vertices of aggregate unselected vertices */
                                const PetscInt bs,          /* (row) block size */
                                const PetscInt nSAvec,      /* column bs */
                                const PetscInt my0crs,      /* global index of start of locals */
                                const PetscInt data_stride, /* (nloc+nghost)*bs */
                                PetscReal      data_in[],   /* [data_stride][nSAvec] */
                                const PetscInt flid_fgid[], /* [data_stride/bs] */
                                PetscReal **a_data_out,
                                Mat a_Prol /* prolongation operator (output)*/
                                )
{
  PetscErrorCode ierr;
  PetscInt  Istart,my0,Iend,nloc,clid,flid,aggID,kk,jj,ii,mm,ndone,nSelected,minsz,nghosts,out_data_stride;
  MPI_Comm       wcomm = ((PetscObject)a_Prol)->comm;
  PetscMPIInt    mype, npe;
  PetscReal      *out_data;
  PetscCDPos         pos;
  GAMGHashTable  fgid_flid;

/* #define OUT_AGGS */
#ifdef OUT_AGGS
  static PetscInt llev = 0; char fname[32]; FILE *file = PETSC_NULL; PetscInt pM;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( a_Prol, &Istart, &Iend );    CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; my0 = Istart/bs; assert((Iend-Istart)%bs==0);
  Iend /= bs;
  nghosts = data_stride/bs - nloc;

  ierr = GAMGTableCreate( 2*nghosts, &fgid_flid ); CHKERRQ(ierr);
  for(kk=0;kk<nghosts;kk++) {
    ierr = GAMGTableAdd( &fgid_flid, flid_fgid[nloc+kk], nloc+kk ); CHKERRQ(ierr);
  }

#ifdef OUT_AGGS
  sprintf(fname,"aggs_%d_%d.m",llev++,mype);
  if(llev==1) {
    file = fopen(fname,"w");
  }
  MatGetSize( a_Prol, &pM, &jj );
#endif

  /* count selected -- same as number of cols of P */
  for(nSelected=mm=0;mm<nloc;mm++) {
    PetscBool ise;
    ierr = PetscCDEmptyAt( agg_llists, mm, &ise ); CHKERRQ(ierr);
    if( !ise ) nSelected++;
  }
  ierr = MatGetOwnershipRangeColumn( a_Prol, &ii, &jj ); CHKERRQ(ierr);
  assert((ii/nSAvec)==my0crs); assert(nSelected==(jj-ii)/nSAvec);

  /* aloc space for coarse point data (output) */
  out_data_stride = nSelected*nSAvec;
  ierr = PetscMalloc( out_data_stride*nSAvec*sizeof(PetscReal), &out_data ); CHKERRQ(ierr);
  for(ii=0;ii<out_data_stride*nSAvec;ii++) {
    out_data[ii]=1.e300;
  }
  *a_data_out = out_data; /* output - stride nSelected*nSAvec */

  /* find points and set prolongation */
  minsz = 100;
  ndone = 0;
  for( mm = clid = 0 ; mm < nloc ; mm++ ){
    ierr = PetscCDSizeAt( agg_llists, mm, &jj ); CHKERRQ(ierr);
    if( jj > 0 ) {
      const PetscInt lid = mm, cgid = my0crs + clid;
      PetscInt cids[100]; /* max bs */
      PetscBLASInt asz=jj,M=asz*bs,N=nSAvec,INFO;
      PetscBLASInt   Mdata=M+((N-M>0)?N-M:0),LDA=Mdata,LWORK=N*bs;
      PetscScalar    *qqc,*qqr,*TAU,*WORK;
      PetscInt       *fids;
      PetscReal      *data;
      /* count agg */
      if( asz<minsz ) minsz = asz;

      /* get block */
      ierr = PetscMalloc( (Mdata*N)*sizeof(PetscScalar), &qqc ); CHKERRQ(ierr);
      ierr = PetscMalloc( (M*N)*sizeof(PetscScalar), &qqr ); CHKERRQ(ierr);
      ierr = PetscMalloc( N*sizeof(PetscScalar), &TAU ); CHKERRQ(ierr);
      ierr = PetscMalloc( LWORK*sizeof(PetscScalar), &WORK ); CHKERRQ(ierr);
      ierr = PetscMalloc( M*sizeof(PetscInt), &fids ); CHKERRQ(ierr);

      aggID = 0;
      ierr = PetscCDGetHeadPos(agg_llists,lid,&pos); CHKERRQ(ierr);
      while(pos){              
        PetscInt gid1; 
        ierr = PetscLLNGetID( pos, &gid1 ); CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg_llists,lid,&pos); CHKERRQ(ierr);

        if( gid1 >= my0 && gid1 < Iend ) flid = gid1 - my0;
        else {
          ierr = GAMGTableFind( &fgid_flid, gid1, &flid ); CHKERRQ(ierr);
          assert(flid>=0);
        }
        /* copy in B_i matrix - column oriented */
        data = &data_in[flid*bs];
        for( kk = ii = 0; ii < bs ; ii++ ) {
          for( jj = 0; jj < N ; jj++ ) {
            PetscReal d = data[jj*data_stride + ii];
            qqc[jj*Mdata + aggID*bs + ii] = d;
          }
        }
#ifdef OUT_AGGS
        if(llev==1) {
          char str[] = "plot(%e,%e,'r*'), hold on,\n", col[] = "rgbkmc", sim[] = "*os+h>d<vx^";
          PetscInt MM,pi,pj;
          str[12] = col[(clid+17*mype)%6]; str[13] = sim[(clid+17*mype)%11];
          MM = (PetscInt)(PetscSqrtReal((PetscReal)pM));
          pj = gid1/MM; pi = gid1%MM;
          fprintf(file,str,(double)pi,(double)pj);
          /* fprintf(file,str,data[2*data_stride+1],-data[2*data_stride]); */
        }
#endif
        /* set fine IDs */
        for(kk=0;kk<bs;kk++) fids[aggID*bs + kk] = flid_fgid[flid]*bs + kk;
        
        aggID++;
      }

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
            assert(data[jj*out_data_stride + ii] == 1.e300);
            if( ii <= jj ) data[jj*out_data_stride + ii] = PetscRealPart(qqc[jj*Mdata + ii]);
	    else data[jj*out_data_stride + ii] = 0.;
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
      clid++;
    } /* coarse agg */
  } /* for all fine nodes */
  ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

/* ierr = MPI_Allreduce( &ndone, &ii, 1, MPIU_INT, MPIU_SUM, wcomm ); */
/* MatGetSize( a_Prol, &kk, &jj ); */
/* ierr = MPI_Allreduce( &minsz, &jj, 1, MPIU_INT, MPIU_MIN, wcomm ); */
/* PetscPrintf(wcomm," **** [%d]%s %d total done, %d nodes (%d local done), min agg. size = %d\n",mype,__FUNCT__,ii,kk/bs,ndone,jj); */

#ifdef OUT_AGGS
  if(llev==1) fclose(file);
#endif
  ierr = GAMGTableDestroy( &fgid_flid ); CHKERRQ(ierr);

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
  Mat            Gmat;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  PetscBool  set,flg,symm;

  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGGgraph_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);

  ierr = MatIsSymmetricKnown(Amat, &set, &flg);        CHKERRQ(ierr);
  symm = (PetscBool)(pc_gamg_agg->sym_graph || !(set && flg));

  ierr  = PCGAMGCreateGraph( Amat, &Gmat ); CHKERRQ( ierr );
  ierr  = PCGAMGFilterGraph( &Gmat, vfilter, symm, verbose ); CHKERRQ( ierr );
  
  *a_Gmat = Gmat;

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGGgraph_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCoarsen_AGG

  Input Parameter:
   . a_pc - this
  Input/Output Parameter:
   . a_Gmat1 - graph on this fine level - coarsening can change this (squares it)
  Output Parameter:
   . agg_lists - list of aggregates
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_AGG"
PetscErrorCode PCGAMGCoarsen_AGG( PC a_pc,
                                  Mat *a_Gmat1,
                                  PetscCoarsenData **agg_lists
                                  )
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat             mat,Gmat2, Gmat1 = *a_Gmat1; /* squared graph */
  IS              perm;
  PetscInt        Ii,nloc,bs,n,m;
  PetscInt *permute;
  PetscBool *bIndexSet;
  MatCoarsen crs;
  MPI_Comm        wcomm = ((PetscObject)Gmat1)->comm;
  PetscMPIInt     mype,npe;

  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGCoarsen_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);
  ierr = MatGetLocalSize( Gmat1, &n, &m ); CHKERRQ(ierr);
  ierr = MatGetBlockSize( Gmat1, &bs ); CHKERRQ(ierr); assert(bs==1);
  nloc = n/bs;
  
  if( pc_gamg_agg->square_graph ) {
    ierr = MatTransposeMatMult( Gmat1, Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2 );
    CHKERRQ(ierr);
  }
  else Gmat2 = Gmat1;

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
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MatCoarsenCreate( wcomm, &crs ); CHKERRQ(ierr);
  /* ierr = MatCoarsenSetType( crs, MATCOARSENMIS ); CHKERRQ(ierr); */
  ierr = MatCoarsenSetFromOptions( crs ); CHKERRQ(ierr);
  ierr = MatCoarsenSetGreedyOrdering( crs, perm ); CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency( crs, Gmat2 ); CHKERRQ(ierr);
  ierr = MatCoarsenSetVerbose( crs, pc_gamg->verbose ); CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs( crs, PETSC_TRUE ); CHKERRQ(ierr);
  ierr = MatCoarsenApply( crs ); CHKERRQ(ierr);
  ierr = MatCoarsenGetData( crs, agg_lists ); CHKERRQ(ierr); /* output */
  ierr = MatCoarsenDestroy( &crs ); CHKERRQ(ierr);

  ierr = ISDestroy( &perm );                    CHKERRQ(ierr);
  ierr = PetscFree( permute );  CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  /* smooth aggs */
  if( Gmat2 != Gmat1 ) {
    const PetscCoarsenData *llist = *agg_lists;
    ierr = smoothAggs( Gmat2, Gmat1, *agg_lists ); CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat1 );  CHKERRQ(ierr);
    *a_Gmat1 = Gmat2; /* output */
    ierr = PetscCDGetMat( llist, &mat );  CHKERRQ(ierr);
    if(mat) SETERRQ(wcomm,PETSC_ERR_ARG_WRONG, "Auxilary matrix with squared graph????");
  }
  else {
    const PetscCoarsenData *llist = *agg_lists;
    /* see if we have a matrix that takes pecedence (returned from MatCoarsenAppply) */
    ierr = PetscCDGetMat( llist, &mat );   CHKERRQ(ierr);
    if( mat ) {
      ierr = MatDestroy( &Gmat1 );  CHKERRQ(ierr);
      *a_Gmat1 = mat; /* output */
    }
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGCoarsen_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCGAMGProlongator_AGG
 
 Input Parameter:
 . pc - this
 . Amat - matrix on this fine level
 . Graph - used to get ghost data for nodes in 
 . agg_lists - list of aggregates
 Output Parameter:
 . a_P_out - prolongation operator to the next level
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_AGG"
PetscErrorCode PCGAMGProlongator_AGG( PC pc,
                                      const Mat Amat,
                                      const Mat Gmat,
                                      PetscCoarsenData *agg_lists,
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
  const PetscInt col_bs=data_cols;
  PetscReal      *data_w_ghost;
  PetscInt       myCrs0, nbnodes=0, *flid_fgid;

  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  ierr  = MatGetBlockSize( Amat, &bs ); CHKERRQ( ierr ); 
  nloc = (Iend-Istart)/bs; my0 = Istart/bs; assert((Iend-Istart)%bs==0);

  /* get 'nLocalSelected' */
  for( ii=0, nLocalSelected = 0 ; ii < nloc ; ii++ ){
    PetscBool ise;
    /* filter out singletons 0 or 1? */
    ierr = PetscCDEmptyAt( agg_lists, ii, &ise ); CHKERRQ(ierr);
    if( !ise ) nLocalSelected++;
  }

  /* create prolongator, create P matrix */
  ierr = MatCreateAIJ( wcomm,
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
    PetscPrintf(wcomm,"\t\t[%d]%s New grid %d nodes\n",mype,__FUNCT__,ii/col_bs);
  }
  ierr = MatGetOwnershipRangeColumn( Prol, &myCrs0, &kk ); CHKERRQ(ierr);

  assert((kk-myCrs0)%col_bs==0);
  myCrs0 = myCrs0/col_bs; 
  assert((kk/col_bs-myCrs0)==nLocalSelected);

  /* create global vector of data in 'data_w_ghost' */
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
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
        ierr = PCGAMGGetDataWithGhosts( Gmat, 1, tmp_ldata, &nnodes, &tmp_gdata );
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
    ierr = PCGAMGGetDataWithGhosts( Gmat, 1, fid_glid_loc, &nnodes, &fiddata );
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
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  {
    PetscReal *data_out = PETSC_NULL;
    ierr = formProl0( agg_lists, bs, data_cols, myCrs0, nbnodes,
                      data_w_ghost, flid_fgid, &data_out, Prol );
    CHKERRQ(ierr);
    ierr = PetscFree( pc_gamg->data ); CHKERRQ( ierr );
    pc_gamg->data = data_out;
    pc_gamg->data_cell_rows = data_cols;
    pc_gamg->data_sz = data_cols*data_cols*nLocalSelected;
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  if (npe > 1) ierr = PetscFree( data_w_ghost );      CHKERRQ(ierr);
  ierr = PetscFree( flid_fgid ); CHKERRQ(ierr);
  
  /* attach block size of columns */
  if( pc_gamg->col_bs_id == -1 ) {
    ierr = PetscObjectComposedDataRegister( &pc_gamg->col_bs_id ); assert(pc_gamg->col_bs_id != -1 );
  }
  ierr = PetscObjectComposedDataSetInt( (PetscObject)Prol, pc_gamg->col_bs_id, data_cols ); CHKERRQ(ierr);

  *a_P_out = Prol;  /* out */
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGOptprol_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
 In/Output Parameter:
   . a_P_out - prolongation operator to the next level
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGOptprol_AGG"
PetscErrorCode PCGAMGOptprol_AGG( PC pc,
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
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGOptprol_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);

  /* smooth P0 */
  for( jj = 0 ; jj < pc_gamg_agg->nsmooths ; jj++ ){
    Mat tMat; 
    Vec diag;
    PetscReal alpha, emax, emin;
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
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
      ierr = KSPAppendOptionsPrefix( eksp, "gamg_est_");         CHKERRQ(ierr);
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
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGOptprol_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
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
  pc_gamg->prolongator = PCGAMGProlongator_AGG;
  pc_gamg->optprol = PCGAMGOptprol_AGG;
 
  pc_gamg->createdefaultdata = PCSetData_AGG;

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
                                            "PCSetCoordinates_C",
                                            "PCSetCoordinates_AGG",
                                            PCSetCoordinates_AGG);
  PetscFunctionReturn(0);
}
