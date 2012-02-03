/* 
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>
#include <private/kspimpl.h>

#include <assert.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt smooths;
} PC_GAMG_AGG;

#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetNSmooths_AGG"
/*@
   PCGAMGSetNSmooths_AGG - Set number of smoothing steps (1 is typical)

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_sa_nsmooths

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode  PCGAMGSetNSmooths_AGG(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetNSmooths_AGG_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGAMGSetNSmooths_AGG_GAMG"
PetscErrorCode PCGAMGSetNSmooths_AGG_GAMG(PC pc, PetscInt n)
{
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_sa = (PC_GAMG_AGG*)pc_gamg->subctx;
  
  PetscFunctionBegin;
  if(n>=0) pc_gamg_sa->smooths = n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*
   PCSetFromOptions_AGG

  Input Parameter:
   . pc - 
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_AGG"
PetscErrorCode PCSetFromOptions_AGG( PC pc )
{
  PetscErrorCode  ierr;
  PC_MG           *mg = (PC_MG*)pc->data;
  PC_GAMG         *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG      *pc_gamg_sa = (PC_GAMG_AGG*)pc_gamg->subctx;
  PetscBool        flag;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG-SA options"); CHKERRQ(ierr);
  {
    /* -pc_gamg_sa_nsmooths */
    pc_gamg_sa->smooths = 0;
    ierr = PetscOptionsInt("-pc_gamg_agg_nsmooths",
                           "smoothing steps for smoothed aggregation, usually 1 (0)",
                           "PCGAMGSetNSmooths_AGG",
                           pc_gamg_sa->smooths,
                           &pc_gamg_sa->smooths,
                           &flag); 
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
  /* call base class */
  ierr = PCSetFromOptions_GAMG( pc ); CHKERRQ(ierr);

  if( pc_gamg->verbose ) {
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
  PC_GAMG_AGG      *pc_gamg_sa = (PC_GAMG_AGG*)pc_gamg->subctx;
  
  PetscFunctionBegin;
  if( pc_gamg_sa ) {
    ierr = PetscFree(pc_gamg_sa);CHKERRQ(ierr);
    pc_gamg_sa = 0;
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
  if( coords && bs==1 ) pc_gamg->data_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if( coords ) pc_gamg->data_cols = (ndm==2 ? 3 : 6); /* elasticity */
  else pc_gamg->data_cols = bs; /* no data, force SA with constant null space vectors */
  pc_gamg->data_rows = bs;

  arrsz = nloc*pc_gamg->data_rows*pc_gamg->data_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (pc_gamg->data==0 || (pc_gamg->data_sz != arrsz)) {
    ierr = PetscFree( pc_gamg->data );  CHKERRQ(ierr);
    ierr = PetscMalloc((arrsz+1)*sizeof(PetscReal), &pc_gamg->data ); CHKERRQ(ierr);
  }
  for(kk=0;kk<arrsz;kk++)pc_gamg->data[kk] = -999999999.;
  pc_gamg->data[arrsz] = -99.;
  /* copy data in - column oriented */
  for(kk=0;kk<nloc;kk++){
    const PetscInt M = Iend - my0;
    PetscReal *data = &pc_gamg->data[kk*bs];
    if( pc_gamg->data_cols==1 ) *data = 1.0;
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
  assert(pc_gamg->data[arrsz] == -99.);

  pc_gamg->data_sz = arrsz;

  PetscFunctionReturn(0);
}
EXTERN_C_END


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
  PC_GAMG_AGG      *pc_gamg_sa;

  PetscFunctionBegin;
  /* create sub context for SA */
  ierr = PetscNewLog( pc, PC_GAMG_AGG, &pc_gamg_sa ); CHKERRQ(ierr);
  assert(!pc_gamg->subctx);
  pc_gamg->subctx = pc_gamg_sa;
  
  pc->ops->setfromoptions = PCSetFromOptions_AGG;
  pc->ops->destroy        = PCDestroy_AGG;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->createprolongator = PCGAMGcreateProl_AGG;
  pc_gamg->createdefaultdata = PCSetData_AGG;

  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
                                            "PCSetCoordinates_C",
                                            "PCSetCoordinates_AGG",
                                            PCSetCoordinates_AGG);
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
PetscErrorCode formProl0(IS selected, /* list of selected local ID, includes selected ghosts */
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
  PetscInt  Istart,Iend,nFineLoc,clid,flid,aggID,kk,jj,ii,nLocalSelected,ndone,nSelected;
  MPI_Comm       wcomm = ((PetscObject)a_Prol)->comm;
  PetscMPIInt    mype, npe;
  const PetscInt *selected_idx,*llist_idx;
  PetscReal      *out_data;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wcomm,&mype);CHKERRQ(ierr);
  ierr = MPI_Comm_size(wcomm,&npe);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( a_Prol, &Istart, &Iend );    CHKERRQ(ierr);
  nFineLoc = (Iend-Istart)/bs; assert((Iend-Istart)%bs==0);
  
  ierr = ISGetLocalSize( selected, &nSelected );        CHKERRQ(ierr);
  ierr = ISGetIndices( selected, &selected_idx );       CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<nSelected;kk++){
    PetscInt lid = selected_idx[kk];
    if(lid<nFineLoc) nLocalSelected++;
  }

  /* aloc space for coarse point data (output) */
#define DATA_OUT_STRIDE (nLocalSelected*nSAvec)
  ierr = PetscMalloc( (DATA_OUT_STRIDE*nSAvec+1)*sizeof(PetscReal), &out_data ); CHKERRQ(ierr);
  for(ii=0;ii<DATA_OUT_STRIDE*nSAvec+1;ii++) out_data[ii]=1.e300;
  *a_data_out = out_data; /* output - stride nLocalSelected*nSAvec */

  /* find points and set prolongation */
  ndone = 0;
  ierr = ISGetIndices( locals_llist, &llist_idx );      CHKERRQ(ierr);
  for( clid = 0 ; clid < nLocalSelected ; clid++ ){
    PetscInt cgid = my0crs + clid, cids[100];

    /* count agg */
    aggID = 0;
    flid = selected_idx[clid]; assert(flid != -1);
    do{
      aggID++;
    } while( (flid=llist_idx[flid]) != -1 );

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

      flid = selected_idx[clid];
      aggID = 0;
      do{
        /* copy in B_i matrix - column oriented */
        PetscReal *data = &data_in[flid*bs];
        for( kk = ii = 0; ii < bs ; ii++ ) {
          for( jj = 0; jj < N ; jj++ ) {
            qqc[jj*Mdata + aggID*bs + ii] = data[jj*data_stride + ii];
          }
        }

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
      for(kk=0;kk<N;kk++) cids[kk] = N*cgid + kk; /* global col IDs in P0 */
      ierr = MatSetValues(a_Prol,M,fids,N,cids,qqr,INSERT_VALUES); CHKERRQ(ierr);

      ierr = PetscFree( qqc );  CHKERRQ(ierr);
      ierr = PetscFree( qqr );  CHKERRQ(ierr);
      ierr = PetscFree( TAU );  CHKERRQ(ierr);
      ierr = PetscFree( WORK );  CHKERRQ(ierr);
      ierr = PetscFree( fids );  CHKERRQ(ierr);
    } /* scoping */
  } /* for all coarse nodes */
  assert(out_data[nSAvec*DATA_OUT_STRIDE]==1.e300);

/* ierr = MPI_Allreduce( &ndone, &ii, 1, MPIU_INT, MPIU_SUM, wcomm ); /\* synchronous version *\/ */
/* MatGetSize( a_Prol, &kk, &jj ); */
/* PetscPrintf(PETSC_COMM_WORLD," **** [%d]%s %d total done, N=%d (%d local done)\n",mype,__FUNCT__,ii,kk/bs,ndone); */

  ierr = ISRestoreIndices( selected, &selected_idx );     CHKERRQ(ierr);
  ierr = ISRestoreIndices( locals_llist, &llist_idx );     CHKERRQ(ierr);
  ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGcreateProl_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
   . data[nloc*data_sz(in)]
  Output Parameter:
   . a_P_out - prolongation operator to the next level
   . a_data_out - data of coarse grid points (num local columns in 'a_P_out')
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGcreateProl_AGG"
PetscErrorCode PCGAMGcreateProl_AGG( PC pc,
                                    const Mat Amat,
                                    const PetscReal data[],
                                    Mat *a_P_out,
                                    PetscReal **a_data_out
                                   )
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_sa = (PC_GAMG_AGG*)pc_gamg->subctx;
  const PetscInt verbose = pc_gamg->verbose;
  const PetscInt data_cols = pc_gamg->data_cols;
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,nloc,jj,kk,my0,nLocalSelected,NN,MM,bs_in;
  Mat            Prol, Gmat, AuxMat;
  PetscMPIInt    mype, npe;
  MPI_Comm       wcomm = ((PetscObject)Amat)->comm;
  IS             permIS, llist_1, selected_1;
  const PetscInt *selected_idx,col_bs=data_cols;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank( wcomm, &mype);  CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe);   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange( Amat, &Istart, &Iend ); CHKERRQ(ierr);
  ierr = MatGetSize( Amat, &MM, &NN );  CHKERRQ(ierr);
  ierr  = MatGetBlockSize( Amat, &bs_in ); CHKERRQ( ierr ); 
  nloc = (Iend-Istart)/bs_in; my0 = Istart/bs_in; assert((Iend-Istart)%bs_in==0);

  ierr = createGraph( pc, Amat, &Gmat, &AuxMat, &permIS ); CHKERRQ(ierr);

  /* SELECT COARSE POINTS */
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif

  ierr = maxIndSetAgg( permIS, Gmat, AuxMat, PETSC_TRUE, &selected_1, &llist_1 );
  CHKERRQ(ierr);
  ierr = MatDestroy( &AuxMat );  CHKERRQ(ierr); 

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = ISDestroy(&permIS); CHKERRQ(ierr);

  /* get 'nLocalSelected' */
  ierr = ISGetLocalSize( selected_1, &NN );        CHKERRQ(ierr);
  ierr = ISGetIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
  for(kk=0,nLocalSelected=0;kk<NN;kk++){
    PetscInt lid = selected_idx[kk];
    if(lid<nloc) nLocalSelected++;
  }
  ierr = ISRestoreIndices( selected_1, &selected_idx );     CHKERRQ(ierr);
  
  /* create prolongator, create P matrix */
  ierr = MatCreateMPIAIJ(wcomm, 
                         nloc*bs_in, nLocalSelected*col_bs,
                         PETSC_DETERMINE, PETSC_DETERMINE,
                         data_cols, PETSC_NULL, data_cols, PETSC_NULL,
                         &Prol );
  CHKERRQ(ierr);

  /* can get all points "removed" */
  ierr =  MatGetSize( Prol, &kk, &NN ); CHKERRQ(ierr);
  if( NN==0 ) {
    if( verbose ) {
      PetscPrintf(PETSC_COMM_WORLD,"[%d]%s no selected points on coarse grid\n",mype,__FUNCT__);
    }
    ierr = MatDestroy( &Prol );  CHKERRQ(ierr);
    ierr = ISDestroy( &llist_1 ); CHKERRQ(ierr);
    ierr = ISDestroy( &selected_1 ); CHKERRQ(ierr);
    ierr = MatDestroy( &Gmat );  CHKERRQ(ierr);    
    *a_P_out = PETSC_NULL;  /* out */
    PetscFunctionReturn(0);
  }

  { /* SA */
    PetscReal *data_w_ghost;
    PetscInt  myCrs0, nbnodes=0, *flid_fgid;
    
    /* create global vector of data in 'data_w_ghost' */
#if defined PETSC_USE_LOG
    ierr = PetscLogEventBegin(gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
#endif
    if (npe > 1) {
      PetscReal *tmp_gdata,*tmp_ldata,*tp2;
      
      ierr = PetscMalloc( nloc*sizeof(PetscReal), &tmp_ldata ); CHKERRQ(ierr);
      for( jj = 0 ; jj < data_cols ; jj++ ){
        for( kk = 0 ; kk < bs_in ; kk++) {
          PetscInt ii,nnodes;
          const PetscReal *tp = data + jj*bs_in*nloc + kk;
          for( ii = 0 ; ii < nloc ; ii++, tp += bs_in ){
            tmp_ldata[ii] = *tp;
          }
          ierr = getDataWithGhosts( Gmat, 1, tmp_ldata, &nnodes, &tmp_gdata );
          CHKERRQ(ierr);
          if(jj==0 && kk==0) { /* now I know how many todal nodes - allocate */
            ierr = PetscMalloc( nnodes*bs_in*data_cols*sizeof(PetscReal), &data_w_ghost ); CHKERRQ(ierr);
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
      data_w_ghost = (PetscReal*)data;
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
    
    ierr = formProl0(selected_1,llist_1,bs_in,data_cols,myCrs0,nbnodes,
                     data_w_ghost,flid_fgid,a_data_out,Prol);
    CHKERRQ(ierr);
    
    if (npe > 1) ierr = PetscFree( data_w_ghost );      CHKERRQ(ierr);
    ierr = PetscFree( flid_fgid ); CHKERRQ(ierr);
    
    /* smooth P0 */
    for( jj = 0 ; jj < pc_gamg_sa->smooths ; jj++ ){
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
        /* ierr = KSPSetType( eksp, KSPCG );                         CHKERRQ(ierr); */
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
          PetscPrintf(PETSC_COMM_WORLD,"\t\t\t%s smooth P0: max eigen=%e min=%e PC=%s\n",
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
  } /* scoping - SA code */
  
  /* attach block size of columns */
  if( pc_gamg->col_bs_id == -1 ) {
    ierr = PetscObjectComposedDataRegister( &pc_gamg->col_bs_id );
    assert(pc_gamg->col_bs_id != -1 );
  }
  ierr = PetscObjectComposedDataSetInt( (PetscObject)Prol, pc_gamg->col_bs_id, data_cols ); CHKERRQ(ierr);

  *a_P_out = Prol;  /* out */
  
  ierr = ISDestroy( &llist_1 ); CHKERRQ(ierr);
  ierr = ISDestroy( &selected_1 ); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
