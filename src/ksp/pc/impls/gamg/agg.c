/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petscblaslapack.h>
#include <petscdm.h>
#include <petsc/private/kspimpl.h>

typedef struct {
  PetscInt  nsmooths;
  PetscBool symmetrize_graph;
  PetscInt  aggressive_coarsening_levels; // number of aggressive coarsening levels (square or MISk)
} PC_GAMG_AGG;

/*@
   PCGAMGSetNSmooths - Set number of smoothing steps (1 is typical)

   Logically Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_agg_nsmooths <nsmooth, default=1> - number of smoothing steps to use with smooth aggregation

   Level: intermediate

.seealso: `()`
@*/
PetscErrorCode PCGAMGSetNSmooths(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n,2);
  PetscTryMethod(pc,"PCGAMGSetNSmooths_C",(PC,PetscInt),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetNSmooths_AGG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->nsmooths = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetSymmetrizeGraph - Symmetrize the graph before computing the aggregation. Some algorithms require the graph be symmetric

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -pc_gamg_symmetrize_graph <true,default=false> - symmetrize the graph before computing the aggregation

   Level: intermediate

.seealso: `PCGAMGSetAggressiveLevels()`
@*/
PetscErrorCode PCGAMGSetSymmetrizeGraph(PC pc, PetscBool n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,n,2);
  PetscTryMethod(pc,"PCGAMGSetSymmetrizeGraph_C",(PC,PetscBool),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetSymmetrizeGraph_AGG(PC pc, PetscBool n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->symmetrize_graph = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetAggressiveLevels -  Aggressive coarsening on first n levels

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - 0, 1 or more

   Options Database Key:
.  -pc_gamg_aggressive_coarsening <n,default = 1> - Number of levels to square the graph on before aggregating it

   Level: intermediate

.seealso: `PCGAMGSetSymmetrizeGraph()`, `PCGAMGSetThreshold()`
@*/
PetscErrorCode PCGAMGSetAggressiveLevels(PC pc, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n,2);
  PetscTryMethod(pc,"PCGAMGSetAggressiveLevels_C",(PC,PetscInt),(pc,n));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetAggressiveLevels_AGG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->aggressive_coarsening_levels = n;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_GAMG_AGG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"GAMG-AGG options");
  {
    PetscBool flg;
    PetscCall(PetscOptionsInt("-pc_gamg_agg_nsmooths","smoothing steps for smoothed aggregation, usually 1","PCGAMGSetNSmooths",pc_gamg_agg->nsmooths,&pc_gamg_agg->nsmooths,NULL));
    PetscCall(PetscOptionsBool("-pc_gamg_symmetrize_graph","Set for asymmetric matrices","PCGAMGSetSymmetrizeGraph",pc_gamg_agg->symmetrize_graph,&pc_gamg_agg->symmetrize_graph,NULL));
    pc_gamg_agg->aggressive_coarsening_levels = 1;
    PetscCall(PetscOptionsInt("-pc_gamg_square_graph","Number of aggressive coarsening (MIS-2) levels from finest (alias for -pc_gamg_aggressive_coarsening, deprecated)","PCGAMGSetAggressiveLevels",pc_gamg_agg->aggressive_coarsening_levels,&pc_gamg_agg->aggressive_coarsening_levels,&flg));
    if (!flg) {
      PetscCall(PetscOptionsInt("-pc_gamg_aggressive_coarsening","Number of aggressive coarsening (MIS-2) levels from finest","PCGAMGSetAggressiveLevels",pc_gamg_agg->aggressive_coarsening_levels,&pc_gamg_agg->aggressive_coarsening_levels,NULL));
    } else {
      PetscCall(PetscOptionsInt("-pc_gamg_aggressive_coarsening","Number of aggressive coarsening (MIS-2) levels from finest","PCGAMGSetAggressiveLevels",pc_gamg_agg->aggressive_coarsening_levels,&pc_gamg_agg->aggressive_coarsening_levels,&flg));
      if (flg) PetscCall(PetscInfo(pc,"Warning: both -pc_gamg_square_graph and -pc_gamg_aggressive_coarsening are used. -pc_gamg_square_graph is deprecated, Number of aggressive levels is %d\n",(int)pc_gamg_agg->aggressive_coarsening_levels));
    }
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode PCDestroy_GAMG_AGG(PC pc)
{
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  PetscCall(PetscFree(pc_gamg->subctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetNSmooths_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetSymmetrizeGraph_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetAggressiveLevels_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_AGG
     - collective

   Input Parameter:
   . pc - the preconditioner context
   . ndm - dimesion of data (used for dof/vertex for Stokes)
   . a_nloc - number of vertices local
   . coords - [a_nloc][ndm] - interleaved coordinate data: {x_0, y_0, z_0, x_1, y_1, ...}
*/

static PetscErrorCode PCSetCoordinates_AGG(PC pc, PetscInt ndm, PetscInt a_nloc, PetscReal *coords)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscInt       arrsz,kk,ii,jj,nloc,ndatarows,ndf;
  Mat            mat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  nloc = a_nloc;

  /* SA: null space vectors */
  PetscCall(MatGetBlockSize(mat, &ndf)); /* this does not work for Stokes */
  if (coords && ndf==1) pc_gamg->data_cell_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if (coords) {
    PetscCheck(ndm <= ndf,PETSC_COMM_SELF,PETSC_ERR_PLIB,"degrees of motion %" PetscInt_FMT " > block size %" PetscInt_FMT,ndm,ndf);
    pc_gamg->data_cell_cols = (ndm==2 ? 3 : 6); /* displacement elasticity */
    if (ndm != ndf) {
      PetscCheck(pc_gamg->data_cell_cols == ndf,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Don't know how to create null space for ndm=%" PetscInt_FMT ", ndf=%" PetscInt_FMT ".  Use MatSetNearNullSpace().",ndm,ndf);
    }
  } else pc_gamg->data_cell_cols = ndf; /* no data, force SA with constant null space vectors */
  pc_gamg->data_cell_rows = ndatarows = ndf;
  PetscCheck(pc_gamg->data_cell_cols > 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"pc_gamg->data_cell_cols %" PetscInt_FMT " <= 0",pc_gamg->data_cell_cols);
  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;

  if (!pc_gamg->data || (pc_gamg->data_sz != arrsz)) {
    PetscCall(PetscFree(pc_gamg->data));
    PetscCall(PetscMalloc1(arrsz+1, &pc_gamg->data));
  }
  /* copy data in - column oriented */
  for (kk=0; kk<nloc; kk++) {
    const PetscInt M     = nloc*pc_gamg->data_cell_rows; /* stride into data */
    PetscReal      *data = &pc_gamg->data[kk*ndatarows]; /* start of cell */
    if (pc_gamg->data_cell_cols==1) *data = 1.0;
    else {
      /* translational modes */
      for (ii=0;ii<ndatarows;ii++) {
        for (jj=0;jj<ndatarows;jj++) {
          if (ii==jj)data[ii*M + jj] = 1.0;
          else data[ii*M + jj] = 0.0;
        }
      }

      /* rotational modes */
      if (coords) {
        if (ndm == 2) {
          data   += 2*M;
          data[0] = -coords[2*kk+1];
          data[1] =  coords[2*kk];
        } else {
          data   += 3*M;
          data[0] = 0.0;             data[M+0] =  coords[3*kk+2]; data[2*M+0] = -coords[3*kk+1];
          data[1] = -coords[3*kk+2]; data[M+1] = 0.0;             data[2*M+1] =  coords[3*kk];
          data[2] =  coords[3*kk+1]; data[M+2] = -coords[3*kk];   data[2*M+2] = 0.0;
        }
      }
    }
  }
  pc_gamg->data_sz = arrsz;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetData_AGG - called if data is not set with PCSetCoordinates.
      Looks in Mat for near null space.
      Does not work for Stokes

  Input Parameter:
   . pc -
   . a_A - matrix to get (near) null space out of.
*/
static PetscErrorCode PCSetData_AGG(PC pc, Mat a_A)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  MatNullSpace   mnull;

  PetscFunctionBegin;
  PetscCall(MatGetNearNullSpace(a_A, &mnull));
  if (!mnull) {
    DM dm;
    PetscCall(PCGetDM(pc, &dm));
    if (!dm) {
      PetscCall(MatGetDM(a_A, &dm));
    }
    if (dm) {
      PetscObject deformation;
      PetscInt    Nf;

      PetscCall(DMGetNumFields(dm, &Nf));
      if (Nf) {
        PetscCall(DMGetField(dm, 0, NULL, &deformation));
        PetscCall(PetscObjectQuery((PetscObject)deformation,"nearnullspace",(PetscObject*)&mnull));
        if (!mnull) {
          PetscCall(PetscObjectQuery((PetscObject)deformation,"nullspace",(PetscObject*)&mnull));
        }
      }
    }
  }

  if (!mnull) {
    PetscInt bs,NN,MM;
    PetscCall(MatGetBlockSize(a_A, &bs));
    PetscCall(MatGetLocalSize(a_A, &MM, &NN));
    PetscCheck(MM % bs == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MM %" PetscInt_FMT " must be divisible by bs %" PetscInt_FMT,MM,bs);
    PetscCall(PCSetCoordinates_AGG(pc, bs, MM/bs, NULL));
  } else {
    PetscReal         *nullvec;
    PetscBool         has_const;
    PetscInt          i,j,mlocal,nvec,bs;
    const Vec         *vecs;
    const PetscScalar *v;

    PetscCall(MatGetLocalSize(a_A,&mlocal,NULL));
    PetscCall(MatNullSpaceGetVecs(mnull, &has_const, &nvec, &vecs));
    pc_gamg->data_sz = (nvec+!!has_const)*mlocal;
    PetscCall(PetscMalloc1((nvec+!!has_const)*mlocal,&nullvec));
    if (has_const) for (i=0; i<mlocal; i++) nullvec[i] = 1.0;
    for (i=0; i<nvec; i++) {
      PetscCall(VecGetArrayRead(vecs[i],&v));
      for (j=0; j<mlocal; j++) nullvec[(i+!!has_const)*mlocal + j] = PetscRealPart(v[j]);
      PetscCall(VecRestoreArrayRead(vecs[i],&v));
    }
    pc_gamg->data           = nullvec;
    pc_gamg->data_cell_cols = (nvec+!!has_const);
    PetscCall(MatGetBlockSize(a_A, &bs));
    pc_gamg->data_cell_rows = bs;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
  formProl0 - collect null space data for each aggregate, do QR, put R in coarse grid data and Q in P_0

  Input Parameter:
   . agg_llists - list of arrays with aggregates -- list from selected vertices of aggregate unselected vertices
   . bs - row block size
   . nSAvec - column bs of new P
   . my0crs - global index of start of locals
   . data_stride - bs*(nloc nodes + ghost nodes) [data_stride][nSAvec]
   . data_in[data_stride*nSAvec] - local data on fine grid
   . flid_fgid[data_stride/bs] - make local to global IDs, includes ghosts in 'locals_llist'

  Output Parameter:
   . a_data_out - in with fine grid data (w/ghosts), out with coarse grid data
   . a_Prol - prolongation operator

*/
static PetscErrorCode formProl0(PetscCoarsenData *agg_llists,PetscInt bs,PetscInt nSAvec,PetscInt my0crs,PetscInt data_stride,PetscReal data_in[],const PetscInt flid_fgid[],PetscReal **a_data_out,Mat a_Prol)
{
  PetscInt        Istart,my0,Iend,nloc,clid,flid = 0,aggID,kk,jj,ii,mm,nSelected,minsz,nghosts,out_data_stride;
  MPI_Comm        comm;
  PetscReal       *out_data;
  PetscCDIntNd    *pos;
  PCGAMGHashTable fgid_flid;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)a_Prol,&comm));
  PetscCall(MatGetOwnershipRange(a_Prol, &Istart, &Iend));
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  PetscCheck((Iend-Istart) % bs == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Iend %" PetscInt_FMT " - Istart %" PetscInt_FMT " must be divisible by bs %" PetscInt_FMT,Iend,Istart,bs);
  Iend   /= bs;
  nghosts = data_stride/bs - nloc;

  PetscCall(PCGAMGHashTableCreate(2*nghosts+1, &fgid_flid));
  for (kk=0; kk<nghosts; kk++) {
    PetscCall(PCGAMGHashTableAdd(&fgid_flid, flid_fgid[nloc+kk], nloc+kk));
  }

  /* count selected -- same as number of cols of P */
  for (nSelected=mm=0; mm<nloc; mm++) {
    PetscBool ise;
    PetscCall(PetscCDEmptyAt(agg_llists, mm, &ise));
    if (!ise) nSelected++;
  }
  PetscCall(MatGetOwnershipRangeColumn(a_Prol, &ii, &jj));
  PetscCheck((ii/nSAvec) == my0crs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ii %" PetscInt_FMT " /nSAvec %" PetscInt_FMT "  != my0crs %" PetscInt_FMT,ii,nSAvec,my0crs);
  PetscCheck(nSelected == (jj-ii)/nSAvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"nSelected %" PetscInt_FMT " != (jj %" PetscInt_FMT " - ii %" PetscInt_FMT ")/nSAvec %" PetscInt_FMT,nSelected,jj,ii,nSAvec);

  /* aloc space for coarse point data (output) */
  out_data_stride = nSelected*nSAvec;

  PetscCall(PetscMalloc1(out_data_stride*nSAvec, &out_data));
  for (ii=0;ii<out_data_stride*nSAvec;ii++) out_data[ii]=PETSC_MAX_REAL;
  *a_data_out = out_data; /* output - stride nSelected*nSAvec */

  /* find points and set prolongation */
  minsz = 100;
  for (mm = clid = 0; mm < nloc; mm++) {
    PetscCall(PetscCDSizeAt(agg_llists, mm, &jj));
    if (jj > 0) {
      const PetscInt lid = mm, cgid = my0crs + clid;
      PetscInt       cids[100]; /* max bs */
      PetscBLASInt   asz  =jj,M=asz*bs,N=nSAvec,INFO;
      PetscBLASInt   Mdata=M+((N-M>0) ? N-M : 0),LDA=Mdata,LWORK=N*bs;
      PetscScalar    *qqc,*qqr,*TAU,*WORK;
      PetscInt       *fids;
      PetscReal      *data;

      /* count agg */
      if (asz<minsz) minsz = asz;

      /* get block */
      PetscCall(PetscMalloc5(Mdata*N, &qqc,M*N, &qqr,N, &TAU,LWORK, &WORK,M, &fids));

      aggID = 0;
      PetscCall(PetscCDGetHeadPos(agg_llists,lid,&pos));
      while (pos) {
        PetscInt gid1;
        PetscCall(PetscCDIntNdGetID(pos, &gid1));
        PetscCall(PetscCDGetNextPos(agg_llists,lid,&pos));

        if (gid1 >= my0 && gid1 < Iend) flid = gid1 - my0;
        else {
          PetscCall(PCGAMGHashTableFind(&fgid_flid, gid1, &flid));
          PetscCheck(flid >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot find gid1 in table");
        }
        /* copy in B_i matrix - column oriented */
        data = &data_in[flid*bs];
        for (ii = 0; ii < bs; ii++) {
          for (jj = 0; jj < N; jj++) {
            PetscReal d = data[jj*data_stride + ii];
            qqc[jj*Mdata + aggID*bs + ii] = d;
          }
        }
        /* set fine IDs */
        for (kk=0; kk<bs; kk++) fids[aggID*bs + kk] = flid_fgid[flid]*bs + kk;
        aggID++;
      }

      /* pad with zeros */
      for (ii = asz*bs; ii < Mdata; ii++) {
        for (jj = 0; jj < N; jj++, kk++) {
          qqc[jj*Mdata + ii] = .0;
        }
      }

      /* QR */
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Mdata, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      PetscCall(PetscFPTrapPop());
      PetscCheck(INFO == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"xGEQRF error");
      /* get R - column oriented - output B_{i+1} */
      {
        PetscReal *data = &out_data[clid*nSAvec];
        for (jj = 0; jj < nSAvec; jj++) {
          for (ii = 0; ii < nSAvec; ii++) {
            PetscCheck(data[jj*out_data_stride + ii] == PETSC_MAX_REAL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"data[jj*out_data_stride + ii] != %e",(double)PETSC_MAX_REAL);
           if (ii <= jj) data[jj*out_data_stride + ii] = PetscRealPart(qqc[jj*Mdata + ii]);
           else data[jj*out_data_stride + ii] = 0.;
          }
        }
      }

      /* get Q - row oriented */
      PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Mdata, &N, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      PetscCheck(INFO == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"xORGQR error arg %" PetscBLASInt_FMT,-INFO);

      for (ii = 0; ii < M; ii++) {
        for (jj = 0; jj < N; jj++) {
          qqr[N*ii + jj] = qqc[jj*Mdata + ii];
        }
      }

      /* add diagonal block of P0 */
      for (kk=0; kk<N; kk++) {
        cids[kk] = N*cgid + kk; /* global col IDs in P0 */
      }
      PetscCall(MatSetValues(a_Prol,M,fids,N,cids,qqr,INSERT_VALUES));
      PetscCall(PetscFree5(qqc,qqr,TAU,WORK,fids));
      clid++;
    } /* coarse agg */
  } /* for all fine nodes */
  PetscCall(MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY));
  PetscCall(PCGAMGHashTableDestroy(&fgid_flid));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_GAMG_AGG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer,"      AGG specific options\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer,"        Symmetric graph %s\n",pc_gamg_agg->symmetrize_graph ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer,"        Number of levels to square graph %d\n",(int)pc_gamg_agg->aggressive_coarsening_levels));
  PetscCall(PetscViewerASCIIPrintf(viewer,"        Number smoothing steps %d\n",(int)pc_gamg_agg->nsmooths));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGGraph_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level

  Output Parameter:
   . a_Gmat -
*/
static PetscErrorCode PCGAMGGraph_AGG(PC pc,Mat Amat,Mat *a_Gmat)
{
  PC_MG                     *mg          = (PC_MG*)pc->data;
  PC_GAMG                   *pc_gamg     = (PC_GAMG*)mg->innerctx;
  const PetscReal           vfilter      = pc_gamg->threshold[pc_gamg->current_level];
  PC_GAMG_AGG               *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat                       Gmat,F=NULL;
  MPI_Comm                  comm;
  PetscBool /* set,flg , */ symm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat,&comm));

  /* PetscCall(MatIsSymmetricKnown(Amat, &set, &flg)); || !(set && flg) -- this causes lot of symm calls */
  symm = (PetscBool)(pc_gamg_agg->symmetrize_graph); /* && !pc_gamg_agg->square_graph; */

  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_GRAPH],0,0,0,0));
  PetscCall(MatCreateGraph(Amat, symm, PETSC_TRUE, &Gmat));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_GRAPH],0,0,0,0));

  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_FILTER],0,0,0,0));
  PetscCall(MatFilter(Gmat, vfilter,&F));
  if (F) {
    PetscCall(MatDestroy(&Gmat));
    Gmat = F;
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_FILTER],0,0,0,0));

  *a_Gmat = Gmat;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCoarsen_AGG - supports squaring the graph (deprecated) and new graph for
     communication of QR data used with HEM and MISk coarsening

  Input Parameter:
   . a_pc - this

  Input/Output Parameter:
   . a_Gmat1 - graph to coarsen (in), graph off processor edges for QR gather scatter (out)

  Output Parameter:
   . agg_lists - list of aggregates

*/
static PetscErrorCode PCGAMGCoarsen_AGG(PC a_pc,Mat *a_Gmat1, PetscCoarsenData **agg_lists)
{
  PC_MG          *mg          = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat            mat, Gmat1 = *a_Gmat1;  /* aggressive graph */
  IS             perm;
  PetscInt       Istart,Iend,Ii,nloc,bs,nn;
  PetscInt       *permute,*degree;
  PetscBool      *bIndexSet;
  MatCoarsen     crs;
  MPI_Comm       comm;
  PetscReal      hashfact;
  PetscInt       iSwapIndex;
  PetscRandom    random;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_COARSEN],0,0,0,0));
  PetscCall(PetscObjectGetComm((PetscObject)Gmat1,&comm));
  PetscCall(MatGetLocalSize(Gmat1, &nn, NULL));
  PetscCall(MatGetBlockSize(Gmat1, &bs));
  PetscCheck(bs == 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"bs %" PetscInt_FMT " must be 1",bs);
  nloc = nn/bs;

  /* get MIS aggs - randomize */
  PetscCall(PetscMalloc2(nloc, &permute,nloc, &degree));
  PetscCall(PetscCalloc1(nloc, &bIndexSet));
  for (Ii = 0; Ii < nloc; Ii++) permute[Ii] = Ii;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&random));
  PetscCall(MatGetOwnershipRange(Gmat1, &Istart, &Iend));
  for (Ii = 0; Ii < nloc; Ii++) {
    PetscInt nc;
    PetscCall(MatGetRow(Gmat1,Istart+Ii,&nc,NULL,NULL));
    degree[Ii] = nc;
    PetscCall(MatRestoreRow(Gmat1,Istart+Ii,&nc,NULL,NULL));
  }
  for (Ii = 0; Ii < nloc; Ii++) {
    PetscCall(PetscRandomGetValueReal(random,&hashfact));
    iSwapIndex = (PetscInt) (hashfact*nloc)%nloc;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii) {
      PetscInt iTemp = permute[iSwapIndex];
      permute[iSwapIndex]   = permute[Ii];
      permute[Ii]           = iTemp;
      iTemp = degree[iSwapIndex];
      degree[iSwapIndex]   = degree[Ii];
      degree[Ii]           = iTemp;
      bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
  // create minimum degree ordering
  PetscCall(PetscSortIntWithArray(nloc,degree,permute));

  PetscCall(PetscFree(bIndexSet));
  PetscCall(PetscRandomDestroy(&random));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_USE_POINTER, &perm));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_MIS],0,0,0,0));
  PetscCall(MatCoarsenCreate(comm, &crs));
  PetscCall(MatCoarsenSetFromOptions(crs));
  PetscCall(MatCoarsenSetGreedyOrdering(crs, perm));
  PetscCall(MatCoarsenSetAdjacency(crs, Gmat1));
  PetscCall(MatCoarsenSetStrictAggs(crs, PETSC_TRUE));
  if (pc_gamg->current_level < pc_gamg_agg->aggressive_coarsening_levels) PetscCall(MatCoarsenMISKSetDistance(crs,2)); // hardwire to MIS-2
  else PetscCall(MatCoarsenMISKSetDistance(crs,1)); // MIS
  PetscCall(MatCoarsenApply(crs));
  PetscCall(MatCoarsenViewFromOptions(crs,NULL,"-mat_coarsen_view"));
  PetscCall(MatCoarsenGetData(crs, agg_lists)); /* output */
  PetscCall(MatCoarsenDestroy(&crs));

  PetscCall(ISDestroy(&perm));
  PetscCall(PetscFree2(permute,degree));
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_MIS],0,0,0,0));

  {
    PetscCoarsenData *llist = *agg_lists;
    /* see if we have a matrix that takes precedence (returned from MatCoarsenApply) */
    PetscCall(PetscCDGetMat(llist, &mat));
    if (mat) {
      PetscCall(MatDestroy(&Gmat1));
      *a_Gmat1 = mat; /* output */
    }
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_COARSEN],0,0,0,0));
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
static PetscErrorCode PCGAMGProlongator_AGG(PC pc,Mat Amat,Mat Gmat,PetscCoarsenData *agg_lists,Mat *a_P_out)
{
  PC_MG          *mg       = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg  = (PC_GAMG*)mg->innerctx;
  const PetscInt col_bs = pc_gamg->data_cell_cols;
  PetscInt       Istart,Iend,nloc,ii,jj,kk,my0,nLocalSelected,bs;
  Mat            Prol;
  PetscMPIInt    size;
  MPI_Comm       comm;
  PetscReal      *data_w_ghost;
  PetscInt       myCrs0, nbnodes=0, *flid_fgid;
  MatType        mtype;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat,&comm));
  PetscCheck(col_bs >= 1,comm,PETSC_ERR_PLIB,"Column bs cannot be less than 1");
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PROL],0,0,0,0));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetOwnershipRange(Amat, &Istart, &Iend));
  PetscCall(MatGetBlockSize(Amat, &bs));
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  PetscCheck((Iend-Istart) % bs == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"(Iend %" PetscInt_FMT " - Istart %" PetscInt_FMT ") not divisible by bs %" PetscInt_FMT,Iend,Istart,bs);

  /* get 'nLocalSelected' */
  for (ii=0, nLocalSelected = 0; ii < nloc; ii++) {
    PetscBool ise;
    /* filter out singletons 0 or 1? */
    PetscCall(PetscCDEmptyAt(agg_lists, ii, &ise));
    if (!ise) nLocalSelected++;
  }

  /* create prolongator, create P matrix */
  PetscCall(MatGetType(Amat,&mtype));
  PetscCall(MatCreate(comm, &Prol));
  PetscCall(MatSetSizes(Prol,nloc*bs,nLocalSelected*col_bs,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetBlockSizes(Prol, bs, col_bs));
  PetscCall(MatSetType(Prol, mtype));
  PetscCall(MatSeqAIJSetPreallocation(Prol,col_bs, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Prol,col_bs, NULL, col_bs, NULL));

  /* can get all points "removed" */
  PetscCall(MatGetSize(Prol, &kk, &ii));
  if (!ii) {
    PetscCall(PetscInfo(pc,"%s: No selected points on coarse grid\n",((PetscObject)pc)->prefix));
    PetscCall(MatDestroy(&Prol));
    *a_P_out = NULL;  /* out */
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROL],0,0,0,0));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscInfo(pc,"%s: New grid %" PetscInt_FMT " nodes\n",((PetscObject)pc)->prefix,ii/col_bs));
  PetscCall(MatGetOwnershipRangeColumn(Prol, &myCrs0, &kk));

  PetscCheck((kk-myCrs0) % col_bs == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"(kk %" PetscInt_FMT " -myCrs0 %" PetscInt_FMT ") not divisible by col_bs %" PetscInt_FMT,kk,myCrs0,col_bs);
  myCrs0 = myCrs0/col_bs;
  PetscCheck((kk/col_bs-myCrs0) == nLocalSelected,PETSC_COMM_SELF,PETSC_ERR_PLIB,"(kk %" PetscInt_FMT "/col_bs %" PetscInt_FMT " - myCrs0 %" PetscInt_FMT ") != nLocalSelected %" PetscInt_FMT ")",kk,col_bs,myCrs0,nLocalSelected);

  /* create global vector of data in 'data_w_ghost' */
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PROLA],0,0,0,0));
  if (size > 1) { /* get ghost null space data */
    PetscReal *tmp_gdata,*tmp_ldata,*tp2;
    PetscCall(PetscMalloc1(nloc, &tmp_ldata));
    for (jj = 0; jj < col_bs; jj++) {
      for (kk = 0; kk < bs; kk++) {
        PetscInt        ii,stride;
        const PetscReal *tp = pc_gamg->data + jj*bs*nloc + kk;
        for (ii = 0; ii < nloc; ii++, tp += bs) tmp_ldata[ii] = *tp;

        PetscCall(PCGAMGGetDataWithGhosts(Gmat, 1, tmp_ldata, &stride, &tmp_gdata));

        if (!jj && !kk) { /* now I know how many total nodes - allocate TODO: move below and do in one 'col_bs' call */
          PetscCall(PetscMalloc1(stride*bs*col_bs, &data_w_ghost));
          nbnodes = bs*stride;
        }
        tp2 = data_w_ghost + jj*bs*stride + kk;
        for (ii = 0; ii < stride; ii++, tp2 += bs) *tp2 = tmp_gdata[ii];
        PetscCall(PetscFree(tmp_gdata));
      }
    }
    PetscCall(PetscFree(tmp_ldata));
  } else {
    nbnodes      = bs*nloc;
    data_w_ghost = (PetscReal*)pc_gamg->data;
  }

  /* get 'flid_fgid' TODO - move up to get 'stride' and do get null space data above in one step (jj loop) */
  if (size > 1) {
    PetscReal *fid_glid_loc,*fiddata;
    PetscInt  stride;

    PetscCall(PetscMalloc1(nloc, &fid_glid_loc));
    for (kk=0; kk<nloc; kk++) fid_glid_loc[kk] = (PetscReal)(my0+kk);
    PetscCall(PCGAMGGetDataWithGhosts(Gmat, 1, fid_glid_loc, &stride, &fiddata));
    PetscCall(PetscMalloc1(stride, &flid_fgid)); /* copy real data to in */
    for (kk=0; kk<stride; kk++) flid_fgid[kk] = (PetscInt)fiddata[kk];
    PetscCall(PetscFree(fiddata));

    PetscCheck(stride == nbnodes/bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"stride %" PetscInt_FMT " != nbnodes %" PetscInt_FMT "/bs %" PetscInt_FMT,stride,nbnodes,bs);
    PetscCall(PetscFree(fid_glid_loc));
  } else {
    PetscCall(PetscMalloc1(nloc, &flid_fgid));
    for (kk=0; kk<nloc; kk++) flid_fgid[kk] = my0 + kk;
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROLA],0,0,0,0));
  /* get P0 */
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_PROLB],0,0,0,0));
  {
    PetscReal *data_out = NULL;
    PetscCall(formProl0(agg_lists, bs, col_bs, myCrs0, nbnodes,data_w_ghost, flid_fgid, &data_out, Prol));
    PetscCall(PetscFree(pc_gamg->data));

    pc_gamg->data           = data_out;
    pc_gamg->data_cell_rows = col_bs;
    pc_gamg->data_sz        = col_bs*col_bs*nLocalSelected;
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROLB],0,0,0,0));
  if (size > 1) {PetscCall(PetscFree(data_w_ghost));}
  PetscCall(PetscFree(flid_fgid));

  *a_P_out = Prol;  /* out */

  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_PROL],0,0,0,0));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGOptProlongator_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
 In/Output Parameter:
   . a_P - prolongation operator to the next level
*/
static PetscErrorCode PCGAMGOptProlongator_AGG(PC pc,Mat Amat,Mat *a_P)
{
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  PetscInt       jj;
  Mat            Prol  = *a_P;
  MPI_Comm       comm;
  KSP            eksp;
  Vec            bb, xx;
  PC             epc;
  PetscReal      alpha, emax, emin;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat,&comm));
  PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_OPT],0,0,0,0));

  /* compute maximum singular value of operator to be used in smoother */
  if (0 < pc_gamg_agg->nsmooths) {
    /* get eigen estimates */
    if (pc_gamg->emax > 0) {
      emin = pc_gamg->emin;
      emax = pc_gamg->emax;
    } else {
      const char *prefix;

      PetscCall(MatCreateVecs(Amat, &bb, NULL));
      PetscCall(MatCreateVecs(Amat, &xx, NULL));
      PetscCall(KSPSetNoisy_Private(bb));

      PetscCall(KSPCreate(comm,&eksp));
      PetscCall(PCGetOptionsPrefix(pc,&prefix));
      PetscCall(KSPSetOptionsPrefix(eksp,prefix));
      PetscCall(KSPAppendOptionsPrefix(eksp,"pc_gamg_esteig_"));
      {
        PetscBool sflg;
        PetscCall(MatGetOption(Amat, MAT_SPD, &sflg));
        if (sflg) PetscCall(KSPSetType(eksp, KSPCG));
      }
      PetscCall(KSPSetErrorIfNotConverged(eksp,pc->erroriffailure));
      PetscCall(KSPSetNormType(eksp, KSP_NORM_NONE));

      PetscCall(KSPSetInitialGuessNonzero(eksp, PETSC_FALSE));
      PetscCall(KSPSetOperators(eksp, Amat, Amat));

      PetscCall(KSPGetPC(eksp, &epc));
      PetscCall(PCSetType(epc, PCJACOBI));  /* smoother in smoothed agg. */

      PetscCall(KSPSetTolerances(eksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT, 10)); // 10 is safer, but 5 is often fine, can override with -pc_gamg_esteig_ksp_max_it -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.2

      PetscCall(KSPSetFromOptions(eksp));
      PetscCall(KSPSetComputeSingularValues(eksp,PETSC_TRUE));
      PetscCall(KSPSolve(eksp, bb, xx));
      PetscCall(KSPCheckSolve(eksp,pc,xx));

      PetscCall(KSPComputeExtremeSingularValues(eksp, &emax, &emin));
      PetscCall(PetscInfo(pc,"%s: Smooth P0: max eigen=%e min=%e PC=%s\n",((PetscObject)pc)->prefix,(double)emax,(double)emin,PCJACOBI));
      PetscCall(VecDestroy(&xx));
      PetscCall(VecDestroy(&bb));
      PetscCall(KSPDestroy(&eksp));
    }
    if (pc_gamg->use_sa_esteig) {
      mg->min_eigen_DinvA[pc_gamg->current_level] = emin;
      mg->max_eigen_DinvA[pc_gamg->current_level] = emax;
      PetscCall(PetscInfo(pc,"%s: Smooth P0: level %" PetscInt_FMT ", cache spectra %g %g\n",((PetscObject)pc)->prefix,pc_gamg->current_level,(double)emin,(double)emax));
    } else {
      mg->min_eigen_DinvA[pc_gamg->current_level] = 0;
      mg->max_eigen_DinvA[pc_gamg->current_level] = 0;
    }
  } else {
    mg->min_eigen_DinvA[pc_gamg->current_level] = 0;
    mg->max_eigen_DinvA[pc_gamg->current_level] = 0;
  }

  /* smooth P0 */
  for (jj = 0; jj < pc_gamg_agg->nsmooths; jj++) {
    Mat tMat;
    Vec diag;

    PetscCall(PetscLogEventBegin(petsc_gamg_setup_events[GAMG_OPTSM],0,0,0,0));

    /* smooth P1 := (I - omega/lam D^{-1}A)P0 */
    PetscCall(PetscLogEventBegin(petsc_gamg_setup_matmat_events[pc_gamg->current_level][2],0,0,0,0));
    PetscCall(MatMatMult(Amat, Prol, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tMat));
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_matmat_events[pc_gamg->current_level][2],0,0,0,0));
    PetscCall(MatProductClear(tMat));
    PetscCall(MatCreateVecs(Amat, &diag, NULL));
    PetscCall(MatGetDiagonal(Amat, diag)); /* effectively PCJACOBI */
    PetscCall(VecReciprocal(diag));
    PetscCall(MatDiagonalScale(tMat, diag, NULL));
    PetscCall(VecDestroy(&diag));

    /* TODO: Set a PCFailedReason and exit the building of the AMG preconditioner */
    PetscCheck(emax != 0.0,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Computed maximum singular value as zero");
    /* TODO: Document the 1.4 and don't hardwire it in this routine */
    alpha = -1.4/emax;

    PetscCall(MatAYPX(tMat, alpha, Prol, SUBSET_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Prol));
    Prol = tMat;
    PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_OPTSM],0,0,0,0));
  }
  PetscCall(PetscLogEventEnd(petsc_gamg_setup_events[GAMG_OPT],0,0,0,0));
  *a_P = Prol;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_AGG

  Input Parameter:
   . pc -
*/
PetscErrorCode  PCCreateGAMG_AGG(PC pc)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg;

  PetscFunctionBegin;
  /* create sub context for SA */
  PetscCall(PetscNewLog(pc,&pc_gamg_agg));
  pc_gamg->subctx = pc_gamg_agg;

  pc_gamg->ops->setfromoptions = PCSetFromOptions_GAMG_AGG;
  pc_gamg->ops->destroy        = PCDestroy_GAMG_AGG;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->graph             = PCGAMGGraph_AGG;
  pc_gamg->ops->coarsen           = PCGAMGCoarsen_AGG;
  pc_gamg->ops->prolongator       = PCGAMGProlongator_AGG;
  pc_gamg->ops->optprolongator    = PCGAMGOptProlongator_AGG;
  pc_gamg->ops->createdefaultdata = PCSetData_AGG;
  pc_gamg->ops->view              = PCView_GAMG_AGG;

  pc_gamg_agg->aggressive_coarsening_levels = 0;
  pc_gamg_agg->symmetrize_graph    = PETSC_FALSE;
  pc_gamg_agg->nsmooths     = 1;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetNSmooths_C",PCGAMGSetNSmooths_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetSymmetrizeGraph_C",PCGAMGSetSymmetrizeGraph_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetAggressiveLevels_C",PCGAMGSetAggressiveLevels_AGG));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_AGG));
  PetscFunctionReturn(0);
}
