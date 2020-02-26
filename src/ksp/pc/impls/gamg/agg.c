/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
/*  Next line needed to deactivate KSP_Solve logging */
#include <petsc/private/kspimpl.h>
#include <petscblaslapack.h>
#include <petscdm.h>

typedef struct {
  PetscInt  nsmooths;
  PetscBool sym_graph;
  PetscInt  square_graph;
} PC_GAMG_AGG;

/*@
   PCGAMGSetNSmooths - Set number of smoothing steps (1 is typical)

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_agg_nsmooths <nsmooth, default=1> - number of smoothing steps to use with smooth aggregation

   Level: intermediate

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
   PCGAMGSetSymGraph - Symmetrize the graph before computing the aggregation. Some algorithms require the graph be symmetric

   Not Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -pc_gamg_sym_graph <true,default=false> - symmetrize the graph before computing the aggregation

   Level: intermediate

.seealso: PCGAMGSetSquareGraph()
@*/
PetscErrorCode PCGAMGSetSymGraph(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSymGraph_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetSymGraph_AGG(PC pc, PetscBool n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->sym_graph = n;
  PetscFunctionReturn(0);
}

/*@
   PCGAMGSetSquareGraph -  Square the graph, ie. compute A'*A before aggregating it

   Not Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  n - PETSC_TRUE or PETSC_FALSE

   Options Database Key:
.  -pc_gamg_square_graph <n,default = 1> - number of levels to square the graph on before aggregating it

   Notes:
   Squaring the graph increases the rate of coarsening (aggressive coarsening) and thereby reduces the complexity of the coarse grids, and generally results in slower solver converge rates. Reducing coarse grid complexity reduced the complexity of Galerkin coarse grid construction considerably.

   Level: intermediate

.seealso: PCGAMGSetSymGraph(), PCGAMGSetThreshold()
@*/
PetscErrorCode PCGAMGSetSquareGraph(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSquareGraph_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGSetSquareGraph_AGG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->square_graph = n;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_GAMG_AGG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"GAMG-AGG options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-pc_gamg_agg_nsmooths","smoothing steps for smoothed aggregation, usually 1","PCGAMGSetNSmooths",pc_gamg_agg->nsmooths,&pc_gamg_agg->nsmooths,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-pc_gamg_sym_graph","Set for asymmetric matrices","PCGAMGSetSymGraph",pc_gamg_agg->sym_graph,&pc_gamg_agg->sym_graph,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_gamg_square_graph","Number of levels to square graph for faster coarsening and lower coarse grid complexity","PCGAMGSetSquareGraph",pc_gamg_agg->square_graph,&pc_gamg_agg->square_graph,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode PCDestroy_GAMG_AGG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       arrsz,kk,ii,jj,nloc,ndatarows,ndf;
  Mat            mat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  nloc = a_nloc;

  /* SA: null space vectors */
  ierr = MatGetBlockSize(mat, &ndf);CHKERRQ(ierr); /* this does not work for Stokes */
  if (coords && ndf==1) pc_gamg->data_cell_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if (coords) {
    if (ndm > ndf) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"degrees of motion %D > block size %D",ndm,ndf);
    pc_gamg->data_cell_cols = (ndm==2 ? 3 : 6); /* displacement elasticity */
    if (ndm != ndf) {
      if (pc_gamg->data_cell_cols != ndf) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Don't know how to create null space for ndm=%D, ndf=%D.  Use MatSetNearNullSpace.",ndm,ndf);
    }
  } else pc_gamg->data_cell_cols = ndf; /* no data, force SA with constant null space vectors */
  pc_gamg->data_cell_rows = ndatarows = ndf;
  if (pc_gamg->data_cell_cols <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"pc_gamg->data_cell_cols %D <= 0",pc_gamg->data_cell_cols);
  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (!pc_gamg->data || (pc_gamg->data_sz != arrsz)) {
    ierr = PetscFree(pc_gamg->data);CHKERRQ(ierr);
    ierr = PetscMalloc1(arrsz+1, &pc_gamg->data);CHKERRQ(ierr);
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

typedef PetscInt NState;
static const NState NOT_DONE=-2;
static const NState DELETED =-1;
static const NState REMOVED =-3;
#define IS_SELECTED(s) (s!=DELETED && s!=NOT_DONE && s!=REMOVED)

/* -------------------------------------------------------------------------- */
/*
   smoothAggs - greedy grab of with G1 (unsquared graph) -- AIJ specific
     - AGG-MG specific: clears singletons out of 'selected_2'

   Input Parameter:
   . Gmat_2 - global matrix of graph (data not defined)   base (squared) graph
   . Gmat_1 - base graph to grab with                 base graph
   Input/Output Parameter:
   . aggs_2 - linked list of aggs with gids)
*/
static PetscErrorCode smoothAggs(PC pc,Mat Gmat_2, Mat Gmat_1,PetscCoarsenData *aggs_2)
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ     *matA_1, *matB_1=0;
  MPI_Comm       comm;
  PetscInt       lid,*ii,*idx,ix,Iend,my0,kk,n,j;
  Mat_MPIAIJ     *mpimat_2 = 0, *mpimat_1=0;
  const PetscInt nloc      = Gmat_2->rmap->n;
  PetscScalar    *cpcol_1_state,*cpcol_2_state,*cpcol_2_par_orig,*lid_parent_gid;
  PetscInt       *lid_cprowID_1;
  NState         *lid_state;
  Vec            ghost_par_orig2;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Gmat_2,&comm);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat_1,&my0,&Iend);CHKERRQ(ierr);

  /* get submatrices */
  ierr = PetscStrbeginswith(((PetscObject)Gmat_1)->type_name,MATMPIAIJ,&isMPI);CHKERRQ(ierr);
  if (isMPI) {
    /* grab matrix objects */
    mpimat_2 = (Mat_MPIAIJ*)Gmat_2->data;
    mpimat_1 = (Mat_MPIAIJ*)Gmat_1->data;
    matA_1   = (Mat_SeqAIJ*)mpimat_1->A->data;
    matB_1   = (Mat_SeqAIJ*)mpimat_1->B->data;

    /* force compressed row storage for B matrix in AuxMat */
    ierr = MatCheckCompressedRow(mpimat_1->B,matB_1->nonzerorowcnt,&matB_1->compressedrow,matB_1->i,Gmat_1->rmap->n,-1.0);CHKERRQ(ierr);

    ierr = PetscMalloc1(nloc, &lid_cprowID_1);CHKERRQ(ierr);
    for (lid = 0; lid < nloc; lid++) lid_cprowID_1[lid] = -1;
    for (ix=0; ix<matB_1->compressedrow.nrows; ix++) {
      PetscInt lid = matB_1->compressedrow.rindex[ix];
      lid_cprowID_1[lid] = ix;
    }
  } else {
    PetscBool        isAIJ;
    ierr = PetscStrbeginswith(((PetscObject)Gmat_1)->type_name,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
    if (!isAIJ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Require AIJ matrix.");
    matA_1        = (Mat_SeqAIJ*)Gmat_1->data;
    lid_cprowID_1 = NULL;
  }
  if (nloc>0) {
    if (matB_1 && !matB_1->compressedrow.use) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"matB_1 && !matB_1->compressedrow.use: PETSc bug???");
  }
  /* get state of locals and selected gid for deleted */
  ierr = PetscMalloc2(nloc, &lid_state,nloc, &lid_parent_gid);CHKERRQ(ierr);
  for (lid = 0; lid < nloc; lid++) {
    lid_parent_gid[lid] = -1.0;
    lid_state[lid]      = DELETED;
  }

  /* set lid_state */
  for (lid = 0; lid < nloc; lid++) {
    PetscCDIntNd *pos;
    ierr = PetscCDGetHeadPos(aggs_2,lid,&pos);CHKERRQ(ierr);
    if (pos) {
      PetscInt gid1;

      ierr = PetscCDIntNdGetID(pos, &gid1);CHKERRQ(ierr);
      if (gid1 != lid+my0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"gid1 %D != lid %D + my0 %D",gid1,lid,my0);
      lid_state[lid] = gid1;
    }
  }

  /* map local to selected local, DELETED means a ghost owns it */
  for (lid=kk=0; lid<nloc; lid++) {
    NState state = lid_state[lid];
    if (IS_SELECTED(state)) {
      PetscCDIntNd *pos;
      ierr = PetscCDGetHeadPos(aggs_2,lid,&pos);CHKERRQ(ierr);
      while (pos) {
        PetscInt gid1;
        ierr = PetscCDIntNdGetID(pos, &gid1);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(aggs_2,lid,&pos);CHKERRQ(ierr);

        if (gid1 >= my0 && gid1 < Iend) lid_parent_gid[gid1-my0] = (PetscScalar)(lid + my0);
      }
    }
  }
  /* get 'cpcol_1/2_state' & cpcol_2_par_orig - uses mpimat_1/2->lvec for temp space */
  if (isMPI) {
    Vec tempVec;
    /* get 'cpcol_1_state' */
    ierr = MatCreateVecs(Gmat_1, &tempVec, 0);CHKERRQ(ierr);
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      PetscScalar v = (PetscScalar)lid_state[kk];
      ierr = VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(mpimat_1->lvec, &cpcol_1_state);CHKERRQ(ierr);
    /* get 'cpcol_2_state' */
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(mpimat_2->lvec, &cpcol_2_state);CHKERRQ(ierr);
    /* get 'cpcol_2_par_orig' */
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      PetscScalar v = (PetscScalar)lid_parent_gid[kk];
      ierr = VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecDuplicate(mpimat_2->lvec, &ghost_par_orig2);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghost_par_orig2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat_2->Mvctx,tempVec, ghost_par_orig2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ghost_par_orig2, &cpcol_2_par_orig);CHKERRQ(ierr);

    ierr = VecDestroy(&tempVec);CHKERRQ(ierr);
  } /* ismpi */

  /* doit */
  for (lid=0; lid<nloc; lid++) {
    NState state = lid_state[lid];
    if (IS_SELECTED(state)) {
      /* steal locals */
      ii  = matA_1->i; n = ii[lid+1] - ii[lid];
      idx = matA_1->j + ii[lid];
      for (j=0; j<n; j++) {
        PetscInt lidj   = idx[j], sgid;
        NState   statej = lid_state[lidj];
        if (statej==DELETED && (sgid=(PetscInt)PetscRealPart(lid_parent_gid[lidj])) != lid+my0) { /* steal local */
          lid_parent_gid[lidj] = (PetscScalar)(lid+my0); /* send this if sgid is not local */
          if (sgid >= my0 && sgid < Iend) {       /* I'm stealing this local from a local sgid */
            PetscInt     hav=0,slid=sgid-my0,gidj=lidj+my0;
            PetscCDIntNd *pos,*last=NULL;
            /* looking for local from local so id_llist_2 works */
            ierr = PetscCDGetHeadPos(aggs_2,slid,&pos);CHKERRQ(ierr);
            while (pos) {
              PetscInt gid;
              ierr = PetscCDIntNdGetID(pos, &gid);CHKERRQ(ierr);
              if (gid == gidj) {
                if (!last) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"last cannot be null");
                ierr = PetscCDRemoveNextNode(aggs_2, slid, last);CHKERRQ(ierr);
                ierr = PetscCDAppendNode(aggs_2, lid, pos);CHKERRQ(ierr);
                hav  = 1;
                break;
              } else last = pos;

              ierr = PetscCDGetNextPos(aggs_2,slid,&pos);CHKERRQ(ierr);
            }
            if (hav!=1) {
              if (!hav) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"found node %D times???",hav);
            }
          } else {            /* I'm stealing this local, owned by a ghost */
            if (sgid != -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mat has an un-symmetric graph. Use '-%spc_gamg_sym_graph true' to symmetrize the graph or '-%spc_gamg_threshold -1' if the matrix is structurally symmetric.",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "",((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "");
            ierr = PetscCDAppendID(aggs_2, lid, lidj+my0);CHKERRQ(ierr);
          }
        }
      } /* local neighbors */
    } else if (state == DELETED && lid_cprowID_1) {
      PetscInt sgidold = (PetscInt)PetscRealPart(lid_parent_gid[lid]);
      /* see if I have a selected ghost neighbor that will steal me */
      if ((ix=lid_cprowID_1[lid]) != -1) {
        ii  = matB_1->compressedrow.i; n = ii[ix+1] - ii[ix];
        idx = matB_1->j + ii[ix];
        for (j=0; j<n; j++) {
          PetscInt cpid   = idx[j];
          NState   statej = (NState)PetscRealPart(cpcol_1_state[cpid]);
          if (IS_SELECTED(statej) && sgidold != (PetscInt)statej) { /* ghost will steal this, remove from my list */
            lid_parent_gid[lid] = (PetscScalar)statej; /* send who selected */
            if (sgidold>=my0 && sgidold<Iend) { /* this was mine */
              PetscInt     hav=0,oldslidj=sgidold-my0;
              PetscCDIntNd *pos,*last=NULL;
              /* remove from 'oldslidj' list */
              ierr = PetscCDGetHeadPos(aggs_2,oldslidj,&pos);CHKERRQ(ierr);
              while (pos) {
                PetscInt gid;
                ierr = PetscCDIntNdGetID(pos, &gid);CHKERRQ(ierr);
                if (lid+my0 == gid) {
                  /* id_llist_2[lastid] = id_llist_2[flid];   /\* remove lid from oldslidj list *\/ */
                  if (!last) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"last cannot be null");
                  ierr = PetscCDRemoveNextNode(aggs_2, oldslidj, last);CHKERRQ(ierr);
                  /* ghost (PetscScalar)statej will add this later */
                  hav = 1;
                  break;
                } else last = pos;

                ierr = PetscCDGetNextPos(aggs_2,oldslidj,&pos);CHKERRQ(ierr);
              }
              if (hav!=1) {
                if (!hav) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
                SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"found node %D times???",hav);
              }
            } else {
              /* ghosts remove this later */
            }
          }
        }
      }
    } /* selected/deleted */
  } /* node loop */

  if (isMPI) {
    PetscScalar     *cpcol_2_parent,*cpcol_2_gid;
    Vec             tempVec,ghostgids2,ghostparents2;
    PetscInt        cpid,nghost_2;
    PCGAMGHashTable gid_cpid;

    ierr = VecGetSize(mpimat_2->lvec, &nghost_2);CHKERRQ(ierr);
    ierr = MatCreateVecs(Gmat_2, &tempVec, 0);CHKERRQ(ierr);

    /* get 'cpcol_2_parent' */
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      ierr = VecSetValues(tempVec, 1, &j, &lid_parent_gid[kk], INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecDuplicate(mpimat_2->lvec, &ghostparents2);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghostparents2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat_2->Mvctx,tempVec, ghostparents2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ghostparents2, &cpcol_2_parent);CHKERRQ(ierr);

    /* get 'cpcol_2_gid' */
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      PetscScalar v = (PetscScalar)j;
      ierr = VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecDuplicate(mpimat_2->lvec, &ghostgids2);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghostgids2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat_2->Mvctx,tempVec, ghostgids2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ghostgids2, &cpcol_2_gid);CHKERRQ(ierr);
    ierr = VecDestroy(&tempVec);CHKERRQ(ierr);

    /* look for deleted ghosts and add to table */
    ierr = PCGAMGHashTableCreate(2*nghost_2+1, &gid_cpid);CHKERRQ(ierr);
    for (cpid = 0; cpid < nghost_2; cpid++) {
      NState state = (NState)PetscRealPart(cpcol_2_state[cpid]);
      if (state==DELETED) {
        PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
        PetscInt sgid_old = (PetscInt)PetscRealPart(cpcol_2_par_orig[cpid]);
        if (sgid_old == -1 && sgid_new != -1) {
          PetscInt gid = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
          ierr = PCGAMGHashTableAdd(&gid_cpid, gid, cpid);CHKERRQ(ierr);
        }
      }
    }

    /* look for deleted ghosts and see if they moved - remove it */
    for (lid=0; lid<nloc; lid++) {
      NState state = lid_state[lid];
      if (IS_SELECTED(state)) {
        PetscCDIntNd *pos,*last=NULL;
        /* look for deleted ghosts and see if they moved */
        ierr = PetscCDGetHeadPos(aggs_2,lid,&pos);CHKERRQ(ierr);
        while (pos) {
          PetscInt gid;
          ierr = PetscCDIntNdGetID(pos, &gid);CHKERRQ(ierr);

          if (gid < my0 || gid >= Iend) {
            ierr = PCGAMGHashTableFind(&gid_cpid, gid, &cpid);CHKERRQ(ierr);
            if (cpid != -1) {
              /* a moved ghost - */
              /* id_llist_2[lastid] = id_llist_2[flid];    /\* remove 'flid' from list *\/ */
              ierr = PetscCDRemoveNextNode(aggs_2, lid, last);CHKERRQ(ierr);
            } else last = pos;
          } else last = pos;

          ierr = PetscCDGetNextPos(aggs_2,lid,&pos);CHKERRQ(ierr);
        } /* loop over list of deleted */
      } /* selected */
    }
    ierr = PCGAMGHashTableDestroy(&gid_cpid);CHKERRQ(ierr);

    /* look at ghosts, see if they changed - and it */
    for (cpid = 0; cpid < nghost_2; cpid++) {
      PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
      if (sgid_new >= my0 && sgid_new < Iend) { /* this is mine */
        PetscInt     gid     = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
        PetscInt     slid_new=sgid_new-my0,hav=0;
        PetscCDIntNd *pos;

        /* search for this gid to see if I have it */
        ierr = PetscCDGetHeadPos(aggs_2,slid_new,&pos);CHKERRQ(ierr);
        while (pos) {
          PetscInt gidj;
          ierr = PetscCDIntNdGetID(pos, &gidj);CHKERRQ(ierr);
          ierr = PetscCDGetNextPos(aggs_2,slid_new,&pos);CHKERRQ(ierr);

          if (gidj == gid) { hav = 1; break; }
        }
        if (hav != 1) {
          /* insert 'flidj' into head of llist */
          ierr = PetscCDAppendID(aggs_2, slid_new, gid);CHKERRQ(ierr);
        }
      }
    }

    ierr = VecRestoreArray(mpimat_1->lvec, &cpcol_1_state);CHKERRQ(ierr);
    ierr = VecRestoreArray(mpimat_2->lvec, &cpcol_2_state);CHKERRQ(ierr);
    ierr = VecRestoreArray(ghostparents2, &cpcol_2_parent);CHKERRQ(ierr);
    ierr = VecRestoreArray(ghostgids2, &cpcol_2_gid);CHKERRQ(ierr);
    ierr = PetscFree(lid_cprowID_1);CHKERRQ(ierr);
    ierr = VecDestroy(&ghostgids2);CHKERRQ(ierr);
    ierr = VecDestroy(&ghostparents2);CHKERRQ(ierr);
    ierr = VecDestroy(&ghost_par_orig2);CHKERRQ(ierr);
  }

  ierr = PetscFree2(lid_state,lid_parent_gid);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  MatNullSpace   mnull;

  PetscFunctionBegin;
  ierr = MatGetNearNullSpace(a_A, &mnull);CHKERRQ(ierr);
  if (!mnull) {
    DM dm;
    ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
    if (!dm) {
      ierr = MatGetDM(a_A, &dm);CHKERRQ(ierr);
    }
    if (dm) {
      PetscObject deformation;
      PetscInt    Nf;

      ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
      if (Nf) {
        ierr = DMGetField(dm, 0, NULL, &deformation);CHKERRQ(ierr);
        ierr = PetscObjectQuery((PetscObject)deformation,"nearnullspace",(PetscObject*)&mnull);CHKERRQ(ierr);
        if (!mnull) {
          ierr = PetscObjectQuery((PetscObject)deformation,"nullspace",(PetscObject*)&mnull);CHKERRQ(ierr);
        }
      }
    }
  }

  if (!mnull) {
    PetscInt bs,NN,MM;
    ierr = MatGetBlockSize(a_A, &bs);CHKERRQ(ierr);
    ierr = MatGetLocalSize(a_A, &MM, &NN);CHKERRQ(ierr);
    if (MM % bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MM %D must be divisible by bs %D",MM,bs);
    ierr = PCSetCoordinates_AGG(pc, bs, MM/bs, NULL);CHKERRQ(ierr);
  } else {
    PetscReal *nullvec;
    PetscBool has_const;
    PetscInt  i,j,mlocal,nvec,bs;
    const Vec *vecs; const PetscScalar *v;

    ierr = MatGetLocalSize(a_A,&mlocal,NULL);CHKERRQ(ierr);
    ierr = MatNullSpaceGetVecs(mnull, &has_const, &nvec, &vecs);CHKERRQ(ierr);
    pc_gamg->data_sz = (nvec+!!has_const)*mlocal;
    ierr = PetscMalloc1((nvec+!!has_const)*mlocal,&nullvec);CHKERRQ(ierr);
    if (has_const) for (i=0; i<mlocal; i++) nullvec[i] = 1.0;
    for (i=0; i<nvec; i++) {
      ierr = VecGetArrayRead(vecs[i],&v);CHKERRQ(ierr);
      for (j=0; j<mlocal; j++) nullvec[(i+!!has_const)*mlocal + j] = PetscRealPart(v[j]);
      ierr = VecRestoreArrayRead(vecs[i],&v);CHKERRQ(ierr);
    }
    pc_gamg->data           = nullvec;
    pc_gamg->data_cell_cols = (nvec+!!has_const);
    ierr = MatGetBlockSize(a_A, &bs);CHKERRQ(ierr);
    pc_gamg->data_cell_rows = bs;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 formProl0

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
  PetscErrorCode  ierr;
  PetscInt        Istart,my0,Iend,nloc,clid,flid = 0,aggID,kk,jj,ii,mm,ndone,nSelected,minsz,nghosts,out_data_stride;
  MPI_Comm        comm;
  PetscMPIInt     rank;
  PetscReal       *out_data;
  PetscCDIntNd    *pos;
  PCGAMGHashTable fgid_flid;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)a_Prol,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Prol, &Istart, &Iend);CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  if ((Iend-Istart) % bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Iend %D - Istart %D must be divisible by bs %D",Iend,Istart,bs);
  Iend   /= bs;
  nghosts = data_stride/bs - nloc;

  ierr = PCGAMGHashTableCreate(2*nghosts+1, &fgid_flid);CHKERRQ(ierr);
  for (kk=0; kk<nghosts; kk++) {
    ierr = PCGAMGHashTableAdd(&fgid_flid, flid_fgid[nloc+kk], nloc+kk);CHKERRQ(ierr);
  }

  /* count selected -- same as number of cols of P */
  for (nSelected=mm=0; mm<nloc; mm++) {
    PetscBool ise;
    ierr = PetscCDEmptyAt(agg_llists, mm, &ise);CHKERRQ(ierr);
    if (!ise) nSelected++;
  }
  ierr = MatGetOwnershipRangeColumn(a_Prol, &ii, &jj);CHKERRQ(ierr);
  if ((ii/nSAvec) != my0crs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ii %D /nSAvec %D  != my0crs %D",ii,nSAvec,my0crs);
  if (nSelected != (jj-ii)/nSAvec) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"nSelected %D != (jj %D - ii %D)/nSAvec %D",nSelected,jj,ii,nSAvec);

  /* aloc space for coarse point data (output) */
  out_data_stride = nSelected*nSAvec;

  ierr = PetscMalloc1(out_data_stride*nSAvec, &out_data);CHKERRQ(ierr);
  for (ii=0;ii<out_data_stride*nSAvec;ii++) out_data[ii]=PETSC_MAX_REAL;
  *a_data_out = out_data; /* output - stride nSelected*nSAvec */

  /* find points and set prolongation */
  minsz = 100;
  ndone = 0;
  for (mm = clid = 0; mm < nloc; mm++) {
    ierr = PetscCDSizeAt(agg_llists, mm, &jj);CHKERRQ(ierr);
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
      ierr = PetscMalloc5(Mdata*N, &qqc,M*N, &qqr,N, &TAU,LWORK, &WORK,M, &fids);CHKERRQ(ierr);

      aggID = 0;
      ierr  = PetscCDGetHeadPos(agg_llists,lid,&pos);CHKERRQ(ierr);
      while (pos) {
        PetscInt gid1;
        ierr = PetscCDIntNdGetID(pos, &gid1);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg_llists,lid,&pos);CHKERRQ(ierr);

        if (gid1 >= my0 && gid1 < Iend) flid = gid1 - my0;
        else {
          ierr = PCGAMGHashTableFind(&fgid_flid, gid1, &flid);CHKERRQ(ierr);
          if (flid < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot find gid1 in table");
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

      ndone += aggID;
      /* QR */
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Mdata, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      if (INFO != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xGEQRF error");
      /* get R - column oriented - output B_{i+1} */
      {
        PetscReal *data = &out_data[clid*nSAvec];
        for (jj = 0; jj < nSAvec; jj++) {
          for (ii = 0; ii < nSAvec; ii++) {
            if (data[jj*out_data_stride + ii] != PETSC_MAX_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"data[jj*out_data_stride + ii] != %e",PETSC_MAX_REAL);
           if (ii <= jj) data[jj*out_data_stride + ii] = PetscRealPart(qqc[jj*Mdata + ii]);
           else data[jj*out_data_stride + ii] = 0.;
          }
        }
      }

      /* get Q - row oriented */
#if defined(PETSC_MISSING_LAPACK_ORGQR)
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"ORGQR - Lapack routine is unavailable.");
#else
      PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Mdata, &N, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      if (INFO != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xORGQR error arg %d",-INFO);
#endif

      for (ii = 0; ii < M; ii++) {
        for (jj = 0; jj < N; jj++) {
          qqr[N*ii + jj] = qqc[jj*Mdata + ii];
        }
      }

      /* add diagonal block of P0 */
      for (kk=0; kk<N; kk++) {
        cids[kk] = N*cgid + kk; /* global col IDs in P0 */
      }
      ierr = MatSetValues(a_Prol,M,fids,N,cids,qqr,INSERT_VALUES);CHKERRQ(ierr);
      ierr = PetscFree5(qqc,qqr,TAU,WORK,fids);CHKERRQ(ierr);
      clid++;
    } /* coarse agg */
  } /* for all fine nodes */
  ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PCGAMGHashTableDestroy(&fgid_flid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_GAMG_AGG(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"      AGG specific options\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"        Symmetric graph %s\n",pc_gamg_agg->sym_graph ? "true" : "false");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"        Number of levels to square graph %D\n",pc_gamg_agg->square_graph);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"        Number smoothing steps %D\n",pc_gamg_agg->nsmooths);CHKERRQ(ierr);
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
  PetscErrorCode            ierr;
  PC_MG                     *mg          = (PC_MG*)pc->data;
  PC_GAMG                   *pc_gamg     = (PC_GAMG*)mg->innerctx;
  const PetscReal           vfilter      = pc_gamg->threshold[pc_gamg->current_level];
  PC_GAMG_AGG               *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat                       Gmat;
  MPI_Comm                  comm;
  PetscBool /* set,flg , */ symm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PC_GAMGGraph_AGG,0,0,0,0);CHKERRQ(ierr);

  /* ierr = MatIsSymmetricKnown(Amat, &set, &flg);CHKERRQ(ierr); || !(set && flg) -- this causes lot of symm calls */
  symm = (PetscBool)(pc_gamg_agg->sym_graph); /* && !pc_gamg_agg->square_graph; */

  ierr = PCGAMGCreateGraph(Amat, &Gmat);CHKERRQ(ierr);
  ierr = PCGAMGFilterGraph(&Gmat, vfilter, symm);CHKERRQ(ierr);
  *a_Gmat = Gmat;
  ierr = PetscLogEventEnd(PC_GAMGGraph_AGG,0,0,0,0);CHKERRQ(ierr);
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
static PetscErrorCode PCGAMGCoarsen_AGG(PC a_pc,Mat *a_Gmat1,PetscCoarsenData **agg_lists)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat            mat,Gmat2, Gmat1 = *a_Gmat1;  /* squared graph */
  IS             perm;
  PetscInt       Istart,Iend,Ii,nloc,bs,n,m;
  PetscInt       *permute;
  PetscBool      *bIndexSet;
  MatCoarsen     crs;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscReal      hashfact;
  PetscInt       iSwapIndex;
  PetscRandom    random;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_GAMGCoarsen_AGG,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)Gmat1,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Gmat1, &n, &m);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Gmat1, &bs);CHKERRQ(ierr);
  if (bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"bs %D must be 1",bs);
  nloc = n/bs;

  if (pc_gamg->current_level < pc_gamg_agg->square_graph) {
    ierr = PetscInfo2(a_pc,"Square Graph on level %D of %D to square\n",pc_gamg->current_level+1,pc_gamg_agg->square_graph);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(Gmat1, Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2);CHKERRQ(ierr);
  } else Gmat2 = Gmat1;

  /* get MIS aggs - randomize */
  ierr = PetscMalloc1(nloc, &permute);CHKERRQ(ierr);
  ierr = PetscCalloc1(nloc, &bIndexSet);CHKERRQ(ierr);
  for (Ii = 0; Ii < nloc; Ii++) {
    permute[Ii]   = Ii;
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&random);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat1, &Istart, &Iend);CHKERRQ(ierr);
  for (Ii = 0; Ii < nloc; Ii++) {
    ierr = PetscRandomGetValueReal(random,&hashfact);CHKERRQ(ierr);
    iSwapIndex = (PetscInt) (hashfact*nloc)%nloc;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii) {
      PetscInt iTemp = permute[iSwapIndex];
      permute[iSwapIndex]   = permute[Ii];
      permute[Ii]           = iTemp;
      bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
  ierr = PetscFree(bIndexSet);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&random);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_USE_POINTER, &perm);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MatCoarsenCreate(comm, &crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetGreedyOrdering(crs, perm);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs, Gmat2);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs, agg_lists);CHKERRQ(ierr); /* output */
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = PetscFree(permute);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif

  /* smooth aggs */
  if (Gmat2 != Gmat1) {
    const PetscCoarsenData *llist = *agg_lists;
    ierr     = smoothAggs(a_pc,Gmat2, Gmat1, *agg_lists);CHKERRQ(ierr);
    ierr     = MatDestroy(&Gmat1);CHKERRQ(ierr);
    *a_Gmat1 = Gmat2; /* output */
    ierr     = PetscCDGetMat(llist, &mat);CHKERRQ(ierr);
    if (mat) SETERRQ(comm,PETSC_ERR_ARG_WRONG, "Auxilary matrix with squared graph????");
  } else {
    const PetscCoarsenData *llist = *agg_lists;
    /* see if we have a matrix that takes precedence (returned from MatCoarsenApply) */
    ierr = PetscCDGetMat(llist, &mat);CHKERRQ(ierr);
    if (mat) {
      ierr     = MatDestroy(&Gmat1);CHKERRQ(ierr);
      *a_Gmat1 = mat; /* output */
    }
  }
  ierr = PetscLogEventEnd(PC_GAMGCoarsen_AGG,0,0,0,0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,nloc,ii,jj,kk,my0,nLocalSelected,bs;
  Mat            Prol;
  PetscMPIInt    rank, size;
  MPI_Comm       comm;
  PetscReal      *data_w_ghost;
  PetscInt       myCrs0, nbnodes=0, *flid_fgid;
  MatType        mtype;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  if (col_bs < 1) SETERRQ(comm,PETSC_ERR_PLIB,"Column bs cannot be less than 1");
  ierr = PetscLogEventBegin(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat, &Istart, &Iend);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Amat, &bs);CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  if ((Iend-Istart) % bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(Iend %D - Istart %D) not divisible by bs %D",Iend,Istart,bs);

  /* get 'nLocalSelected' */
  for (ii=0, nLocalSelected = 0; ii < nloc; ii++) {
    PetscBool ise;
    /* filter out singletons 0 or 1? */
    ierr = PetscCDEmptyAt(agg_lists, ii, &ise);CHKERRQ(ierr);
    if (!ise) nLocalSelected++;
  }

  /* create prolongator, create P matrix */
  ierr = MatGetType(Amat,&mtype);CHKERRQ(ierr);
  ierr = MatCreate(comm, &Prol);CHKERRQ(ierr);
  ierr = MatSetSizes(Prol,nloc*bs,nLocalSelected*col_bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Prol, bs, col_bs);CHKERRQ(ierr);
  ierr = MatSetType(Prol, mtype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Prol,col_bs, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Prol,col_bs, NULL, col_bs, NULL);CHKERRQ(ierr);

  /* can get all points "removed" */
  ierr =  MatGetSize(Prol, &kk, &ii);CHKERRQ(ierr);
  if (!ii) {
    ierr = PetscInfo(pc,"No selected points on coarse grid\n");CHKERRQ(ierr);
    ierr = MatDestroy(&Prol);CHKERRQ(ierr);
    *a_P_out = NULL;  /* out */
    ierr = PetscLogEventEnd(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo1(pc,"New grid %D nodes\n",ii/col_bs);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(Prol, &myCrs0, &kk);CHKERRQ(ierr);

  if ((kk-myCrs0) % col_bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(kk %D -myCrs0 %D) not divisible by col_bs %D",kk,myCrs0,col_bs);
  myCrs0 = myCrs0/col_bs;
  if ((kk/col_bs-myCrs0) != nLocalSelected) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(kk %D/col_bs %D - myCrs0 %D) != nLocalSelected %D)",kk,col_bs,myCrs0,nLocalSelected);

  /* create global vector of data in 'data_w_ghost' */
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
#endif
  if (size > 1) { /*  */
    PetscReal *tmp_gdata,*tmp_ldata,*tp2;
    ierr = PetscMalloc1(nloc, &tmp_ldata);CHKERRQ(ierr);
    for (jj = 0; jj < col_bs; jj++) {
      for (kk = 0; kk < bs; kk++) {
        PetscInt        ii,stride;
        const PetscReal *tp = pc_gamg->data + jj*bs*nloc + kk;
        for (ii = 0; ii < nloc; ii++, tp += bs) tmp_ldata[ii] = *tp;

        ierr = PCGAMGGetDataWithGhosts(Gmat, 1, tmp_ldata, &stride, &tmp_gdata);CHKERRQ(ierr);

        if (!jj && !kk) { /* now I know how many todal nodes - allocate */
          ierr    = PetscMalloc1(stride*bs*col_bs, &data_w_ghost);CHKERRQ(ierr);
          nbnodes = bs*stride;
        }
        tp2 = data_w_ghost + jj*bs*stride + kk;
        for (ii = 0; ii < stride; ii++, tp2 += bs) *tp2 = tmp_gdata[ii];
        ierr = PetscFree(tmp_gdata);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(tmp_ldata);CHKERRQ(ierr);
  } else {
    nbnodes      = bs*nloc;
    data_w_ghost = (PetscReal*)pc_gamg->data;
  }

  /* get P0 */
  if (size > 1) {
    PetscReal *fid_glid_loc,*fiddata;
    PetscInt  stride;

    ierr = PetscMalloc1(nloc, &fid_glid_loc);CHKERRQ(ierr);
    for (kk=0; kk<nloc; kk++) fid_glid_loc[kk] = (PetscReal)(my0+kk);
    ierr = PCGAMGGetDataWithGhosts(Gmat, 1, fid_glid_loc, &stride, &fiddata);CHKERRQ(ierr);
    ierr = PetscMalloc1(stride, &flid_fgid);CHKERRQ(ierr);
    for (kk=0; kk<stride; kk++) flid_fgid[kk] = (PetscInt)fiddata[kk];
    ierr = PetscFree(fiddata);CHKERRQ(ierr);

    if (stride != nbnodes/bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"stride %D != nbnodes %D/bs %D",stride,nbnodes,bs);
    ierr = PetscFree(fid_glid_loc);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(nloc, &flid_fgid);CHKERRQ(ierr);
    for (kk=0; kk<nloc; kk++) flid_fgid[kk] = my0 + kk;
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  {
    PetscReal *data_out = NULL;
    ierr = formProl0(agg_lists, bs, col_bs, myCrs0, nbnodes,data_w_ghost, flid_fgid, &data_out, Prol);CHKERRQ(ierr);
    ierr = PetscFree(pc_gamg->data);CHKERRQ(ierr);

    pc_gamg->data           = data_out;
    pc_gamg->data_cell_rows = col_bs;
    pc_gamg->data_sz        = col_bs*col_bs*nLocalSelected;
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  if (size > 1) {ierr = PetscFree(data_w_ghost);CHKERRQ(ierr);}
  ierr = PetscFree(flid_fgid);CHKERRQ(ierr);

  *a_P_out = Prol;  /* out */

  ierr = PetscLogEventEnd(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
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
  PetscRandom    random;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PC_GAMGOptProlongator_AGG,0,0,0,0);CHKERRQ(ierr);

  /* compute maximum value of operator to be used in smoother */
  if (0 < pc_gamg_agg->nsmooths) {
    ierr = MatCreateVecs(Amat, &bb, 0);CHKERRQ(ierr);
    ierr = MatCreateVecs(Amat, &xx, 0);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&random);CHKERRQ(ierr);
    ierr = VecSetRandom(bb,random);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&random);CHKERRQ(ierr);

    ierr = KSPCreate(comm,&eksp);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(eksp,pc->erroriffailure);CHKERRQ(ierr);
    ierr = KSPSetTolerances(eksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10);CHKERRQ(ierr);
    ierr = KSPSetNormType(eksp, KSP_NORM_NONE);CHKERRQ(ierr);

    ierr = KSPSetInitialGuessNonzero(eksp, PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPSetOperators(eksp, Amat, Amat);CHKERRQ(ierr);
    ierr = KSPSetComputeSingularValues(eksp,PETSC_TRUE);CHKERRQ(ierr);

    ierr = KSPGetPC(eksp, &epc);CHKERRQ(ierr);
    ierr = PCSetType(epc, PCJACOBI);CHKERRQ(ierr);  /* smoother in smoothed agg. */

    ierr = KSPSetOptionsPrefix(eksp,((PetscObject)pc)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(eksp, "gamg_est_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(eksp);CHKERRQ(ierr);

    /* solve - keep stuff out of logging */
    ierr = PetscLogEventDeactivate(KSP_Solve);CHKERRQ(ierr);
    ierr = PetscLogEventDeactivate(PC_Apply);CHKERRQ(ierr);
    ierr = KSPSolve(eksp, bb, xx);CHKERRQ(ierr);
    ierr = KSPCheckSolve(eksp,pc,xx);CHKERRQ(ierr);
    ierr = PetscLogEventActivate(KSP_Solve);CHKERRQ(ierr);
    ierr = PetscLogEventActivate(PC_Apply);CHKERRQ(ierr);

    ierr = KSPComputeExtremeSingularValues(eksp, &emax, &emin);CHKERRQ(ierr);
    ierr = PetscInfo3(pc,"Smooth P0: max eigen=%e min=%e PC=%s\n",emax,emin,PCJACOBI);CHKERRQ(ierr);
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&bb);CHKERRQ(ierr);
    ierr = KSPDestroy(&eksp);CHKERRQ(ierr);
  }

  /* smooth P0 */
  for (jj = 0; jj < pc_gamg_agg->nsmooths; jj++) {
    Mat       tMat;
    Vec       diag;

#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif

    /* smooth P1 := (I - omega/lam D^{-1}A)P0 */
    ierr  = MatMatMult(Amat, Prol, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tMat);CHKERRQ(ierr);
    ierr  = MatCreateVecs(Amat, &diag, 0);CHKERRQ(ierr);
    ierr  = MatGetDiagonal(Amat, diag);CHKERRQ(ierr); /* effectively PCJACOBI */
    ierr  = VecReciprocal(diag);CHKERRQ(ierr);
    ierr  = MatDiagonalScale(tMat, diag, 0);CHKERRQ(ierr);
    ierr  = VecDestroy(&diag);CHKERRQ(ierr);
    alpha = -1.4/emax;
    ierr  = MatAYPX(tMat, alpha, Prol, SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr  = MatDestroy(&Prol);CHKERRQ(ierr);
    Prol  = tMat;
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif
  }
  ierr = PetscLogEventEnd(PC_GAMGOptProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg;

  PetscFunctionBegin;
  /* create sub context for SA */
  ierr            = PetscNewLog(pc,&pc_gamg_agg);CHKERRQ(ierr);
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

  pc_gamg_agg->square_graph = 1;
  pc_gamg_agg->sym_graph    = PETSC_FALSE;
  pc_gamg_agg->nsmooths     = 1;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetNSmooths_C",PCGAMGSetNSmooths_AGG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetSymGraph_C",PCGAMGSetSymGraph_AGG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGSetSquareGraph_C",PCGAMGSetSquareGraph_AGG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_AGG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
