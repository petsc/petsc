#include <petsc/private/matimpl.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

#define MIS_NOT_DONE -2
#define MIS_DELETED  -1
#define MIS_REMOVED  -3
#define MIS_IS_SELECTED(s) (s!=MIS_DELETED && s!=MIS_NOT_DONE && s!=MIS_REMOVED)

/* -------------------------------------------------------------------------- */
/*
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . perm - serial permutation of rows of local to process in MIS
   . Gmat - global matrix of graph (data not defined)
   . strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';

   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - array of list of nodes rooted at selected nodes
*/
PetscErrorCode maxIndSetAgg(IS perm,Mat Gmat,PetscBool strict_aggs,PetscCoarsenData **a_locals_llist)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *matA,*matB=NULL;
  Mat_MPIAIJ       *mpimat=NULL;
  MPI_Comm         comm;
  PetscInt         num_fine_ghosts,kk,n,ix,j,*idx,*ii,iter,Iend,my0,nremoved,gid,lid,cpid,lidj,sgid,t1,t2,slid,nDone,nselected=0,state,statej;
  PetscInt         *cpcol_gid,*cpcol_state,*lid_cprowID,*lid_gid,*cpcol_sel_gid,*icpcol_gid,*lid_state,*lid_parent_gid=NULL;
  PetscBool        *lid_removed;
  PetscBool        isMPI,isAIJ,isOK;
  const PetscInt   *perm_ix;
  const PetscInt   nloc = Gmat->rmap->n;
  PetscCoarsenData *agg_lists;
  PetscLayout      layout;
  PetscSF          sf;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Gmat,&comm);CHKERRQ(ierr);

  /* get submatrices */
  ierr = PetscObjectBaseTypeCompare((PetscObject)Gmat,MATMPIAIJ,&isMPI);CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)Gmat->data;
    matA   = (Mat_SeqAIJ*)mpimat->A->data;
    matB   = (Mat_SeqAIJ*)mpimat->B->data;
    /* force compressed storage of B */
    ierr   = MatCheckCompressedRow(mpimat->B,matB->nonzerorowcnt,&matB->compressedrow,matB->i,Gmat->rmap->n,-1.0);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectBaseTypeCompare((PetscObject)Gmat,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
    if (!isAIJ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Require AIJ matrix.");
    matA = (Mat_SeqAIJ*)Gmat->data;
  }
  ierr = MatGetOwnershipRange(Gmat,&my0,&Iend);CHKERRQ(ierr);
  ierr = PetscMalloc1(nloc,&lid_gid);CHKERRQ(ierr); /* explicit array needed */
  if (mpimat) {
    for (kk=0,gid=my0; kk<nloc; kk++,gid++) {
      lid_gid[kk] = gid;
    }
    ierr = VecGetLocalSize(mpimat->lvec, &num_fine_ghosts);CHKERRQ(ierr);
    ierr = PetscMalloc1(num_fine_ghosts,&cpcol_gid);CHKERRQ(ierr);
    ierr = PetscMalloc1(num_fine_ghosts,&cpcol_state);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)Gmat),&sf);CHKERRQ(ierr);
    ierr = MatGetLayouts(Gmat,&layout,NULL);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,layout,num_fine_ghosts,NULL,PETSC_COPY_VALUES,mpimat->garray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,MPIU_INT,lid_gid,cpcol_gid);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,lid_gid,cpcol_gid);CHKERRQ(ierr);
    for (kk=0;kk<num_fine_ghosts;kk++) {
      cpcol_state[kk]=MIS_NOT_DONE;
    }
  } else num_fine_ghosts = 0;

  ierr = PetscMalloc1(nloc, &lid_cprowID);CHKERRQ(ierr);
  ierr = PetscMalloc1(nloc, &lid_removed);CHKERRQ(ierr); /* explicit array needed */
  if (strict_aggs) {
    ierr = PetscMalloc1(nloc,&lid_parent_gid);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(nloc,&lid_state);CHKERRQ(ierr);

  /* has ghost nodes for !strict and uses local indexing (yuck) */
  ierr = PetscCDCreate(strict_aggs ? nloc : num_fine_ghosts+nloc, &agg_lists);CHKERRQ(ierr);
  if (a_locals_llist) *a_locals_llist = agg_lists;

  /* need an inverse map - locals */
  for (kk=0; kk<nloc; kk++) {
    lid_cprowID[kk] = -1; lid_removed[kk] = PETSC_FALSE;
    if (strict_aggs) {
      lid_parent_gid[kk] = -1.0;
    }
    lid_state[kk] = MIS_NOT_DONE;
  }
  /* set index into cmpressed row 'lid_cprowID' */
  if (matB) {
    for (ix=0; ix<matB->compressedrow.nrows; ix++) {
      lid = matB->compressedrow.rindex[ix];
      lid_cprowID[lid] = ix;
    }
  }
  /* MIS */
  iter = nremoved = nDone = 0;
  ierr = ISGetIndices(perm, &perm_ix);CHKERRQ(ierr);
  while (nDone < nloc || PETSC_TRUE) { /* asyncronous not implemented */
    iter++;
    /* check all vertices */
    for (kk=0; kk<nloc; kk++) {
      lid   = perm_ix[kk];
      state = lid_state[lid];
      if (lid_removed[lid]) continue;
      if (state == MIS_NOT_DONE) {
        /* parallel test, delete if selected ghost */
        isOK = PETSC_TRUE;
        if ((ix=lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
          ii  = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
          idx = matB->j + ii[ix];
          for (j=0; j<n; j++) {
            cpid   = idx[j]; /* compressed row ID in B mat */
            gid    = cpcol_gid[cpid];
            statej = cpcol_state[cpid];
            if (statej == MIS_NOT_DONE && gid >= Iend) { /* should be (pe>rank), use gid as pe proxy */
              isOK = PETSC_FALSE; /* can not delete */
              break;
            }
          }
        } /* parallel test */
        if (isOK) { /* select or remove this vertex */
          nDone++;
          /* check for singleton */
          ii = matA->i; n = ii[lid+1] - ii[lid];
          if (n < 2) {
            /* if I have any ghost adj then not a sing */
            ix = lid_cprowID[lid];
            if (ix==-1 || !(matB->compressedrow.i[ix+1]-matB->compressedrow.i[ix])) {
              nremoved++;
              lid_removed[lid] = PETSC_TRUE;
              /* should select this because it is technically in the MIS but lets not */
              continue; /* one local adj (me) and no ghost - singleton */
            }
          }
          /* SELECTED state encoded with global index */
          lid_state[lid] = lid+my0; /* needed???? */
          nselected++;
          if (strict_aggs) {
            ierr = PetscCDAppendID(agg_lists, lid, lid+my0);CHKERRQ(ierr);
          } else {
            ierr = PetscCDAppendID(agg_lists, lid, lid);CHKERRQ(ierr);
          }
          /* delete local adj */
          idx = matA->j + ii[lid];
          for (j=0; j<n; j++) {
            lidj   = idx[j];
            statej = lid_state[lidj];
            if (statej == MIS_NOT_DONE) {
              nDone++;
              if (strict_aggs) {
                ierr = PetscCDAppendID(agg_lists, lid, lidj+my0);CHKERRQ(ierr);
              } else {
                ierr = PetscCDAppendID(agg_lists, lid, lidj);CHKERRQ(ierr);
              }
              lid_state[lidj] = MIS_DELETED;  /* delete this */
            }
          }
          /* delete ghost adj of lid - deleted ghost done later for strict_aggs */
          if (!strict_aggs) {
            if ((ix=lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
              ii  = matB->compressedrow.i; n = ii[ix+1] - ii[ix];
              idx = matB->j + ii[ix];
              for (j=0; j<n; j++) {
                cpid   = idx[j]; /* compressed row ID in B mat */
                statej = cpcol_state[cpid];
                if (statej == MIS_NOT_DONE) {
                  ierr = PetscCDAppendID(agg_lists, lid, nloc+cpid);CHKERRQ(ierr);
                }
              }
            }
          }
        } /* selected */
      } /* not done vertex */
    } /* vertex loop */

    /* update ghost states and count todos */
    if (mpimat) {
      /* scatter states, check for done */
      ierr = PetscSFBcastBegin(sf,MPIU_INT,lid_state,cpcol_state);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,MPIU_INT,lid_state,cpcol_state);CHKERRQ(ierr);
      ii   = matB->compressedrow.i;
      for (ix=0; ix<matB->compressedrow.nrows; ix++) {
        lid   = matB->compressedrow.rindex[ix]; /* local boundary node */
        state = lid_state[lid];
        if (state == MIS_NOT_DONE) {
          /* look at ghosts */
          n   = ii[ix+1] - ii[ix];
          idx = matB->j + ii[ix];
          for (j=0; j<n; j++) {
            cpid   = idx[j]; /* compressed row ID in B mat */
            statej = cpcol_state[cpid];
            if (MIS_IS_SELECTED(statej)) { /* lid is now deleted, do it */
              nDone++;
              lid_state[lid] = MIS_DELETED; /* delete this */
              if (!strict_aggs) {
                lidj = nloc + cpid;
                ierr = PetscCDAppendID(agg_lists, lidj, lid);CHKERRQ(ierr);
              } else {
                sgid = cpcol_gid[cpid];
                lid_parent_gid[lid] = sgid; /* keep track of proc that I belong to */
              }
              break;
            }
          }
        }
      }
      /* all done? */
      t1   = nloc - nDone;
      ierr = MPIU_Allreduce(&t1, &t2, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr); /* synchronous version */
      if (!t2) break;
    } else break; /* all done */
  } /* outer parallel MIS loop */
  ierr = ISRestoreIndices(perm,&perm_ix);CHKERRQ(ierr);
  ierr = PetscInfo3(Gmat,"\t removed %D of %D vertices.  %D selected.\n",nremoved,nloc,nselected);CHKERRQ(ierr);

  /* tell adj who my lid_parent_gid vertices belong to - fill in agg_lists selected ghost lists */
  if (strict_aggs && matB) {
    /* need to copy this to free buffer -- should do this globaly */
    ierr = PetscMalloc1(num_fine_ghosts, &cpcol_sel_gid);CHKERRQ(ierr);
    ierr = PetscMalloc1(num_fine_ghosts, &icpcol_gid);CHKERRQ(ierr);
    for (cpid=0; cpid<num_fine_ghosts; cpid++) icpcol_gid[cpid] = cpcol_gid[cpid];

    /* get proc of deleted ghost */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,lid_parent_gid,cpcol_sel_gid);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,lid_parent_gid,cpcol_sel_gid);CHKERRQ(ierr);
    for (cpid=0; cpid<num_fine_ghosts; cpid++) {
      sgid = cpcol_sel_gid[cpid];
      gid  = icpcol_gid[cpid];
      if (sgid >= my0 && sgid < Iend) { /* I own this deleted */
        slid = sgid - my0;
        ierr = PetscCDAppendID(agg_lists, slid, gid);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(icpcol_gid);CHKERRQ(ierr);
    ierr = PetscFree(cpcol_sel_gid);CHKERRQ(ierr);
  }
  if (mpimat) {
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = PetscFree(cpcol_gid);CHKERRQ(ierr);
    ierr = PetscFree(cpcol_state);CHKERRQ(ierr);
  }
  ierr = PetscFree(lid_cprowID);CHKERRQ(ierr);
  ierr = PetscFree(lid_gid);CHKERRQ(ierr);
  ierr = PetscFree(lid_removed);CHKERRQ(ierr);
  if (strict_aggs) {
    ierr = PetscFree(lid_parent_gid);CHKERRQ(ierr);
  }
  ierr = PetscFree(lid_state);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MIS coarsen, simple greedy.
*/
static PetscErrorCode MatCoarsenApply_MIS(MatCoarsen coarse)
{
  PetscErrorCode ierr;
  Mat            mat = coarse->graph;

  PetscFunctionBegin;
  if (!coarse->perm) {
    IS       perm;
    PetscInt n,m;
    MPI_Comm comm;

    ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
    ierr = MatGetLocalSize(mat, &m, &n);CHKERRQ(ierr);
    ierr = ISCreateStride(comm, m, 0, 1, &perm);CHKERRQ(ierr);
    ierr = maxIndSetAgg(perm, mat, coarse->strict_aggs, &coarse->agg_lists);CHKERRQ(ierr);
    ierr = ISDestroy(&perm);CHKERRQ(ierr);
  } else {
    ierr = maxIndSetAgg(coarse->perm, mat, coarse->strict_aggs,  &coarse->agg_lists);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCoarsenView_MIS(MatCoarsen coarse,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)coarse),&rank);CHKERRMPI(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] MIS aggregator\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATCOARSENMIS - Creates a coarsen context via the external package MIS.

   Collective

   Input Parameter:
.  coarse - the coarsen context

   Options Database Keys:
.  -mat_coarsen_MIS_xxx -

   Level: beginner

.seealso: MatCoarsenSetType(), MatCoarsenType

M*/

PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MIS(MatCoarsen coarse)
{
  PetscFunctionBegin;
  coarse->ops->apply = MatCoarsenApply_MIS;
  coarse->ops->view  = MatCoarsenView_MIS;
  /* coarse->ops->setfromoptions = MatCoarsenSetFromOptions_MIS; */
  PetscFunctionReturn(0);
}
