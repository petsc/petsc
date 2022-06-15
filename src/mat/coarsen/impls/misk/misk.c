#include <petsc/private/matimpl.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

#define MIS_NOT_DONE -2
#define MIS_DELETED  -1
#define MIS_REMOVED  -3
#define MIS_IS_SELECTED(s) (s >= 0)

/* ********************************************************************** */
/* edge for priority queue */
typedef struct edge_tag {
  PetscReal weight;
  PetscInt  lid0,gid1,cpid1;
} Edge;

static PetscErrorCode PetscCoarsenDataView_private(PetscCoarsenData *agg_lists, PetscViewer viewer)
{
  PetscCDIntNd *pos,*pos2;

  PetscFunctionBegin;
  for (PetscInt kk=0; kk<agg_lists->size; kk++) {
    PetscCall(PetscCDGetHeadPos(agg_lists,kk,&pos));
    if ((pos2=pos)) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"selected local %d: ",(int)kk));
    while (pos) {
      PetscInt gid1;
      PetscCall(PetscCDIntNdGetID(pos, &gid1));
      PetscCall(PetscCDGetNextPos(agg_lists,kk,&pos));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer," %d ",(int)gid1));
    }
    if (pos2) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
  MatCoarsenApply_MISK_private - parallel heavy edge matching

  Input Parameter:
   . perm - permutation
   . Gmat - global matrix of graph (data not defined)

  Output Parameter:
   . a_locals_llist - array of list of local nodes rooted at local node
*/
static PetscErrorCode MatCoarsenApply_MISK_private(IS perm, const PetscInt misk, Mat Gmat,PetscCoarsenData **a_locals_llist)
{
  PetscBool        isMPI;
  MPI_Comm         comm;
  PetscMPIInt      rank,size;
  Mat              cMat,Prols[5],Rtot;
  PetscScalar      one = 1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(perm,IS_CLASSID,1);
  PetscValidHeaderSpecific(Gmat,MAT_CLASSID,3);
  PetscValidPointer(a_locals_llist,4);
  PetscCheck(misk < 5 && misk > 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"too many/few levels: %d",(int)misk);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPI));
  PetscCall(PetscObjectGetComm((PetscObject)Gmat,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscInfo(Gmat,"misk %d\n",(int)misk));
  /* make a copy of the graph, this gets destroyed in iterates */
  PetscCall(MatDuplicate(Gmat,MAT_COPY_VALUES,&cMat));
  for (PetscInt iterIdx=0 ; iterIdx < misk ; iterIdx++) {
    Mat_SeqAIJ       *matA,*matB=NULL;
    Mat_MPIAIJ       *mpimat=NULL;
    const PetscInt   *perm_ix;
    const PetscInt   nloc = cMat->rmap->n;
    PetscCoarsenData *agg_lists;
    PetscInt         *cpcol_gid=NULL,*cpcol_state,*lid_cprowID,*lid_state,*lid_parent_gid=NULL;
    PetscInt         num_fine_ghosts,kk,n,ix,j,*idx,*ai,iter,Iend,my0,nremoved,gid,lid,cpid,lidj,sgid,t1,t2,slid,nDone,nselected=0,state;
    PetscBool        *lid_removed,isOK;
    PetscLayout      layout;
    PetscSF          sf;

    if (isMPI) {
      mpimat = (Mat_MPIAIJ*)cMat->data;
      matA   = (Mat_SeqAIJ*)mpimat->A->data;
      matB   = (Mat_SeqAIJ*)mpimat->B->data;
      /* force compressed storage of B */
      PetscCall(MatCheckCompressedRow(mpimat->B,matB->nonzerorowcnt,&matB->compressedrow,matB->i,cMat->rmap->n,-1.0));
    } else {
      PetscBool isAIJ;
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)cMat,MATSEQAIJ,&isAIJ));
      PetscCheck(isAIJ,PETSC_COMM_SELF,PETSC_ERR_USER,"Require AIJ matrix.");
      matA = (Mat_SeqAIJ*)cMat->data;
    }
    PetscCall(MatGetOwnershipRange(cMat,&my0,&Iend));
    if (mpimat) {
      PetscInt *lid_gid;
      PetscCall(PetscMalloc1(nloc,&lid_gid)); /* explicit array needed */
      for (kk=0,gid=my0; kk<nloc; kk++,gid++) lid_gid[kk] = gid;
      PetscCall(VecGetLocalSize(mpimat->lvec, &num_fine_ghosts));
      PetscCall(PetscMalloc1(num_fine_ghosts,&cpcol_gid));
      PetscCall(PetscMalloc1(num_fine_ghosts,&cpcol_state));
      PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)cMat),&sf));
      PetscCall(MatGetLayouts(cMat,&layout,NULL));
      PetscCall(PetscSFSetGraphLayout(sf,layout,num_fine_ghosts,NULL,PETSC_COPY_VALUES,mpimat->garray));
      PetscCall(PetscSFBcastBegin(sf,MPIU_INT,lid_gid,cpcol_gid,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf,MPIU_INT,lid_gid,cpcol_gid,MPI_REPLACE));
      for (kk=0;kk<num_fine_ghosts;kk++) cpcol_state[kk]=MIS_NOT_DONE;
      PetscCall(PetscFree(lid_gid));
    } else num_fine_ghosts = 0;

    PetscCall(PetscMalloc1(nloc, &lid_cprowID));
    PetscCall(PetscMalloc1(nloc, &lid_removed)); /* explicit array needed */
    PetscCall(PetscMalloc1(nloc, &lid_parent_gid));
    PetscCall(PetscMalloc1(nloc, &lid_state));

    /* the data structure */
    PetscCall(PetscCDCreate(nloc, &agg_lists));
    /* need an inverse map - locals */
    for (kk=0; kk<nloc; kk++) {
      lid_cprowID[kk] = -1; lid_removed[kk] = PETSC_FALSE;
      lid_parent_gid[kk] = -1.0;
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
    if (!iterIdx) PetscCall(ISGetIndices(perm, &perm_ix)); // use permutation on first MIS
    else perm_ix = NULL;
    while (nDone < nloc || PETSC_TRUE) { /* asynchronous not implemented */
      iter++;
      /* check all vertices */
      for (kk=0; kk<nloc; kk++) {
        lid   = perm_ix ? perm_ix[kk] : kk;
        state = lid_state[lid];
        if (lid_removed[lid]) continue;
        if (state == MIS_NOT_DONE) {
          /* parallel test, delete if selected ghost */
          isOK = PETSC_TRUE;
          /* parallel test */
          if ((ix=lid_cprowID[lid]) != -1) { /* if I have any ghost neighbors */
            ai  = matB->compressedrow.i; n = ai[ix+1] - ai[ix];
            idx = matB->j + ai[ix];
            for (j=0; j<n; j++) {
              cpid   = idx[j]; /* compressed row ID in B mat */
              gid    = cpcol_gid[cpid];
              if (cpcol_state[cpid] == MIS_NOT_DONE && gid >= Iend) { /* or pe>rank */
                isOK = PETSC_FALSE; /* can not delete */
                break;
              }
            }
          }
          if (isOK) { /* select or remove this vertex if it is a true singleton like a BC */
            nDone++;
            /* check for singleton */
            ai = matA->i; n = ai[lid+1] - ai[lid];
            if (n < 2) {
              /* if I have any ghost adj then not a singleton */
              ix = lid_cprowID[lid];
              if (ix==-1 || !(matB->compressedrow.i[ix+1]-matB->compressedrow.i[ix])) {
                nremoved++;
                lid_removed[lid] = PETSC_TRUE;
                /* should select this because it is technically in the MIS but lets not */
                continue; /* one local adj (me) and no ghost - singleton */
              }
            }
            /* SELECTED state encoded with global index */
            lid_state[lid] = nselected; // >= 0  is selected, cache for ordering coarse grid
            nselected++;
            PetscCall(PetscCDAppendID(agg_lists, lid, lid+my0));
            /* delete local adj */
            idx = matA->j + ai[lid];
            for (j=0; j<n; j++) {
              lidj   = idx[j];
              if (lid_state[lidj] == MIS_NOT_DONE) {
                nDone++;
                PetscCall(PetscCDAppendID(agg_lists, lid, lidj+my0));
                lid_state[lidj] = MIS_DELETED;  /* delete this */
              }
            }
          } /* selected */
        } /* not done vertex */
      } /* vertex loop */

      /* update ghost states and count todos */
      if (mpimat) {
        /* scatter states, check for done */
        PetscCall(PetscSFBcastBegin(sf,MPIU_INT,lid_state,cpcol_state,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(sf,MPIU_INT,lid_state,cpcol_state,MPI_REPLACE));
        ai = matB->compressedrow.i;
        for (ix=0; ix<matB->compressedrow.nrows; ix++) {
          lid = matB->compressedrow.rindex[ix]; /* local boundary node */
          state = lid_state[lid];
          if (state == MIS_NOT_DONE) {
            /* look at ghosts */
            n   = ai[ix+1] - ai[ix];
            idx = matB->j + ai[ix];
            for (j=0; j<n; j++) {
              cpid   = idx[j]; /* compressed row ID in B mat */
              if (MIS_IS_SELECTED(cpcol_state[cpid])) { /* lid is now deleted by ghost */
                nDone++;
                lid_state[lid] = MIS_DELETED; /* delete this */
                sgid = cpcol_gid[cpid];
                lid_parent_gid[lid] = sgid; /* keep track of proc that I belong to */
                break;
              }
            }
          }
        }
        /* all done? */
        t1   = nloc - nDone;
        PetscCall(MPIU_Allreduce(&t1, &t2, 1, MPIU_INT, MPI_SUM, comm)); /* synchronous version */
        if (!t2) break;
      } else break; /* no mpi - all done */
    } /* outer parallel MIS loop */
    if (!iterIdx) PetscCall(ISRestoreIndices(perm,&perm_ix));
    PetscCall(PetscInfo(Gmat,"\t removed %" PetscInt_FMT " of %" PetscInt_FMT " vertices.  %" PetscInt_FMT " selected.\n",nremoved,nloc,nselected));

    /* tell adj who my lid_parent_gid vertices belong to - fill in agg_lists selected ghost lists */
    if (matB) {
      PetscInt *cpcol_sel_gid,*icpcol_gid;
      /* need to copy this to free buffer -- should do this globally */
      PetscCall(PetscMalloc1(num_fine_ghosts, &cpcol_sel_gid));
      PetscCall(PetscMalloc1(num_fine_ghosts, &icpcol_gid));
      for (cpid=0; cpid<num_fine_ghosts; cpid++) icpcol_gid[cpid] = cpcol_gid[cpid];
      /* get proc of deleted ghost */
      PetscCall(PetscSFBcastBegin(sf,MPIU_INT,lid_parent_gid,cpcol_sel_gid,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf,MPIU_INT,lid_parent_gid,cpcol_sel_gid,MPI_REPLACE));
      for (cpid=0; cpid<num_fine_ghosts; cpid++) {
        sgid = cpcol_sel_gid[cpid];
        gid  = icpcol_gid[cpid];
        if (sgid >= my0 && sgid < Iend) { /* I own this deleted */
          slid = sgid - my0;
          PetscCall(PetscCDAppendID(agg_lists, slid, gid));
        }
      }
      // done - cleanup
      PetscCall(PetscFree(icpcol_gid));
      PetscCall(PetscFree(cpcol_sel_gid));
      PetscCall(PetscSFDestroy(&sf));
      PetscCall(PetscFree(cpcol_gid));
      PetscCall(PetscFree(cpcol_state));
    }
    PetscCall(PetscFree(lid_cprowID));
    PetscCall(PetscFree(lid_removed));
    PetscCall(PetscFree(lid_parent_gid));
    PetscCall(PetscFree(lid_state));

    /* MIS done - make projection matrix - P */
    MatType jtype;
    PetscCall(MatGetType(Gmat,&jtype));
    PetscCall(MatCreate(comm,&Prols[iterIdx]));
    PetscCall(MatSetType(Prols[iterIdx], jtype));
    PetscCall(MatSetSizes(Prols[iterIdx],nloc, nselected, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSeqAIJSetPreallocation(Prols[iterIdx], 1, NULL));
    PetscCall(MatMPIAIJSetPreallocation(Prols[iterIdx], 1, NULL, 1, NULL));
    //PetscCall(MatCreateAIJ(comm, nloc, nselected, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &Prols[iterIdx]));
    {
      PetscCDIntNd *pos,*pos2;
      PetscInt     colIndex,Iend,fgid;
      PetscCall(MatGetOwnershipRangeColumn(Prols[iterIdx],&colIndex,&Iend));
      // TODO - order with permutation in lid_selected (reversed)
      for (PetscInt lid=0; lid < agg_lists->size; lid++) {
        PetscCall(PetscCDGetHeadPos(agg_lists,lid,&pos));
        pos2 = pos;
        while (pos) {
          PetscCall(PetscCDIntNdGetID(pos, &fgid));
          PetscCall(PetscCDGetNextPos(agg_lists,lid,&pos));
          PetscCall(MatSetValues(Prols[iterIdx],1,&fgid,1,&colIndex,&one,INSERT_VALUES));
        }
        if (pos2) colIndex++;
      }
      PetscCheck(Iend==colIndex,PETSC_COMM_SELF,PETSC_ERR_SUP,"Iend!=colIndex: %d %d",(int)Iend,(int)colIndex);
    }
    PetscCall(MatAssemblyBegin(Prols[iterIdx],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Prols[iterIdx],MAT_FINAL_ASSEMBLY));
    /* project to make new graph for next MIS, skip if last */
    if (iterIdx < misk-1) {
      Mat new_mat;
      PetscCall(MatPtAP(cMat,Prols[iterIdx],MAT_INITIAL_MATRIX,PETSC_DEFAULT,&new_mat));
      PetscCall(MatDestroy(&cMat));
      cMat = new_mat; // next iter
    } else {
      PetscCall(MatDestroy(&cMat));
    }
    // cleanup
    PetscCall(PetscCDDestroy(agg_lists));
  } /* MIS-k iteration */
  /* make total prolongator Rtot = P_0 * P_1 * ... */
  Rtot = Prols[misk-1]; // compose P then transpose to get R
  for (PetscInt iterIdx = misk - 1 ; iterIdx > 0 ; iterIdx--) {
    Mat P;
    PetscCall(MatMatMult(Prols[iterIdx-1], Rtot, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &P));
    PetscCall(MatDestroy(&Prols[iterIdx-1]));
    PetscCall(MatDestroy(&Rtot));
    Rtot = P;
  }
  PetscCall(MatTranspose(Rtot, MAT_INPLACE_MATRIX, &Rtot)); // R now
  PetscCall(MatViewFromOptions(Rtot,NULL,"-misk_aggregation_view"));
  /* make aggregates with Rtot - could use Rtot directly in theory but have to go through the aggrate list data structure */
  {
    PetscInt         Istart,Iend,ncols, NN, MM, jj = 0, max_osz = 0;
    const PetscInt   nloc = Gmat->rmap->n;
    PetscCoarsenData *agg_lists;
    Mat              mat;
    PetscCall(PetscCDCreate(nloc, &agg_lists));
    *a_locals_llist = agg_lists; // return
    PetscCall(MatGetOwnershipRange(Rtot,&Istart,&Iend));
    for (int grow=Istart,lid=0; grow<Iend; grow++,lid++) {
      const PetscInt    *idx;
      PetscCall(MatGetRow(Rtot,grow,&ncols,&idx,NULL));
      for (int jj = 0; jj < ncols; jj++) {
        PetscInt gcol = idx[jj];
        PetscCall(PetscCDAppendID(agg_lists, lid, gcol)); // local row, global column
      }
      PetscCall(MatRestoreRow(Rtot,grow,&ncols,&idx,NULL));
    }
    PetscCall(MatDestroy(&Rtot));

    /* make fake matrix, get largest */
    for (int lid=0; lid<nloc; lid++) {
      PetscCall(PetscCDSizeAt(agg_lists, lid, &jj));
      if (jj > max_osz) max_osz = jj;
    }
    PetscCall(MatGetSize(Gmat, &MM, &NN));
    if (max_osz > MM-nloc) max_osz = MM-nloc;
    PetscCall(MatGetOwnershipRange(Gmat,&Istart,NULL));
    PetscCall(MatCreateAIJ(comm, nloc, nloc,PETSC_DETERMINE, PETSC_DETERMINE, 0, NULL, max_osz, NULL, &mat));
    for (PetscInt lid=0,gidi=Istart; lid<nloc; lid++,gidi++) {
      PetscCDIntNd *pos;
      PetscCall(PetscCDGetHeadPos(agg_lists,lid,&pos));
      while (pos) {
        PetscInt gidj;
        PetscCall(PetscCDIntNdGetID(pos, &gidj));
        PetscCall(PetscCDGetNextPos(agg_lists,lid,&pos));
        if (gidj < Istart || gidj >= Istart+nloc) {
          PetscCall(MatSetValues(mat,1,&gidi,1,&gidj,&one,ADD_VALUES));
        }
      }
    }
    PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
    PetscCall(PetscCDSetMat(agg_lists, mat));
  }

  PetscFunctionReturn(0);
}

/*
   Distance k MIS. k is in 'subctx'
*/
static PetscErrorCode MatCoarsenApply_MISK(MatCoarsen coarse)
{
  Mat       mat = coarse->graph;
  PetscInt  k;

  PetscFunctionBegin;
  PetscCall(MatCoarsenMISKGetDistance(coarse,&k));
  PetscCheck(k > 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"too few levels: %d",(int)k);
  if (!coarse->perm) {
    IS       perm;
    PetscInt n,m;

    PetscCall(MatGetLocalSize(mat, &m, &n));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat), m, 0, 1, &perm));
    PetscCall(MatCoarsenApply_MISK_private(perm, (PetscInt)k, mat, &coarse->agg_lists));
    PetscCall(ISDestroy(&perm));
  } else {
    PetscCall(MatCoarsenApply_MISK_private(coarse->perm,  (PetscInt)k, mat, &coarse->agg_lists));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCoarsenView_MISK(MatCoarsen coarse, PetscViewer viewer)
{
  PetscMPIInt    rank;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)coarse),&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] MISK aggregator\n",rank));
    if (!rank) {
      PetscCall(PetscCoarsenDataView_private(coarse->agg_lists,viewer));
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCoarsenSetFromOptions_MISK(PetscOptionItems *PetscOptionsObject,MatCoarsen coarse)
{
  PetscInt  k = 1;
  PetscBool flg;
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"MatCoarsen-MISk options");
  PetscCall(PetscOptionsInt("-mat_coarsen_misk_distance","k distance for MIS","",k,&k,&flg));
  if (flg) coarse->subctx = (void*)(size_t)k;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*MC
   MATCOARSENMISK - A coarsener that uses MISK a simple greedy coarsener

   Level: beginner

.seealso: `MatCoarsenSetType()`, `MatCoarsenType`, `MatCoarsenCreate()`

M*/

PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MISK(MatCoarsen coarse)
{
  PetscFunctionBegin;
  coarse->ops->apply   = MatCoarsenApply_MISK;
  coarse->ops->view    = MatCoarsenView_MISK;
  coarse->subctx = (void*)(size_t)1;
  coarse->ops->setfromoptions = MatCoarsenSetFromOptions_MISK;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCoarsenMISKSetDistance(MatCoarsen crs,PetscInt k)
{
  PetscFunctionBegin;
  crs->subctx = (void*)(size_t)k;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCoarsenMISKGetDistance(MatCoarsen crs,PetscInt*k)
{
  PetscFunctionBegin;
  *k = (PetscInt)(size_t)crs->subctx;
  PetscFunctionReturn(0);
}
