#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

typedef struct {
  PetscReal dummy; /* empty struct; save for later */
} PC_GAMG_Classical;


#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalCreateGhostVector_Private"
PetscErrorCode PCGAMGClassicalCreateGhostVector_Private(Mat G,Vec *gvec,PetscInt **global)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    if (gvec)ierr = VecDuplicate(aij->lvec,gvec);CHKERRQ(ierr);
    if (global)*global = aij->garray;
  } else {
    /* no off-processor nodes */
    if (gvec)*gvec = NULL;
    if (global)*global = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalGraphSplitting_Private"
/*
 Split the relevant graph into diagonal and off-diagonal parts in local numbering; for now this
 a roundabout private interface to the mats' internal diag and offdiag mats.
 */
PetscErrorCode PCGAMGClassicalGraphSplitting_Private(Mat G,Mat *Gd, Mat *Go)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    *Gd = aij->A;
    *Go = aij->B;
  } else {
    *Gd = G;
    *Go = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGGraph_Classical"
PetscErrorCode PCGAMGGraph_Classical(PC pc,const Mat A,Mat *G)
{
  PetscInt          s,f,idx;
  PetscInt          r,c,ncols;
  const PetscInt    *rcol;
  const PetscScalar *rval;
  PetscInt          *gcol;
  PetscScalar       *gval;
  PetscReal         rmax;
  PetscInt          ncolstotal,cmax = 0;
  PC_MG             *mg;
  PC_GAMG           *gamg;
  PetscErrorCode    ierr;
  PetscInt          *gsparse,*lsparse;
  PetscScalar       *Amax;
  Mat               lA,gA;
  MatType           mtype;

  PetscFunctionBegin;
  mg   = (PC_MG *)pc->data;
  gamg = (PC_GAMG *)mg->innerctx;

  ierr = MatGetOwnershipRange(A,&s,&f);CHKERRQ(ierr);

  ierr = PCGAMGClassicalGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(PetscInt)*(f - s),&lsparse);CHKERRQ(ierr);
  if (gA) {ierr = PetscMalloc(sizeof(PetscInt)*(f - s),&gsparse);CHKERRQ(ierr);}
  else {
    gsparse = NULL;
  }
  ierr = PetscMalloc(sizeof(PetscScalar)*(f - s),&Amax);CHKERRQ(ierr);

  for (r = 0;r < f-s;r++) {
    lsparse[r] = 0;
    if (gsparse) gsparse[r] = 0;
  }

  for (r = 0;r < f-s;r++) {
    /* determine the maximum off-diagonal in each row */
    rmax = 0.;
    ierr = MatGetRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    ncolstotal = ncols;
    for (c = 0; c < ncols; c++) {
      if (PetscAbsScalar(rval[c]) > rmax && rcol[c] != r) {
        rmax = PetscAbsScalar(rval[c]);
      }
    }
    ierr = MatRestoreRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);

    if (gA) {
      ierr = MatGetRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
      ncolstotal += ncols;
      for (c = 0; c < ncols; c++) {
        if (PetscAbsScalar(rval[c]) > rmax) {
          rmax = PetscAbsScalar(rval[c]);
        }
      }
      ierr = MatRestoreRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    }
    Amax[r] = rmax;
    if (ncolstotal > cmax) cmax = ncolstotal;

    ierr = MatGetRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    idx = 0;

    /* create the local and global sparsity patterns */
    for (c = 0; c < ncols; c++) {
      if (PetscAbsScalar(rval[c]) > gamg->threshold*PetscRealPart(Amax[r])) {
        idx++;
      }
    }
    ierr = MatRestoreRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    lsparse[r] = idx;
    if (gA) {
      idx = 0;
      ierr = MatGetRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (c = 0; c < ncols; c++) {
        if (PetscAbsScalar(rval[c]) > gamg->threshold*PetscRealPart(Amax[r])) {
          idx++;
        }
      }
      ierr = MatRestoreRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
      gsparse[r] = idx;
    }
  }
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&gval);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&gcol);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),G); CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*G,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(*G,f-s,f-s,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*G,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*G,0,lsparse);CHKERRQ(ierr);
  for (r = s;r < f;r++) {
    ierr = MatGetRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    idx = 0;
    for (c = 0; c < ncols; c++) {
      /* classical strength of connection */
      if (PetscAbsScalar(rval[c]) > gamg->threshold*PetscRealPart(Amax[r-s])) {
        gcol[idx] = rcol[c];
        gval[idx] = rval[c];
        idx++;
      }
    }
    ierr = MatSetValues(*G,1,&r,idx,gcol,gval,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(gval);CHKERRQ(ierr);
  ierr = PetscFree(gcol);CHKERRQ(ierr);
  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = PetscFree(Amax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_Classical"
PetscErrorCode PCGAMGCoarsen_Classical(PC pc,Mat *G,PetscCoarsenData **agg_lists)
{
  PetscErrorCode   ierr;
  MatCoarsen       crs;
  MPI_Comm         fcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;


  /* construct the graph if necessary */
  if (!G) {
    SETERRQ(fcomm,PETSC_ERR_ARG_WRONGSTATE,"Must set Graph in PC in PCGAMG before coarsening");
  }

  ierr = MatCoarsenCreate(fcomm,&crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs,*G);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs,agg_lists);CHKERRQ(ierr);
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalGhost_Private"
/*
 Find all ghost nodes that are coarse and output the fine/coarse splitting for those as well

 Input:
 G - graph;
 gvec - Global Vector
 avec - Local part of the scattered vec
 bvec - Global part of the scattered vec

 Output:
 findx - indirection t

 */
PetscErrorCode PCGAMGClassicalGhost_Private(Mat G,Vec v,Vec gv)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    ierr = VecScatterBegin(aij->Mvctx,v,gv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(aij->Mvctx,v,gv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_Classical"
PetscErrorCode PCGAMGProlongator_Classical(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  Mat               lG,gG,lA,gA;     /* on and off diagonal matrices */
  PetscInt          fn;                        /* fine local blocked sizes */
  PetscInt          cn;                        /* coarse local blocked sizes */
  PetscInt          gn;                        /* size of the off-diagonal fine vector */
  PetscInt          fs,fe;                     /* fine (row) ownership range*/
  PetscInt          cs,ce;                     /* coarse (column) ownership range */
  PetscInt          i,j,k;                     /* indices! */
  PetscBool         iscoarse;                  /* flag for determining if a node is coarse */
  PetscInt          *lcid,*gcid;               /* on and off-processor coarse unknown IDs */
  PetscInt          *lsparse,*gsparse;         /* on and off-processor sparsity patterns for prolongator */
  PetscScalar       pij;
  const PetscScalar *rval;
  const PetscInt    *rcol;
  PetscScalar       g_pos,g_neg,a_pos,a_neg,diag,invdiag,alpha,beta;
  Vec               F;   /* vec of coarse size */
  Vec               C;   /* vec of fine size */
  Vec               gF;  /* vec of off-diagonal fine size */
  MatType           mtype;
  PetscInt          c_indx;
  const PetscScalar *vcols;
  const PetscInt    *icols;
  PetscScalar       c_scalar;
  PetscInt          ncols,col;
  PetscInt          row_f,row_c;
  PetscInt          cmax=0,ncolstotal,idx;
  PetscScalar       *pvals;
  PetscInt          *pcols;

  PetscFunctionBegin;
  comm = ((PetscObject)pc)->comm;
  ierr = MatGetOwnershipRange(A,&fs,&fe); CHKERRQ(ierr);
  fn = (fe - fs);

  ierr = MatGetVecs(A,&F,NULL);CHKERRQ(ierr);

  /* get the number of local unknowns and the indices of the local unknowns */

  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&gsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lcid);CHKERRQ(ierr);

  /* count the number of coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    /* filter out singletons */
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse); CHKERRQ(ierr);
    lcid[i] = -1;
    if (!iscoarse) {
      cn++;
    }
  }

   /* create the coarse vector */
  ierr = VecCreateMPI(comm,cn,PETSC_DECIDE,&C);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(C,&cs,&ce);CHKERRQ(ierr);

  /* construct a global vector indicating the global indices of the coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse); CHKERRQ(ierr);
    if (!iscoarse) {
      lcid[i] = cs+cn;
      cn++;
    } else {
      lcid[i] = -1;
    }
    c_scalar = (PetscScalar)lcid[i];
    c_indx = fs+i;
    ierr = VecSetValues(F,1,&c_indx,&c_scalar,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);

  /* split the graph into two */
  ierr = PCGAMGClassicalGraphSplitting_Private(G,&lG,&gG);CHKERRQ(ierr);
  ierr = PCGAMGClassicalGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  /* scatter to the ghost vector */
  ierr = PCGAMGClassicalCreateGhostVector_Private(G,&gF,NULL);CHKERRQ(ierr);
  ierr = PCGAMGClassicalGhost_Private(G,F,gF);CHKERRQ(ierr);

  if (gG) {
    ierr = VecGetSize(gF,&gn);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*gn,&gcid);CHKERRQ(ierr);
    for (i=0;i<gn;i++) {
      ierr = VecGetValues(gF,1,&i,&c_scalar);CHKERRQ(ierr);
      gcid[i] = (PetscInt)PetscRealPart(c_scalar);
    }
  }

  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&gF);CHKERRQ(ierr);
  ierr = VecDestroy(&C);CHKERRQ(ierr);

  /* count the on and off processor sparsity patterns for the prolongator */

  for (i=0;i<fn;i++) {
    /* on */
    ierr = MatGetRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    ncolstotal = ncols;
    lsparse[i] = 0;
    gsparse[i] = 0;
    if (lcid[i] >= 0) {
      lsparse[i] = 1;
      gsparse[i] = 0;
    } else {
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          lsparse[i] += 1;
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      ncolstotal += ncols;
      /* off */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = icols[j];
          if (gcid[col] >= 0 && vcols[j] != 0.) {
            gsparse[i] += 1;
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      }
      if (ncolstotal > cmax) cmax = ncolstotal;
    }
  }

  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&pcols);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&pvals);CHKERRQ(ierr);

  /* preallocate and create the prolongator */
  ierr = MatCreate(comm,P); CHKERRQ(ierr);
  ierr = MatGetType(G,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*P,mtype);CHKERRQ(ierr);

  ierr = MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,0,lsparse);CHKERRQ(ierr);

  /* loop over local fine nodes -- get the diagonal, the sum of positive and negative strong and weak weights, and set up the row */
  for (i = 0;i < fn;i++) {
    /* determine on or off */
    row_f = i + fs;
    row_c = lcid[i];
    if (row_c >= 0) {
      pij = 1.;
      ierr = MatSetValues(*P,1,&row_f,1,&row_c,&pij,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt nstrong=0,ntotal=0;
      g_pos = 0.;
      g_neg = 0.;
      a_pos = 0.;
      a_neg = 0.;
      diag = 0.;

      /* local strong connections */
      ierr = MatGetRow(lG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (k = 0; k < ncols; k++) {
        if (lcid[rcol[k]] >= 0) {
          if (PetscRealPart(rval[k]) > 0) {
            g_pos += rval[k];
          } else {
            g_neg += rval[k];
          }
          nstrong++;
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted strong connections */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (k = 0; k < ncols; k++) {
          if (gcid[rcol[k]] >= 0) {
            if (PetscRealPart(rval[k]) > 0.) {
              g_pos += rval[k];
            } else {
              g_neg += rval[k];
            }
            nstrong++;
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }

      /* local all connections */
      ierr = MatGetRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (k = 0; k < ncols; k++) {
        if (rcol[k] != i) {
          if (PetscRealPart(rval[k]) > 0) {
            a_pos += rval[k];
          } else {
            a_neg += rval[k];
          }
          ntotal++;
        } else diag = rval[k];
      }
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted all connections */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (k = 0; k < ncols; k++) {
          if (PetscRealPart(rval[k]) > 0.) {
            a_pos += PetscRealPart(rval[k]);
          } else {
            a_neg += PetscRealPart(rval[k]);
          }
          ntotal++;
        }
        ierr = MatRestoreRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }

      if (g_neg == 0.) {
        alpha = 0.;
      } else {
        alpha = -a_neg/g_neg;
      }

      if (g_pos == 0.) {
        diag += a_pos;
        beta = 0.;
      } else {
        beta = -a_pos/g_pos;
      }
      if (diag == 0.) {
        invdiag = 0.;
      } else invdiag = 1. / diag;
      /* on */
      ierr = MatGetRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      idx = 0;
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          row_f = i + fs;
          row_c = lcid[col];
          /* set the values for on-processor ones */
          if (PetscRealPart(vcols[j]) < 0.) {
            pij = vcols[j]*alpha*invdiag;
          } else {
            pij = vcols[j]*beta*invdiag;
          }
          if (PetscAbsScalar(pij) != 0.) {
            pvals[idx] = pij;
            pcols[idx] = row_c;
            idx++;
          }
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      /* off */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = icols[j];
          if (gcid[col] >= 0 && vcols[j] != 0.) {
            row_f = i + fs;
            row_c = gcid[col];
            /* set the values for on-processor ones */
            if (PetscRealPart(vcols[j]) < 0.) {
              pij = vcols[j]*alpha*invdiag;
            } else {
              pij = vcols[j]*beta*invdiag;
            }
            if (PetscAbsScalar(pij) != 0.) {
              pvals[idx] = pij;
              pcols[idx] = row_c;
              idx++;
            }
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      }
      ierr = MatSetValues(*P,1,&row_f,idx,pcols,pvals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = PetscFree(pcols);CHKERRQ(ierr);
  ierr = PetscFree(pvals);CHKERRQ(ierr);
  ierr = PetscFree(lcid);CHKERRQ(ierr);
  if (gG) {ierr = PetscFree(gcid);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDestroy_Classical"
PetscErrorCode PCGAMGDestroy_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetFromOptions_Classical"
PetscErrorCode PCGAMGSetFromOptions_Classical(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetData_Classical"
PetscErrorCode PCGAMGSetData_Classical(PC pc, Mat A)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  /* no data for classical AMG */
  pc_gamg->data           = NULL;
  pc_gamg->data_cell_cols = 1;
  pc_gamg->data_cell_rows = 1;
  pc_gamg->data_sz = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_Classical

*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_Classical"
PetscErrorCode  PCCreateGAMG_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *pc_gamg_classical;

  PetscFunctionBegin;
  if (pc_gamg->subctx) {
    /* call base class */
    ierr = PCDestroy_GAMG(pc);CHKERRQ(ierr);
  }

  /* create sub context for SA */
  ierr = PetscNewLog(pc, PC_GAMG_Classical, &pc_gamg_classical);CHKERRQ(ierr);
  pc_gamg->subctx = pc_gamg_classical;
  pc->ops->setfromoptions = PCGAMGSetFromOptions_Classical;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->destroy     = PCGAMGDestroy_Classical;
  pc_gamg->ops->graph       = PCGAMGGraph_Classical;
  pc_gamg->ops->coarsen     = PCGAMGCoarsen_Classical;
  pc_gamg->ops->prolongator = PCGAMGProlongator_Classical;
  pc_gamg->ops->optprol     = NULL;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Classical;
  PetscFunctionReturn(0);
}
