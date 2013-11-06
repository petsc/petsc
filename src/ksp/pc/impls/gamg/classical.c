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
  PetscInt          s,f,n,idx,lidx,gidx;
  PetscInt          r,c,ncols;
  const PetscInt    *rcol;
  const PetscScalar *rval;
  PetscInt          *gcol;
  PetscScalar       *gval;
  PetscReal         rmax;
  PetscInt          cmax = 0;
  PC_MG             *mg;
  PC_GAMG           *gamg;
  PetscErrorCode    ierr;
  PetscInt          *gsparse,*lsparse;
  PetscScalar       *Amax;
  MatType           mtype;

  PetscFunctionBegin;
  mg   = (PC_MG *)pc->data;
  gamg = (PC_GAMG *)mg->innerctx;

  ierr = MatGetOwnershipRange(A,&s,&f);CHKERRQ(ierr);
  n=f-s;
  ierr = PetscMalloc(sizeof(PetscInt)*n,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*n,&gsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*n,&Amax);CHKERRQ(ierr);

  for (r = 0;r < n;r++) {
    lsparse[r] = 0;
    gsparse[r] = 0;
  }

  for (r = s;r < f;r++) {
    /* determine the maximum off-diagonal in each row */
    rmax = 0.;
    ierr = MatGetRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    for (c = 0; c < ncols; c++) {
      if (PetscRealPart(-rval[c]) > rmax && rcol[c] != r) {
        rmax = PetscRealPart(-rval[c]);
      }
    }
    Amax[r-s] = rmax;
    if (ncols > cmax) cmax = ncols;
    lidx = 0;
    gidx = 0;
    /* create the local and global sparsity patterns */
    for (c = 0; c < ncols; c++) {
      if (PetscRealPart(-rval[c]) > gamg->threshold*PetscRealPart(Amax[r-s])) {
        if (rcol[c] < f && rcol[c] >= s) {
          lidx++;
        } else {
          gidx++;
        }
      }
    }
    ierr = MatRestoreRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    lsparse[r-s] = lidx;
    gsparse[r-s] = gidx;
  }
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&gval);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&gcol);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),G); CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*G,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(*G,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*G,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*G,0,lsparse);CHKERRQ(ierr);
  for (r = s;r < f;r++) {
    ierr = MatGetRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    idx = 0;
    for (c = 0; c < ncols; c++) {
      /* classical strength of connection */
      if (PetscRealPart(-rval[c]) > gamg->threshold*PetscRealPart(Amax[r-s])) {
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
  PetscReal         *Amax_pos,*Amax_neg;
  Mat               lA,gA;                     /* on and off diagonal matrices */
  PetscInt          fn;                        /* fine local blocked sizes */
  PetscInt          cn;                        /* coarse local blocked sizes */
  PetscInt          gn;                        /* size of the off-diagonal fine vector */
  PetscInt          fs,fe;                     /* fine (row) ownership range*/
  PetscInt          cs,ce;                     /* coarse (column) ownership range */
  PetscInt          i,j;                       /* indices! */
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
  PetscScalar       c_scalar;
  PetscInt          ncols,col;
  PetscInt          row_f,row_c;
  PetscInt          cmax=0,idx;
  PetscScalar       *pvals;
  PetscInt          *pcols;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *gamg        = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  comm = ((PetscObject)pc)->comm;
  ierr = MatGetOwnershipRange(A,&fs,&fe); CHKERRQ(ierr);
  fn = (fe - fs);

  ierr = MatGetVecs(A,&F,NULL);CHKERRQ(ierr);

  /* get the number of local unknowns and the indices of the local unknowns */

  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&gsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lcid);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*fn,&Amax_pos);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*fn,&Amax_neg);CHKERRQ(ierr);

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
    *((PetscInt *)&c_scalar) = lcid[i];
    c_indx = fs+i;
    ierr = VecSetValues(F,1,&c_indx,&c_scalar,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);

  /* determine the biggest off-diagonal entries in each row */
  for (i=fs;i<fe;i++) {
    Amax_pos[i-fs] = 0.;
    Amax_neg[i-fs] = 0.;
    ierr = MatGetRow(A,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
    for(j=0;j<ncols;j++){
      if ((PetscRealPart(-rval[j]) > Amax_neg[i-fs]) && i != rcol[j]) Amax_neg[i-fs] = PetscAbsScalar(rval[j]);
      if ((PetscRealPart(rval[j])  > Amax_pos[i-fs]) && i != rcol[j]) Amax_pos[i-fs] = PetscAbsScalar(rval[j]);
    }
    if (ncols > cmax) cmax = ncols;
    ierr = MatRestoreRow(A,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&pcols);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&pvals);CHKERRQ(ierr);

  /* split the operator into two */
  ierr = PCGAMGClassicalGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  /* scatter to the ghost vector */
  ierr = PCGAMGClassicalCreateGhostVector_Private(A,&gF,NULL);CHKERRQ(ierr);
  ierr = PCGAMGClassicalGhost_Private(A,F,gF);CHKERRQ(ierr);

  if (gA) {
    ierr = VecGetSize(gF,&gn);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*gn,&gcid);CHKERRQ(ierr);
    for (i=0;i<gn;i++) {
      ierr = VecGetValues(gF,1,&i,&c_scalar);CHKERRQ(ierr);
      gcid[i] = *((PetscInt *)&c_scalar);
    }
  }

  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&gF);CHKERRQ(ierr);
  ierr = VecDestroy(&C);CHKERRQ(ierr);

  /* count the on and off processor sparsity patterns for the prolongator */
  for (i=0;i<fn;i++) {
    /* on */
    lsparse[i] = 0;
    gsparse[i] = 0;
    if (lcid[i] >= 0) {
      lsparse[i] = 1;
      gsparse[i] = 0;
    } else {
      ierr = MatGetRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (j = 0;j < ncols;j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold*Amax_neg[i])) {
          lsparse[i] += 1;
        }
      }
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      /* off */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold*Amax_neg[i])) {
            gsparse[i] += 1;
          }
        }
        ierr = MatRestoreRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }
    }
  }

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
      g_pos = 0.;
      g_neg = 0.;
      a_pos = 0.;
      a_neg = 0.;
      diag = 0.;

      /* local connections */
      ierr = MatGetRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (j = 0; j < ncols; j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold*Amax_neg[i])) {
          if (PetscRealPart(rval[j]) > 0.) {
            g_pos += rval[j];
          } else {
            g_neg += rval[j];
          }
        }
        if (col != i) {
          if (PetscRealPart(rval[j]) > 0.) {
            a_pos += rval[j];
          } else {
            a_neg += rval[j];
          }
        } else {
          diag = rval[j];
        }
      }
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted connections */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold*Amax_neg[i])) {
            if (PetscRealPart(rval[j]) > 0.) {
              g_pos += rval[j];
            } else {
              g_neg += rval[j];
            }
          }
          if (PetscRealPart(rval[j]) > 0.) {
            a_pos += rval[j];
          } else {
            a_neg += rval[j];
          }
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
      ierr = MatGetRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      idx = 0;
      for (j = 0;j < ncols;j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold*Amax_neg[i])) {
          row_f = i + fs;
          row_c = lcid[col];
          /* set the values for on-processor ones */
          if (PetscRealPart(rval[j]) < 0.) {
            pij = rval[j]*alpha*invdiag;
          } else {
            pij = rval[j]*beta*invdiag;
          }
          if (PetscAbsScalar(pij) != 0.) {
            pvals[idx] = pij;
            pcols[idx] = row_c;
            idx++;
          }
        }
      }
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      /* off */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold*Amax_neg[i])) {
            row_f = i + fs;
            row_c = gcid[col];
            /* set the values for on-processor ones */
            if (PetscRealPart(rval[j]) < 0.) {
              pij = rval[j]*alpha*invdiag;
            } else {
              pij = rval[j]*beta*invdiag;
            }
            if (PetscAbsScalar(pij) != 0.) {
              pvals[idx] = pij;
              pcols[idx] = row_c;
              idx++;
            }
          }
        }
        ierr = MatRestoreRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
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
  ierr = PetscFree(Amax_pos);CHKERRQ(ierr);
  ierr = PetscFree(Amax_neg);CHKERRQ(ierr);
  ierr = PetscFree(lcid);CHKERRQ(ierr);
  if (gA) {ierr = PetscFree(gcid);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_Standard_Classical"
PetscErrorCode PCGAMGProlongator_Standard_Classical(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
  Mat               *lA;
  Vec               lv,v,cv;
  PetscScalar       *lcid;
  IS                lis;
  PetscInt          fs,fe,cs,ce,nl,i,j,k,li,lni,ci;
  VecScatter        lscat;
  PetscInt          fn,cn,cid,c_indx;
  PetscBool         iscoarse;
  PetscScalar       c_scalar;
  const PetscScalar *vcol;
  const PetscInt    *icol;
  const PetscInt    *gidx;
  PetscInt          ncols;
  PetscInt          *lsparse,*gsparse;
  MatType           mtype;
  PetscInt          maxcols;
  PetscReal         g_pos,g_neg,a_pos,a_neg,diag,invdiag,alpha,beta;
  /* PetscReal         jdiag,invjdiag; */
  PetscReal         *amax_pos,*amax_neg;
  PetscScalar       *pvcol,vi;
  PetscInt          *picol;
  PetscInt          pncols;
  PetscScalar       *pcontrib,pentry;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *gamg        = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;

  ierr = MatGetOwnershipRange(A,&fs,&fe);CHKERRQ(ierr);
  fn = fe-fs;
  ierr = MatGetVecs(A,NULL,&v);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,fe-fs,fs,1,&lis);CHKERRQ(ierr);
  /* increase the overlap by two to get neighbors of neighbors */
  ierr = MatIncreaseOverlap(A,1,&lis,2);CHKERRQ(ierr);
  ierr = ISSort(lis);CHKERRQ(ierr);
  /* get the local part of A */
  ierr = MatGetSubMatrices(A,1,&lis,&lis,MAT_INITIAL_MATRIX,&lA);CHKERRQ(ierr);
  /* build the scatter out of it */
  ierr = ISGetLocalSize(lis,&nl);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nl,&lv);CHKERRQ(ierr);
  ierr = VecScatterCreate(v,lis,lv,NULL,&lscat);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&gsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nl,&amax_pos);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nl,&amax_neg);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nl,&pcontrib);CHKERRQ(ierr);

  /* create coarse vector */
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse);CHKERRQ(ierr);
    if (!iscoarse) {
      cn++;
    }
  }
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)A),cn,PETSC_DECIDE,&cv);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(cv,&cs,&ce);CHKERRQ(ierr);
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse); CHKERRQ(ierr);
    if (!iscoarse) {
      cid = cs+cn;
      cn++;
    } else {
      cid = -1;
    }
    c_scalar = (PetscScalar)cid;
    c_indx = fs+i;
    ierr = VecSetValues(v,1,&c_indx,&c_scalar,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(lscat,v,lv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(lscat,v,lv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* count to preallocate the prolongator */
  ierr = ISGetIndices(lis,&gidx);CHKERRQ(ierr);
  ierr = VecGetArray(lv,&lcid);CHKERRQ(ierr);
  maxcols = 0;
  for (i=0;i<nl;i++) {
    amax_pos[i] = 0.;
    amax_neg[i] = 0.;
    pcontrib[i] = 0.;
    ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
    for (j=0;j<ncols;j++) {
      if (i != icol[j]) {
        if (PetscRealPart(vcol[j]) > 0.) {
          if (amax_pos[i] < PetscAbsScalar(vcol[j])) amax_pos[i] = PetscAbsScalar(vcol[j]);
        } else {
          if (amax_neg[i] < PetscAbsScalar(vcol[j])) amax_neg[i] = PetscAbsScalar(vcol[j]);
        }
      }
    }
    ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
  }
  /* count the number of unique contributing coarse cells for each fine */
  for (i=0;i<nl;i++) {
    if (gidx[i] >= fs && gidx[i] < fe) {
      li = gidx[i] - fs;
      lsparse[li] = 0;
      gsparse[li] = 0;
      cid = (PetscInt)lcid[i];
      if (cid >= 0) {
        lsparse[li] = 1;
      } else {
        ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
        }
        for (j=0;j<ncols;j++) {
          if ((PetscInt)lcid[icol[j]] >= 0) {
            pcontrib[icol[j]] = 1.;
          } else {
            ci = icol[j];
            ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if ((PetscInt)lcid[icol[k]] >= 0) {
                pcontrib[icol[k]] = 1.;
              }
            }
            ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
          }
        }
        for (j=0;j<ncols;j++) {
          if (lcid[icol[j]] >= 0 && pcontrib[icol[j]] != 0.) {
            /* the neighbor is a coarse node */
            lni = (PetscInt)lcid[icol[j]];
            if (lni >= cs && lni < ce) {
              lsparse[li]++;
            } else {
              gsparse[li]++;
            }
            pcontrib[icol[j]] = 0.;
          } else {
            /* the neighbor is a strongly connected fine node */
            ci = icol[j];
            ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if (lcid[icol[k]] >= 0 && pcontrib[icol[k]] != 0.) {
                lni = (PetscInt)lcid[icol[k]];
                if (lni >= cs && lni < ce) {
                  lsparse[li]++;
                } else {
                  gsparse[li]++;
                }
                pcontrib[icol[k]] = 0.;
              }
            }
            ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
          }
        }
        if (lsparse[li] + gsparse[li] > maxcols) maxcols = lsparse[li] + gsparse[li];
        ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscMalloc(sizeof(PetscInt)*maxcols,&picol);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*maxcols,&pvcol);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),P);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*P,mtype);CHKERRQ(ierr);

  ierr = MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,0,lsparse);CHKERRQ(ierr);
  for (i=0;i<nl;i++) {
    if (gidx[i] >= fs && gidx[i] < fe) {
      a_pos = 0.;
      a_neg = 0.;
      g_pos = 0.;
      g_neg = 0.;
      li = gidx[i] - fs;
      pncols=0;
      cid = (PetscInt)lcid[i];
      if (cid >= 0) {
        pncols = 1;
        picol[0] = cid;
        pvcol[0] = 1.;
      } else {
        ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if (icol[j] == i) {
            diag = vcol[j];
          } else {
            if ((PetscInt)lcid[icol[j]] >= 0 && (PetscRealPart(vcol[j]) > gamg->threshold*amax_pos[i] || PetscRealPart(-vcol[j]) > gamg->threshold*amax_neg[i])) {
              if (PetscRealPart(vcol[j]) > 0.) {
                g_pos += vcol[j];
              } else {
                g_neg += vcol[j];
              }
            }
            if (PetscRealPart(vcol[j]) > 0.) {
              a_pos += vcol[j];
            } else {
              a_neg += vcol[j];
            }
          }
        }
        if (g_neg == 0.) {
          alpha = 0.;
        } else {
          alpha = a_neg/g_neg;
        }
        if (g_pos == 0.) {
          diag += a_pos;
          beta = 0.;
        } else {
          beta = a_pos/g_pos;
        }
        invdiag = 0;
        if (diag != 0.) {
          invdiag = 1./diag;
        }
        for (j=0;j<ncols;j++) {
          if (PetscRealPart(vcol[j]) > gamg->threshold*amax_pos[i] || PetscRealPart(-vcol[j]) > gamg->threshold*amax_neg[i]) {
            if (PetscRealPart(vcol[j]) < 0.) {
              pentry = -vcol[j]*invdiag*alpha;
            } else {
              pentry = -vcol[j]*invdiag*beta;
            }
            if ((PetscInt)lcid[icol[j]] >= 0) {
              /* coarse neighbor */
              pcontrib[icol[j]] = pentry;
            } else {
              /* the neighbor is a strongly connected fine node */
              ci = icol[j];
              vi = vcol[j];
              ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
              ierr = MatGetRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
              jdiag = 0.;
              invjdiag = 0.;
              for (k=0;k<ncols;k++) {
                if (ci == icol[k]) jdiag = PetscRealPart(vcol[k]);
              }
              if (jdiag != 0) {
                invjdiag = 1. / jdiag;
              }
              for (k=0;k<ncols;k++) {
                if ((PetscInt)lcid[icol[k]] >= 0 && (PetscAbsScalar(vcol[k]) > gamg->threshold*amax_pos[ci] || PetscRealPart(vcol[k]) < gamg->threshold*amax_neg[ci])) {
                  /* pcontrib[icol[k]] += -pentry*vcol[k]*invjdiag; */
                }
              }
              ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
              ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            }
          }
        }
        for (j=0;j<ncols;j++) {
          if (lcid[icol[j]] >= 0 && pcontrib[icol[j]] != 0.) {
            /* the neighbor is a coarse node */
            lni = (PetscInt)lcid[icol[j]];
            pvcol[pncols] = pcontrib[icol[j]];
            picol[pncols] = lni;
            pcontrib[icol[j]] = 0.;
            pncols++;
          } else {
            /* the neighbor is a strongly connected fine node */
            ci = icol[j];
            ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if (lcid[icol[k]] >= 0 && pcontrib[icol[k]] != 0.) {
                lni = (PetscInt)lcid[icol[k]];
                pvcol[pncols] = pcontrib[icol[k]];
                picol[pncols] = lni;
                pcontrib[icol[k]] = 0.;
                pncols++;
              }
            }
            ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
          }
        }
        ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
      }
      ci = gidx[i];
      li = gidx[i] - fs;
      if (pncols > 0) {
        ierr = MatSetValues(*P,1,&ci,pncols,picol,pvcol,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = ISRestoreIndices(lis,&gidx);CHKERRQ(ierr);
  ierr = VecRestoreArray(lv,&lcid);CHKERRQ(ierr);

  ierr = PetscFree(amax_pos);CHKERRQ(ierr);
  ierr = PetscFree(amax_neg);CHKERRQ(ierr);
  ierr = PetscFree(pcontrib);CHKERRQ(ierr);
  ierr = PetscFree(picol);CHKERRQ(ierr);
  ierr = PetscFree(pvcol);CHKERRQ(ierr);
  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = ISDestroy(&lis);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&lA);CHKERRQ(ierr);
  ierr = VecDestroy(&lv);CHKERRQ(ierr);
  ierr = VecDestroy(&cv);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&lscat);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
  Mat Pold;
  ierr = PCGAMGProlongator_Classical(pc,A,G,agg_lists,&Pold);CHKERRQ(ierr);
  ierr = MatView(Pold,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(*P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&Pold);CHKERRQ(ierr);
   */

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
  pc_gamg->data_cell_cols = 0;
  pc_gamg->data_cell_rows = 0;
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
  pc_gamg->ops->prolongator = PCGAMGProlongator_Standard_Classical;
  pc_gamg->ops->optprol     = NULL;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Classical;
  PetscFunctionReturn(0);
}
