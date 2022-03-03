#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petscsf.h>

PetscFunctionList PCGAMGClassicalProlongatorList    = NULL;
PetscBool         PCGAMGClassicalPackageInitialized = PETSC_FALSE;

typedef struct {
  PetscReal interp_threshold; /* interpolation threshold */
  char      prolongtype[256];
  PetscInt  nsmooths;         /* number of jacobi smoothings on the prolongator */
} PC_GAMG_Classical;

/*@C
   PCGAMGClassicalSetType - Sets the type of classical interpolation to use

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_classical_type <direct,standard> - set type of Classical AMG prolongation

   Level: intermediate

.seealso: ()
@*/
PetscErrorCode PCGAMGClassicalSetType(PC pc, PCGAMGClassicalType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCGAMGClassicalSetType_C",(PC,PCGAMGClassicalType),(pc,type)));
  PetscFunctionReturn(0);
}

/*@C
   PCGAMGClassicalGetType - Gets the type of classical interpolation to use

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  type - the type used

   Level: intermediate

.seealso: ()
@*/
PetscErrorCode PCGAMGClassicalGetType(PC pc, PCGAMGClassicalType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCGAMGClassicalGetType_C",(PC,PCGAMGClassicalType*),(pc,type)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGClassicalSetType_GAMG(PC pc, PCGAMGClassicalType type)
{
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;

  PetscFunctionBegin;
  CHKERRQ(PetscStrcpy(cls->prolongtype,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGClassicalGetType_GAMG(PC pc, PCGAMGClassicalType *type)
{
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;

  PetscFunctionBegin;
  *type = cls->prolongtype;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGGraph_Classical(PC pc,Mat A,Mat *G)
{
  PetscInt          s,f,n,idx,lidx,gidx;
  PetscInt          r,c,ncols;
  const PetscInt    *rcol;
  const PetscScalar *rval;
  PetscInt          *gcol;
  PetscScalar       *gval;
  PetscReal         rmax;
  PetscInt          cmax = 0;
  PC_MG             *mg = (PC_MG *)pc->data;
  PC_GAMG           *gamg = (PC_GAMG *)mg->innerctx;
  PetscInt          *gsparse,*lsparse;
  PetscScalar       *Amax;
  MatType           mtype;

  PetscFunctionBegin;
  CHKERRQ(MatGetOwnershipRange(A,&s,&f));
  n=f-s;
  CHKERRQ(PetscMalloc3(n,&lsparse,n,&gsparse,n,&Amax));

  for (r = 0;r < n;r++) {
    lsparse[r] = 0;
    gsparse[r] = 0;
  }

  for (r = s;r < f;r++) {
    /* determine the maximum off-diagonal in each row */
    rmax = 0.;
    CHKERRQ(MatGetRow(A,r,&ncols,&rcol,&rval));
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
      if (PetscRealPart(-rval[c]) > gamg->threshold[0]*PetscRealPart(Amax[r-s]) || rcol[c] == r) {
        if (rcol[c] < f && rcol[c] >= s) {
          lidx++;
        } else {
          gidx++;
        }
      }
    }
    CHKERRQ(MatRestoreRow(A,r,&ncols,&rcol,&rval));
    lsparse[r-s] = lidx;
    gsparse[r-s] = gidx;
  }
  CHKERRQ(PetscMalloc2(cmax,&gval,cmax,&gcol));

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),G));
  CHKERRQ(MatGetType(A,&mtype));
  CHKERRQ(MatSetType(*G,mtype));
  CHKERRQ(MatSetSizes(*G,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatMPIAIJSetPreallocation(*G,0,lsparse,0,gsparse));
  CHKERRQ(MatSeqAIJSetPreallocation(*G,0,lsparse));
  for (r = s;r < f;r++) {
    CHKERRQ(MatGetRow(A,r,&ncols,&rcol,&rval));
    idx = 0;
    for (c = 0; c < ncols; c++) {
      /* classical strength of connection */
      if (PetscRealPart(-rval[c]) > gamg->threshold[0]*PetscRealPart(Amax[r-s]) || rcol[c] == r) {
        gcol[idx] = rcol[c];
        gval[idx] = rval[c];
        idx++;
      }
    }
    CHKERRQ(MatSetValues(*G,1,&r,idx,gcol,gval,INSERT_VALUES));
    CHKERRQ(MatRestoreRow(A,r,&ncols,&rcol,&rval));
  }
  CHKERRQ(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscFree2(gval,gcol));
  CHKERRQ(PetscFree3(lsparse,gsparse,Amax));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGCoarsen_Classical(PC pc,Mat *G,PetscCoarsenData **agg_lists)
{
  MatCoarsen       crs;
  MPI_Comm         fcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;
  PetscCheck(G,fcomm,PETSC_ERR_ARG_WRONGSTATE,"Must set Graph in PC in PCGAMG before coarsening");

  CHKERRQ(MatCoarsenCreate(fcomm,&crs));
  CHKERRQ(MatCoarsenSetFromOptions(crs));
  CHKERRQ(MatCoarsenSetAdjacency(crs,*G));
  CHKERRQ(MatCoarsenSetStrictAggs(crs,PETSC_TRUE));
  CHKERRQ(MatCoarsenApply(crs));
  CHKERRQ(MatCoarsenGetData(crs,agg_lists));
  CHKERRQ(MatCoarsenDestroy(&crs));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical_Direct(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *gamg        = (PC_GAMG*)mg->innerctx;
  PetscBool         iscoarse,isMPIAIJ,isSEQAIJ;
  PetscInt          fn,cn,fs,fe,cs,ce,i,j,ncols,col,row_f,row_c,cmax=0,idx,noff;
  PetscInt          *lcid,*gcid,*lsparse,*gsparse,*colmap,*pcols;
  const PetscInt    *rcol;
  PetscReal         *Amax_pos,*Amax_neg;
  PetscScalar       g_pos,g_neg,a_pos,a_neg,diag,invdiag,alpha,beta,pij;
  PetscScalar       *pvals;
  const PetscScalar *rval;
  Mat               lA,gA=NULL;
  MatType           mtype;
  Vec               C,lvec;
  PetscLayout       clayout;
  PetscSF           sf;
  Mat_MPIAIJ        *mpiaij;

  PetscFunctionBegin;
  CHKERRQ(MatGetOwnershipRange(A,&fs,&fe));
  fn = fe-fs;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSEQAIJ));
  PetscCheckFalse(!isMPIAIJ && !isSEQAIJ,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Classical AMG requires MPIAIJ matrix");
  if (isMPIAIJ) {
    mpiaij = (Mat_MPIAIJ*)A->data;
    lA = mpiaij->A;
    gA = mpiaij->B;
    lvec = mpiaij->lvec;
    CHKERRQ(VecGetSize(lvec,&noff));
    colmap = mpiaij->garray;
    CHKERRQ(MatGetLayouts(A,NULL,&clayout));
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)A),&sf));
    CHKERRQ(PetscSFSetGraphLayout(sf,clayout,noff,NULL,PETSC_COPY_VALUES,colmap));
    CHKERRQ(PetscMalloc1(noff,&gcid));
  } else {
    lA = A;
  }
  CHKERRQ(PetscMalloc5(fn,&lsparse,fn,&gsparse,fn,&lcid,fn,&Amax_pos,fn,&Amax_neg));

  /* count the number of coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    /* filter out singletons */
    CHKERRQ(PetscCDEmptyAt(agg_lists,i,&iscoarse));
    lcid[i] = -1;
    if (!iscoarse) {
      cn++;
    }
  }

   /* create the coarse vector */
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)A),cn,PETSC_DECIDE,&C));
  CHKERRQ(VecGetOwnershipRange(C,&cs,&ce));

  cn = 0;
  for (i=0;i<fn;i++) {
    CHKERRQ(PetscCDEmptyAt(agg_lists,i,&iscoarse));
    if (!iscoarse) {
      lcid[i] = cs+cn;
      cn++;
    } else {
      lcid[i] = -1;
    }
  }

  if (gA) {
    CHKERRQ(PetscSFBcastBegin(sf,MPIU_INT,lcid,gcid,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf,MPIU_INT,lcid,gcid,MPI_REPLACE));
  }

  /* determine the largest off-diagonal entries in each row */
  for (i=fs;i<fe;i++) {
    Amax_pos[i-fs] = 0.;
    Amax_neg[i-fs] = 0.;
    CHKERRQ(MatGetRow(A,i,&ncols,&rcol,&rval));
    for (j=0;j<ncols;j++) {
      if ((PetscRealPart(-rval[j]) > Amax_neg[i-fs]) && i != rcol[j]) Amax_neg[i-fs] = PetscAbsScalar(rval[j]);
      if ((PetscRealPart(rval[j])  > Amax_pos[i-fs]) && i != rcol[j]) Amax_pos[i-fs] = PetscAbsScalar(rval[j]);
    }
    if (ncols > cmax) cmax = ncols;
    CHKERRQ(MatRestoreRow(A,i,&ncols,&rcol,&rval));
  }
  CHKERRQ(PetscMalloc2(cmax,&pcols,cmax,&pvals));
  CHKERRQ(VecDestroy(&C));

  /* count the on and off processor sparsity patterns for the prolongator */
  for (i=0;i<fn;i++) {
    /* on */
    lsparse[i] = 0;
    gsparse[i] = 0;
    if (lcid[i] >= 0) {
      lsparse[i] = 1;
      gsparse[i] = 0;
    } else {
      CHKERRQ(MatGetRow(lA,i,&ncols,&rcol,&rval));
      for (j = 0;j < ncols;j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
          lsparse[i] += 1;
        }
      }
      CHKERRQ(MatRestoreRow(lA,i,&ncols,&rcol,&rval));
      /* off */
      if (gA) {
        CHKERRQ(MatGetRow(gA,i,&ncols,&rcol,&rval));
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
            gsparse[i] += 1;
          }
        }
        CHKERRQ(MatRestoreRow(gA,i,&ncols,&rcol,&rval));
      }
    }
  }

  /* preallocate and create the prolongator */
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),P));
  CHKERRQ(MatGetType(G,&mtype));
  CHKERRQ(MatSetType(*P,mtype));
  CHKERRQ(MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse));
  CHKERRQ(MatSeqAIJSetPreallocation(*P,0,lsparse));

  /* loop over local fine nodes -- get the diagonal, the sum of positive and negative strong and weak weights, and set up the row */
  for (i = 0;i < fn;i++) {
    /* determine on or off */
    row_f = i + fs;
    row_c = lcid[i];
    if (row_c >= 0) {
      pij = 1.;
      CHKERRQ(MatSetValues(*P,1,&row_f,1,&row_c,&pij,INSERT_VALUES));
    } else {
      g_pos = 0.;
      g_neg = 0.;
      a_pos = 0.;
      a_neg = 0.;
      diag  = 0.;

      /* local connections */
      CHKERRQ(MatGetRow(lA,i,&ncols,&rcol,&rval));
      for (j = 0; j < ncols; j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
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
      CHKERRQ(MatRestoreRow(lA,i,&ncols,&rcol,&rval));

      /* ghosted connections */
      if (gA) {
        CHKERRQ(MatGetRow(gA,i,&ncols,&rcol,&rval));
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
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
        CHKERRQ(MatRestoreRow(gA,i,&ncols,&rcol,&rval));
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
      CHKERRQ(MatGetRow(lA,i,&ncols,&rcol,&rval));
      idx = 0;
      for (j = 0;j < ncols;j++) {
        col = rcol[j];
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
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
      CHKERRQ(MatRestoreRow(lA,i,&ncols,&rcol,&rval));
      /* off */
      if (gA) {
        CHKERRQ(MatGetRow(gA,i,&ncols,&rcol,&rval));
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
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
        CHKERRQ(MatRestoreRow(gA,i,&ncols,&rcol,&rval));
      }
      CHKERRQ(MatSetValues(*P,1,&row_f,idx,pcols,pvals,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscFree5(lsparse,gsparse,lcid,Amax_pos,Amax_neg));

  CHKERRQ(PetscFree2(pcols,pvals));
  if (gA) {
    CHKERRQ(PetscSFDestroy(&sf));
    CHKERRQ(PetscFree(gcid));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGTruncateProlongator_Private(PC pc,Mat *P)
{
  PetscInt          j,i,ps,pf,pn,pcs,pcf,pcn,idx,cmax;
  const PetscScalar *pval;
  const PetscInt    *pcol;
  PetscScalar       *pnval;
  PetscInt          *pncol;
  PetscInt          ncols;
  Mat               Pnew;
  PetscInt          *lsparse,*gsparse;
  PetscReal         pmax_pos,pmax_neg,ptot_pos,ptot_neg,pthresh_pos,pthresh_neg;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;
  MatType           mtype;

  PetscFunctionBegin;
  /* trim and rescale with reallocation */
  CHKERRQ(MatGetOwnershipRange(*P,&ps,&pf));
  CHKERRQ(MatGetOwnershipRangeColumn(*P,&pcs,&pcf));
  pn = pf-ps;
  pcn = pcf-pcs;
  CHKERRQ(PetscMalloc2(pn,&lsparse,pn,&gsparse));
  /* allocate */
  cmax = 0;
  for (i=ps;i<pf;i++) {
    lsparse[i-ps] = 0;
    gsparse[i-ps] = 0;
    CHKERRQ(MatGetRow(*P,i,&ncols,&pcol,&pval));
    if (ncols > cmax) {
      cmax = ncols;
    }
    pmax_pos = 0.;
    pmax_neg = 0.;
    for (j=0;j<ncols;j++) {
      if (PetscRealPart(pval[j]) > pmax_pos) {
        pmax_pos = PetscRealPart(pval[j]);
      } else if (PetscRealPart(pval[j]) < pmax_neg) {
        pmax_neg = PetscRealPart(pval[j]);
      }
    }
    for (j=0;j<ncols;j++) {
      if (PetscRealPart(pval[j]) >= pmax_pos*cls->interp_threshold || PetscRealPart(pval[j]) <= pmax_neg*cls->interp_threshold) {
        if (pcol[j] >= pcs && pcol[j] < pcf) {
          lsparse[i-ps]++;
        } else {
          gsparse[i-ps]++;
        }
      }
    }
    CHKERRQ(MatRestoreRow(*P,i,&ncols,&pcol,&pval));
  }

  CHKERRQ(PetscMalloc2(cmax,&pnval,cmax,&pncol));

  CHKERRQ(MatGetType(*P,&mtype));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)*P),&Pnew));
  CHKERRQ(MatSetType(Pnew, mtype));
  CHKERRQ(MatSetSizes(Pnew,pn,pcn,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSeqAIJSetPreallocation(Pnew,0,lsparse));
  CHKERRQ(MatMPIAIJSetPreallocation(Pnew,0,lsparse,0,gsparse));

  for (i=ps;i<pf;i++) {
    CHKERRQ(MatGetRow(*P,i,&ncols,&pcol,&pval));
    pmax_pos = 0.;
    pmax_neg = 0.;
    for (j=0;j<ncols;j++) {
      if (PetscRealPart(pval[j]) > pmax_pos) {
        pmax_pos = PetscRealPart(pval[j]);
      } else if (PetscRealPart(pval[j]) < pmax_neg) {
        pmax_neg = PetscRealPart(pval[j]);
      }
    }
    pthresh_pos = 0.;
    pthresh_neg = 0.;
    ptot_pos = 0.;
    ptot_neg = 0.;
    for (j=0;j<ncols;j++) {
      if (PetscRealPart(pval[j]) >= cls->interp_threshold*pmax_pos) {
        pthresh_pos += PetscRealPart(pval[j]);
      } else if (PetscRealPart(pval[j]) <= cls->interp_threshold*pmax_neg) {
        pthresh_neg += PetscRealPart(pval[j]);
      }
      if (PetscRealPart(pval[j]) > 0.) {
        ptot_pos += PetscRealPart(pval[j]);
      } else {
        ptot_neg += PetscRealPart(pval[j]);
      }
    }
    if (PetscAbsReal(pthresh_pos) > 0.) ptot_pos /= pthresh_pos;
    if (PetscAbsReal(pthresh_neg) > 0.) ptot_neg /= pthresh_neg;
    idx=0;
    for (j=0;j<ncols;j++) {
      if (PetscRealPart(pval[j]) >= pmax_pos*cls->interp_threshold) {
        pnval[idx] = ptot_pos*pval[j];
        pncol[idx] = pcol[j];
        idx++;
      } else if (PetscRealPart(pval[j]) <= pmax_neg*cls->interp_threshold) {
        pnval[idx] = ptot_neg*pval[j];
        pncol[idx] = pcol[j];
        idx++;
      }
    }
    CHKERRQ(MatRestoreRow(*P,i,&ncols,&pcol,&pval));
    CHKERRQ(MatSetValues(Pnew,1,&i,idx,pncol,pnval,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(Pnew, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pnew, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatDestroy(P));

  *P = Pnew;
  CHKERRQ(PetscFree2(lsparse,gsparse));
  CHKERRQ(PetscFree2(pnval,pncol));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical_Standard(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  Mat               lA,*lAs;
  MatType           mtype;
  Vec               cv;
  PetscInt          *gcid,*lcid,*lsparse,*gsparse,*picol;
  PetscInt          fs,fe,cs,ce,nl,i,j,k,li,lni,ci,ncols,maxcols,fn,cn,cid;
  PetscMPIInt       size;
  const PetscInt    *lidx,*icol,*gidx;
  PetscBool         iscoarse;
  PetscScalar       vi,pentry,pjentry;
  PetscScalar       *pcontrib,*pvcol;
  const PetscScalar *vcol;
  PetscReal         diag,jdiag,jwttotal;
  PetscInt          pncols;
  PetscSF           sf;
  PetscLayout       clayout;
  IS                lis;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  CHKERRQ(MatGetOwnershipRange(A,&fs,&fe));
  fn = fe-fs;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,fe-fs,fs,1,&lis));
  if (size > 1) {
    CHKERRQ(MatGetLayouts(A,NULL,&clayout));
    /* increase the overlap by two to get neighbors of neighbors */
    CHKERRQ(MatIncreaseOverlap(A,1,&lis,2));
    CHKERRQ(ISSort(lis));
    /* get the local part of A */
    CHKERRQ(MatCreateSubMatrices(A,1,&lis,&lis,MAT_INITIAL_MATRIX,&lAs));
    lA = lAs[0];
    /* build an SF out of it */
    CHKERRQ(ISGetLocalSize(lis,&nl));
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)A),&sf));
    CHKERRQ(ISGetIndices(lis,&lidx));
    CHKERRQ(PetscSFSetGraphLayout(sf,clayout,nl,NULL,PETSC_COPY_VALUES,lidx));
    CHKERRQ(ISRestoreIndices(lis,&lidx));
  } else {
    lA = A;
    nl = fn;
  }
  /* create a communication structure for the overlapped portion and transmit coarse indices */
  CHKERRQ(PetscMalloc3(fn,&lsparse,fn,&gsparse,nl,&pcontrib));
  /* create coarse vector */
  cn = 0;
  for (i=0;i<fn;i++) {
    CHKERRQ(PetscCDEmptyAt(agg_lists,i,&iscoarse));
    if (!iscoarse) {
      cn++;
    }
  }
  CHKERRQ(PetscMalloc1(fn,&gcid));
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)A),cn,PETSC_DECIDE,&cv));
  CHKERRQ(VecGetOwnershipRange(cv,&cs,&ce));
  cn = 0;
  for (i=0;i<fn;i++) {
    CHKERRQ(PetscCDEmptyAt(agg_lists,i,&iscoarse));
    if (!iscoarse) {
      gcid[i] = cs+cn;
      cn++;
    } else {
      gcid[i] = -1;
    }
  }
  if (size > 1) {
    CHKERRQ(PetscMalloc1(nl,&lcid));
    CHKERRQ(PetscSFBcastBegin(sf,MPIU_INT,gcid,lcid,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf,MPIU_INT,gcid,lcid,MPI_REPLACE));
  } else {
    lcid = gcid;
  }
  /* count to preallocate the prolongator */
  CHKERRQ(ISGetIndices(lis,&gidx));
  maxcols = 0;
  /* count the number of unique contributing coarse cells for each fine */
  for (i=0;i<nl;i++) {
    pcontrib[i] = 0.;
    CHKERRQ(MatGetRow(lA,i,&ncols,&icol,NULL));
    if (gidx[i] >= fs && gidx[i] < fe) {
      li = gidx[i] - fs;
      lsparse[li] = 0;
      gsparse[li] = 0;
      cid = lcid[i];
      if (cid >= 0) {
        lsparse[li] = 1;
      } else {
        for (j=0;j<ncols;j++) {
          if (lcid[icol[j]] >= 0) {
            pcontrib[icol[j]] = 1.;
          } else {
            ci = icol[j];
            CHKERRQ(MatRestoreRow(lA,i,&ncols,&icol,NULL));
            CHKERRQ(MatGetRow(lA,ci,&ncols,&icol,NULL));
            for (k=0;k<ncols;k++) {
              if (lcid[icol[k]] >= 0) {
                pcontrib[icol[k]] = 1.;
              }
            }
            CHKERRQ(MatRestoreRow(lA,ci,&ncols,&icol,NULL));
            CHKERRQ(MatGetRow(lA,i,&ncols,&icol,NULL));
          }
        }
        for (j=0;j<ncols;j++) {
          if (lcid[icol[j]] >= 0 && pcontrib[icol[j]] != 0.) {
            lni = lcid[icol[j]];
            if (lni >= cs && lni < ce) {
              lsparse[li]++;
            } else {
              gsparse[li]++;
            }
            pcontrib[icol[j]] = 0.;
          } else {
            ci = icol[j];
            CHKERRQ(MatRestoreRow(lA,i,&ncols,&icol,NULL));
            CHKERRQ(MatGetRow(lA,ci,&ncols,&icol,NULL));
            for (k=0;k<ncols;k++) {
              if (lcid[icol[k]] >= 0 && pcontrib[icol[k]] != 0.) {
                lni = lcid[icol[k]];
                if (lni >= cs && lni < ce) {
                  lsparse[li]++;
                } else {
                  gsparse[li]++;
                }
                pcontrib[icol[k]] = 0.;
              }
            }
            CHKERRQ(MatRestoreRow(lA,ci,&ncols,&icol,NULL));
            CHKERRQ(MatGetRow(lA,i,&ncols,&icol,NULL));
          }
        }
      }
      if (lsparse[li] + gsparse[li] > maxcols) maxcols = lsparse[li]+gsparse[li];
    }
    CHKERRQ(MatRestoreRow(lA,i,&ncols,&icol,&vcol));
  }
  CHKERRQ(PetscMalloc2(maxcols,&picol,maxcols,&pvcol));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),P));
  CHKERRQ(MatGetType(A,&mtype));
  CHKERRQ(MatSetType(*P,mtype));
  CHKERRQ(MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse));
  CHKERRQ(MatSeqAIJSetPreallocation(*P,0,lsparse));
  for (i=0;i<nl;i++) {
    diag = 0.;
    if (gidx[i] >= fs && gidx[i] < fe) {
      pncols=0;
      cid = lcid[i];
      if (cid >= 0) {
        pncols = 1;
        picol[0] = cid;
        pvcol[0] = 1.;
      } else {
        CHKERRQ(MatGetRow(lA,i,&ncols,&icol,&vcol));
        for (j=0;j<ncols;j++) {
          pentry = vcol[j];
          if (lcid[icol[j]] >= 0) {
            /* coarse neighbor */
            pcontrib[icol[j]] += pentry;
          } else if (icol[j] != i) {
            /* the neighbor is a strongly connected fine node */
            ci = icol[j];
            vi = vcol[j];
            CHKERRQ(MatRestoreRow(lA,i,&ncols,&icol,&vcol));
            CHKERRQ(MatGetRow(lA,ci,&ncols,&icol,&vcol));
            jwttotal=0.;
            jdiag = 0.;
            for (k=0;k<ncols;k++) {
              if (ci == icol[k]) {
                jdiag = PetscRealPart(vcol[k]);
              }
            }
            for (k=0;k<ncols;k++) {
              if (lcid[icol[k]] >= 0 && jdiag*PetscRealPart(vcol[k]) < 0.) {
                pjentry = vcol[k];
                jwttotal += PetscRealPart(pjentry);
              }
            }
            if (jwttotal != 0.) {
              jwttotal = PetscRealPart(vi)/jwttotal;
              for (k=0;k<ncols;k++) {
                if (lcid[icol[k]] >= 0 && jdiag*PetscRealPart(vcol[k]) < 0.) {
                  pjentry = vcol[k]*jwttotal;
                  pcontrib[icol[k]] += pjentry;
                }
              }
            } else {
              diag += PetscRealPart(vi);
            }
            CHKERRQ(MatRestoreRow(lA,ci,&ncols,&icol,&vcol));
            CHKERRQ(MatGetRow(lA,i,&ncols,&icol,&vcol));
          } else {
            diag += PetscRealPart(vcol[j]);
          }
        }
        if (diag != 0.) {
          diag = 1./diag;
          for (j=0;j<ncols;j++) {
            if (lcid[icol[j]] >= 0 && pcontrib[icol[j]] != 0.) {
              /* the neighbor is a coarse node */
              if (PetscAbsScalar(pcontrib[icol[j]]) > 0.0) {
                lni = lcid[icol[j]];
                pvcol[pncols] = -pcontrib[icol[j]]*diag;
                picol[pncols] = lni;
                pncols++;
              }
              pcontrib[icol[j]] = 0.;
            } else {
              /* the neighbor is a strongly connected fine node */
              ci = icol[j];
              CHKERRQ(MatRestoreRow(lA,i,&ncols,&icol,&vcol));
              CHKERRQ(MatGetRow(lA,ci,&ncols,&icol,&vcol));
              for (k=0;k<ncols;k++) {
                if (lcid[icol[k]] >= 0 && pcontrib[icol[k]] != 0.) {
                  if (PetscAbsScalar(pcontrib[icol[k]]) > 0.0) {
                    lni = lcid[icol[k]];
                    pvcol[pncols] = -pcontrib[icol[k]]*diag;
                    picol[pncols] = lni;
                    pncols++;
                  }
                  pcontrib[icol[k]] = 0.;
                }
              }
              CHKERRQ(MatRestoreRow(lA,ci,&ncols,&icol,&vcol));
              CHKERRQ(MatGetRow(lA,i,&ncols,&icol,&vcol));
            }
            pcontrib[icol[j]] = 0.;
          }
          CHKERRQ(MatRestoreRow(lA,i,&ncols,&icol,&vcol));
        }
      }
      ci = gidx[i];
      if (pncols > 0) {
        CHKERRQ(MatSetValues(*P,1,&ci,pncols,picol,pvcol,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(ISRestoreIndices(lis,&gidx));
  CHKERRQ(PetscFree2(picol,pvcol));
  CHKERRQ(PetscFree3(lsparse,gsparse,pcontrib));
  CHKERRQ(ISDestroy(&lis));
  CHKERRQ(PetscFree(gcid));
  if (size > 1) {
    CHKERRQ(PetscFree(lcid));
    CHKERRQ(MatDestroyMatrices(1,&lAs));
    CHKERRQ(PetscSFDestroy(&sf));
  }
  CHKERRQ(VecDestroy(&cv));
  CHKERRQ(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGOptProlongator_Classical_Jacobi(PC pc,Mat A,Mat *P)
{

  PetscInt          f,s,n,cf,cs,i,idx;
  PetscInt          *coarserows;
  PetscInt          ncols;
  const PetscInt    *pcols;
  const PetscScalar *pvals;
  Mat               Pnew;
  Vec               diag;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;

  PetscFunctionBegin;
  if (cls->nsmooths == 0) {
    CHKERRQ(PCGAMGTruncateProlongator_Private(pc,P));
    PetscFunctionReturn(0);
  }
  CHKERRQ(MatGetOwnershipRange(*P,&s,&f));
  n = f-s;
  CHKERRQ(MatGetOwnershipRangeColumn(*P,&cs,&cf));
  CHKERRQ(PetscMalloc1(n,&coarserows));
  /* identify the rows corresponding to coarse unknowns */
  idx = 0;
  for (i=s;i<f;i++) {
    CHKERRQ(MatGetRow(*P,i,&ncols,&pcols,&pvals));
    /* assume, for now, that it's a coarse unknown if it has a single unit entry */
    if (ncols == 1) {
      if (pvals[0] == 1.) {
        coarserows[idx] = i;
        idx++;
      }
    }
    CHKERRQ(MatRestoreRow(*P,i,&ncols,&pcols,&pvals));
  }
  CHKERRQ(MatCreateVecs(A,&diag,NULL));
  CHKERRQ(MatGetDiagonal(A,diag));
  CHKERRQ(VecReciprocal(diag));
  for (i=0;i<cls->nsmooths;i++) {
    CHKERRQ(MatMatMult(A,*P,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Pnew));
    CHKERRQ(MatZeroRows(Pnew,idx,coarserows,0.,NULL,NULL));
    CHKERRQ(MatDiagonalScale(Pnew,diag,NULL));
    CHKERRQ(MatAYPX(Pnew,-1.0,*P,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatDestroy(P));
    *P  = Pnew;
    Pnew = NULL;
  }
  CHKERRQ(VecDestroy(&diag));
  CHKERRQ(PetscFree(coarserows));
  CHKERRQ(PCGAMGTruncateProlongator_Private(pc,P));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    (*f)(PC,Mat,Mat,PetscCoarsenData*,Mat*);
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListFind(PCGAMGClassicalProlongatorList,cls->prolongtype,&f));
  PetscCheck(f,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot find PCGAMG Classical prolongator type");
  CHKERRQ((*f)(pc,A,G,agg_lists,P));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGDestroy_Classical(PC pc)
{
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(pc_gamg->subctx));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalSetType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalGetType_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGSetFromOptions_Classical(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;
  char              tname[256];
  PetscBool         flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"GAMG-Classical options"));
  CHKERRQ(PetscOptionsFList("-pc_gamg_classical_type","Type of Classical AMG prolongation","PCGAMGClassicalSetType",PCGAMGClassicalProlongatorList,cls->prolongtype, tname, sizeof(tname), &flg));
  if (flg) {
    CHKERRQ(PCGAMGClassicalSetType(pc,tname));
  }
  CHKERRQ(PetscOptionsReal("-pc_gamg_classical_interp_threshold","Threshold for classical interpolator entries","",cls->interp_threshold,&cls->interp_threshold,NULL));
  CHKERRQ(PetscOptionsInt("-pc_gamg_classical_nsmooths","Threshold for classical interpolator entries","",cls->nsmooths,&cls->nsmooths,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGSetData_Classical(PC pc, Mat A)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  /* no data for classical AMG */
  pc_gamg->data           = NULL;
  pc_gamg->data_cell_cols = 0;
  pc_gamg->data_cell_rows = 0;
  pc_gamg->data_sz        = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGClassicalFinalizePackage(void)
{
  PetscFunctionBegin;
  PCGAMGClassicalPackageInitialized = PETSC_FALSE;
  CHKERRQ(PetscFunctionListDestroy(&PCGAMGClassicalProlongatorList));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGClassicalInitializePackage(void)
{
  PetscFunctionBegin;
  if (PCGAMGClassicalPackageInitialized) PetscFunctionReturn(0);
  CHKERRQ(PetscFunctionListAdd(&PCGAMGClassicalProlongatorList,PCGAMGCLASSICALDIRECT,PCGAMGProlongator_Classical_Direct));
  CHKERRQ(PetscFunctionListAdd(&PCGAMGClassicalProlongatorList,PCGAMGCLASSICALSTANDARD,PCGAMGProlongator_Classical_Standard));
  CHKERRQ(PetscRegisterFinalize(PCGAMGClassicalFinalizePackage));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_Classical

*/
PetscErrorCode  PCCreateGAMG_Classical(PC pc)
{
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *pc_gamg_classical;

  PetscFunctionBegin;
  CHKERRQ(PCGAMGClassicalInitializePackage());
  if (pc_gamg->subctx) {
    /* call base class */
    CHKERRQ(PCDestroy_GAMG(pc));
  }

  /* create sub context for SA */
  CHKERRQ(PetscNewLog(pc,&pc_gamg_classical));
  pc_gamg->subctx = pc_gamg_classical;
  pc->ops->setfromoptions = PCGAMGSetFromOptions_Classical;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->destroy        = PCGAMGDestroy_Classical;
  pc_gamg->ops->graph          = PCGAMGGraph_Classical;
  pc_gamg->ops->coarsen        = PCGAMGCoarsen_Classical;
  pc_gamg->ops->prolongator    = PCGAMGProlongator_Classical;
  pc_gamg->ops->optprolongator = PCGAMGOptProlongator_Classical_Jacobi;
  pc_gamg->ops->setfromoptions = PCGAMGSetFromOptions_Classical;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Classical;
  pc_gamg_classical->interp_threshold = 0.2;
  pc_gamg_classical->nsmooths         = 0;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalSetType_C",PCGAMGClassicalSetType_GAMG));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalGetType_C",PCGAMGClassicalGetType_GAMG));
  CHKERRQ(PCGAMGClassicalSetType(pc,PCGAMGCLASSICALSTANDARD));
  PetscFunctionReturn(0);
}
