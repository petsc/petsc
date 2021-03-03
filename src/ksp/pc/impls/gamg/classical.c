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
.  -pc_gamg_classical_type

   Level: intermediate

.seealso: ()
@*/
PetscErrorCode PCGAMGClassicalSetType(PC pc, PCGAMGClassicalType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGClassicalSetType_C",(PC,PCGAMGClassicalType),(pc,type));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCGAMGClassicalGetType_C",(PC,PCGAMGClassicalType*),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCGAMGClassicalSetType_GAMG(PC pc, PCGAMGClassicalType type)
{
  PetscErrorCode    ierr;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;

  PetscFunctionBegin;
  ierr = PetscStrcpy(cls->prolongtype,type);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          *gsparse,*lsparse;
  PetscScalar       *Amax;
  MatType           mtype;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(A,&s,&f);CHKERRQ(ierr);
  n=f-s;
  ierr = PetscMalloc3(n,&lsparse,n,&gsparse,n,&Amax);CHKERRQ(ierr);

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
      if (PetscRealPart(-rval[c]) > gamg->threshold[0]*PetscRealPart(Amax[r-s]) || rcol[c] == r) {
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
  ierr = PetscMalloc2(cmax,&gval,cmax,&gcol);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),G);CHKERRQ(ierr);
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
      if (PetscRealPart(-rval[c]) > gamg->threshold[0]*PetscRealPart(Amax[r-s]) || rcol[c] == r) {
        gcol[idx] = rcol[c];
        gval[idx] = rval[c];
        idx++;
      }
    }
    ierr = MatSetValues(*G,1,&r,idx,gcol,gval,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(gval,gcol);CHKERRQ(ierr);
  ierr = PetscFree3(lsparse,gsparse,Amax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PCGAMGCoarsen_Classical(PC pc,Mat *G,PetscCoarsenData **agg_lists)
{
  PetscErrorCode   ierr;
  MatCoarsen       crs;
  MPI_Comm         fcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;
  if (!G) SETERRQ(fcomm,PETSC_ERR_ARG_WRONGSTATE,"Must set Graph in PC in PCGAMG before coarsening");

  ierr = MatCoarsenCreate(fcomm,&crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs,*G);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs,agg_lists);CHKERRQ(ierr);
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical_Direct(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
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
  ierr = MatGetOwnershipRange(A,&fs,&fe);CHKERRQ(ierr);
  fn = fe-fs;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  if (!isMPIAIJ && !isSEQAIJ) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Classical AMG requires MPIAIJ matrix");
  if (isMPIAIJ) {
    mpiaij = (Mat_MPIAIJ*)A->data;
    lA = mpiaij->A;
    gA = mpiaij->B;
    lvec = mpiaij->lvec;
    ierr = VecGetSize(lvec,&noff);CHKERRQ(ierr);
    colmap = mpiaij->garray;
    ierr = MatGetLayouts(A,NULL,&clayout);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)A),&sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,clayout,noff,NULL,PETSC_COPY_VALUES,colmap);CHKERRQ(ierr);
    ierr = PetscMalloc1(noff,&gcid);CHKERRQ(ierr);
  } else {
    lA = A;
  }
  ierr = PetscMalloc5(fn,&lsparse,fn,&gsparse,fn,&lcid,fn,&Amax_pos,fn,&Amax_neg);CHKERRQ(ierr);

  /* count the number of coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    /* filter out singletons */
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse);CHKERRQ(ierr);
    lcid[i] = -1;
    if (!iscoarse) {
      cn++;
    }
  }

   /* create the coarse vector */
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)A),cn,PETSC_DECIDE,&C);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(C,&cs,&ce);CHKERRQ(ierr);

  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse);CHKERRQ(ierr);
    if (!iscoarse) {
      lcid[i] = cs+cn;
      cn++;
    } else {
      lcid[i] = -1;
    }
  }

  if (gA) {
    ierr = PetscSFBcastBegin(sf,MPIU_INT,lcid,gcid,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,lcid,gcid,MPI_REPLACE);CHKERRQ(ierr);
  }

  /* determine the largest off-diagonal entries in each row */
  for (i=fs;i<fe;i++) {
    Amax_pos[i-fs] = 0.;
    Amax_neg[i-fs] = 0.;
    ierr = MatGetRow(A,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
    for (j=0;j<ncols;j++){
      if ((PetscRealPart(-rval[j]) > Amax_neg[i-fs]) && i != rcol[j]) Amax_neg[i-fs] = PetscAbsScalar(rval[j]);
      if ((PetscRealPart(rval[j])  > Amax_pos[i-fs]) && i != rcol[j]) Amax_pos[i-fs] = PetscAbsScalar(rval[j]);
    }
    if (ncols > cmax) cmax = ncols;
    ierr = MatRestoreRow(A,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(cmax,&pcols,cmax,&pvals);CHKERRQ(ierr);
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
        if (lcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
          lsparse[i] += 1;
        }
      }
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      /* off */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = rcol[j];
          if (gcid[col] >= 0 && (PetscRealPart(rval[j]) > gamg->threshold[0]*Amax_pos[i] || PetscRealPart(-rval[j]) > gamg->threshold[0]*Amax_neg[i])) {
            gsparse[i] += 1;
          }
        }
        ierr = MatRestoreRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }
    }
  }

  /* preallocate and create the prolongator */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),P);CHKERRQ(ierr);
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
      diag  = 0.;

      /* local connections */
      ierr = MatGetRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
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
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted connections */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
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
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      /* off */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
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
        ierr = MatRestoreRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }
      ierr = MatSetValues(*P,1,&row_f,idx,pcols,pvals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree5(lsparse,gsparse,lcid,Amax_pos,Amax_neg);CHKERRQ(ierr);

  ierr = PetscFree2(pcols,pvals);CHKERRQ(ierr);
  if (gA) {
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = PetscFree(gcid);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGTruncateProlongator_Private(PC pc,Mat *P)
{
  PetscInt          j,i,ps,pf,pn,pcs,pcf,pcn,idx,cmax;
  PetscErrorCode    ierr;
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
  ierr = MatGetOwnershipRange(*P,&ps,&pf);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(*P,&pcs,&pcf);CHKERRQ(ierr);
  pn = pf-ps;
  pcn = pcf-pcs;
  ierr = PetscMalloc2(pn,&lsparse,pn,&gsparse);CHKERRQ(ierr);
  /* allocate */
  cmax = 0;
  for (i=ps;i<pf;i++) {
    lsparse[i-ps] = 0;
    gsparse[i-ps] = 0;
    ierr = MatGetRow(*P,i,&ncols,&pcol,&pval);CHKERRQ(ierr);
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
    ierr = MatRestoreRow(*P,i,&ncols,&pcol,&pval);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(cmax,&pnval,cmax,&pncol);CHKERRQ(ierr);

  ierr = MatGetType(*P,&mtype);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)*P),&Pnew);CHKERRQ(ierr);
  ierr = MatSetType(Pnew, mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(Pnew,pn,pcn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Pnew,0,lsparse);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Pnew,0,lsparse,0,gsparse);CHKERRQ(ierr);

  for (i=ps;i<pf;i++) {
    ierr = MatGetRow(*P,i,&ncols,&pcol,&pval);CHKERRQ(ierr);
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
    ierr = MatRestoreRow(*P,i,&ncols,&pcol,&pval);CHKERRQ(ierr);
    ierr = MatSetValues(Pnew,1,&i,idx,pncol,pnval,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(Pnew, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pnew, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);

  *P = Pnew;
  ierr = PetscFree2(lsparse,gsparse);CHKERRQ(ierr);
  ierr = PetscFree2(pnval,pncol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical_Standard(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = MatGetOwnershipRange(A,&fs,&fe);CHKERRQ(ierr);
  fn = fe-fs;
  ierr = ISCreateStride(PETSC_COMM_SELF,fe-fs,fs,1,&lis);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatGetLayouts(A,NULL,&clayout);CHKERRQ(ierr);
    /* increase the overlap by two to get neighbors of neighbors */
    ierr = MatIncreaseOverlap(A,1,&lis,2);CHKERRQ(ierr);
    ierr = ISSort(lis);CHKERRQ(ierr);
    /* get the local part of A */
    ierr = MatCreateSubMatrices(A,1,&lis,&lis,MAT_INITIAL_MATRIX,&lAs);CHKERRQ(ierr);
    lA = lAs[0];
    /* build an SF out of it */
    ierr = ISGetLocalSize(lis,&nl);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)A),&sf);CHKERRQ(ierr);
    ierr = ISGetIndices(lis,&lidx);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,clayout,nl,NULL,PETSC_COPY_VALUES,lidx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(lis,&lidx);CHKERRQ(ierr);
  } else {
    lA = A;
    nl = fn;
  }
  /* create a communication structure for the overlapped portion and transmit coarse indices */
  ierr = PetscMalloc3(fn,&lsparse,fn,&gsparse,nl,&pcontrib);CHKERRQ(ierr);
  /* create coarse vector */
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse);CHKERRQ(ierr);
    if (!iscoarse) {
      cn++;
    }
  }
  ierr = PetscMalloc1(fn,&gcid);CHKERRQ(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)A),cn,PETSC_DECIDE,&cv);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(cv,&cs,&ce);CHKERRQ(ierr);
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse);CHKERRQ(ierr);
    if (!iscoarse) {
      gcid[i] = cs+cn;
      cn++;
    } else {
      gcid[i] = -1;
    }
  }
  if (size > 1) {
    ierr = PetscMalloc1(nl,&lcid);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,MPIU_INT,gcid,lcid,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,gcid,lcid,MPI_REPLACE);CHKERRQ(ierr);
  } else {
    lcid = gcid;
  }
  /* count to preallocate the prolongator */
  ierr = ISGetIndices(lis,&gidx);CHKERRQ(ierr);
  maxcols = 0;
  /* count the number of unique contributing coarse cells for each fine */
  for (i=0;i<nl;i++) {
    pcontrib[i] = 0.;
    ierr = MatGetRow(lA,i,&ncols,&icol,NULL);CHKERRQ(ierr);
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
            ierr = MatRestoreRow(lA,i,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA,ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if (lcid[icol[k]] >= 0) {
                pcontrib[icol[k]] = 1.;
              }
            }
            ierr = MatRestoreRow(lA,ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA,i,&ncols,&icol,NULL);CHKERRQ(ierr);
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
            ierr = MatRestoreRow(lA,i,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA,ci,&ncols,&icol,NULL);CHKERRQ(ierr);
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
            ierr = MatRestoreRow(lA,ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA,i,&ncols,&icol,NULL);CHKERRQ(ierr);
          }
        }
      }
      if (lsparse[li] + gsparse[li] > maxcols) maxcols = lsparse[li]+gsparse[li];
    }
    ierr = MatRestoreRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(maxcols,&picol,maxcols,&pvcol);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),P);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*P,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,0,lsparse);CHKERRQ(ierr);
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
        ierr = MatGetRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          pentry = vcol[j];
          if (lcid[icol[j]] >= 0) {
            /* coarse neighbor */
            pcontrib[icol[j]] += pentry;
          } else if (icol[j] != i) {
            /* the neighbor is a strongly connected fine node */
            ci = icol[j];
            vi = vcol[j];
            ierr = MatRestoreRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA,ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
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
            ierr = MatRestoreRow(lA,ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
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
              ierr = MatRestoreRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
              ierr = MatGetRow(lA,ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
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
              ierr = MatRestoreRow(lA,ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
              ierr = MatGetRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            }
            pcontrib[icol[j]] = 0.;
          }
          ierr = MatRestoreRow(lA,i,&ncols,&icol,&vcol);CHKERRQ(ierr);
        }
      }
      ci = gidx[i];
      if (pncols > 0) {
        ierr = MatSetValues(*P,1,&ci,pncols,picol,pvcol,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISRestoreIndices(lis,&gidx);CHKERRQ(ierr);
  ierr = PetscFree2(picol,pvcol);CHKERRQ(ierr);
  ierr = PetscFree3(lsparse,gsparse,pcontrib);CHKERRQ(ierr);
  ierr = ISDestroy(&lis);CHKERRQ(ierr);
  ierr = PetscFree(gcid);CHKERRQ(ierr);
  if (size > 1) {
    ierr = PetscFree(lcid);CHKERRQ(ierr);
    ierr = MatDestroyMatrices(1,&lAs);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&cv);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGOptProlongator_Classical_Jacobi(PC pc,Mat A,Mat *P)
{

  PetscErrorCode    ierr;
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
    ierr = PCGAMGTruncateProlongator_Private(pc,P);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MatGetOwnershipRange(*P,&s,&f);CHKERRQ(ierr);
  n = f-s;
  ierr = MatGetOwnershipRangeColumn(*P,&cs,&cf);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&coarserows);CHKERRQ(ierr);
  /* identify the rows corresponding to coarse unknowns */
  idx = 0;
  for (i=s;i<f;i++) {
    ierr = MatGetRow(*P,i,&ncols,&pcols,&pvals);CHKERRQ(ierr);
    /* assume, for now, that it's a coarse unknown if it has a single unit entry */
    if (ncols == 1) {
      if (pvals[0] == 1.) {
        coarserows[idx] = i;
        idx++;
      }
    }
    ierr = MatRestoreRow(*P,i,&ncols,&pcols,&pvals);CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(A,&diag,NULL);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  for (i=0;i<cls->nsmooths;i++) {
    ierr = MatMatMult(A,*P,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Pnew);CHKERRQ(ierr);
    ierr = MatZeroRows(Pnew,idx,coarserows,0.,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDiagonalScale(Pnew,diag,NULL);CHKERRQ(ierr);
    ierr = MatAYPX(Pnew,-1.0,*P,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(P);CHKERRQ(ierr);
    *P  = Pnew;
    Pnew = NULL;
  }
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  ierr = PetscFree(coarserows);CHKERRQ(ierr);
  ierr = PCGAMGTruncateProlongator_Private(pc,P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGProlongator_Classical(PC pc, Mat A, Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
  PetscErrorCode    (*f)(PC,Mat,Mat,PetscCoarsenData*,Mat*);
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;

  PetscFunctionBegin;
  ierr = PetscFunctionListFind(PCGAMGClassicalProlongatorList,cls->prolongtype,&f);CHKERRQ(ierr);
  if (!f)SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Cannot find PCGAMG Classical prolongator type");
  ierr = (*f)(pc,A,G,agg_lists,P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGDestroy_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalGetType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGSetFromOptions_Classical(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;
  char              tname[256];
  PetscErrorCode    ierr;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"GAMG-Classical options");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-pc_gamg_classical_type","Type of Classical AMG prolongation","PCGAMGClassicalSetType",PCGAMGClassicalProlongatorList,cls->prolongtype, tname, sizeof(tname), &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCGAMGClassicalSetType(pc,tname);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_gamg_classical_interp_threshold","Threshold for classical interpolator entries","",cls->interp_threshold,&cls->interp_threshold,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_gamg_classical_nsmooths","Threshold for classical interpolator entries","",cls->nsmooths,&cls->nsmooths,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PCGAMGClassicalPackageInitialized = PETSC_FALSE;
  ierr = PetscFunctionListDestroy(&PCGAMGClassicalProlongatorList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGClassicalInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PCGAMGClassicalPackageInitialized) PetscFunctionReturn(0);
  ierr = PetscFunctionListAdd(&PCGAMGClassicalProlongatorList,PCGAMGCLASSICALDIRECT,PCGAMGProlongator_Classical_Direct);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PCGAMGClassicalProlongatorList,PCGAMGCLASSICALSTANDARD,PCGAMGProlongator_Classical_Standard);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PCGAMGClassicalFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_Classical

*/
PetscErrorCode  PCCreateGAMG_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *pc_gamg_classical;

  PetscFunctionBegin;
  ierr = PCGAMGClassicalInitializePackage();CHKERRQ(ierr);
  if (pc_gamg->subctx) {
    /* call base class */
    ierr = PCDestroy_GAMG(pc);CHKERRQ(ierr);
  }

  /* create sub context for SA */
  ierr = PetscNewLog(pc,&pc_gamg_classical);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalSetType_C",PCGAMGClassicalSetType_GAMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalGetType_C",PCGAMGClassicalGetType_GAMG);CHKERRQ(ierr);
  ierr = PCGAMGClassicalSetType(pc,PCGAMGCLASSICALSTANDARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
