#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

PetscFunctionList PCGAMGClassicalProlongatorList    = NULL;
PetscBool         PCGAMGClassicalPackageInitialized = PETSC_FALSE;

typedef struct {
  PetscReal interp_threshold; /* interpolation threshold */
  char      prolongtype[256];
  PetscInt  nsmooths;         /* number of jacobi smoothings on the prolongator */
} PC_GAMG_Classical;

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalSetType"
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
  ierr = PetscTryMethod(pc,"PCGAMGClassicalSetType_C",(PC,PCGAMGType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalSetType_GAMG"
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
#define __FUNCT__ "PCGAMGProlongator_Classical_Direct"
PetscErrorCode PCGAMGProlongator_Classical_Direct(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
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
#define __FUNCT__ "PCGAMGTruncateProlongator_Private"
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

  PetscFunctionBegin;
  /* trim and rescale with reallocation */
  ierr = MatGetOwnershipRange(*P,&ps,&pf);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(*P,&pcs,&pcf);CHKERRQ(ierr);
  pn = pf-ps;
  pcn = pcf-pcs;
  ierr = PetscMalloc(sizeof(PetscInt)*pn,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*pn,&gsparse);CHKERRQ(ierr);
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

  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&pnval);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&pncol);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)*P),&Pnew);CHKERRQ(ierr);
  ierr = MatSetType(Pnew, MATAIJ);CHKERRQ(ierr);
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
    if (PetscAbsScalar(pthresh_pos) > 0.) ptot_pos /= pthresh_pos;
    if (PetscAbsScalar(pthresh_neg) > 0.) ptot_neg /= pthresh_neg;
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
  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = PetscFree(pncol);CHKERRQ(ierr);
  ierr = PetscFree(pnval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_Classical_Standard"
PetscErrorCode PCGAMGProlongator_Classical_Standard(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
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
  PetscReal         diag,jdiag,jwttotal;
  PetscScalar       *pvcol,vi;
  PetscInt          *picol;
  PetscInt          pncols;
  PetscScalar       *pcontrib,pentry,pjentry;
  /* PC_MG             *mg          = (PC_MG*)pc->data; */
  /* PC_GAMG           *gamg        = (PC_GAMG*)mg->innerctx; */

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
    *(PetscInt*)&c_scalar = cid;
    c_indx = fs+i;
    ierr = VecSetValues(v,1,&c_indx,&c_scalar,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(lscat,v,lv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(lscat,v,lv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* count to preallocate the prolongator */
  ierr = ISGetIndices(lis,&gidx);CHKERRQ(ierr);
  ierr = VecGetArray(lv,&lcid);CHKERRQ(ierr);
  maxcols = 0;
  /* count the number of unique contributing coarse cells for each fine */
  for (i=0;i<nl;i++) {
    pcontrib[i] = 0.;
    ierr = MatGetRow(lA[0],i,&ncols,&icol,NULL);CHKERRQ(ierr);
    if (gidx[i] >= fs && gidx[i] < fe) {
      li = gidx[i] - fs;
      lsparse[li] = 0;
      gsparse[li] = 0;
      cid = *(PetscInt*)&(lcid[i]);
      if (cid >= 0) {
        lsparse[li] = 1;
      } else {
        for (j=0;j<ncols;j++) {
          if (*(PetscInt*)&(lcid[icol[j]]) >= 0) {
            pcontrib[icol[j]] = 1.;
          } else {
            ci = icol[j];
            ierr = MatRestoreRow(lA[0],i,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if (*(PetscInt*)&(lcid[icol[k]]) >= 0) {
                pcontrib[icol[k]] = 1.;
              }
            }
            ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],i,&ncols,&icol,NULL);CHKERRQ(ierr);
          }
        }
        for (j=0;j<ncols;j++) {
          if (*(PetscInt*)&(lcid[icol[j]]) >= 0 && pcontrib[icol[j]] != 0.) {
            lni = *(PetscInt*)&(lcid[icol[j]]);
            if (lni >= cs && lni < ce) {
              lsparse[li]++;
            } else {
              gsparse[li]++;
            }
            pcontrib[icol[j]] = 0.;
          } else {
            ci = icol[j];
            ierr = MatRestoreRow(lA[0],i,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if (*(PetscInt*)&(lcid[icol[k]]) >= 0 && pcontrib[icol[k]] != 0.) {
                lni = *(PetscInt*)&(lcid[icol[k]]);
                if (lni >= cs && lni < ce) {
                  lsparse[li]++;
                } else {
                  gsparse[li]++;
                }
                pcontrib[icol[k]] = 0.;
              }
            }
            ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,NULL);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],i,&ncols,&icol,NULL);CHKERRQ(ierr);
          }
        }
      }
      if (lsparse[li] + gsparse[li] > maxcols) maxcols = lsparse[li]+gsparse[li];
    }
    ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
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
    diag = 0.;
    if (gidx[i] >= fs && gidx[i] < fe) {
      li = gidx[i] - fs;
      pncols=0;
      cid = *(PetscInt*)&(lcid[i]);
      if (cid >= 0) {
        pncols = 1;
        picol[0] = cid;
        pvcol[0] = 1.;
      } else {
        ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          pentry = vcol[j];
          if (*(PetscInt*)&(lcid[icol[j]]) >= 0) {
            /* coarse neighbor */
            pcontrib[icol[j]] += pentry;
          } else if (icol[j] != i) {
            /* the neighbor is a strongly connected fine node */
            ci = icol[j];
            vi = vcol[j];
            ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            jwttotal=0.;
            jdiag = 0.;
            for (k=0;k<ncols;k++) {
              if (ci == icol[k]) {
                jdiag = PetscRealPart(vcol[k]);
              }
            }
            for (k=0;k<ncols;k++) {
              if (*(PetscInt*)&(lcid[icol[k]]) >= 0 && jdiag*PetscRealPart(vcol[k]) < 0.) {
                pjentry = vcol[k];
                jwttotal += PetscRealPart(pjentry);
              }
            }
            if (jwttotal != 0.) {
              jwttotal = PetscRealPart(vi)/jwttotal;
              for (k=0;k<ncols;k++) {
                if (*(PetscInt*)&(lcid[icol[k]]) >= 0 && jdiag*PetscRealPart(vcol[k]) < 0.) {
                  pjentry = vcol[k]*jwttotal;
                  pcontrib[icol[k]] += pjentry;
                }
              }
            } else {
              diag += PetscRealPart(vi);
            }
            ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
            ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
          } else {
            diag += PetscRealPart(vcol[j]);
          }
        }
        if (diag != 0.) {
          diag = 1./diag;
          for (j=0;j<ncols;j++) {
            if (*(PetscInt*)&(lcid[icol[j]]) >= 0 && pcontrib[icol[j]] != 0.) {
              /* the neighbor is a coarse node */
              if (PetscAbsScalar(pcontrib[icol[j]]) > 0.0) {
                lni = *(PetscInt*)&(lcid[icol[j]]);
                pvcol[pncols] = -pcontrib[icol[j]]*diag;
                picol[pncols] = lni;
                pncols++;
              }
              pcontrib[icol[j]] = 0.;
            } else {
              /* the neighbor is a strongly connected fine node */
              ci = icol[j];
              ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
              ierr = MatGetRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
              for (k=0;k<ncols;k++) {
                if (*(PetscInt*)&(lcid[icol[k]]) >= 0 && pcontrib[icol[k]] != 0.) {
                  if (PetscAbsScalar(pcontrib[icol[k]]) > 0.0) {
                    lni = *(PetscInt*)&(lcid[icol[k]]);
                    pvcol[pncols] = -pcontrib[icol[k]]*diag;
                    picol[pncols] = lni;
                    pncols++;
                  }
                  pcontrib[icol[k]] = 0.;
                }
              }
              ierr = MatRestoreRow(lA[0],ci,&ncols,&icol,&vcol);CHKERRQ(ierr);
              ierr = MatGetRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
            }
            pcontrib[icol[j]] = 0.;
          }
          ierr = MatRestoreRow(lA[0],i,&ncols,&icol,&vcol);CHKERRQ(ierr);
        }
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
#define __FUNCT__ "PCGAMGOptProl_Classical_Jacobi"
PetscErrorCode PCGAMGOptProl_Classical_Jacobi(PC pc,Mat A,Mat *P)
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
  ierr = PetscMalloc(sizeof(PetscInt)*n,&coarserows);CHKERRQ(ierr);
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
  ierr = MatGetVecs(A,&diag,0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  for (i=0;i<cls->nsmooths;i++) {
    ierr = MatMatMult(A,*P,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Pnew);CHKERRQ(ierr);
    ierr = MatZeroRows(Pnew,idx,coarserows,0.,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDiagonalScale(Pnew,diag,0);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_Classical"
PetscErrorCode PCGAMGProlongator_Classical(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
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

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDestroy_Classical"
PetscErrorCode PCGAMGDestroy_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalSetType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetFromOptions_Classical"
PetscErrorCode PCGAMGSetFromOptions_Classical(PC pc)
{
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls         = (PC_GAMG_Classical*)pc_gamg->subctx;
  char              tname[256];
  PetscErrorCode    ierr;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG-Classical options");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-pc_gamg_classical_type","Type of Classical AMG prolongation",
                          "PCGAMGClassicalSetType",PCGAMGClassicalProlongatorList,cls->prolongtype, tname, sizeof(tname), &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCGAMGClassicalSetType(pc,tname);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_gamg_classical_interp_threshold","Threshold for classical interpolator entries","",cls->interp_threshold,&cls->interp_threshold,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_gamg_classical_nsmooths","Threshold for classical interpolator entries","",cls->nsmooths,&cls->nsmooths,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  pc_gamg->data = NULL;
  pc_gamg->data_cell_cols = 0;
  pc_gamg->data_cell_rows = 0;
  pc_gamg->data_sz        = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalFinalizePackage"
PetscErrorCode PCGAMGClassicalFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PCGAMGClassicalPackageInitialized = PETSC_FALSE;
  ierr = PetscFunctionListDestroy(&PCGAMGClassicalProlongatorList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalInitializePackage"
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
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_Classical"
PetscErrorCode  PCCreateGAMG_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *pc_gamg_classical;

  PetscFunctionBegin;
  ierr = PCGAMGClassicalInitializePackage();
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
  pc_gamg->ops->optprol        = PCGAMGOptProl_Classical_Jacobi;
  pc_gamg->ops->setfromoptions = PCGAMGSetFromOptions_Classical;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Classical;
  pc_gamg_classical->interp_threshold = 0.2;
  pc_gamg_classical->nsmooths         = 0;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGAMGClassicalSetType_C",PCGAMGClassicalSetType_GAMG);CHKERRQ(ierr);
  ierr = PCGAMGClassicalSetType(pc,PCGAMGCLASSICALSTANDARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
