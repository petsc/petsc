#define PETSCKSP_DLL

/*

*/
#include "private/pcimpl.h"     /*I "petscpc.h" I*/

const char *PCFieldSplitSchurPreTypes[] = {"self","diag","user","PCFieldSplitSchurPreType","PC_FIELDSPLIT_SCHUR_PRE_",0};

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  KSP               ksp;
  Vec               x,y;
  PetscInt          nfields;
  PetscInt          *fields;
  VecScatter        sctx;
  IS                is;
  PC_FieldSplitLink next,previous;
};

typedef struct {
  PCCompositeType   type;
  PetscTruth        defaultsplit; /* Flag for a system with a set of 'k' scalar fields with the same layout (and bs = k) */
  PetscTruth        realdiagonal; /* Flag to use the diagonal blocks of mat preconditioned by pmat, instead of just pmat */
  PetscInt          bs;           /* Block size for IS and Mat structures */
  PetscInt          nsplits;      /* Number of field divisions defined */
  Vec               *x,*y,w1,w2;
  Mat               *mat;         /* The diagonal block for each split */
  Mat               *pmat;        /* The preconditioning diagonal block for each split */
  Mat               *Afield;      /* The rows of the matrix associated with each split */
  PetscTruth        issetup;
  /* Only used when Schur complement preconditioning is used */
  Mat               B;            /* The (0,1) block */
  Mat               C;            /* The (1,0) block */
  Mat               schur;        /* The Schur complement S = D - C A^{-1} B */
  Mat               schur_user;   /* User-provided preconditioning matrix for the Schur complement */
  PCFieldSplitSchurPreType schurpre; /* Determines which preconditioning matrix is used for the Schur complement */
  KSP               kspschur;     /* The solver for S */
  PC_FieldSplitLink head;
} PC_FieldSplit;

/* 
    Notes: there is no particular reason that pmat, x, and y are stored as arrays in PC_FieldSplit instead of 
   inside PC_FieldSplitLink, just historical. If you want to be able to add new fields after already using the 
   PC you could change this.
*/

/* This helper is so that setting a user-provided preconditioning matrix is orthogonal to choosing to use it.  This way the
* application-provided FormJacobian can provide this matrix without interfering with the user's (command-line) choices. */
static Mat FieldSplitSchurPre(PC_FieldSplit *jac)
{
  switch (jac->schurpre) {
    case PC_FIELDSPLIT_SCHUR_PRE_SELF: return jac->schur;
    case PC_FIELDSPLIT_SCHUR_PRE_DIAG: return jac->pmat[1];
    case PC_FIELDSPLIT_SCHUR_PRE_USER: /* Use a user-provided matrix if it is given, otherwise diagonal block */
    default:
      return jac->schur_user ? jac->schur_user : jac->pmat[1];
  }
}


#undef __FUNCT__  
#define __FUNCT__ "PCView_FieldSplit"
static PetscErrorCode PCView_FieldSplit(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with %s composition: total splits = %D, blocksize = %D\n",PCCompositeTypes[jac->type],jac->nsplits,jac->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following KSP objects:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=0; i<jac->nsplits; i++) {
      if (ilink->fields) {
	ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",i);CHKERRQ(ierr);
	ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
	for (j=0; j<ilink->nfields; j++) {
	  if (j > 0) {
	    ierr = PetscViewerASCIIPrintf(viewer,",");CHKERRQ(ierr);
	  }
	  ierr = PetscViewerASCIIPrintf(viewer," %D",ilink->fields[j]);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      } else {
	ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Defined by IS\n",i);CHKERRQ(ierr);
      }
      ierr = KSPView(ilink->ksp,viewer);CHKERRQ(ierr);
      ilink = ilink->next;
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCFieldSplit",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_FieldSplit_Schur"
static PetscErrorCode PCView_FieldSplit_Schur(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscInt          i,j;
  PC_FieldSplitLink ilink = jac->head;
  KSP               ksp;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit with Schur preconditioner, blocksize = %D\n",jac->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Split info:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=0; i<jac->nsplits; i++) {
      if (ilink->fields) {
	ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",i);CHKERRQ(ierr);
	ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
	for (j=0; j<ilink->nfields; j++) {
	  if (j > 0) {
	    ierr = PetscViewerASCIIPrintf(viewer,",");CHKERRQ(ierr);
	  }
	  ierr = PetscViewerASCIIPrintf(viewer," %D",ilink->fields[j]);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      } else {
	ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Defined by IS\n",i);CHKERRQ(ierr);
      }
      ilink = ilink->next;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for A block \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (jac->schur) {
      ierr = MatSchurComplementGetKSP(jac->schur,&ksp);CHKERRQ(ierr);
      ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  not yet available\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"KSP solver for S = D - C inv(A) B \n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (jac->kspschur) {
      ierr = KSPView(jac->kspschur,viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  not yet available\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCFieldSplit",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetDefaults"
static PetscErrorCode PCFieldSplitSetDefaults(PC pc)
{
  PC_FieldSplit     *jac  = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          i = 0,*ifields,nfields;
  PetscTruth        flg = PETSC_FALSE,*fields,flg2;
  char              optionname[128];

  PetscFunctionBegin;
  if (!ilink) { 

    if (jac->bs <= 0) {
      if (pc->pmat) {
        ierr   = MatGetBlockSize(pc->pmat,&jac->bs);CHKERRQ(ierr);
      } else {
        jac->bs = 1;
      }
    }

    ierr = PetscOptionsGetTruth(((PetscObject)pc)->prefix,"-pc_fieldsplit_default",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (!flg) {
      /* Allow user to set fields from command line,  if bs was known at the time of PCSetFromOptions_FieldSplit()
         then it is set there. This is not ideal because we should only have options set in XXSetFromOptions(). */
      flg = PETSC_TRUE; /* switched off automatically if user sets fields manually here */
      ierr = PetscMalloc(jac->bs*sizeof(PetscInt),&ifields);CHKERRQ(ierr);
      while (PETSC_TRUE) {
        sprintf(optionname,"-pc_fieldsplit_%d_fields",(int)i++);
        nfields = jac->bs;
        ierr    = PetscOptionsGetIntArray(((PetscObject)pc)->prefix,optionname,ifields,&nfields,&flg2);CHKERRQ(ierr);
        if (!flg2) break;
        if (!nfields) SETERRQ(PETSC_ERR_USER,"Cannot list zero fields");
        flg = PETSC_FALSE;
        ierr = PCFieldSplitSetFields(pc,nfields,ifields);CHKERRQ(ierr);
      }
      ierr = PetscFree(ifields);CHKERRQ(ierr);
    }
    
    if (flg) {
      ierr = PetscInfo(pc,"Using default splitting of fields\n");CHKERRQ(ierr);
      ierr = PetscMalloc(jac->bs*sizeof(PetscTruth),&fields);CHKERRQ(ierr);
      ierr = PetscMemzero(fields,jac->bs*sizeof(PetscTruth));CHKERRQ(ierr);
      while (ilink) {
	for (i=0; i<ilink->nfields; i++) {
	  fields[ilink->fields[i]] = PETSC_TRUE;
	}
	ilink = ilink->next;
      }
      jac->defaultsplit = PETSC_TRUE;
      for (i=0; i<jac->bs; i++) {
	if (!fields[i]) {
	  ierr = PCFieldSplitSetFields(pc,1,&i);CHKERRQ(ierr);
	} else {
	  jac->defaultsplit = PETSC_FALSE;
	}
      }
      ierr = PetscFree(fields);CHKERRQ(ierr);
    }
  } else if (jac->nsplits == 1) {
    if (ilink->is) {
      IS       is2;
      PetscInt nmin,nmax;

      ierr = MatGetOwnershipRange(pc->mat,&nmin,&nmax);CHKERRQ(ierr);
      ierr = ISComplement(ilink->is,nmin,nmax,&is2);CHKERRQ(ierr);
      ierr = PCFieldSplitSetIS(pc,is2);CHKERRQ(ierr);
      ierr = ISDestroy(is2);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Must provide at least two sets of fields to PCFieldSplit()");
    }
  }
  if (jac->nsplits < 2) {
    SETERRQ(PETSC_ERR_PLIB,"Unhandled case, must have at least two fields");
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_FieldSplit"
static PetscErrorCode PCSetUp_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink;
  PetscInt          i,nsplit,ccsize;
  MatStructure      flag = pc->flag;
  PetscTruth        sorted;

  PetscFunctionBegin;
  ierr   = PCFieldSplitSetDefaults(pc);CHKERRQ(ierr);
  nsplit = jac->nsplits;
  ilink  = jac->head;

  /* get the matrices for each split */
  if (!jac->issetup) {
    PetscInt rstart,rend,nslots,bs;

    jac->issetup = PETSC_TRUE;

    /* This is done here instead of in PCFieldSplitSetFields() because may not have matrix at that point */
    bs     = jac->bs;
    ierr   = MatGetOwnershipRange(pc->pmat,&rstart,&rend);CHKERRQ(ierr);
    ierr   = MatGetLocalSize(pc->pmat,PETSC_NULL,&ccsize);CHKERRQ(ierr);
    nslots = (rend - rstart)/bs;
    for (i=0; i<nsplit; i++) {
      if (jac->defaultsplit) {
        ierr = ISCreateStride(((PetscObject)pc)->comm,nslots,rstart+i,nsplit,&ilink->is);CHKERRQ(ierr);
      } else if (!ilink->is) {
        if (ilink->nfields > 1) {
          PetscInt   *ii,j,k,nfields = ilink->nfields,*fields = ilink->fields;
          ierr = PetscMalloc(ilink->nfields*nslots*sizeof(PetscInt),&ii);CHKERRQ(ierr);
          for (j=0; j<nslots; j++) {
            for (k=0; k<nfields; k++) {
              ii[nfields*j + k] = rstart + bs*j + fields[k];
            }
          }
          ierr = ISCreateGeneral(((PetscObject)pc)->comm,nslots*nfields,ii,&ilink->is);CHKERRQ(ierr);       
          ierr = PetscFree(ii);CHKERRQ(ierr);
        } else { 
          ierr = ISCreateStride(((PetscObject)pc)->comm,nslots,rstart+ilink->fields[0],bs,&ilink->is);CHKERRQ(ierr);
        }
      } 
      ierr = ISSorted(ilink->is,&sorted);CHKERRQ(ierr);
      if (!sorted) SETERRQ(PETSC_ERR_USER,"Fields must be sorted when creating split");
      ilink = ilink->next;
    }
  }
  
  ilink  = jac->head;
  if (!jac->pmat) {
    ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->pmat);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      ierr = MatGetSubMatrix(pc->pmat,ilink->is,ilink->is,MAT_INITIAL_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
      ilink = ilink->next;
    }
  } else {
    for (i=0; i<nsplit; i++) {
      ierr = MatGetSubMatrix(pc->pmat,ilink->is,ilink->is,MAT_REUSE_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
      ilink = ilink->next;
    }
  }
  if (jac->realdiagonal) {
    ilink = jac->head;
    if (!jac->mat) {
      ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->mat);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,ilink->is,MAT_INITIAL_MATRIX,&jac->mat[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,ilink->is,MAT_REUSE_MATRIX,&jac->mat[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  } else {
    jac->mat = jac->pmat;
  }

  if (jac->type != PC_COMPOSITE_ADDITIVE  && jac->type != PC_COMPOSITE_SCHUR) {
    /* extract the rows of the matrix associated with each field: used for efficient computation of residual inside algorithm */
    ilink  = jac->head;
    if (!jac->Afield) {
      ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->Afield);CHKERRQ(ierr);
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,PETSC_NULL,MAT_INITIAL_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    } else {
      for (i=0; i<nsplit; i++) {
        ierr = MatGetSubMatrix(pc->mat,ilink->is,PETSC_NULL,MAT_REUSE_MATRIX,&jac->Afield[i]);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  }

  if (jac->type == PC_COMPOSITE_SCHUR) {
    IS       ccis;
    PetscInt rstart,rend;
    if (nsplit != 2) SETERRQ(PETSC_ERR_ARG_INCOMP,"To use Schur complement preconditioner you must have exactly 2 fields");

    /* When extracting off-diagonal submatrices, we take complements from this range */
    ierr  = MatGetOwnershipRangeColumn(pc->mat,&rstart,&rend);CHKERRQ(ierr);

    /* need to handle case when one is resetting up the preconditioner */
    if (jac->schur) {
      ilink = jac->head;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->B);CHKERRQ(ierr);
      ierr  = ISDestroy(ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_REUSE_MATRIX,&jac->C);CHKERRQ(ierr);
      ierr  = ISDestroy(ccis);CHKERRQ(ierr);
      ierr  = MatSchurComplementUpdate(jac->schur,jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->pmat[1],pc->flag);CHKERRQ(ierr);
      ierr  = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),pc->flag);CHKERRQ(ierr);

     } else {
      KSP ksp;

      /* extract the B and C matrices */
      ilink = jac->head;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->B);CHKERRQ(ierr);
      ierr  = ISDestroy(ccis);CHKERRQ(ierr);
      ilink = ilink->next;
      ierr  = ISComplement(ilink->is,rstart,rend,&ccis);CHKERRQ(ierr);
      ierr  = MatGetSubMatrix(pc->mat,ilink->is,ccis,MAT_INITIAL_MATRIX,&jac->C);CHKERRQ(ierr);
      ierr  = ISDestroy(ccis);CHKERRQ(ierr);
      /* Better would be to use 'mat[0]' (diagonal block of the real matrix) preconditioned by pmat[0] */
      ierr  = MatCreateSchurComplement(jac->mat[0],jac->pmat[0],jac->B,jac->C,jac->mat[1],&jac->schur);CHKERRQ(ierr);
      ierr  = MatSchurComplementGetKSP(jac->schur,&ksp);CHKERRQ(ierr);
      ierr  = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,2);CHKERRQ(ierr);
      ierr  = MatSetFromOptions(jac->schur);CHKERRQ(ierr);

      ierr  = KSPCreate(((PetscObject)pc)->comm,&jac->kspschur);CHKERRQ(ierr);
      ierr  = PetscObjectIncrementTabLevel((PetscObject)jac->kspschur,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr  = KSPSetOperators(jac->kspschur,jac->schur,FieldSplitSchurPre(jac),DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      if (jac->schurpre == PC_FIELDSPLIT_SCHUR_PRE_SELF) {
        PC pc;
        ierr = KSPGetPC(jac->kspschur,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
        /* Note: This is bad if there exist preconditioners for MATSCHURCOMPLEMENT */
      }
      ierr  = KSPSetOptionsPrefix(jac->kspschur,((PetscObject)pc)->prefix);CHKERRQ(ierr);
      ierr  = KSPAppendOptionsPrefix(jac->kspschur,"fieldsplit_1_");CHKERRQ(ierr);
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      ierr = KSPSetFromOptions(jac->kspschur);CHKERRQ(ierr);

      ierr = PetscMalloc2(2,Vec,&jac->x,2,Vec,&jac->y);CHKERRQ(ierr);
      ierr = MatGetVecs(jac->pmat[0],&jac->x[0],&jac->y[0]);CHKERRQ(ierr);
      ierr = MatGetVecs(jac->pmat[1],&jac->x[1],&jac->y[1]);CHKERRQ(ierr);
      ilink = jac->head;
      ilink->x = jac->x[0]; ilink->y = jac->y[0];
      ilink = ilink->next;
      ilink->x = jac->x[1]; ilink->y = jac->y[1];
    } 
  } else {
    /* set up the individual PCs */
    i    = 0;
    ilink = jac->head;
    while (ilink) {
      ierr = KSPSetOperators(ilink->ksp,jac->mat[i],jac->pmat[i],flag);CHKERRQ(ierr);
      /* really want setfromoptions called in PCSetFromOptions_FieldSplit(), but it is not ready yet */
      ierr = KSPSetFromOptions(ilink->ksp);CHKERRQ(ierr);
      ierr = KSPSetUp(ilink->ksp);CHKERRQ(ierr);
      i++;
      ilink = ilink->next;
    }
  
    /* create work vectors for each split */
    if (!jac->x) {
      ierr = PetscMalloc2(nsplit,Vec,&jac->x,nsplit,Vec,&jac->y);CHKERRQ(ierr);
      ilink = jac->head;
      for (i=0; i<nsplit; i++) {
        Vec *vl,*vr;

        ierr      = KSPGetVecs(ilink->ksp,1,&vr,1,&vl);CHKERRQ(ierr);
        ilink->x  = *vr;
        ilink->y  = *vl;
        ierr      = PetscFree(vr);CHKERRQ(ierr);
        ierr      = PetscFree(vl);CHKERRQ(ierr);
        jac->x[i] = ilink->x;
        jac->y[i] = ilink->y;
        ilink     = ilink->next;
      }
    }
  }


  if (!jac->head->sctx) {
    Vec xtmp;

    /* compute scatter contexts needed by multiplicative versions and non-default splits */
    
    ilink = jac->head;
    ierr = MatGetVecs(pc->pmat,&xtmp,PETSC_NULL);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      ierr = VecScatterCreate(xtmp,ilink->is,jac->x[i],PETSC_NULL,&ilink->sctx);CHKERRQ(ierr);
      ilink = ilink->next;
    }
    ierr = VecDestroy(xtmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define FieldSplitSplitSolveAdd(ilink,xx,yy) \
    (VecScatterBegin(ilink->sctx,xx,ilink->x,INSERT_VALUES,SCATTER_FORWARD) || \
     VecScatterEnd(ilink->sctx,xx,ilink->x,INSERT_VALUES,SCATTER_FORWARD) || \
     KSPSolve(ilink->ksp,ilink->x,ilink->y) || \
     VecScatterBegin(ilink->sctx,ilink->y,yy,ADD_VALUES,SCATTER_REVERSE) || \
     VecScatterEnd(ilink->sctx,ilink->y,yy,ADD_VALUES,SCATTER_REVERSE))

#undef __FUNCT__  
#define __FUNCT__ "PCApply_FieldSplit_Schur"
static PetscErrorCode PCApply_FieldSplit_Schur(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  KSP               ksp;
  PC_FieldSplitLink ilinkA = jac->head, ilinkD = ilinkA->next;

  PetscFunctionBegin;
  ierr = MatSchurComplementGetKSP(jac->schur,&ksp);CHKERRQ(ierr);

  ierr = VecScatterBegin(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkA->sctx,x,ilinkA->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
  ierr = MatMult(jac->C,ilinkA->y,ilinkD->x);CHKERRQ(ierr);
  ierr = VecScale(ilinkD->x,-1.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkD->sctx,x,ilinkD->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = KSPSolve(jac->kspschur,ilinkD->x,ilinkD->y);CHKERRQ(ierr);  
  ierr = VecScatterBegin(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkD->sctx,ilinkD->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = MatMult(jac->B,ilinkD->y,ilinkA->y);CHKERRQ(ierr);
  ierr = VecAXPY(ilinkA->x,-1.0,ilinkA->y);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,ilinkA->x,ilinkA->y);CHKERRQ(ierr);
  ierr = VecScatterBegin(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ilinkA->sctx,ilinkA->y,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_FieldSplit"
static PetscErrorCode PCApply_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          cnt;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecSetBlockSize(x,jac->bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(y,jac->bs);CHKERRQ(ierr);

  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
      while (ilink) {
        ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
        ilink = ilink->next;
      }
      ierr = VecStrideScatterAll(jac->y,y,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSet(y,0.0);CHKERRQ(ierr);
      while (ilink) {
        ierr = FieldSplitSplitSolveAdd(ilink,x,y);CHKERRQ(ierr);
        ilink = ilink->next;
      }
    }
  } else if (jac->type == PC_COMPOSITE_MULTIPLICATIVE || jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    if (!jac->w1) {
      ierr = VecDuplicate(x,&jac->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&jac->w2);CHKERRQ(ierr);
    }
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = FieldSplitSplitSolveAdd(ilink,x,y);CHKERRQ(ierr);
    cnt = 1;
    while (ilink->next) {
      ilink = ilink->next;
      /* compute the residual only over the part of the vector needed */
      ierr = MatMult(jac->Afield[cnt++],y,ilink->x);CHKERRQ(ierr);
      ierr = VecScale(ilink->x,-1.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
      ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      cnt -= 2;
      while (ilink->previous) {
        ilink = ilink->previous;
        /* compute the residual only over the part of the vector needed */
        ierr = MatMult(jac->Afield[cnt--],y,ilink->x);CHKERRQ(ierr);
        ierr = VecScale(ilink->x,-1.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(ilink->sctx,x,ilink->x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = KSPSolve(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
        ierr = VecScatterBegin(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(ilink->sctx,ilink->y,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      }
    }
  } else SETERRQ1(PETSC_ERR_SUP,"Unsupported or unknown composition",(int) jac->type);
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#define FieldSplitSplitSolveAddTranspose(ilink,xx,yy) \
    (VecScatterBegin(ilink->sctx,xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD) || \
     VecScatterEnd(ilink->sctx,xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD) || \
     KSPSolveTranspose(ilink->ksp,ilink->y,ilink->x) || \
     VecScatterBegin(ilink->sctx,ilink->x,yy,ADD_VALUES,SCATTER_REVERSE) || \
     VecScatterEnd(ilink->sctx,ilink->x,yy,ADD_VALUES,SCATTER_REVERSE))

#undef __FUNCT__  
#define __FUNCT__ "PCApply_FieldSplit"
static PetscErrorCode PCApplyTranspose_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecSetBlockSize(x,jac->bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(y,jac->bs);CHKERRQ(ierr);

  if (jac->type == PC_COMPOSITE_ADDITIVE) {
    if (jac->defaultsplit) {
      ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
      while (ilink) {
	ierr = KSPSolveTranspose(ilink->ksp,ilink->x,ilink->y);CHKERRQ(ierr);
	ilink = ilink->next;
      }
      ierr = VecStrideScatterAll(jac->y,y,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSet(y,0.0);CHKERRQ(ierr);
      while (ilink) {
        ierr = FieldSplitSplitSolveAddTranspose(ilink,x,y);CHKERRQ(ierr);
	ilink = ilink->next;
      }
    }
  } else {
    if (!jac->w1) {
      ierr = VecDuplicate(x,&jac->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&jac->w2);CHKERRQ(ierr);
    }
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      ierr = FieldSplitSplitSolveAddTranspose(ilink,x,y);CHKERRQ(ierr);
      while (ilink->next) {
        ilink = ilink->next;
        ierr  = MatMultTranspose(pc->mat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
      while (ilink->previous) {
        ilink = ilink->previous;
        ierr  = MatMultTranspose(pc->mat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
    } else {
      while (ilink->next) {   /* get to last entry in linked list */
	ilink = ilink->next;
      }
      ierr = FieldSplitSplitSolveAddTranspose(ilink,x,y);CHKERRQ(ierr);
      while (ilink->previous) {
	ilink = ilink->previous;
	ierr  = MatMultTranspose(pc->mat,y,jac->w1);CHKERRQ(ierr);
	ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
	ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
    }
  }
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_FieldSplit"
static PetscErrorCode PCDestroy_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head,next;

  PetscFunctionBegin;
  while (ilink) {
    ierr = KSPDestroy(ilink->ksp);CHKERRQ(ierr);
    if (ilink->x) {ierr = VecDestroy(ilink->x);CHKERRQ(ierr);}
    if (ilink->y) {ierr = VecDestroy(ilink->y);CHKERRQ(ierr);}
    if (ilink->sctx) {ierr = VecScatterDestroy(ilink->sctx);CHKERRQ(ierr);}
    if (ilink->is) {ierr = ISDestroy(ilink->is);CHKERRQ(ierr);}
    next = ilink->next;
    ierr = PetscFree(ilink->fields);CHKERRQ(ierr);
    ierr = PetscFree(ilink);CHKERRQ(ierr);
    ilink = next;
  }
  ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);
  if (jac->mat && jac->mat != jac->pmat) {ierr = MatDestroyMatrices(jac->nsplits,&jac->mat);CHKERRQ(ierr);}
  if (jac->pmat) {ierr = MatDestroyMatrices(jac->nsplits,&jac->pmat);CHKERRQ(ierr);}
  if (jac->Afield) {ierr = MatDestroyMatrices(jac->nsplits,&jac->Afield);CHKERRQ(ierr);}
  if (jac->w1) {ierr = VecDestroy(jac->w1);CHKERRQ(ierr);}
  if (jac->w2) {ierr = VecDestroy(jac->w2);CHKERRQ(ierr);}
  if (jac->schur) {ierr = MatDestroy(jac->schur);CHKERRQ(ierr);}
  if (jac->schur_user) {ierr = MatDestroy(jac->schur_user);CHKERRQ(ierr);}
  if (jac->kspschur) {ierr = KSPDestroy(jac->kspschur);CHKERRQ(ierr);}
  if (jac->B) {ierr = MatDestroy(jac->B);CHKERRQ(ierr);}
  if (jac->C) {ierr = MatDestroy(jac->C);CHKERRQ(ierr);}
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_FieldSplit"
static PetscErrorCode PCSetFromOptions_FieldSplit(PC pc)
{
  PetscErrorCode  ierr;
  PetscInt        i = 0,nfields,*fields,bs;
  PetscTruth      flg;
  char            optionname[128];
  PC_FieldSplit   *jac = (PC_FieldSplit*)pc->data;
  PCCompositeType ctype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("FieldSplit options");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-pc_fieldsplit_real_diagonal","Use diagonal blocks of the operator","PCFieldSplitSetRealDiagonal",jac->realdiagonal,&jac->realdiagonal,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_fieldsplit_block_size","Blocksize that defines number of fields","PCFieldSplitSetBlockSize",jac->bs,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFieldSplitSetBlockSize(pc,bs);CHKERRQ(ierr);
  }

  ierr = PetscOptionsEnum("-pc_fieldsplit_type","Type of composition","PCFieldSplitSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&ctype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFieldSplitSetType(pc,ctype);CHKERRQ(ierr);
  }

  /* Only setup fields once */
  if ((jac->bs > 0) && (jac->nsplits == 0)) {
    /* only allow user to set fields from command line if bs is already known.
       otherwise user can set them in PCFieldSplitSetDefaults() */
    ierr = PetscMalloc(jac->bs*sizeof(PetscInt),&fields);CHKERRQ(ierr);
    while (PETSC_TRUE) {
      sprintf(optionname,"-pc_fieldsplit_%d_fields",(int)i++);
      nfields = jac->bs;
      ierr    = PetscOptionsIntArray(optionname,"Fields in this split","PCFieldSplitSetFields",fields,&nfields,&flg);CHKERRQ(ierr);
      if (!flg) break;
      if (!nfields) SETERRQ(PETSC_ERR_USER,"Cannot list zero fields");
      ierr = PCFieldSplitSetFields(pc,nfields,fields);CHKERRQ(ierr);
    }
    ierr = PetscFree(fields);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnum("-pc_fieldsplit_schur_precondition","How to build preconditioner for Schur complement","PCFieldSplitSchurPrecondition",PCFieldSplitSchurPreTypes,(PetscEnum)jac->schurpre,(PetscEnum*)&jac->schurpre,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetFields_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetFields_FieldSplit(PC pc,PetscInt n,PetscInt *fields)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink,next = jac->head;
  char              prefix[128];
  PetscInt          i;

  PetscFunctionBegin;
  if (n <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative number of fields requested");
  for (i=0; i<n; i++) {
    if (fields[i] >= jac->bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Field %D requested but only %D exist",fields[i],jac->bs);
    if (fields[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative field %D requested",fields[i]);
  }
  ierr = PetscNew(struct _PC_FieldSplitLink,&ilink);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&ilink->fields);CHKERRQ(ierr);
  ierr = PetscMemcpy(ilink->fields,fields,n*sizeof(PetscInt));CHKERRQ(ierr);
  ilink->nfields = n;
  ilink->next    = PETSC_NULL;
  ierr           = KSPCreate(((PetscObject)pc)->comm,&ilink->ksp);CHKERRQ(ierr);
  ierr           = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr           = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);

  if (((PetscObject)pc)->prefix) {
    sprintf(prefix,"%sfieldsplit_%d_",((PetscObject)pc)->prefix,(int)jac->nsplits);
  } else {
    sprintf(prefix,"fieldsplit_%d_",(int)jac->nsplits);
  }
  ierr = KSPSetOptionsPrefix(ilink->ksp,prefix);CHKERRQ(ierr);

  if (!next) {
    jac->head       = ilink;
    ilink->previous = PETSC_NULL;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  jac->nsplits++;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitGetSubKSP_FieldSplit_Schur"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitGetSubKSP_FieldSplit_Schur(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->nsplits*sizeof(KSP),subksp);CHKERRQ(ierr);
  ierr = MatSchurComplementGetKSP(jac->schur,*subksp);CHKERRQ(ierr);
  (*subksp)[1] = jac->kspschur;
  *n = jac->nsplits;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitGetSubKSP_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitGetSubKSP_FieldSplit(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          cnt = 0;
  PC_FieldSplitLink ilink = jac->head;

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->nsplits*sizeof(KSP*),subksp);CHKERRQ(ierr);
  while (ilink) {
    (*subksp)[cnt++] = ilink->ksp;
    ilink = ilink->next;
  }
  if (cnt != jac->nsplits) SETERRQ2(PETSC_ERR_PLIB,"Corrupt PCFIELDSPLIT object: number splits in linked list %D in object %D",cnt,jac->nsplits);
  *n = jac->nsplits;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetIS_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetIS_FieldSplit(PC pc,IS is)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink, next = jac->head;
  char              prefix[128];

  PetscFunctionBegin;
  ierr = PetscNew(struct _PC_FieldSplitLink,&ilink);CHKERRQ(ierr);
  ilink->is      = is;
  ierr           = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  ilink->next    = PETSC_NULL;
  ierr           = KSPCreate(((PetscObject)pc)->comm,&ilink->ksp);CHKERRQ(ierr);
  ierr           = PetscObjectIncrementTabLevel((PetscObject)ilink->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr           = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);

  if (((PetscObject)pc)->prefix) {
    sprintf(prefix,"%sfieldsplit_%d_",((PetscObject)pc)->prefix,(int)jac->nsplits);
  } else {
    sprintf(prefix,"fieldsplit_%d_",(int)jac->nsplits);
  }
  ierr = KSPSetOptionsPrefix(ilink->ksp,prefix);CHKERRQ(ierr);

  if (!next) {
    jac->head       = ilink;
    ilink->previous = PETSC_NULL;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  jac->nsplits++;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetFields"
/*@
    PCFieldSplitSetFields - Sets the fields for one particular split in the field split preconditioner

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   n - the number of fields in this split
.   fields - the fields in this split

    Level: intermediate

    Notes: Use PCFieldSplitSetIS() to set a completely general set of indices as a field. 

     The PCFieldSplitSetFields() is for defining fields as a strided blocks. For example, if the block
     size is three then one can define a field as 0, or 1 or 2 or 0,1 or 0,2 or 1,2 which mean
     0xx3xx6xx9xx12 ... x1xx4xx7xx ... xx2xx5xx8xx.. 01x34x67x... 0x1x3x5x7.. x12x45x78x....
     where the numbered entries indicate what is in the field. 

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetBlockSize(), PCFieldSplitSetIS()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetFields(PC pc,PetscInt n, PetscInt *fields)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,PetscInt *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitSetFields_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,fields);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetIS"
/*@
    PCFieldSplitSetIS - Sets the exact elements for field

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   is - the index set that defines the vector elements in this field


    Notes:
    Use PCFieldSplitSetFields(), for fields defined by strided types.

    This function is called once per split (it creates a new split each time).

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetBlockSize()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetIS(PC pc,IS is)
{
  PetscErrorCode ierr,(*f)(PC,IS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitSetIS_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,is);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetBlockSize"
/*@
    PCFieldSplitSetBlockSize - Sets the block size for defining where fields start in the 
      fieldsplit preconditioner. If not set the matrix block size is used.

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   bs - the block size

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetBlockSize(PC pc,PetscInt bs)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitSetBlockSize_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,bs);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitGetSubKSP"
/*@C
   PCFieldSplitGetSubKSP - Gets the KSP contexts for all splits
   
   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n - the number of split
-  pc - the array of KSP contexts

   Note:  
   After PCFieldSplitGetSubKSP() the array of KSPs IS to be freed by the user
   (not the KSP just the array that contains them).

   You must call KSPSetUp() before calling PCFieldSplitGetSubKSP().

   Level: advanced

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitGetSubKSP(PC pc,PetscInt *n,KSP *subksp[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*,KSP **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(n,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,subksp);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get subksp for this type of PC");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSchurPrecondition"
/*@
    PCFieldSplitSchurPrecondition -  Indicates if the Schur complement is preconditioned by a preconditioner constructed by the
      D matrix. Otherwise no preconditioner is used.

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   ptype - which matrix to use for preconditioning the Schur complement
-   userpre - matrix to use for preconditioning, or PETSC_NULL

    Notes:
    The default is to use the block on the diagonal of the preconditioning matrix.  This is D, in the (1,1) position.
    There are currently no preconditioners that work directly with the Schur complement so setting
    PC_FIELDSPLIT_SCHUR_PRE_SELF is observationally equivalent to -fieldsplit_1_pc_type none.

    Options Database:
.     -pc_fieldsplit_schur_precondition <self,user,diag> default is diag

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetFields(), PCFieldSplitSchurPreType

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSchurPrecondition(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PetscErrorCode ierr,(*f)(PC,PCFieldSplitSchurPreType,Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitSchurPrecondition_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ptype,pre);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSchurPrecondition_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSchurPrecondition_FieldSplit(PC pc,PCFieldSplitSchurPreType ptype,Mat pre)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  jac->schurpre = ptype;
  if (pre) {
    if (jac->schur_user) {ierr = MatDestroy(jac->schur_user);CHKERRQ(ierr);}
    jac->schur_user = pre;
    ierr = PetscObjectReference((PetscObject)jac->schur_user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitGetSchurBlocks"
/*@C
   PCFieldSplitGetSchurBlocks - Gets the all matrix blocks for the Schur complement
   
   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  A - the (0,0) block
.  B - the (0,1) block
.  C - the (1,0) block
-  D - the (1,1) block

   Level: advanced

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitGetSchurBlocks(PC pc,Mat *A,Mat *B,Mat *C, Mat *D)
{
  PC_FieldSplit *jac = (PC_FieldSplit *) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (jac->type != PC_COMPOSITE_SCHUR) {SETERRQ(PETSC_ERR_ARG_WRONG, "FieldSplit is not using a Schur complement approach.");}
  if (A) *A = jac->pmat[0];
  if (B) *B = jac->B;
  if (C) *C = jac->C;
  if (D) *D = jac->pmat[1];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetType_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetType_FieldSplit(PC pc,PCCompositeType type)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac->type = type;
  if (type == PC_COMPOSITE_SCHUR) {
    pc->ops->apply = PCApply_FieldSplit_Schur;
    pc->ops->view  = PCView_FieldSplit_Schur;
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit_Schur",PCFieldSplitGetSubKSP_FieldSplit_Schur);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSchurPrecondition_C","PCFieldSplitSchurPrecondition_FieldSplit",PCFieldSplitSchurPrecondition_FieldSplit);CHKERRQ(ierr);

  } else {
    pc->ops->apply = PCApply_FieldSplit;
    pc->ops->view  = PCView_FieldSplit;
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit",PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSchurPrecondition_C","",0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetBlockSize_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetBlockSize_FieldSplit(PC pc,PetscInt bs)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Blocksize must be positive, you gave %D",bs);
  if (jac->bs > 0 && jac->bs != bs) SETERRQ2(PETSC_ERR_ARG_WRONGSTATE,"Cannot change fieldsplit blocksize from %D to %D after it has been set",jac->bs,bs);
  jac->bs = bs;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetType"
/*@
   PCFieldSplitSetType - Sets the type of fieldsplit preconditioner.
   
   Collective on PC

   Input Parameter:
.  pc - the preconditioner context
.  type - PC_COMPOSITE_ADDITIVE, PC_COMPOSITE_MULTIPLICATIVE (default), PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL, PC_COMPOSITE_SCHUR

   Options Database Key:
.  -pc_fieldsplit_type <type: one of multiplicative, additive, symmetric_multiplicative, special, schur> - Sets fieldsplit preconditioner type

   Level: Developer

.keywords: PC, set, type, composite preconditioner, additive, multiplicative

.seealso: PCCompositeSetType()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetType(PC pc,PCCompositeType type)
{
  PetscErrorCode ierr,(*f)(PC,PCCompositeType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCFIELDSPLIT - Preconditioner created by combining separate preconditioners for individual
                  fields or groups of fields


     To set options on the solvers for each block append -fieldsplit_ to all the PC
        options database keys. For example, -fieldsplit_pc_type ilu -fieldsplit_pc_factor_levels 1
        
     To set the options on the solvers separate for each block call PCFieldSplitGetSubKSP()
         and set the options directly on the resulting KSP object

   Level: intermediate

   Options Database Keys:
+   -pc_fieldsplit_%d_fields <a,b,..> - indicates the fields to be used in the %d'th split
.   -pc_fieldsplit_default - automatically add any fields to additional splits that have not
                              been supplied explicitly by -pc_fieldsplit_%d_fields
.   -pc_fieldsplit_block_size <bs> - size of block that defines fields (i.e. there are bs fields)
.    -pc_fieldsplit_type <additive,multiplicative,schur,symmetric_multiplicative>
.    -pc_fieldsplit_schur_precondition <true,false> default is true

-    Options prefix for inner solvers when using Schur complement preconditioner are -fieldsplit_0_ and -fieldsplit_1_
     for all other solvers they are -fieldsplit_%d_ for the dth field, use -fieldsplit_ for all fields


   Notes: use PCFieldSplitSetFields() to set fields defined by "strided" entries and PCFieldSplitSetIS()
     to define a field by an arbitrary collection of entries.

      If no fields are set the default is used. The fields are defined by entries strided by bs,
      beginning at 0 then 1, etc to bs-1. The block size can be set with PCFieldSplitSetBlockSize(),
      if this is not called the block size defaults to the blocksize of the second matrix passed
      to KSPSetOperators()/PCSetOperators().

      Currently for the multiplicative version, the updated residual needed for the next field
     solve is computed via a matrix vector product over the entire array. An optimization would be
     to update the residual only for the part of the right hand side associated with the next field
     solve. (This would involve more MatGetSubMatrix() calls or some other mechanism to compute the 
     part of the matrix needed to just update part of the residual).

     For the Schur complement preconditioner if J = ( A B )
                                                    ( C D )
     the preconditioner is 
              (I   -B inv(A)) ( inv(A)   0    ) (I         0  )
              (0    I       ) (   0    inv(S) ) (-C inv(A) I  )
     where the action of inv(A) is applied using the KSP solver with prefix -fieldsplit_0_. The action of 
     inv(S) is computed using the KSP solver with prefix -schur_. For PCFieldSplitGetKSP() when field number is
     0 it returns the KSP associated with -fieldsplit_0_ while field number 1 gives -fieldsplit_1_ KSP. By default
     D is used to construct a preconditioner for S, use PCFieldSplitSchurPrecondition() to turn on or off this
     option.
     
     If only one set of indices (one IS) is provided with PCFieldSplitSetIS() then the complement of that IS
     is used automatically for a second block.

   Concepts: physics based preconditioners, block preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, Block_Preconditioners
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields(), PCFieldSplitSetType(), PCFieldSplitSetIS(), PCFieldSplitSchurPrecondition()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_FieldSplit(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_FieldSplit,&jac);CHKERRQ(ierr);
  jac->bs        = -1;
  jac->nsplits   = 0;
  jac->type      = PC_COMPOSITE_MULTIPLICATIVE;
  jac->schurpre  = PC_FIELDSPLIT_SCHUR_PRE_USER; /* Try user preconditioner first, fall back on diagonal */

  pc->data     = (void*)jac;

  pc->ops->apply             = PCApply_FieldSplit;
  pc->ops->applytranspose    = PCApplyTranspose_FieldSplit;
  pc->ops->setup             = PCSetUp_FieldSplit;
  pc->ops->destroy           = PCDestroy_FieldSplit;
  pc->ops->setfromoptions    = PCSetFromOptions_FieldSplit;
  pc->ops->view              = PCView_FieldSplit;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit",
                    PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetFields_C","PCFieldSplitSetFields_FieldSplit",
                    PCFieldSplitSetFields_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetIS_C","PCFieldSplitSetIS_FieldSplit",
                    PCFieldSplitSetIS_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetType_C","PCFieldSplitSetType_FieldSplit",
                    PCFieldSplitSetType_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetBlockSize_C","PCFieldSplitSetBlockSize_FieldSplit",
                    PCFieldSplitSetBlockSize_FieldSplit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


/*MC
   Block_Preconditioners - PETSc provides support for a variety of "block preconditioners", this provides an
          overview of these methods. 

      Consider the solution to ( A B ) (x_1)  =  (b_1)
                               ( C D ) (x_2)     (b_2)

      Important special cases, the Stokes equation: C = B' and D = 0  (A   B) (x_1) = (b_1)
                                                                       B'  0) (x_2)   (b_2) 

      One of the goals of the PCFieldSplit preconditioner in PETSc is to provide a variety of preconditioners
      for this block system.
   
      Consider an additional matrix (Ap  Bp)
                                    (Cp  Dp) where some or all of the entries may be the same as
      in the original matrix (for example Ap == A).

      In the following, A^ denotes the approximate application of the inverse of A, possibly using Ap in the 
      approximation. In PETSc this simply means one has called KSPSetOperators(ksp,A,Ap,...) or KSPSetOperators(ksp,Ap,Ap,...)

      Block Jacobi:   x_1 = A^ b_1
                      x_2 = D^ b_2

      Lower block Gauss-Seidel:   x_1 = A^ b_1
                            x_2 = D^ (b_2 - C x_1)       variant x_2 = D^ (b_2 - Cp x_1)

      Symmetric Gauss-Seidel:  x_1 = x_1 + A^(b_1 - A x_1 - B x_2)    variant  x_1 = x_1 + A^(b_1 - Ap x_1 - Bp x_2)
          Interestingly this form is not actually a symmetric matrix, the symmetric version is 
                              x_1 = A^(b_1 - B x_2)      variant x_1 = A^(b_1 - Bp x_2)

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCFIELDSPLIT
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields(), PCFieldSplitSetType(), PCFieldSplitSetIS(), PCFieldSplitSchurPrecondition()
M*/
