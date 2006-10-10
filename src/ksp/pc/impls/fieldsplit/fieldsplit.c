#define PETSCKSP_DLL

/*

*/
#include "private/pcimpl.h"     /*I "petscpc.h" I*/

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  KSP               ksp;
  Vec               x,y;
  PetscInt          nfields;
  PetscInt          *fields;
  VecScatter        sctx;
  PC_FieldSplitLink next,previous;
};

typedef struct {
  PCCompositeType   type;              /* additive or multiplicative */
  PetscTruth        defaultsplit;
  PetscInt          bs;
  PetscInt          nsplits,*csize;
  Vec               *x,*y,w1,w2;
  Mat               *pmat;
  IS                *is,*cis;
  PC_FieldSplitLink head;
} PC_FieldSplit;

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
#define __FUNCT__ "PCFieldSplitSetDefaults"
static PetscErrorCode PCFieldSplitSetDefaults(PC pc)
{
  PC_FieldSplit     *jac  = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          i;
  PetscTruth        flg = PETSC_FALSE,*fields;

  PetscFunctionBegin;
  ierr = PetscOptionsGetTruth(pc->prefix,"-pc_fieldsplit_default",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (!ilink || flg) { 
    ierr = PetscInfo(pc,"Using default splitting of fields\n");CHKERRQ(ierr);
    if (jac->bs <= 0) {
      ierr   = MatGetBlockSize(pc->pmat,&jac->bs);CHKERRQ(ierr);
    }
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
  if (!jac->is) {
    PetscInt rstart,rend,nslots,bs;

    bs     = jac->bs;
    ierr   = MatGetOwnershipRange(pc->pmat,&rstart,&rend);CHKERRQ(ierr);
    ierr   = MatGetLocalSize(pc->pmat,PETSC_NULL,&ccsize);CHKERRQ(ierr);
    nslots = (rend - rstart)/bs;
    ierr   = PetscMalloc(nsplit*sizeof(IS),&jac->is);CHKERRQ(ierr);
    ierr   = PetscMalloc(nsplit*sizeof(IS),&jac->cis);CHKERRQ(ierr);
    ierr   = PetscMalloc(nsplit*sizeof(PetscInt),&jac->csize);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      if (jac->defaultsplit) {
	ierr     = ISCreateStride(pc->comm,nslots,rstart+i,nsplit,&jac->is[i]);CHKERRQ(ierr);
        jac->csize[i] = ccsize/nsplit;
      } else {
        if (ilink->nfields > 1) {
	  PetscInt   *ii,j,k,nfields = ilink->nfields,*fields = ilink->fields;
	  ierr = PetscMalloc(ilink->nfields*nslots*sizeof(PetscInt),&ii);CHKERRQ(ierr);
	  for (j=0; j<nslots; j++) {
	    for (k=0; k<nfields; k++) {
	      ii[nfields*j + k] = rstart + bs*j + fields[k];
	    }
	  }
	  ierr = ISCreateGeneral(pc->comm,nslots*nfields,ii,&jac->is[i]);CHKERRQ(ierr);       
	  ierr = PetscFree(ii);CHKERRQ(ierr);
        } else { 
          ierr = ISCreateStride(pc->comm,nslots,ilink->fields[0],bs,&jac->is[i]);CHKERRQ(ierr);
        }
        jac->csize[i] = (ccsize/bs)*ilink->nfields;
        ierr = ISSorted(jac->is[i],&sorted);CHKERRQ(ierr);
        if (!sorted) SETERRQ(PETSC_ERR_USER,"Fields must be sorted when creating split");
        ilink = ilink->next;
      }
      ierr = ISAllGather(jac->is[i],&jac->cis[i]);CHKERRQ(ierr);
    }
  }
  
  if (!jac->pmat) {
    ierr = PetscMalloc(nsplit*sizeof(Mat),&jac->pmat);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      ierr = MatGetSubMatrix(pc->pmat,jac->is[i],jac->cis[i],jac->csize[i],MAT_INITIAL_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
    }
  } else {
    for (i=0; i<nsplit; i++) {
      ierr = MatGetSubMatrix(pc->pmat,jac->is[i],jac->cis[i],jac->csize[i],MAT_REUSE_MATRIX,&jac->pmat[i]);CHKERRQ(ierr);
    }
  }

  /* set up the individual PCs */
  i    = 0;
  ilink = jac->head;
  while (ilink) {
    ierr = KSPSetOperators(ilink->ksp,jac->pmat[i],jac->pmat[i],flag);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ilink->ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ilink->ksp);CHKERRQ(ierr);
    i++;
    ilink = ilink->next;
  }
  
  /* create work vectors for each split */
  if (!jac->x) {
    Vec xtmp;
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
    /* compute scatter contexts needed by multiplicative versions and non-default splits */
    
    ilink = jac->head;
    ierr = MatGetVecs(pc->pmat,&xtmp,PETSC_NULL);CHKERRQ(ierr);
    for (i=0; i<nsplit; i++) {
      ierr = VecScatterCreate(xtmp,jac->is[i],jac->x[i],PETSC_NULL,&ilink->sctx);CHKERRQ(ierr);
      ilink = ilink->next;
    }
    ierr = VecDestroy(xtmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define FieldSplitSplitSolveAdd(ilink,xx,yy) \
    (VecScatterBegin(xx,ilink->x,INSERT_VALUES,SCATTER_FORWARD,ilink->sctx) || \
     VecScatterEnd(xx,ilink->x,INSERT_VALUES,SCATTER_FORWARD,ilink->sctx) || \
     KSPSolve(ilink->ksp,ilink->x,ilink->y) || \
     VecScatterBegin(ilink->y,yy,ADD_VALUES,SCATTER_REVERSE,ilink->sctx) || \
     VecScatterEnd(ilink->y,yy,ADD_VALUES,SCATTER_REVERSE,ilink->sctx))

#undef __FUNCT__  
#define __FUNCT__ "PCApply_FieldSplit"
static PetscErrorCode PCApply_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          bs;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
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
  } else {
    if (!jac->w1) {
      ierr = VecDuplicate(x,&jac->w1);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&jac->w2);CHKERRQ(ierr);
    }
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = FieldSplitSplitSolveAdd(ilink,x,y);CHKERRQ(ierr);
    while (ilink->next) {
      ilink = ilink->next;
      ierr  = MatMult(pc->pmat,y,jac->w1);CHKERRQ(ierr);
      ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
      ierr  = FieldSplitSplitSolveAdd(ilink,jac->w2,y);CHKERRQ(ierr);
    }
    if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
      while (ilink->previous) {
        ilink = ilink->previous;
        ierr  = MatMult(pc->pmat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAdd(ilink,jac->w2,y);CHKERRQ(ierr);
      }
    }
  }
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#define FieldSplitSplitSolveAddTranspose(ilink,xx,yy) \
    (VecScatterBegin(xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD,ilink->sctx) || \
     VecScatterEnd(xx,ilink->y,INSERT_VALUES,SCATTER_FORWARD,ilink->sctx) || \
     KSPSolveTranspose(ilink->ksp,ilink->y,ilink->x) || \
     VecScatterBegin(ilink->x,yy,ADD_VALUES,SCATTER_REVERSE,ilink->sctx) || \
     VecScatterEnd(ilink->x,yy,ADD_VALUES,SCATTER_REVERSE,ilink->sctx))

#undef __FUNCT__  
#define __FUNCT__ "PCApply_FieldSplit"
static PetscErrorCode PCApplyTranspose_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink ilink = jac->head;
  PetscInt          bs;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
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
        ierr  = MatMultTranspose(pc->pmat,y,jac->w1);CHKERRQ(ierr);
        ierr  = VecWAXPY(jac->w2,-1.0,jac->w1,x);CHKERRQ(ierr);
        ierr  = FieldSplitSplitSolveAddTranspose(ilink,jac->w2,y);CHKERRQ(ierr);
      }
      while (ilink->previous) {
        ilink = ilink->previous;
        ierr  = MatMultTranspose(pc->pmat,y,jac->w1);CHKERRQ(ierr);
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
	ierr  = MatMultTranspose(pc->pmat,y,jac->w1);CHKERRQ(ierr);
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
    next = ilink->next;
    ierr = PetscFree2(ilink,ilink->fields);CHKERRQ(ierr);
    ilink = next;
  }
  ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);
  if (jac->pmat) {ierr = MatDestroyMatrices(jac->nsplits,&jac->pmat);CHKERRQ(ierr);}
  if (jac->is) {
    PetscInt i;
    for (i=0; i<jac->nsplits; i++) {ierr = ISDestroy(jac->is[i]);CHKERRQ(ierr);}
    ierr = PetscFree(jac->is);CHKERRQ(ierr);
  }
  if (jac->cis) {
    PetscInt i;
    for (i=0; i<jac->nsplits; i++) {ierr = ISDestroy(jac->cis[i]);CHKERRQ(ierr);}
    ierr = PetscFree(jac->cis);CHKERRQ(ierr);
  }
  if (jac->w1) {ierr = VecDestroy(jac->w1);CHKERRQ(ierr);}
  if (jac->w2) {ierr = VecDestroy(jac->w2);CHKERRQ(ierr);}
  ierr = PetscFree(jac->csize);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_FieldSplit"
static PetscErrorCode PCSetFromOptions_FieldSplit(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       i = 0,nfields,*fields,bs;
  PetscTruth     flg;
  char           optionname[128];
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("FieldSplit options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_fieldsplit_block_size","Blocksize that defines number of fields","PCFieldSplitSetBlockSize",jac->bs,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFieldSplitSetBlockSize(pc,bs);CHKERRQ(ierr);
  }
  if (jac->bs <= 0) {
    ierr = PCFieldSplitSetBlockSize(pc,1);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnum("-pc_fieldsplit_type","Type of composition","PCFieldSplitSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg);CHKERRQ(ierr);
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
  ierr = PetscMalloc2(1,struct _PC_FieldSplitLink,&ilink,n,PetscInt,&ilink->fields);CHKERRQ(ierr);
  ierr = PetscMemcpy(ilink->fields,fields,n*sizeof(PetscInt));CHKERRQ(ierr);
  ilink->nfields = n;
  ilink->next    = PETSC_NULL;
  ierr           = KSPCreate(pc->comm,&ilink->ksp);CHKERRQ(ierr);
  ierr           = KSPSetType(ilink->ksp,KSPPREONLY);CHKERRQ(ierr);

  if (pc->prefix) {
    sprintf(prefix,"%sfieldsplit_%d_",pc->prefix,(int)jac->nsplits);
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

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT, PCFieldSplitSetBlockSize()

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
   After PCFieldSplitGetSubKSP() the array of KSPs IS to be freed

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetType_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitSetType_FieldSplit(PC pc,PCCompositeType type)
{
  PC_FieldSplit  *jac = (PC_FieldSplit*)pc->data;

  PetscFunctionBegin;
  jac->type = type;
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
/*@C
   PCFieldSplitSetType - Sets the type of fieldsplit preconditioner.
   
   Collective on PC

   Input Parameter:
.  pc - the preconditioner context
.  type - PC_COMPOSITE_ADDITIVE (default), PC_COMPOSITE_MULTIPLICATIVE

   Options Database Key:
.  -pc_fieldsplit_type <type: one of multiplicative, additive, special> - Sets fieldsplit preconditioner type

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


     To set options on the solvers for each block append -sub_ to all the PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1
        
     To set the options on the solvers separate for each block call PCFieldSplitGetSubKSP()
         and set the options directly on the resulting KSP object

   Level: intermediate

   Options Database Keys:
+   -pc_splitfield_%d_fields <a,b,..> - indicates the fields to be used in the %d'th split
.   -pc_splitfield_default - automatically add any fields to additional splits that have not
                              been supplied explicitly by -pc_splitfield_%d_fields
.   -pc_splitfield_block_size <bs> - size of block that defines fields (i.e. there are bs fields)
-   -pc_splitfield_type <additive,multiplicative>

   Concepts: physics based preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields(),PCFieldSplitSetType()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_FieldSplit"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_FieldSplit(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  ierr = PetscNew(PC_FieldSplit,&jac);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_FieldSplit));CHKERRQ(ierr);
  jac->bs        = -1;
  jac->nsplits   = 0;
  jac->type      = PC_COMPOSITE_ADDITIVE;

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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetType_C","PCFieldSplitSetType_FieldSplit",
                    PCFieldSplitSetType_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetBlockSize_C","PCFieldSplitSetBlockSize_FieldSplit",
                    PCFieldSplitSetBlockSize_FieldSplit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


