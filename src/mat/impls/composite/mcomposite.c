
#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

const char *const MatCompositeMergeTypes[] = {"left","right","MatCompositeMergeType","MAT_COMPOSITE_",NULL};

typedef struct _Mat_CompositeLink *Mat_CompositeLink;
struct _Mat_CompositeLink {
  Mat               mat;
  Vec               work;
  Mat_CompositeLink next,prev;
};

typedef struct {
  MatCompositeType      type;
  Mat_CompositeLink     head,tail;
  Vec                   work;
  PetscScalar           scale;        /* scale factor supplied with MatScale() */
  Vec                   left,right;   /* left and right diagonal scaling provided with MatDiagonalScale() */
  Vec                   leftwork,rightwork,leftwork2,rightwork2; /* Two pairs of working vectors */
  PetscInt              nmat;
  PetscBool             merge;
  MatCompositeMergeType mergetype;
  MatStructure          structure;

  PetscScalar           *scalings;
  PetscBool             merge_mvctx;  /* Whether need to merge mvctx of component matrices */
  Vec                   *lvecs;       /* [nmat] Basically, they are Mvctx->lvec of each component matrix */
  PetscScalar           *larray;      /* [len] Data arrays of lvecs[] are stored consecutively in larray */
  PetscInt              len;          /* Length of larray[] */
  Vec                   gvec;         /* Union of lvecs[] without duplicated entries */
  PetscInt              *location;    /* A map that maps entries in garray[] to larray[] */
  VecScatter            Mvctx;
} Mat_Composite;

PetscErrorCode MatDestroy_Composite(Mat mat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink next   = shell->head,oldnext;
  PetscInt          i;

  PetscFunctionBegin;
  while (next) {
    PetscCall(MatDestroy(&next->mat));
    if (next->work && (!next->next || next->work != next->next->work)) {
      PetscCall(VecDestroy(&next->work));
    }
    oldnext = next;
    next    = next->next;
    PetscCall(PetscFree(oldnext));
  }
  PetscCall(VecDestroy(&shell->work));
  PetscCall(VecDestroy(&shell->left));
  PetscCall(VecDestroy(&shell->right));
  PetscCall(VecDestroy(&shell->leftwork));
  PetscCall(VecDestroy(&shell->rightwork));
  PetscCall(VecDestroy(&shell->leftwork2));
  PetscCall(VecDestroy(&shell->rightwork2));

  if (shell->Mvctx) {
    for (i=0; i<shell->nmat; i++) PetscCall(VecDestroy(&shell->lvecs[i]));
    PetscCall(PetscFree3(shell->location,shell->larray,shell->lvecs));
    PetscCall(PetscFree(shell->larray));
    PetscCall(VecDestroy(&shell->gvec));
    PetscCall(VecScatterDestroy(&shell->Mvctx));
  }

  PetscCall(PetscFree(shell->scalings));
  PetscCall(PetscFree(mat->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Composite_Multiplicative(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  Vec               in,out;
  PetscScalar       scale;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCheck(next,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      PetscCall(VecDuplicate(shell->right,&shell->rightwork));
    }
    PetscCall(VecPointwiseMult(shell->rightwork,shell->right,in));
    in   = shell->rightwork;
  }
  while (next->next) {
    if (!next->work) { /* should reuse previous work if the same size */
      PetscCall(MatCreateVecs(next->mat,NULL,&next->work));
    }
    out  = next->work;
    PetscCall(MatMult(next->mat,in,out));
    in   = out;
    next = next->next;
  }
  PetscCall(MatMult(next->mat,in,y));
  if (shell->left) {
    PetscCall(VecPointwiseMult(y,shell->left,y));
  }
  scale = shell->scale;
  if (shell->scalings) {for (i=0; i<shell->nmat; i++) scale *= shell->scalings[i];}
  PetscCall(VecScale(y,scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Composite_Multiplicative(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink tail   = shell->tail;
  Vec               in,out;
  PetscScalar       scale;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCheck(tail,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      PetscCall(VecDuplicate(shell->left,&shell->leftwork));
    }
    PetscCall(VecPointwiseMult(shell->leftwork,shell->left,in));
    in   = shell->leftwork;
  }
  while (tail->prev) {
    if (!tail->prev->work) { /* should reuse previous work if the same size */
      PetscCall(MatCreateVecs(tail->mat,NULL,&tail->prev->work));
    }
    out  = tail->prev->work;
    PetscCall(MatMultTranspose(tail->mat,in,out));
    in   = out;
    tail = tail->prev;
  }
  PetscCall(MatMultTranspose(tail->mat,in,y));
  if (shell->right) {
    PetscCall(VecPointwiseMult(y,shell->right,y));
  }

  scale = shell->scale;
  if (shell->scalings) {for (i=0; i<shell->nmat; i++) scale *= shell->scalings[i];}
  PetscCall(VecScale(y,scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Composite(Mat mat,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink cur = shell->head;
  Vec               in,y2,xin;
  Mat               A,B;
  PetscInt          i,j,k,n,nuniq,lo,hi,mid,*gindices,*buf,*tmp,tot;
  const PetscScalar *vals;
  const PetscInt    *garray;
  IS                ix,iy;
  PetscBool         match;

  PetscFunctionBegin;
  PetscCheck(cur,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      PetscCall(VecDuplicate(shell->right,&shell->rightwork));
    }
    PetscCall(VecPointwiseMult(shell->rightwork,shell->right,in));
    in   = shell->rightwork;
  }

  /* Try to merge Mvctx when instructed but not yet done. We did not do it in MatAssemblyEnd() since at that time
     we did not know whether mat is ADDITIVE or MULTIPLICATIVE. Only now we are assured mat is ADDITIVE and
     it is legal to merge Mvctx, because all component matrices have the same size.
   */
  if (shell->merge_mvctx && !shell->Mvctx) {
    /* Currently only implemented for MATMPIAIJ */
    for (cur=shell->head; cur; cur=cur->next) {
      PetscCall(PetscObjectTypeCompare((PetscObject)cur->mat,MATMPIAIJ,&match));
      if (!match) {
        shell->merge_mvctx = PETSC_FALSE;
        goto skip_merge_mvctx;
      }
    }

    /* Go through matrices first time to count total number of nonzero off-diag columns (may have dups) */
    tot = 0;
    for (cur=shell->head; cur; cur=cur->next) {
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat,NULL,&B,NULL));
      PetscCall(MatGetLocalSize(B,NULL,&n));
      tot += n;
    }
    PetscCall(PetscMalloc3(tot,&shell->location,tot,&shell->larray,shell->nmat,&shell->lvecs));
    shell->len = tot;

    /* Go through matrices second time to sort off-diag columns and remove dups */
    PetscCall(PetscMalloc1(tot,&gindices)); /* No Malloc2() since we will give one to petsc and free the other */
    PetscCall(PetscMalloc1(tot,&buf));
    nuniq = 0; /* Number of unique nonzero columns */
    for (cur=shell->head; cur; cur=cur->next) {
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat,NULL,&B,&garray));
      PetscCall(MatGetLocalSize(B,NULL,&n));
      /* Merge pre-sorted garray[0,n) and gindices[0,nuniq) to buf[] */
      i = j = k = 0;
      while (i < n && j < nuniq) {
        if (garray[i] < gindices[j]) buf[k++] = garray[i++];
        else if (garray[i] > gindices[j]) buf[k++] = gindices[j++];
        else {buf[k++] = garray[i++]; j++;}
      }
      /* Copy leftover in garray[] or gindices[] */
      if (i < n) {
        PetscCall(PetscArraycpy(buf+k,garray+i,n-i));
        nuniq = k + n-i;
      } else if (j < nuniq) {
        PetscCall(PetscArraycpy(buf+k,gindices+j,nuniq-j));
        nuniq = k + nuniq-j;
      } else nuniq = k;
      /* Swap gindices and buf to merge garray of the next matrix */
      tmp      = gindices;
      gindices = buf;
      buf      = tmp;
    }
    PetscCall(PetscFree(buf));

    /* Go through matrices third time to build a map from gindices[] to garray[] */
    tot = 0;
    for (cur=shell->head,j=0; cur; cur=cur->next,j++) { /* j-th matrix */
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat,NULL,&B,&garray));
      PetscCall(MatGetLocalSize(B,NULL,&n));
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,NULL,&shell->lvecs[j]));
      /* This is an optimized PetscFindInt(garray[i],nuniq,gindices,&shell->location[tot+i]), using the fact that garray[] is also sorted */
      lo   = 0;
      for (i=0; i<n; i++) {
        hi = nuniq;
        while (hi - lo > 1) {
          mid = lo + (hi - lo)/2;
          if (garray[i] < gindices[mid]) hi = mid;
          else lo = mid;
        }
        shell->location[tot+i] = lo; /* gindices[lo] = garray[i] */
        lo++; /* Since garray[i+1] > garray[i], we can safely advance lo */
      }
      tot += n;
    }

    /* Build merged Mvctx */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nuniq,gindices,PETSC_OWN_POINTER,&ix));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,nuniq,0,1,&iy));
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&xin));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,nuniq,&shell->gvec));
    PetscCall(VecScatterCreate(xin,ix,shell->gvec,iy,&shell->Mvctx));
    PetscCall(VecDestroy(&xin));
    PetscCall(ISDestroy(&ix));
    PetscCall(ISDestroy(&iy));
  }

skip_merge_mvctx:
  PetscCall(VecSet(y,0));
  if (!shell->leftwork2) PetscCall(VecDuplicate(y,&shell->leftwork2));
  y2 = shell->leftwork2;

  if (shell->Mvctx) { /* Have a merged Mvctx */
    /* Suppose we want to compute y = sMx, where s is the scaling factor and A, B are matrix M's diagonal/off-diagonal part. We could do
       in y = s(Ax1 + Bx2) or y = sAx1 + sBx2. The former incurs less FLOPS than the latter, but the latter provides an oppertunity to
       overlap communication/computation since we can do sAx1 while communicating x2. Here, we use the former approach.
     */
    PetscCall(VecScatterBegin(shell->Mvctx,in,shell->gvec,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(shell->Mvctx,in,shell->gvec,INSERT_VALUES,SCATTER_FORWARD));

    PetscCall(VecGetArrayRead(shell->gvec,&vals));
    for (i=0; i<shell->len; i++) shell->larray[i] = vals[shell->location[i]];
    PetscCall(VecRestoreArrayRead(shell->gvec,&vals));

    for (cur=shell->head,tot=i=0; cur; cur=cur->next,i++) { /* i-th matrix */
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat,&A,&B,NULL));
      PetscCall((*A->ops->mult)(A,in,y2));
      PetscCall(MatGetLocalSize(B,NULL,&n));
      PetscCall(VecPlaceArray(shell->lvecs[i],&shell->larray[tot]));
      PetscCall((*B->ops->multadd)(B,shell->lvecs[i],y2,y2));
      PetscCall(VecResetArray(shell->lvecs[i]));
      PetscCall(VecAXPY(y,(shell->scalings ? shell->scalings[i] : 1.0),y2));
      tot += n;
    }
  } else {
    if (shell->scalings) {
      for (cur=shell->head,i=0; cur; cur=cur->next,i++) {
        PetscCall(MatMult(cur->mat,in,y2));
        PetscCall(VecAXPY(y,shell->scalings[i],y2));
      }
    } else {
      for (cur=shell->head; cur; cur=cur->next) PetscCall(MatMultAdd(cur->mat,in,y,y));
    }
  }

  if (shell->left) PetscCall(VecPointwiseMult(y,shell->left,y));
  PetscCall(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Composite(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  Vec               in,y2 = NULL;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCheck(next,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      PetscCall(VecDuplicate(shell->left,&shell->leftwork));
    }
    PetscCall(VecPointwiseMult(shell->leftwork,shell->left,in));
    in   = shell->leftwork;
  }

  PetscCall(MatMultTranspose(next->mat,in,y));
  if (shell->scalings) {
    PetscCall(VecScale(y,shell->scalings[0]));
    if (!shell->rightwork2) PetscCall(VecDuplicate(y,&shell->rightwork2));
    y2 = shell->rightwork2;
  }
  i = 1;
  while ((next = next->next)) {
    if (!shell->scalings) PetscCall(MatMultTransposeAdd(next->mat,in,y,y));
    else {
      PetscCall(MatMultTranspose(next->mat,in,y2));
      PetscCall(VecAXPY(y,shell->scalings[i++],y2));
    }
  }
  if (shell->right) {
    PetscCall(VecPointwiseMult(y,shell->right,y));
  }
  PetscCall(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Composite(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;

  PetscFunctionBegin;
  if (y != z) {
    PetscCall(MatMult(A,x,z));
    PetscCall(VecAXPY(z,1.0,y));
  } else {
    if (!shell->leftwork) {
      PetscCall(VecDuplicate(z,&shell->leftwork));
    }
    PetscCall(MatMult(A,x,shell->leftwork));
    PetscCall(VecCopy(y,z));
    PetscCall(VecAXPY(z,1.0,shell->leftwork));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Composite(Mat A,Vec x,Vec y, Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;

  PetscFunctionBegin;
  if (y != z) {
    PetscCall(MatMultTranspose(A,x,z));
    PetscCall(VecAXPY(z,1.0,y));
  } else {
    if (!shell->rightwork) {
      PetscCall(VecDuplicate(z,&shell->rightwork));
    }
    PetscCall(MatMultTranspose(A,x,shell->rightwork));
    PetscCall(VecCopy(y,z));
    PetscCall(VecAXPY(z,1.0,shell->rightwork));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Composite(Mat A,Vec v)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next   = shell->head;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCheck(next,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  PetscCheck(!shell->right && !shell->left,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get diagonal if left or right scaling");

  PetscCall(MatGetDiagonal(next->mat,v));
  if (shell->scalings) PetscCall(VecScale(v,shell->scalings[0]));

  if (next->next && !shell->work) {
    PetscCall(VecDuplicate(v,&shell->work));
  }
  i = 1;
  while ((next = next->next)) {
    PetscCall(MatGetDiagonal(next->mat,shell->work));
    PetscCall(VecAXPY(v,(shell->scalings ? shell->scalings[i++] : 1.0),shell->work));
  }
  PetscCall(VecScale(v,shell->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Composite(Mat Y,MatAssemblyType t)
{
  Mat_Composite  *shell = (Mat_Composite*)Y->data;

  PetscFunctionBegin;
  if (shell->merge) {
    PetscCall(MatCompositeMerge(Y));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_Composite(Mat inA,PetscScalar alpha)
{
  Mat_Composite *a = (Mat_Composite*)inA->data;

  PetscFunctionBegin;
  a->scale *= alpha;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_Composite(Mat inA,Vec left,Vec right)
{
  Mat_Composite  *a = (Mat_Composite*)inA->data;

  PetscFunctionBegin;
  if (left) {
    if (!a->left) {
      PetscCall(VecDuplicate(left,&a->left));
      PetscCall(VecCopy(left,a->left));
    } else {
      PetscCall(VecPointwiseMult(a->left,left,a->left));
    }
  }
  if (right) {
    if (!a->right) {
      PetscCall(VecDuplicate(right,&a->right));
      PetscCall(VecCopy(right,a->right));
    } else {
      PetscCall(VecPointwiseMult(a->right,right,a->right));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_Composite(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_Composite  *a = (Mat_Composite*)A->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"MATCOMPOSITE options");
  PetscCall(PetscOptionsBool("-mat_composite_merge","Merge at MatAssemblyEnd","MatCompositeMerge",a->merge,&a->merge,NULL));
  PetscCall(PetscOptionsEnum("-mat_composite_merge_type","Set composite merge direction","MatCompositeSetMergeType",MatCompositeMergeTypes,(PetscEnum)a->mergetype,(PetscEnum*)&a->mergetype,NULL));
  PetscCall(PetscOptionsBool("-mat_composite_merge_mvctx","Merge MatMult() vecscat contexts","MatCreateComposite",a->merge_mvctx,&a->merge_mvctx,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*@
   MatCreateComposite - Creates a matrix as the sum or product of one or more matrices

  Collective

   Input Parameters:
+  comm - MPI communicator
.  nmat - number of matrices to put in
-  mats - the matrices

   Output Parameter:
.  mat - the matrix

   Options Database Keys:
+  -mat_composite_merge         - merge in MatAssemblyEnd()
.  -mat_composite_merge_mvctx   - merge Mvctx of component matrices to optimize communication in MatMult() for ADDITIVE matrices
-  -mat_composite_merge_type    - set merge direction

   Level: advanced

   Notes:
     Alternative construction
$       MatCreate(comm,&mat);
$       MatSetSizes(mat,m,n,M,N);
$       MatSetType(mat,MATCOMPOSITE);
$       MatCompositeAddMat(mat,mats[0]);
$       ....
$       MatCompositeAddMat(mat,mats[nmat-1]);
$       MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
$       MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);

     For the multiplicative form the product is mat[nmat-1]*mat[nmat-2]*....*mat[0]

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCompositeGetMat(), MatCompositeMerge(), MatCompositeSetType(), MATCOMPOSITE

@*/
PetscErrorCode MatCreateComposite(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscInt       m,n,M,N,i;

  PetscFunctionBegin;
  PetscCheck(nmat >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,4);

  PetscCall(MatGetLocalSize(mats[0],PETSC_IGNORE,&n));
  PetscCall(MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE));
  PetscCall(MatGetSize(mats[0],PETSC_IGNORE,&N));
  PetscCall(MatGetSize(mats[nmat-1],&M,PETSC_IGNORE));
  PetscCall(MatCreate(comm,mat));
  PetscCall(MatSetSizes(*mat,m,n,M,N));
  PetscCall(MatSetType(*mat,MATCOMPOSITE));
  for (i=0; i<nmat; i++) {
    PetscCall(MatCompositeAddMat(*mat,mats[i]));
  }
  PetscCall(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeAddMat_Composite(Mat mat,Mat smat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink ilink,next = shell->head;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(mat,&ilink));
  ilink->next = NULL;
  PetscCall(PetscObjectReference((PetscObject)smat));
  ilink->mat  = smat;

  if (!next) shell->head = ilink;
  else {
    while (next->next) {
      next = next->next;
    }
    next->next  = ilink;
    ilink->prev = next;
  }
  shell->tail =  ilink;
  shell->nmat += 1;

  /* Retain the old scalings (if any) and expand it with a 1.0 for the newly added matrix */
  if (shell->scalings) {
    PetscCall(PetscRealloc(sizeof(PetscScalar)*shell->nmat,&shell->scalings));
    shell->scalings[shell->nmat-1] = 1.0;
  }
  PetscFunctionReturn(0);
}

/*@
    MatCompositeAddMat - Add another matrix to a composite matrix.

   Collective on Mat

    Input Parameters:
+   mat - the composite matrix
-   smat - the partial matrix

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeGetMat(), MATCOMPOSITE
@*/
PetscErrorCode MatCompositeAddMat(Mat mat,Mat smat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(smat,MAT_CLASSID,2);
  PetscUseMethod(mat,"MatCompositeAddMat_C",(Mat,Mat),(mat,smat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeSetType_Composite(Mat mat,MatCompositeType type)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  b->type = type;
  if (type == MAT_COMPOSITE_MULTIPLICATIVE) {
    mat->ops->getdiagonal   = NULL;
    mat->ops->mult          = MatMult_Composite_Multiplicative;
    mat->ops->multtranspose = MatMultTranspose_Composite_Multiplicative;
    b->merge_mvctx          = PETSC_FALSE;
  } else {
    mat->ops->getdiagonal   = MatGetDiagonal_Composite;
    mat->ops->mult          = MatMult_Composite;
    mat->ops->multtranspose = MatMultTranspose_Composite;
  }
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetType - Indicates if the matrix is defined as the sum of a set of matrices or the product.

   Logically Collective on Mat

   Input Parameters:
.  mat - the composite matrix

   Level: advanced

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCreateComposite(), MatCompositeGetType(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetType(Mat mat,MatCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,type,2);
  PetscUseMethod(mat,"MatCompositeSetType_C",(Mat,MatCompositeType),(mat,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetType_Composite(Mat mat,MatCompositeType *type)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  *type = b->type;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetType - Returns type of composite.

   Not Collective

   Input Parameter:
.  mat - the composite matrix

   Output Parameter:
.  type - type of composite

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeSetType(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetType(Mat mat,MatCompositeType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(type,2);
  PetscUseMethod(mat,"MatCompositeGetType_C",(Mat,MatCompositeType*),(mat,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeSetMatStructure_Composite(Mat mat,MatStructure str)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  b->structure = str;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetMatStructure - Indicates structure of matrices in the composite matrix.

   Not Collective

   Input Parameters:
+  mat - the composite matrix
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN (default) or SUBSET_NONZERO_PATTERN

   Level: advanced

   Notes:
    Information about the matrices structure is used in MatCompositeMerge() for additive composite matrix.

.seealso: MatAXPY(), MatCreateComposite(), MatCompositeMerge() MatCompositeGetMatStructure(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetMatStructure(Mat mat,MatStructure str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatCompositeSetMatStructure_C",(Mat,MatStructure),(mat,str));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetMatStructure_Composite(Mat mat,MatStructure *str)
{
  Mat_Composite  *b = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  *str = b->structure;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetMatStructure - Returns the structure of matrices in the composite matrix.

   Not Collective

   Input Parameter:
.  mat - the composite matrix

   Output Parameter:
.  str - structure of the matrices

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeSetMatStructure(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetMatStructure(Mat mat,MatStructure *str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(str,2);
  PetscUseMethod(mat,"MatCompositeGetMatStructure_C",(Mat,MatStructure*),(mat,str));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeSetMergeType_Composite(Mat mat,MatCompositeMergeType type)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  shell->mergetype = type;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetMergeType - Sets order of MatCompositeMerge().

   Logically Collective on Mat

   Input Parameters:
+  mat - the composite matrix
-  type - MAT_COMPOSITE_MERGE RIGHT (default) to start merge from right with the first added matrix (mat[0]),
          MAT_COMPOSITE_MERGE_LEFT to start merge from left with the last added matrix (mat[nmat-1])

   Level: advanced

   Notes:
    The resulting matrix is the same regardles of the MergeType. Only the order of operation is changed.
    If set to MAT_COMPOSITE_MERGE_RIGHT the order of the merge is mat[nmat-1]*(mat[nmat-2]*(...*(mat[1]*mat[0])))
    otherwise the order is (((mat[nmat-1]*mat[nmat-2])*mat[nmat-3])*...)*mat[0].

.seealso: MatCreateComposite(), MatCompositeMerge(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetMergeType(Mat mat,MatCompositeMergeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(mat,type,2);
  PetscUseMethod(mat,"MatCompositeSetMergeType_C",(Mat,MatCompositeMergeType),(mat,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeMerge_Composite(Mat mat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink next   = shell->head, prev = shell->tail;
  Mat               tmat,newmat;
  Vec               left,right;
  PetscScalar       scale;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCheck(next,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  scale = shell->scale;
  if (shell->type == MAT_COMPOSITE_ADDITIVE) {
    if (shell->mergetype == MAT_COMPOSITE_MERGE_RIGHT) {
      i = 0;
      PetscCall(MatDuplicate(next->mat,MAT_COPY_VALUES,&tmat));
      if (shell->scalings) PetscCall(MatScale(tmat,shell->scalings[i++]));
      while ((next = next->next)) {
        PetscCall(MatAXPY(tmat,(shell->scalings ? shell->scalings[i++] : 1.0),next->mat,shell->structure));
      }
    } else {
      i = shell->nmat-1;
      PetscCall(MatDuplicate(prev->mat,MAT_COPY_VALUES,&tmat));
      if (shell->scalings) PetscCall(MatScale(tmat,shell->scalings[i--]));
      while ((prev = prev->prev)) {
        PetscCall(MatAXPY(tmat,(shell->scalings ? shell->scalings[i--] : 1.0),prev->mat,shell->structure));
      }
    }
  } else {
    if (shell->mergetype == MAT_COMPOSITE_MERGE_RIGHT) {
      PetscCall(MatDuplicate(next->mat,MAT_COPY_VALUES,&tmat));
      while ((next = next->next)) {
        PetscCall(MatMatMult(next->mat,tmat,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat));
        PetscCall(MatDestroy(&tmat));
        tmat = newmat;
      }
    } else {
      PetscCall(MatDuplicate(prev->mat,MAT_COPY_VALUES,&tmat));
      while ((prev = prev->prev)) {
        PetscCall(MatMatMult(tmat,prev->mat,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat));
        PetscCall(MatDestroy(&tmat));
        tmat = newmat;
      }
    }
    if (shell->scalings) {for (i=0; i<shell->nmat; i++) scale *= shell->scalings[i];}
  }

  if ((left = shell->left)) PetscCall(PetscObjectReference((PetscObject)left));
  if ((right = shell->right)) PetscCall(PetscObjectReference((PetscObject)right));

  PetscCall(MatHeaderReplace(mat,&tmat));

  PetscCall(MatDiagonalScale(mat,left,right));
  PetscCall(MatScale(mat,scale));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscFunctionReturn(0);
}

/*@
   MatCompositeMerge - Given a composite matrix, replaces it with a "regular" matrix
     by summing or computing the product of all the matrices inside the composite matrix.

  Collective

   Input Parameter:
.  mat - the composite matrix

   Options Database Keys:
+  -mat_composite_merge - merge in MatAssemblyEnd()
-  -mat_composite_merge_type - set merge direction

   Level: advanced

   Notes:
      The MatType of the resulting matrix will be the same as the MatType of the FIRST
    matrix in the composite matrix.

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCreateComposite(), MatCompositeSetMatStructure(), MatCompositeSetMergeType(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeMerge(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatCompositeMerge_C",(Mat),(mat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetNumberMat_Composite(Mat mat,PetscInt *nmat)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;

  PetscFunctionBegin;
  *nmat = shell->nmat;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetNumberMat - Returns the number of matrices in the composite matrix.

   Not Collective

   Input Parameter:
.  mat - the composite matrix

   Output Parameter:
.  nmat - number of matrices in the composite matrix

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeGetMat(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetNumberMat(Mat mat,PetscInt *nmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidIntPointer(nmat,2);
  PetscUseMethod(mat,"MatCompositeGetNumberMat_C",(Mat,PetscInt*),(mat,nmat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCompositeGetMat_Composite(Mat mat,PetscInt i,Mat *Ai)
{
  Mat_Composite     *shell = (Mat_Composite*)mat->data;
  Mat_CompositeLink ilink;
  PetscInt          k;

  PetscFunctionBegin;
  PetscCheck(i < shell->nmat,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"index out of range: %" PetscInt_FMT " >= %" PetscInt_FMT,i,shell->nmat);
  ilink = shell->head;
  for (k=0; k<i; k++) {
    ilink = ilink->next;
  }
  *Ai = ilink->mat;
  PetscFunctionReturn(0);
}

/*@
   MatCompositeGetMat - Returns the ith matrix from the composite matrix.

   Logically Collective on Mat

   Input Parameters:
+  mat - the composite matrix
-  i - the number of requested matrix

   Output Parameter:
.  Ai - ith matrix in composite

   Level: advanced

.seealso: MatCreateComposite(), MatCompositeGetNumberMat(), MatCompositeAddMat(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeGetMat(Mat mat,PetscInt i,Mat *Ai)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(mat,i,2);
  PetscValidPointer(Ai,3);
  PetscUseMethod(mat,"MatCompositeGetMat_C",(Mat,PetscInt,Mat*),(mat,i,Ai));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCompositeSetScalings_Composite(Mat mat,const PetscScalar *scalings)
{
  Mat_Composite  *shell = (Mat_Composite*)mat->data;
  PetscInt       nmat;

  PetscFunctionBegin;
  PetscCall(MatCompositeGetNumberMat(mat,&nmat));
  if (!shell->scalings) PetscCall(PetscMalloc1(nmat,&shell->scalings));
  PetscCall(PetscArraycpy(shell->scalings,scalings,nmat));
  PetscFunctionReturn(0);
}

/*@
   MatCompositeSetScalings - Sets separate scaling factors for component matrices.

   Logically Collective on Mat

   Input Parameters:
+  mat      - the composite matrix
-  scalings - array of scaling factors with scalings[i] being factor of i-th matrix, for i in [0, nmat)

   Level: advanced

.seealso: MatScale(), MatDiagonalScale(), MATCOMPOSITE

@*/
PetscErrorCode MatCompositeSetScalings(Mat mat,const PetscScalar *scalings)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidScalarPointer(scalings,2);
  PetscValidLogicalCollectiveScalar(mat,*scalings,2);
  PetscUseMethod(mat,"MatCompositeSetScalings_C",(Mat,const PetscScalar*),(mat,scalings));
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {NULL,
                                       NULL,
                                       NULL,
                                       MatMult_Composite,
                                       MatMultAdd_Composite,
                               /*  5*/ MatMultTranspose_Composite,
                                       MatMultTransposeAdd_Composite,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 15*/ NULL,
                                       NULL,
                                       MatGetDiagonal_Composite,
                                       MatDiagonalScale_Composite,
                                       NULL,
                               /* 20*/ NULL,
                                       MatAssemblyEnd_Composite,
                                       NULL,
                                       NULL,
                               /* 24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 34*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 39*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 44*/ NULL,
                                       MatScale_Composite,
                                       MatShift_Basic,
                                       NULL,
                                       NULL,
                               /* 49*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 54*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 59*/ NULL,
                                       MatDestroy_Composite,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       NULL,
                                       MatSetFromOptions_Composite,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*99*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*104*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*144*/NULL,
                                       NULL,
                                       NULL,
                                       NULL
};

/*MC
   MATCOMPOSITE - A matrix defined by the sum (or product) of one or more matrices.
    The matrices need to have a correct size and parallel layout for the sum or product to be valid.

   Notes:
    to use the product of the matrices call MatCompositeSetType(mat,MAT_COMPOSITE_MULTIPLICATIVE);

  Level: advanced

.seealso: MatCreateComposite(), MatCompositeSetScalings(), MatCompositeAddMat(), MatSetType(), MatCompositeSetType(), MatCompositeGetType(), MatCompositeSetMatStructure(), MatCompositeGetMatStructure(), MatCompositeMerge(), MatCompositeSetMergeType(), MatCompositeGetNumberMat(), MatCompositeGetMat()
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Composite(Mat A)
{
  Mat_Composite  *b;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(A,&b));
  A->data = (void*)b;
  PetscCall(PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps)));

  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_TRUE;
  b->type         = MAT_COMPOSITE_ADDITIVE;
  b->scale        = 1.0;
  b->nmat         = 0;
  b->merge        = PETSC_FALSE;
  b->mergetype    = MAT_COMPOSITE_MERGE_RIGHT;
  b->structure    = DIFFERENT_NONZERO_PATTERN;
  b->merge_mvctx  = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeAddMat_C",MatCompositeAddMat_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetType_C",MatCompositeSetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetType_C",MatCompositeGetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetMergeType_C",MatCompositeSetMergeType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetMatStructure_C",MatCompositeSetMatStructure_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetMatStructure_C",MatCompositeGetMatStructure_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeMerge_C",MatCompositeMerge_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetNumberMat_C",MatCompositeGetNumberMat_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeGetMat_C",MatCompositeGetMat_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCompositeSetScalings_C",MatCompositeSetScalings_Composite));

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATCOMPOSITE));
  PetscFunctionReturn(0);
}
