
#include "matnestimpl.h" /*I   "petscmat.h"   I*/

static PetscErrorCode MatSetUp_NestIS_Private(Mat,PetscInt,const IS[],PetscInt,const IS[]);
static PetscErrorCode MatGetVecs_Nest(Mat A,Vec *right,Vec *left);

/* private functions */
#undef __FUNCT__ 
#define __FUNCT__ "MatNestGetSizes_Private"
static PetscErrorCode MatNestGetSizes_Private(Mat A,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *m = *n = *M = *N = 0;
  for (i=0; i<bA->nr; i++) {  /* rows */
    PetscInt sm,sM;
    ierr = ISGetLocalSize(bA->isglobal.row[i],&sm);CHKERRQ(ierr);
    ierr = ISGetSize(bA->isglobal.row[i],&sM);CHKERRQ(ierr);
    *m += sm;
    *M += sM;
  }
  for (j=0; j<bA->nc; j++) {  /* cols */
    PetscInt sn,sN;
    ierr = ISGetLocalSize(bA->isglobal.col[j],&sn);CHKERRQ(ierr);
    ierr = ISGetSize(bA->isglobal.col[j],&sN);CHKERRQ(ierr);
    *n += sn;
    *N += sN;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PETSc_MatNest_CheckMatVecCompatibility2"
PETSC_UNUSED
static PetscErrorCode PETSc_MatNest_CheckMatVecCompatibility2(Mat A,Vec x,Vec y)
{
  PetscBool      isANest, isxNest, isyNest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  isANest = isxNest = isyNest = PETSC_FALSE;
  ierr = PetscTypeCompare( (PetscObject)A, MATNEST, &isANest );CHKERRQ(ierr);
  ierr = PetscTypeCompare( (PetscObject)x, VECNEST, &isxNest );CHKERRQ(ierr);
  ierr = PetscTypeCompare( (PetscObject)y, VECNEST, &isyNest );CHKERRQ(ierr);

  if (isANest && isxNest && isyNest) {
    PetscFunctionReturn(0);
  } else {
    SETERRQ( ((PetscObject)A)->comm, PETSC_ERR_SUP, "Operation only valid for Mat(Nest)-Vec(Nest) combinations");
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

/*
 This function is called every time we insert a sub matrix the Nest.
 It ensures that every Nest along row r, and col c has the same dimensions
 as the submat being inserted.
*/
#undef __FUNCT__  
#define __FUNCT__ "PETSc_MatNest_CheckConsistency"
PETSC_UNUSED
static PetscErrorCode PETSc_MatNest_CheckConsistency(Mat A,Mat submat,PetscInt r,PetscInt c)
{
  Mat_Nest       *b = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscInt       nr,nc;
  PetscInt       sM,sN,M,N;
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = b->nr;
  nc = b->nc;
  ierr = MatGetSize(submat,&sM,&sN);CHKERRQ(ierr);
  for (i=0; i<nr; i++) {
    mat = b->m[i][c];
    if (mat) {
      ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
      /* Check columns match */
      if (sN != N) {
        SETERRQ3(((PetscObject)A)->comm,PETSC_ERR_SUP,"Inserted incompatible submatrix into Nest at (%D,%D). Submatrix must have %D rows",r,c,N );
      }
    }
  }

  for (j=0; j<nc; j++) {
    mat = b->m[r][j];
    if (mat) {
      ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
      /* Check rows match */
      if (sM != M) {
        SETERRQ3(((PetscObject)A)->comm,PETSC_ERR_SUP,"Inserted incompatible submatrix into Nest at (%D,%D). Submatrix must have %D cols",r,c,M );
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
 Checks if the row/col's contain a complete set of IS's.
 Once they are filled, the offsets are computed. This saves having to update
 the index set entries each time we insert something new.
 */
#undef __FUNCT__  
#define __FUNCT__ "PETSc_MatNest_UpdateStructure"
PETSC_UNUSED
static PetscErrorCode PETSc_MatNest_UpdateStructure(Mat A,PetscInt ridx,PetscInt cidx)
{
  Mat_Nest       *b = (Mat_Nest*)A->data;
  PetscInt       m,n,i,j;
  PetscBool      fullrow,fullcol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(b->m[ridx][cidx],&m,&n);CHKERRQ(ierr);
  b->row_len[ridx] = m;
  b->col_len[cidx] = n;

  fullrow = PETSC_TRUE;
  for (i=0; i<b->nr; i++) {
    if (b->row_len[i]==-1) { fullrow = PETSC_FALSE; break; }
  }
  if ( (fullrow) && (!b->isglobal.row[0]) ){
    PetscInt cnt = 0;
    for (i=0; i<b->nr; i++) {
      ierr = ISCreateStride(PETSC_COMM_SELF,b->row_len[i],cnt,1,&b->isglobal.row[i]);CHKERRQ(ierr);
      cnt = cnt + b->row_len[i];
    }
  }

  fullcol = PETSC_TRUE;
  for (j=0; j<b->nc; j++) {
    if (b->col_len[j]==-1) { fullcol = PETSC_FALSE; break; }
  }
  if( (fullcol) && (!b->isglobal.col[0]) ){
    PetscInt cnt = 0;
    for (j=0; j<b->nc; j++) {
      ierr = ISCreateStride(PETSC_COMM_SELF,b->col_len[j],cnt,1,&b->isglobal.col[j]);CHKERRQ(ierr);
      cnt = cnt + b->col_len[j];
    }
  }
  PetscFunctionReturn(0);
}

/* operations */
#undef __FUNCT__  
#define __FUNCT__ "MatMult_Nest"
static PetscErrorCode MatMult_Nest(Mat A,Vec x,Vec y)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bx = bA->right,*by = bA->left;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<nr; i++) {ierr = VecGetSubVector(y,bA->isglobal.row[i],&by[i]);CHKERRQ(ierr);}
  for (i=0; i<nc; i++) {ierr = VecGetSubVector(x,bA->isglobal.col[i],&bx[i]);CHKERRQ(ierr);}
  for (i=0; i<nr; i++) {
    ierr = VecZeroEntries(by[i]);CHKERRQ(ierr);
    for (j=0; j<nc; j++) {
      if (!bA->m[i][j]) continue;
      /* y[i] <- y[i] + A[i][j] * x[j] */
      ierr = MatMultAdd(bA->m[i][j],bx[j],by[i],by[i]);CHKERRQ(ierr);
    }
  }
  for (i=0; i<nr; i++) {ierr = VecRestoreSubVector(y,bA->isglobal.row[i],&by[i]);CHKERRQ(ierr);}
  for (i=0; i<nc; i++) {ierr = VecRestoreSubVector(x,bA->isglobal.col[i],&bx[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Nest"
static PetscErrorCode MatMultTranspose_Nest(Mat A,Vec x,Vec y)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bx = bA->left,*by = bA->right;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->symmetric) {
    ierr = MatMult_Nest(A,x,y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (i=0; i<nr; i++) {ierr = VecGetSubVector(y,bA->isglobal.row[i],&bx[i]);CHKERRQ(ierr);}
  for (i=0; i<nc; i++) {ierr = VecGetSubVector(x,bA->isglobal.col[i],&by[i]);CHKERRQ(ierr);}
  for (i=0; i<nr; i++) {
    ierr = VecZeroEntries(by[i]);CHKERRQ(ierr);
    for (j=0; j<nc; j++) {
      if (!bA->m[j][i]) continue;
      /* y[i] <- y[i] + A^T[i][j] * x[j], so we swap i,j in mat[][] */
      ierr = MatMultTransposeAdd(bA->m[j][i],bx[j],by[i],by[i]);CHKERRQ(ierr);
    }
  }
  for (i=0; i<nr; i++) {ierr = VecRestoreSubVector(y,bA->isglobal.row[i],&bx[i]);CHKERRQ(ierr);}
  for (i=0; i<nc; i++) {ierr = VecRestoreSubVector(x,bA->isglobal.col[i],&by[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNestDestroyISList"
static PetscErrorCode MatNestDestroyISList(PetscInt n,IS **list)
{
  PetscErrorCode ierr;
  IS             *lst = *list;
  PetscInt       i;

  PetscFunctionBegin;
  if (!lst) PetscFunctionReturn(0);
  for (i=0; i<n; i++) if (lst[i]) {ierr = ISDestroy(&lst[i]);CHKERRQ(ierr);}
  ierr = PetscFree(lst);CHKERRQ(ierr);
  *list = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Nest"
static PetscErrorCode MatDestroy_Nest(Mat A)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* release the matrices and the place holders */
  ierr = MatNestDestroyISList(vs->nr,&vs->isglobal.row);CHKERRQ(ierr);
  ierr = MatNestDestroyISList(vs->nc,&vs->isglobal.col);CHKERRQ(ierr);
  ierr = MatNestDestroyISList(vs->nr,&vs->islocal.row);CHKERRQ(ierr);
  ierr = MatNestDestroyISList(vs->nc,&vs->islocal.col);CHKERRQ(ierr);

  ierr = PetscFree(vs->row_len);CHKERRQ(ierr);
  ierr = PetscFree(vs->col_len);CHKERRQ(ierr);

  ierr = PetscFree2(vs->left,vs->right);CHKERRQ(ierr);

  /* release the matrices and the place holders */
  if (vs->m) {
    for (i=0; i<vs->nr; i++) {
      for (j=0; j<vs->nc; j++) {
        ierr = MatDestroy(&vs->m[i][j]);CHKERRQ(ierr);
      }
      ierr = PetscFree( vs->m[i] );CHKERRQ(ierr);
    }
    ierr = PetscFree(vs->m);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestGetSubMat_C",   "",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestGetSubMats_C",  "",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestGetSize_C",     "",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestSetVecType_C",  "",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestSetSubMats_C",   "",0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_Nest"
static PetscErrorCode MatAssemblyBegin_Nest(Mat A,MatAssemblyType type)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<vs->nr; i++) {
    for (j=0; j<vs->nc; j++) {
      if (vs->m[i][j]) { ierr = MatAssemblyBegin(vs->m[i][j],type);CHKERRQ(ierr); }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_Nest"
static PetscErrorCode MatAssemblyEnd_Nest(Mat A, MatAssemblyType type)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<vs->nr; i++) {
    for (j=0; j<vs->nc; j++) {
      if (vs->m[i][j]) { ierr = MatAssemblyEnd(vs->m[i][j],type);CHKERRQ(ierr); }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNestFindNonzeroSubMatRow"
static PetscErrorCode MatNestFindNonzeroSubMatRow(Mat A,PetscInt row,Mat *B)
{
  PetscErrorCode ierr;
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       j;
  Mat            sub;

  PetscFunctionBegin;
  sub = (row < vs->nc) ? vs->m[row][row] : PETSC_NULL; /* Prefer to find on the diagonal */
  for (j=0; !sub && j<vs->nc; j++) sub = vs->m[row][j];
  if (sub) {ierr = MatPreallocated(sub);CHKERRQ(ierr);} /* Ensure that the sizes are available */
  *B = sub;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNestFindNonzeroSubMatCol"
static PetscErrorCode MatNestFindNonzeroSubMatCol(Mat A,PetscInt col,Mat *B)
{
  PetscErrorCode ierr;
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i;
  Mat            sub;

  PetscFunctionBegin;
  sub = (col < vs->nr) ? vs->m[col][col] : PETSC_NULL; /* Prefer to find on the diagonal */
  for (i=0; !sub && i<vs->nr; i++) sub = vs->m[i][col];
  if (sub) {ierr = MatPreallocated(sub);CHKERRQ(ierr);} /* Ensure that the sizes are available */
  *B = sub;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNestFindIS"
static PetscErrorCode MatNestFindIS(Mat A,PetscInt n,const IS list[],IS is,PetscInt *found)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidPointer(list,3);
  PetscValidHeaderSpecific(is,IS_CLASSID,4);
  PetscValidIntPointer(found,5);
  *found = -1;
  for (i=0; i<n; i++) {
    if (!list[i]) continue;
    ierr = ISEqual(list[i],is,&flg);CHKERRQ(ierr);
    if (flg) {
      *found = i;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Could not find index set");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNestGetRow"
/* Get a block row as a new MatNest */
static PetscErrorCode MatNestGetRow(Mat A,PetscInt row,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            C;
  char           keyname[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *B = PETSC_NULL;
  ierr = PetscSNPrintf(keyname,sizeof keyname,"NestRow_%D",row);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,keyname,(PetscObject*)B);CHKERRQ(ierr);
  if (*B) PetscFunctionReturn(0);

  ierr = MatCreateNest(((PetscObject)A)->comm,1,PETSC_NULL,vs->nc,vs->isglobal.col,vs->m[row],B);CHKERRQ(ierr);
  (*B)->assembled = A->assembled;
  ierr = PetscObjectCompose((PetscObject)A,keyname,(PetscObject)*B);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)*B);CHKERRQ(ierr); /* Leave the only remaining reference in the composition */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNestFindSubMat"
static PetscErrorCode MatNestFindSubMat(Mat A,struct MatNestISPair *is,IS isrow,IS iscol,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,col,i;
  IS             *iscopy;
  Mat            *matcopy;
  PetscBool      same,isFullCol;

  PetscFunctionBegin;
  /* Check if full column space. This is a hack */
  isFullCol = PETSC_FALSE;
  ierr = PetscTypeCompare((PetscObject)iscol,ISSTRIDE,&same);CHKERRQ(ierr);
  if (same) {
    PetscInt n,first,step;
    ierr = ISStrideGetInfo(iscol,&first,&step);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iscol,&n);CHKERRQ(ierr);
    if (A->cmap->n == n && A->cmap->rstart == first && step == 1) isFullCol = PETSC_TRUE;
  }

  if (isFullCol) {
    PetscInt row;
    ierr = MatNestFindIS(A,vs->nr,is->row,isrow,&row);CHKERRQ(ierr);
    ierr = MatNestGetRow(A,row,B);CHKERRQ(ierr);
  } else {
    ierr = MatNestFindIS(A,vs->nr,is->row,isrow,&row);CHKERRQ(ierr);
    ierr = MatNestFindIS(A,vs->nc,is->col,iscol,&col);CHKERRQ(ierr);
    *B = vs->m[row][col];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_Nest"
static PetscErrorCode MatGetSubMatrix_Nest(Mat A,IS isrow,IS iscol,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            sub;

  PetscFunctionBegin;
  ierr = MatNestFindSubMat(A,&vs->isglobal,isrow,iscol,&sub);CHKERRQ(ierr);
  switch (reuse) {
  case MAT_INITIAL_MATRIX:
    if (sub) { ierr = PetscObjectReference((PetscObject)sub);CHKERRQ(ierr); }
    *B = sub;
    break;
  case MAT_REUSE_MATRIX:
    if (sub != *B) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Submatrix was not used before in this call");
    break;
  case MAT_IGNORE_MATRIX:       /* Nothing to do */
    break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetLocalSubMatrix_Nest"
PetscErrorCode MatGetLocalSubMatrix_Nest(Mat A,IS isrow,IS iscol,Mat *B)
{
  PetscErrorCode ierr;
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            sub;

  PetscFunctionBegin;
  ierr = MatNestFindSubMat(A,&vs->islocal,isrow,iscol,&sub);CHKERRQ(ierr);
  /* We allow the submatrix to be NULL, perhaps it would be better for the user to return an empty matrix instead */
  if (sub) {ierr = PetscObjectReference((PetscObject)sub);CHKERRQ(ierr);}
  *B = sub;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreLocalSubMatrix_Nest"
static PetscErrorCode MatRestoreLocalSubMatrix_Nest(Mat A,IS isrow,IS iscol,Mat *B)
{
  PetscErrorCode ierr;
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            sub;

  PetscFunctionBegin;
  ierr = MatNestFindSubMat(A,&vs->islocal,isrow,iscol,&sub);CHKERRQ(ierr);
  if (*B != sub) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Local submatrix has not been gotten");
  if (sub) {
    if (((PetscObject)sub)->refct <= 1) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Local submatrix has had reference count decremented too many times");
    ierr = MatDestroy(B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_Nest"
static PetscErrorCode MatGetDiagonal_Nest(Mat A,Vec v)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bdiag;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  ierr = MatGetVecs_Nest(A,&diag,PETSC_NULL);CHKERRQ(ierr); */
  /*  ierr = VecNestGetSubVecs(diag,PETSC_NULL,&bdiag);CHKERRQ(ierr); */
  ierr = VecNestGetSubVecs(v,PETSC_NULL,&bdiag);CHKERRQ(ierr);
  for (i=0; i<bA->nr; i++) {
    if (bA->m[i][i]) {
      ierr = MatGetDiagonal(bA->m[i][i],bdiag[i]);CHKERRQ(ierr);
    } else {
      ierr = VecSet(bdiag[i],1.0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_Nest2"
static PetscErrorCode MatGetDiagonal_Nest2(Mat A,Vec v)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            diag,*bdiag;
  VecScatter     *vscat;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetVecs_Nest(A,&diag,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetDiagonal_Nest(A,diag);CHKERRQ(ierr);

  /* scatter diag into v */
  ierr = VecNestGetSubVecs(diag,PETSC_NULL,&bdiag);CHKERRQ(ierr);
  ierr = PetscMalloc( sizeof(VecScatter) * bA->nr, &vscat );CHKERRQ(ierr);
  for (i=0; i<bA->nr; i++) {
    ierr = VecScatterCreate(v,bA->isglobal.row[i], bdiag[i],PETSC_NULL,&vscat[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscat[i],bdiag[i],v,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  for (i=0; i<bA->nr; i++) {
    ierr = VecScatterEnd(vscat[i],bdiag[i],v,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  for (i=0; i<bA->nr; i++) {
    ierr = VecScatterDestroy(&vscat[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(vscat);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_Nest"
static PetscErrorCode MatDiagonalScale_Nest(Mat A,Vec l,Vec r)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bl,*br;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVecs(l,PETSC_NULL,&bl);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(r,PETSC_NULL,&br);CHKERRQ(ierr);
  for (i=0; i<bA->nr; i++) {
    for (j=0; j<bA->nc; j++) {
      if (bA->m[i][j]) {
        ierr = MatDiagonalScale(bA->m[i][j],bl[i],br[j]);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_Nest2"
static PetscErrorCode MatDiagonalScale_Nest2(Mat A,Vec l,Vec r)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            bl,br,*ble,*bre;
  VecScatter     *vscatl,*vscatr;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* scatter l,r into bl,br */
  ierr = MatGetVecs_Nest(A,&bl,&br);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(bl,PETSC_NULL,&ble);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(br,PETSC_NULL,&bre);CHKERRQ(ierr);

  /* row */
  ierr = PetscMalloc( sizeof(VecScatter) * bA->nr, &vscatl );CHKERRQ(ierr);
  for (i=0; i<bA->nr; i++) {
    ierr = VecScatterCreate(l,bA->isglobal.row[i], ble[i],PETSC_NULL,&vscatl[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscatl[i],l,ble[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  for (i=0; i<bA->nr; i++) {
    ierr = VecScatterEnd(vscatl[i],l,ble[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  /* col */
  ierr = PetscMalloc( sizeof(VecScatter) * bA->nc, &vscatr );CHKERRQ(ierr);
  for (i=0; i<bA->nc; i++) {
    ierr = VecScatterCreate(r,bA->isglobal.col[i], bre[i],PETSC_NULL,&vscatr[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscatr[i],l,bre[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  for (i=0; i<bA->nc; i++) {
    ierr = VecScatterEnd(vscatr[i],r,bre[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  ierr = MatDiagonalScale_Nest(A,bl,br);CHKERRQ(ierr);

  for (i=0; i<bA->nr; i++) {
    ierr = VecScatterDestroy(&vscatl[i]);CHKERRQ(ierr);
  }
  for (i=0; i<bA->nc; i++) {
    ierr = VecScatterDestroy(&vscatr[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(vscatl);CHKERRQ(ierr);
  ierr = PetscFree(vscatr);CHKERRQ(ierr);
  ierr = VecDestroy(&bl);CHKERRQ(ierr);
  ierr = VecDestroy(&br);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetVecs_Nest"
static PetscErrorCode MatGetVecs_Nest(Mat A,Vec *right,Vec *left)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *L,*R;
  MPI_Comm       comm;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = ((PetscObject)A)->comm;
  if (right) {
    /* allocate R */
    ierr = PetscMalloc( sizeof(Vec) * bA->nc, &R );CHKERRQ(ierr);
    /* Create the right vectors */
    for (j=0; j<bA->nc; j++) {
      for (i=0; i<bA->nr; i++) {
        if (bA->m[i][j]) {
          ierr = MatGetVecs(bA->m[i][j],&R[j],PETSC_NULL);CHKERRQ(ierr);
          break;
        }
      }
      if (i==bA->nr) {
        /* have an empty column */
        SETERRQ( ((PetscObject)A)->comm, PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null column.");
      }
    }
    ierr = VecCreateNest(comm,bA->nc,bA->isglobal.col,R,right);CHKERRQ(ierr);
    /* hand back control to the nest vector */
    for (j=0; j<bA->nc; j++) {
      ierr = VecDestroy(&R[j]);CHKERRQ(ierr);
    }
    ierr = PetscFree(R);CHKERRQ(ierr);
  }

  if (left) {
    /* allocate L */
    ierr = PetscMalloc( sizeof(Vec) * bA->nr, &L );CHKERRQ(ierr);
    /* Create the left vectors */
    for (i=0; i<bA->nr; i++) {
      for (j=0; j<bA->nc; j++) {
        if (bA->m[i][j]) {
          ierr = MatGetVecs(bA->m[i][j],PETSC_NULL,&L[i]);CHKERRQ(ierr);
          break;
        }
      }
      if (j==bA->nc) {
        /* have an empty row */
        SETERRQ( ((PetscObject)A)->comm, PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null row.");
      }
    }

    ierr = VecCreateNest(comm,bA->nr,bA->isglobal.row,L,left);CHKERRQ(ierr);
    for (i=0; i<bA->nr; i++) {
      ierr = VecDestroy(&L[i]);CHKERRQ(ierr);
    }

    ierr = PetscFree(L);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Nest"
static PetscErrorCode MatView_Nest(Mat A,PetscViewer viewer)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscBool      isascii;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {

    PetscViewerASCIIPrintf(viewer,"Matrix object: \n" );
    PetscViewerASCIIPushTab( viewer );    /* push0 */
    PetscViewerASCIIPrintf( viewer, "type=nest, rows=%d, cols=%d \n",bA->nr,bA->nc);

    PetscViewerASCIIPrintf(viewer,"MatNest structure: \n" );
    for (i=0; i<bA->nr; i++) {
      for (j=0; j<bA->nc; j++) {
        const MatType type;
        const char *name;
        PetscInt NR,NC;
        PetscBool isNest = PETSC_FALSE;

        if (!bA->m[i][j]) {
          PetscViewerASCIIPrintf( viewer, "(%D,%D) : PETSC_NULL \n",i,j);
          continue;
        }
        ierr = MatGetSize(bA->m[i][j],&NR,&NC);CHKERRQ(ierr);
        ierr = MatGetType( bA->m[i][j], &type );CHKERRQ(ierr);
        name = ((PetscObject)bA->m[i][j])->prefix;
        ierr = PetscTypeCompare((PetscObject)bA->m[i][j],MATNEST,&isNest);CHKERRQ(ierr);

        PetscViewerASCIIPrintf( viewer, "(%D,%D) : name=\"%s\", type=%s, rows=%D, cols=%D \n",i,j,name,type,NR,NC);

        if (isNest) {
          PetscViewerASCIIPushTab( viewer );  /* push1 */
          ierr = MatView( bA->m[i][j], viewer );CHKERRQ(ierr);
          PetscViewerASCIIPopTab(viewer);    /* pop1 */
        }
      }
    }
    PetscViewerASCIIPopTab(viewer);    /* pop0 */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_Nest"
static PetscErrorCode MatZeroEntries_Nest(Mat A)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    for (j=0; j<bA->nc; j++) {
      if (!bA->m[i][j]) continue;
      ierr = MatZeroEntries(bA->m[i][j]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_Nest"
static PetscErrorCode MatDuplicate_Nest(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Mat            *b;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(nr*nc*sizeof(Mat),&b);CHKERRQ(ierr);
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (bA->m[i][j]) {
        ierr = MatDuplicate(bA->m[i][j],op,&b[i*nc+j]);CHKERRQ(ierr);
      } else {
        b[i*nc+j] = PETSC_NULL;
      }
    }
  }
  ierr = MatCreateNest(((PetscObject)A)->comm,nr,bA->isglobal.row,nc,bA->isglobal.col,b,B);CHKERRQ(ierr);
  /* Give the new MatNest exclusive ownership */
  for (i=0; i<nr*nc; i++) {
    ierr = MatDestroy(&b[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* nest api */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatNestGetSubMat_Nest"
PetscErrorCode MatNestGetSubMat_Nest(Mat A,PetscInt idxm,PetscInt jdxm,Mat *mat)
{
  Mat_Nest *bA = (Mat_Nest*)A->data;
  PetscFunctionBegin;
  if (idxm >= bA->nr) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",idxm,bA->nr-1);
  if (jdxm >= bA->nc) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Col too large: row %D max %D",jdxm,bA->nc-1);
  *mat = bA->m[idxm][jdxm];
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatNestGetSubMat"
/*@C
 MatNestGetSubMat - Returns a single, sub-matrix from a nest matrix.

 Not collective

 Input Parameters:
 .  A  - nest matrix
 .  idxm - index of the matrix within the nest matrix
 .  jdxm - index of the matrix within the nest matrix

 Output Parameter:
 .  sub - matrix at index idxm,jdxm within the nest matrix

 Notes:

 Level: developer

 .seealso: MatNestGetSize(), MatNestGetSubMats()
@*/
PetscErrorCode  MatNestGetSubMat(Mat A,PetscInt idxm,PetscInt jdxm,Mat *sub)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatNestGetSubMat_C",(Mat,PetscInt,PetscInt,Mat*),(A,idxm,jdxm,sub));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatNestGetSubMats_Nest"
PetscErrorCode MatNestGetSubMats_Nest(Mat A,PetscInt *M,PetscInt *N,Mat ***mat)
{
  Mat_Nest *bA = (Mat_Nest*)A->data;
  PetscFunctionBegin;
  if (M)   { *M   = bA->nr; }
  if (N)   { *N   = bA->nc; }
  if (mat) { *mat = bA->m;  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatNestGetSubMats"
/*@C
 MatNestGetSubMats - Returns the entire two dimensional array of matrices defining a nest matrix.

 Not collective

 Input Parameters:
 .  A  - nest matri

 Output Parameter:
 .  M - number of rows in the nest matrix
 .  N - number of cols in the nest matrix
 .  mat - 2d array of matrices

 Notes:

 The user should not free the array mat.

 Level: developer

 .seealso: MatNestGetSize(), MatNestGetSubMat()
@*/
PetscErrorCode  MatNestGetSubMats(Mat A,PetscInt *M,PetscInt *N,Mat ***mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatNestGetSubMats_C",(Mat,PetscInt*,PetscInt*,Mat***),(A,M,N,mat));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatNestGetSize_Nest"
PetscErrorCode  MatNestGetSize_Nest(Mat A,PetscInt *M,PetscInt *N)
{
  Mat_Nest  *bA = (Mat_Nest*)A->data;

  PetscFunctionBegin;
  if (M) { *M  = bA->nr; }
  if (N) { *N  = bA->nc; }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatNestGetSize"
/*@C
 MatNestGetSize - Returns the size of the nest matrix.

 Not collective

 Input Parameters:
 .  A  - nest matrix

 Output Parameter:
 .  M - number of rows in the nested mat
 .  N - number of cols in the nested mat

 Notes:

 Level: developer

 .seealso: MatNestGetSubMat(), MatNestGetSubMats()
@*/
PetscErrorCode  MatNestGetSize(Mat A,PetscInt *M,PetscInt *N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatNestGetSize_C",(Mat,PetscInt*,PetscInt*),(A,M,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatNestSetVecType_Nest"
PetscErrorCode  MatNestSetVecType_Nest(Mat A,const VecType vtype)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscStrcmp(vtype,VECNEST,&flg);CHKERRQ(ierr);
  /* In reality, this only distinguishes VECNEST and "other" */
  A->ops->getvecs       = flg ? MatGetVecs_Nest       : 0;
  A->ops->getdiagonal   = flg ? MatGetDiagonal_Nest   : 0;
  A->ops->diagonalscale = flg ? MatDiagonalScale_Nest : 0;
 
 PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatNestSetVecType"
/*@C
 MatNestSetVecType - Sets the type of Vec returned by MatGetVecs()

 Not collective

 Input Parameters:
+  A  - nest matrix
-  vtype - type to use for creating vectors

 Notes:

 Level: developer

 .seealso: MatGetVecs()
@*/
PetscErrorCode  MatNestSetVecType(Mat A,const VecType vtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"MatNestSetVecType_C",(Mat,const VecType),(A,vtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatNestSetSubMats_Nest"
PetscErrorCode MatNestSetSubMats_Nest(Mat A,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[])
{
  Mat_Nest       *s = (Mat_Nest*)A->data;
  PetscInt       i,j,m,n,M,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  s->nr = nr;
  s->nc = nc;

  /* Create space for submatrices */
  ierr = PetscMalloc(sizeof(Mat*)*nr,&s->m);CHKERRQ(ierr);
  for (i=0; i<nr; i++) {
    ierr = PetscMalloc(sizeof(Mat)*nc,&s->m[i]);CHKERRQ(ierr);
  }
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      s->m[i][j] = a[i*nc+j];
      if (a[i*nc+j]) {
        ierr = PetscObjectReference((PetscObject)a[i*nc+j]);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatSetUp_NestIS_Private(A,nr,is_row,nc,is_col);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(PetscInt)*nr,&s->row_len);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nc,&s->col_len);CHKERRQ(ierr);
  for (i=0; i<nr; i++) s->row_len[i]=-1;
  for (j=0; j<nc; j++) s->col_len[j]=-1;

  ierr = MatNestGetSizes_Private(A,&m,&n,&M,&N);CHKERRQ(ierr);

  ierr = PetscLayoutSetSize(A->rmap,M);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(A->rmap,m);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(A->cmap,N);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(A->cmap,n);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);

  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  ierr = PetscMalloc2(nr,Vec,&s->left,nc,Vec,&s->right);CHKERRQ(ierr);
  ierr = PetscMemzero(s->left,nr*sizeof(Vec));CHKERRQ(ierr);
  ierr = PetscMemzero(s->right,nc*sizeof(Vec));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatNestSetSubMats"
/*@
   MatNestSetSubMats - Sets the nested submatrices

   Collective on Mat

   Input Parameter:
+  N - nested matrix
.  nr - number of nested row blocks
.  is_row - index sets for each nested row block, or PETSC_NULL to make contiguous
.  nc - number of nested column blocks
.  is_col - index sets for each nested column block, or PETSC_NULL to make contiguous
-  a - row-aligned array of nr*nc submatrices, empty submatrices can be passed using PETSC_NULL

   Level: advanced

.seealso: MatCreateNest(), MATNEST
@*/
PetscErrorCode MatNestSetSubMats(Mat A,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[])
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (nr < 0) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of rows cannot be negative");
  if (nr && is_row) {
    PetscValidPointer(is_row,3);
    for (i=0; i<nr; i++) PetscValidHeaderSpecific(is_row[i],IS_CLASSID,3);
  }
  if (nc < 0) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of columns cannot be negative");
  if (nc && is_row) {
    PetscValidPointer(is_col,5);
    for (i=0; i<nr; i++) PetscValidHeaderSpecific(is_col[i],IS_CLASSID,5);
  }
  if (nr*nc) PetscValidPointer(a,6);
  ierr = PetscUseMethod(A,"MatNestSetSubMats_C",(Mat,PetscInt,const IS[],PetscInt,const IS[],const Mat[]),(A,nr,is_row,nc,is_col,a));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* If an IS was provided, there is nothing Nest needs to do, otherwise Nest will build a strided IS */
/*
  nprocessors = NP
  Nest x^T = ( (g_0,g_1,...g_nprocs-1), (h_0,h_1,...h_NP-1) )
       proc 0: => (g_0,h_0,)
       proc 1: => (g_1,h_1,)
       ...
       proc nprocs-1: => (g_NP-1,h_NP-1,)

            proc 0:                      proc 1:                    proc nprocs-1:
    is[0] = ( 0,1,2,...,nlocal(g_0)-1 )  ( 0,1,...,nlocal(g_1)-1 )  ( 0,1,...,nlocal(g_NP-1) )

            proc 0:
    is[1] = ( nlocal(g_0),nlocal(g_0)+1,...,nlocal(g_0)+nlocal(h_0)-1 )
            proc 1:
    is[1] = ( nlocal(g_1),nlocal(g_1)+1,...,nlocal(g_1)+nlocal(h_1)-1 )

            proc NP-1:
    is[1] = ( nlocal(g_NP-1),nlocal(g_NP-1)+1,...,nlocal(g_NP-1)+nlocal(h_NP-1)-1 )
*/
#undef __FUNCT__  
#define __FUNCT__ "MatSetUp_NestIS_Private"
static PetscErrorCode MatSetUp_NestIS_Private(Mat A,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[])
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j,offset,n,nsum,bs;
  PetscErrorCode ierr;
  Mat            sub;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(IS)*nr,&vs->isglobal.row);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(IS)*nc,&vs->isglobal.col);CHKERRQ(ierr);
  if (is_row) { /* valid IS is passed in */
    /* refs on is[] are incremeneted */
    for (i=0; i<vs->nr; i++) {
      ierr = PetscObjectReference((PetscObject)is_row[i]);CHKERRQ(ierr);
      vs->isglobal.row[i] = is_row[i];
    }
  } else {                      /* Create the ISs by inspecting sizes of a submatrix in each row */
    nsum = 0;
    for (i=0; i<vs->nr; i++) {  /* Add up the local sizes to compute the aggregate offset */
      ierr = MatNestFindNonzeroSubMatRow(A,i,&sub);CHKERRQ(ierr);
      if (!sub) SETERRQ1(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"No nonzero submatrix in row %D",i);
      ierr = MatGetLocalSize(sub,&n,PETSC_NULL);CHKERRQ(ierr);
      if (n < 0) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Sizes have not yet been set for submatrix");
      nsum += n;
    }
    offset = 0;
    ierr = MPI_Exscan(&nsum,&offset,1,MPIU_INT,MPI_SUM,((PetscObject)A)->comm);CHKERRQ(ierr);
    for (i=0; i<vs->nr; i++) {
      ierr = MatNestFindNonzeroSubMatRow(A,i,&sub);CHKERRQ(ierr);
      ierr = MatGetLocalSize(sub,&n,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetBlockSize(sub,&bs);CHKERRQ(ierr);
      ierr = ISCreateStride(((PetscObject)sub)->comm,n,offset,1,&vs->isglobal.row[i]);CHKERRQ(ierr);
      ierr = ISSetBlockSize(vs->isglobal.row[i],bs);CHKERRQ(ierr);
      offset += n;
    }
  }

  if (is_col) { /* valid IS is passed in */
    /* refs on is[] are incremeneted */
    for (j=0; j<vs->nc; j++) {
      ierr = PetscObjectReference((PetscObject)is_col[j]);CHKERRQ(ierr);
      vs->isglobal.col[j] = is_col[j];
    }
  } else {                      /* Create the ISs by inspecting sizes of a submatrix in each column */
    offset = A->cmap->rstart;
    nsum = 0;
    for (j=0; j<vs->nc; j++) {
      ierr = MatNestFindNonzeroSubMatCol(A,j,&sub);CHKERRQ(ierr);
      if (!sub) SETERRQ1(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"No nonzero submatrix in column %D",i);
      ierr = MatGetLocalSize(sub,PETSC_NULL,&n);CHKERRQ(ierr);
      if (n < 0) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONGSTATE,"Sizes have not yet been set for submatrix");
      nsum += n;
    }
    offset = 0;
    ierr = MPI_Exscan(&nsum,&offset,1,MPIU_INT,MPI_SUM,((PetscObject)A)->comm);CHKERRQ(ierr);
    for (j=0; j<vs->nc; j++) {
      ierr = MatNestFindNonzeroSubMatCol(A,j,&sub);CHKERRQ(ierr);
      ierr = MatGetLocalSize(sub,PETSC_NULL,&n);CHKERRQ(ierr);
      ierr = MatGetBlockSize(sub,&bs);CHKERRQ(ierr);
      ierr = ISCreateStride(((PetscObject)sub)->comm,n,offset,1,&vs->isglobal.col[j]);CHKERRQ(ierr);
      ierr = ISSetBlockSize(vs->isglobal.col[j],bs);CHKERRQ(ierr);
      offset += n;
    }
  }

  /* Set up the local ISs */
  ierr = PetscMalloc(vs->nr*sizeof(IS),&vs->islocal.row);CHKERRQ(ierr);
  ierr = PetscMalloc(vs->nc*sizeof(IS),&vs->islocal.col);CHKERRQ(ierr);
  for (i=0,offset=0; i<vs->nr; i++) {
    IS                     isloc;
    ISLocalToGlobalMapping rmap = PETSC_NULL;
    PetscInt               nlocal,bs;
    ierr = MatNestFindNonzeroSubMatRow(A,i,&sub);CHKERRQ(ierr);
    if (sub) {ierr = MatGetLocalToGlobalMapping(sub,&rmap,PETSC_NULL);CHKERRQ(ierr);}
    if (rmap) {
      ierr = MatGetBlockSize(sub,&bs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetSize(rmap,&nlocal);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,nlocal,offset,1,&isloc);CHKERRQ(ierr);
      ierr = ISSetBlockSize(isloc,bs);CHKERRQ(ierr);
    } else {
      nlocal = 0;
      isloc  = PETSC_NULL;
    }
    vs->islocal.row[i] = isloc;
    offset += nlocal;
  }
  for (i=0,offset=0; i<vs->nc; i++) {
    IS                     isloc;
    ISLocalToGlobalMapping cmap = PETSC_NULL;
    PetscInt               nlocal,bs;
    ierr = MatNestFindNonzeroSubMatCol(A,i,&sub);CHKERRQ(ierr);
    if (sub) {ierr = MatGetLocalToGlobalMapping(sub,PETSC_NULL,&cmap);CHKERRQ(ierr);}
    if (cmap) {
      ierr = MatGetBlockSize(sub,&bs);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetSize(cmap,&nlocal);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,nlocal,offset,1,&isloc);CHKERRQ(ierr);
      ierr = ISSetBlockSize(isloc,bs);CHKERRQ(ierr);
    } else {
      nlocal = 0;
      isloc  = PETSC_NULL;
    }
    vs->islocal.col[i] = isloc;
    offset += nlocal;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateNest"
/*@
   MatCreateNest - Creates a new matrix containing several nested submatrices, each stored separately

   Collective on Mat

   Input Parameter:
+  comm - Communicator for the new Mat
.  nr - number of nested row blocks
.  is_row - index sets for each nested row block, or PETSC_NULL to make contiguous
.  nc - number of nested column blocks
.  is_col - index sets for each nested column block, or PETSC_NULL to make contiguous
-  a - row-aligned array of nr*nc submatrices, empty submatrices can be passed using PETSC_NULL

   Output Parameter:
.  B - new matrix

   Level: advanced

.seealso: MatCreate(), VecCreateNest(), DMGetMatrix(), MATNEST
@*/
PetscErrorCode MatCreateNest(MPI_Comm comm,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[],Mat *B)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *B = 0;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATNEST);CHKERRQ(ierr);
  ierr = MatNestSetSubMats(A,nr,is_row,nc,is_col,a);CHKERRQ(ierr);
  *B = A;
  PetscFunctionReturn(0);
}

/*MC
  MATNEST - MATNEST = "nest" - Matrix type consisting of nested submatrices, each stored separately.

  Level: intermediate

  Notes:
  This matrix type permits scalable use of PCFieldSplit and avoids the large memory costs of extracting submatrices.
  It allows the use of symmetric and block formats for parts of multi-physics simulations.
  It is usually used with DMComposite and DMGetMatrix()

.seealso: MatCreate(), MatType, MatCreateNest()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Nest"
PetscErrorCode MatCreate_Nest(Mat A)
{
  Mat_Nest       *s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,Mat_Nest,&s);CHKERRQ(ierr);
  A->data         = (void*)s;
  s->nr = s->nc   = -1;
  s->m            = PETSC_NULL;

  ierr = PetscMemzero(A->ops,sizeof(*A->ops));CHKERRQ(ierr);
  A->ops->mult                  = MatMult_Nest;
  A->ops->multadd               = 0;
  A->ops->multtranspose         = MatMultTranspose_Nest;
  A->ops->multtransposeadd      = 0;
  A->ops->assemblybegin         = MatAssemblyBegin_Nest;
  A->ops->assemblyend           = MatAssemblyEnd_Nest;
  A->ops->zeroentries           = MatZeroEntries_Nest;
  A->ops->duplicate             = MatDuplicate_Nest;
  A->ops->getsubmatrix          = MatGetSubMatrix_Nest;
  A->ops->destroy               = MatDestroy_Nest;
  A->ops->view                  = MatView_Nest;
  A->ops->getvecs               = 0; /* Use VECNEST by calling MatNestSetVecType(A,VECNEST) */
  A->ops->getlocalsubmatrix     = MatGetLocalSubMatrix_Nest;
  A->ops->restorelocalsubmatrix = MatRestoreLocalSubMatrix_Nest;
  A->ops->getdiagonal           = MatGetDiagonal_Nest2; /* VECNEST version activated by calling MatNestSetVecType(A,VECNEST) */
  A->ops->diagonalscale         = MatDiagonalScale_Nest2; /* VECNEST version activated by calling MatNestSetVecType(A,VECNEST) */

  A->spptr        = 0;
  A->same_nonzero = PETSC_FALSE;
  A->assembled    = PETSC_FALSE;

  /* expose Nest api's */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestGetSubMat_C",   "MatNestGetSubMat_Nest",   MatNestGetSubMat_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestGetSubMats_C",  "MatNestGetSubMats_Nest",  MatNestGetSubMats_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestGetSize_C",     "MatNestGetSize_Nest",     MatNestGetSize_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestSetVecType_C",  "MatNestSetVecType_Nest",  MatNestSetVecType_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatNestSetSubMats_C",  "MatNestSetSubMats_Nest",  MatNestSetSubMats_Nest);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATNEST);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
