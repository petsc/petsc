#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dense.c,v 1.4 2000/01/10 03:13:10 knepley Exp $";
#endif
/*
     Defines the basic bilinear operations for sequential dense.
*/

#include "src/bilinear/impls/dense/seq/bldense.h"
#include "pinclude/pviewer.h"

/*--------------------------------------------- Basic Functions -----------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearView_SeqDense_ASCII"
static int BilinearView_SeqDense_ASCII(Bilinear B, Viewer viewer)
{
  FILE   *fd;
  int     format;
  Scalar *array;
  int     n_i, n_j, n_k;
  int     i, j, k;
  int     ierr;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer, &fd);                                                             CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer, &format);                                                               CHKERRQ(ierr);
  ierr = BilinearGetSize(B, &n_i, &n_j, &n_k);                                                           CHKERRQ(ierr);
  ierr = BilinearGetArray(B, &array);                                                                    CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    PetscFunctionReturn(0);  /* do nothing for now */
  } else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    ierr = BilinearGetArray(B, &array);                                                                  CHKERRQ(ierr);
    /* Output the n_i rows of B */
    for(i = 0; i < n_i; i++) {
      PetscFPrintf(B->comm, fd, "row %d:\n", i);
      /* Output the n_j vectors of row i */
      for(j = 0; j < n_j; j++) {
        PetscFPrintf(B->comm, fd, "col %d: ", j);
        /* Output vectors of length n_k */
        for(k = 0; k < n_k; k++)
          PetscFPrintf(B->comm, fd, "%d %g ", k, array[i*n_j*n_k+j*n_k+k]);
        PetscFPrintf(B->comm, fd, " ");
      }
      PetscFPrintf(B->comm, fd, "\n");
    }
    ierr = BilinearRestoreArray(B, &array);                                                              CHKERRQ(ierr);
  } else {
    ierr = BilinearGetArray(B, &array);                                                                  CHKERRQ(ierr);
    /* Output the n_i rows of B */
    for(i = 0; i < n_i; i++) {
      /* Output the n_j vectors of row i */
      for(j = 0; j < n_j; j++) {
        /* Output vectors of length n_k */
        for(k = 0; k < n_k; k++)
          PetscFPrintf(B->comm, fd, "%6.4g ", array[i*n_j*n_k+j*n_k+k]);
        PetscFPrintf(B->comm, fd, " ");
      }
      PetscFPrintf(B->comm, fd, "\n");
    }
    ierr = BilinearRestoreArray(B, &array);                                                              CHKERRQ(ierr);
  }
  fflush(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearView_SeqDense_Draw"
int BilinearView_SeqDense_Draw(Bilinear B, Viewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearView_SeqDense_Mathematica"
int BilinearView_SeqDense_Mathematica(Bilinear B, Viewer viewer)
{
#ifdef HAVE_MATHEMATICA
  MLINK   link;
  char   *name;
  Scalar *array;
  int     n_i, n_j, n_k;
  int     i, j, k;
  int     ierr;

  PetscFunctionBegin;
  ierr = ViewerMathematicaGetLink(viewer, &link);                                                        CHKERRQ(ierr);
  ierr = BilinearGetSize(B, &n_i, &n_j, &n_k);                                                           CHKERRQ(ierr);
  ierr = BilinearGetArray(B, &array);                                                                    CHKERRQ(ierr);
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      ierr = ViewerMathematicaGetName(viewer, &name);                                                    CHKERRQ(ierr);
      MLPutSymbol(link, name);
      MLPutFunction(link, "List", n_i);
      for(i = 0; i < n_i; i++) {
        MLPutFunction(link, "List", n_j);
        for(j = 0; j < n_j; j++) {
          MLPutFunction(link, "List", n_k);
          for(k = 0; k < n_k; k++) {
            MLPutReal(link, array[i*n_j*n_k+j*n_k+k]);
          }
        }
      }
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = ViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNC__  
#define __FUNC__ "BilinearView_SeqDense"
int BilinearView_SeqDense(Bilinear B, Viewer viewer)
{
  PetscTruth isascii, isdraw, ismathematica;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, ASCII_VIEWER,       &isascii);                           CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, DRAW_VIEWER,        &isdraw);                            CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, MATHEMATICA_VIEWER, &ismathematica);                     CHKERRQ(ierr);
  if (isascii) {
    ierr = BilinearView_SeqDense_ASCII(B, viewer);                                                       CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = BilinearView_SeqDense_Draw(B, viewer);                                                        CHKERRQ(ierr);
  } else if (ismathematica) {
    ierr = BilinearView_SeqDense_Mathematica(B, viewer);                                                 CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP, 0, "Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearDestroy_SeqDense"
int BilinearDestroy_SeqDense(Bilinear B)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;
  int                ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject) B, "Rows %d Cols %d SubCols: %d", b->n_i, b->n_j, b->n_k);
#endif
  if (b->pivots) {
    ierr = PetscFree(b->pivots);                                                                         CHKERRQ(ierr);
  }
  if (b->user_alloc != PETSC_TRUE) {
    ierr = PetscFree(b->v);                                                                              CHKERRQ(ierr);
  }
  ierr = PetscFree(b);                                                                                   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearConvertSameType_SeqDense"
int BilinearConvertSameType_SeqDense(Bilinear B, Bilinear *newB, int cpvalues)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data, *newb;
  Bilinear           newOp;
  int                ierr;

  PetscFunctionBegin;
  ierr = BilinearCreateSeqDense(B->comm, B->N_i, B->N_j, B->N_k, PETSC_NULL, &newOp);                    CHKERRQ(ierr);
  newb = (Bilinear_SeqDense *) newOp->data;
  if (cpvalues == COPY_VALUES) {
    ierr = PetscMemcpy(newb->v, b->v, b->n_i*b->n_j*b->n_k * sizeof(Scalar));                            CHKERRQ(ierr);
  }
  newOp->assembled = PETSC_TRUE;
  *newB = newOp;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearCopy_SeqDense"
int BilinearCopy_SeqDense(Bilinear A, Bilinear B)
{
  Bilinear_SeqDense *a = (Bilinear_SeqDense *) A->data;
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;
  int                ierr;

  PetscFunctionBegin;
  if (B->type != BILINEAR_SEQDENSE) {
    ierr = BilinearCopy_Basic(A, B);                                                                     CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (a->n_i != b->n_i || a->n_j != b->n_j || a->n_k != b->n_k)
    SETERRQ(PETSC_ERR_ARG_SIZ, 0, "Bilinear A, Bilinear B: local dim");
  ierr = PetscMemcpy(b->v, a->v, a->n_i*a->n_j*a->n_k * sizeof(Scalar));                                 CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearSetOption_SeqDense"
int BilinearSetOption_SeqDense(Bilinear B, BilinearOption op)
{
  PetscFunctionBegin;
  if (op == MAT_SYMMETRIC                  ||
      op == MAT_NO_NEW_NONZERO_LOCATIONS   ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS  ||
      op == MAT_NEW_NONZERO_LOCATION_ERR   ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERR ||
      op == MAT_IGNORE_OFF_PROC_ENTRIES
      )
    PLogInfo(B, "BilinearSetOption_SeqDense: Option ignored\n");
  else
    SETERRQ(PETSC_ERR_SUP, 0, "Unknown option");
  PetscFunctionReturn(0);
}

/*------------------------------------- Generation and Assembly Functions -------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearSetValues_SeqDense"
int BilinearSetValues_SeqDense(Bilinear B, int i, int *idxi, int j, int *idxj, int k, int *idxk, Scalar *v, InsertMode addv)
{ 
  Bilinear_SeqDense *b   = (Bilinear_SeqDense *) B->data;
  int                n_i = b->n_i;
  int                n_j = b->n_j;
  int                n_k = b->n_k;
  int                ii, jj, kk;

  PetscFunctionBegin;
  if (addv == INSERT_VALUES)
  {
    for(ii = 0; ii < i; ii++) {
#if defined(PETSC_USE_BOPT_g)
      if (idxi[ii] < 0)    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Negative row");
      if (idxi[ii] >= n_i) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Row too large");
#endif
      for(jj = 0; jj < j; jj++) {
#if defined(PETSC_USE_BOPT_g)
        if (idxj[jj] < 0)    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Negative column");
        if (idxj[jj] >= n_j) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Column too large");
#endif
        for(kk = 0; kk < k; kk++) {
#if defined(PETSC_USE_BOPT_g)
          if (idxk[kk] < 0)    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Negative subcolumn");
          if (idxk[kk] >= n_k) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Subcolumn too large");
#endif
          b->v[idxi[ii]*n_k*n_k+idxj[jj]*n_k+idxk[kk]] = *v++;
        }
      }
    }
  } else {
    for(ii = 0; ii < i; ii++) {
#if defined(PETSC_USE_BOPT_g)
      if (idxi[ii] < 0)    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Negative row");
      if (idxi[ii] >= n_i) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Row too large");
#endif
      for(jj = 0; jj < j; jj++) {
#if defined(PETSC_USE_BOPT_g)
        if (idxj[jj] < 0)    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Negative column");
        if (idxj[jj] >= n_j) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Column too large");
#endif
        for(kk = 0; kk < k; kk++) {
#if defined(PETSC_USE_BOPT_g)
          if (idxk[kk] < 0)    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Negative subcolumn");
          if (idxk[kk] >= n_k) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, 0, "Subcolumn too large");
#endif
          b->v[idxi[ii]*n_k*n_k+idxj[jj]*n_k+idxk[kk]] += *v++;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetArray_SeqDense"
int BilinearGetArray_SeqDense(Bilinear B, Scalar **array)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;

  PetscFunctionBegin;
  *array = b->v;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearRestoreArray_SeqDense"
int BilinearRestoreArray_SeqDense(Bilinear B, Scalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearZeroEntries_SeqDense"
int BilinearZeroEntries_SeqDense(Bilinear B)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;
  int                ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(b->v, b->n_i*b->n_j*b->n_k * sizeof(Scalar));                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*-------------------------------------------- Application Functions ------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearMult_SeqDense"
int BilinearMult_SeqDense(Bilinear B, Vec x, Mat M)
{
  Bilinear_SeqDense *b   = (Bilinear_SeqDense *) B->data;
  int                n_i = b->n_i;
  int                n_j = b->n_j;
  int                n_k = b->n_k;
  PetscTruth         hasArray;
  int               *idx;
  Scalar            *array, *array2;
  int                i, j, k;
  int                ierr;

  PetscFunctionBegin;
  if (!b->n_i || !b->n_j || !b->n_k) PetscFunctionReturn(0);
  ierr = MatHasOperation(M, MATOP_GET_ARRAY, &hasArray);                                                 CHKERRQ(ierr);
  if (hasArray == PETSC_TRUE) {
    /* THIS IS VERY DANGEROUS:
         We assume the storage format is column-oriented. There should be a way
         to query this.
    */
    ierr = VecGetArray(x, &array);                                                                       CHKERRQ(ierr);
    ierr = MatGetArray(M, &array2);                                                                      CHKERRQ(ierr);
    for(i = 0; i < n_i; i++) {
      for(j = 0; j < n_j; j++) {
        array2[j*n_i+i]    = 0.0;
        for(k = 0; k < n_k; k++)
          array2[j*n_i+i] += b->v[i*n_j*n_k+j*n_k+k]*array[k];
      }
    }
    ierr = VecRestoreArray(x, &array);                                                                   CHKERRQ(ierr);
    ierr = MatRestoreArray(M, &array2);                                                                  CHKERRQ(ierr);
  } else {
    idx    = (int *)    PetscMalloc(PetscMax(n_i, n_j) * sizeof(int));    CHKPTRQ(idx);
    array2 = (Scalar *) PetscMalloc(n_i*n_j            * sizeof(Scalar)); CHKPTRQ(array2);
    ierr = VecGetArray(x, &array);                                                                       CHKERRQ(ierr);
    for(i = 0; i < PetscMax(n_i, n_j); i++)
      idx[i] = i;
    for(i = 0; i < n_i; i++) {
      for(j = 0; j < n_j; j++) {
        array2[i*n_j+j]    = 0.0;
        for(k = 0; k < n_k; k++)
          array2[i*n_j+j] += b->v[i*n_j*n_k+j*n_k+k]*array[k];
      }
    }
    ierr = MatSetValues(M, n_i, idx, n_j, idx, array2, INSERT_VALUES);                                   CHKERRQ(ierr);
    ierr = VecRestoreArray(x, &array);                                                                   CHKERRQ(ierr);
    ierr = PetscFree(array2);                                                                            CHKERRQ(ierr);
    ierr = PetscFree(idx);                                                                               CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);                                                        CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);                                                          CHKERRQ(ierr);
  PLogFlops(2*n_i*n_j*n_k - n_j*n_k);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearFullMult_SeqDense"
int BilinearFullMult_SeqDense(Bilinear B, Vec x, Vec y, Vec z)
{
  Bilinear_SeqDense *b   = (Bilinear_SeqDense *) B->data;
  int                n_i = b->n_i;
  int                n_j = b->n_j;
  int                n_k = b->n_k;
  Scalar            *array, *array2, *array3;
  int                i, j, k;
  int                ierr;

  PetscFunctionBegin;
  if (!b->n_i || !b->n_j || !b->n_k) PetscFunctionReturn(0);
  ierr = VecGetArray(x, &array);                                                                         CHKERRQ(ierr);
  ierr = VecGetArray(y, &array2);                                                                        CHKERRQ(ierr);
  ierr = VecGetArray(z, &array3);                                                                        CHKERRQ(ierr);
  for(i = 0; i < n_i; i++) {
    array3[i] = 0.0;
    for(j = 0; j < n_j; j++)
      for(k = 0; k < n_k; k++)
        array3[i] += b->v[i*n_j*n_k+j*n_k+k]*array[j]*array2[k];
  }
  ierr = VecRestoreArray(x, &array);                                                                     CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &array2);                                                                    CHKERRQ(ierr);
  ierr = VecRestoreArray(z, &array3);                                                                    CHKERRQ(ierr);
  PLogFlops(3*n_i*n_j*n_k - n_j*n_k);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearDiamond_SeqDense"
int BilinearDiamond_SeqDense(Bilinear B, Vec x, Vec y)
{
  Bilinear_SeqDense *b   = (Bilinear_SeqDense *) B->data;
  int                n_i = b->n_i;
  int                n_j = b->n_j;
  int                n_k = b->n_k;
  Scalar            *array, *array2;
  int                i, j, k;
  int                ierr;

  PetscFunctionBegin;
  if (!b->n_i || !b->n_j || !b->n_k) PetscFunctionReturn(0);
  ierr = VecGetArray(x, &array);                                                                         CHKERRQ(ierr);
  ierr = VecGetArray(y, &array2);                                                                        CHKERRQ(ierr);
  for(i = 0; i < n_i; i++) {
    array2[i] = 0.0;
    for(j = 0; j < n_j; j++)
      for(k = 0; k < n_k; k++)
        array2[i] += b->v[i*n_j*n_k+j*n_k+k]*array[j]*array[k];
  }
  ierr = VecRestoreArray(x, &array);                                                                     CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &array2);                                                                    CHKERRQ(ierr);
  PLogFlops(3*n_i*n_j*n_k - n_j*n_k);
  PetscFunctionReturn(0);
}

/*------------------------------------------- Interrogation Functions -----------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearGetInfo_SeqDense"
int BilinearGetInfo_SeqDense(Bilinear B, InfoType flag, BilinearInfo *info)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;
  int                N = b->n_i*b->n_j*b->n_k;
  Scalar            *v = b->v;
  int                i, count;

  PetscFunctionBegin;
  for(i = 0, count =  0; i < N; i++, v++) {
    if (*v != 0.0)
      count++;
  }

  info->rows_global       = (double) b->n_i;
  info->cols_global       = (double) b->n_j;
  info->subcols_global    = (double) b->n_k;
  info->rows_local        = (double) b->n_i;
  info->cols_local        = (double) b->n_j;
  info->subcols_local     = (double) b->n_k;
  info->nz_allocated      = (double) N;
  info->nz_used           = (double) count;
  info->nz_unneeded       = (double) (N - count);
  info->assemblies        = (double) B->num_ass;
  info->mallocs           = 0;
  info->memory            = B->mem;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetSize_SeqDense"
int BilinearGetSize_SeqDense(Bilinear B, int *N_i, int *N_j, int *N_k)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;

  PetscFunctionBegin;
  *N_i = b->n_i;
  *N_j = b->n_j;
  *N_k = b->n_k;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearGetOwnershipRange_SeqDense"
int BilinearGetOwnershipRange_SeqDense(Bilinear B, int *rowStart, int *rowEnd)
{
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;

  PetscFunctionBegin;
  *rowStart = 0;
  *rowEnd   = b->n_i;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearEqual_SeqDense"
int BilinearEqual_SeqDense(Bilinear A, Bilinear B, PetscTruth *eq)
{
  Bilinear_SeqDense *a = (Bilinear_SeqDense *) A->data;
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;

  PetscFunctionBegin;
  if (B->type != BILINEAR_SEQDENSE)
    SETERRQ(PETSC_ERR_SUP, 0, "Bilinear operators must be of same type: BILINEAR_SEQDENSE");
  if ((a->n_i != b->n_i) || (a->n_j != b->n_j) || (a->n_k != b->n_k)) {
    *eq = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (PetscMemcmp(a->v, b->v, a->n_i*a->n_j*a->n_k * sizeof(Scalar)))
    *eq = PETSC_TRUE;
  else
    *eq = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "BilinearNorm_SeqDense"
int BilinearNorm_SeqDense(Bilinear B, NormType type, double *norm)
{
  Bilinear_SeqDense *b   = (Bilinear_SeqDense *) B->data;
  int                n_i = b->n_i;
  int                n_j = b->n_j;
  int                n_k = b->n_k;
  Scalar            *v   = b->v;
  double             sum;
  int                i, j, k;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for(i = 0, sum = 0.0; i < n_i*n_j*n_k; i++, v++)
      sum += (*v)*(*v);
    *norm = sqrt(sum);
    PLogFlops(2*n_i*n_j*n_k+1);
  } else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for(i = 0, sum = 0.0; i < n_i; i++) {
      for(j = 0; j < n_j; j++)
        for(k = 0; k < n_k; k++, v++)
          sum += PetscAbsScalar(*v);
      if (sum > *norm)
        *norm = sum;
    }
    PLogFlops(n_i*n_j*n_k);
  } else if (type == NORM_1) {
    SETERRQ(PETSC_ERR_SUP, 0, "No one norm");
  } else if (type == NORM_2) {
    SETERRQ(PETSC_ERR_SUP, 0, "No two norm");
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG, 0, "Unknown norm");
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------- Factorization Functions -----------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "BilinearLUFactor_SeqDense"
int BilinearLUFactor_SeqDense(Bilinear B, IS row, IS col, IS subcol, double f)
{
#if 1
  SETERRQ(PETSC_ERR_SUP, 0, "Do not know how yet");
#else
  Bilinear_SeqDense *mat = (Bilinear_SeqDense *) A->data;
  int          info;

  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  if (!mat->pivots) {
    mat->pivots = (int *) PetscMalloc((mat->m+1)*sizeof(int));CHKPTRQ(mat->pivots);
    PLogObjectMemory(A,mat->m*sizeof(int));
  }
  LAgetrf_(&mat->m,&mat->n,mat->v,&mat->m,mat->pivots,&info);
  if (info<0) SETERRQ(PETSC_ERR_LIB,0,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,0,"Bad LU factorization");
  A->factor = FACTOR_LU;
  PLogFlops((2*mat->n*mat->n*mat->n)/3);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNC__  
#define __FUNC__ "BilinearCholeskyFactor_SeqDense"
int BilinearCholeskyFactor_SeqDense(Bilinear B, IS perm, double f)
{
#if 1
  SETERRQ(PETSC_ERR_SUP, 0, "Do not know how yet");
#else
  Bilinear_SeqDense  *mat = (Bilinear_SeqDense *) A->data;
  int           info;

  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  if (mat->pivots) {
    PetscFree(mat->pivots);
    PLogObjectMemory(A,-mat->m*sizeof(int));
    mat->pivots = 0;
  }
  LApotrf_("L",&mat->n,mat->v,&mat->m,&info);
  if (info) SETERRQ(PETSC_ERR_MAT_CH_ZRPVT,0,"Bad factorization");
  A->factor = FACTOR_CHOLESKY;
  PLogFlops((mat->n*mat->n*mat->n)/3);
  PetscFunctionReturn(0);
#endif
}

/* -------------------------------------------------------------------*/
static struct _BilinearOps BilinearOps = {0 /* BilinearPrintHelp_SeqDense */,
                                          BilinearSetOption_SeqDense,
                                          BilinearCopy_SeqDense,
                                          BilinearConvertSameType_SeqDense,
                                          0 /* BilinearSerialize_SeqDense */,
                                          BilinearGetInfo_SeqDense,
                                          BilinearGetSize_SeqDense,
                                          BilinearGetSize_SeqDense,
                                          BilinearGetOwnershipRange_SeqDense,
                                          BilinearEqual_SeqDense,
                                          BilinearNorm_SeqDense,
                                          BilinearGetArray_SeqDense, 
                                          BilinearRestoreArray_SeqDense,
                                          BilinearSetValues_SeqDense,
                                          0 /* BilinearAssemblyBegin_SeqDense */,
                                          0 /* BilinearAssemblyEnd_SeqDense */,
                                          BilinearZeroEntries_SeqDense,
                                          BilinearMult_SeqDense, 
                                          BilinearFullMult_SeqDense, 
                                          BilinearDiamond_SeqDense, 
                                          BilinearLUFactor_SeqDense,
                                          BilinearCholeskyFactor_SeqDense};

static int BilinearAllocate_Dense_Seq(Bilinear B) {
  Bilinear_SeqDense *b = (Bilinear_SeqDense *) B->data;
  int                i = B->N_i;
  int                j = B->N_j;
  int                k = B->N_k;
  int                ierr;

  PetscFunctionBegin;
  if (i*j*k > 0) {
    ierr = PetscMalloc(i*j*k * sizeof(PetscScalar), &b->v);                                               CHKERRQ(ierr);
    ierr = PetscMemzero(b->v, i*j*k * sizeof(PetscScalar));                                               CHKERRQ(ierr);
    PetscLogObjectMemory(B, i*j*k * sizeof(PetscScalar));
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "BilinearCreate_Dense_Seq"
int BilinearCreate_Dense_Seq(Bilinear B) {
  Bilinear_SeqDense *b;
  int                size;
  int                ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm, &size);                                                                   CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG, 0, "Communicator must have only one process");

  ierr = PetscNew(Bilinear_SeqDense, &b);                                                                 CHKERRQ(ierr);
  ierr = PetscMemcpy(B->ops, &BilinearOps, sizeof(struct _BilinearOps));                                  CHKERRQ(ierr);

  B->data = (void *) b;

  b->v          = PETSC_NULL;
  b->user_alloc = PETSC_FALSE;
  b->pivots     = PETSC_NULL;

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "BilinearSerialize_Dense_Seq"
int BilinearSerialize_Dense_Seq(MPI_Comm comm, Bilinear *B, Viewer viewer, PetscTruth store) {
  SETERRQ(PETSC_ERR_SUP, "");
}
EXTERN_C_END
