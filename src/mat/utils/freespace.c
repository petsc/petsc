
#include <../src/mat/utils/freespace.h>

#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceGet"
PetscErrorCode PetscFreeSpaceGet(PetscInt n,PetscFreeSpaceList *list)
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _Space),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&(a->array_head));CHKERRQ(ierr);
  a->array            = a->array_head;
  a->local_remaining  = n;
  a->local_used       = 0;
  a->total_array_size = 0;
  a->more_space       = PETSC_NULL;

  if (*list) {
    (*list)->more_space = a;
    a->total_array_size = (*list)->total_array_size;
  }

  a->total_array_size += n;
  *list               =  a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceContiguous"
PetscErrorCode PetscFreeSpaceContiguous(PetscFreeSpaceList *head,PetscInt *space) 
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while ((*head)) {
    a     =  (*head)->more_space;
    ierr  =  PetscMemcpy(space,(*head)->array_head,((*head)->local_used)*sizeof(PetscInt));CHKERRQ(ierr);
    space += (*head)->local_used;
    ierr  =  PetscFree((*head)->array_head);CHKERRQ(ierr);
    ierr  =  PetscFree(*head);CHKERRQ(ierr);
    *head =  a;
  }
  PetscFunctionReturn(0);
}

/*
  PetscFreeSpaceContiguous_LU -
    Copy a linket list obtained from matrix symbolic ILU or LU factorization into a contiguous array 
  that enables an efficient matrix triangular solve.

   Input Parameters:
+  head - linked list of column indices obtained from matrix symbolic ILU or LU factorization
.  space - an allocated int array with length nnz of factored matrix. 
.  n - order of the matrix
.  bi - row pointer of factored matrix L with length n+1.
-  bdiag - int array of length n+1. bdiag[i] points to diagonal of U(i,:), and bdiag[n] points to entry of U(n-1,0)-1.

   Output Parameter:
.  space - column indices are copied into this int array with contiguous layout of L and U

   See MatILUFactorSymbolic_SeqAIJ_ilu0() for detailed data structure of L and U
*/
#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceContiguous_LU"
PetscErrorCode PetscFreeSpaceContiguous_LU(PetscFreeSpaceList *head,PetscInt *space,PetscInt n,PetscInt *bi,PetscInt *bdiag) 
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;
  PetscInt           row,nnz,*bj,*array,total,bi_temp;
  PetscInt           nnzL,nnzU;

  PetscFunctionBegin;
  bi_temp = bi[n];
  row       = 0; 
  total     = 0; 
  nnzL  = bdiag[0];
  while ((*head)) {
    total += (*head)->local_used;
    array  = (*head)->array_head;
  
    while (row < n) {
      if (bi[row+1] > total) break;
      /* copy array entries into bj for this row */  
      nnz  = bi[row+1] - bi[row];
      /* set bi[row] for new datastruct */
      if (row == 0 ){
        bi[row] = 0;
      } else {
        bi[row] = bi[row-1] + nnzL; /* nnzL of previous row */
      } 

      /* L part */
      nnzL = bdiag[row];
      bj   = space+bi[row];
      ierr = PetscMemcpy(bj,array,nnzL*sizeof(PetscInt));CHKERRQ(ierr);
    
      /* diagonal entry */
      bdiag[row] = bi_temp - 1;
      space[bdiag[row]] = row;

      /* U part */
      nnzU        = nnz - nnzL; 
      bi_temp = bi_temp - nnzU;
      nnzU --;      /* exclude diagonal */
      bj = space + bi_temp;
      ierr = PetscMemcpy(bj,array+nnzL+1,nnzU*sizeof(PetscInt));CHKERRQ(ierr);
      array += nnz; 
      row++;
    }

    a     = (*head)->more_space;
    ierr  = PetscFree((*head)->array_head);CHKERRQ(ierr);
    ierr  = PetscFree(*head);CHKERRQ(ierr);
    *head = a;
  }
  if (n) {
    bi[n] = bi[n-1] + nnzL;
    bdiag[n] = bdiag[n-1]-1;
  }
  PetscFunctionReturn(0);
}

/*
  PetscFreeSpaceContiguous_Cholesky -
    Copy a linket list obtained from matrix symbolic ICC or Cholesky factorization into a contiguous array 
  that enables an efficient matrix triangular solve.

   Input Parameters:
+  head - linked list of column indices obtained from matrix symbolic ICC or Cholesky factorization
.  space - an allocated int array with length nnz of factored matrix. 
.  n - order of the matrix
.  ui - row pointer of factored matrix with length n+1. All entries are set based on the traditional layout U matrix.
-  udiag - int array of length n.

   Output Parameter:
+  space - column indices are copied into this int array with contiguous layout of U, with diagonal located as the last entry in each row
-  udiag - indices of diagonal entries

   See MatICCFactorSymbolic_SeqAIJ_newdatastruct() for detailed description.
*/

#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceContiguous_Cholesky"
PetscErrorCode PetscFreeSpaceContiguous_Cholesky(PetscFreeSpaceList *head,PetscInt *space,PetscInt n,PetscInt *ui,PetscInt *udiag) 
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;
  PetscInt           row,nnz,*uj,*array,total;

  PetscFunctionBegin;
  row   = 0; 
  total = 0; 
  while (*head) {
    total += (*head)->local_used;
    array  = (*head)->array_head;
  
    while (row < n){
      if (ui[row+1] > total) break;
      udiag[row] = ui[row+1] - 1;     /* points to the last entry of U(row,:) */     
      nnz  = ui[row+1] - ui[row] - 1; /* exclude diagonal */
      uj   = space + ui[row];
      ierr = PetscMemcpy(uj,array+1,nnz*sizeof(PetscInt));CHKERRQ(ierr);
      uj[nnz] = array[0]; /* diagonal */
      array += nnz + 1; 
      row++;
    }

    a     = (*head)->more_space;
    ierr  = PetscFree((*head)->array_head);CHKERRQ(ierr);
    ierr  = PetscFree(*head);CHKERRQ(ierr);
    *head = a;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceDestroy"
PetscErrorCode PetscFreeSpaceDestroy(PetscFreeSpaceList head) 
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while ((head)) {
    a    = (head)->more_space;
    ierr = PetscFree((head)->array_head);CHKERRQ(ierr);
    ierr = PetscFree(head);CHKERRQ(ierr);
    head = a;
  }
  PetscFunctionReturn(0);
}
