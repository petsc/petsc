#define PETSCMAT_DLL

#include "../src/mat/utils/freespace.h"

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
  a->more_space       = NULL;

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
  while ((*head)!=NULL) {
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
  Copy a linket list obtained from matrix symbolic ilu or lu factorization into a contiguous array that enables 
  an efficient matrix solve.

   Input Parameters:
+  head - linked list of column indices obtained from matrix symbolic ilu or lu factorization
.  space - an allocated int arry with length nnz of factored matrix. 
.  n - order of the matrix
.  bi - row pointer of factored matrix with length 2n+2. First n+1 entries are set based on the traditional layout of L and U matrices
.
-  bdiag - int array holding the number of nonzeros in each row of L matrix, excluding diagonals.

   Output Parameter:
.  space - column indices are copied into this int array with contiguous layout of L and U
   See MatILUFactorSymbolic_SeqAIJ_ilu0_newdatastruct() for detailed description.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceContiguous_newdatastruct"
PetscErrorCode PetscFreeSpaceContiguous_newdatastruct(PetscFreeSpaceList *head,PetscInt *space,PetscInt n,PetscInt *bi,PetscInt *bdiag) 
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;
  PetscInt           row,nnz,*bj,*array,total;
  PetscInt           nnzL,nnzU;

  PetscFunctionBegin;
  bi[2*n+1] = bi[n];
  row       = 0; 
  total     = 0; 
  nnzL  = bdiag[0];
  while ((*head)!=NULL) {
    total += (*head)->local_used;
    array  = (*head)->array_head;
  
    while (bi[row+1] <= total && row < n){
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
      bdiag[row]        = bi[2*n-row+1]-1;
      space[bdiag[row]] = row;

      /* U part */
      nnzU        = nnz - nnzL; 
      bi[2*n-row] = bi[2*n-row+1] - nnzU;
      nnzU --;      /* exclude diagonal */
      bj   = space + bi[2*n-(row)];
      ierr = PetscMemcpy(bj,array+nnzL+1,nnzU*sizeof(PetscInt));CHKERRQ(ierr);
  
      array += nnz; 
      row++;
    }

    a     = (*head)->more_space;
    ierr  = PetscFree((*head)->array_head);CHKERRQ(ierr);
    ierr  = PetscFree(*head);CHKERRQ(ierr);
    *head = a;
  }
  bi[n] = bi[n-1] + nnzL;
  if (bi[n] != bi[n+1]) SETERRQ2(1,"bi[n] %d != bi[n+1] %d",bi[n],bi[n+1]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFreeSpaceDestroy"
PetscErrorCode PetscFreeSpaceDestroy(PetscFreeSpaceList head) 
{
  PetscFreeSpaceList a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while ((head)!=NULL) {
    a    = (head)->more_space;
    ierr = PetscFree((head)->array_head);CHKERRQ(ierr);
    ierr = PetscFree(head);CHKERRQ(ierr);
    head = a;
  }
  PetscFunctionReturn(0);
}
