#define PETSCMAT_DLL

#include "src/mat/utils/freespace.h"

#undef __FUNCT__
#define __FUNCT__ "GetMoreSpace"
PetscErrorCode GetMoreSpace(PetscInt n,FreeSpaceList *list)
{
  FreeSpaceList  a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(FreeSpace),&a);CHKERRQ(ierr);
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
#define __FUNCT__ "MakeSpaceContiguous"
PetscErrorCode MakeSpaceContiguous(FreeSpaceList *head,PetscInt *space) 
{
  FreeSpaceList  a;
  PetscErrorCode ierr;

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

#undef __FUNCT__
#define __FUNCT__ "DestroySpace"
PetscErrorCode DestroySpace(FreeSpaceList head) 
{
  FreeSpaceList  a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while ((head)!=NULL) {
    a    = (head)->more_space;
    ierr = PetscFree((head)->array_head);CHKERRQ(ierr);
    ierr = PetscFree(head);CHKERRQ(ierr);
    head = a;
  }
  PetscFunctionReturn(0);
}
