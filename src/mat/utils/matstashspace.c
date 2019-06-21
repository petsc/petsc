
#include <petsc/private/matimpl.h>

/* Get new PetscMatStashSpace into the existing space */
PetscErrorCode PetscMatStashSpaceGet(PetscInt bs2,PetscInt n,PetscMatStashSpace *space)
{
  PetscMatStashSpace a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = PetscMalloc(sizeof(struct _MatStashSpace),&a);CHKERRQ(ierr);
  ierr = PetscMalloc3(n*bs2,&(a->space_head),n,&a->idx,n,&a->idy);CHKERRQ(ierr);

  a->val              = a->space_head;
  a->local_remaining  = n;
  a->local_used       = 0;
  a->total_space_size = 0;
  a->next             = NULL;

  if (*space) {
    (*space)->next      = a;
    a->total_space_size = (*space)->total_space_size;
  }
  a->total_space_size += n;
  *space               = a;
  PetscFunctionReturn(0);
}

/* Copy the values in space into arrays val, idx and idy. Then destroy space */
PetscErrorCode PetscMatStashSpaceContiguous(PetscInt bs2,PetscMatStashSpace *space,PetscScalar *val,PetscInt *idx,PetscInt *idy)
{
  PetscMatStashSpace a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while ((*space)) {
    a    = (*space)->next;
    ierr = PetscArraycpy(val,(*space)->val,(*space)->local_used*bs2);CHKERRQ(ierr);
    val += bs2*(*space)->local_used;
    ierr = PetscArraycpy(idx,(*space)->idx,(*space)->local_used);CHKERRQ(ierr);
    idx += (*space)->local_used;
    ierr = PetscArraycpy(idy,(*space)->idy,(*space)->local_used);CHKERRQ(ierr);
    idy += (*space)->local_used;

    ierr   =  PetscFree3((*space)->space_head,(*space)->idx,(*space)->idy);CHKERRQ(ierr);
    ierr   =  PetscFree(*space);CHKERRQ(ierr);
    *space = a;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace *space)
{
  PetscMatStashSpace a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while (*space) {
    a      = (*space)->next;
    ierr   = PetscFree3((*space)->space_head,(*space)->idx,(*space)->idy);CHKERRQ(ierr);
    ierr   = PetscFree((*space));CHKERRQ(ierr);
    *space = a;
  }
  *space = NULL;
  PetscFunctionReturn(0);
}
