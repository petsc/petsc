
#include <petsc/private/matimpl.h>

/* Get new PetscMatStashSpace into the existing space */
PetscErrorCode PetscMatStashSpaceGet(PetscInt bs2, PetscInt n, PetscMatStashSpace *space)
{
  PetscMatStashSpace a;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscMalloc(sizeof(struct _MatStashSpace), &a));
  PetscCall(PetscMalloc3(n * bs2, &(a->space_head), n, &a->idx, n, &a->idy));

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
  *space = a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Copy the values in space into arrays val, idx and idy. Then destroy space */
PetscErrorCode PetscMatStashSpaceContiguous(PetscInt bs2, PetscMatStashSpace *space, PetscScalar *val, PetscInt *idx, PetscInt *idy)
{
  PetscMatStashSpace a;

  PetscFunctionBegin;
  while ((*space)) {
    a = (*space)->next;
    PetscCall(PetscArraycpy(val, (*space)->val, (*space)->local_used * bs2));
    val += bs2 * (*space)->local_used;
    PetscCall(PetscArraycpy(idx, (*space)->idx, (*space)->local_used));
    idx += (*space)->local_used;
    PetscCall(PetscArraycpy(idy, (*space)->idy, (*space)->local_used));
    idy += (*space)->local_used;

    PetscCall(PetscFree3((*space)->space_head, (*space)->idx, (*space)->idy));
    PetscCall(PetscFree(*space));
    *space = a;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace *space)
{
  PetscMatStashSpace a;

  PetscFunctionBegin;
  while (*space) {
    a = (*space)->next;
    PetscCall(PetscFree3((*space)->space_head, (*space)->idx, (*space)->idy));
    PetscCall(PetscFree((*space)));
    *space = a;
  }
  *space = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
