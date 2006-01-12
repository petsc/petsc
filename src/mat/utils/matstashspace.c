
#define PETSCMAT_DLL

#include "src/mat/matimpl.h"
#include "src/mat/utils/matstashspace.h"

/* Get new PetscMatStashSpace into the existing space */
#undef __FUNCT__
#define __FUNCT__ "PetscMatStashSpaceGet"
PetscErrorCode PetscMatStashSpaceGet(PetscInt bs2,PetscInt n,PetscMatStashSpace *space)
{
  PetscMatStashSpace a;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = PetscMalloc(sizeof(struct _MatStashSpace),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(n*(bs2*sizeof(MatScalar)+2*sizeof(PetscInt)),&(a->space_head));CHKERRQ(ierr);
  a->val              = a->space_head;
  a->idx              = (PetscInt*)(a->val + bs2*n);
  a->idy              = (PetscInt*)(a->idx + n); 
  a->local_remaining  = n;
  a->local_used       = 0;
  a->total_space_size = 0;
  a->next             = PETSC_NULL;

  if (*space){
    (*space)->next      = a;
    a->total_space_size = (*space)->total_space_size; 
  }
  a->total_space_size += n;
  *space              = a;
  /* printf(" 2. total_space_size : %d\n",a->total_space_size); */
  /*
  PetscMPIInt rank;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  printf("[%d] MatStashSpaceCreate space %p, next %p, total_space_size: %d, space->val %p\n",rank,*space,(*space)->next,(*space)->total_space_size,(*space)->val);
  */
  PetscFunctionReturn(0);
}

/* Copy the values in space into arrays val, idx and idy. Then destroy space */
#undef __FUNCT__
#define __FUNCT__ "PetscMatStashSpaceContiguous"
PetscErrorCode PetscMatStashSpaceContiguous(PetscInt bs2,PetscMatStashSpace *space,PetscScalar *val,PetscInt *idx,PetscInt *idy) 
{
  PetscMatStashSpace a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while ((*space) != PETSC_NULL){
    a    = (*space)->next;
    ierr = PetscMemcpy(val,(*space)->val,((*space)->local_used*bs2)*sizeof(MatScalar));CHKERRQ(ierr);
    val += bs2*(*space)->local_used;
    ierr = PetscMemcpy(idx,(*space)->idx,((*space)->local_used)*sizeof(PetscInt));CHKERRQ(ierr);
    idx += (*space)->local_used;
    ierr = PetscMemcpy(idy,(*space)->idy,((*space)->local_used)*sizeof(PetscInt));CHKERRQ(ierr);
    idy += (*space)->local_used;
   
    ierr =  PetscFree((*space)->space_head);CHKERRQ(ierr);
    ierr =  PetscFree(*space);CHKERRQ(ierr);
    *space = a;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMatStashSpaceDestroy"
PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace space) 
{
  PetscMatStashSpace a;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while (space != PETSC_NULL){
    a    = space->next;
    ierr = PetscFree(space->space_head);CHKERRQ(ierr);
    ierr = PetscFree(space);CHKERRQ(ierr);
    space = a;
  }
  PetscFunctionReturn(0);
}
