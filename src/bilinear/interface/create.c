#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: create.c,v 1.7 2000/01/10 03:54:25 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"      /*I "bilinear.h"  I*/

#undef  __FUNCT__
#define __FUNCT__ "BilinearCreate"
/*@ 
  BilinearCreate - This function creates an empty bilinear operator. The type can then be set with BilinearSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm     - The communicator

  Output Parameter:
. bilinear - The bilinear

  Level: beginner

.keywords: Bilinear, create
.seealso: BilinearSetType(), BilinearSetUp(), BilinearDestroy()
@*/
int BilinearCreate(MPI_Comm comm, Bilinear *bilinear) {
  Bilinear b;
  int      ierr;

  PetscFunctionBegin;
  PetscValidPointer(bilinear);
  *bilinear = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = BilinearInitializePackage(PETSC_NULL);                                                           CHKERRQ(ierr);
#endif

  PetscHeaderCreate(b, _p_Bilinear, struct _BilinearOps, BILINEAR_COOKIE, -1, "Bilinear", comm, BilinearDestroy, BilinearView);
  PetscLogObjectCreate(b);
  PetscLogObjectMemory(b, sizeof(struct _Bilinear));
  ierr = PetscMemzero(b->ops, sizeof(struct _BilinearOps));                                               CHKERRQ(ierr);
  b->bops->publish    = PETSC_NULL /* BilinearPublish_Petsc */;
  b->type_name        = PETSC_NULL;
  b->serialize_name   = PETSC_NULL;

  /* Size variables */
  b->N_i = 0;
  b->N_j = 0;
  b->N_k = 0;
  b->n_i = 0;
  b->n_j = 0;
  b->n_k = 0;
  /* Assembly variables */
  b->assembled     = PETSC_FALSE;
  b->was_assembled = PETSC_FALSE;
  b->num_ass       = 0;
  b->same_nonzero  = PETSC_FALSE;
  b->insertmode    = INSERT_VALUES;
  /* Factorization variables */
  b->factor           = BILINEAR_FACTOR_NONE;
  /* Query variables */
  b->info = PETSC_NULL;

  *bilinear = b;
  PetscFunctionReturn(0);
}
