#ifndef PETSC_DMPRODUCTIMPL_H
#define PETSC_DMPRODUCTIMPL_H

#include <petscdmproduct.h> /*I "petscdmproduct.h" I*/
#include <petsc/private/dmimpl.h>

#define DMPRODUCT_MAX_DIM 3

typedef struct {
  DM       dm[DMPRODUCT_MAX_DIM];
  PetscInt dim[DMPRODUCT_MAX_DIM]; /* Which dimension in the sub DM for this slot? */
} DM_Product;

#endif // PETSC_DMPRODUCTIMPL_H
