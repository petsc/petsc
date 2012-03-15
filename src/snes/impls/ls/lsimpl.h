/* 
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#ifndef __SNES_LS_H
#define __SNES_LS_H
#include <private/snesimpl.h>
#include <petsclinesearch.h>

typedef struct {
  PetscLineSearch linesearch;
} SNES_LS;

#endif
