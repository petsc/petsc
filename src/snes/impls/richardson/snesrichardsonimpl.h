/* 
   Private context for Richardson iteration
*/

#ifndef __SNES_RICHARDSON_H
#define __SNES_RICHARDSON_H
#include <private/snesimpl.h>
#include <petsclinesearch.h>

typedef struct {

  PetscLineSearch linesearch;

  int dummy;
} SNES_NRichardson;

#endif
