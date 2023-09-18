/*
   Private context for Richardson iteration
*/

#pragma once
#include <petsc/private/snesimpl.h>
#include <petsc/private/linesearchimpl.h>

typedef struct {
  SNESNCGType type;    /* Fletcher-Reeves, Polak-Ribiere-Polyak, Hestenes-Steifel, Dai-Yuan, Conjugate Descent */
  PetscViewer monitor; /* monitor for ncg (prints out the alpha and beta parameters at each iteration) */
} SNES_NCG;
