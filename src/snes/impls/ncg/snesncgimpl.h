/* 
   Private context for Richardson iteration
*/

#ifndef __SNES_NCG_H
#define __SNES_NCG_H
#include <petsc-private/snesimpl.h>
#include <petsc-private/linesearchimpl.h>

typedef struct {
  PetscInt    betatype;     /* 0 = Fletcher-Reeves, 1 = Polak-Ribiere-Polyak, 2 = Hestenes-Steifel, 3 = Dai-Yuan, 4 = Conjugate Descent */
  PetscViewer monitor;   /* monitor for ncg (prints out the alpha and beta parameters at each interation) */
} SNES_NCG;

#endif
