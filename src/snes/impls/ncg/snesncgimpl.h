/* 
   Private context for Richardson iteration
*/

#ifndef __SNES_NCG_H
#define __SNES_NCG_H
#include <private/snesimpl.h>

typedef struct {
  /* Line Search Parameters */
  PetscInt    betatype;     /* 0 = Fletcher-Reeves, 1 = Polak-Ribiere-Polyak, 2 = Hestenes-Steifel, 3 = Dai-Yuan, 4 = Conjugate Descent */
  PetscViewer monitor;   /* monitor for ncg (prints out the alpha and beta parameters at each interation) */
} SNES_NCG;

#endif
