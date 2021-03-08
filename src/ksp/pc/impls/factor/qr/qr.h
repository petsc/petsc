/*
   Private data structure for QR preconditioner.
*/
#if !defined(QR_H)
#define QR_H

#include <../src/ksp/pc/impls/factor/factor.h>

typedef struct {
  PC_Factor hdr;
  IS        col;            /* index sets used for reordering */
} PC_QR;

#endif
