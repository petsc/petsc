/*
   Private data structure for QR preconditioner.
*/
#pragma once

#include <../src/ksp/pc/impls/factor/factor.h>

typedef struct {
  PC_Factor hdr;
  IS        col; /* index sets used for reordering */
} PC_QR;
