#ifndef MM_LOADER_H
#define MM_LOADER_H
#include <petscmat.h>
#include "mmio.h"

PetscErrorCode MatCreateFromMTX(Mat *A, const char *filein, PetscBool aijonly);
#endif
