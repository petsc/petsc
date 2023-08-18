#pragma once
#include <petscmat.h>
#include "mmio.h"

PetscErrorCode MatCreateFromMTX(Mat *A, const char *filein, PetscBool aijonly);
