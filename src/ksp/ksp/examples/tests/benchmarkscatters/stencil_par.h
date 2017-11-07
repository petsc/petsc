/*
 * stencil_par.h
 *
 *  Created on: Jan 4, 2012
 *      Author: htor
 */

#ifndef STENCIL_PAR_H_
#define STENCIL_PAR_H_

#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// row-major order
#define ind(i,j) (j)*(bx+2)+(i)

void printarr_par(int iter, double* array, int size, int px, int py, int rx, int ry, int bx, int by, int offx, int offy, MPI_Comm comm);


#endif /* STENCIL_PAR_H_ */
