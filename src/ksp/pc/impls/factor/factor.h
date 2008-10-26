/* 
   Private data structure for ILU/ICC/LU/Cholesky preconditioners.
*/
#if !defined(__FACTOR_H)
#define __FACTOR_H

#include "private/pcimpl.h"                /*I "petscpc.h" I*/

typedef struct {
  Mat               fact;             /* factored matrix */
  MatFactorInfo     info;
  MatOrderingType   ordering;         /* matrix reordering */
  MatSolverPackage  solvertype;
} PC_Factor;

#endif
