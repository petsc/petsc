#ifndef _SNESNGMRES_H
#define _SNESNGMRES_H

#include <private/snesimpl.h>

/*  Data structure for the Nonlinear GMRES method.  */
typedef struct {
 
  // Solver parameters and counters
  PetscInt     msize;          // maximum size of krylov space
  PetscInt     k_rmax;         // maximum number of iterations before restart
  PetscBool    debug;          // debugging output for NGMRES
  // History and subspace data
  Vec          *rdot;          // residual history -- length msize 
  Vec          *xdot;          // solution history -- length msize
  PetscReal     *r_norms;         // the residual norm history

  // General minimization problem context
  PetscScalar  *h;              // the constraint matrix
  PetscScalar  *beta;          // rhs for the minimization problem
  PetscScalar    *xi;            // the norms of the various residual histories

  // Selection constants
  PetscReal gammaA;          // Criterion A residual tolerance
  PetscReal epsilonB;        // Criterion B difference tolerance
  PetscReal deltaB;          // Criterion B residual tolerance
  PetscReal gammaC;          // Restart residual tolerance

  // LS Minimization solve context
  PetscScalar  *q;    // the matrix formed as q_ij = (rdot_i, rdot_j)
  PetscBLASInt m;     // matrix dimension
  PetscBLASInt n;     // matrix dimension
  PetscBLASInt nrhs;  // the number of right hand sides
  PetscBLASInt lda;   // the padded matrix dimension
  PetscBLASInt ldb;   // the padded vector dimension
  PetscReal * s;      // the singular values
  PetscReal rcond;    // the condition
  PetscBLASInt rank;  // the effective rank
  PetscScalar * work; // the work vector
  PetscReal  * rwork; // the real work vector used for complex
  PetscBLASInt lwork; // the size of the work vector
  PetscBLASInt info;  // the output condition

} SNES_NGMRES;

#define H(i,j)  ngmres->h[i*ngmres->msize + j]
#define Q(i,j)  ngmres->q[i*ngmres->msize + j]

#endif
