#ifndef __TAOLINESEARCH_ARMIJO_H
#define __TAOLINESEARCH_ARMIJO_H

/* Context for an Armijo (nonmonotone) linesearch for unconstrained
   minimization.

   Given a function f, the current iterate x, and a descent direction d:
   Find the smallest i in 0, 1, 2, ..., such that:

      f(x + (beta**i)d) <= f(x) + (sigma*beta**i)<grad f(x),d>

   The nonmonotone modification of this linesearch replaces the f(x) term
   with a reference value, R, and seeks to find the smallest i such that:

      f(x + (beta**i)d) <= R + (sigma*beta**i)<grad f(x),d>

   This modification does effect neither the convergence nor rate of
   convergence of an algorithm when R is chosen appropriately.  Essentially,
   R must decrease on average in some sense.  The benefit of a nonmonotone
   linesearch is that local minimizers can be avoided (by allowing increase
   in function value), and typically, fewer iterations are performed in
   the main code.

   The reference value is chosen based upon some historical information
   consisting of function values for previous iterates.  The amount of
   historical information used is determined by the memory size where the
   memory is used to store the previous function values.  The memory is
   initialized to alpha*f(x^0) for some alpha >= 1, with alpha=1 signifying
   that we always force decrease from the initial point.

   The reference value can be the maximum value in the memory or can be
   chosen to provide some mean descent.  Elements are removed from the
   memory with a replacement policy that either removes the oldest
   value in the memory (FIFO), or the largest value in the memory (MRU).

   Additionally, we can add a watchdog strategy to the search, which
   essentially accepts small directions and only checks the nonmonotonic
   descent criteria every m-steps.  This strategy is NOT implemented in
   the code.

   Finally, care must be taken when steepest descent directions are used.
   For example, when the Newton direction does not satisfy a sufficient
   descent criteria.  The code will apply the same test regardless of
   the direction.  This type of search may not be appropriate for all
   algorithms.  For example, when a gradient direction is used, we may
   want to revert to the best point found and reset the memory so that
   we stay in an appropriate level set after using a gradient steps.
   This type of search is currently NOT supported by the code.

   References:
    Armijo, "Minimization of Functions Having Lipschitz Continuous
      First-Partial Derivatives," Pacific Journal of Mathematics, volume 16,
      pages 1-3, 1966.
    Ferris and Lucidi, "Nonmonotone Stabilization Methods for Nonlinear
      Equations," Journal of Optimization Theory and Applications, volume 81,
      pages 53-71, 1994.
    Grippo, Lampariello, and Lucidi, "A Nonmonotone Line Search Technique
      for Newton's Method," SIAM Journal on Numerical Analysis, volume 23,
      pages 707-716, 1986.
    Grippo, Lampariello, and Lucidi, "A Class of Nonmonotone Stabilization
      Methods in Unconstrained Optimization," Numerische Mathematik, volume 59,
      pages 779-805, 1991. */
#include <petsc/private/taolinesearchimpl.h>
typedef struct {
  PetscReal *memory;

  PetscReal alpha;                      /* Initial reference factor >= 1 */
  PetscReal beta;                       /* Steplength determination < 1 */
  PetscReal beta_inf;           /* Steplength determination < 1 */
  PetscReal sigma;                      /* Acceptance criteria < 1) */
  PetscReal minimumStep;                /* Minimum step size */
  PetscReal lastReference;              /* Reference value of last iteration */

  PetscInt memorySize;          /* Number of functions kept in memory */
  PetscInt current;                     /* Current element for FIFO */
  PetscInt referencePolicy;             /* Integer for reference calculation rule */
  PetscInt replacementPolicy;   /* Policy for replacing values in memory */

  PetscBool nondescending;
  PetscBool memorySetup;

  Vec x;        /* Maintain reference to variable vector to check for changes */
  Vec work;
} TaoLineSearch_ARMIJO;

#endif
