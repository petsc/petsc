#ifndef __TAOLINESEARCH_MORETHUENTE_H
#define __TAOLINESEARCH_MORETHUENTE_H

typedef struct {
  double maxstep;	     /* maximum step size */
  double rtol;		     /* relative tol for acceptable step (rtol>0) */
  double ftol;		     /* tol for sufficient decr. condition (ftol>0) */
  double gtol;		     /* tol for curvature condition (gtol>0)*/
  double stepmin;	     /* lower bound for step */
  double stepmax;	     /* upper bound for step */
  PetscInt    maxfev;	     /* maximum funct evals per line search call */
  PetscInt    nfev;	     /* number of funct evals per line search call */
  PetscInt    bracket;
  PetscInt    infoc;
  PetscTruth  setupcalled;
  Vec Work;

} TAOLINESEARCH;

#endif
