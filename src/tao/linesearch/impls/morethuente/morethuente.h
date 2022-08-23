#ifndef __TAOLINESEARCH_MORETHUENTE_H
#define __TAOLINESEARCH_MORETHUENTE_H

typedef struct {
  PetscInt  bracket;
  PetscInt  infoc;
  PetscReal initstep;
  Vec       x; /* used to see if work needs to be reformed */
  Vec       work;

  PetscReal stx, fx, dgx;
  PetscReal sty, fy, dgy;

} TaoLineSearch_MT;

#endif
