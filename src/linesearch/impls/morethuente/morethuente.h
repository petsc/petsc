#ifndef __TAOLINESEARCH_MORETHUENTE_H
#define __TAOLINESEARCH_MORETHUENTE_H

typedef struct {
  PetscInt    bracket;
  PetscInt    infoc;
  Vec x; // used to see if work needs to be reformed
  Vec work;

} TAOLINESEARCH_MT_CTX;

#endif
