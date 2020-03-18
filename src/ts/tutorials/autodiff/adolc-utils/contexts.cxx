#include <petscsys.h>

/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/

#ifndef ADOLCCTX
#define ADOLCCTX
typedef struct {
  /* No ADOL-C annotation */
  PetscBool   no_an;

  /* Compressed Jacobian computation */
  PetscBool   sparse,sparse_view,sparse_view_done;
  PetscScalar **Seed,**Rec,*rec;
  PetscInt    p;

  /* Matrix dimensions, number of parameters */
  PetscInt    m,n,num_params;
} AdolcCtx;
#endif

/* Matrix (free) context */
#ifndef MATCTX
#define MATCTX
typedef struct {
  PetscReal time;
  Vec           X;
  Vec           Xdot;
  Vec           localX0;
  PetscReal     shift;
  PetscInt      m,n;
  PetscInt      tag1,tag2;
  TS            ts;
  PetscBool     flg;
  PetscLogEvent event1,event2,event3,event4;
} AdolcMatCtx;
#endif
