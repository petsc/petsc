#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>


#ifndef ADOLCCTX
#define ADOLCCTX
typedef struct {
  /* Zero Order Scalar (ZOS) test */
  PetscBool   zos,zos_view;

  /* No ADOL-C annotation */
  PetscBool   no_an;

  /* Compressed Jacobian computation */
  PetscBool   sparse,sparse_view,sparse_view_done;
  PetscScalar **Seed,**Rec,*rec;
  PetscInt    p;

  /* Matrix dimensions */
  PetscInt    m,n;
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
} MatCtx;
#endif
