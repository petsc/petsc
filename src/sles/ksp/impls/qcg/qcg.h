/* $Id: qcg.h,v 1.2 2000/01/11 21:02:12 bsmith Exp bsmith $ */

/*
    Context for using preconditioned CG to minimize a quadratic function 
 */

#ifndef __QCG
#define __QCG

typedef struct {
  PetscReal quadratic;
  PetscReal ltsnrm;
  PetscReal delta;
} KSP_QCG;

#endif
