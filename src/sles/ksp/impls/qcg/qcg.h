/* $Id: qcg.h,v 1.3 2001/08/06 21:16:48 bsmith Exp $ */

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
