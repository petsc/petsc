/* $Id: qcg.h,v 1.1 1995/07/20 21:09:50 curfman Exp bsmith $ */

/*
    Context for using preconditioned CG to minimize a quadratic function 
 */

#ifndef __QCG
#define __QCG

typedef struct {
  double quadratic;
  double ltsnrm;
  double delta;
} KSP_QCG;

#endif
