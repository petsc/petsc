/* $Id: qcg.h,v 1.1 1995/07/20 16:49:20 curfman Exp curfman $ */

/*
    Context for using preconditioned CG to minimize a quadratic function 
 */

#ifndef __QCG
#define __QCG

typedef struct {
  double quadratic;
  double ltsnrm;
  double delta;
  int    info;
} KSP_QCG;

#endif
