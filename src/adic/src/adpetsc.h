
/* static char vcid[] = "$Id: adpetsc.h,v 1.3 1997/05/23 18:35:23 balay Exp $"; */

#if !defined(__ADPETSC_H)
#define __ADPETSC_H

struct _n_PetscADICFunction{
  MPI_Comm comm;
  int      m,n;
  Vec      din, dout;
  int      (*FunctionInitialize)(void **);        /* user function initialize */
  int      (*Function)(Vec, Vec);                 /* user function */
  void     *ctx;                                  /* user function context */
  int      (*ad_FunctionInitialize)(void **);     /* user AD function initialize */
  int      (*ad_Function)(Vec, Vec);              /* user AD function */
  void     *ad_ctx;                               /* user function context */
};

#endif
