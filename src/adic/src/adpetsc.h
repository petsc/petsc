
/* static char vcid[] = "$Id: adpetsc.h,v 1.2 1997/05/23 17:20:26 balay Exp balay $"; */

#if !defined(__ADPETSC_H)
#define __ADPETSC_H

struct _p_PetscADICFunction{
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
