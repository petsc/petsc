
/* static char vcid[] = "$Id: petscadic.c,v 1.1 1997/03/28 04:08:42 bsmith Exp bsmith $"; */

#if !defined(__ADPETSC_H)
#define __ADPETSC_H

struct _PetscADICFunction{
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
