/*$Id: color.h,v 1.5 2000/05/10 16:41:33 bsmith Exp $*/

#if !defined(_MINPACK_COLOR_H)
#define _MINPACK_COLOR_H

/*
     Prototypes for Minpack coloring routines 
*/
EXTERN int MINPACKdegr(int *,int *,int *,int *,int *,int *,int *);
EXTERN int MINPACKdsm(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *,int *,int *,int *,int *);
EXTERN int MINPACKido(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *,int *,int *,int *,int *);
EXTERN int MINPACKnumsrt(int *,int *,int *,int *,int *,int *,int *);
EXTERN int MINPACKseq(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *);
EXTERN int MINPACKsetr(int*,int*,int*,int*,int*,int*,int*);
EXTERN int MINPACKslo(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *,int *,int *,int *);

#endif
