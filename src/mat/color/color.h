/*$Id: color.c,v 1.42 1999/11/05 14:46:11 bsmith Exp $*/

#if !defined(_MINPACK_COLOR_H)
#define _MINPACK_COLOR_H

/*
     Prototypes for Minpack coloring routines 
*/
extern int MINPACKdegr(int *,int *,int *,int *,int *,int *,int *);
extern int MINPACKdsm(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *,int *,int *,int *,int *);
extern int MINPACKido(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *, int *, int *, int *, int *);
extern int MINPACKnumsrt(int *,int *,int *,int *,int *,int *,int *);
extern int MINPACKseq(int *,int *,int *,int *,int *,int *,int *,
                      int *,int *);
extern int MINPACKsetr(int*,int*,int*,int*,int*, int*,int*);
extern int MINPACKslo(int *,int *,int *,int *, int *, int *,int *,
                      int *, int *,int *,int *,int *);

#endif
