/* $Id: sys.h,v 1.5 1995/06/07 16:30:27 bsmith Exp bsmith $ */
#if !defined(__SYS_PACKAGE)
#define __SYS_PACKAGE

extern void   SYGetArchType(char*,int);
extern int    SYIsort(int,int*);
extern int    SYIsortperm(int,int*,int*);
extern int    SYDsort(int,double*);

#endif      

