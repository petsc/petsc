/* $Id: sys.h,v 1.6 1995/08/22 16:33:42 bsmith Exp bsmith $ */
#if !defined(__SYS_PACKAGE)
#define __SYS_PACKAGE

extern int   SYGetArchType(char*,int);
extern int    SYIsort(int,int*);
extern int    SYIsortperm(int,int*,int*);
extern int    SYDsort(int,double*);

#endif      

