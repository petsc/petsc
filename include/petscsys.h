/* $Id: sys.h,v 1.8 1995/10/24 21:55:05 bsmith Exp bsmith $ */
#if !defined(__SYS_PACKAGE)
#define __SYS_PACKAGE

extern int    SYGetArchType(char*,int);
extern int    SYIsort(int,int*);
extern int    SYIsortperm(int,int*,int*);
extern int    SYDsort(int,double*);
extern char   *SYGetDate();
extern int    TrDebugLevel(int);
extern int    TrValid();
#endif      

