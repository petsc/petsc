/* $Id: sys.h,v 1.7 1995/09/04 17:26:02 bsmith Exp bsmith $ */
#if !defined(__SYS_PACKAGE)
#define __SYS_PACKAGE

extern int    SYGetArchType(char*,int);
extern int    SYIsort(int,int*);
extern int    SYIsortperm(int,int*,int*);
extern int    SYDsort(int,double*);
extern char   *SYGetDate();

#endif      

