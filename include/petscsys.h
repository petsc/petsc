/* $Id: sys.h,v 1.9 1995/11/09 22:33:28 bsmith Exp bsmith $ */
/*
    Provides access to a small number of system related and general utility routines.
*/
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

