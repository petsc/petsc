#if !defined(__SYS)
#define __SYS

extern int    SYGetCPUTime(); 
extern int    SYGetElapsedTime();
extern int    SYGetResidentSetSize();
extern int    SYGetPageFaults(); 


extern int    SYExit();
extern int    SYSetUserTimerRoutine();

extern int    SYGetDayTime( ), SYGetDate();

extern int    SYIsort(int,int*);
extern int    SYIsortperm(int,int*,int*);
extern int    SYDsort(int,double*);

extern int    SYCreateRndm();
extern int    SYDRndm();
extern int    SYFreeRndm();

#endif      

