#if !defined(__SYS)
#define __SYS

extern int    SYArgGetInt();
extern int    SYArgGetDouble();
extern int    SYArgHasName();
extern int    SYArgGetString();
extern int    SYArgGetIntList();
extern int    SYArgGetIntVector();

extern int    SYGetCPUTime(); 
extern int    SYGetElapsedTime();
extern int    SYGetResidentSetSize();
extern int    SYGetPageFaults(); 


extern int    SYExit();
extern int    SYSetUserTimerRoutine();

extern int    SYGetDayTime( ), SYGetDate();

extern int    SYIsort(), SYDsort();

extern int    SYCreateRndm();
extern int    SYDRndm();
extern int    SYFreeRndm();

#endif      

