#if !defined(__MM_PACKAGE)
#define __MM_PACKAGE

/* Type of model */
typedef enum { MMEULER, MMFP, MMHYBRID_EF1, MMHYBRID_E, MMHYBRID_F, MMNEW } MMType;

typedef struct _p_MM* MM;

extern int MMCreate(MPI_Comm,MM*);
extern int MMSetType(MM,MMType);
extern int MMSetUp(MM);
extern int MMRegister(MMType,MMType*,char *,int (*)(MM));
extern int MMRegisterDestroy();
extern int MMRegisterAll();
extern int MMRegisterAllCalled;
extern int MMDestroy(MM);
extern int MMSetFromOptions(MM);
extern int MMGetType(MM,MMType*,char**);
extern int MMPrintHelp(MM);
extern int MMView(MM,Viewer);
extern int MMSetOptionsPrefix(MM,char*);
extern int MMAppendOptionsPrefix(MM,char*);
extern int MMGetOptionsPrefix(MM,char**);

extern int MMGetNumberOfComponents(MM,int*);

#endif
