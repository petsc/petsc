#if !defined(__MM_PACKAGE)
#define __MM_PACKAGE

/* Type of model */
typedef char *MMType;

#define MMEULER      "euler"
#define MMFP         "fp"
#define MMHYBRID_E   "hybrid_e"
#define MMHYBRID_F   "hybrid_f"
#define MMHYBRID_EF1 "hybrid_ef1"

typedef struct _p_MM* MM;

extern int MMCreate(MPI_Comm,MM*);
extern int MMSetType(MM,MMType);
extern int MMSetUp(MM);
extern int MMDestroy(MM);
extern int MMSetFromOptions(MM);
extern int MMGetType(MM,MMType*);
extern int MMPrintHelp(MM);
extern int MMView(MM,Viewer);
extern int MMSetOptionsPrefix(MM,char*);
extern int MMAppendOptionsPrefix(MM,char*);
extern int MMGetOptionsPrefix(MM,char**);

extern int MMGetNumberOfComponents(MM,int*);

extern FList MMList;
extern int MMRegisterAll(char *);
extern int MMRegisterDestroy(void);

extern int MMRegister_Private(char*,char*,char*,int(*)(MM));
#if defined(USE_DYNAMIC_LIBRARIES)
#define MMRegister(a,b,c,d) MMRegister_Private(a,b,c,0)
#else
#define MMRegister(a,b,c,d) MMRegister_Private(a,b,c,d)
#endif

/* 
   Temporary kluge for compatibility with Fortran
 */

typedef enum { MMEULER_INT, MMFP_INT, MMHYBRID_EF1_INT} MMTypeInt;

#endif
