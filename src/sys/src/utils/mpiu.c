#ifndef lint
static char vcid[] = "$Id: try.c,v 1.6 1995/03/06 04:32:59 bsmith Exp bsmith $";
#endif
#include "petsc.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
/*@
    MPE_fopen - The first process in the communicator opens a file,
                all others do nothing.

  Input Parameters:
.  comm - the communicator
.  name - the filename
.  mode - usually "w"
@*/
FILE *MPE_fopen(MPI_Comm comm,char *name,char *mode)
{
  int  mytid;
  FILE *fd;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) fd = fopen(name,mode);
  else fd = 0;
  return fd;
}
/*@
     MPE_fclose - The first processor in the communicator closes a 
                  file, all others do nothing.

  Input Parameters:
.  comm - the communicator
.  fd - the file, opened with MPE_fopen()

@*/
int MPE_fclose(MPI_Comm comm,FILE *fd)
{
  int  mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) return fclose(fd);
  else return 0;
}

/*@
      MPE_fprintf - Single print to a file only from the first
                    processor in the communicator.

  Input Parameters:
.  comm - the communicator
.  fd - the file pointer
.  format - the usual printf() format string 
@*/
int MPE_fprintf(MPI_Comm comm,FILE* fd,char *format,...)
{
  int mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) {
    va_list Argp;
    va_start( Argp, format );
    vfprintf(fd,format,Argp);
    va_end( Argp );
  }
  return 0;
}
/*@
      MPE_printf - Single print to standard out, only from the first
                    processor in the communicator.

  Input Parameters:
.  comm - the communicator
.  format - the usual printf() format string 
@*/
int MPE_printf(MPI_Comm comm,char *format,...)
{
  int mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) {
    va_list Argp;
    va_start( Argp, format );
    vfprintf(stdout,format,Argp);
    va_end( Argp );
  }
  return 0;
}

#if defined(__cplusplus)
extern "C" {
#endif
extern char *getenv(char*);
#if defined(__cplusplus)
};
#endif

/*@
     MPE_Set_display - Tries to set the display variable for all processors.

  Input Parameters:
.   comm - the communicatior, probably MPI_COMM_WORLD

  Output Parameters:
.   display - the display string, may (and should) be freed.

@*/
int MPE_Set_display(MPI_Comm comm,char **display)
{
  int  MPI_Used,numtid,mytid,len;
  char *string,*str;
  MPI_Initialized(&MPI_Used);
  if (!MPI_Used) { *display = 0; return 0;}
  MPI_Comm_size(comm,&numtid);
  MPI_Comm_rank(comm,&mytid);  
  if (!mytid) {
    str = getenv("DISPLAY");
    if (!str || str[0] == ':') {
      string = (char *) MALLOC( 256*sizeof(char) ); CHKPTR(string);
      MPI_Get_processor_name(string,&len);
      *display = (char *) MALLOC( (5+len)*sizeof(char) ); CHKPTR(*display);
      strcpy(*display,string); FREE(string);
      strcat(*display,":0.0");
    }
    else {
      len = strlen(str);
      *display = (char *) MALLOC( (5+len)*sizeof(char) ); CHKPTR(*display);
      strcpy(*display,str);
    }
    len = strlen(*display);
    MPI_Bcast(&len,1,MPI_INT,0,comm);
    MPI_Bcast(*display,len,MPI_CHAR,0,comm);
  }
  else {
    MPI_Bcast(&len,1,MPI_INT,0,comm);
    *display = (char *) MALLOC( (len+1)*sizeof(char) ); CHKPTR(*display);
    MPI_Bcast(*display,len,MPI_CHAR,0,comm);
    (*display)[len] = 0;
  }
  return 0;  
}
