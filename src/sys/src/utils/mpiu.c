
#include "petsc.h"
#include <stdio.h>
#include <stdarg.h>
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



