#include <stdio.h>
/*
        Prints the size of various C data types
*/
int main(int argc,char *args)
{
  fprintf(stdout,"long double : %lu\n",(unsigned long)sizeof(long double));
  fprintf(stdout,"double      : %lu\n",(unsigned long)sizeof(double));
  fprintf(stdout,"int         : %lu\n",(unsigned long)sizeof(int));
  fprintf(stdout,"char        : %lu\n",(unsigned long)sizeof(char));
  fprintf(stdout,"short       : %lu\n",(unsigned long)sizeof(short));
  fprintf(stdout,"long        : %lu\n",(unsigned long)sizeof(long));
  fprintf(stdout,"long long   : %lu\n",(unsigned long)sizeof(long long));
  fprintf(stdout,"int *       : %lu\n",(unsigned long)sizeof(int*));
  fprintf(stdout,"size_t      : %lu\n",(unsigned long)sizeof(size_t));

  return 0;
}
