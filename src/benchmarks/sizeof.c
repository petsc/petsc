#include <stdio.h>
/*
        Prints the size of various C data types
*/
int main(int argc,char *args)
{
  fprintf(stdout,"long double : %d\n",sizeof(long double));
  fprintf(stdout,"double      : %d\n",sizeof(double));
  fprintf(stdout,"int         : %d\n",sizeof(int));
  fprintf(stdout,"char        : %d\n",sizeof(char));
  fprintf(stdout,"short       : %d\n",sizeof(short));
  fprintf(stdout,"long        : %d\n",sizeof(long));
  fprintf(stdout,"long long   : %d\n",sizeof(long long));
  fprintf(stdout,"int *       : %d\n",sizeof(int*));

  return 0;
}
