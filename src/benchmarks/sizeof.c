#include <stdio.h>
/*
        Prints the size of various C data types
*/
int main(int argc,char *args)
{
  printf("Long Double %d\n",sizeof(long double));
  printf("double %d\n",sizeof(double));
  printf("int %d\n",sizeof(int));
  printf("char %d\n",sizeof(char));
  printf("short %d\n",sizeof(short));
  printf("long %d\n",sizeof(long));
  printf("long long %d\n",sizeof(long long));
  printf("int * %d\n",sizeof(int*));

  return 0;
}
