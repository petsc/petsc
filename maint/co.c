#include <sys/param.h>

int main(int argc, char **argv)
{
  int i;
  char resolved_path[MAXPATHLEN];
  char path[MAXPATHLEN];
  char command[4*MAXPATHLEN];

  if (argc !=4) {
    printf(" Wrong no of arg: %d \n", argc);
    for (i =0; i < argc; i++)
      printf("%s ",argv[i]);
    printf("\n");
    return 1;
  }

  strcpy(path, argv[3]);
  realpath(path, resolved_path);
  sscanf(resolved_path,"%[^RCS]",path);


  strcpy(command,"cd ");
  strcat(command, path );
  strcat(command,"; /usr/local/bin/co ");
  strcat(command, argv[1]);
  strcat (command, " ");
  strcat(command, argv[2]);
  strcat (command, " ");
  strcat (command, resolved_path);
  printf("com:%s\n",command);
  system(command);
  return 0;
}
    
