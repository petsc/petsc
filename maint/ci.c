#include <sys/param.h>
#include <string.h>
int main(int argc, char **argv)
{
  int i, len;
  char resolved_path[MAXPATHLEN];
  char path[MAXPATHLEN];
  char command[4*MAXPATHLEN];
  char *c =NULL;

  
  /* Print the comand as it was entered */
  for (i =0; i < argc; i++)
    printf("%s ",argv[i]);
  printf("\n");
  
  /* Assume the last arfument is the filename */
  strcpy(path, argv[argc-1]);
  
  /* Find the actual file corresponding to the link */
  realpath(path, resolved_path);
  printf("resolved file: %s\n",resolved_path);
  
  /* Now extrext the dir so that I can "cd" to it */
  strcpy(path, resolved_path);
  c = strstr(path,"RCS");
  
  /* If "RCS" is not in the path, find the dir by brute force. 
     Travel the string up to the last occurence of "/" */
  if ( c != NULL) { *c = NULL; }
  else {
    len = strlen(path) -1;
    while ((len >= 0 ) &&  ( path[len]  != '/' )) len --;
    path[len+1] = NULL;
  }


  strcpy(command,"cd ");
  strcat(command, path );
  strcat(command,"; /usr/local/bin/ci ");

  for (i=1; i< argc-1; i++) {
    /* if -m option is entered, take it away, and also the string
       associated with it */
    if (!strncmp( argv[i], "-m", 2)) { i++; }
    else {
      strcat(command, argv[i]);
      strcat (command, " ");
    }
  }
    
  strcat (command, resolved_path);
  printf("command:%s\n",command);
  system(command);
  return 0;

}
    
