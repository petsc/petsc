/*
    Parses list of people who have ftped PETSc
*/
#include <stdio.h>
#include <string.h>
#include <strings.h>
void ToLower();

char (*Month[]) = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug",
                   "Sep","Oct","Nov","Dec"};
int Monthsize[] = {0,31,59,90,120,151,181,212,243,273,304,334};
main(argc,args)
int  argc;
char **args;
{
  FILE *fd;
  char *tmp,line[1024], month[12],time[12], address[100], package[100];
  int  p, cnt,i, n, mon, year,date[10000],day, j, wanted = 30;
  int  exactdate[10000];
  char name[10000][100],dow[3];

  fd = fopen(args[1],"r");
  if (!fd) {
    fprintf(stderr,"Could not open %s\n",args[1]);
    return;
  }
  if (argc > 2) sscanf(args[2],"%d",&wanted);
 
  n = 0;
  while (fgets(line,1024,fd)) {

    sscanf(line,"%s %s %d %s %d %s %s",dow,month,&day,time,&year,name+n,package);

    if (strstr(name+n,"bsmith"))       continue;
    if (strstr(name+n,"gropp"))        continue;
    if (strstr(name+n,"curfman"))      continue;
    if (strstr(name+n,"mcinnes"))      continue;
    if (n && !strcmp(name+n-1,name+n)) continue;
    if      (!strcmp(month,"Jan")) mon = 0;
    else if (!strcmp(month,"Feb")) mon = 1;
    else if (!strcmp(month,"Mar")) mon = 2;
    else if (!strcmp(month,"Apr")) mon = 3;
    else if (!strcmp(month,"May")) mon = 4;
    else if (!strcmp(month,"Jun")) mon = 5;
    else if (!strcmp(month,"Jul")) mon = 6;
    else if (!strcmp(month,"Aug")) mon = 7;
    else if (!strcmp(month,"Sep")) mon = 8;
    else if (!strcmp(month,"Oct")) mon = 9;
    else if (!strcmp(month,"Nov")) mon = 10;
    else if (!strcmp(month,"Dec")) mon = 11;
    else {
      fprintf(stderr,"Bad Month %s\n",month);
    }
    date[n] = 12*(year - 1995) + mon;
    exactdate[n] = 365*(year - 1995) + Monthsize[mon] + day;
    n++; 
  }
  printf("%%Total retrieves %d \n",n);
  printf("%%Month Year Number \n");
  j = 0;
  cnt = 0;
  for ( i=0; i<30; i++) {
    while (date[j] == i) {j++; cnt++; }
    if (cnt) {
      printf("%% %s    %d  %4d \n",Month[((i)%12)],1995+(i)/12,cnt);
    }
    cnt = 0;
    if ( j>= n) break;
  } 
  j = 0;
  cnt = 0;
  printf(" x = [\n");
  for ( i=150; i<800; i++) {
    while (exactdate[j] == i) {cnt++; j++;}
    printf("%d  %4d \n",i,cnt);
    cnt = 0;
    if ( j>= n) break;
  } 
  printf("];\n x = [x(:,2)*sum(x(:,2))/(5*max(x(:,2))),cumsum(x(:,2))];\n");
}
void ToLower(name)
char *name;
{
  int i,N = strlen(name); 
  for ( i=0; i<N; i++ ) {
     if (isupper(name[i])) name[i] = (char) tolower(name[i]);
  }
}
