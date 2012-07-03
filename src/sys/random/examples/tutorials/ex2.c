static char help[] = "Tests PetscRandom functions.\n\n";

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>

#include <petscsys.h>

#define PETSC_MAXBSIZE     40
#define PI           3.1415926535897
#define DATAFILENAME "ex2_stock.txt"

struct himaInfoTag {
  int           n;
  double        r;
  double        dt;
  int           totalNumSim;
  double        *St0; 
  double        *vol; 
};
typedef struct himaInfoTag himaInfo;

/* function protype */
PetscErrorCode readData(MPI_Comm comm,himaInfo *hinfo);
double mcVal(double St, double r, double vol, double dt, double eps);
void exchange(double *a, double *b);
double basketPayoff(double vol[], double St0[], int n, double r,double dt, double eps[]);
void stdNormalArray(double *eps, int size,PetscRandom ran);
unsigned long divWork(int id, unsigned long num, int np);

/* 
   Contributed by Xiaoyan Zeng <zengxia@iit.edu> and Liu, Kwong Ip" <kiliu@math.hkbu.edu.hk>

   Example of usage: 
     mpiexec -n 4 ./ex2 -num_of_stocks 30 -interest_rate 0.4 -time_interval 0.01 -num_of_simulations 10000
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
    double         r,dt;
    int            n;
    unsigned long  i,myNumSim,totalNumSim,numdim;
    /* double         payoff; */
    double         *vol, *St0, x, totalx;
    int            np,myid;
    time_t         start,stop;
    double         *eps;
    himaInfo       hinfo;
    PetscRandom    ran;
    PetscErrorCode ierr;
    PetscBool      flg;

    PetscInitialize(&argc,&argv,(char *)0,help);
#if defined(PETSC_USE_COMPLEX)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This example does not work for scalar type complex\n");
#endif
    time(&start);
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&ran);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SPRNG)
    ierr = PetscRandomSetType(ran,PETSCSPRNG);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_RAND)
    ierr = PetscRandomSetType(ran,PETSCRAND);CHKERRQ(ierr);
#endif
    ierr = PetscRandomSetFromOptions(ran);CHKERRQ(ierr);

    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &np);CHKERRQ(ierr);     /* number of nodes */
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &myid);CHKERRQ(ierr);   /* my ranking */   

    ierr = PetscOptionsHasName(PETSC_NULL, "-check_generators", &flg);CHKERRQ(ierr);
    if (flg){
      ierr = PetscRandomGetValue(ran,(PetscScalar *)&r);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] rval: %g\n",myid,r);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);
    }
    
    hinfo.n           = 31;
    hinfo.r           = 0.04;
    hinfo.dt          = 1.0/12; /* a month as a period */
    hinfo.totalNumSim = 1000;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-num_of_stocks",&(hinfo.n),PETSC_NULL);CHKERRQ(ierr); 
    if (hinfo.n <1 || hinfo.n > 31) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only 31 stocks listed in stock.txt. num_of_stocks %D must between 1 and 31",hinfo.n);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-interest_rate",&(hinfo.r),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-time_interval",&(hinfo.dt),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-num_of_simulations",&(hinfo.totalNumSim),PETSC_NULL);CHKERRQ(ierr);

    n           = hinfo.n;
    r           = hinfo.r;
    dt          = hinfo.dt;
    totalNumSim = hinfo.totalNumSim;
    vol         = hinfo.vol = (double *)malloc(sizeof(double)*(2*n+1));
    St0         = hinfo.St0 = hinfo.vol + n;
    ierr = readData(PETSC_COMM_WORLD,&hinfo);CHKERRQ(ierr);

    numdim = n*(n+1)/2;
    if (numdim%2 == 1){
      numdim++;
    }
    eps = (double *)malloc(sizeof(double)*numdim);

    myNumSim = divWork(myid,totalNumSim,np);

    x = 0;
    for (i=0;i<myNumSim;i++){
        stdNormalArray(eps,numdim,ran);
        x += basketPayoff(vol,St0,n,r,dt,eps);
    }

    ierr = MPI_Reduce(&x, &totalx, 1, MPI_DOUBLE, MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    time(&stop);
    /* payoff = exp(-r*dt*n)*(totalx/totalNumSim);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Option price = $%.3f using %ds of %s computation with %d %s for %d stocks, %d trading period per year, %.2f%% interest rate\n",
     payoff,(int)(stop - start),"parallel",np,"processors",n,(int)(1/dt),r);CHKERRQ(ierr); */
    
    free(vol);
    free(eps);
    ierr = PetscRandomDestroy(&ran);CHKERRQ(ierr);
    PetscFinalize();   
    return 0;
}

void stdNormalArray(double *eps, int size, PetscRandom ran)
{
  int            i;
  double         u1,u2,t;
  PetscErrorCode ierr;

  for (i=0;i<size;i+=2){
    ierr = PetscRandomGetValue(ran,(PetscScalar*)&u1);CHKERRABORT(PETSC_COMM_WORLD,ierr);
    ierr = PetscRandomGetValue(ran,(PetscScalar*)&u2);CHKERRABORT(PETSC_COMM_WORLD,ierr);
    
    t = sqrt(-2*log(u1));
    eps[i] = t * cos(2*PI*u2);
    eps[i+1] = t * sin(2*PI*u2);
  }
}


double basketPayoff(double vol[], double St0[], int n, double r,double dt, double eps[])
{
  double Stk[PETSC_MAXBSIZE], temp;
  double payoff;
  int    maxk,i,j;
  int    pointcount=0;
    
  for (i=0;i<n;i++) {
    Stk[i] = St0[i];
  }

  for (i=0;i<n;i++){
    maxk = 0;
    for (j=0;j<(n-i);j++){
      Stk[j] = mcVal(Stk[j],r,vol[j],dt,eps[pointcount++]);
      if ((Stk[j]/St0[j]) > (Stk[maxk]/St0[maxk])){
        maxk = j;
      }
    }
    exchange(Stk+j-1,Stk+maxk);
    exchange(St0+j-1,St0+maxk);
    exchange(vol+j-1,vol+maxk);
  }
    
  payoff = 0;
  for (i=0;i<n;i++){
    temp = (Stk[i]/St0[i]) - 1 ;
    if (temp > 0) payoff += temp;
  }
  return payoff;
}

#undef __FUNCT__
#define __FUNCT__ "readData"
PetscErrorCode readData(MPI_Comm comm,himaInfo *hinfo)
{
  int            i;
  FILE           *fd;
  char           temp[50];
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  double         *v = hinfo->vol, *t = hinfo->St0; 
  int            num=hinfo->n;
    
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFOpen(PETSC_COMM_SELF,DATAFILENAME,"r",&fd);CHKERRQ(ierr);
    for (i=0;i<num;i++){
      fscanf(fd,"%s%lf%lf",temp,v+i,t+i);
    }
    fclose(fd);
  }
  ierr = MPI_Bcast(v,2*num,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] vol %g, ... %g; St0 %g, ... %g\n",rank,hinfo->vol[0],hinfo->vol[num-1],hinfo->St0 [0],hinfo->St0[num-1]);
  PetscFunctionReturn(0);
}

void exchange(double *a, double *b)
{
  double t;
    
  t = *a;
  *a = *b;
  *b = t;
}

double mcVal(double St, double r, double vol, double dt, double eps)
{
  return (St * exp((r-0.5*vol*vol)*dt + vol*sqrt(dt)*eps));
}

unsigned long divWork(int id, unsigned long num, int np)
{
  unsigned long numit;

  numit = (unsigned long)(((double)num)/np);
  numit++;
  return numit;
}


