#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
//#include <python2.7/Python.h>

using namespace std;

#include "revolve.h"

Revolve *r = NULL;

extern "C" int wrap_revolve(int* check,int* capo,int* fine,int *snaps_in,int* info, int *rank)
  {
    enum ACTION::action whatodo;
    int snaps = *snaps_in;
    int ret=-1;

    if (! r )
    {
      r = new Revolve(*fine,snaps,*info, *rank);

    }

    whatodo = r->revolve(check, capo, fine, snaps, info);
    switch(whatodo) {
      case ACTION::advance:
        ret=1;
        break;
      case ACTION::takeshot:
        ret=2;
        break;
      case ACTION::firsturn:
        ret=3;
        break;
      case ACTION::youturn:
        ret=4;
        break;
      case ACTION::restore:
        ret=5;
        break;
      case ACTION::terminate:
        ret=6;
        break;
      case ACTION::error:
        ret=-1;
        break;
    }
  return ret;
  }

extern "C"
  int wrap_revolve_(int* check,int* capo,int* fine,int *snaps_in,int* info, int *rank)
  {
  return wrap_revolve(check,capo,fine,snaps_in,info, rank);
  }

extern "C" void wrap_revolve_reset() {
  delete r;
  r=NULL;

}

extern "C" void wrap_revolve_reset_() {
  delete r;
  r=NULL;

}
