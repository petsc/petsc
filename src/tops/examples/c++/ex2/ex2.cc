#include "sidl.hh"
#include "TOPS.hh"
#include "server/c++/Ex2.hh"
#include <iostream>

int main(int argc,char **argv)
{
  TOPS::Solver solver = TOPS::Solver_Structured::_create();
  TOPS::System system = Ex2::System::_create();

  solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
  solver.setSystem(system);
  solver.solve();

  return 0;
}

