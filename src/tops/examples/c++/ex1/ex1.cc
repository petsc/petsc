#include "sidl.hh"
#include "TOPS.hh"
#include "server/c++/Ex1.hh"
#include <iostream>

int main(int argc,char **argv)
{
  TOPS::Solver solver = TOPS::StructuredSolver::_create();
  TOPS::System::System system = Ex1::System::_create();

  solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
  solver.setSystem(system);
  solver.solve();

  return 0;
}

