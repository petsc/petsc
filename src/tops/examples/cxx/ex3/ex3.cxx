#include "sidl.hxx"
#include "TOPS.hxx"
#include "server/c++/Ex3.hxx"
#include <iostream>

int main(int argc,char **argv)
{
  TOPS::Solver solver = TOPS::StructuredSolver::_create();
  TOPS::System::System system = Ex3::System::_create();

  solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
  solver.setSystem(system);
  solver.solve();

  return 0;
}

