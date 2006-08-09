#include "sidl.hxx"
#include "TOPS.hxx"
#include "server/cxx/glue/Ex4.hxx"
#include <iostream>

int main(int argc,char **argv)
{
  TOPS::Solver solver = TOPS::UnstructuredSolver::_create();
  TOPS::System::System system = Ex4::System::_create();

  solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
  solver.setSystem(system);
  solver.solve();

  return 0;
}

