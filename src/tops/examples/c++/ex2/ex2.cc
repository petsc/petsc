#include "sidl.hh"
#include "TOPS.hh"
#include "server/c++/Ex2.hh"
#include <iostream>

int main(int argc,char **argv)
{
  TOPS::State state = TOPS::State::_create();
  state.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));

  TOPS::Solver solver = TOPS::Solver_Structured::_create();
  TOPS::System system = Ex2::System::_create();

  solver.setSystem(system);
  solver.solve();

  return 0;
}

