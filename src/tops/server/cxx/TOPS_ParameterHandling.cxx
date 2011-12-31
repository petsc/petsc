// This file is not generated from SIDL, it's for internal impl use only

#include <iostream>
#include "TOPS_ParameterHandling.hxx"

void processTOPSOptions(std::string options) {
  std::string key = "", val = ""; 
  bool inKey = true, inVal = false, newOption = true;
  int len = options.length();
  int start=0, end=len;

  // options can be in the form "options"
  if (options[0] == '"' && options[len-1] =='"') {
    start = 1; end = len-1;
  }

  for (int i = start; i <= end; ++i) {
    if ((i == end) || (newOption && (options[i] == '-'))) { 
      //std::cout << "Setting petsc option: " << key << " " << val << std::endl;
      if (key != "") {
      	if (val != "") PetscOptionsSetValue(key.c_str(), val.c_str());
      	else PetscOptionsSetValue(key.c_str(), 0);
      }
      inKey = true; inVal = false; 
      key = "-"; val = ""; continue; 
    } else if (options[i] == ' ') { 
      newOption = true; inKey = false; inVal = false;
    } else {
      newOption = false;
      if (inKey) key += options[i];   
      else { inVal = true; val += options[i]; }
    }
  }
}

void TOPSPetscOptionsMonitor(const char name[], const char value[], void* ctxt) {
}
