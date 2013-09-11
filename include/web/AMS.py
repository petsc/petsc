#
#   Defines an interface from Pyjs to the AMS memory snooper.  This version downloads ALL the memories and fields when the AMS_Comm is attached
#  thus all memory and field value inquiries are local and immediate and do not involve accessing the publisher. To get fresh values one simply
#  creates another AMS_Comm() which makes a freash connection to the publisher
#

import pyjd
from pyjamas.JSONService import JSONProxy

args   = {}  # Arguments to each remote call
sent   = 0   # Number of calls sent to server
recv   = 0   # Number of calls received from server

class ServicePython(JSONProxy):
    def __init__(self):
        JSONProxy.__init__(self, "No service name", ["YAML_echo", "YAML_AMS_Connect", "YAML_AMS_Comm_attach", "YAML_AMS_Comm_get_memory_list","YAML_AMS_Memory_attach","YAML_AMS_Memory_get_field_list","YAML_AMS_Memory_get_field_info","YAML_AMS_Memory_set_field_info","YAML_AMS_Memory_update_send_begin"])

# ---------------------------------------------------------
class AMS_Memory(JSONProxy):
    def __init__(self,comm,memory):
        global args,sent,recv
        self.comm             = comm
        self.name             = memory   # string name of memory
        self.fieldlist        = []
        self.fields           = {}
        self.remote           = ServicePython()
        id = self.remote.YAML_AMS_Memory_attach(comm,memory,self)
        args[id] = ['YAML_AMS_Memory_attach',comm,memory]
        sent += 1

    def get_field_list(self,func = null):
        '''If called with func (net yet supported) then calls func asynchronously with latest memory list;
           otherwise returns current (possibly out-dated) memory list'''
        return self.fieldlist

    def get_field_info(self,field, func = null):
        '''Pass in string name of AMS field
           If called with func (not yet done) then first updates comm with latest field list and then calls func with field'''
        if not self.fields.has_key(field):
            return 'Memory does not have field named '+field
        return self.fields[field]

    def set_field_info(self,field,value,funct = null):
        '''Pass in string name of AMS field and value to be set back on publisher
           If called with func (not yet done) then first updates remove and then calls func with any error information'''
        if not self.fields.has_key(field):
            return 'Memory does not have field named '+field+' thus value not set'
        id = self.remote.YAML_AMS_Memory_set_field_info(self.memory,field,value,self)
        args[id] = ['YAML_AMS_Memory_set_field_info',comm,memory,field]

    def update_send_begin(self,funct = null):
        '''Tells the accessor to update the values in the memory on the publisher
           If called with func (not yet done) then first updates remove and then calls func with any error information'''
        id = self.remote.YAML_AMS_Memory_update_send_begin(self.memory,self)

    def onRemoteResponse(self, response, request_info):
        global args,sent,recv,statusbar
        recv += 1
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Memory_attach":
            self.memory = response[0]
            id = self.remote.YAML_AMS_Memory_get_field_list(self.memory,self)
            args[id] = ['YAML_AMS_Memory_get_field_list',self.memory]
            sent += 1
        elif method == "YAML_AMS_Memory_get_field_list":
            self.fieldlist = response
            if not isinstance(self.fieldlist,list): self.fieldlist = [self.fieldlist]
            for i in self.fieldlist:
                id = self.remote.YAML_AMS_Memory_get_field_info(self.memory,i,self)
                args[id] = ['YAML_AMS_Memory_get_field_info',self.memory,i]
                sent += 1
        elif method == "YAML_AMS_Memory_get_field_info":
            self.fields[args[rid][2]] = response
        elif method == "YAML_AMS_Memory_set_field_info":
          self.update_send_begin()
#          statusbar.setText("Value updated on server")


    def onRemoteError(self, code, errobj, request_info):
        global statusbar,recv
        recv += 1
        method = str(request_info.method)
        if method == "YAML_AMS_Memory_update_send_begin":
          self.comm.comm = -1;
#          statusbar.setText("Publisher is no longer accessable")
        else:
#          statusbar.setText("Error "+str(errobj))
          pass

# ---------------------------------------------------------
class AMS_Comm(JSONProxy):
    def __init__(self):
        global args,sent,recv
        self.comm             = -1
        self.commname         = ''
        self.memlist          = []
        self.memories         = {}
        self.memory_list_func = null
        self.remote           = ServicePython()
        id = self.remote.YAML_AMS_Connect('No argument', self)
        args[id] = ['YAML_AMS_Connect']
        sent += 1

    def get_memory_list(self,func = null):
        '''If called with func then calls func asynchronously with latest memory list;
           otherwise returns current (possibly out-dated) memory list'''
        if func:
            self.memory_list_func = func
            id = self.remote.YAML_AMS_Comm_get_memory_list(self.comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',self.comm]
            sent += 1
        else:
            return self.memlist

    def memory_attach(self,memory,func = null):
        '''Pass in string name of AMS memory object
           If called with func (not yet done) then first updates comm with latest memory list and then calls func with memory'''
        return self.memories[memory]

    def onRemoteResponse(self, response, request_info):
        global args,sent,recv
        recv += 1
        method = str(request_info.method)
        rid    = request_info.id
        if method == "YAML_AMS_Connect":
            if isinstance(response,list):
#             Currently always connects to the zeroth communicator published, no way to connect to others.
              self.commname = str(response[0])
            else:
              self.commname = str(response)
            if self.commname == 'No AMS publisher running':
                 pass
            else:
                 id = self.remote.YAML_AMS_Comm_attach(self.commname,self)
                 args[id] = ['YAML_AMS_Comm_attach',self.commname]
                 sent += 1
        elif method == "YAML_AMS_Comm_attach":
            self.comm = str(response)
            id = self.remote.YAML_AMS_Comm_get_memory_list(self.comm,self)
            args[id] = ['YAML_AMS_Comm_get_memory_list',self.comm]
            sent += 1
        elif method == "YAML_AMS_Comm_get_memory_list":
            if not isinstance(response,list):response = [response]
            self.memlist = response
            for i in self.memlist:
                self.memories[i] = AMS_Memory(self.comm,i)
            if self.memory_list_func:
                self.memory_list_func(response)
                self.memory_list_func = null

    def onRemoteError(self, code, errobj, request_info):
        global statusbar,recv
        recv += 1
#        statusbar.setText("Error "+str(errobj))


