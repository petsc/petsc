#   
#   Run python pyjsbuild --output . JSONRPCExample.py to generate the needed HTML and Javascript
#

import pyjd 

from pyjamas.ui.RootPanel import RootPanel
from pyjamas.ui.TextArea import TextArea
from pyjamas.ui.Label import Label
from pyjamas.ui.Button import Button
from pyjamas.ui.HTML import HTML
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.ListBox import ListBox
from pyjamas.JSONService import JSONProxy

commname = 'Set at program initialization'

class JSONRPCExample:
    def onModuleLoad(self):
        self.TEXT_WAITING = "Waiting for response..."
        self.TEXT_ERROR = "Server Error"
        self.remote_py = ServicePython()
        global commname
        commname  = 'Set in onModuleLoad()'
        self.status=Label()
        self.text_area = TextArea()
        self.text_area.setText("Simple text")
        self.text_area.setCharacterWidth(80)
        self.text_area.setVisibleLines(8)
        
        self.button_echo = Button("Send to echo Service", self)
        self.button_ams_memlist = Button("Get AMS memory list", self)

        buttons = HorizontalPanel()
        buttons.add(self.button_echo)
        buttons.add(self.button_ams_memlist)
        buttons.setSpacing(8)
        
        info = """<h2>JSON-RPC Example</h2>
        <p>This example demonstrates the calling of AMS server in PETSc with <a href="http://json-rpc.org/">JSON-RPC</a>.
        </p>"""
        
        panel = VerticalPanel()
        panel.add(HTML(info))
        panel.add(self.text_area)
        panel.add(buttons)
        panel.add(self.status)
        
        RootPanel().add(panel)

    def onClick(self, sender):
        self.status.setText(self.TEXT_WAITING)
        text = self.text_area.getText()

        if sender == self.button_echo:
            id = self.remote_py.YAML_echo(text, self)
        elif sender == self.button_ams_memlist:
            id = self.remote_py.YAML_AMS_Connect(text, self)

    def onRemoteResponse(self, response, request_info):
        self.status.setText(response)
        method = str(request_info.method)
        if method == "YAML_AMS_Connect":
            id = self.remote_py.YAML_AMS_Comm_attach(str(response),self)
        elif method == "YAML_AMS_Comm_attach":
            id = self.remote_py.YAML_AMS_Comm_get_memory_list(str(response),self)
        elif method == "YAML_AMS_Comm_get_memory_list":
            pass


    def onRemoteError(self, code, errobj, request_info):
        # onRemoteError gets the HTTP error code or 0 and
        # errobj is an jsonrpc 2.0 error dict:
        #     {
        #       'code': jsonrpc-error-code (integer) ,
        #       'message': jsonrpc-error-message (string) ,
        #       'data' : extra-error-data
        #     }
        message = errobj['message']
        if code != 0:
            self.status.setText("HTTP error %d: %s" %(code, message))
        else:
            code = errobj['code']
            self.status.setText("JSONRPC Error %s: %s" %(code, message))



class ServicePython(JSONProxy):
    def __init__(self):
        JSONProxy.__init__(self, "No service name", ["YAML_echo", "YAML_AMS_Connect", "YAML_AMS_Comm_attach", "YAML_AMS_Comm_get_memory_list"])

if __name__ == '__main__':
    # for pyjd, set up a web server and load the HTML from there:
    # this convinces the browser engine that the AJAX will be loaded
    # from the same URI base as the URL, it's all a bit messy...
    pyjd.setup()
    app = JSONRPCExample()
    app.onModuleLoad()
    pyjd.run()

