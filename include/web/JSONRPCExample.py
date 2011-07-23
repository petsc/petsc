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

commname = 'jeff'

class JSONRPCExample:
    def onModuleLoad(self):
        self.TEXT_WAITING = "Waiting for response..."
        self.TEXT_ERROR = "Server Error"
        self.METHOD_ECHO = "echo"
        self.METHOD_REVERSE = "Reverse"
        self.METHOD_UPPERCASE = "UPPERCASE"
        self.METHOD_LOWERCASE = "lowercase"
        self.METHOD_NONEXISTANT = "Non existant"
        self.methods = [self.METHOD_ECHO, self.METHOD_REVERSE, 
                        self.METHOD_UPPERCASE, self.METHOD_LOWERCASE, 
                        self.METHOD_NONEXISTANT]
        self.remote_py = ServicePython()
        global commname
        commname  = 'heff'
        self.status=Label()
        self.text_area = TextArea()
        self.text_area.setText("Simple text")
        self.text_area.setCharacterWidth(80)
        self.text_area.setVisibleLines(8)
        
        self.method_list = ListBox()
        self.method_list.setName("hello")
        self.method_list.setVisibleItemCount(1)
        for method in self.methods:
            self.method_list.addItem(method)
        self.method_list.setSelectedIndex(0)

        method_panel = HorizontalPanel()
        method_panel.add(HTML("Remote string method to call: "))
        method_panel.add(self.method_list)
        method_panel.setSpacing(8)

        self.button_py = Button("Send to echo Service", self)
        self.button_ams_connect = Button("Connect to AMS", self)
        self.button_ams_attach  = Button("Attach to AMS communicator", self)

        buttons = HorizontalPanel()
        buttons.add(self.button_py)
        buttons.add(self.button_ams_connect)
        buttons.add(self.button_ams_attach)
        buttons.setSpacing(8)
        
        info = """<h2>JSON-RPC Example</h2>
        <p>This example demonstrates the calling of server services with
           <a href="http://json-rpc.org/">JSON-RPC</a>.
        </p>
        <p>Enter some text below, and press a button to send the text
           to an echo service on your server. An echo service simply sends the exact same text back that it receives.
           </p>"""
        
        panel = VerticalPanel()
        panel.add(HTML(info))
        panel.add(self.text_area)
        panel.add(method_panel)
        panel.add(buttons)
        panel.add(self.status)
        
        RootPanel().add(panel)

    def onClick(self, sender):
        self.status.setText(self.TEXT_WAITING)
        method = self.methods[self.method_list.getSelectedIndex()]
        text = self.text_area.getText()

        # demonstrate proxy & callMethod()
        if sender == self.button_py:
            id = self.remote_py.YAML_echo(text, self)
        elif sender == self.button_ams_connect:
            id = self.remote_py.YAML_AMS_Connect(text, self)
        else:
            self.status.setText('trying comm attach'+commname)
            id = self.remote_py.YAML_AMS_Comm_attach(commname, self)
        self.lastsender = sender

    def onRemoteResponse(self, response, request_info):
        global commname
        self.status.setText(response)
        if self.lastsender == self.button_ams_connect:
            commname = str(response)
            self.status.setText('was ams connect'+commname)
        else:
            commname = "joe"

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
        JSONProxy.__init__(self, "No service name", ["YAML_echo", "YAML_AMS_Connect", "YAML_AMS_Comm_attach"])

if __name__ == '__main__':
    # for pyjd, set up a web server and load the HTML from there:
    # this convinces the browser engine that the AJAX will be loaded
    # from the same URI base as the URL, it's all a bit messy...
    pyjd.setup()
    app = JSONRPCExample()
    app.onModuleLoad()
    pyjd.run()

