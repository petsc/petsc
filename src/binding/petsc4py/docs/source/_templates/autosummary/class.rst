{{ fullname | escape | underline}}

{%- if autotype is defined %}
{%- set objtype = autotype.get(name) or objtype %}
{%- endif %}

.. auto{{ objtype }}:: {{ module}}.{{ objname }}
   :show-inheritance:

   {%- for item in ['__new__', '__init__'] %}
     {%- if item in members and item not in inherited_members %}
     {%- endif %}
   {%- endfor %}

   {%- for item in ['__new__', '__init__'] %}
     {%- if item in methods %}
       {%- set dummy = methods.remove(item) %}
     {%- endif %}
   {%- endfor %}

   {%- for item in inherited_members %}
     {%- if item in methods %}
       {%- set dummy = methods.remove(item) %}
     {%- endif %}
     {%- if item in attributes %}
       {%- set dummy = attributes.remove(item) %}
     {%- endif %}
   {%- endfor %}

   {%- set enumerations = [] %}
   {%- for item in members %}
     {%- if item not in inherited_members and item not in all_methods and item not in all_attributes %}
       {%- set dummy = enumerations.append(item) %}
     {%- endif %}
   {%- endfor %}

   {% block enumerations_documentation %}
   {%- if enumerations %}
   .. rubric:: Enumerations
   .. autosummary::
      :toctree:
   {%+ for item in enumerations %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block methods_summary %}
   {%- if methods %}
   .. rubric:: Methods Summary
   .. autosummary::
   {%+ for item in methods %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes_summary %}
   {%- if attributes %}
   .. rubric:: Attributes Summary
   .. autosummary::
   {%+ for item in attributes %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block methods_documentation %}
   {%- if methods %}
   .. rubric:: Methods Documentation
   {%+ for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes_documentation %}
   {%- if attributes %}
   .. rubric:: Attributes Documentation
   {%+ for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}
{# #}
