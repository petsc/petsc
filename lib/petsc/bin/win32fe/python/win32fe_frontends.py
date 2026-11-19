"""Tool registry and entry point for the Python port of win32fe (mirrors
create.cpp and win32fe.cpp)."""

from win32fe_tool import Tool
from win32fe_compiler import Cl

TOOLS = {
    "cl": Cl,
}


def main(argv):
    """Dispatch a named tool or handle standalone win32fe options."""
    if len(argv) > 1 and not argv[1].startswith("--") and argv[1] not in TOOLS:
        return Tool.unknown_tool(argv[1])
    tool = TOOLS.get(argv[1], Tool)() if len(argv) > 1 else Tool()
    tool.get_args(argv)
    ierr = tool.parse()
    if not ierr:
        ierr = tool.execute()
    return ierr
