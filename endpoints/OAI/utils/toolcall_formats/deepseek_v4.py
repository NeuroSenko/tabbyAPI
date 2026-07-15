import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
DeepSeek-V4 "DSML" pseudo-XML tool-call format. The DSML token is U+FF5C DSML U+FF5C
(the same full-width bars as the model's other special tokens).

Raw format:
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="__FUNCTION_NAME__">
    <｜DSML｜parameter name="__P1__" string="true">literal string value</｜DSML｜parameter>
    <｜DSML｜parameter name="__P2__" string="false">[1, 2, 3]</｜DSML｜parameter>
    </｜DSML｜invoke>
    ...
    </｜DSML｜tool_calls>

`string="true"` marks a literal string parameter; `string="false"` marks a JSON-encoded
value (number, bool, array, object, null).
"""

_D = "｜DSML｜"

TOOLCALL_START = "<" + _D + "tool_calls>"
TOOLCALL_END = "</" + _D + "tool_calls>"

_INVOKE = re.compile(
    re.escape("<" + _D + 'invoke name="') + r'([^"]+)">' + r"(.*?)" + re.escape("</" + _D + "invoke>"),
    re.DOTALL,
)
_PARAM = re.compile(
    re.escape("<" + _D + 'parameter name="')
    + r'([^"]+)"\s+string="(true|false)"\s*>'
    + r"(.*?)"
    + re.escape("</" + _D + "parameter>"),
    re.DOTALL,
)


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    for im in _INVOKE.finditer(text):
        func_name = im.group(1).strip()
        body = im.group(2)
        args: dict[str, any] = {}
        for pm in _PARAM.finditer(body):
            key = pm.group(1).strip()
            is_string = pm.group(2) == "true"
            raw = pm.group(3).strip()
            # string="true" is a literal; otherwise the value is JSON (fall back to string).
            args[key] = raw if is_string else coerce_param_value(raw)
        args_json = json.dumps(args, ensure_ascii=False)
        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"deepseek_v4: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
