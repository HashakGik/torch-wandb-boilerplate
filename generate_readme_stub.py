"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import utils
import os
import re
from subprocess import check_output

# This script automatically generates a README.md file, filling it with a table of hyper-parameters and the list of
# requirements. If a README.md is already present, it throws an error and terminates.
# For it to work correctly, pipreqs must be installed, and hyper-parameters must be handled by and argparse.ArgumentParser instance.
# Note: when pushing your project, either delete this script or .gitignore it!


def parse_arg(a):
    if a.default == "==SUPPRESS==":
        return None

    name = "--{}".format(a.dest)

    if a.default is None:
        default = ""
    else:
        default = str(a.default)

    if isinstance(a.type, type(str)):
        if a.choices is None:
            values = "str"
        else:
            values = ", ".join(a.choices)
    elif isinstance(a.type, utils.ArgBoolean):
        values = "bool"
    elif isinstance(a.type, utils.ArgNumber):
        values = str(a.type._ArgNumber__number_type.__name__)
    else:
        values = str(a.type)

    description = a.help
    description = re.sub(r"default:[^)]*", "", description)
    description = re.sub(r"\(\W*\)", "", description)
    description = re.sub(r";\W*\)", ")", description)
    if values != "str": # Remove alternatives, only if already specified as choices (which is not always possible, eg. in "mlp:N").
        description = re.sub(r" in \{\}", "", description)
    else: # If that is not the case, first extract choices, then remove them from descriptions.
        m = re.search(r"\{(.*)\}", description)
        if m is not None:
            tmp = [t.strip().replace("'", "") for t in m.groups()[0].split(",")]
            values = ", ".join(tmp)
            description = re.sub(r" in \{.*\}", "", description)

    description = description.strip()

    return [name, values, default, description]

def format_table(params):
    longest_val = [0, 0, 0, 0]
    for p in params:
        for i in range(len(p)):
            if len(p[i]) > longest_val[i]:
                longest_val[i] = len(p[i])

    header = ["Argument", "Values", "Default", "Description"]
    table = "| " + " | ".join(["{{:<{}}}".format(longest_val[i]).format(h) for i, h in enumerate(header)]) + " |\n"
    table += "|{}|{}|{}|{}|\n".format(*["-" * (l+2) for l in longest_val])
    for p in params:
        table += "| " + " | ".join(["{{:<{}}}".format(longest_val[i]).format(pp) for i, pp in enumerate(p)]) + " |\n"

    return table

if __name__ == "__main__":
    ap = utils.get_arg_parser()
    dependencies = check_output(["pipreqs", ".", "--print"])

    params = []
    for a in ap._actions:
        aa = parse_arg(a)
        if aa is not None:
            params.append(aa)

    # Markdown stub for the readme file.
    markdown = """
# Project name
This is the code for our paper [Name](https://duckduckgo.com).
If applicable also link [datasets](url).

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Description
TODO: Short description of the project...

## Requirements

```
{}
```

TODO: Additional requirements to be installed manually...

## Usage

{}
""".format(dependencies.decode("utf-8"), format_table(params))



    assert not os.path.exists("README.md"), "README.md already exists. Refusing to overwrite it."

    with open("README.md", "w") as file:
        file.write(markdown)
