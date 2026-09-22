# -*- coding: utf-8 -*-
"""Print the CHANGELOG entry for one version, dedented, for use as GitHub Release notes.

The CHANGELOG is a flat text file whose entries look like::

    v4.3.3  22.09.2026 -- first line of the entry
                          continuation lines, indented to the same column

Usage: python .github/changelog_extract.py 4.3.3
"""
import re
import sys

version = sys.argv[1]
lines = open("CHANGELOG", encoding="utf-8").read().splitlines()

start = next((i for i, l in enumerate(lines) if l.startswith("v%s " % version)), None)
if start is None:
    sys.exit("no CHANGELOG entry for version %s" % version)
end = next((i for i in range(start + 1, len(lines)) if re.match(r"^v\d", lines[i])), len(lines))

head = re.sub(r"^v\S+\s+\S+\s+--\s*", "", lines[start])  # strip "v4.3.3  22.09.2026 -- "
body = [head] + [l[22:] if l.startswith(" " * 22) else l.strip() for l in lines[start + 1 : end]]
while body and not body[-1].strip():
    body.pop()

print("\n".join(body))
print()
print("Install: `pip install modlamp==%s` -- https://pypi.org/project/modlamp/%s/" % (version, version))
