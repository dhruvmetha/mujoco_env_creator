#!/usr/bin/env python3
"""Inspect a generated MuJoCo XML and print geom/site information.

Read-only diagnostics: reports geom names, types, pos, size and sites.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Inspect XML geoms and sites")
    p.add_argument("xml", help="Path to XML file to inspect")
    args = p.parse_args()

    path = Path(args.xml)
    if not path.exists():
        print(f"XML not found: {path}")
        return

    tree = ET.parse(str(path))
    root = tree.getroot()

    print(f"Inspecting: {path}\n")

    print("Geoms:")
    for geom in root.iter('geom'):
        name = geom.get('name', '<noname>')
        gtype = geom.get('type', '<notype>')
        pos = geom.get('pos', '<n/a>')
        size = geom.get('size', '<n/a>')
        euler = geom.get('euler', geom.get('quat', '<n/a>'))
        print(f"  {name}: type={gtype} pos={pos} size={size} euler={euler}")

    print('\nSites:')
    for site in root.iter('site'):
        name = site.get('name', '<noname>')
        pos = site.get('pos', '<n/a>')
        size = site.get('size', '<n/a>')
        print(f"  {name}: pos={pos} size={size}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Inspect a generated MuJoCo XML and print geom/site information.

Read-only diagnostics: reports geom names, types, pos, size and sites.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Inspect XML geoms and sites")
    p.add_argument("xml", help="Path to XML file to inspect")
    args = p.parse_args()

    path = Path(args.xml)
    if not path.exists():
        print(f"XML not found: {path}")
        return

    tree = ET.parse(str(path))
    root = tree.getroot()

    print(f"Inspecting: {path}\n")

    print("Geoms:")
    for geom in root.iter('geom'):
        name = geom.get('name', '<noname>')
        gtype = geom.get('type', '<notype>')
        pos = geom.get('pos', '<n/a>')
        size = geom.get('size', '<n/a>')
        euler = geom.get('euler', geom.get('quat', '<n/a>'))
        print(f"  {name}: type={gtype} pos={pos} size={size} euler={euler}")

    print('\nSites:')
    for site in root.iter('site'):
        name = site.get('name', '<noname>')
        pos = site.get('pos', '<n/a>')
        size = site.get('size', '<n/a>')
        print(f"  {name}: pos={pos} size={size}")


if __name__ == '__main__':
    main()
