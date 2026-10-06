"""PSLG-NILM test package.

Making ``tests`` a regular package guarantees that ``from tests.test_m3_...``
resolves to this directory even when a third-party ``tests`` package exists in
site-packages: a regular package on the current working directory always wins
over a namespace package from the interpreter's search path.
"""
