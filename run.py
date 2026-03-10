import py_compile
try:
    py_compile.compile('app.py', doraise=True)
    print("Syntax OK")
except Exception as e:
    print(e)
