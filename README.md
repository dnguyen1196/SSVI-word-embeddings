


# Common problems

libcublas.so.9.0: cannot open shared object file: No such file or directory

First make sure that the available cuda matches with the available tensorflow
Depending on the version of tensorflow, it will ask for different cuda library
(Downgrade or upgrade if necessary either cuda or tensorflow)
(Also remember to do export path)


