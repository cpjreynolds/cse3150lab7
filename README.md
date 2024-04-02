# CSE3150 - Lab 7

## Compilation

`make all` to compile both testing and `main()` binaries.

`make check` to compile and run tests.

`make run` to compile and execute `./lab7.out`

If you have compilation errors relating to `format` or `views::zip`, compile
with NOFORMAT=1 and NOZIP=1 respectively.

e.g.: `make run NOFORMAT=1`

### Usage

`./lab7.out <input-file>` where `<input-file>` defaults to `test.txt` if not
specified.
