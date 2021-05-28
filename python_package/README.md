## SQUIC for Python


if libSQUIC(.dylib/.so) is in home directory 
package will find it, otherwise set

```angular2
import pySQUIC as pyS
pyS.set_path("/path/to/libSQUIC")
```

To run tests:

```angular2
import pySQUIC.test 
pySQUIC.test.run_SQUIC_S()
pySQUIC.test.run_SQUIC()
pySQUIC.test.run_SQUIC_M()
```

For regular SQUIC calls:

```angular2
import pySQUIC as pyS
import pySQUIC.test
Y = pySQUIC.test.generate_sample(8, 200)
l = 0.5
pyS.SQUIC_S(Y,l)
pyS.SQUIC(Y,l)
```