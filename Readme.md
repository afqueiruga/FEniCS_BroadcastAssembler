BroadcastAssembler for FEniCS.

Routines with more control over assembly.
It can be given an arbitrarily sized matrix and will 
assemble forms onto the DOFs its told.
It can also handle contacts via special integrals 
(requires afq's patches of FFC and Dolfin.)
