#!/usr/bin/python

from dolfin import *
import os
import numpy as np

srcdir = str(os.path.dirname(os.path.realpath(__file__))+"/src/")
header_file = open(srcdir+"/BroadcastAssembler.h", "r")
code = header_file.read()
header_file.close()
compiled_module = compile_extension_module(
    code=code, source_directory=srcdir, sources=["BroadcastAssembler.cpp"],
    include_dirs=[".",os.path.abspath(srcdir)],
    additional_declarations="""
%feature("notabstract") BroadcastAssembler;

%template(SizetVector) std::vector<std::size_t>;
"""
)

BroadcastAssembler = compiled_moduled.BroadcastAssembler()


def test():
    from IPython import embed
    from matplotlib import pylab as plt
    assem = compiled_module.BroadcastAssembler()
    
    me = UnitSquareMesh(5,5)
    V = FunctionSpace(me,"CG",1)
    u = TrialFunction(V)
    du = TestFunction(V)
    a = (inner(grad(u),grad(du))*dx)
    Fa = Form(a)

    me2 = UnitSquareMesh(2,10)
    V2 = FunctionSpace(me2,"CG",1)
    u2 = TrialFunction(V2)
    du2 = TestFunction(V2)
    a2 = (inner(grad(u2),grad(du2))*dx)
    Fa2 = Form(a2)

    mdof = MultiMeshDofMap()
    mdof.add(V.dofmap())
    mdof.add(V2.dofmap())
    mmfs = MultiMeshFunctionSpace()
    mdof.build(mmfs,np.array([],dtype=np.intc))

    
    A = Matrix() #PETScMatrix()
    
    N = mdof.global_dimension() 
    dim = np.array([N,N],dtype=np.intc)
    local_dofs = np.array([0,N,0,N],dtype=np.intc)

    assem.init_global_tensor(A,dim,2,0,local_dofs,mdof.off_process_owner())

    # assem.init_global_tensor(A,N,2,MPI_Comm(),local_dofs)
    
    assem.sparsity_form(Fa,mdof.part(0))
    assem.sparsity_form(Fa2,mdof.part(1))
    assem.sparsity_cell_pair(Fa,me,mdof.part(0),me2,mdof.part(1),np.array([1,1,2,2],dtype=np.intc))
    assem.sparsity_apply()
    
    assem.assemble_form(Fa,mdof.part(0))
    assem.assemble_form(Fa2,mdof.part(1))
    assem.assemble_cell_pair(Fa,me,mdof.part(0),me2,mdof.part(1),np.array([1,1,2,2],dtype=np.intc))

    A.apply('add')
    embed()
    
    # print A.size(0)
    # print assem.A.size(0)


    # pairs = np.array([1,25, 25, 30],dtype=np.intc)
    
    # MTB.build_cell_pair_sparsity(A,Fa,pairs)
    # MTB.apply_sparsity_pattern(A)

    # # assemble(Fa,tensor=A,finalize_tensor=False,add_values=True)
    # MTB.assemble_cell_pair(A,Fa,pairs)
    # # A.add(np.array([77.7],dtype=np.double),np.array([7],dtype=np.intc),np.array([3],dtype=np.intc))
    # A.apply("add")
    
    # plt.spy(A.array())
    # plt.show()
    # embed()


if __name__=="__main__":
    test()
