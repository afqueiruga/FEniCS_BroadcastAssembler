
#include <memory>
#include <iterator>
#include <vector>

#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

#include "dolfin/fem/FiniteElement.h"
#include "dolfin/fem/Form.h"
#include "dolfin/fem/GenericDofMap.h"
#include "dolfin/fem/DofMap.h"
#include "dolfin/fem/SparsityPatternBuilder.h"
#include "dolfin/fem/Assembler.h"

#include "dolfin/fem/UFC.h"

#include "BroadcastAssembler.h"
#include "BroadcastDofMap.h"

using namespace dolfin;

BroadcastAssembler::BroadcastAssembler() {
  /// Should I take in the the tensor here?
}

/// TODO: local ownership....
void BroadcastAssembler::init_global_tensor(
    GenericTensor &newA, 
    const Array<int> & global_dim, int rank, 
    const MPI_Comm mpi_comm,
    const Array<int> & local_range,
    const Array<int> & off_process
    )
{
  A = &newA;
  // std::vector<std::size_t> global_dim_vec;
  // std::vector<std::pair<std::size_t, std::size_t> > local_range_vec;
  std::vector<const std::vector<std::size_t>* > local_to_global(rank);
  std::vector<std::vector<std::size_t> > tmp_local_to_global(rank);
  // std::vector<const std::vector<int>* > off_process_owner;
  for(int i=0; i<global_dim.size(); i++) {
    global_dim_vec.push_back( global_dim[i] );
    local_range_vec.push_back(std::pair<std::size_t, std::size_t>(local_range[2*i],local_range[2*i+1]));
    local_to_global[i] = &tmp_local_to_global[i];
    std::vector<int>tmp = std::vector<int>(off_process[0],off_process[0]+off_process.size());
    off_process_owner.push_back( &tmp );
  }

  if(A->empty()) {
    // Is it a new tensor?
    tensor_layout = A->factory().create_layout(rank);
    dolfin_assert(tensor_layout);
    // Block size has to be 1 to deal with all forms...
    tensor_layout->init(mpi_comm, global_dim_vec, 1, local_range_vec);
    // Initialize the sparsity pattern
    GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();
    pattern.init(mpi_comm, global_dim_vec, local_range_vec, local_to_global, off_process_owner,1);
    
  } else {
    // Make sure it's the right size
    if(A->rank() != rank) {
      dolfin_error("BroadcastAssembler.cpp",
		   "init_global_tensor",
		   "A does not rank");
    }
    for(int i=0; i<rank; i++) {
      if(A->size(i) != global_dim[i]) {
	dolfin_error("BroadcastAssembler.cpp",
		     "init_global_tensor",
		     "A does not match dim %d",i);
      }
    }
  }

}

void BroadcastAssembler::sparsity_form(const Form &a,
				       const GenericDofMap & mappeddof
				       )

{
  /// Build the form_dof maps:
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < a.rank(); ++i)
    dofmaps.push_back(&mappeddof);
  std::cout << dofmaps.size();

  GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();
  SparsityPatternBuilder::build(pattern,
                                a.mesh(), dofmaps,
                                a.ufc_form()->has_cell_integrals(),
                                a.ufc_form()->has_interior_facet_integrals(),
                                a.ufc_form()->has_exterior_facet_integrals(),
                                false,
				false,
				false);
}

void BroadcastAssembler::sparsity_cell_pair(const Form &a,
					    const Mesh & meshA, const GenericDofMap & mdofA,
					    const Mesh & meshB, const GenericDofMap & mdofB,
					    const Array<int>& pairs)
{

  GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();

  const std::size_t rank = a.rank();

  std::cout << "Inserting " << pairs.size()/2 << " pair entries \n";
  std::cout << "rank: " << rank << "\n";
  std::cout << "globdim: " << global_dim_vec[0] << ", " << global_dim_vec[1] << "\n";

  // Create vector to point to dofs
  std::vector<std::vector<dolfin::la_index>* > dofs(rank);
  std::vector<const std::vector<dolfin::la_index>* > dofsconst(rank);
  for(std::size_t i=0; i<rank; ++i)
    {
      dofs[i] = new std::vector<dolfin::la_index>();
      dofsconst[i] = (const std::vector<dolfin::la_index>* )dofs[i];
    }


  const std::vector<unsigned int>& cellsA = meshA.cells();
  const std::vector<unsigned int>& cellsB = meshB.cells();
  std::cout << meshA.cells().size() << " vs. " << meshA.num_cells() << "\n";
  std::cout << meshB.cells().size() << " vs. " << meshB.num_cells() << "\n";

  for( int p=0; p<pairs.size(); p+= 2) {
    int ca = pairs[p];
    int cb = pairs[p+1];
    std::cout << pairs[p] << ", " << pairs[p+1] << "\n";
    for (std::size_t i = 0; i < rank; ++i)
      {
	const std::vector<dolfin::la_index>* first;
	const std::vector<dolfin::la_index>* second;

        first  = &mdofA[i]->cell_dofs(ca);
	second = &mdofB[i]->cell_dofs(cb);

	dofs[i]->clear();
	dofs[i]->insert(dofs[i]->end(),first->begin(),first->end());
	dofs[i]->insert(dofs[i]->end(),second->begin(),second->end());
	for( int d = 0; d < dofs[i]->size() ; ++d) {
	  std::cout << (*dofs[i])[d] << ", ";
	}
	std::cout << "\n";
      }
    tensor_layout->sparsity_pattern()->insert(dofsconst);
    
  }
}

void BroadcastAssembler::sparsity_apply()
{
  GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();
  pattern.apply();
  A->init(*tensor_layout);
}



void BroadcastAssembler::assemble_form(const Form &a,
				       const GenericDofMap & mappeddof)
{
  const Mesh& mesh = a.mesh();
  const std::size_t form_rank = a.rank();

  std::vector<std::vector<dolfin::la_index>* > dofs(form_rank);
  std::vector<const std::vector<dolfin::la_index>* > dofsconst(form_rank);

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(&mappeddof);

  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  UFC ufc_data(a);
  ufc::cell_integral* cell_integral = ufc_data.default_cell_integral.get();
  if(!cell_integral) return;

  for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      cell->get_cell_data(ufc_cell);
      cell->get_vertex_coordinates(vertex_coordinates);
      ufc_data.update(*cell, vertex_coordinates,ufc_cell);
      for(std::size_t i=0; i<form_rank; ++i)
	{
	  dofsconst[i] = &(dofmaps[i]->cell_dofs(cell->index()));
	}
      
      cell_integral->tabulate_tensor(ufc_data.A.data(), ufc_data.w(),
				vertex_coordinates.data(),
				ufc_cell.orientation);
      add_to_global_tensor(*A, ufc_data.A, dofsconst);
    }
}

void BroadcastAssembler::assemble_cell_pair(const Form &a,
			    const GenericDofMap & mdofA,
			    const GenericDofMap & mdofB,
			    const Array<int>& pairs)
{


}

//-----------------------------------------------------------------------------
void BroadcastAssembler::add_to_global_tensor(GenericTensor& A,
                                     std::vector<double>& cell_tensor,
                                     std::vector<const std::vector<dolfin::la_index>* >& dofs)
{
  A.add_local(&cell_tensor[0], dofs);
}
//-----------------------------------------------------------------------------



// const DofMap& BroadcastAssembler::map_to_giant(GenericDofMap & gdof, int offset)
// {
//   DofMap * dof = dynamic_cast<DofMap*>(&gdof);
//   std::vector<std::vector<dolfin::la_index> > _dofmap = dof->data();
//   std::vector<std::vector<dolfin::la_index> >::iterator it;
//   std::vector<dolfin::la_index>::iterator jt;
//   for (it = _dofmap.begin(); it != _dofmap.end(); ++it)
//     for (jt = it->begin(); jt != it->end(); ++jt) {
//       *jt += offset;
//       std::cout << *jt << "\n";
//     }
//   return *dof;
// }

// const GenericDofMap& BroadcastAssembler::map_to_giant(DofMap & dof, const Array<int>& giantmap)
// {
//   return dof;

// }
