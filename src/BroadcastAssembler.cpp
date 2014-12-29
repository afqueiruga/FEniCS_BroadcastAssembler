
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
// #include "BroadcastDofMap.h"

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
    std::vector<int>tmp = std::vector<int>();
    for(int j=0; j<off_process.size(); j++) {
      tmp.push_back(off_process[j]);
    }

    off_process_owner.push_back( &tmp );
  }

  if(A->empty()) {
    // Is it a new tensor?
    tensor_layout = A->factory().create_layout(rank);
    dolfin_assert(tensor_layout);

    // Block size has to be 1 to deal with all forms...
    tensor_layout->init(mpi_comm, global_dim_vec, 1, local_range_vec);

    // Initialize the sparsity pattern
    if(tensor_layout->sparsity_pattern()) {
      GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();

      pattern.init(mpi_comm, global_dim_vec, local_range_vec, local_to_global, off_process_owner,1);
    }

    
  } else {
    // Make sure it's the right size
    if(A->rank() != rank) {
      dolfin_error("BroadcastAssembler.cpp",
		   "init_global_tensor",
		   "A does not match in rank");
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

  // std::cout << "Inserting " << pairs.size()/2 << " pair entries \n";
  // std::cout << "rank: " << rank << "\n";
  // std::cout << "globdim: " << global_dim_vec[0] << ", " << global_dim_vec[1] << "\n";

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

  for( int p=0; p<pairs.size(); p+= 2) {
    int ca = pairs[p];
    int cb = pairs[p+1];
    for (std::size_t i = 0; i < rank; ++i)
      {
	const std::vector<dolfin::la_index>* first;
	const std::vector<dolfin::la_index>* second;

        first  = & mdofA.cell_dofs(ca);
	second = & mdofB.cell_dofs(cb);

	dofs[i]->clear();
	dofs[i]->insert(dofs[i]->end(),first->begin(),first->end());
	dofs[i]->insert(dofs[i]->end(),second->begin(),second->end());

      }
    // TODO: IDK?? Depends on how I define the mdof's
    pattern.insert_global(dofsconst);
    
  }
}

void BroadcastAssembler::sparsity_apply()
{
  if(tensor_layout->sparsity_pattern()) {
    GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();
    pattern.apply();
  }
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
					    const Mesh & meshA, const GenericDofMap & mdofA,
					    const Form &b,
					    const Mesh & meshB, const GenericDofMap & mdofB,
					    const Array<int>& pairs,
					    const Array<double>& chi,
					    const int chi_n_max)
{
  /* assumptions for now: chi is 3*chi_n_max * npairs */ 

  GenericSparsityPattern& pattern = * tensor_layout->sparsity_pattern();

  const std::size_t rank = a.rank();
  // std::cout << "Inserting " << pairs.size()/2 << " pair entries \n";
  // std::cout << "rank: " << rank << "\n";
  // std::cout << "globdim: " << global_dim_vec[0] << ", " << global_dim_vec[1] << "\n";

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

  std::size_t num_local_entries = 1;
  for(std::size_t i=0; i < rank ; i++) {
    num_local_entries *= 2*3; // TODO: extract this somehow.
  }
  std::vector<double> AE;
  AE.resize(num_local_entries);
  for(std::size_t i=0; i < num_local_entries ; ++i) {
    AE[i] = 100.0;
  }


  // Initialize variables that will be reused throughout assembly
  ufc::cell ufc_cell[2];
  std::vector<double> vertex_coordinates[2];
  std::vector<double> macro_vertex_coordinates;
  const double * q_points = NULL;
  const double * q_weights = NULL;
  int num_q_points = 0;

  double q_weight_table[chi_n_max];
  for(std::size_t i=0; i<chi_n_max; i++) {
    q_weight_table[i] = 1.0 / ((double) chi_n_max);
  }
  // Make the ufc data
  UFC ufc_data(a); // FIXME: The problem is that the ufc_data only works with one form, but the data for the other mesh is on a different form. Or is it?
  // NONONO: The problem is that the UFC can't pack in the data since it only has the data to one form, need to feed it both FunctionSpaces/Functions
  UFC ufc_data_b(b);
  // Extract the integral.
  // FIXME: I'm assuming it's the zeroth because I don't know any better.
  ufc::custom_integral * custom_integral = ufc_data.get_custom_integral(0);
  if(!custom_integral) {
    printf("Error... No custom integral in slot 0\n");
    custom_integral = ufc_data.get_custom_integral(1);
    if(!custom_integral) {
      printf("Error... No custom integral in slot 1\n");
      return;
    } else {
      printf("Found integral in slot 1\n");
    }
  } else {
    printf("Found integral in slot 0\n");
  }

  // Iterate over the pairs
  for( int p=0; p<pairs.size(); p+= 2) {
    int ca = pairs[p];
    int cb = pairs[p+1];
    const Cell cellA(meshA, ca);
    const Cell cellB(meshB, cb);
    // printf("%d : %d,%d\n", p, pairs[p],pairs[p+1]);
    // Update the ufc data to the current cell
    cellA.get_cell_data(ufc_cell[0], 0);
    cellB.get_cell_data(ufc_cell[1], 0);

    cellA.get_vertex_coordinates(vertex_coordinates[0]);
    cellB.get_vertex_coordinates(vertex_coordinates[1]);
    // for( int vi = 0; vi< 2; vi++ ) {
    //   for( int d = 0; d < vertex_coordinates[vi].size() ; ++d) {
    // 	std::cout << (vertex_coordinates[vi])[d] << ", ";
    //   }
    //   std::cout << "\n";
    // }
    ufc_data.update(cellA, vertex_coordinates[0], ufc_cell[0],
		    ufc_data_b,cellB, vertex_coordinates[1], ufc_cell[1]);

    // Collect vertex coordinates
    macro_vertex_coordinates.resize(vertex_coordinates[0].size() +
				    vertex_coordinates[0].size()); //TODO: is this an error?
    std::copy(vertex_coordinates[0].begin(),
	      vertex_coordinates[0].end(),
	      macro_vertex_coordinates.begin());
    std::copy(vertex_coordinates[1].begin(),
	      vertex_coordinates[1].end(),
	      macro_vertex_coordinates.begin() + vertex_coordinates[0].size());


    // Select the quadrature points for this pair
    num_q_points = chi_n_max;
    q_points = &(chi[0]) + (p/2)*chi_n_max;
    double q_points_ready[6*chi_n_max];
    printf("max : %d\n",chi_n_max);
    for (std::size_t i = 0 ; i < chi_n_max ; i++ ) {
      for(std::size_t j = 0 ; j < 3 ; j++ ) {
	q_points_ready[3*i+j] = q_points[6*i+j];
	q_points_ready[3*chi_n_max+3*i+j] = q_points[6*i+j+3];
	// printf("%  lf , %lf\n",q_points_ready[3*i+j],q_points_ready[3*chi_n_max+3*i+j]);
      }
    }
    q_weights = q_weight_table;

    // Tabulate the DOFs of both cells in one vector
    // std::cout << pairs[p] << ", " << pairs[p+1] << "\n";
    for (std::size_t i = 0; i < rank; ++i)
      {
	const std::vector<dolfin::la_index>* first;
	const std::vector<dolfin::la_index>* second;

        first  = &mdofA.cell_dofs(ca);
	second = &mdofB.cell_dofs(cb);

	dofs[i]->clear();
	dofs[i]->insert(dofs[i]->end(),first->begin(),first->end());
	dofs[i]->insert(dofs[i]->end(),second->begin(),second->end());
	// for( int d = 0; d < dofs[i]->size() ; ++d) {
	//   std::cout << (*dofs[i])[d] << ", ";
	// }
	// std::cout << "\n";
      }
    
    // Cell orientation not supported...
    const int cell_orientation = ufc_cell[0].orientation;

    // Tabulate cell tensor
    custom_integral->tabulate_tensor(ufc_data.macro_A.data(),
				     ufc_data.macro_w(),
				     macro_vertex_coordinates.data(),
				     num_q_points,
				     q_points_ready,
				     q_weights,
				     0,
				     cell_orientation);
    // for( int m = 0; m < dofs[0]->size() ; ++m) {
      // for( int n = 0; n < dofs[0]->size() ; ++n) {
      // 	std::cout << ufc_data.macro_A.data()[n] << ", ";
      // }
      // std::cout << "\n";
    // }
    // std::cout << "\n";
    // Add to matrix
    A->add(ufc_data.macro_A.data(), dofsconst);  

  }
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
