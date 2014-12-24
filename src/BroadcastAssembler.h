
#ifndef __BROADCAST_ASSEMBLER_BASE_H
#define __BROADCAST_ASSEMBLER_BASE_H

#include <string>
#include <utility>
#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

#include "dolfin/fem/DofMap.h"


namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class Form;


  /// This class is a reimplimentation of Assembler that just
  /// broadcasts element matrices into an arbitrary matrix with
  /// an additional DOF map to assemble the global matrix
  /// into a "more" global matrix. e.g., assembling multiple meshes
  /// or coupling to another non-FEM problem
  class BroadcastAssembler
  {
  public:
    GenericTensor* A;
    std::shared_ptr<TensorLayout> tensor_layout;

    // std::size_t rank;
    std::vector<std::size_t> global_dim_vec;
    std::vector<std::pair<std::size_t, std::size_t> > local_range_vec;
    std::vector<const std::vector<int>* > off_process_owner;


    BroadcastAssembler();

    void init_global_tensor(GenericTensor &newA, 
			    const Array<int> &global_dim, int rank, 
			    const MPI_Comm mpi_comm,
			    const Array<int> &local_range,
			    const Array<int> & off_process
			    );
    void sparsity_form(const Form &a,
		       const GenericDofMap & dofmaps);
    void sparsity_cell_pair(const Form &a,
			    const Mesh & meshA, const GenericDofMap & mdofA,
			    const Mesh & meshB, const GenericDofMap & mdofB,
					    
			    const Array<int>& pairs);
    void sparsity_apply();

    void assemble_form(const Form &a,
		       const GenericDofMap & mappeddof);
    void assemble_cell_pair(const Form &a,
			    const Mesh & meshA, const GenericDofMap & mdofA,
			    const Form &b,
			    const Mesh & meshB, const GenericDofMap & mdofB,
			    const Array<int>& pairs,
			    const Array<double>& chi,
			    const int chi_n);


    void add_to_global_tensor(GenericTensor& A,
			      std::vector<double>& cell_tensor,
			      std::vector<const std::vector<dolfin::la_index>* >& dofs);

    /* const DofMap& map_to_giant(GenericDofMap & dof, int offset); */
    /* const GenericDofMap& map_to_giant(DofMap & dof, const Array<int>& giantmap); */

  };

}


#endif
