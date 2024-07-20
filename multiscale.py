# Import modules
from typing import Tuple

import basix
import cffi
import dolfinx
import gmsh
import meshio
import numpy as np
import pyvista as pv
import time
import ufl
from dolfinx import fem, mesh, plot, geometry
from dolfinx.cpp.mesh import to_string
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.lib.PETSc import ScalarType
from scipy.linalg import inv
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

# Some constants
proc = MPI.COMM_WORLD.rank

cell_marker_1 = 3
cell_marker_2 = 50
boundary_marker = 1

kappa_1 = lambda x: 3  # kappa in large cells
kappa_2 = lambda x: 50  # kappa in small cells

n = 4  # number of subdomain per row/column
d = 1.0 / n  # size of square subdomain
lc_outer = 1e-1 * 1.0
lc_inner = 1e-1 * 0.5
lc_debug = 0.75

kappa_x = 0.6875 * d
kappa_y = 0.6875 * d
kappa_a2 = np.power(10.0, -4)


# Local assembler for a bilinear form
# https://fenicsproject.discourse.group/t/how-to-enumerate-the-cells-of-a-mesh-with-dolfinx-v0-6-0/11661/7
_type_to_offset_index = {fem.IntegralType.cell: 0, fem.IntegralType.exterior_facet: 1}


class LocalAssembler:

    def __init__(self, form, integral_type: fem.IntegralType = fem.IntegralType.cell):
        self.consts = None
        self.coeffs = None
        self.form = fem.form(form)
        self.integral_type = integral_type
        self.update_coefficients()
        self.update_constants()

        subdomain_ids = self.form._cpp_object.integral_ids(integral_type)
        assert (len(subdomain_ids) == 1)
        assert (subdomain_ids[0] == -1)
        is_complex = np.issubdtype(ScalarType, np.complexfloating)
        nptype = "complex128" if is_complex else "float64"
        o_s = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type]]
        o_e = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type] + 1]
        assert o_e - o_s == 1

        self.kernel = getattr(self.form.ufcx_form.form_integrals[o_s], f"tabulate_tensor_{nptype}")
        self.active_cells = self.form._cpp_object.domains(integral_type, -1)
        assert len(self.form.function_spaces) == 2
        self.local_shape = [0, 0]
        for i, V in enumerate(self.form.function_spaces):
            self.local_shape[i] = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs

        e0 = self.form.function_spaces[0].element
        e1 = self.form.function_spaces[1].element
        needs_transformation_data = e0.needs_dof_transformations or e1.needs_dof_transformations or \
                                    self.form._cpp_object.needs_facet_permutations
        if needs_transformation_data:
            raise NotImplementedError("Dof transformations not implemented")

        self.ffi = cffi.FFI()
        V = self.form.function_spaces[0]
        self.x_dofs = V.mesh.geometry.dofmap

    def update_coefficients(self):
        self.coeffs = fem.assemble.pack_coefficients(self.form._cpp_object)[(self.integral_type, -1)]

    def update_constants(self):
        self.consts = fem.assemble.pack_constants(self.form._cpp_object)

    def update(self):
        self.update_coefficients()
        self.update_constants()

    def assemble_matrix(self, i: int, local_index: int = 0):

        x = self.form.function_spaces[0].mesh.geometry.x
        x_dofs = self.x_dofs[i]
        geometry = np.zeros((len(x_dofs), 3), dtype=np.float64)
        geometry[:, :] = x[x_dofs]

        A_local = np.zeros((self.local_shape[0], self.local_shape[0]), dtype=ScalarType)
        facet_index = np.array([local_index], dtype=np.intc)
        facet_perm = np.zeros(0, dtype=np.uint8)
        if self.coeffs.shape == (0, 0):
            coeffs = np.zeros(0, dtype=ScalarType)
        else:
            coeffs = self.coeffs[i, :]
        ffi_fb = self.ffi.from_buffer
        self.kernel(ffi_fb(A_local), ffi_fb(coeffs), ffi_fb(self.consts), ffi_fb(geometry),
                    ffi_fb(facet_index), ffi_fb(facet_perm))
        return A_local


def interpolation_matrix_non_matching_meshes(V_1, V_0):  # Function spaces from non-matching meshes
    msh_0 = V_0.mesh
    x_0 = V_0.tabulate_dof_coordinates()
    x_1 = V_1.tabulate_dof_coordinates()

    bb_tree = geometry.bb_tree(msh_0, msh_0.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, x_1)
    cells = []
    points_on_proc = []
    index_points = []
    colliding_cells = geometry.compute_colliding_cells(msh_0, cell_candidates, x_1)

    for i, point in enumerate(x_1):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            index_points.append(i)

    index_points_ = np.array(index_points)
    points_on_proc_ = np.array(points_on_proc, dtype=np.float64)
    cells_ = np.array(cells)

    ct = to_string(msh_0.topology.cell_types[0])
    element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.cell.string_to_type(ct), V_0.ufl_element().degree(), basix.LagrangeVariant.equispaced)

    x_ref = np.zeros((len(cells_), 2))

    for i in range(0, len(cells_)):
        geom_dofs = msh_0.geometry.dofmap[cells_[i]]
        x_ref[i, :] = msh_0.geometry.cmaps[0].pull_back([points_on_proc_[i, :]], msh_0.geometry.x[geom_dofs])

    basis_matrix = element.tabulate(0, x_ref)[0, :, :, 0]

    cell_dofs = np.zeros((len(x_1), len(basis_matrix[0, :])))
    basis_matrix_full = np.zeros((len(x_1), len(basis_matrix[0, :])))

    for nn in range(0, len(cells_)):
        cell_dofs[index_points_[nn], :] = V_0.dofmap.cell_dofs(cells_[nn])
        basis_matrix_full[index_points_[nn], :] = basis_matrix[nn, :]

    cell_dofs_ = cell_dofs.astype(int)  # REDUCE HERE

    I = np.zeros((len(x_1), len(x_0)))

    for i in range(0, len(x_1)):
        for j in range(0, len(basis_matrix[0, :])):
            I[i, cell_dofs_[i, j]] = basis_matrix_full[i, j]

    return I


def create_simple_gmsh(marker_cell: int, marker_facet: int):
    gmsh.initialize()

    p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc_debug)
    p2 = gmsh.model.occ.addPoint(1.0, 0.0, 0.0, lc_debug)
    p3 = gmsh.model.occ.addPoint(1.0, 1.0, 0.0, lc_debug)
    p4 = gmsh.model.occ.addPoint(0.0, 1.0, 0.0, lc_debug)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    ps = gmsh.model.occ.addPlaneSurface([cl])

    gmsh.model.occ.synchronize()

    # Tag the cells
    gmsh.model.addPhysicalGroup(2, [ps], marker_cell)

    # Tag the facets
    facets = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0.0) or np.isclose(com[0], 1.0) or np.isclose(com[1], 0.0) or np.isclose(com[1], 1.0):
            facets.append(line[1])
    gmsh.model.addPhysicalGroup(1, facets, marker_facet)

    gmsh.model.mesh.generate(2)
    gmsh.write("MS/mesh_c.msh")

    gmsh.finalize()


# Create the mesh via gmsh
def create_gmsh(marker_cell_outer: int, marker_cell_inner: int, marker_facet_boundary: int, fine: bool = False):
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 2)
    outer_cell_tags = []
    inner_cell_tags = []
    ones, others = [], []
    boundary_tags = []
    if proc == 0:
        for i in range(n):
            for j in range(n):
                # We create one large cell, and one small cell within it
                # tag1 = i*n+j
                # tag2 = i*n+j + n*n

                # Create inner rectangle
                # gmsh.model.occ.addRectangle(d*(i+0.5), d*(j+0.5), 0, d*3/8, d*3/8, tag=tag2)
                p1 = gmsh.model.occ.addPoint(d * (i + 0.500), d * (j + 0.500), 0, lc_inner)
                p2 = gmsh.model.occ.addPoint(d * (i + 0.875), d * (j + 0.500), 0, lc_inner)
                p3 = gmsh.model.occ.addPoint(d * (i + 0.875), d * (j + 0.875), 0, lc_inner)
                p4 = gmsh.model.occ.addPoint(d * (i + 0.500), d * (j + 0.875), 0, lc_inner)
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p4)
                l4 = gmsh.model.occ.addLine(p4, p1)

                cl_inner = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
                ps_inner = gmsh.model.occ.addPlaneSurface([cl_inner])

                # Create outer rectangle
                # gmsh.model.occ.addRectangle(d*i, d*j, 0, d, d, tag=tag1)
                p1 = gmsh.model.occ.addPoint(d * i, d * j, 0, lc_outer)
                p2 = gmsh.model.occ.addPoint(d * (i + 1), d * j, 0, lc_outer)
                p3 = gmsh.model.occ.addPoint(d * (i + 1), d * (j + 1), 0, lc_outer)
                p4 = gmsh.model.occ.addPoint(d * i, d * (j + 1), 0, lc_outer)
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p4)
                l4 = gmsh.model.occ.addLine(p4, p1)

                cl_outer = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
                ps_outer = gmsh.model.occ.addPlaneSurface([cl_outer, cl_inner])

                # We add the rectangles in the subdomain list
                outer_cell_tags.append(ps_outer)
                inner_cell_tags.append(ps_inner)

                # We add the appropriate rectangles to appropriate list for fragmenting
                if (i + j) % 2 == 0:
                    ones.append((2, ps_outer))
                    others.append((2, ps_inner))
                else:
                    ones.append((2, ps_inner))
                    others.append((2, ps_outer))

        gmsh.model.occ.fragment(ones, others)

        gmsh.model.occ.synchronize()

        # print(outer_tags)
        # print(inner_tags)
        gmsh.model.addPhysicalGroup(2, outer_cell_tags, marker_cell_outer)
        gmsh.model.addPhysicalGroup(2, inner_cell_tags, marker_cell_inner)

        # Tag the dirichlet boundary facets
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0.0) or np.isclose(com[0], 1.0) or np.isclose(com[1], 0.0) or np.isclose(com[1], 1.0):
                boundary_tags.append(line[1])
        gmsh.model.addPhysicalGroup(1, boundary_tags, marker_facet_boundary)

        gmsh.model.mesh.generate(2)
        if fine:
            gmsh.model.mesh.refine()
        gmsh.write("MS/mesh_" + ("f" if fine else "c") + ".msh")

    gmsh.finalize()


# A convenience function for extracting data for a single cell type, and creating a new meshio mesh,
# including physical markers for the given type.
def create_mesh(in_mesh, cell_type, prune_z=False) -> meshio.Mesh:
    cells = in_mesh.get_cells_type(cell_type)
    cell_data = in_mesh.get_cell_data("gmsh:physical", cell_type)
    points = in_mesh.points[:, :2] if prune_z else in_mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh


# We have now written the mesh and the cell markers to one file, and the facet markers in a separate file.
# We can now read this data in DOLFINx using XDMFFile.read_mesh and XDMFFile.read_meshtags.
# The dolfinx.MeshTags stores the index of the entity, along with the value of the marker in two one dimensional arrays.
def read_mesh(fine: bool) -> Tuple[mesh.Mesh, mesh.MeshTags, mesh.MeshTags]:
    suffix = "f" if fine else "c"
    # Read in mesh
    msh = meshio.read("MS/mesh_" + suffix + ".msh")

    # Create and save one file for the mesh, and one file for the facets
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("MS/mesh_" + suffix + ".xdmf", triangle_mesh)
    meshio.write("MS/mt_" + suffix + ".xdmf", line_mesh)
    MPI.COMM_WORLD.barrier()

    # We read the mesh in parallel
    with XDMFFile(MPI.COMM_WORLD, "MS/mesh_" + suffix + ".xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")  # Mesh
        ct = xdmf.read_meshtags(mesh, name="Grid")  # Cell tags
    # Create connectivity cell -> facet
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "MS/mt_" + suffix + ".xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")  # Facet tags

    # Create connectivity cell <-> point
    mesh.topology.create_connectivity(0, mesh.topology.dim)

    # print(mesh/ct/ft.indices)
    # print(mesh/ct/ft.values)

    return mesh, ct, ft


def create_kappa(
        msh: mesh.Mesh,
        ct: mesh.MeshTags,
        marker_1: int,
        marker_2: int
) -> Tuple[fem.Function, fem.Function]:
    # Create a discontinuous function for tagging the cells
    FS_DG = fem.functionspace(msh, ("DG", 0))
    q = fem.Function(FS_DG)
    cells_1 = ct.find(marker_1)
    q.x.array[cells_1] = np.full_like(cells_1, marker_1, dtype=dolfinx.default_scalar_type)
    cells_2 = ct.find(marker_2)
    q.x.array[cells_2] = np.full_like(cells_2, marker_2, dtype=dolfinx.default_scalar_type)

    # Out of q, we can create an arbitrary kappa:
    # We can now create our discontinuous function for tagging subdomains:
    FS_CG = fem.functionspace(msh, ("CG", 1))
    fx, fy, fz = fem.Function(FS_CG), fem.Function(FS_CG), fem.Function(FS_CG)
    fq, kappa = fem.Function(FS_CG), fem.Function(FS_CG)
    fx.interpolate(lambda x: x[0])
    fy.interpolate(lambda x: x[1])
    fz.interpolate(lambda x: x[2])
    fq.interpolate(q)
    for i in range(len(kappa.x.array)):
        x = [fx.x.array[i], fy.x.array[i], fz.x.array[i]]
        if fq.x.array[i] == marker_2:
            kappa.x.array[i] = kappa_2(x)
        else:
            kappa.x.array[i] = kappa_1(x)

    return q, kappa


def create_point_source(
        msh: mesh.Mesh,
        x0: float,
        y0: float,
        a2: float
) -> fem.Function:
    FS_ps = fem.functionspace(msh, ("CG", 1))
    f = fem.Function(FS_ps)
    dirac = lambda x: 1.0 / np.sqrt(a2 * np.pi) * np.exp(
        -(np.power(x[0] - x0, 2) + np.power(x[1] - y0, 2)) / a2
    )
    f.interpolate(dirac)
    return f


def compute_correction_operator(
        msh_c: mesh.Mesh,
        ct_c: mesh.MeshTags,
        parent_cells: np.ndarray,
        FS_c: fem.FunctionSpaceBase,
        FS_f: fem.FunctionSpaceBase,
        coarse_boundary_dofs: np.ndarray,
        fine_boundary_dofs: np.ndarray,
        A_h: csr_matrix,
        P_h: csr_matrix,
        C_h: csr_matrix,
        fine_stiffness_assembler: LocalAssembler
) -> lil_matrix:

    num_dofs_c = FS_c.dofmap.index_map.size_local * FS_c.dofmap.index_map_bs  # N_H
    num_dofs_f = FS_f.dofmap.index_map.size_local * FS_f.dofmap.index_map_bs  # N_h

    Q_h = lil_matrix((num_dofs_c, num_dofs_f))

    # For each coarse cell K_l
    for l in ct_c.indices:
        # Create local patch U_l consisting of K_l with one layer of neighboring cells (k = 1, for now)
        # https://docs.fenicsproject.org/dolfinx/main/cpp/mesh.html#_CPPv4N7dolfinx4mesh25compute_incident_entitiesERK8TopologyNSt4spanIKNSt7int32_tEEEii
        incident_facets = mesh.compute_incident_entities(msh_c.topology, l, 2, 1)
        incident_vertices = mesh.compute_incident_entities(msh_c.topology, l, 2, 0)
        coarse_patch_1 = mesh.compute_incident_entities(msh_c.topology, incident_facets, 1, 2)
        coarse_patch_2 = mesh.compute_incident_entities(msh_c.topology, incident_vertices, 0, 2)
        coarse_patch = np.unique(np.concatenate((coarse_patch_1, coarse_patch_2)))

        # Find coarse dofs on patch
        coarse_dofs_local = fem.locate_dofs_topological(FS_c, 2, coarse_patch)  # Z_l[i] = coarse_dofs_local[i]
        coarse_dofs_local = np.setdiff1d(coarse_dofs_local, coarse_boundary_dofs, assume_unique=True)
        num_coarse_dofs_local = coarse_dofs_local.size  # N_H_l

        # Create restriction matrix R_H_l (N_H_l x N_H)
        R_H_l = csr_matrix(
            (
                np.ones_like(coarse_dofs_local),
                (
                    np.arange(num_coarse_dofs_local),
                    coarse_dofs_local
                ),
            ),
            shape=(num_coarse_dofs_local, num_dofs_c)
        )
#        print("R_H_l:")
#        print(R_H_l)

        # Find fine cells on patch
        fine_patch = np.where(np.isin(parent_cells, coarse_patch))[0]

        # Find fine dofs on patch
        fine_dofs_local = fem.locate_dofs_topological(FS_f, 2, fine_patch)  # z_l[i] = fine_dofs_local[i]
        fine_dofs_local = np.setdiff1d(fine_dofs_local, fine_boundary_dofs, assume_unique=True)
        num_fine_dofs_local = fine_dofs_local.size  # N_h_l

        # Create restriction matrix R_h_l (N_h_l x N_h)
        R_h_l = csr_matrix(
            (
                np.ones_like(fine_dofs_local),
                (
                    np.arange(num_fine_dofs_local),
                    fine_dofs_local
                ),
            ),
            shape=(num_fine_dofs_local, num_dofs_f)
        )
        assert R_h_l.shape == (num_fine_dofs_local, num_dofs_f)

#        print("R_h_l:")
#        print(R_h_l)

        # Create local coarse-node-to-coarse-element restriction matrix T_H_l (c_d x N_H)
        l_global_dofs = fem.locate_dofs_topological(FS_c, 2, l)  # p[i] = l_dofs[i]
        assert l_global_dofs.size == msh_c.topology.cell_types[0].value
        T_H_l = csr_matrix(
            (
                np.ones_like(l_global_dofs),
                (
                    np.arange(l_global_dofs.size),
                    l_global_dofs
                ),
            ),
            shape=(l_global_dofs.size, num_dofs_c)
        )
        assert T_H_l.shape == (l_global_dofs.size, num_dofs_c)

#        print("T_H_l:")
#        print(T_H_l)

        # Calculate local stiffness matrix and constraints matrix
        A_l = R_h_l @ A_h @ R_h_l.transpose()
        C_l = R_H_l @ C_h @ R_h_l.transpose()

        # In order to create local load vector matrix,
        # we need the contributions of local stiffness matrices on fine cells
        sigma_A_sigmaT_l = lil_matrix((num_dofs_f, num_dofs_f))
        # Find fine cells only on coarse cell l
        fine_cells_on_l = np.where(parent_cells == l)[0]
        for t in fine_cells_on_l:
            # Assemble local stiffness matrix A_t
            A_t = fine_stiffness_assembler.assemble_matrix(t)
            A_t = csr_matrix(A_t)

            # Find global fine dofs on fine cell t
            fine_dofs_global_t = fem.locate_dofs_topological(FS_f, 2, t)
            # Create local-to-global-mapping sigma_t
            sigma_t = csr_matrix(
                (
                    np.ones_like(fine_dofs_global_t),
                    (
                        np.arange(fine_dofs_global_t.size),
                        fine_dofs_global_t
                    ),
                ),
                shape=(fine_dofs_global_t.size, num_dofs_f)
            )

            # Add the contribution
            sigma_A_sigmaT_l += sigma_t.transpose() @ A_t @ sigma_t

        # Create local load vector matrix
        r_l = - (T_H_l @ P_h @ sigma_A_sigmaT_l @ R_h_l.transpose())

        # Compute the inverse of the local stiffness matrix
        A_l_inv = csr_matrix(inv(A_l.todense()))

        # Precomputations related to the operator
        Y_l = A_l_inv @ C_l.transpose()

        # Compute inverse Schur complement
        S_l_inv = csr_matrix(inv((C_l @ Y_l).todense()))

        # Compute correction for each coarse space function with support on K_l
        w_l = lil_matrix((l_global_dofs.size, num_fine_dofs_local))
        for i in range(l_global_dofs.size):
            q_i = A_l_inv @ r_l[i].transpose()
            lambda_i = S_l_inv @ (C_l @ q_i)
            w_l_i = q_i - Y_l @ lambda_i
            w_l[i] = w_l_i.transpose()

        # Update the corrector matrix
        Q_h += T_H_l.transpose() @ w_l @ R_h_l

    return Q_h


def main():

    t0 = time.time()

    # Create coarse mesh
    create_gmsh(cell_marker_1, cell_marker_2, boundary_marker, False)
    # create_simple_gmsh(cell_marker_1, boundary_marker)

    # Read coarse mesh
    msh_c, ct_c, ft_c = read_mesh(False)

    # Create fine mesh
    # https://fenicsproject.discourse.group/t/input-for-mesh-refinement-with-refine-plaza/13426/4
    msh_f, parent_cells, _ = mesh.refine_plaza(
        msh_c,
        edges=None,  # Mesh refinement is uniform
        redistribute=False,  # Do not redistribute across processes
        option=RefinementOption.parent_cell  # We are interested in parent cells and parent facets
    )

    # Fix parent cells
    marker = np.arange(msh_c.topology.index_map(msh_c.topology.dim).size_local, dtype=np.int32)
    ct_parent_c = dolfinx.mesh.meshtags(msh_c, msh_c.topology.dim, marker, marker)
    ct_parent_f = dolfinx.mesh.MeshTags(
        dolfinx.cpp.refinement.transfer_cell_meshtag(ct_parent_c._cpp_object, msh_f.topology, parent_cells)
    )

    # Transfer cell mesh tags from coarse to fine mesh
    msh_f.topology.create_connectivity(msh_f.topology.dim, msh_f.topology.dim - 1)
    ct_f = dolfinx.mesh.MeshTags(
        dolfinx.cpp.refinement.transfer_cell_meshtag(ct_c._cpp_object, msh_f.topology, parent_cells)
    )

    # Setup parent cells to reuse later
    parent_cells = []
    for i in range(ct_parent_f.indices.shape[0]):
        cell = ct_parent_f.indices[i]
        value = ct_parent_f.values[i]
        parent_cells.insert(cell, value)
    parent_cells = np.array(parent_cells)

    # Make new facet tags on fine mesh (until we figure out why transferring doesn't work)
    facets = mesh.locate_entities_boundary(msh_f, msh_f.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    ft_f = mesh.meshtags(msh_f, 1, facets, boundary_marker)

    # Number of cells on coarse mesh
    num_cells_c = ct_c.indices.shape[0]  # N_T_H
    index_map_c = msh_c.topology.index_map(2)
    assert index_map_c.size_local == index_map_c.size_global
    assert num_cells_c == index_map_c.size_local

    # P1 function space on coarse mesh for solution
    FS_c = fem.functionspace(msh_c, ("CG", 1))
    assert FS_c.dofmap.index_map.size_local == FS_c.dofmap.index_map.size_global
    num_dofs_c = FS_c.dofmap.index_map.size_local * FS_c.dofmap.index_map_bs  # N_H

    # P1 function space on fine mesh for solution
    FS_f = fem.functionspace(msh_f, ("CG", 1))
    assert FS_f.dofmap.index_map.size_local == FS_f.dofmap.index_map.size_global
    num_dofs_f = FS_f.dofmap.index_map.size_local * FS_f.dofmap.index_map_bs  # N_h

    # Define Dirichlet boundary condition on fine mesh
    boundary_facets_f = ft_f.find(boundary_marker)
    boundary_dofs_f = fem.locate_dofs_topological(FS_f, msh_f.topology.dim - 1, boundary_facets_f)
    bcs_f = [fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_f, FS_f)]

    # Create boundary correction (restriction) matrix B_H
    boundary_facets_c = ft_c.find(boundary_marker)
    boundary_dofs_c = fem.locate_dofs_topological(FS_c, 1, boundary_facets_c)
    not_boundary_dofs_c = np.setdiff1d(np.arange(num_dofs_c), boundary_dofs_c, assume_unique=True)
    B_H = csr_matrix(
        (
            np.ones_like(not_boundary_dofs_c),
            (
                not_boundary_dofs_c,
                not_boundary_dofs_c
            )
        ),
        shape=(num_dofs_c, num_dofs_c)
    )
    assert B_H.shape == (num_dofs_c, num_dofs_c)

    # Remove all-zero rows from B_H
    ind_ptr = np.unique(B_H.indptr)
    B_H = csr_matrix((B_H.data, B_H.indices, ind_ptr), shape=(len(ind_ptr) - 1, B_H.shape[1]))

    # Create kappa and rhs source term f
    q = create_kappa(msh_f, ct_f, cell_marker_1, cell_marker_2)[0]
    f = create_point_source(msh_f, kappa_x, kappa_y, kappa_a2)

    # Plot q and point source
    grid_q = pv.UnstructuredGrid(*plot.vtk_mesh(msh_f, msh_f.topology.dim))
    grid_q.cell_data["q"] = q.x.array.real
    grid_q.set_active_scalars("q")
    plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
    plotter.show_axes()
    plotter.show_grid()
    plotter.add_mesh(grid_q, show_edges=True)
    plotter.view_xy()
    plotter.screenshot("MS/q.png")

    grid_ps = pv.UnstructuredGrid(*plot.vtk_mesh(msh_f, msh_f.topology.dim))
    grid_ps.point_data["f"] = f.x.array.real
    grid_ps.set_active_scalars("f")
    plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
    plotter.show_axes()
    plotter.show_grid()
    plotter.add_mesh(grid_ps, show_edges=True)
    plotter.view_xy()
    plotter.screenshot("MS/f.png")

    # Define our problem on fine mesh using Unified Form Language (UFL)
    # and assemble the stiffness and mass matrices A_h and M_h
    u_f = ufl.TrialFunction(FS_f)
    v_f = ufl.TestFunction(FS_f)
    a = ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
    m = ufl.inner(u_f, v_f) * ufl.dx
    L = ufl.inner(f, v_f) * ufl.dx

    fine_stiffness_assembler = LocalAssembler(a)

    A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
    assert A_h.shape == (num_dofs_f, num_dofs_f)
    M_h = fem.assemble_matrix(fem.form(m), bcs_f).to_scipy()
    assert M_h.shape == (num_dofs_f, num_dofs_f)
    f_h = fem.assemble_vector(fem.form(L)).array
    assert f_h.shape == (num_dofs_f,)

    # Create projection matrix P_h from coarse mesh Lagrange space to fine mesh Lagrange space
    P_h = csr_matrix(interpolation_matrix_non_matching_meshes(FS_f, FS_c).transpose())

    # Calculate constraint matrix C_h
    C_h = P_h @ M_h

    print(f"Setup time: {time.time() - t0}")
    t0 = time.time()

    # Create corrector matrix Q_h
    Q_h = compute_correction_operator(msh_c, ct_c, parent_cells,
                                      FS_c, FS_f,
                                      boundary_dofs_c, boundary_dofs_f,
                                      A_h, P_h, C_h,
                                      fine_stiffness_assembler)

    print(f"Time to compute correction operator: {time.time() - t0}")
    t0 = time.time()

    # Add the corrector matrix to the solution and solve the system
    A_H_LOD = B_H @ (P_h + Q_h) @ A_h @ (P_h + Q_h).transpose() @ B_H.transpose()
    f_H_LOD = B_H @ (P_h + Q_h) @ f_h
    u_H_LOD = B_H.transpose() @ spsolve(A_H_LOD, f_H_LOD)
    u_h_LOD = (P_h + Q_h).transpose() @ u_H_LOD

    print(f"Time to solve LOD system: {time.time() - t0}")
    t0 = time.time()

    # u_h_LOD is now a vector of size num_dofs_f, which we can wrap in a Function object using dolfinx
    # and display using pyvista or ParaView
    uhLOD = fem.Function(FS_f)
    uhLOD.x.array.real = u_h_LOD  # spsolve(A_h, f_h)

    grid_uhLOD = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
    grid_uhLOD.point_data["uhLOD"] = uhLOD.x.array.real
    grid_uhLOD.set_active_scalars("uhLOD")

    with dolfinx.io.XDMFFile(msh_f.comm, "MS/solution_LOD.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh_f)
        xdmf.write_function(uhLOD)

    # Coarse solution for comparison
    A_H = B_H @ P_h @ A_h @ P_h.transpose() @ B_H.transpose()
    f_H = B_H @ P_h @ f_h
    uh_c = fem.Function(FS_c)
    uh_c.x.array.real = B_H.transpose() @ spsolve(A_H, f_H)
    print(f"Time to solve just the coarse system: {time.time() - t0}")
    t0 = time.time()
    grid_uh_c = pv.UnstructuredGrid(*plot.vtk_mesh(FS_c))
    grid_uh_c.point_data["uh_c"] = uh_c.x.array.real
    grid_uh_c.set_active_scalars("uh_c")

    with dolfinx.io.XDMFFile(msh_c.comm, "MS/solution_coarse.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh_c)
        xdmf.write_function(uh_c)

    # Fine solution for comparison
    uh_f = fem.Function(FS_f)
    uh_f.x.array.real = spsolve(A_h, f_h)
    print(f"Time to solve whole fine system: {time.time() - t0}")
    grid_uh_f = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
    grid_uh_f.point_data["uh_f"] = uh_f.x.array.real
    grid_uh_f.set_active_scalars("uh_f")

    with dolfinx.io.XDMFFile(msh_f.comm, "MS/solution_fine.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh_f)
        xdmf.write_function(uh_f)

    # pv.start_xvfb()
    for grid, suffix in [(grid_uh_c, "coarse"), (grid_uh_f, "fine"), (grid_uhLOD, "LOD")]:
        plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
        plotter.show_axes()
        plotter.show_grid()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.screenshot(f"MS/solution_{suffix}.png")

    # Interpolate coarse solution to fine mesh
    # https://github.com/FEniCS/dolfinx/blob/v0.8.0/python/test/unit/fem/test_interpolation.py
    uh_c_to_f = fem.Function(FS_f)
    uh_c_to_f.interpolate(uh_c, nmm_interpolation_data=dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        uh_c_to_f.function_space.mesh._cpp_object,
        uh_c_to_f.function_space.element,
        uh_c.function_space.mesh._cpp_object,
        padding=1e-14
    ))

    # Calculate L2 errors
    L2_c_f = fem.assemble_scalar(fem.form(
        ufl.inner(uh_c_to_f - uh_f, uh_c_to_f - uh_f) * ufl.dx
    ))
    L2_c_f = np.sqrt(msh_f.comm.allreduce(L2_c_f, op=MPI.SUM))

    L2_LOD_c = fem.assemble_scalar(fem.form(
        ufl.inner(uhLOD - uh_c_to_f, uhLOD - uh_c_to_f) * ufl.dx
    ))
    L2_LOD_c = np.sqrt(msh_f.comm.allreduce(L2_LOD_c, op=MPI.SUM))

    L2_LOD_f = fem.assemble_scalar(fem.form(
        ufl.inner(uhLOD - uh_f, uhLOD - uh_f) * ufl.dx
    ))
    L2_LOD_f = np.sqrt(msh_f.comm.allreduce(L2_LOD_f, op=MPI.SUM))

    print(f"L2 error between coarse and fine solution: {L2_c_f}")
    print(f"L2 error between LOD and coarse solution: {L2_LOD_c}")
    print(f"L2 error between LOD and fine solution: {L2_LOD_f}")

    # Plot absolute difference between LOD and fine solution
    diff = fem.Function(FS_f)
    diff.x.array.real = uhLOD.x.array.real - uh_f.x.array.real
    for i in range(diff.x.array.real.size):
        if diff.x.array.real[i] < 0.0:
            diff.x.array.real[i] = -diff.x.array.real[i]
    grid_diff = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
    grid_diff.point_data["diff"] = diff.x.array.real
    grid_diff.set_active_scalars("diff")

    with dolfinx.io.XDMFFile(msh_f.comm, "MS/solution_diff.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh_f)
        xdmf.write_function(diff)

    plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
    plotter.show_axes()
    plotter.show_grid()
    plotter.add_mesh(grid_diff, show_edges=True)
    plotter.view_xy()
    plotter.screenshot("MS/solution_diff.png")


if __name__ == "__main__":
    main()
