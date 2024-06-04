# Import modules
import dolfinx
import gmsh
import meshio
import numpy as np
import pyvista as pv
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.fem import functionspace, Function, FunctionSpaceBase, assemble_matrix, form, locate_dofs_topological, \
    assemble_vector
from dolfinx.fem.petsc import interpolation_matrix, assemble_matrix_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh, MeshTags, compute_incident_entities, refine_plaza, meshtags
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
from scipy.linalg import inv
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, inv as sp_inv
from typing import Tuple, List
from ufl import TestFunction, TrialFunction, inner, grad, dx, Measure

# Some constants
proc = MPI.COMM_WORLD.rank

f_1 = lambda x: 3  # kappa in large cells
f_2 = lambda x: 47  # kappa in small cells

n = 2  # number of subdomain per row/column
d = 1.0 / n  # size of square subdomain
lc_outer = 1e-1
lc_inner = 1e-1


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
def read_mesh(fine: bool) -> Tuple[Mesh, MeshTags, MeshTags]:
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
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "MS/mt_" + suffix + ".xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")  # Facet tags
    mesh.topology.create_connectivity(0, 2)
    mesh.topology.create_connectivity(2, 0)

    # print(mesh/ct/ft.indices)
    # print(mesh/ct/ft.values)

    return mesh, ct, ft


def create_kappa(
        mesh: Mesh,
        ct: MeshTags,
        marker_1: int,
        marker_2: int,
        FS: FunctionSpaceBase
) -> Function:
    # Create a discontinuous function for tagging the cells
    FS_tags = functionspace(mesh, ("DG", 0))
    q = Function(FS_tags)
    cells_1 = ct.find(marker_1)
    q.x.array[cells_1] = np.full_like(cells_1, marker_1, dtype=dolfinx.default_scalar_type)
    cells_2 = ct.find(marker_2)
    q.x.array[cells_2] = np.full_like(cells_2, marker_2, dtype=dolfinx.default_scalar_type)

    # Out of q, we can create an arbitrary kappa:
    # We can now create our discontinuous function for tagging subdomains:
    FS_kappa = FS
    fx, fy, fz = Function(FS_kappa), Function(FS_kappa), Function(FS_kappa)
    fq, kappa = Function(FS_kappa), Function(FS_kappa)
    fx.interpolate(lambda x: x[0])
    fy.interpolate(lambda x: x[1])
    fz.interpolate(lambda x: x[2])
    fq.interpolate(q)
    for i in range(len(kappa.x.array)):
        x = [fx.x.array[i], fy.x.array[i], fz.x.array[i]]
        if fq.x.array[i] == marker_2:
            kappa.x.array[i] = f_2(x)
        else:
            kappa.x.array[i] = f_1(x)

    return kappa


def create_point_source(
        FS: FunctionSpaceBase,
        x0: float,
        y0: float,
        a2: float
) -> Function:
    f = Function(FS)
    dirac = lambda x: 1 / np.sqrt(a2 * np.pi) * np.exp(
        -(np.power(x[0] - x0, 2) + np.power(x[1] - y0, 2)) / a2
    )
    f.interpolate(dirac)
    return f


def compute_correction_operator(
        mesh_c: Mesh,
        mesh_f: Mesh,
        FS_c: FunctionSpaceBase,
        FS_f: FunctionSpaceBase,
        ct_c: MeshTags,
        parent_cells: np.ndarray,
        coarse_boundary_dofs: np.ndarray,
        fine_boundary_dofs: np.ndarray,
        A_h: csr_matrix,
        B_H: csr_matrix,
        P_h: csr_matrix,
        C_h: csr_matrix
) -> lil_matrix:

    num_dofs_c = FS_c.dofmap.index_map.size_local * FS_c.dofmap.index_map_bs  # N_H
    num_dofs_f = FS_f.dofmap.index_map.size_local * FS_f.dofmap.index_map_bs  # N_h

    Q_h = lil_matrix((num_dofs_c, num_dofs_f))

    # For each coarse cell K_l
    for l in ct_c.indices:
        # Create local patch U_l consisting of K_l with one layer of neighboring cells (k = 1, for now)
        # https://docs.fenicsproject.org/dolfinx/main/cpp/mesh.html#_CPPv4N7dolfinx4mesh25compute_incident_entitiesERK8TopologyNSt4spanIKNSt7int32_tEEEii
        incident_facets = compute_incident_entities(mesh_c.topology, l, 2, 1)
        incident_vertices = compute_incident_entities(mesh_c.topology, l, 2, 0)
        coarse_patch_1 = compute_incident_entities(mesh_c.topology, incident_facets, 1, 2)
        coarse_patch_2 = compute_incident_entities(mesh_c.topology, incident_vertices, 0, 2)
        coarse_patch = np.unique(np.concatenate((coarse_patch_1, coarse_patch_2)))

        # Find coarse dofs on patch
        coarse_dofs_local = locate_dofs_topological(FS_c, 2, coarse_patch)  # Z_l[i] = coarse_dofs_local[i]
        coarse_dofs_local = np.setdiff1d(coarse_dofs_local, coarse_boundary_dofs, assume_unique=True)
        num_coarse_dofs_local = coarse_dofs_local.size  # N_H_l

        # Create restriction matrix R_H_l (N_H_l x N_H)
        R_H_l = lil_matrix((num_coarse_dofs_local, num_dofs_c))
        for i in range(num_coarse_dofs_local):
            R_H_l[i, coarse_dofs_local[i]] = 1

        # Find fine cells on patch
        fine_patch = np.where(np.isin(parent_cells, coarse_patch))[0]

        # Find fine dofs on patch
        fine_dofs_local = locate_dofs_topological(FS_f, 2, fine_patch)  # z_l[i] = fine_dofs_local[i]
        fine_dofs_local = np.setdiff1d(fine_dofs_local, fine_boundary_dofs, assume_unique=True)
        num_fine_dofs_local = fine_dofs_local.size  # N_h_l

        # Create restriction matrix R_h_l (N_h_l x N_h)
        R_h_l = lil_matrix((num_fine_dofs_local, num_dofs_f))
        for i in range(num_fine_dofs_local):
            R_h_l[i, fine_dofs_local[i]] = 1

        # Create local coarse-node-to-coarse-element restriction matrix T_H_l (c_d x N_H)
        l_dofs = locate_dofs_topological(FS_c, 2, l)  # p[i] = l_dofs[i]
        assert l_dofs.size == mesh_c.topology.cell_types[0].value
        T_H_l = lil_matrix((l_dofs.size, num_dofs_c))
        for i in range(l_dofs.size):
            T_H_l[i, l_dofs[i]] = 1

        # Calculate local stiffness matrix and constraints matrix
        A_l = R_h_l @ A_h @ R_h_l.transpose()
        C_l = R_H_l @ C_h @ R_h_l.transpose()

        # In order to create local load vector matrix,
        # we need the contributions of local stiffness matrices on fine cells
        sigma_A_sigmaT_l = lil_matrix((num_dofs_f, num_dofs_f))
        # Find fine cells only on coarse cell l
        fine_cells_on_l = np.where(parent_cells == l)[0]
        for t in fine_cells_on_l:
            # Create submesh containing only that one fine cell t
            # https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.mesh.html#dolfinx.mesh.create_submesh
            mesh_t, entity_map, vertex_map, node_map = dolfinx.mesh.create_submesh(mesh_f, 2, t)
            ct_t = dolfinx.mesh.meshtags(mesh_t, 2, np.array([0]), np.array([ct_c.values[l]]))
            FS_t = functionspace(mesh_t, ("CG", 1))
            v_t = TestFunction(FS_t)
            u_t = TrialFunction(FS_t)
            kappa_t = create_kappa(mesh_t, ct_t, 1, 2, FS_t)
            a_t = inner(kappa_t * grad(u_t), grad(v_t)) * dx
            # Assemble local stiffness matrix
            A_t = assemble_matrix(form(a_t)).to_scipy()

            # Find fine dofs on fine cell t
            fine_dofs_local_t = locate_dofs_topological(FS_f, 2, t)
            # Create local-to-global-mapping sigma_t
            sigma_t = lil_matrix((fine_dofs_local_t.size, num_dofs_f))
            for i in range(fine_dofs_local_t.size):
                sigma_t[i, fine_dofs_local_t[i]] = 1

            # Add the contribution
            sigma_A_sigmaT_l += sigma_t.transpose() @ A_t @ sigma_t

        # Create local load vector matrix
        r_l = - (T_H_l @ B_H @ P_h @ sigma_A_sigmaT_l @ R_h_l.transpose())

        # Compute the inverse of the local stiffness matrix
        A_l_inv = sp_inv(A_l)

        # Precomputations related to the operator
        Y_l = A_l_inv @ C_l.transpose()

        # Compute inverse Schur complement
        S_l_inv = sp_inv(C_l @ Y_l)

        # Compute correction for each coarse space function with support on K_l
        w_l = lil_matrix((l_dofs.size, num_fine_dofs_local))
        for i in range(l_dofs.size):
            q_i = A_l_inv @ r_l[i].transpose()
            lambda_i = S_l_inv @ (C_l @ q_i)
            w_l_i = q_i - Y_l @ lambda_i
            w_l[i] = w_l_i.transpose()

        # Update the corrector matrix
        Q_h += T_H_l.transpose() @ w_l @ R_h_l

    return Q_h


def main():
    cell_marker_1 = 1
    cell_marker_2 = 2
    boundary_marker = 1

    # Create coarse mesh
    create_gmsh(cell_marker_1, cell_marker_2, boundary_marker, False)

    # Read coarse mesh
    mesh_c, ct_c, ft_c = read_mesh(False)

    # Create fine mesh
    # https://fenicsproject.discourse.group/t/input-for-mesh-refinement-with-refine-plaza/13426/4
    mesh_f, parent_cells, parent_facets = refine_plaza(
        mesh_c,
        redistribute=False,
        option=RefinementOption.parent_cell_and_facet
    )

    # Transfer cell mesh tags from coarse to fine mesh
    fine_cell_entities = np.arange(len(parent_cells))
    fine_cell_values = ct_c.values[parent_cells]
    ct_f = meshtags(mesh_f, 2, fine_cell_entities, fine_cell_values)

    # Transfer facet mesh tags from coarse to fine mesh
    fine_facet_entities = []
    fine_facet_values = []
    for child_facet, parent_facet in enumerate(parent_facets):
        if parent_facet > -1:
            fine_facet_entities.append(child_facet)
            fine_facet_values.append(ft_c.values[parent_facet])
    fine_facet_entities = np.array(fine_facet_entities)
    fine_facet_values = np.array(fine_facet_values)
    ft_f = meshtags(mesh_f, 1, fine_facet_entities, fine_facet_values)

    # Number of cells on coarse mesh
    num_cells_c = ct_c.indices.shape[0]  # N_T_H
    index_map_c = mesh_c.topology.index_map(2)
    assert index_map_c.size_local == index_map_c.size_global
    assert num_cells_c == index_map_c.size_local

    # P1 function space on coarse mesh
    FS_c = functionspace(mesh_c, ("CG", 1))
    assert FS_c.dofmap.index_map.size_local == FS_c.dofmap.index_map.size_global
    num_dofs_c = FS_c.dofmap.index_map.size_local * FS_c.dofmap.index_map_bs  # N_H

    # P1 function space on fine mesh
    FS_f = functionspace(mesh_f, ("CG", 1))
    assert FS_f.dofmap.index_map.size_local == FS_f.dofmap.index_map.size_global
    num_dofs_f = FS_f.dofmap.index_map.size_local * FS_f.dofmap.index_map_bs  # N_h

    # Create kappa
    kappa = create_kappa(mesh_f, ct_f, 1, 2, FS_f)
    f = create_point_source(FS_f, 0.3, 0.3, 1e-6)

    # Define our problem on fine mesh using Unified Form Language (UFL)
    # and assemble the stiffness and mass matrices A_h and M_h
    u_f = TrialFunction(FS_f)
    v_f = TestFunction(FS_f)
    a = inner(kappa * grad(u_f), grad(v_f)) * dx
    m = inner(u_f, v_f) * dx
    L = inner(f, v_f) * dx

    A_h = assemble_matrix(form(a)).to_scipy()
    M_h = assemble_matrix(form(m)).to_scipy()
    f_h = assemble_vector(form(L))

    assert A_h.shape == (num_dofs_f, num_dofs_f)
    assert M_h.shape == (num_dofs_f, num_dofs_f)

    # Create boundary correction (restriction) matrix B_H
    boundary_facets_c = ft_c.find(boundary_marker)
    boundary_dofs_c = locate_dofs_topological(FS_c, 1, boundary_facets_c)
    B_H = csr_matrix((np.ones_like(boundary_dofs_c), (boundary_dofs_c, boundary_dofs_c)),
                     shape=(num_dofs_c, num_dofs_c))

    # Find boundary dofs on fine mesh.
    # We need boundary dofs on both meshes in order to remove them from local dofs on each patch.
    boundary_facets_f = ft_f.find(boundary_marker)
    boundary_dofs_f = locate_dofs_topological(FS_f, 1, boundary_facets_f)

    # Create projection matrix P_h from coarse mesh Lagrange space to fine mesh Lagrange space
    P_h_petsc = interpolation_matrix(FS_c, FS_f)
    P_h_petsc.assemble()
    # https://fenicsproject.discourse.group/t/converting-to-scipy-sparse-matrix-without-eigen-backend/847/2
    P_h = csr_matrix(P_h_petsc.getValuesCSR()[::-1], shape=P_h_petsc.size).transpose()

    # Calculate constraint matrix C_h
    C_h = P_h @ M_h

    # Create corrector matrix Q_h
    Q_h = compute_correction_operator(mesh_c, mesh_f, FS_c, FS_f, ct_c, parent_cells, boundary_dofs_c, boundary_dofs_f,
                                      A_h, B_H, P_h, C_h)

    # Add the corrector matrix to the solution and solve the system
    A_H_LOD = B_H @ (P_h + Q_h) @ A_h @ (P_h + Q_h).transpose() @ B_H
    f_H = B_H @ (P_h + Q_h) @ f_h
    u_H_LOD = spsolve(A_H_LOD, f_H)
    u_h_LOD = (P_h + Q_h).transpose() @ u_H_LOD

    # u_h_LOD is now a vector of size num_dofs_f, which we can wrap in a Function object using dolfinx
    # and display using pyvista or ParaView
    uhLOD = Function(FS_f)
    uhLOD.x.array.real = u_h_LOD

    grid_uhLOD = pv.UnstructuredGrid(*vtk_mesh(FS_f))
    grid_uhLOD.point_data["uhLOD"] = uhLOD.x.array.real
    grid_uhLOD.set_active_scalars("uhLOD")

    # pv.start_xvfb()
    plot = pv.Plotter(window_size=[1000, 1000])
    plot.show_axes()
    plot.show_grid()
    plot.add_mesh(grid_uhLOD, show_edges=True)
    plot.show()


if __name__ == "__main__":
    main()
