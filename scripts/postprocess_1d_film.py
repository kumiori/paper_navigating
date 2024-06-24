from petsc4py import PETSc
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI


def read_fields_from_npz(npz_file, time_step, num_points=-1):
    """
    Read mode data for a given timestep and x_values from an npz file.

    Parameters:
    - npz_file (numpy.lib.npyio.NpzFile): The npz file containing mode shapes data.
    - time_step (int): The timestep to read.
    - num_points (int): The number of domain nodes.

    Returns:
    - mode_data (dict): A dictionary containing mode-specific fields for the given timestep.
    """
    mode_data = {}
    mode_data["mesh"] = npz_file["mesh"]
    if 'time_steps' not in npz_file or time_step not in npz_file['time_steps']:
        print(f"No data available for timestep {time_step}.")
        return None

    index = np.where(npz_file['time_steps'] == time_step)[0][0]

    for mode in range(1, num_modes + 1):
        mode_key = f'mode_{mode}'

        if mode_key not in npz_file['point_values'].item():
            print(
                f"No data available for mode {mode} at timestep {time_step}.")
            continue

        fields = npz_file['point_values'].item()[mode_key]
        if 'bifurcation_β' not in fields or 'stability_β' not in fields:
            print(f"Incomplete data for mode {mode} at timestep {time_step}.")
            continue

        field_β_bif_values = np.array(fields['bifurcation_β'][index])
        field_v_bif_values = np.array(fields['bifurcation_v'][index])
        field_β_stab_values = np.array(fields['stability_β'][index])
        field_v_stab_values = np.array(fields['stability_v'][index])

        # Assuming x_values is known or can be obtained
        if num_points == -1:
            num_points = len(npz_file["mesh"])
        # Replace with actual x_values
        x_values = np.linspace(0, 1, num_points)

        mode_data["fields"] = {
            'bifurcation_β': field_β_bif_values,
            'bifurcation_v': field_v_bif_values,
            'stability_β': field_β_stab_values,
            'stability_v': field_v_stab_values,
        }

        mode_data["time_step"] = time_step
        mode_data["lambda_bifurcation"] = np.nan
        mode_data["lambda_stability"] = np.nan

    return mode_data


def save_binary_data(filename, data):
    viewer = PETSc.Viewer().createBinary(filename, "w")

    if isinstance(data, list):
        for item in data:
            item.view(viewer)
    elif isinstance(data, PETSc.Mat):
        data.view(viewer)
    elif isinstance(data, PETSc.Vec):
        data.view(viewer)
    else:
        raise ValueError("Unsupported data type for saving")


def load_binary_data(filename):
    viewer = PETSc.Viewer().createBinary(filename, "r")
    data = []
    vectors = []

    i = 0
    while True:
        try:
            vec = PETSc.Vec().load(viewer)

            vectors.append(vec)
            i += 1
        except PETSc.Error as e:
            # __import__('pdb').set_trace()
            # if e.ierr == PETSc.Error.S_ARG_WRONG:
            # print(f"Error {e.ierr}: {translatePETScERROR.get(e.ierr, 'Unknown error')}")
            print(f"Error {e.ierr}")
            break
            # else:
            # raise

    return data


def load_binary_vector(filename):
    """
    Load a binary file containing a PETSc vector.

    Args:
        filename (str): Path to the binary file.

    Returns:
        PETSc.Vec: Loaded PETSc vector.
    """
    try:
        # Create a PETSc viewer for reading
        viewer = PETSc.Viewer().createBinary(filename, "r")

        # Load the vector from the viewer
        vector = PETSc.Vec().load(viewer)

        # Close the viewer
        viewer.destroy()

        return vector

    except PETSc.Error as e:
        print(f"Error: {e}")
        return None


def load_binary_matrix(filename):
    """
    Load a binary file containing a PETSc Matrix.

    Args:
        filename (str): Path to the binary file.

    Returns:
        PETSc.Mat: Loaded PETSc Matrix.
    """
    try:
        # Create a PETSc viewer for reading
        viewer = PETSc.Viewer().createBinary(filename, "r")

        # Load the vector from the viewer
        vector = PETSc.Mat().load(viewer)

        # Close the viewer
        viewer.destroy()

        return vector

    except PETSc.Error as e:
        print(f"Error: {e}")
        return None

# Open XDMF file and read the mesh
filename = "../data/irreversible/7f4361886184f3c6791fe16bf4f4b3f2/1d-bar.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf_file:
    mesh = xdmf_file.read_mesh()
    
# Create function spaces for the fields
V = dolfinx.fem.functionspace(mesh, ("CG", 1))  # Example for scalar field

v = dolfinx.fem.Function(V)

v.vector.set(1.)

output = "../data/irreversible/test.xdmf" 

save_binary_data(output, [v.vector])
b = load_binary_vector(output)

print(b[:])
