# This file was taken from: https://gitlab.com/ase/ase/-/merge_requests/2581
# The copyright to the unmodified portions of the code belongs to Michael Waters
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

# pylint: skip-file


def _write_n2p2(fid, atoms, comment="", with_charges=False, with_energy_and_forces="auto"):
    # float_form = "{: 13.8f}" # looks nicer but, results can be very sensitive
    float_form = "{: 1.16E}"  # {: 1.16E} is what N2P2 uses in outout.data so we should match them
    lattice_format = "lattice  %s  %s  %s\n" % tuple(3 * [float_form])
    atom_format = "atom %s %s %s   {} %s %s %s %s %s\n" % tuple(8 * [float_form])
    unused_column = [0.0]

    fid.write("begin\n")
    if comment != "":
        fid.write(f"comment {comment}\n")

    for i in range(3):
        fid.write(lattice_format.format(*atoms.cell[i]))

    # deciding what to do about energies and forces
    fillzeros = True
    if with_energy_and_forces:
        fillzeros = False

    if isinstance(with_energy_and_forces, str):
        if with_energy_and_forces.lower() == "auto":
            # if hasattr(atoms, 'calc'):
            if atoms._calc is not None:
                fillzeros = False
            else:
                fillzeros = True

    if fillzeros:
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
    else:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    unuseds = unused_column * len(atoms)

    if with_charges:
        charges = atoms.get_charges()
    else:
        charges = np.zeros(len(atoms))

    for data in zip(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        symbols,
        charges,
        unuseds,
        forces[:, 0],
        forces[:, 1],
        forces[:, 2],
    ):
        fid.write(atom_format.format(*data))
    if not fillzeros:
        fid.write(f"energy {float_form}\n".format(energy))
    fid.write("end\n")


def write_n2p2(
    filename,  # ='input.data',
    images,
    comment="",
    with_charges=False,
    with_energy_and_forces="auto",
):

    # filename is typically 'input.data'

    fid = open(filename, "w")
    if type(images) is Atoms:
        atoms = images
        _write_n2p2(fid, atoms, comment, with_charges, with_energy_and_forces)
    else:
        for atoms in images:
            _write_n2p2(fid, atoms, comment, with_charges, with_energy_and_forces)
    fid.close()


class N2P2_data:
    def __init__(self, filename="input.data"):
        self.fid = open(filename, "w")

    def write(self, images, comment="", with_charges=False, with_energy_and_forces=True):
        if type(images) is Atoms:
            atoms = images
            write_atoms(self.fid, atoms, comment, with_charges, with_energy_and_forces)
        else:
            for atoms in images:
                write_atoms(self.fid, atoms, comment, with_charges, with_energy_and_forces)

    def close(self):
        self.fid.close()


def read_n2p2(
    filename="output.data",
    index=-1,
    with_energy_and_forces="auto",
    model_length_units=units.Ang,  # n2p2 doesn't have units, so you should specify if not eV and Ang
    model_energy_units=units.eV,
):
    """Import n2p2 .data file
    I'm not sure this correct
    """
    fd = open(filename, "r")  # @reader decorator ensures this is a file descriptor???
    images = list()

    line = fd.readline()
    while "begin" in line:
        line = fd.readline()
        if "comment" in line:
            # comment = line[7:]
            line = fd.readline()

        cell = np.zeros((3, 3))
        for ii in range(3):
            cell[ii] = [float(jj) for jj in line.split()[1:4]]
            line = fd.readline()

        positions = []
        symbols = []
        charges = []  # not used yet
        nn = []  # not used
        forces = []
        energy = 0.0
        # charge = 0.0

        while "atom" in line:
            sline = line.split()
            positions.append([float(pos) for pos in sline[1:4]])
            symbols.append(sline[4])
            nn.append(float(sline[5]))
            charges.append(float(sline[6]))
            forces.append([float(pos) for pos in sline[7:10]])
            line = fd.readline()

        while "end" not in line:
            if "energy" in line:
                energy = float(line.split()[-1])
            if "charge" in line:
                # charge = float(line.split()[-1])
                pass
            line = fd.readline()

        image = Atoms(
            symbols=symbols,
            positions=np.array(positions) * model_length_units,
            cell=cell * model_length_units,
            pbc=[True, True, True],
        )

        store_energy_and_forces = False
        if with_energy_and_forces:
            store_energy_and_forces = True
        elif with_energy_and_forces == "auto":
            if energy != 0.0 or np.absolute(forces).sum() > 1e-8:
                store_energy_and_forces = True

        if store_energy_and_forces:
            image.calc = SinglePointCalculator(
                atoms=image,
                energy=energy * model_energy_units,
                free_energy=energy * model_energy_units,
                forces=np.array(forces) * model_energy_units / model_length_units,
                charges=charges,
            )
            # charge  = charge)
        images.append(image)
        # to start the next section
        line = fd.readline()

    if index == ":" or index is None:
        return images
    else:
        return images[index]
