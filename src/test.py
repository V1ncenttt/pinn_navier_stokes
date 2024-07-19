import h5py

def inspect_h5(filename):
    def print_attrs(name, obj):
        print(f"{name}: {dict(obj.attrs) if hasattr(obj, 'attrs') else ''}")
    with h5py.File(filename, 'r') as f:
        f.visititems(print_attrs)

# Inspect velocity and pressure HDF5 files
inspect_h5('navier_stokes_cylinder/velocity_series.h5')
inspect_h5('navier_stokes_cylinder/pressure_series.h5')
