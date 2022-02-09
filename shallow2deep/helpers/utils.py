import h5py

def load_h5_volume(path, internal_path):
    print(f"Loading volume {internal_path} from {path}...")
    with h5py.File(path,"r") as f:
        vol = f[internal_path][:]
    return vol

def write_h5_volume(path, internal_path, volume):
    print(f"Saving volume {internal_path} to {path}...")
    with h5py.File(path,"a") as f:
        f.create_dataset(internal_path,data=volume,compression="gzip")
