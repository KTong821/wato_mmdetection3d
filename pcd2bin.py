from pypcd import pypcd
import numpy as np
import os

def to_bin(old_dir, new_dir):
    for filename in os.listdir(old_dir):
        try:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            f = os.path.join(old_dir, filename)
            if os.path.isfile(f):
                print(f)      
                pc = pypcd.PointCloud.from_path(f)

                ## Get data from pcd (x, y, z, intensity)
                np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
                np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
                np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
                np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(
                    np.float32
                ) / 256

                ## Stack all data
                points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

                # print(points_32[:5])
                # print(points_32.shape)

                ## Save bin old_file
                # points_32.tofile(new_file)
                np.save(os.path.join(new_dir, filename)[:-4], points_32)

        except:
            pass

if (__name__ == "__main__"):
    name = "cyclist_forward"
    to_bin("/wato/host/wato_monorepo/cyclist/" + name,
        "/wato/host/wato_monorepo/cyclist/npy/" + name)