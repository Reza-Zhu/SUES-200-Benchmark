import os
import pickle
from VLADlib.VLAD import indexBallTree


def build_ball_tree(gallery_filepath):
    with open(gallery_filepath, "rb") as f:
        VLAD_DS = pickle.load(f)
    imageID = VLAD_DS[0]
    V = VLAD_DS[1]
    pathImageData = VLAD_DS[2]
    # print(V)
    print(V.shape)
    tree = indexBallTree(V, Leaf_Size)
    view = ""
    if "satellite" in gallery_filepath:
        view = "satellite"
    elif "drone" in gallery_filepath:
        view = "drone"
    tree_save_path = "tree_" + view + "_" + str(Height) + ".pickle"
    tree_save_path = os.path.join("./Data", tree_save_path)

    with open(tree_save_path, "wb") as f:
        pickle.dump([imageID, tree, pathImageData], f, pickle.HIGHEST_PROTOCOL)


Leaf_Size = 40

for Height in [150, 200, 250, 300]:

    gallery_satellite_path = "./Data/gallery_satellite_%s_VLADDictionary" % str(Height) + ".pickle"
    gallery_drone_path = "./Data/gallery_drone_%s_VLADDictionary" % str(Height) + ".pickle"

    build_ball_tree(gallery_satellite_path)
    build_ball_tree(gallery_drone_path)
    break
