import sys
try:
    import grasp_nms
    print("grasp_nms found in", grasp_nms.__file__)
except ImportError:
    print("grasp_nms NOT FOUND!")
