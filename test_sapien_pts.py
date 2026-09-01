import sapien.core as sapien
import numpy as np

def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    context = renderer._internal_context
    points = np.random.rand(100, 3).astype(np.float32)
    colors = np.ones((100, 4)).astype(np.float32)
    print(dir(context))
    try:
        pc = context.create_point_cloud(points, colors)
        print("Success:", pc)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
