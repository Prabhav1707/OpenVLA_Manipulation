import pybullet as p
import pybullet_data
import numpy as np
import time
from PIL import Image

# Connect to PyBullet in GUI mode — opens a window you can see
physicsClient = p.connect(p.GUI)

# Tell PyBullet where to find its built-in models (plane, robot etc)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity — same as real world
p.setGravity(0, 0, 9.81)

# Load ground plane
planeId = p.loadURDF("plane.urdf")

# Load a table
tableId = p.loadURDF(
    "table/table.urdf",
    basePosition=[0.5, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)

# Load Franka Panda robot arm
robotId = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0.63],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True
)

# Load a small red cube on the table
cubeId = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.5, 0, 0.7],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)

# Make the cube red
p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

print("Simulation loaded!")
print("You should see a window with a robot arm, table, and red cube")

# Let simulation settle for a moment
for _ in range(100):
    p.stepSimulation()
    time.sleep(1/240)

# Camera setup — positioned above and in front looking down at the scene
def get_camera_image():
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.5, -0.5, 1.2],   # where camera is
        cameraTargetPosition=[0.5, 0, 0.65],   # what it looks at
        cameraUpVector=[0, 0, 1]               # which way is up
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,           # field of view in degrees
        aspect=1.0,       # square image
        nearVal=0.1,      # near clipping plane
        farVal=10.0       # far clipping plane
    )
    width, height, rgb, depth, seg = p.getCameraImage(
        width=224,
        height=224,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )
    # Convert to PIL image — drop alpha channel
    rgb_array = np.array(rgb, dtype=np.uint8).reshape(224, 224, 4)[:, :, :3]
    return Image.fromarray(rgb_array)

# Take a camera shot and save it so we can see what the model sees
image = get_camera_image()
image.save("scene.png")
print("Camera image saved as scene.png — open it to see what the model will see")

# Keep window open for 5 seconds so you can look at it
for _ in range(5 * 240):
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
print("Done")
