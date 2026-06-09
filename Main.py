import pybullet as p
import pybullet_data
import numpy as np
import time
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import os
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"  # Force OpenGL 3.3 for better compatibility
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"  # Match GLSL version to OpenGL
# ── STEP 1: Load OpenVLA ──────────────────────────────────────────
print("Loading OpenVLA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

print("OpenVLA loaded!")

# ── STEP 2: Set up PyBullet ───────────────────────────────────────
print("Setting up simulation...")

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")
tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
robotId = p.loadURDF("franka_panda/panda.urdf",
                      basePosition=[0, 0, 0.63],
                      useFixedBase=True)
cubeId  = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.1, 0.7])
p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

# Franka Panda end effector is link 11
END_EFFECTOR_LINK = 11

# Franka has 7 arm joints (indices 0-6) + 2 finger joints (7-8)
NUM_ARM_JOINTS = 7

# Starting pose — arm pointing forward over the table
# These are joint angles in radians for a natural reaching position
START_JOINT_ANGLES = [0, -0.5, 0, -2.0, 0, 1.5, 0.8]

# Apply starting position
for i in range(NUM_ARM_JOINTS):
    p.resetJointState(robotId, i, START_JOINT_ANGLES[i])

# Let everything settle
for _ in range(200):
    p.stepSimulation()

print("Simulation ready!")

# ── STEP 3: Camera ────────────────────────────────────────────────
def get_camera_image():
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.5, -0.5, 1.2],
        cameraTargetPosition=[0.5, 0, 0.65],
        cameraUpVector=[0, 0, 1]
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width=224, height=224,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )
    rgb_array = np.array(rgb, dtype=np.uint8).reshape(224, 224, 4)[:, :, :3]
    return Image.fromarray(rgb_array)

# ── STEP 4: OpenVLA inference ─────────────────────────────────────
def get_action(image, instruction):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False
        )
    return action
def scripted_approach(target_pos, num_steps=300):
    """
    Phase 1 — Move arm directly above cube using simple IK.
    No OpenVLA involved — pure scripted motion.
    Like a crane positioning itself before the operator takes over.
    """
    print("\nPhase 1: Scripted approach — moving arm above cube...")

    # Target is directly above the cube, 15cm up
    # We go above first so we don't knock it over approaching from the side
    above_cube = [target_pos[0], target_pos[1], target_pos[2] + 0.15]

    # Orientation — gripper pointing straight down
    target_orn = p.getQuaternionFromEuler([3.14159, 0, 0])

    for step in range(num_steps):
        # Calculate IK for above-cube position
        joint_angles = p.calculateInverseKinematics(
            robotId,
            END_EFFECTOR_LINK,
            above_cube,
            target_orn
        )

        # Command all arm joints
        for i in range(NUM_ARM_JOINTS):
            p.setJointMotorControl2(
                bodyIndex=robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=100
            )

        # Keep gripper open during approach
        for finger_joint in [9, 10]:
            p.setJointMotorControl2(
                bodyIndex=robotId,
                jointIndex=finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.04,
                force=20
            )

        # Step physics smoothly
        p.stepSimulation()
        time.sleep(1/240)

        # Check how close we are every 50 steps
        if step % 50 == 0:
            ee_state = p.getLinkState(robotId, END_EFFECTOR_LINK)
            ee_pos = ee_state[0]
            dist = np.sqrt(sum((ee_pos[i] - above_cube[i])**2 for i in range(3)))
            print(f"  Step {step:03d} | EE pos: "
                  f"X:{ee_pos[0]:.3f} Y:{ee_pos[1]:.3f} Z:{ee_pos[2]:.3f} | "
                  f"Distance to target: {dist:.4f}m")

            # Close enough — stop early
            if dist < 0.01:
                print("  Reached position!")
                break

    print("Phase 1 complete — arm is above cube\n")

def scripted_descend(target_pos, num_steps=200):
    """
    Phase 1.5 — Descend straight down onto the cube.
    Moves from above the cube to just touching it.
    Like an elevator going straight down.
    """
    print("Phase 1.5: Descending onto cube...")

    # Target is right at cube height
    at_cube = [target_pos[0], target_pos[1], target_pos[2] + 0.02]
    target_orn = p.getQuaternionFromEuler([3.14159, 0, 0])

    for step in range(num_steps):
        joint_angles = p.calculateInverseKinematics(
            robotId,
            END_EFFECTOR_LINK,
            at_cube,
            target_orn
        )

        for i in range(NUM_ARM_JOINTS):
            p.setJointMotorControl2(
                bodyIndex=robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=100
            )

        # Keep gripper open while descending
        for finger_joint in [9, 10]:
            p.setJointMotorControl2(
                bodyIndex=robotId,
                jointIndex=finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.04,
                force=20
            )

        p.stepSimulation()
        time.sleep(1/240)

        if step % 50 == 0:
            ee_state = p.getLinkState(robotId, END_EFFECTOR_LINK)
            ee_pos = ee_state[0]
            dist = np.sqrt(sum((ee_pos[i] - at_cube[i])**2 for i in range(3)))
            print(f"  Step {step:03d} | "
                  f"EE Z:{ee_pos[2]:.3f} | "
                  f"Distance: {dist:.4f}m")
            if dist < 0.01:
                print("  Reached cube!")
                break

    print("Phase 1.5 complete — arm is at cube\n")

def scripted_place(drop_pos, num_steps=300):
    """
    Phase 3 — Carry cube to target position and drop it.
    Scripted motion — lift up, move to target, descend, release.
    """
    print("\nPhase 3: Moving to drop position...")

    target_orn = p.getQuaternionFromEuler([3.14159, 0, 0])

    # Step 1 — lift up first so we don't drag the cube across the table
    lift_pos = list(drop_pos)
    lift_pos[2] += 0.2  # lift 20cm above drop target

    print("  Lifting cube...")
    for step in range(num_steps):
        joint_angles = p.calculateInverseKinematics(
            robotId, END_EFFECTOR_LINK, lift_pos, target_orn
        )
        for i in range(NUM_ARM_JOINTS):
            p.setJointMotorControl2(
                bodyIndex=robotId, jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i], force=100
            )
        # Keep gripper closed — holding the cube
        for finger_joint in [9, 10]:
            p.setJointMotorControl2(
                bodyIndex=robotId, jointIndex=finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0, force=20
            )
        p.stepSimulation()
        time.sleep(1/240)

        if step % 50 == 0:
            ee_state = p.getLinkState(robotId, END_EFFECTOR_LINK)
            ee_pos = ee_state[0]
            dist = np.sqrt(sum((ee_pos[i] - lift_pos[i])**2 for i in range(3)))
            print(f"  Step {step:03d} | Distance to lift point: {dist:.4f}m")
            if dist < 0.02:
                print("  Lifted!")
                break

    # Step 2 — move horizontally to above drop position
    print("  Moving to drop position...")
    for step in range(num_steps):
        joint_angles = p.calculateInverseKinematics(
            robotId, END_EFFECTOR_LINK, lift_pos, target_orn
        )
        # same as lift_pos but now targeting drop x,y
        above_drop = [drop_pos[0], drop_pos[1], lift_pos[2]]
        joint_angles = p.calculateInverseKinematics(
            robotId, END_EFFECTOR_LINK, above_drop, target_orn
        )
        for i in range(NUM_ARM_JOINTS):
            p.setJointMotorControl2(
                bodyIndex=robotId, jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i], force=100
            )
        for finger_joint in [9, 10]:
            p.setJointMotorControl2(
                bodyIndex=robotId, jointIndex=finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0, force=20
            )
        p.stepSimulation()
        time.sleep(1/240)

        if step % 50 == 0:
            ee_state = p.getLinkState(robotId, END_EFFECTOR_LINK)
            ee_pos = ee_state[0]
            dist = np.sqrt(sum((ee_pos[i] - above_drop[i])**2 for i in range(3)))
            print(f"  Step {step:03d} | Distance to drop point: {dist:.4f}m")
            if dist < 0.02:
                print("  Above drop position!")
                break

    # Step 3 — open gripper and drop
    print("  Dropping cube...")
    for step in range(60):
        for finger_joint in [9, 10]:
            p.setJointMotorControl2(
                bodyIndex=robotId, jointIndex=finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.04, force=20
            )
        p.stepSimulation()
        time.sleep(1/240)

    print("Phase 3 complete — cube dropped!\n")
# ── STEP 5: Apply action to robot ────────────────────────────────
def apply_action(action):
    """
    Converts OpenVLA's end-effector delta into joint angles using IK,
    then commands each joint to move there.
    """
    # Get current end effector position and orientation
    ee_state = p.getLinkState(robotId, END_EFFECTOR_LINK)
    current_pos = list(ee_state[0])    # [x, y, z]
    current_orn = list(ee_state[1])    # [qx, qy, qz, qw]

    # Apply deltas from OpenVLA to get new target position
    # Scale factor — actions are small normalized values, scale them to meters
    SCALE = 0.15
    target_pos = [
        current_pos[0] + action[0] * SCALE,
        current_pos[1] + action[1] * SCALE,
        current_pos[2] + action[2] * SCALE,
    ]

    # Keep orientation change small too
    # Convert current euler, add delta, convert back to quaternion
    current_euler = list(p.getEulerFromQuaternion(current_orn))
    target_euler = [
        current_euler[0] + action[3] * SCALE,
        current_euler[1] + action[4] * SCALE,
        current_euler[2] + action[5] * SCALE,
    ]
    target_orn = p.getQuaternionFromEuler(target_euler)

    # Use PyBullet's built-in IK to find joint angles for target pose
    # This is the same IK concept you used on the UR10 — give target pose,
    # get back joint angles
    joint_angles = p.calculateInverseKinematics(
        robotId,
        END_EFFECTOR_LINK,
        target_pos,
        target_orn
    )

    # Command each arm joint to its new angle
    # POSITION_CONTROL is like telling each joint "go to this angle"
    # as opposed to TORQUE_CONTROL which says "apply this force"
    for i in range(NUM_ARM_JOINTS):
        p.setJointMotorControl2(
            bodyIndex=robotId,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_angles[i],
            force=100       # max force the motor can apply, in Newtons
        )

    # Handle gripper — action[6] close to 1 means close, close to 0 means open
    gripper_open  = 0.04   # 4cm open
    gripper_close = 0.0    # fully closed
    gripper_pos = gripper_open if action[6] < 0.5 else gripper_close

    # Franka finger joints are index 9 and 10
    for finger_joint in [9, 10]:
        p.setJointMotorControl2(
            bodyIndex=robotId,
            jointIndex=finger_joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripper_pos,
            force=20
        )

cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
cube_pos = list(cube_pos)
print(f"Cube position: X:{cube_pos[0]:.3f} Y:{cube_pos[1]:.3f} Z:{cube_pos[2]:.3f}")


scripted_approach(cube_pos)  # move above the cube before starting
scripted_descend(cube_pos)   # descend straight down onto the cube



# ── STEP 6: Main loop ─────────────────────────────────────────────
# Phase 2 — OpenVLA grasping
instruction = "pick up the red cube"
print(f"Phase 2: OpenVLA taking over — '{instruction}'")
print("Running OpenVLA — will stop once gripper closes...\n")

step = 0
gripper_closed_steps = 0

try:
    while True:
        image = get_camera_image()

        if step % 10 == 0:
            image.save(f"frame_{step:04d}.png")

        action = get_action(image, instruction)
        apply_action(action)

        ee_state = p.getLinkState(robotId, END_EFFECTOR_LINK)
        ee_pos = ee_state[0]
        print(f"Step {step:03d} | "
              f"Action X:{action[0]:+.3f} Y:{action[1]:+.3f} Z:{action[2]:+.3f} "
              f"Gripper:{action[6]:+.3f} | "
              f"EE X:{ee_pos[0]:.3f} Y:{ee_pos[1]:.3f} Z:{ee_pos[2]:.3f}")

        for _ in range(48):
            p.stepSimulation()
            time.sleep(1/240)

        # If gripper stays closed for 5 consecutive steps — assume grasp succeeded
        if action[6] > 0.5:
            gripper_closed_steps += 1
        else:
            gripper_closed_steps = 0

        if gripper_closed_steps >= 5:
            print("\nGrasp detected! Moving to place phase...")
            break

        step += 1

except KeyboardInterrupt:
    print("\nStopped.")
    p.disconnect()

# Phase 3 — place the cube
DROP_POSITION = [0.4, -0.2, 0.75]
scripted_place(DROP_POSITION)

print("\nFull pick and place complete!")
p.disconnect()